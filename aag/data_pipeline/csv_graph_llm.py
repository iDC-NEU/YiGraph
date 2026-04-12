"""
文本知识库内 CSV 边表：由 LLM 推断列语义，并生成与 Text2Graph 一致的
process_text/{name}_accounts.csv / {name}_transactions.csv 及 graph_schemas 条目结构。
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from aag.utils.parse_json import extract_json_from_response
from aag.reasoner.model_deployment import OllamaEnv, OpenAIEnv


CSV_INFER_PROMPT = """你是一个结构化数据分析助手。下面是一张边表 CSV 的列名与若干样例行。
请推断：哪一列表示边的**源端点**、哪一列表示**目标端点**；哪一列（若有）表示**关系类型/谓词**；
还有哪些列应作为**边的附加属性**（非端点、非关系名本身时）。
再从整张表角度推断：生成顶点表时，用端点字符串作为展示名是否足够；若表中有明显可作为「顶点类型」的列名可指出（通常没有则 null）。

**重要**：source_field、target_field、edge_relation_field、edge_attribute_fields 中的列名必须**逐字**来自「列名」列表，不要编造。

## 列名（按顺序）
{columns}

## 样例（最多 12 行，JSON 数组）
{sample_json}

## 输出格式
仅输出一个 JSON 对象（不要 markdown 代码块），字段如下：
{{
  "source_field": "列名",
  "target_field": "列名",
  "edge_relation_field": "列名或 null",
  "edge_attribute_fields": ["可选列名", "..."],
  "vertex_id_semantics": "端点字符串即顶点主键，输出时为整型 id",
  "vertex_display_semantics": "用端点原始字符串作为展示名",
  "vertex_type_default": "OTHER",
  "vertex_extra_attribute_fields": [],
  "is_directed": true,
  "confidence": "high|medium|low",
  "notes": "一句话说明"
}}

vertex_extra_attribute_fields：若你认为某些列应合并进顶点属性（除端点外），填列名列表；多数边表可填空数组 []。
"""


def _make_llm(llm_type: str, llm_name: str, api_key: str, base_url: Optional[str]):
    if llm_type == "ollama":
        return OllamaEnv(llm_mode_name=llm_name)
    if llm_type == "openai":
        return OpenAIEnv(api_key=api_key, model_name=llm_name, base_url=base_url)
    raise ValueError(f"不支持的 LLM 类型: {llm_type}")


def _llm_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return (getattr(raw, "text", None) or "") or ""


def read_user_csv(path: str, max_body_rows: int = 8000) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    读取 CSV，返回表头与行字典列表。
    依次尝试多种编码：UTF-8 与 GB 系列混用时，仅试 GBK 常会误伤 UTF-8 文件（如 0xBB 为 UTF-8 多字节片段），
    因此增加 gb18030、latin-1 等兜底（latin-1 永不因解码失败抛错）。
    """
    def _read(enc: str) -> Tuple[List[str], List[Dict[str, str]]]:
        with open(path, "r", encoding=enc, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV 无表头")
            fields = [h.strip() for h in reader.fieldnames if h is not None]
            rows = []
            for i, row in enumerate(reader):
                if i >= max_body_rows:
                    break
                rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k})
            return fields, rows

    last_err: Optional[Exception] = None
    for enc in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "big5",
        "cp936",
        "latin-1",
    ):
        try:
            return _read(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def infer_csv_schema_with_llm(
    columns: List[str],
    sample_rows: List[Dict[str, str]],
    llm_type: str,
    llm_name: str,
    api_key: str,
    base_url: Optional[str],
    max_retries: int = 4,
) -> Dict[str, Any]:
    llm = _make_llm(llm_type, llm_name, api_key, base_url)
    sample = sample_rows[:12]
    prompt = CSV_INFER_PROMPT.format(
        columns=json.dumps(columns, ensure_ascii=False),
        sample_json=json.dumps(sample, ensure_ascii=False),
    )
    last_err = None
    for attempt in range(1, max_retries + 1):
        raw = llm.generate_response(query=prompt)
        text = _llm_text(raw)
        if not text:
            last_err = "模型无返回"
            continue
        try:
            data = extract_json_from_response(text)
        except ValueError as e:
            last_err = str(e)
            continue
        if not isinstance(data, dict):
            last_err = "JSON 不是对象"
            continue
        return data
    raise RuntimeError(f"CSV 列推断失败（已重试 {max_retries} 次）: {last_err}")


def _heuristic_schema(columns: List[str]) -> Dict[str, Any]:
    """LLM 失败时的极简启发式：前两列为源、目标。"""
    if len(columns) < 2:
        raise ValueError("CSV 至少需要两列才能推断边")
    return {
        "source_field": columns[0],
        "target_field": columns[1],
        "edge_relation_field": None,
        "edge_attribute_fields": columns[2:] if len(columns) > 2 else [],
        "vertex_type_default": "OTHER",
        "vertex_extra_attribute_fields": [],
        "is_directed": True,
        "confidence": "low",
        "notes": "heuristic_fallback",
    }


def _validate_infer(columns: List[str], infer: Dict[str, Any]) -> Dict[str, Any]:
    colset = set(columns)
    src = infer.get("source_field")
    tgt = infer.get("target_field")
    if not isinstance(src, str) or src not in colset:
        raise ValueError(f"无效的 source_field: {src}")
    if not isinstance(tgt, str) or tgt not in colset:
        raise ValueError(f"无效的 target_field: {tgt}")
    if src == tgt:
        raise ValueError("源列与目标列不能为同一列")
    rel = infer.get("edge_relation_field")
    if rel is not None and (not isinstance(rel, str) or rel not in colset):
        raise ValueError(f"无效的 edge_relation_field: {rel}")
    attrs = infer.get("edge_attribute_fields") or []
    if not isinstance(attrs, list):
        attrs = []
    attrs = [a for a in attrs if isinstance(a, str) and a in colset and a not in (src, tgt)]
    if rel:
        attrs = [a for a in attrs if a != rel]
    infer["edge_attribute_fields"] = attrs
    vex = infer.get("vertex_extra_attribute_fields") or []
    if not isinstance(vex, list):
        vex = []
    vex = [a for a in vex if isinstance(a, str) and a in colset and a not in (src, tgt)]
    infer["vertex_extra_attribute_fields"] = vex
    return infer


def materialize_text_csv_graph(
    *,
    user_csv_path: str,
    process_text_dir: str,
    graph_name: str,
    rows: List[Dict[str, str]],
    infer: Dict[str, Any],
) -> Dict[str, Any]:
    """
    生成 process_text 下 *_accounts.csv / *_transactions.csv 与 graph_schemas 边条目。

    边表列顺序：**tran_id**（边序号）, **orig_acct**（头结点 id）, **bene_acct**（尾结点 id）,
    其后依次为 LLM 推断的 **关系列**（若有）与 **边属性列**，列名写入 schema 的 **attribute_fields**。
    """
    os.makedirs(process_text_dir, exist_ok=True)
    src_f = infer["source_field"]
    tgt_f = infer["target_field"]
    rel_f = infer.get("edge_relation_field")
    edge_attrs = infer.get("edge_attribute_fields") or []

    # 边文件尾部列：关系列在前，其余边属性按推断顺序；与 tran_id/orig_acct/bene_acct 不重名
    tail_specs: List[str] = []
    if rel_f:
        tail_specs.append(rel_f)
    for a in edge_attrs:
        if a not in tail_specs:
            tail_specs.append(a)
    only_synthetic_tail = False
    if not tail_specs:
        tail_specs = ["predicate"]
        only_synthetic_tail = True

    nodes: Dict[str, Dict[str, str]] = {}
    edge_records: List[Tuple[str, str, Dict[str, str]]] = []

    for row in rows:
        s = (row.get(src_f) or "").strip()
        t = (row.get(tgt_f) or "").strip()
        if not s or not t:
            continue
        for nid in (s, t):
            if nid not in nodes:
                nodes[nid] = {"dsply_nm": nid, "type": infer.get("vertex_type_default") or "OTHER"}
        edge_records.append((s, t, row))

    sorted_ids = sorted(nodes.keys(), key=lambda x: (len(x), x))
    entity2id = {nid: i + 1 for i, nid in enumerate(sorted_ids)}

    entities_csv_path = os.path.join(process_text_dir, f"{graph_name}_accounts.csv")
    triplets_csv_path = os.path.join(process_text_dir, f"{graph_name}_transactions.csv")

    for p in (entities_csv_path, triplets_csv_path):
        if os.path.exists(p):
            os.remove(p)

    with open(entities_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["acct_id", "dsply_nm", "type"])
        for nid in sorted_ids:
            w.writerow([entity2id[nid], nodes[nid]["dsply_nm"], nodes[nid]["type"]])

    header = ["tran_id", "orig_acct", "bene_acct"] + tail_specs
    with open(triplets_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for eid, (s, t, row) in enumerate(edge_records, start=1):
            tail_vals: List[str] = []
            for _col in tail_specs:
                if only_synthetic_tail:
                    tail_vals.append("RELATED_TO")
                else:
                    v = row.get(_col, "")
                    tail_vals.append("" if v is None else str(v).strip())
            w.writerow([eid, entity2id[s], entity2id[t]] + tail_vals)

    n_edges = len(edge_records)
    if n_edges == 0:
        raise ValueError("未解析到任何有效边（请检查源/目标列或过滤空行后是否仍有数据）")
    n_verts = len(entity2id)
    dr = infer.get("is_directed", True)
    if isinstance(dr, str):
        dr = dr.strip().lower() in ("true", "1", "yes", "y")
    directed_str = "true" if dr else "false"
    # 用于三元组展示的「关系」列：取尾部第一列（与 read_graph_triplets 的 label_field 一致）
    label_col = tail_specs[0] if tail_specs else "predicate"
    schema_dict = {
        "description": f"{graph_name} graph generated from CSV {user_csv_path}",
        "name": graph_name,
        "type": "graph",
        "graph_status": "completed",
        "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema": {
            "vertex": [
                {
                    "attribute_fields": [],
                    "format": "csv",
                    "id_field": "acct_id",
                    "label_field": "dsply_nm",
                    "path": entities_csv_path,
                    "original_path": user_csv_path,
                    "type": "account",
                }
            ],
            "edge": [
                {
                    "attribute_fields": list(tail_specs),
                    "format": "csv",
                    "label_field": label_col,
                    "path": triplets_csv_path,
                    "original_path": user_csv_path,
                    "source_field": "orig_acct",
                    "target_field": "bene_acct",
                    "type": "transfer",
                    "weight_field": None,
                }
            ],
            "graph": {
                "directed": directed_str,
                "heterogeneous": "false",
                "multigraph": "false",
                "weighted": "false",
            },
            "graph_store_info": {
                "backend": "nebula_graph",
                "edge_count": n_edges,
                "space_name": graph_name,
                "status": "success",
                "version": "null",
                "vertex_count": n_verts,
            },
        },
    }
    return schema_dict


def merge_graph_schema_yaml(schema_path: str, schema_dict: Dict[str, Any]) -> None:
    """将 schema_dict 按 name 合并写入 graph_schemas.yaml（单文档）。"""
    import yaml

    name = schema_dict.get("name")
    existing: List[Dict[str, Any]] = []
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if isinstance(doc, dict):
            ds = doc.get("datasets")
            if isinstance(ds, list):
                existing = [d for d in ds if d.get("name") != name]
    existing.append(schema_dict)
    with open(schema_path, "w", encoding="utf-8") as f:
        yaml.dump({"datasets": existing}, f, allow_unicode=True, sort_keys=False)
