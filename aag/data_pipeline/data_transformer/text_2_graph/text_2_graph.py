import re
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import datetime
import json

from aag.utils.file_operation import file_exist
from aag.utils.parse_json import extract_json_from_response
from aag.reasoner.model_deployment import OllamaEnv
from aag.reasoner.model_deployment import OpenAIEnv


def _llm_completion_text(raw: Any) -> str:
    """将 Ollama completion（含 .text）、OpenAI 的 str、或 None 统一为纯文本。"""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return (getattr(raw, "text", None) or "") or ""


prompt_template_str = """
#### Process
**Identify Named Entities**: Extract entities based on the given entity types, ensuring they appear in the order they are mentioned in the text.
**Establish Triplets**: Form triples with reference to the provided predicates, again in the order they appear in the text.

Your final response should follow this format (must be a json format):


**Output:**
```json
{{
    "entities": # type: Dict
    {{
        "Entity Type": ["entity_name"]
    }},
    "triplets": # type: List
    [
        ["subject", "predicate", "object"]
    ]
}}
```

### Example:
**Entity Types:**
ORGANIZATION
COMPANY
CITY
STATE
COUNTRY
OTHER
PERSON
YEAR
MONTH
DAY
OTHER
QUANTITY
EVENT

**Predicates:**
FOUNDED_BY
HEADQUARTERED_IN
OPERATES_IN
OWNED_BY
ACQUIRED_BY
HAS_EMPLOYEE_COUNT
GENERATED_REVENUE
LISTED_ON
INCORPORATED
HAS_DIVISION
ALIAS
ANNOUNCED
HAS_QUANTITY
AS_OF


**Input:**
Walmart Inc. (formerly Wal-Mart Stores, Inc.) is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas.[10] The company was founded by brothers Sam and James "Bud" Walton in nearby Rogers, Arkansas in 1962 and incorporated under Delaware General Corporation Law on October 31, 1969. It also owns and operates Sam's Club retail warehouses.[11][12]

As of October 31, 2022, Walmart has 10,586 stores and clubs in 24 countries, operating under 46 different names.[2][3][4] The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, and as Flipkart Wholesale in India.

**Output:**
```json
{{
"entities": {{
    "COMPANY": ["Walmart Inc.", "Sam's Club", "Flipkart Wholesale"],
    "PERSON": ["Sam Walton", "James 'Bud' Walton"],
    "COUNTRY": ["United States", "Canada", "Mexico", "Central America", "India"],
    "CITY": ["Bentonville", "Rogers"],
    "STATE": ["Arkansas"],
    "DATE": ["1962", "October 31, 1969", "October 31, 2022"],
    "ORGANIZATION": ["Delaware General Corporation Law"]
}},
"triplets": [
    ["Walmart Inc.", "FOUNDED_BY", "Sam Walton"],
    ["Walmart Inc.", "FOUNDED_BY", "James 'Bud' Walton"],
    ["Walmart Inc.", "HEADQUARTERED_IN", "Bentonville, Arkansas"],
    ["Walmart Inc.", "FOUNDED_IN", "1962"],
    ["Walmart Inc.", "INCORPORATED", "October 31, 1969"],
    ["Sam Walton", "FOUNDED", "Walmart Inc."],
    ["James \"Bud\" Walton", "CO-FOUNDED", "Walmart Inc."],
    ["Walmart Inc.", "OWNS", "Sam's Club"],
    ["Flipkart Wholesale", "OWNED_BY", "Walmart Inc."],
    ["Walmart Inc.", "OPERATES_IN", "United States"],
    ["Walmart Inc.", "OPERATES_IN", "Canada"],
    ["Walmart Inc.", "OPERATES_IN", "Mexico"],
    ["Walmart Inc.", "OPERATES_IN", "Central America"],
    ["Walmart Inc.", "OPERATES_IN", "India"]
]
}}
```

### Task:
Your task is to perform Named Entity Recognition (NER) and knowledge graph triplet extraction on the text that follows below.

**Input:**
{context}

**Output:**
"""

prompt_entity_type = """
#### Task
You are asked to determine the type of a given entity. Only provide the entity type as the output. Do not include any extra information or explanations.

**Entity Types:**
ORGANIZATION
COMPANY
CITY
STATE
COUNTRY
PERSON
YEAR
MONTH
DAY
EVENT
QUANTITY
OTHER

**Reference Text:**
{text_context}

**Input:**
{entity}

**Output:**
"""


class Text2Graph:
    def __init__(
        self,
        type: str,
        file_path: str,
        graph_name: str,
        llm_name: str,
        api_key: str,
        chunk_size: int = 512,
        base_url: str = None,
        thread_count: int = 1,
    ):
        """
        初始化文本到图转换器

        Args:
            file_path: 输入的文本文件路径
        """
        self.file_path = file_path
        self.graph_name = graph_name
        self.thread_count = thread_count
        self.chunk_size = chunk_size
        self.provider = type
        self.model_name = llm_name

        if not file_exist(self.file_path):
            raise FileNotFoundError(f"文本文件未找到: {self.file_path}")

        # sentences = re.split(r'(?<=[。！？\n])', self.read_file_path().strip())  # 按句子切分
        markdown_text = self.read_file_path_markitdown()
        self.markdown_text_chars = len(markdown_text)
        self.markdown_text_bytes = len(markdown_text.encode("utf-8"))
        self.source_file_size_bytes = (
            os.path.getsize(self.file_path) if os.path.exists(self.file_path) else None
        )
        self.markdown_file_size_bytes = (
            os.path.getsize(self.md_file_path)
            if os.path.exists(self.md_file_path)
            else None
        )
        sentences = re.split(r"(?<=[。！？\n])", markdown_text.strip())  # 按句子切分
        self.text_chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                self.text_chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            self.text_chunks.append(current_chunk)

        print(
            f"文本已加载，共 {len(self.text_chunks)} 个块, 每块约 {chunk_size} 字符。"
        )
        if type == "ollama":
            self.llm = OllamaEnv(llm_mode_name=llm_name)
        elif type == "openai":
            self.llm = OpenAIEnv(
                api_key=api_key, model_name=llm_name, base_url=base_url
            )
            print(f"使用 OpenAI 模型: {llm_name}")
        else:
            raise ValueError(f"不支持的类型: {type}")

    @staticmethod
    def _empty_parse_metrics() -> Dict[str, Any]:
        return {
            "llm_elapsed_ms": 0,
            "llm_call_count": 0,
            "llm_success_count": 0,
            "llm_fail_count": 0,
            "retry_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "entity_type_infer_calls": 0,
        }

    @staticmethod
    def _merge_parse_metrics(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k in (
            "llm_elapsed_ms",
            "llm_call_count",
            "llm_success_count",
            "llm_fail_count",
            "retry_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "entity_type_infer_calls",
        ):
            dst[k] = (dst.get(k, 0) or 0) + (src.get(k, 0) or 0)

    def _invoke_llm_with_metrics(
        self, prompt: str
    ) -> Tuple[str, Dict[str, Optional[int]], int]:
        """
        统一执行一次 LLM 调用并返回文本、token usage 与耗时（毫秒）。
        OpenAI 模型可拿到 usage；其他模型 usage 为空值。
        """
        t0 = time.perf_counter()
        usage: Dict[str, Optional[int]] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        text = ""
        if hasattr(self.llm, "client") and hasattr(self.llm, "model"):
            resp = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (
                (resp.choices[0].message.content or "")
                if getattr(resp, "choices", None)
                else ""
            )
            u = getattr(resp, "usage", None)
            if u is not None:
                usage["prompt_tokens"] = getattr(u, "prompt_tokens", None)
                usage["completion_tokens"] = getattr(u, "completion_tokens", None)
                usage["total_tokens"] = getattr(u, "total_tokens", None)
        else:
            raw = self.llm.generate_response(query=prompt)
            text = _llm_completion_text(raw)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return text, usage, elapsed_ms

    def _generate_with_metrics(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        统一执行一次 LLM 调用并返回文本与用量指标。
        - OpenAI: 直接走 client.chat.completions，获取 usage token。
        - 其他提供方: 回退 generate_response，仅记录耗时。
        """
        t0 = time.perf_counter()
        usage_obj = None
        if self.provider == "openai" and hasattr(self.llm, "client"):
            resp = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
            )
            usage_obj = getattr(resp, "usage", None)
            text = (
                (resp.choices[0].message.content or "")
                if getattr(resp, "choices", None)
                else ""
            )
        else:
            raw = self.llm.generate_response(query=prompt)
            text = _llm_completion_text(raw)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        metric = {
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None)
            if usage_obj is not None
            else None,
            "completion_tokens": getattr(usage_obj, "completion_tokens", None)
            if usage_obj is not None
            else None,
            "total_tokens": getattr(usage_obj, "total_tokens", None)
            if usage_obj is not None
            else None,
        }
        return text, metric

    def read_file_path(self) -> str:
        import os

        ext = os.path.splitext(self.file_path)[1].lower()
        text = ""

        if ext == ".txt":
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()

        elif ext == ".docx":
            from docx import Document

            doc = Document(self.file_path)
            text = "\n".join([p.text for p in doc.paragraphs])

        elif ext == ".doc":
            import mammoth

            with open(self.file_path, "rb") as f:
                result = mammoth.extract_raw_text(f)
                text = result.value

        elif ext == ".pdf":
            import fitz  # PyMuPDF

            # 打开 PDF
            doc = fitz.open(self.file_path)
            # 提取每页文本
            text_list = []
            for page in doc:
                page_text = page.get_text("text")  # 按文本顺序提取
                if page_text:  # 有文字才保留
                    text_list.append(page_text)
            # 拼接所有页
            text = "\n".join(text_list)
            # 清理多余空白和换行，保证句子连续
            text = re.sub(r"\r\n|\r", "\n", text)  # 统一换行
            text = re.sub(r"\n+", "\n", text)  # 多个换行合并为一个
            text = re.sub(r"[ \t]+", " ", text)  # 多空格缩成一个空格
            text = text.strip()  # 去掉首尾空白

        else:
            raise ValueError(f"不支持的文件类型: {ext}")

        return text

    def read_file_path_markitdown(self) -> str:
        """
        将用户上传的文件转为 Markdown，并在 schema YAML 中追加一条 text dataset。
        """
        import os

        base_path, _ = os.path.splitext(self.file_path)
        self.md_file_path = f"{base_path}.md"

        # 1️⃣ 转成 Markdown
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(self.file_path)
        markdown_text = result.text_content

        # 2️⃣ 保存 Markdown 文件
        with open(self.md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        return markdown_text

    def extract_graph(self, MAX_RETRIES=5):
        """
        从文本数据中提取知识图谱

        Returns:
            三元组列表
        """
        print("开始从文本中提取知识图谱...")

        triplets = []
        entities = []

        for idx, chunk in enumerate(tqdm(self.text_chunks)):
            for attempt in range(1, MAX_RETRIES + 1):
                raw = self.llm.generate_response(
                    query=prompt_template_str.format(context=chunk)
                )
                text = _llm_completion_text(raw)
                if not text:
                    print(f"⚠️ 块 {idx} 第 {attempt} 次尝试未获得响应")
                    continue
                # 尝试解析 JSON
                response_parsed = extract_json_from_response(text)
                if isinstance(response_parsed, str):
                    print(f"⚠️ 块 {idx} 第 {attempt} 次响应解析失败")
                    continue

                # 成功获取合法响应，跳出重试循环
                response = response_parsed
                break
            else:
                # 5 次都不成功，跳过该块
                print(f"❌ 块 {idx} 超过 {MAX_RETRIES} 次尝试仍失败，跳过")
                continue

            # 注释掉是应为提取的实体和三元组不一样，有可能对应不上，我们只用提取到的三元组就OK了
            # entity_label = {}
            # if isinstance(response["entities"], List):
            #     for entities in response["entities"]:
            #         for entity_type, names in entities.items():
            #             for name in names:
            #                 if not isinstance(name, str) or not isinstance(
            #                     entity_type, str
            #                 ):
            #                     continue
            #                 entity_label[name.capitalize()] = entity_type.capitalize()
            # else:
            #     assert isinstance(response["entities"], Dict)
            #     for entity_type, names in response["entities"].items():
            #         for name in names:
            #             if not isinstance(name, str) or not isinstance(entity_type, str):
            #                 continue
            #             entity_label[name.capitalize()] = entity_type.capitalize()

            new_triplets = []

            for triplet in response.get("triplets", []):
                # 1️⃣ 必须是列表/元组且长度为3
                if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                    # print(f"⚠️ 无效三元组，跳过: {triplet}")
                    continue

                # 2️⃣ 遍历每个元素，处理字符串大小写
                processed_triplet = []
                for phrase in triplet:
                    if isinstance(phrase, str) and phrase.strip():
                        processed_triplet.append(phrase.strip().capitalize())
                    else:
                        processed_triplet.append(phrase)  # 保留原样（可能是数字或空）

                new_triplets.append(processed_triplet)

            triplets.extend(new_triplets)

        # print(f"提取完成，共获得 {len(triplets)} 个三元组。")
        return triplets

    def save_triplet(self, triplets: List[List[str]]):
        import os, csv

        dir_path = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        entities_csv_path = os.path.join(dir_path, f"{base_name}_accounts.csv")
        triplets_csv_path = os.path.join(dir_path, f"{base_name}_transactions.csv")

        valid_triplets = []
        for t in triplets:
            if len(t) != 3:
                # print(f"⚠️ 无效三元组，跳过: {t}")
                continue
            head, rel, tail = t
            if not head or not tail or not rel:
                # print(f"⚠️ 三元组元素为空，跳过: {t}")
                continue
            valid_triplets.append([head, rel, tail])

        if not valid_triplets:
            # print("⚠️ 没有合法三元组，退出。")
            return

        entities = {}
        next_id = 1
        for head, rel, tail in valid_triplets:
            for ent in [head, tail]:
                if ent not in entities:
                    entities[ent] = next_id
                    next_id += 1

        with open(entities_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["acct_id", "dsply_nm"])
            for ent, eid in entities.items():
                writer.writerow([eid, ent])

        with open(triplets_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tran_id", "orig_acct", "bene_acct"])
            for head, rel, tail in valid_triplets:
                writer.writerow([entities[head], entities[tail], rel])

        # print(f"✅ 实体 CSV 保存到: {entities_csv_path}")
        # print(f"✅ 边 CSV 保存到: {triplets_csv_path}")

        schema_path = os.path.join(dir_path, f"{base_name}_graph_schemas.yaml")
        graph_name = f"{base_name}_Graph"

        schema_dict = {
            "datasets": [
                {
                    "name": graph_name,
                    "description": f"{graph_name} graph generated from {self.file_path}",
                    "type": "graph",
                    "schema": {
                        "vertex": [
                            {
                                "attribute_fields": [],
                                "format": "csv",
                                "id_field": "acct_id",
                                "label_field": "dsply_nm",
                                "path": entities_csv_path,
                                "type": "account",
                            }
                        ],
                        "edge": [
                            {
                                "attribute_fields": [],
                                "format": "csv",
                                "label_field": "bene_acct",
                                "path": triplets_csv_path,
                                "source_field": "tran_id",
                                "target_field": "orig_acct",
                                "type": "transfer",
                                "weight_field": None,
                            }
                        ],
                        "graph": {
                            "directed": "true",
                            "heterogeneous": "false",
                            "multigraph": "false",
                            "weighted": "false",
                        },
                        "graph_store_info": {
                            "backend": "nebula_graph",
                            "edge_count": len(valid_triplets),
                            "space_name": graph_name,
                            "status": "success",
                            "version": "null",
                            "vertex_count": len(entities),
                        },
                    },
                }
            ]
        }
        import yaml

        with open(schema_path, "w", encoding="utf-8") as f:
            yaml.dump(schema_dict, f, sort_keys=False, allow_unicode=True)

        #  print(f"✅ Graph schema YAML: {schema_path}")

    def extract_graph_by_openie(self):
        print("开始使用OpenIE从文本中提取知识图谱...")
        try:
            from openie import StanfordOpenIE
        except ImportError:
            import subprocess, sys

            package_name = "stanford-openie"
            # print(f"⚙️ 检测到未安装依赖 '{package_name}'，正在自动安装...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package_name]
                )
                # print(f"✅ 成功安装 {package_name}")
                from openie import StanfordOpenIE  # 再次导入
            except subprocess.CalledProcessError as e:
                # print(f"❌ 安装 {package_name} 失败，请手动安装：pip install {package_name}")
                raise e

        triplets = []
        for idx, chunk in enumerate(tqdm(self.text_chunks)):
            properties = {
                "openie.affinity_probability_cap": 2 / 3,
            }
            with StanfordOpenIE(properties=properties) as client:
                for triple in client.annotate(chunk):
                    head = triple["subject"].strip().capitalize()
                    relation = triple["relation"].strip().capitalize()
                    tail = triple["object"].strip().capitalize()
                    triplet = [head, relation, tail]
                    triplets.append(triplet)
        # print(f"提取完成，共获得 {len(triplets)} 个三元组。")
        print(triplets)
        return triplets

    def extract_graph_and_entity_by_LLM(
        self, each_dataset, file_name, each_dataset_schema_file_path, MAX_RETRIES=5
    ):
        """
        从文本数据中提取三元组与实体类型

        Returns:
            triplets: 合法三元组列表
            entity2id: 实体 -> ID
            entity2type: 实体 -> 类型
        """
        print(f"开始使用 LLM 从文本中提取知识图谱...")
        t_total = time.perf_counter()
        triplets: List[List[str]] = []
        entity2type: Dict[str, str] = {}
        total_calls = 0
        total_retries = 0
        total_llm_elapsed_ms = 0
        prompt_tokens_sum = 0
        completion_tokens_sum = 0
        total_tokens_sum = 0
        token_observed = False

        for idx, chunk in enumerate(tqdm(self.text_chunks)):
            for attempt in range(1, MAX_RETRIES + 1):
                total_calls += 1
                response, call_metric = self._generate_with_metrics(
                    prompt_template_str.format(context=chunk)
                )
                total_llm_elapsed_ms += call_metric.get("elapsed_ms", 0) or 0
                pt = call_metric.get("prompt_tokens")
                ct = call_metric.get("completion_tokens")
                tt = call_metric.get("total_tokens")
                if isinstance(pt, int):
                    prompt_tokens_sum += pt
                    token_observed = True
                if isinstance(ct, int):
                    completion_tokens_sum += ct
                    token_observed = True
                if isinstance(tt, int):
                    total_tokens_sum += tt
                    token_observed = True
                if not response:
                    # print(f"⚠️ 块 {idx} 第 {attempt} 次无响应")
                    if attempt > 1:
                        total_retries += 1
                    continue

                cleaned = response.strip()

                # remove markdown mark
                cleaned = re.sub(
                    r"^```[a-zA-Z]*\n?|```$", "", cleaned, flags=re.MULTILINE
                ).strip()

                # match {} content
                match = re.search(r"\{[\s\S]*\}", cleaned)
                if not match:
                    # print(f"❌ 未找到 JSON 内容，原始响应:\n{cleaned}")
                    if attempt > 1:
                        total_retries += 1
                    continue

                json_str = match.group(0)

                try:
                    response_parsed = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # JSON 解析失败，记录日志并跳过当前条目
                    # print(f"[Warning] JSON parsing failed at index {idx} of {attempt}: {e}")
                    if attempt > 1:
                        total_retries += 1
                    continue

                if not isinstance(response_parsed, dict):
                    # print(f"⚠️ 块 {idx} 第 {attempt} 次 JSON 解析失败")
                    if attempt > 1:
                        total_retries += 1
                    continue
                response = response_parsed
                break
            else:
                # print(f"❌ 块 {idx} 超过 {MAX_RETRIES} 次尝试失败，跳过")
                continue

            # ======================
            # 1️⃣ 解析 entities 部分
            # ======================
            entities_from_response = {}

            if isinstance(response.get("entities"), dict):
                for entity_type, names in response["entities"].items():
                    for name in names:
                        if isinstance(name, str) and name.strip():
                            entities_from_response[name.strip()] = (
                                entity_type.strip().capitalize()
                            )

            # ======================
            # 2️⃣ 解析 triplets 部分
            # ======================
            new_triplets = []
            for triplet in response.get("triplets", []):
                if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                    # print(f"⚠️ 无效三元组，跳过: {triplet}")
                    continue
                # 去掉空字符串，只保留非空字符串和数字/其他类型
                clean_triplet = []
                for x in triplet:
                    if isinstance(x, str):
                        x_clean = x.strip()
                        if x_clean:  # 非空字符串才保留
                            clean_triplet.append(x_clean.capitalize())
                    else:
                        if x is not None:
                            clean_triplet.append(str(x))

                # 长度必须为3才保留
                if len(clean_triplet) == 3:
                    new_triplets.append(clean_triplet)
                else:
                    pass

            triplets.extend(new_triplets)

            # ======================
            # 3️⃣ 更新实体类型映射
            # ======================
            for head, rel, tail in new_triplets:
                for ent in [head, tail]:
                    if ent not in entity2type:
                        if ent in entities_from_response:
                            entity2type[ent] = entities_from_response[ent]
                        else:
                            # 调用大模型补全实体类型
                            entity_type, type_metric = self._ask_entity_type(
                                ent, chunk, MAX_RETRIES
                            )
                            total_calls += type_metric.get("call_count", 0)
                            total_retries += type_metric.get("retry_count", 0)
                            total_llm_elapsed_ms += (
                                type_metric.get("llm_elapsed_ms", 0) or 0
                            )
                            tpt = type_metric.get("prompt_tokens")
                            tct = type_metric.get("completion_tokens")
                            ttt = type_metric.get("total_tokens")
                            if isinstance(tpt, int):
                                prompt_tokens_sum += tpt
                                token_observed = True
                            if isinstance(tct, int):
                                completion_tokens_sum += tct
                                token_observed = True
                            if isinstance(ttt, int):
                                total_tokens_sum += ttt
                                token_observed = True
                            entity2type[ent] = entity_type

            each_dataset[file_name]["parsing_rate"] = (idx + 1) / len(self.text_chunks)
            import yaml

            output_file = []
            for key in each_dataset:
                output_file.append(each_dataset[key])
            final_schema = {"datasets": output_file}
            with open(each_dataset_schema_file_path, "w", encoding="utf-8") as f:
                yaml.dump(final_schema, f, sort_keys=False, allow_unicode=True)
            f.close()
        # ======================
        # 4️⃣ 生成实体-ID 映射
        # ======================
        entity2id = {ent: i + 1 for i, ent in enumerate(entity2type.keys())}

        #  print(f"✅ 提取完成，共 {len(triplets)} 个三元组，{len(entity2id)} 个实体")
        elapsed_ms = int((time.perf_counter() - t_total) * 1000)
        text_bytes = (
            os.path.getsize(self.file_path) if os.path.exists(self.file_path) else None
        )
        text_chars = sum(len(c) for c in self.text_chunks)
        parse_metrics = {
            "provider": self.provider,
            "model": self.model_name,
            "chunk_count": len(self.text_chunks),
            "llm_call_count": total_calls,
            "llm_elapsed_ms": total_llm_elapsed_ms,
            "retry_count": total_retries,
            "elapsed_ms": elapsed_ms,
            "text_bytes": text_bytes,
            "text_chars": text_chars,
            "markdown_text_bytes": self.markdown_text_bytes,
            "markdown_text_chars": self.markdown_text_chars,
            "markdown_file_size_bytes": self.markdown_file_size_bytes,
            "prompt_tokens": prompt_tokens_sum if token_observed else None,
            "completion_tokens": completion_tokens_sum if token_observed else None,
            "total_tokens": total_tokens_sum if token_observed else None,
        }
        return triplets, entity2id, entity2type, parse_metrics

    def extract_graph_and_entity_by_LLM_with_mutiple(
        self, each_dataset, file_name, each_dataset_schema_file_path, MAX_RETRIES=5
    ):
        """
        从文本数据中提取三元组与实体类型（多线程）。

        Returns:
            triplets: 合法三元组列表
            entity2id: 实体 -> ID
            entity2type: 实体 -> 类型
            parse_metrics: 聚合解析指标
        """
        import yaml

        t_total = time.perf_counter()
        print(
            f"开始使用 LLM 从文本中提取知识图谱（多线程，最多 {self.thread_count} 个 chunk 并行）..."
        )
        n_chunks = len(self.text_chunks)
        if n_chunks == 0:
            return (
                [],
                {},
                {},
                {
                    "provider": self.provider,
                    "model": self.model_name,
                    "chunk_count": 0,
                    "llm_call_count": 0,
                    "llm_elapsed_ms": 0,
                    "retry_count": 0,
                    "elapsed_ms": 0,
                    "text_bytes": self.source_file_size_bytes,
                    "text_chars": 0,
                    "markdown_text_bytes": self.markdown_text_bytes,
                    "markdown_text_chars": self.markdown_text_chars,
                    "markdown_file_size_bytes": self.markdown_file_size_bytes,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            )

        def process_one_chunk(
            idx: int, chunk: str
        ) -> Tuple[int, List[List[str]], Dict[str, str], Dict[str, Any]]:
            new_triplets: List[List[str]] = []
            entity2type_local: Dict[str, str] = {}
            local_calls = 0
            local_retries = 0
            local_elapsed_ms = 0
            local_prompt_tokens = 0
            local_completion_tokens = 0
            local_total_tokens = 0
            local_token_observed = False

            for attempt in range(1, MAX_RETRIES + 1):
                local_calls += 1
                response, call_metric = self._generate_with_metrics(
                    prompt_template_str.format(context=chunk)
                )
                local_elapsed_ms += call_metric.get("elapsed_ms", 0) or 0
                pt = call_metric.get("prompt_tokens")
                ct = call_metric.get("completion_tokens")
                tt = call_metric.get("total_tokens")
                if isinstance(pt, int):
                    local_prompt_tokens += pt
                    local_token_observed = True
                if isinstance(ct, int):
                    local_completion_tokens += ct
                    local_token_observed = True
                if isinstance(tt, int):
                    local_total_tokens += tt
                    local_token_observed = True

                if not response:
                    if attempt > 1:
                        local_retries += 1
                    continue

                cleaned = response.strip()
                cleaned = re.sub(
                    r"^```[a-zA-Z]*\n?|```$", "", cleaned, flags=re.MULTILINE
                ).strip()
                match = re.search(r"\{[\s\S]*\}", cleaned)
                if not match:
                    if attempt > 1:
                        local_retries += 1
                    continue

                json_str = match.group(0)
                try:
                    response_parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    if attempt > 1:
                        local_retries += 1
                    continue

                if not isinstance(response_parsed, dict):
                    if attempt > 1:
                        local_retries += 1
                    continue
                response = response_parsed
                break
            else:
                return (
                    idx,
                    [],
                    {},
                    {
                        "llm_call_count": local_calls,
                        "llm_elapsed_ms": local_elapsed_ms,
                        "retry_count": local_retries,
                        "prompt_tokens": local_prompt_tokens
                        if local_token_observed
                        else None,
                        "completion_tokens": local_completion_tokens
                        if local_token_observed
                        else None,
                        "total_tokens": local_total_tokens
                        if local_token_observed
                        else None,
                    },
                )

            entities_from_response = {}
            if isinstance(response.get("entities"), dict):
                for entity_type, names in response["entities"].items():
                    for name in names:
                        if isinstance(name, str) and name.strip():
                            entities_from_response[name.strip()] = (
                                entity_type.strip().capitalize()
                            )

            for triplet in response.get("triplets", []):
                if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                    continue
                clean_triplet = []
                for x in triplet:
                    if isinstance(x, str):
                        x_clean = x.strip()
                        if x_clean:
                            clean_triplet.append(x_clean.capitalize())
                    else:
                        if x is not None:
                            clean_triplet.append(str(x))
                if len(clean_triplet) == 3:
                    new_triplets.append(clean_triplet)

            for head, _, tail in new_triplets:
                for ent in [head, tail]:
                    if ent not in entity2type_local:
                        if ent in entities_from_response:
                            entity2type_local[ent] = entities_from_response[ent]
                        else:
                            entity_type, type_metric = self._ask_entity_type(
                                ent, chunk, MAX_RETRIES
                            )
                            local_calls += type_metric.get("call_count", 0)
                            local_retries += type_metric.get("retry_count", 0)
                            local_elapsed_ms += (
                                type_metric.get("llm_elapsed_ms", 0) or 0
                            )
                            tpt = type_metric.get("prompt_tokens")
                            tct = type_metric.get("completion_tokens")
                            ttt = type_metric.get("total_tokens")
                            if isinstance(tpt, int):
                                local_prompt_tokens += tpt
                                local_token_observed = True
                            if isinstance(tct, int):
                                local_completion_tokens += tct
                                local_token_observed = True
                            if isinstance(ttt, int):
                                local_total_tokens += ttt
                                local_token_observed = True
                            entity2type_local[ent] = entity_type

            return (
                idx,
                new_triplets,
                entity2type_local,
                {
                    "llm_call_count": local_calls,
                    "llm_elapsed_ms": local_elapsed_ms,
                    "retry_count": local_retries,
                    "prompt_tokens": local_prompt_tokens
                    if local_token_observed
                    else None,
                    "completion_tokens": local_completion_tokens
                    if local_token_observed
                    else None,
                    "total_tokens": local_total_tokens
                    if local_token_observed
                    else None,
                },
            )

        results: Dict[int, Tuple[List[List[str]], Dict[str, str]]] = {}
        yaml_lock = threading.Lock()
        completed = 0
        total_calls = 0
        total_retries = 0
        total_llm_elapsed_ms = 0
        prompt_tokens_sum = 0
        completion_tokens_sum = 0
        total_tokens_sum = 0
        token_observed = False

        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            future_to_idx = {
                executor.submit(process_one_chunk, idx, chunk): idx
                for idx, chunk in enumerate(self.text_chunks)
            }
            for fut in tqdm(
                as_completed(future_to_idx), total=n_chunks, desc="LLM chunks"
            ):
                idx, new_triplets, entity2type_local, metric = fut.result()
                results[idx] = (new_triplets, entity2type_local)

                total_calls += metric.get("llm_call_count", 0)
                total_retries += metric.get("retry_count", 0)
                total_llm_elapsed_ms += metric.get("llm_elapsed_ms", 0) or 0
                pt = metric.get("prompt_tokens")
                ct = metric.get("completion_tokens")
                tt = metric.get("total_tokens")
                if isinstance(pt, int):
                    prompt_tokens_sum += pt
                    token_observed = True
                if isinstance(ct, int):
                    completion_tokens_sum += ct
                    token_observed = True
                if isinstance(tt, int):
                    total_tokens_sum += tt
                    token_observed = True

                with yaml_lock:
                    completed += 1
                    each_dataset[file_name]["parsing_rate"] = completed / n_chunks
                    output_file = [each_dataset[k] for k in each_dataset]
                    final_schema = {"datasets": output_file}
                    with open(
                        each_dataset_schema_file_path, "w", encoding="utf-8"
                    ) as f:
                        yaml.dump(final_schema, f, sort_keys=False, allow_unicode=True)

        triplets: List[List[str]] = []
        entity2type: Dict[str, str] = {}
        for idx in range(n_chunks):
            if idx not in results:
                continue
            new_triplets, entity2type_local = results[idx]
            triplets.extend(new_triplets)
            for ent, typ in entity2type_local.items():
                if ent not in entity2type:
                    entity2type[ent] = typ

        entity2id = {ent: i + 1 for i, ent in enumerate(entity2type.keys())}
        elapsed_ms = int((time.perf_counter() - t_total) * 1000)
        text_bytes = self.source_file_size_bytes
        text_chars = sum(len(c) for c in self.text_chunks)
        parse_metrics = {
            "provider": self.provider,
            "model": self.model_name,
            "chunk_count": len(self.text_chunks),
            "llm_call_count": total_calls,
            "llm_elapsed_ms": total_llm_elapsed_ms,
            "retry_count": total_retries,
            "elapsed_ms": elapsed_ms,
            "text_bytes": text_bytes,
            "text_chars": text_chars,
            "markdown_text_bytes": self.markdown_text_bytes,
            "markdown_text_chars": self.markdown_text_chars,
            "markdown_file_size_bytes": self.markdown_file_size_bytes,
            "prompt_tokens": prompt_tokens_sum if token_observed else None,
            "completion_tokens": completion_tokens_sum if token_observed else None,
            "total_tokens": total_tokens_sum if token_observed else None,
        }
        return triplets, entity2id, entity2type, parse_metrics

    def _ask_entity_type(
        self, entity: str, chunk: str, MAX_RETRIES=5
    ) -> Tuple[str, Dict[str, Any]]:
        """向 LLM 查询实体类型，并返回该步骤的调用指标。"""
        prompt = prompt_entity_type.format(entity=entity, text_context=chunk)
        call_count = 0
        retry_count = 0
        llm_elapsed_ms = 0
        prompt_tokens_sum = 0
        completion_tokens_sum = 0
        total_tokens_sum = 0
        token_observed = False

        for attempt in range(1, MAX_RETRIES + 1):
            call_count += 1
            response, call_metric = self._generate_with_metrics(prompt)
            llm_elapsed_ms += call_metric.get("elapsed_ms", 0) or 0
            pt = call_metric.get("prompt_tokens")
            ct = call_metric.get("completion_tokens")
            tt = call_metric.get("total_tokens")
            if isinstance(pt, int):
                prompt_tokens_sum += pt
                token_observed = True
            if isinstance(ct, int):
                completion_tokens_sum += ct
                token_observed = True
            if isinstance(tt, int):
                total_tokens_sum += tt
                token_observed = True
            if response and isinstance(response, str):
                metric = {
                    "call_count": call_count,
                    "retry_count": retry_count,
                    "llm_elapsed_ms": llm_elapsed_ms,
                    "prompt_tokens": prompt_tokens_sum if token_observed else None,
                    "completion_tokens": completion_tokens_sum
                    if token_observed
                    else None,
                    "total_tokens": total_tokens_sum if token_observed else None,
                }
                return response.strip().capitalize(), metric
            if attempt > 1:
                retry_count += 1

        metric = {
            "call_count": call_count,
            "retry_count": retry_count,
            "llm_elapsed_ms": llm_elapsed_ms,
            "prompt_tokens": prompt_tokens_sum if token_observed else None,
            "completion_tokens": completion_tokens_sum if token_observed else None,
            "total_tokens": total_tokens_sum if token_observed else None,
        }
        return "Unknown", metric

    def save_graph_with_entity(
        self,
        triplets: List[List[str]],
        entity2id: Dict[str, int],
        entity2type: Dict[str, str],
        graph_schema_file: str,
    ) -> Dict[str, Any]:
        """
        保存实体/三元组到 CSV，并合并写入 graph_schemas.yaml。
        返回新生成的 schema 字典（含 create_time）。
        """
        import csv
        import yaml

        process_dir = os.path.dirname(self.file_path)
        os.makedirs(process_dir, exist_ok=True)

        entities_csv_path = os.path.join(process_dir, f"{self.graph_name}_accounts.csv")
        triplets_csv_path = os.path.join(
            process_dir, f"{self.graph_name}_transactions.csv"
        )

        # 过滤非法三元组，并确保端点实体在 entity2id 中有映射
        valid_triplets: List[List[str]] = []
        for t in triplets:
            if not isinstance(t, (list, tuple)) or len(t) != 3:
                continue
            head, rel, tail = t
            if not head or not rel or not tail:
                continue
            head_s = str(head).strip()
            rel_s = str(rel).strip()
            tail_s = str(tail).strip()
            if not head_s or not rel_s or not tail_s:
                continue
            valid_triplets.append([head_s, rel_s, tail_s])

        if not entity2id:
            # 兜底：从三元组重建实体 ID
            entity2id = {}
            for head, _, tail in valid_triplets:
                if head not in entity2id:
                    entity2id[head] = len(entity2id) + 1
                if tail not in entity2id:
                    entity2id[tail] = len(entity2id) + 1
        else:
            # 如果三元组里出现新实体，补齐 ID 映射，避免 KeyError
            for head, _, tail in valid_triplets:
                if head not in entity2id:
                    entity2id[head] = len(entity2id) + 1
                if tail not in entity2id:
                    entity2id[tail] = len(entity2id) + 1

        with open(entities_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["acct_id", "dsply_nm", "type"])
            for entity, eid in sorted(entity2id.items(), key=lambda x: x[1]):
                writer.writerow([eid, entity, entity2type.get(entity, "OTHER")])

        with open(triplets_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tran_id", "orig_acct", "bene_acct", "predicate"])
            for idx, (head, rel, tail) in enumerate(valid_triplets, start=1):
                writer.writerow([idx, entity2id[head], entity2id[tail], rel])

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_schema = {
            "description": f"{self.graph_name} graph generated from {self.file_path}",
            "name": self.graph_name,
            "type": "graph",
            "graph_status": "completed",
            "create_time": now_str,
            "schema": {
                "vertex": [
                    {
                        "attribute_fields": ["type"],
                        "format": "csv",
                        "id_field": "acct_id",
                        "label_field": "dsply_nm",
                        "path": entities_csv_path,
                        "original_path": self.file_path,
                        "type": "account",
                    }
                ],
                "edge": [
                    {
                        "attribute_fields": [],
                        "format": "csv",
                        "label_field": "predicate",
                        "path": triplets_csv_path,
                        "original_path": self.file_path,
                        "source_field": "orig_acct",
                        "target_field": "bene_acct",
                        "type": "transfer",
                        "weight_field": None,
                    }
                ],
                "graph": {
                    "directed": "true",
                    "heterogeneous": "false",
                    "multigraph": "false",
                    "weighted": "false",
                },
                "graph_store_info": {
                    "backend": "nebula_graph",
                    "edge_count": len(valid_triplets),
                    "space_name": self.graph_name,
                    "status": "success",
                    "version": "null",
                    "vertex_count": len(entity2id),
                },
            },
        }

        existing = []
        if os.path.exists(graph_schema_file):
            with open(graph_schema_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                datasets = loaded.get("datasets")
                if isinstance(datasets, list):
                    existing = [d for d in datasets if d.get("name") != self.graph_name]

        existing.append(new_schema)
        with open(graph_schema_file, "w", encoding="utf-8") as f:
            yaml.dump({"datasets": existing}, f, sort_keys=False, allow_unicode=True)

        return new_schema
