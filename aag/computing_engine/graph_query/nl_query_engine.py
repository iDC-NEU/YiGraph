# /home/gaojq/AAG_duan/AAG/aag/computing_engine/graph_query/nl_query_engine.py

import json
import re
import os
import sys

# add gjq
# Add project root directory to Python path to enable importing aag module
# Get current file directory, then navigate up to find AAG directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# add gjq
from openai import OpenAI
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# 类型别名
JsonDict = Dict[str, Any]

# add gjq
# Import Reasoner for LLM calls
from aag.reasoner.model_deployment import Reasoner
from aag.config.engine_config import ReasonerConfig

# ===== 导入你的模板模块 =====
from aag.computing_engine.graph_query.graph_query import Neo4jGraphClient, Neo4jConfig
#from graph_query import Neo4jGraphClient, Neo4jConfig

# add gjq
# Removed direct OpenAI client initialization
# os.environ['OPENAI_API_KEY'] = 'sk-G30rFStBigqXtuyIOkOo7Zh4QNxO8ZAjfZQ5DYPCgMXbPv8q'
# os.environ['OPENAI_BASE_URL'] = 'https://gitaigc.com/v1/'
# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     base_url=os.environ.get("OPENAI_BASE_URL")
# )

# ==========================================
# 1. Basic Query Types (Core Query Logic)
# ==========================================
QUERY_TEMPLATES = {
    "node_lookup": {
        "description": "Find a single node by unique key",
        "method": "get_node_by_unique_key",
        "required_params": ["label", "key", "value"],
        "optional_params": [],
        "example": "Find user Collins Steven"
    },
    "neighbor_query": {
        "description": "Query neighbors or N-hop relationships of a node",
        "method": "neighbors_n_hop",
        "required_params": ["label", "key", "value"],
        "optional_params": ["hops", "rel_type", "direction"],
        "example": "Query neighbors of Collins Steven"
    },
    "path_query": {
        "description": "Query paths between two nodes",
        "method": "paths_between",
        "required_params": ["label", "key", "v1", "v2"],
        "optional_params": ["rel_type", "direction", "min_hops", "max_hops"],
        "example": "Path from Collins Steven to Nunez Mitchell"
    },
    "common_neighbor": {
        "description": "Query common neighbors of two nodes",
        "method": "common_neighbors",
        "required_params": ["label", "key", "v1", "v2"],
        "optional_params": ["rel_type", "direction"],
        "example": "Common neighbors of Collins Steven and Nunez Mitchell"
    },
    "subgraph": {
        "description": "Extract subgraph (single center node)",
        "method": "subgraph_extract",
        "required_params": ["label", "key", "value"],
        "optional_params": ["hops", "rel_type", "direction", "limit_paths"],
        "example": "Subgraph within 2 hops around Collins Steven"
    },
    "subgraph_by_nodes": {
        "description": "Extract subgraph based on node list (multiple specified nodes and their mutual relationships)",
        "method": "subgraph_extract_by_nodes",
        "required_params": ["label", "key", "values"],
        "optional_params": ["include_internal", "rel_type", "direction"],
        "example": "Extract accounts A, B, C and their transfer relationships"
    }
}

# ==========================================
# 2. General Modifiers (Applicable to any query)
# ==========================================
QUERY_MODIFIERS = {
    "order_by": {
        "description": "Sort field (relationship property or node property)",
        "format": "firstRel.property_name or nbr.property_name",
        "example": "firstRel.base_amt, nbr.name"
    },
    "order_direction": {
        "description": "Sort direction",
        "values": ["ASC", "DESC"],
        "default": "ASC"
    },
    "limit": {
        "description": "Limit on number of results returned",
        "type": "integer",
        "example": 10
    },
    "where": {
        "description": "Filter condition (node or relationship property)",
        "format": "n.property operator value or r.property operator value",
        "example": "n.balance > 1000, r.amount > 500"
    },
    "aggregate": {
        "description": "Aggregation function",
        "values": ["COUNT", "SUM", "AVG", "MAX", "MIN"],
        "note": "Returns statistical results instead of raw data after applying aggregation"
    },
    "aggregate_field": {
        "description": "Aggregation field (required only when aggregate is not COUNT)",
        "format": "rel.property_name or neighbor.property_name",
        "example": "rel.base_amt"
    }
}

# Standard JSON output template
STANDARD_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["params", "modifiers"],
    "properties": {
        "params": {
            "type": "object",
            "description": "Query-specific required and optional parameters"
        },
        "modifiers": {
            "type": "object",
            "description": "General modifiers to use (only include those needed)",
            "properties": {
                "order_by": {"type": "string", "description": "Sort field"},
                "order_direction": {"type": "string", "enum": ["ASC", "DESC"]},
                "limit": {"type": "integer", "description": "Result count limit"},
                "where": {"type": "string", "description": "Filter condition"},
                "aggregate": {"type": "string", "enum": ["count", "sum", "avg", "max", "min"]},
                "aggregate_field": {"type": "string", "description": "Aggregation field"}
            }
        }
    }
}

# ==========================================
# 2. LLM 接口抽象（你需要替换成真实实现）
# ==========================================
# add gjq
class LLMInterface:
    """LLM 调用接口（使用 Reasoner 统一调用）"""
    
    def __init__(self, reasoner: Reasoner):
        """
        初始化 LLM 接口
        
        Args:
            reasoner: Reasoner 实例，用于统一调用不同的 LLM
        """
        self.reasoner = reasoner
    
    def call(self, prompt: str, **kwargs) -> str:
        """
        调用 LLM，返回文本响应
        
        使用 Reasoner 的 generate_response 方法
        """
        response = self.reasoner.generate_response(prompt)
        if hasattr(response, 'text'):
            return response.text
        return str(response)


# ==========================================
# 3. LLM1：查询类型分类器
# ==========================================
# add gjq
class QueryTypeClassifier:
    """使用 LLM 判断查询类型"""
    
    def __init__(self, reasoner: Reasoner):
        """
        初始化查询类型分类器
        
        Args:
            reasoner: Reasoner 实例，用于调用 LLM
        """
        self.reasoner = reasoner
    
    def classify(self, question: str) -> str:
        """
        返回查询类型（如 "node_lookup"）
        """
        # add gjq
        # 使用 Reasoner 的 nl_query_classify_type 方法
        query_type = self.reasoner.nl_query_classify_type(question, QUERY_TEMPLATES)
        
        logging.info(f"LLM 分类响应: {query_type}")
        
        if query_type not in QUERY_TEMPLATES:
            # 降级到规则匹配
            logging.warning(f"未知查询类型: {query_type}，使用规则匹配降级")
            query_type = self._fallback_classify(question)
        
        return query_type
    
    def _fallback_classify(self, question: str) -> str:
        """规则匹配降级方案"""
        q = question.lower()
        
        # 优先级从高到低
        if "公共" in q or "共同" in q:
            return "common_neighbor"
        if any(kw in q for kw in ["路径", "怎么到", "到达"]):
            return "path_query"
        # 检测多节点子图查询（包含多个节点名称或"之间"关键词）
        if "子图" in q:
            # 检测是否包含多个节点标识（如 "A、B、C" 或 "之间"）
            if any(kw in q for kw in ["、", "和", "及", "之间", "相互", "互相"]):
                return "subgraph_by_nodes"
            return "subgraph"
        if any(kw in q for kw in ["邻居", "朋友", "关注", "关系"]):
            return "neighbor_query"
        if any(kw in q for kw in ["找", "查", "获取"]):
            return "node_lookup"
        
        # 默认使用邻居查询
        return "neighbor_query"


# ==========================================
# 4. Schema 工具
# ==========================================
class SchemaAnalyzer:
    """图 Schema 分析工具"""
    
    def __init__(self, client: Neo4jGraphClient):
        self.client = client
        self._schema_cache = None
    
    def get_schema(self) -> Dict:
        """获取并缓存 Schema 信息"""
        if self._schema_cache is None:
            self._schema_cache = self.client.get_schema()
        return self._schema_cache
    
    def format_for_llm(self) -> str:
        """格式化 Schema 信息给 LLM（增强版）"""
        schema = self.get_schema()
        
        formatted = "## 图数据库 Schema 信息\n\n"
        
        # 节点类型及属性（包含示例值）
        formatted += "### 节点类型及属性\n"
        for label, info in schema["node_labels"].items():
            props = info.get("properties", [])
            samples = info.get("sample_values", {})
            formatted += f"- **{label}**:\n"
            for prop in props:
                sample = samples.get(prop, "")
                sample_str = f" (示例: {sample})" if sample else ""
                formatted += f"  - `{prop}`{sample_str}\n"
        
        # 关系类型及属性（包含示例值）
        formatted += "\n### 关系类型及属性\n"
        for rel_type, info in schema["relationship_types"].items():
            props = info.get("properties", [])
            samples = info.get("sample_values", {})
            if props:
                formatted += f"- **{rel_type}**:\n"
                for prop in props:
                    sample = samples.get(prop, "")
                    sample_str = f" (示例: {sample})" if sample else ""
                    formatted += f"  - `{prop}`{sample_str}\n"
            else:
                formatted += f"- **{rel_type}**: 无属性\n"
        
        # 关系模式
        formatted += "\n### 常见关系模式\n"
        for pattern in schema["patterns"][:10]:
            formatted += f"- {pattern}\n"
        
        return formatted


# ==========================================
# 5. LLM2：参数填充器
# ==========================================
# add gjq
class ParameterExtractor:
    """使用 LLM 根据模板要求和 Schema 填充参数"""
    
    def __init__(self, reasoner: Reasoner, schema_analyzer: SchemaAnalyzer):
        """
        初始化参数提取器
        
        Args:
            reasoner: Reasoner 实例，用于调用 LLM
            schema_analyzer: Schema 分析器
        """
        self.reasoner = reasoner
        self.schema_analyzer = schema_analyzer
    
    def extract(self, question: str, query_type: str) -> Dict:
        """
        返回填充好的参数字典（包含需要使用的修饰符列表）
        """
        template = QUERY_TEMPLATES[query_type]
        schema = self.schema_analyzer.get_schema()
        schema_info = self.schema_analyzer.format_for_llm()
        logging.info(f"Schema 已加载{schema_info}")
        
        # add gjq
        # 使用 Reasoner 的 nl_query_extract_params 方法
        result = self.reasoner.nl_query_extract_params(
            question=question,
            query_type=query_type,
            template=template,
            schema_info=schema_info,
            query_modifiers=QUERY_MODIFIERS
        )
        
        # add gjq
        # Reasoner 已经返回解析好的 JSON，不需要再次解析
        # 解析新的 JSON 格式
        if isinstance(result, dict) and "params" in result and "modifiers" in result:
            # 新格式：{"params": {...}, "modifiers": {...}}
            params = result["params"]
            modifiers = result["modifiers"]
        else:
            # 旧格式兼容或解析失败：降级到规则提取
            logging.warning(f"LLM 返回格式不正确，使用规则提取降级")
            result = self._fallback_extract(question, query_type)
            params = result.get("params", result)
            modifiers = result.get("modifiers", {})
        
        # 验证必需参数
        self._validate_params(params, template)
        
        # ⭐ 核心改进：验证和清理修饰符
        cleaned_modifiers = self._validate_and_clean_modifiers(
            modifiers,
            query_type,
            schema
        )
        
        # 合并参数
        final_params = {**params, **cleaned_modifiers}
        
        logging.info(f"提取的参数: {params}")
        logging.info(f"原始修饰符: {modifiers}")
        logging.info(f"清理后修饰符: {cleaned_modifiers}")
        logging.info(f"最终参数: {final_params}")
        
        return final_params
    def _validate_and_clean_modifiers(
        self, 
        modifiers: Dict, 
        query_type: str,
        schema: Dict
    ) -> Dict:
        """
        验证并清理修饰符（核心功能）
        
        功能：
        1. 移除不适用的修饰符
        2. 验证字段名是否存在于 Schema
        3. 修正格式错误
        4. 添加默认值
        """
        cleaned = {}
        
        # 1. 验证 order_by
        if "order_by" in modifiers:
            order_field = modifiers["order_by"]
            
            # 根据查询类型验证字段格式
            if query_type == "neighbor_query":
                if self._validate_order_field(order_field, ["firstRel", "nbr"], schema):
                    cleaned["order_by"] = order_field
                    cleaned["order_direction"] = modifiers.get("order_direction", "ASC")
                else:
                    logging.warning(f"无效的 order_by 字段: {order_field}，已忽略")
            
            elif query_type == "common_neighbor":
                if self._validate_order_field(order_field, ["rA", "rB", "C"], schema):
                    cleaned["order_by"] = order_field
                    cleaned["order_direction"] = modifiers.get("order_direction", "ASC")
                else:
                    logging.warning(f"无效的 order_by 字段: {order_field}，已忽略")
            
            elif query_type == "path_query":
                if order_field == "hops" or self._validate_order_field(order_field, [], schema):
                    cleaned["order_by"] = order_field
                    cleaned["order_direction"] = modifiers.get("order_direction", "ASC")
                else:
                    logging.warning(f"无效的 order_by 字段: {order_field}，已忽略")
        
        # 2. 验证 limit
        if "limit" in modifiers:
            try:
                limit_val = int(modifiers["limit"])
                if limit_val > 0:
                    cleaned["limit"] = limit_val
                else:
                    logging.warning(f"limit 必须大于 0: {limit_val}，已忽略")
            except (ValueError, TypeError):
                logging.warning(f"无效的 limit 值: {modifiers['limit']}，已忽略")
        
        # 3. 验证 where
        if "where" in modifiers:
            where_clause = modifiers["where"]
            
            if self._validate_where_clause(where_clause, query_type, schema):
                cleaned["where"] = where_clause
            else:
                logging.warning(f"无效的 where 子句: {where_clause}，已忽略")
        
        # 4. 验证聚合
        if "aggregate" in modifiers:
            agg_type = modifiers["aggregate"].lower()
            
            if agg_type in ["count", "sum", "avg", "max", "min"]:
                cleaned["aggregate"] = agg_type
                
                if agg_type != "count":
                    if "aggregate_field" in modifiers:
                        agg_field = modifiers["aggregate_field"]
                        if self._validate_aggregate_field(agg_field, query_type, schema):
                            cleaned["aggregate_field"] = agg_field
                        else:
                            logging.warning(f"无效的聚合字段: {agg_field}，聚合已取消")
                            cleaned.pop("aggregate", None)
                    else:
                        logging.warning(f"聚合类型 {agg_type} 需要指定 aggregate_field，聚合已取消")
            else:
                logging.warning(f"未知的聚合类型: {agg_type}，已忽略")
        
        return cleaned

    def _validate_order_field(
        self, 
        field: str, 
        allowed_prefixes: List[str], 
        schema: Dict
    ) -> bool:
        """验证排序字段是否有效"""
        if not field:
            return False
        
        # 如果是简单字段（如 "hops"），直接允许
        if "." not in field:
            return field in ["hops", "minHops"]
        
        # 分割前缀和属性名
        parts = field.split(".", 1)
        if len(parts) != 2:
            return False
        
        prefix, prop = parts
        
        # 验证前缀
        if allowed_prefixes and prefix not in allowed_prefixes:
            logging.warning(f"字段前缀 '{prefix}' 不在允许列表 {allowed_prefixes} 中")
            return False
        
        # 验证属性是否存在于 Schema
        all_props = set()
        for label_info in schema.get("node_labels", {}).values():
            all_props.update(label_info.get("properties", []))
        for rel_info in schema.get("relationship_types", {}).values():
            all_props.update(rel_info.get("properties", []))
        
        if prop not in all_props:
            logging.warning(f"属性 '{prop}' 不在 Schema 中")
            return False
        
        return True

    def _validate_where_clause(
        self, 
        where: str, 
        query_type: str, 
        schema: Dict
    ) -> bool:
        """验证 WHERE 子句格式"""
        if not where:
            return False
        
        # 定义每种查询类型允许的变量前缀
        allowed_vars = {
            "neighbor_query": ["nbr", "firstRel"],
            "common_neighbor": ["C", "rA", "rB"],
            "path_query": ["hops"],
            "filter_query": ["n", "r"]
        }
        
        vars_for_type = allowed_vars.get(query_type, [])
        
        # 检查是否包含允许的变量
        if not any(var in where for var in vars_for_type):
            logging.warning(f"WHERE 子句 '{where}' 中的变量不符合查询类型 {query_type}")
            return False
        
        # 检查是否包含操作符
        operators = [">", "<", ">=", "<=", "=", "!=", "IN", "CONTAINS"]
        if not any(op in where for op in operators):
            logging.warning(f"WHERE 子句 '{where}' 缺少操作符")
            return False
        
        return True

    def _validate_aggregate_field(
        self, 
        field: str, 
        query_type: str, 
        schema: Dict
    ) -> bool:
        """验证聚合字段"""
        allowed_prefixes = {
            "neighbor_query": ["rel", "neighbor", "firstRel", "nbr"],
            "common_neighbor": ["rA", "rB", "commonNeighbor", "C"],
            "filter_query": ["rel", "node", "r", "n"]
        }
        
        prefixes = allowed_prefixes.get(query_type, [])
        return self._validate_order_field(field, prefixes, schema)
    def _fallback_extract(self, question: str, query_type: str) -> Dict:
        """规则提取降级方案（返回新格式）"""
        schema = self.schema_analyzer.get_schema()
        
        # 检测 label
        label = ""
        for l in schema["node_labels"].keys():
            if l.lower() in question.lower():
                label = l
                break
        if not label:
            label = list(schema["node_labels"].keys())[0]
        
        # 检测 key - 优先使用 node_key
        label_info = schema["node_labels"].get(label, {})
        props = label_info.get("properties", [])
        key = "id"
        # 优先级: node_key > id > acct_id > userId > user_id > name
        for k in ["node_key", "id", "acct_id", "userId", "user_id", "name"]:
            if k in props:
                key = k
                break
        
        # 提取值 - 支持人名和ID格式
        # 1. 先尝试提取人名（英文名格式：姓 名）
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        names = re.findall(name_pattern, question)
        
        # 2. 提取ID格式的值
        id_values = re.findall(r'\b([a-zA-Z]+\d+|[a-zA-Z]+_\d+)\b', question)
        
        # 优先使用人名，如果没有则使用ID
        values = names if names else id_values
        
        params = {
            "label": label,
            "key": key
        }
        
        if query_type in ["node_lookup", "neighbor_query", "subgraph", "filter_query"]:
            params["value"] = values[0] if values else None
        elif query_type == "subgraph_by_nodes":
            # 多节点子图查询：提取所有节点值
            params["values"] = values if values else []
        elif query_type in ["common_neighbor", "path_query"]:
            params["v1"] = values[0] if len(values) > 0 else None
            params["v2"] = values[1] if len(values) > 1 else None
        elif query_type == "global_stats":
            for field in ["country", "city", "type"]:
                if field in props:
                    params["group_by"] = field
                    break
        
        # 提取跳数
        hop_match = re.search(r'(\d+)\s*[跳度层]', question)
        if hop_match:
            params["hops"] = int(hop_match.group(1))
        
        # 检测修饰符需求
        modifiers = {}
        
        # 检测聚合需求
        if any(kw in question.lower() for kw in ["统计", "数量", "多少", "总和", "平均"]):
            if "数量" in question.lower() or "多少" in question.lower():
                modifiers["aggregate"] = "count"
            elif "总和" in question.lower() or "总计" in question.lower():
                modifiers["aggregate"] = "sum"
            elif "平均" in question.lower():
                modifiers["aggregate"] = "avg"
        
        # 检测排序需求
        if any(kw in question.lower() for kw in ["排序", "从大到小", "从小到大", "最大", "最小"]):
            if "从大到小" in question.lower() or "最大" in question.lower():
                modifiers["order_direction"] = "DESC"
            else:
                modifiers["order_direction"] = "ASC"
        
        # 检测数量限制
        limit_match = re.search(r'前(\d+)|最多(\d+)|(\d+)个', question)
        if limit_match and "所有" not in question.lower() and "全部" not in question.lower():
            limit_num = limit_match.group(1) or limit_match.group(2) or limit_match.group(3)
            modifiers["limit"] = int(limit_num)
        
        # 返回新格式
        return {
            "params": params,
            "modifiers": modifiers
        }
    
    def _validate_params(self, params: Dict, template: Dict) -> None:
        """验证必需参数"""
        for req in template["required_params"]:
            if req not in params or params[req] is None:
                raise ValueError(f"缺少必需参数: {req}")


# ==========================================
# 6. 查询执行器
# ==========================================
class QueryExecutor:
    """执行查询模板（支持聚合）"""
    
    def __init__(self, client: Neo4jGraphClient):
        self.client = client
    
    def _apply_aggregation(self, results: List[JsonDict], aggregate_type: str, aggregate_field: Optional[str] = None) -> List[JsonDict]:
        """
        对查询结果应用聚合
        
        Args:
            results: 原始查询结果
            aggregate_type: 聚合类型 (count, sum, avg, max, min)
            aggregate_field: 聚合字段（可选，count 不需要）
            
        Returns:
            聚合后的结果
        """
        # 统一转换为小写，避免大小写不匹配问题
        aggregate_type = aggregate_type.lower()
        
        if not results:
            return [{"aggregate_type": aggregate_type, "value": 0}]
        
        if aggregate_type == "count":
            return [{"aggregate_type": "count", "value": len(results)}]
        
        # 其他聚合类型需要指定字段
        if not aggregate_field:
            return [{"error": f"聚合类型 {aggregate_type} 需要指定 aggregate_field"}]
        
        # 提取字段值（支持嵌套字段，如 rel.base_amt）
        values = []
        for item in results:
            try:
                # 支持嵌套访问，如 "rel.base_amt"
                value = item
                for key in aggregate_field.split('.'):
                    value = value.get(key, {})
                
                if isinstance(value, (int, float)):
                    values.append(value)
            except (AttributeError, TypeError):
                continue
        
        if not values:
            return [{"aggregate_type": aggregate_type, "value": None, "note": "没有找到有效的数值"}]
        
        # 执行聚合
        if aggregate_type == "sum":
            result_value = sum(values)
        elif aggregate_type == "avg":
            result_value = sum(values) / len(values)
        elif aggregate_type == "max":
            result_value = max(values)
        elif aggregate_type == "min":
            result_value = min(values)
        else:
            return [{"error": f"未知的聚合类型: {aggregate_type}"}]
        
        return [{
            "aggregate_type": aggregate_type,
            "field": aggregate_field,
            "value": result_value,
            "count": len(values)
        }]
    
    def execute(self, query_type: str, params: Dict) -> Dict:
        """根据类型和参数执行查询（支持聚合）"""
        template = QUERY_TEMPLATES[query_type]
        method_name = template["method"]
        method = getattr(self.client, method_name)
        
        # 检查是否需要聚合
        aggregate_type = params.pop("aggregate", None)
        aggregate_field = params.pop("aggregate_field", None)
        
        try:
            # 根据不同方法调整参数格式
            if query_type == "node_lookup":
                result = method(params["label"], params["key"], params["value"])
                results = [result] if result else []
            
            elif query_type == "neighbor_query":
                results = method(
                    params["label"],
                    params["key"],
                    params["value"],
                    hops=params.get("hops", 1),
                    rel_type=params.get("rel_type"),
                    direction=params.get("direction", "both"),
                    where=params.get("where"),
                    order_by=params.get("order_by"),
                    order_direction=params.get("order_direction", "ASC"),
                    limit=params.get("limit")
                )
                
                # 如果需要聚合，处理结果
                if aggregate_type:
                    results = self._apply_aggregation(results, aggregate_type, aggregate_field)
            
            elif query_type == "common_neighbor":
                results = method(
                    (params["label"], params["key"], params["v1"]),
                    (params["label"], params["key"], params["v2"]),
                    rel_type=params.get("rel_type"),
                    direction=params.get("direction", "both"),
                    where=params.get("where"),
                    order_by=params.get("order_by"),
                    order_direction=params.get("order_direction", "ASC"),
                    limit=params.get("limit")
                )
                
                # 如果需要聚合，处理结果
                if aggregate_type:
                    results = self._apply_aggregation(results, aggregate_type, aggregate_field)
            
            elif query_type == "path_query":
                results = method(
                    (params["label"], params["key"], params["v1"]),
                    (params["label"], params["key"], params["v2"]),
                    rel_type=params.get("rel_type"),
                    direction=params.get("direction", "both"),
                    min_hops=params.get("min_hops", 1),
                    max_hops=params.get("max_hops", 5),
                    where=params.get("where"),
                    order_by=params.get("order_by"),
                    order_direction=params.get("order_direction", "ASC"),
                    limit=params.get("limit")
                )
                
                # 如果需要聚合，处理结果
                if aggregate_type:
                    results = self._apply_aggregation(results, aggregate_type, aggregate_field)
            
            elif query_type == "global_stats":
                results = method(
                    params["label"],
                    group_by=params.get("group_by"),
                    where=params.get("where"),
                    params=params.get("params"),
                    metrics=params.get("metrics"),
                    limit=params.get("limit")
                )
            
            elif query_type == "subgraph":
                result = method(
                    (params["label"], params["key"], params["value"]),
                    hops=params.get("hops", 2),
                    rel_type=params.get("rel_type"),
                    direction=params.get("direction", "both"),
                    where=params.get("where"),
                    limit_paths=params.get("limit_paths", 200)
                )
                results = [result]
            
            elif query_type == "subgraph_by_nodes":
                result = method(
                    params["label"],
                    params["key"],
                    params["values"],
                    include_internal=params.get("include_internal", True),
                    rel_type=params.get("rel_type"),
                    direction=params.get("direction", "both"),
                    where=params.get("where")
                )
                results = [result]
            
            elif query_type == "filter_query":
                results = method(
                    (params["label"], params["key"], params["value"]),
                    rel_type=params.get("rel_type"),
                    node_label=params.get("node_label"),
                    direction=params.get("direction", "out"),
                    node_where=params.get("node_where"),
                    rel_where=params.get("rel_where"),
                    params=params.get("params"),
                    limit=params.get("limit")
                )
                
                # 如果需要聚合，处理结果
                if aggregate_type:
                    results = self._apply_aggregation(results, aggregate_type, aggregate_field)
            
            else:
                return {"success": False, "error": f"未知查询类型: {query_type}"}
            
            return {
                "success": True,
                "query_type": query_type,
                "params": params,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query_type": query_type,
                "params": params
            }


# ==========================================
# 7. 主引擎（双 LLM 架构）
# ==========================================
# add gjq
class NaturalLanguageQueryEngine:
    """自然语言查询引擎（双 LLM + Schema 工具）"""
    
    def __init__(self, db_client: Neo4jGraphClient, reasoner: Reasoner):
        """
        初始化自然语言查询引擎
        
        Args:
            db_client: Neo4j 数据库客户端
            reasoner: Reasoner 实例，用于调用 LLM
        """
        self.client = db_client
        self.reasoner = reasoner
        
        # 初始化各组件
        self.schema_analyzer = SchemaAnalyzer(db_client)
        self.type_classifier = QueryTypeClassifier(reasoner)
        self.param_extractor = ParameterExtractor(reasoner, self.schema_analyzer)
        self.executor = QueryExecutor(db_client)
    
    def initialize(self):
        """初始化：加载 Schema"""
        print("\n🔍 正在分析图数据库结构...")
        schema = self.schema_analyzer.get_schema()
        
        print(f"✅ Schema 加载完成")
        print(f"   节点类型: {list(schema['node_labels'].keys())}")
        print(f"   关系类型: {list(schema['relationship_types'].keys())}\n")
    
    def ask(self, question: str) -> Dict:
        """
        主入口：处理自然语言问题
        
        流程：
        1. LLM1 判断查询类型
        2. 获取 Schema 信息
        3. LLM2 填充参数
        4. 执行查询
        """
        print(f"\n💬 用户问题: {question}")
        print("=" * 80)
        
        # Step 1: LLM1 判断查询类型
        print("🤖 LLM1 正在判断查询类型...")
        query_type = self.type_classifier.classify(question)
        print(f"   查询类型: {query_type}")
        print(f"   说明: {QUERY_TEMPLATES[query_type]['description']}")
        
        # Step 2: 获取 Schema（已缓存，不会重复查询）
        print("\n📊 获取图 Schema 信息...")
        schema_info = self.schema_analyzer.format_for_llm()
        print("   Schema 已加载")
        
        # 调试：打印 Schema 信息（可选）
        if os.environ.get("DEBUG_SCHEMA"):
            print("\n" + "="*80)
            print(schema_info)
            print("="*80)
        
        # Step 3: LLM2 提取参数
        print("\n🤖 LLM2 正在提取参数...")
        try:
            params = self.param_extractor.extract(question, query_type)
            print(f"   提取参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"   ❌ 参数提取失败: {e}")
            return {"success": False, "error": f"参数提取失败: {e}"}
        
        # Step 4: 执行查询
        print("\n⚙️  执行查询...")
        result = self.executor.execute(query_type, params)
        
        if result["success"]:
            print(f"✅ 查询成功！返回 {result['count']} 条结果\n")
            
            # 显示结果预览
            for i, item in enumerate(result["results"][:3], 1):
                print(f"{i}. {item}")
            
            if result["count"] > 3:
                print(f"... 还有 {result['count'] - 3} 条结果")
        else:
            print(f"❌ 查询失败: {result['error']}")
        
        return result


# ==========================================
# 8. 交互式命令行
# ==========================================
# add gjq
def main():
    """命令行入口"""
    print("=" * 80)
    print("🚀 Neo4j 自然语言查询引擎（双 LLM 架构）")
    print("=" * 80)
    
    # 数据库配置
    uri = input("\nNeo4j URI (默认 bolt://localhost:7687): ").strip() or "bolt://localhost:7687"
    user = input("用户名 (默认 neo4j): ").strip() or "neo4j"
    password = input("密码: ").strip()
    
    try:
        # 初始化数据库客户端
        config = Neo4jConfig(uri=uri, user=user, password=password)
        db_client = Neo4jGraphClient(config)
        
        # add gjq
        # 初始化 Reasoner（使用配置文件或默认配置）
        # 这里需要根据实际情况配置 ReasonerConfig
        # 示例：使用 OpenAI
        from aag.config.engine_config import LLMConfig
        
        llm_config = LLMConfig(
            provider='openai',
            ollama={},
            openai={
                'base_url': os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                'api_key': os.environ.get('OPENAI_API_KEY'),
                'model': 'gpt-4o-mini'
            }
        )
        reasoner_config = ReasonerConfig(llm=llm_config)
        reasoner = Reasoner(reasoner_config)
        
        # 创建查询引擎
        engine = NaturalLanguageQueryEngine(db_client, reasoner)
        engine.initialize()
        
        print("\n✨ 已连接！输入 'exit' 退出，'help' 查看示例")
        print("=" * 80)
        
        while True:
            question = input("\n❓ 请输入问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', '退出']:
                print("\n👋 再见！")
                break
            
            if question.lower() == 'help':
                print("\n📚 示例问题：")
                for qtype, info in QUERY_TEMPLATES.items():
                    print(f"  - {info['example']}")
                continue
            
            engine.ask(question)
        
        db_client.close()
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
