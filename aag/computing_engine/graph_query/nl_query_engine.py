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
from aag.computing_engine.graph_query.nl_query_engine_refactored import ParameterExtractorRouter
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
from aag.computing_engine.graph_query.templates import QUERY_TEMPLATES, QUERY_MODIFIERS
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
        # 使用 Reasoner 的 nl_query_classify_type 方法，直接调用 model_deployment.py 中的方法
        query_type = self.reasoner.nl_query_classify_type(question, QUERY_TEMPLATES)
        logging.info(f"LLM 分类响应: {query_type}")
        # ⚠️ CRITICAL FIX: 修正 LLM 可能返回的错误类型名
        # LLM 可能返回 "subgraph_extract_by_nodes" 而不是正确的 "subgraph_by_nodes"
        if query_type == "subgraph_extract_by_nodes":
            logging.warning(f"LLM 返回了错误的类型名 '{query_type}'，自动修正为 'subgraph_by_nodes'")
            query_type = "subgraph_by_nodes"
        
        if query_type not in QUERY_TEMPLATES:
            # 降级到规则匹配
            logging.warning(f"未知查询类型: {query_type}，使用规则匹配降级")
            query_type = self._fallback_classify(question)
        
        return query_type
    
    def _fallback_classify(self, question: str) -> str:
        """规则匹配降级方案"""
        q = question.lower()
        
        # 优先级从高到低
        # 0. 检测是否有具体的起点节点（人名、ID等）
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        has_specific_node = bool(re.search(name_pattern, question))
        
        # 1. 检测聚合查询（最高优先级）
        # 1.1 分组聚合：包含"每个"关键词
        if any(kw in q for kw in ["统计每个", "计算每个", "每个", "各个", "分别"]):
            # ⚠️ CRITICAL: 即使有"每个"，如果有具体节点，也应该是 neighbor_query
            if has_specific_node:
                logging.info(f"检测到具体节点 + '每个'关键词，使用 neighbor_query + aggregate")
                return "neighbor_query"
            return "aggregation_query"
        
        # 1.2 全局聚合：包含聚合关键词（统计、计算、数量、总额等）
        if any(kw in q for kw in ["统计", "计算", "数量", "次数", "总数", "总金额", "金额合计", "平均", "最大", "最小", "top", "前", "排序", "排名", "占比"]):
            # ⚠️ CRITICAL: 如果有具体的起点节点，应该是 neighbor_query + aggregate
            if has_specific_node:
                logging.info(f"检测到具体节点 + 聚合关键词，使用 neighbor_query + aggregate")
                return "neighbor_query"
            
            # 排除"找出"、"列出"、"返回"等返回列表的关键词
            if not any(kw in q for kw in ["找出", "列出", "返回...的交易", "查询...的交易"]):
                return "aggregation_query"
        
        # 2. 检测关系过滤查询（关键词：交易、转账、关系属性）
        if any(kw in q for kw in ["交易", "转账", "金额", "is_sar", "tran_"]):
            # ⚠️ 关键区分：如果包含人名或具体节点标识，是邻居查询
            name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
            if re.search(name_pattern, question):
                # 包含人名，很可能是邻居查询
                if any(kw in q for kw in ["对手", "邻居", "直接", "发生过"]):
                    return "neighbor_query"
            
            # 如果包含关系属性条件（大于、小于等），且要求返回列表
            if any(kw in q for kw in ["大于", "小于", "等于", ">", "<", "="]):
                if any(kw in q for kw in ["找出", "列出", "返回", "查询"]):
                    return "relationship_filter"
            
            # 如果包含"所有"且没有明确起点，是关系过滤
            if "所有" in q and not any(kw in q for kw in ["的", "作为"]):
                return "relationship_filter"
        
        # 检测公共邻居查询（支持简单模式和过滤模式）
        if "公共" in q or "共同" in q or ("既" in q and "又" in q):
            # 检查是否是节点属性条件（如 "为true"、"为false"）
            # 这种情况目前不支持，需要自定义Cypher
            if any(kw in q for kw in ["为true", "为false"]):
                logging.warning(f"检测到复杂的节点属性条件公共邻居查询，当前模板不支持，建议使用自定义Cypher")
                return "node_lookup"  # 降级处理
            
            # 统一返回 common_neighbor，由参数提取器判断是简单模式还是过滤模式
            return "common_neighbor"
        
        if any(kw in q for kw in ["路径", "怎么到", "到达"]):
            return "path_query"
        
        # ⚠️ CRITICAL FIX: 优先检测多节点子图查询（在检测单节点邻居查询之前）
        # 检测多节点子图查询的关键特征：
        # 1. 包含"子图"关键词
        # 2. 包含节点列表标识（如 "A、B、C" 或 "[A,B,C]" 或 "之间"）
        # 3. 包含"包含"、"及其"、"相互"、"连边"等关键词
        if "子图" in q or any(kw in q for kw in ["包含节点", "节点列表", "相互之间", "连边"]):
            # 检测是否包含多个节点标识
            # 方式1：逗号分隔（如 "A、B、C" 或 "A,B,C"）
            # 方式2：列表格式（如 "[A,B,C]"）
            # 方式3：关键词（如 "之间"、"相互"、"互相"）
            if any(kw in q for kw in ["、", ",", "[", "之间", "相互", "互相", "及其", "连边"]):
                logging.info("检测到多节点子图查询关键词，返回 subgraph_by_nodes")
                return "subgraph_by_nodes"
            # 如果只有"子图"但没有多节点标识，是单中心子图
            if "子图" in q:
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


# add gjq
class CypherValidator:
    """Use LLM to validate generated Cypher statements"""
    
    def __init__(self, reasoner: Reasoner, schema_analyzer: Optional['SchemaAnalyzer'] = None):
        self.reasoner = reasoner
        self.schema_analyzer = schema_analyzer
    
    def validate(self, cypher: str, question: str, query_type: str, params: Dict) -> Dict:
        """
        Validate if Cypher statement conforms to syntax rules and user question requirements
        
        Args:
            cypher: Generated Cypher statement
            question: User's original question
            query_type: Query type
            params: Extracted parameters
            
        Returns:
            Validation result dictionary {
                "is_valid": bool,
                "issues": List[str],  # List of issues found
                "suggestions": List[str],  # Modification suggestions
                "corrected_cypher": str  # Corrected Cypher (if needed)
            }
        """
        # Get Schema information
        schema_info = ""
        if self.schema_analyzer:
            schema_info = self.schema_analyzer.format_for_llm()
        
        # Get query template information
        template_info = ""
        template_cypher_example = ""
        if query_type in QUERY_TEMPLATES:
            template = QUERY_TEMPLATES[query_type]
            template_info = f"""
## Query Template Information
- **Template Name**: {query_type}
- **Description**: {template.get('description', '')}
- **Method Used**: {template.get('method', '')}
- **Required Parameters**: {template.get('required_params', [])}
- **Optional Parameters**: {template.get('optional_params', [])}

⚠️ **Important**: Please refer to this template information to validate parameters and generate Cypher statements.
"""
            
            # Provide Cypher examples based on query type
            if query_type == "subgraph":
                template_cypher_example = """
### Correct Cypher Pattern for subgraph Query

For "N-hop neighbor" queries, the correct approach is to use variable-length path patterns:

```cypher
// Correct example: Return all 1-hop and 2-hop neighbors
MATCH (a:Account {{node_key: "Collins Steven"}})-[*1..2]-(b)
RETURN DISTINCT b
ORDER BY b.acct_id

// Or use UNION to query 1-hop and 2-hop separately
MATCH (a:Account {{node_key: "Collins Steven"}})-[r]-(b)
RETURN DISTINCT b
UNION
MATCH (a:Account {{node_key: "Collins Steven"}})-[*2]-(b)
WHERE NOT (a)--(b)  // Exclude 1-hop neighbors
RETURN DISTINCT b
ORDER BY b.acct_id
```

❌ **Wrong Example**: Only returns 2-hop paths
```cypher
// This is wrong! Only returns two-hop paths, missing one-hop neighbors
MATCH (a:Account {{node_key: "Collins Steven"}})-[rel:TRANSFER]-(b)
WITH a, b
MATCH (b)-[rel2:TRANSFER]-(c)
RETURN a, b, c
ORDER BY c.acct_id
```
"""
        
        # add gjq
        # Use Reasoner's nl_query_validate_cypher method, directly call model_deployment.py method
        try:
            result = self.reasoner.nl_query_validate_cypher(
                cypher=cypher,
                question=question,
                query_type=query_type,
                params=params,
                schema_info=schema_info,
                template_info=template_info,
                template_cypher_example=template_cypher_example
            )
            return result
        except Exception as e:
            logging.error(f"Cypher validation failed: {e}")
            # When error occurs, default to valid to avoid blocking queries
            return {
                "is_valid": True,
                "issues": [],
                "suggestions": [],
                "corrected_cypher": None
            }



# ==========================================
# 6. 查询执行器
# ==========================================
class QueryExecutor:
    """执行查询模板（支持聚合）"""
    
    def __init__(self, client: Neo4jGraphClient, schema_analyzer: Optional['SchemaAnalyzer'] = None):
        self.client = client
        self.schema_analyzer = schema_analyzer
    
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
    
    # add gjq
    def execute(self, query_type: str, params: Dict) -> Dict:
        """根据类型和参数执行查询（支持聚合）"""
        template = QUERY_TEMPLATES[query_type]
        
        # ⚠️ CRITICAL: 对于 aggregation_query，不要 pop aggregate_field
        # 因为后续转换为 relationship_filter 时还需要使用
        if query_type == "aggregation_query":
            # 不 pop，保留在 params 中
            aggregate_type = params.get("aggregate", None)
            aggregate_field = params.get("aggregate_field", None)
        else:
            # 其他查询类型，pop 出来
            aggregate_type = params.pop("aggregate", None)
            aggregate_field = params.pop("aggregate_field", None)
        
        try:
            # 根据不同方法调整参数格式
            if query_type == "node_lookup":
                # 判断是单节点查找还是多节点筛选
                if "conditions" in params:
                    # ⚠️ FIX: 处理 order_by 参数格式
                    # LLM 可能返回 {"field": "direction"} 格式，需要拆分
                    order_by_param = params.get("order_by")
                    order_direction_param = params.get("order_direction", "ASC")
                    
                    if isinstance(order_by_param, dict):
                        # 如果 order_by 是字典格式 {"field": "direction"}
                        # 提取字段名和方向
                        if order_by_param:
                            field_name = list(order_by_param.keys())[0]
                            direction = order_by_param[field_name]
                            order_by_param = field_name
                            order_direction_param = direction
                        else:
                            order_by_param = None
                    
                    # 多节点筛选模式
                    results = self.client.filter_nodes_by_properties(
                        params["label"],
                        params["conditions"],
                        return_fields=params.get("return_fields"),
                        order_by=order_by_param,
                        order_direction=order_direction_param,
                        limit=params.get("limit")
                    )
                else:
                    # 单节点查找模式（支持return_fields）
                    result = self.client.get_node_by_unique_key(
                        params["label"],
                        params["key"],
                        params["value"],
                        return_fields=params.get("return_fields")
                    )
                    results = [result] if result else []
            
            elif query_type == "relationship_filter":
                # 关系过滤查询
                results = self.client.filter_relationships(
                    params["rel_type"],
                    start_label=params.get("start_label"),
                    end_label=params.get("end_label"),
                    rel_conditions=params.get("rel_conditions"),
                    return_fields=params.get("return_fields"),
                    aggregate=params.get("aggregate"),
                    aggregate_field=params.get("aggregate_field"),
                    order_by=params.get("order_by"),
                    order_direction=params.get("order_direction", "ASC"),
                    limit=params.get("limit")
                )
            
            elif query_type == "aggregation_query":
                # 聚合统计查询
                # 检查是否有分组参数
                has_grouping = params.get("group_by_node") or params.get("group_by_property")
                
                # 如果没有分组参数，这应该是全局聚合查询
                if not has_grouping:
                    logging.warning("检测到 aggregation_query 缺少分组参数，转换为全局聚合查询")
                    
                    # 场景1：节点属性聚合（有 conditions）
                    if params.get("conditions"):
                        logging.info("转换为 node_lookup + aggregate")
                        # 转换为 node_lookup 查询
                        results = self.client.filter_nodes_by_properties(
                            params["node_label"],
                            params["conditions"],
                            return_fields=params.get("return_fields"),
                            order_by=params.get("order_by"),
                            order_direction=params.get("order_direction", "ASC"),
                            limit=params.get("limit")
                        )
                        
                        # 应用聚合
                        agg_type = params.get("aggregate_type", "COUNT").upper()
                        if agg_type == "COUNT":
                            results = [{"aggregate_type": "count", "value": len(results)}]
                        else:
                            # 其他聚合类型需要 aggregate_field
                            agg_field = params.get("aggregate_field")
                            if agg_field:
                                results = self._apply_aggregation(results, agg_type, agg_field)
                            else:
                                results = [{"error": f"聚合类型 {agg_type} 需要指定 aggregate_field"}]
                    
                    # 场景2：关系属性聚合（有 rel_type）
                    elif params.get("rel_type"):
                        logging.info("转换为 relationship_filter + aggregate")
                        
                        # ⚠️ CRITICAL: 获取聚合类型和字段
                        # 注意：aggregate_field 可能在 params 中，也可能已经被 pop 出去了
                        agg_type = params.get("aggregate_type")
                        agg_field = params.get("aggregate_field")
                        
                        # 如果 aggregate_field 不在 params 中，尝试从外层获取
                        if not agg_field and aggregate_field:
                            agg_field = aggregate_field
                            logging.info(f"从外层获取 aggregate_field: {agg_field}")
                        
                        logging.info(f"aggregate_type: {agg_type}, aggregate_field: {agg_field}")
                        
                        # ⚠️ CRITICAL: 确保 aggregate_field 正确传递
                        # 对于 SUM/AVG/MAX/MIN，aggregate_field 是必需的
                        if agg_type and agg_type.upper() in ["SUM", "AVG", "MAX", "MIN"]:
                            if not agg_field:
                                # 尝试从 rel_type 的 Schema 中推断默认字段
                                if self.schema_analyzer:
                                    schema = self.schema_analyzer.get_schema()
                                    rel_info = schema.get("relationship_types", {}).get(params["rel_type"], {})
                                    rel_props = rel_info.get("properties", [])
                                    
                                    # 优先使用常见的金额字段
                                    for common_field in ["base_amt", "amount", "value", "total"]:
                                        if common_field in rel_props:
                                            agg_field = common_field
                                            logging.warning(f"aggregate_field 未指定，自动推断为: {agg_field}")
                                            break
                                    
                                    if not agg_field:
                                        # 如果还是没有，使用第一个数值型属性
                                        for prop in rel_props:
                                            if prop not in ["tran_id", "alert_id", "is_sar", "tx_type", "tran_timestamp"]:
                                                agg_field = prop
                                                logging.warning(f"aggregate_field 未指定，使用第一个可能的数值字段: {agg_field}")
                                                break
                                else:
                                    logging.error("schema_analyzer 未初始化，无法自动推断 aggregate_field")
                        
                        # 转换为 relationship_filter 查询
                        results = self.client.filter_relationships(
                            params["rel_type"],
                            start_label=params.get("start_label") or params.get("node_label"),
                            end_label=params.get("end_label") or params.get("node_label"),
                            rel_conditions=params.get("rel_conditions"),
                            aggregate=agg_type,
                            aggregate_field=agg_field,
                            return_fields=params.get("return_fields"),
                            order_by=params.get("order_by"),
                            order_direction=params.get("order_direction", "ASC"),
                            limit=params.get("limit")
                        )
                    
                    else:
                        results = [{"error": "aggregation_query 缺少必要参数：需要 group_by_node/group_by_property 或 conditions 或 rel_type"}]
                
                else:
                    # 正常的分组聚合查询
                    results = self.client.aggregation_query(
                        params["aggregate_type"],
                        group_by_node=params.get("group_by_node"),
                        group_by_property=params.get("group_by_property"),
                        node_label=params.get("node_label"),
                        rel_type=params.get("rel_type"),
                        direction=params.get("direction", "out"),
                        aggregate_field=params.get("aggregate_field"),
                        return_fields=params.get("return_fields"),
                        where=params.get("where"),
                        order_by=params.get("order_by"),
                        order_direction=params.get("order_direction", "DESC"),
                        limit=params.get("limit")
                    )
            
            elif query_type == "neighbor_query":
                results = self.client.neighbors_n_hop(
                    params["label"],
                    params["key"],
                    params["value"],
                    hops=params.get("hops", 1),
                    rel_type=params.get("rel_type"),
                    direction=params.get("direction", "both"),
                    where=params.get("where"),
                    return_fields=params.get("return_fields"),
                    order_by=params.get("order_by"),
                    order_direction=params.get("order_direction", "ASC"),
                    limit=params.get("limit"),
                    return_distinct=params.get("return_distinct", False),
                    exclude_start=params.get("exclude_start", False),
                    return_path_length=params.get("return_path_length", False)
                )
                
                # 如果需要聚合，处理结果
                if aggregate_type:
                    results = self._apply_aggregation(results, aggregate_type, aggregate_field)
            
            elif query_type == "common_neighbor":
                # 判断是简单模式还是过滤模式
                if "rel_conditions" in params and params["rel_conditions"]:
                    # 过滤模式：使用 common_neighbors_with_rel_filter
                    results = self.client.common_neighbors_with_rel_filter(
                        (params["label"], params["key"], params["v1"]),
                        (params["label"], params["key"], params["v2"]),
                        rel_type=params.get("rel_type"),
                        direction=params.get("direction", "both"),
                        rel_conditions=params.get("rel_conditions"),
                        neighbor_where=params.get("neighbor_where"),
                        return_fields=params.get("return_fields"),
                        order_by=params.get("order_by"),
                        order_direction=params.get("order_direction", "ASC"),
                        limit=params.get("limit")
                    )
                else:
                    # 简单模式：使用 common_neighbors
                    # ⚠️ CRITICAL: 检测是否需要聚合排序
                    # 如果有 aggregate 参数且 order_by 是 "count"，启用聚合模式
                    enable_aggregate = aggregate_type and aggregate_type.lower() == "count"
                    
                    results = self.client.common_neighbors(
                        (params["label"], params["key"], params["v1"]),
                        (params["label"], params["key"], params["v2"]),
                        rel_type=params.get("rel_type"),
                        direction=params.get("direction", "both"),
                        where=params.get("where"),
                        order_by=params.get("order_by"),
                        order_direction=params.get("order_direction", "ASC"),
                        limit=params.get("limit"),
                        aggregate=enable_aggregate
                    )
                
                # ⚠️ CRITICAL: 公共邻居的聚合处理
                # 如果需要聚合，但不是 count 类型（已经在上面处理了）
                if aggregate_type and aggregate_type.lower() != "count":
                    # 其他聚合类型（SUM、AVG等）
                    results = self._apply_aggregation(results, aggregate_type, aggregate_field)
            
            elif query_type == "path_query":
                results = self.client.paths_between(
                    (params["label"], params["key"], params["v1"]),
                    (params["label"], params["key"], params["v2"]),
                    rel_type=params.get("rel_type"),
                    direction=params.get("direction", "both"),
                    min_hops=params.get("min_hops", 1),
                    max_hops=params.get("max_hops", 5),
                    where=params.get("where"),
                    return_fields=params.get("return_fields"),
                    order_by=params.get("order_by"),
                    order_direction=params.get("order_direction", "ASC"),
                    limit=params.get("limit")
                )
                
                # 如果需要聚合，处理结果
                if aggregate_type:
                    results = self._apply_aggregation(results, aggregate_type, aggregate_field)
            
            elif query_type == "global_stats":
                results = self.client.aggregate_stats(
                    params["label"],
                    group_by=params.get("group_by"),
                    where=params.get("where"),
                    params=params.get("params"),
                    metrics=params.get("metrics"),
                    limit=params.get("limit")
                )
            
            elif query_type == "subgraph":
                # 判断是单中心节点模式还是关系属性过滤模式
                if "rel_conditions" in params and params["rel_conditions"]:
                    # 关系属性过滤模式：使用 subgraph_extract_by_rel_filter
                    result = self.client.subgraph_extract_by_rel_filter(
                        params["rel_type"],
                        params["rel_conditions"],
                        start_label=params.get("start_label"),
                        end_label=params.get("end_label"),
                        direction=params.get("direction", "both"),
                        limit=params.get("limit")
                    )
                else:
                    # 单中心节点模式：使用 subgraph_extract
                    result = self.client.subgraph_extract(
                        (params["label"], params["key"], params["value"]),
                        hops=params.get("hops", 2),
                        rel_type=params.get("rel_type"),
                        direction=params.get("direction", "both"),
                        where=params.get("where"),
                        limit_paths=params.get("limit_paths", 200)
                    )
                
                # 如果需要聚合（如计算交易总数），应用聚合
                if aggregate_type:
                    if aggregate_type.lower() == "count":
                        # 计算关系总数（交易总数）
                        rel_count = result.get("relationship_count", len(result.get("relationships", [])))
                        results = [{"aggregate_type": "count", "value": rel_count}]
                    else:
                        # 其他聚合类型（如 SUM、AVG 等）
                        results = self._apply_aggregation([result], aggregate_type, aggregate_field)
                else:
                    results = [result]
            
            elif query_type == "subgraph_by_nodes":
                result = self.client.subgraph_extract_by_nodes(
                    params["label"],
                    params["key"],
                    params["values"],
                    include_internal=params.get("include_internal", True),
                    rel_type=params.get("rel_type"),
                    direction=params.get("direction", "both"),
                    where=params.get("where")
                )
                
                # ⚠️ CRITICAL FIX: 如果需要聚合（如计算节点总数），应用聚合
                if aggregate_type:
                    if aggregate_type.lower() == "count":
                        # 计算节点总数
                        node_count = result.get("node_count", len(result.get("nodes", [])))
                        results = [{"aggregate_type": "count", "value": node_count}]
                    else:
                        # 其他聚合类型（如 SUM、AVG 等）
                        results = self._apply_aggregation([result], aggregate_type, aggregate_field)
                else:
                    results = [result]
            
            elif query_type == "filter_query":
                results = self.client.filter_query(
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
    
    def __init__(self, db_client: Neo4jGraphClient, llm: LLMInterface, enable_validation: bool = True):
        self.client = db_client
        self.llm = llm
        # add gjq: log_manager 未定义，移除此行
        # self.log_manager = log_manager
        self.enable_validation = enable_validation
        
        # 初始化各组件
        self.schema_analyzer = SchemaAnalyzer(db_client)
        # add gjq: QueryTypeClassifier 需要 Reasoner 对象，从 LLMInterface 中获取
        self.type_classifier = QueryTypeClassifier(llm.reasoner)
        
        
        self.param_extractor = ParameterExtractorRouter(llm, self.schema_analyzer)
        logging.info("✅ 使用重构后的参数提取器路由器")
        
        # 初始化 Cypher 验证器（传入 schema_analyzer）
        # add gjq: CypherValidator 需要 Reasoner 对象，从 LLMInterface 中获取
        if enable_validation:
            self.cypher_validator = CypherValidator(llm.reasoner, self.schema_analyzer)
            logging.info("✅ Cypher 验证器已启用（包含 Schema 验证）")
        else:
            self.cypher_validator = None
            logging.info("⚠️  Cypher 验证器已禁用")
        
        self.executor = QueryExecutor(db_client, self.schema_analyzer)
    
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
            error_msg = str(e)
            print(f"   ❌ 参数提取失败: {error_msg}")
            
            # ⚠️ CRITICAL FIX: 检测查询类型判断错误，自动重试
            if "查询类型判断错误" in error_msg and "neighbor_query" in error_msg:
                print("\n🔄 检测到查询类型判断错误，自动切换到 neighbor_query 重试...")
                query_type = "neighbor_query"
                print(f"   新查询类型: {query_type}")
                print(f"   说明: {QUERY_TEMPLATES[query_type]['description']}")
                
                try:
                    params = self.param_extractor.extract(question, query_type)
                    print(f"   提取参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
                except Exception as e2:
                    print(f"   ❌ 重试后仍然失败: {e2}")
                    return {"success": False, "error": f"参数提取失败: {e2}"}
            else:
                return {"success": False, "error": f"参数提取失败: {error_msg}"}
        
        # Step 3.5: 验证参数（可选）
        corrected_cypher = None
        # ⚠️ 对于 path_query、subgraph 和 common_neighbor 查询类型，跳过 LLM3 验证
        # 原因：这些查询类型使用固定的模板，不需要 LLM3 动态修正
        # 解决方案：直接增强模板本身的能力，支持更多参数
        skip_validation_types = ["path_query", "subgraph"]
        
        if self.enable_validation and self.cypher_validator and query_type not in skip_validation_types:
            print("\n🔍 LLM3 正在验证参数...")
            # 构建一个模拟的 Cypher 语句用于验证
            # 注意：这里我们验证的是参数的完整性和正确性，而不是实际的 Cypher 语句
            validation_result = self._validate_params(question, query_type, params)
            
            if not validation_result["is_valid"]:
                print(f"   ⚠️  发现问题:")
                for issue in validation_result["issues"]:
                    print(f"      - {issue}")
                
                if validation_result["suggestions"]:
                    print(f"   💡 修改建议:")
                    for suggestion in validation_result["suggestions"]:
                        print(f"      - {suggestion}")
                
                # 如果验证器提供了修正后的 Cypher 语句，使用它
                if validation_result.get("corrected_cypher"):
                    print(f"\n   🔧 使用修正后的 Cypher 语句:")
                    print(f"      {validation_result['corrected_cypher']}")
                    corrected_cypher = validation_result["corrected_cypher"]
                else:
                    print("\n   ⚠️  参数可能不完整，但仍将继续执行查询...")
            else:
                print(f"   ✅ 参数验证通过")
        elif query_type in skip_validation_types:
            print(f"\n⏭️  跳过 LLM3 验证（查询类型: {query_type}）")
        
        # Step 4: 执行查询
        print("\n⚙️  执行查询...")
        
        # 如果有修正后的 Cypher 语句，直接执行它
        if corrected_cypher:
            print(f"   使用修正后的 Cypher 语句执行...")
            try:
                # ⚠️ FIX: 使用 run() 方法而不是 execute_cypher()
                results = self.client.run(corrected_cypher)
                result = {
                    "success": True,
                    "query_type": query_type,
                    "params": params,
                    "results": results,
                    "count": len(results),
                    "corrected": True
                }
            except Exception as e:
                result = {
                    "success": False,
                    "error": str(e),
                    "query_type": query_type,
                    "params": params,
                    "corrected_cypher": corrected_cypher
                }
        else:
            # 使用原始的模板方法执行
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
    
    def _validate_params(self, question: str, query_type: str, params: Dict) -> Dict:
        """
        验证提取的参数是否完整和正确
        
        Args:
            question: 用户原始问题
            query_type: 查询类型
            params: 提取的参数
            
        Returns:
            验证结果字典
        """
        # 构建一个描述性的"Cypher"用于验证
        cypher_description = self._build_cypher_description(query_type, params)
        
        # 调用验证器
        return self.cypher_validator.validate(cypher_description, question, query_type, params)
    
    def _build_cypher_description(self, query_type: str, params: Dict) -> str:
        """
        根据查询类型和参数构建 Cypher 描述
        
        这不是实际的 Cypher 语句，而是一个描述性的文本，
        用于让 LLM 理解我们要执行的查询
        """
        description = f"查询类型: {query_type}\n\n"
        description += "查询参数:\n"
        
        for key, value in params.items():
            description += f"  - {key}: {value}\n"
        
        # 根据查询类型添加特定的描述
        if query_type == "common_neighbor":
            description += "\n预期行为:\n"
            description += f"  1. 找出节点 {params.get('v1')} 和 {params.get('v2')} 的公共邻居\n"
            
            if "rel_conditions" in params and params["rel_conditions"]:
                description += f"  2. 对两条关系都应用过滤条件: {params['rel_conditions']}\n"
                description += f"  3. 只返回满足条件的公共邻居\n"
            else:
                description += f"  2. 返回所有公共邻居（无过滤条件）\n"
        
        elif query_type == "relationship_filter":
            description += "\n预期行为:\n"
            description += f"  1. 筛选关系类型: {params.get('rel_type')}\n"
            
            if "rel_conditions" in params and params["rel_conditions"]:
                description += f"  2. 应用关系属性过滤: {params['rel_conditions']}\n"
            
            if "return_fields" in params:
                description += f"  3. 返回字段: {params['return_fields']}\n"
        
        elif query_type == "aggregation_query":
            description += "\n预期行为:\n"
            description += f"  1. 聚合类型: {params.get('aggregate_type')}\n"
            description += f"  2. 分组依据: {params.get('group_by_node')}\n"
            
            if "aggregate_field" in params:
                description += f"  3. 聚合字段: {params['aggregate_field']}\n"
        
        elif query_type == "subgraph":
            description += "\n预期行为:\n"
            description += f"  1. 抽取以节点 {params.get('value')} 为中心的子图\n"
            description += f"  2. 跳数: {params.get('hops', 2)}\n"
            description += f"  3. 节点标签: {params.get('label')}\n"
            description += f"  4. 节点属性键: {params.get('key')}\n"
            
            if "rel_type" in params:
                description += f"  5. 关系类型: {params.get('rel_type')}\n"
            
            if "direction" in params:
                description += f"  6. 关系方向: {params.get('direction')}\n"
        
        elif query_type == "neighbor_query":
            description += "\n预期行为:\n"
            description += f"  1. 查询节点 {params.get('value')} 的邻居\n"
            description += f"  2. 跳数: {params.get('hops', 1)}\n"
            description += f"  3. 节点标签: {params.get('label')}\n"
            description += f"  4. 节点属性键: {params.get('key')}\n"
            
            if "rel_type" in params:
                description += f"  5. 关系类型: {params.get('rel_type')}\n"
            
            if "direction" in params:
                description += f"  6. 关系方向: {params.get('direction')}\n"
            
            if "order_by" in params:
                description += f"  7. 排序字段: {params.get('order_by')}\n"
        
        return description


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
