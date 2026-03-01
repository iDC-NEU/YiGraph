# /home/gaojq/AAG_duan/AAG/aag/computing_engine/graph_query/nl_query_engine.py

import json
import re
import os
import sys

 
# Add project root directory to Python path to enable importing aag module
# Get current file directory, then navigate up to find AAG directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
 
from openai import OpenAI
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

JsonDict = Dict[str, Any]

 
# Import Reasoner for LLM calls
from aag.reasoner.model_deployment import Reasoner
from aag.config.engine_config import ReasonerConfig

from aag.computing_engine.graph_query.graph_query import Neo4jGraphClient, Neo4jConfig
#from graph_query import Neo4jGraphClient, Neo4jConfig

 
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
# 2. LLM interface (via Reasoner)
# ==========================================

class LLMInterface:
    """LLM call interface (unified via Reasoner)."""
    
    def __init__(self, reasoner: Reasoner):
        """
        Args:
            reasoner: Reasoner instance for LLM calls.
        """
        self.reasoner = reasoner
    
    def call(self, prompt: str, **kwargs) -> str:
        """Call LLM and return text response (uses Reasoner.generate_response)."""
        response = self.reasoner.generate_response(prompt)
        if hasattr(response, 'text'):
            return response.text
        return str(response)


# ==========================================
# 3. LLM1: Query type classifier
# ==========================================

class QueryTypeClassifier:
    """Use LLM to classify query type."""
    
    def __init__(self, reasoner: Reasoner):
        """
        Args:
            reasoner: Reasoner instance for LLM calls.
        """
        self.reasoner = reasoner
    
    def classify(self, question: str) -> str:
        """Return query type (e.g. 'node_lookup')."""
        query_type = self.reasoner.nl_query_classify_type(question, QUERY_TEMPLATES)
        logging.info(f"LLM classification: {query_type}")
        
        if query_type not in QUERY_TEMPLATES:
            logging.warning(f"Unknown query type: {query_type}; using rule-based fallback")
            query_type = self._fallback_classify(question)
        
        return query_type
    
    def _fallback_classify(self, question: str) -> str:
        """Rule-based fallback for query type."""
        q = question.lower()
        if any(kw in q for kw in ["公共", "共同", "common"]):
            return "common_neighbor"
        if any(kw in q for kw in ["路径", "怎么到", "到达", "path", "between", "route"]):
            return "path_query"
        if any(kw in q for kw in ["子图", "subgraph"]):
            if any(kw in q for kw in ["、", "和", "及", "之间", "相互", "互相", " and ", " between ", " among "]):
                return "subgraph_by_nodes"
            return "subgraph"
        if any(kw in q for kw in ["邻居", "朋友", "关注", "关系", "neighbor", "neighbour", "friend", "related"]):
            return "neighbor_query"
        if any(kw in q for kw in ["找", "查", "获取", "find", "get", "lookup", "fetch"]):
            return "node_lookup"
        return "neighbor_query"


# ==========================================
# 4. Schema utilities
# ==========================================
class SchemaAnalyzer:
    """Graph schema analysis for Neo4j."""
    
    def __init__(self, client: Neo4jGraphClient):
        self.client = client
        self._schema_cache = None
    
    def get_schema(self) -> Dict:
        """Get and cache schema."""
        if self._schema_cache is None:
            self._schema_cache = self.client.get_schema()
        return self._schema_cache
    
    def format_for_llm(self) -> str:
        """Format schema for LLM (with sample values)."""
        schema = self.get_schema()
        formatted = "## Graph database schema\n\n"
        formatted += "### Node types and properties\n"
        for label, info in schema["node_labels"].items():
            props = info.get("properties", [])
            samples = info.get("sample_values", {})
            formatted += f"- **{label}**:\n"
            for prop in props:
                sample = samples.get(prop, "")
                sample_str = f" (e.g. {sample})" if sample else ""
                formatted += f"  - `{prop}`{sample_str}\n"
        
        formatted += "\n### Relationship types and properties\n"
        for rel_type, info in schema["relationship_types"].items():
            props = info.get("properties", [])
            samples = info.get("sample_values", {})
            if props:
                formatted += f"- **{rel_type}**:\n"
                for prop in props:
                    sample = samples.get(prop, "")
                    sample_str = f" (e.g. {sample})" if sample else ""
                    formatted += f"  - `{prop}`{sample_str}\n"
            else:
                formatted += f"- **{rel_type}**: no properties\n"
        
        formatted += "\n### Common relationship patterns\n"
        for pattern in schema["patterns"][:10]:
            formatted += f"- {pattern}\n"
        
        return formatted


# ==========================================
# 5. LLM2: Parameter extractor
# ==========================================

class ParameterExtractor:
    """Use LLM to fill parameters from template and schema."""
    
    def __init__(self, reasoner: Reasoner, schema_analyzer: SchemaAnalyzer):
        """
        Args:
            reasoner: Reasoner instance for LLM calls.
            schema_analyzer: Schema analyzer.
        """
        self.reasoner = reasoner
        self.schema_analyzer = schema_analyzer
    
    def extract(self, question: str, query_type: str) -> Dict:
        """Return filled params dict (including modifiers)."""
        template = QUERY_TEMPLATES[query_type]
        schema = self.schema_analyzer.get_schema()
        schema_info = self.schema_analyzer.format_for_llm()
        logging.info("Schema loaded for param extraction")
        
        result = self.reasoner.nl_query_extract_params(
            question=question,
            query_type=query_type,
            template=template,
            schema_info=schema_info,
            query_modifiers=QUERY_MODIFIERS
        )
        
        if isinstance(result, dict) and "params" in result and "modifiers" in result:
            params = result["params"]
            modifiers = result["modifiers"]
        else:
            logging.warning("LLM result format invalid; using rule-based fallback")
            result = self._fallback_extract(question, query_type)
            params = result.get("params", result)
            modifiers = result.get("modifiers", {})
        
        self._validate_params(params, template)
        cleaned_modifiers = self._validate_and_clean_modifiers(
            modifiers,
            query_type,
            schema
        )
        
        final_params = {**params, **cleaned_modifiers}
        logging.info(f"Extracted params: {params}")
        logging.info(f"Raw modifiers: {modifiers}")
        logging.info(f"Cleaned modifiers: {cleaned_modifiers}")
        logging.info(f"Final params: {final_params}")
        
        return final_params

    def _validate_and_clean_modifiers(
        self, 
        modifiers: Dict, 
        query_type: str,
        schema: Dict
    ) -> Dict:
        """
        Validate and clean modifiers: drop invalid ones, check schema, fix format, add defaults.
        """
        cleaned = {}
        
        if "order_by" in modifiers:
            order_field = modifiers["order_by"]
            if query_type == "neighbor_query":
                if self._validate_order_field(order_field, ["firstRel", "nbr"], schema):
                    cleaned["order_by"] = order_field
                    cleaned["order_direction"] = modifiers.get("order_direction", "ASC")
                else:
                    logging.warning(f"Invalid order_by: {order_field}, ignored")
            elif query_type == "common_neighbor":
                if self._validate_order_field(order_field, ["rA", "rB", "C"], schema):
                    cleaned["order_by"] = order_field
                    cleaned["order_direction"] = modifiers.get("order_direction", "ASC")
                else:
                    logging.warning(f"Invalid order_by: {order_field}, ignored")
            elif query_type == "path_query":
                if order_field == "hops" or self._validate_order_field(order_field, [], schema):
                    cleaned["order_by"] = order_field
                    cleaned["order_direction"] = modifiers.get("order_direction", "ASC")
                else:
                    logging.warning(f"Invalid order_by: {order_field}, ignored")
        
        if "limit" in modifiers:
            try:
                limit_val = int(modifiers["limit"])
                if limit_val > 0:
                    cleaned["limit"] = limit_val
                else:
                    logging.warning(f"limit must be > 0: {limit_val}, ignored")
            except (ValueError, TypeError):
                logging.warning(f"Invalid limit: {modifiers['limit']}, ignored")
        
        if "where" in modifiers:
            where_clause = modifiers["where"]
            if self._validate_where_clause(where_clause, query_type, schema):
                cleaned["where"] = where_clause
            else:
                logging.warning(f"Invalid where clause: {where_clause}, ignored")
        
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
                            logging.warning(f"Invalid aggregate field: {agg_field}, aggregate dropped")
                            cleaned.pop("aggregate", None)
                    else:
                        logging.warning(f"Aggregate {agg_type} requires aggregate_field, dropped")
            else:
                logging.warning(f"Unknown aggregate type: {agg_type}, ignored")
        
        return cleaned

    def _validate_order_field(
        self, 
        field: str, 
        allowed_prefixes: List[str], 
        schema: Dict
    ) -> bool:
        """Check if sort field is valid."""
        if not field:
            return False
        if "." not in field:
            return field in ["hops", "minHops"]
        parts = field.split(".", 1)
        if len(parts) != 2:
            return False
        prefix, prop = parts
        if allowed_prefixes and prefix not in allowed_prefixes:
            logging.warning(f"Field prefix '{prefix}' not in allowed list {allowed_prefixes}")
            return False
        all_props = set()
        for label_info in schema.get("node_labels", {}).values():
            all_props.update(label_info.get("properties", []))
        for rel_info in schema.get("relationship_types", {}).values():
            all_props.update(rel_info.get("properties", []))
        if prop not in all_props:
            logging.warning(f"Property '{prop}' not in schema")
            return False
        return True

    def _validate_where_clause(
        self, 
        where: str, 
        query_type: str, 
        schema: Dict
    ) -> bool:
        """Validate WHERE clause format."""
        if not where:
            return False
        allowed_vars = {
            "neighbor_query": ["nbr", "firstRel"],
            "common_neighbor": ["C", "rA", "rB"],
            "path_query": ["hops"],
            "filter_query": ["n", "r"]
        }
        vars_for_type = allowed_vars.get(query_type, [])
        if not any(var in where for var in vars_for_type):
            logging.warning(f"WHERE clause '{where}' variables not valid for query type {query_type}")
            return False
        operators = [">", "<", ">=", "<=", "=", "!=", "IN", "CONTAINS"]
        if not any(op in where for op in operators):
            logging.warning(f"WHERE clause '{where}' missing operator")
            return False
        return True

    def _validate_aggregate_field(
        self, 
        field: str, 
        query_type: str, 
        schema: Dict
    ) -> bool:
        """Validate aggregate field."""
        allowed_prefixes = {
            "neighbor_query": ["rel", "neighbor", "firstRel", "nbr"],
            "common_neighbor": ["rA", "rB", "commonNeighbor", "C"],
            "filter_query": ["rel", "node", "r", "n"]
        }
        
        prefixes = allowed_prefixes.get(query_type, [])
        return self._validate_order_field(field, prefixes, schema)
    def _fallback_extract(self, question: str, query_type: str) -> Dict:
        """Rule-based fallback extraction (returns params + modifiers)."""
        schema = self.schema_analyzer.get_schema()
        label = ""
        for l in schema["node_labels"].keys():
            if l.lower() in question.lower():
                label = l
                break
        if not label:
            label = list(schema["node_labels"].keys())[0]
        
        label_info = schema["node_labels"].get(label, {})
        props = label_info.get("properties", [])
        key = "id"
        for k in ["node_key", "id", "acct_id", "userId", "user_id", "name"]:
            if k in props:
                key = k
                break
        
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        names = re.findall(name_pattern, question)
        id_values = re.findall(r'\b([a-zA-Z]+\d+|[a-zA-Z]+_\d+)\b', question)
        values = names if names else id_values
        
        params = {
            "label": label,
            "key": key
        }
        
        if query_type in ["node_lookup", "neighbor_query", "subgraph", "filter_query"]:
            params["value"] = values[0] if values else None
        elif query_type == "subgraph_by_nodes":
            params["values"] = values if values else []
        elif query_type in ["common_neighbor", "path_query"]:
            params["v1"] = values[0] if len(values) > 0 else None
            params["v2"] = values[1] if len(values) > 1 else None
        elif query_type == "global_stats":
            for field in ["country", "city", "type"]:
                if field in props:
                    params["group_by"] = field
                    break
        
        hop_match = re.search(r'(\d+)\s*[跳度层]|(\d+)\s*hops?', question, re.I)
        if hop_match:
            params["hops"] = int(hop_match.group(1) or hop_match.group(2))
        
        modifiers = {}
        ql = question.lower()
        if any(kw in ql for kw in ["统计", "数量", "多少", "总和", "平均", "count", "total", "sum", "average", "avg"]):
            if any(kw in ql for kw in ["数量", "多少", "count", "how many"]):
                modifiers["aggregate"] = "count"
            elif any(kw in ql for kw in ["总和", "总计", "sum", "total"]):
                modifiers["aggregate"] = "sum"
            elif any(kw in ql for kw in ["平均", "average", "avg"]):
                modifiers["aggregate"] = "avg"
        
        if any(kw in ql for kw in ["排序", "从大到小", "从小到大", "最大", "最小", "sort", "desc", "asc", "largest", "smallest"]):
            if any(kw in ql for kw in ["从大到小", "最大", "desc", "largest", "highest"]):
                modifiers["order_direction"] = "DESC"
            else:
                modifiers["order_direction"] = "ASC"
        
        limit_match = re.search(r'前(\d+)|最多(\d+)|(\d+)个|top\s*(\d+)|first\s*(\d+)|limit\s*(\d+)', question, re.I)
        if limit_match and not any(kw in ql for kw in ["所有", "全部", "all"]):
            limit_num = (limit_match.group(1) or limit_match.group(2) or limit_match.group(3) or
                        limit_match.group(4) or limit_match.group(5) or limit_match.group(6))
            modifiers["limit"] = int(limit_num)
        
        return {
            "params": params,
            "modifiers": modifiers
        }
    
    def _validate_params(self, params: Dict, template: Dict) -> None:
        """Validate required parameters."""
        for req in template["required_params"]:
            if req not in params or params[req] is None:
                raise ValueError(f"Missing required parameter: {req}")


# ==========================================
# 6. Query executor
# ==========================================
class QueryExecutor:
    """Execute query templates (with aggregation support)."""
    
    def __init__(self, client: Neo4jGraphClient):
        self.client = client
    
    def _apply_aggregation(self, results: List[JsonDict], aggregate_type: str, aggregate_field: Optional[str] = None) -> List[JsonDict]:
        """
        Apply aggregation to query results.
        aggregate_type: count, sum, avg, max, min.
        aggregate_field: required for non-count.
        """
        aggregate_type = aggregate_type.lower()
        
        if not results:
            return [{"aggregate_type": aggregate_type, "value": 0}]
        
        if aggregate_type == "count":
            return [{"aggregate_type": "count", "value": len(results)}]
        
        if not aggregate_field:
            return [{"error": f"Aggregate type {aggregate_type} requires aggregate_field"}]
        
        values = []
        for item in results:
            try:
                value = item
                for key in aggregate_field.split('.'):
                    value = value.get(key, {})
                if isinstance(value, (int, float)):
                    values.append(value)
            except (AttributeError, TypeError):
                continue
        
        if not values:
            return [{"aggregate_type": aggregate_type, "value": None, "note": "No valid numeric values found"}]
        
        if aggregate_type == "sum":
            result_value = sum(values)
        elif aggregate_type == "avg":
            result_value = sum(values) / len(values)
        elif aggregate_type == "max":
            result_value = max(values)
        elif aggregate_type == "min":
            result_value = min(values)
        else:
            return [{"error": f"Unknown aggregate type: {aggregate_type}"}]
        
        return [{
            "aggregate_type": aggregate_type,
            "field": aggregate_field,
            "value": result_value,
            "count": len(values)
        }]
    
    def execute(self, query_type: str, params: Dict) -> Dict:
        """Execute query by type and params (supports aggregation)."""
        template = QUERY_TEMPLATES[query_type]
        method_name = template["method"]
        method = getattr(self.client, method_name)
        aggregate_type = params.pop("aggregate", None)
        aggregate_field = params.pop("aggregate_field", None)
        
        try:
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
                if aggregate_type:
                    results = self._apply_aggregation(results, aggregate_type, aggregate_field)
            
            else:
                return {"success": False, "error": f"Unknown query type: {query_type}"}
            
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
# 7. Main engine (dual-LLM + schema)
# ==========================================

class NaturalLanguageQueryEngine:
    """Natural language query engine (dual LLM + schema tools)."""
    
    def __init__(self, db_client: Neo4jGraphClient, reasoner: Reasoner):
        """
        Args:
            db_client: Neo4j client.
            reasoner: Reasoner for LLM calls.
        """
        self.client = db_client
        self.reasoner = reasoner
        self.schema_analyzer = SchemaAnalyzer(db_client)
        self.type_classifier = QueryTypeClassifier(reasoner)
        self.param_extractor = ParameterExtractor(reasoner, self.schema_analyzer)
        self.executor = QueryExecutor(db_client)
    
    def initialize(self):
        """Load schema."""
        print("\n🔍 Analyzing graph database schema...")
        schema = self.schema_analyzer.get_schema()
        print("✅ Schema loaded")
        print(f"   Node types: {list(schema['node_labels'].keys())}")
        print(f"   Relationship types: {list(schema['relationship_types'].keys())}\n")
    
    def ask(self, question: str) -> Dict:
        """
        Main entry: process natural language question.
        Steps: 1) LLM1 classify type 2) Get schema 3) LLM2 extract params 4) Execute.
        """
        print(f"\n💬 Question: {question}")
        print("=" * 80)
        
        print("🤖 LLM1 classifying query type...")
        query_type = self.type_classifier.classify(question)
        print(f"   Query type: {query_type}")
        print(f"   Description: {QUERY_TEMPLATES[query_type]['description']}")
        
        print("\n📊 Loading graph schema...")
        schema_info = self.schema_analyzer.format_for_llm()
        print("   Schema loaded")
        
        if os.environ.get("DEBUG_SCHEMA"):
            print("\n" + "="*80)
            print(schema_info)
            print("="*80)
        
        print("\n🤖 LLM2 extracting parameters...")
        try:
            params = self.param_extractor.extract(question, query_type)
            print(f"   Params: {json.dumps(params, ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"   ❌ Parameter extraction failed: {e}")
            return {"success": False, "error": f"Parameter extraction failed: {e}"}
        
        print("\n⚙️  Executing query...")
        result = self.executor.execute(query_type, params)
        
        if result["success"]:
            print(f"✅ Query succeeded; {result['count']} result(s)\n")
            for i, item in enumerate(result["results"][:3], 1):
                print(f"{i}. {item}")
            if result["count"] > 3:
                print(f"... and {result['count'] - 3} more")
        else:
            print(f"❌ Query failed: {result['error']}")
        
        return result


# ==========================================
# 8. Interactive CLI
# ==========================================

def main():
    """CLI entry."""
    print("=" * 80)
    print("🚀 Neo4j natural language query engine (dual-LLM)")
    print("=" * 80)
    
    uri = input("\nNeo4j URI (default bolt://localhost:7687): ").strip() or "bolt://localhost:7687"
    user = input("Username (default neo4j): ").strip() or "neo4j"
    password = input("Password: ").strip()
    
    try:
        config = Neo4jConfig(uri=uri, user=user, password=password)
        db_client = Neo4jGraphClient(config)
        
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
        
        engine = NaturalLanguageQueryEngine(db_client, reasoner)
        engine.initialize()
        
        print("\n✨ Connected. Type 'exit' to quit, 'help' for examples")
        print("=" * 80)
        
        while True:
            question = input("\n❓ Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', '退出']:
                print("\n👋 Bye!")
                break
            
            if question.lower() == 'help':
                print("\n📚 Example questions:")
                for qtype, info in QUERY_TEMPLATES.items():
                    print(f"  - {info['example']}")
                continue
            
            engine.ask(question)
        
        db_client.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
