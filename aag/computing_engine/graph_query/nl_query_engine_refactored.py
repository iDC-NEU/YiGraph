"""
NL Query Engine Refactored Version
Split parameter extractors by query type, each extractor only contains prompts and examples for that type

Refactoring Goals:
1. Prompt simplification: Reduce prompt length by over 80% for each extractor
2. Accuracy improvement: LLM only needs to focus on current query type, reducing confusion
3. Maintainability: Each extractor is independently maintained
4. Testability: Can test each query type independently
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any

# add gjq - Import model deployment for LLM calls
from aag.reasoner.model_deployment import Reasoner
from aag.config.engine_config import ReasonerConfig

# 类型别名
JsonDict = Dict[str, Any]

# ==========================================
# 查询模板定义
# ==========================================
QUERY_TEMPLATES = {
    "node_lookup": {
        "description": "Find a single node by unique key, or filter multiple nodes by property conditions",
        "method": "get_node_by_unique_key / filter_nodes_by_properties",
        "required_params": ["label"],
        "optional_params": ["key", "value", "conditions", "return_fields"],
        "example": "Find user Collins Steven or Query customers living in US with state VT",
        "note": "Supports two modes: 1) Single node exact lookup (requires key+value, uses get_node_by_unique_key) 2) Multiple node property filtering (requires conditions, uses filter_nodes_by_properties)"
    },
    "relationship_filter": {
        "description": "Filter relationships based on relationship property conditions, return relationships and related node information that meet the conditions",
        "method": "filter_relationships",
        "required_params": ["rel_type"],
        "optional_params": ["start_label", "end_label", "rel_conditions", "return_fields", "aggregate"],
        "example": "Find transactions with amount greater than 400 or List the count of transactions where is_sar is False",
        "note": "Used for relationship property filtering, relationship statistics, etc., not for node queries or path queries"
    },
    "aggregation_query": {
        "description": "Aggregation statistical query, supports grouping by node or property, calculates COUNT/SUM/AVG etc., used for ranking and TOP-N queries",
        "method": "aggregation_query",
        "required_params": ["aggregate_type"],
        "optional_params": ["group_by_node", "group_by_property", "node_label", "rel_type", "direction", "aggregate_field", "return_fields"],
        "example": "Count the number of transactions per account or Calculate the total amount of outgoing transactions per account or Count the number of accounts under each branch_id",
        "note": "Used for GROUP BY + aggregation function scenarios, the difference from relationship_filter is: aggregation_query returns grouped aggregation results, relationship_filter returns raw data or global aggregation"
    },
    "neighbor_query": {
        "description": "Query neighbors or N-hop relationships of a node, supports returning detailed information for each edge",
        "method": "neighbors_n_hop",
        "required_params": ["label", "key", "value"],
        "optional_params": ["hops", "rel_type", "direction", "return_fields"],
        "example": "Query neighbors of Collins Steven or Query all transaction details where Collins Steven is the source account",
        "note": "When hops=1 and detailed information for each edge is needed, use return_fields to specify fields to avoid data loss from path aggregation"
    },
    "path_query": {
        "description": "Query paths between two nodes",
        "method": "paths_between",
        "required_params": ["label", "key", "v1", "v2"],
        "optional_params": ["rel_type", "direction", "min_hops", "max_hops"],
        "example": "Path from Collins Steven to Nunez Mitchell"
    },
    "common_neighbor": {
        "description": "Query common neighbors of two nodes, supports relationship property filtering",
        "method": "common_neighbors / common_neighbors_with_rel_filter",
        "required_params": ["label", "key", "v1", "v2"],
        "optional_params": ["rel_type", "direction", "rel_conditions", "return_fields"],
        "example": "Common neighbors of Collins Steven and Nunez Mitchell or Find common transaction neighbors of Steven Collins and Samantha Cook where transaction amounts are all greater than 400",
        "note": "Supports two modes: 1) Simple common neighbor query (no rel_conditions, uses common_neighbors) 2) With relationship property filtering (has rel_conditions, uses common_neighbors_with_rel_filter)"
    },
    "subgraph": {
        "description": "Extract subgraph (supports three modes)",
        "method": "subgraph_extract / subgraph_extract_by_rel_filter",
        "required_params": [],  # Varies by mode
        "optional_params": ["label", "key", "value", "hops", "rel_type", "direction", "limit_paths", "rel_conditions", "start_label", "end_label", "limit"],
        "example": "Subgraph around Collins Steven within 2 hops or Extract subgraph of all transactions on 2025-05-01 or Extract subgraph of transactions with amounts between 300 and 500",
        "note": "Supports three modes: 1) Single center node (requires label+key+value, uses subgraph_extract) 2) Relationship property filtering (requires rel_type+rel_conditions, uses subgraph_extract_by_rel_filter)"
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
# 通用修饰符定义
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
        "description": "Filter conditions (node or relationship properties)",
        "format": "n.property operator value or r.property operator value",
        "example": "n.balance > 1000, r.amount > 500"
    },
    "aggregate": {
        "description": "Aggregation function",
        "values": ["COUNT", "SUM", "AVG", "MAX", "MIN"],
        "note": "Returns statistical results rather than raw data after applying aggregation"
    },
    "aggregate_field": {
        "description": "Aggregation field (required only when aggregate is not COUNT)",
        "format": "rel.property_name or neighbor.property_name",
        "example": "rel.base_amt"
    }
}

# ==========================================
# LLM 接口抽象
# ==========================================
# add gjq - Modified to use Reasoner from model_deployment
class LLMInterface:
    """LLM 调用接口 - 使用 Reasoner 统一调用"""
    
    def __init__(self, reasoner: Reasoner = None):
        """
        初始化 LLM 接口
        
        Args:
            reasoner: Reasoner 实例，如果为 None 则创建默认实例
        """
        if reasoner is None:
            # add gjq - 创建默认的 Reasoner 配置
            config = ReasonerConfig()
            self.reasoner = Reasoner(config)
        else:
            self.reasoner = reasoner
    
    def call(self, prompt: str, **kwargs) -> str:
        """调用 LLM，返回文本响应"""
        # add gjq - 使用 Reasoner 的 generate_response 方法
        return self.reasoner.generate_response(prompt)

# ==========================================
# 1. Base Class: BaseParameterExtractor
# ==========================================
# add gjq - Simplified base class, removed _build_prompt abstract method
class BaseParameterExtractor:
    """Parameter extractor base class"""
    
    def __init__(self, llm: LLMInterface, schema_analyzer: 'SchemaAnalyzer'):
        """
        Initialize parameter extractor
        
        Args:
            llm: LLM interface instance
            schema_analyzer: Schema analyzer instance (from nl_query_engine)
        """
        self.llm = llm
        self.schema_analyzer = schema_analyzer
    
    # add gjq - Modified extract method to use reasoner's specific prompt methods
    def extract(self, question: str, query_type: str) -> Dict:
        """
        Main process for parameter extraction
        
        Args:
            question: User question
            query_type: Query type
            
        Returns:
            Parameter dictionary (merged params + modifiers)
        """
        # 1. Get Schema information
        schema = self.schema_analyzer.get_schema()
        schema_info = self.schema_analyzer.format_for_llm()
        
        # 2. Get template information
        template = QUERY_TEMPLATES[query_type]
        
        # 3. Call LLM using reasoner's query-type-specific method
        # add gjq - 使用 reasoner 的查询类型特定方法替代通用方法
        result = self._call_llm_for_extraction(question, query_type, template, schema_info)
        
        # 4. Validate parameters
        self._validate_params(result.get("params", {}), template)
        
        # 5. Clean modifiers
        cleaned_modifiers = self._validate_and_clean_modifiers(
            result.get("modifiers", {}),
            query_type,
            schema
        )
        
        # 6. Merge parameters
        final_params = {**result.get("params", {}), **cleaned_modifiers}
        
        logging.info(f"[{self.__class__.__name__}] Extracted parameters: {final_params}")
        
        return final_params
    
    # add gjq - Abstract method for calling LLM, to be overridden by subclasses
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """
        Call LLM for parameter extraction - to be overridden by subclasses
        
        Args:
            question: User question
            query_type: Query type
            template: Query template
            schema_info: Schema information
            
        Returns:
            Extracted parameters dictionary
        """
        # Default implementation uses generic extraction
        return self.llm.reasoner.nl_query_extract_params(
            question=question,
            query_type=query_type,
            template=template,
            schema_info=schema_info,
            query_modifiers=QUERY_MODIFIERS
        )
    
    # add gjq - Removed _build_prompt abstract method as we now use reasoner directly
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response (common logic)"""
        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parsing failed: {e}, attempting to clean response...")
            
            # Try to clean response
            cleaned_response = response.strip()
            
            # Remove code block markers
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Remove comments (// and /* */)
            cleaned_response = re.sub(r'//.*?$', '', cleaned_response, flags=re.MULTILINE)
            cleaned_response = re.sub(r'/\*.*?\*/', '', cleaned_response, flags=re.DOTALL)
            
            # Remove trailing commas
            cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
            
            try:
                result = json.loads(cleaned_response)
            except json.JSONDecodeError as e2:
                logging.warning(f"Still failed after cleaning: {e2}, attempting to extract JSON fragment...")
                
                # Try to extract JSON fragment
                match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if match:
                    json_str = match.group()
                    # Clean extracted JSON again
                    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError as e3:
                        logging.error(f"Final parsing failed: {e3}")
                        logging.error(f"Original response: {response}")
                        logging.error(f"Cleaned response: {json_str}")
                        raise ValueError(f"Unable to parse LLM response: {response}")
                else:
                    logging.error(f"Unable to extract JSON fragment")
                    logging.error(f"Original response: {response}")
                    raise ValueError(f"Unable to parse LLM response: {response}")
        
        return result
    
    def _validate_params(self, params: Dict, template: Dict) -> None:
        """Validate required parameters"""
        required_params = template.get("required_params", [])
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
    
    def _validate_and_clean_modifiers(self, modifiers: Dict, query_type: str, schema: Dict) -> Dict:
        """Validate and clean modifiers"""
        cleaned = {}
        
        # Clean aggregate_field: remove prefix
        if "aggregate_field" in modifiers:
            agg_field = modifiers["aggregate_field"]
            if "." in agg_field:
                cleaned["aggregate_field"] = agg_field.split(".")[-1]
            else:
                cleaned["aggregate_field"] = agg_field
        
        # ⚠️ CRITICAL FIX: Handle order_by parameter format errors
        # LLM may incorrectly merge sort field and direction into one string (e.g., "base_amt DESC")
        # Or return dictionary format (e.g., {"base_amt": "DESC"})
        if "order_by" in modifiers:
            order_by_value = modifiers["order_by"]
            
            # Case 1: Dictionary format {"field": "direction"}
            if isinstance(order_by_value, dict):
                if order_by_value:
                    # Extract field name and direction
                    field_name = list(order_by_value.keys())[0]
                    direction = order_by_value[field_name]
                    cleaned["order_by"] = field_name
                    cleaned["order_direction"] = direction.upper()
                    logging.info(f"Fixed order_by format: split dictionary {order_by_value} into field={field_name}, direction={direction}")
            
            # Case 2: String format "field DESC" or "field ASC"
            elif isinstance(order_by_value, str):
                # Check if contains direction keyword
                parts = order_by_value.strip().split()
                if len(parts) == 2 and parts[1].upper() in ["ASC", "DESC"]:
                    # Split field and direction
                    field_name = parts[0]
                    direction = parts[1].upper()
                    cleaned["order_by"] = field_name
                    cleaned["order_direction"] = direction
                    logging.info(f"Fixed order_by format: split string '{order_by_value}' into field={field_name}, direction={direction}")
                else:
                    # Normal field name
                    cleaned["order_by"] = order_by_value
            
            else:
                # Other types, copy directly
                cleaned["order_by"] = order_by_value
        
        # Copy order_direction (if not set during order_by processing)
        if "order_direction" in modifiers and "order_direction" not in cleaned:
            cleaned["order_direction"] = modifiers["order_direction"]
        
        # Copy other modifiers (including new neighbor_query modifiers)
        for key in ["limit", "where", "aggregate", "return_fields",
                    "neighbor_where", "rel_conditions", "return_distinct", "exclude_start", "return_path_length"]:
            if key in modifiers:
                cleaned[key] = modifiers[key]
        
        return cleaned


# ==========================================
# 2. Specific Extractors for Each Query Type
# ==========================================

# add gjq - Node lookup extractor
class NodeLookupExtractor(BaseParameterExtractor):
    """Node lookup parameter extractor"""
    
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """Call LLM using node_lookup specific prompt"""
        # add gjq - 调用 reasoner 的 node_lookup 专用方法
        from aag.reasoner.prompt_template.llm_prompt_en import nl_query_node_lookup_prompt
        prompt = nl_query_node_lookup_prompt.format(
            schema_info=schema_info,
            question=question
        )
        response = self.llm.reasoner.generate_response(prompt)
        return self._parse_response(response)


# add gjq - Relationship filter extractor
class RelationshipFilterExtractor(BaseParameterExtractor):
    """Relationship property filter parameter extractor"""
    
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """Call LLM using relationship_filter specific prompt"""
        # add gjq - 调用 reasoner 的 relationship_filter 专用方法
        from aag.reasoner.prompt_template.llm_prompt_en import nl_query_relationship_filter_prompt
        prompt = nl_query_relationship_filter_prompt.format(
            schema_info=schema_info,
            question=question
        )
        response = self.llm.reasoner.generate_response(prompt)
        return self._parse_response(response)


# add gjq - Aggregation query extractor  
class AggregationQueryExtractor(BaseParameterExtractor):
    """Aggregation query parameter extractor"""
    
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """Call LLM using aggregation_query specific prompt"""
        # add gjq - 调用 reasoner 的 aggregation_query 专用方法
        from aag.reasoner.prompt_template.llm_prompt_en import nl_query_aggregation_query_prompt
        prompt = nl_query_aggregation_query_prompt.format(
            schema_info=schema_info,
            question=question
        )
        response = self.llm.reasoner.generate_response(prompt)
        return self._parse_response(response)


# add gjq - Neighbor query extractor
class NeighborQueryExtractor(BaseParameterExtractor):
    """Neighbor query parameter extractor"""
    
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """Call LLM using neighbor_query specific prompt"""
        # add gjq - 调用 reasoner 的 neighbor_query 专用方法
        from aag.reasoner.prompt_template.llm_prompt_en import nl_query_neighbor_query_prompt
        prompt = nl_query_neighbor_query_prompt.format(
            schema_info=schema_info,
            question=question
        )
        response = self.llm.reasoner.generate_response(prompt)
        return self._parse_response(response)


# add gjq - Path query extractor
class PathQueryExtractor(BaseParameterExtractor):
    """Path query parameter extractor"""
    
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """Call LLM using path_query specific prompt"""
        # add gjq - 调用 reasoner 的 path_query 专用方法
        from aag.reasoner.prompt_template.llm_prompt_en import nl_query_path_query_prompt
        prompt = nl_query_path_query_prompt.format(
            schema_info=schema_info,
            question=question
        )
        response = self.llm.reasoner.generate_response(prompt)
        return self._parse_response(response)


# add gjq - Common neighbor extractor
class CommonNeighborExtractor(BaseParameterExtractor):
    """Common neighbor parameter extractor"""
    
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """Call LLM using common_neighbor specific prompt"""
        # add gjq - 调用 reasoner 的 common_neighbor 专用方法
        from aag.reasoner.prompt_template.llm_prompt_en import nl_query_common_neighbor_prompt
        prompt = nl_query_common_neighbor_prompt.format(
            schema_info=schema_info,
            question=question
        )
        response = self.llm.reasoner.generate_response(prompt)
        return self._parse_response(response)


# add gjq - Subgraph extractor
class SubgraphExtractor(BaseParameterExtractor):
    """Subgraph extraction parameter extractor"""
    
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """Call LLM using subgraph specific prompt"""
        # add gjq - 调用 reasoner 的 subgraph 专用方法
        from aag.reasoner.prompt_template.llm_prompt_en import nl_query_subgraph_prompt
        prompt = nl_query_subgraph_prompt.format(
            schema_info=schema_info,
            question=question
        )
        response = self.llm.reasoner.generate_response(prompt)
        return self._parse_response(response)


# add gjq - Subgraph by nodes extractor
class SubgraphByNodesExtractor(BaseParameterExtractor):
    """Multi-node subgraph extraction parameter extractor"""
    
    def _call_llm_for_extraction(self, question: str, query_type: str, template: Dict, schema_info: str) -> Dict:
        """Call LLM using subgraph_by_nodes specific prompt"""
        # add gjq - 调用 reasoner 的 subgraph_by_nodes 专用方法
        from aag.reasoner.prompt_template.llm_prompt_en import nl_query_subgraph_by_nodes_prompt
        prompt = nl_query_subgraph_by_nodes_prompt.format(
            schema_info=schema_info,
            question=question
        )
        response = self.llm.reasoner.generate_response(prompt)
        return self._parse_response(response)


# ==========================================
# 3. Router: ParameterExtractorRouter
# ==========================================
# add gjq - Router that dispatches to specific extractors
class ParameterExtractorRouter:
    """Parameter extractor router"""
    
    def __init__(self, llm: LLMInterface, schema_analyzer: 'SchemaAnalyzer'):
        """
        Initialize parameter extractor router
        
        Args:
            llm: LLM interface instance
            schema_analyzer: Schema analyzer instance (from nl_query_engine)
        """
        self.llm = llm
        self.schema_analyzer = schema_analyzer
        
        # add gjq - Register all extractors with their specific prompts
        self.extractors = {
            "node_lookup": NodeLookupExtractor(llm, schema_analyzer),
            "relationship_filter": RelationshipFilterExtractor(llm, schema_analyzer),
            "aggregation_query": AggregationQueryExtractor(llm, schema_analyzer),
            "neighbor_query": NeighborQueryExtractor(llm, schema_analyzer),
            "path_query": PathQueryExtractor(llm, schema_analyzer),
            "common_neighbor": CommonNeighborExtractor(llm, schema_analyzer),
            "subgraph": SubgraphExtractor(llm, schema_analyzer),
            "subgraph_by_nodes": SubgraphByNodesExtractor(llm, schema_analyzer),
        }
    
    def extract(self, question: str, query_type: str) -> Dict:
        """Route to corresponding extractor"""
        if query_type not in self.extractors:
            raise ValueError(f"Unknown query type: {query_type}")
        
        extractor = self.extractors[query_type]
        return extractor.extract(question, query_type)
