from enum import Enum


class GraphAnalysisType(Enum):
    GRAPH_ALGORITHM = "graph_algorithm"
    NUMERIC_ANALYSIS = "numeric_analysis"
    # add gjq: 添加图查询相关的任务类型（基于 templates.py 中的 QUERY_TEMPLATES）
    GRAPH_QUERY = "graph_query"  # 通用图查询类型
    NODE_LOOKUP = "node_lookup"  # 节点查找（单节点或多节点筛选）
    RELATIONSHIP_FILTER = "relationship_filter"  # 关系过滤查询
    AGGREGATION_QUERY = "aggregation_query"  # 聚合统计查询
    NEIGHBOR_QUERY = "neighbor_query"  # 邻居查询（N跳关系）
    PATH_QUERY = "path_query"  # 路径查询
    COMMON_NEIGHBOR = "common_neighbor"  # 公共邻居查询
    SUBGRAPH = "subgraph"  # 子图抽取
    SUBGRAPH_BY_NODES = "subgraph_by_nodes"  # 基于节点列表的子图抽取

class GraphAnalysisSubType(Enum):
    GRAPH_ALGORITHM = "graph_algorithm"
    POST_PROCESSING = "post_processing"
    LLM_REASONING = "llm_reasoning"
    NUMERIC_COMPUTATION = "numeric_computation"
    SUBGRAPH_EXTRACTION = "subgraph_extraction"