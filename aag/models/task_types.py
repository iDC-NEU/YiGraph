from enum import Enum


class GraphAnalysisType(Enum):
    GRAPH_ALGORITHM = "graph_algorithm"
    NUMERIC_ANALYSIS = "numeric_analysis"

class GraphAnalysisSubType(Enum):
    GRAPH_ALGORITHM = "graph_algorithm"
    POST_PROCESSING = "post_processing"
    LLM_REASONING = "llm_reasoning"
    NUMERIC_COMPUTATION = "numeric_computation"
    SUBGRAPH_EXTRACTION = "subgraph_extraction"