"""
错误类型定义
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional


class DependencyErrorType(Enum):
    """数据依赖错误的细粒度类型"""
    # Step 1 错误
    LLM_DEPENDENCY_ANALYSIS_FAILED = "llm_dependency_analysis_failed"
    INVALID_DEPENDENCY_TYPE = "invalid_dependency_type"
    OUTPUT_ID_NOT_FOUND = "output_id_not_found"
    FIELD_KEY_NOT_FOUND = "field_key_not_found"
    
    # Step 3.1 图依赖转换错误
    LLM_GRAPH_CONVERSION_CODE_FAILED = "llm_graph_conversion_code_failed"
    GRAPH_CONVERSION_CODE_EXEC_FAILED = "graph_conversion_code_exec_failed"
    TRANSFORM_GRAPH_FUNCTION_MISSING = "transform_graph_function_missing"
    GLOBAL_GRAPH_NOT_INITIALIZED = "global_graph_not_initialized"
    
    # Step 3.2 参数依赖转换错误
    LLM_PARAMETER_MAPPING_FAILED = "llm_parameter_mapping_failed"
    DEPENDENCY_FIELD_NOT_FOUND = "dependency_field_not_found"
    EXTRACT_CODE_EXEC_FAILED = "extract_code_exec_failed"
    
    # 通用错误
    UNKNOWN = "unknown"


@dataclass
class DependencyErrorInfo:
    """数据依赖错误的详细信息"""
    error_type: DependencyErrorType
    error_message: str
    error_location: str  # 如 "Step 1.1", "Step 3.1.2"
    parent_step_id: Optional[int] = None
    field_key: Optional[str] = None
    output_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None  # 错误发生时的上下文
    previous_result: Optional[Dict[str, Any]] = None  # 之前尝试的结果


class DependencyResolutionError(Exception):
    """数据依赖解析错误"""
    
    def __init__(
        self,
        error_type: DependencyErrorType,
        error_info: DependencyErrorInfo,
        location: str
    ):
        self.error_type = error_type
        self.error_info = error_info
        self.location = location
        super().__init__(f"[{location}] {error_info.error_message}")
