"""
重试辅助工具
提供简洁的错误恢复辅助方法
"""

import logging
from typing import Callable, Any, Dict, Optional
import traceback

logger = logging.getLogger(__name__)


def should_retry(retry_count: int, max_retries: int, error_recovery) -> bool:
    """判断是否应该重试"""
    return retry_count < max_retries and error_recovery is not None


def prepare_error_info(error: Exception, result: Any = None) -> Dict[str, Any]:
    """准备错误信息"""
    error_msg = str(error)
    error_type_name = type(error).__name__
    
    error_info = {
        "error": error_msg,
        "error_type": error_type_name,
    }
    
    # 如果有 traceback，添加进去
    try:
        error_info["traceback"] = traceback.format_exc()
    except:
        pass
    
    # 如果 result 是字典且包含错误信息，提取出来
    if isinstance(result, dict):
        if "error" in result:
            error_info["result_error"] = result.get("error")
        if "traceback" in result:
            error_info["traceback"] = result.get("traceback")
        if "error_type" in result:
            error_info["error_type"] = result.get("error_type")
    
    return error_info


def classify_error_type(error_msg: str) -> str:
    """根据错误信息分类错误类型（code 或 parameter）"""
    error_msg_lower = error_msg.lower()
    
    # 代码相关错误
    code_keywords = ["nameerror", "typeerror", "keyerror", "attributeerror", 
                     "syntax", "indentation", "line", "code", "function"]
    if any(keyword in error_msg_lower for keyword in code_keywords):
        return "code"
    
    # 参数相关错误
    param_keywords = ["parameter", "argument", "missing", "required", "invalid"]
    if any(keyword in error_msg_lower for keyword in param_keywords):
        return "parameter"
    
    return "unknown"
