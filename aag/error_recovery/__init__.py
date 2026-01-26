"""
错误纠正模块
提供错误恢复和修复补丁功能
"""

from .error_recovery_module import ErrorRecoveryModule
from .error_types import (
    DependencyErrorType,
    DependencyErrorInfo,
    DependencyResolutionError
)
from .patch_factory import RepairPatchFactory
from .base_patch import RepairPatch
from .retry_helper import should_retry, prepare_error_info, classify_error_type

__all__ = [
    "ErrorRecoveryModule",
    "DependencyErrorType",
    "DependencyErrorInfo",
    "DependencyResolutionError",
    "RepairPatchFactory",
    "RepairPatch",
    "should_retry",
    "prepare_error_info",
    "classify_error_type",
]
