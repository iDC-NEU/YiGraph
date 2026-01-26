"""
修复补丁工厂
根据错误类型创建相应的补丁
"""

from typing import Dict, Any, Optional
from .base_patch import (
    RepairPatch,
    ParameterExtractionPatch,
    PostprocessCodePatch,
    DependencyRecognitionPatch,
    NumericAnalysisCodePatch
)


class RepairPatchFactory:
    """
    修复补丁工厂
    根据错误类型创建相应的补丁
    """
    
    _patch_registry = {
        "parameter_extraction": ParameterExtractionPatch,
        "postprocess_code": PostprocessCodePatch,
        "dependency_recognition": DependencyRecognitionPatch,
        "numeric_analysis_code": NumericAnalysisCodePatch,
    }
    
    @classmethod
    def create_patch(
        cls,
        error_type: str,
        error_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[RepairPatch]:
        """
        创建修复补丁
        
        Args:
            error_type: 错误类型
            error_info: 错误信息
            context: 上下文信息
            
        Returns:
            修复补丁实例，如果类型不存在则返回None
        """
        patch_class = cls._patch_registry.get(error_type)
        if patch_class:
            return patch_class(error_info, context)
        return None
    
    @classmethod
    def register_patch(cls, error_type: str, patch_class: type):
        """
        注册新的补丁类型（支持扩展）
        
        Args:
            error_type: 错误类型名称
            patch_class: 补丁类（必须继承 RepairPatch）
        """
        if not issubclass(patch_class, RepairPatch):
            raise ValueError(f"补丁类必须继承 RepairPatch: {patch_class}")
        cls._patch_registry[error_type] = patch_class
    
    @classmethod
    def list_registered_patches(cls) -> list:
        """列出所有已注册的补丁类型"""
        return list(cls._patch_registry.keys())
