"""
错误恢复模块
提供错误恢复和重试功能
"""

import logging
from typing import Dict, Any, Optional
from aag.models.graph_workflow_dag import WorkflowStep
from aag.reasoner.model_deployment import Reasoner
from .patch_factory import RepairPatchFactory
from .error_types import DependencyErrorType, DependencyErrorInfo, DependencyResolutionError

logger = logging.getLogger(__name__)


class ErrorRecoveryModule:
    """
    错误恢复模块
    负责捕获错误、创建修复补丁、重试失败的操作
    """
    
    def __init__(self, reasoner: Reasoner, max_retries: int = 3):
        """
        初始化错误恢复模块
        
        Args:
            reasoner: Reasoner 实例，用于调用 LLM
            max_retries: 最大重试次数，默认 3 次
        """
        self.reasoner = reasoner
        self.max_retries = max_retries
        self.patch_factory = RepairPatchFactory()
        self.retry_count = {}  # step_id -> retry_count
    
    def _get_retry_count(self, step_id: str) -> int:
        """获取当前步骤的重试次数"""
        return self.retry_count.get(step_id, 0)
    
    def _increment_retry_count(self, step_id: str):
        """增加重试次数"""
        self.retry_count[step_id] = self.retry_count.get(step_id, 0) + 1
    
    def _reset_retry_count(self, step_id: str):
        """重置重试次数"""
        self.retry_count[step_id] = 0
    
    def can_retry(self, step_id: str) -> bool:
        """检查是否可以重试"""
        return self._get_retry_count(step_id) < self.max_retries
    
    async def recover_parameter_error(
        self,
        step: WorkflowStep,
        error_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        修复参数提取错误
        
        Args:
            step: 当前步骤
            error_info: 错误信息
            context: 上下文信息（包含 tool_description, vertex_schema, edge_schema 等）
            
        Returns:
            修复后的参数提取结果
        """
        # 创建补丁
        patch = self.patch_factory.create_patch(
            error_type="parameter_extraction",
            error_info=error_info,
            context=context
        )
        
        if not patch:
            raise ValueError("无法创建参数提取修复补丁")
        
        logger.info(f"🔧 应用修复补丁: {patch.get_patch_metadata()}")
        
        # 调用原有的参数提取方法，传入补丁
        # 注意：这里需要 reasoner 的方法支持 repair_patch 参数
        # 如果 reasoner 方法还不支持，可以先构建增强后的提示词，然后调用
        return await self._extract_parameters_with_patch(
            step=step,
            patch=patch,
            context=context
        )
    
    async def _extract_parameters_with_patch(
        self,
        step: WorkflowStep,
        patch: Any,  # RepairPatch
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用补丁增强提示词后提取参数
        
        这是一个辅助方法，用于在 reasoner 方法支持补丁之前
        手动构建增强后的提示词并调用
        """
        # 构建原始提示词（这里需要根据实际的 reasoner 方法调整）
        # 暂时返回一个占位符，实际实现需要调用 reasoner 的相应方法
        logger.warning("⚠️ _extract_parameters_with_patch 需要根据实际 reasoner 接口实现")
        return {}
    
    async def recover_postprocess_code_error(
        self,
        step: WorkflowStep,
        error_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        修复后处理代码错误
        
        Args:
            step: 当前步骤
            error_info: 错误信息
            context: 上下文信息（包含 previous_code, algorithm_result 等）
            
        Returns:
            修复后的代码
        """
        patch = self.patch_factory.create_patch(
            error_type="postprocess_code",
            error_info=error_info,
            context=context
        )
        
        if not patch:
            raise ValueError("无法创建后处理代码修复补丁")
        
        logger.info(f"🔧 应用修复补丁: {patch.get_patch_metadata()}")
        
        # 调用原有的代码生成方法，传入补丁
        return await self._generate_code_with_patch(
            step=step,
            patch=patch,
            context=context
        )
    
    async def _generate_code_with_patch(
        self,
        step: WorkflowStep,
        patch: Any,
        context: Dict[str, Any]
    ) -> str:
        """使用补丁增强提示词后生成代码"""
        logger.warning("⚠️ _generate_code_with_patch 需要根据实际 reasoner 接口实现")
        return ""
    
    async def recover_dependency_error(
        self,
        step: WorkflowStep,
        error_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        修复数据依赖识别错误
        
        Args:
            step: 当前步骤
            error_info: 错误信息（可能包含 dependency_error_info）
            context: 上下文信息（包含 data_dependency_parents, tool_metadata 等）
            
        Returns:
            修复后的依赖上下文
        """
        patch = self.patch_factory.create_patch(
            error_type="dependency_recognition",
            error_info=error_info,
            context=context
        )
        
        if not patch:
            raise ValueError("无法创建数据依赖识别修复补丁")
        
        logger.info(f"🔧 应用修复补丁: {patch.get_patch_metadata()}")
        
        # 调用依赖解析方法，传入补丁
        return await self._resolve_dependencies_with_patch(
            step=step,
            patch=patch,
            context=context
        )
    
    async def _resolve_dependencies_with_patch(
        self,
        step: WorkflowStep,
        patch: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用补丁增强提示词后解析依赖"""
        logger.warning("⚠️ _resolve_dependencies_with_patch 需要根据实际接口实现")
        return {
            "graph_dependencies": [],
            "parameter_dependencies": [],
            "graph_input_adapter_result": None,
            "parameter_input_adapter_result": None
        }
    
    async def recover_numeric_code_error(
        self,
        step: WorkflowStep,
        error_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        修复数值分析代码错误
        
        Args:
            step: 当前步骤
            error_info: 错误信息
            context: 上下文信息（包含 previous_code, execution_data 等）
            
        Returns:
            修复后的代码
        """
        patch = self.patch_factory.create_patch(
            error_type="numeric_analysis_code",
            error_info=error_info,
            context=context
        )
        
        if not patch:
            raise ValueError("无法创建数值分析代码修复补丁")
        
        logger.info(f"🔧 应用修复补丁: {patch.get_patch_metadata()}")
        
        # 调用代码生成方法，传入补丁
        return await self._generate_numeric_code_with_patch(
            step=step,
            patch=patch,
            context=context
        )
    
    async def _generate_numeric_code_with_patch(
        self,
        step: WorkflowStep,
        patch: Any,
        context: Dict[str, Any]
    ) -> str:
        """使用补丁增强提示词后生成数值分析代码"""
        logger.warning("⚠️ _generate_numeric_code_with_patch 需要根据实际 reasoner 接口实现")
        return ""
