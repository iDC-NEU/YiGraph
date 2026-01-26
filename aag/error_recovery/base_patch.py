"""
修复补丁基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import json


class RepairPatch(ABC):
    """
    修复补丁基类
    所有修复补丁都应该继承这个类
    """
    
    def __init__(self, error_info: Dict[str, Any], context: Dict[str, Any]):
        self.error_info = error_info
        self.context = context
    
    @abstractmethod
    def enhance_prompt(self, original_prompt: str, **kwargs) -> str:
        """
        增强原始提示词
        
        Args:
            original_prompt: 原始提示词
            **kwargs: 其他参数
            
        Returns:
            增强后的提示词
        """
        pass
    
    @abstractmethod
    def get_patch_metadata(self) -> Dict[str, Any]:
        """
        获取补丁元数据（用于日志和调试）
        """
        pass


class ParameterExtractionPatch(RepairPatch):
    """
    参数提取错误的修复补丁
    """
    
    def enhance_prompt(self, original_prompt: str, **kwargs) -> str:
        """在原有提示词基础上添加错误信息和修复指导"""
        error_message = self.error_info.get("error", "未知错误")
        previous_parameters = self.error_info.get("previous_parameters", {})
        
        repair_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 修复模式：之前的参数提取失败了
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【之前的尝试】
提取的参数：{json.dumps(previous_parameters, ensure_ascii=False, indent=2)}

【错误信息】
{error_message}

【修复要求】
1. 仔细分析错误原因（可能是参数类型不匹配、缺少必需参数、参数值不合理等）
2. 根据工具描述和图schema重新提取正确的参数
3. 确保所有必需参数都已提供，且类型和格式正确

请重新提取参数：
"""
        return original_prompt + repair_section
    
    def get_patch_metadata(self) -> Dict[str, Any]:
        return {
            "patch_type": "parameter_extraction",
            "error_type": self.error_info.get("error_type", "unknown")
        }


class PostprocessCodePatch(RepairPatch):
    """
    后处理代码错误的修复补丁
    """
    
    def enhance_prompt(self, original_prompt: str, **kwargs) -> str:
        """增强代码生成提示词，添加错误分析和修复指导"""
        error_message = self.error_info.get("error", "未知错误")
        error_type = self.error_info.get("error_type", "RuntimeError")
        error_location = self.error_info.get("location", "")
        previous_code = self.context.get("previous_code", "")
        algorithm_result = self.context.get("algorithm_result", {})
        
        repair_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 修复模式：之前的后处理代码执行失败了
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【之前的代码】
```python
{previous_code}
```

【执行错误】
错误类型：{error_type}
错误信息：{error_message}
错误位置：{error_location}

【算法执行结果（用于参考）】
{json.dumps(algorithm_result, ensure_ascii=False, indent=2) if algorithm_result else "无"}

【修复要求】
1. 分析错误原因：
   - 如果是 NameError：检查变量名是否正确
   - 如果是 TypeError：检查数据类型是否匹配
   - 如果是 KeyError：检查字典键是否存在
   - 如果是逻辑错误：检查代码逻辑是否正确
2. 修复代码，确保：
   - 所有变量名正确
   - 数据类型匹配
   - 能正确处理算法结果
   - 输出格式符合要求

请提供修复后的代码：
"""
        return original_prompt + repair_section
    
    def get_patch_metadata(self) -> Dict[str, Any]:
        return {
            "patch_type": "postprocess_code",
            "error_type": self.error_info.get("error_type", "unknown"),
            "error_location": self.error_info.get("location", "")
        }


class DependencyRecognitionPatch(RepairPatch):
    """
    数据依赖识别错误的修复补丁
    """
    
    def enhance_prompt(self, original_prompt: str, **kwargs) -> str:
        """增强依赖识别提示词"""
        error_message = self.error_info.get("error", "未知错误")
        previous_context = self.context.get("previous_dependency_context", {})
        dependency_parents = self.context.get("data_dependency_parents", [])
        
        # 格式化依赖父步骤信息
        parents_info = self._format_dependency_parents(dependency_parents)
        
        repair_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 修复模式：之前的数据依赖识别失败了
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【之前的识别结果】
{json.dumps(previous_context, ensure_ascii=False, indent=2)}

【错误信息】
{error_message}

【上游依赖步骤】
{parents_info}

【修复要求】
1. 分析为什么之前的依赖识别失败
2. 重新识别需要哪些图依赖和参数依赖
3. 确保依赖关系正确，符合算法要求

请重新识别数据依赖：
"""
        return original_prompt + repair_section
    
    def _format_dependency_parents(self, parents):
        """格式化依赖父步骤信息"""
        if not parents:
            return "无上游依赖"
        formatted = []
        for parent in parents:
            formatted.append(f"- 步骤 {parent.step_id}: {parent.question}")
        return "\n".join(formatted)
    
    def get_patch_metadata(self) -> Dict[str, Any]:
        return {
            "patch_type": "dependency_recognition",
            "error_type": self.error_info.get("error_type", "unknown")
        }


class NumericAnalysisCodePatch(RepairPatch):
    """
    数值分析代码错误的修复补丁
    """
    
    def enhance_prompt(self, original_prompt: str, **kwargs) -> str:
        """增强数值分析代码生成提示词"""
        error_message = self.error_info.get("error", "未知错误")
        error_type = self.error_info.get("error_type", "RuntimeError")
        traceback_info = self.error_info.get("traceback", "")
        previous_code = self.context.get("previous_code", "")
        execution_data = self.context.get("execution_data", {})
        
        repair_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 修复模式：之前的数值分析代码执行失败了
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【之前的代码】
```python
{previous_code}
```

【执行数据】
{json.dumps(execution_data, ensure_ascii=False, indent=2) if execution_data else "无"}

【执行错误】
错误类型：{error_type}
错误信息：{error_message}
堆栈信息：
{traceback_info if traceback_info else "无"}

【修复要求】
1. 分析错误原因（变量名、类型、逻辑等）
2. 修复代码，确保能正确处理输入数据
3. 保持输出格式正确

请提供修复后的代码：
"""
        return original_prompt + repair_section
    
    def get_patch_metadata(self) -> Dict[str, Any]:
        return {
            "patch_type": "numeric_analysis_code",
            "error_type": self.error_info.get("error_type", "unknown")
        }
