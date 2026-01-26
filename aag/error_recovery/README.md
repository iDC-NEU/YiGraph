# 错误纠正模块

## 概述

错误纠正模块提供错误恢复和修复补丁功能，用于在执行图算法流程时自动修复错误。

## 目录结构

```
error_recovery/
├── __init__.py                    # 导出主要接口
├── error_types.py                 # 错误类型定义
├── base_patch.py                  # 补丁基类和具体实现
├── patch_factory.py               # 补丁工厂
├── error_recovery_module.py       # 错误恢复模块主类
└── README.md                      # 本文档
```

## 核心组件

### 1. RepairPatch (修复补丁)

修复补丁用于增强原始提示词，添加错误信息和修复指导。

**已实现的补丁类型：**
- `ParameterExtractionPatch`: 参数提取错误修复
- `PostprocessCodePatch`: 后处理代码错误修复
- `DependencyRecognitionPatch`: 数据依赖识别错误修复
- `NumericAnalysisCodePatch`: 数值分析代码错误修复

### 2. ErrorRecoveryModule (错误恢复模块)

错误恢复模块负责：
- 捕获错误
- 创建相应的修复补丁
- 重试失败的操作

### 3. RepairPatchFactory (补丁工厂)

根据错误类型创建相应的补丁实例。

## 使用示例

### 基本使用

```python
from aag.error_recovery import ErrorRecoveryModule
from aag.reasoner.model_deployment import Reasoner

# 初始化
reasoner = Reasoner(config)
error_recovery = ErrorRecoveryModule(reasoner, max_retries=3)

# 在 scheduler 中使用
try:
    extraction_result = self._prepare_parameters_for_execution(...)
except Exception as param_err:
    if error_recovery.can_retry(step_id):
        # 准备错误信息
        error_info = {
            "error": str(param_err),
            "error_type": type(param_err).__name__,
            "previous_parameters": context.get("previous_parameters", {})
        }
        
        # 修复并重试
        extraction_result = await error_recovery.recover_parameter_error(
            step=step,
            error_info=error_info,
            context={
                "tool_description": tool_description,
                "vertex_schema": vertex_schema,
                "edge_schema": edge_schema
            }
        )
        error_recovery._increment_retry_count(step_id)
    else:
        raise
```

## 扩展

### 添加新的补丁类型

1. 在 `base_patch.py` 中创建新的补丁类：

```python
class MyCustomPatch(RepairPatch):
    def enhance_prompt(self, original_prompt: str, **kwargs) -> str:
        # 实现增强逻辑
        pass
    
    def get_patch_metadata(self) -> Dict[str, Any]:
        return {"patch_type": "my_custom"}
```

2. 在 `patch_factory.py` 中注册：

```python
RepairPatchFactory.register_patch("my_custom", MyCustomPatch)
```

## 注意事项

1. **当前实现状态**: 部分方法（如 `_extract_parameters_with_patch`）需要根据实际的 `reasoner` 接口进行实现。

2. **集成到 scheduler**: 需要在 `scheduler.py` 的 `_run_algorithm_pipeline2` 方法中集成错误恢复逻辑。

3. **错误信息结构**: 确保错误信息包含足够的上下文，以便补丁能够生成有效的修复指导。

## 下一步

- [ ] 实现与 `reasoner` 的完整集成
- [ ] 在 `scheduler.py` 中集成错误恢复逻辑
- [ ] 添加更细粒度的错误定位（如 `dependency_resolver` 内部错误）
- [ ] 添加错误恢复的日志和监控
