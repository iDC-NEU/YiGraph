import ast
import json
import logging
from typing import Any, Dict, Optional

_LOGGER = logging.getLogger(__name__)

_NODE_PARAM_NAMES = {
    "source", "target", "node", "u", "v", "s", "t", "nodes", "root", "start", "end",
}

_NORMALIZATION_BLACKLIST = {
    "initialize_graph",
    "get_graph_info",
    # 其他无需规范化的工具放这里
}

def should_normalize(tool_name: str) -> bool:
    return tool_name not in _NORMALIZATION_BLACKLIST


def annotate_schema(input_schema: dict) -> dict:
    """Schema annotation logic specific to the NetworkX engine."""
    annotated = input_schema.copy()

    if 'parameters' in annotated:
        parameters = annotated['parameters'].copy()

        # Annotate parameter G
        if 'G' in parameters:
            parameters['G'] = {
                **parameters['G'],
                'description': f"[System auto-injected] {parameters['G'].get('description', 'Graph object instance')}"
            }

        # Annotate backend_kwargs
        if 'backend_kwargs' in parameters:
            parameters['backend_kwargs'] = {
                **parameters['backend_kwargs'],
                'description': f"[System auto-injected, default={{}}] {parameters['backend_kwargs'].get('description', 'Backend execution parameters')}"
            }

        annotated['parameters'] = parameters

    return annotated



def normalize_parameters(
    tool_name: str,
    parameters: Dict[str, Any],
    tool_info: Optional[Dict[str, Any]],
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    规范化 LLM 抽取的参数，使其符合 NetworkX 工具的 schema 要求。
    - tool_info: 来自 MCP 客户端的工具定义（包含 input_schema）。
    """
    log = logger or _LOGGER
    
    if not parameters:
        return {}

    if not tool_info:
        log.warning("⚠️ 未找到工具 '%s' 的定义，跳过参数规范化", tool_name)
        return parameters

    input_schema = tool_info.get("input_schema") or {}
    properties = input_schema.get("parameters") or {}

    normalized: Dict[str, Any] = {}

    for param_name, param_value in parameters.items():
        param_schema = properties.get(param_name, {})
        param_type = param_schema.get('type', '')
        param_description = param_schema.get('description', '').lower()
        expected_type = param_schema.get('type', None)
        
        # 🔴 先检查是否需要类型转换（优先级最高）
        if expected_type:
            # 解析 expected_type（处理 "dict, optional" 这种格式）
            base_type = expected_type.split(',')[0].strip()
            
            original_type = type(param_value).__name__
            converted_value = _smart_convert(param_value, base_type, param_name)
            
            if type(converted_value).__name__ != original_type:
                logger.info(
                    f"🔄 自动类型转换: {param_name} = {repr(param_value)[:50]} "
                    f"({original_type}) → {repr(converted_value)[:50]} ({type(converted_value).__name__})"
                )
            
            normalized[param_name] = converted_value
            continue  # ⭐ 转换后直接跳过后续判断
        
        # 🔴 判断是否为节点参数（仅当未定义 expected_type 时）
        is_node_param = (
            param_name in _NODE_PARAM_NAMES or
            param_type == 'node' or
            'node' in param_description or
            'vertex' in param_description
        )
        
        # ⭐ 排除字典和列表类型
        if is_node_param and not isinstance(param_value, (str, dict, list)):
            original_value = param_value
            original_type = type(param_value).__name__
            normalized[param_name] = str(param_value)
            logger.info(
                f"🔄 参数类型转换: {param_name} = {original_value} "
                f"({original_type}) → '{normalized[param_name]}' (str)"
            )
        elif is_node_param and isinstance(param_value, list):
            # 列表中的元素转字符串
            normalized[param_name] = [str(v) if not isinstance(v, str) else v for v in param_value]
            logger.info(f"🔄 参数类型转换: {param_name} (list) 中的元素转换为字符串")
        else:
            # 保持原值
            normalized[param_name] = param_value

    return normalized


def _smart_convert(
    value: Any, target_type: str, param_name: str, log: logging.Logger
) -> Any:
    """按照 schema 类型做一次宽松转换。"""
    if value is None:
        return None

    target_type = target_type.lower().strip()

    if target_type == "node":
        return value if isinstance(value, str) else str(value)

    if target_type in {"int", "integer"}:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                log.warning("⚠️ 无法将 '%s' 转换为 int，保持原值", value)
                return value
        if isinstance(value, float):
            return int(value)
        return value

    if target_type in {"float", "number"}:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                log.warning("⚠️ 无法将 '%s' 转换为 float，保持原值", value)
                return value
        return value

    if target_type in {"bool", "boolean"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes"}
        return bool(value)

    if target_type in {"dict", "object", "dictionary"}:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            for loader, name in ((json.loads, "json"), (ast.literal_eval, "literal_eval")):
                try:
                    return loader(value)
                except Exception:
                    continue
            log.warning("⚠️ 无法将字符串 '%s' 转换为 dict，保持原值", value)
        return value

    if target_type in {"list", "array"}:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            for loader, name in ((json.loads, "json"), (ast.literal_eval, "literal_eval")):
                try:
                    return loader(value)
                except Exception:
                    continue
            log.warning("⚠️ 无法将字符串 '%s' 转换为 list，保持原值", value)
        return value

    if target_type in {"str", "string"}:
        return value if isinstance(value, str) else str(value)

    log.warning("⚠️ 未知类型 '%s' for %s，保持原值", target_type, param_name)