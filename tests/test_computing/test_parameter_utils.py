"""测试参数工具函数 —— 类型转换与参数规范化。

注意：原规范中提到的 validate_and_convert_param 在代码库中为私有函数 _smart_convert。
本测试直接测试 _smart_convert 与 normalize_parameters 的实际行为。
"""

import logging

import pytest
from aag.computing_engine.networkx_server.parameter_utils import (
    _smart_convert,
    normalize_parameters,
)


class TestParameterUtils:
    """覆盖参数类型转换与规范化逻辑。"""

    # ── _smart_convert 测试 ──────────────────────────────

    def test_validate_int_param(self):
        """整数字符串 "42" 应被正确转换为 int 42。"""
        log = logging.getLogger("test")
        result = _smart_convert("42", "integer", "test_param", log)
        # TODO: 验证结果是 int 类型且值为 42
        assert result == 42
        assert isinstance(result, int)

    def test_validate_float_param(self):
        """浮点字符串 "3.14" 应被正确转换为 float。"""
        log = logging.getLogger("test")
        result = _smart_convert("3.14", "float", "test_param", log)
        # TODO: 验证结果是 float 类型且近似 3.14
        assert result == 3.14
        assert isinstance(result, float)

    def test_validate_bool_param(self):
        """ "true"/"True"/"1" → True, "false" → False。"""
        log = logging.getLogger("test")
        # TODO: 验证布尔转换的多种输入形式
        assert _smart_convert("true", "boolean", "p", log) is True
        assert _smart_convert("True", "boolean", "p", log) is True
        assert _smart_convert("1", "boolean", "p", log) is True
        assert _smart_convert("false", "boolean", "p", log) is False
        assert _smart_convert("no", "boolean", "p", log) is False

    def test_none_passthrough(self):
        """None 值应原样返回。"""
        log = logging.getLogger("test")
        result = _smart_convert(None, "integer", "test_param", log)
        # TODO: 验证 None 不被转换为字符串 "None"
        assert result is None

    def test_invalid_param_type(self):
        """未知类型应保持原值并输出警告日志。"""
        log = logging.getLogger("test")
        result = _smart_convert("hello", "unknown_type_xyz", "test_param", log)
        # TODO: 验证未知类型不会崩溃，返回原值
        assert result == "hello"

    # ── normalize_parameters 测试 ───────────────────────

    def test_normalize_parameters_converts_types(self):
        """normalize_parameters 应对 schema 中声明的类型执行 _smart_convert。"""
        tool_info = {
            "input_schema": {
                "parameters": {
                    "k": {"type": "integer"},
                }
            }
        }
        result = normalize_parameters("test_tool", {"k": "10"}, tool_info)
        # TODO: 验证 k 从 "10" 转换为 int 10
        assert result["k"] == 10
        assert isinstance(result["k"], int)

    def test_normalize_parameters_skips_unknown_tools(self):
        """未找到工具定义时，应原样返回参数。"""
        # tool_info 为 None 时直接返回原 parameters
        result = normalize_parameters("unknown", {"k": 5}, None)
        # TODO: 验证无 tool_info 时参数未被修改
        assert result["k"] == 5

    def test_normalize_parameters_empty_params(self):
        """空参数字典应返回空字典。"""
        result = normalize_parameters("any", {}, None)
        # TODO: 验证空输入 → 空输出
        assert result == {}
