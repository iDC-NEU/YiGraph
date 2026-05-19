"""测试 parse_json 工具函数 —— JSON 提取与容错解析。"""

import pytest


class TestParseJson:
    """覆盖 extract_json_from_response / parse_openai_json_response 的边界行为。"""

    def test_extract_clean_json(self):
        """从纯净 JSON 字符串中提取应直接返回。"""
        # TODO: 调用 extract_json_from_response('{"key": "value"}')
        #       验证返回 {"key": "value"}

    def test_extract_json_from_markdown_block(self):
        """从 Markdown 代码块中提取 JSON。"""
        # TODO: 调用 extract_json_from_response('```json\n{"key": "value"}\n```')
        #       验证返回 {"key": "value"}

    def test_extract_json_with_think_tag(self):
        """忽略 <think> 标签，提取后续 JSON 对象。"""
        # TODO: 输入包含 <think> 等干扰内容的字符串，验证仍能正确提取

    def test_extract_missing_json_raises(self):
        """完全不包含 JSON 时，应抛出 ValueError。"""
        # TODO: 调用 extract_json_from_response("pure text without braces")
        #       with pytest.raises(ValueError, match="JSON")

    def test_parse_empty_response(self):
        """空字符串应返回 error 字典。"""
        # TODO: 调用 parse_openai_json_response("")
        #       验证返回 {"error": "Empty response from OpenAI API"}

    def test_parse_json_decode_error(self):
        """非法 JSON 应返回包含 raw_response 的错误字典。"""
        # TODO: 调用 parse_openai_json_response("{invalid}")
        #       验证返回的 dict 包含 "error" 和 "raw_response" 键
