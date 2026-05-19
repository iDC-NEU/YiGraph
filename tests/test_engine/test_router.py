"""测试 QueryRouter —— 查询分类与回退行为。

注意：所有路由测试均依赖 Reasoner（LLM），当前为 TODO 骨架。
实际运行需要 mock Reasoner.chat() 或提供测试专用的 LLM 端点。
"""

import pytest


class TestRouter:
    """覆盖 QueryRouter 的分类逻辑与非法 JSON 回退行为。"""

    def test_graph_query_classification(self):
        """图查询分类 —— 模拟 Router.classify() 返回 RouteDecision(GRAPH)。"""
        # TODO: 需要 mock Reasoner.chat() 返回 '{"type":"graph","reason":"..."}'
        # from aag.engine.router import QueryRouter, RouteDecision, QueryType
        # router = QueryRouter(mock_reasoner)
        # decision = router.route("分析图的社区结构")
        # assert decision.query_type == QueryType.GRAPH

    def test_rag_query_classification(self):
        """RAG 查询分类 —— 模拟 Router.classify() 返回 RouteDecision(RAG)。"""
        # TODO: 需要 mock Reasoner.chat() 返回 '{"type":"rag","reason":"..."}'
        # from aag.engine.router import QueryRouter, RouteDecision, QueryType
        # router = QueryRouter(mock_reasoner)
        # decision = router.route("根据文档内容回答问题")
        # assert decision.query_type == QueryType.RAG

    def test_general_query_classification(self):
        """通用查询分类 —— 模拟 Router.classify() 返回 RouteDecision(GENERAL)。"""
        # TODO: 需要 mock Reasoner.chat() 返回 '{"type":"general","reason":"..."}'
        # from aag.engine.router import QueryRouter, RouteDecision, QueryType
        # router = QueryRouter(mock_reasoner)
        # decision = router.route("你好，介绍一下你自己")
        # assert decision.query_type == QueryType.GENERAL

    def test_invalid_json_response(self):
        """LLM 返回非法 JSON 时，应回退到 GENERAL 类型。"""
        # TODO: 需要 mock Reasoner.chat() 返回非法 JSON 字符串
        # from aag.engine.router import QueryRouter, RouteDecision, QueryType
        # router = QueryRouter(mock_reasoner)
        # decision = router.route("任意查询")
        # assert decision.query_type == QueryType.GENERAL
        # assert "Failed to parse" in decision.reason
