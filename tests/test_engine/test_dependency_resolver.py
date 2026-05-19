"""测试 DataDependencyResolver —— 依赖分析、子图构造与参数映射。

注意：DataDependencyResolver 依赖 Reasoner 进行 LLM 调用，
当前所有测试均为 TODO 骨架，需要 mock Reasoner 或启动集成测试环境。
"""

import pytest


class TestDependencyResolver:
    """覆盖依赖解析器的核心行为：依赖分类、数据定位、图/参数转换。"""

    def test_resolve_no_parents_returns_empty(self):
        """无父步骤时 resolve_dependencies 应返回空依赖。"""
        # TODO: mock DataDependencyResolver + 空父步骤列表
        # result = await resolver.resolve_dependencies(step_id, step, alg_info, [])
        # assert result["graph_dependencies"] == []
        # assert result["parameter_dependencies"] == []

    def test_classify_graph_dependency(self):
        """LLM 返回 dependency_type=graph 时，应产生 graph 类别依赖项。"""
        # TODO: mock reasoner.analyze_dependency_type_and_locate_dependency_data()
        #       返回 dependency_type="graph" 及 selected_outputs
        # dep_info = await resolver._classify_and_locate_dependency(...)
        # assert dep_info.dependency_type == DataDependencyType.GRAPH

    def test_classify_parameter_dependency(self):
        """LLM 返回 dependency_type=parameter 时，应产生 parameter 类别依赖项。"""
        # TODO: mock reasoner 返回 dependency_type="parameter"

    def test_locate_specific_output_field(self):
        """LLM 指定 output_id + field_key 时，应精确定位到真实数据。"""
        # TODO: 构造父步骤包含多个 output，验证 SingleDependencyItem 的 value 正确

    def test_convert_graph_dependencies_code_execution(self):
        """图依赖转换：LLM 生成代码 → 执行 → 返回子图。"""
        # TODO: mock reasoner.generate_graph_conversion_code()
        #       返回包含有效 Python 代码的字典
        # mock resolver.global_vertices / global_edges

    def test_set_global_graph(self):
        """set_global_graph 应正确存储全局图数据。"""
        # TODO: 直接调用 set_global_graph 并验证存储
