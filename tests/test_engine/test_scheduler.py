"""测试 Scheduler —— DAG 构建、查询执行与模式切换。

注意：Scheduler 依赖完整的 Reasoner/ComputingEngine/ExpertSearchEngine 等组件，
当前所有测试均为 TODO 骨架，需要 mock 外部依赖或启动集成测试环境。
"""

import pytest


class TestScheduler:
    """覆盖 Scheduler 的核心流程：DAG 构建、拓扑执行、模式切换。"""

    def test_build_dag_from_query(self):
        """从用户查询构建 DAG —— 需要 mock reasoner.plan_subqueries()。"""
        # TODO: 创建 mock Reasoner，注入 plan_subqueries 返回固定计划
        # scheduler = Scheduler(mock_config)
        # dag = scheduler._build_dag_from_query("分析图社区")
        # assert len(dag.steps) > 0

    def test_execute_graph_mode(self):
        """正常图分析模式下的完整执行流程。"""
        # TODO: mock 完整组件链，验证 execute() 返回有效结果

    def test_execute_with_decompose_false(self):
        """decompose=False 时不拆分子查询。"""
        # TODO: 验证 decompose=False 时 query 不被拆解

    def test_mode_validation(self):
        """不支持的 mode 应抛出 ValueError。"""
        # TODO: 直接调用 _normalize_graph_mode("invalid")
        #       验证抛出 ValueError

    def test_ready_steps_after_success(self):
        """步骤完成后，ready_steps 应包含下一个无阻塞步骤。"""
        # TODO: 设置 sample_dag 中 step1 为 success，验证 ready_steps 包含 step2

    def test_topological_order_is_valid(self, sample_dag):
        """拓扑排序结果应是无环的合法顺序。"""
        # 复用 conftest 中的 sample_dag fixture
        topo = sample_dag.topological_order()
        # TODO: 验证拓扑序非空且无重复
        assert len(topo) == 3
        assert len(set(topo)) == 3
