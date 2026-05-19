"""测试 GraphWorkflowDAG —— DAG 构建、环检测、拓扑排序与就绪判断。"""

import pytest
from aag.models.graph_workflow_dag import GraphWorkflowDAG


class TestWorkflowDAG:
    """覆盖 GraphWorkflowDAG 的核心行为：增删步骤、依赖、环检测、拓扑排序、就绪判断。"""

    def test_add_step(self, sample_dag):
        """添加步骤后 DAG 应包含对应数量的节点。"""
        # TODO: 验证 sample_dag 包含 3 个步骤
        assert len(sample_dag.steps) == 3

    def test_add_dependency(self, sample_dag):
        """依赖关系添加后，出入边应正确反映父子关系。"""
        # TODO: 验证依赖关系正确建立
        # sample_dag 结构: step1 → step2 → step3（ID 为 1, 2, 3 的整数）
        assert len(sample_dag.out_edges[1]) == 1  # step1 → step2
        assert 2 in sample_dag.out_edges[1]
        assert len(sample_dag.in_edges[2]) == 1  # step2 ← step1
        assert 1 in sample_dag.in_edges[2]

    def test_circular_dependency_detection(self):
        """添加形成环的依赖时，应抛出包含"环"的 ValueError。"""
        dag = GraphWorkflowDAG()
        a_id = dag.add_step("A")
        b_id = dag.add_step("B")
        c_id = dag.add_step("C")
        dag.add_dependency(a_id, b_id)
        dag.add_dependency(b_id, c_id)
        # 添加 A ← C 会形成 A → B → C → A 环
        with pytest.raises(ValueError, match="环"):
            dag.add_dependency(c_id, a_id)

    def test_topological_order(self, sample_dag):
        """拓扑排序中 step1 在 step2 之前，step2 在 step3 之前。"""
        # TODO: 验证拓扑顺序中 step1(1) 在 step2(2) 之前，step2 在 step3(3) 之前
        topo = sample_dag.topological_order()
        idx1 = topo.index(1)
        idx2 = topo.index(2)
        idx3 = topo.index(3)
        assert idx1 < idx2 < idx3

    def test_ready_steps(self, sample_dag):
        """初始 ready_steps 应仅包含无父节点的步骤（step1）。"""
        # TODO: 验证初始 ready_steps 仅包含 step1（ID=1）
        ready = sample_dag.ready_steps()
        assert ready == [1]

    def test_empty_dag(self):
        """空 DAG 的拓扑排序结果应为空列表。"""
        dag = GraphWorkflowDAG()
        assert len(dag.topological_order()) == 0
        assert len(dag.steps) == 0

    def test_duplicate_step_id_unique(self):
        """每次 add_step 应返回不重复的整数 ID。"""
        dag = GraphWorkflowDAG()
        ids = {dag.add_step(f"step_{i}") for i in range(10)}
        assert len(ids) == 10  # 10 次添加应产生 10 个唯一 ID

    def test_self_loop_rejected(self):
        """不允许自环：add_dependency(a, a) 必须抛出 ValueError。"""
        dag = GraphWorkflowDAG()
        a_id = dag.add_step("A")
        with pytest.raises(ValueError, match="自环"):
            dag.add_dependency(a_id, a_id)

    def test_topological_order_detects_cycle(self):
        """手工破坏 DAG 不变式后，topological_order 应抛出 ValueError。"""
        dag = GraphWorkflowDAG()
        a_id = dag.add_step("A")
        b_id = dag.add_step("B")
        dag.add_dependency(a_id, b_id)
        # 手工注入一条反向边，破坏 DAG 性质
        dag.in_edges[a_id].add(b_id)
        dag.out_edges[b_id].add(a_id)
        with pytest.raises(ValueError, match="环"):
            dag.topological_order()
