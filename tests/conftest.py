"""pytest 全局 fixtures 与共享配置。"""

import sys
from pathlib import Path

import pytest

# 将项目根目录添加到 sys.path，确保测试可导入 aag 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """返回项目根目录的绝对路径。"""
    return PROJECT_ROOT


@pytest.fixture
def sample_dag():
    """
    构建一个包含 3 个步骤的简单示例 DAG，用于 DAG 模型单元测试。

    结构:
        step1 (无父节点) → step2 → step3
    """
    from aag.models.graph_workflow_dag import GraphWorkflowDAG

    dag = GraphWorkflowDAG()

    # 添加步骤，返回整数 step_id
    step1_id = dag.add_step("社区检测：使用Louvain算法检测社区结构")
    step2_id = dag.add_step("中心性分析：计算PageRank中心性")
    step3_id = dag.add_step("可视化：生成PNG格式网络图")

    # 依据真实 API：add_dependency(parent_id, child_id) 使用整数 ID
    dag.add_dependency(step1_id, step2_id)
    dag.add_dependency(step2_id, step3_id)

    return dag
