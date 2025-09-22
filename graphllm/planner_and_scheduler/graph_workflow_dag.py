"""
简化版本的GraphWorkflowDAG实现
专注于DAG构建、拓扑排序和依赖关系管理
"""

from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import collections


@dataclass
class WorkflowStep:
    """DAG节点，代表一个子问题"""
    step_id: int
    question: str                           # 该节点代表子问题的问题描述
    graph_algorithm: Optional[str] = None   # 该节点子问题解决的图算法（设置为None）
    status: str = "pending"                 # 节点状态: pending, running, success, failed
    result: Any = None                      # 节点执行结果
    
    def __str__(self):
        return f"Step({self.step_id}): {self.question[:50]}..."


class GraphWorkflowDAG:
    """
    简化版本的GraphWorkflowDAG实现
    支持拓扑排序和返回拓扑序，维护每个节点的入边和出边
    """
    
    def __init__(self):
        self.steps: Dict[int, WorkflowStep] = {}
        self.out_edges: Dict[int, Set[int]] = collections.defaultdict(set)  # 出边: step_id -> {child_steps}
        self.in_edges: Dict[int, Set[int]] = collections.defaultdict(set)   # 入边: step_id -> {parent_steps}
        self.next_step_id: int = 1
    
    def add_step(self, question: str, graph_algorithm: Optional[str] = None) -> int:
        """
        添加新步骤
        
        Args:
            question: 子问题的问题描述
            graph_algorithm: 图算法名称，默认为None
            
        Returns:
            新步骤的ID
        """
        step_id = self.next_step_id
        self.next_step_id += 1
        
        step = WorkflowStep(
            step_id=step_id,
            question=question,
            graph_algorithm=graph_algorithm
        )
        
        self.steps[step_id] = step
        return step_id
    
    def add_dependency(self, parent_id: int, child_id: int):
        """
        添加依赖边（从parent到child）
        
        Args:
            parent_id: 父步骤ID
            child_id: 子步骤ID
        """
        if parent_id not in self.steps:
            raise ValueError(f"父步骤 {parent_id} 不存在")
        if child_id not in self.steps:
            raise ValueError(f"子步骤 {child_id} 不存在")
        if parent_id == child_id:
            raise ValueError("不允许自环")
        
        self.out_edges[parent_id].add(child_id)
        self.in_edges[child_id].add(parent_id)
        
        # 检查是否产生环
        if self._has_cycle():
            # 回滚添加的边
            self.out_edges[parent_id].discard(child_id)
            self.in_edges[child_id].discard(parent_id)
            raise ValueError(f"添加边 {parent_id} -> {child_id} 会产生环")
    
    def parents_of(self, step_id: int) -> List[int]:
        """获取步骤的所有父步骤（入边）"""
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        return list(self.in_edges[step_id])
    
    def children_of(self, step_id: int) -> List[int]:
        """获取步骤的所有子步骤（出边）"""
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        return list(self.out_edges[step_id])
    
    def topological_order(self) -> List[int]:
        """
        返回DAG的拓扑序
        
        Returns:
            拓扑排序后的步骤ID列表
            
        Raises:
            ValueError: 如果图中存在环
        """
        # Kahn算法进行拓扑排序
        in_degree = {step_id: len(self.in_edges[step_id]) for step_id in self.steps}
        queue = collections.deque([step_id for step_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            step_id = queue.popleft()
            result.append(step_id)
            
            # 处理当前步骤的所有子步骤
            for child_id in self.out_edges[step_id]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
        
        # 检查是否所有步骤都被访问（即是否存在环）
        if len(result) != len(self.steps):
            raise ValueError("图中存在环，无法进行拓扑排序")
        
        return result
    
    def _has_cycle(self) -> bool:
        """检查图中是否存在环"""
        try:
            self.topological_order()
            return False
        except ValueError:
            return True
    
    def ready_steps(self) -> List[int]:
        """
        获取当前可以执行的步骤（所有父步骤都已完成）
        
        Returns:
            可执行步骤ID列表
        """
        ready = []
        for step_id, step in self.steps.items():
            if step.status != "pending":
                continue
            
            # 检查所有父步骤是否都已完成
            parents = self.in_edges[step_id]
            if all(self.steps[parent_id].status == "success" for parent_id in parents):
                ready.append(step_id)
        
        return ready
    
    def set_success(self, step_id: int, output_data: Any = None):
        """设置步骤为成功状态"""
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        
        self.steps[step_id].status = "success"
        if output_data is not None:
            self.steps[step_id].result = output_data
    
    def set_running(self, step_id: int):
        """设置步骤为运行状态"""
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        
        self.steps[step_id].status = "running"
    
    def set_failed(self, step_id: int, error: str):
        """设置步骤为失败状态"""
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        
        self.steps[step_id].status = "failed"
        self.steps[step_id].result = error
    
    def get_step_info(self, step_id: int) -> Dict[str, Any]:
        """获取步骤的详细信息"""
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        
        step = self.steps[step_id]
        return {
            "step_id": step.step_id,
            "question": step.question,
            "graph_algorithm": step.graph_algorithm,
            "status": step.status,
            "parents": list(self.in_edges[step_id]),
            "children": list(self.out_edges[step_id]),
            "result": step.result
        }
    
    def print_dag_info(self):
        """打印DAG的详细信息"""
        print(f"DAG包含 {len(self.steps)} 个步骤")
        print("\n步骤详情:")
        
        try:
            topo_order = self.topological_order()
            print(f"拓扑序: {topo_order}")
            
            for step_id in topo_order:
                info = self.get_step_info(step_id)
                print(f"\n步骤 {step_id}:")
                print(f"  问题: {info['question']}")
                print(f"  图算法: {info['graph_algorithm']}")
                print(f"  状态: {info['status']}")
                print(f"  父步骤: {info['parents']}")
                print(f"  子步骤: {info['children']}")
                
        except ValueError as e:
            print(f"拓扑排序失败: {e}")
    
    def export_as_dict(self) -> Dict[str, Any]:
        """将DAG转换为字典格式"""
        steps_info = []
        for step_id in self.steps:
            info = self.get_step_info(step_id)
            steps_info.append(info)
        
        edges_info = []
        for parent_id, children in self.out_edges.items():
            for child_id in children:
                edges_info.append({"from": parent_id, "to": child_id})
        
        return {
            "nodes": steps_info,
            "edges": edges_info,
            "step_count": len(self.steps),
            "edge_count": len(edges_info)
        }
    
    def to_dot(self) -> str:
        """
        生成 Graphviz DOT 字符串，便于可视化
        """
        lines = ["digraph G {"]
        for step_id, step in self.steps.items():
            label = f"{step_id}:{step.question[:20]}...\\n{step.status}"
            lines.append(f'  {step_id} [label="{label}"];')
        for parent_id, children in self.out_edges.items():
            for child_id in children:
                lines.append(f"  {parent_id} -> {child_id};")
        lines.append("}")
        return "\n".join(lines)