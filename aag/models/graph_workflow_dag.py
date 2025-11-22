"""
简化版本的GraphWorkflowDAG实现
专注于DAG构建、拓扑排序和依赖关系管理
"""

import collections
# from pickle import DICT
from pydantic import BaseModel
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
from aag.models.task_types import GraphAnalysisType, GraphAnalysisSubType
from aag.engine.dependency_resolver import DataDependencyInfo

class OutputField(BaseModel):
    type: str
    field_description: Optional[str] = None

class OutputSchema(BaseModel):
    description: Optional[str] = None
    type: str = "dict"
    fields: Dict[str, OutputField]

@dataclass
class StepOutputItem:
    """单个步骤输出项（用于描述算法执行结果的结构化内容）"""
    output_id: int          # 在同一dag节点中的顺序编号，用于表示结果生成顺序（从1开始）
    task_type: GraphAnalysisSubType
    source: str             # 算法名或处理动作名
    output_schema: Optional[OutputSchema] = None
    value: Dict[str, Any] = None             # 实际计算结果
    path: Optional[str] = None

    def to_meta(self) -> dict:
        """返回仅包含用于语义判断的元信息"""
        meta = {
            "output_id": self.output_id,
            "source": self.source,
        }

        if self.output_schema and self.task_type==GraphAnalysisSubType.POST_PROCESSING:
            meta["type"] = self.output_schema.type
            meta["description"] = self.output_schema.description
            fields = []
            for k, v in self.output_schema.fields.items():
                fields.append({
                    "key": k,
                    "type": v.get("type", "unknown"),
                    "desc": v.get("field_description", "")
                })
            if fields:
                meta["fields"] = fields
        elif self.output_schema and self.task_type==GraphAnalysisSubType.GRAPH_ALGORITHM:
            fields = []
            for k, v in self.output_schema.fields.items():
                fields.append({
                    "key": k,
                    "type": v.get("type", "unknown"),
                    "desc": v.get("field_description", "")
                })
            if fields:
                meta["fields"] = fields

        return meta
    


@dataclass
class WorkflowStep:
    """DAG节点，代表一个子问题"""
    step_id: int
    question: str                           # 该节点表示问题的问题描述
    task_type: Optional[GraphAnalysisType] = None         # 该节点表示问题解决的任务类型（graph processing / numeric_analysis）
    graph_algorithm: Optional[str] = None   # 该节点表示问题解决的图算法（设置为None）
    status: str = "pending"                 # 节点状态: pending, running, success, failed
    result: Optional[Dict[int, StepOutputItem]] = None                      # 节点执行结果  
    llm_analysis: Optional[str] = None

    next_output_id: int = 1

    def __str__(self):
        return f"Step({self.step_id}): {self.question[:50]}..."

    def get_result_meta(self) -> List[dict]:
        """
        获取当前步骤结果的元信息视图，仅用于依赖判定或 LLM 语义输入。
        如果 result 为空，返回空列表。
        """
        if not self.result:
            return []
        return [item.to_meta() for item in self.result]
    
    def add_output(
    self,
    task_type: GraphAnalysisSubType,
    source: str,
    output_schema: Optional[OutputSchema] = None,
    value: Any = None,
    path: Optional[str] = None,
    validate_schema: bool = False,
    ) -> StepOutputItem:
        """
        向当前 WorkflowStep 插入一个 StepOutputItem，并自动分配 output_id。

        参数：
            task_type: 当前输出是哪类任务的结果
            source: 执行来源（算法名、后处理代码名等）
            value: 实际计算输出
            output_schema: 输出的结构化描述（可选）
            path: 若输出保存在文件中，则是文件路径
            validate_schema: 是否对 value 与 output_schema 做结构验证
        """

        if self.result is None:
            self.result = []

        output_id = self.next_output_id
        self.next_output_id += 1

        # -------- 可选：验证 value 是否符合 output_schema -------
        if validate_schema and output_schema is not None:
            if output_schema.type == "dict" and not isinstance(value, dict):
                raise ValueError(f"value 必须为 dict，但实际为 {type(value)}")

            # 如果 fields 存在，则逐项验证
            if isinstance(value, dict) and output_schema.fields:
                for field_name, field_schema in output_schema.fields.items():
                    if field_name not in value:
                        raise ValueError(f"value 缺少字段 {field_name}")
                    # 可继续加更多验证逻辑（例如类型检查）

        # -------- 创建 StepOutputItem -------
        item = StepOutputItem(
            output_id=output_id,
            task_type=task_type,
            source=source,
            output_schema=output_schema,
            value=value,
            path=path,
        )

        self.result[output_id] = item
        return item
    
    def get_output(self, output_id: int) -> Optional[StepOutputItem]:
        if not self.result:
            return None
        return self.result.get(output_id)


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
    
    def set_success(self, step_id: int, output_data: Optional[List[StepOutputItem]] = None, llm_analysis: str = None):
        """设置步骤为成功状态"""
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        
        self.steps[step_id].status = "success"
        if output_data is not None:
            self.steps[step_id].result = output_data
        if llm_analysis is not None:
            self.steps[step_id].llm_analysis =  llm_analysis
    
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