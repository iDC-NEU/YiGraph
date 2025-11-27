"""
简化版本的GraphWorkflowDAG实现
专注于DAG构建、拓扑排序和依赖关系管理
"""

import collections
import collections
import copy
# from pickle import DICT
from pydantic import BaseModel
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
from aag.models.task_types import GraphAnalysisType, GraphAnalysisSubType
from typing import Dict, List, Set, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from aag.reasoner.model_deployment import Reasoner


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
                    "type": v.type,
                    "desc": v.field_description
                })
            if fields:
                meta["fields"] = fields
        elif self.output_schema and self.task_type==GraphAnalysisSubType.GRAPH_ALGORITHM:
            fields = []
            for k, v in self.output_schema.fields.items():
                fields.append({
                    "key": k,
                    "type": v.type,
                    "desc": v.field_description
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
        return [item.to_meta() for item in self.result.values()]
    
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
            self.result = {}

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
    
    def add_algorithm_result(
        self,
        tool_name: str,
        tool_result_data: Dict[str, Any],
        output_schema: Dict[str, Any],
        is_has_extract_code: bool = False,
    ) -> None:
        """
        添加算法执行结果到当前步骤。
        
        参数:
            tool_name: 算法名称
            tool_result_data: 算法执行结果数据，格式: {"original_result": value, "filed1": value, ...}
            output_schema: 输出schema定义，包含fields字段
            is_has_extract_code: 是否有后处理代码
        """
        schema_fields = output_schema.get("fields", {})
        
        # 1. 添加原始图算法计算结果 (original_result 字段)
        if "original_result" in schema_fields:
            original_result_field = schema_fields["original_result"]
            original_result_value = tool_result_data.get("original_result") if isinstance(tool_result_data, dict) else tool_result_data
            
            self.add_output(
                task_type=GraphAnalysisSubType.GRAPH_ALGORITHM,
                source=tool_name,
                output_schema=OutputSchema(
                    description=f"The {tool_name} algorithm is executed",
                    type="dict",
                    fields={
                        "original_result": OutputField(
                            type=original_result_field.get("type", ""),
                            field_description=original_result_field.get("field_description", "")
                        )
                    }
                ),
                value={"original_result": original_result_value},
                path=None,  # 需要一个保存中间结果的模块
                validate_schema=True
            )
        
        # 2. 如果有后处理代码，添加后处理计算结果（排除 original_result 字段）
        if is_has_extract_code:
            post_processing_fields = {
                name: OutputField(
                    type=info.get("type", ""),
                    field_description=info.get("field_description", "")
                )
                for name, info in schema_fields.items()
                if name != "original_result"
            }
            
            # 从结果中移除 original_result
            if isinstance(tool_result_data, dict) and "original_result" in tool_result_data:
                tool_result_data.pop("original_result")
            
            self.add_output(
                task_type=GraphAnalysisSubType.POST_PROCESSING,
                source="python code",
                output_schema=OutputSchema(
                    description=output_schema.get("description", ""),
                    type=output_schema.get("type", "dict"),
                    fields=post_processing_fields
                ),
                value=tool_result_data,
                path=None,  # 需要一个保存中间结果的模块
                validate_schema=True
            )


class GraphWorkflowDAG:
    """
    简化版本的GraphWorkflowDAG实现
    支持拓扑排序和返回拓扑序，维护每个节点的入边和出边
    """
    
    def __init__(self):
        self.subquery_plan: Dict[str, Any] = {"subqueries": []}
        self._reset_structure()

    def _reset_structure(self):
        """重置 DAG 节点、边和状态信息。"""
        self.steps: Dict[int, WorkflowStep] = {}
        self.out_edges: Dict[int, Set[int]] = collections.defaultdict(set)
        self.in_edges: Dict[int, Set[int]] = collections.defaultdict(set)
        self.data_dependency: Dict[int, Set[int]] = collections.defaultdict(set)
        self.next_step_id: int = 1
        self.query_id_mapping: Dict[str, int] = {}
    
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
    
    def ancestors_of(self, step_id: int) -> Set[int]:
        """获取步骤的所有祖先步骤"""
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        ancestors: Set[int] = set()
        stack = list(self.in_edges[step_id])
        while stack:
            current = stack.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            stack.extend(self.in_edges[current])
        return ancestors
    
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
            "task_type": step.task_type,
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
                print(f"\nstep {step_id}:")
                print(f"  question: {info['question']}")
                print(f"  task_type: {info['task_type']}")
                print(f"  algorithm: {info['graph_algorithm']}")
                print(f"  status: {info['status']}")
                print(f"  parents: {info['parents']}")
                print(f"  children: {info['children']}")
                
        except ValueError as e:
            print(f"拓扑排序失败: {e}")
            
    def get_dag_info(self) -> Dict[str, Any]:
        """
        获取当前DAG的详细信息
        
        Returns:
            包含DAG信息的字典
        """

        # 获取步骤信息
        steps_info = {}
        topological_order = []
        
        try:
            topological_order = self.topological_order()
            for step_id in topological_order:
                step = self.steps[step_id]
                steps_info[str(step_id)] = {
                    "question": step.question,
                    "task_type": step.task_type if step.task_type else None,
                    "algorithm": step.graph_algorithm if step.graph_algorithm else None,
                    "status": step.status.value if hasattr(step.status, 'value') else str(step.status)
                }
        except Exception as e:
            # 如果获取信息时出错，返回部分信息
            print(f"获取DAG信息时出错: {e}")
        
        return {
            "subquery_plan": self.get_subquery_plan(),
            "steps": steps_info,
            "topological_order": [str(sid) for sid in topological_order]
        }
    
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
    
    def refresh_data_dependency(self, reasoner) -> None:
        """
        遍历所有(q1, q2)对，其中q1是q2的祖先节点，并使用reasoner判断q2是否依赖q1。
        如果需要依赖，则在self.data_dependency[q2]中加入q1。
        """
        
        self.data_dependency.clear()
        
        for q2_id, q2_step in self.steps.items():
            ancestors = self.ancestors_of(q2_id)
            if not ancestors:
                continue
            
            q2_question = q2_step.question or ""
            q2_algorithm = q2_step.graph_algorithm or ""
            
            for q1_id in ancestors:
                q1_step = self.steps[q1_id]
                q1_question = q1_step.question or ""
                q1_algorithm = q1_step.graph_algorithm or ""
                
                try:
                    depends = reasoner.check_data_dependency(
                        q1_question=q1_question,
                        q1_algorithm=q1_algorithm,
                        q2_question=q2_question,
                        q2_algorithm=q2_algorithm,
                    )
                except Exception as exc:
                    print(f"检查数据依赖时出现异常: q1={q1_id}, q2={q2_id}, error={exc}")
                    depends = False
                
                if depends:
                    self.data_dependency[q2_id].add(q1_id)
    
    def get_data_dependency(self, step_id: int) -> Set[int]:
        if step_id not in self.steps:
            raise ValueError(f"步骤 {step_id} 不存在")
        return self.data_dependency.get(step_id, set())
    
    def print_data_dependency(self):
        """打印数据依赖关系"""
        print("数据依赖关系:")
        for step_id, dependencies in self.data_dependency.items():
            print(f"步骤 {step_id} 数据依赖于步骤: {list(dependencies)}")

    def set_subquery_plan(self, subquery_plan: Dict[str, Any]) -> None:
        """
        保存最新的 subquery_plan 副本，供后续修改时使用。
        """
        self._validate_subquery_plan_structure(subquery_plan)
        self.subquery_plan = copy.deepcopy(subquery_plan)

    def get_subquery_plan(self) -> Dict[str, Any]:
        """
        返回当前缓存的 subquery_plan（副本），避免外部直接修改原对象。
        """
        return copy.deepcopy(self.subquery_plan)

    def build_from_subquery_plan(self, subquery_plan: Dict[str, Any]) -> Dict[str, int]:
        """
        根据 subquery_plan 重构整个 DAG，并返回 query_id -> step_id 的映射。
        """
        self._validate_subquery_plan_structure(subquery_plan)
        subqueries = subquery_plan.get("subqueries", [])
        if not subqueries:
            raise ValueError("子查询计划为空，必须包含至少一个子查询")

        self._reset_structure()
        self.subquery_plan = copy.deepcopy(subquery_plan)

        query_id_to_step_id: Dict[str, int] = {}
        for subquery in subqueries:
            query_id = subquery.get("id")
            question = subquery.get("query")
            if query_id is None:
                raise ValueError("子查询缺少必需的'id'字段")
            if question is None:
                raise ValueError("子查询缺少必需的'query'字段")
            step_id = self.add_step(question=question, graph_algorithm=None)
            query_id_to_step_id[query_id] = step_id

        for subquery in subqueries:
            query_id = subquery["id"]
            depends_on = subquery.get("depends_on", [])
            current_step_id = query_id_to_step_id[query_id]
            for parent_query_id in depends_on:
                if parent_query_id not in query_id_to_step_id:
                    raise ValueError(f"依赖的查询ID '{parent_query_id}' 在子查询列表中不存在")
                parent_step_id = query_id_to_step_id[parent_query_id]
                self.add_dependency(parent_step_id, current_step_id)

        # 验证拓扑序，确保没有产生环
        self.topological_order()
        self.query_id_mapping = copy.deepcopy(query_id_to_step_id)
        return copy.deepcopy(query_id_to_step_id)

    def get_query_id_mapping(self) -> Dict[str, int]:
        """返回 query_id -> step_id 映射的副本。"""
        return copy.deepcopy(self.query_id_mapping)

    def modify_dag(self, reasoner: Reasoner, user_request: str) -> Dict[str, Any]:
        """
        Based on the user's natural language modification request, regenerate the subquery_plan.
        """
        if reasoner is None:
            raise ValueError("reasoner 不能为空")

        normalized_request = (user_request or "").strip()
        if not normalized_request:
            raise ValueError("user_request 不能为空字符串")

        if not self.subquery_plan or not self.subquery_plan.get("subqueries"):
            raise ValueError("当前 DAG 还没有可用的 subquery_plan")

        updated_plan = reasoner.revise_subquery_plan(
            current_plan=self.get_subquery_plan(),
            user_request=normalized_request
        )
        self.build_from_subquery_plan(updated_plan)
        return updated_plan

    def _validate_subquery_plan_structure(self, plan: Dict[str, Any]) -> None:
        """
        基础结构校验，确保 plan 至少包含合法的 subqueries 字段。
        """
        if not isinstance(plan, dict):
            raise ValueError("subquery_plan 必须是 dict")

        subqueries = plan.get("subqueries")
        if not isinstance(subqueries, list):
            raise ValueError("subquery_plan['subqueries'] 必须是 list")

        for idx, item in enumerate(subqueries):
            if not isinstance(item, dict):
                raise ValueError(f"subqueries[{idx}] 必须是 dict")
            if "id" not in item:
                raise ValueError(f"subqueries[{idx}] 缺少 'id'")
            if "query" not in item:
                raise ValueError(f"subqueries[{idx}] 缺少 'query'")
            depends_on = item.get("depends_on", [])
            if depends_on is None:
                continue
            if not isinstance(depends_on, list):
                raise ValueError(
                    f"subqueries[{idx}]['depends_on'] 必须是 list 或省略")
