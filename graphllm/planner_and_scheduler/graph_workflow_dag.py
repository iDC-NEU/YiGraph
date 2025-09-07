# graph_workflow_dag.py
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Iterable, Literal
from datetime import datetime
import json
import collections

# ---- Step 定义 --------------------------------------------------------------

class StepType(str, Enum):
    RETRIEVAL = "retrieval"
    PLANNING = "planning"
    GRAPH_ALGORITHM = "graph_algorithm"
    LLM_INTERACTION = "llm_interaction"
    AGGREGATION = "aggregation"


@dataclass
class WorkflowStep:
    step_id: int
    step_type: StepType                 # "retrieval" | "planning" | "graph_algorithm" | "llm_interaction" | "aggregation" | ...
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_data: Any = None
    output_data: Any = None
    status: str = "pending"        # "pending" | "running" | "success" | "failed" | "skipped"
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # input/output 可能是不可序列化对象，这里尽量转成字符串，避免导出报错
        if not _is_jsonable(d["input_data"]):
            d["input_data"] = str(d["input_data"])
        if not _is_jsonable(d["output_data"]):
            d["output_data"] = str(d["output_data"])
        return d


def _is_jsonable(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


def _frozen_params(step_type: StepType, params: Dict[str, Any]) -> str:
    """
    生成一个可哈希的签名，用于去重/命中缓存：
    - 步骤类型 + 排序后的参数 JSON
    """
    return json.dumps({"t": step_type, "p": params}, sort_keys=True, ensure_ascii=False)


# ---- DAG 管理器 ------------------------------------------------------------

class GraphWorkflowDAG:
    """
    面向图任务的 LLM+Graph 工作流内存（DAG 模式）：
    - 记录每一步（节点）
    - 维护依赖（有向无环图）
    - 支持拓扑排序/就绪队列/回滚/导出/可解释
    - 简单去重缓存：相同(step_type, parameters) 可复用结果
    """

    def __init__(self):
        self.steps: Dict[int, WorkflowStep] = {}
        self.out_edges: Dict[int, Set[int]] = collections.defaultdict(set)  # parent -> children
        self.in_edges: Dict[int, Set[int]] = collections.defaultdict(set)   # child  -> parents
        self.next_step_id: int = 0
        # （可选）结果缓存：signature -> step_id
        self.signature_index: Dict[str, int] = {}

    # -- 增加步骤 & 依赖 ------------------------------------------------------

    def add_step(
        self,
        step_type: StepType,
        description: str,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        input_data: Any = None,
        parents: Optional[Iterable[int]] = None,
        allow_dedup: bool = True
    ) -> int:
        """
        新增步骤节点，可一次性指定多个父依赖。
        若 allow_dedup=True，会尝试用 (step_type, parameters) 去重并复用历史成功结果。
        返回 step_id。
        """
        parameters = parameters or {}

        if allow_dedup:
            sig = _frozen_params(step_type, parameters)
            hit = self.signature_index.get(sig)
            if hit is not None and self.steps[hit].status == "success":
                # 复用节点：只需在 DAG 上把 parents 指向这个已存在节点
                sid = hit
                if parents:
                    for p in parents:
                        self._assert_step_exists(p)
                        self._add_edge(p, sid)
                return sid

        self.next_step_id += 1
        sid = self.next_step_id
        step = WorkflowStep(
            step_id=sid,
            step_type=step_type,
            description=description,
            parameters=parameters,
            input_data=input_data,
            status="pending"
        )
        self.steps[sid] = step
        if parents:
            for p in parents:
                self._assert_step_exists(p)
                self._add_edge(p, sid)

        # 插入后检查是否形成环
        self._assert_acyclic()

        # 记录签名索引（即便未完成，也可用于避免重复创建节点；真正复用结果看 status）
        self.signature_index[_frozen_params(step_type, parameters)] = sid
        return sid

    def add_dependency(self, parent_id: int, child_id: int):
        """后补一条依赖边。"""
        self._assert_step_exists(parent_id)
        self._assert_step_exists(child_id)
        self._add_edge(parent_id, child_id)
        self._assert_acyclic()

    def _add_edge(self, parent_id: int, child_id: int):
        if parent_id == child_id:
            raise ValueError("不允许自环依赖")
        self.out_edges[parent_id].add(child_id)
        self.in_edges[child_id].add(parent_id)

    def _assert_step_exists(self, sid: int):
        if sid not in self.steps:
            raise KeyError(f"step_id {sid} 不存在")

    def _assert_acyclic(self):
        # Kahn 拓扑检测：若剩余节点不为 0 则有环
        indeg = {v: len(self.in_edges[v]) for v in self.steps.keys()}
        q = collections.deque([v for v, d in indeg.items() if d == 0])
        visited = 0
        while q:
            u = q.popleft()
            visited += 1
            for w in self.out_edges.get(u, []):
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
        if visited != len(self.steps):
            raise ValueError("检测到循环依赖（DAG 不能有环）")

    # -- 更新状态/写回结果 ----------------------------------------------------

    def set_running(self, step_id: int):
        self._assert_step_exists(step_id)
        self.steps[step_id].status = "running"

    def set_success(self, step_id: int, output_data: Any = None):
        self._assert_step_exists(step_id)
        s = self.steps[step_id]
        s.status = "success"
        s.output_data = output_data

    def set_failed(self, step_id: int, error: str):
        self._assert_step_exists(step_id)
        s = self.steps[step_id]
        s.status = "failed"
        s.error = error

    def set_skipped(self, step_id: int, reason: str = ""):
        self._assert_step_exists(step_id)
        s = self.steps[step_id]
        s.status = "skipped"
        s.error = reason or s.error

    # -- 查询/计划 ------------------------------------------------------------

    def topological_order(self) -> List[int]:
        """返回 DAG 的一个拓扑序。"""
        indeg = {v: len(self.in_edges[v]) for v in self.steps.keys()}
        q = collections.deque([v for v, d in indeg.items() if d == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for w in self.out_edges.get(u, []):
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
        if len(order) != len(self.steps):
            raise ValueError("检测到循环依赖（无法拓扑排序）")
        return order

    def ready_steps(self) -> List[int]:
        """
        返回“就绪步骤”：自身 pending 且 所有父节点 status 为 success。
        可用于调度器按批执行。
        """
        ready = []
        for sid, step in self.steps.items():
            if step.status != "pending":
                continue
            parents = self.in_edges.get(sid, set())
            if all(self.steps[p].status == "success" for p in parents):
                ready.append(sid)
        return ready

    def parents_of(self, step_id: int) -> List[int]:
        self._assert_step_exists(step_id)
        return list(self.in_edges.get(step_id, set()))

    def children_of(self, step_id: int) -> List[int]:
        self._assert_step_exists(step_id)
        return list(self.out_edges.get(step_id, set()))

    def ancestors_of(self, step_id: int) -> Set[int]:
        self._assert_step_exists(step_id)
        seen, stack = set(), list(self.in_edges.get(step_id, set()))
        while stack:
            u = stack.pop()
            if u in seen: 
                continue
            seen.add(u)
            stack.extend(self.in_edges.get(u, set()))
        return seen

    def descendants_of(self, step_id: int) -> Set[int]:
        self._assert_step_exists(step_id)
        seen, stack = set(), list(self.out_edges.get(step_id, set()))
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(self.out_edges.get(u, set()))
        return seen

    # -- 回滚 ---------------------------------------------------------------

    def rollback_to(self, keep_upto_step_id: int):
        """
        回滚：只保留 step_id <= keep_upto_step_id 的节点及其边，删除其后的所有后继。
        """
        self._assert_step_exists(keep_upto_step_id)
        # 要保留的集合
        keep = {sid for sid in self.steps if sid <= keep_upto_step_id}
        # 同时清理这些节点之间的边之外的其它所有节点
        remove = [sid for sid in self.steps.keys() if sid not in keep]
        for sid in remove:
            # 从图中摘除边
            for p in self.in_edges.get(sid, set()):
                self.out_edges[p].discard(sid)
            for c in self.out_edges.get(sid, set()):
                self.in_edges[c].discard(sid)
            self.in_edges.pop(sid, None)
            self.out_edges.pop(sid, None)
            # 删除节点
            self.steps.pop(sid, None)
        # 修正 next_step_id（不回退编号，以免与历史冲突；也可以选择回退）
        # self.next_step_id = max(keep) if keep else 0

    # -- 导出/可视化 ---------------------------------------------------------

    def export_as_dict(self) -> Dict[str, Any]:
        """
        导出 DAG：节点列表 + 边列表
        """
        nodes = [self.steps[sid].to_dict() for sid in sorted(self.steps)]
        edges = []
        for u, outs in self.out_edges.items():
            for v in outs:
                edges.append({"from": u, "to": v})
        return {"nodes": nodes, "edges": edges}

    def export_as_json(self, filepath: str):
        obj = self.export_as_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def to_dot(self) -> str:
        """
        生成 Graphviz DOT 字符串，便于可视化。
        """
        lines = ["digraph G {"]
        for sid, s in self.steps.items():
            label = f"{sid}:{s.step_type}\\n{s.status}"
            lines.append(f'  {sid} [label="{label}"];')
        for u, outs in self.out_edges.items():
            for v in outs:
                lines.append(f"  {u} -> {v};")
        lines.append("}")
        return "\n".join(lines)