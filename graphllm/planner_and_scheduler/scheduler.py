# scheduler_with_dag.py
from __future__ import annotations
from typing import Any, Dict, Optional, Callable, List
import time
import traceback
from graphllm.planner_and_scheduler.graph_workflow_dag import GraphWorkflowDAG, WorkflowStep

class Scheduler:
    """
    统一调度器：内部包含4类执行能力（检索 / 图计算 / 图学习 / LLM），
    并持有 DAG。根据 DAG 的依赖关系决定执行顺序。
    """
    def __init__(
        self,
        *,
        graph_engine: Any = None,   # 检索/子图/图算法等底层图后端
        gnn_engine: Any = None,     # 图学习后端（PyG/DGL等）
        llm_client: Any = None,     # LLM 客户端
        max_retries: int = 0,
        retry_sleep_sec: float = 0.0,
    ):
        self.dag: Optional[GraphWorkflowDAG] = None
        self.graph_engine = graph_engine
        self.gnn_engine = gnn_engine
        self.llm_client = llm_client

        self.max_retries = max_retries
        self.retry_sleep_sec = retry_sleep_sec

        # 可选回调：用于埋点/日志
        self.on_step_start: Optional[Callable[["WorkflowStep"], None]] = None
        self.on_step_end: Optional[Callable[["WorkflowStep"], None]] = None

        # 内部映射：step_type -> 执行方法
        # 你可以根据系统里定义的 step_type 自行扩展/调整
        self._dispatch = {
            "retrieval":        self._exec_retrieval,
            "graph_algorithm":  self._exec_graph_algorithm,
            "graph_learning":   self._exec_graph_learning,
            "llm_interaction":  self._exec_llm_interaction,
            "planning":         self._exec_llm_planning,   # 若有规划型步骤
            "aggregation":      self._exec_aggregation,    # 若需要聚合/汇总
        }

    
    def build_dag_from_subquery_plan(self, subquery: Dict[str, Any]) -> GraphWorkflowDAG:
        """
        根据 JSON 格式的 subquery 构造 DAG。
        TODO: 实现从 subquery 到 GraphWorkflowDAG 的转换逻辑。
        """
        pass

    # ----------------- 外部入口 -----------------

    def run(self, *, stop_on_fail: bool = True) -> None:
        """
        顺序调度执行：不断消费 DAG 的就绪步骤。
        """
        while True:
            ready: List[int] = self.dag.ready_steps()
            if not ready:
                # 没有就绪节点：若还有 pending，则说明被失败阻塞或已无可前进
                pending_exists = any(s.status == "pending" for s in self.dag.steps.values())
                if not pending_exists:
                    # 全部完成
                    break
                # 无就绪但仍有 pending：结束（等待外部处理/回滚/修复）
                break

            for sid in ready:
                step = self.dag.steps[sid]
                fn = self._dispatch.get(step.step_type)
                if fn is None:
                    self.dag.set_failed(sid, error=f"Unsupported step_type={step.step_type}")
                    if stop_on_fail:
                        return
                    else:
                        continue

                if self.on_step_start:
                    self.on_step_start(step)

                self.dag.set_running(sid)

                tries = 0
                while True:
                    try:
                        output = fn(step_id=sid)  # 执行内部“执行器”
                        self.dag.set_success(sid, output_data=output)
                        break
                    except Exception as e:
                        tries += 1
                        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                        if tries <= self.max_retries:
                            time.sleep(self.retry_sleep_sec)
                            continue
                        self.dag.set_failed(sid, error=err)
                        if stop_on_fail:
                            return
                        break

                if self.on_step_end:
                    self.on_step_end(step)

    # ----------------- 内部执行器 -----------------

    def _exec_retrieval(self, *, step_id: int) -> Any:
        """
        检索/子图构建等。约定 parameters 里包含 node/depth 等。
        从父节点拿输入（如上游 planning 的结果）也可以。
        """
        step  = self.dag.steps[step_id]
        params = step.parameters
        node  = params.get("node")
        depth = int(params.get("depth", 1))

        # 从父节点拿到上游结果（如需）
        parents = self.dag.parents_of(step_id)
        parent_outputs = [self.dag.get_result(pid) for pid in parents]

        # 具体调用你的图引擎（此处示例）
        # subgraph = self.graph_engine.k_hop_neighbors(node=node, k=depth)
        subgraph = {"nodes": ["C_1", "C_2"], "edges": [("C_0","C_1"),("C_1","C_2")], "src": node, "depth": depth,
                    "evidence_from_parents": parent_outputs}
        return subgraph

    def _exec_graph_algorithm(self, *, step_id: int) -> Any:
        """
        图算法执行：如 connected_components、pagerank、shortest_path 等。
        约定 parameters["algo"] 指明算法类型。
        """
        step = self.dag.steps[step_id]
        algo = step.parameters.get("algo")
        parents = self.dag.parents_of(step_id)
        # 通常取第一个父的子图输入；或做多父合并
        subgraph = self.dag.get_result(parents[0]) if parents else None

        # 具体算法分派（示例）
        if algo in ("connected_components", "cc"):
            # result = self.graph_engine.connected_components(subgraph)
            result = {"components": 3, "labels": {"C_1":0, "C_2":0}}
        elif algo in ("pagerank", "pr"):
            # result = self.graph_engine.pagerank(subgraph)
            result = {"scores": {"C_1":0.42, "C_2":0.58}}
        elif algo == "shortest_path":
            src = step.parameters.get("src"); dst = step.parameters.get("dst")
            # result = self.graph_engine.shortest_path(subgraph, src, dst)
            result = {"path": ["C_0","C_1","C_2"], "length": 2, "src": src, "dst": dst}
        else:
            raise ValueError(f"Unknown graph algorithm: {algo}")
        return result

    def _exec_graph_learning(self, *, step_id: int) -> Any:
        """
        图学习：节点分类/链路预测/图分类等。把父步骤产出的子图/特征喂给 GNN。
        约定 parameters["task"] 指定任务。
        """
        step = self.dag.steps[step_id]
        task = step.parameters.get("task", "node_classification")
        parents = self.dag.parents_of(step_id)
        subgraph = self.dag.get_result(parents[0]) if parents else None

        # 具体的训练/推理调用替换成你的 gnn_engine
        # logits = self.gnn_engine.infer(subgraph, **step.parameters)
        if task == "node_classification":
            logits = {"preds": {"C_1": 1, "C_2": 0}}
        elif task == "link_prediction":
            logits = {"links": [( "C_1","C_2", 0.91)]}
        else:
            raise ValueError(f"Unknown graph learning task: {task}")
        return logits

    def _exec_llm_interaction(self, *, step_id: int) -> Any:
        """
        LLM 解释/总结/对话：把证据链（祖先步骤）作为上下文，结合参数提示词，让 LLM 输出说明。
        """
        step = self.dag.steps[step_id]
        chain = self.dag.explain_path_to(step_id)
        context = [
            {"id": s.step_id, "type": s.step_type, "params": s.parameters, "out": s.output_data}
            for s in chain
        ]
        prompt = step.parameters.get("prompt", "请基于步骤证据给出解释。")
        # reply = self.llm_client.summarize(context=context, prompt=prompt)
        reply = f"[LLM] 依据 {len(context)} 步证据生成解释：节点 C_2 风险较高。"
        return {"explanation": reply, "context_size": len(context)}

    def _exec_llm_planning(self, *, step_id: int) -> Any:
        """
        若你让 LLM 生成或优化 plan，这里执行规划（也可把新步骤动态加入 DAG）。
        """
        step = self.dag.steps[step_id]
        question = step.parameters.get("q", "")
        # plan = self.llm_client.plan(question)
        plan = {"plan": ["retrieval(depth=2)", "cc", "pagerank", "llm_interaction"]}
        return plan

    def _exec_aggregation(self, *, step_id: int) -> Any:
        """
        聚合多个父节点输出，做最终汇总/打分/组装返回。
        """
        parents = self.dag.parents_of(step_id)
        outs = [self.dag.get_result(pid) for pid in parents]
        # 简单合并示例
        return {"aggregated": outs}