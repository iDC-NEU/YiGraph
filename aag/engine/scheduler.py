# scheduler_with_dag.py
import time
import traceback
import logging
import json
from argparse import OPTIONAL
from dataclasses import asdict
from typing import Any, Dict, Optional, Callable, List
from aag.config.engine_config import *
from aag.models.graph_workflow_dag import GraphWorkflowDAG, WorkflowStep
from aag.reasoner.model_deployment import Reasoner
from aag.engine.router import QueryRouter, QueryType
from aag.computing_engine.computing_engine import ComputingEngine
from aag.expert_search_engine.rag import ExpertSearchEngine
from aag.data_pipeline.data_transformer.dataset_manager import DatasetManager
from aag.config.data_upload_config import DatasetConfig

logger = logging.getLogger(__name__)

class Scheduler:
    """
    统一调度器：内部包含4类执行能力（检索 / 图计算 / 图学习 / LLM），
    并持有 DAG。根据 DAG 的依赖关系决定执行顺序。
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.reasoner: Optional[Reasoner] = None
        self.computing_engine: Optional[ComputingEngine] = None
        self.expert_search_engine: Optional[ExpertSearchEngine] = None
        self.dag: Optional[GraphWorkflowDAG] = None
        self.dataset_manager: Optional[DatasetManager] = None

        self.current_dataset : DatasetConfig = None
        
        self._initialize_components()

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


    def _initialize_components(self):
        """初始化各个组件"""
        print("Initializing Scheduler components...")

        self._init_reasoner()

        self._init_computing_engine()

        self._init_expert_search_engine()

        self._init_dataset_manager()

        self._init_router()

    
    def _init_reasoner(self):
        """初始化 reasoner 连接"""
        try:
            self.reasoner = Reasoner(self.config.reasoner)
            print("✓ Reasoner initialized")
        except Exception as e:
            print(f"✗ Reasoner initialization failed: {e}")
            raise
    
    def _init_computing_engine(self):
        """初始化 computing engine 连接"""
        try:
            self.computing_engine = ComputingEngine()
            print("✓ ComputingEngine initialized")
        except Exception as e:
            print(f"✗ ComputingEngine initialization failed: {e}")
            raise
    
    def _init_expert_search_engine(self):
        """初始化 expert search engine 连接"""
        try:
            self.expert_search_engine = ExpertSearchEngine(self.config.retrieval)
            print("✓ ExpertSearchEngine initialized")
        except Exception as e:
            print(f"✗ ExpertSearchEngine initialization failed: {e}")
            raise
    
    def _init_dataset_manager(self):
        """初始化 dataset manager 连接"""
        try:
            self.dataset_manager = DatasetManager()
            print("✓ DatasetManager initialized")
        except Exception as e:
            print(f"✗ DatasetManager initialization failed: {e}")
            raise


    def _init_router(self):
        try:
            self.router = QueryRouter(reasoner=self.reasoner)
            print("✓ Reasoner initialized")
        except Exception as e:
            print(f"✗ Reasoner initialization failed: {e}")
            raise

    def list_datasets(self, dtype: Optional[str] = None) -> Dict[str, List[str]]:
        return self.dataset_manager.list_datasets(dtype)

    def specific_analysis_dataset(self, name: str, dtype: Optional[str] = None) -> DatasetConfig: 
        self.current_dataset  = self.dataset_manager.get_dataset_info(name, dtype)
        return self.current_dataset

    async def execute(self, query: str, decompose: bool = True) -> str:
        
        decision = self.router.route(
            query=query,
        )

        logger.info(f"🚦[Router] query_type={decision.query_type}, reason={decision.reason}")

        # 1) graph analysis
        if decision.query_type == QueryType.GRAPH:
            return await self._execute_graph(query, decompose=decompose)

        # 2) rag task
        elif decision.query_type == QueryType.RAG:

            raise NotImplementedError
            # return await self.rag_engine.answer(query)

        # 3) general query
        return self.reasoner.general_query_response(query)
    

    async def _execute_graph(self, query: str, decompose: bool = True) -> str:
        if not self.computing_engine._initialized:
            await self.computing_engine.initialize()

        # 补充代码：首先验证 self.current_dataset 是否为空，如果为空的话，输出 没有指定分析的数据
        # 若未指定分析数据集，先提醒用户
        if not self.current_dataset:
            return "⚠️ 未指定分析数据，请先设置分析对象"
        
        # 🔹 若是图数据，先通过 GraphDataLoader 获取顶点和边
        if self.current_dataset.type == "graph":
            try:
                vertices, edges = self.dataset_manager.get_dataset_content(self.current_dataset)

                result = await self.computing_engine.run_algorithm("initialize_graph", {
                "vertices": [v.to_dict() for v in vertices],
                "edges": [e.to_dict() for e in edges],
                "dataset_config": self.current_dataset.to_dict()
                })

                if not result.get('success'):
                    raise Exception(result.get('error', '初始化失败'))
            except Exception as e:
                return f"❌ 加载图数据失败：{e}"

        # step1. 根据 query 解析 生成 dag
        self._build_dag_from_query(query, decompose)

        
        # step2. 遍历每个 dag 的节点，确定算法
        self._find_algorithm()
        self.dag.print_dag_info()
        # step3. 根据每个节点问题内容和图算法得到数据依赖
        self.dag.refresh_data_dependency(self.reasoner)
        self.dag.print_data_dependency()
        print("✅ DAG 构建与算法选择完成，准备执行计算流程")

        # step4. 根据每个问题确定的算法，调度算法执行
        return await self._run_algorithm_pipeline()

      # step5. 整理计算结果，输出报告


        # dag = self.reasoner.parse_query(query)
        # for node in dag.topological_sort():
        #     algo = self.reasoner.refine_subtask(node)
        #     if algo.needs_expert_knowledge:
        #         knowledge = self.expert_search_engine.search_algorithm_docs(algo)
        #     result = self.computing_engine.run_algorithm(node, algo, knowledge)
        #     self.reasoner.explain_result(node, result)
        # return self.reasoner.generate_report(dag)

    def build_dag_from_subquery_plan(self, subquery_plan: Dict[str, Any]) -> GraphWorkflowDAG:
        """
        根据 JSON 格式的 subquery 构造 DAG。
        
        Args:
            subquery_plan: 子查询计划，格式为:
                {
                    "subqueries": [
                        {
                            "id": "q1",
                            "query": "问题描述",
                            "depends_on": ["q0"]  # 依赖的其他子查询ID列表
                        },
                        ...
                    ]
                }
        
        Returns:
            GraphWorkflowDAG: 构建完成的DAG对象
            
        Raises:
            ValueError: 如果输入格式不正确或存在循环依赖
        """
        # 创建新的DAG实例
        self.dag = GraphWorkflowDAG()
        
        # 提取子查询列表
        subqueries = subquery_plan.get("subqueries", [])
        if not subqueries:
            raise ValueError("子查询计划为空，必须包含至少一个子查询")
        
        # 步骤1: 建立查询ID到步骤ID的映射
        query_id_to_step_id = {}
        
        # 步骤2: 为每个子查询创建DAG步骤
        logger.info(f"⚙️ 正在创建 {len(subqueries)} 个DAG步骤·...")
        for subquery in subqueries:
            # 验证子查询格式
            if "id" not in subquery:
                raise ValueError("子查询缺少必需的'id'字段")
            if "query" not in subquery:
                raise ValueError("子查询缺少必需的'query'字段")
            
            query_id = subquery["id"]
            question = subquery["query"]
            
            # 创建步骤（图算法设置为None）
            step_id = self.dag.add_step(
                question=question,
                graph_algorithm=None
            )
            
            query_id_to_step_id[query_id] = step_id
            logger.info(f"⚙️  创建步骤 {step_id} for 查询 '{query_id}': {question[:50]}...")
        
        # 步骤3: 建立依赖关系
        logger.info("⚙️ 正在建立依赖关系...")
        for subquery in subqueries:
            query_id = subquery["id"]
            depends_on = subquery.get("depends_on", [])
            
            current_step_id = query_id_to_step_id[query_id]
            
            # 为每个依赖关系添加边
            for parent_query_id in depends_on:
                if parent_query_id not in query_id_to_step_id:
                    raise ValueError(f"依赖的查询ID '{parent_query_id}' 在子查询列表中不存在")
                
                parent_step_id = query_id_to_step_id[parent_query_id]
                print(f"  添加依赖: {parent_step_id}({parent_query_id}) -> {current_step_id}({query_id})")
                
                # 添加边，如果产生环会自动抛出异常
                self.dag.add_dependency(parent_step_id, current_step_id)
        
        # 步骤4: 验证DAG并获取拓扑序
        try:
            topological_order = self.dag.topological_order()
            logger.info(f"✅ DAG构建成功！拓扑序: {topological_order}")
        except ValueError as e:
            raise ValueError(f"DAG构建失败，检测到循环依赖: {e}")
        
        # 步骤5: 存储查询ID映射（用于后续查询）
        self.query_id_mapping = query_id_to_step_id
        
        return self.dag


    def _build_dag_from_query(self, query: str, decompose: bool = True) -> GraphWorkflowDAG:
        """
            将用户查询转换为DAG
        """
        subquery_plan = self.reasoner.plan_subqueries(decompose, query)
        return self.build_dag_from_subquery_plan(subquery_plan)


    def _find_algorithm(self):
        """
           遍历dag，对每个节点，确定适合执行的图算法
           函数逻辑：
             1. 遍历每个dag的节点，对每个节点做：
                 获取每个节点的 question, 确定每个节点适合执行的算法，并设置每个节点的graph_algorithm字段
        """
        for step in self.dag.steps.values():
            task_type_list = self.expert_search_engine.retrieve_task_type(step.question)
            selected_task_type_id = self.reasoner.select_task_type(step.question, task_type_list).get("id")
            algorithm_list = self.expert_search_engine.retrieve_algorithm(step.question, selected_task_type_id)
            selected_algorithm_id = self.reasoner.select_algorithm(step.question, algorithm_list).get("id")
            step.task_type = selected_task_type_id
            step.graph_algorithm = selected_algorithm_id
            logger.info(f"✅ 算法已选择 | ❓ 问题:{step.question}, 🧩 选择的任务类型: {step.task_type}, 🔍 选择的算法: {step.graph_algorithm}")

    async def _run_algorithm_pipeline(self):
        #  按照 dag 的顺序，对每个节点，执行算法
        #  对于每个节点来说：
        #   （1） 首先获取这个节点的算法，然后调用  computing_engine的get_algorithm_description 函数，获取注册的tool 信息。
        #   （2） 将这个节点的问题 和 tool信息 一起喂给 self.reasoner 的 xxx函数，该函数的作用是根据问题和tool的信息，从问题中提取参数并生成后处理代码。 根据这个函数的作用，给这个函数起个名字。 这个函数的内容先不用写
        #   （3） 调用computing_engine的run_algorithm 执行函数，获取执行结果 tool_result
        #   （4）  调用  self.reasoner 和分析结果生成回答， 同样起名字，不写函数内容
        analysis_result = ""
        for step_id in self.dag.topological_order():
            step = self.dag.steps[step_id]
            if not step.graph_algorithm:
                self.dag.set_failed(step_id, "未为该节点选择图算法")
                raise RuntimeError(f"节点 {step_id} 缺少 graph_algorithm")

            # 获取上层依赖节点的计算结果
            # parents = self.dag.parents_of(step_id)
            # upstream_results = [self.dag.steps[parent_id].result for parent_id in parents]

            tool_description = await self.computing_engine.get_algorithm_description(
                step.graph_algorithm
            )

            extraction_result = self.reasoner.extract_parameters_with_postprocess(
                question=step.question,
                tool_description=tool_description
            )

            # print(f"extraction_result:{extraction_result}")

            logger.info(
                f"✅ LLM 提取的参数: {json.dumps(extraction_result.get('parameters'), ensure_ascii=False)}")

            if extraction_result.get("post_processing_code"):
                logger.info(
                    f"✅ 后处理代码:\n{extraction_result.get('post_processing_code','')[:200]}...")

            tool_result = await self.computing_engine.run_algorithm(
                step.graph_algorithm,
                extraction_result.get("parameters", {}),
                extraction_result.get("post_processing_code", "")
            )

            logger.info(f"✅ 工具执行: {tool_result.get('summary','完成')}")

            if not tool_result.get("success", False):
                error_msg = tool_result.get("error", "算法执行失败")
                self.dag.set_failed(step_id, error_msg)
                raise RuntimeError(f"节点 {step_id} 执行失败：{error_msg}")

            answer = self.reasoner.generate_answer_from_algorithm_result(
                question=step.question,
                tool_description=tool_description,
                tool_result=tool_result,
            )

            analysis_result += answer

            self.dag.set_success(
                step_id,
                {
                    "tool_result": tool_result,
                    "answer": answer
                },
            )

        return analysis_result

    async def shutdown(self):
        if self.computing_engine:
            await self.computing_engine.shutdown()
        


    # 
    # 
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
        # subgraph = self.graph_computing_engine.k_hop_neighbors(node=node, k=depth)
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
            # result = self.graph_computing_engine.connected_components(subgraph)
            result = {"components": 3, "labels": {"C_1":0, "C_2":0}}
        elif algo in ("pagerank", "pr"):
            # result = self.graph_computing_engine.pagerank(subgraph)
            result = {"scores": {"C_1":0.42, "C_2":0.58}}
        elif algo == "shortest_path":
            src = step.parameters.get("src"); dst = step.parameters.get("dst")
            # result = self.graph_computing_engine.shortest_path(subgraph, src, dst)
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