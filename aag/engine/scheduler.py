# scheduler_with_dag.py
import time
import traceback
import logging
import json
from argparse import OPTIONAL
from dataclasses import asdict
from typing import Any, Dict, Optional, Callable, List, Tuple, Union

from sympy import N
from aag.config.engine_config import *
from aag.models.graph_workflow_dag import GraphWorkflowDAG, WorkflowStep, OutputSchema, OutputField
from aag.models.task_types import GraphAnalysisSubType, GraphAnalysisType
from aag.reasoner.model_deployment import Reasoner
from aag.engine.router import QueryRouter, QueryType
from aag.computing_engine.computing_engine import ComputingEngine
from aag.expert_search_engine.search import ExpertSearchEngine
from aag.data_pipeline.data_transformer.dataset_manager import DatasetManager
from aag.config.data_upload_config import DatasetConfig
from aag.engine.dependency_resolver import DataDependencyResolver
from aag.expert_search_engine.database.datatype import VertexData, EdgeData, GraphData
from aag.utils.graph_conversion import flatten_graph, reconstruct_graph
from aag.utils.data_utils import take_sample
from aag.rag_engine.vector_rag import VectorRAG
from aag.reasoner.prompt_template.llm_prompt_en import rag_prompt

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
        self.data_dependency_resolver: Optional[DataDependencyResolver] = None

        self.current_dataset_name: Optional[str] = None  # Dataset-level name (for text datasets, this is the dataset name)
        self.current_dataset: Optional[List[DatasetConfig]] = None  # Original dataset config (text/graph, never changes)
        self.current_graph_dataset: Optional[DatasetConfig] = None  # Current graph dataset config (may be converted graph)
        self.global_graph: Optional[GraphData] = None

        self._initialize_components()

        # 可选回调：用于埋点/日志
        self.on_step_start: Optional[Callable[["WorkflowStep"], None]] = None
        self.on_step_end: Optional[Callable[["WorkflowStep"], None]] = None



    def _initialize_components(self):
        """初始化各个组件"""
        print("Initializing Scheduler components...")

        self._init_reasoner()

        self._init_computing_engine()

        self._init_expert_search_engine()

        self._init_dataset_manager()

        self._init_data_dependency_resolver()
        
        self._init_router()

        self._init_rag_engine()

    
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
    
    def _init_data_dependency_resolver(self):
        """初始化 data dependency resolver 连接"""
        try:
            self.data_dependency_resolver = DataDependencyResolver(self.reasoner)
            print("✓ DataDependencyResolver initialized")
        except Exception as e:
            print(f"✗ DataDependencyResolver initialization failed: {e}")
            raise

    def _init_router(self):
        try:
            self.router = QueryRouter(reasoner=self.reasoner)
            print("✓ Reasoner initialized")
        except Exception as e:
            print(f"✗ Reasoner initialization failed: {e}")
            raise

    def _init_rag_engine(self):
        try:
            self.rag_engine = VectorRAG(self.config.retrieval)
            print(f"✓ RAGEngine initialized")
        except Exception as e:
            print(f"✗ RAGEngineinitialization failed: {e}")
            raise

    def list_datasets(self, dtype: Optional[str] = None) -> Dict[str, List[str]]:
        return self.dataset_manager.list_datasets(dtype)

    def specific_analysis_dataset(self, name: str, dtype: Optional[str] = None) -> Optional[List[DatasetConfig]]: 
        """
        Set the original dataset for analysis
        
        Args:
            name: Dataset name (dataset-level)
            dtype: Dataset type (optional)
            
        Returns:
            List[DatasetConfig] - list of configs (single for graph/table, multiple for text)
            None if not found
        """
        self.current_dataset = self.dataset_manager.get_dataset_info(name, dtype)
        
        # Store dataset-level name
        self.current_dataset_name = name
        if self.current_dataset is None or len(self.current_dataset) == 0:
            self.current_dataset = None
            self.current_graph_dataset = None
            self.global_graph = None
            return None
        
        # Handle graph datasets
        if self.dataset_manager.get_dataset_original_type(self.current_dataset_name) == "graph":
            self.current_graph_dataset = self.current_dataset[0]
            global_vertices, global_edges = self.dataset_manager.get_dataset_content(self.current_graph_dataset)
            self.global_graph = GraphData(vertices=global_vertices, edges=global_edges)
            graph_nodes, graph_edges = flatten_graph(global_vertices, global_edges)
            self.data_dependency_resolver.set_global_graph(graph_nodes, graph_edges)
        else:
            # Text/table dataset
            self.current_graph_dataset = None
            self.global_graph = None
        
        return self.current_dataset

    async def execute(self, query: str, decompose: bool = True) -> str:
        """
        Execute query with dataset type validation
        
        Validates:
        1. RAG query + graph dataset → Error (graph data cannot do retrieval)
        2. GRAPH query + text dataset → Check if converted graph exists
        """
        decision = self.router.route(query=query)
        logger.info(f"🚦[Router] query_type={decision.query_type}, reason={decision.reason}")

        if not self.current_dataset or not self.current_dataset_name:
            return "⚠️ 未指定数据集，请先设置分析对象"
        
        original_type = self.dataset_manager.get_dataset_original_type(self.current_dataset_name)
        
        # 1) RAG query validation
        if decision.query_type == QueryType.RAG:
            if original_type == "graph":
                return "❌ 错误：当前数据集是图数据，不支持检索任务。图数据只能用于图分析任务。"
            return await self._execute_rag(query)
        
        # 2) GRAPH query validation and graph dataset setup
        elif decision.query_type == QueryType.GRAPH:
            if original_type == "text":
                # Get converted graph dataset config (returns None if not converted)
                converted_graph_config = self.dataset_manager.get_converted_graph_dataset(self.current_dataset_name)
                if not converted_graph_config:
                    return "❌ 错误：当前数据集是文本数据，且尚未转换为图数据。请先进行文本到图的转换。"
                
                self.current_graph_dataset = converted_graph_config
                global_vertices, global_edges = self.dataset_manager.get_dataset_content(self.current_graph_dataset)
                self.global_graph = GraphData(vertices=global_vertices, edges=global_edges)
                graph_nodes, graph_edges = flatten_graph(global_vertices, global_edges)
                self.data_dependency_resolver.set_global_graph(graph_nodes, graph_edges)         
            
            return await self._execute_graph(query, decompose=decompose)
        
        # 3) General query
        return self.reasoner.general_query_response(query)
    

    async def _execute_graph(self, query: str, decompose: bool = True) -> str:
        """
        Execute graph analysis query
        
        Uses self.current_graph_dataset (which may be converted graph for text datasets)
        """
        if not self.computing_engine._initialized:
            await self.computing_engine.initialize()

        # Validate graph dataset is available
        if not self.current_graph_dataset:
            return "⚠️ 未指定图数据，请先设置分析对象"
        
        if not self.global_graph:
            return "⚠️ 图数据未加载，请先加载图数据"
        
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
        return await self._run_algorithm_pipeline2()

        # step5. 整理计算结果，输出报告



    async def _execute_rag(self, query: str) -> str:
        """
        Execute RAG query
        
        Uses self.current_dataset (original text dataset config, never changes)
        For text datasets, current_dataset is the first file config from the list
        """
        if not self.current_dataset:
            return "⚠️ 未指定分析数据，请先设置分析对象"
        
        # Validate dataset type
        if self.dataset_manager.get_dataset_original_type(self.current_dataset_name) != "text":
            return f"❌ 错误：RAG任务需要文本数据，但当前数据集类型是 {self.current_dataset.type}"
        
        # Use dataset-level name stored in specific_analysis_dataset
        # For RAG, we might need to handle multiple files
        # Currently use first file, but could be extended to handle all files
        first_file_config = self.current_dataset[0]
        file_path = first_file_config.schema.path

        if not self.rag_engine._initialized:
            self.rag_engine.initialize(db_name=self.current_dataset_name, file_path=file_path)

        retrieved_context, _ = self.rag_engine.retrieve(query)

        prompt = rag_prompt.format(context=retrieved_context, query=query)

        return self.reasoner.generate_response(prompt)


      # step5. 整理计算结果，输出报告

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
            # Step 1: Classify question type (graph_algorithm or numeric_analysis)
            question_classification = self.reasoner.classify_question_type(step.question)
            question_type = question_classification.get("type", "graph_algorithm")
            
            if question_type == "numeric_analysis":
                # If numeric analysis, set task type and skip graph algorithm selection
                step.task_type = GraphAnalysisType.NUMERIC_ANALYSIS
                step.graph_algorithm = None
                logger.info(f"✅ 问题分类为数值分析 | ❓ 问题:{step.question}, 🧩 任务类型: {step.task_type}, 原因: {question_classification.get('reason', '')}")
            else:
                # If graph algorithm, proceed with normal algorithm selection flow
                task_type_list = self.expert_search_engine.retrieve_task_type(step.question)
                selected_task_type_id = self.reasoner.select_task_type(step.question, task_type_list).get("id")
                algorithm_list = self.expert_search_engine.retrieve_algorithm(step.question, selected_task_type_id)
                selected_algorithm_id = self.reasoner.select_algorithm(step.question, algorithm_list).get("id")
                step.task_type = GraphAnalysisType.GRAPH_ALGORITHM
                step.graph_algorithm = selected_algorithm_id
                logger.info(f"✅ 算法已选择 | ❓ 问题:{step.question}, 🧩 选择的任务类型: {selected_task_type_id}, 🔍 选择的算法: {step.graph_algorithm}")

    async def _run_algorithm_pipeline(self):
        analysis_result = ""

        for step_id in self.dag.topological_order(): 
            step = self.dag.steps[step_id]

            if not step.graph_algorithm:
                self.dag.set_failed(step_id, "未为该节点选择图算法")
                raise RuntimeError(f"节点 {step_id} 缺少 graph_algorithm")

            # step 1. 获取算法描述（doc， tool_metadata)， doc 是 tool_metadata 的字符串形式
            tool_description,  tool_metadata = await self.computing_engine.get_algorithm_description(
                step.graph_algorithm
            )

            logger.info(f"tool_description:{tool_description}")

            extraction_result = self.reasoner.extract_parameters_with_postprocess(
                question=step.question,
                tool_description=tool_description
            )
            
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

            llm_analysis = self.reasoner.generate_answer_from_algorithm_result(
                question=step.question,
                tool_description=tool_description,
                tool_result=tool_result,
            )

            analysis_result += llm_analysis

            self.dag.set_success(
                step_id,
                output_data = None,
                llm_analysis= llm_analysis
            )

        return analysis_result
    
    async def _run_algorithm_pipeline2(self):
        analysis_result = ""

        for step_id in self.dag.topological_order(): 
            step = self.dag.steps[step_id]
            tool_description = None
            tool_metadata = None

            if step.task_type == GraphAnalysisType.GRAPH_ALGORITHM:
                if not step.graph_algorithm:
                    self.dag.set_failed(step_id, "未为该节点选择图算法")
                    raise RuntimeError(f"节点 {step_id} 缺少 graph_algorithm")

                 # step 1. 获取算法描述（doc， tool_metadata)， doc 是 tool_metadata 的字符串形式
                tool_description,  tool_metadata = await self.computing_engine.get_algorithm_description(
                    step.graph_algorithm
                )   
                # logger.info(f"tool_description:{tool_description}")

            # step 2. 上游依赖解析
            data_dependency_ids = self.dag.get_data_dependency(step_id)
            data_dependency_parents = [self.dag.steps[pid] for pid in data_dependency_ids]
            data_dependency_context = {
                "graph_dependencies": [],
                "parameter_dependencies": [],
                "graph_input_adapter_result": None,
                "parameter_input_adapter_result": None,
            }
            
            if data_dependency_parents:
                data_dependency_context = self.data_dependency_resolver.resolve_dependencies(
                    step_id=step_id,
                    step=step,
                    alg_des_info=tool_metadata,
                    data_dependency_parents=data_dependency_parents
                )
                logger.info(
                    "📊 依赖上下文 | 图依赖:%s | 参数依赖:%s",
                    len(data_dependency_context.get("graph_dependencies", [])),
                    len(data_dependency_context.get("parameter_dependencies", [])),
                )
                

            if step.task_type == GraphAnalysisType.GRAPH_ALGORITHM:
                try:
                    await self._prepare_graph_for_execution(
                        graph_dependencies=data_dependency_context.get("graph_dependencies") or [],
                        graph_adapter_result=data_dependency_context.get("graph_input_adapter_result"),
                    )
                except Exception as graph_err:
                    self.dag.set_failed(step_id, f"初始化工作图失败: {graph_err}")
                    raise

                try:
                    extraction_result = self._prepare_parameters_for_execution(
                        step=step,
                        tool_description=tool_description,
                        vertex_schema=self.global_graph.get_vertex_properties_schema(),
                        edge_schema=self.global_graph.get_edge_properties_schema(),
                        dependency_parameters=data_dependency_context.get("parameter_input_adapter_result") or {},
                    )
                except Exception as param_err:
                    self.dag.set_failed(step_id, f"参数准备失败: {param_err}")
                    raise

                logger.info(
                    "✅ LLM/依赖提取的参数: %s",
                    json.dumps(extraction_result.get("parameters", {}), ensure_ascii=False)
                )

                post_processing_info = extraction_result.get("post_processing_code") or {}  
                is_has_extract_code = bool(post_processing_info.get("is_calculate"))
                output_schema = post_processing_info.get("output_schema") or {}

                if post_processing_info.get("code"):
                    logger.info(
                        f"✅ 后处理代码:\n{post_processing_info.get('code','')}...")
        
                # 记录计算结果
                tool_result = await self.computing_engine.run_algorithm(
                    step.graph_algorithm,
                    extraction_result.get("parameters", {}),
                    post_processing_info.get("code"),
                    global_graph=self.global_graph
                )

                if tool_result.get("result").get("error"):
                    error_msg = tool_result.get("result").get("error")
                    self.dag.set_failed(step_id, error_msg)
                    raise RuntimeError(f"节点 {step_id}: {step.question} 执行失败: {error_msg}")

                if not tool_result.get("success", False):
                    error_msg = tool_result.get("error", "算法执行失败")
                    self.dag.set_failed(step_id, error_msg)
                    raise RuntimeError(f"节点 {step_id} 执行失败：{error_msg}")

                ## todo: 保存中间计算结果的模块: 如果图计算的结果规模特别大, 把结果写到一个文件里 
                # 添加算法执行结果到 step
                step.add_algorithm_result(
                    tool_name=tool_metadata.get("name", ""),
                    tool_result_data=tool_result.get("result", {}),
                    output_schema=output_schema,
                    is_has_extract_code=is_has_extract_code
                )
                logger.info(f"✅ 工具执行: {tool_result.get('summary','完成')}")

                self.dag.set_success(step_id)

            elif step.task_type == GraphAnalysisType.NUMERIC_ANALYSIS:
                logger.info("🧮 当前任务类型：Numeric Analysis")
                try:
                    # 1. 合并 graph_dependencies 和 parameter_dependencies 成依赖参数项
                    graph_deps = data_dependency_context.get("graph_dependencies", [])
                    param_deps = data_dependency_context.get("parameter_dependencies", [])
                    
                    dependency_items = []
                    execution_data = {}
                    for dep_item in graph_deps + param_deps:   
                        value = take_sample(dep_item.value)
                        execution_data[dep_item.field_key] = value
                        dependency_items.append({
                            "field_key": dep_item.field_key,
                            "field_type": str(dep_item.field_type) if dep_item.field_type else "unknown",
                            "field_desc": dep_item.field_desc,
                            "value": value
                        })
                
                    vertex_schema = {}
                    edge_schema = {}
                    if self.global_graph:
                        vertex_schema = self.global_graph.get_vertex_properties_schema()
                        edge_schema = self.global_graph.get_edge_properties_schema()
                        
                    code_result = self.reasoner.generate_numeric_analysis_code(
                        question=step.question,
                        dependency_items=dependency_items,
                        vertex_schema=vertex_schema,
                        edge_schema=edge_schema
                    )
                    
                    numeric_analysis_code = code_result.get("numeric_analysis_code", {})
                    generated_code = numeric_analysis_code.get("code", "")
                    output_schema = numeric_analysis_code.get("output_schema", {})
                    logger.info(f"✅ 生成的数值分析代码: {generated_code[:200]}...")
                    
                    code_result_value = self.computing_engine.execute_code(
                        code=generated_code,
                        data=execution_data,
                        global_graph=self.global_graph
                    )

                    if isinstance(code_result_value, dict) and "error" in code_result_value:
                        error_msg = code_result_value.get("error")
                        self.dag.set_failed(step_id, error_msg)
                        raise RuntimeError(f"节点 {step_id} 数值分析失败：{error_msg}")

                    step.add_output(
                        task_type=GraphAnalysisSubType.NUMERIC_COMPUTATION,
                        source="numeric analysis code",
                        output_schema=OutputSchema(
                            description=output_schema.get("description", ""),
                            type=output_schema.get("type", "dict"),
                            fields={
                                name: OutputField(
                                    type=info.get("type", ""),
                                    field_description=info.get("field_description", "")
                                )
                                for name, info in output_schema.get("fields", {}).items()
                            }
                        ) if output_schema else None,
                        value=code_result_value if isinstance(code_result_value, dict) else {"result": code_result_value},
                        path=None,
                        validate_schema=True
                    )
                    logger.info(f"✅ 数值分析执行完成")
                    self.dag.set_success(step_id)
                    
                except Exception as numeric_err:
                    error_msg = f"数值分析执行失败: {numeric_err}"
                    logger.error(error_msg, exc_info=True)
                    self.dag.set_failed(step_id, error_msg)
                    raise RuntimeError(f"节点 {step_id} 数值分析失败：{numeric_err}")

            llm_analysis = self.reasoner.generate_answer_from_algorithm_result(
                question=step.question,
                tool_description=tool_description,
                tool_result=tool_result,
            )
            analysis_result += llm_analysis

        return analysis_result

    async def _prepare_graph_for_execution(
        self,
        *,
        graph_dependencies: List[Any],
        graph_adapter_result: Optional[Dict[str, Any]],
    ) -> Tuple[List[VertexData], List[EdgeData]]:
        """
        根据依赖上下文初始化当前应使用的图（全图或子图）。
        """
        if self.global_graph is None:
            raise RuntimeError("全局图数据尚未加载，无法执行图算法。")

        use_subgraph = bool(graph_dependencies) and bool(graph_adapter_result)
        vertices_to_use = self.global_graph.vertices
        edges_to_use = self.global_graph.edges

        if use_subgraph:
            node_ids_raw = (graph_adapter_result or {}).get("nodes") or []
            edge_pairs_raw = (graph_adapter_result or {}).get("edges") or []
            # Convert to string and remove duplicates while preserving order
            node_ids = list(dict.fromkeys([str(n) for n in node_ids_raw]))
            edge_pairs: List[Tuple[str, str]] = []
            seen_edges = set()
            for edge in edge_pairs_raw:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    edge_pair = (str(edge[0]), str(edge[1]))
                    if edge_pair not in seen_edges:
                        edge_pairs.append(edge_pair)
                        seen_edges.add(edge_pair)
            if node_ids or edge_pairs:
                vertices_to_use, edges_to_use = reconstruct_graph(
                    node_ids,
                    edge_pairs,
                    self.global_graph.vertices,
                    self.global_graph.edges,
                )
            else:
                logger.warning("⚠️ 子图依赖缺少有效的节点或边数据，回退至全图。")
                use_subgraph = False

        payload = {
            "vertices": [v.to_dict() for v in vertices_to_use or []],
            "edges": [e.to_dict() for e in edges_to_use or []],
        }
        if self.current_graph_dataset:
            payload["dataset_config"] = self.current_graph_dataset.to_dict()

        result = await self.computing_engine.run_algorithm("initialize_graph", payload)
        if not result.get("success", False):
            raise RuntimeError(result.get("error", "初始化图数据失败"))

        logger.info(
            "🗺️ 初始化工作图 | 使用子图:%s | 节点:%s | 边:%s",
            use_subgraph,
            len(vertices_to_use or []),
            len(edges_to_use or []),
        )

        return vertices_to_use, edges_to_use

    def _prepare_parameters_for_execution(
        self,
        *,
        step: WorkflowStep,
        tool_description: Optional[str],
        vertex_schema: Dict[str, str],
        edge_schema: Dict[str, str],
        dependency_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
            根据数据依赖或 LLM 结果准备图算法输入参数。
        """
        dependency_parameters = dependency_parameters or {}
        result = None
        
        try:
            if dependency_parameters:
                result = self.reasoner.merge_parameters_from_dependencies(
                    question=step.question,
                    tool_description=tool_description,
                    vertex_schema=vertex_schema,
                    edge_schema=edge_schema,
                    dependency_parameters=dependency_parameters
                )
            else:
                result = self.reasoner.extract_parameters_with_postprocess_new(
                    question=step.question,
                    tool_description=tool_description,
                    vertex_schema=vertex_schema,
                    edge_schema=edge_schema,
                )
        except Exception as e:
            logger.info(f"Parameter processing failed: {e}")
            result = None

        return result

    async def shutdown(self):
        if self.computing_engine:
            await self.computing_engine.shutdown()
        