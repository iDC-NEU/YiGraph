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
from aag.error_recovery.error_manager import ErrorRecovery

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
        self.error_recovery: Optional[ErrorRecovery] = None  # 错误恢复模块

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
        
        self._init_error_recovery()

    
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
            
            # add gjq: 初始化computing_engine的图查询引擎
            neo4j_config_dict = self.config.retrieval.database.neo4j
            if neo4j_config_dict.get("enabled", False):
                self.computing_engine.initialize_graph_query_engine(
                    neo4j_config=neo4j_config_dict,
                    reasoner=self.reasoner
                )
                print("✓ ComputingEngine with GraphQueryEngine initialized")
            else:
                print("✓ ComputingEngine initialized (GraphQueryEngine disabled)")
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

    def _init_error_recovery(self):
        """初始化错误恢复模块"""
        try:
            self.error_recovery = ErrorRecovery()
            print("✓ ErrorRecoveryModule initialized")
        except Exception as e:
            print(f"✗ ErrorRecoveryModule initialization failed: {e}")
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
            
            # add gjq: 自动将图数据加载到Neo4j（调用DatasetManager的方法）
            neo4j_config_dict = self.config.retrieval.database.neo4j
            self.dataset_manager.load_graph_to_neo4j(
                self.current_graph_dataset,
                self.current_dataset_name,
                neo4j_config_dict
            )
        else:
            # Text/table dataset
            self.current_graph_dataset = None
            self.global_graph = None
        
        return self.current_dataset

    async def execute(
        self, 
        query: str, 
        decompose: bool = True, 
        mode: str = "normal",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute query with dataset type validation
        
        Args:
            query: 用户查询
            decompose: 是否分解查询
            mode: 执行模式 "normal" | "expert"
            callback: 可选的回调函数，用于实时发送数据（如DAG信息）
                     签名: callback(data: Dict[str, Any])
        
        Returns:
            普通模式: 返回分析结果字符串
            专家模式: 返回DAG信息字典（包含 dag_info, message 等）
        
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

        # 2) GRAPH_QUERY - 图查询（模板匹配查询）
        elif decision.query_type == QueryType.GRAPH_QUERY:
            if original_type != "graph":
                # 检查是否有转换后的图数据
                converted_graph_config = self.dataset_manager.get_converted_graph_dataset(self.current_dataset_name)
                if not converted_graph_config:
                    return "❌ 错误：图查询需要图数据，当前数据集不是图数据且未转换为图数据。"
                self.current_graph_dataset = converted_graph_config
            
            return await self._execute_graph_query(query)
        
        # 3) GRAPH query validation and graph dataset setup
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
            
            return await self._execute_graph(query, decompose=decompose, mode=mode, callback=callback)
        
        # 3) General query
        return self.reasoner.general_query_response(query)
    

    async def _execute_graph(
        self, 
        query: str, 
        decompose: bool = True, 
        mode: str = "normal",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute graph analysis query
        
        Args:
            query: 用户查询
            decompose: 是否分解查询
            mode: 执行模式 "normal" | "expert"
            callback: 可选的回调函数，用于实时发送数据（如DAG信息）
                     签名: callback(data: Dict[str, Any])
        
        Returns:
            普通模式: 返回分析结果字符串
            专家模式: 返回DAG信息字典
        
        Uses self.current_graph_dataset (which may be converted graph for text datasets)
        """
        if not self.computing_engine._initialized:
            await self.computing_engine.initialize()

        # Validate graph dataset is available
        if not self.current_graph_dataset:
            return "⚠️ 未指定图数据，请先设置分析对象"
        
        if not self.global_graph:
            return "⚠️ 图数据未加载，请先加载图数据"
        

        self._build_dag_from_query(query, decompose)
        self._find_algorithm()
        print("✅ 初始 DAG 构建与算法选择完成")
        self.dag.print_dag_info()
        
        if mode == "expert":
            dag_info = self.dag.get_dag_info()
            return {
                "message": "DAG已生成，请选择下一步操作",
                "dag_info": dag_info
            }
        
        # 普通模式：继续执行完整流程
        self.dag.refresh_data_dependency(self.reasoner)
        self.dag.print_data_dependency()
        print("✅ DAG 构建与算法选择完成，准备执行计算流程")
        
        # 执行算法流程
        analysis_result = await self._run_algorithm_pipeline2()
        
        # 如果提供了 callback（Web 调用），返回包含 DAG 信息的字典
        # 否则返回字符串（保持终端兼容性）
        if callback and mode == "normal":
            dag_info = self.dag.get_dag_info()
            return {
                "analysis_result": analysis_result,
                "dag_info": dag_info
            }
        
        return analysis_result

    async def expert_modify_dag(self, modification_request: str) -> Dict[str, Any]:
        """
        专家模式：根据用户需求修改DAG
        
        Args:
            modification_request: 用户修改需求（自然语言）
        
        Returns:
            包含更新后DAG信息的字典
        """
        if not self.dag:
            return {
                "error": "DAG尚未构建，请先输入问题生成DAG"
            }
        
        try:
            self.dag.modify_dag(self.reasoner, modification_request)
            self._find_algorithm()
            dag_info = self.dag.get_dag_info()
            return {
                "message": "DAG已更新",
                "dag_info": dag_info
            }
        except Exception as e:
            logger.error(f"修改DAG失败: {e}", exc_info=True)
            return {
                "error": f"修改DAG失败: {str(e)}"
            }

    async def expert_start_analysis(self) -> str:
        """
        专家模式：开始执行分析
        
        Returns:
            分析结果字符串
        """
        if not self.dag:
            return "❌ 错误：DAG尚未构建，请先输入问题生成DAG"
                
        self.dag.refresh_data_dependency(self.reasoner)
        self.dag.print_data_dependency()
        print("✅ DAG 构建与算法选择完成，准备执行计算流程")
        return await self._run_algorithm_pipeline2()

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
        
        file_paths = [config.schema.path for config in self.current_dataset]

        if not self.rag_engine._initialized:
            self.rag_engine.initialize(db_name=self.current_dataset_name, file_paths=file_paths)

        retrieved_context, _ = self.rag_engine.retrieve(query)

        prompt = rag_prompt.format(context=retrieved_context, query=query)

        return self.reasoner.generate_response(prompt)

    # add gjq: 修改为通过 computing_engine 调用图查询
    async def _execute_graph_query(self, query: str) -> str:
        """
        执行图查询（通过 computing_engine）
        
        Args:
            query: 用户查询
            
        Returns:
            查询结果字符串
        """
        try:
            # 通过 computing_engine 执行图查询
            result = self.computing_engine.execute_graph_query(query)
            
            if result.get("success"):
                # 格式化查询结果
                results = result.get("results", [])
                count = result.get("count", 0)
                query_type = result.get("query_type", "unknown")
                
                response = f"✅ 图查询成功！\n"
                response += f"查询类型: {query_type}\n"
                response += f"返回结果数: {count}\n\n"
                
                if count > 0:
                    response += "查询结果:\n"
                    for i, item in enumerate(results[:10], 1):  # 最多显示10条
                        # 将 Neo4j Node 对象转换为可序列化的字典
                        serializable_item = self._convert_neo4j_to_dict(item)
                        response += f"{i}. {json.dumps(serializable_item, ensure_ascii=False, indent=2)}\n"
                    
                    if count > 10:
                        response += f"\n... 还有 {count - 10} 条结果未显示"
                else:
                    response += "未找到匹配的结果"
                
                return response
            else:
                error_msg = result.get("error", "未知错误")
                return f"❌ 图查询失败: {error_msg}"
                
        except Exception as e:
            logger.error(f"图查询执行失败: {e}", exc_info=True)
            return f"❌ 图查询执行失败: {str(e)}"
    
    def _convert_neo4j_to_dict(self, obj):
        """
        将 Neo4j 对象转换为可 JSON 序列化的字典
        
        Args:
            obj: Neo4j 对象（Node, Relationship, Path 等）或普通对象
            
        Returns:
            可序列化的字典或原始对象
        """
        # 检查是否是 Neo4j Node 对象（通过检查是否有 labels 和 id 属性）
        if hasattr(obj, 'labels') and hasattr(obj, 'id') and callable(getattr(obj, 'items', None)):
            return {
                'id': obj.id,
                'labels': list(obj.labels),
                'properties': dict(obj)
            }
        # 检查是否是 Neo4j Relationship 对象（通过检查是否有 type 和 id 属性）
        elif hasattr(obj, 'type') and hasattr(obj, 'id') and callable(getattr(obj, 'items', None)):
            return {
                'id': obj.id,
                'type': obj.type,
                'start_node': obj.start_node.id if hasattr(obj, 'start_node') else None,
                'end_node': obj.end_node.id if hasattr(obj, 'end_node') else None,
                'properties': dict(obj)
            }
        # 检查是否是 Neo4j Path 对象（通过检查是否有 nodes 和 relationships 属性）
        elif hasattr(obj, 'nodes') and hasattr(obj, 'relationships'):
            return {
                'nodes': [self._convert_neo4j_to_dict(node) for node in obj.nodes],
                'relationships': [self._convert_neo4j_to_dict(rel) for rel in obj.relationships]
            }
        # 如果是字典，递归转换其值
        elif isinstance(obj, dict):
            return {k: self._convert_neo4j_to_dict(v) for k, v in obj.items()}
        # 如果是列表，递归转换其元素
        elif isinstance(obj, list):
            return [self._convert_neo4j_to_dict(item) for item in obj]
        # 其他类型直接返回
        else:
            return obj

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
        # 创建新的DAG实例并构建
        self.dag = GraphWorkflowDAG()
        query_id_to_step_id = self.dag.build_from_subquery_plan(subquery_plan)
        logger.info(f"✅ DAG构建成功！拓扑序: {self.dag.topological_order()}")
        self.query_id_mapping = query_id_to_step_id

        return self.dag


    def _get_algorithm_library_info(self) -> str:
        """
        Extract algorithm library information from expert_search_engine's knowledge base.
        
        Returns:
            Formatted string containing task types and their algorithms
        """
        if not self.expert_search_engine:
            return "Algorithm library not available"
        
        task_index = self.expert_search_engine.task_index
        algo_index = self.expert_search_engine.algo_index
        
        if not task_index or not algo_index:
            return "Algorithm library not loaded"
        
        library_info = []
        for task_id, task_data in task_index.items():
            task_type = task_data.get("task_type", "Unknown")
            description = task_data.get("description", "")
            algorithms = task_data.get("algorithm", [])
            
            # Get algorithm names
            algo_names = []
            for algo_id in algorithms[:5]:  # Limit to first 5 for brevity
                if algo_id in algo_index:
                    algo_names.append(algo_id)
            
            if algorithms:
                algo_list = ", ".join(algo_names)
                if len(algorithms) > 5:
                    algo_list += f" (and {len(algorithms) - 5} more)"
                library_info.append(
                    f"- **{task_type}**: {description}\n  Algorithms: {algo_list}"
                )
        
        return "\n".join(library_info)
    
    def _get_graph_schema_summary(self) -> Optional[str]:
        """
        Get a summary of the current graph dataset schema.
        
        Returns:
            Formatted string with dataset schema information, or None if not available
        """
        if not self.current_graph_dataset:
            return None
        
        try:
            schema_info = []
            schema_info.append(f"Dataset: {self.current_dataset_name}")
            
            # Graph properties
            if hasattr(self.current_graph_dataset.schema, 'graph'):
                graph_props = self.current_graph_dataset.schema.graph
                graph_type = []
                if graph_props.directed:
                    graph_type.append("Directed")
                else:
                    graph_type.append("Undirected")
                if graph_props.heterogeneous:
                    graph_type.append("Heterogeneous")
                if graph_props.weighted:
                    graph_type.append("Weighted")
                schema_info.append(f"Graph Type: {', '.join(graph_type)}")
            
            # Vertex types
            if hasattr(self.current_graph_dataset.schema, 'vertex'):
                vertex_types = [v.type for v in self.current_graph_dataset.schema.vertex]
                schema_info.append(f"Vertex Types: {', '.join(vertex_types)}")
            
            # Edge types
            if hasattr(self.current_graph_dataset.schema, 'edge'):
                edge_types = [e.type for e in self.current_graph_dataset.schema.edge]
                schema_info.append(f"Edge Types: {', '.join(edge_types)}")
            
            return "\n".join(schema_info)
        except Exception as e:
            logger.warning(f"Failed to get graph schema summary: {e}")
            return None

    def _build_dag_from_query(self, query: str, decompose: bool = True) -> GraphWorkflowDAG:
        """
        将用户查询转换为DAG，包含查询重写步骤
        
        Args:
            query: 用户原始查询
            decompose: 是否分解查询
            
        Returns:
            构建完成的DAG对象
        """
        # Step 1: Get algorithm library information
        algorithm_library_info = self._get_algorithm_library_info()
        logger.info("📚 Algorithm library information extracted")
        
        # Step 2: Get dataset schema information (optional)
        dataset_info = self._get_graph_schema_summary()
        if dataset_info:
            logger.info("📊 Dataset schema information extracted")
        
        # Step 3: Rewrite query with algorithm context
        try:
            rewrite_result = self.reasoner.rewrite_query(
                original_query=query,
                algorithm_library_info=algorithm_library_info,
                dataset_info=dataset_info
            )
            
            rewritten_query = rewrite_result.get("rewritten_query", query)
            reasoning = rewrite_result.get("reasoning", "")
            mapped_concepts = rewrite_result.get("mapped_concepts", [])
            
            logger.info(f"✍️ Query rewritten successfully")
            logger.info(f"Original query: {query}")
            logger.info(f"Rewritten query: {rewritten_query}")
            logger.info(f"Reasoning: {reasoning}")
            
            if mapped_concepts:
                logger.info("🔗 Concept mappings:")
                for mapping in mapped_concepts:
                    logger.info(f"  - {mapping.get('original_concept')} → {mapping.get('mapped_to')}")
            
            # Use rewritten query for planning
            query_to_use = rewritten_query
            
        except Exception as e:
            logger.warning(f"⚠️ Query rewriting failed: {e}, using original query")
            query_to_use = query
        
        # Step 4: Continue with existing flow (plan subqueries)
        subquery_plan = self.reasoner.plan_subqueries(decompose, query_to_use)
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
            logger.info(f"🔍 问题分类结果: {question_classification}")

            if question_classification is None:
                error_msg = f"问题分类失败：返回了 None"
                logger.error(f"❌ {error_msg} | 问题: {step.question}")
                raise RuntimeError(error_msg)
            
            if "error" in question_classification:
                error_msg = f"问题分类失败：{question_classification.get('error', 'Unknown error')}"
                logger.error(f"❌ {error_msg} | 问题: {step.question} | 详情: {question_classification}")
                raise RuntimeError(error_msg)
            
            question_type = question_classification.get("type", "graph_algorithm")
            
            if question_type == "graph_query":
                # 图查询类型：使用nl_query_engine
                step.task_type = GraphAnalysisType.GRAPH_QUERY
                step.graph_algorithm = None
                logger.info(f"✅ 问题分类为图查询 | ❓ 问题:{step.question}, 🧩 任务类型: {step.task_type}, 原因: {question_classification.get('reason', '')}")
            elif question_type == "numeric_analysis":
                # If numeric analysis, set task type and skip graph algorithm selection
                step.task_type = GraphAnalysisType.NUMERIC_ANALYSIS
                step.graph_algorithm = None
                logger.info(f"✅ 问题分类为数值分析 | ❓ 问题:{step.question}, 🧩 任务类型: {step.task_type}, 原因: {question_classification.get('reason', '')}")
            else:
                # If graph algorithm, proceed with normal algorithm selection flow
                logger.info(f"📋 开始任务类型选择 | 问题: {step.question}")
                task_type_list = self.expert_search_engine.retrieve_task_type(step.question)
                logger.info(f"📋 检索到的任务类型列表: {task_type_list}")
                
                task_type_result = self.reasoner.select_task_type(step.question, task_type_list)
                logger.info(f"📋 任务类型选择结果: {task_type_result}")
                
                # Check if task_type_result is None or contains error
                if task_type_result is None:
                    error_msg = f"任务类型选择失败：LLM 返回了 None"
                    logger.error(f"❌ {error_msg} | 问题: {step.question}")
                    raise RuntimeError(error_msg)
                
                if "error" in task_type_result:
                    error_msg = f"任务类型选择失败：{task_type_result.get('error', 'Unknown error')}"
                    logger.error(f"❌ {error_msg} | 问题: {step.question} | 详情: {task_type_result}")
                    raise RuntimeError(error_msg)
                
                selected_task_type_id = task_type_result.get("id")
                if not selected_task_type_id:
                    error_msg = f"任务类型选择失败：返回结果中没有 'id' 字段"
                    logger.error(f"❌ {error_msg} | 返回结果: {task_type_result}")
                    raise RuntimeError(error_msg)
                
                logger.info(f"🔍 开始算法选择 | 任务类型: {selected_task_type_id}")
                algorithm_list = self.expert_search_engine.retrieve_algorithm(step.question, selected_task_type_id)
                logger.info(f"🔍 检索到的算法列表: {algorithm_list}")
                
                # 获取当前数据集的完整schema信息
                graph_schema_info = None
                if self.current_graph_dataset:
                    graph_schema_info = {
                        "dataset_name": self.current_dataset_name,
                        "graph_properties": {
                            "directed": self.current_graph_dataset.schema.graph.directed if hasattr(self.current_graph_dataset.schema, 'graph') else True,
                            "heterogeneous": self.current_graph_dataset.schema.graph.heterogeneous if hasattr(self.current_graph_dataset.schema, 'graph') else False,
                            "multigraph": self.current_graph_dataset.schema.graph.multigraph if hasattr(self.current_graph_dataset.schema, 'graph') else False,
                            "weighted": self.current_graph_dataset.schema.graph.weighted if hasattr(self.current_graph_dataset.schema, 'graph') else False,
                        },
                        "vertex_types": [v.type for v in self.current_graph_dataset.schema.vertex] if hasattr(self.current_graph_dataset.schema, 'vertex') else [],
                        "edge_types": [e.type for e in self.current_graph_dataset.schema.edge] if hasattr(self.current_graph_dataset.schema, 'edge') else [],
                        "vertex_configs": [{"type": v.type, "query_field": v.query_field, "attribute_fields": v.attribute_fields}
                                          for v in self.current_graph_dataset.schema.vertex] if hasattr(self.current_graph_dataset.schema, 'vertex') else [],
                        "edge_configs": [{"type": e.type, "source_field": e.source_field, "target_field": e.target_field, "weight_field": e.weight_field}
                                        for e in self.current_graph_dataset.schema.edge] if hasattr(self.current_graph_dataset.schema, 'edge') else [],
                    }
                
                algorithm_result = self.reasoner.select_algorithm(step.question, algorithm_list, graph_schema=graph_schema_info)
                logger.info(f"🔍 算法选择结果: {algorithm_result}")
                
                # Check if algorithm_result is None or contains error
                if algorithm_result is None:
                    error_msg = f"算法选择失败：LLM 返回了 None"
                    logger.error(f"❌ {error_msg} | 问题: {step.question} | 任务类型: {selected_task_type_id}")
                    raise RuntimeError(error_msg)
                
                if "error" in algorithm_result:
                    error_msg = f"算法选择失败：{algorithm_result.get('error', 'Unknown error')}"
                    logger.error(f"❌ {error_msg} | 问题: {step.question} | 任务类型: {selected_task_type_id} | 详情: {algorithm_result}")
                    raise RuntimeError(error_msg)
                
                selected_algorithm_id = algorithm_result.get("id")
                if not selected_algorithm_id:
                    error_msg = f"算法选择失败：返回结果中没有 'id' 字段"
                    logger.error(f"❌ {error_msg} | 返回结果: {algorithm_result}")
                    raise RuntimeError(error_msg)
                
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
            tool_result = None 

            if step.task_type == GraphAnalysisType.GRAPH_ALGORITHM:
                if not step.graph_algorithm:
                    self.dag.set_failed(step_id, "未为该节点选择图算法")
                    raise RuntimeError(f"节点 {step_id} 缺少 graph_algorithm")

                 # step 1. 获取算法描述（doc， tool_metadata)， doc 是 tool_metadata 的字符串形式
                tool_description,  tool_metadata = await self.computing_engine.get_algorithm_description(
                    step.graph_algorithm
                )
                
                # 检查是否成功获取算法描述
                if tool_metadata is None:
                    error_msg = f"无法获取算法 '{step.graph_algorithm}' 的描述信息"
                    logger.error(f"❌ {error_msg}")
                    self.dag.set_failed(step_id, error_msg)
                    raise RuntimeError(f"节点 {step_id}: {error_msg}")
                
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

                # extraction_result = self._prepare_parameters_for_execution(
                #     step=step,
                #     tool_description=tool_description,
                #     vertex_schema=self.global_graph.get_vertex_properties_schema(),
                #     edge_schema=self.global_graph.get_edge_properties_schema(),
                #     dependency_parameters=data_dependency_context.get("parameter_input_adapter_result") or {},
                # )
                
                # if extraction_result is None:
                #     error_msg = "参数提取返回了 None"
                #     logger.error(f"❌ {error_msg}")
                #     self.dag.set_failed(step_id, error_msg)
                #     raise ValueError(error_msg)

                # logger.info(
                #     "✅ LLM/依赖提取的参数: %s",
                #     json.dumps(extraction_result.get("parameters", {}), ensure_ascii=False)
                # )

                # post_processing_info = extraction_result.get("post_processing_code") or {}  
                # is_has_extract_code = bool(post_processing_info.get("is_calculate"))
                # output_schema = post_processing_info.get("output_schema") or {}

                # if post_processing_info.get("code"):
                #     logger.info(
                #         f"✅ 后处理代码:\n{post_processing_info.get('code','')}...")
        
                # tool_result = await self.computing_engine.run_algorithm(
                #     step.graph_algorithm,
                #     extraction_result.get("parameters", {}),
                #     post_processing_info.get("code"),
                #     global_graph=self.global_graph
                # )
                
                # if tool_result is None:
                #     error_msg = "算法执行返回了 None"
                #     logger.error(f"❌ {error_msg}")
                #     self.dag.set_failed(step_id, error_msg)
                #     raise ValueError(error_msg)
                
                # # 检查嵌套的 result.error
                # result_data = tool_result.get("result")
                # if result_data is not None and isinstance(result_data, dict) and "error" in result_data:
                #     error_msg = result_data.get("error")
                #     logger.error(f"❌ {error_msg}")
                #     self.dag.set_failed(step_id, error_msg)
                #     raise RuntimeError(error_msg)
                
                # if not tool_result.get("success", False):
                #     error_msg = tool_result.get("error", "算法执行失败")
                #     logger.error(f"❌ {error_msg}")
                #     self.dag.set_failed(step_id, error_msg)
                #     raise RuntimeError(error_msg)
                async def op_prepare_params_and_postprocess_then_execute(error_history):
                    extraction_result = self._prepare_parameters_for_execution(
                        step=step,
                        tool_description=tool_description,
                        vertex_schema=self.global_graph.get_vertex_properties_schema(),
                        edge_schema=self.global_graph.get_edge_properties_schema(),
                        dependency_parameters=data_dependency_context.get("parameter_input_adapter_result") or {},
                        error_history=error_history
                    )

                    if extraction_result is None:
                        raise ValueError("Parameter adaptation and post-processing code generation failed, returning an empty result.")

                    post_processing_info = extraction_result.get("post_processing_code") or {} 
                    is_has_extract_code = bool(post_processing_info.get("is_calculate"))
                    output_schema = post_processing_info.get("output_schema") or {}


                    tool_result = await self.computing_engine.run_algorithm(
                        step.graph_algorithm,
                        extraction_result.get("parameters", {}),
                        post_processing_info.get("code"),
                        global_graph=self.global_graph
                    )

                    if tool_result is None:
                        raise ValueError("算法执行返回了 None")

                    result_data = tool_result.get("result")
                    if isinstance(result_data, dict) and "error" in result_data:
                        raise RuntimeError(result_data.get("error") or "tool_result.result.error")

                    # if not tool_result.get("success", False):
                    #     raise RuntimeError(tool_result.get("error", "算法执行失败"))

                    return is_has_extract_code, output_schema,tool_result


                is_has_extract_code, output_schema,tool_result = await self.error_recovery.run(
                    op_prepare_params_and_postprocess_then_execute,
                    name=f"prepare_params+postprocess+execute(step={step_id})"
                )
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
                
                # 生成数值分析代码
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
                
                # 执行代码
                code_result_value = self.computing_engine.execute_code(
                    code=generated_code,
                    data=execution_data,
                    global_graph=self.global_graph,
                    is_numeric_analysis=True
                )

                # 检查执行结果
                if isinstance(code_result_value, dict) and "error" in code_result_value:
                    error_msg = code_result_value.get("error")
                    logger.error(f"❌ {error_msg}")
                    self.dag.set_failed(step_id, error_msg)
                    raise RuntimeError(error_msg)

                # 成功执行
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

            elif step.task_type == GraphAnalysisType.GRAPH_QUERY:
                logger.info("🔍 当前任务类型：Graph Query")
                try:
                    # 通过 computing_engine 执行图查询
                    query_result = self.computing_engine.execute_graph_query(step.question)
                    
                    if query_result.get("success"):
                        # 将查询结果添加到step的输出
                        step.add_output(
                            task_type=GraphAnalysisSubType.SUBGRAPH_EXTRACTION,
                            source="graph_query",
                            output_schema=OutputSchema(
                                description="图查询结果",
                                type="list",
                                fields={
                                    "results": OutputField(
                                        type="list",
                                        field_description="查询返回的结果列表"
                                    ),
                                    "count": OutputField(
                                        type="int",
                                        field_description="结果数量"
                                    ),
                                    "query_type": OutputField(
                                        type="str",
                                        field_description="查询类型"
                                    )
                                }
                            ),
                            value={
                                "results": query_result.get("results", []),
                                "count": query_result.get("count", 0),
                                "query_type": query_result.get("query_type", "unknown")
                            },
                            path=None,
                            validate_schema=False
                        )
                        logger.info(f"✅ 图查询执行完成，返回 {query_result.get('count', 0)} 条结果")
                        self.dag.set_success(step_id)
                        
                        # 设置tool_result用于后续的LLM分析
                        tool_result = {
                            "success": True,
                            "result": query_result.get("results", []),
                            "summary": f"图查询成功，返回 {query_result.get('count', 0)} 条结果"
                        }
                    else:
                        error_msg = query_result.get("error", "图查询失败")
                        logger.error(f"❌ 图查询失败 | 错误: {error_msg}")
                        self.dag.set_failed(step_id, error_msg)
                        raise RuntimeError(f"节点 {step_id} 图查询失败：{error_msg}")
                        
                except Exception as query_err:
                    error_msg = f"图查询执行失败: {query_err}"
                    logger.error(error_msg, exc_info=True)
                    self.dag.set_failed(step_id, error_msg)
                    raise RuntimeError(f"节点 {step_id} 图查询失败：{query_err}")

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
        dependency_parameters: Dict[str, Any],
        error_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
            根据数据依赖或 LLM 结果准备图算法输入参数。
        """
        dependency_parameters = dependency_parameters or {}
        trace = {"step_id": getattr(step, "step_id", None), "op": "prepare_params"}

        if dependency_parameters:
            return self.reasoner.merge_parameters_from_dependencies(
                question=step.question,
                tool_description=tool_description or "",
                vertex_schema=vertex_schema,
                edge_schema=edge_schema,
                dependency_parameters=dependency_parameters,
                error_history=error_history,
                error_recovery=self.error_recovery,
                trace=trace,
            )

        return self.reasoner.extract_parameters_with_postprocess_new(
            question=step.question,
            tool_description=tool_description or "",
            vertex_schema=vertex_schema,
            edge_schema=edge_schema,
            error_history=error_history,
            error_recovery=self.error_recovery,
            trace=trace,
        )
        # result = None
        # try:
        #     if dependency_parameters:
        #         result = self.reasoner.merge_parameters_from_dependencies(
        #             question=step.question,
        #             tool_description=tool_description,
        #             vertex_schema=vertex_schema,
        #             edge_schema=edge_schema,
        #             dependency_parameters=dependency_parameters,
        #             error_history=error_history, 
        #             error_recovery=self.error_recovery
        #         )
        #     else:
        #         result = self.reasoner.extract_parameters_with_postprocess_new(
        #             question=step.question,
        #             tool_description=tool_description,
        #             vertex_schema=vertex_schema,
        #             edge_schema=edge_schema,
        #             error_history=error_history, 
        #             error_recovery=self.error_recovery
        #         )
        # except Exception as e:
        #     logger.info(f"Parameter processing failed: {e}")
        #     result = None

        # return result
    
    def _extract_dag_structure(self) -> Dict[str, Any]:
        """
        从当前DAG中提取subquery结构
        
        Returns:
            包含subqueries列表的字典
        """
        subqueries = []
        for step_id in self.dag.topological_order():
            step = self.dag.steps[step_id]
            
            # 将内部step_id映射回query_id (如q1, q2等)
            query_id = None
            for qid, sid in self.query_id_mapping.items():
                if sid == step_id:
                    query_id = qid
                    break
            
            if query_id is None:
                query_id = f"q{step_id}"
            
            # 获取依赖的query_ids - 使用DAG的in_edges而不是step.depends_on
            depends_on = []
            parent_step_ids = self.dag.parents_of(step_id)
            for dep_step_id in parent_step_ids:
                for qid, sid in self.query_id_mapping.items():
                    if sid == dep_step_id:
                        depends_on.append(qid)
                        break
            
            subqueries.append({
                "id": query_id,
                "query": step.question,
                "depends_on": depends_on,
                "task_type": str(step.task_type) if step.task_type else None,
                "algorithm": step.graph_algorithm
            })
        
        return {"subqueries": subqueries}
    
    def _is_dag_changed(self, old_structure: Dict[str, Any], new_structure: Dict[str, Any]) -> bool:
        """
        检查DAG结构是否发生变化
        
        Args:
            old_structure: 旧的DAG结构
            new_structure: 新的DAG结构
            
        Returns:
            True if changed, False otherwise
        """
        old_queries = old_structure.get("subqueries", [])
        new_queries = new_structure.get("subqueries", [])
        
        # 检查节点数量是否变化
        if len(old_queries) != len(new_queries):
            return True
        
        # 检查每个节点的query和依赖关系是否变化
        for old_q, new_q in zip(old_queries, new_queries):
            if old_q.get("query") != new_q.get("query"):
                return True
            if set(old_q.get("depends_on", [])) != set(new_q.get("depends_on", [])):
                return True
        
        return False
    
    def _refine_dag_with_retry(self, max_retries: int = 1) -> Dict[str, Any]:
        """
        使用LLM优化DAG,确保任务类型边界清晰
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            优化后的DAG信息
        """
        logger.info("🔄 开始DAG优化流程...")
        
        for retry_count in range(max_retries + 1):
            try:
                # 获取当前DAG的subquery结构
                current_dag_structure = self._extract_dag_structure()
                
                logger.info(f"📋 当前DAG结构 (尝试 {retry_count + 1}/{max_retries + 1}):")
                logger.info(json.dumps(current_dag_structure, ensure_ascii=False, indent=2))
                
                # 调用LLM进行DAG优化
                refined_structure = self.reasoner.refine_subqueries(current_dag_structure)
                
                logger.info("✅ LLM返回的优化DAG:")
                logger.info(json.dumps(refined_structure, ensure_ascii=False, indent=2))
                
                # 检查是否有实质性变化
                if self._is_dag_changed(current_dag_structure, refined_structure):
                    logger.info("🔄 检测到DAG结构变化,重新构建DAG...")
                    
                    # 重新构建DAG
                    self.build_dag_from_subquery_plan(refined_structure)
                    
                    # 重新为新的节点找算法
                    self._find_algorithm()
                    
                    logger.info(f"✅ DAG优化完成 (第 {retry_count + 1} 次迭代)")
                else:
                    logger.info("✅ DAG结构已优化,无需进一步调整")
                    break
                    
            except Exception as e:
                logger.error(f"❌ DAG优化失败 (尝试 {retry_count + 1}/{max_retries + 1}): {e}")
                if retry_count == max_retries:
                    logger.warning("⚠️ 达到最大重试次数,使用当前DAG继续执行")
                continue
        
        return self.dag.get_dag_info()

    async def shutdown(self):
        if self.computing_engine:
            await self.computing_engine.shutdown()
        
