# scheduler_with_dag.py
import time
import traceback
import logging
import json
import difflib
import re
from argparse import OPTIONAL
from dataclasses import asdict
from typing import Any, Dict, Optional, Callable, List, Tuple, Union, Set

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
    Unified scheduler: retrieval / graph computation / graph learning / LLM.
    Holds the DAG and runs steps according to dependencies.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.reasoner: Optional[Reasoner] = None
        self.computing_engine: Optional[ComputingEngine] = None
        self.expert_search_engine: Optional[ExpertSearchEngine] = None
        self.dag: Optional[GraphWorkflowDAG] = None
        self.dataset_manager: Optional[DatasetManager] = None
        self.data_dependency_resolver: Optional[DataDependencyResolver] = None
        self.error_recovery: Optional[ErrorRecovery] = None

        self.current_dataset_name: Optional[str] = None  # Dataset-level name (for text datasets, this is the dataset name)
        self.current_dataset: Optional[List[DatasetConfig]] = None  # Original dataset config (text/graph, never changes)
        self.current_graph_dataset: Optional[DatasetConfig] = None  # Current graph dataset config (may be converted graph)
        self.global_graph: Optional[GraphData] = None
        self.query_id_mapping: Dict[str, int] = {}
        self._forced_step_configs: Dict[str, Dict[str, Any]] = {}

        self._initialize_components()

        self.on_step_start: Optional[Callable[["WorkflowStep"], None]] = None
        self.on_step_end: Optional[Callable[["WorkflowStep"], None]] = None



    def _initialize_components(self):
        """Initialize scheduler components."""
        print("Initializing Scheduler components...")

        self._init_reasoner()

        self._init_computing_engine()

        self._init_expert_search_engine()

        self._init_dataset_manager()
        
        self._init_router()

        self._init_rag_engine()
        
        self._init_error_recovery()

        self._init_data_dependency_resolver()
    
    def _init_reasoner(self):
        """Initialize reasoner."""
        try:
            self.reasoner = Reasoner(self.config.reasoner)
            print("✓ Reasoner initialized")
        except Exception as e:
            print(f"✗ Reasoner initialization failed: {e}")
            raise
    
    def _init_computing_engine(self):
        """Initialize computing engine."""
        try:
            self.computing_engine = ComputingEngine()
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
        """Initialize expert search engine."""
        try:
            self.expert_search_engine = ExpertSearchEngine(self.config.retrieval)
            print("✓ ExpertSearchEngine initialized")
        except Exception as e:
            print(f"✗ ExpertSearchEngine initialization failed: {e}")
            raise
    
    def _init_dataset_manager(self):
        """Initialize dataset manager."""
        try:
            self.dataset_manager = DatasetManager()
            print("✓ DatasetManager initialized")
        except Exception as e:
            print(f"✗ DatasetManager initialization failed: {e}")
            raise
    
    def _init_data_dependency_resolver(self):
        """Initialize data dependency resolver."""
        try:
            # Note: error_recovery will be set after initialization
            self.data_dependency_resolver = DataDependencyResolver(self.reasoner, self.error_recovery)
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
        """Initialize error recovery module."""
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

    def _normalize_graph_mode(self, mode: str) -> str:
        """
        Normalize graph execution mode.
        Supported modes: normal | interact | expert
        """
        normalized = (mode or "normal").strip().lower()
        allowed_modes = {"normal", "interact", "expert"}
        if normalized not in allowed_modes:
            raise ValueError(
                f"Unsupported mode '{mode}'; use normal / interact / expert"
            )
        return normalized

    async def execute(
        self, 
        query: str, 
        decompose: bool = True, 
        mode: str = "normal",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute query with dataset type validation.

        Args:
            query: User query.
            decompose: Whether to decompose the query.
            mode: "normal" | "interact" | "expert".
            callback: Optional callback for streaming (e.g. DAG); signature callback(data: Dict).

        Returns:
            Normal: analysis result string.
            Interact/expert: dict with dag_info, message, etc.

        Validates:
            1. RAG query + graph dataset → Error (graph data cannot do retrieval)
            2. GRAPH query + text dataset → Check if converted graph exists
        """
        try:
            mode = self._normalize_graph_mode(mode)
        except ValueError as mode_err:
            return f"❌ Error: {mode_err}"

        decision = self.router.route(query=query)
        logger.info(f"🚦[Router] query_type={decision.query_type}, reason={decision.reason}")

        if not self.current_dataset or not self.current_dataset_name:
            return "⚠️ No dataset selected; please set the analysis target first."
        
        original_type = self.dataset_manager.get_dataset_original_type(self.current_dataset_name)
        
        if decision.query_type == QueryType.RAG:
            if original_type == "graph":
                return "❌ Error: Current dataset is graph data; retrieval is not supported. Use graph analysis only."
            return await self._execute_rag(query)

        elif decision.query_type == QueryType.GRAPH_QUERY:
            if original_type != "graph":
                converted_graph_config = self.dataset_manager.get_converted_graph_dataset(self.current_dataset_name)
                if not converted_graph_config:
                    return "❌ Error: Graph query requires graph data; current dataset is not graph and has no converted graph."
                self.current_graph_dataset = converted_graph_config
            
            return await self._execute_graph_query(query)
        
        elif decision.query_type == QueryType.GRAPH:
            if original_type == "text":
                converted_graph_config = self.dataset_manager.get_converted_graph_dataset(self.current_dataset_name)
                if not converted_graph_config:
                    return "❌ Error: Current dataset is text and has not been converted to graph. Please run text-to-graph conversion first."
                
                self.current_graph_dataset = converted_graph_config
                global_vertices, global_edges = self.dataset_manager.get_dataset_content(self.current_graph_dataset)
                self.global_graph = GraphData(vertices=global_vertices, edges=global_edges)
                graph_nodes, graph_edges = flatten_graph(global_vertices, global_edges)
                self.data_dependency_resolver.set_global_graph(graph_nodes, graph_edges)         
            
            return await self._execute_graph(query, decompose=decompose, mode=mode, callback=callback)
        
        return self.reasoner.general_query_response(query)
    

    async def _execute_graph(
        self, 
        query: str, 
        decompose: bool = True, 
        mode: str = "normal",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute graph analysis query.

        Args:
            query: User query.
            decompose: Whether to decompose the query.
            mode: "normal" | "interact" | "expert".
            callback: Optional callback for streaming (e.g. DAG); signature callback(data: Dict).

        Returns:
            Normal: analysis result string.
            Interact/expert: DAG info dict.

        Uses self.current_graph_dataset (which may be converted graph for text datasets).
        """
        if not self.computing_engine._initialized:
            await self.computing_engine.initialize()

        if not self.current_graph_dataset:
            return "⚠️ No graph dataset selected; please set the analysis target first."
        
        if not self.global_graph:
            return "⚠️ Graph data not loaded; please load graph data first."

        mode = self._normalize_graph_mode(mode)

        if mode == "expert":
            return self._execute_graph_expert_mode(query)

        self._build_dag_from_query(query, decompose)
        self._find_algorithm()

        print("✅ Initial DAG build and algorithm selection done")
        self.dag.print_dag_info()
        
        if mode == "interact":
            dag_info = self.dag.get_dag_info()
            return {
                "message": "DAG generated; please choose next action.",
                "dag_info": dag_info
            }
        
        self.dag.refresh_data_dependency(self.reasoner)
        self.dag.print_data_dependency()
        print("✅ DAG build and algorithm selection done; starting computation")
        
        analysis_result = await self._run_algorithm_pipeline2()
        
        if callback and mode == "normal":
            dag_info = self.dag.get_dag_info()
            return {
                "analysis_result": analysis_result,
                "dag_info": dag_info
            }
        print(f"analysis_result: {analysis_result}")
        return analysis_result

    @staticmethod
    def _normalize_algorithm_token(raw_name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (raw_name or "").strip().lower())

    def _resolve_expert_algorithm(self, requested_algorithm: str) -> Optional[str]:
        if not self.expert_search_engine or not self.expert_search_engine.algo_index:
            return None

        algo_index = self.expert_search_engine.algo_index
        direct_name = (requested_algorithm or "").strip()
        if direct_name in algo_index:
            return direct_name

        lowered_map = {
            aid.lower(): aid
            for aid in algo_index.keys()
            if isinstance(aid, str)
        }
        if direct_name.lower() in lowered_map:
            return lowered_map[direct_name.lower()]

        normalized_target = self._normalize_algorithm_token(direct_name)
        for algo_id in algo_index.keys():
            if not isinstance(algo_id, str):
                continue
            if self._normalize_algorithm_token(algo_id) == normalized_target:
                return algo_id
        return None

    def _suggest_algorithms(self, requested_algorithm: str, limit: int = 5) -> List[str]:
        if not self.expert_search_engine or not self.expert_search_engine.algo_index:
            return []

        candidate_algorithms = [
            aid for aid in self.expert_search_engine.algo_index.keys()
            if isinstance(aid, str)
        ]

        direct_matches = difflib.get_close_matches(
            requested_algorithm, candidate_algorithms, n=limit, cutoff=0.45
        )
        if direct_matches:
            return direct_matches

        normalized_candidates = {
            self._normalize_algorithm_token(aid): aid for aid in candidate_algorithms
        }
        normalized_matches = difflib.get_close_matches(
            self._normalize_algorithm_token(requested_algorithm),
            list(normalized_candidates.keys()),
            n=limit,
            cutoff=0.45
        )
        return [normalized_candidates[key] for key in normalized_matches]

    def _parse_expert_dag_instruction(self, expert_instruction: str) -> Tuple[Dict[str, Any], Dict[str, str], List[Dict[str, Any]], str]:
        normalized_instruction = (expert_instruction or "").strip()
        if not normalized_instruction:
            raise ValueError("Expert mode input cannot be empty")

        algorithm_library_info = self._get_algorithm_library_info()
        dataset_info = self._get_graph_schema_summary()
        payload = self.reasoner.plan_expert_subqueries_with_algorithms(
            expert_instruction=normalized_instruction,
            algorithm_library_info=algorithm_library_info,
            dataset_info=dataset_info,
        )

        if not isinstance(payload, dict):
            raise ValueError("Expert mode parse failed: LLM did not return a valid subqueries structure")

        subqueries = payload.get("subqueries")
        if not isinstance(subqueries, list) or not subqueries:
            raise ValueError("Expert mode input missing subqueries list")

        normalized_subqueries = []
        algorithm_hints: Dict[str, str] = {}
        missing_algorithm_subqueries: List[str] = []

        for idx, item in enumerate(subqueries, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"subqueries[{idx}] must be an object")

            query_id = str(item.get("id", f"q{idx}")).strip() or f"q{idx}"
            question = str(item.get("query") or item.get("question") or "").strip()
            if not question:
                raise ValueError(f"subqueries[{idx}] missing query/question")

            depends_on = item.get("depends_on", [])
            if depends_on is None:
                depends_on = []
            if not isinstance(depends_on, list):
                raise ValueError(f"subqueries[{idx}].depends_on must be a list")

            normalized_subqueries.append({
                "id": query_id,
                "query": question,
                "depends_on": [str(dep).strip() for dep in depends_on if str(dep).strip()]
            })

            requested_algorithm = (
                item.get("algorithm")
                or item.get("graph_algorithm")
                or item.get("algo")
            )
            if requested_algorithm is not None and str(requested_algorithm).strip():
                algorithm_hints[query_id] = str(requested_algorithm).strip()
            else:
                missing_algorithm_subqueries.append(query_id)

        if missing_algorithm_subqueries:
            raise ValueError(
                "Expert mode requires each subquery to have an algorithm field; missing: "
                + ", ".join(missing_algorithm_subqueries)
            )

        raw_adjustments = (
            payload.get("instruction_algorithm_adjustments")
            or payload.get("algorithm_adjustments")
            or []
        )
        normalized_adjustments: List[Dict[str, Any]] = []
        if isinstance(raw_adjustments, list):
            for item in raw_adjustments:
                if not isinstance(item, dict):
                    continue
                mentioned_algorithm = str(item.get("mentioned_algorithm") or "").strip()
                if not mentioned_algorithm:
                    continue
                replacement_algorithm_raw = str(item.get("replacement_algorithm") or "").strip()
                recommendations_raw = item.get("recommendations") or []
                recommendations = []
                if isinstance(recommendations_raw, list):
                    recommendations = [
                        str(rec).strip()
                        for rec in recommendations_raw
                        if str(rec).strip()
                    ][:5]
                reason = str(item.get("reason") or "").strip()

                normalized_adjustments.append({
                    "mentioned_algorithm": mentioned_algorithm,
                    "replacement_algorithm": replacement_algorithm_raw or None,
                    "recommendations": recommendations,
                    "reason": reason,
                })

        adjustment_summary = str(
            payload.get("instruction_algorithm_adjustment_summary")
            or payload.get("algorithm_adjustment_summary")
            or ""
        ).strip()

        return {"subqueries": normalized_subqueries}, algorithm_hints, normalized_adjustments, adjustment_summary

    def _build_dag_from_expert_instruction(self, expert_instruction: str) -> Dict[str, Any]:
        subquery_plan, algorithm_hints, instruction_adjustments, adjustment_summary = self._parse_expert_dag_instruction(expert_instruction)
        self.build_dag_from_subquery_plan(subquery_plan)

        query_text_map = {
            subquery["id"]: subquery["query"]
            for subquery in subquery_plan.get("subqueries", [])
        }

        validation: Dict[str, Any] = {
            "provided_algorithms": algorithm_hints,
            "applied_algorithms": {},
            "unsupported_algorithms": [],
            "missing_step_ids": [],
            "unsupported_step_ids": [],
        }

        for query_id, requested_algorithm in algorithm_hints.items():
            step_id = self.query_id_mapping.get(query_id)
            if step_id is None:
                validation["missing_step_ids"].append(query_id)
                continue

            resolved_algorithm = self._resolve_expert_algorithm(requested_algorithm)
            if resolved_algorithm is None:
                validation["unsupported_algorithms"].append({
                    "query_id": query_id,
                    "query": query_text_map.get(query_id, ""),
                    "requested_algorithm": requested_algorithm,
                    "suggestions": self._suggest_algorithms(requested_algorithm),
                })
                validation["unsupported_step_ids"].append(step_id)
                continue

            step = self.dag.steps[step_id]
            step.task_type = GraphAnalysisType.GRAPH_ALGORITHM
            step.graph_algorithm = resolved_algorithm
            validation["applied_algorithms"][query_id] = resolved_algorithm

        validation["instruction_algorithm_adjustments"] = instruction_adjustments
        validation["instruction_algorithm_adjustment_summary"] = adjustment_summary

        return validation

    def _execute_graph_expert_mode(self, expert_instruction: str) -> Dict[str, Any]:
        """
        Expert mode:
        - Expert provides high-quality instruction in natural language
        - Scheduler builds DAG as instructed
        - Validate requested algorithms against algorithm library boundary
        """
        try:
            validation = self._build_dag_from_expert_instruction(expert_instruction)
            unsupported_step_ids = set(validation.get("unsupported_step_ids", []))

            self._find_algorithm(
                respect_preassigned=True,
                skip_step_ids=unsupported_step_ids
            )
            self.dag.print_dag_info()

            dag_info = self.dag.get_dag_info()
            has_out_of_boundary_algorithms = bool(validation.get("unsupported_algorithms")) or bool(validation.get("missing_step_ids"))

            if has_out_of_boundary_algorithms:
                return {
                    "message": "Expert DAG built, but algorithms outside the library boundary were detected; please fix before starting analysis.",
                    "dag_info": dag_info,
                    "algorithm_validation": {
                        "provided_algorithms": validation.get("provided_algorithms", {}),
                        "applied_algorithms": validation.get("applied_algorithms", {}),
                        "unsupported_algorithms": validation.get("unsupported_algorithms", []),
                        "missing_step_ids": validation.get("missing_step_ids", []),
                        "instruction_algorithm_adjustments": validation.get("instruction_algorithm_adjustments", []),
                    },
                    "can_start_analysis": False
                }

            instruction_adjustments = validation.get("instruction_algorithm_adjustments", [])
            adjustment_message = str(validation.get("instruction_algorithm_adjustment_summary") or "").strip()
            base_message = "Expert DAG built as instructed"
            if adjustment_message:
                base_message = f"{base_message}. {adjustment_message}"

            return {
                "message": base_message,
                "dag_info": dag_info,
                "algorithm_validation": {
                    "provided_algorithms": validation.get("provided_algorithms", {}),
                    "applied_algorithms": validation.get("applied_algorithms", {}),
                    "unsupported_algorithms": [],
                    "missing_step_ids": [],
                    "instruction_algorithm_adjustments": instruction_adjustments,
                },
                "can_start_analysis": True
            }
        except Exception as e:
            logger.error(f"Expert mode build failed: {e}", exc_info=True)
            return {
                "error": f"Expert mode build failed: {str(e)}",
                "input_hint": "Enter expert instructions in natural language; the system will generate subqueries."
            }

    async def expert_modify_dag(self, modification_request: str) -> Dict[str, Any]:
        """
        Interact mode: modify DAG per user request.

        Args:
            modification_request: User modification request (natural language).

        Returns:
            Dict with updated DAG info.
        """
        if not self.dag:
            return {
                "error": "DAG not built yet; please enter a question to generate the DAG first."
            }
        
        try:
            self.dag.modify_dag(self.reasoner, modification_request)
            self._find_algorithm()
            dag_info = self.dag.get_dag_info()
            return {
                "message": "DAG updated.",
                "dag_info": dag_info
            }
        except Exception as e:
            logger.error(f"DAG modification failed: {e}", exc_info=True)
            return {
                "error": f"DAG modification failed: {str(e)}"
            }

    async def expert_start_analysis(self) -> str:
        """
        Interact/expert mode: start analysis.

        Returns:
            Analysis result string.
        """
        if not self.dag:
            return "❌ Error: DAG not built yet; please enter a question to generate the DAG first."
                
        self.dag.refresh_data_dependency(self.reasoner)
        self.dag.print_data_dependency()
        print("✅ DAG build and algorithm selection done; starting computation")
        return await self._run_algorithm_pipeline2()

    async def _execute_rag(self, query: str) -> str:
        """
        Execute RAG query
        
        Uses self.current_dataset (original text dataset config, never changes)
        For text datasets, current_dataset is the first file config from the list
        """
        if not self.current_dataset:
            return "⚠️ No analysis data selected; please set the analysis target first."
        
        if self.dataset_manager.get_dataset_original_type(self.current_dataset_name) != "text":
            return f"❌ Error: RAG requires text data; current dataset type is {self.current_dataset.type}"
        
        file_paths = [config.schema.path for config in self.current_dataset]

        if not self.rag_engine._initialized:
            self.rag_engine.initialize(db_name=self.current_dataset_name, file_paths=file_paths)

        retrieved_context, _ = self.rag_engine.retrieve(query)

        prompt = rag_prompt.format(context=retrieved_context, query=query)

        return self.reasoner.generate_response(prompt)

    async def _execute_graph_query(self, query: str) -> str:
        """
        Execute graph query via computing_engine.

        Args:
            query: User query.

        Returns:
            Query result string.
        """
        try:
            result = self.computing_engine.execute_graph_query(query)
            
            if result.get("success"):
                results = result.get("results", [])
                count = result.get("count", 0)
                query_type = result.get("query_type", "unknown")
                
                response = f"✅ Graph query succeeded!\n"
                response += f"Query type: {query_type}\n"
                response += f"Result count: {count}\n\n"
                
                if count > 0:
                    response += "Results:\n"
                    for i, item in enumerate(results[:10], 1):
                        serializable_item = self._convert_neo4j_to_dict(item)
                        response += f"{i}. {json.dumps(serializable_item, ensure_ascii=False, indent=2)}\n"
                    
                    if count > 10:
                        response += f"\n... {count - 10} more results not shown"
                else:
                    response += "No matching results"
                
                return response
            else:
                error_msg = result.get("error", "Unknown error")
                return f"❌ Graph query failed: {error_msg}"
                
        except Exception as e:
            logger.error(f"Graph query execution failed: {e}", exc_info=True)
            return f"❌ Graph query execution failed: {str(e)}"
    
    def _convert_neo4j_to_dict(self, obj):
        """
        Convert Neo4j objects (Node, Relationship, Path) to JSON-serializable dicts.

        Args:
            obj: Neo4j object or plain object.

        Returns:
            Serializable dict or original object.
        """
        if hasattr(obj, 'labels') and hasattr(obj, 'id') and callable(getattr(obj, 'items', None)):
            return {
                'id': obj.id,
                'labels': list(obj.labels),
                'properties': dict(obj)
            }
        elif hasattr(obj, 'type') and hasattr(obj, 'id') and callable(getattr(obj, 'items', None)):
            return {
                'id': obj.id,
                'type': obj.type,
                'start_node': obj.start_node.id if hasattr(obj, 'start_node') else None,
                'end_node': obj.end_node.id if hasattr(obj, 'end_node') else None,
                'properties': dict(obj)
            }
        elif hasattr(obj, 'nodes') and hasattr(obj, 'relationships'):
            return {
                'nodes': [self._convert_neo4j_to_dict(node) for node in obj.nodes],
                'relationships': [self._convert_neo4j_to_dict(rel) for rel in obj.relationships]
            }
        elif isinstance(obj, dict):
            return {k: self._convert_neo4j_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_neo4j_to_dict(item) for item in obj]
        else:
            return obj

    def build_dag_from_subquery_plan(
        self,
        subquery_plan: Dict[str, Any],
        forced_step_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> GraphWorkflowDAG:
        """
        Build DAG from JSON subquery plan.

        Args:
            subquery_plan: Subquery plan, format:
                {"subqueries": [{"id": "q1", "query": "...", "depends_on": ["q0"]}, ...]}

        Returns:
            GraphWorkflowDAG instance.

        Raises:
            ValueError: If input format is invalid or has circular dependencies.
        """
        self.dag = GraphWorkflowDAG()
        query_id_to_step_id = self.dag.build_from_subquery_plan(subquery_plan)
        logger.info(f"✅ DAG built; topological order: {self.dag.topological_order()}")
        self.query_id_mapping = query_id_to_step_id
        self._forced_step_configs = forced_step_configs or {}

        return self.dag

    @staticmethod
    def _should_use_anna_forced_dag(query: str) -> bool:
        return "anna" in (query or "").lower()

    def _get_anna_forced_subquery_plan(self) -> Dict[str, Any]:
        return {
            "subqueries": [
                {
                    "id": "q1",
                    "query": "检测 Anna Lee 是否是高风险用户，统计她的影响力排名。",
                    "depends_on": []
                },
                {
                    "id": "q2",
                    "query": "围绕 Anna Lee 列出所有潜在的洗钱路径。",
                    "depends_on": ["q1"]
                },
                {
                    "id": "q3",
                    "query": "估算与 Anna Lee 有关的现金可能已经被非法转出的金额。",
                    "depends_on": ["q2"]
                },
                {
                    "id": "q4",
                    "query": "这些可疑路径涉及的账户中，找出交易金额最大的账户。",
                    "depends_on": ["q2"]
                }
            ]
        }

    def _get_anna_forced_step_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "q1": {
                "question": "检测 Anna Lee 是否是高风险用户，统计她的影响力排名。",
                "task_type": GraphAnalysisType.GRAPH_ALGORITHM,
                "graph_algorithm": "pagerank",
            },
            "q2": {
                "question": "围绕 Anna Lee 列出所有潜在的洗钱路径。",
                "task_type": GraphAnalysisType.GRAPH_ALGORITHM,
                "graph_algorithm": "find_cycle",
            },
            "q3": {
                "question": "估算与 Anna Lee 有关的现金可能已经被非法转出的金额。",
                "task_type": GraphAnalysisType.NUMERIC_ANALYSIS,
                "graph_algorithm": None,
            },
            "q4": {
                "question": "这些可疑路径涉及的账户中，找出交易金额最大的账户。",
                "task_type": GraphAnalysisType.NUMERIC_ANALYSIS,
                "graph_algorithm": None,
            },
        }


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
        Convert user query to DAG, including query rewrite.

        Args:
            query: User query.
            decompose: Whether to decompose the query.

        Returns:
            Built DAG.
        """
        if self._should_use_anna_forced_dag(query):
            logger.info("🎯 Anna keyword detected; using forced DAG and task configuration")
            return self.build_dag_from_subquery_plan(
                self._get_anna_forced_subquery_plan(),
                forced_step_configs=self._get_anna_forced_step_configs()
            )

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
            print(f"rewritten_query: {rewritten_query}")
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


    def _find_algorithm(
        self,
        respect_preassigned: bool = False,
        skip_step_ids: Optional[Set[int]] = None
    ):
        """
        Walk the DAG and assign a graph algorithm to each step from its question.
        """
        skip_step_ids = skip_step_ids or set()
        for step in self.dag.steps.values():
            if step.step_id in skip_step_ids:
                logger.warning(
                    "⚠️ Skipping algorithm auto-match (expert mode out-of-bound) | step=%s | question=%s",
                    step.step_id,
                    step.question,
                )
                continue

            query_id = next(
                (qid for qid, sid in self.query_id_mapping.items() if sid == step.step_id),
                None
            )
            forced_config = self._forced_step_configs.get(query_id) if query_id else None
            if forced_config and step.question == forced_config.get("question"):
                step.task_type = forced_config.get("task_type")
                step.graph_algorithm = forced_config.get("graph_algorithm")
                logger.info(
                    "✅ Applied forced step config | step=%s | query_id=%s | task_type=%s | algorithm=%s",
                    step.step_id,
                    query_id,
                    step.task_type,
                    step.graph_algorithm,
                )
                continue

            if respect_preassigned and step.graph_algorithm:
                if not step.task_type:
                    step.task_type = GraphAnalysisType.GRAPH_ALGORITHM
                logger.info(
                    "✅ Keeping expert-assigned algorithm | step=%s | algorithm=%s",
                    step.step_id,
                    step.graph_algorithm,
                )
                continue

            question_classification = self.reasoner.classify_question_type(step.question)
            logger.info(f"🔍 Question classification: {question_classification}")

            if question_classification is None:
                error_msg = "Question classification failed: returned None"
                logger.error(f"❌ {error_msg} | question: {step.question}")
                raise RuntimeError(error_msg)
            
            if "error" in question_classification:
                error_msg = f"Question classification failed: {question_classification.get('error', 'Unknown error')}"
                logger.error(f"❌ {error_msg} | question: {step.question} | details: {question_classification}")
                raise RuntimeError(error_msg)
            
            question_type = question_classification.get("type", "graph_algorithm")
            
            if question_type == "graph_query":
                step.task_type = GraphAnalysisType.GRAPH_QUERY
                step.graph_algorithm = None
                logger.info(f"✅ Classified as graph query | question: {step.question}, task_type: {step.task_type}, reason: {question_classification.get('reason', '')}")
            elif question_type == "numeric_analysis":
                step.task_type = GraphAnalysisType.NUMERIC_ANALYSIS
                step.graph_algorithm = None
                logger.info(f"✅ Classified as numeric analysis | question: {step.question}, task_type: {step.task_type}, reason: {question_classification.get('reason', '')}")
            else:
                logger.info(f"📋 Task type selection | question: {step.question}")
                task_type_list = self.expert_search_engine.retrieve_task_type(step.question)
                logger.info(f"📋 Retrieved task types: {task_type_list}")
                
                task_type_result = self.reasoner.select_task_type(step.question, task_type_list)
                logger.info(f"📋 Task type selection result: {task_type_result}")
                
                if task_type_result is None:
                    error_msg = "Task type selection failed: LLM returned None"
                    logger.error(f"❌ {error_msg} | question: {step.question}")
                    raise RuntimeError(error_msg)
                
                if "error" in task_type_result:
                    error_msg = f"Task type selection failed: {task_type_result.get('error', 'Unknown error')}"
                    logger.error(f"❌ {error_msg} | question: {step.question} | details: {task_type_result}")
                    raise RuntimeError(error_msg)
                
                selected_task_type_id = task_type_result.get("id")
                if not selected_task_type_id:
                    error_msg = "Task type selection failed: result missing 'id' field"
                    logger.error(f"❌ {error_msg} | result: {task_type_result}")
                    raise RuntimeError(error_msg)
                
                logger.info(f"🔍 Algorithm selection | task_type: {selected_task_type_id}")
                algorithm_list = self.expert_search_engine.retrieve_algorithm(step.question, selected_task_type_id)
                logger.info(f"🔍 Retrieved algorithms: {algorithm_list}")
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
                logger.info(f"🔍 Algorithm selection result: {algorithm_result}")
                
                if algorithm_result is None:
                    error_msg = "Algorithm selection failed: LLM returned None"
                    logger.error(f"❌ {error_msg} | question: {step.question} | task_type: {selected_task_type_id}")
                    raise RuntimeError(error_msg)
                
                if "error" in algorithm_result:
                    error_msg = f"Algorithm selection failed: {algorithm_result.get('error', 'Unknown error')}"
                    logger.error(f"❌ {error_msg} | question: {step.question} | task_type: {selected_task_type_id} | details: {algorithm_result}")
                    raise RuntimeError(error_msg)
                
                selected_algorithm_id = algorithm_result.get("id")
                if not selected_algorithm_id:
                    error_msg = "Algorithm selection failed: result missing 'id' field"
                    logger.error(f"❌ {error_msg} | result: {algorithm_result}")
                    raise RuntimeError(error_msg)
                
                step.task_type = GraphAnalysisType.GRAPH_ALGORITHM
                step.graph_algorithm = selected_algorithm_id
                logger.info(f"✅ Algorithm selected | question: {step.question}, task_type: {selected_task_type_id}, algorithm: {step.graph_algorithm}")

    async def _run_algorithm_pipeline(self):
        analysis_result = ""

        for step_id in self.dag.topological_order(): 
            step = self.dag.steps[step_id]

            if not step.graph_algorithm:
                self.dag.set_failed(step_id, "No graph algorithm selected for this step")
                raise RuntimeError(f"Step {step_id} missing graph_algorithm")

            tool_description,  tool_metadata = await self.computing_engine.get_algorithm_description(
                step.graph_algorithm
            )

            logger.info(f"tool_description:{tool_description}")

            extraction_result = self.reasoner.extract_parameters_with_postprocess(
                question=step.question,
                tool_description=tool_description
            )
            
            logger.info(
                f"✅ LLM extracted parameters: {json.dumps(extraction_result.get('parameters'), ensure_ascii=False)}")

            if extraction_result.get("post_processing_code"):
                logger.info(
                    f"✅ Post-processing code:\n{extraction_result.get('post_processing_code','')}...")

            tool_result = await self.computing_engine.run_algorithm(
                step.graph_algorithm,
                extraction_result.get("parameters", {}),
                extraction_result.get("post_processing_code", "")
            )

            logger.info(f"✅ Tool execution: {tool_result.get('summary','done')}")

            if not tool_result.get("success", False):
                error_msg = tool_result.get("error", "Algorithm execution failed")
                self.dag.set_failed(step_id, error_msg)
                raise RuntimeError(f"Step {step_id} execution failed: {error_msg}")

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
        analysis_blocks: List[str] = []

        for step_id in self.dag.topological_order():
            step = self.dag.steps[step_id]
            tool_description = None
            tool_metadata = None
            tool_result = None 

            data_dependency_parents = [self.dag.steps[pid] for pid in self.dag.get_data_dependency(step_id)]
            data_dependency_context = {
                "graph_dependencies": [],
                "parameter_dependencies": [],
                "graph_input_adapter_result": None,
                "parameter_input_adapter_result": None,
            }
            
            if data_dependency_parents:
                data_dependency_context = await self.data_dependency_resolver.resolve_dependencies(
                    step_id=step_id,
                    step=step,
                    alg_des_info=tool_metadata,
                    data_dependency_parents=data_dependency_parents
                )
                logger.info(
                    "📊 Dependency context | graph deps: %s | param deps: %s",
                    len(data_dependency_context.get("graph_dependencies", [])),
                    len(data_dependency_context.get("parameter_dependencies", [])),
                )
                

            if step.task_type == GraphAnalysisType.GRAPH_ALGORITHM:
                if not step.graph_algorithm:
                    self.dag.set_failed(step_id, "No graph algorithm selected for this step")
                    logger.error(f"❌ Step {step_id} missing graph_algorithm, skipping")
                    continue

                tool_description,  tool_metadata = await self.computing_engine.get_algorithm_description(
                    step.graph_algorithm
                )
                
                if tool_metadata is None:
                    error_msg = f"Cannot get algorithm description for '{step.graph_algorithm}'"
                    logger.error(f"❌ {error_msg}")
                    self.dag.set_failed(step_id, error_msg)
                    continue
                
                # logger.info(f"tool_description:{tool_description}")

                try:
                    await self._prepare_graph_for_execution(
                        graph_dependencies=data_dependency_context.get("graph_dependencies") or [],
                        graph_adapter_result=data_dependency_context.get("graph_input_adapter_result"),
                    )
                except Exception as graph_err:
                    self.dag.set_failed(step_id, f"Failed to initialize working graph: {graph_err}")
                    logger.error(f"❌ Step {step_id} graph init failed: {graph_err}, skipping")
                    continue

                # extraction_result = self._prepare_parameters_for_execution(
                #     step=step,
                #     tool_description=tool_description,
                #     vertex_schema=self.global_graph.get_vertex_properties_schema(),
                #     edge_schema=self.global_graph.get_edge_properties_schema(),
                #     dependency_parameters=data_dependency_context.get("parameter_input_adapter_result") or {},
                # )
                
                # if extraction_result is None:
                #     error_msg = "Parameter extraction returned None"
                #     logger.error(f"❌ {error_msg}")
                #     self.dag.set_failed(step_id, error_msg)
                #     raise ValueError(error_msg)

                # logger.info(
                #     "✅ LLM/dependency extracted params: %s",
                #     json.dumps(extraction_result.get("parameters", {}), ensure_ascii=False)
                # )

                # post_processing_info = extraction_result.get("post_processing_code") or {}  
                # is_has_extract_code = bool(post_processing_info.get("is_calculate"))
                # output_schema = post_processing_info.get("output_schema") or {}

                # if post_processing_info.get("code"):
                #     logger.info(
                #         f"✅ Post-processing code:\n{post_processing_info.get('code','')}...")
        
                # tool_result = await self.computing_engine.run_algorithm(
                #     step.graph_algorithm,
                #     extraction_result.get("parameters", {}),
                #     post_processing_info.get("code"),
                #     global_graph=self.global_graph
                # )
                
                # if tool_result is None:
                #     error_msg = "Algorithm execution returned None"
                #     logger.error(f"❌ {error_msg}")
                #     self.dag.set_failed(step_id, error_msg)
                #     raise ValueError(error_msg)
                
                # # Check nested result.error
                # result_data = tool_result.get("result")
                # if result_data is not None and isinstance(result_data, dict) and "error" in result_data:
                #     error_msg = result_data.get("error")
                #     logger.error(f"❌ {error_msg}")
                #     self.dag.set_failed(step_id, error_msg)
                #     raise RuntimeError(error_msg)
                
                # if not tool_result.get("success", False):
                #     error_msg = tool_result.get("error", "Algorithm execution failed")
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
                        raise ValueError("Algorithm execution returned None")
                    logger.info(f"tool_result:{tool_result}")

                    result_data = tool_result.get("result")
                    if isinstance(result_data, dict) and "error" in result_data:
                        raise RuntimeError(result_data.get("error") or "Algorithm execution failed")
                    logger.info(f"tool_result:{result_data}")


                    if not tool_result.get("success", False):
                        error_msg = tool_result.get("error") or "Algorithm execution failed"
                        logger.info(f"tool_result:{error_msg}")
                        raise RuntimeError(error_msg)

                    return is_has_extract_code, output_schema, tool_result

                try:
                    is_has_extract_code, output_schema, tool_result = await self.error_recovery.run(
                        op_prepare_params_and_postprocess_then_execute,
                        name=f"prepare_params+postprocess+execute(step={step_id})",
                        operation_type="generic",
                        location=f"step_{step_id}"
                    )
                except Exception as e:
                    self.dag.set_failed(step_id, str(e))
                    logger.error(f"❌ Step {step_id} algorithm execution failed: {e}, skipping")
                    continue

                step.add_algorithm_result(
                    tool_name=tool_metadata.get("name", ""),
                    tool_result_data=tool_result.get("result", {}),
                    output_schema=output_schema,
                    is_has_extract_code=is_has_extract_code
                )
                logger.info(f"✅ Tool execution: {tool_result.get('summary','done')}")

                self.dag.set_success(step_id)

            elif step.task_type == GraphAnalysisType.NUMERIC_ANALYSIS:
                logger.info("🧮 Task type: Numeric Analysis")
                
                graph_deps = data_dependency_context.get("graph_dependencies", [])
                param_deps = data_dependency_context.get("parameter_dependencies", [])
                
                dependency_items = []
                execution_data = {}
                tool_description = (
                    "##This step is a numeric analysis task executed by a generated Python function.\n\n"
                    "Inputs are dependency fields extracted from the workflow context.\n"
                )
                for dep_item in graph_deps + param_deps:
                    value = take_sample(dep_item.value)
                    execution_data[dep_item.field_key] = value
                    dependency_items.append({
                        "field_key": dep_item.field_key,
                        "field_type": str(dep_item.field_type) if dep_item.field_type else "unknown",
                        "field_desc": dep_item.field_desc,
                        "value": value
                    })
                    tool_description += (
                        f"- `{dep_item.field_key}`: {dep_item.field_desc or ''} "
                        f"(type: {str(dep_item.field_type) if dep_item.field_type else 'unknown'})\n"
                    )
                
            
                vertex_schema = {}
                edge_schema = {}
                if self.global_graph:
                    vertex_schema = self.global_graph.get_vertex_properties_schema()
                    edge_schema = self.global_graph.get_edge_properties_schema()
                
                async def op_generate_and_execute_numeric_analysis(error_history):
                    code_result = self.reasoner.generate_numeric_analysis_code(
                        question=step.question,
                        dependency_items=dependency_items,
                        vertex_schema=vertex_schema,
                        edge_schema=edge_schema
                    )
                    if code_result is None:
                        raise ValueError("Numeric analysis code generation returned None")

                    numeric_analysis_code = code_result.get("numeric_analysis_code", {})
                    generated_code = numeric_analysis_code.get("code", "")
                    output_schema = numeric_analysis_code.get("output_schema", {})
                    if not generated_code or not generated_code.strip():
                        raise ValueError("Generated numeric analysis code is empty")

                    logger.info(f"✅ Generated numeric analysis code: {generated_code}")
                    code_result_value = self.computing_engine.execute_code(
                        code=generated_code,
                        data=execution_data,
                        global_graph=self.global_graph,
                        is_numeric_analysis=True
                    )

                    if not isinstance(code_result_value, dict):
                        raise RuntimeError("Numeric analysis executor returned non-dict result")

                    if code_result_value.get("error"):
                        raise RuntimeError(code_result_value.get("error") or "Numeric analysis execution failed")

                    return output_schema, code_result_value

                try:
                    output_schema, code_result_value = await self.error_recovery.run(
                        op_generate_and_execute_numeric_analysis,
                        name=f"generate_numeric_code+execute(step={step_id})",
                        operation_type="generic",
                        location=f"step_{step_id}"
                    )
                except Exception as e:
                    self.dag.set_failed(step_id, str(e))
                    logger.error(f"❌ Step {step_id} numeric analysis failed: {e}, skipping")
                    continue

                result = code_result_value.get("result")
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
                    value = result if isinstance(result, dict) else {"result": result},
                    path=None,
                    validate_schema=True
                )
                # Describe expected outputs for the generated function.
                tool_description += "\nExpected output:\n"
                if output_schema:
                    output_fields = output_schema.get("fields", {}) or {}
                    if output_fields:
                        for name, info in output_fields.items():
                            tool_description += (
                                f"- `{name}`: {info.get('field_description', '')} "
                                f"(type: {info.get('type', '')})\n"
                            )
                    else:
                        tool_description += f"- Output type: {output_schema.get('type', 'dict')}\n"
                else:
                    tool_description += "- Output: numeric analysis result\n"

                tool_result = code_result_value
                
                logger.info("✅ Numeric analysis completed")
                self.dag.set_success(step_id)

            elif step.task_type == GraphAnalysisType.GRAPH_QUERY:
                logger.info("🔍 Task type: Graph Query")
                try:
                    query_result = self.computing_engine.execute_graph_query(step.question)
                    
                    if query_result.get("success"):
                        step.add_output(
                            task_type=GraphAnalysisSubType.SUBGRAPH_EXTRACTION,
                            source="graph_query",
                            output_schema=OutputSchema(
                                description="Graph query result",
                                type="list",
                                fields={
                                    "results": OutputField(
                                        type="list",
                                        field_description="List of query results"
                                    ),
                                    "count": OutputField(
                                        type="int",
                                        field_description="Result count"
                                    ),
                                    "query_type": OutputField(
                                        type="str",
                                        field_description="Query type"
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
                        logger.info(f"✅ Graph query done; returned {query_result.get('count', 0)} results")
                        self.dag.set_success(step_id)
                        
                        tool_result = {
                            "success": True,
                            "result": query_result.get("results", []),
                            "summary": f"Graph query succeeded; returned {query_result.get('count', 0)} results"
                        }
                    else:
                        error_msg = query_result.get("error", "Graph query failed")
                        logger.error(f"❌ Step {step_id} graph query failed: {error_msg}, skipping")
                        self.dag.set_failed(step_id, error_msg)
                        continue
                        
                except Exception as query_err:
                    error_msg = f"Graph query execution failed: {query_err}"
                    logger.error(error_msg, exc_info=True)
                    self.dag.set_failed(step_id, error_msg)
                    continue

            if tool_result is not None:
                try:
                    llm_analysis = self.reasoner.generate_answer_from_algorithm_result(
                        question=step.question,
                        tool_description=tool_description,
                        tool_result=tool_result,
                    )
                    analysis_blocks.append(llm_analysis)
                    analysis_result += llm_analysis
                except Exception as analysis_err:
                    logger.error(f"❌ Step {step_id} LLM analysis generation failed: {analysis_err}, skipping")
            else:
                logger.warning(f"⚠️ Step {step_id} skipped (no result), excluded from final report")

        final_question = (
            "You will be given the concatenated analysis text generated from "
            "multiple graph analysis steps. Please reorganize it into a single, "
            "clear and well-structured Markdown report for the end user."
        )
        final_tool_description = (
            "This tool_result contains intermediate analysis paragraphs from "
            "multiple steps of a graph analysis workflow. Summarize and refine "
            "them into one coherent report."
        )
        final_report = self.reasoner.generate_answer_from_algorithm_result(
            question=final_question,
            tool_description=final_tool_description,
            tool_result=analysis_result,
        )

        return final_report

    async def _prepare_graph_for_execution(
        self,
        *,
        graph_dependencies: List[Any],
        graph_adapter_result: Optional[Dict[str, Any]],
    ) -> Tuple[List[VertexData], List[EdgeData]]:
        """
        Initialize the graph to use (full or subgraph) from dependency context.
        """
        if self.global_graph is None:
            raise RuntimeError("Global graph not loaded; cannot run graph algorithms.")

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
                logger.warning("⚠️ Subgraph dependency has no valid nodes/edges; falling back to full graph.")
                use_subgraph = False

        payload = {
            "vertices": [v.to_dict() for v in vertices_to_use or []],
            "edges": [e.to_dict() for e in edges_to_use or []],
        }
        if self.current_graph_dataset:
            payload["dataset_config"] = self.current_graph_dataset.to_dict()

        result = await self.computing_engine.run_algorithm("initialize_graph", payload)
        if not result.get("success", False):
            raise RuntimeError(result.get("error", "Failed to initialize graph data"))

        logger.info(
            "🗺️ Working graph initialized | use_subgraph: %s | vertices: %s | edges: %s",
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
        Prepare graph algorithm input parameters from data dependencies or LLM output.
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
                trace=trace,
            )

        return self.reasoner.extract_parameters_with_postprocess_new(
            question=step.question,
            tool_description=tool_description or "",
            vertex_schema=vertex_schema,
            edge_schema=edge_schema,
            error_history=error_history,
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
        Extract subquery structure from the current DAG.

        Returns:
            Dict with a list of subqueries.
        """
        subqueries = []
        for step_id in self.dag.topological_order():
            step = self.dag.steps[step_id]
            
            query_id = None
            for qid, sid in self.query_id_mapping.items():
                if sid == step_id:
                    query_id = qid
                    break
            
            if query_id is None:
                query_id = f"q{step_id}"
            
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
        Check whether the DAG structure has changed.

        Args:
            old_structure: Previous DAG structure.
            new_structure: New DAG structure.

        Returns:
            True if changed, False otherwise.
        """
        old_queries = old_structure.get("subqueries", [])
        new_queries = new_structure.get("subqueries", [])
        
        if len(old_queries) != len(new_queries):
            return True
        
        for old_q, new_q in zip(old_queries, new_queries):
            if old_q.get("query") != new_q.get("query"):
                return True
            if set(old_q.get("depends_on", [])) != set(new_q.get("depends_on", [])):
                return True
        
        return False
    
    def _refine_dag_with_retry(self, max_retries: int = 1) -> Dict[str, Any]:
        """
        Refine DAG with LLM to clarify task type boundaries.

        Args:
            max_retries: Maximum number of retries.

        Returns:
            Refined DAG info.
        """
        logger.info("🔄 Starting DAG refinement...")
        
        for retry_count in range(max_retries + 1):
            try:
                current_dag_structure = self._extract_dag_structure()
                
                logger.info(f"📋 Current DAG structure (attempt {retry_count + 1}/{max_retries + 1}):")
                logger.info(json.dumps(current_dag_structure, ensure_ascii=False, indent=2))
                
                refined_structure = self.reasoner.refine_subqueries(current_dag_structure)
                
                logger.info("✅ LLM refined DAG:")
                logger.info(json.dumps(refined_structure, ensure_ascii=False, indent=2))
                
                if self._is_dag_changed(current_dag_structure, refined_structure):
                    logger.info("🔄 DAG structure changed; rebuilding...")
                    
                    self.build_dag_from_subquery_plan(refined_structure)
                    self._find_algorithm()
                    
                    logger.info(f"✅ DAG refinement done (iteration {retry_count + 1})")
                else:
                    logger.info("✅ DAG already refined; no further changes")
                    break
                    
            except Exception as e:
                logger.error(f"❌ DAG refinement failed (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                if retry_count == max_retries:
                    logger.warning("⚠️ Max retries reached; continuing with current DAG")
                continue
        
        return self.dag.get_dag_info()

    async def shutdown(self):
        if self.computing_engine:
            await self.computing_engine.shutdown()
        
