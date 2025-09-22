"""
GraphLLM Engine
端到端的图分析、检索增强生成和大语言模型框架
"""

import os
import time
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 导入各个组件
from graphllm.database.nebulagraph import NebulaDB
from graphllm.database.milvus import MilvusDB
from rag_engine.rag import RAG_Engine
from graphllm.model_deploy.model_deployment import OpenAIEnv, OllamaEnv, EmbeddingEnv
from graphllm.graph_engine.graphcomputation_processor import GraphProcessor
from graphllm.graph_engine.graphlearning_processor import GraphLearningProcessor
from graphllm.planner_and_scheduler.scheduler import Scheduler



@dataclass
class EngineConfig:
    """Engine配置类"""
    # 数据库配置
    graph_db_config: Dict[str, Any]
    vector_db_config: Dict[str, Any]
    
    # 模型配置
    llm_config: Dict[str, Any]
    embedding_config: Dict[str, Any]
    
    # RAG配置
    graph_rag_config: Dict[str, Any]
    vector_rag_config: Dict[str, Any]
    
    # 数据处理配置
    data_process_config: Dict[str, Any]


class GraphLLMEngine:
    """
    端到端的Graph Analysis + RAG + LLM Engine
    
    主要功能：
    1. 图算法定位
    2. 多模态检索（图检索 + 向量检索）, 向量检索返回论文, 图检索返回图数据
    3. 图计算框架执行图计算
    4. LLM生成回答
    5. 结果后处理和评估
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.graph_db: Optional[NebulaDB] = None
        self.vector_db: Optional[MilvusDB] = None
        self.rag_engine: Optional[RAG_Engine] = None
        self.llm_env: Optional[Union[OpenAIEnv, OllamaEnv]] = None
        self.embedding_env: Optional[EmbeddingEnv] = None
        self.graph_processor: Optional[GraphProcessor] = None
        self.gnn_engine: Optional[GraphLearningProcessor] = None
        self.scheduler: Optional[Scheduler] = None
        
        # 性能监控
        self.metrics = {
            "retrieval_time": [],
            "generation_time": [],
            "total_time": [],
            "retrieval_quality": [],
            "generation_quality": []
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化各个组件"""
        print("Initializing GraphLLM Engine components...")
        
        # 1. 初始化数据库
        self._init_databases()
        
        # 2. 初始化模型环境
        self._init_models()
        
        # 3. 初始化RAG引擎
        self._init_rag_engines()

        # 4. 初始化图处理器
        self._init_graph_processor()

        # 5. 初始化图学习处理器
        self._init_graph_learning_processor()

        # 6. 初始化调度器
        self._init_scheduler()
        
        print("Engine initialization completed!")
    
    def _init_databases(self):
        """初始化数据库连接"""
        try:
            # 初始化图数据库
            self.graph_db = NebulaDB(
                space_name=self.config.graph_db_config["space_name"],
                server_ip=self.config.graph_db_config.get("server_ip", "127.0.0.1"),
                server_port=self.config.graph_db_config.get("server_port", "9669"),
                create=self.config.graph_db_config.get("create", False),
                verbose=self.config.graph_db_config.get("verbose", False)
            )
            print(f"✓ Graph database initialized: {self.config.graph_db_config['space_name']}")
            
            # 初始化向量数据库
            self.vector_db = MilvusDB(
                collection_name=self.config.vector_db_config["collection_name"],
                dim=self.config.vector_db_config["dim"],
                host=self.config.vector_db_config.get("host", "localhost"),
                port=self.config.vector_db_config.get("port", 19530),
                store=True,
                retriever=True
            )
            print(f"✓ Vector database initialized: {self.config.vector_db_config['collection_name']}")
            
        except Exception as e:
            print(f"✗ Database initialization failed: {e}")
            raise
    
    def _init_models(self):
        """初始化模型环境"""
        try:
            llm_type = self.config.llm_config.get("type", "ollama")
            
            if llm_type == "ollama":
                self.llm_env = OllamaEnv(
                    llm_mode_name=self.config.llm_config["model_name"],
                    llm_embed_name=self.config.embedding_config["model_name"],
                    chunk_size=self.config.llm_config.get("chunk_size", 512),
                    chunk_overlap=self.config.llm_config.get("chunk_overlap", 20),
                    embed_batch_size=self.config.embedding_config.get("batch_size", 20),
                    device=self.config.llm_config.get("device", "cuda:0"),
                    timeout=self.config.llm_config.get("timeout", 150000),
                    port=self.config.llm_config.get("port", 11434)
                )
            elif llm_type == "openai":
                self.llm_env = OpenAIEnv(
                    self.config.llm_config["openai_base_url"],
                    self.config.llm_config["openai_api_key"],
                    self.config.llm_config["openai_model"]
                )
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")
            
            print(f"✓ LLM environment initialized: {llm_type}")
            
            # 初始化嵌入模型环境
            self.embedding_env = EmbeddingEnv(
                embed_name=self.config.embedding_config["model_name"],
                embed_batch_size=self.config.embedding_config.get("batch_size", 20),
                device=self.config.embedding_config.get("device", "cuda:0")
            )
            print(f"✓ Embedding environment initialized: {self.config.embedding_config['model_name']}")
            
        except Exception as e:
            print(f"✗ Model initialization failed: {e}")
            raise
    
    def _init_rag_engines(self):
        """初始化RAG引擎"""
        try:
            self.rag_engine = RAG_Engine(
                vector_db=self.vector_db,
                vector_k_similarity=self.config.vector_rag_config.get("k_similarity", 5),
                graph_db=self.graph_db,
                vector_rag_llm_env_=self.embedding_env,
                graph_rag_llm_env_=self.llm_env
            )  
            print("✓ RAG engine initialized")
            
        except Exception as e:
            print(f"✗ RAG engine initialization failed: {e}")
            raise
    
    def _init_graph_processor(self):
        """初始化图处理器"""
        self.graph_processor = GraphProcessor()

    def _init_graph_learning_processor(self):
        """初始化图学习处理器"""
        try:
            self.gnn_engine = GraphLearningProcessor()
            print("✓ Graph learning processor initialized")
        except Exception as e:
            print(f"✗ Graph learning processor initialization failed: {e}")
            raise

    def _init_scheduler(self):
        """初始化调度器"""
        try:
            self.scheduler = Scheduler(
                graph_engine=self.graph_processor,
                gnn_engine=self.gnn_engine,
                llm_client=self.llm_env,
                max_retries=0,
                retry_sleep_sec=0.0,
            )
            print("✓ Scheduler initialized")
        except Exception as e:
            print(f"✗ Scheduler initialization failed: {e}")
            raise

 
    def query(self, question: str) -> str:
        """
        查询处理：多模态检索 + LLM生成
        
        Args:
            question: 查询问题
            
        Returns:
            回答
        """
        start_time = time.time()
        
        # TODO(zihan): 根据输入问题，生成 subquery 构成的 plan，并将 plan 转换为 DAG
        query_plan =  self.llm_env.plan_subqueries(True, question)  # 在 /home/chency/GraphLLM/graphllm/model_deploy/model_deployment.py 中  OllamaEnv 和  OpenAIEnv 分别实现该函数
        self.scheduler.build_dag_from_subquery_plan(query_plan)


        # 1. 检索阶段: 调用self.rag_engine.vector_rag.retrieve() 检索与问题相关的论文
        retrieval_result, retrieve_information = self.rag_engine.vector_rag.retrieve(question)

        # 2. 图算法和问题实体确定阶段，先调用 self.llm_env.get_graph_algorithm 确定图算法执行计划; 再调用 self.llm_env.get_question_entity 确定问题实体
        graph_algorithm_plan = self.llm_env.get_graph_algorithm(question, retrieval_result)
        # question_entity_list = self.llm_env.get_question_entity(question)   # 暂时不使用

        # 3. 根据问题实体列表，调用self.rag_engine.graph_rag.query_with_entity() 获取这些子图的边表形式
        # graph_data_list = self.rag_engine.graph_rag.query_with_entity(question_entity_list)
        graph_data_list = self.rag_engine.graph_rag.get_all_edges()

        # 4. 将边表和算法执行计划传入self.graph_processor.graph_algorithm_execution() 执行图算法
        results  = self.graph_processor.execute_plan(graph_data_list, graph_algorithm_plan)
        
        # 5. 将图算法结果list中的最后一个结果和问题实体列表传入self.llm_env.get_graph_algorithm_result() 生成回答
        response = self.llm_env.get_graph_algorithm_result(results[-1], question_entity_list)

        return response

    
    def _record_metrics(self, retrieval_time: float, generation_time: float, total_time: float):
        """记录性能指标"""
        self.metrics["retrieval_time"].append(retrieval_time)
        self.metrics["generation_time"].append(generation_time)
        self.metrics["total_time"].append(total_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics["total_time"]:
            return {"message": "No queries processed yet"}
        
        return {
            "total_queries": len(self.metrics["total_time"]),
            "avg_retrieval_time": sum(self.metrics["retrieval_time"]) / len(self.metrics["retrieval_time"]),
            "avg_generation_time": sum(self.metrics["generation_time"]) / len(self.metrics["generation_time"]),
            "avg_total_time": sum(self.metrics["total_time"]) / len(self.metrics["total_time"]),
            "total_time": sum(self.metrics["total_time"])
        }
    
    def clear_metrics(self):
        """清空性能指标"""
        for key in self.metrics:
            self.metrics[key].clear()
    
    def shutdown(self):
        """关闭Engine，释放资源"""
        print("Shutting down GraphLLM Engine...")
        
        if self.graph_db:
            # 图数据库连接会在析构函数中自动关闭
            pass
        
        if self.vector_db:
            # 向量数据库连接会在析构函数中自动关闭
            pass
        
        print("✓ Engine shutdown completed")


def load_config_from_yaml(config_path: str) -> EngineConfig:
    """从YAML配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return EngineConfig(
            graph_db_config=config_data.get("database", {}).get("graph", {}),
            vector_db_config=config_data.get("database", {}).get("vector", {}),
            llm_config=config_data.get("models", {}).get("llm", {}),
            embedding_config=config_data.get("models", {}).get("embedding", {}),
            graph_rag_config=config_data.get("rag", {}).get("graph", {}),
            vector_rag_config=config_data.get("rag", {}).get("vector", {})
        )
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


# 便捷的配置创建函数
def create_engine_config(
    graph_space_name: str = "graphllm_space",
    vector_collection_name: str = "graphllm_collection",
    llm_model: str = "llama3.1:70b",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    **kwargs
) -> EngineConfig:
    """创建Engine配置"""
    
    return EngineConfig(
        graph_db_config={
            "space_name": graph_space_name,
            "server_ip": kwargs.get("graph_server_ip", "127.0.0.1"),
            "server_port": kwargs.get("graph_server_port", "9669"),
            "create": kwargs.get("create_graph_space", True),
            "verbose": kwargs.get("graph_verbose", False)
        },
        vector_db_config={
            "collection_name": vector_collection_name,
            "dim": kwargs.get("vector_dim", 1024),
            "host": kwargs.get("vector_host", "localhost"),
            "port": kwargs.get("vector_port", 19530)
        },
        llm_config={
            "type": kwargs.get("llm_type", "ollama"),
            "model_name": llm_model,
            "chunk_size": kwargs.get("chunk_size", 512),
            "chunk_overlap": kwargs.get("chunk_overlap", 20),
            "device": kwargs.get("llm_device", "cuda:0"),
            "timeout": kwargs.get("llm_timeout", 150000),
            "port": kwargs.get("ollama_port", 11434)
        },
        embedding_config={
            "model_name": embedding_model,
            "batch_size": kwargs.get("embed_batch_size", 20),
            "device": kwargs.get("embed_device", "cuda:0")
        },
        graph_rag_config={
            "k_hop": kwargs.get("graph_k_hop", 2),
            "pruning": kwargs.get("graph_pruning", 30),
            "data_type": kwargs.get("graph_data_type", "qa"),
            "pruning_mode": kwargs.get("graph_pruning_mode", "embedding_for_perentity")
        },
        vector_rag_config={
            "k_similarity": kwargs.get("vector_k_similarity", 5),
            "data_type": kwargs.get("vector_data_type", "summary")
        },
        data_process_config={
            "openai_api_key": kwargs.get("openai_api_key"),
            "extraction_model": kwargs.get("extraction_model", "gpt-3.5-turbo")
        }
    ) 