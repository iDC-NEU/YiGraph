"""
AAG Engine
端到端的图分析、检索增强生成和大语言模型框架
"""

import os
import time
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 导入各个组件
from aag.config.engine_config import EngineConfig   
from aag.engine.scheduler import Scheduler



class AAGEngine:
    """
    Analytics Augmented Generation Engine - 端到端分析增强生成框架
    
    主要功能：
    1. 图算法定位
    2. 多模态检索（图检索 + 向量检索）, 向量检索返回论文, 图检索返回图数据
    3. 图计算框架执行图计算
    4. LLM生成回答
    5. 结果后处理和评估
    """

    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.scheduler: Optional[Scheduler] = None
    
        # 性能监控
        self.metrics = {
            "retrieval_time": [],
            "generation_time": [],
            "total_time": [],
            "retrieval_quality": [],
            "generation_quality": []
        }

        self._init_scheduler()
        

    def _init_scheduler(self):
        """初始化调度器"""
        try:
            self.scheduler = Scheduler(config = self.config)
            print("✓ Scheduler initialized")
        except Exception as e:
            print(f"✗ Scheduler initialization failed: {e}")
            raise
    
    def _initialize_components(self):
        """初始化各个组件"""
        print("Initializing AAG components...")
        
        # 1. 初始化调度器
        self._init_scheduler()
        
        print("Engine initialization completed!")
    

    async def run(self, query: str) -> str:
        return await self.scheduler.execute(query)


    def list_datasets(self, dtype: Optional[str] = None) -> Dict[str, List[str]]:
        return self.scheduler.list_datasets(dtype)

    def specific_dataset(self, name: str, dtype: Optional[str] = None) -> Optional[Any]:
        result = self.scheduler.specific_analysis_dataset(name, dtype)
        return result if result is not None else None

 
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

        # TODO(chaoyi): 
        # 1.遍历 dag, 对每个子问题确定适合的图算法， 从层次化的领域知识库里确定
        self.scheduler.find_graph_algorithm()

        # 2.调度器 run DAG


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
     
    async def shutdown(self):
        """关闭Engine，释放资源"""
        print("Shutting down GraphLLM Engine...")
        
        #TODO(chaoyi): 释放资源
        await self.scheduler.shutdown()
        
        print("✓ Engine shutdown completed")
