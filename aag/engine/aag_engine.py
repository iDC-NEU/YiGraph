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
