"""
AAG Engine: end-to-end graph analysis, retrieval-augmented generation, and LLM framework.
"""

import os
import time
import yaml
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from aag.config.engine_config import EngineConfig   
from aag.engine.scheduler import Scheduler


class AAGEngine:
    """
    Analytics Augmented Generation Engine: end-to-end analysis-augmented generation.

    Main capabilities:
    1. Graph algorithm selection
    2. Multi-modal retrieval (graph + vector); vector returns docs, graph returns graph data
    3. Graph computation execution
    4. LLM answer generation
    5. Post-processing and evaluation
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.scheduler: Optional[Scheduler] = None
        # Cache for dataset init: only re-run heavy init when name or dtype changes
        self._current_dataset_name: Optional[str] = None
        self._current_dataset_type: Optional[str] = None

        self.metrics = {
            "retrieval_time": [],
            "generation_time": [],
            "total_time": [],
            "retrieval_quality": [],
            "generation_quality": []
        }
        self._init_scheduler()
        

    def _init_scheduler(self):
        """Initialize the scheduler."""
        try:
            self.scheduler = Scheduler(config = self.config)
            print("✓ Scheduler initialized")
        except Exception as e:
            print(f"✗ Scheduler initialization failed: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize engine components."""
        print("Initializing AAG components...")
        self._init_scheduler()
        print("Engine initialization completed!")
    

    async def run(
        self,
        query: str,
        mode: str = "normal",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Run a query (normal or expert mode).

        Args:
            query: User query.
            mode: "normal" | "interact" | "expert".
            callback: Optional callback for streaming (e.g. DAG); signature callback(data: Dict).
                     Used in Web flow only; omit for CLI.

        Returns:
            Normal: analysis result string.
            Interact/expert: dict with dag_info, message, etc.
        """
        return await self.scheduler.execute(query, mode=mode, callback=callback)
    
    async def expert_modify_dag(self, modification_request: str) -> Dict[str, Any]:
        """
        Expert mode: modify the DAG.

        Args:
            modification_request: User modification request.

        Returns:
            Dict with updated DAG info.
        """
        return await self.scheduler.expert_modify_dag(modification_request)
    
    async def expert_start_analysis(self) -> str:
        """
        Expert mode: start analysis.

        Returns:
            Analysis result string.
        """
        return await self.scheduler.expert_start_analysis()

        
    def list_datasets(self, dtype: Optional[str] = None) -> Dict[str, List[str]]:
        return self.scheduler.list_datasets(dtype)

    def specific_dataset(self, name: str, dtype: Optional[str] = None) -> Optional[Any]:
        dtype = (dtype or "").strip() or None
        if name == self._current_dataset_name and dtype == self._current_dataset_type:
            return None
        result = self.scheduler.specific_analysis_dataset(name, dtype)
        self._current_dataset_name = name
        self._current_dataset_type = dtype
        return result if result is not None else None


    def _record_metrics(self, retrieval_time: float, generation_time: float, total_time: float):
        """Record performance metrics."""
        self.metrics["retrieval_time"].append(retrieval_time)
        self.metrics["generation_time"].append(generation_time)
        self.metrics["total_time"].append(total_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Return a performance summary."""
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
        """Clear performance metrics."""
        for key in self.metrics:
            self.metrics[key].clear()
     
    async def shutdown(self):
        """Shut down the engine and release resources."""
        print("Shutting down GraphLLM Engine...")
        await self.scheduler.shutdown()
        print("✓ Engine shutdown completed")
