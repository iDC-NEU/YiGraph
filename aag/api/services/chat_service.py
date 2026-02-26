"""
聊天服务
封装AAG引擎的聊天功能，处理用户消息
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from .engine_service import EngineService

logger = logging.getLogger(__name__)

# 聊天页统一错误提示（以 type: result 形式返回，便于在聊天框内展示，不弹窗）
CHAT_FRIENDLY_ERROR_MSG = "Request failed. Please try again later."


def _is_scheduler_error_text(text: str) -> bool:
    """判断是否为 scheduler 返回的业务错误字符串（应统一为友好提示）。"""
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    return t.startswith("⚠️") or t.startswith("❌") or "错误：" in t


class ChatService:
    """
    聊天服务 - 封装AAG引擎的聊天功能
    支持 normal / interact / expert 三种模式
    """
    
    def __init__(self):
        """初始化聊天服务"""
        self.engine_service = EngineService.get_instance()
    
    async def process_message(
        self,
        message: str,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        mode: str = "normal",
        expert_mode: bool = False
    ) -> Dict[str, Any]:
        """
        处理用户消息（同步接口）
        
        Args:
            message: 用户消息
            model: 模型名称（可选，用于日志）
            dataset: 数据集名称（可选）
            mode: 模式 "normal" | "interact" | "expert"
            expert_mode: 兼容旧前端开关（True时使用 interact）
        
        Returns:
            处理结果字典
        """
        try:
            # 兼容旧前端：expert_mode 开关对应新的 interact 模式
            if expert_mode:
                mode = "interact"
            
            engine = self.engine_service.get_engine()
            
            # 如果指定了数据集，先选择数据集
            if dataset:
                logger.info(f"选择数据集: {dataset}")
                engine.specific_dataset(dataset)
            
            logger.info(f"处理消息: mode={mode}, model={model}, message={message[:50]}...")
            
            # 执行查询
            result = await engine.run(message, mode=mode)
            
            return {
                "success": True,
                "result": result,
                "mode": mode
            }
        except Exception as e:
            logger.error(f"处理消息失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mode": mode
            }
    
    async def process_dag_modification(
        self,
        modification_request: str
    ) -> Dict[str, Any]:
        """
        处理DAG修改请求（专家模式）
        
        Args:
            modification_request: 用户修改需求
        
        Returns:
            包含更新后DAG信息的字典
        """
        try:
            engine = self.engine_service.get_engine()
            logger.info(f"处理DAG修改请求: {modification_request[:50]}...")
            
            result = await engine.expert_modify_dag(modification_request)
            
            return {
                "success": True,
                **result
            }
        except Exception as e:
            logger.error(f"DAG修改失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def start_expert_analysis(self) -> Dict[str, Any]:
        """
        开始专家模式分析
        
        Returns:
            分析结果字典
        """
        try:
            engine = self.engine_service.get_engine()
            logger.info("开始执行专家模式分析")
            
            result = await engine.expert_start_analysis()
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"专家模式分析失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_streaming_chat(
        self,
        message: str,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        dataset_type: Optional[str] = None,
        mode: str = "normal",
        expert_mode: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        流式处理聊天（用于WebSocket）
        
        Args:
            message: 用户消息
            model: 模型名称（可选）
            dataset: 数据集名称（可选）
            dataset_type: 数据集类型 "text" | "graph"（可选）
            mode: 模式 "normal" | "interact" | "expert"
            expert_mode: 兼容旧前端开关（True时使用 interact）
            callback: 回调函数，用于发送流式数据
                     签名: callback(data: Dict[str, Any])
        """
        try:
            # 兼容旧前端：expert_mode 开关对应新的 interact 模式
            if expert_mode:
                mode = "interact"
            
            engine = self.engine_service.get_engine()
            
            # 发送思考过程
            if callback:
                callback({
                    "type": "thinking",
                    "contentType": "text",
                    "content": "正在分析您的问题..."
                })
            
            # 如果指定了数据集，先选择数据集
            if dataset:
                logger.info(f"选择数据集: {dataset}, type: {dataset_type}")
                engine.specific_dataset(dataset, dataset_type)
            
            logger.info(f"流式处理消息: mode={mode}, message={message[:50]}...")
            
            # 执行查询（传递 callback，用于在 scheduler 中判断是否返回 DAG 信息）
            result = await engine.run(message, mode=mode, callback=callback)
            
            # 根据模式返回不同格式
            if mode in {"interact", "expert"} and isinstance(result, dict) and "dag_info" in result:
                # interact / expert 模式返回 DAG
                dag_content = self._convert_dag_to_frontend_format(result)
                if callback:
                    callback({
                        "type": "result",
                        "contentType": "dag",
                        "content": dag_content
                    })
            elif mode in {"interact", "expert"} and isinstance(result, dict) and "error" in result:
                if callback:
                    callback({
                        "type": "result",
                        "contentType": "text",
                        "content": CHAT_FRIENDLY_ERROR_MSG
                    })
            elif mode == "normal" and isinstance(result, dict) and "dag_info" in result:
                # 普通模式：如果返回了包含 DAG 信息的字典（Web 调用）
                # 先发送 DAG 信息
                dag_content = self._convert_dag_to_frontend_format({
                    "dag_info": result.get("dag_info", {})
                })
                if callback:
                    callback({
                        "type": "result",
                        "contentType": "dag",
                        "content": dag_content
                    })
                
                # 再发送分析结果文本
                result_text = result.get("analysis_result", "")
                if callback and result_text:
                    paragraphs = [p.strip() for p in result_text.split('\n') if p.strip()]
                    for para in paragraphs:
                        callback({
                            "type": "result",
                            "contentType": "text",
                            "content": para
                        })
            else:
                # 普通模式：返回字符串（向后兼容，或没有 callback 的情况）
                result_text = str(result) if result else "未获取到结果"
                if _is_scheduler_error_text(result_text):
                    result_text = CHAT_FRIENDLY_ERROR_MSG
                if callback:
                    paragraphs = [p.strip() for p in result_text.split('\n') if p.strip()]
                    for para in paragraphs:
                        callback({
                            "type": "result",
                            "contentType": "text",
                            "content": para
                        })
            
            # 发送结束信号
            if callback:
                callback({
                    "type": "stream_end"
                })
                
        except Exception as e:
            logger.error(f"流式处理失败: {str(e)}", exc_info=True)
            if callback:
                callback({
                    "type": "result",
                    "contentType": "text",
                    "content": CHAT_FRIENDLY_ERROR_MSG
                })
                callback({"type": "stream_end"})
            return
    
    def _convert_dag_to_frontend_format(self, dag_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将DAG结果转换为前端需要的格式
        
        Args:
            dag_result: AAG引擎返回的DAG结果
        
        Returns:
            前端格式的DAG数据
        """
        try:
            dag_info = dag_result.get("dag_info", {})
            steps = dag_info.get("steps", {})
            
            nodes = []
            edges = []
            
            # 构建节点
            for step_id, step_info in steps.items():
                # 格式化 tasktype 为前端期望的格式
                task_type = step_info.get("task_type", "")
                algorithm = step_info.get("algorithm", "")
                
                # 根据 task_type 和 algorithm 构建格式化的字符串
                if task_type == "graph_algorithm":
                    if algorithm:
                        tasktype_str = f"Graph Algorithm ({algorithm})"
                    else:
                        tasktype_str = "Graph Algorithm"
                elif task_type == "numeric_analysis":
                    tasktype_str = "Numeric Analysis"
                else:
                    tasktype_str = "Unknown"
                
                nodes.append({
                    "id": str(step_id),
                    "label": step_info.get("question", ""),
                    "tasktype": tasktype_str
                })
            
            # 构建边（使用DAG返回的边信息，如果没有则根据拓扑顺序构建）
            edges_info = dag_info.get("edges", [])
            if edges_info:
                # 使用DAG返回的边信息
                edges = edges_info
            else:
                # 如果没有边信息，根据拓扑顺序构建（向后兼容）
                topological_order = dag_info.get("topological_order", [])
                for i in range(len(topological_order) - 1):
                    edges.append({
                        "from": str(topological_order[i]),
                        "to": str(topological_order[i + 1])
                    })
            
            return {
                "nodes": nodes,
                "edges": edges
            }
        except Exception as e:
            logger.error(f"转换DAG格式失败: {str(e)}")
            return {
                "nodes": [],
                "edges": []
            }
    
    def process_dag_confirmation(
        self,
        dag_confirm: str,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        处理DAG确认（yes/no）
        
        Args:
            dag_confirm: "yes" 或 "no"
            callback: 回调函数
        
        Returns:
            处理结果
        """
        if dag_confirm == "yes":
            # 确认DAG，开始执行分析
            return self.start_expert_analysis()
        else:
            # 拒绝DAG，返回提示
            if callback:
                callback({
                    "type": "result",
                    "contentType": "text",
                    "content": "DAG已拒绝，请修改后重新提交"
                })
            return {
                "success": True,
                "message": "DAG已拒绝"
            }
