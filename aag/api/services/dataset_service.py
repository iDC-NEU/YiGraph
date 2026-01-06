"""
数据集管理服务
封装DocumentAPI，提供更简洁的接口
"""

import logging
import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from aag.api.DocumentAPI import DocumentAPIServer, DummySocket

logger = logging.getLogger(__name__)


class CollectingSocket:
    """用于收集DocumentAPI响应的Socket类"""
    
    def __init__(self, msgs, collector):
        self.msgs = msgs
        self.collector = collector
    
    async def send(self, msg):
        self.collector.add_response(msg)
        logger.debug(f"[DatasetService] Response: {msg}")
    
    async def __aiter__(self):
        for m in self.msgs:
            yield m


class DatasetService:
    """
    数据集管理服务 - 封装DocumentAPI
    提供知识库的创建、删除、查询等功能
    """
    
    def __init__(self):
        """初始化数据集服务"""
        self.document_api = DocumentAPIServer()
        self._response_cache = {}  # 用于存储异步响应结果
    
    async def create_knowledge_base(
        self,
        file_path: str,
        graph_name: str,
        db_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        创建知识库（从文本文件）
        
        Args:
            file_path: 文件路径
            graph_name: 图名称
            db_name: 数据库名称（可选）
        
        Returns:
            创建结果字典
        """
        try:
            logger.info(f"创建知识库: graph_name={graph_name}, file_path={file_path}")
            
            message = {
                "action": "create_kb",
                "file_path": file_path,
                "graph_name": graph_name
            }
            
            if db_name:
                message["db_name"] = db_name
            
            # 创建响应收集器
            response_collector = ResponseCollector()
            dummy_socket = CollectingSocket([json.dumps(message)], response_collector)
            
            # 执行处理
            await self.document_api.handler(dummy_socket)
            
            # 获取响应
            responses = response_collector.get_responses()
            
            # 解析响应
            for response in responses:
                try:
                    data = json.loads(response)
                    if data.get("type") == "data" and data.get("contentType") == "json":
                        content = data.get("content", {})
                        if content.get("success"):
                            return {
                                "success": True,
                                "data": content.get("data", {})
                            }
                    elif data.get("type") == "error":
                        return {
                            "success": False,
                            "error": data.get("content", "未知错误")
                        }
                except json.JSONDecodeError:
                    continue
            
            return {
                "success": True,
                "message": "知识库创建请求已提交"
            }
            
        except Exception as e:
            logger.error(f"创建知识库失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_knowledge_base(
        self,
        graph_name: str,
        db_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        删除知识库
        
        Args:
            graph_name: 图名称
            db_name: 数据库名称（可选）
        
        Returns:
            删除结果字典
        """
        try:
            logger.info(f"删除知识库: graph_name={graph_name}")
            
            message = {
                "action": "delete_kb",
                "graph_name": graph_name
            }
            
            if db_name:
                message["db_name"] = db_name
            
            # 创建响应收集器
            response_collector = ResponseCollector()
            dummy_socket = CollectingSocket([json.dumps(message)], response_collector)
            
            # 执行处理
            await self.document_api.handler(dummy_socket)
            
            # 获取响应
            responses = response_collector.get_responses()
            
            # 解析响应
            for response in responses:
                try:
                    data = json.loads(response)
                    if data.get("type") == "data" and data.get("contentType") == "json":
                        content = data.get("content", {})
                        if content.get("success"):
                            return {
                                "success": True,
                                "data": content.get("data", {})
                            }
                    elif data.get("type") == "error":
                        return {
                            "success": False,
                            "error": data.get("content", "未知错误")
                        }
                except json.JSONDecodeError:
                    continue
            
            return {
                "success": True,
                "message": "知识库删除请求已提交"
            }
            
        except Exception as e:
            logger.error(f"删除知识库失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_triplets(
        self,
        graph_name: str
    ) -> Dict[str, Any]:
        """
        获取知识库的三元组
        
        Args:
            graph_name: 图名称
        
        Returns:
            三元组列表
        """
        try:
            logger.info(f"获取三元组: graph_name={graph_name}")
            
            message = {
                "action": "get_triplets",
                "graph_name": graph_name
            }
            
            # 创建响应收集器
            response_collector = ResponseCollector()
            dummy_socket = CollectingSocket([json.dumps(message)], response_collector)
            
            # 执行处理
            await self.document_api.handler(dummy_socket)
            
            # 获取响应
            responses = response_collector.get_responses()
            
            # 解析响应
            for response in responses:
                try:
                    data = json.loads(response)
                    if data.get("type") == "data" and data.get("contentType") == "json":
                        content = data.get("content", {})
                        if content.get("success"):
                            return {
                                "success": True,
                                "data": content.get("data", [])
                            }
                    elif data.get("type") == "error":
                        return {
                            "success": False,
                            "error": data.get("content", "未知错误")
                        }
                except json.JSONDecodeError:
                    continue
            
            return {
                "success": False,
                "error": "未获取到响应"
            }
            
        except Exception as e:
            logger.error(f"获取三元组失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_kb_from_graph(
        self,
        graph_name: str,
        edge_file: str,
        source_field: str,
        target_field: str,
        relation_field: str,
        vertex_file: Optional[str] = None,
        vertex_id_field: Optional[str] = None,
        vertex_name_field: Optional[str] = None,
        weight_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从图文件创建知识库
        
        Args:
            graph_name: 图名称
            edge_file: 边文件路径
            source_field: 源节点字段名
            target_field: 目标节点字段名
            relation_field: 关系字段名
            vertex_file: 顶点文件路径（可选）
            vertex_id_field: 顶点ID字段名（可选）
            vertex_name_field: 顶点名称字段名（可选）
            weight_field: 权重字段名（可选）
        
        Returns:
            创建结果字典
        """
        try:
            logger.info(f"从图文件创建知识库: graph_name={graph_name}")
            
            message = {
                "action": "upload_graph",
                "graph_name": graph_name,
                "edge_file": edge_file,
                "source_field": source_field,
                "target_field": target_field,
                "relation_field": relation_field
            }
            
            if vertex_file:
                message["vertex_file"] = vertex_file
                message["vertex_id_field"] = vertex_id_field
                message["vertex_name_field"] = vertex_name_field
            
            if weight_field:
                message["weight_field"] = weight_field
            
            # 创建响应收集器
            response_collector = ResponseCollector()
            dummy_socket = CollectingSocket([json.dumps(message)], response_collector)
            
            # 执行处理
            await self.document_api.handler(dummy_socket)
            
            # 获取响应
            responses = response_collector.get_responses()
            
            # 解析响应
            for response in responses:
                try:
                    data = json.loads(response)
                    if data.get("type") == "data" and data.get("contentType") == "json":
                        content = data.get("content", {})
                        if content.get("success"):
                            return {
                                "success": True,
                                "data": content.get("data", {})
                            }
                    elif data.get("type") == "error":
                        return {
                            "success": False,
                            "error": data.get("content", "未知错误")
                        }
                except json.JSONDecodeError:
                    continue
            
            return {
                "success": True,
                "message": "知识库创建请求已提交"
            }
            
        except Exception as e:
            logger.error(f"从图文件创建知识库失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


class ResponseCollector:
    """用于收集异步响应的辅助类"""
    
    def __init__(self):
        self.responses = []
    
    def add_response(self, response: str):
        """添加响应"""
        self.responses.append(response)
    
    def get_responses(self) -> List[str]:
        """获取所有响应"""
        return self.responses
    
    def clear(self):
        """清空响应"""
        self.responses = []

