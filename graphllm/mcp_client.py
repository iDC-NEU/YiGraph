"""
MCP Client - 仅负责与MCP Server通信
"""
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import asyncio
import json
import logging
import csv
from typing import Any, Dict, List, Optional
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphMCPClient:
    """图计算MCP客户端 - 仅负责MCP通信"""
    
    def __init__(self, server_command: str = "python", server_args: List[str] = None):
        self.server_command = server_command
        self.server_args = server_args or ["graphllm/mcp_server.py"]
        self.session = None
        
    async def connect(self):
        """连接到MCP服务器"""
        try:
            server_params = StdioServerParameters(
                command=self.server_command,
                args=self.server_args
            )
            
            self.stdio_client = stdio_client(server_params)
            self.read_stream, self.write_stream = await self.stdio_client.__aenter__()
            self.session = ClientSession(self.read_stream, self.write_stream)
            await self.session.__aenter__()
            await self.session.initialize()
            
            logger.info("✅ 成功连接到MCP服务器")
            return True
            
        except Exception as e:
            logger.error(f"❌ 连接MCP服务器失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if hasattr(self, 'stdio_client'):
                await self.stdio_client.__aexit__(None, None, None)
            logger.info("👋 已断开与MCP服务器的连接")
        except Exception as e:
            logger.error(f"❌ 断开连接时发生错误: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用的工具"""
        try:
            tools_response = await self.session.list_tools()
            tools = []
            for tool in tools_response.tools:
                tools.append({
                    'name': tool.name,
                    'description': tool.description,
                    'input_schema': tool.inputSchema
                })
            return tools
        except Exception as e:
            logger.error(f"❌ 获取工具列表失败: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP工具"""
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return json.loads(content.text)
            
            return {"error": "无法解析工具调用结果"}
            
        except Exception as e:
            logger.error(f"❌ 工具调用失败: {e}")
            return {"error": str(e)}
    
    def load_data_from_csv(self, accounts_file: str, transactions_file: str) -> tuple:
        """从CSV文件加载图数据"""
        vertices = []
        edges = []
        
        try:
            logger.info(f"📖 读取账户数据: {accounts_file}")
            with open(accounts_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vertex = {
                        'vid': row['acct_id'],
                        'properties': {k: v for k, v in row.items() if k != 'acct_id'}
                    }
                    vertices.append(vertex)
            
            logger.info(f"📖 读取交易数据: {transactions_file}")
            with open(transactions_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    edge = {
                        'src': row['orig_acct'],
                        'dst': row['bene_acct'],
                        'rank': 0,
                        'properties': {k: v for k, v in row.items() 
                                     if k not in ['orig_acct', 'bene_acct']}
                    }
                    edges.append(edge)
            
            logger.info(f"✅ 加载了 {len(vertices)} 个顶点和 {len(edges)} 条边")
            return vertices, edges
            
        except Exception as e:
            logger.error(f"❌ CSV数据加载失败: {e}")
            raise


async def create_client_and_connect() -> GraphMCPClient:
    """便捷函数:创建并连接客户端"""
    client = GraphMCPClient()
    await client.connect()
    return client
