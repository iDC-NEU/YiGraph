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
        self.server_args = server_args or ["graphllm/mcp_server.py"] # 默认MCP服务器脚本路径
        self.session = None
        self.available_tools = {}  # 缓存工具定义
        
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
            
            # 连接后自动获取工具列表
            await self._load_available_tools()
            
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
    
    async def _load_available_tools(self):
        """加载并缓存可用工具列表"""
        try:
            tools_response = await self.session.list_tools()
            for tool in tools_response.tools:
                self.available_tools[tool.name] = {
                    'name': tool.name,
                    'description': tool.description,
                    'input_schema': tool.inputSchema
                }
            logger.info(f"📋 加载了 {len(self.available_tools)} 个可用工具")
        except Exception as e:
            logger.error(f"❌ 加载工具列表失败: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用的工具"""
        if not self.available_tools:
            await self._load_available_tools()
        return list(self.available_tools.values())
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取指定工具的 input_schema"""
        tool = self.available_tools.get(tool_name)
        if tool:
            return tool.get('input_schema')
        return None
    
    def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """验证工具参数是否符合 schema 定义
        
        Returns:
            (is_valid, error_message)
        """
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return False, f"工具 '{tool_name}' 不存在"
        
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        # 检查必需参数
        for req_param in required:
            if req_param not in arguments:
                return False, f"缺少必需参数: {req_param}"
        
        # 检查参数类型（基础验证）
        for param_name, param_value in arguments.items():
            if param_name not in properties:
                # 忽略我们自定义的内部参数
                if param_name == "__post_processing_code__":
                    continue
                logger.warning(f"⚠️ 参数 '{param_name}' 不在 schema 定义中")
                continue
            
            expected_type = properties[param_name].get('type')
            if expected_type:
                actual_type = self._get_json_type(param_value)
                if actual_type != expected_type:
                    return False, f"参数 '{param_name}' 类型错误: 期望 {expected_type}, 实际 {actual_type}"
        
        return True, None
    
    def _get_json_type(self, value: Any) -> str:
        """获取 Python 值对应的 JSON Schema 类型"""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        elif value is None:
            return "null"
        else:
            return "unknown"
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], 
                        post_processing_code: Optional[str] = None,
                        validate: bool = True) -> Dict[str, Any]:
        """调用MCP工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            post_processing_code: (新增) 在服务端执行的后处理代码
            validate: 是否在调用前验证参数
        """
        try:
            # ⭐ 新增逻辑：将后处理代码添加到参数中
            if post_processing_code:
                # 使用特殊名称以避免与工具参数冲突
                arguments["__post_processing_code__"] = post_processing_code

            # 可选的参数验证
            if validate:
                is_valid, error_msg = self.validate_arguments(tool_name, arguments)
                if not is_valid:
                    logger.error(f"❌ 参数验证失败: {error_msg}")
                    return {
                        "success": False,
                        "error": f"参数验证失败: {error_msg}",
                        "summary": "参数格式不正确"
                    }
            
            # 调用工具
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            # 解析结果
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return json.loads(content.text)
            
            return {"error": "无法解析工具调用结果"}
            
        except Exception as e:
            logger.error(f"❌ 工具调用失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "工具调用异常"
            }
    
    def print_tool_info(self, tool_name: str):
        """打印工具的详细信息（包括 schema）"""
        tool = self.available_tools.get(tool_name)
        if not tool:
            print(f"❌ 工具 '{tool_name}' 不存在")
            return
        
        print(f"\n{'='*60}")
        print(f"工具名称: {tool['name']}")
        print(f"描述: {tool['description'][:100]}...")
        print(f"\n输入参数 Schema:")
        print(json.dumps(tool['input_schema'], indent=2, ensure_ascii=False))
        print(f"{'='*60}\n")
    
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
    success = await client.connect()
    if not success:
        raise ConnectionError("无法连接到 MCP 服务器")
    return client


# 使用示例
async def example_usage():
    """使用示例"""
    client = await create_client_and_connect()
    
    try:
        # 1. 查看所有工具
        tools = await client.list_tools()
        print(f"可用工具数量: {len(tools)}")
        
        # 2. 查看特定工具的 schema
        client.print_tool_info("initialize_graph")
        
        # 3. 验证参数（不调用）
        test_args = {
            "vertices": [{"vid": "A", "properties": {}}],
            "edges": [{"src": "A", "dst": "B"}]
        }
        is_valid, error = client.validate_arguments("initialize_graph", test_args)
        print(f"参数验证: {'✅ 通过' if is_valid else f'❌ 失败 - {error}'}")
        
        # 4. 调用工具（不带后处理）
        result = await client.call_tool("get_graph_info", {}, validate=True)
        print(f"无后处理结果: {result}")

        # 5. 调用工具（带后处理代码）
        # 假设我们已经初始化了图，现在调用pagerank并要求返回Top 3
        pagerank_args = {"alpha": 0.85}
        post_process_code = "def process(data):\n    return dict(sorted(data.items(), key=lambda item: item[1], reverse=True)[:3])"
        
        # 注意：此示例直接调用，可能因未初始化图而出错，仅为演示方法调用
        print("\n演示带后处理代码的调用（可能因图未初始化而失败）:")
        processed_result = await client.call_tool(
            "run_pagerank", 
            pagerank_args,
            post_processing_code=post_process_code
        )
        print(f"带后处理结果: {processed_result}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    # 提示：直接运行此文件中的 example_usage 可能会因为找不到 mcp_server.py 或依赖的图引擎而出错
    # 此文件主要作为模块被 SmartQA 系统导入使用
    # asyncio.run(example_usage())
    print("GraphMCPClient (客户端) 代码已更新。")

