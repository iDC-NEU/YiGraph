"""
MCP Client - 修复版
关键修复:
1. 移除不必要的 G 参数过滤
2. 简化参数验证逻辑
3. 增强 CSV 数据加载时的类型处理
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
    """图计算MCP客户端"""
    
    def __init__(self, server_command: str = "python", server_args: List[str] = None):
        self.server_command = server_command
        self.server_args = server_args or ["graphllm/mcp_server.py"]
        self.session = None
        self.available_tools = {}
        
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
            self.output_schemas_cache = self._load_output_schemas()
            await self.session.__aenter__()
            await self.session.initialize()
            
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
    
    def _load_output_schemas(self) -> Dict[str, Dict[str, Any]]:
        """从文件加载预生成的 output schemas"""
        from pathlib import Path
        import json
        
        # ✅ 修复：使用正确的相对路径
        # 假设 mcp_client.py 和 dynamic_tools_registrar.py 在同一目录
        schema_file = Path(__file__).parent / "generated_output_schemas.json"
        
        if schema_file.exists():
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schemas = json.load(f)
                logger.info(f"✅ 已加载 {len(schemas)} 个工具的 output schemas")
                logger.info(f"   文件位置: {schema_file.absolute()}")
                return schemas
            except Exception as e:
                logger.warning(f"⚠️ 加载 output schemas 失败: {e}")
        else:
            logger.warning(f"⚠️ 未找到 output schemas 文件: {schema_file}")
            logger.info("   提示：先运行 MCP Server 以生成 schemas")
        
        return {}
    async def _load_available_tools(self):
        """加载并缓存可用工具列表"""
        try:
            tools_response = await self.session.list_tools()
            for tool in tools_response.tools:
                tool_name = tool.name
                
                # ✅ 从预生成的文件中获取 output_schema
                output_schema = self.output_schemas_cache.get(tool_name)
                
                self.available_tools[tool_name] = {
                    'name': tool_name,
                    'description': tool.description,
                    'input_schema': tool.inputSchema,
                    'output_schema': output_schema  # ✅ 使用预加载的 schema
                }
            
            logger.info(f"✅ 已加载 {len(self.available_tools)} 个可用工具")
        except Exception as e:
            logger.error(f"❌ 加载工具列表失败: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用的工具"""
        if not self.available_tools:
            await self._load_available_tools()
        return list(self.available_tools.values())
    
    def get_tool_schema(self, tool_name: str, schema_type: str = 'input') -> Optional[Dict[str, Any]]:
        """获取指定工具的 schema"""
        tool = self.available_tools.get(tool_name)
        if not tool:
            return None
        
        if schema_type == 'output':
            output_schema = tool.get('output_schema')
            
            # ✅ 如果没有预加载的 schema，返回默认结构
            if not output_schema:
                logger.debug(f"⚠️ 工具 '{tool_name}' 没有预定义的 output_schema,使用默认")
                return self._get_default_output_schema()
            
            return output_schema
        
        return tool.get('input_schema')
    
    def _get_default_output_schema(self) -> Dict[str, Any]:
        """返回默认的 output schema"""
        return {
            "type": "object",
            "properties": {
                "algorithm": {"type": "string"},
                "success": {"type": "boolean"},
                "result": {
                    "description": "Algorithm result (type varies)"
                },
                "error": {"type": ["string", "null"]},
                "summary": {"type": ["string", "null"]}
            },
            "required": ["algorithm", "success"]
        }
    
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
        
        # 检查额外参数（允许 __post_processing_code__）
        for param_name in arguments.keys():
            if param_name not in properties and param_name != "__post_processing_code__":
                logger.warning(f"⚠️ 参数 '{param_name}' 不在 schema 定义中，但将尝试传递")
        
        return True, None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], 
                        post_processing_code: Optional[str] = None,
                        validate: bool = True) -> Dict[str, Any]:
        """调用MCP工具"""
        try:
            if post_processing_code:
                arguments["__post_processing_code__"] = post_processing_code

            if validate:
                is_valid, error_msg = self.validate_arguments(tool_name, arguments)
                if not is_valid:
                    logger.error(f"❌ 参数验证失败: {error_msg}")
                    return {
                        "success": False,
                        "error": f"参数验证失败: {error_msg}",
                        "summary": "参数格式不正确"
                    }
            
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError as e:
                        raw_content = content.text if content.text else "Empty response"
                        logger.error(f"❌ JSON 解析失败: {e} - 原始响应: {raw_content}")
                        return {
                            "success": False,
                            "error": raw_content,
                            "summary": "服务器返回非 JSON 响应"
                        }
            
            return {"error": "无法解析工具调用结果"}
            
        except Exception as e:
            logger.error(f"❌ 工具调用失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "工具调用异常"
            }
    
    def print_tool_info(self, tool_name: str):
        """打印工具的详细信息"""
        tool = self.available_tools.get(tool_name)
        if not tool:
            print(f"❌ 工具 '{tool_name}' 不存在")
            return
        
        print(f"\n{'='*60}")
        print(f"工具名称: {tool['name']}")
        print(f"描述: {tool['description']}")
        
        print(f"\n--- 输入参数 Schema ---")
        input_schema = tool.get('input_schema', {})
        print(json.dumps(input_schema, indent=2, ensure_ascii=False))
        
        print(f"\n--- 返回结构 Schema ---")
        if tool.get('output_schema'):
            print(json.dumps(tool.get('output_schema'), indent=2, ensure_ascii=False))
        else:
            print("未提供 Output Schema。")
            
        print(f"{'='*60}\n")
    
    def load_data_from_csv(self, accounts_file: str, transactions_file: str) -> tuple:
        """⭐ 从CSV文件加载图数据（增强类型处理）"""
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
                    properties = {}
                    for k, v in row.items():
                        if k not in ['orig_acct', 'bene_acct']:
                            # ⭐ 尝试转换数值类型
                            try:
                                if '.' in v:
                                    properties[k] = float(v)
                                else:
                                    properties[k] = int(v)
                            except (ValueError, TypeError):
                                properties[k] = v
                    
                    edge = {
                        'src': row['orig_acct'],
                        'dst': row['bene_acct'],
                        'rank': 0,
                        'properties': properties
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


if __name__ == "__main__":
    print("GraphMCPClient (修复版本) - 已修复类型转换问题")