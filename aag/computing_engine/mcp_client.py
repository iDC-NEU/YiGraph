"""
MCP Client - Schema 文件加载版
关键改进:
1. ✅ 从文件加载 input_schema 和 output_schema
2. ✅ 移除运行时 schema 生成逻辑
3. ✅ 优化参数验证性能
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
    """图计算MCP客户端 - Schema 文件版"""

    def __init__(self, server_command: str = "python", server_args: List[str] = None, schema_dir: Optional[str] = None):
        self.server_command = server_command
        self.server_args = server_args or ["networkx_server/mcp_server.py"]
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.session = None
        self.available_tools = {}
        
        # ✅ 新增：预加载 schemas
        self.input_schemas_cache = {}
        self.output_schemas_cache = {}
        
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
            
            # ✅ 先加载 schemas 缓存
            self.input_schemas_cache = self._load_schemas("input")
            self.output_schemas_cache = self._load_schemas("output")
            
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
    
    def _load_schemas(self, schema_type: str) -> Dict[str, Dict[str, Any]]:
        """
        ✅ 统一的 schema 加载函数
        
        Args:
            schema_type: "input" 或 "output"
            
        Returns:
            工具名称到 schema 的映射字典
        """

        if self.schema_dir:
            schema_file = self.schema_dir / f"generated_{schema_type}_schemas.json"
        elif self.server_args:
            # 如果是 -m 参数，server_args = ["-m", "module.name"]
            if len(self.server_args) >= 2 and self.server_args[0] == "-m":
                # 从模块名获取路径
                # 例如: "aag.computing_engine.networkx_server.mcp_server"
                # 转换为: aag/computing_engine/networkx_server/
                module_name = self.server_args[1]
                module_path = module_name.replace('.', '/')
                # 去掉最后的文件名
                module_dir = '/'.join(module_path.split('/')[:-1])
                # 使用相对于项目根目录的路径
                schema_file = Path(__file__).parent.parent.parent / module_dir / f"generated_{schema_type}_schemas.json"
            else:
                # 传统方式：直接使用文件路径
                schema_file = Path(self.server_args[0]).parent / f"generated_{schema_type}_schemas.json"
        else:
            schema_file = Path(__file__).parent / f"generated_{schema_type}_schemas.json"
        
        if schema_file.exists():
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schemas = json.load(f)
                logger.info(f"✅ 已加载 {len(schemas)} 个工具的 {schema_type} schemas")
                logger.info(f"   文件位置: {schema_file.absolute()}")
                return schemas
            except Exception as e:
                logger.warning(f"⚠️ 加载 {schema_type} schemas 失败: {e}")
        else:
            logger.warning(f"⚠️ 未找到 {schema_type} schemas 文件: {schema_file}")
            logger.info("   提示：先运行 MCP Server 以生成 schemas")
        
        return {}
    
    async def _load_available_tools(self):
        """加载并缓存可用工具列表（使用预加载的 schemas）"""
        try:
            tools_response = await self.session.list_tools()
            for tool in tools_response.tools:
                tool_name = tool.name
                
                # ✅ 优先使用预加载的 schemas
                input_schema = self.input_schemas_cache.get(tool_name)
                output_schema = self.output_schemas_cache.get(tool_name)
                
                # 如果缓存中没有,使用服务器返回的 schema (兜底)
                if not input_schema:
                    input_schema = tool.inputSchema
                    logger.debug(f"⚠️ 工具 '{tool_name}' 使用服务器返回的 input_schema")
                
                self.available_tools[tool_name] = {
                    'name': tool_name,
                    'description': tool.description,
                    'input_schema': input_schema,
                    'output_schema': output_schema
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
        """
        获取指定工具的 schema
        
        Args:
            tool_name: 工具名称
            schema_type: "input" 或 "output"
        """
        tool = self.available_tools.get(tool_name)
        if not tool:
            logger.warning(f"⚠️ 工具 '{tool_name}' 不存在")
            return None
        
        schema_key = f"{schema_type}_schema"
        schema = tool.get(schema_key)
        
        if not schema:
            logger.debug(f"⚠️ 工具 '{tool_name}' 没有 {schema_type}_schema")
            if schema_type == 'output':
                return self._get_default_output_schema()
        
        return schema
    
    def _get_default_output_schema(self) -> Dict[str, Any]:
        """返回默认的 output schema"""
        return {
            "type": "object",
            "properties": {
                "algorithm": {"type": "string"},
                "success": {"type": "boolean"},
                "result": {"description": "Algorithm result (type varies)"},
                "error": {"type": ["string", "null"]},
                "summary": {"type": ["string", "null"]}
            },
            "required": ["algorithm", "success"]
        }
    
    def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """验证工具参数是否符合 schema 定义"""
        schema = self.get_tool_schema(tool_name, 'input')

        if not schema:
            return False, f"工具 '{tool_name}' 不存在或缺少 input schema"
        
        properties =schema.get('parameters') or schema.get('properties') or {}
        required = schema.get('required', [])

        # 检查必需参数
        for req_param in required:
            if req_param not in arguments:
                return False, f"缺少必需参数: {req_param}"
        
        # 检查额外参数（允许 __post_processing_code__）
        for param_name in arguments.keys():
            if param_name not in properties and param_name != "__post_processing_code__":
                logger.warning(f"⚠️ 参数 '{param_name}' 不在 schema 定义中")
        
        return True, None
    
    # [其他方法保持不变: call_tool, print_tool_info, load_data_from_csv...]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], post_processing_code: Optional[str] = None, validate: bool = True) -> Dict[str, Any]: 
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
        """打印工具的详细信息（增强版）"""
        tool = self.available_tools.get(tool_name)
        if not tool:
            print(f"❌ 工具 '{tool_name}' 不存在")
            return
        
        print(f"\n{'='*60}")
        print(f"工具名称: {tool['name']}")
        print(f"描述: {tool['description']}")
        
        print(f"\n--- 输入参数 Schema ---")
        input_schema = tool.get('input_schema')
        if input_schema:
            print(json.dumps(input_schema, indent=2, ensure_ascii=False))
        else:
            print("⚠️ 未提供 Input Schema")
        
        print(f"\n--- 返回结构 Schema ---")
        output_schema = tool.get('output_schema')
        if output_schema:
            print(json.dumps(output_schema, indent=2, ensure_ascii=False))
        else:
            print("⚠️ 未提供 Output Schema")
            
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