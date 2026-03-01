"""
MCP Client – schema loaded from files.
- Load input_schema and output_schema from files.
- No runtime schema generation.
- Optimized argument validation.
"""

# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'

import logging
import csv
import asyncio
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from aag.computing_engine.code_executor import DynamicCodeExecutor
from aag.expert_search_engine.database.datatype import GraphData

logger = logging.getLogger(__name__)


class GraphMCPClient:
    """Graph computation MCP client with schema loaded from files."""

    def __init__(self, server_command: str = "python", server_args: List[str] = None, schema_dir: Optional[str] = None):
        self.server_command = server_command
        self.server_args = server_args or ["networkx_server/mcp_server.py"]
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.session = None
        self.available_tools = {}
        self.input_schemas_cache = {}
        self.output_schemas_cache = {}
        self.code_executor = DynamicCodeExecutor(
                timeout=120,
                auto_install=True
            )
        
    async def connect(self):
        """Connect to the MCP server."""
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
            await self._load_available_tools()
            
            logger.info("✅ Connected to MCP server")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if hasattr(self, 'stdio_client'):
                await self.stdio_client.__aexit__(None, None, None)
            logger.info("👋 Disconnected from MCP server")
        except Exception as e:
            logger.error(f"❌ Error while disconnecting: {e}")
    
    def _load_schemas(self, schema_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Load schemas from file (input or output).

        Args:
            schema_type: "input" or "output"

        Returns:
            Map of tool name -> schema.
        """
        if self.schema_dir:
            schema_file = self.schema_dir / f"generated_{schema_type}_schemas.json"
        elif self.server_args:
            if len(self.server_args) >= 2 and self.server_args[0] == "-m":
                module_name = self.server_args[1]
                module_path = module_name.replace('.', '/')
                module_dir = '/'.join(module_path.split('/')[:-1])
                schema_file = Path(__file__).parent.parent.parent / module_dir / f"generated_{schema_type}_schemas.json"
            else:
                schema_file = Path(self.server_args[0]).parent / f"generated_{schema_type}_schemas.json"
        else:
            schema_file = Path(__file__).parent / f"generated_{schema_type}_schemas.json"
        
        if schema_file.exists():
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schemas = json.load(f)
                logger.info(f"✅ Loaded {schema_type} schemas for {len(schemas)} tools")
                logger.info(f"   File: {schema_file.absolute()}")
                return schemas
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {schema_type} schemas: {e}")
        else:
            logger.warning(f"⚠️ {schema_type} schemas file not found: {schema_file}")
            logger.info("   Hint: run MCP Server first to generate schemas")
        
        return {}
    
    async def _load_available_tools(self):
        """Load and cache available tools (using preloaded schemas)."""
        try:
            tools_response = await self.session.list_tools()
            self.input_schemas_cache = self._load_schemas("input")
            self.output_schemas_cache = self._load_schemas("output")
            
            for tool in tools_response.tools:
                tool_name = tool.name
                input_schema = self.input_schemas_cache.get(tool_name)
                output_schema = self.output_schemas_cache.get(tool_name)
                if not input_schema:
                    input_schema = tool.inputSchema
                    logger.debug(f"⚠️ Tool '{tool_name}' using server-provided input_schema")
                
                self.available_tools[tool_name] = {
                    'name': tool_name,
                    'description': tool.description,
                    'input_schema': input_schema,
                    'output_schema': output_schema
                }
            
            logger.info(f"✅ Loaded {len(self.available_tools)} available tools")
        except Exception as e:
            logger.error(f"❌ Failed to load tool list: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        if not self.available_tools:
            await self._load_available_tools()
        return list(self.available_tools.values())
    
    def get_tool_schema(self, tool_name: str, schema_type: str = 'input') -> Optional[Dict[str, Any]]:
        """
        Get schema for a tool.

        Args:
            tool_name: Tool name.
            schema_type: "input" or "output".
        """
        tool = self.available_tools.get(tool_name)
        if not tool:
            logger.warning(f"⚠️ Tool '{tool_name}' not found")
            return None
        
        schema_key = f"{schema_type}_schema"
        schema = tool.get(schema_key)
        
        if not schema:
            logger.debug(f"⚠️ Tool '{tool_name}' has no {schema_type}_schema")
            if schema_type == 'output':
                return self._get_default_output_schema()
        
        return schema
    
    def _get_default_output_schema(self) -> Dict[str, Any]:
        """Return default output schema."""
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
        """Validate tool arguments against schema."""
        schema = self.get_tool_schema(tool_name, 'input')

        if not schema:
            return False, f"Tool '{tool_name}' not found or missing input schema"
        
        properties = schema.get('parameters') or schema.get('properties') or {}
        required = schema.get('required', [])

        for req_param in required:
            if req_param not in arguments:
                return False, f"Missing required parameter: {req_param}"
        
        for param_name in arguments.keys():
            if param_name not in properties and param_name != "__post_processing_code__":
                logger.warning(f"⚠️ Parameter '{param_name}' not in schema")
        
        return True, None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], post_processing_code: Optional[str] = None, global_graph: Optional[GraphData] = None, validate: bool = True) -> Dict[str, Any]: 
        """
        Call MCP tool (supports optional post-processing code)
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            post_processing_code: Optional post-processing code
            validate: Whether to validate arguments
        
        Returns:
            Tool execution result dictionary
        """
        try:
            # ========== New: Argument cleaning ==========
            cleaned_arguments = {}
            for key, value in arguments.items():
                # Convert empty containers to None (avoid empty container traps in libraries like NetworkX)
                if isinstance(value, (dict, list)) and len(value) == 0:
                    cleaned_arguments[key] = None
                    logger.warning(f"⚠️ Parameter '{key}' is an empty container, converted to None")
                else:
                    cleaned_arguments[key] = value
            # ========== Argument validation ==========
            if validate:
                is_valid, error_msg = self.validate_arguments(tool_name, arguments)
                if not is_valid:
                    logger.error(f"❌ Argument validation failed: {error_msg}")
                    return {
                        "success": False,
                        "error": f"Argument validation failed: {error_msg}",
                        "summary": "Invalid argument format"
                    }
            
            # ========== Step 1: Call original tool ==========
            logger.info(f"📤 Calling tool {tool_name}...")
            result = await self.session.call_tool(tool_name, arguments=arguments)
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    try:
                        original_response = json.loads(content.text)
                        # logger.info(f"✅ Tool execution completed: {original_response.get('summary', '')}")
                    except json.JSONDecodeError as e:
                        raw_content = content.text if content.text else "Empty response"
                        logger.error(f"JSON parsing failed: {e} - Raw response: {raw_content}")
                        return {
                            "success": False, 
                            "error": raw_content,
                            "summary": "Server returned non-JSON response"
                        }
                else:
                    return {"success": False, "error": "Tool call failed: Unable to parse tool call result"}
            else:
                return {"success": False, "error": "Tool call failed: Unable to parse tool call result"}

            # ========== Step 2: If post-processing code exists, call post-processing tool ==========
            if post_processing_code:
                logger.info(f"📤 Applying post-processing code...")
                try:
                    # ✅ Key change: Use local executor for processing
                    processed_data = self.code_executor.execute(
                        post_processing_code, 
                        original_response.get("result"),  # Only pass the result part
                        global_graph=global_graph
                    )
                    
                    # Update return result
                    original_response["result"] = processed_data
                    original_response["summary"] = original_response.get("summary", "") + " (Local post-processing applied)"
                    
                    logger.info(f"✅ Post-processing execution completed")
                    # logger.info(f"Key results extracted: {processed_data}")
                    logger.info(f"post_processing_status:{original_response}")
                    return original_response
                    
                except Exception as post_error:
                    logger.error(f"Post-processing execution failed: {post_error}")
                    # logger.warning("⚠️ Post-processing failed, returning original result")
                    # return original_response
                    return {"success": False, "error": f"❌ Post-processing code execution failed:  {post_error}"}
            
            return original_response
        
        except Exception as e:
            logger.error(f"❌ Tool call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "Tool call error"
            }
            
    def print_tool_info(self, tool_name: str):
        """Print detailed tool info."""
        tool = self.available_tools.get(tool_name)
        if not tool:
            print(f"❌ Tool '{tool_name}' not found")
            return
        
        print(f"\n{'='*60}")
        print(f"Tool: {tool['name']}")
        print(f"Description: {tool['description']}")
        
        print(f"\n--- Input Schema ---")
        input_schema = tool.get('input_schema')
        if input_schema:
            print(json.dumps(input_schema, indent=2, ensure_ascii=False))
        else:
            print("⚠️ No Input Schema")
        
        print(f"\n--- Output Schema ---")
        output_schema = tool.get('output_schema')
        if output_schema:
            print(json.dumps(output_schema, indent=2, ensure_ascii=False))
        else:
            print("⚠️ No Output Schema")
            
        print(f"{'='*60}\n")
    

    def load_data_from_csv(self, accounts_file: str, transactions_file: str) -> tuple:
        """Load graph data from CSV files (with type handling)."""
        vertices = []
        edges = []
        
        try:
            logger.info(f"📖 Reading accounts: {accounts_file}")
            with open(accounts_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vertex = {
                        'vid': row['acct_id'],
                        'properties': {k: v for k, v in row.items() if k != 'acct_id'}
                    }
                    vertices.append(vertex)
            
            logger.info(f"📖 Reading transactions: {transactions_file}")
            with open(transactions_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    properties = {}
                    for k, v in row.items():
                        if k not in ['orig_acct', 'bene_acct']:
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
            
            logger.info(f"✅ Loaded {len(vertices)} vertices and {len(edges)} edges")
            return vertices, edges
            
        except Exception as e:
            logger.error(f"❌ CSV data load failed: {e}")
            raise


async def create_client_and_connect() -> GraphMCPClient:
    """Create and connect the client."""
    client = GraphMCPClient()
    success = await client.connect()
    if not success:
        raise ConnectionError("Failed to connect to MCP server")
    return client


if __name__ == "__main__":
    print("GraphMCPClient – schema file version")