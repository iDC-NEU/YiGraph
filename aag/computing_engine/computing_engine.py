import importlib
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from aag.utils.path_utils import DEFAULT_CONFIG_SERVER_PATH

from aag.computing_engine.mcp_client import GraphMCPClient
from aag.computing_engine.code_executor import DynamicCodeExecutor
from aag.expert_search_engine.database.datatype import GraphData
# add gjq: 导入图查询相关类
from aag.computing_engine.graph_query.nl_query_engine import NaturalLanguageQueryEngine
from aag.computing_engine.graph_query.graph_query import Neo4jGraphClient, Neo4jConfig


logger = logging.getLogger(__name__)


class ComputingEngine:
    """
    🔧 统一的计算调度引擎层
    -------------------------------------
    - 管理多个计算引擎（NetworkX / PyG / Nebula）
    - 向 scheduler 提供统一的执行接口
    - 封装 MCP 客户端通信逻辑
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_SERVER_PATH):
        self.config_path = Path(config_path)
        self.clients: Dict[str, GraphMCPClient] = {}
        self.engine_supported_algorithms = {}   #  algorithm_type → engine_name
        self.algorithm_tool_mapping = {}        #   algorithm_name → tool_name
        self.parameter_modules = {}
        self._initialized = False
        self.code_executor = DynamicCodeExecutor(timeout=120, auto_install=True)
        # add gjq: 添加图查询引擎字段
        self.nl_query_engine: Optional[NaturalLanguageQueryEngine] = None
        self.neo4j_config: Optional[Dict[str, Any]] = None
        self.reasoner = None  # 将在 initialize_graph_query_engine 中设置

    async def initialize(self):
        """初始化：加载配置并连接所有 MCP servers"""
        config = self._load_config()
        # 构建算法到引擎映射表
        self.engine_supported_algorithms = config.get("engine_supported_algorithms", {})
        
        # 解析算法到工具名的映射
        self._parse_algorithm_tool_mapping(config.get("engine_supported_algorithms", {}))
        
        for engine_name, server in config.get("servers", {}).items():
            client = GraphMCPClient(
                server_command=server.get("command", "python"),
                server_args=server.get("args", [])
            )
            ok = await client.connect()
            if ok:
                self.clients[engine_name] = client
                logger.info(f"✅ Engine '{engine_name}' connected")
            else:
                logger.warning(f"⚠️ Engine '{engine_name}' failed to connect")

            # 自动加载 annotate_utils.py
            try:
                module_path = f"aag.computing_engine.{engine_name}_server.parameter_utils"
                module = importlib.import_module(module_path)
                self.parameter_modules[engine_name] = module
                logger.info(f"🧩 Loaded parameter module: {module_path}")
            except ImportError as e:
                logger.warning(f"⚠️ No parameter_utils.py found for engine '{engine_name}' ({e})")
        self._initialized = True
        logger.info(f"✅ Loaded {len(self.clients)} computing engines")
        # logger.info(f"🧠 Annotation modules loaded: {list(self.parameter_modules.keys())}")


    def _load_config(self) -> dict:
        """从配置文件加载引擎定义"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _parse_algorithm_tool_mapping(self, engine_algorithms: dict):
        """
        engine_supported_algorithms:
          networkx:
            bfs:
              - tool: bfs_edges
        """
        
        for engine_name, algorithms in engine_algorithms.items():
            engine_mapping = {}
            
            for algo_name, tool_configs in algorithms.items():
                # tool_configs 是一个列表
                # 目前只取第一个，未来可以扩展支持多候选
                if isinstance(tool_configs, list) and len(tool_configs) > 0:
                    tool_config = tool_configs[0]
                    tool_base_name = tool_config.get("tool")
                    
                    # ✅ 添加空值检查
                    if tool_base_name and isinstance(tool_base_name, str):
                        engine_mapping[algo_name] = f"run_{tool_base_name}"
                    else:
                        logger.warning(f"⚠️ Invalid tool name for algorithm '{algo_name}' in engine '{engine_name}': {tool_base_name}")
            
            self.algorithm_tool_mapping[engine_name] = engine_mapping
        
        logger.info(f"✅ Parsed {sum(len(m) for m in self.algorithm_tool_mapping.values())} algorithm-tool mappings")
        logger.debug(f"Algorithm-tool mapping: {self.algorithm_tool_mapping}")

    def _resolve_engine(self, algo_name: str) -> str:
        """根据算法名查找所属引擎"""
        algo_name = algo_name.lower()
        for engine, algorithms in self.engine_supported_algorithms.items():
            # 适配新格式：检查算法名是否在字典的 keys 中
            if isinstance(algorithms, dict):
                if algo_name in algorithms:
                    return engine

        
        logger.warning(f"⚠️ Algorithm '{algo_name}' not found in config; fallback to 'networkx'")
        return "networkx"  # 默认引擎

    def _resolve_tool_name(self, algo_name: str, engine_name: str) -> str:
        """
        根据算法名和引擎名解析实际工具名
        
        Args:
            algo_name: 算法名（如 "bfs"）
            engine_name: 引擎名（如 "networkx"）
        
        Returns:
            工具名（如 "run_bfs_edges"）
        
        Raises:
            ValueError: 如果无法解析工具名
        """
        # 优先使用映射表
        engine_mapping = self.algorithm_tool_mapping.get(engine_name, {})
        if algo_name in engine_mapping:
            return engine_mapping[algo_name]
        
        logger.error(f"❌ Cannot resolve tool name for algorithm '{algo_name}' in engine '{engine_name}'")
        raise ValueError(f"Algorithm '{algo_name}' has no tool mapping in engine '{engine_name}'. "
                         f"Please check config_servers.yaml")

    async def run_algorithm(self, algo_name: str, parameters: Dict[str, Any], post_processing_code: Optional[str] = None, global_graph: Optional[GraphData] = None) -> Dict[str, Any]:
        """
        执行指定算法
        """
        try:
            engine_name = self._resolve_engine(algo_name)
            if engine_name not in self.clients:
                return {"success": False, "error": f"Engine '{engine_name}' not connected"}

            client = self.clients[engine_name]
            
            # ✅ 使用新方法解析工具名
            tool_name = self._resolve_tool_name(algo_name, engine_name)
            
            prepared_params = parameters or {}
            if self._should_normalize(engine_name, tool_name):
                prepared_params = self._normalize_parameters(engine_name, tool_name, prepared_params)
            
            logger.info(f"🚀 Running '{algo_name}' (tool: {tool_name}) on engine [{engine_name}]")
            result = await client.call_tool(tool_name, prepared_params, post_processing_code, global_graph)
            return result

        except Exception as e:
            logger.error(f"❌ Algorithm '{algo_name}' failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_algorithm_description(self, algo_name: str) -> tuple[str, Optional[dict]]:
        """
        获取指定算法的工具描述信息（包含 input/output schema）
        """
        engine = self._resolve_engine(algo_name)
        client = self.clients.get(engine)
        if not client:
            return f"⚠️ Engine '{engine}' not connected.\n", None

        # ✅ 使用新方法解析工具名
        tool_name = self._resolve_tool_name(algo_name, engine)
        
        tool_info = client.available_tools.get(tool_name)
        if not tool_info:
            return f"⚠️ Tool '{tool_name}' not found in engine '{engine}'.\n", None

        # 取出 schema
        input_schema = tool_info.get("input_schema", {})
        output_schema = tool_info.get("output_schema", {})

        # 调用引擎自定义的 schema 注释逻辑
        annotated_input = self._annotate_schema(input_schema, engine)

        # 构建格式化文档
        doc = (
            f"### `{tool_name}` (Engine: {engine})\n"
            f"**Description:** {tool_info.get('description', 'No description')}\n\n"
            f"**Input Parameters** *(with system-injected annotations)*:\n"
            f"```json\n{json.dumps(annotated_input, indent=2, ensure_ascii=False)}\n```\n\n"
            f"**Output Structure** *(data received by your `process(data)` in the `result` field)*:\n"
            f"```json\n{json.dumps(output_schema, indent=2, ensure_ascii=False)}\n```\n"
            f"{'-'*50}\n"
        )

        tool_metadata = {
            "name": tool_name,
            "engine": engine,
            "description": tool_info.get("description", "No description"),
            "input_params": annotated_input,
            "output_params": output_schema,
        }
        logger.debug(f"🧩 Generated description for '{tool_name}':\n{doc}")
        return doc, tool_metadata

    def _annotate_schema(self, input_schema: Dict, engine_name: str) -> Dict:
        """
            调度不同计算引擎的 schema 注释函数。
            每个引擎在各自 server 模块里定义 annotate_schema()。
        """
        module = self.parameter_modules.get(engine_name)
        if not module:
            logger.debug(f"⚠️ No annotation module for engine '{engine_name}'")
            return input_schema  # 未注册引擎，直接返回原 schema

        annotate_func = getattr(module, "annotate_schema", None)

        if callable(annotate_func):
            try:
                return annotate_func(input_schema)
            except Exception as e:
                logger.error(f"❌ Error running annotate_schema() for '{engine_name}': {e}")
        return input_schema
    

    def _should_normalize(self, engine_name: str, tool_name: str) -> bool:
        module = self.parameter_modules.get(engine_name)
        if not module:
            return False
        guard = getattr(module, "should_normalize", None)
        if callable(guard):
            return guard(tool_name)
        return hasattr(module, "normalize_parameters")

    def _normalize_parameters(self, engine_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        module = self.parameter_modules.get(engine_name)
        if not module:
            return parameters
        normalize_fn = getattr(module, "normalize_parameters", None)
        if not callable(normalize_fn):
            return parameters

        client = self.clients.get(engine_name)
        tool_info = client.available_tools.get(tool_name) if client else None
        try:
            return normalize_fn(tool_name, dict(parameters or {}), tool_info, logger=logger)
        except Exception as exc:
            logger.warning("⚠️ 参数规范化失败 (%s): %s", tool_name, exc)
            return parameters

    def execute_code(self, code: str, data: Any, global_graph: Optional[GraphData] = None, fallback_to_direct_exec: bool = True, is_numeric_analysis: bool = False) -> Any:
        """
        执行动态生成的代码（数值分析、后处理等）
        
        Args:
            code: 要执行的 Python 代码字符串
            data: 传递给代码的数据（可以是 dict 或其他类型）
            fallback_to_direct_exec: 如果代码不是 process 函数格式，是否直接执行
            is_numeric_analysis: 是否为数值分析场景（True：使用 **data 解包；False：直接传递 data）
            
        Returns:
            代码执行结果
            
        Raises:
            RuntimeError: 代码执行失败
        """
        try:
            return self.code_executor.execute(code, data, global_graph=global_graph, is_numeric_analysis=is_numeric_analysis)
        except (ValueError, AttributeError) as e:
            if not fallback_to_direct_exec:
                raise RuntimeError(f"代码执行失败: {e}")
            
            # 如果代码不是 process 函数格式，直接执行
            if isinstance(data, dict):
                namespace = {**data, "__builtins__": __builtins__}
            else:
                namespace = {"data": data, "__builtins__": __builtins__}
            
            exec(code, namespace)
            # 假设代码会定义一个 result 变量
            return namespace.get("result", data)

    # add gjq: 添加图查询引擎初始化方法
    def initialize_graph_query_engine(self, neo4j_config: Dict[str, Any], reasoner):
        """
        初始化图查询引擎
        
        Args:
            neo4j_config: Neo4j配置字典，包含 enabled, uri, user, password
            reasoner: Reasoner实例，用于LLM调用
        """
        try:
            self.neo4j_config = neo4j_config
            self.reasoner = reasoner
            
            # add gjq: 添加调试日志
            logger.info(f"📝 开始初始化图查询引擎，配置: {neo4j_config}")
            
            # 检查是否启用neo4j
            if not neo4j_config.get("enabled", False):
                logger.info("ℹ NaturalLanguageQueryEngine disabled in config")
                self.nl_query_engine = None
                return
            
            # add gjq: 添加调试日志
            logger.info(f"📝 Neo4j已启用，开始创建连接...")
            
            # 从config中构建Neo4jConfig对象
            config = Neo4jConfig(
                uri=neo4j_config.get("uri", "bolt://localhost:7687"),
                user=neo4j_config.get("user", "neo4j"),
                password=neo4j_config.get("password", "")
            )
            
            # add gjq: 添加调试日志
            logger.info(f"📝 创建Neo4jGraphClient，uri={config.uri}")
            
            # 创建数据库客户端和查询引擎
            db_client = Neo4jGraphClient(config)
            
            # add gjq: 添加调试日志
            logger.info(f"📝 创建NaturalLanguageQueryEngine")
            
            self.nl_query_engine = NaturalLanguageQueryEngine(db_client, reasoner)
            self.nl_query_engine.initialize()
            logger.info("✓ NaturalLanguageQueryEngine initialized in ComputingEngine")
        except Exception as e:
            logger.error(f"✗ NaturalLanguageQueryEngine initialization failed: {e}", exc_info=True)
            # 不抛出异常，允许系统在没有图查询功能的情况下运行
            self.nl_query_engine = None
    
    # add gjq: 添加图查询执行方法
    def execute_graph_query(self, query: str) -> Dict[str, Any]:
        """
        执行图查询
        
        Args:
            query: 用户查询
            
        Returns:
            查询结果字典，包含 success, results, count, query_type 等字段
        """
        if not self.nl_query_engine:
            return {
                "success": False,
                "error": "图查询引擎未初始化，请检查Neo4j配置"
            }
        
        try:
            result = self.nl_query_engine.ask(query)
            return result
        except Exception as e:
            logger.error(f"图查询执行失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"图查询执行失败: {str(e)}"
            }

    async def shutdown(self):
        """断开所有引擎连接"""
        for name, client in self.clients.items():
            await client.disconnect()
        logger.info("👋 所有计算引擎已断开连接")
