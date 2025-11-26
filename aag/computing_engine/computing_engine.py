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

    async def get_algorithm_description(self, algo_name: str) -> tuple[str, dict]:
        """
        获取指定算法的工具描述信息（包含 input/output schema）
        """
        engine = self._resolve_engine(algo_name)
        client = self.clients.get(engine)
        if not client:
            return f"⚠️ Engine '{engine}' not connected.\n"

        # ✅ 使用新方法解析工具名
        tool_name = self._resolve_tool_name(algo_name, engine)
        
        tool_info = client.available_tools.get(tool_name)
        if not tool_info:
            return f"⚠️ Tool '{tool_name}' not found in engine '{engine}'.\n"

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

    def execute_code(self, code: str, data: Any, global_graph: Optional[GraphData] = None, fallback_to_direct_exec: bool = True) -> Any:
        """
        执行动态生成的代码（数值分析、后处理等）
        
        Args:
            code: 要执行的 Python 代码字符串
            data: 传递给代码的数据（可以是 dict 或其他类型）
            fallback_to_direct_exec: 如果代码不是 process 函数格式，是否直接执行
            
        Returns:
            代码执行结果
            
        Raises:
            RuntimeError: 代码执行失败
        """
        try:
            # 尝试使用 execute 方法（期望代码定义 process 函数）
            return self.code_executor.execute(code, data, global_graph=global_graph)
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

    async def shutdown(self):
        """断开所有引擎连接"""
        for name, client in self.clients.items():
            await client.disconnect()
        logger.info("👋 所有计算引擎已断开连接")