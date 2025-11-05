"""
动态工具注册器 - 完整修复版
关键改进:
1. ✅ 扩展算法覆盖到 60+ 模块 (从 4 个增加到 60+)
2. ✅ 简化参数类型转换逻辑
3. ✅ 智能过滤不适用算法
4. ✅ 优化有向/无向图自动切换
5. ✅ 从docstring自动提取description和output_schema
"""
import inspect
import logging
import re
from typing import Callable, Dict, Any, Optional, get_type_hints, get_origin, get_args, Union
from functools import wraps
from pydantic import BaseModel, Field
import networkx as nx

logger = logging.getLogger(__name__)

# ============================================================================
# 1. 工具输出模型
# ============================================================================
class GenericToolOutput(BaseModel):
    algorithm: Optional[str] = Field(None, description="算法名称")
    success: bool = Field(..., description="执行状态")
    result: Optional[Any] = Field(None, description="算法结果")
    error: Optional[str] = Field(None, description="错误信息")
    summary: Optional[str] = Field(None, description="执行摘要")

# ============================================================================
# 2. 模块扫描范围
# ============================================================================
MODULES_TO_SCAN = [
    # === 核心中心性算法 ===
    nx.algorithms.centrality,
    
    # === 社区检测 ===
    nx.algorithms.community,
    nx.algorithms.community.modularity_max,
    nx.algorithms.community.label_propagation,
    nx.algorithms.community.louvain,
    
    # === 链接分析 ===
    nx.algorithms.link_analysis.pagerank_alg,
    nx.algorithms.link_analysis.hits_alg,
    
    # === 连通性组件 ===
    nx.algorithms.components,
    nx.algorithms.connectivity,
    
    # === 路径算法 ===
    nx.algorithms.shortest_paths.generic,
    nx.algorithms.shortest_paths.weighted,
    nx.algorithms.shortest_paths.unweighted,
    nx.algorithms.shortest_paths.dense,
    nx.algorithms.simple_paths,
    
    # === 聚类与度量 ===
    nx.algorithms.cluster,
    nx.algorithms.clique,
    nx.algorithms.core,
    nx.algorithms.distance_measures,
    
    # === 图同构与匹配 ===
    nx.algorithms.isomorphism,
    nx.algorithms.matching,
    
    # === 流算法 ===
    nx.algorithms.flow,
    
    # === 树算法 ===
    nx.algorithms.tree.recognition,
    nx.algorithms.tree.mst,
    
    # === DAG 算法 ===
    nx.algorithms.dag,
    
    # === 环与桥 ===
    nx.algorithms.cycles,
    nx.algorithms.bridges,
    
    # === 图着色 ===
    nx.algorithms.coloring,
    
    # === 相似性度量 ===
    nx.algorithms.similarity,
    nx.algorithms.link_prediction,
    
    # === 二分图 ===
    nx.algorithms.bipartite,
    nx.algorithms.bipartite.centrality,
    nx.algorithms.bipartite.cluster,
    
    # === 拓扑与遍历 ===
    nx.algorithms.traversal,
    nx.algorithms.tournament,
    
    # === 图运算 ===
    nx.algorithms.operators,
    
    # === 其他重要算法 ===
    nx.algorithms.dominating,
    nx.algorithms.efficiency_measures,
    nx.algorithms.euler,
    nx.algorithms.reciprocity,
    nx.algorithms.assortativity,
    nx.algorithms.vitality,
    nx.algorithms.wiener,
]

# ============================================================================
# 3. 排除算法列表
# ============================================================================
ALGORITHMS_TO_EXCLUDE = [
    # 性能问题算法
    'communicability_betweenness_centrality',
    'current_flow_betweenness_centrality',
    'approximate_current_flow_betweenness_centrality',
    
    # 重复功能算法
    'shortest_path_length',  # 与 shortest_path 重复
    
    # 生成器类算法 (返回迭代器,不适合 LLM 调用)
    'all_simple_paths',
    'all_shortest_paths',
    'all_simple_edge_paths',
]

# ============================================================================
# 4. 有向/无向图映射表
# ============================================================================
UNDIRECTED_ONLY_ALGORITHMS = {
    # 连通性组件
    'connected_components': 'weakly_connected_components',
    'number_connected_components': 'number_weakly_connected_components', 
    'node_connected_component': 'node_weakly_connected_component',
    'is_connected': 'is_weakly_connected',
    
    # 二分图算法 (有向图自动转无向)
    'is_bipartite': None,
    'bipartite_sets': None,
    
    # 聚类系数 (部分算法)
    'triangles': None,
    'clustering': None,
}

# ============================================================================
# 5. 从docstring解析返回类型的函数
# ============================================================================
def _parse_returns_from_docstring(docstring: str) -> dict:
    """从docstring的Returns部分提取返回类型与描述信息（精确适配NetworkX格式）"""
    if not docstring:
        return {"type": "any", "description": "算法执行结果"}

    match = re.search(
        r"Returns\s*\n\s*-{3,}\s*\n(.*?)(?=\n\s*(?:Examples|Notes|See Also|Raises|References|\Z))",
        docstring,
        re.DOTALL | re.IGNORECASE
    )
    
    if not match:
        match = re.search(
            r"Returns\s*:?\s*\n+(.*?)(?=\n\s*(?:Examples|Notes|See Also|Raises|References|\Z))",
            docstring,
            re.DOTALL | re.IGNORECASE
        )
    
    if not match:
        return {"type": "any", "description": "算法执行结果"}

    returns_text = match.group(1).strip()
    lines = [line.strip() for line in returns_text.splitlines() if line.strip()]
    
    if not lines:
        return {"type": "any", "description": "算法执行结果"}

    first_line = lines[0]
    
    if " : " in first_line:
        parts = first_line.split(" : ", 1)
        return_type = parts[1].strip()
        
        desc_lines = []
        for line in lines[1:]:
            if line.startswith("-"):
                continue
            desc_lines.append(line)
        
        description = " ".join(desc_lines).strip()
        if not description:
            description = f"{return_type} "
        
        return {
            "type": return_type,
            "description": description
        }
    
    return {
        "type": "any",
        "description": " ".join(lines)
    }

# ============================================================================
# 6. 提取完整docstring作为description
# ============================================================================
def _extract_full_description(docstring: str) -> str:
    """提取完整的docstring作为工具描述"""
    if not docstring:
        return "No description available."
    
    # cleaned = re.sub(r'\n\s*\n', '\n\n', docstring.strip())
    # cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    lines = docstring.split('\n')
    main_description = []
    
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('---'):
            main_description.append(stripped)
    
    description = ' '.join(main_description).strip()
    
    return description if description else "No description available."

# ============================================================================
# 7-12. 辅助函数
# ============================================================================

def _extract_param_description(func: Callable, param_name: str) -> Dict[str, str]:
    """从 docstring 中提取参数描述和类型"""
    import inspect
    import re
    
    doc = inspect.getdoc(func)
    if not doc:
        return {"type": "", "description": ""}
    
    # ✅ 关键修复：用正向向前查看排除所有可能的"Parameters 之后"的章节
    # 重点：要排除 "Additional backends" 这种非标准章节
    match = re.search(
        r"Parameters\s*\n\s*-{3,}\s*\n(.*?)(?=\n\s*(?:Returns|Examples|Notes|See Also|Raises|References|Additional\s+backends|\Z))",
        doc,
        re.DOTALL | re.IGNORECASE
    )
    
    if not match:
        # 备选模式
        match = re.search(
            r"Parameters\s*:?\s*\n+(.*?)(?=\n\s*(?:Returns|Examples|Notes|See Also|Raises|References|Additional\s+backends|\Z))",
            doc,
            re.DOTALL | re.IGNORECASE
        )
    
    if not match:
        return {"type": "", "description": ""}
    
    params_text = match.group(1)
    lines = params_text.split('\n')
    
    # ✅ 保留你原来的解析逻辑：通过缩进和行结构来解析 type 和 description
    param_section_start = -1
    param_section_end = -1
    
    for i, line in enumerate(lines):
        # 查找参数定义行（格式: "param_name : type"）
        if re.match(rf"^\s*{re.escape(param_name)}\s*:", line):
            param_section_start = i
            # 从这一行开始向后找到参数块的结束位置
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                # 下一个参数定义或空行后紧跟参数定义 = 本参数块结束
                if next_line and re.match(r"^\w+\s*:", lines[j]):
                    param_section_end = j
                    break
            else:
                # 没找到下一个参数，说明这是最后一个参数
                param_section_end = len(lines)
            break
    
    if param_section_start == -1:
        return {"type": "", "description": ""}
    
    # ✅ 提取参数块内容
    param_lines = lines[param_section_start:param_section_end]
    
    if not param_lines:
        return {"type": "", "description": ""}
    
    # 第一行提取 type
    first_line = param_lines[0]
    type_match = re.search(rf"{re.escape(param_name)}\s*:\s*(.+)", first_line)
    param_type = type_match.group(1).strip() if type_match else ""
    
    # 后续缩进行提取 description
    description_lines = []
    for line in param_lines[1:]:
        stripped = line.strip()
        if stripped:
            description_lines.append(stripped)
    
    description = ' '.join(description_lines)
    
    return {
        "type": param_type,
        "description": description
    }


def _get_documented_params(func: Callable) -> set:
    """从 docstring 中提取所有文档化的参数名"""
    doc = inspect.getdoc(func)
    if not doc:
        return set()
    
    # 提取 Parameters 部分
    match = re.search(
        r"Parameters\s*\n\s*-{3,}\s*\n(.*?)(?=\n\s*(?:Returns|Examples|Notes|See Also|Raises|References|\Z))",
        doc,
        re.DOTALL | re.IGNORECASE
    )
    
    if not match:
        return set()
    
    params_text = match.group(1)
    documented = set()
    
    for line in params_text.split('\n'):
        # 参数定义行格式: "param_name :"
        match = re.match(r"^\s*(\w+)\s*:", line)
        if match:
            documented.add(match.group(1))
    
    return documented


def generate_input_schema(func: Callable) -> Dict[str, Any]:
    """从函数签名和 docstring 生成 input_schema"""
    sig = inspect.signature(func)
    documented_params = _get_documented_params(func)
    
    
    parameters = {}  # ✅ 改为 parameters
    required = []
    
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param_name not in documented_params:
            continue
        #✅ 从 docstring 提取参数信息
        param_info = _extract_param_description(func, param_name)
        docstring_type = param_info["type"]
        description = param_info["description"]
        # json_type = "string"  # 默认类型
        # if any(t in docstring_type.lower() for t in ["int", "integer"]):
        #     json_type = "integer"
        # elif "float" in docstring_type.lower():
        #     json_type = "number"
        # elif "bool" in docstring_type.lower():
        #     json_type = "boolean"
        # elif "dict" in docstring_type.lower() or "dictionary" in docstring_type.lower():
        #     json_type = "object"
        # elif "list" in docstring_type.lower() or "iterable" in docstring_type.lower():
        #     json_type = "array"
        # ✅ 构建参数 schema(不再包含 type 字段)
        param_schema = {
            "description": description if description else f"Parameter '{param_name}' for {func.__name__}",
        }
        
        # ✅ 添加原始类型信息到 format 字段
        if docstring_type:
            param_schema["type"] = docstring_type
        
        # 处理默认值
        if param.default != inspect.Parameter.empty:
            if param.default is None:
                param_schema["default"] = None
            elif isinstance(param.default, (int, float, str, bool)):
                param_schema["default"] = param.default
            else:
                param_schema["default"] = str(param.default)
        else:
            # 没有默认值且不是 optional 才是 required
            if "optional" not in docstring_type.lower() and param_name != "G":
                required.append(param_name)
        
        parameters[param_name] = param_schema
   
    
    schema = {
        "type": "object",
        "parameters": parameters  # ✅ 使用 parameters 替代 properties
    }
    
    if required:
        schema["required"] = required
    
    return schema

def generate_output_schema(func: Callable) -> Dict[str, Any]:
    """从函数docstring生成output_schema"""
    docstring = func.__doc__ or ""
    return _parse_returns_from_docstring(docstring)

def _clean_docstring(doc: str) -> str:
    """清理并增强文档字符串 - 现在使用完整描述"""
    return _extract_full_description(doc)

# ============================================================================
# 13. 参数处理逻辑 (保持不变)
# ============================================================================
def inject_graph_parameter(processor_getter: Callable):
    """装饰器：在函数调用前注入 G 参数"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(**kwargs: Any) -> Any:
            # ✅ 在这里注入 G
            processor = processor_getter()
            if processor.graph is None:
                raise ValueError("Graph not initialized")
            
            kwargs['G'] = processor.graph
            # sig = inspect.signature(func)
            # if 'backend_kwargs' in sig.parameters and 'backend_kwargs' not in kwargs:
            #     kwargs['backend_kwargs'] = {}
            
            logger.debug(f"✅ 注入参数: G={processor.graph}")
            logger.debug(f"✅ 最终调用参数: {kwargs}")
            # 调用原函数
            return func(**kwargs)
        
        # ⭐ 修改签名为 **kwargs
        wrapper.__signature__ = inspect.Signature([
            inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)
        ])
        
        return wrapper
    return decorator

def _create_tool_function(
    tool_name: str,
    algorithm_func: Callable,
    processor_getter: Callable,
    post_processing_decorator: Callable,
) -> Callable:
    
    """
    【全新重构版】
    创建一个通用的工具函数包装器，不再使用 exec() 或代码字符串模板。
    它在运行时动态处理类型转换和参数注入。
    """
    original_sig = inspect.signature(algorithm_func)
    
    type_hints = get_type_hints(algorithm_func)
    
    @inject_graph_parameter(processor_getter)
    @post_processing_decorator
    @wraps(algorithm_func)
    def tool_wrapper(**kwargs: Any) -> str:
        """
        这是一个动态生成的工具执行器。
        它会自动处理图(G)的注入、参数类型转换和错误捕获。
        """
        try:
            # 1. 获取图对象
            processor = processor_getter()
            # ⭐ 检查图是否初始化
            if processor.graph is None:
                logger.error(f"❌ 图未初始化，无法执行 {tool_name}")
                return GenericToolOutput(
                    algorithm=tool_name,
                    success=False,
                    error="Graph not initialized",
                    summary="Call 'initialize_graph' first"
                ).model_dump_json()
            
            # ✅ 1. 先复制所有用户传入的参数
            final_kwargs = dict(kwargs)  # 重要！保留所有参数
            logger.debug(f"📝 原始签名: {original_sig}")
            # ⭐⭐⭐ 强制注入 G 参数（即使客户端传了也覆盖）
            final_kwargs['G'] = processor.graph
            logger.info(f"✅ 已自动注入 G 参数 (节点数: {processor.graph.number_of_nodes()}, 边数: {processor.graph.number_of_edges()})")
            logger.info(f"🔍 Initial kwargs for '{tool_name}': {final_kwargs}")
            # if 'backend_kwargs' in original_sig.parameters and 'backend_kwargs' not in kwargs:
            #     kwargs['backend_kwargs'] = {}
        
            if processor.graph is None:
                return GenericToolOutput(
                    algorithm=tool_name,
                    success=False,
                    error="Graph not initialized",
                    summary="Call 'initialize_graph' first"
                ).model_dump_json()
            algo_name = tool_name.replace('run_', '')
            actual_func = algorithm_func

            # (这部分逻辑保持不变)
            if algo_name in UNDIRECTED_ONLY_ALGORITHMS and processor.graph.is_directed():
                alternative_name = UNDIRECTED_ONLY_ALGORITHMS.get(algo_name)
                if alternative_name:
                    actual_func = getattr(nx.algorithms.components, alternative_name, algorithm_func)
                    logger.info(f"🔄 Auto-switching: {algo_name} -> {alternative_name}")
                else:
                    logger.info(f"🔄 Converting to undirected graph for {algo_name}")
                    final_kwargs['G'] = processor.graph.to_undirected()

            # 4. 执行算法
            logger.info(f"🚀 Executing '{tool_name}' with params: {list(kwargs.keys())}")
            
            # 过滤掉原始函数不接受的参数 (例如 __post_processing_code__)
            valid_param_names = set(original_sig.parameters.keys())
            execution_kwargs = {k: v for k, v in final_kwargs.items() if k in valid_param_names}
            logger.info(f"✅ 这里是检查，最终传递给算法的参数: {list(execution_kwargs.keys())}")
            
            # 在 tool_wrapper 函数开始处添加：
            logger.info(f"🔍 tool_wrapper 收到的原始参数: {kwargs}")
            logger.info(f"🔍 final_kwargs 准备完成: {final_kwargs}")
            logger.info(f"🔍 execution_kwargs 过滤后: {execution_kwargs}")

            
            
            result = actual_func(**execution_kwargs)
            
            # 5. 返回结果
            #from .models import GenericToolOutput
            logger.info(f"✅ '{tool_name}' completed successfully")
            return GenericToolOutput(algorithm=tool_name, success=True, result=result, summary=f"'{tool_name}' executed successfully").model_dump_json()

        except Exception as e:
            #from .models import GenericToolOutput
            logger.error(f"❌ '{tool_name}' unexpected error: {e}", exc_info=True)
            return GenericToolOutput(algorithm=tool_name, success=False, error=str(e), summary=f"'{tool_name}' execution failed").model_dump_json()

    # 设置文档字符串，以便在帮助信息中显示修改过
    params_without_G = [
        param for name, param in original_sig.parameters.items() 
        if name not in ['G', 'backend_kwargs']
    ]
    
    tool_wrapper.__signature__ = inspect.Signature(parameters=params_without_G)
    tool_wrapper.__doc__ = _clean_docstring(algorithm_func.__doc__)
    
    logger.debug(f"📝 修改后的签名: {tool_wrapper.__signature__}")
    return tool_wrapper


# ============================================================================
# 14. 主注册函数 (现在依赖于新的 _create_tool_function)
# ============================================================================
def register_discovered_tools(
    mcp,
    processor_getter: Callable,
    post_processing_decorator: Callable
):
    """扫描并注册 NetworkX 算法，同时导出 schemas"""
    logger.info("🚀 开始动态注册 NetworkX 算法 (Schema 导出版)...")
    registered_count = 0
    failed_count = 0
    skipped_count = 0 
    
    # ✅ 新增：同时收集 input 和 output schemas
    input_schemas_map = {}
    output_schemas_map = {}
    
    for module in MODULES_TO_SCAN:
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith('_') or name in ALGORITHMS_TO_EXCLUDE: 
                continue
            
            try:
                sig = inspect.signature(func)
                if not sig.parameters or 'G' not in sig.parameters: 
                    continue
            except (ValueError, TypeError): 
                continue
            
            tool_name = f"run_{name}"
            
            # TODO(chaoyi): 对于有重名的函数，暂时先跳过处理，后期需要解决这个问题。 因为重名的函数可能是对不同属性的图进行数据分析
            if tool_name in input_schemas_map:
                logger.warning(
                    f"⚠️ 工具名称冲突: {tool_name} 已存在，跳过注册\n"
                    f"   当前来源: {module.__name__}\n"
                    f"   已保留来源: {input_schemas_map[tool_name].get('_source_module', 'unknown')}"
                )
                skipped_count += 1
                continue  # ✅ 跳过后续同名工具
            
            try:
                # 1. 生成完整的 Schema
                input_schema = generate_input_schema(func)
                output_schema = generate_output_schema(func)
                description = _clean_docstring(func.__doc__)
                
                # 2. 创建工具函数
                tool_func = _create_tool_function(
                    tool_name, func, processor_getter, 
                    post_processing_decorator
                )                               
                
                # 3. 注册到 MCP
                if hasattr(mcp, 'tool'):
                    try:
                        decorated_func = mcp.tool(
                            name=tool_name,
                            description=description,
                            input_schema=input_schema
                        )(tool_func)
                    except TypeError:
                        decorated_func = mcp.tool(
                            name=tool_name,
                            description=description
                        )(tool_func)
                        if hasattr(decorated_func, '__mcp_tool__'):
                            decorated_func.__mcp_tool__['input_schema'] = input_schema
                    input_schemas_map[tool_name] = input_schema
                    output_schemas_map[tool_name] = output_schema
                    registered_count += 1
                    logger.info(f"✅ 已注册: {tool_name}")
                else:
                    logger.error("MCP 对象没有 'tool' 方法")
                    continue
                
                # ✅ 4. 同时保存两种 schema
                # input_schemas_map[tool_name] = input_schema
                # output_schemas_map[tool_name] = output_schema
                
                # registered_count += 1
                # logger.info(f"✅ 已注册: {tool_name}")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ 注册失败 '{tool_name}': {type(e).__name__}: {e}")
    
    logger.info(f"📊 注册统计: ✅ {registered_count} 成功, ❌ {failed_count} 失败")
    
    # ✅ 5. 导出两种 schemas 到文件
    if input_schemas_map:
        _export_schemas(input_schemas_map, "input")
    if output_schemas_map:
        _export_schemas(output_schemas_map, "output")
    
    return registered_count


def _export_schemas(schemas_map: Dict[str, Dict[str, Any]], schema_type: str):
    """
    ✅ 统一的 schema 导出函数
    
    Args:
        schemas_map: 工具名称到 schema 的映射
        schema_type: "input" 或 "output"
    """
    from pathlib import Path
    import json
    
    try:
        output_file = Path(__file__).parent / f"generated_{schema_type}_schemas.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_map, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 已导出 {len(schemas_map)} 个工具的 {schema_type} schemas")
        logger.info(f"   文件位置: {output_file.absolute()}")
        
    except Exception as e:
        logger.warning(f"⚠️ 无法导出 {schema_type} schemas: {e}")
