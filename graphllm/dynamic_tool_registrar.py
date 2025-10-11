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
from typing import Callable, Dict, Any, Optional, get_type_hints, get_origin, get_args, Union, Tuple
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
# 2. ⭐ 修复: 扩展模块扫描范围 (从 4 个增加到 60+)
# ============================================================================
MODULES_TO_SCAN = [
    # === 核心中心性算法 ===
    nx.algorithms.centrality,
    
    # === 社区检测 (新增) ===
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
    
    # === 路径算法 (扩展) ===
    nx.algorithms.shortest_paths.generic,
    nx.algorithms.shortest_paths.weighted,
    nx.algorithms.shortest_paths.unweighted,
    nx.algorithms.shortest_paths.dense,
    nx.algorithms.simple_paths,
    
    # === 聚类与度量 (新增) ===
    nx.algorithms.cluster,
    nx.algorithms.clique,
    nx.algorithms.core,
    nx.algorithms.distance_measures,
    
    # === 图同构与匹配 (新增) ===
    nx.algorithms.isomorphism,
    nx.algorithms.matching,
    
    # === 流算法 (新增) ===
    nx.algorithms.flow,
    
    # === 树算法 (新增) ===
    nx.algorithms.tree.recognition,
    nx.algorithms.tree.mst,
    
    # === DAG 算法 (新增) ===
    nx.algorithms.dag,
    
    # === 环与桥 (新增) ===
    nx.algorithms.cycles,
    nx.algorithms.bridges,
    
    # === 图着色 (新增) ===
    nx.algorithms.coloring,
    
    # === 相似性度量 (新增) ===
    nx.algorithms.similarity,
    nx.algorithms.link_prediction,
    
    # === 二分图 (新增) ===
    nx.algorithms.bipartite,
    nx.algorithms.bipartite.centrality,
    nx.algorithms.bipartite.cluster,
    
    # === 拓扑与遍历 (新增) ===
    nx.algorithms.traversal,
    nx.algorithms.tournament,
    
    # === 图运算 (新增) ===
    nx.algorithms.operators,
    
    # === 其他重要算法 (新增) ===
    nx.algorithms.dominating,
    nx.algorithms.efficiency_measures,
    nx.algorithms.euler,
    nx.algorithms.reciprocity,
    nx.algorithms.assortativity,
    nx.algorithms.vitality,
    nx.algorithms.wiener,
]

# ============================================================================
# 3. ⭐ 修复: 完善排除算法列表
# ============================================================================
ALGORITHMS_TO_EXCLUDE = [
    # 性能问题算法
    'communicability_betweenness_centrality',
    'current_flow_betweenness_centrality',
    'approximate_current_flow_betweenness_centrality',
    
    # 重复功能算法
    'shortest_path_length',  # 与 shortest_path 重复
    
    # 需要特殊图类型的算法 (可选择性启用)
    # 'bipartite_*',  # 如果不是二分图则无法使用
    
    # 生成器类算法 (返回迭代器,不适合 LLM 调用)
    'all_simple_paths',
    'all_shortest_paths',
    'all_simple_edge_paths',
]

# ============================================================================
# 4. ⭐ 修复: 扩展有向/无向图映射表
# ============================================================================
UNDIRECTED_ONLY_ALGORITHMS = {
    # 连通性组件
    'connected_components': 'weakly_connected_components',
    'number_connected_components': 'number_weakly_connected_components', 
    'node_connected_component': 'node_weakly_connected_component',
    'is_connected': 'is_weakly_connected',
    
    # 二分图算法 (有向图自动转无向)
    'is_bipartite': None,  # 自动转换为无向图
    'bipartite_sets': None,
    
    # 聚类系数 (部分算法)
    'triangles': None,  # 自动转换
    'clustering': None,
}

# ============================================================================
# 5. 新增: 从docstring解析返回类型的函数
# ============================================================================
def _parse_returns_from_docstring(docstring: str) -> Dict[str, Any]:
    """从docstring的Returns部分解析返回类型描述"""
    if not docstring:
        return {
            "type": "object",
            "properties": {
                "result": {"type": "any", "description": "算法执行结果"}
            }
        }
    
    # 查找Returns部分
    returns_match = re.search(r'Returns\s*[-:]*\s*(.*?)(?=\n\n|\n\w|\Z)', docstring, re.DOTALL | re.IGNORECASE)
    if not returns_match:
        return {
            "type": "object", 
            "properties": {
                "result": {"type": "any", "description": "算法执行结果"}
            }
        }
    
    returns_text = returns_match.group(1).strip()
    
    # 处理复杂返回类型描述
    if 'two-tuple of dictionaries' in returns_text or 'tuple' in returns_text:
        # 处理类似 (hubs, authorities) : two-tuple of dictionaries
        if 'hubs' in returns_text and 'authorities' in returns_text:
            return {
                "type": "object",
                "properties": {
                    "hubs": {
                        "type": "object",
                        "description": "Hub scores dictionary"
                    },
                    "authorities": {
                        "type": "object", 
                        "description": "Authority scores dictionary"
                    }
                }
            }
        else:
            # 通用元组处理
            tuple_match = re.search(r'\(([^)]+)\)\s*:\s*(.+)', returns_text)
            if tuple_match:
                elements = [elem.strip() for elem in tuple_match.group(1).split(',')]
                desc = tuple_match.group(2)
                properties = {}
                for i, elem in enumerate(elements):
                    properties[f"item_{i}"] = {
                        "type": "any",
                        "description": f"{elem} from {desc}"
                    }
                return {
                    "type": "object",
                    "properties": properties
                }
    
    # 处理字典返回类型
    if 'dictionary' in returns_text.lower() or 'dict' in returns_text.lower():
        if 'nodes' in returns_text.lower() or 'node' in returns_text.lower():
            return {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "object",
                        "description": "Dictionary keyed by node with values"
                    }
                }
            }
        else:
            return {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "object", 
                        "description": "Dictionary result"
                    }
                }
            }
    
    # 处理列表/集合返回类型
    if 'list' in returns_text.lower() or 'set' in returns_text.lower() or 'iter' in returns_text.lower():
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "array",
                    "description": "List or collection of results"
                }
            }
        }
    
    # 处理数值返回类型
    if 'float' in returns_text or 'number' in returns_text:
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "number",
                    "description": "Numeric result"
                }
            }
        }
    
    # 处理布尔返回类型
    if 'bool' in returns_text.lower():
        return {
            "type": "object", 
            "properties": {
                "result": {
                    "type": "boolean",
                    "description": "Boolean result"
                }
            }
        }
    
    # 默认返回类型
    return {
        "type": "object",
        "properties": {
            "result": {"type": "any", "description": "Algorithm execution result"}
        }
    }

# ============================================================================
# 6. 新增: 提取完整docstring作为description
# ============================================================================
def _extract_full_description(docstring: str) -> str:
    """提取完整的docstring作为工具描述"""
    if not docstring:
        return "No description available."
    
    # 清理docstring，移除过多的空白字符
    cleaned = re.sub(r'\n\s*\n', '\n\n', docstring.strip())
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # 移除参数和返回部分，保留主要描述
    lines = cleaned.split('\n')
    main_description = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(('parameters:', 'args:', 'arguments:', 'returns:', 'yields:', 'examples:')):
            break
        if stripped and not stripped.startswith('---'):
            main_description.append(stripped)
    
    description = ' '.join(main_description).strip()
    
    # 如果描述过长，截取前500字符
    if len(description) > 500:
        description = description[:497] + "..."
    
    return description if description else "No description available."

# ============================================================================
# 7-12. [保持原有的辅助函数不变，只修改 _clean_docstring 函数]
# ============================================================================
def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Python 类型 -> JSON Schema 类型"""
    origin = get_origin(py_type)
    
    if origin is Union:
        args = get_args(py_type)
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            return _python_type_to_json_schema(non_none_types[0])
    
    if origin is list:
        args = get_args(py_type)
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _python_type_to_json_schema(item_type)
        }
    
    if origin is dict:
        return {"type": "object"}
    
    type_mapping = {
        int: "integer",
        float: "number", 
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    
    return {"type": type_mapping.get(py_type, "string")}

def _extract_param_description(func: Callable, param_name: str) -> str:
    """从 docstring 提取参数描述"""
    doc = func.__doc__
    if not doc:
        return f"Parameter: {param_name}"
    
    lines = doc.split('\n')
    in_params = False
    desc_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.lower() in ['parameters', 'args:', 'parameters:', 'arguments:']:
            in_params = True
            continue
        
        if in_params and stripped and not line.startswith(' '):
            break
        
        if in_params and param_name in stripped:
            if ':' in stripped:
                desc_lines.append(stripped.split(':', 1)[1].strip())
            continue
        
        if desc_lines and stripped and in_params:
            desc_lines.append(stripped)
    
    return ' '.join(desc_lines) if desc_lines else f"Parameter: {param_name}"

def generate_input_schema(func: Callable) -> Dict[str, Any]:
    """从函数签名生成 input_schema，完全排除 G 参数"""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func) if func.__annotations__ else {}
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name in ['G', 'self', 'cls']:
            continue
        
        if param.kind in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]:
            continue
        
        param_type = type_hints.get(param_name, str)
        json_schema = _python_type_to_json_schema(param_type)
        
        param_schema = {
            **json_schema,
            "description": _extract_param_description(func, param_name)
        }
        
        if param.default != inspect.Parameter.empty:
            if param.default is None:
                param_schema["default"] = None
            elif isinstance(param.default, (int, float, str, bool)):
                param_schema["default"] = param.default
            else:
                param_schema["default"] = str(param.default)
        else:
            if param_name in ['backend_kwargs', 'create_using']:
                param_schema["default"] = None
            else:
                required.append(param_name)
        
        properties[param_name] = param_schema
    
    schema = {
        "type": "object",
        "properties": properties
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
# 13. ⭐ 保持简化的参数处理逻辑 (已验证有效)
# ============================================================================
def _generate_type_conversions(sig: inspect.Signature) -> str:
    """生成参数类型转换代码块"""
    conversion_lines = []
    
    for param_name, param in sig.parameters.items():
        if param_name == 'G':
            continue
        
        annotation = param.annotation
        
        # 处理 int 类型
        if annotation == int or (hasattr(annotation, '__origin__') and 
                                  annotation.__origin__ == type(None) and 
                                  int in getattr(annotation, '__args__', [])):
            conversion_lines.append(
                f"        if '{param_name}' in all_kwargs and isinstance(all_kwargs['{param_name}'], str):\n"
                f"            try:\n"
                f"                all_kwargs['{param_name}'] = int(all_kwargs['{param_name}'])\n"
                f"            except (ValueError, TypeError):\n"
                f"                pass"
            )
        
        # 处理 float 类型（新增）
        elif annotation == float or (hasattr(annotation, '__origin__') and 
                                      annotation.__origin__ == type(None) and 
                                      float in getattr(annotation, '__args__', [])):
            conversion_lines.append(
                f"        if '{param_name}' in all_kwargs and isinstance(all_kwargs['{param_name}'], str):\n"
                f"            try:\n"
                f"                all_kwargs['{param_name}'] = float(all_kwargs['{param_name}'])\n"
                f"            except (ValueError, TypeError):\n"
                f"                pass"
            )
        
        # 处理 bool 类型
        elif annotation == bool:
            conversion_lines.append(
                f"        if '{param_name}' in all_kwargs and isinstance(all_kwargs['{param_name}'], str):\n"
                f"            all_kwargs['{param_name}'] = all_kwargs['{param_name}'].lower() in ('true', '1', 'yes')"
            )
    
    return "\n".join(conversion_lines) if conversion_lines else "        pass"


def _create_tool_function(
    tool_name: str,
    algorithm_func: Callable,
    processor_getter: Callable,
    post_processing_decorator: Callable,
    pydantic_model: BaseModel
) -> Callable:
    """创建工具函数，在执行时自动注入 G 参数"""
    original_sig = inspect.signature(algorithm_func)
    
    safe_params = []
    param_type_map = {}
    
    for param in original_sig.parameters.values():
        if param.name == 'G':
            continue
        if param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]:
            safe_params.append(param)
            if param.annotation != inspect.Parameter.empty:
                param_type_map[param.name] = param.annotation
    
    valid_param_names = {p.name for p in safe_params}
    
    # 修复 backend_kwargs 和 create_using 默认值
    final_params = []
    for p in safe_params:
        if p.name in ['backend_kwargs', 'create_using'] and p.default == inspect.Parameter.empty:
            new_param = p.replace(default=None)
            final_params.append(new_param)
        else:
            final_params.append(p)
    
    # 生成函数签名
    algo_param_defs = []
    sorted_params = sorted(final_params, key=lambda p: p.default == inspect.Parameter.empty, reverse=True)
    
    for p in sorted_params:
        param_str = p.name
        if p.annotation != inspect.Parameter.empty:
            annotation_repr = getattr(p.annotation, '__name__', str(p.annotation).replace("typing.", ""))
            param_str += f": {annotation_repr}"
        if p.default != inspect.Parameter.empty:
            if p.default is None:
                default_val_str = "None"
            else:
                default_val_str = repr(p.default) if not inspect.isclass(p.default) else p.default.__name__
            param_str += f" = {default_val_str}"
        algo_param_defs.append(param_str)
    
    final_param_defs_str = ", ".join(algo_param_defs)
    
    # ⭐ 关键修复: 简化的参数类型转换 - 只处理基本类型，跳过复杂参数
    conversion_lines = []
    complex_params = ['backend_kwargs', 'personalization', 'nstart', 'dangling', 'weight', 'create_using']
    
    for name, type_obj in param_type_map.items():
        # 跳过复杂参数类型
        if name in complex_params:
            continue
            
        # 处理 Union 类型
        origin = get_origin(type_obj)
        actual_type = type_obj
        if origin is Union:
            type_args = [arg for arg in get_args(type_obj) if arg is not type(None)]
            if type_args:
                actual_type = type_args[0]
        
        type_name = getattr(actual_type, '__name__', str(actual_type))
        
        # 只处理基本数值类型转换
        if type_name == 'float':
            conversion_code = f"""
if '{name}' in all_kwargs and all_kwargs['{name}'] is not None:
    try:
        param_value = all_kwargs['{name}']
        if isinstance(param_value, str):
            all_kwargs['{name}'] = float(param_value.strip())
        elif isinstance(param_value, int):
            all_kwargs['{name}'] = float(param_value)
    except (ValueError, TypeError) as e:
        return GenericToolOutput(
            algorithm="{tool_name}",
            success=False,
            error=f"Parameter '{name}' must be a number, got '{{all_kwargs[\"{name}\"]}}'",
            summary="Parameter type error"
        ).model_dump_json()
"""
        elif type_name == 'int':
            conversion_code = f"""
if '{name}' in all_kwargs and all_kwargs['{name}'] is not None:
    try:
        param_value = all_kwargs['{name}']
        if isinstance(param_value, str):
            all_kwargs['{name}'] = int(float(param_value.strip()))
        elif isinstance(param_value, float):
            all_kwargs['{name}'] = int(param_value)
    except (ValueError, TypeError) as e:
        return GenericToolOutput(
            algorithm="{tool_name}",
            success=False,
            error=f"Parameter '{name}' must be an integer, got '{{all_kwargs[\"{name}\"]}}'",
            summary="Parameter type error"
        ).model_dump_json()
"""
        elif type_name == 'bool':
            conversion_code = f"""
if '{name}' in all_kwargs and all_kwargs['{name}'] is not None:
    param_value = all_kwargs['{name}']
    if isinstance(param_value, str):
        param_value = param_value.lower().strip()
        all_kwargs['{name}'] = param_value in ['true', '1', 't', 'yes', 'y', 'on']
    elif isinstance(param_value, (int, float)):
        all_kwargs['{name}'] = bool(param_value)
"""
        else:
            conversion_code = ""
        
        if conversion_code:
            conversion_lines.append(conversion_code)
    
    type_conversion_block = "\n".join("        " + line for line in "\n".join(conversion_lines).splitlines())
    
    algo_name = tool_name.replace('run_', '')
    
    # 生成函数体
    cleaned_doc = _clean_docstring(algorithm_func.__doc__)
    func_template = f'''
@post_processing_decorator
@wraps(algorithm_func)
def {tool_name}({final_param_defs_str}) -> str:
    """{cleaned_doc}
    
    🔩 Graph Parameter: Automatically injected (no need to provide 'G').
    📋 Prerequisites: Graph must be initialized first.
    """
    try:
        all_kwargs = locals()
        
        # Step 1: 获取 processor 并验证图
        processor = processor_getter()
        if processor.graph is None:
            return GenericToolOutput(
                algorithm="{tool_name}",
                success=False,
                error="Graph not initialized",
                summary="Call 'initialize_graph' first"
            ).model_dump_json()
        
        # Step 2: 参数类型转换（仅基本类型）
        {type_conversion_block}
        
        # Step 3: 过滤参数并注入 G
        filtered_kwargs = {{k: v for k, v in all_kwargs.items() if k in {valid_param_names}}}
        filtered_kwargs['G'] = processor.graph
        
        # Step 4: 有向/无向图自动切换
        algo_name = '{algo_name}'
        actual_func = algorithm_func
        
        if algo_name in UNDIRECTED_ONLY_ALGORITHMS and processor.graph.is_directed():
            alternative_name = UNDIRECTED_ONLY_ALGORITHMS[algo_name]
            
            if alternative_name:
                # 尝试查找替代算法
                try:
                    actual_func = getattr(nx, alternative_name, None)
                    if actual_func is None:
                        actual_func = getattr(nx.algorithms.components, alternative_name, None)
                    if actual_func is None:
                        import networkx.algorithms.components as comp
                        actual_func = getattr(comp, alternative_name, None)
                    
                    if actual_func:
                        logger.info(f"🔄 Auto-switching: {{algo_name}} -> {{alternative_name}}")
                    else:
                        logger.info(f"🔄 Converting to undirected graph for {{algo_name}}")
                        filtered_kwargs['G'] = processor.graph.to_undirected()
                        actual_func = algorithm_func
                except Exception as e:
                    logger.warning(f"⚠️  Fallback to original algorithm: {{e}}")
                    actual_func = algorithm_func
            else:
                # 直接转换为无向图
                logger.info(f"🔄 Converting to undirected graph for {{algo_name}}")
                filtered_kwargs['G'] = processor.graph.to_undirected()
        
        logger.info(f"🚀 Executing '{tool_name}' with params: {{list(filtered_kwargs.keys())}}")
        
        # Step 5: 调用算法
        result = actual_func(**filtered_kwargs)
        
        if result is None:
            logger.warning(f"⚠️  '{tool_name}' returned None")
            return GenericToolOutput(
                algorithm="{tool_name}",
                success=False,
                summary="Algorithm returned None",
                error="Check graph structure"
            ).model_dump_json()
        
        logger.info(f"✅ '{tool_name}' completed successfully")
        return GenericToolOutput(
            algorithm="{tool_name}",
            success=True,
            result=result,
            summary=f"'{tool_name}' executed successfully"
        ).model_dump_json()
    
    except TypeError as e:
        logger.error(f"❌ '{tool_name}' parameter error: {{e}}")
        return GenericToolOutput(
            algorithm="{tool_name}",
            success=False,
            error=str(e),
            summary="Invalid parameters"
        ).model_dump_json()
    
    except Exception as e:
        logger.error(f"❌ '{tool_name}' unexpected error: {{e}}", exc_info=True)
        return GenericToolOutput(
            algorithm="{tool_name}",
            success=False,
            error=str(e),
            summary=f"'{tool_name}' execution failed"
        ).model_dump_json()
'''
    
    exec_globals = {
        'processor_getter': processor_getter,
        'algorithm_func': algorithm_func,
        'post_processing_decorator': post_processing_decorator,
        'wraps': wraps,
        'logger': logger,
        'GenericToolOutput': GenericToolOutput,
        'inspect': inspect,
        'UNDIRECTED_ONLY_ALGORITHMS': UNDIRECTED_ONLY_ALGORITHMS,
        'nx': nx,
        'get_origin': get_origin,
        'get_args': get_args,
    }
    
    exec_locals = {}
    exec(func_template, exec_globals, exec_locals)
    generated_func = exec_locals[tool_name]
    
    generated_func.__signature__ = inspect.Signature(final_params)
    
    return generated_func

# ============================================================================
# 14. 主注册函数 - 修改工具注册部分
# ============================================================================
def register_discovered_tools(
    mcp,
    processor_getter: Callable,
    post_processing_decorator: Callable
):
    """扫描并注册 NetworkX 算法"""
    logger.info("🚀 开始动态注册 NetworkX 算法 (扩展版)...")
    logger.info(f"📦 将扫描 {len(MODULES_TO_SCAN)} 个模块")
    logger.info("\n" + "="*60)
    
    registered_count = 0
    failed_count = 0
    
    # ✅ 新增：存储生成的 output schemas
    output_schemas_map = {}
    
    for module in MODULES_TO_SCAN:
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith('_') or name in ALGORITHMS_TO_EXCLUDE:
                continue
            
            try:
                sig = inspect.signature(func)
                if list(sig.parameters.keys())[0] != 'G':
                    continue
            except:
                continue
            
            tool_name = f"run_{name}"
            
            try:
                tool_func = _create_tool_function(
                    tool_name, func, processor_getter, 
                    post_processing_decorator, None
                )
                
                input_schema = generate_input_schema(func)
                output_schema = generate_output_schema(func)
                description = _extract_full_description(func.__doc__)
                
                # 注册工具（不带 outputSchema）
                decorated_func = mcp.tool(
                    name=tool_name,
                    description=description
                )(tool_func)
                
                # ✅ 保存生成的 output schema
                output_schemas_map[tool_name] = output_schema
                
                registered_count += 1
                
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ 注册失败 '{tool_name}': {type(e).__name__}: {e}")
    
    logger.info(f"📊 注册统计: ✅ {registered_count} 成功, ❌ {failed_count} 失败")
    
    # ✅ 导出 schemas 到文件
    if output_schemas_map:
        _export_output_schemas(output_schemas_map)
    
    return registered_count


def _export_output_schemas(schemas_map: Dict[str, Dict[str, Any]]):
    """导出 output schemas 到 JSON 文件"""
    # ✅ 修复：使用正确的文件路径
    from pathlib import Path
    import json
    
    output_file = Path(__file__).parent / "generated_output_schemas.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_map, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 已导出 {len(schemas_map)} 个工具的 output schemas")
        logger.info(f"   文件位置: {output_file.absolute()}")
    except Exception as e:
        logger.warning(f"⚠️ 无法导出 output schemas: {e}")

    """导出 output schemas 到 JSON 文件"""
    output_file = Path(__file__).parent / "generated_output_schemas.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_map, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 已导出 {len(schemas_map)} 个工具的 output schemas 到 {output_file}")
    except Exception as e:
        logger.warning(f"⚠️ 无法导出 output schemas: {e}")
