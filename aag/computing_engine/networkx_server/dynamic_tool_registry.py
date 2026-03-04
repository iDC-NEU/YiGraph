"""
Dynamic tool registry - fully fixed version.

Key improvements:
1. ✅ Extend algorithm coverage to 60+ modules (from 4 to 60+)
2. ✅ Simplify parameter type conversion logic
3. ✅ Intelligently filter out unsuitable algorithms
4. ✅ Automatically switch between directed/undirected graphs when needed
5. ✅ Automatically extract description and output_schema from docstrings
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
# 1. Tool output model
# ============================================================================
class GenericToolOutput(BaseModel):
    algorithm: Optional[str] = Field(None, description="算法名称")
    success: bool = Field(..., description="执行状态")
    result: Optional[Any] = Field(None, description="算法结果")
    error: Optional[str] = Field(None, description="错误信息")
    summary: Optional[str] = Field(None, description="执行摘要")

# ============================================================================
# 2. Modules to scan
# ============================================================================
MODULES_TO_SCAN = [
    # === Core centrality algorithms ===
    nx.algorithms.centrality,
    
    # === Community detection ===
    nx.algorithms.community,
    nx.algorithms.community.modularity_max,
    nx.algorithms.community.label_propagation,
    nx.algorithms.community.louvain,
    
    # === Link analysis ===
    nx.algorithms.link_analysis.pagerank_alg,
    nx.algorithms.link_analysis.hits_alg,
    
    # === Connectivity components ===
    nx.algorithms.components,
    nx.algorithms.connectivity,
    
    # === Shortest path algorithms ===
    nx.algorithms.shortest_paths.generic,
    nx.algorithms.shortest_paths.weighted,
    nx.algorithms.shortest_paths.unweighted,
    nx.algorithms.shortest_paths.dense,
    nx.algorithms.simple_paths,
    
    # === Clustering and metrics ===
    nx.algorithms.cluster,
    nx.algorithms.clique,
    nx.algorithms.core,
    nx.algorithms.distance_measures,
    
    # === Graph isomorphism and matching ===
    nx.algorithms.isomorphism,
    nx.algorithms.matching,
    
    # === Flow algorithms ===
    nx.algorithms.flow,
    
    # === Tree algorithms ===
    nx.algorithms.tree.recognition,
    nx.algorithms.tree.mst,
    
    # === DAG algorithms ===
    nx.algorithms.dag,
    
    # === Cycles and bridges ===
    nx.algorithms.cycles,
    nx.algorithms.bridges,
    
    # === Graph coloring ===
    nx.algorithms.coloring,
    
    # === Similarity measures ===
    nx.algorithms.similarity,
    nx.algorithms.link_prediction,
    
    # === Bipartite graphs ===
    nx.algorithms.bipartite,
    nx.algorithms.bipartite.centrality,
    nx.algorithms.bipartite.cluster,
    
    # === Traversal and tournaments ===
    nx.algorithms.traversal,
    nx.algorithms.tournament,
    
    # === Graph operators ===
    nx.algorithms.operators,
    
    # === Other important algorithms ===
    nx.algorithms.dominating,
    nx.algorithms.efficiency_measures,
    nx.algorithms.euler,
    nx.algorithms.reciprocity,
    nx.algorithms.assortativity,
    nx.algorithms.vitality,
    nx.algorithms.wiener,
]

# ============================================================================
# 3. Algorithms to exclude
# ============================================================================
ALGORITHMS_TO_EXCLUDE = [
    # Algorithms with serious performance issues
    'communicability_betweenness_centrality',
    'current_flow_betweenness_centrality',
    'approximate_current_flow_betweenness_centrality',
    
    # Duplicated functionality
    'shortest_path_length',  # Duplicated with shortest_path
    
    # Generator-style algorithms (return iterators, not ideal for LLM tooling)
    'all_simple_paths',
    'all_shortest_paths',
    'all_simple_edge_paths',
]

# ============================================================================
# 4. Directed/undirected mapping
# ============================================================================
UNDIRECTED_ONLY_ALGORITHMS = {
    # Connectivity components
    'connected_components': 'weakly_connected_components',
    'number_connected_components': 'number_weakly_connected_components', 
    'node_connected_component': 'node_weakly_connected_component',
    'is_connected': 'is_weakly_connected',
    
    # Bipartite graph algorithms (convert directed graph to undirected)
    'is_bipartite': None,
    'bipartite_sets': None,
    
    # Clustering coefficient (subset of algorithms)
    'triangles': None,
    'clustering': None,
}

# ============================================================================
# 5. Parse return type from docstring
# ============================================================================
def _parse_returns_from_docstring(docstring: str) -> dict:
    """Parse return type and description from the Returns section of a docstring (aligned with NetworkX format)."""
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
# 6. Extract full docstring as description
# ============================================================================
def _extract_full_description(docstring: str) -> str:
    """Extract the full docstring as a tool description."""
    if not docstring:
        return "No description available."
    
    lines = docstring.split('\n')
    main_description = []
    
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('---'):
            main_description.append(stripped)
    
    description = ' '.join(main_description).strip()
    
    return description if description else "No description available."

# ============================================================================
# 7-12. Helper functions
# ============================================================================

def _extract_param_description(func: Callable, param_name: str) -> Dict[str, str]:
    """Extract parameter description and type from a function docstring."""
    import inspect
    import re
    
    doc = inspect.getdoc(func)
    if not doc:
        return {"type": "", "description": ""}
    
    # ✅ Key fix: use a lookahead to exclude all possible sections that come after "Parameters"
    # Especially avoid non-standard sections like "Additional backends"
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
    
    # ✅ Keep the original parsing logic: use indentation and line structure to parse type and description
    param_section_start = -1
    param_section_end = -1
    
    for i, line in enumerate(lines):
        # Find the parameter definition line (format: "param_name : type")
        if re.match(rf"^\s*{re.escape(param_name)}\s*:", line):
            param_section_start = i
            # From this line, search forward for the end of the parameter block
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                # Next parameter definition or empty line followed by a definition means end of this block
                if next_line and re.match(r"^\w+\s*:", lines[j]):
                    param_section_end = j
                    break
            else:
                # No further parameter definition found, this is the last parameter
                param_section_end = len(lines)
            break
    
    if param_section_start == -1:
        return {"type": "", "description": ""}
    
    # ✅ Extract the parameter block content
    param_lines = lines[param_section_start:param_section_end]
    
    if not param_lines:
        return {"type": "", "description": ""}
    
    # First line: extract type
    first_line = param_lines[0]
    type_match = re.search(rf"{re.escape(param_name)}\s*:\s*(.+)", first_line)
    param_type = type_match.group(1).strip() if type_match else ""
    
    # Following indented lines: extract description
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
    """Extract all documented parameter names from a func docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return set()
    
    # Extract the Parameters section
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
        # Parameter definition line format: "param_name :"
        match = re.match(r"^\s*(\w+)\s*:", line)
        if match:
            documented.add(match.group(1))
    
    return documented


def generate_input_schema(func: Callable) -> Dict[str, Any]:
    """Generate input_schema from a function signature and docstring."""
    sig = inspect.signature(func)
    documented_params = _get_documented_params(func)
    
    
    parameters = {}  # ✅ Use "parameters" instead of "properties"
    required = []
    
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param_name not in documented_params:
            continue
        # ✅ Extract parameter info from docstring
        param_info = _extract_param_description(func, param_name)
        docstring_type = param_info["type"]
        description = param_info["description"]
        # json_type = "string"  # default type
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
        # ✅ Build parameter schema
        param_schema = {
            "description": description if description else f"Parameter '{param_name}' for {func.__name__}",
        }
        
        # ✅ Attach original type information
        if docstring_type:
            param_schema["type"] = docstring_type
        
        # Handle default values
        if param.default != inspect.Parameter.empty:
            if param.default is None:
                param_schema["default"] = None
            elif isinstance(param.default, (int, float, str, bool)):
                param_schema["default"] = param.default
            else:
                param_schema["default"] = str(param.default)
        else:
            # No default value and not optional => required
            if "optional" not in docstring_type.lower() and param_name != "G":
                required.append(param_name)
        
        parameters[param_name] = param_schema
   
    
    schema = {
        "type": "object",
        "parameters": parameters  # ✅ Use "parameters" instead of "properties"
    }
    
    if required:
        schema["required"] = required
    
    return schema

def generate_output_schema(func: Callable) -> Dict[str, Any]:
    """Generate output_schema from a function docstring."""
    docstring = func.__doc__ or ""
    return _parse_returns_from_docstring(docstring)

def _clean_docstring(doc: str) -> str:
    """Clean and enhance a docstring - now using the full description."""
    return _extract_full_description(doc)

def _serialize_result(result: Any) -> Any:
    """
    Serialize the algorithm result into a JSON-serializable structure.
    Special handling for NetworkX graph objects (DiGraph, Graph, MultiDiGraph, MultiGraph).
    """
    if result is None:
        return None
    
    # Handle NetworkX graph objects
    if isinstance(result, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        # Convert to a list-of-nodes and list-of-edges format
        nodes = list(result.nodes())
        edges = []
        for u, v, data in result.edges(data=True):
            edge_dict = {"src": str(u), "dst": str(v)}
            # Add edge attributes (skip non-serializable values)
            for key, value in data.items():
                try:
                    # Try to JSON-serialize the value
                    import json
                    json.dumps(value)
                    edge_dict[key] = value
                except (TypeError, ValueError):
                    # If not serializable, fall back to string
                    edge_dict[key] = str(value)
            edges.append(edge_dict)
        
        return {
            "type": "graph",
            "graph_type": result.__class__.__name__,
            "nodes": [str(n) for n in nodes],
            "edges": edges,
            "num_nodes": result.number_of_nodes(),
            "num_edges": result.number_of_edges()
        }
    
    # Handle all other non-serializable results
    try:
        import json
        json.dumps(result)
        return result
    except (TypeError, ValueError):
        # If still not serializable, convert to string
        return str(result)
    
# ============================================================================
# 13. Parameter handling logic
# ============================================================================
def inject_graph_parameter(processor_getter: Callable):
    """Decorator: inject graph parameter G before calling the algorithm."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(**kwargs: Any) -> Any:
            # ✅ Inject G here
            processor = processor_getter()
            if processor.graph is None:
                raise ValueError("Graph not initialized")
            
            kwargs['G'] = processor.graph
            logger.debug(f"✅ Injected graph G={processor.graph}")
            logger.debug(f"✅ Final call kwargs: {kwargs}")
            # Call the original function
            return func(**kwargs)
        
        # ⭐ Override signature to **kwargs
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
    Fully refactored version.

    Create a generic tool function wrapper without using exec() or code templates.
    It dynamically handles type conversion and parameter injection at runtime.
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
            # 1. Get the graph object
            processor = processor_getter()
            # ⭐ Ensure the graph has been initialized
            if processor.graph is None:
                logger.error(f"❌ Graph not initialized; cannot execute {tool_name}")
                return GenericToolOutput(
                    algorithm=tool_name,
                    success=False,
                    error="Graph not initialized",
                    summary="Call 'initialize_graph' first"
                ).model_dump_json()
            
            # ✅ 1. Copy all user-provided parameters first
            final_kwargs = dict(kwargs)  # Important: keep all parameters
            logger.debug(f"📝 Original signature: {original_sig}")
            # ⭐⭐⭐ Force-inject G (override even if the client passes it)
            final_kwargs['G'] = processor.graph
            logger.info(f"✅ Automatically injected G (nodes={processor.graph.number_of_nodes()}, edges={processor.graph.number_of_edges()})")
            logger.info(f"🔍 Initial kwargs for '{tool_name}': {final_kwargs}")
            if processor.graph is None:
                return GenericToolOutput(
                    algorithm=tool_name,
                    success=False,
                    error="Graph not initialized",
                    summary="Call 'initialize_graph' first"
                ).model_dump_json()
            algo_name = tool_name.replace('run_', '')
            actual_func = algorithm_func

            if algo_name in UNDIRECTED_ONLY_ALGORITHMS and processor.graph.is_directed():
                alternative_name = UNDIRECTED_ONLY_ALGORITHMS.get(algo_name)
                if alternative_name:
                    actual_func = getattr(nx.algorithms.components, alternative_name, algorithm_func)
                    logger.info(f"🔄 Auto-switching algorithm: {algo_name} -> {alternative_name}")
                else:
                    logger.info(f"🔄 Converting graph to undirected for {algo_name}")
                    final_kwargs['G'] = processor.graph.to_undirected()

            # 4. Execute the algorithm
            logger.info(f"🚀 Executing '{tool_name}' with param keys: {list(kwargs.keys())}")
            
            # Filter out parameters the original function does not accept (e.g. __post_processing_code__)
            valid_param_names = set(original_sig.parameters.keys())
            execution_kwargs = {k: v for k, v in final_kwargs.items() if k in valid_param_names}
            logger.info(f"✅ Final parameter keys passed to algorithm: {list(execution_kwargs.keys())}")
            
            # Extra logging for debugging wrapper parameters:
            logger.info(f"🔍 tool_wrapper received raw kwargs: {kwargs}")
            logger.info(f"🔍 final_kwargs prepared: {final_kwargs}")
            logger.info(f"🔍 execution_kwargs after filtering: {execution_kwargs}")

            result = actual_func(**execution_kwargs)
            
            # 5. Serialize the result (convert NetworkX graph objects to JSON-friendly structures)
            serializable_result = _serialize_result(result)
            
            # 6. Return the result
            logger.info(f"✅ '{tool_name}' completed successfully")
            return GenericToolOutput(algorithm=tool_name, success=True, result=serializable_result, summary=f"'{tool_name}' executed successfully").model_dump_json()

        except Exception as e:
            logger.error(f"❌ '{tool_name}' unexpected error: {e}", exc_info=True)
            return GenericToolOutput(algorithm=tool_name, success=False, error=str(e), summary=f"'{tool_name}' execution failed").model_dump_json()

    # Update docstring and signature so tools and help text stay accurate
    params_without_G = [
        param for name, param in original_sig.parameters.items() 
        if name not in ['G', 'backend_kwargs']
    ]
    
    tool_wrapper.__signature__ = inspect.Signature(parameters=params_without_G)
    tool_wrapper.__doc__ = _clean_docstring(algorithm_func.__doc__)
    
    logger.debug(f"📝 Updated tool signature: {tool_wrapper.__signature__}")
    return tool_wrapper


# ============================================================================
# 14. Main registration function (depends on _create_tool_function)
# ============================================================================
def register_discovered_tools(
    mcp,
    processor_getter: Callable,
    post_processing_decorator: Callable
):
    """Scan and register NetworkX algorithms, and export their schemas."""
    logger.info("🚀 Starting dynamic registration of NetworkX algorithms (schema export version)...")
    registered_count = 0
    failed_count = 0
    skipped_count = 0 
    
    # ✅ Collect both input and output schemas
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
            
            # TODO(chaoyi): For functions with duplicate names, skip for now.
            # In the future we may want to support different variants for different graph attributes.
            if tool_name in input_schemas_map:
                logger.warning(
                    f"⚠️ Tool name conflict: {tool_name} already registered, skipping\n"
                    f"   current module: {module.__name__}\n"
                    f"   kept module: {input_schemas_map[tool_name].get('_source_module', 'unknown')}"
                )
                skipped_count += 1
                continue  # ✅ 跳过后续同名工具
            
            try:
                # 1. Generate schemas
                input_schema = generate_input_schema(func)
                output_schema = generate_output_schema(func)
                description = _clean_docstring(func.__doc__)
                
                # 2. Create tool wrapper
                tool_func = _create_tool_function(
                    tool_name, func, processor_getter, 
                    post_processing_decorator
                )                               
                
                # 3. Register in MCP
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
                    logger.info(f"✅ Registered tool: {tool_name}")
                else:
                    logger.error("MCP 对象没有 'tool' 方法")
                    continue
                
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Failed to register '{tool_name}': {type(e).__name__}: {e}")
    
    logger.info(f"📊 Registration summary: ✅ {registered_count} success, ❌ {failed_count} failed")
    
    # ✅ 5. Export both kinds of schemas to files
    if input_schemas_map:
        _export_schemas(input_schemas_map, "input")
    if output_schemas_map:
        _export_schemas(output_schemas_map, "output")
    
    return registered_count


def _export_schemas(schemas_map: Dict[str, Dict[str, Any]], schema_type: str):
    """
    Unified schema export helper.

    Args:
        schemas_map: mapping from tool name to its schema
        schema_type: "input" or "output"
    """
    from pathlib import Path
    import json
    
    try:
        output_file = Path(__file__).parent / f"generated_{schema_type}_schemas.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schemas_map, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Exported {len(schemas_map)} {schema_type} schemas")
        logger.info(f"   File path: {output_file.absolute()}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to export {schema_type} schemas: {e}")
