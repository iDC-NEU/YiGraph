# mcp_server.py (MODIFIED)

import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps

from mcp.server.fastmcp import FastMCP
# 导入新的动态注册器，并导入 GenericToolOutput 以支持 outputSchema
from aag.computing_engine.networkx_server.dynamic_tool_registry import register_discovered_tools, GenericToolOutput
from aag.computing_engine.networkx_server.graph_computation_processor import GraphComputationProcessor
from aag.expert_search_engine.database.datatype import VertexData, EdgeData
from aag.config.data_upload_config import DatasetConfig

logger = logging.getLogger(__name__)

mcp = FastMCP("Graph Computation Server")

# --- 全局状态与辅助函数 (基本不变) ---

graph_processor = None

def get_processor():
    """延迟获取图处理器实例"""
    global graph_processor
    if graph_processor is None:
        graph_processor = GraphComputationProcessor()
    return graph_processor

def get_datatype_classes():
    """延迟获取数据类型类"""
    return VertexData, EdgeData

# --- 后处理逻辑 (完全不变) ---
# @mcp.tool()
# def apply_post_processing(
#     original_response: Dict[str, Any],
#     post_processing_code: str
# ) -> Dict[str, Any]:
#     """
#     ✅ 独立的后处理工具
    
#     Args:
#         original_response: 原始工具的返回结果
#         post_processing_code: 后处理代码字符串
    
#     Returns:
#         处理后的结果
#     """
#     # if not original_response.get("success"):
#     #     logger.warning("原始执行失败，跳过后处理")
#     #     return original_response
    
#     try:
#         logger.info("🔄 开始执行后处理代码...")
#         processed_result = _execute_dynamic_code(
#             post_processing_code, 
#             original_response.get("result")
#         )
        
#         original_response["result"] = processed_result
#         original_response["summary"] += " (已应用动态后处理)"
#         logger.info("✅ 后处理执行成功")
#         return original_response
        
#     except Exception as e:
#         logger.error(f"❌ 后处理失败: {e}")
#         return {
#             "success": False,
#             "error": f"后处理失败: {str(e)}",
#             "summary": "后处理脚本执行失败",
#             "original_response": original_response
#         }


# def _execute_dynamic_code(code: str, data: Any) -> Any:
#     """在受限的沙箱中执行后处理代码"""
    
#     # ✅ 定义允许使用的内置函数（白名单）
#     safe_builtins = {
#         # 基本函数
#         "len": len,
#         "sorted": sorted,
#         "reversed": reversed,
#         "enumerate": enumerate,
#         "zip": zip,
#         "map": map,
#         "filter": filter,
#         "sum": sum,
#         "min": min,
#         "max": max,
#         "abs": abs,
#         "round": round,
        
#         # 类型转换
#         "int": int,
#         "float": float,
#         "str": str,
#         "bool": bool,
#         "list": list,
#         "dict": dict,
#         "set": set,
#         "tuple": tuple,
        
#         # 其他常用
#         "range": range,
#         "any": any,
#         "all": all,
#         "isinstance": isinstance,
#         "type": type,
        
#         # ❌ 禁止危险函数
#         # "eval": eval,    # 禁止
#         # "exec": exec,    # 禁止
#         # "open": open,    # 禁止
#         # "__import__": __import__,  # 禁止
#     }
    
#     # ✅ 创建安全的全局作用域
#     global_namespace = {
#         "__builtins__": safe_builtins,
        
#         # 可选：提供常用模块（只读）
#         "json": __import__("json"),
#         "math": __import__("math"),
#         "statistics": __import__("statistics"),
#     }
    
#     local_namespace = {"data": data}
    
#     try:
#         # 执行代码
#         exec(code, global_namespace, local_namespace)
        
#         # 检查 process 函数
#         if "process" not in local_namespace:
#             raise ValueError("后处理代码必须定义 'process(data)' 函数")
        
#         process_func = local_namespace["process"]
        
#         if not callable(process_func):
#             raise ValueError("'process' 必须是一个函数")
        
#         # 执行
#         result = process_func(data)
#         logger.info(f"✅ 后处理执行成功")
        
#         return result
        
#     except Exception as e:
#         logger.error(f"❌ 后处理失败: {e}", exc_info=True)
#         raise RuntimeError(f"后处理代码执行错误: {e}")



def apply_processing(func):
    """通用后处理装饰器 (完全不变)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        #post_processing_code = kwargs.pop("__post_processing_code__", None)
        original_response = func(*args, **kwargs)
        # if not post_processing_code or not original_response.get("success"):
        return original_response
        # try:
        #     processed_result = _execute_dynamic_code(post_processing_code, original_response.get("result"))
        #     original_response["result"] = processed_result
        #     original_response["summary"] += " (已应用动态后处理)"
        #     return original_response
        # except Exception as e:
        #     original_response['success'] = False
        #     original_response['error'] = f"后处理失败: {str(e)}"
        #     original_response['summary'] = "算法执行成功，但后处理脚本执行失败。"
        #     return original_response
    return wrapper

# --- 保留的手动定义的核心工具 ---
# 这些工具因其独特的逻辑或初始化性质而被保留
# 新增: 添加 outputSchema 以匹配动态工具（使用 camelCase）

@mcp.tool()
def run_initialize_graph(vertices: List[Dict[str, Any]], edges: List[Dict[str, Any]], dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """初始化图数据结构，加载顶点和边数据。必须首先调用此工具才能运行任何图算法。"""
    try:
        dataset_obj = DatasetConfig.from_dict(dataset_config)
        directed = dataset_obj.schema.graph_structure.directed
        multiedge = dataset_obj.schema.graph_structure.multigraph

        vertex_objs = [VertexData.from_dict(v) for v in vertices]
        edge_objs = [EdgeData.from_dict(e) for e in edges]
        processor = get_processor()
        processor.create_graph_from_edges(vertex_objs, edge_objs, directed, multiedge)

        info = processor.get_graph_info()
        summary = f"图初始化成功: {info.get('节点数', 0)}个节点, {info.get('边数', 0)}条边"
        logger.info(f"✅ {summary}")
        return {"algorithm": "initialize_graph", "success": True, "result": info, "summary": summary}
    except Exception as e:
        return {"algorithm": "initialize_graph", "success": False, "error": str(e), "summary": "图初始化失败"}

@mcp.tool()
def run_get_graph_info() -> Dict[str, Any]:
    """获取当前图的基本统计信息（节点数、边数、图类型等）。"""
    try:
        result = get_processor().get_graph_info()
        if not result:
            return {"success": False, "summary": "图未初始化", "error": "Graph not initialized"}
        logger.info("✅ 图信息查询完成")
        return {"algorithm": "get_graph_info", "success": True, "result": result, "summary": f"图信息查询成功"}
    except Exception as e:
        return {"algorithm": "get_graph_info", "success": False, "error": str(e), "summary": "图信息查询失败"}
        

# --- 【可选】保留部分逻辑复杂的工具，例如社区发现 ---
# (你可以将 run_louvain_community_detection 等复杂工具的代码粘贴回这里)
# @mcp.tool() ...

# ==============================================================================
# ▼▼▼ 核心改造：调用动态工具注册器 ▼▼▼
# 移除了所有旧的硬编码算法工具 (run_pagerank, run_degree_centrality, etc.)
# 替换为对动态注册器的一次性调用。
# ==============================================================================

register_discovered_tools(
    mcp=mcp, 
    processor_getter=get_processor, 
    post_processing_decorator=apply_processing
)

if __name__ == "__main__":
    mcp.run()
