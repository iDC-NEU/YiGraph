# pyg/mcp_server.py

import csv
import logging
from typing import Any, Dict, List
from functools import wraps

from mcp.server.fastmcp import FastMCP

from aag.computing_engine.networkx_server.graph_computation_processor import GraphComputationProcessor
from aag.computing_engine.pyg.dynamic_tool_registry import register_pyg_tools
from aag.computing_engine.pyg.algorithm_tool_registry import register_pyg_algorithm_tools
from aag.expert_search_engine.database.datatype import VertexData, EdgeData

logger = logging.getLogger(__name__)

# 创建 MCP 服务器实例
mcp = FastMCP("PyG Graph Computation Server")

# 全局图处理器（延迟初始化）
graph_processor = None


def get_processor() -> GraphComputationProcessor:
    """延迟获取图处理器实例"""
    global graph_processor
    if graph_processor is None:
        graph_processor = GraphComputationProcessor()
    return graph_processor


def apply_processing(func):
    """后处理装饰器（直通，保留扩展接口）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@mcp.tool()
def run_initialize_graph(
    vertices: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    dataset_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    初始化图数据结构，加载顶点和边数据。
    必须首先调用此工具，才能运行任何 PyG 卷积。
    """
    try:
        graph_structure = dataset_config.get("schema", {}).get("graph_structure", {})
        directed = graph_structure.get("directed", False)
        multiedge = graph_structure.get("multigraph", False)

        vertex_objs = [VertexData.from_dict(v) for v in vertices]
        edge_objs = [EdgeData.from_dict(e) for e in edges]

        processor = get_processor()
        processor.create_graph_from_edges(vertex_objs, edge_objs, directed, multiedge)

        info = processor.get_graph_info()
        summary = f"图初始化成功: {info.get('节点数', 0)} 个节点, {info.get('边数', 0)} 条边"
        logger.info(f"✅ {summary}")
        return {"algorithm": "initialize_graph", "success": True, "result": info, "summary": summary}

    except Exception as e:
        logger.error(f"❌ 图初始化失败: {e}", exc_info=True)
        return {"algorithm": "initialize_graph", "success": False, "error": str(e), "summary": "图初始化失败"}


@mcp.tool()
def run_initialize_graph_from_file(
    vertices_path: str,
    edges_path: str,
    directed: bool = False,
    multigraph: bool = False,
    vertex_id_field: str = "",
    edge_src_field: str = "src",
    edge_dst_field: str = "dst",
) -> Dict[str, Any]:
    """
    从 CSV 文件路径加载图数据并初始化，适合加载大型数据集。
    vertex_id_field: 顶点 ID 列名，留空则自动取第一列。
    edge_src_field: 边起点列名，默认 src。
    edge_dst_field: 边终点列名，默认 dst。
    """
    try:
        with open(vertices_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("顶点文件为空")
        id_field = vertex_id_field if vertex_id_field else list(rows[0].keys())[0]
        vertex_objs = [VertexData(vid=r[id_field], properties=dict(r)) for r in rows]

        with open(edges_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        edge_objs = [EdgeData(src=r[edge_src_field], dst=r[edge_dst_field], properties=dict(r)) for r in rows]

        processor = get_processor()
        processor.create_graph_from_edges(vertex_objs, edge_objs, directed, multigraph)

        info = processor.get_graph_info()
        summary = f"图初始化成功: {info.get('节点数', 0)} 个节点, {info.get('边数', 0)} 条边"
        logger.info(f"✅ {summary}")
        return {"algorithm": "initialize_graph_from_file", "success": True, "result": info, "summary": summary}

    except Exception as e:
        logger.error(f"❌ 文件加载失败: {e}", exc_info=True)
        return {"algorithm": "initialize_graph_from_file", "success": False, "error": str(e), "summary": "文件加载失败"}


@mcp.tool()
def run_get_graph_info() -> Dict[str, Any]:
    """获取当前图的基本统计信息（节点数、边数、图类型等）。"""
    try:
        result = get_processor().get_graph_info()
        if not result:
            return {"success": False, "summary": "图未初始化", "error": "Graph not initialized"}
        logger.info("✅ 图信息查询完成")
        return {"algorithm": "get_graph_info", "success": True, "result": result, "summary": "图信息查询成功"}

    except Exception as e:
        logger.error(f"❌ 图信息查询失败: {e}", exc_info=True)
        return {"algorithm": "get_graph_info", "success": False, "error": str(e), "summary": "图信息查询失败"}


# 批量注册所有 PyG GNN 卷积工具（算子级）
register_pyg_tools(
    mcp=mcp,
    processor_getter=get_processor,
    post_processing_decorator=apply_processing,
)

# 批量注册算法级工具（backbone × task 组合）
register_pyg_algorithm_tools(
    mcp=mcp,
    processor_getter=get_processor,
    post_processing_decorator=apply_processing,
)

if __name__ == "__main__":
    mcp.run()
