# pyg/mcp_server.py

import logging
from typing import Any, Dict, List
from functools import wraps

from mcp.server.fastmcp import FastMCP

from aag.computing_engine.networkx_server.graph_computation_processor import GraphComputationProcessor
from aag.computing_engine.pyg.dynamic_tool_registry import register_pyg_tools
from aag.expert_search_engine.database.datatype import VertexData, EdgeData
from aag.config.data_upload_config import DatasetConfig

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
        dataset_obj = DatasetConfig.from_dict(dataset_config)
        directed = dataset_obj.schema.graph_structure.directed
        multiedge = dataset_obj.schema.graph_structure.multigraph

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


# 批量注册所有 PyG GNN 卷积工具
register_pyg_tools(
    mcp=mcp,
    processor_getter=get_processor,
    post_processing_decorator=apply_processing,
)

if __name__ == "__main__":
    mcp.run()
