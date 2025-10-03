"""
MCP Server for Graph Computation Processor
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Graph Computation Server")

graph_processor = None

def get_processor():
    """延迟获取图处理器实例"""
    global graph_processor
    if graph_processor is None:
        from graph_engine.graphcomputation_processor import GraphComputationProcessor
        graph_processor = GraphComputationProcessor()
    return graph_processor

def get_datatype_classes():
    """延迟获取数据类型类"""
    from database.datatype import VertexData, EdgeData
    return VertexData, EdgeData


@mcp.tool(
    description="初始化图数据结构,加载顶点和边数据。必须在运行其他算法前调用此工具。参数:vertices(顶点列表),edges(边列表),directed(是否有向图),multiedge(是否多重边)"
)
def initialize_graph(
    vertices: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    directed: bool = True,
    multiedge: bool = False
) -> Dict[str, Any]:
    """初始化图数据结构"""
    try:
        VertexData, EdgeData = get_datatype_classes()
        processor = get_processor()
        
        vertex_objs = [VertexData(vid=v['vid'], properties=v.get('properties', {})) for v in vertices]
        edge_objs = [EdgeData(src=e['src'], dst=e['dst'], rank=e.get('rank'), properties=e.get('properties', {})) 
                     for e in edges]
        
        processor.create_graph_from_edges(vertex_objs, edge_objs, directed, multiedge)
        info = processor.get_graph_info()
        
        return {
            "algorithm": "initialize_graph",
            "success": True,
            "result": info,
            "summary": f"图初始化成功:{info.get('节点数', 0)}个节点,{info.get('边数', 0)}条边"
        }
        
    except Exception as e:
        return {
            "algorithm": "initialize_graph",
            "success": False,
            "result": None,
            "summary": "图初始化失败",
            "error": str(e)
        }


@mcp.tool(
    description="运行PageRank算法计算节点重要性和影响力。用于回答'哪些节点最重要'、'节点权威性'等问题。参数:alpha(阻尼系数,默认0.85),max_iter(最大迭代次数),tol(收敛容差)"
)
def run_pagerank(
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """运行PageRank算法"""
    try:
        result = get_processor().run_pagerank(alpha=alpha, max_iter=max_iter, tol=tol)
        top_nodes = sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "algorithm": "pagerank",
            "success": True,
            "result": dict(top_nodes),
            "full_result_count": len(result),
            "summary": f"PageRank计算完成,共{len(result)}个节点"
        }
        
    except Exception as e:
        return {
            "algorithm": "pagerank",
            "success": False,
            "result": None,
            "summary": "PageRank算法执行失败",
            "error": str(e)
        }


@mcp.tool(
    description="运行连通分量算法,找出图中所有连通的子图。用于回答'图的连通性'、'有多少个分组'等问题"
)
def run_connected_components() -> Dict[str, Any]:
    """运行连通分量算法"""
    try:
        result = get_processor().run_connected_components()
        component_sizes = [len(comp) for comp in result]
        
        return {
            "algorithm": "connected_components",
            "success": True,
            "result": {
                "component_count": len(result),
                "largest_component": max(component_sizes) if component_sizes else 0,
                "component_sizes": component_sizes[:10]
            },
            "summary": f"发现{len(result)}个连通分量"
        }
        
    except Exception as e:
        return {
            "algorithm": "connected_components",
            "success": False,
            "result": None,
            "summary": "连通分量算法执行失败",
            "error": str(e)
        }


@mcp.tool(
    description="计算两个节点之间的最短路径。用于回答'从A到B的最短距离'、'路径长度'等问题。参数:source(源节点),target(目标节点)"
)
def run_shortest_path(source: Union[int, str], target: Union[int, str]) -> Dict[str, Any]:
    """计算最短路径"""
    try:
        result = get_processor().run_shortest_path(source=source, target=target)
        
        if result is None:
            summary = f"从节点{source}到节点{target}不存在路径"
            path_length = None
        else:
            path_length = len(result) - 1
            summary = f"最短路径长度为{path_length}"
        
        return {
            "algorithm": "shortest_path",
            "success": True,
            "result": {"path": result, "length": path_length},
            "summary": summary
        }
        
    except Exception as e:
        return {
            "algorithm": "shortest_path",
            "success": False,
            "result": None,
            "summary": "最短路径算法执行失败",
            "error": str(e)
        }


@mcp.tool(
    description="计算介数中心性,找出网络中起桥梁作用的关键节点。用于回答'哪些节点是关键连接点'等问题"
)
def run_betweenness_centrality() -> Dict[str, Any]:
    """计算介数中心性"""
    try:
        result = get_processor().run_betweenness_centrality()
        top_nodes = sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "algorithm": "betweenness_centrality",
            "success": True,
            "result": dict(top_nodes),
            "full_result_count": len(result),
            "summary": f"介数中心性计算完成,共{len(result)}个节点"
        }
    except Exception as e:
        return {
            "algorithm": "betweenness_centrality",
            "success": False,
            "result": None,
            "error": str(e)
        }


@mcp.tool(
    description="计算接近中心性,找出与其他节点距离最近的中心节点。用于回答'哪些节点最中心'等问题"
)
def run_closeness_centrality() -> Dict[str, Any]:
    """计算接近中心性"""
    try:
        result = get_processor().run_closeness_centrality()
        top_nodes = sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "algorithm": "closeness_centrality",
            "success": True,
            "result": dict(top_nodes),
            "full_result_count": len(result),
            "summary": f"接近中心性计算完成,共{len(result)}个节点"
        }
    except Exception as e:
        return {
            "algorithm": "closeness_centrality",
            "success": False,
            "result": None,
            "error": str(e)
        }


@mcp.tool(
    description="计算度中心性,找出连接最多的节点。用于回答'哪些节点连接最多'等问题"
)
def run_degree_centrality() -> Dict[str, Any]:
    """计算度中心性"""
    try:
        result = get_processor().run_degree_centrality()
        top_nodes = sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "algorithm": "degree_centrality",
            "success": True,
            "result": dict(top_nodes),
            "full_result_count": len(result),
            "summary": f"度中心性计算完成,共{len(result)}个节点"
        }
    except Exception as e:
        return {
            "algorithm": "degree_centrality",
            "success": False,
            "result": None,
            "error": str(e)
        }


@mcp.tool(
    description="使用Louvain算法进行社区检测,找出网络中的群体结构。用于回答'有哪些社区'、'群体划分'等问题。参数:resolution(分辨率),random_state(随机种子)"
)
def run_louvain_community_detection(
    resolution: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """运行Louvain社区检测"""
    try:
        result = get_processor().run_louvain_community_detection(
            resolution=resolution,
            random_state=random_state
        )
        
        stats = get_processor().get_community_statistics(result)
        
        return {
            "algorithm": "louvain_community_detection",
            "success": True,
            "result": {
                "community_count": result['community_count'],
                "modularity": result['modularity'],
                "statistics": stats
            },
            "summary": f"发现{result['community_count']}个社区,模块度:{result['modularity']:.4f}"
        }
        
    except Exception as e:
        return {
            "algorithm": "louvain_community_detection",
            "success": False,
            "result": None,
            "error": str(e)
        }


@mcp.tool(
    description="获取指定节点的社区信息,包括社区成员、邻居等。用于回答'节点X属于哪个社区'等问题。参数:vertex_id(节点ID)"
)
def get_community_by_vertex(vertex_id: int) -> Dict[str, Any]:
    """获取指定节点的社区信息"""
    try:
        result = get_processor().get_community_by_specific_id(vertex_id)
        
        return {
            "algorithm": "get_community_by_vertex",
            "success": True,
            "result": result,
            "summary": f"节点{vertex_id}的社区信息查询完成"
        }
        
    except Exception as e:
        return {
            "algorithm": "get_community_by_vertex",
            "success": False,
            "result": None,
            "error": str(e)
        }


@mcp.tool(
    description="获取图的基本统计信息,包括节点数、边数、图类型等。用于回答'图有多大'、'基本信息'等问题"
)
def get_graph_info() -> Dict[str, Any]:
    """获取图的基本信息"""
    try:
        result = get_processor().get_graph_info()
        
        return {
            "algorithm": "get_graph_info",
            "success": True,
            "result": result,
            "summary": f"图信息:{result.get('节点数', 0)}个节点,{result.get('边数', 0)}条边"
        }
        
    except Exception as e:
        return {
            "algorithm": "get_graph_info",
            "success": False,
            "result": None,
            "error": str(e)
        }


if __name__ == "__main__":
    mcp.run()
