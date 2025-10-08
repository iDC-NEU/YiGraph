"""
MCP Server for Graph Computation Processor with Dynamic Post-Processing
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Graph Computation Server")

graph_processor = None

def get_processor():
    """延迟获取图处理器实例"""
    global graph_processor
    if graph_processor is None:
        # 假设您的图引擎代码在这个路径下
        from graph_engine.graphcomputation_processor import GraphComputationProcessor
        graph_processor = GraphComputationProcessor()
    return graph_processor

def get_datatype_classes():
    """延迟获取数据类型类"""
    # 假设您的数据类型定义在这个路径下
    from database.datatype import VertexData, EdgeData
    return VertexData, EdgeData

# ⭐ 新增：安全代码执行器
def _execute_dynamic_code(code_string: str, data: Any) -> Any:
    """
    安全地执行LLM生成的后处理代码。
    
    安全警告:
    直接使用 exec() 存在严重的安全风险，可能导致任意代码执行。
    在生产环境中，必须使用成熟的沙箱技术，例如:
    - RestrictedPython库: https://github.com/zopefoundation/RestrictedPython
    - Docker容器: 将每次执行隔离在独立的、无权限的容器中。
    - WebAssembly (WASM): 将代码编译成安全的WASM模块执行。

    此处的实现仅为原型演示，绝不可直接用于生产！
    """
    if not isinstance(code_string, str) or "def process(data):" not in code_string:
        raise ValueError("后处理代码格式无效，必须包含 'def process(data):' 函数定义")

    try:
        # 创建一个受限的执行命名空间
        execution_scope = {}
        
        # 执行代码，这会使 process 函数在 execution_scope 中被定义
        exec(code_string, {"__builtins__": {}}, execution_scope)
        
        # 从命名空间中获取 process 函数
        process_func = execution_scope.get('process')
        
        if callable(process_func):
            # 使用原始数据调用该函数
            logger.info("🚀 正在执行动态后处理代码...")
            processed_data = process_func(data)
            logger.info("✅ 动态后处理代码执行成功")
            return processed_data
        else:
            raise ValueError("在后处理代码中未找到可调用的 'process' 函数")
            
    except Exception as e:
        logger.error(f"❌ 执行动态后处理代码失败: {e}", exc_info=True)
        # 如果执行失败，返回原始数据或抛出异常
        raise RuntimeError(f"后处理代码执行错误: {e}")


# ⭐ 新增：通用后处理装饰器
def apply_post_processing(func):
    """
    一个装饰器，用于从参数中提取后处理代码，并在工具函数执行后应用它。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 弹出后处理代码，使其不传递给原始工具函数
        post_processing_code = kwargs.pop("__post_processing_code__", None)

        # 调用原始的工具函数（例如 run_pagerank）
        original_response = func(*args, **kwargs)

        # 如果没有后处理代码，或工具执行失败，则直接返回原始结果
        if not post_processing_code or not original_response.get("success"):
            return original_response
        
        try:
            raw_result = original_response.get("result")
            
            # 执行动态代码进行后处理
            processed_result = _execute_dynamic_code(post_processing_code, raw_result)
            
            # 用处理后的结果更新响应
            original_response["result"] = processed_result
            original_response["summary"] += " (已应用动态后处理)"
            original_response["post_processing_applied"] = True

            return original_response

        except Exception as e:
            # 如果后处理失败，返回一个包含错误的响应
            logger.error(f"后处理阶段失败: {e}")
            original_response['success'] = False
            original_response['error'] = f"后处理失败: {str(e)}"
            original_response['summary'] = "算法执行成功，但后处理脚本执行失败。"
            return original_response

    return wrapper


@mcp.tool()
# @apply_post_processing # 初始化函数通常不需要后处理
def initialize_graph(
    vertices: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    directed: bool = True,
    multiedge: bool = False
) -> Dict[str, Any]:
    # Docstring (保持不变)
    """初始化图数据结构，加载顶点和边数据。
**必须首先调用**: 在运行任何图算法前必须先调用此工具初始化图。

**适用场景**:

- 用户提供了新的图数据（顶点/边列表）

- 需要重新构建图结构

- 开始新的分析任务



**参数说明**:

- vertices: 顶点列表，每个顶点需包含 'vid' 和 'properties'。例如: [{"vid": "A", "properties": {"name": "Alice"}}, {"vid": "B", "properties": {"name": "Bob"}}]

- edges: 边列表，每个边需包含 'src', 'dst', 可选 'rank' 和 'properties'。例如: [{"src": "A", "dst": "B", "rank": 1, "properties": {"weight": 10}}]

- directed: 是否为有向图(默认True)

- multiedge: 是否允许多重边(默认False)



**返回**: 图的基本信息（节点数、边数、图类型）

Args:

    vertices: 顶点列表，每个元素为包含'vid'和'properties'的字典

    edges: 边列表，每个元素为包含'src'、'dst'的字典，可选'rank'和'properties'

    directed: 是否为有向图

    multiedge: 是否允许多重边"""
    try:
        VertexData, EdgeData = get_datatype_classes()
        processor = get_processor()
        
        if not vertices:
            return {"success": False, "summary": "顶点列表不能为空", "error": "Empty vertices list"}
        
        vertex_objs = [VertexData(vid=v['vid'], properties=v.get('properties', {})) for v in vertices]
        edge_objs = [EdgeData(src=e['src'], dst=e['dst'], rank=e.get('rank'), properties=e.get('properties', {})) for e in edges]
        
        processor.create_graph_from_edges(vertex_objs, edge_objs, directed, multiedge)
        info = processor.get_graph_info()
        
        logger.info(f"✅ 图初始化成功: {info.get('节点数', 0)} 节点, {info.get('边数', 0)} 边")
        
        return {
            "algorithm": "initialize_graph", "success": True, "result": info,
            "summary": f"图初始化成功: {info.get('节点数', 0)}个节点, {info.get('边数', 0)}条边"
        }
    except Exception as e:
        logger.error(f"❌ 图初始化失败: {e}")
        return {"algorithm": "initialize_graph", "success": False, "error": str(e), "summary": "图初始化失败"}


@mcp.tool()
@apply_post_processing # 应用装饰器
def run_pagerank(
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    计算图中每个节点的PageRank值，衡量节点的重要性和影响力。



**适用问题类型**:

- "哪些账户/节点最重要？"

- "找出最有影响力的实体"

- "节点权威性排名"

- "哪些节点在网络中最核心？"



**算法原理**: 模拟随机游走，重要节点会被更多其他重要节点指向。



**参数说明**:

- alpha: 阻尼系数（0-1之间），默认0.85。表示随机游走继续的概率

- max_iter: 最大迭代次数，默认100

- tol: 收敛容差，默认1e-6


**注意**: 需要先调用 initialize_graph 初始化图



Args:

    alpha: 阻尼系数，范围(0, 1)

    max_iter: 最大迭代次数

    tol: 收敛容差
    """
    try:
        result = get_processor().run_pagerank(alpha=alpha, max_iter=max_iter, tol=tol)
        
        if not result:
            return {"success": False, "summary": "图未初始化或为空", "error": "Graph not initialized"}
        
        logger.info(f"✅ PageRank计算完成: {len(result)} 个节点")
        
        # ⭐ 修改：返回完整结果，移除 hardcoded top-10
        return {
            "algorithm": "pagerank", "success": True, "result": result,
            "summary": f"PageRank计算完成, 共{len(result)}个节点"
        }
    except Exception as e:
        logger.error(f"❌ PageRank计算失败: {e}")
        return {"algorithm": "pagerank", "success": False, "error": str(e), "summary": "PageRank算法执行失败"}


@mcp.tool()
@apply_post_processing # 应用装饰器
def run_connected_components() -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    找出图中所有连通分量（互相连通的节点集合）。



**适用问题类型**:

- "图是否连通？"

- "有多少个独立的子网络/群组？"

- "节点A和节点B是否在同一个连通分量中？"

- "最大的连通分量有多大？"



**算法原理**: 使用深度优先搜索或广度优先搜索找出所有互相可达的节点集合。



**返回**:

- 连通分量数量

- 每个分量的大小



**注意**:

- 对于有向图，计算的是弱连通分量（忽略边的方向）

- 需要先调用 initialize_graph 初始化图
    """
    try:
        result = get_processor().run_connected_components() # result 是一个 list of sets/lists
        
        if not result:
            return {"success": False, "summary": "图未初始化或为空", "error": "Graph not initialized"}
        
        # ⭐ 修改：返回完整结果，后处理代码可以自行计算统计信息
        # 原始结果格式: [['A', 'B'], ['C']]
        logger.info(f"✅ 连通分量分析完成: {len(result)} 个分量")
        
        return {
            "algorithm": "connected_components", "success": True, "result": result,
            "summary": f"发现{len(result)}个连通分量"
        }
    except Exception as e:
        logger.error(f"❌ 连通分量计算失败: {e}")
        return {"algorithm": "connected_components", "success": False, "error": str(e), "summary": "连通分量算法执行失败"}


@mcp.tool()
@apply_post_processing # 应用装饰器
def run_shortest_path(source: Union[int, str], target: Union[int, str]) -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    计算两个节点之间的最短路径。



**适用问题类型**:

- "从账户A到账户B的最短路径是什么？"

- "节点X和节点Y之间隔了几步？"

- "如何从A最快到达B？"

- "两个节点之间的距离是多少？"



**算法原理**: 使用BFS（广度优先搜索）或Dijkstra算法找最短路径。



**参数说明**:

- source: 起始节点ID（必需）

- target: 目标节点ID（必需）



**返回**:

- 完整路径（节点ID序列）

- 路径长度（边的数量）



**注意**:

- 如果节点不存在或不可达，返回 None

- 需要先调用 initialize_graph 初始化图



Args:

    source: 起始节点ID

    target: 目标节点ID
    """
    try:
        path = get_processor().run_shortest_path(source=source, target=target)
        
        if path is None:
            summary = f"从节点{source}到节点{target}不存在路径"
            length = -1
        else:
            length = len(path) - 1
            summary = f"找到从 {source} 到 {target} 的最短路径，长度为 {length}"
        
        logger.info(f"✅ 最短路径计算完成: {source} -> {target}")
        
        # ⭐ 修改：结果包装在字典里，方便后处理
        result = {"path": path, "length": length, "source": source, "target": target}

        return {
            "algorithm": "shortest_path", "success": True, "result": result, "summary": summary
        }
    except Exception as e:
        logger.error(f"❌ 最短路径计算失败: {e}")
        return {"algorithm": "shortest_path", "success": False, "error": str(e), "summary": "最短路径算法执行失败"}

# ... 对 run_betweenness_centrality, run_closeness_centrality, run_degree_centrality 执行类似的修改 ...

@mcp.tool()
@apply_post_processing # 应用装饰器
def run_betweenness_centrality() -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    计算介数中心性，识别在网络中起"桥梁"作用的关键节点。



**适用问题类型**:

- "哪些节点是信息传播的关键枢纽？"

- "找出网络中的中介节点"

- "哪些账户控制了资金流动？"

- "移除哪个节点会最大程度破坏网络连通性？"



**算法原理**: 计算通过每个节点的最短路径数量。介数高的节点是网络中的"桥梁"。



**应用场景**:

- 社交网络: 找意见领袖

- 交易网络: 识别资金中转站

- 通信网络: 找关键路由节点



**返回**:
所有节点的介数中心性值



**注意**:

- 计算量较大，对大图可能较慢

- 需要先调用 initialize_graph 初始化图
    """
    try:
        result = get_processor().run_betweenness_centrality()
        if not result:
            return {"success": False, "summary": "图未初始化或为空", "error": "Graph not initialized"}
        logger.info(f"✅ 介数中心性计算完成: {len(result)} 个节点")
        return {
            "algorithm": "betweenness_centrality", "success": True, "result": result,
            "summary": f"介数中心性计算完成, 共{len(result)}个节点"
        }
    except Exception as e:
        return {"algorithm": "betweenness_centrality", "success": False, "error": str(e), "summary": "介数中心性计算失败"}


@mcp.tool()
@apply_post_processing # 应用装饰器
def run_closeness_centrality() -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    计算接近中心性，找出距离其他所有节点最近的"中心"节点。



**适用问题类型**:

- "哪些节点最容易接触到其他节点？"

- "信息传播最快的节点是哪些？"

- "找出网络的地理中心"

- "哪些账户到其他账户的距离最短？"



**算法原理**: 计算每个节点到其他所有节点的平均最短路径长度的倒数。



**应用场景**:

- 找信息传播源头

- 识别便于协调的中心节点

- 物流/交通网络优化



**返回**: 
所有节点的接近中心性值



**注意**:

- 只对连通图有效

- 需要先调用 initialize_graph 初始化图
    """
    try:
        result = get_processor().run_closeness_centrality()
        if not result:
            return {"success": False, "summary": "图未初始化或为空", "error": "Graph not initialized"}
        logger.info(f"✅ 接近中心性计算完成: {len(result)} 个节点")
        return {
            "algorithm": "closeness_centrality", "success": True, "result": result,
            "summary": f"接近中心性计算完成, 共{len(result)}个节点"
        }
    except Exception as e:
        return {"algorithm": "closeness_centrality", "success": False, "error": str(e), "summary": "接近中心性计算失败"}


@mcp.tool()
@apply_post_processing # 应用装饰器
def run_degree_centrality() -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    计算度中心性，找出连接数量最多的"社交明星"节点。



**适用问题类型**:

- "哪些节点的连接最多？"

- "找出最活跃的账户"

- "谁的朋友/关注者最多？"

- "哪些节点有最多的直接邻居？"



**算法原理**: 简单计算每个节点的度（连接边数）。



**返回**: 
所有节点的度中心性



**注意**:

- 这是最简单的中心性指标

- 有向图会分别计算入度和出度

- 需要先调用 initialize_graph 初始化图
    """
    try:
        result = get_processor().run_degree_centrality()
        if not result:
            return {"success": False, "summary": "图未初始化或为空", "error": "Graph not initialized"}
        logger.info(f"✅ 度中心性计算完成: {len(result)} 个节点")
        return {
            "algorithm": "degree_centrality", "success": True, "result": result,
            "summary": f"度中心性计算完成, 共{len(result)}个节点"
        }
    except Exception as e:
        return {"algorithm": "degree_centrality", "success": False, "error": str(e), "summary": "度中心性计算失败"}


@mcp.tool()
@apply_post_processing # 应用装饰器
def run_louvain_community_detection(resolution: float = 1.0, random_state: Optional[int] = None) -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    使用Louvain算法检测网络中的社区结构（群组/聚类）。



**适用问题类型**:

- "网络中有哪些群组/社区？"

- "如何对节点进行聚类划分？"

- "找出紧密连接的节点集合"

- "账户之间形成了哪些团体？"



**算法原理**: 优化模块度（modularity）来划分社区，使得社区内部连接紧密，社区之间连接稀疏。



**参数说明**:

- resolution: 分辨率参数（默认1.0）。值越大，检测到的社区越多越小；值越小，社区越少越大

- random_state: 随机种子，用于结果可复现



**返回**:

- 社区数量

- 模块度（衡量社区划分质量，范围-0.5到1，越高越好）

- 每个社区的统计信息



**注意**:

- 适用于无向图，有向图会被转为无向

- 需要先调用 initialize_graph 初始化图



Args:

    resolution: 分辨率参数，必须大于0

    random_state: 随机种子
    """
    try:
        # 假设这个函数返回类似 {'community_count': N, 'modularity': M, 'communities': {'node1': 0, 'node2': 1}}
        result = get_processor().run_louvain_community_detection(resolution=resolution, random_state=random_state)
        if not result:
            return {"success": False, "summary": "图未初始化或社区检测失败", "error": "Detection failed"}
        
        logger.info(f"✅ Louvain社区检测完成: {result.get('community_count')} 个社区")

        return {
            "algorithm": "louvain_community_detection", "success": True, "result": result,
            "summary": f"发现{result.get('community_count')}个社区, 模块度: {result.get('modularity', 0):.4f}"
        }
    except Exception as e:
        return {"algorithm": "louvain_community_detection", "success": False, "error": str(e), "summary": "社区检测失败"}


@mcp.tool()
@apply_post_processing # 应用装饰器
def get_community_by_vertex(vertex_id: Union[int, str]) -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    查询指定节点所属的社区及其详细信息。



**适用问题类型**:

- "节点X属于哪个社区？"

- "这个账户在哪个群组里？"

- "社区Y包含哪些成员？"

- "某个节点的社区邻居有哪些？"



**参数说明**:

- vertex_id: 要查询的节点ID(必需)



**返回**:

- 节点所属的社区ID

- 该社区的所有成员

- 该社区的统计信息



**注意**:

- 必须先运行 run_louvain_community_detection 进行社区检测

- 需要先调用 initialize_graph 初始化图



Args:
vertex_id: 节点ID(字符串或整数)
    """
    try:
        result = get_processor().get_community_by_specific_id(vertex_id)
        if not result:
            return {"success": False, "summary": f"节点{vertex_id}不存在或未进行社区检测", "error": "Vertex not found or community detection not run"}
        
        logger.info(f"✅ 节点 {vertex_id} 的社区信息查询完成")
        
        return {
            "algorithm": "get_community_by_vertex", "success": True, "result": result,
            "summary": f"节点{vertex_id}属于社区{result.get('community_id', 'N/A')}"
        }
    except Exception as e:
        return {"algorithm": "get_community_by_vertex", "success": False, "error": str(e), "summary": "社区查询失败"}


@mcp.tool()
# @apply_post_processing # 获取图信息通常不需要后处理
def get_graph_info() -> Dict[str, Any]:
    # Docstring (保持不变)
    """
    获取当前图的基本统计信息和元数据。



**适用问题类型**:

- "这个图有多大？"

- "图的基本信息是什么？"

- "图是有向的还是无向的？"

- "图中有多少节点和边？"



**返回**:

- 节点总数

- 边总数

- 图类型（有向/无向）

- 是否允许多重边

- 平均度数等统计信息



**注意**: 需要先调用 initialize_graph 初始化图
    """
    try:
        result = get_processor().get_graph_info()
        if not result:
            return {"success": False, "summary": "图未初始化", "error": "Graph not initialized"}
        
        logger.info(f"✅ 图信息查询完成")
        
        return {
            "algorithm": "get_graph_info", "success": True, "result": result,
            "summary": f"图信息: {result.get('节点数', 0)}个节点, {result.get('边数', 0)}条边"
        }
    except Exception as e:
        return {"algorithm": "get_graph_info", "success": False, "error": str(e), "summary": "图信息查询失败"}


if __name__ == "__main__":
    mcp.run()
