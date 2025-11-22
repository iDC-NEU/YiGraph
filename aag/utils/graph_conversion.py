from typing import Dict, Iterable, List, Optional, Tuple
from aag.expert_search_engine.database.datatype import VertexData, EdgeData

def flatten_graph(
    vertices: Optional[Iterable[VertexData]],
    edges: Optional[Iterable[EdgeData]],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    将完整的顶点/边对象转换为轻量级结构，便于下游只依赖 ID 的模块使用。
    """
    node_ids: List[str] = []
    seen_nodes = set()

    if vertices:
        for vertex in vertices:
            if vertex is None or vertex.vid is None:
                continue
            vid_str = str(vertex.vid)
            if vid_str in seen_nodes:
                continue
            seen_nodes.add(vid_str)
            node_ids.append(vid_str)

    edge_pairs: List[Tuple[str, str]] = []
    if edges:
        for edge in edges:
            if edge is None or edge.src is None or edge.dst is None:
                continue
            edge_pairs.append((str(edge.src), str(edge.dst)))

    return node_ids, edge_pairs


def reconstruct_graph(
    node_ids: Iterable[str],
    edge_pairs: Iterable[Tuple[str, str]],
    vertices: Iterable[VertexData],
    edges: Iterable[EdgeData],
) -> Tuple[List[VertexData], List[EdgeData]]:
    """
    根据节点/边 ID 列表，从完整缓存中恢复对应的顶点和边对象。
    """
    vertex_index: Dict[str, VertexData] = {}
    for vertex in vertices or []:
        if vertex is None or vertex.vid is None:
            continue
        vertex_index[str(vertex.vid)] = vertex

    edge_index: Dict[Tuple[str, str], List[EdgeData]] = {}
    for edge in edges or []:
        if edge is None or edge.src is None or edge.dst is None:
            continue
        edge_index.setdefault((str(edge.src), str(edge.dst)), []).append(edge)

    reconstructed_vertices = [vertex_index[nid] for nid in node_ids if nid in vertex_index]

    reconstructed_edges: List[EdgeData] = []
    for pair in edge_pairs:
        for edge in edge_index.get(pair, []):
            reconstructed_edges.append(edge)

    return reconstructed_vertices, reconstructed_edges

