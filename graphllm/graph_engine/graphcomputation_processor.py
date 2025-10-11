import networkx as nx
import logging
import community 
from typing import List, Tuple, Dict, Any, Optional, Union
from database.datatype import *
from database.nebulagraph import *
from graph_engine.graph_processor import GraphProcessor
#修改
import inspect
from typing import Callable, Any


logger = logging.getLogger(__name__)


class GraphComputationProcessor(GraphProcessor):
    """
    图处理器类：处理从图数据库提取的图数据，转换为networkx格式并运行图算法
    """
    
    def __init__(self):
        self.vertices = None
        self.edges = None
        self.graph = None
        self.is_directed = False
        self.is_multiedge = False
        self.query_vertices = None
        self.query_vertices =  15
    
    def create_graph_from_edges(self, vertices: List[VertexData], edges: List[EdgeData], directed: bool = True, multiedge: bool = False):
        """
        从边列表创建networkx图
        
        Args:
            vertices: 顶点列表
            edges: 边列表
            directed: 是否为有向图
            multiedge: 是否为多重边图
            
        Returns:
            networkx图对象
        """
        try:
            self.get_vertex_map(vertices)
            self.is_directed = directed
            self.is_multiedge = multiedge
            if self.is_directed:
                self.graph = nx.MultiDiGraph() if self.is_multiedge else nx.DiGraph()
            else:
                self.graph = nx.MultiGraph() if self.is_multiedge else nx.Graph()
            
            # 添加顶点及属性
            for v in vertices:
                self.graph.add_node(v.vid, **v.properties)

            # 添加边及属性
            for idx, e in enumerate(edges):
                edge_attrs = e.properties.copy()
                if e.rank is not None:
                    edge_attrs['rank'] = e.rank

                if self.is_multiedge:
                    key = e.rank if e.rank is not None else 0
                    self.graph.add_edge(e.src, e.dst, key=key, **edge_attrs)
                else:
                    self.graph.add_edge(e.src, e.dst, **edge_attrs)

            print(f"✅ 图构建完成: {self.graph.number_of_nodes()} 个节点，{self.graph.number_of_edges()} 条边")
            return self.graph
            
        except Exception as e:
            logger.error(f"创建图时发生错误: {e}")
            raise
    
    def run_pagerank(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
        """
        运行PageRank算法
        
        Args:
            alpha: 阻尼系数，默认为0.85
            max_iter: 最大迭代次数，默认为100
            tol: 收敛容差，默认为1e-6
            
        Returns:
            PageRank分数字典，键为节点，值为分数
        """
        if self.graph is None:
            raise ValueError("图未初始化，请先调用create_graph_from_edges方法")
        
        try:
            pagerank_scores = nx.pagerank(
                self.graph, 
                alpha=alpha, 
                max_iter=max_iter, 
                tol=tol
            )
            logger.info(f"PageRank算法完成，计算了 {len(pagerank_scores)} 个节点的分数")
            return pagerank_scores
            
        except Exception as e:
            logger.error(f"运行PageRank算法时发生错误: {e}")
            raise
    
    def run_connected_components(self) -> List[set]:
        """
        运行连通分量算法
        
        Returns:
            连通分量列表，每个连通分量是一个节点集合
        """
        if self.graph is None:
            raise ValueError("图未初始化，请先调用create_graph_from_edges方法")
        
        try:
            if self.is_directed:
                # 对于有向图，使用强连通分量
                components = list(nx.strongly_connected_components(self.graph))
                logger.info(f"找到 {len(components)} 个强连通分量")
            else:
                # 对于无向图，使用连通分量
                components = list(nx.connected_components(self.graph))
                logger.info(f"找到 {len(components)} 个连通分量")
            
            return components
            
        except Exception as e:
            logger.error(f"运行连通分量算法时发生错误: {e}")
            raise
    
    def run_shortest_path(self, source: Any, target: Any) -> Optional[List[Any]]:
        """
        计算两个节点之间的最短路径
        
        Args:
            source: 源节点
            target: 目标节点
            
        Returns:
            最短路径节点列表，如果不存在路径则返回None
        """
        if self.graph is None:
            raise ValueError("图未初始化，请先调用create_graph_from_edges方法")
        
        try:
            if self.is_directed:
                path = nx.shortest_path(self.graph, source, target)
            else:
                path = nx.shortest_path(self.graph, source, target)
            
            logger.info(f"从 {source} 到 {target} 的最短路径长度为 {len(path) - 1}")
            return path
            
        except nx.NetworkXNoPath:
            logger.warning(f"从 {source} 到 {target} 不存在路径")
            return None
        except Exception as e:
            logger.error(f"计算最短路径时发生错误: {e}")
            raise
    
    def run_betweenness_centrality(self) -> Dict[Any, float]:
        """
        计算介数中心性
        
        Returns:
            介数中心性分数字典
        """
        if self.graph is None:
            raise ValueError("图未初始化，请先调用create_graph_from_edges方法")
        
        try:
            betweenness_scores = nx.betweenness_centrality(self.graph)
            logger.info(f"介数中心性计算完成，计算了 {len(betweenness_scores)} 个节点的分数")
            return betweenness_scores
            
        except Exception as e:
            logger.error(f"计算介数中心性时发生错误: {e}")
            raise
    
    def run_closeness_centrality(self) -> Dict[Any, float]:
        """
        计算接近中心性
        
        Returns:
            接近中心性分数字典
        """
        if self.graph is None:
            raise ValueError("图未初始化，请先调用create_graph_from_edges方法")
        
        try:
            closeness_scores = nx.closeness_centrality(self.graph)
            logger.info(f"接近中心性计算完成，计算了 {len(closeness_scores)} 个节点的分数")
            return closeness_scores
            
        except Exception as e:
            logger.error(f"计算接近中心性时发生错误: {e}")
            raise
    
    def run_degree_centrality(self) -> Dict[Any, float]:
        """
        计算度中心性
        
        Returns:
            度中心性分数字典
        """
        if self.graph is None:
            raise ValueError("图未初始化，请先调用create_graph_from_edges方法")
        
        try:
            degree_scores = nx.degree_centrality(self.graph)
            logger.info(f"度中心性计算完成，计算了 {len(degree_scores)} 个节点的分数")
            return degree_scores
            
        except Exception as e:
            logger.error(f"计算度中心性时发生错误: {e}")
            raise

    def run_louvain_community_detection(self, resolution: float = 1.0, random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        使用Louvain算法进行社区检测
        
        Args:
            resolution: 分辨率参数，控制社区大小。值越大，社区越小
            random_state: 随机种子，用于结果的可重现性
            
        Returns:
            包含社区检测结果的字典：
            - 'communities': 社区列表，每个社区是一个节点集合
            - 'modularity': 模块度分数
            - 'node_communities': 节点到社区的映射字典
            - 'community_count': 社区数量
        """
        if self.graph is None:
            raise ValueError("图未初始化,请先调用create_graph_from_edges方法")
        
        try:
            # 将图转换为无向图（Louvain算法通常用于无向图）
            if self.is_directed:
                undirected_graph = self.graph.to_undirected()
            else:
                undirected_graph = self.graph
            
            # 运行Louvain算法
            partition = community.best_partition(undirected_graph, resolution=resolution, random_state=random_state)
            
            # 计算模块度
            modularity = self._calculate_modularity(undirected_graph, partition)
            
            # 构建社区列表
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = set()
                communities[community_id].add(node)
            
            # 转换为列表格式
            community_list = list(communities.values())
            
            # 构建节点到社区的映射
            node_communities = {node: comm_id for node, comm_id in partition.items()}
            
            result = {
                'communities': community_list,
                'modularity': modularity,
                'node_communities': node_communities,
                'community_count': len(community_list),
                'resolution': resolution
            }
            
            logger.info(f"Louvain社区检测完成，发现 {len(community_list)} 个社区，模块度: {modularity:.4f}")
            print(f"Louvain社区检测完成，发现 {len(community_list)} 个社区，模块度: {modularity:.4f}")
            return result
            
        except ImportError:
            logger.error("未安装community模块，请运行: pip install python-louvain")
            raise
        except Exception as e:
            logger.error(f"Louvain社区检测时发生错误: {e}")
            raise
    
    def get_community_by_specific_id(self, query_vertex_id: Optional[int] = None) -> Dict[str, Any]:
        """
        获取指定节点ID的社区结果
        
        Args:
            query_vertex_id: 查询的节点ID，如果为None则使用self.query_vertices
            
        Returns:
            包含指定节点社区信息的字典：
            - 'vertex_id': 查询的节点ID
            - 'community_id': 节点所属的社区ID
            - 'community_members': 该社区的所有成员节点
            - 'community_size': 社区大小
            - 'neighbors_in_community': 社区内的邻居节点
            - 'neighbors_outside_community': 社区外的邻居节点
        """
        if query_vertex_id is None:
            query_vertex_id = self.query_vertices
            
        if query_vertex_id is None:
            raise ValueError("未指定查询节点ID，请设置query_vertex_id参数或self.query_vertices")
            
        if self.graph is None:
            raise ValueError("图未初始化，请先调用create_graph_from_edges方法")
            
        # 运行Louvain算法获取社区检测结果
        louvain_result = self.run_louvain_community_detection()
        
        # 获取节点到社区的映射
        node_communities = louvain_result['node_communities']
        
        # 检查查询节点是否存在于图中
        if query_vertex_id not in node_communities:
            return {
                'error': f'节点 {query_vertex_id} 不存在于图中',
                'vertex_id': query_vertex_id,
                'available_nodes': list(node_communities.keys())
            }
        
        # 获取节点所属的社区ID
        community_id = node_communities[query_vertex_id]
        
        # 找到该社区的所有成员
        community_members = set()
        for node, comm_id in node_communities.items():
            if comm_id == community_id:
                community_members.add(node)
        
        # 获取节点的邻居
        if query_vertex_id in self.graph:
            undirected_graph = self.graph.to_undirected()
            all_neighbors = set(undirected_graph.neighbors(query_vertex_id))
            
            # 分类邻居：社区内和社区外
            neighbors_in_community = all_neighbors.intersection(community_members)
            neighbors_outside_community = all_neighbors - community_members
        else:
            neighbors_in_community = set()
            neighbors_outside_community = set()
        
        result = {
            'vertex_id': query_vertex_id,
            'community_id': community_id,
            'community_members': sorted(list(community_members)),
            'community_size': len(community_members),
            'neighbors_in_community': sorted(list(neighbors_in_community)),
            'neighbors_outside_community': sorted(list(neighbors_outside_community)),
            'total_neighbors': len(neighbors_in_community) + len(neighbors_outside_community),
            'community_cohesion': len(neighbors_in_community) / max(1, len(all_neighbors)) if 'all_neighbors' in locals() else 0.0
        }
        
        logger.info(f"节点 {query_vertex_id} 的社区信息: 社区ID={community_id}, 社区大小={len(community_members)}")
        print(f"节点 {query_vertex_id} 的社区信息: 社区ID={community_id}, 社区大小={len(community_members)}")
        
        return result
    

    def _calculate_modularity(self, graph: nx.Graph, partition: Dict[Any, int]) -> float:
        """
        计算模块度
        
        Args:
            graph: 无向图
            partition: 节点到社区的映射
            
        Returns:
            模块度分数
        """
        try:
            return community.modularity(partition, graph)
        except ImportError:
            # 如果community模块不可用，使用自定义实现
            return self._custom_modularity(graph, partition)
    
    def _custom_modularity(self, graph: nx.Graph, partition: Dict[Any, int]) -> float:
        """
        自定义模块度计算实现
        
        Args:
            graph: 无向图
            partition: 节点到社区的映射
            
        Returns:
            模块度分数
        """
        m = graph.number_of_edges()
        if m == 0:
            return 0.0
        
        # 计算总度数
        total_degree = sum(dict(graph.degree()).values())
        
        # 按社区分组
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = set()
            communities[comm_id].add(node)
        
        modularity = 0.0
        
        for community in communities.values():
            # 计算社区内边数
            internal_edges = 0
            community_degree = 0
            
            for node in community:
                community_degree += graph.degree(node)
                for neighbor in graph.neighbors(node):
                    if neighbor in community:
                        internal_edges += 1
            
            # 避免重复计算边
            internal_edges //= 2
            
            # 计算模块度贡献
            expected_edges = (community_degree ** 2) / (2 * m)
            modularity += (internal_edges - expected_edges) / m
        
        return modularity
    
    def get_community_statistics(self, louvain_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取社区检测的统计信息
        
        Args:
            louvain_result: Louvain算法的结果
            
        Returns:
            统计信息字典
        """
        communities = louvain_result['communities']
        
        if not communities:
            return {"error": "没有找到社区"}
        
        # 计算社区大小
        community_sizes = [len(comm) for comm in communities]
        
        # 计算统计信息
        stats = {
            'total_communities': len(communities),
            'largest_community_size': max(community_sizes),
            'smallest_community_size': min(community_sizes),
            'average_community_size': sum(community_sizes) / len(community_sizes),
            'modularity': louvain_result['modularity'],
            'size_distribution': {
                'small': len([s for s in community_sizes if s <= 5]),
                'medium': len([s for s in community_sizes if 5 < s <= 20]),
                'large': len([s for s in community_sizes if s > 20])
            }
        }
        
        return stats
    
    def run_algorithm(self, algorithm: str, **kwargs) -> Union[Dict, List, Any]:
        """
        运行指定的图算法
        
        Args:
            algorithm: 算法名称，支持 'pagerank', 'cc', 'shortest_path', 'betweenness', 'closeness', 'degree'
            **kwargs: 算法特定参数
            
        Returns:
            算法结果
        """
        algorithm_map = {
            'pagerank': self.run_pagerank,
            'cc': self.run_connected_components,
            'shortest_path': self.run_shortest_path,
            'betweenness': self.run_betweenness_centrality,
            'closeness': self.run_closeness_centrality,
            'degree': self.run_degree_centrality,
            'louvain': self.run_louvain_community_detection
        }
        
        if algorithm not in algorithm_map:
            raise ValueError(f"不支持的算法: {algorithm}。支持的算法: {list(algorithm_map.keys())}")
        
        try:
            return algorithm_map[algorithm](**kwargs)
        except Exception as e:
            logger.error(f"运行算法 {algorithm} 时发生错误: {e}")
            raise    
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        获取图的基本信息
        
        Returns:
            包含图信息的字典
        """
        if self.graph is None:
            return {"error": "图未初始化"}
        
        info = {
            "节点数": self.graph.number_of_nodes(),
            "边数": self.graph.number_of_edges(),
            "图类型": "有向图" if self.is_directed else "无向图",
            "是否连通": nx.is_connected(self.graph) if not self.is_directed else None,
            "是否强连通": nx.is_strongly_connected(self.graph) if self.is_directed else None
        }
        
        return info
    
    def execute_plan(self, edges: List[Tuple], plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        根据执行计划按顺序执行算法
        
        Args:
            plan: 执行计划列表，每个元素是一个字典，包含：
                - 'algorithm': 算法名称
                - 'params': 算法参数字典（可选）
                - 'name': 结果名称（可选，默认为算法名称）
                
        Returns:
            包含所有算法结果的字典
        """
        self.create_graph_from_edges(edges)
        if self.graph is None:
            raise ValueError("图未初始化，请先调用create_graph_from_edges方法")
        
        results = {}
        
        try:
            for i, step in enumerate(plan):
                algorithm = step.get('algorithm')
                params = step.get('params', {})
                result_name = step.get('name', algorithm)
                
                if not algorithm:
                    logger.error(f"步骤 {i+1} 缺少算法名称")
                    raise ValueError(f"步骤 {i+1} 缺少算法名称")
                
                logger.info(f"执行步骤 {i+1}: {algorithm}")
                
                # 执行算法
                result = self.run_algorithm(algorithm, **params)
                results[result_name] = result
                
                logger.info(f"步骤 {i+1} 完成: {algorithm}")
            
            logger.info(f"执行计划完成，共执行 {len(results)} 个算法")
            return results
            
        except Exception as e:
            logger.error(f"执行计划时发生错误: {e}")
            raise
    
    def create_execution_plan(self, algorithms: List[str], 
                            algorithm_params: Optional[Dict[str, Dict]] = None,
                            result_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        创建执行计划
        
        Args:
            algorithms: 算法名称列表
            algorithm_params: 算法参数字典，键为算法名称，值为参数字典
            result_names: 结果名称列表，如果为None则使用算法名称
            
        Returns:
            执行计划列表
        """
        plan = []
        algorithm_params = algorithm_params or {}
        
        for i, algorithm in enumerate(algorithms):
            step = {
                'algorithm': algorithm,
                'params': algorithm_params.get(algorithm, {}),
                'name': result_names[i] if result_names and i < len(result_names) else algorithm
            }
            plan.append(step)
        
        return plan 
    
    def get_vertex_map(self, vertices: List[VertexData]):
        """
        将 self.vertices 转换为以 vid 为键、VertexData 为值的字典。
        """
        self.vertices =  {vertex.vid: vertex for vertex in vertices}
    def run_generic_algorithm(self, algorithm_func: Callable, **kwargs) -> Any:
        """
        一个通用的方法，用于运行任何以图 'G' 作为首个参数的NetworkX算法。

        Args:
            algorithm_func: 要调用的实际NetworkX函数 (例如, nx.pagerank)。
            **kwargs: 要传递给该算法函数的参数。

        Returns:
            来自NetworkX算法的原始结果。
        """
        # ▼▼▼ FIX #1: 使用 self.graph 进行检查 ▼▼▼
        # (同时也优化了检查方式，使其更Pythonic)
        if self.graph is None:
            # 返回一个错误字典，而不是抛出异常，这样更安全
            return {"error": "图尚未初始化，请先调用 initialize_graph。"}
            
        if self.graph.number_of_nodes() == 0:
            logger.warning("尝试在空图上运行算法，返回空结果。")
            return {} # 对空图返回一个安全的结果

        try:
            # ▼▼▼ FIX #2: 将 self.graph 传递给算法 ▼▼▼
            # 核心调用：将存储的图对象 self.graph 和其他参数传递给指定的算法函数
            # 例如: result = nx.pagerank(self.graph, alpha=0.85, max_iter=100)
            result = algorithm_func(self.graph, **kwargs)
            
            # 特殊处理：NetworkX的连通分量等函数返回的是生成器，需转换为列表才能被JSON序列化
            if inspect.isgenerator(result):
                # 将集合(set)也转换为列表，确保JSON兼容性
                return [list(c) for c in result]
                
            return result
        except Exception as e:
            logger.error(f"在执行通用算法 '{algorithm_func.__name__}' 时发生错误: {e}", exc_info=True)
            # 在算法执行层也返回一个错误字典，增加系统的健壮性
            return {"error": f"算法执行失败: {str(e)}"}



if __name__ == '__main__':
    testclient = NebulaGraphClient(space_name="AMLSim1K")
    vertices = testclient.get_all_vertices()
    edges = testclient.get_all_edges()
    print(f"vertices number: {len(vertices)}, edge number: {len(edges)}")
    
    graphprocessor = GraphComputationProcessor()
    graphprocessor.create_graph_from_edges(vertices, edges, True, True)
    info = graphprocessor.get_graph_info()
    print(info)

    components = graphprocessor.run_pagerank()
    # print(sum(components.values()) == 1.0)
    # tt = 0
    # total_score = 0
    # for key, value in components.items():
    #     tt = tt + 1
    #     total_score += value
    #     if tt < 10:
    #         print(f"{key}:{value}")
    # print(total_score)

    # community1 = graphprocessor.run_louvain_community_detection()
    # # print(graphprocessor.get_community_statistics(community))
    # result = graphprocessor.get_community_by_specific_id()
    # for k,v in result.items():
    #     print(f"{k}: {v}")

    com_member  = [6, 15, 23, 28, 68, 86, 93, 118, 124, 136, 139, 155, 171, 221, 243, 244, 310, 360, 371, 385, 386, 402, 409, 437, 462, 467, 508, 545, 570, 585, 622, 623, 630, 631, 635, 654, 675, 679, 744, 746, 769, 797, 799, 802, 815, 839, 847, 854, 878, 884, 895, 934, 956, 959, 965, 982, 983, 1040, 1045, 1048, 1052, 1079, 1104, 1130, 1138, 1164, 1189, 1202, 1264, 1273, 1280, 1282, 1293, 1297, 1307, 1329, 1351, 1379, 1407, 1418]
    sar_count = 0
    print(len(com_member))
    for  v in vertices:
        if v.vid in com_member:
            if v.properties['prior_sar_count']:
                sar_count+=1
    print(sar_count)