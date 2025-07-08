import networkx as nx
from typing import List, Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class GraphProcessor:
    """
    图处理器类：处理从图数据库提取的图数据，转换为networkx格式并运行图算法
    """
    
    def __init__(self):
        """初始化图处理器"""
        self.graph = None
        self.is_directed = False
    
    def create_graph_from_edges(self, edges: List[Tuple], directed: bool = True) -> nx.DiGraph:
        """
        从边列表创建networkx图
        
        Args:
            edges: 边列表，每个边是一个元组 (source, target) 或 (source, target, weight)
            directed: 是否创建有向图，默认为True
            
        Returns:
            networkx图对象
        """
        try:
            if directed:
                self.graph = nx.DiGraph()
                self.is_directed = True
            else:
                self.graph = nx.Graph()
                self.is_directed = False
            
            # 添加边到图中
            for edge in edges:
                if len(edge) == 2:
                    source, target = edge
                    self.graph.add_edge(source, target)
                elif len(edge) == 3:
                    source, target, weight = edge
                    self.graph.add_edge(source, target, weight=weight)
                else:
                    logger.warning(f"跳过无效的边格式: {edge}")
            
            logger.info(f"成功创建{'有向' if directed else '无向'}图，包含 {self.graph.number_of_nodes()} 个节点和 {self.graph.number_of_edges()} 条边")
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
            'degree': self.run_degree_centrality
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