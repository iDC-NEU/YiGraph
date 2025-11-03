#!/usr/bin/env python3
"""
GraphProcessor使用示例
演示如何从图数据库提取的边列表创建图并运行各种图算法
"""

import logging
from aag.computing_engine.graphcomputation_processor import GraphProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建图处理器实例
    processor = GraphProcessor()
    
    # 模拟从图数据库提取的边列表
    # 格式：(source, target) 或 (source, target, weight)
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('D', 'A'),
        ('A', 'C'),
        ('B', 'D'),
        ('E', 'F'),
        ('F', 'G'),
        ('G', 'E')
    ]
    
    # 创建有向图
    graph = processor.create_graph_from_edges(edges, directed=True)
    
    # 获取图信息
    info = processor.get_graph_info()
    print(f"图信息: {info}")
    
    # 运行PageRank算法
    pagerank_scores = processor.run_pagerank()
    print(f"PageRank结果: {pagerank_scores}")
    
    # 运行连通分量算法
    components = processor.run_connected_components()
    print(f"连通分量: {components}")
    
    # 计算最短路径
    path = processor.run_shortest_path('A', 'D')
    print(f"从A到D的最短路径: {path}")


def example_weighted_graph():
    """带权重的图示例"""
    print("\n=== 带权重的图示例 ===")
    
    processor = GraphProcessor()
    
    # 带权重的边列表
    weighted_edges = [
        ('A', 'B', 2.0),
        ('B', 'C', 1.5),
        ('C', 'D', 3.0),
        ('D', 'A', 1.0),
        ('A', 'C', 4.0),
        ('B', 'D', 2.5)
    ]
    
    # 创建有向图
    graph = processor.create_graph_from_edges(weighted_edges, directed=True)
    
    # 运行各种中心性算法
    betweenness = processor.run_betweenness_centrality()
    print(f"介数中心性: {betweenness}")
    
    closeness = processor.run_closeness_centrality()
    print(f"接近中心性: {closeness}")
    
    degree = processor.run_degree_centrality()
    print(f"度中心性: {degree}")


def example_undirected_graph():
    """无向图示例"""
    print("\n=== 无向图示例 ===")
    
    processor = GraphProcessor()
    
    # 无向图的边列表
    undirected_edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('D', 'E'),
        ('E', 'A'),
        ('F', 'G'),
        ('G', 'H'),
        ('H', 'F')
    ]
    
    # 创建无向图
    graph = processor.create_graph_from_edges(undirected_edges, directed=False)
    
    # 运行连通分量算法（对于无向图）
    components = processor.run_connected_components()
    print(f"连通分量: {components}")
    
    # 计算最短路径
    path = processor.run_shortest_path('A', 'E')
    print(f"从A到E的最短路径: {path}")


def example_algorithm_runner():
    """使用通用算法运行器"""
    print("\n=== 通用算法运行器示例 ===")
    
    processor = GraphProcessor()
    
    # 创建图
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'A'),
        ('D', 'E'),
        ('E', 'F')
    ]
    
    processor.create_graph_from_edges(edges, directed=True)
    
    # 使用通用方法运行不同算法
    algorithms = ['pagerank', 'cc', 'betweenness', 'closeness', 'degree']
    
    for algo in algorithms:
        try:
            if algo == 'shortest_path':
                result = processor.run_algorithm(algo, source='A', target='C')
            else:
                result = processor.run_algorithm(algo)
            print(f"{algo.upper()} 结果: {result}")
        except Exception as e:
            print(f"运行 {algo} 时出错: {e}")


def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    processor = GraphProcessor()
    
    # 尝试在未初始化图的情况下运行算法
    try:
        processor.run_pagerank()
    except ValueError as e:
        print(f"预期的错误: {e}")
    
    # 尝试运行不存在的算法
    try:
        processor.create_graph_from_edges([('A', 'B')])
        processor.run_algorithm('nonexistent_algorithm')
    except ValueError as e:
        print(f"预期的错误: {e}")


def example_execution_plan():
    """执行计划示例"""
    print("\n=== 执行计划示例 ===")
    
    processor = GraphProcessor()
    
    # 创建图
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('D', 'A'),
        ('A', 'C'),
        ('B', 'D')
    ]
    
    processor.create_graph_from_edges(edges, directed=True)
    
    # 方法1: 直接创建执行计划
    plan = [
        {
            'algorithm': 'pagerank',
            'name': 'pagerank_scores',
            'params': {'alpha': 0.85}
        },
        {
            'algorithm': 'cc',
            'name': 'connected_components'
        },
        {
            'algorithm': 'betweenness',
            'name': 'betweenness_centrality'
        },
        {
            'algorithm': 'shortest_path',
            'name': 'path_a_to_d',
            'params': {'source': 'A', 'target': 'D'}
        }
    ]
    
    results = processor.execute_plan(plan)
    print(f"执行计划结果: {results}")
    
    # 方法2: 使用create_execution_plan创建计划
    algorithms = ['pagerank', 'cc', 'closeness', 'degree']
    algorithm_params = {
        'pagerank': {'alpha': 0.9},
        'cc': {}
    }
    result_names = ['pagerank_result', 'components', 'closeness_scores', 'degree_scores']
    
    plan2 = processor.create_execution_plan(algorithms, algorithm_params, result_names)
    results2 = processor.execute_plan(plan2)
    print(f"计划2结果: {results2}")


def example_complex_plan():
    """复杂执行计划示例"""
    print("\n=== 复杂执行计划示例 ===")
    
    processor = GraphProcessor()
    
    # 创建更复杂的图
    complex_edges = [
        ('A', 'B', 2.0),
        ('B', 'C', 1.5),
        ('C', 'D', 3.0),
        ('D', 'A', 1.0),
        ('A', 'C', 4.0),
        ('B', 'D', 2.5),
        ('E', 'F', 1.0),
        ('F', 'G', 2.0),
        ('G', 'E', 1.5)
    ]
    
    processor.create_graph_from_edges(complex_edges, directed=True)
    
    # 创建复杂的执行计划
    complex_plan = [
        {
            'algorithm': 'pagerank',
            'name': 'initial_pagerank',
            'params': {'alpha': 0.85}
        },
        {
            'algorithm': 'cc',
            'name': 'strong_components'
        },
        {
            'algorithm': 'betweenness',
            'name': 'betweenness_scores'
        },
        {
            'algorithm': 'closeness',
            'name': 'closeness_scores'
        },
        {
            'algorithm': 'degree',
            'name': 'degree_scores'
        },
        {
            'algorithm': 'shortest_path',
            'name': 'path_a_to_g',
            'params': {'source': 'A', 'target': 'G'}
        }
    ]
    
    results = processor.execute_plan(complex_plan)
    
    # 分析结果
    print("复杂执行计划结果分析:")
    for name, result in results.items():
        if isinstance(result, dict):
            print(f"{name}: {len(result)} 个节点的分数")
        elif isinstance(result, list):
            print(f"{name}: {len(result)} 个组件")
        elif isinstance(result, list) and len(result) > 0:
            print(f"{name}: 路径长度 {len(result)}")
        else:
            print(f"{name}: {result}")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_weighted_graph()
    example_undirected_graph()
    example_algorithm_runner()
    example_error_handling()
    example_execution_plan()
    example_complex_plan()
    
    print("\n=== 所有示例完成 ===") 