#!/usr/bin/env python3
"""
执行计划功能演示脚本
展示如何使用GraphProcessor的执行计划功能
"""

import logging
from graph_processor import GraphProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def demo_basic_execution_plan():
    """演示基本执行计划功能"""
    print("=" * 50)
    print("基本执行计划演示")
    print("=" * 50)
    
    # 创建图处理器
    processor = GraphProcessor()
    
    # 创建示例图
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('D', 'A'),
        ('A', 'C'),
        ('B', 'D'),
        ('E', 'F'),
        ('F', 'G')
    ]
    
    processor.create_graph_from_edges(edges, directed=True)
    print(f"创建了包含 {processor.graph.number_of_nodes()} 个节点和 {processor.graph.number_of_edges()} 条边的图")
    
    # 定义执行计划
    plan = [
        {
            'algorithm': 'pagerank',
            'name': 'importance_scores',
            'params': {'alpha': 0.85}
        },
        {
            'algorithm': 'cc',
            'name': 'community_structure'
        },
        {
            'algorithm': 'betweenness',
            'name': 'bridge_nodes'
        },
        {
            'algorithm': 'shortest_path',
            'name': 'a_to_d_path',
            'params': {'source': 'A', 'target': 'D'}
        }
    ]
    
    print("\n执行计划:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step['algorithm']} -> {step['name']}")
    
    # 执行计划
    print("\n开始执行计划...")
    results = processor.execute_plan(plan)
    
    # 分析结果
    print("\n执行结果分析:")
    for name, result in results.items():
        if isinstance(result, dict):
            print(f"  {name}: {len(result)} 个节点的分数")
            # 显示前3个最高分数的节点
            sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]
            for node, score in sorted_items:
                print(f"    {node}: {score:.4f}")
        elif isinstance(result, list):
            print(f"  {name}: {len(result)} 个组件")
            for i, component in enumerate(result):
                print(f"    组件 {i+1}: {component}")
        else:
            print(f"  {name}: {result}")


def demo_advanced_execution_plan():
    """演示高级执行计划功能"""
    print("\n" + "=" * 50)
    print("高级执行计划演示")
    print("=" * 50)
    
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
        ('G', 'E', 1.5),
        ('H', 'I', 1.0),
        ('I', 'J', 2.0)
    ]
    
    processor.create_graph_from_edges(complex_edges, directed=True)
    print(f"创建了包含 {processor.graph.number_of_nodes()} 个节点和 {processor.graph.number_of_edges()} 条边的复杂图")
    
    # 使用create_execution_plan创建计划
    algorithms = ['pagerank', 'cc', 'closeness', 'degree', 'shortest_path']
    algorithm_params = {
        'pagerank': {'alpha': 0.9},
        'shortest_path': {'source': 'A', 'target': 'G'}
    }
    result_names = ['node_importance', 'communities', 'centrality', 'degree_centrality', 'longest_path']
    
    plan = processor.create_execution_plan(algorithms, algorithm_params, result_names)
    
    print("\n自动生成的执行计划:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step['algorithm']} -> {step['name']}")
        if step['params']:
            print(f"     参数: {step['params']}")
    
    # 执行计划
    print("\n开始执行计划...")
    results = processor.execute_plan(plan)
    
    # 详细分析结果
    print("\n详细结果分析:")
    
    # PageRank结果
    if 'node_importance' in results:
        pagerank = results['node_importance']
        print(f"\n节点重要性排名 (PageRank):")
        sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        for i, (node, score) in enumerate(sorted_pagerank, 1):
            print(f"  {i}. {node}: {score:.4f}")
    
    # 连通分量结果
    if 'communities' in results:
        communities = results['communities']
        print(f"\n社区结构 (连通分量):")
        for i, community in enumerate(communities, 1):
            print(f"  社区 {i}: {community}")
    
    # 中心性结果
    if 'centrality' in results:
        centrality = results['centrality']
        print(f"\n接近中心性排名:")
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        for i, (node, score) in enumerate(sorted_centrality, 1):
            print(f"  {i}. {node}: {score:.4f}")
    
    # 度中心性结果
    if 'degree_centrality' in results:
        degree = results['degree_centrality']
        print(f"\n度中心性排名:")
        sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
        for i, (node, score) in enumerate(sorted_degree, 1):
            print(f"  {i}. {node}: {score:.4f}")
    
    # 最短路径结果
    if 'longest_path' in results and results['longest_path']:
        path = results['longest_path']
        print(f"\n从A到G的最短路径:")
        print(f"  路径: {' -> '.join(path)}")
        print(f"  长度: {len(path) - 1}")


def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 50)
    print("错误处理演示")
    print("=" * 50)
    
    processor = GraphProcessor()
    
    # 测试未初始化图的情况
    print("1. 测试未初始化图:")
    try:
        plan = [{'algorithm': 'pagerank'}]
        processor.execute_plan(plan)
    except ValueError as e:
        print(f"   错误: {e}")
    
    # 测试无效算法
    print("\n2. 测试无效算法:")
    try:
        processor.create_graph_from_edges([('A', 'B')])
        plan = [{'algorithm': 'invalid_algorithm'}]
        processor.execute_plan(plan)
    except ValueError as e:
        print(f"   错误: {e}")
    
    # 测试缺少算法名称
    print("\n3. 测试缺少算法名称:")
    try:
        plan = [
            {'algorithm': 'pagerank', 'name': 'valid'},
            {'name': 'invalid'},  # 缺少algorithm
            {'algorithm': 'cc', 'name': 'valid2'}
        ]
        results = processor.execute_plan(plan)
        print(f"   结果: 成功执行了 {len(results)} 个算法")
    except ValueError as e:
        print(f"   错误: {e}")


if __name__ == "__main__":
    print("GraphProcessor 执行计划功能演示")
    print("=" * 60)
    
    # 运行所有演示
    demo_basic_execution_plan()
    demo_advanced_execution_plan()
    demo_error_handling()
    
    print("\n" + "=" * 60)
    print("演示完成！") 