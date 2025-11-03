#!/usr/bin/env python3
"""
测试社区查询功能
展示如何使用get_community_by_specific_id和get_communities_for_multiple_vertices方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aag.computing_engine.graphcomputation_processor import GraphProcessor
from aag.expert_search_engine.database.datatype import VertexData, EdgeData
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_graph():
    """创建一个测试图"""
    
    # 创建顶点数据
    vertices = [
        VertexData(vid=1, properties={'name': 'Alice', 'type': 'user'}),
        VertexData(vid=2, properties={'name': 'Bob', 'type': 'user'}),
        VertexData(vid=3, properties={'name': 'Charlie', 'type': 'user'}),
        VertexData(vid=4, properties={'name': 'David', 'type': 'user'}),
        VertexData(vid=5, properties={'name': 'Eve', 'type': 'user'}),
        VertexData(vid=6, properties={'name': 'Frank', 'type': 'user'}),
        VertexData(vid=7, properties={'name': 'Grace', 'type': 'user'}),
        VertexData(vid=8, properties={'name': 'Henry', 'type': 'user'}),
        VertexData(vid=9, properties={'name': 'Ivy', 'type': 'user'}),
        VertexData(vid=10, properties={'name': 'Jack', 'type': 'user'}),
        VertexData(vid=11, properties={'name': 'Kate', 'type': 'user'}),
        VertexData(vid=12, properties={'name': 'Liam', 'type': 'user'}),
        VertexData(vid=13, properties={'name': 'Mia', 'type': 'user'}),
        VertexData(vid=14, properties={'name': 'Noah', 'type': 'user'}),
        VertexData(vid=15, properties={'name': 'Olivia', 'type': 'user'}),
    ]
    
    # 创建边数据 - 模拟三个社区
    edges = [
        # 社区1: 1-5 (紧密连接)
        EdgeData(src=1, dst=2, properties={'weight': 1.0}),
        EdgeData(src=1, dst=3, properties={'weight': 1.0}),
        EdgeData(src=2, dst=3, properties={'weight': 1.0}),
        EdgeData(src=2, dst=4, properties={'weight': 1.0}),
        EdgeData(src=3, dst=4, properties={'weight': 1.0}),
        EdgeData(src=4, dst=5, properties={'weight': 1.0}),
        EdgeData(src=1, dst=5, properties={'weight': 1.0}),
        
        # 社区2: 6-10 (紧密连接)
        EdgeData(src=6, dst=7, properties={'weight': 1.0}),
        EdgeData(src=6, dst=8, properties={'weight': 1.0}),
        EdgeData(src=7, dst=8, properties={'weight': 1.0}),
        EdgeData(src=7, dst=9, properties={'weight': 1.0}),
        EdgeData(src=8, dst=9, properties={'weight': 1.0}),
        EdgeData(src=9, dst=10, properties={'weight': 1.0}),
        EdgeData(src=6, dst=10, properties={'weight': 1.0}),
        
        # 社区3: 11-15 (紧密连接)
        EdgeData(src=11, dst=12, properties={'weight': 1.0}),
        EdgeData(src=11, dst=13, properties={'weight': 1.0}),
        EdgeData(src=12, dst=13, properties={'weight': 1.0}),
        EdgeData(src=12, dst=14, properties={'weight': 1.0}),
        EdgeData(src=13, dst=14, properties={'weight': 1.0}),
        EdgeData(src=14, dst=15, properties={'weight': 1.0}),
        EdgeData(src=11, dst=15, properties={'weight': 1.0}),
        
        # 社区间的少量连接
        EdgeData(src=3, dst=6, properties={'weight': 0.5}),
        EdgeData(src=5, dst=8, properties={'weight': 0.5}),
        EdgeData(src=7, dst=11, properties={'weight': 0.5}),
        EdgeData(src=10, dst=12, properties={'weight': 0.5}),
    ]
    
    return vertices, edges

def test_single_vertex_community():
    """测试单个节点的社区查询"""
    
    logger.info("=== 测试单个节点的社区查询 ===")
    
    # 创建图处理器
    processor = GraphProcessor()
    
    # 创建测试图
    vertices, edges = create_test_graph()
    
    # 从边创建图
    graph = processor.create_graph_from_edges(vertices, edges, directed=False)
    
    # 设置查询节点
    processor.query_vertices = 15
    
    # 查询节点15的社区信息
    result = processor.get_community_by_specific_id()
    
    print("\n节点15的社区信息:")
    print(f"  节点ID: {result['vertex_id']}")
    print(f"  社区ID: {result['community_id']}")
    print(f"  社区大小: {result['community_size']}")
    print(f"  社区成员: {result['community_members']}")
    print(f"  社区内邻居: {result['neighbors_in_community']}")
    print(f"  社区外邻居: {result['neighbors_outside_community']}")
    print(f"  总邻居数: {result['total_neighbors']}")
    print(f"  社区凝聚力: {result['community_cohesion']:.3f}")
    
    return result

def test_multiple_vertices_community():
    """测试多个节点的社区查询"""
    
    logger.info("\n=== 测试多个节点的社区查询 ===")
    
    # 创建图处理器
    processor = GraphProcessor()
    
    # 创建测试图
    vertices, edges = create_test_graph()
    
    # 从边创建图
    graph = processor.create_graph_from_edges(vertices, edges, directed=False)
    
    # 查询多个节点的社区信息
    query_vertices = [1, 6, 11, 15, 20]  # 包含一个不存在的节点20
    results = processor.get_communities_for_multiple_vertices(query_vertices)
    
    print(f"\n批量查询结果 (查询节点: {query_vertices}):")
    
    for vertex_id, result in results.items():
        print(f"\n节点 {vertex_id}:")
        if 'error' in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  社区ID: {result['community_id']}")
            print(f"  社区大小: {result['community_size']}")
            print(f"  社区内邻居: {result['neighbors_in_community']}")
            print(f"  社区外邻居: {result['neighbors_outside_community']}")
            print(f"  社区凝聚力: {result['community_cohesion']:.3f}")
    
    return results

def test_community_analysis():
    """测试社区分析功能"""
    
    logger.info("\n=== 测试社区分析功能 ===")
    
    # 创建图处理器
    processor = GraphProcessor()
    
    # 创建测试图
    vertices, edges = create_test_graph()
    
    # 从边创建图
    graph = processor.create_graph_from_edges(vertices, edges, directed=False)
    
    # 运行Louvain算法
    louvain_result = processor.run_louvain_community_detection()
    
    # 获取统计信息
    stats = processor.get_community_statistics(louvain_result)
    
    print("\n社区检测统计信息:")
    print(f"  总社区数: {stats['total_communities']}")
    print(f"  最大社区大小: {stats['largest_community_size']}")
    print(f"  最小社区大小: {stats['smallest_community_size']}")
    print(f"  平均社区大小: {stats['average_community_size']:.2f}")
    print(f"  模块度: {stats['modularity']:.4f}")
    print(f"  大小分布: {stats['size_distribution']}")
    
    # 分析每个社区
    print("\n各社区详细信息:")
    for i, community in enumerate(louvain_result['communities']):
        print(f"  社区 {i+1}: 大小={len(community)}, 成员={sorted(community)}")
    
    return louvain_result, stats

if __name__ == "__main__":
    try:
        # 测试单个节点社区查询
        single_result = test_single_vertex_community()
        
        # 测试多个节点社区查询
        multiple_results = test_multiple_vertices_community()
        
        # 测试社区分析
        louvain_result, stats = test_community_analysis()
        
        logger.info("所有测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc() 