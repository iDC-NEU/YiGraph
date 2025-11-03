#!/usr/bin/env python3
"""
GraphProcessor测试文件
测试图处理器的各种功能
"""

import unittest
import logging
from aag.computing_engine.graphcomputation_processor import GraphProcessor

# 设置日志
logger = logging.getLogger(__name__)


class TestGraphProcessor(unittest.TestCase):
    """GraphProcessor测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.processor = GraphProcessor()
        self.simple_edges = [
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'D'),
            ('D', 'A')
        ]
        self.weighted_edges = [
            ('A', 'B', 2.0),
            ('B', 'C', 1.5),
            ('C', 'D', 3.0),
            ('D', 'A', 1.0)
        ]
    
    def test_create_directed_graph(self):
        """测试创建有向图"""
        graph = self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        
        self.assertIsNotNone(graph)
        self.assertTrue(self.processor.is_directed)
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph.number_of_edges(), 4)
    
    def test_create_undirected_graph(self):
        """测试创建无向图"""
        graph = self.processor.create_graph_from_edges(self.simple_edges, directed=False)
        
        self.assertIsNotNone(graph)
        self.assertFalse(self.processor.is_directed)
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph.number_of_edges(), 4)
    
    def test_create_weighted_graph(self):
        """测试创建带权重的图"""
        graph = self.processor.create_graph_from_edges(self.weighted_edges, directed=True)
        
        self.assertIsNotNone(graph)
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph.number_of_edges(), 4)
        
        # 检查权重
        self.assertEqual(graph['A']['B']['weight'], 2.0)
        self.assertEqual(graph['B']['C']['weight'], 1.5)
    
    def test_pagerank(self):
        """测试PageRank算法"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        pagerank_scores = self.processor.run_pagerank()
        
        self.assertIsInstance(pagerank_scores, dict)
        self.assertEqual(len(pagerank_scores), 4)
        
        # 检查所有分数都在0和1之间
        for score in pagerank_scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_connected_components_directed(self):
        """测试有向图的连通分量"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        components = self.processor.run_connected_components()
        
        self.assertIsInstance(components, list)
        self.assertGreater(len(components), 0)
        
        # 检查所有节点都在某个连通分量中
        all_nodes = set()
        for component in components:
            all_nodes.update(component)
        
        expected_nodes = {'A', 'B', 'C', 'D'}
        self.assertEqual(all_nodes, expected_nodes)
    
    def test_connected_components_undirected(self):
        """测试无向图的连通分量"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=False)
        components = self.processor.run_connected_components()
        
        self.assertIsInstance(components, list)
        self.assertGreater(len(components), 0)
    
    def test_shortest_path(self):
        """测试最短路径"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        path = self.processor.run_shortest_path('A', 'D')
        
        self.assertIsInstance(path, list)
        self.assertEqual(path[0], 'A')
        self.assertEqual(path[-1], 'D')
    
    def test_shortest_path_nonexistent(self):
        """测试不存在路径的情况"""
        # 创建不连通的图
        edges = [('A', 'B'), ('C', 'D')]
        self.processor.create_graph_from_edges(edges, directed=True)
        path = self.processor.run_shortest_path('A', 'D')
        
        self.assertIsNone(path)
    
    def test_betweenness_centrality(self):
        """测试介数中心性"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        betweenness = self.processor.run_betweenness_centrality()
        
        self.assertIsInstance(betweenness, dict)
        self.assertEqual(len(betweenness), 4)
        
        # 检查所有分数都在0和1之间
        for score in betweenness.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_closeness_centrality(self):
        """测试接近中心性"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        closeness = self.processor.run_closeness_centrality()
        
        self.assertIsInstance(closeness, dict)
        self.assertEqual(len(closeness), 4)
        
        # 检查所有分数都在0和1之间
        for score in closeness.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_degree_centrality(self):
        """测试度中心性"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        degree = self.processor.run_degree_centrality()
        
        self.assertIsInstance(degree, dict)
        self.assertEqual(len(degree), 4)
        
        # 检查所有分数都在0和1之间
        for score in degree.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_run_algorithm(self):
        """测试通用算法运行器"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        
        # 测试PageRank
        result = self.processor.run_algorithm('pagerank')
        self.assertIsInstance(result, dict)
        
        # 测试连通分量
        result = self.processor.run_algorithm('cc')
        self.assertIsInstance(result, list)
        
        # 测试最短路径
        result = self.processor.run_algorithm('shortest_path', source='A', target='D')
        self.assertIsInstance(result, list)
    
    def test_run_algorithm_invalid(self):
        """测试无效算法"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        
        with self.assertRaises(ValueError):
            self.processor.run_algorithm('invalid_algorithm')
    
    def test_get_graph_info(self):
        """测试获取图信息"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        info = self.processor.get_graph_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('节点数', info)
        self.assertIn('边数', info)
        self.assertIn('图类型', info)
        self.assertEqual(info['节点数'], 4)
        self.assertEqual(info['边数'], 4)
    
    def test_get_graph_info_no_graph(self):
        """测试未初始化图时获取信息"""
        info = self.processor.get_graph_info()
        self.assertIn('error', info)
    
    def test_error_handling_no_graph(self):
        """测试未初始化图时的错误处理"""
        with self.assertRaises(ValueError):
            self.processor.run_pagerank()
        
        with self.assertRaises(ValueError):
            self.processor.run_connected_components()
        
        with self.assertRaises(ValueError):
            self.processor.run_shortest_path('A', 'B')
    
    def test_create_execution_plan(self):
        """测试创建执行计划"""
        algorithms = ['pagerank', 'cc', 'betweenness']
        algorithm_params = {
            'pagerank': {'alpha': 0.9},
            'cc': {}
        }
        result_names = ['pagerank_result', 'components', 'betweenness_scores']
        
        plan = self.processor.create_execution_plan(algorithms, algorithm_params, result_names)
        
        self.assertIsInstance(plan, list)
        self.assertEqual(len(plan), 3)
        
        # 检查第一个步骤
        self.assertEqual(plan[0]['algorithm'], 'pagerank')
        self.assertEqual(plan[0]['name'], 'pagerank_result')
        self.assertEqual(plan[0]['params'], {'alpha': 0.9})
        
        # 检查第二个步骤
        self.assertEqual(plan[1]['algorithm'], 'cc')
        self.assertEqual(plan[1]['name'], 'components')
        self.assertEqual(plan[1]['params'], {})
    
    def test_execute_plan(self):
        """测试执行计划"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        
        plan = [
            {
                'algorithm': 'pagerank',
                'name': 'pagerank_scores'
            },
            {
                'algorithm': 'cc',
                'name': 'components'
            }
        ]
        
        results = self.processor.execute_plan(plan)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn('pagerank_scores', results)
        self.assertIn('components', results)
        
        # 检查结果类型
        self.assertIsInstance(results['pagerank_scores'], dict)
        self.assertIsInstance(results['components'], list)
    
    def test_execute_plan_with_params(self):
        """测试带参数的执行计划"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        
        plan = [
            {
                'algorithm': 'pagerank',
                'name': 'pagerank_scores',
                'params': {'alpha': 0.9}
            },
            {
                'algorithm': 'shortest_path',
                'name': 'path',
                'params': {'source': 'A', 'target': 'D'}
            }
        ]
        
        results = self.processor.execute_plan(plan)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn('pagerank_scores', results)
        self.assertIn('path', results)
    
    def test_execute_plan_error_handling(self):
        """测试执行计划的错误处理"""
        # 未初始化图
        plan = [{'algorithm': 'pagerank'}]
        with self.assertRaises(ValueError):
            self.processor.execute_plan(plan)
        
        # 初始化图后测试无效算法
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        invalid_plan = [{'algorithm': 'invalid_algorithm'}]
        with self.assertRaises(ValueError):
            self.processor.execute_plan(invalid_plan)
    
    def test_execute_plan_missing_algorithm(self):
        """测试缺少算法名称的执行计划"""
        self.processor.create_graph_from_edges(self.simple_edges, directed=True)
        
        plan = [
            {'algorithm': 'pagerank', 'name': 'pagerank_scores'},
            {'name': 'missing_algorithm'},  # 缺少算法名称
            {'algorithm': 'cc', 'name': 'components'}
        ]
        
        # 现在应该抛出异常而不是跳过
        with self.assertRaises(ValueError):
            self.processor.execute_plan(plan)


if __name__ == '__main__':
    unittest.main() 