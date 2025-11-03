#!/usr/bin/env python3
"""
测试YAML文件读取功能

这个文件用于测试读取 task_types.yaml 和 algorithms.yaml 文件的功能
"""

import yaml
import os
import sys
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append('/home/chency/GraphLLM')

try:
    from aag.utils.file_operation import read_yaml
except ImportError:
    print("警告: 无法导入 aag.utils.file_operation.read_yaml，将使用备用方法")


def read_yaml_fallback(file_path: str, key: str = None):
    """备用YAML读取函数"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    if key:
        return data[key] if key in data else data
    return data


def test_read_task_types_yaml():
    """测试读取 task_types.yaml 文件"""
    print("=== 测试读取 task_types.yaml ===")
    
    file_path = "/home/chency/GraphLLM/aag/knowledge_base/task_types.yaml"
    
    try:
        # 方法1：使用自定义的 read_yaml 函数
        print("方法1：使用自定义 read_yaml 函数")
        try:
            task_types = read_yaml(file_path, "task_types")
        except:
            task_types = read_yaml_fallback(file_path)
        
        print(f"成功读取 task_types，共 {len(task_types)} 个任务类型")
        
        # 显示所有任务类型
        for i, task_type in enumerate(task_types):
            print(f"\n任务类型 {i+1}:")
            print(f"  ID: {task_type.get('id', 'N/A')}")
            print(f"  类型: {task_type.get('task_type', 'N/A')}")
            print(f"  描述: {task_type.get('description', 'N/A')}")
            print(f"  算法: {task_type.get('algorithm', [])}")
        
        return task_types
        
    except Exception as e:
        print(f"读取失败: {e}")
        return None


def test_read_algorithms_yaml():
    """测试读取 algorithms.yaml 文件"""
    print("\n=== 测试读取 algorithms.yaml ===")
    
    file_path = "/home/chency/GraphLLM/aag/knowledge_base/algorithms.yaml"
    
    try:
        # 使用备用方法读取
        print("使用 yaml.safe_load 读取")
        algorithms = read_yaml_fallback(file_path)
        print(f"成功读取 algorithms，共 {len(algorithms)} 个算法")
        
        # 显示所有算法
        for i, algorithm in enumerate(algorithms):
            print(f"\n算法 {i+1}:")
            print(f"  ID: {algorithm.get('id', 'N/A')}")
            print(f"  类别: {algorithm.get('category', 'N/A')}")
            print(f"  支持引擎: {algorithm.get('support_engines', 'N/A')}")
            
            # 显示描述信息
            description = algorithm.get('description', {})
            if isinstance(description, dict):
                print(f"  原理: {description.get('principle', 'N/A')[:100]}...")
                print(f"  意义: {description.get('meaning', 'N/A')[:100]}...")
            
            # 显示输入参数
            input_schema = algorithm.get('inputSchema', {})
            if input_schema:
                required_params = input_schema.get('required', [])
                print(f"  必需参数: {required_params}")
        
        return algorithms
        
    except Exception as e:
        print(f"读取失败: {e}")
        return None


def test_yaml_structure_analysis():
    """分析YAML文件结构"""
    print("\n=== YAML文件结构分析 ===")
    
    # 分析 task_types.yaml
    task_types_file = "/home/chency/GraphLLM/aag/knowledge_base/task_types.yaml"
    algorithms_file = "/home/chency/GraphLLM/aag/knowledge_base/algorithms.yaml"
    
    for file_path, file_name in [(task_types_file, "task_types.yaml"), (algorithms_file, "algorithms.yaml")]:
        print(f"\n分析 {file_name}:")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            print(f"  数据类型: {type(data)}")
            
            if isinstance(data, list):
                print(f"  列表长度: {len(data)}")
                if data:
                    print(f"  第一个元素类型: {type(data[0])}")
                    if isinstance(data[0], dict):
                        print(f"  第一个元素的键: {list(data[0].keys())}")
            elif isinstance(data, dict):
                print(f"  字典键: {list(data.keys())}")
            
        except Exception as e:
            print(f"  分析失败: {e}")


def test_algorithm_categorization():
    """测试算法分类功能"""
    print("\n=== 算法分类测试 ===")
    
    try:
        # 读取两个文件
        task_types_file = "/home/chency/GraphLLM/aag/knowledge_base/task_types.yaml"
        algorithms_file = "/home/chency/GraphLLM/aag/knowledge_base/algorithms.yaml"
        
        with open(task_types_file, 'r', encoding='utf-8') as file:
            task_types = yaml.safe_load(file)
        
        with open(algorithms_file, 'r', encoding='utf-8') as file:
            algorithms = yaml.safe_load(file)
        
        # 创建算法ID到类别的映射
        algorithm_to_category = {}
        for algorithm in algorithms:
            algorithm_to_category[algorithm['id']] = algorithm.get('category', 'unknown')
        
        # 验证任务类型中的算法是否在算法文件中存在
        print("验证任务类型中的算法:")
        for task_type in task_types:
            task_id = task_type['id']
            task_name = task_type['task_type']
            task_algorithms = task_type.get('algorithm', [])
            
            print(f"\n任务类型: {task_name} ({task_id})")
            print(f"  包含算法: {task_algorithms}")
            
            for alg_id in task_algorithms:
                if alg_id in algorithm_to_category:
                    category = algorithm_to_category[alg_id]
                    print(f"    ✓ {alg_id} -> 类别: {category}")
                else:
                    print(f"    ✗ {alg_id} -> 未找到")
        
        # 统计各类别的算法数量
        print("\n算法类别统计:")
        category_count = {}
        for algorithm in algorithms:
            category = algorithm.get('category', 'unknown')
            category_count[category] = category_count.get(category, 0) + 1
        
        for category, count in category_count.items():
            print(f"  {category}: {count} 个算法")
        
    except Exception as e:
        print(f"算法分类测试失败: {e}")


def test_search_functionality():
    """测试搜索功能"""
    print("\n=== 搜索功能测试 ===")
    
    try:
        # 读取算法文件
        algorithms_file = "/home/chency/GraphLLM/aag/knowledge_base/algorithms.yaml"
        
        with open(algorithms_file, 'r', encoding='utf-8') as file:
            algorithms = yaml.safe_load(file)
        
        # 测试按类别搜索
        def search_by_category(category: str):
            return [alg for alg in algorithms if alg.get('category') == category]
        
        # 测试按ID搜索
        def search_by_id(algorithm_id: str):
            return next((alg for alg in algorithms if alg.get('id') == algorithm_id), None)
        
        # 测试按关键词搜索
        def search_by_keyword(keyword: str):
            results = []
            for alg in algorithms:
                # 在描述中搜索关键词
                description = alg.get('description', {})
                if isinstance(description, dict):
                    principle = description.get('principle', '').lower()
                    meaning = description.get('meaning', '').lower()
                    if keyword.lower() in principle or keyword.lower() in meaning:
                        results.append(alg)
            return results
        
        # 执行搜索测试
        print("1. 按类别搜索 'traversal':")
        traversal_algs = search_by_category('traversal')
        for alg in traversal_algs:
            print(f"   - {alg['id']}: {alg.get('description', {}).get('principle', '')[:50]}...")
        
        print("\n2. 按ID搜索 'pagerank':")
        pagerank = search_by_id('pagerank')
        if pagerank:
            print(f"   找到: {pagerank['id']} - {pagerank.get('category', 'unknown')}")
        
        print("\n3. 按关键词搜索 'shortest path':")
        shortest_path_algs = search_by_keyword('shortest path')
        for alg in shortest_path_algs:
            print(f"   - {alg['id']}: {alg.get('category', 'unknown')}")
        
    except Exception as e:
        print(f"搜索功能测试失败: {e}")


def test_specific_algorithms():
    """测试特定算法的详细信息"""
    print("\n=== 特定算法详细信息测试 ===")
    
    try:
        algorithms_file = "/home/chency/GraphLLM/aag/knowledge_base/algorithms.yaml"
        
        with open(algorithms_file, 'r', encoding='utf-8') as file:
            algorithms = yaml.safe_load(file)
        
        # 测试几个重要算法
        test_algorithm_ids = ['pagerank', 'bfs', 'dijkstra_path', 'louvain']
        
        for alg_id in test_algorithm_ids:
            algorithm = next((alg for alg in algorithms if alg.get('id') == alg_id), None)
            if algorithm:
                print(f"\n算法: {alg_id}")
                print(f"  类别: {algorithm.get('category', 'N/A')}")
                print(f"  支持引擎: {algorithm.get('support_engines', 'N/A')}")
                
                # 输入参数
                input_schema = algorithm.get('inputSchema', {})
                if input_schema:
                    required = input_schema.get('required', [])
                    parameters = input_schema.get('parameters', {})
                    print(f"  必需参数: {required}")
                    print(f"  参数数量: {len(parameters)}")
                
                # 输出
                output = algorithm.get('output', {})
                if output:
                    print(f"  输出类型: {output.get('type', 'N/A')}")
                    print(f"  输出描述: {output.get('description', 'N/A')[:100]}...")
            else:
                print(f"\n算法 {alg_id} 未找到")
        
    except Exception as e:
        print(f"特定算法测试失败: {e}")


def main():
    """主测试函数"""
    print("开始YAML文件读取测试...")
    
    # 检查文件是否存在
    task_types_file = "/home/chency/GraphLLM/aag/knowledge_base/task_types.yaml"
    algorithms_file = "/home/chency/GraphLLM/aag/knowledge_base/algorithms.yaml"
    
    print(f"检查文件存在性:")
    print(f"  task_types.yaml: {'存在' if os.path.exists(task_types_file) else '不存在'}")
    print(f"  algorithms.yaml: {'存在' if os.path.exists(algorithms_file) else '不存在'}")
    
    # 执行各项测试
    task_types = test_read_task_types_yaml()
    algorithms = test_read_algorithms_yaml()
    test_yaml_structure_analysis()
    test_algorithm_categorization()
    # test_search_functionality()
    # test_specific_algorithms()
    
    print("\n=== 测试总结 ===")
    print(f"task_types.yaml 读取: {'成功' if task_types else '失败'}")
    print(f"algorithms.yaml 读取: {'成功' if algorithms else '失败'}")
    
    if task_types and algorithms:
        print("所有测试完成！YAML文件读取功能正常。")
        print(f"共读取 {len(task_types)} 个任务类型和 {len(algorithms)} 个算法。")
    else:
        print("部分测试失败，请检查文件路径和格式。")


if __name__ == "__main__":
    main()
