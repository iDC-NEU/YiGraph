#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试基于节点列表的子图抽取功能

演示三种应用场景：
1. 交易转账网络：提取账户 A、B、C 及其之间的转账记录
2. 社交网络：导出以"某大V"为中心的局部网络（使用原有方法）
3. 指定节点子图：获取包含节点列表 [A, B, C] 及其相互连边的子图
"""

from graph_query import Neo4jGraphClient, Neo4jConfig
import json


def test_scenario_1_transaction_network(client: Neo4jGraphClient):
    """
    场景1：交易转账网络
    提取账户 A、B、C 及其之间所有的相互转账记录，形成一个小型的资金互动网络
    """
    print("\n" + "="*80)
    print("场景1：交易转账网络 - 提取多个账户及其相互转账关系")
    print("="*80)
    
    # 假设我们要提取这三个账户之间的转账关系
    account_names = ["Collins Steven", "Nunez Mitchell", "Lee Alex"]
    
    print(f"\n目标账户: {account_names}")
    print("\n执行查询...")
    
    # 使用新的 subgraph_extract_by_nodes 方法
    result = client.subgraph_extract_by_nodes(
        label="Account",
        key="node_key",
        values=account_names,
        rel_type="TRANSFER",  # 只关注转账关系
        direction="both",     # 双向（A->B 和 B->A 都要）
        include_internal=False  # 不包含自己转给自己的情况
    )
    
    print(f"\n✅ 查询完成！")
    print(f"   节点数量: {result['node_count']}")
    print(f"   关系数量: {result['relationship_count']}")
    
    # 显示节点信息
    print(f"\n📊 节点详情:")
    for i, node in enumerate(result['nodes'], 1):
        print(f"   {i}. {node.get('node_key', 'N/A')} (余额: {node.get('balance', 'N/A')})")
    
    # 显示关系信息
    print(f"\n🔗 转账关系:")
    for i, rel in enumerate(result['relationships'], 1):
        print(f"   {i}. 金额: {rel.get('base_amt', 'N/A')}, 时间: {rel.get('timestamp', 'N/A')}")
    
    return result


def test_scenario_2_social_network_center(client: Neo4jGraphClient):
    """
    场景2：社交网络
    导出以"某大V"为中心，包括其关注者及关注者之间关系的局部网络拓扑
    """
    print("\n" + "="*80)
    print("场景2：社交网络 - 以某大V为中心的局部网络")
    print("="*80)
    
    # 使用原有的 subgraph_extract 方法（单中心节点）
    big_v_name = "Collins Steven"  # 假设这是大V
    
    print(f"\n中心节点: {big_v_name}")
    print(f"抽取范围: 2跳邻居")
    print("\n执行查询...")
    
    result = client.subgraph_extract(
        center=("Account", "node_key", big_v_name),
        hops=2,              # 2跳邻居
        direction="both",    # 双向关系
        limit_paths=100      # 限制路径数量
    )
    
    print(f"\n✅ 查询完成！")
    print(f"   节点数量: {len(result['nodes'])}")
    print(f"   关系数量: {len(result['relationships'])}")
    
    # 显示部分节点
    print(f"\n📊 部分节点 (前5个):")
    for i, node in enumerate(result['nodes'][:5], 1):
        print(f"   {i}. {node.get('node_key', 'N/A')}")
    
    if len(result['nodes']) > 5:
        print(f"   ... 还有 {len(result['nodes']) - 5} 个节点")
    
    return result


def test_scenario_3_specified_nodes_subgraph(client: Neo4jGraphClient):
    """
    场景3：指定节点子图
    获取包含节点列表 [A, B, C] 及其相互之间所有连边的子图
    """
    print("\n" + "="*80)
    print("场景3：指定节点子图 - 提取指定节点及其相互连边")
    print("="*80)
    
    # 指定要提取的节点列表
    node_list = ["Collins Steven", "Nunez Mitchell", "Lee Alex", "Smith John"]
    
    print(f"\n目标节点列表: {node_list}")
    print("\n执行查询...")
    
    # 使用新的 subgraph_extract_by_nodes 方法
    result = client.subgraph_extract_by_nodes(
        label="Account",
        key="node_key",
        values=node_list,
        direction="both",        # 双向关系
        include_internal=True    # 包含自环（如果有）
        # rel_type=None          # 不限制关系类型，获取所有类型的关系
    )
    
    print(f"\n✅ 查询完成！")
    print(f"   节点数量: {result['node_count']}")
    print(f"   关系数量: {result['relationship_count']}")
    
    # 显示节点信息
    print(f"\n📊 节点详情:")
    for i, node in enumerate(result['nodes'], 1):
        print(f"   {i}. {node.get('node_key', 'N/A')}")
    
    # 显示关系信息（按类型分组）
    print(f"\n🔗 关系详情:")
    rel_types = {}
    for rel in result['relationships']:
        rel_type = rel.get('type', 'UNKNOWN')
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    for rel_type, count in rel_types.items():
        print(f"   {rel_type}: {count} 条")
    
    return result


def test_with_natural_language(client: Neo4jGraphClient):
    """
    使用自然语言查询引擎测试
    """
    print("\n" + "="*80)
    print("自然语言查询测试")
    print("="*80)
    
    from nl_query_engine import NaturalLanguageQueryEngine, LLMInterface
    
    # 初始化引擎
    llm = LLMInterface()
    engine = NaturalLanguageQueryEngine(client, llm)
    engine.initialize()
    
    # 测试问题
    test_questions = [
        "提取账户 Collins Steven、Nunez Mitchell、Lee Alex 之间的转账关系",
        "获取 Collins Steven 周围2跳的子图",
        "抽取包含 Collins Steven 和 Nunez Mitchell 及其相互关系的子图"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        result = engine.ask(question)
        print(f"结果: {result.get('count', 0)} 条记录")


def main():
    """主测试函数"""
    print("="*80)
    print("基于节点列表的子图抽取功能测试")
    print("="*80)
    
    # 连接数据库
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password"  # 请修改为实际密码
    )
    
    try:
        with Neo4jGraphClient(config) as client:
            # 测试三种场景
            test_scenario_1_transaction_network(client)
            test_scenario_2_social_network_center(client)
            test_scenario_3_specified_nodes_subgraph(client)
            
            # 可选：测试自然语言查询
            # test_with_natural_language(client)
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
