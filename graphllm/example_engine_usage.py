"""
GraphLLM Engine 使用示例
展示如何使用端到端的Graph Analysis + RAG + LLM框架
"""

import os
import sys
from typing import List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphllm.graphllm_engine import GraphLLMEngine, create_engine_config, load_config_from_yaml


def example_basic_usage():
    """基础使用示例"""
    print("=== GraphLLM Engine 基础使用示例 ===")
    
    # 1. 创建配置
    config = create_engine_config(
        graph_space_name="example_space",
        vector_collection_name="example_collection",
        llm_model="llama3.1:70b",
        embedding_model="BAAI/bge-large-en-v1.5",
        openai_api_key="your-openai-api-key",  # 替换为你的API key
        llm_device="cuda:0",
        embed_device="cuda:0"
    )
    
    # 2. 初始化Engine
    engine = GraphLLMEngine(config)
    
    # 3. 示例文档
    sample_documents = [
        "Graph Neural Networks (GNNs) are a class of neural networks designed to work with graph-structured data.",
        "GNNs can be used for node classification, link prediction, and graph classification tasks.",
        "Popular GNN architectures include Graph Convolutional Networks (GCN), GraphSAGE, and Graph Attention Networks (GAT).",
        "Knowledge graphs represent information as entities and relationships between them.",
        "RAG (Retrieval-Augmented Generation) combines retrieval and generation for better question answering."
    ]
    
    # 4. 处理文档（构建知识图谱和向量索引）
    print("\n--- 处理文档 ---")
    engine.process_documents(sample_documents)
    
    # 5. 查询示例
    questions = [
        "What are Graph Neural Networks?",
        "What are the main applications of GNNs?",
        "How does RAG work?",
        "What is the relationship between GNNs and knowledge graphs?"
    ]
    
    print("\n--- 查询示例 ---")
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        
        # 使用图检索和向量检索
        result = engine.query(question, use_graph=True, use_vector=True)
        
        print(f"回答: {result['answer']}")
        print(f"检索节点数: {result['retrieved_nodes']}")
        print(f"检索时间: {result['retrieval_time']:.3f}s")
        print(f"生成时间: {result['generation_time']:.3f}s")
        print(f"总时间: {result['total_time']:.3f}s")
    
    # 6. 性能摘要
    print("\n--- 性能摘要 ---")
    summary = engine.get_performance_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 7. 清理资源
    engine.shutdown()


def example_config_file_usage():
    """使用配置文件示例"""
    print("\n=== GraphLLM Engine 配置文件使用示例 ===")
    
    # 假设配置文件路径
    config_file = "config_template.yaml"
    
    try:
        # 从配置文件加载
        config = load_config_from_yaml(config_file)
        engine = GraphLLMEngine(config)
        
        print("✓ 从配置文件成功初始化Engine")
        
        # 测试查询
        result = engine.query("What is Graph Neural Network?")
        print(f"测试查询结果: {result['answer']}")
        
        engine.shutdown()
        
    except Exception as e:
        print(f"配置文件使用失败: {e}")
        print("请确保配置文件存在且格式正确")


def example_advanced_usage():
    """高级使用示例 - 自定义配置和策略"""
    print("\n=== GraphLLM Engine 高级使用示例 ===")
    
    # 自定义配置
    config = create_engine_config(
        graph_space_name="advanced_space",
        vector_collection_name="advanced_collection",
        llm_model="llama3.1:70b",
        embedding_model="BAAI/bge-large-en-v1.5",
        # 图RAG配置
        graph_k_hop=3,  # 3跳图遍历
        graph_pruning=50,  # 更严格的剪枝
        graph_data_type="qa",
        graph_pruning_mode="embedding_for_perentity",
        # 向量RAG配置
        vector_k_similarity=10,  # 更多相似文档
        vector_data_type="summary",
        # 模型配置
        llm_device="cuda:0",
        embed_device="cuda:0",
        chunk_size=1024,  # 更大的chunk
        chunk_overlap=50,
        embed_batch_size=32
    )
    
    engine = GraphLLMEngine(config)
    
    # 示例：只使用图检索
    print("\n--- 仅图检索示例 ---")
    question = "What are the key components of Graph Neural Networks?"
    result_graph_only = engine.query(question, use_graph=True, use_vector=False)
    print(f"问题: {question}")
    print(f"回答: {result_graph_only['answer']}")
    print(f"图检索节点数: {result_graph_only['retrieved_nodes']}")
    
    # 示例：只使用向量检索
    print("\n--- 仅向量检索示例 ---")
    result_vector_only = engine.query(question, use_graph=False, use_vector=True)
    print(f"问题: {question}")
    print(f"回答: {result_vector_only['answer']}")
    print(f"向量检索节点数: {result_vector_only['retrieved_nodes']}")
    
    # 示例：混合检索
    print("\n--- 混合检索示例 ---")
    result_hybrid = engine.query(question, use_graph=True, use_vector=True)
    print(f"问题: {question}")
    print(f"回答: {result_hybrid['answer']}")
    print(f"混合检索节点数: {result_hybrid['retrieved_nodes']}")
    
    engine.shutdown()


def example_batch_processing():
    """批处理示例"""
    print("\n=== GraphLLM Engine 批处理示例 ===")
    
    config = create_engine_config(
        graph_space_name="batch_space",
        vector_collection_name="batch_collection"
    )
    
    engine = GraphLLMEngine(config)
    
    # 批量问题
    batch_questions = [
        "Explain the concept of graph convolution",
        "What is the difference between GCN and GAT?",
        "How do knowledge graphs enhance RAG systems?",
        "What are the challenges in graph neural networks?",
        "Explain the attention mechanism in GAT"
    ]
    
    print("--- 批量查询处理 ---")
    results = []
    
    for i, question in enumerate(batch_questions, 1):
        print(f"\n处理问题 {i}/{len(batch_questions)}: {question}")
        result = engine.query(question)
        results.append(result)
        
        print(f"✓ 完成 - 总时间: {result['total_time']:.3f}s")
    
    # 统计结果
    total_time = sum(r['total_time'] for r in results)
    avg_time = total_time / len(results)
    total_nodes = sum(r['retrieved_nodes'] for r in results)
    
    print(f"\n--- 批处理统计 ---")
    print(f"总问题数: {len(batch_questions)}")
    print(f"总处理时间: {total_time:.3f}s")
    print(f"平均处理时间: {avg_time:.3f}s")
    print(f"总检索节点数: {total_nodes}")
    
    engine.shutdown()


def example_error_handling():
    """错误处理示例"""
    print("\n=== GraphLLM Engine 错误处理示例 ===")
    
    try:
        # 尝试使用错误的配置
        config = create_engine_config(
            graph_space_name="error_space",
            vector_collection_name="error_collection",
            llm_model="invalid_model",  # 无效模型
            embedding_model="invalid_embedding"  # 无效嵌入模型
        )
        
        engine = GraphLLMEngine(config)
        
    except Exception as e:
        print(f"预期的错误: {e}")
        print("✓ 错误处理正常工作")
    
    # 正常配置
    try:
        config = create_engine_config(
            graph_space_name="safe_space",
            vector_collection_name="safe_collection"
        )
        
        engine = GraphLLMEngine(config)
        
        # 测试空查询
        result = engine.query("")
        print(f"空查询结果: {result['answer']}")
        
        # 测试无相关信息的查询
        result = engine.query("What is the meaning of life?")
        print(f"无相关信息查询结果: {result['answer']}")
        
        engine.shutdown()
        
    except Exception as e:
        print(f"意外错误: {e}")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_config_file_usage()
    example_advanced_usage()
    example_batch_processing()
    example_error_handling()
    
    print("\n=== 所有示例完成 ===") 