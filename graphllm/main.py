"""
GraphLLM Engine 主入口点
端到端的Graph Analysis + RAG + LLM框架
"""

import os
import sys
import argparse
from typing import List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphllm.graphllm_engine import GraphLLMEngine, create_engine_config, load_config_from_yaml


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GraphLLM Engine - 端到端图分析+RAG+LLM框架")
    
    # 基础配置
    parser.add_argument("--mode", choices=["interactive", "batch", "process"], 
                       default="interactive", help="运行模式")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 数据库配置
    parser.add_argument("--graph-space", type=str, default="graphllm_space",
                       help="图数据库空间名称")
    parser.add_argument("--vector-collection", type=str, default="graphllm_collection",
                       help="向量数据库集合名称")
    
    # 模型配置
    parser.add_argument("--llm-model", type=str, default="llama3.1:70b",
                       help="LLM模型名称")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5",
                       help="嵌入模型名称")
    parser.add_argument("--llm-type", choices=["ollama", "openai"], default="ollama",
                       help="LLM类型")
    
    # 设备配置
    parser.add_argument("--llm-device", type=str, default="cuda:0",
                       help="LLM设备")
    parser.add_argument("--embed-device", type=str, default="cuda:0",
                       help="嵌入模型设备")
    
    # RAG配置
    parser.add_argument("--graph-k-hop", type=int, default=2,
                       help="图遍历跳数")
    parser.add_argument("--vector-k-similarity", type=int, default=5,
                       help="向量相似度检索数量")
    
    # 输入输出
    parser.add_argument("--input-file", type=str, help="输入文件路径")
    parser.add_argument("--output-file", type=str, help="输出文件路径")
    parser.add_argument("--questions", nargs="+", help="问题列表")
    
    return parser.parse_args()


def interactive_mode(engine: GraphLLMEngine):
    """交互模式"""
    print("=== GraphLLM Engine 交互模式 ===")
    print("输入问题执行回答")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'stats' 查看性能统计")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式")
                break
            elif question.lower() == 'stats':
                reponse = engine.get_performance_summary()
                print("\n--- 性能统计 ---")
                for key, value in summary.items():
                    print(f"{key}: {value}")
                continue
            elif not question:
                continue
            
            # 处理查询
            print("正在处理查询...")
            result = engine.query(question)
            
            print(f"\n回答: {result}")

        except KeyboardInterrupt:
            print("\n\n用户中断，退出交互模式")
            break
        except Exception as e:
            print(f"处理查询时出错: {e}")


def batch_mode(engine: GraphLLMEngine, questions: List[str], output_file: Optional[str] = None):
    """批处理模式"""
    print(f"=== GraphLLM Pipeline 批处理模式 ===")
    print(f"处理 {len(questions)} 个问题")
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n处理问题 {i}/{len(questions)}: {question}")
        
        try:
            result = engine.query(question)
            results.append(result)
            
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            results.append({
                "question": question,
                "raise error": f"处理失败: {e}",
            })
    
    # 输出结果
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
    



def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        # 创建配置 - 优先从配置文件读取，否则使用命令行参数
        if args.config:
            print(f"从配置文件加载配置: {args.config}")
            config = load_config_from_yaml(args.config)
        else:
            print("使用命令行参数创建配置")
            config = create_engine_config(
                graph_space_name=args.graph_space,
                vector_collection_name=args.vector_collection,
                llm_model=args.llm_model,
                embedding_model=args.embedding_model,
                llm_type=args.llm_type,
                llm_device=args.llm_device,
                embed_device=args.embed_device,
                graph_k_hop=args.graph_k_hop,
                vector_k_similarity=args.vector_k_similarity
            )
        
        # 初始化Engine
        print("正在初始化 GraphLLM Engine...")
        engine = GraphLLMEngine(config)
        print("✓ Engine 初始化完成")
        
        # 根据模式运行
        if args.mode == "interactive":
            interactive_mode(engine)
        elif args.mode == "batch":
            if not args.questions:
                print("批处理模式需要提供问题列表，使用 --questions 参数")
                return
            batch_mode(engine, args.questions, args.output_file)
        
        # 清理资源
        engine.shutdown()
        
    except Exception as e:
        print(f"运行出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
