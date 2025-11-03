import os
import sys
import argparse
from typing import List, Optional
import time
import platform
import datetime
import asyncio

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aag.config.engine_config import create_engine_config, load_config_from_yaml
from aag.engine.aag_engine import AAGEngine
from aag.utils.path_utils import DEFAULT_CONFIG_PATH


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
GRAY = "\033[90m"


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Analytics Augmented Generation Engine - 端到端分析增强生成框架")
    
    # 基础配置
    parser.add_argument("--mode", choices=["interactive", "batch", "process"], 
                       default="interactive", help="运行模式")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 数据库配置
    parser.add_argument("--graph-space", type=str, default="graph_space",
                       help="图数据库空间名称")
    parser.add_argument("--vector-collection", type=str, default="vector_collection",
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


async def interactive_mode(engine: AAGEngine):
    """交互模式"""

    while True:
        try:
            # 👤 用户输入
            question = input("\n👤 用户 > ").strip()
            
            # 🔚 退出命令
            if question.lower() in ['quit', 'exit', 'q']:
                print("🐾 AAG小助手退下喵~ 再见! (ฅ'ω'ฅ)")
                break

            # 📊 性能统计
            elif question.lower() == 'stats':
                response = engine.get_performance_summary()
                print("\n📊 --- 性能统计 ---")
                for key, value in response.items():
                    print(f"• {key}: {value}")
                print("-" * 74)
                continue

            # 📁 数据集列表
            elif question.lower() in ['datasets', 'list', 'list datasets']:
                try:
                    ds_map = engine.list_datasets()
                    print("\n📁 --- 可用数据集 ---")
                    for dtype, names in ds_map.items():
                        print(f"{dtype} ({len(names)}): {', '.join(names) if names else '(empty)'}")
                    print("-" * 74)
                except Exception as e:
                    print(f"⚠️ 列出数据集失败: {e}")
                    print("-" * 74)
                continue

            # 🗂 选择数据集
            elif question.lower().startswith('use '):
                cmd = question[4:].strip()
                dtype = None
                name = cmd

                if ':' in cmd:
                    parts = cmd.split(':', 1)
                    if len(parts) == 2:
                        dtype, name = parts[0].strip(), parts[1].strip()
                else:
                    toks = cmd.split()
                    if len(toks) >= 2:
                        name = ' '.join(toks[:-1]).strip()
                        dtype = toks[-1].strip()

                try:
                    data = engine.specific_dataset(name, dtype)
                    if data is None:
                        scope = dtype if dtype else "graph/table/text"
                        print(f"❌ 未找到数据集: '{name}' (搜索范围: {scope})")
                    else:
                        print(f"✅ 已选择数据集: {name}" + (f" ({dtype})" if dtype else ""))
                    print("-" * 74)
                except Exception as e:
                    print(f"⚠️ 选择数据集失败: {e}")
                    print("-" * 74)
                continue

            # 🆘 帮助命令
            elif question.lower() in ['help', 'h']:
                print("\n📌 === 帮助菜单 ===")
                print("可用命令：")
                print("  📊 stats                         显示性能统计")
                print("  📁 datasets | list               列出所有可用数据集")
                print("  🗂 use <name>                    选定数据集 (自动推断类型)")
                print("  🗂 use <name> <dtype>            指定类型 (graph/table/text)")
                print("  🗂 use <dtype>:<name>            dtype:name 形式选择")
                print("  ❓ help | h                 显示帮助")
                print("  👋 quit | exit | q               退出系统")
                print("\n示例：")
                print("  use AMLSim1K")
                print("  use AMLSim1K graph")
                print("  use graph:AMLSim1K")
                print("-" * 74)
                continue

            elif not question:
                continue
            
            # 🤖 执行查询
            print("🤖 正在思考中，请稍等… 🧠✨", flush=True)
            result = await engine.run(question)

            print(f"\n🤖 分析报告\n：{result}")
            print("-" * 74)

        except KeyboardInterrupt:
            print("\n\n⛔ 用户中断，系统退出")
            break
        except Exception as e:
            print(f"⚠️ 处理查询时出错: {e}")

    # while True:
    #     try:
    #         question = input("\n请输入问题: ").strip()
            
    #         if question.lower() in ['quit', 'exit', 'q']:
    #             print("退出交互模式")
    #             break
    #         elif question.lower() == 'stats':
    #             reponse = engine.get_performance_summary()
    #             print("\n--- 性能统计 ---")
    #             for key, value in reponse.items():
    #                 print(f"{key}: {value}")
    #             print("-" * 74)
    #             continue
    #         elif question.lower() in ['datasets', 'list', 'list datasets']:
    #             # 列出当前可用数据集
    #             try:
    #                 ds_map = engine.list_datasets()
    #                 print("\n--- 可用数据集 ---")
    #                 for dtype, names in ds_map.items():
    #                     print(f"{dtype}: {', '.join(names) if names else '(empty)'}")
    #                     print(f" - {dtype} ({len(names)}): {', '.join(names)}")
    #                 print("-" * 74)
    #             except Exception as e:
    #                 print(f"列出数据集失败: {e}")
    #                 print("-" * 74)
    #             continue
    #         elif question.lower().startswith('use '):
    #             # 选择要分析的数据集
    #             # 支持格式：use <name>、use <name> <dtype>、use <dtype>:<name>
    #             cmd = question[4:].strip()
    #             dtype = None
    #             name = cmd
    #             if ':' in cmd:
    #                 # dtype:name 形式
    #                 parts = cmd.split(':', 1)
    #                 if len(parts) == 2:
    #                     dtype, name = parts[0].strip(), parts[1].strip()
    #             else:
    #                 # 尝试用空格分离 name 和 dtype（末尾为 dtype 更自然）
    #                 toks = cmd.split()
    #                 if len(toks) >= 2:
    #                     name = ' '.join(toks[:-1]).strip()
    #                     dtype = toks[-1].strip()
    #             try:
    #                 data = engine.specific_dataset(name, dtype)
    #                 if data is None:
    #                     scope = dtype if dtype else "graph/table/text"
    #                     print(f"未找到数据集: name='{name}', type_scope='{scope}'")
    #                 else:
    #                     print(f"已选择数据集: name='{name}'" + (f", type='{dtype}'" if dtype else ""))
    #                 print("-" * 74)
    #             except Exception as e:
    #                 print(f"选择数据集失败: {e}")
    #                 print("-" * 74)
    #             continue
    #         # 补充一个 help， 输出可用的所有命令和对应的输出格式，参考成熟的输出方案
    #         elif question.lower() in ['help', 'h']:
    #             print("\n=== 帮助 (Help) ===")
    #             print("可用命令：")
    #             print("  stats                         显示性能统计")
    #             print("  datasets | list               列出所有可用数据集，按类型分组")
    #             print("  use <name>                    选择要分析的数据集（类型自动推断）")
    #             print("  use <name> <dtype>            选择数据集并指定类型（graph/table/text）")
    #             print("  use <dtype>:<name>            以 dtype:name 形式选择数据集")
    #             print("  help | h                 显示本帮助")
    #             print("  quit | exit | q               退出系统")
    #             print("\n示例：")
    #             print("  use AMLSim1K")
    #             print("  use AMLSim1K graph")
    #             print("  use graph:AMLSim1K")
    #             print("-" * 74)
    #             continue
    #         elif not question:
    #             continue
            
    #         # 处理查询（非命令输入时，交由引擎进行分析）
    #         print("🤔思考中...", flush=True)
    #         result = await engine.run(question)
            
    #         print(f"\n回答: {result}")
    #         print("-" * 74)

    #     except KeyboardInterrupt:
    #         print("\n\n用户中断，退出交互模式")
    #         break
    #     except Exception as e:
    #         print(f"处理查询时出错: {e}")


def batch_mode(engine: AAGEngine, questions: List[str], output_file: Optional[str] = None):
    """批处理模式"""
    print(f"=== AAG Pipeline 批处理模式 ===")
    print(f"处理 {len(questions)} 个问题")
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n处理问题 {i}/{len(questions)}: {question}")
        
        try:
            result = engine.run(question)
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
    


async def main():
    """主函数：直接从配置文件加载参数"""
    try:
        # 确定配置文件路径
        config_path = DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(f"默认配置文件未找到: {config_path}")

        print(f"{CYAN}\n{'=' * 74}{RESET}")
        print(f"{BOLD} 🧠  欢迎使用 AAG 智能分析系统  (Analytics Augmented Generation Engine) {RESET}")
        print(f"{CYAN}{'=' * 74}{RESET}")
        print(f" 配置文件: {config_path}")
        print(f" 启动时间 : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 74)

        # 从配置文件加载配置
        config = load_config_from_yaml(config_path)

        # 初始化引擎
        print(" 正在初始化引擎，请稍候 ... \n", end="", flush=True)
        start = time.time()
        engine = AAGEngine(config)
        elapsed = time.time() - start
        print(f"\r {GREEN}✅ 引擎初始化完成{RESET}  用时 {elapsed:.2f}s")
        print(f"{CYAN}{'=' * 74}{RESET}\n")

        # 判断运行模式（从配置文件读取）
        mode = config.mode.lower()

        if mode == "interactive":
            print(f"{BOLD}💬 当前运行模式：交互模式 (Interactive Mode){RESET}")
            print("-" * 74)
            print(" 输入问题按 Enter 分析；命令：stats | datasets | use <name> [dtype] | help | quit")
            print(f"{CYAN}{'=' * 74}{RESET}")
            # interactive_mode(engine)
            await interactive_mode(engine)

        elif mode == "batch":
            # TODO(chaoyi): 批处理模式尚未完善，当前不可用，需后续补充实现
            print(f"{YELLOW}⚠️  当前运行模式：批处理模式 (Batch Mode){RESET}")
            print("-" * 74)
            print(" 该模式尚未实现，请使用交互模式运行。")
            print(f"{CYAN}{'=' * 74}{RESET}")
            questions = config.get("questions")
            output_file = config.get("output_file", "results.json")
            if not questions:
                print(f"{YELLOW}错误：批处理模式需要在配置文件中提供 'questions' 字段{RESET}")
                return 1
            batch_mode(engine, questions, output_file)

        else:
            print(f"{YELLOW}❌ 未知的运行模式: {mode}{RESET}")
            print(f"{CYAN}{'=' * 74}{RESET}")

        # 清理资源
        await engine.shutdown()
        print(f"{GREEN}✓ 程序运行完成，资源已释放{RESET}")

    except Exception as e:
        print(f"{YELLOW}运行出错: {e}{RESET}")
        return 1

    return 0


def main_delay():
    """主函数"""
    args = parse_arguments()
    
    try:
        # 创建配置 - 优先从配置文件读取，否则使用命令行参数
        if args.config:
            print(f"从配置文件加载配置: {args.config}")
            config = load_config_from_yaml(args.config)
        else:
            # TODO(chaoyi): 数过多， llm 现在不支持通过命令行适配 可选的llm，后期需要修正 
            print("使用命令行参数创建配置")
            config = create_engine_config(
                graph_space_name=args.graph_space,
                vector_collection_name=args.vector_collection,
                llm_model=args.llm_model,
                embedding_model=args.embedding_model,
                llm_provider=args.llm_type,
                ollama_device=args.llm_device,
                embed_device=args.embed_device,
                graph_k_hop=args.graph_k_hop,
                vector_k_similarity=args.vector_k_similarity
            )
        
        # 初始化Engine
        print("正在初始化 AAG Engine...")
        engine = AAGEngine(config)
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
    # exit(main())
    asyncio.run(main())
