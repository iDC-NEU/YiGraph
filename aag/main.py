import os
import sys
import argparse
from typing import List, Optional, Dict, Any, Union
import time
import platform
import datetime
import asyncio
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aag.config.engine_config import create_engine_config, load_config_from_yaml
from aag.engine.aag_engine import AAGEngine
from aag.utils.path_utils import DEFAULT_CONFIG_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(funcName)s(): %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
GRAY = "\033[90m"


def get_user_prompt(current_mode: str) -> str:
    """
    根据当前模式生成用户输入提示符
    
    Args:
        current_mode: 当前模式 "normal" | "interact" | "expert"
    
    Returns:
        格式化的提示符字符串
    """
    if current_mode == "interact":
        return "(interact mode) 👤 用户 > "
    if current_mode == "expert":
        return "(expert mode) 👤 用户 > "
    return "👤 用户 > "


def print_dag_info(dag_info: Dict[str, Any]) -> None:
    """打印DAG信息（格式化输出）"""
    if not dag_info:
        return
    
    print("\n📊 --- DAG 信息 ---")
    
    # 打印子查询计划
    if "subquery_plan" in dag_info:
        plan = dag_info["subquery_plan"]
        if "subqueries" in plan:
            print("\n子查询列表：")
            for i, subq in enumerate(plan["subqueries"], 1):
                print(f"  {i}. [{subq.get('id', '?')}] {subq.get('query', '')}")
                deps = subq.get('depends_on', [])
                if deps:
                    print(f"     依赖: {', '.join(deps)}")
    
    # 打印步骤信息
    if "steps" in dag_info:
        print("\n步骤详情：")
        for step_id, step_info in dag_info["steps"].items():
            print(f"  [{step_id}] {step_info.get('question', '')}")
            if step_info.get('algorithm'):
                print(f"      算法: {step_info['algorithm']}")
            if step_info.get('task_type'):
                print(f"      类型: {step_info['task_type']}")
    
    # 打印拓扑顺序
    if "topological_order" in dag_info:
        order = dag_info["topological_order"]
        print(f"\n执行顺序: {' → '.join(order)}")
    
    print("-" * 74)


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
    """交互模式 - 支持 normal / interact / expert 三种模式"""
    current_mode = "normal"
    dag_built = False

    mode_label = {
        "normal": "普通",
        "interact": "交互",
        "expert": "专家",
    }

    while True:
        try:
            question = input(f"\n{get_user_prompt(current_mode)}").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("🐾 AAG小助手退下喵~ 再见! (ฅ'ω'ฅ)")
                break

            elif question.lower().startswith('mode '):
                mode_arg = question[5:].strip().lower()
                if mode_arg in ["normal", "interact", "expert"]:
                    old_mode = current_mode
                    current_mode = mode_arg
                    dag_built = False
                    print(f"✅ 已切换到{mode_label[current_mode]}模式")
                    if old_mode != current_mode:
                        if current_mode == "normal":
                            print("   提示：直接生成DAG并执行，返回分析报告")
                        elif current_mode == "interact":
                            print("   提示：生成DAG后可 modify/start 交互调整")
                        else:
                            print("   提示：输入自然语言专家指令，系统会构建DAG并校验算法边界")
                    print("-" * 74)
                else:
                    print("⚠️ 无效模式，请使用 'mode normal' / 'mode interact' / 'mode expert'")
                    print("-" * 74)
                continue

            elif question.lower() == 'stats':
                response = engine.get_performance_summary()
                print("\n📊 --- 性能统计 ---")
                for key, value in response.items():
                    print(f"• {key}: {value}")
                print("-" * 74)
                continue

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
                    dag_built = False
                    print("-" * 74)
                except Exception as e:
                    print(f"⚠️ 选择数据集失败: {e}")
                    print("-" * 74)
                continue

            elif question.lower() in ['help', 'h']:
                print("\n📌 === 帮助菜单 ===")
                print("通用命令：")
                print("  📊 stats                                显示性能统计")
                print("  📁 datasets | list                      列出所有可用数据集")
                print("  🗂 use <name>                           选定数据集 (自动推断类型)")
                print("  🗂 use <name> <dtype>                   指定类型 (graph/table/text)")
                print("  🗂 use <dtype>:<name>                   dtype:name 形式选择")
                print("  🔄 mode normal|interact|expert          切换执行模式")
                print("  ❓ help | h                             显示帮助")
                print("  👋 quit | exit | q                      退出系统")

                if current_mode == "interact":
                    print("\n交互模式命令：")
                    print("  🔧 modify <request>                    修改DAG（可多次）")
                    print("  ▶️  start | analyze                     开始执行分析")

                if current_mode == "expert":
                    print("\n专家模式输入示例：")
                    print("  自然语言: 先找节点23所在社区，再在社区里用pagerank找前10个关键节点")

                print(f"\n当前模式: {mode_label[current_mode]}模式")
                print("\n问题前缀示例：")
                print("  normal: 找出节点45的社区")
                print("  interact: 找出节点45的社区")
                print("  expert: 先找节点23所在社区，再在社区里用pagerank找前10关键节点")
                print("-" * 74)
                continue

            elif not question:
                continue

            actual_question = question
            question_mode = current_mode
            if question.lower().startswith("normal:"):
                actual_question = question[7:].strip()
                question_mode = "normal"
            elif question.lower().startswith("interact:"):
                actual_question = question[9:].strip()
                question_mode = "interact"
            elif question.lower().startswith("expert:"):
                actual_question = question[7:].strip()
                question_mode = "expert"

            if not actual_question:
                print("⚠️ 问题不能为空")
                continue

            if question_mode != current_mode:
                print(f"ℹ️ 本次查询使用{mode_label[question_mode]}模式（通过前缀指定）")
                current_mode = question_mode
                dag_built = False

            if current_mode == "interact" and actual_question.lower().startswith("modify "):
                if not dag_built:
                    print("⚠️ 请先输入问题生成DAG")
                    continue
                modification_request = actual_question[7:].strip()
                if not modification_request:
                    print("⚠️ 请输入修改需求")
                    continue

                print("🔧 正在修改DAG...", flush=True)
                try:
                    result = await engine.expert_modify_dag(modification_request)
                    if isinstance(result, dict) and "error" in result:
                        print(f"❌ {result['error']}")
                    else:
                        print(f"✅ {result.get('message', 'DAG已更新')}")
                        print_dag_info(result.get('dag_info', {}))
                        print("\n请选择下一步操作：")
                        print("  🔧 modify <request>  修改DAG")
                        print("  ▶️  start            开始分析")
                        dag_built = True
                except Exception as e:
                    print(f"⚠️ 修改DAG失败: {e}")
                print("-" * 74)
                continue

            if current_mode in {"interact", "expert"} and actual_question.lower() in ['start', 'analyze', '开始分析']:
                if not dag_built:
                    print("⚠️ 请先输入问题生成DAG")
                    continue
                print("▶️ 开始执行分析...", flush=True)
                try:
                    result = await engine.expert_start_analysis()
                    print(f"\n🤖 分析报告\n：{result}")
                    dag_built = False
                except Exception as e:
                    print(f"⚠️ 分析执行失败: {e}")
                print("-" * 74)
                continue

            print("🤖 正在思考中，请稍等… 🧠✨", flush=True)

            if current_mode == "normal":
                result = await engine.run(actual_question, mode="normal")
                print(f"\n🤖 分析报告\n：{result}")
                print("-" * 74)
                continue

            if current_mode == "interact":
                result = await engine.run(actual_question, mode="interact")
                if isinstance(result, dict):
                    if "error" in result:
                        print(f"❌ {result['error']}")
                        dag_built = False
                    else:
                        print(f"✅ {result.get('message', 'DAG已生成')}")
                        print_dag_info(result.get('dag_info', {}))
                        print("\n请选择下一步操作：")
                        print("  🔧 modify <request>  修改DAG")
                        print("  ▶️  start            开始分析")
                        dag_built = True
                else:
                    print(f"\n{result}")
                    dag_built = False
                print("-" * 74)
                continue

            result = await engine.run(actual_question, mode="expert")
            if isinstance(result, dict):
                if "error" in result:
                    print(f"❌ {result['error']}")
                    dag_built = False
                else:
                    print(f"✅ {result.get('message', '专家DAG处理完成')}")
                    print_dag_info(result.get('dag_info', {}))

                    validation = result.get("algorithm_validation", {})
                    unsupported = validation.get("unsupported_algorithms", [])
                    if unsupported:
                        print("\n⚠️ 以下算法不在算法库中：")
                        for item in unsupported:
                            print(f"  - {item.get('query_id')}: {item.get('requested_algorithm')}")
                            suggestions = item.get("suggestions") or []
                            if suggestions:
                                print(f"    建议: {', '.join(suggestions)}")

                    instruction_adjustments = validation.get("instruction_algorithm_adjustments", [])
                    if instruction_adjustments:
                        print("\nℹ️ 专家指令算法替换说明：")
                        for item in instruction_adjustments:
                            mentioned_algorithm = item.get("mentioned_algorithm")
                            replacement_algorithm = item.get("replacement_algorithm")
                            recommendations = item.get("recommendations") or []

                            line = f"  - 算法库中没有 {mentioned_algorithm}"
                            if replacement_algorithm:
                                line += f"，已改为 {replacement_algorithm}"
                            print(line)
                            if recommendations:
                                print(f"    推荐: {', '.join(recommendations)}")

                    if result.get("can_start_analysis", False):
                        print("\n可执行后续分析：")
                        print("  ▶️  start            开始分析")
                        dag_built = True
                    else:
                        dag_built = False
            else:
                print(f"\n{result}")
                dag_built = False
            print("-" * 74)

        except KeyboardInterrupt:
            print("\n\n⛔ 用户中断，系统退出")
            break
        except Exception as e:
            print(f"⚠️ 处理分析时出错: {e}")


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
            print(" 输入问题按 Enter 分析；命令：stats | datasets | use <name> [dtype] | mode | help | quit")
            print(" 支持 normal / interact / expert 三种模式，使用 'mode <name>' 切换")
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
