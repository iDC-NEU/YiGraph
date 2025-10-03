"""
智能问答系统 - 适配 openai>=1.0.0
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from openai import OpenAI  # 新版导入方式

from mcp_client import GraphMCPClient, create_client_and_connect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置OpenAI
os.environ['OPENAI_API_KEY'] = 'sk-G30rFStBigqXtuyIOkOo7Zh4QNxO8ZAjfZQ5DYPCgMXbPv8q'
os.environ['OPENAI_BASE_URL'] = 'https://gitaigc.com/v1/'

# 创建 OpenAI 客户端
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=os.environ['OPENAI_BASE_URL']
)


@dataclass
class QuestionAnalysis:
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str


class GraphSmartQA:
    def __init__(self, mcp_client: Optional[GraphMCPClient] = None):
        self.client = mcp_client
        self._client_created = False
        self.available_tools = []
        self.conversation_history: List[Dict[str, str]] = []
        self.graph_initialized = False
        
    async def initialize(self):
        """初始化系统"""
        if self.client is None:
            self.client = await create_client_and_connect()
            self._client_created = True
        
        self.available_tools = await self.client.list_tools()
        logger.info(f"✅ 加载了 {len(self.available_tools)} 个可用工具")
        
    async def cleanup(self):
        if self._client_created and self.client:
            await self.client.disconnect()
    
    async def load_graph_data(self, accounts_file: str, transactions_file: str):
        """加载并初始化图数据"""
        try:
            logger.info("📊 开始加载图数据...")
            vertices, edges = self.client.load_data_from_csv(accounts_file, transactions_file)
            
            result = await self.client.call_tool("initialize_graph", {
                "vertices": vertices,
                "edges": edges,
                "directed": True
            })
            
            if result.get('success'):
                self.graph_initialized = True
                logger.info(f"✅ {result['summary']}")
                return result['summary']
            else:
                raise Exception(result.get('error', '初始化失败'))
                
        except Exception as e:
            logger.error(f"❌ 图数据加载失败: {e}")
            raise
    
    async def ask(self, question: str) -> str:
        """回答用户问题(支持对话历史)"""
        if not self.graph_initialized and "初始化" not in question and "加载" not in question:
            return "⚠️ 请先使用 'load' 命令加载图数据,或输入包含'初始化'的问题。"
        
        try:
            self.conversation_history.append({"role": "user", "content": question})
            
            # Step 1: LLM分析问题
            analysis = await self._analyze_question_with_llm(question)
            logger.info(f"🤖 选择工具: {analysis.tool_name}")
            logger.info(f"📝 推理: {analysis.reasoning}")
            
            # Step 2: 调用MCP工具
            tool_result = await self.client.call_tool(
                analysis.tool_name, 
                analysis.parameters
            )
            
            # Step 3: LLM生成回答
            answer = await self._generate_answer_with_llm(question, tool_result)
            
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ 错误: {e}")
            error_msg = f"抱歉,处理问题时出错: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    async def _analyze_question_with_llm(self, question: str) -> QuestionAnalysis:
        """使用LLM分析问题(新版API)"""
        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}" 
            for t in self.available_tools
        ])
        
        context = ""
        if len(self.conversation_history) > 1:
            recent_history = self.conversation_history[-4:]
            context = "最近的对话:\n" + "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_history if msg['role'] == 'user'
            ]) + "\n\n"
        
        prompt = f"""{context}当前问题: {question}

可用工具:
{tools_desc}

请根据当前问题和对话历史,选择最合适的工具并提取参数。

以JSON格式回答:
{{
    "tool_name": "工具名称",
    "parameters": {{}},
    "reasoning": "选择原因"
}}

注意:
- 如果用户说"它"、"这个节点"等指代词,需要从对话历史中推断具体内容
- 优先考虑当前问题的主要意图"""
        
        # 使用新版API调用方式
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 或者你的API支持的模型名
            messages=[
                {"role": "system", "content": "你是图算法专家,擅长理解用户意图和对话上下文。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        return QuestionAnalysis(**result_json)
    
    async def _generate_answer_with_llm(self, question: str, 
                                       tool_result: Dict[str, Any]) -> str:
        """使用LLM生成回答(新版API)"""
        history_summary = ""
        if len(self.conversation_history) > 2:
            history_summary = "\n对话历史摘要:\n" + "\n".join([
                f"- {msg['content'][:50]}..." 
                for msg in self.conversation_history[-4:-1]
            ]) + "\n\n"
        
        prompt = f"""{history_summary}用户当前问题: {question}

工具执行结果:
{json.dumps(tool_result, ensure_ascii=False, indent=2)}

请根据工具结果回答用户问题。要求:
1. 简洁明了,突出关键信息
2. 如果问题与之前对话有关联,可以适当承接
3. 用自然对话的方式回答,不要过于正式
4. 不要重复用户问题,直接给出答案"""
        
        # 使用新版API调用方式
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是友好的数据分析助手,擅长多轮对话。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        logger.info("🗑️  对话历史已清除")


async def interactive_mode():
    """多轮对话交互模式"""
    print("=" * 60)
    print("🤖 图智能问答系统 - 交互模式")
    print("=" * 60)
    print("\n可用命令:")
    print("  load <accounts_file> <transactions_file> - 加载图数据")
    print("  clear - 清除对话历史")
    print("  help - 显示帮助信息")
    print("  exit/quit - 退出系统")
    print("\n提示: 可以直接提问,例如 '哪些节点最重要?'\n")
    
    qa = GraphSmartQA()
    await qa.initialize()
    
    default_accounts = "graphllm/graph_data/AMLSim/1K/accounts.csv"
    default_transactions = "graphllm/graph_data/AMLSim/1K/transactions.csv"
    
    try:
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 再见!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n📖 帮助信息:")
                    print("  - 直接输入问题,例如: '哪些节点最重要?'")
                    print("  - 使用 'load' 命令加载数据")
                    print("  - 使用 'clear' 清除对话历史重新开始")
                    continue
                
                elif user_input.lower() == 'clear':
                    qa.clear_history()
                    print("✅ 对话历史已清除")
                    continue
                
                elif user_input.lower().startswith('load'):
                    parts = user_input.split()
                    if len(parts) == 3:
                        accounts_file = parts[1]
                        transactions_file = parts[2]
                    else:
                        accounts_file = default_accounts
                        transactions_file = default_transactions
                        print(f"使用默认数据文件:\n  - {accounts_file}\n  - {transactions_file}")
                    
                    try:
                        summary = await qa.load_graph_data(accounts_file, transactions_file)
                        print(f"\n✅ {summary}")
                    except Exception as e:
                        print(f"\n❌ 加载失败: {e}")
                    continue
                
                print("\n🤖 思考中...", end='', flush=True)
                answer = await qa.ask(user_input)
                print(f"\r🤖 助手: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断,正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                logger.exception("详细错误:")
    
    finally:
        await qa.cleanup()


async def main():
    """主函数 - 启动交互模式"""
    await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
