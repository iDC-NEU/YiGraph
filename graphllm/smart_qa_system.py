"""
智能问答系统 - 支持动态后处理代码生成
"""
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from openai import OpenAI

# 假设 client 和 server 文件都在当前目录
from mcp_client import GraphMCPClient, create_client_and_connect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置OpenAI (请替换为您自己的key和endpoint)
os.environ['OPENAI_API_KEY'] = 'sk-G30rFStBigqXtuyIOkOo7Zh4QNxO8ZAjfZQ5DYPCgMXbPv8q'
os.environ['OPENAI_BASE_URL'] = 'https://gitaigc.com/v1/'

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

@dataclass
class QuestionAnalysis:
    tool_name: str
    parameters: Dict[str, Any]
    post_processing_code: Optional[str]  # ⭐ 新增字段
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
        logger.info(f"加载了 {len(self.available_tools)} 个可用工具")
        
    async def cleanup(self):
        if self._client_created and self.client:
            await self.client.disconnect()
    
    async def load_graph_data(self, accounts_file: str, transactions_file: str):
        """加载并初始化图数据"""
        try:
            logger.info("开始加载图数据...")
            vertices, edges = self.client.load_data_from_csv(accounts_file, transactions_file)
            
            result = await self.client.call_tool("initialize_graph", {
                "vertices": vertices,
                "edges": edges,
                "directed": True
            })
            
            if result.get('success'):
                self.graph_initialized = True
                logger.info(f"{result['summary']}")
                return result['summary']
            else:
                raise Exception(result.get('error', '初始化失败'))
                
        except Exception as e:
            logger.error(f"图数据加载失败: {e}")
            raise
    
    async def ask(self, question: str) -> str:
        """回答用户问题"""
        if not self.graph_initialized and "初始化" not in question and "加载" not in question:
            return "请先使用 'load' 命令加载图数据。"
        
        try:
            self.conversation_history.append({"role": "user", "content": question})
            
            # Step 1: 分析问题，选择工具，提取参数，并生成后处理代码
            analysis = await self._analyze_question_with_llm(question)
            logger.info(f"工具选择: {analysis.tool_name}")
            logger.info(f"参数提取: {json.dumps(analysis.parameters, ensure_ascii=False)}")
            logger.info(f"后处理代码:\n{analysis.post_processing_code}")
            logger.debug(f"推理过程: {analysis.reasoning}")
            
            # Step 2: 调用工具，并传入后处理代码
            tool_result = await self.client.call_tool(
                analysis.tool_name, 
                analysis.parameters,
                post_processing_code=analysis.post_processing_code  # ⭐ 传递代码
            )
            
            logger.info(f"工具执行结果: {tool_result.get('summary', '执行完成')}")
            
            # Step 3: 基于精简后的结果生成最终回答
            answer = await self._generate_answer_with_llm(question, analysis.tool_name, tool_result)
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            error_msg = f"处理过程中发生错误: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    async def _analyze_question_with_llm(self, question: str) -> QuestionAnalysis:
        """使用LLM分析问题、提取参数并生成后处理代码"""
        
        tools_documentation = self._build_tools_documentation()
        conversation_context = self._build_conversation_context()
        
        # ⭐ 修改后的系统提示词，增加了后处理代码生成的要求
        system_prompt = """你是一个专业的图计算调度AI。你的任务是分析用户问题，选择最合适的图算法工具，提取参数，并生成一段在服务器端执行的Python代码来后处理结果。

## 核心职责

### 1. 工具选择和参数提取
- 根据用户意图，从可用工具列表中选择最匹配的工具。
- 严格按照工具的Schema提取参数，确保类型正确。
- 用户未提及的参数，如果存在默认值，则不要在 `parameters` 中输出。

### 2. ⭐ 生成后处理代码 (post_processing_code)
这是最关键的任务。你需要根据用户问题的具体需求（如"Top 5"、"大于0.5的"、"有多少个"），生成一段Python代码。

#### 代码要求:
- **必须**包含一个名为 `process(data)` 的函数。
- `data` 参数是图算法返回的原始、完整结果。
- 函数**必须**返回一个经过处理的、精简的结果（通常是 `dict` 或 `list`）。
- 代码中不应包含任何不安全的库导入 (`os`, `sys` 等)。只进行数据操作。
- 如果用户没有提出筛选或聚合需求，返回一个默认的、仅用于透传数据的代码。

#### `data` 参数格式示例 (不同算法的原始输出):
- `run_pagerank`: `{'nodeA': 0.15, 'nodeB': 0.12, ...}` (字典)
- `run_connected_components`: `[['nodeA', 'nodeB'], ['nodeC'], ...]` (列表的列表)
- `run_shortest_path`: `{'path': ['A', 'B', 'C'], 'length': 2, ...}` (字典)
- `run_louvain_community_detection`: `{'community_count': 3, 'modularity': 0.7, 'communities': {'A':0, 'B':0, 'C':1}}` (字典)

#### 代码生成示例:
- **用户问题: "最重要的5个节点是什么?"** (排序和切片)
  ```python
  def process(data):
      # data is a dict like {'A': 0.15, 'B': 0.12, ...}
      sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
      return dict(sorted_items[:5])
  ```

- **用户问题: "PageRank分数高于0.01的节点有哪些?"** (过滤)
  ```python
  def process(data):
      # data is a dict like {'A': 0.015, 'B': 0.009, ...}
      return {node: score for node, score in data.items() if score > 0.01}
  ```

- **用户问题: "图中有多少个连通分量?"** (聚合)
  ```python
  def process(data):
      # data is a list of lists like [['A', 'B'], ['C']]
      return {'component_count': len(data)}
  ```

- **用户问题: "节点A和B的最短路径长度是多少?"** (提取特定值)
  ```python
  def process(data):
      # data is a dict like {'path': ['A', 'B'], 'length': 1}
      return {'path_length': data.get('length')}
  ```

- **用户问题: "计算PageRank"** (无具体筛选要求) (默认透传)
  ```python
  def process(data):
      # No specific filtering, but maybe limit to top 10 by default
      if isinstance(data, dict):
          sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
          return dict(sorted_items[:10])
      return data  # Or just return raw data if not a dict
  ```

### 3. 输出格式要求
必须返回有效的JSON对象，格式如下 (不要用markdown代码块包裹):

```json
{
  "tool_name": "选择的工具名称",
  "parameters": {
    "param1": "value1"
  },
  "post_processing_code": "def process(data):\\n    # Your python code here",
  "reasoning": "详细说明你的选择、参数提取和代码生成逻辑。"
}
```"""

        user_prompt = f"""## 对话上下文
{conversation_context}

当前用户问题
{question}

可用工具列表
{tools_documentation}

任务
请分析用户问题，完成以下任务：

选择最合适的工具。

提取所有相关参数。

生成用于筛选/聚合结果的 post_processing_code。

提供清晰的 reasoning。

现在请返回JSON格式的分析结果。"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # 设为0以保证代码生成的稳定性
                max_tokens=1024
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM原始返回:\n{result_text}")
            
            result_text = re.sub(r'^```(?:json)?\s*', '', result_text, flags=re.MULTILINE)
            result_text = re.sub(r'\s*```$', '', result_text, flags=re.MULTILINE)
            result_text = result_text.strip()
            
            result_json = json.loads(result_text)
            
            # 字段验证和默认值
            tool_name = result_json.get('tool_name')
            if not tool_name:
                raise ValueError("LLM返回缺少tool_name字段")
            
            return QuestionAnalysis(
                tool_name=tool_name,
                parameters=result_json.get('parameters', {}),
                post_processing_code=result_json.get('post_processing_code'),
                reasoning=result_json.get('reasoning', "未提供推理说明")
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"LLM分析或JSON解析失败: {e}\n原始文本:\n{result_text}")
            # 在失败时提供一个默认的、无操作的分析结果
            return QuestionAnalysis(
                tool_name="get_graph_info",
                parameters={},
                post_processing_code="def process(data):\n    return data",
                reasoning=f"LLM分析失败，回退到默认操作。错误: {e}"
            )

    def _build_tools_documentation(self) -> str:
        docs = []
        for tool in self.available_tools:
            doc = f"### 工具: {tool['name']}\n{tool['description']}\n"
            docs.append(doc)
        return "\n".join(docs)

    def _build_conversation_context(self) -> str:
        if not self.conversation_history:
            return "（这是对话的第一轮）"
        context_lines = []
        for msg in self.conversation_history[-6:]:
            role = "用户" if msg['role'] == 'user' else "助手"
            content = msg['content']
            context_lines.append(f"{role}: {content[:150]}")
        return "\n".join(context_lines)

    async def _generate_answer_with_llm(self, question: str, tool_name: str,
                                      tool_result: Dict[str, Any]) -> str:
        """使用LLM基于精简后的结果生成回答"""
        
        if not tool_result.get('success'):
            error_msg = tool_result.get('error', '未知错误')
            summary = tool_result.get('summary', '工具执行失败')
            return f"{summary}。 错误详情: {error_msg}"
        
        summary = tool_result.get('summary', '')
        # ⭐ 结果已经是精简过的，不再需要截断
        result_data = tool_result.get('result', {})
        result_data_str = json.dumps(result_data, ensure_ascii=False, indent=2)

        if len(result_data_str) > 2000:
            result_data_str = result_data_str[:2000] + "\n... (结果过长，已截断)"

        system_prompt = """你是一个专业的数据分析师，负责向用户解释图算法的执行结果。你的回答应简洁、清晰、直接。

回答要求:
1. 直接回答: 根据用户问题和精简后的数据，直接给出答案。
2. 数据呈现: 清晰地列出关键数据点。
3. 语言风格: 通俗易懂，避免重复摘要里的信息。
4. 保持简洁: 回答应控制在1-3句话内。"""

        user_prompt = f"""## 用户原始问题
{question}

执行的工具
{tool_name}

工具执行摘要
{summary}

⭐ 精简后的结果数据
{result_data_str}

任务
请基于上述信息，用通俗易懂的语言回答用户的问题。"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM生成回答失败: {e}", exc_info=True)
            return summary  # 后备：直接返回服务器的摘要

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清除")

async def interactive_mode():
    """交互模式"""
    print("=" * 60)
    print("图智能问答系统 (v2 - 动态后处理)")
    print("=" * 60)
    print("命令: load <accounts.csv> <transactions.csv>, clear, exit\n")

    qa = GraphSmartQA()
    await qa.initialize()

    default_accounts = "graphllm/graph_data/AMLSim/1K/accounts.csv" 
    default_transactions = "graphllm/graph_data/AMLSim/1K/transactions.csv"

    try:
        while True:
            try:
                user_input = input("\n您: ").strip()
                
                if not user_input: 
                    continue
                if user_input.lower() in ['exit', 'quit', 'q']: 
                    break
                
                if user_input.lower() == 'clear':
                    qa.clear_history()
                    print("对话历史已清除")
                    continue
                
                if user_input.lower().startswith('load'):
                    parts = user_input.split()
                    accounts = parts[1] if len(parts) > 1 else default_accounts
                    transactions = parts[2] if len(parts) > 2 else default_transactions
                    try:
                        print("\n正在加载数据...", end='', flush=True)
                        summary = await qa.load_graph_data(accounts, transactions)
                        print(f"\r✅ {summary}")
                    except Exception as e:
                        print(f"\n❌ 加载失败: {e}")
                    continue
                
                print("\n思考中...", end='', flush=True)
                answer = await qa.ask(user_input)
                print(f"\r助手: {answer}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n发生错误: {e}")
                logger.error("交互模式错误", exc_info=True)

    finally:
        print("\n\n正在断开连接...再见!")
        await qa.cleanup()

if __name__ == "__main__":
    # 确保在运行前，你已经安装了所需依赖，并且相关文件路径正确
    # pip install mcp-client openai
    asyncio.run(interactive_mode())
