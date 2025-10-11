
"""
智能问答系统 - 支持动态后处理代码生成 (最终修复版)
核心改进:
1. 强化 LLM 提示词,明确忽略 G 参数
2. 从工具文档中过滤 G 参数
3. 双重安全检查,确保 G 参数不会被传递
4. 增强错误处理和日志记录
"""
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from openai import OpenAI

from mcp_client import GraphMCPClient, create_client_and_connect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    post_processing_code: Optional[str]
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
                logger.info(f"✅ {result['summary']}")
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
            
            # Step 1: 分析问题
            analysis = await self._analyze_question_with_llm(question)
            logger.info(f"✅ 工具选择: {analysis.tool_name}")
            logger.info(f"✅ 参数提取: {json.dumps(analysis.parameters, ensure_ascii=False)}")
            if analysis.post_processing_code:
                logger.info(f"✅ 后处理代码:\n{analysis.post_processing_code[:200]}...")
            
            # ⭐ MODIFIED: 额外验证 - 确保没有 G 参数泄漏
            if 'G' in analysis.parameters:
                logger.warning("⚠️  检测到 G 参数泄漏,已自动移除")
                del analysis.parameters['G']
            
            # Step 2: 调用工具
            tool_result = await self.client.call_tool(
                analysis.tool_name, 
                analysis.parameters,
                post_processing_code=analysis.post_processing_code
            )
            
            logger.info(f"✅ 工具执行: {tool_result.get('summary', '完成')}")
            
            # Step 3: 生成回答
            answer = await self._generate_answer_with_llm(question, analysis.tool_name, tool_result)
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}", exc_info=True)
            error_msg = f"处理过程中发生错误: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    async def _analyze_question_with_llm(self, question: str) -> QuestionAnalysis:
        """使用LLM分析问题、提取参数并生成后处理代码"""
        
        tools_documentation = self._build_tools_documentation()
        conversation_context = self._build_conversation_context()
        
        # ⭐ MODIFIED: 强化提示词,明确三个关键点
        system_prompt = """你是一个专业的图计算调度AI。你的任务是分析用户问题,选择工具,提取参数,并生成后处理代码。

## 🚨 核心规则 (必须严格遵守)

### 规则1: 工具名称必须精确匹配
- 从"可用工具列表"中**精确复制**工具名称
- 例如: 必须写 'run_pagerank' 而不是 'pagerank' 或 'PageRank'
- ⚠️  错误的名称会导致工具调用失败

### 规则2: 永远不要提取或包含 'G' 参数
- 'G' 代表图对象,由服务器端自动注入,无需传递
- **即使工具 schema 中有 'G' 参数,也必须忽略它**
- parameters 字典中**绝对不能**出现 'G' 键
- 示例正确输出: `{"parameters": {"alpha": "0.85"}}`
- 示例错误输出: `{"parameters": {"G": "current", "alpha": "0.85"}}` ❌

### 规则3: 只提取用户明确指定的参数
- 不要为缺失的参数填充 null、空字符串或占位符
- 如果用户没提到某个参数,就不要在 parameters 中包含它
- **数值参数**: 
  - 整数类型保持整数: `max_iter: 100`
  - 浮点数类型保持浮点数: `alpha: 0.85`

## 后处理代码生成

根据用户需求(如"Top 5"、"大于0.5"、"有多少个")生成 Python 代码:

**必需格式:**
```python
def process(data):
    # data 的结构严格遵循工具的 output_schema['result'] 字段
    # 进行筛选、排序、聚合等操作
    return processed_result  # 返回 dict 或 list
```

示例1: 排序和截取
用户问: "最重要的5个节点?"
假设 output_schema 的 result 是 {"node_id": score} 格式

```python
def process(data):
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:5])
```

示例2: 计数
用户问: "有多少个连通分量?"
假设 result 是 [['A','B'], ['C']] 格式

```python
def process(data):
    return {'component_count': len(data)}
```

示例3: 无特殊需求(默认截取)
用户问: "计算 PageRank"

```python
def process(data):
    if isinstance(data, dict):
        sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:10])
    return data
```

输出格式
返回标准 JSON (不要用 markdown 代码块包裹):

```json
{
  "tool_name": "run_pagerank",
  "parameters": {
    "alpha": 0.85
  },
  "post_processing_code": "def process(data):\\n    return data",
  "reasoning": "选择 run_pagerank 因为..."
}
```

🔴 再次强调: 绝对不要在 parameters 中包含 'G' 参数!"""

        user_prompt = f"""## 对话历史
{conversation_context}

用户问题
{question}

可用工具
{tools_documentation}

任务
请分析问题并返回 JSON,确保:

tool_name 精确匹配列表中的名称

parameters 中绝对不包含 'G' 键

根据 output_schema 生成后处理代码

提供清晰的 reasoning

现在返回 JSON 分析结果:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1500
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM 原始返回:\n{result_text}")
            
            # 清理 markdown 代码块
            result_text = re.sub(r'^```(?:json)?\s*', '', result_text, flags=re.MULTILINE)
            result_text = re.sub(r'\s*```$', '', result_text, flags=re.MULTILINE)
            result_text = result_text.strip()
            
            result_json = json.loads(result_text)
            
            tool_name = result_json.get('tool_name')
            if not tool_name:
                raise ValueError("LLM返回缺少 tool_name 字段")
            
            parameters = result_json.get('parameters', {})
            
            # ⭐ MODIFIED: 安全检查 - 移除任何 G 参数
            if 'G' in parameters:
                logger.warning("⚠️  LLM 错误地提取了 G 参数,已自动移除")
                del parameters['G']
            
            return QuestionAnalysis(
                tool_name=tool_name,
                parameters=parameters,
                post_processing_code=result_json.get('post_processing_code'),
                reasoning=result_json.get('reasoning', "未提供推理说明")
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"❌ LLM 分析失败: {e}\n原始文本:\n{result_text}")
            # 回退到安全的默认工具
            return QuestionAnalysis(
                tool_name="get_graph_info",
                parameters={},
                post_processing_code="def process(data):\n    return data",
                reasoning=f"LLM 分析失败,回退到默认操作。错误: {e}"
            )

    def _build_tools_documentation(self) -> str:
        """构建工具文档 - ⭐ MODIFIED: 过滤掉 G 参数以减少混淆"""
        docs = []
        for tool in self.available_tools:
            input_schema = tool.get('input_schema', {})
            output_schema = tool.get('output_schema', {})
            
            # ⭐ MODIFIED: 从文档中移除 G 参数
            filtered_input = self._filter_g_parameter(input_schema)
            
            doc = (
                f"### 工具: `{tool['name']}`\n"
                f"**描述**: {tool.get('description', '无描述')}\n\n"
                f"**输入参数**:\n"
                f"```json\n{json.dumps(filtered_input, indent=2, ensure_ascii=False)}\n```\n\n"
                f"**返回结构** (你的 `process(data)` 将接收 `result` 字段的数据):\n"
                f"```json\n{json.dumps(output_schema, indent=2, ensure_ascii=False)}\n```\n"
                f"{'-'*50}"
            )
            docs.append(doc)
        return "\n".join(docs)

    def _filter_g_parameter(self, input_schema: Dict) -> Dict:
        """⭐ 新增: 从 input_schema 中移除 G 参数"""
        filtered = input_schema.copy()
        
        # 移除 properties 中的 G
        if 'properties' in filtered and 'G' in filtered['properties']:
            filtered['properties'] = {
                k: v for k, v in filtered['properties'].items() if k != 'G'
            }
        
        # 从 required 列表中移除 G
        if 'required' in filtered and 'G' in filtered['required']:
            filtered['required'] = [r for r in filtered['required'] if r != 'G']
        
        return filtered

    def _build_conversation_context(self) -> str:
        """构建对话上下文"""
        if not self.conversation_history:
            return "(这是对话的第一轮)"
        context_lines = []
        for msg in self.conversation_history[-6:]:
            role = "用户" if msg['role'] == 'user' else "助手"
            content = msg['content']
            context_lines.append(f"{role}: {content[:150]}")
        return "\n".join(context_lines)

    async def _generate_answer_with_llm(self, question: str, tool_name: str,
                                      tool_result: Dict[str, Any]) -> str:
        """使用LLM生成最终回答"""
        
        if not tool_result.get('success'):
            error_msg = tool_result.get('error', '未知错误')
            summary = tool_result.get('summary', '工具执行失败')
            return f"{summary}。错误详情: {error_msg}"

        summary = tool_result.get('summary', '')
        result_data = tool_result.get('result', {})
        result_data_str = json.dumps(result_data, ensure_ascii=False, indent=2)

        # 截断过长的结果
        if len(result_data_str) > 2000:
            result_data_str = result_data_str[:2000] + "\n... (结果过长,已截断)"

        system_prompt = """你是一个专业的数据分析师,负责解释图算法结果。
回答要求:

直接回答用户问题,基于精简后的数据

清晰列出关键数据点

语言通俗易懂,避免重复摘要

控制在1-3句话内"""

        user_prompt = f"""## 用户问题
{question}

执行工具
{tool_name}

执行摘要
{summary}

处理后的结果
{result_data_str}

请用简洁的语言回答用户问题。"""

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
            logger.error(f"❌ LLM 生成回答失败: {e}")
            return summary

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清除")

async def interactive_mode():
    """交互模式"""
    print("=" * 60)
    print("图智能问答系统 (v3 - 完全修复版)")
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
                    print("✅ 对话历史已清除")
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
    asyncio.run(interactive_mode())
