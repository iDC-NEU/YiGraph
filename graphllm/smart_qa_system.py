"""
智能问答系统 - 完整参数提取版
核心改进:
1. LLM 看到完整的 input_schema（包括 G）
2. 提示词明确告知 G 由系统注入
3. 调用工具前自动注入 G 参数
4. 确保不漏掉任何参数（如 backend_kwargs）
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
    
    async def load_graph_data(self, accounts_file: str, transactions_file: str, directed: bool = True):
        """加载并初始化图数据"""
        try:
            logger.info("开始加载图数据...")
            vertices, edges = self.client.load_data_from_csv(accounts_file, transactions_file)
            
            result = await self.client.call_tool("initialize_graph", {
                "vertices": vertices,
                "edges": edges,
                "directed": directed
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
            
            # Step 1: 分析问题（LLM 会看到完整的 schema）
            analysis = await self._analyze_question_with_llm(question)
            logger.info(f"✅ 工具选择: {analysis.tool_name}")
            logger.info(f"✅ LLM 提取的参数: {json.dumps(analysis.parameters, ensure_ascii=False)}")
            
            if analysis.post_processing_code:
                logger.info(f"✅ 后处理代码:\n{analysis.post_processing_code[:200]}...")
            
            # ⭐ 核心修改：在这里注入 G 参数
            # final_parameters = await self._prepare_tool_parameters(
            #     analysis.tool_name, 
            #     analysis.parameters
            # )
            
            logger.info(f"✅ 最终参数（含注入）: {json.dumps(analysis.parameters, ensure_ascii=False)}")
            
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
        
        # ⭐ 修改后的提示词：明确告知 G 的处理方式
        system_prompt = """你是一个专业的图计算调度AI。你的任务是分析用户问题，选择工具，提取参数，并生成后处理代码。

## 🚨 核心规则 (必须严格遵守)

### 规则1: 工具名称必须精确匹配
- 从"可用工具列表"中**精确复制**工具名称
- 例如: 必须写 'run_pagerank' 而不是 'pagerank' 或 'PageRank'
-根据工具中的描述信息准确找到合适的工具

1. **路径/连接性问题**
   - 关键词: "最短路径"、"路径"、"从A到B"、"如何到达"

2. **社区/聚类问题**
   - 关键词: "社区"、"分组"、"聚类"、"模块化"
   → Louvain: `run_louvain_communities` (基于模块化)

3. **连通性/分量问题**
   - 关键词: "连通分量"、"孤立"、"分裂"、"有多少个独立部分"
   → 选择 `run_connected_components`

4. **重要性/影响力问题**（最复杂，需仔细区分）
   - **PageRank**: "全局影响力"、"最重要"、"权威性"、"排名"
     → 选择 `run_pagerank`

### 规则2: 参数提取规则（重要！）
#### 🔴 G 参数的特殊处理
- 你会在工具的 input_schema 中看到 `G` 参数
- **绝对不要**在 `parameters` 中包含 `G` 键
- `G` 参数会由系统在后台自动注入
- 即使 `G` 在 `required` 列表中，也不要提取它

#### ✅ 其他参数的提取规则
- **提取所有用户相关的参数**（如 `alpha`, `max_iter`, `personalization` 等）


### 规则3: 后处理代码生成
根据用户需求(如"Top 5"、"大于0.5"、"有多少个"等)生成 Python 代码:

必需格式:
```python
def process(data):
    # data 的结构严格遵循工具的 output_schema['result'] 字段
    # 进行筛选、排序、聚合等操作
    return processed_result  # 返回 dict 或 list
```

#### 🔴 排序规则
- "前N"、"Top N"、"最高"、"最大" → reverse=True (降序)
- "后N"、"Bottom N"、"最低"、"最小" → reverse=False (升序)

#### 示例1: 排序和截取
用户问: "最重要的5个节点?"
```python
def process(data):
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:5])
```

#### 示例2: 计数
用户问: "有多少个连通分量?"
```python
def process(data):
    return {'component_count': len(data)}
```

### 输出格式
返回标准 JSON (不要用 markdown 代码块包裹):
```json
{
  "tool_name": "run_pagerank",
  "parameters": {
    "alpha": 0.85,
    "max_iter": 100
  },
  "post_processing_code": "def process(data):\\n    return data",
  "reasoning": "选择 run_pagerank 因为..."
}
```

🔴 再次强调:
- 绝对不要在 parameters 中包含 'G'
- 不要在 parameters 中包含 'backend_kwargs'
- 提取所有用户明确指定的其他参数"""

        user_prompt = f"""## 对话历史
{conversation_context}

## 用户问题
{question}

## 可用工具
{tools_documentation}

## 任务
请分析问题并返回 JSON，确保:
- tool_name 精确匹配列表中的名称
- parameters 中不包含 'G' 和 'backend_kwargs'
- 提取所有用户相关的参数
- 根据需求生成后处理代码
- 提供清晰的 reasoning

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
            
            # ⭐ 安全检查：移除不应该提取的参数
            # forbidden_params = {'G', 'backend_kwargs'}
            # removed_params = []
            # for param in forbidden_params:
            #     if param in parameters:
            #         removed_params.append(param)
            #         del parameters[param]
            
            # if removed_params:
            #     logger.warning(f"⚠️ LLM 错误地提取了以下参数，已自动移除: {removed_params}")
            #⭐ 新增：参数类型规范化（核心修改）
            parameters = self._normalize_parameters(tool_name, parameters)
            # 详细参数日志
            self._log_extracted_parameters(tool_name, parameters)
            
            return QuestionAnalysis(
                tool_name=tool_name,
                parameters=parameters,
                post_processing_code=result_json.get('post_processing_code'),
                reasoning=result_json.get('reasoning', "未提供推理说明")
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"❌ LLM 分析失败: {e}\n原始文本:\n{result_text}")
            return QuestionAnalysis(
                tool_name="get_graph_info",
                parameters={},
                post_processing_code="def process(data):\n    return data",
                reasoning=f"LLM 分析失败，回退到默认操作。错误: {e}"
            )

    def _normalize_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """⭐ 新增方法：规范化参数类型，确保节点参数为字符串"""

        # 定义常见的节点参数名称
        node_param_names = {
            'source', 'target', 'node', 'u', 'v', 
            's', 't', 'nodes', 'root', 'start', 'end'
        }

        # 获取工具的参数定义
        tool_info = next((t for t in self.available_tools if t['name'] == tool_name), None)

        if not tool_info:
            logger.warning(f"⚠️ 未找到工具 '{tool_name}' 的定义，跳过参数规范化")
            return parameters

        input_schema = tool_info.get('input_schema', {})
        properties = input_schema.get('parameters', {})

        normalized = {}

        for param_name, param_value in parameters.items():
            param_schema = properties.get(param_name, {})
            param_type = param_schema.get('type', '')
            param_description = param_schema.get('description', '').lower()
            
            # 🔴 判断是否为节点参数的逻辑
            is_node_param = (
                # 1. 参数名在预定义的节点参数列表中
                param_name in node_param_names or
                # 2. 参数类型标注为 'node' 或描述中包含 'node'
                param_type == 'node' or
                'node' in param_description or
                'vertex' in param_description or
                'vertices' in param_description or
                # 3. 特殊情况：参数类型无法判断时，默认转为字符串
                param_type in ['', 'string', 'any']
            )
            
            # 如果是节点参数且当前不是字符串，则转换
            if is_node_param and not isinstance(param_value, str):
                original_value = param_value
                original_type = type(param_value).__name__
                normalized[param_name] = str(param_value)
                logger.info(
                    f"🔄 参数类型转换: {param_name} = {original_value} "
                    f"({original_type}) → '{normalized[param_name]}' (str)"
                )
            # 如果是列表类型的节点参数，转换列表中的每个元素
            elif is_node_param and isinstance(param_value, list):
                normalized[param_name] = [str(v) if not isinstance(v, str) else v for v in param_value]
                logger.info(f"🔄 参数类型转换: {param_name} (list) 中的元素转换为字符串")
            else:
                # 保持原值
                normalized[param_name] = param_value

        return normalized

    def _log_extracted_parameters(self, tool_name: str, parameters: Dict[str, Any]):
        """⭐ 新增：详细打印提取的参数"""
        logger.info("=" * 70)
        logger.info("📋 LLM 提取的参数详情")
        logger.info("=" * 70)
        
        tool_info = next((t for t in self.available_tools if t['name'] == tool_name), None)
        
        if tool_info:
            input_schema = tool_info.get('input_schema', {})
            properties = input_schema.get('parameters', {})
            required = input_schema.get('required', [])
            if parameters:
                logger.info(f"✅ 共提取了 {len(parameters)} 个参数:\n")
                
                for param_name, param_value in parameters.items():
                    param_schema = properties.get(param_name, {})
                    param_type = param_schema.get('type', '未定义')
                    param_description = param_schema.get('description', '无描述')
                    is_required = "✓ 必需" if param_name in required else "可选"
                    
                    actual_type = type(param_value).__name__
                    logger.info(f"  参数名: {param_name} ({is_required})")
                    logger.info(f"    ├─ 值: {param_value}")
                    logger.info(f"    ├─ 预期类型: {param_type}")
                    logger.info(f"    ├─ 实际类型: {actual_type}")
                    logger.info(f"    └─ 描述: {param_description}\n")
            else:
                logger.info("ℹ️ 未提取任何参数（可能所有参数都有默认值）")
            
            # 检查遗漏的必需参数（排除系统注入的）
            system_params = {'G'}
            extracted = set(parameters.keys())
            required_set = set(required) - system_params
            missing = required_set - extracted
            
            if missing:
                logger.warning(f"⚠️ 遗漏的必需参数: {missing}")
            else:
                logger.info("✅ 所有必需参数都已提取")
        else:
            logger.warning(f"⚠️ 未找到工具 '{tool_name}' 的定义信息")
        
        logger.info("=" * 70)

    def _build_tools_documentation(self) -> str:
        """构建工具文档 - 保持完整的 schema（包括 G）"""
        docs = []
        for tool in self.available_tools:
            name=tool['name']
            if (name == 'run_bellman_ford_path' or name == 'run_connected_components' or name == 'run_louvain_communities' or name == 'run_pagerank' or name == 'get_graph_info') :
            # 获取完整的 input/output schema
                input_schema = tool.get('input_schema', {})
                output_schema = tool.get('output_schema', {})
                    
                    # ⭐ 添加标注：哪些参数是系统注入的
                annotated_input = self._annotate_schema(input_schema)
                    
                doc = (
                        f"### 工具: `{tool['name']}`\n"
                        f"**描述**: {tool.get('description', '无描述')}\n\n"
                        f"**输入参数** (标注了系统注入的参数):\n"
                        f"```json\n{json.dumps(annotated_input, indent=2, ensure_ascii=False)}\n```\n\n"
                        f"**返回结构** (你的 `process(data)` 将接收 `result` 字段的数据):\n"
                        f"```json\n{json.dumps(output_schema, indent=2, ensure_ascii=False)}\n```\n"
                        f"{'-'*50}"
                    )
                docs.append(doc)
                logger.debug(f"📝 工具文档:\n{doc}")
        return "\n".join(docs)

    def _annotate_schema(self, input_schema: Dict) -> Dict:
        """⭐ 新增：为 schema 添加标注，说明哪些参数由系统注入"""
        annotated = input_schema.copy()
        
        if 'parameters' in annotated:
            parameters = annotated['parameters'].copy()
            
            # 标注 G 参数
            if 'G' in parameters:
                parameters['G'] = {
                    **parameters['G'],
                    'description': f"[系统自动注入] {parameters['G'].get('description', '图对象')}"
                }
            
            # 标注 backend_kwargs
            if 'backend_kwargs' in parameters:
                parameters['backend_kwargs'] = {
                    **parameters['backend_kwargs'],
                    'description': f"[系统自动注入，默认为{{}}] {parameters['backend_kwargs'].get('description', '后端参数')}"
                }
            
            annotated['parameters'] = parameters
        
        return annotated

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
            result_data_str = result_data_str[:2000] + "\n... (结果过长，已截断)"

        system_prompt = """你是一个专业的数据分析师，负责解释图算法结果。
回答要求:
- 直接回答用户问题，基于精简后的数据
- 清晰列出关键数据点
- 语言通俗易懂，避免重复摘要
- 如果数据不足以回答问题，说明原因"""

        user_prompt = f"""## 用户问题
{question}

## 执行工具
{tool_name}

## 执行摘要
{summary}

## 处理后的结果
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
    print("图智能问答系统 (v4 - 完整参数提取版)")
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
