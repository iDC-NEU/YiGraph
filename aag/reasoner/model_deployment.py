import numpy as np
import os
import openai
import json
import re
import requests # deepseek
from typing import Dict, Literal, Any, List, Optional
from llama_index.core import Settings
from llama_index.core.utils import print_text
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage, MessageRole

from aag.config.engine_config import ReasonerConfig
from aag.reasoner.prompt_template.llm_prompt_en import *
from aag.reasoner.prompt_template.llm_prompt_zh import *
from aag.utils.parse_json import extract_json_from_response, parse_openai_json_response
from aag.error_recovery.enhancer import enhance_prompt




EMBEDD_DIMS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384
}

DAG_REVISION_SYSTEM_PROMPT = (
    "You are an expert workflow planner for a graph-computation pipeline. "
    "You will modify an existing subquery DAG plan based on user edit instructions. "
    "\n\n### Responsibilities:\n"
    "1. Interpret user edits precisely (add / delete / modify nodes or dependencies). "
    "2. Maintain logical correctness and valid dependencies (no cycles, causal order only). "
    "3. Each node must contain exactly: id, query, depends_on. "
    "4. Output must strictly follow the JSON schema below, with NO explanations and NO extra text.\n\n"
    "### Required Output JSON Schema:\n"
    "{\n"
    "  \"subqueries\": [\n"
    "    {\n"
    "      \"id\": \"qX\",\n"
    "      \"query\": \"text\",\n"
    "      \"depends_on\": []\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "### Example Input (Before Modification):\n"
    "{\n"
    "  \"subqueries\": [\n"
    "    {\"id\": \"q1\", \"query\": \"Check if user Anna is a high-risk user.\", \"depends_on\": []},\n"
    "    {\"id\": \"q2\", \"query\": \"List all potential money laundering pathways around Anna.\", \"depends_on\": [\"q1\"]},\n"
    "    {\"id\": \"q3\", \"query\": \"Estimate the amount of cash that may have been illegally transferred out in relation to Anna.\", \"depends_on\": [\"q2\"]},\n"
    "    {\"id\": \"q4\", \"query\": \"Find the account with the largest transaction amount in the suspicious paths.\", \"depends_on\": [\"q2\"]}\n"
    "  ]\n"
    "}\n\n"
    "### Example User Edit Instruction:\n"
    "Add a step between node 1 and node 2 to identify Anna's fraud community. "
    "Modify node 4 so that it becomes: identify the account with the largest transaction amount within Anna’s community.\n\n"
    "### Example Output:\n"
    "{\n"
    "  \"subqueries\": [\n"
    "    {\"id\": \"q1\", \"query\": \"Check if user Anna is a high-risk user.\", \"depends_on\": []},\n"
    "    {\"id\": \"q2\", \"query\": \"Identify the potential fraud community in which Anna resides to narrow the scope of subsequent risk monitoring.\", \"depends_on\": [\"q1\"]},\n"
    "    {\"id\": \"q3\", \"query\": \"List all potential money laundering pathways within the high-risk community where Anna is located.\", \"depends_on\": [\"q2\"]},\n"
    "    {\"id\": \"q4\", \"query\": \"Estimate the amount of cash that may have been illegally transferred out in relation to Anna.\", \"depends_on\": [\"q3\"]},\n"
    "    {\"id\": \"q5\", \"query\": \"Identify the account with the largest transaction amount within Anna's fraud community.\", \"depends_on\": [\"q2\"]}\n"
    "  ]\n"
    "}"
)


def build_dag_revision_user_prompt(current_plan: Dict[str, Any], user_request: str) -> str:
    plan_text = json.dumps(current_plan, ensure_ascii=False, indent=2)
    normalized_request = user_request.strip()
    return (
        "Current subquery_plan:\n"
        f"{plan_text}\n\n"
        "User request:\n"
        f"{normalized_request}\n\n"
        "Update the plan so it satisfies the request. Rules:\n"
        "1. Output JSON only with the shape {\"subqueries\": [...]}.\n"
        "2. Each entry needs \"id\", \"query\", and \"depends_on\" (list) fields.\n"
        "3. Preserve existing ids when editing content; introduce new ids only for new steps.\n"
        "4. Keep dependencies acyclic and align them with the described changes.\n"
        "5. If the user removes or inserts nodes between two ids, reflect that explicitly.\n"
        "Return JSON only."
    )


class EmbeddingEnv:

    def __init__(self,
                 embed_name="BAAI/bge-large-en-v1.5",
                 embed_batch_size=20,
                 device="cuda:0"):
        self.embed_name = embed_name
        self.embed_batch_size = embed_batch_size

        assert embed_name in EMBEDD_DIMS
        self.dim = EMBEDD_DIMS[embed_name]

        if 'BAAI' in embed_name:
            print(f"use huggingface embedding {embed_name}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=embed_name,
                embed_batch_size=embed_batch_size,
                device=device)
        else:
            print(f"use openai embedding {embed_name}")
            self.embed_model = OpenAIEmbedding(
                model=embed_name, embed_batch_size=embed_batch_size)

        Settings.embed_model = self.embed_model
        print_text(
            f"EmbeddingEnv: embed_name {embed_name}, embed_batch_size {self.embed_batch_size}, dim {self.dim}\n",
            color='red')

    def __str__(self):
        return f"{self.embed_name} {self.embed_batch_size}"

    def get_embedding(self, query_str):
        embedding = self.embed_model._get_text_embedding(query_str)
        return embedding

    def get_embeddings(self, query_str_list):
        embeddings = self.embed_model._get_text_embeddings(query_str_list)
        return embeddings

    def calculate_similarity(self, query1, query2):
        # Cosine similarity
        embedding1 = self.embed_model.get_embedding(query1)
        embedding2 = self.embed_model.get_embedding(query2)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = round(dot_product / (norm1 * norm2), 6)
        return similarity

    # def parallel_generate_embedding(self,
    #                                 entities,
    #                                 step,
    #                                 devices=["cuda:0", "cuda:2"]):
    #     nproc = len(devices)
    #     job_len = len(entities) // nproc
    #     jobs = [
    #         entities[i:i + job_len] for i in range(0, len(entities), job_len)
    #     ]

    #     # multi processing
    #     for i, job in enumerate(range(jobs)):
    #         save_path = './embedding_{i}.json'
    #         process_one_job(job, step, save_path)


class OllamaEnv:

    ROLE_MAP = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }

    def __init__(self,
                 llm_mode_name="llama3.1:70b",
                 llm_embed_name="BAAI/bge-large-en-v1.5",
                 chunk_size=512,
                 chunk_overlap=20,
                 embed_batch_size=20,
                 device='cuda:2',
                 timeout=150000,
                 port=11434,
                 verbose=False):

        base_url = f"http://localhost:{port}"
        Settings.llm = Ollama(model=llm_mode_name, request_timeout=timeout,
                              temperature=0.0, base_url=base_url)  # , device=device
        self.verbose = verbose

        if 'BAAI' in llm_embed_name:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=llm_embed_name,
                embed_batch_size=embed_batch_size,
                device=device)
        else:
            print(f"use openai embedding {llm_embed_name}")
            Settings.embed_model = OpenAIEmbedding(
                model=llm_embed_name, embed_batch_size=embed_batch_size)

        if llm_embed_name not in EMBEDD_DIMS.keys():
            raise NotImplementedError('embed model not support!')

        self.llm = Settings.llm
        self.embed_model = Settings.embed_model
        self.dim = EMBEDD_DIMS[llm_embed_name]
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        print(
            f"llm_mode_name: {llm_mode_name}, llm_embed_name: {llm_embed_name}, chunk_size: {Settings.chunk_size}, chunk_overlap: {chunk_overlap}")


    def chat(self, messages: list) -> str:
        if messages and isinstance(messages[0], dict):
            messages = [
                ChatMessage(
                    role=self.ROLE_MAP.get(m["role"], MessageRole.USER),
                    content=m.get("content", "")
                )
                for m in messages
            ]
        response = self.llm.chat(messages=messages)
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content
        return response.raw

    def generate_response(self, query: str):
        response = self.llm.complete(query)
        return response
    
    def execute_prompt(self, full_prompt: str, parse_json: bool = True, response_format: Optional[Dict] = None) -> Any:
        """Execute a formatted prompt and return the response.
        
        Args:
            full_prompt: The fully formatted prompt string
            parse_json: Whether to parse the response as JSON
            response_format: Optional response format dict (for OpenAI compatibility)
        
        Returns:
            Parsed JSON dict if parse_json=True, otherwise raw response text
        """
        response = self.llm.complete(full_prompt)
        if parse_json:
            return extract_json_from_response(response.text)
        return response.text
        
    def check_data_dependency(
            self,
            q1_question: str,
            q1_algorithm: str,
            q2_question: str,
            q2_algorithm: str) -> bool:
        """Determine whether Q2 depends on the result of Q1 using the LLM."""
        try:
            full_prompt = check_data_dependency_prompt.format(
                q1_question=q1_question,
                q1_algorithm=q1_algorithm,
                q2_question=q2_question,
                q2_algorithm=q2_algorithm
            )
            result = self.execute_prompt(full_prompt, parse_json=True)
            depends = result.get("q2_depends_on_q1")
            if isinstance(depends, bool):
                return depends
            if isinstance(depends, str):
                normalized = depends.strip().lower()
                if normalized in {"true", "yes"}:
                    return True
                if normalized in {"false", "no"}:
                    return False
        except Exception as e:
            print(f"Error determining data dependency with Ollama: {e}")
        return False
        
    def plan_subqueries(self, decompose: bool, query: str) -> dict:
        if decompose == False: 
            # Do not decompose, treat as a single question
            return{"subqueries": [{
                        "id": "q1",
                        "query": query,
                        "depends_on": []
                    }]}  
        full_prompt = plan_subqueries_prompt.format(query=query)
        return self.execute_prompt(full_prompt, parse_json=True)

    def classify_question_type(self, question: str) -> dict:
        """Classify whether a question requires graph algorithm or numeric analysis."""
        full_prompt = classify_question_type_prompt.format(question=question)
        return self.execute_prompt(full_prompt, parse_json=True)


    def revise_subquery_plan(self, current_plan: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        full_prompt = revise_subquery_plan_prompt.format(
            current_plan=json.dumps(current_plan, ensure_ascii=False),
            user_request=user_request
        )
        return self.execute_prompt(full_prompt, parse_json=True)

    def select_task_type(self, question: str, task_type_list: list) -> dict:
        full_prompt = select_task_type_prompt.format(
            question=question,
            task_type_list=task_type_list
        )
        return self.execute_prompt(full_prompt, parse_json=True)


    def select_algorithm(self, question: str, algorithm_list: list, graph_schema: Optional[Dict[str, Any]] = None) -> dict:
        schema_context = ""
        if graph_schema:
            schema_context = f"""

Current Graph Dataset Schema:
- Dataset: {graph_schema.get('dataset_name', 'Unknown')}
- Graph Type: {'Directed' if graph_schema.get('graph_properties', {}).get('directed') else 'Undirected'}, {'Heterogeneous' if graph_schema.get('graph_properties', {}).get('heterogeneous') else 'Homogeneous'}, {'Multigraph' if graph_schema.get('graph_properties', {}).get('multigraph') else 'Simple'}, {'Weighted' if graph_schema.get('graph_properties', {}).get('weighted') else 'Unweighted'}
- Vertex Types: {', '.join(graph_schema.get('vertex_types', []))}
- Edge Types: {', '.join(graph_schema.get('edge_types', []))}
- Vertex Configurations: {json.dumps(graph_schema.get('vertex_configs', []), ensure_ascii=False, indent=2)}
- Edge Configurations: {json.dumps(graph_schema.get('edge_configs', []), ensure_ascii=False, indent=2)}

Please consider this schema when selecting the algorithm to ensure compatibility.
"""
        
        response = self.llm.complete(select_algorithm_prompt.format(
                question=question,
                algorithm_list=algorithm_list
            ) + schema_context)
        return extract_json_from_response(response.text)
        
    
    def extract_parameters_with_postprocess(self, question: str, tool_description: str) -> dict:
        full_prompt = extract_parameters_with_postprocess_promt.format(
            question=question,
            tool_description=tool_description
        )
        return self.execute_prompt(full_prompt, parse_json=True)
    
    def extract_parameters_with_postprocess_new(self, question: str, tool_description: str, vertex_schema: Dict[str, str], edge_schema: Dict[str, str]) -> dict:
        """Extract parameters and generate post-processing code with vertex and edge schema information."""
        response = self.llm.complete(extract_parameters_with_postprocess_promt_new.format(
            question=question,
            tool_description=tool_description,
            vertex_schema=json.dumps(vertex_schema, indent=2),
            edge_schema=json.dumps(edge_schema, indent=2)
        ))
        return extract_json_from_response(response.text)
    
    def merge_parameters_from_dependencies(
        self, 
        question: str, 
        tool_description: str, 
        vertex_schema: Dict[str, str], 
        edge_schema: Dict[str, str],
        dependency_parameters: Dict[str, Any]
    ) -> dict:
        """Merge dependency parameters with extracted parameters and generate post-processing code."""
        full_prompt = merge_parameters_with_dependencies_prompt.format(
            question=question,
            tool_description=tool_description,
            dependency_parameters=json.dumps(dependency_parameters, indent=2),
            vertex_schema=json.dumps(vertex_schema, indent=2),
            edge_schema=json.dumps(edge_schema, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)
        

    def generate_answer_from_algorithm_result(self, question: str, tool_description: str, tool_result: Dict[str, Any]) -> str:
        prompt = f"""
你是一名专业的数据分析师，负责解读图算法的计算结果。你的任务是基于以下内容，对某个工具的计算输出进行分析：
- **自然语言用户问题**（*question*）
- **对应的工具描述**（*tool_description*）
- **该工具的实际执行结果**（*tool_result*）

每个工具都代表一种特定的图算法（例如 PageRank、Louvain、最短路径等）。你需要生成一份简洁、易懂的分析报告，直接回答用户的问题，并参考下面给出的分析报告格式示例。

----------------------------
## 响应要求
- 你的解释必须严格基于给定的计算结果。
- 清晰突出并解释关键数据点（例如排名靠前的节点、聚类数量、路径长度等）。
- 使用简单、直观的语言，避免术语堆砌，也不要重复同样的总结。
- 如果现有数据不足以完整回答问题，要明确指出限制或缺失的信息。
- 输出格式必须为 **Markdown**，以保证展示清晰。
- 每一步分析都必须有明确编号和简短标题。例如：## 1. **Anna Lee 节点中心性评估（PageRank 结果）**
- 最终输出**必须包含 Summary（总结）部分和 Recommendations（建议）部分**。
- 请按照下面给出的分析报告模板来组织你的回答。

----------------------------
## 输入：
### 用户问题：
{question}

### 工具描述：
{tool_description}

### 工具执行结果：
{tool_result}

----------------------------
## 输出

请用通俗易懂的英文写出一份清晰、简洁的解释，直接回答用户的问题，并按照以下分析报告格式组织内容：

----------------------------
**分析报告示例：**

## 1. **Anna Lee 节点中心性评估（PageRank 结果）**

首先，我们使用 **PageRank 图算法** 对交易网络中所有账户节点的“重要性”和“风险关联性”进行了量化评估。PageRank 的基本思想是：如果一个账户频繁与多个高风险账户发生交易，或者处于交易网络中的关键结构位置，那么它的得分会显著升高，从而反映出它在网络中的潜在影响力和风险暴露程度。

结果显示，**Anna Lee** 的 PageRank 得分位于所有客户中的前 2%，显著高于其他用户。这说明该账户在交易网络中被其他节点高度“依赖”，表现出较强的风险扩散潜力。因此，可以初步将 Anna Lee 识别为高风险客户或重点监控对象。

## 2. **可疑路径识别（DFS 结果）**

接下来，我们使用 **DFS（深度优先搜索）图算法** 对与 Anna Lee 相关的资金流进行了全链路穿透分析。执行 DFS 算法后，我们识别出一条以 Anna Lee 为起点的闭环可疑路径：

- **Anna Lee → Gill Zachary → Garcia Marcus → Nunez Mitchell → Robinson David → Anna Lee**

这条路径具有典型的闭环结构：资金从 Anna Lee 转出，经过多个中间账户多步流转后，又回流到 Anna Lee 账户，形成完整的“转出 → 中转 → 回流”链路。该结构在反洗钱风险识别中通常被视为高风险模式，因为它常被用来掩盖交易来源或隐藏真实资金流向。

## 3. **金额评估（Python 代码统计结果）**

基于步骤 2 中识别出的路径，AAG 自动生成了 Python 数据处理代码，对该闭环路径中涉及的所有交易金额进行了提取和汇总。各步交易金额如下：

- Anna Lee → Gill Zachary，交易金额为 879.94 $
- Gill Zachary → Garcia Marcus，交易金额为 210.43 $
- Garcia Marcus → Nunez Mitchell，交易金额为 472.69 $
- Nunez Mitchell → Robinson David，交易金额为 606.94 $
- Robinson David → Anna Lee，交易金额为 825.22 $

该路径的总交易金额为 **2995.22 $**。

## 4. **最大交易账户识别（Python 代码统计结果）**

基于步骤 2 中识别出的路径，AAG 自动生成了 Python 数据处理代码，对路径中所有相关账户的历史交易总额进行了统计和排序。结果如下：

- Anna Lee 的总交易额为 161272.07 $
- Nunez Mitchell 的总交易额为 122984.06 $
- Garcia Marcus 的总交易额为 115367.59 $
- Robinson David 的总交易额为 96550.65 $
- Gill Zachary 的总交易额为 64109.86 $

结果表明，Anna Lee 和 Nunez Mitchell 的历史交易总额明显更高，说明它们可能在资金流动中扮演资金归集点的角色，需要进一步核查其业务活动和资金流向。

## **Summary**

本次分析利用图算法对 Anna Lee 的洗钱风险进行了综合评估，主要使用了 **PageRank 中心性算法** 和 **DFS（深度优先搜索）路径算法**。首先，PageRank 对网络中所有账户的重要性和风险关联性进行了量化，结果显示 **Anna Lee 的 PageRank 得分位于前 2%**，说明其处于高度关键的节点位置，具有较强的风险扩散潜力。随后，DFS 对以 Anna Lee 为起点的资金流进行了穿透分析，识别出一条闭环可疑路径：**Anna Lee → Gill Zachary → Garcia Marcus → Nunez Mitchell → Robinson David → Anna Lee**，总交易金额为 **2995.22 $**，表现出典型的“资金转出—多步中转—回流原账户”的高风险结构。此外，金额统计结果显示 Anna Lee（161272.07 $）和 Nunez Mitchell（122984.06 $）的历史交易总额显著偏高，进一步表明它们可能是资金归集或中转的关键节点，整体可疑程度较高。

## **Recommendations**

1. 建议将 Anna Lee 及闭环路径中的相关账户（尤其是 Nunez Mitchell 和 Robinson David）纳入高风险名单，并提升监控等级。对其后续的大额、循环型交易设置严格的实时预警和限额控制。
2. 建议对 Anna Lee 及关键交易对手开展专项尽职调查，重点核查其资金来源、交易目的和业务背景。同时结合更长周期的交易记录与外部信息，排查是否存在结构化拆分、循环转账等洗钱特征。

----------------------------"""
        response = self.llm.complete(prompt)
        response_text = str(response.text).strip()
        if not response_text:
            return "Unable to generate answer from the algorithm result."
        return response_text
    
    def analyze_dependency_type_and_locate_dependency_data(self, current_question:str, task_type:str, current_algo_desc:str, parent_question: str,  parent_outputs_meta:list)-> dict:
        full_prompt = analyze_dependency_type_and_locate_dependency_data_prompt.format(
            current_question=current_question,
            task_type=task_type,
            current_algo_desc=current_algo_desc,
            parent_question=parent_question,
            parent_outputs_meta=parent_outputs_meta
        )
        return self.execute_prompt(full_prompt, parse_json=True)
 
    def map_parameters(self, current_question: str, current_algo_desc: str, dependency_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_prompt = map_parameters_prompt.format(
            current_question=current_question,
            algo_desc=current_algo_desc,
            dependency_items=json.dumps(dependency_items, ensure_ascii=False, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)
    
    def generate_graph_conversion_code(self, current_question: str, dependency_items: List[Dict[str, Any]])-> Dict[str, Any]:
        full_prompt = generate_graph_conversion_code_prompt.format(
            current_question=current_question,
            dependency_items=json.dumps(dependency_items, ensure_ascii=False, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)
    
    def generate_numeric_analysis_code(
        self, 
        question: str, 
        dependency_items: List[Dict[str, Any]], 
        vertex_schema: Dict[str, str], 
        edge_schema: Dict[str, str]
    ) -> Dict[str, Any]:
        full_prompt = generate_numeric_analysis_code_prompt.format(
            question=question,
            dependency_data_items=json.dumps(dependency_items, ensure_ascii=False, indent=2),
            vertex_schema=json.dumps(vertex_schema, ensure_ascii=False, indent=2),
            edge_schema=json.dumps(edge_schema, ensure_ascii=False, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)
    
    def nl_query_classify_type(self, question: str, query_templates: dict) -> str:
        """Classify the query type for natural language query engine."""
        full_prompt = nl_query_classify_type_prompt.format(
            question=question,
            query_templates=json.dumps(query_templates, ensure_ascii=False, indent=2)
        )
        response_text = self.execute_prompt(full_prompt, parse_json=False)
        return response_text.strip().strip('"').strip("'")
    
    def nl_query_extract_params(self, question: str, query_type: str, template: dict,
                                schema_info: str, query_modifiers: dict) -> dict:
        """Extract parameters for natural language query engine."""
        full_prompt = nl_query_extract_params_prompt.format(
            schema_info=schema_info,
            query_type=query_type,
            template_description=template['description'],
            template_method=template['method'],
            required_params=template['required_params'],
            optional_params=template.get('optional_params', []),
            query_modifiers=json.dumps(query_modifiers, ensure_ascii=False, indent=2),
            question=question
        )
        return self.execute_prompt(full_prompt, parse_json=True)
 


class OpenAIEnv:

    def __init__(self, 
                 base_url,
                 api_key,
                 model_name):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model_name

        openai.api_key = self.api_key
        openai.base_url = self.base_url
        self.client = openai

    def execute_prompt(self, full_prompt: str, parse_json: bool = True, response_format: Optional[Dict] = None) -> Any:
        """Execute a formatted prompt and return the response.
        
        Args:
            full_prompt: The fully formatted prompt string
            parse_json: Whether to parse the response as JSON
            response_format: Optional response format dict (for OpenAI compatibility)
        
        Returns:
            Parsed JSON dict if parse_json=True, otherwise raw response text
        """
        messages = [{"role": "user", "content": full_prompt}]
        request_kwargs = {"model": self.model, "messages": messages}
        
        if response_format:
            request_kwargs["response_format"] = response_format
        elif parse_json:
            # Use JSON mode for better JSON parsing
            request_kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**request_kwargs)
        response_text = response.choices[0].message.content
        
        if parse_json:
            return parse_openai_json_response(response_text, "execute_prompt")
        return response_text

    def check_data_dependency(
            self,
            q1_question: str,
            q1_algorithm: str,
            q2_question: str,
            q2_algorithm: str) -> bool:
        """Use the llm to assess whether Q2 depends on Q1."""
        try:
            full_prompt = check_data_dependency_prompt.format(
                q1_question=q1_question,
                q1_algorithm=q1_algorithm,
                q2_question=q2_question,
                q2_algorithm=q2_algorithm
            )
            result = self.execute_prompt(full_prompt, parse_json=True, response_format={"type": "json_object"})
            depends = result.get("q2_depends_on_q1")
            if isinstance(depends, bool):
                return depends
            if isinstance(depends, str):
                normalized = depends.strip().lower()
                if normalized in {"true", "yes"}:
                    return True
                if normalized in {"false", "no"}:
                    return False
        except Exception as e:
            print(f"Error determining data dependency with OpenAI: {e}")
        return False

    def generate_response(self, query: str):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI generate_response: {e}")
            return None
    
    
    def plan_subqueries(self, decompose: bool, query: str) -> dict:
        if decompose == False: 
            # Do not decompose, treat as a single question
            return{"subqueries": [{
                        "id": "q1",
                        "query": query,
                        "depends_on": []
                    }]}  
        full_prompt = plan_subqueries_prompt.format(query=query)
        return self.execute_prompt(full_prompt, parse_json=True)

    def revise_subquery_plan(
            self,
            current_plan: Dict[str, Any],
            user_request: str) -> Dict[str, Any]:
        full_prompt = revise_subquery_plan_prompt.format(
            current_plan=json.dumps(current_plan, ensure_ascii=False), 
            user_request=user_request
        )
        return self.execute_prompt(full_prompt, parse_json=True)

    def classify_question_type(self, question: str) -> dict:
        """Classify whether a question requires graph algorithm or numeric analysis."""
        full_prompt = classify_question_type_prompt.format(question=question)
        return self.execute_prompt(full_prompt, parse_json=True)
    
    def select_task_type(self, question: str, task_type_list: list) -> dict:
        full_prompt = select_task_type_prompt.format(
            question=question,
            task_type_list=task_type_list
        )
        return self.execute_prompt(full_prompt, parse_json=True)
    
    def select_algorithm(self, question: str, algorithm_list: list, graph_schema: Optional[Dict[str, Any]] = None) -> dict:
        schema_context = ""
        if graph_schema:
            schema_context = f"""

Current Graph Dataset Schema:
- Dataset: {graph_schema.get('dataset_name', 'Unknown')}
- Graph Type: {'Directed' if graph_schema.get('graph_properties', {}).get('directed') else 'Undirected'}, {'Heterogeneous' if graph_schema.get('graph_properties', {}).get('heterogeneous') else 'Homogeneous'}, {'Multigraph' if graph_schema.get('graph_properties', {}).get('multigraph') else 'Simple'}, {'Weighted' if graph_schema.get('graph_properties', {}).get('weighted') else 'Unweighted'}
- Vertex Types: {', '.join(graph_schema.get('vertex_types', []))}
- Edge Types: {', '.join(graph_schema.get('edge_types', []))}
- Vertex Configurations: {json.dumps(graph_schema.get('vertex_configs', []), ensure_ascii=False, indent=2)}
- Edge Configurations: {json.dumps(graph_schema.get('edge_configs', []), ensure_ascii=False, indent=2)}

Please consider this schema when selecting the algorithm to ensure compatibility.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": select_algorithm_prompt.format(
                question=question,
                algorithm_list=algorithm_list
            ) + schema_context}]
        )
        response_text = response.choices[0].message.content
        return parse_openai_json_response(response_text, "select_algorithm")
    
    def extract_parameters_with_postprocess(self, question: str, tool_description: str) -> dict:
        full_prompt = extract_parameters_with_postprocess_promt.format(
            question=question,
            tool_description=tool_description
        )
        return self.execute_prompt(full_prompt, parse_json=True)
    
    def extract_parameters_with_postprocess_new(self, question: str, tool_description: str, vertex_schema: Dict[str, str], edge_schema: Dict[str, str]) -> dict:
        """Extract parameters and generate post-processing code with vertex and edge schema information."""
        full_prompt = extract_parameters_with_postprocess_promt_new.format(
            question=question,
            tool_description=tool_description,
            vertex_schema=json.dumps(vertex_schema, indent=2),
            edge_schema=json.dumps(edge_schema, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)

    
    def merge_parameters_from_dependencies(
        self, 
        question: str, 
        tool_description: str, 
        vertex_schema: Dict[str, str], 
        edge_schema: Dict[str, str],
        dependency_parameters: Dict[str, Any]
    ) -> dict:
        """Merge dependency parameters with extracted parameters and generate post-processing code."""
        full_prompt = merge_parameters_with_dependencies_prompt.format(
            question=question,
            tool_description=tool_description,
            dependency_parameters=json.dumps(dependency_parameters, indent=2),
            vertex_schema=json.dumps(vertex_schema, indent=2),
            edge_schema=json.dumps(edge_schema, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)

    def generate_answer_from_algorithm_result(self, question: str, tool_description: str, tool_result: Dict[str, Any]) -> str:
        # prompt = f"""
        # You are a professional data analyst responsible for interpreting graph algorithm results.
        # Your task is to analyze the computation output of a given tool based on:
        # - a natural-language user question (*question*),
        # - the corresponding tool description (*tool_description*), and
        # - the actual execution result of that tool (*tool_result*).

        # Each tool represents a specific graph algorithm (e.g., PageRank, Louvain, Shortest Path, etc.).
        # You should generate a concise, human-understandable analytical explanation that directly answers the user question.

        # ----------------------------
        # ## Response Requirements
        # - Base your explanation strictly on the provided computation result.
        # - Clearly highlight and explain key data points (e.g., top-ranked nodes, cluster counts, path lengths).
        # - Use simple, intuitive language — avoid jargon and avoid repeating the same summary.
        # - If the available data is insufficient to fully answer the question, explicitly state the limitation or missing information.

        # ----------------------------
        # ## Inputs:
        # ### User Question:
        # {question}
        # ### Tool Description:
        # {tool_description}
        # ### Tool Execution Result:
        # {tool_result}

        # ----------------------------
        # ## Output
        # Provide a clear and concise written explanation in plain English that directly addresses the user’s question.
        # Do NOT return JSON or code — only natural language.
        # """

        # 
        prompt = f"""
你是一名专业的数据分析师，负责解读图算法的计算结果。你的任务是基于以下内容，对某个工具的计算输出进行分析：
- **自然语言用户问题**（*question*）
- **对应的工具描述**（*tool_description*）
- **该工具的实际执行结果**（*tool_result*）

每个工具都代表一种特定的图算法（例如 PageRank、Louvain、最短路径等）。你需要生成一份简洁、易懂的分析报告，直接回答用户的问题，并参考下面给出的分析报告格式示例。

----------------------------
## 响应要求
- 你的解释必须严格基于给定的计算结果。
- 清晰突出并解释关键数据点（例如排名靠前的节点、聚类数量、路径长度等）。
- 使用简单、直观的语言，避免术语堆砌，也不要重复同样的总结。
- 如果现有数据不足以完整回答问题，要明确指出限制或缺失的信息。
- 输出格式必须为 **Markdown**，以保证展示清晰。
- 每一步分析都必须有明确编号和简短标题。例如：## 1. **Anna Lee 节点中心性评估（PageRank 结果）**
- 最终输出**必须包含 Summary（总结）部分和 Recommendations（建议）部分**。
- 请按照下面给出的分析报告模板来组织你的回答。

----------------------------
## 输入：
### 用户问题：
{question}

### 工具描述：
{tool_description}

### 工具执行结果：
{tool_result}

----------------------------
## 输出

请用通俗易懂的英文写出一份清晰、简洁的解释，直接回答用户的问题，并按照以下分析报告格式组织内容：

----------------------------
**分析报告示例：**

## 1. **Anna Lee 节点中心性评估（PageRank 结果）**

首先，我们使用 **PageRank 图算法** 对交易网络中所有账户节点的“重要性”和“风险关联性”进行了量化评估。PageRank 的基本思想是：如果一个账户频繁与多个高风险账户发生交易，或者处于交易网络中的关键结构位置，那么它的得分会显著升高，从而反映出它在网络中的潜在影响力和风险暴露程度。

结果显示，**Anna Lee** 的 PageRank 得分位于所有客户中的前 2%，显著高于其他用户。这说明该账户在交易网络中被其他节点高度“依赖”，表现出较强的风险扩散潜力。因此，可以初步将 Anna Lee 识别为高风险客户或重点监控对象。

## 2. **可疑路径识别（DFS 结果）**

接下来，我们使用 **DFS（深度优先搜索）图算法** 对与 Anna Lee 相关的资金流进行了全链路穿透分析。执行 DFS 算法后，我们识别出一条以 Anna Lee 为起点的闭环可疑路径：

- **Anna Lee → Gill Zachary → Garcia Marcus → Nunez Mitchell → Robinson David → Anna Lee**

这条路径具有典型的闭环结构：资金从 Anna Lee 转出，经过多个中间账户多步流转后，又回流到 Anna Lee 账户，形成完整的“转出 → 中转 → 回流”链路。该结构在反洗钱风险识别中通常被视为高风险模式，因为它常被用来掩盖交易来源或隐藏真实资金流向。

## 3. **金额评估（Python 代码统计结果）**

基于步骤 2 中识别出的路径，AAG 自动生成了 Python 数据处理代码，对该闭环路径中涉及的所有交易金额进行了提取和汇总。各步交易金额如下：

- Anna Lee → Gill Zachary，交易金额为 879.94 $
- Gill Zachary → Garcia Marcus，交易金额为 210.43 $
- Garcia Marcus → Nunez Mitchell，交易金额为 472.69 $
- Nunez Mitchell → Robinson David，交易金额为 606.94 $
- Robinson David → Anna Lee，交易金额为 825.22 $

该路径的总交易金额为 **2995.22 $**。

## 4. **最大交易账户识别（Python 代码统计结果）**

基于步骤 2 中识别出的路径，AAG 自动生成了 Python 数据处理代码，对路径中所有相关账户的历史交易总额进行了统计和排序。结果如下：

- Anna Lee 的总交易额为 161272.07 $
- Nunez Mitchell 的总交易额为 122984.06 $
- Garcia Marcus 的总交易额为 115367.59 $
- Robinson David 的总交易额为 96550.65 $
- Gill Zachary 的总交易额为 64109.86 $

结果表明，Anna Lee 和 Nunez Mitchell 的历史交易总额明显更高，说明它们可能在资金流动中扮演资金归集点的角色，需要进一步核查其业务活动和资金流向。

## **Summary**

本次分析利用图算法对 Anna Lee 的洗钱风险进行了综合评估，主要使用了 **PageRank 中心性算法** 和 **DFS（深度优先搜索）路径算法**。首先，PageRank 对网络中所有账户的重要性和风险关联性进行了量化，结果显示 **Anna Lee 的 PageRank 得分位于前 2%**，说明其处于高度关键的节点位置，具有较强的风险扩散潜力。随后，DFS 对以 Anna Lee 为起点的资金流进行了穿透分析，识别出一条闭环可疑路径：**Anna Lee → Gill Zachary → Garcia Marcus → Nunez Mitchell → Robinson David → Anna Lee**，总交易金额为 **2995.22 $**，表现出典型的“资金转出—多步中转—回流原账户”的高风险结构。此外，金额统计结果显示 Anna Lee（161272.07 $）和 Nunez Mitchell（122984.06 $）的历史交易总额显著偏高，进一步表明它们可能是资金归集或中转的关键节点，整体可疑程度较高。

## **Recommendations**

1. 建议将 Anna Lee 及闭环路径中的相关账户（尤其是 Nunez Mitchell 和 Robinson David）纳入高风险名单，并提升监控等级。对其后续的大额、循环型交易设置严格的实时预警和限额控制。
2. 建议对 Anna Lee 及关键交易对手开展专项尽职调查，重点核查其资金来源、交易目的和业务背景。同时结合更长周期的交易记录与外部信息，排查是否存在结构化拆分、循环转账等洗钱特征。

----------------------------"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
        if not response_text:
            return "Unable to generate answer from the algorithm result."
        return response_text
    
    def chat(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        response_text = response.choices[0].message.content
        if not response_text:
            return "Unable to generate answer."
        return response_text

    def analyze_dependency_type_and_locate_dependency_data(self, current_question:str, task_type:str, current_algo_desc:str, parent_question: str,  parent_outputs_meta:list)-> dict:
        full_prompt = analyze_dependency_type_and_locate_dependency_data_prompt.format(
            current_question=current_question,
            task_type=task_type,
            current_algo_desc=current_algo_desc,
            parent_question=parent_question,
            parent_outputs_meta=parent_outputs_meta
        )
        return self.execute_prompt(full_prompt, parse_json=True)


    def map_parameters(self, current_question: str, current_algo_desc: str, dependency_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_prompt = map_parameters_prompt.format(
            current_question=current_question,
            algo_desc=current_algo_desc,
            dependency_items=json.dumps(dependency_items, ensure_ascii=False, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)

    def generate_graph_conversion_code(self, current_question: str, dependency_items: List[Dict[str, Any]])-> Dict[str, Any]:
        full_prompt = generate_graph_conversion_code_prompt.format(
            current_question=current_question,
            dependency_items=json.dumps(dependency_items, ensure_ascii=False, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)
    
    def generate_numeric_analysis_code(
        self, 
        question: str, 
        dependency_items: List[Dict[str, Any]], 
        vertex_schema: Dict[str, str], 
        edge_schema: Dict[str, str]
    ) -> Dict[str, Any]:
        full_prompt = generate_numeric_analysis_code_prompt.format(
            question=question,
            dependency_data_items=json.dumps(dependency_items, ensure_ascii=False, indent=2),
            vertex_schema=json.dumps(vertex_schema, ensure_ascii=False, indent=2),
            edge_schema=json.dumps(edge_schema, ensure_ascii=False, indent=2)
        )
        return self.execute_prompt(full_prompt, parse_json=True)
    
    # add gjq
    def nl_query_classify_type(self, question: str, query_templates: dict) -> str:
        """Classify the query type for natural language query engine."""
        full_prompt = nl_query_classify_type_prompt.format(
            question=question,
            query_templates=json.dumps(query_templates, ensure_ascii=False, indent=2)
        )
        response_text = self.execute_prompt(full_prompt, parse_json=False)
        return response_text.strip().strip('"').strip("'")
    
    # add gjq
    def nl_query_extract_params(self, question: str, query_type: str, template: dict,
                                schema_info: str, query_modifiers: dict) -> dict:
        """Extract parameters for natural language query engine."""
        full_prompt = nl_query_extract_params_prompt.format(
            schema_info=schema_info,
            query_type=query_type,
            template_description=template['description'],
            template_method=template['method'],
            required_params=template['required_params'],
            optional_params=template.get('optional_params', []),
            query_modifiers=json.dumps(query_modifiers, ensure_ascii=False, indent=2),
            question=question
        )
        return self.execute_prompt(full_prompt, parse_json=True)


class Reasoner:

    def __init__(self, config: ReasonerConfig):
        """Initialize Reasoner by provider selection.

        Args:
            config: ReasonerConfig with llm.provider in {"ollama","openai"} and provider-specific fields.
            fallback_embed_model: used only for OllamaEnv when no embedding is provided from outside.
        """
        if config is None or config.llm is None:
            raise ValueError("Reasoner requires a valid ReasonerConfig with llm settings")

        self.config = config
        provider = (config.llm.provider or "ollama").lower()

        if provider == "ollama":
            ollama_cfg = config.llm.ollama or {}
            self.env = OllamaEnv(
                llm_mode_name=ollama_cfg.get("model_name", "llama3.1:70b"),
                llm_embed_name="BAAI/bge-large-en-v1.5",
                chunk_size=ollama_cfg.get("chunk_size", 512),
                chunk_overlap=ollama_cfg.get("chunk_overlap", 20),
                embed_batch_size=ollama_cfg.get("embed_batch_size", 20),
                device=ollama_cfg.get("device", "cuda:0"),
                timeout=ollama_cfg.get("timeout", 150000),
                port=ollama_cfg.get("port", 11434),
                verbose=ollama_cfg.get("verbose", False),
            )
        elif provider == "openai":
            openai_cfg = config.llm.openai or {}
            base_url = openai_cfg.get("base_url") or "https://api.openai.com/v1"
            api_key = openai_cfg.get("api_key")
            model = openai_cfg.get("model") or "gpt-4o"
            if not api_key:
                # Allow environment variable fallback
                api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI provider requires an API key via config.reasoner.llm.openai.api_key or env OPENAI_API_KEY")
            self.env = OpenAIEnv(base_url=base_url, api_key=api_key, model_name=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


    def plan_subqueries(self, decompose: bool, query: str) -> dict:
        if decompose == False: 
            # Do not decompose, treat as a single question
            return{"subqueries": [{
                        "id": "q1",
                        "query": query,
                        "depends_on": []
                    }]}  
        full_prompt = plan_subqueries_prompt_zh.format(query=query)
        return self.env.execute_prompt(full_prompt, parse_json=True)

    def plan_expert_subqueries_with_algorithms(
        self,
        expert_instruction: str,
        algorithm_library_info: str = "",
        dataset_info: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        将专家自然语言指令转为带 algorithm 字段的 subqueries。
        """
        full_prompt = expert_subqueries_with_algorithms_prompt_zh.format(
            expert_instruction=expert_instruction,
            algorithm_library_info=algorithm_library_info or "Algorithm library not available",
            dataset_info=dataset_info or "N/A"
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)
    
    def revise_subquery_plan(self, current_plan: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        full_prompt = revise_subquery_plan_prompt.format(
            current_plan=json.dumps(current_plan, ensure_ascii=False),
            user_request=user_request
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)
    
    def refine_subqueries(self, current_dag: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据找到的算法信息优化DAG,确保子查询严格按照任务类型边界划分
        
        Args:
            current_dag: 当前的DAG结构,包含subqueries列表
            
        Returns:
            优化后的DAG结构
        """
        full_prompt = refine_subqueries_prompt_en.format(
            current_dag=json.dumps(current_dag, ensure_ascii=False, indent=2)
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)
    
    def select_task_type(self, question: str, task_type_list: list) -> dict:
        full_prompt = select_task_type_prompt.format(
            question=question,
            task_type_list=task_type_list
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)
    
    def select_algorithm(self, question: str, algorithm_list: list, graph_schema: Optional[Dict[str, Any]] = None) -> dict:
        return self.env.select_algorithm(question, algorithm_list, graph_schema)
    
    def extract_parameters_with_postprocess(self, question: str, tool_description: str) -> dict:
        full_prompt = extract_parameters_with_postprocess_promt.format(
            question=question,
            tool_description=tool_description
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)
    
    def check_data_dependency(self,  q1_question: str, q1_algorithm: str, q2_question: str, q2_algorithm: str) -> bool:
        try:
            full_prompt = check_data_dependency_prompt.format(
                q1_question=q1_question,
                q1_algorithm=q1_algorithm,
                q2_question=q2_question,
                q2_algorithm=q2_algorithm
            )
            result = self.env.execute_prompt(full_prompt, parse_json=True)
            depends = result.get("q2_depends_on_q1")
            if isinstance(depends, bool):
                return depends
            if isinstance(depends, str):
                normalized = depends.strip().lower()
                if normalized in {"true", "yes"}:
                    return True
                if normalized in {"false", "no"}:
                    return False
        except Exception as e:
            print(f"Error determining data dependency: {e}")
        return False

    def generate_answer_from_algorithm_result(self, question: str, tool_description: str, tool_result: Dict[str, Any]):
        return self.env.generate_answer_from_algorithm_result(question, tool_description, tool_result)

    def analyze_dependency_type_and_locate_dependency_data(self, current_question:str, task_type:str, current_algo_desc:str, parent_question: str,  parent_outputs_meta:list) -> dict:
        full_prompt = analyze_dependency_type_and_locate_dependency_data_prompt.format(
            current_question=current_question,
            task_type=task_type,
            current_algo_desc=current_algo_desc,
            parent_question=parent_question,
            parent_outputs_meta=parent_outputs_meta
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)

    def classify_question_type(self, question: str) -> Dict[str, Any]:
        full_prompt = classify_question_type_prompt.format(question=question)
        return self.env.execute_prompt(full_prompt, parse_json=True)

    def map_parameters(self, current_question: str, current_algo_desc: str, dependency_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_prompt = map_parameters_prompt.format(
            current_question=current_question,
            algo_desc=current_algo_desc,
            dependency_items=json.dumps(dependency_items, ensure_ascii=False, indent=2)
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)

    def generate_graph_conversion_code(self, current_question: str, dependency_items: List[Dict[str, Any]])-> Dict[str, Any]:
        full_prompt = generate_graph_conversion_code_prompt.format(
            current_question=current_question,
            dependency_items=json.dumps(dependency_items, ensure_ascii=False, indent=2)
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)

    def generate_numeric_analysis_code(
        self, 
        question: str, 
        dependency_items: List[Dict[str, Any]], 
        vertex_schema: Dict[str, str], 
        edge_schema: Dict[str, str]
    ) -> Dict[str, Any]:
        full_prompt = generate_numeric_analysis_code_prompt.format(
            question=question,
            dependency_data_items=json.dumps(dependency_items, ensure_ascii=False, indent=2),
            vertex_schema=json.dumps(vertex_schema, ensure_ascii=False, indent=2),
            edge_schema=json.dumps(edge_schema, ensure_ascii=False, indent=2)
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)

    def get_question_entity(self, question, language="en"):
        if hasattr(self.env, "get_question_entity"):
            return self.env.get_question_entity(question, language)
        return []

    def get_quetion_response(self, question, graph_result, language="en"):
        if hasattr(self.env, "get_quetion_response"):
            return self.env.get_quetion_response(question, graph_result, language)
        return None

    def generate_response(self, prompt: str):
        if hasattr(self.env, "generate_response"):
            return self.env.generate_response(prompt)
        raise NotImplementedError("Underlying environment does not support generate_response/complete")

    def extract_parameters_with_postprocess_new(
        self, 
        question: str, 
        tool_description: str, 
        vertex_schema: Dict[str, str], 
        edge_schema: Dict[str, str],
        error_history: Optional[List[Dict[str, Any]]] = None, 
        trace: Optional[Dict[str, Any]] = None, # 用于记录step_id等
    ) -> dict:
        full_prompt = extract_parameters_with_postprocess_promt_new.format(
            question=question,
            tool_description=tool_description,
            vertex_schema=json.dumps(vertex_schema, ensure_ascii=False, indent=2),
            edge_schema=json.dumps(edge_schema, ensure_ascii=False, indent=2)
        )

        if error_history:
            full_prompt = enhance_prompt(
                base_prompt=full_prompt,
                error_history=error_history
            )

        return self.env.execute_prompt(full_prompt, parse_json=True)

    def merge_parameters_from_dependencies(
        self, 
        question: str, 
        tool_description: str, 
        vertex_schema: Dict[str, str], 
        edge_schema: Dict[str, str],
        dependency_parameters: Dict[str, Any],
        error_history: Optional[List[Dict[str, Any]]] = None,
        trace: Optional[Dict[str, Any]] = None, # 用于记录step_id等
    ) -> dict:
        """Merge dependency parameters with extracted parameters and generate post-processing code."""
        full_prompt = merge_parameters_with_dependencies_prompt.format(
            question=question,
            tool_description=tool_description,
            dependency_parameters=json.dumps(dependency_parameters, indent=2),
            vertex_schema=json.dumps(vertex_schema, indent=2),
            edge_schema=json.dumps(edge_schema, indent=2)
        )

        if error_history:
            full_prompt = enhance_prompt(
                base_prompt=full_prompt,
                error_history=error_history
            )


        return self.env.execute_prompt(full_prompt, parse_json=True)

    def chat(self, messages: list):
        if hasattr(self.env, "chat"):
            return self.env.chat(messages)
        raise NotImplementedError("Underlying environment does not support chat")

    def general_query_response(self, query):
        messages = [
            {"role": "system", "content": general_query_prompt},
            {
                "role": "user",
                "content": query
            },
        ]
        return self.chat(messages)
    
    def nl_query_classify_type(self, question: str, query_templates: dict) -> str:
        """Classify the query type for natural language query engine."""
        full_prompt = nl_query_classify_type_prompt.format(
            question=question,
            query_templates=json.dumps(query_templates, ensure_ascii=False, indent=2)
        )
        response_text = self.env.execute_prompt(full_prompt, parse_json=False)
        return response_text.strip().strip('"').strip("'")
    
    def nl_query_extract_params(self, question: str, query_type: str, template: dict,
                                schema_info: str, query_modifiers: dict) -> dict:
        """Extract parameters for natural language query engine."""
        full_prompt = nl_query_extract_params_prompt.format(
            schema_info=schema_info,
            query_type=query_type,
            template_description=template['description'],
            template_method=template['method'],
            required_params=template['required_params'],
            optional_params=template.get('optional_params', []),
            query_modifiers=json.dumps(query_modifiers, ensure_ascii=False, indent=2),
            question=question
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)

    def rewrite_query(
        self,
        original_query: str,
        algorithm_library_info: str,
        dataset_info: Optional[str] = None,
        use_chinese: bool = True
    ) -> Dict[str, Any]:
        """
        Rewrite a vague user query into a concrete, executable query.
        
        Args:
            original_query: The user's original vague question
            algorithm_library_info: Information about available task types and algorithms
            dataset_info: Optional information about the current graph dataset
            use_chinese: Whether to use Chinese prompt (default: True)
            
        Returns:
            Dict containing:
                - rewritten_query: The concrete, executable query
                - reasoning: Explanation of changes made
                - mapped_concepts: List of concept mappings
        """
        dataset_context = dataset_info if dataset_info else ("暂无数据集信息" if use_chinese else "No dataset information available")
        
        # Choose prompt based on language preference
        if use_chinese:
            from aag.reasoner.prompt_template.llm_prompt_zh import rewrite_query_prompt_zh
            prompt_template = rewrite_query_prompt_zh
        else:
            prompt_template = rewrite_query_prompt
        
        full_prompt = prompt_template.format(
            original_query=original_query,
            algorithm_library_info=algorithm_library_info,
            dataset_info=dataset_context
        )
        return self.env.execute_prompt(full_prompt, parse_json=True)



