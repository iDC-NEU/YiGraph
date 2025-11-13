import numpy as np
import os
import openai
import json
import re
import requests # deepseek
from typing import Dict, Literal, Any, List
from llama_index.core import Settings
from llama_index.core.utils import print_text
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from aag.config.engine_config import ReasonerConfig
from aag.reasoner.prompt_template.llm_prompt import prompt_select_graph_algorithm_str_en, prompt_select_graph_algorithm_str_zh

from aag.reasoner.prompt_template.prompt import (
    command_keyword_extract_prompt_template,
    command_synonym_expand_prompt_template,
    gemma_keyword_extract_prompt_template,
    gemma_synonym_expand_prompt_template,
)

from aag.utils.parse_json import extract_json_from_response


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(),
                      re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()

def get_pasrse_output(output_str, field=Literal["algorithm"]):
    retry = 3
    while retry > 0:
        try:
            output_data = json.loads(extract_json_str(output_str))
            assert field in output_data
            if field == "algorithm":
                algorithm = output_data[field]
                return algorithm
            else:
                raise ValueError(f"Field {field} is not supported")
        except json.JSONDecodeError as e:
            retry -= 1
            if retry == 0:
                return {"error": "JSONDecodeError", "message": str(e), "output": output_str}

EMBEDD_DIMS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    # "text-embedding-ada-002": 1536,
}


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

        # base_url = "http://localhost:11434"
        base_url = f"http://localhost:{port}"
        Settings.llm = Ollama(model=llm_mode_name, request_timeout=timeout,
                              temperature=0.0, base_url=base_url)  # , device=device
        self.verbose = verbose

        if 'llama' in llm_mode_name:
            self.keyword_extract_prompt_template = command_keyword_extract_prompt_template
            self.synonym_expand_prompt_template = command_synonym_expand_prompt_template
        elif 'gemma' in llm_mode_name and 'instruct' in llm_mode_name:
            self.keyword_extract_prompt_template = gemma_keyword_extract_prompt_template
            self.synonym_expand_prompt_template = gemma_synonym_expand_prompt_template
        elif 'command' in llm_mode_name:
            self.keyword_extract_prompt_template = command_keyword_extract_prompt_template
            self.synonym_expand_prompt_template = command_synonym_expand_prompt_template
        else:
            self.keyword_extract_prompt_template = None
            self.synonym_expand_prompt_template = None

        if 'BAAI' in llm_embed_name:
            # print(f"use huggingface embedding {llm_embed_name}")
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
        # print(f"using {self.dim}")
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        print(
            f"llm_mode_name: {llm_mode_name}, llm_embed_name: {llm_embed_name}, chunk_size: {Settings.chunk_size}, chunk_overlap: {chunk_overlap}")

    def complete(self, prompt, info=""):
        response = self.llm.complete(prompt)
        print_text(f'{prompt}\n', color='yellow')
        print_text(f'{response.text}\n', color='green')

        generate_time, load_time, prefill_time, decode_time, prompt_len, generate_len = self.parse_response_time(
            response)

        # if self.logger is not None:
        #     self.logger.add(f'{info}generate_time', generate_time)
        #     self.logger.add(f'{info}load_time', load_time)
        #     self.logger.add(f'{info}prefill_time', prefill_time)
        #     self.logger.add(f'{info}decode_time', decode_time)
        #     self.logger.add(f'{info}prompt_len', prompt_len)
        #     self.logger.add(f'{info}generate_len', generate_len)
        if self.verbose:
            print_text(
                f"generate_time {generate_time:.3f}s, load_time {load_time:.3f}s, prefill_time {prefill_time:.3f}s, decode_time {decode_time:.3f}s, prompt_len {prompt_len}, generate_len {generate_len}\n",
                color='red')
            
        return response.text

    def parse_response_time(self, response, verbose=False):
        # total_duration: time spent generating the response
        total_time = response.raw['total_duration'] / 1e9

        # load_duration: time spent in nanoseconds loading the model
        load_time = response.raw['load_duration'] / 1e9

        # (prefill): prompt_eval_duration: time spent in nanoseconds evaluating the prompt
        prefill_time = response.raw['prompt_eval_duration'] / 1e9

        # (generation): eval_duration: time in nanoseconds spent generating the response
        decode_time = response.raw['eval_duration'] / 1e9

        # prompt_eval_count: number of tokens in the prompt
        prompt_len = response.raw[
            'prompt_eval_count'] if 'prompt_eval_count' in response.raw else 0

        # eval_count: number of tokens in the response
        generate_len = response.raw['eval_count']

        # print(f'total_duration: {total_duration:.3f}ms, load_duration: {load_duration:.3f}ms, prompt_eval_duration: {prompt_eval_duration:.3f}ms, eval_duration: {eval_duration:.3f}ms')
        if verbose:
            print_text(
                f'total_duration: {total_time:.3f}ms, load, prompt, eval: ({load_time:.3f}, {prefill_time:.3f}, {decode_time:.3f})ms\n',
                color='red')
            print_text(
                f'prompt_eval_count: {prompt_len} tokens, eval_count: {generate_len} tokens\n',
                color='red')

            # print(total_duration, prompt_eval_duration + eval_duration + load_time)

        return (total_time, load_time, prefill_time, decode_time, prompt_len,
                generate_len)

    def generate_response(self, query: str):
        response = self.llm.complete(query)
        return response
    
    def get_graph_algorithm(self, question, context, language="en"):
        try:
            prompt_mapping = {
                "en": prompt_select_graph_algorithm_str_en,
                "zh": prompt_select_graph_algorithm_str_zh,
            }
            prompt_extract_triplest = prompt_mapping.get(language.lower())
            if not prompt_extract_triplest:
                raise NotImplementedError(
                    f"Language '{language}' is not supported.")
            full_prompt = prompt_extract_triplest.format(
                input_question=question,
                context=context
            )
            response = self.complete(full_prompt, info="graph_algorithm_selection")
            algorithm = get_pasrse_output(response.strip(), field="algorithm")
            return algorithm     
        except Exception as e:
            print(f"Error in Ollama graph algorithm selection: {e}")
            return None
        
    def check_data_dependency(
            self,
            q1_question: str,
            q1_algorithm: str,
            q2_question: str,
            q2_algorithm: str) -> bool:
        """Determine whether Q2 depends on the result of Q1 using the LLM."""
        prompt = f"""
You are a graph analytics planning assistant. Decide whether a second sub-question (Q2) requires running after a first sub-question (Q1) because Q2 needs Q1's computed result.

Dependency definition:
- Return true only when information produced by solving Q1 is a required input for solving Q2.
- If Q2 can be solved directly from the original data or context without first executing Q1, return false.

###Example 1
Original query: "Recently I discovered that Anna's transaction behavior is anomalous and she might be a potential fraud user. I want to find the potential fraud community around her, suggest possible suspicious transaction paths, and determine how much cash has likely been illegally transferred out."
Q1 question: "Is Anna a fraud user based on her anomalous transaction behavior?"
Q1 algorithm: "detect_fraud_user"
Q2 question: "Find the potential fraud community centered around Anna."
Q2 algorithm: "detect_fraud_community"
Expected dependency: false
Reason: Community detection around Anna can use the transaction graph directly without first proving she is fraudulent.

###Example 2
Original query (same as above).
Q1 question: "Find the potential fraud community centered around Anna."
Q1 algorithm: "detect_fraud_community"
Q2 question: "What are the possible suspicious transaction paths associated with Anna?"
Q2 algorithm: "discover_suspicious_paths"
Expected dependency: true
Reason: Identifying suspicious paths should leverage the community detected in Q1 to limit the search space.

###Input To Evaluate
Q1 question: {q1_question}
Q1 algorithm: {q1_algorithm}
Q2 question: {q2_question}
Q2 algorithm: {q2_algorithm}

###Output Format
Respond with JSON only:
{{
  "q2_depends_on_q1": true or false,
  "reason": "Brief justification."
}}
"""
        try:
            response = self.llm.complete(prompt)
            result = extract_json_from_response(response.text)
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
        
    def get_question_entity(self, question, language="en"):  #TODO: 需要补充一个函数，从问题中确定需要查询的实体, 如果查询实体为空，则返回none（表示要查询全图）
        """从问题中确定需要查询的实体, 如果查询实体为空，则返回none（表示要查询全图）"""
        """返回类型是list，表示需要查询的实体"""
        return []

    def get_quetion_response(self, question, graph_result, language="en"):  #TODO: 需要补充一个函数
        """根据问题和图算法结果，生成响应"""
        pass
    
    def plan_subqueries(self, decompose: bool, query: str) -> dict:
        if decompose == False: 
            # Do not decompose, treat as a single question
            return{"subqueries": [{
                        "id": "q1",
                        "query": query,
                        "depends_on": []
                    }]}  
        prompt = f"""**Role**:You are an AI assistant specialized in decomposing complex queries. Your task is to break down a complex question into multiple sub-queries that have logical dependencies.
        **Available Resources**:
        1. **Complete Algorithm Library**: Supports all graph algorithms in NetworkX, including but not limited to:
        - Traversal Algorithms: BFS, DFS
        - Shortest Path: Dijkstra, A*, Bellman-Ford
        - Community Detection: Louvain, Leiden, Girvan-Newman
        - Centrality Metrics: Degree Centrality, Betweenness Centrality, Proximity Centrality, Eigenvector Centrality, PageRank
        - Matching Algorithms: Maximum Matching, Minimum Weight Matching
        - Connectivity: Strongly Connected Components, Weakly Connected Components
        - And all other NetworkX algorithms
        2. **Powerful Post-Processing Capabilities**: Can perform the following operations on the results of any graph algorithm:
        - Sorting, Filtering, Intersection/Union
        - Mathematical Operations: Weighted Summation, Normalization, Standardization
        - Statistical Analysis: Maximum, Minimum, Average, Percentiles
        - Logical Operations: Conditional Filtering, Multiple Result Fusion
        **Decomposition Principles**:
        1. **Single Algorithm Principle**: Each subproblem uses only one core graph algorithm.
        2. **Pipeline Thinking**: The results of preceding algorithms, after post-processing, can be used as input for subsequent algorithms.
        Each sub-query must have a unique ID (e.g., "q1", "q2"), the query text itself, and a depends_on list specifying which other sub-query IDs must be resolved before this one can be answered.
        Infer dependencies based on logical necessity. If answering a sub-query requires the answer from another sub-query, specify that ID in the depends_on field. Dependencies should be based on prerequisites and the flow of information.
        Example 1 for Guidance:
        Input Query:
        "Recently I discovered that Anna's transaction behavior is anomalous and she might be a potential fraud user. I want to find the potential fraud community around her, suggest possible suspicious transaction paths, and determine how much cash has likely been illegally transferred out."
        Output:{{
        "subqueries": [
            {{
            "id": "q1",
            "query": "Is Anna a fraud user based on her anomalous transaction behavior?",
            "depends_on": []
            }},
            {{
            "id": "q2",
            "query": "Find the potential fraud community centered around Anna.",
            "depends_on": ["q1"]
            }},
            {{
            "id": "q3",
            "query": "What are the possible suspicious transaction paths associated with Anna?",
            "depends_on": ["q2"]
            }},
            {{
            "id": "q4",
            "query": "Determine how much cash has likely been illegally transferred out.",
            "depends_on": ["q2", "q3"]    
            }}
        ]
        }} 
        Example 2 for Guidance:
        Input Query:
        "Please help me identify the ten most influential people in the picture, calculate their sum, and find out who has the least influence."
        Output:{{
        "subqueries": [
            {{
                "id": "q1",
                "query": "Identify the ten most influential people in the picture, calculate their sum, and find out who has the least influence.",
                "depends_on": []
            }}
        ]
        }}
        Example 3 for Guidance:
        Input Query:
        "In a wartime supply network, cities are linked by roads of varying maintenance costs, and command needs the most reliable evacuation blueprint. First locate the city whose influence score is the lowest. Then enumerate every minimum spanning tree of the road network graph and, using that least influential city as the root, add up the distances from it to every other city inside each tree. Return the minimum total distance observed among all the spanning trees."
        Output{{
        "subqueries": [
            {{
                "id": "q1",
                "query": "Identify the city with the lowest influence score in the network.",
                "depends_on": []
            }},
            {{
                "id": "q2",
                "query": "Enumerate all minimum spanning trees of the road network graph.",
                "depends_on": []
            }},
            {{
                "id": "q3",
                "query": "For each minimum spanning tree, sum the distances from the least influential city to all other cities and keep the smallest total.",
                "depends_on": ["q1", "q2"]
            }}
        ]
        }}
        Now, based on the instructions and example above, decompose the new complex query provided by the user. Your output must be the valid JSON object only.The query is : {query}"""
        response = self.llm.complete(prompt)
        return extract_json_from_response(response.text)


    def select_task_type(self, question: str, task_type_list: list) -> dict:
        prompt = f""" You are a graph algorithm expert skilled at identifying the most appropriate graph task type based on a natural language question and a provided task type list.You will be given an input question and a task type list.
        Each task type in the list includes the following fields:
        - id: a unique identifier for the task type
        - task_type: the name of the task type
        - description: a detailed explanation of what the task type does
        Each task type represents a category of graph algorithms that can solve certain types of problems.Your goal is to infer the most suitable task type to solve the given input question, based on the semantic meaning of the question and the descriptions in the task type list.
        Example for Guidance::
        Input Query : "Which nodes in the graph are the most influential?"
        Input Task Type List:[
            {{
                "id": "traversal",
                "task_type": "Traversal",
                "description": "Visit all nodes or edges in a graph sequentially, usually for exploring graph structures or serving as the foundation for other algorithms."
            }},
            {{
                "id": "centrality_importance",
                "task_type": "Centrality and Importance Measures",
                "description": "Evaluate the importance or influence of nodes in a graph, used for ranking, identifying key nodes, and network analysis."
            }}
        ]
        Output:
        {{
            "id": "centrality_importance",
        }}
        Now, based on the instructions and example above, determine the most appropriate task type.Your output must be a valid JSON object only — no additional text or explanation.The query is: {question}. The task type list is: {task_type_list}
        """
        response = self.llm.complete(prompt)
        return extract_json_from_response(response.text)


    def select_algorithm(self, question: str, algorithm_list: list) -> dict:
        prompt = f"""
        You are a graph algorithm expert skilled at identifying the most appropriate graph algorithm based on a natural language question and a provided algorithm list.You will be given an input question and an algorithm list.
        Each algorithm in the list includes the following fields:
        - id: a unique identifier for the algorithm
        - description_principle: the theoretical or operational principle of the algorithm
        - description_meaning: the semantic purpose or interpretation of what the algorithm achieves

        Each algorithm represents a concrete computational method that can solve certain types of graph problems.Your goal is to infer the **most suitable algorithm** to solve the given input question, based on the semantic meaning of the question and the algorithm descriptions.

        Example for Guidance:
        Input Query:
        "Which nodes in the graph are the most influential?"

        Input Algorithm List:[
            {{
                "id": "pagerank",
                "description_principle": "PageRank is based on the random surfer model. A random walker traverses the graph by following outgoing edges with probability alpha, or jumps to a random node with probability (1 - alpha). The score of a node is accumulated through its incoming edges: each source node distributes its PageRank score equally across its outgoing edges, and the target node sums up these contributions.",
                "description_meaning": "PageRank measures the relative importance or influence of nodes in a network. Higher scores indicate nodes that are more likely to be visited during random walks, making it useful for ranking, identifying key nodes, and analyzing information flow."
            }},
            {{
                "id": "degree_centrality",
                "description_principle": "Degree centrality measures the importance of a node based on its degree, i.e., the number of connections it has. The value is normalized by dividing the degree of the node by the maximum possible degree (n-1), where n is the number of nodes in the graph.",
                "description_meaning": "Degree centrality reflects the direct influence of a node within the network. Nodes with higher degree centrality are more connected, indicating stronger ability to spread information or interact with other nodes."
            }},
            {{
                "id": "betweenness_centrality",
                "description_principle": "Betweenness centrality is based on shortest paths. For a given node v, it is calculated as the fraction of all-pairs shortest paths in the graph that pass through v. Nodes that frequently occur on many shortest paths between other nodes will have high betweenness scores.",
                "description_meaning": "Betweenness centrality measures a node’s role as a bridge or intermediary in the network. High scores indicate nodes that control information flow, making them critical for communication, influence, or vulnerability in the graph."
            }}
        ]
        Output:
        {{
            "id": "pagerank"
        }}
        Now, based on the instructions and example above, determine the most appropriate algorithm. Your output must be a valid JSON object only — no additional text or explanation. The query is: {question}. The algorithm list is: {algorithm_list}
        """
        response = self.llm.complete(prompt)
        return extract_json_from_response(response.text)
        
    
    def extract_parameters_with_postprocess(self, question: str, tool_description: str) -> dict:
        prompt = f"""
        You are an intelligent scheduling expert for graph computation tasks. Your responsibility is to analyze user questions, extract tool parameters, and generate post-processing code.
        Your task is to analyze a given natural-language *question* together with a *tool_description*, extract the parameters that match the specified tool, and produce the corresponding Python post-processing function.
        You must return a valid JSON object in the exact format shown below. Do NOT wrap the result in markdown code blocks or backticks.
        ##Output Format (Must Follow Exactly)
        ```json
        {{
            "tool_name": "run_pagerank",
            "parameters": {{
                "alpha": 0.85,
                "max_iter": 100
            }},
            "post_processing_code": "def process(data):\\n    return data",
            "reasoning": "Selected run_pagerank because..."
        }}
        ```
        ##Core Parsing Rules (Strictly Enforced)
        ###Rule 1: Fixed Tool
        -The given tool_description is fixed; you must not choose or invent any other tool.
        -The tool_name in your output must exactly match the one defined in the tool_description.

        ###Rule 2: Parameter Extraction
        -Extract only the parameters explicitly mentioned or implied in the user question.
        -Parameter names must align with the tool’s input_schema.
        -Never include 'G' or 'backend_kwargs', even if G appears in the required list.
        
        ###Rule 3: Post-Processing Code Generation
        -Generate appropriate Python post-processing code based on user intent (e.g., “Top 5”, “greater than 0.5”, “how many”, etc.).
        -The function must strictly follow this format:
        ```python
        def process(data):
            # The structure of 'data' strictly follows output_schema['result']
            # Perform filtering, sorting, or aggregation operations as needed
            return processed_result 
        ```
        #### Sorting Rules
        -“Top N”, “highest”, “largest”, “maximum” → reverse=True (descending)
        -“Bottom N”, “lowest”, “smallest”, “minimum” → reverse=False (ascending)
        ####Example 1: Sorting and Selecting
        User question: “How many connected components are there?”
        ```python
        def process(data):
            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_items[:5])
        ```
        ###Example 2: Counting
        User question: “How many connected components are there?”
        ```python
        def process(data):
            return {{'component_count': len(data)}}
        ```

        ###Rule 4: If the user does not specify any special post-processing requirements, simply return the data as-is:
        ```python
        def process(data):
            return data
        ``` 

        ##Inputs:
        ###User question:
        {question}
        ###Tool Description:
        {tool_description}
        
        ##Instruction:
        Follow all rules and examples above strictly. Your output must be a valid JSON object only — no additional text, markdown, or explanation.
        """

        response = self.llm.complete(prompt)
        return extract_json_from_response(response.text)
        

    def generate_answer_from_algorithm_result(self, question: str, tool_description: str, tool_result: Dict[str, Any]) -> str:
        prompt = f"""
        You are a professional data analyst responsible for interpreting graph algorithm results.
        Your task is to analyze the computation output of a given tool based on:
        - a natural-language user question (*question*),
        - the corresponding tool description (*tool_description*), and
        - the actual execution result of that tool (*tool_result*).

        Each tool represents a specific graph algorithm (e.g., PageRank, Louvain, Shortest Path, etc.).
        You should generate a concise, human-understandable analytical explanation that directly answers the user question.

        ----------------------------
        ## Response Requirements
        - Base your explanation strictly on the provided computation result.
        - Clearly highlight and explain key data points (e.g., top-ranked nodes, cluster counts, path lengths).
        - Use simple, intuitive language — avoid jargon and avoid repeating the same summary.
        - If the available data is insufficient to fully answer the question, explicitly state the limitation or missing information.

        ----------------------------
        ## Inputs:
        ### User Question:
        {question}
        ### Tool Description:
        {tool_description}
        ### Tool Execution Result:
        {tool_result}

        ----------------------------
        ## Output
        Provide a clear and concise written explanation in plain English that directly addresses the user’s question.
        Do NOT return JSON or code — only natural language.
        """
        response = self.llm.complete(prompt)
        response_text = str(response.text).strip()
        if not response_text:
            return "Unable to generate answer from the algorithm result."
        return response_text


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

    def get_graph_algorithm(self, question, context, language="en"):
        try:
            prompt_mapping = {
                "en": prompt_select_graph_algorithm_str_en,
                "zh": prompt_select_graph_algorithm_str_zh,
            }
            prompt_extract_triplest = prompt_mapping.get(language.lower())
            if not prompt_extract_triplest:
                raise NotImplementedError(
                    f"Language '{language}' is not supported.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_extract_triplest.format(
                    context=context)}]
            )
            response = response.choices[0].message.content
            return get_pasrse_output(response.strip(), field="algorithm")
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None
        
    def check_data_dependency(
            self,
            q1_question: str,
            q1_algorithm: str,
            q2_question: str,
            q2_algorithm: str) -> bool:
        """Use the llm to assess whether Q2 depends on Q1."""
        prompt = f"""
You are a graph analytics planning assistant. Decide whether a second sub-question (Q2) requires running after a first sub-question (Q1) because Q2 needs Q1's computed result.

Dependency definition:
- Return true only when information produced by solving Q1 is a required input for solving Q2.
- If Q2 can be solved directly from the original data or context without first executing Q1, return false.

###Example 1
Original query: "Recently I discovered that Anna's transaction behavior is anomalous and she might be a potential fraud user. I want to find the potential fraud community around her, suggest possible suspicious transaction paths, and determine how much cash has likely been illegally transferred out."
Q1 question: "Is Anna a fraud user based on her anomalous transaction behavior?"
Q1 algorithm: "detect_fraud_user"
Q2 question: "Find the potential fraud community centered around Anna."
Q2 algorithm: "detect_fraud_community"
Expected dependency: false
Reason: Community detection around Anna can use the transaction graph directly without first proving she is fraudulent.

###Example 2
Original query (same as above).
Q1 question: "Find the potential fraud community centered around Anna."
Q1 algorithm: "detect_fraud_community"
Q2 question: "What are the possible suspicious transaction paths associated with Anna?"
Q2 algorithm: "discover_suspicious_paths"
Expected dependency: true
Reason: Identifying suspicious paths should leverage the community detected in Q1 to limit the search space.

###Input To Evaluate
Q1 question: {q1_question}
Q1 algorithm: {q1_algorithm}
Q2 question: {q2_question}
Q2 algorithm: {q2_algorithm}

###Output Format
Respond with JSON only:
{{
  "q2_depends_on_q1": true or false,
  "reason": "Brief justification."
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
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

    def get_question_entity(self, question, context, language="en"):  #TODO: 需要补充一个函数，从问题中确定需要查询的实体, 如果查询实体为空，则返回none（表示要查询全图）
        """从问题中确定需要查询的实体, 如果查询实体为空，则返回none（表示要查询全图）"""
        """返回类型是list，表示需要查询的实体"""
        return []

    def get_quetion_response(self, question, graph_result, language="en"):  #TODO: 需要补充一个函数
        """根据问题和图算法结果，生成响应"""
        pass

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
        prompt = f"""**Role**:You are an AI assistant specialized in decomposing complex queries. Your task is to break down a complex question into multiple sub-queries that have logical dependencies.
        **Available Resources**:
        1. **Complete Algorithm Library**: Supports all graph algorithms in NetworkX, including but not limited to:
        - Traversal Algorithms: BFS, DFS
        - Shortest Path: Dijkstra, A*, Bellman-Ford
        - Community Detection: Louvain, Leiden, Girvan-Newman
        - Centrality Metrics: Degree Centrality, Betweenness Centrality, Proximity Centrality, Eigenvector Centrality, PageRank
        - Matching Algorithms: Maximum Matching, Minimum Weight Matching
        - Connectivity: Strongly Connected Components, Weakly Connected Components
        - And all other NetworkX algorithms
        2. **Powerful Post-Processing Capabilities**: Can perform the following operations on the results of any graph algorithm:
        - Sorting, Filtering, Intersection/Union
        - Mathematical Operations: Weighted Summation, Normalization, Standardization
        - Statistical Analysis: Maximum, Minimum, Average, Percentiles
        - Logical Operations: Conditional Filtering, Multiple Result Fusion
        **Decomposition Principles**:
        1. **Single Algorithm Principle**: Each subproblem uses only one core graph algorithm.
        2. **Pipeline Thinking**: The results of preceding algorithms, after post-processing, can be used as input for subsequent algorithms.
        Each sub-query must have a unique ID (e.g., "q1", "q2"), the query text itself, and a depends_on list specifying which other sub-query IDs must be resolved before this one can be answered.
        Infer dependencies based on logical necessity. If answering a sub-query requires the answer from another sub-query, specify that ID in the depends_on field. Dependencies should be based on prerequisites and the flow of information.
        Example 1 for Guidance:
        Input Query:
        "Recently I discovered that Anna's transaction behavior is anomalous and she might be a potential fraud user. I want to find the potential fraud community around her, suggest possible suspicious transaction paths, and determine how much cash has likely been illegally transferred out."
        Output:{{
        "subqueries": [
            {{
            "id": "q1",
            "query": "Is Anna a fraud user based on her anomalous transaction behavior?",
            "depends_on": []
            }},
            {{
            "id": "q2",
            "query": "Find the potential fraud community centered around Anna.",
            "depends_on": ["q1"]
            }},
            {{
            "id": "q3",
            "query": "What are the possible suspicious transaction paths associated with Anna?",
            "depends_on": ["q2"]
            }},
            {{
            "id": "q4",
            "query": "Determine how much cash has likely been illegally transferred out.",
            "depends_on": ["q2", "q3"]    
            }}
        ]
        }} 
        Example 2 for Guidance:
        Input Query:
        "Please help me identify the ten most influential people in the picture, calculate their sum, and find out who has the least influence."
        Output:{{
        "subqueries": [
            {{
                "id": "q1",
                "query": "Identify the ten most influential people in the picture, calculate their sum, and find out who has the least influence.",
                "depends_on": []
            }}
        ]
        }}
        Example 3 for Guidance:
        Input Query:
        "In a wartime supply network, cities are linked by roads of varying maintenance costs, and command needs the most reliable evacuation blueprint. First locate the city whose influence score is the lowest. Then enumerate every minimum spanning tree of the road network graph and, using that least influential city as the root, add up the distances from it to every other city inside each tree. Return the minimum total distance observed among all the spanning trees."
        Output{{
        "subqueries": [
            {{
                "id": "q1",
                "query": "Identify the city with the lowest influence score in the network.",
                "depends_on": []
            }},
            {{
                "id": "q2",
                "query": "Enumerate all minimum spanning trees of the road network graph.",
                "depends_on": []
            }},
            {{
                "id": "q3",
                "query": "For each minimum spanning tree, sum the distances from the least influential city to all other cities and keep the smallest total.",
                "depends_on": ["q1", "q2"]
            }}
        ]
        }}
        Now, based on the instructions and example above, decompose the new complex query provided by the user. Your output must be the valid JSON object only.The query is : {query}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        response = response.choices[0].message.content
        return json.loads(response)
    
    def select_task_type(self, question: str, task_type_list: list) -> dict:
        prompt = f""" You are a graph algorithm expert skilled at identifying the most appropriate graph task type based on a natural language question and a provided task type list.You will be given an input question and a task type list.
        Each task type in the list includes the following fields:
        - id: a unique identifier for the task type
        - task_type: the name of the task type
        - description: a detailed explanation of what the task type does
        Each task type represents a category of graph algorithms that can solve certain types of problems.Your goal is to infer the most suitable task type to solve the given input question, based on the semantic meaning of the question and the descriptions in the task type list.
        Example for Guidance:
        Input Query : "Which nodes in the graph are the most influential?"
        Input Task Type List:[
            {{
                "id": "traversal",
                "task_type": "Traversal",
                "description": "Visit all nodes or edges in a graph sequentially, usually for exploring graph structures or serving as the foundation for other algorithms."
            }},
            {{
                "id": "centrality_importance",
                "task_type": "Centrality and Importance Measures",
                "description": "Evaluate the importance or influence of nodes in a graph, used for ranking, identifying key nodes, and network analysis."
            }}
        ]
        Output:
        {{
            "id": "centrality_importance"
        }}
        Now, based on the instructions and example above, determine the most appropriate task type.Your output must be a valid JSON object only — no additional text or explanation.The query is: {question}. The task type list is: {task_type_list}
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        response = response.choices[0].message.content
        return json.loads(response)
    
    def select_algorithm(self, question: str, algorithm_list: list) -> dict:
        prompt = f"""
        You are a graph algorithm expert skilled at identifying the most appropriate graph algorithm based on a natural language question and a provided algorithm list.You will be given an input question and an algorithm list.
        Each algorithm in the list includes the following fields:
        - id: a unique identifier for the algorithm
        - description_principle: the theoretical or operational principle of the algorithm
        - description_meaning: the semantic purpose or interpretation of what the algorithm achieves

        Each algorithm represents a concrete computational method that can solve certain types of graph problems.Your goal is to infer the **most suitable algorithm** to solve the given input question, based on the semantic meaning of the question and the algorithm descriptions.

        Example for Guidance:
        Input Query:
        "Which nodes in the graph are the most influential?"

        Input Algorithm List:[
            {{
                "id": "pagerank",
                "description_principle": "PageRank is based on the random surfer model. A random walker traverses the graph by following outgoing edges with probability alpha, or jumps to a random node with probability (1 - alpha). The score of a node is accumulated through its incoming edges: each source node distributes its PageRank score equally across its outgoing edges, and the target node sums up these contributions.",
                "description_meaning": "PageRank measures the relative importance or influence of nodes in a network. Higher scores indicate nodes that are more likely to be visited during random walks, making it useful for ranking, identifying key nodes, and analyzing information flow."
            }},
            {{
                "id": "degree_centrality",
                "description_principle": "Degree centrality measures the importance of a node based on its degree, i.e., the number of connections it has. The value is normalized by dividing the degree of the node by the maximum possible degree (n-1), where n is the number of nodes in the graph.",
                "description_meaning": "Degree centrality reflects the direct influence of a node within the network. Nodes with higher degree centrality are more connected, indicating stronger ability to spread information or interact with other nodes."
            }},
            {{
                "id": "betweenness_centrality",
                "description_principle": "Betweenness centrality is based on shortest paths. For a given node v, it is calculated as the fraction of all-pairs shortest paths in the graph that pass through v. Nodes that frequently occur on many shortest paths between other nodes will have high betweenness scores.",
                "description_meaning": "Betweenness centrality measures a node’s role as a bridge or intermediary in the network. High scores indicate nodes that control information flow, making them critical for communication, influence, or vulnerability in the graph."
            }}
        ]
        Output:
        {{
            "id": "pagerank"
        }}
        Now, based on the instructions and example above, determine the most appropriate algorithm. Your output must be a valid JSON object only — no additional text or explanation. The query is: {question}. The algorithm list is: {algorithm_list}
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        response = response.choices[0].message.content
        return json.loads(response)
    
    
    def extract_parameters_with_postprocess(self, question: str, tool_description: str) -> dict:
        prompt = f"""
        You are an intelligent scheduling expert for graph computation tasks. Your responsibility is to analyze user questions, extract tool parameters, and generate post-processing code.
        Your task is to analyze a given natural-language *question* together with a *tool_description*, extract the parameters that match the specified tool, and produce the corresponding Python post-processing function.
        You must return a valid JSON object in the exact format shown below. Do NOT wrap the result in markdown code blocks or backticks.
        ##Output Format (Must Follow Exactly)
        ```json
        {{
            "tool_name": "run_pagerank",
            "parameters": {{
                "alpha": 0.85,
                "max_iter": 100
            }},
            "post_processing_code": "def process(data):\\n    return data",
            "reasoning": "Selected run_pagerank because..."
        }}
        ```
        ##Core Parsing Rules (Strictly Enforced)
        ###Rule 1: Fixed Tool
        -The given tool_description is fixed; you must not choose or invent any other tool.
        -The tool_name in your output must exactly match the one defined in the tool_description.

        ###Rule 2: Parameter Extraction
        -Extract only the parameters explicitly mentioned or implied in the user question.
        -Parameter names must align with the tool’s input_schema.
        -Never include 'G' or 'backend_kwargs', even if G appears in the required list.
        
        ###Rule 3: Post-Processing Code Generation
        -Generate appropriate Python post-processing code based on user intent (e.g., “Top 5”, “greater than 0.5”, “how many”, etc.).
        -The function must strictly follow this format:
        ```python
        def process(data):
            # The structure of 'data' strictly follows output_schema['result']
            # Perform filtering, sorting, or aggregation operations as needed
            return processed_result 
        ```
        
        #### Sorting Rules
        -“Top N”, “highest”, “largest”, “maximum” → reverse=True (descending)
        -“Bottom N”, “lowest”, “smallest”, “minimum” → reverse=False (ascending)
        ####Example 1: Sorting and Selecting
        User question: “How many connected components are there?”
        ```python
        def process(data):
            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_items[:5])
        ```
        ###Example 2: Counting
        User question: “How many connected components are there?”
        ```python
        def process(data):
            return {{'component_count': len(data)}}
        ```

        ###Rule 4: If the user does not specify any special post-processing requirements, simply return the data as-is:
        ```python
        def process(data):
            return data
        ``` 

        ##Inputs:
        ###User question:
        {question}
        ###Tool Description:
        {tool_description}
        
        ##Instruction:
        Follow all rules and examples above strictly. Your output must be a valid JSON object only — no additional text, markdown, or explanation.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
        if not response_text:
            return {"error": "Empty response from OpenAI API"}
        result_text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.MULTILINE)
        result_text = re.sub(r'\s*```$', '', result_text, flags=re.MULTILINE)
        result_text = result_text.strip()    
        try:
            result_json = json.loads(result_text)
            return result_json
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in extract_parameters_with_postprocess: {e}")
            print(f"Response text: {result_text}")
            return {"error": "JSONDecodeError", "message": str(e), "raw_response": result_text}

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

        prompt = f"""
        You are a professional data analyst responsible for interpreting graph algorithm results.
        Your task is to analyze the computation output of a given tool based on:
        - a natural-language user question (*question*),
        - the corresponding tool description (*tool_description*), and
        - the actual execution result of that tool (*tool_result*).

        Each tool represents a specific graph algorithm (e.g., PageRank representing influence score,
        Louvain representing community structure, Shortest Path representing connectivity / distance, etc.).
        You should generate a concise, human-understandable analytical explanation that directly answers
        the user’s question and helps them understand the real-world meaning behind the numbers.

        ------------------------------------------------
        ## Response Requirements
        - Base your explanation strictly on the provided computation result — do not hallucinate.
        - Answer all information needs contained in the user’s question.
        - When showing ordered results (e.g., top-K ranking), preserve the original ordering.
        - Explain what the metric/value represents in practical terms (e.g., “higher PageRank means 
        the user is more influential in the network”).
        - If the question asks about **paths**, explicitly show the path in the form:`NodeA -> NodeB -> NodeC -> ...`
        - If the available data is insufficient to fully answer the question, explicitly state the limitation or missing information.

        ------------------------------------------------
        ## Inputs:
        ### User Question:
        {question}
        ### Tool Description:
        {tool_description}
        ### Tool Execution Result:
        {tool_result}

        ------------------------------------------------
        ## Output
        Provide a clear, concise explanation in plain English that directly answers the user’s question.
        Use only natural language — do NOT output code or JSON.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
        if not response_text:
            return "Unable to generate answer from the algorithm result."
        return response_text



# 写一个 reasoner 类， 根据传入的配置参数ReasonerConfig 来初始化参数， 要求实现 根据provider 来选择切换对应的大模型OllamaEnv 和  OpenAIEnv
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

    # Delegate operations to underlying environment
    def get_graph_algorithm(self, question, context, language="en"):
        return self.env.get_graph_algorithm(question, context, language)

    def plan_subqueries(self, decompose: bool, query: str) -> dict:
        return self.env.plan_subqueries(decompose, query)
    
    def select_task_type(self, question: str, task_type_list: list) -> dict:
        return self.env.select_task_type(question, task_type_list)
    
    def select_algorithm(self, question: str, algorithm_list: list) -> dict:
        return self.env.select_algorithm(question, algorithm_list)
    
    def extract_parameters_with_postprocess(self, question: str, tool_description: str) -> dict:
        return self.env.extract_parameters_with_postprocess(question, tool_description)
    
    def check_data_dependency(
            self,
            q1_question: str,
            q1_algorithm: str,
            q2_question: str,
            q2_algorithm: str) -> bool:
        if hasattr(self.env, "check_data_dependency"):
            return self.env.check_data_dependency(
                q1_question=q1_question,
                q1_algorithm=q1_algorithm,
                q2_question=q2_question,
                q2_algorithm=q2_algorithm)
        return False

    def generate_answer_from_algorithm_result(self, question: str, tool_description: str, tool_result: Dict[str, Any]):
        return self.env.generate_answer_from_algorithm_result(question, tool_description, tool_result)

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



if __name__ == '__main__':

    llm_env = OpenAIEnv("https://gitaigc.com/v1/", "sk-G30rFStBigqXtuyIOkOo7Zh4QNxO8ZAjfZQ5DYPCgMXbPv8q", "gpt-4o-mini")
    queries = "Recently I discovered that Anna's transaction behavior is anomalous and she might be a potential fraud user. I want to find the potential fraud community around her, suggest possible suspicious transaction paths, and determine how much cash has likely been illegally transferred out."
    sub_queries = llm_env.plan_subqueries(True,queries)
    print(type(sub_queries))
    print(sub_queries)

    # llm_env = OllamaEnv()
    # queries = "What is the name of the deskmate of the US President's son?"
    # print(llm_env.plan_subqueries(True,queries))
