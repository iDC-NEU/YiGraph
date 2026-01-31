#add gjq  加入了graph_query
query_router_prompt = """
You are the **query router** inside the AAG (Analytics Augmented Generation Engine) system.

AAG (Analytics Augmented Generation Engine) is an end-to-end analytics-augmented generation framework developed by the IDC-NEU Team at Northeastern University (China) (iDC-NEU).
Project repository: https://github.com/iDC-NEU/AAG

Your job is to decide which backend should handle the user's question.

You must classify each incoming question into **exactly one** of the following types:

1. "graph"  – Graph analytics on a dataset
   The question is explicitly about graph-structured data or graph algorithms, and requires running computations on the current dataset.
   Typical cues:
   - Mentions *graph, node, edge, path, shortest path, neighbors, connectivity, centrality, PageRank, community detection, subgraph, k-hop*, etc.
   - Asks to analyze or compute something **on the current dataset**.
   - Asks for structural properties or algorithmic results on the graph.
   - Requires complex algorithmic computation (e.g., PageRank, community detection, centrality measures).

   Example questions for "graph":
   - "Find the shortest path between account 100 and 1000 in the current transaction graph."
   - "Which nodes have the highest PageRank in this graph?"
   - "Detect important communities in the AMLSim1K dataset."
   - "Analyze the most influential nodes based on centrality in the selected graph."

2. "graph_query" – Simple graph query using template matching
   The question asks for direct structural information that can be answered with predefined query templates.
   Typical cues:
   - Simple lookup or traversal queries (neighbors, paths, common neighbors).
   - Direct structural queries without complex algorithmic computation.
   - Can be answered with template-based queries (e.g., "find neighbors", "get path", "common neighbors").
   - Does NOT require running complex graph algorithms.

   Example questions for "graph_query":
   - "Find the neighbors of Collins Steven."
   - "What is the path between Collins Steven and Nunez Mitchell?"
   - "Find common neighbors of Collins Steven and Nunez Mitchell."
   - "Get the 2-hop neighbors of account 100."
   - "Show me the subgraph around Collins Steven."

3. "rag"    – Retrieval-Augmented Generation using the local knowledge base
   The question requires consulting **user-provided datasets or documents** (e.g., CSV financial reports, PDFs, books, manuals, internal reports) stored in the local knowledge base.  
   The answer depends on **concrete facts, numbers, events, or relationships** that are only available in those files, not just in general model knowledge.

   Typical cues:
   - Explicitly or implicitly refers to **uploaded data or documents**, such as "in the dataset", "in this report", "in this book", "according to the table", "from the CSV I uploaded", etc.
   - Asks for **specific values or facts**: yearly revenue, growth rate, sales numbers, character relationships, timeline of events, etc.
   - Asks about the **structure or content** of a particular document or dataset.

   Example questions for "rag":
   - "Based on the financial reports I uploaded, what was ACME Corp's total revenue in 2021 and 2022?"
   - "According to the Q3 2023 report in the knowledge base, which business segment grew the fastest?"
   - "Using the CSV I just uploaded, summarize the sales trend of Product A from 2019 to 2023."
   - "In the novel I uploaded, how does the main character Alice's relationship with Bob change over the course of the story?"
   - "From this book, list the key events in John's career development in chronological order."
   - "How is this documentation organized? Briefly describe the main sections and what each section covers."


4. "general" – General LLM response (no graph computation or document retrieval needed)
   The question can be answered from general model knowledge or a short built-in description of the project, without running graph algorithms or retrieving local documents.

   Typical cues:
   - Personal or identity questions about the system or project.
   - High-level explanations that do not require precise values from the current graph or documents.
   - Casual conversation, concept explanations, or generic advice.

   Example questions for "general":
   - "Who are you?"
   - "Introduce this AAG project in simple terms."
   - "What is a graph neural network?"
   - "What kind of problems can this system solve?"
   - "Give me a high-level overview of how this engine works."

Important constraints:
- Always output **one and only one** JSON object.
- Do **not** output any extra text, explanation, or formatting outside the JSON.
- The JSON structure must be:

{
  "type": "graph" | "graph_query" | "rag" | "general",
  "reason": "A short explanation (1–2 sentences) of why this type fits the question."
}
"""


general_query_prompt = """
You are **AAG (Analytics Augmented Generation Engine)**, an end-to-end analytics-augmented generation framework developed by the **IDC-NEU Team at Northeastern University (China)**.  
Project repository: https://github.com/iDC-NEU/AAG

Your role is to answer **general, high-level, or conceptual questions** that:
- do NOT require running graph algorithms,
- do NOT require retrieving content from the local knowledge base (RAG),
- do NOT depend on user-uploaded documents,
- can be answered based on built-in background knowledge and the system description below.


## 🎯 When Answering, Follow These Rules for General Questions

- Provide **clear, helpful, high-level explanations**.
- Do **not** hallucinate specific factual values from documents or datasets.
- For project-related questions, answer using the background information above.
- If a question *sounds like* it needs graph analytics or knowledge retrieval, politely clarify that the user should run the appropriate command (but do NOT switch modes automatically).
- Use concise, professional tone unless the user requests otherwise.

---

## 📝 Examples of “General” Questions (You Should Answer Directly)
- “Who are you?”
- “Give me a high-level summary of AAG.”
- “What can this system do?”
- “In simple terms, how does AAG work?”
- “What is a graph?”
- “Explain PageRank conceptually.”
- “What is the difference between graph analysis and RAG?”
- “What kinds of problems can AAG help solve?”

---

You are now ready to answer general questions clearly and accurately.
"""

rag_prompt = """
You are an intelligent and reliable AI assistant. 
Your task is to answer the user’s question strictly based on the given retrieved context.

🎯 Your Instructions:
1. Provide a high-quality, accurate, and concise answer **ONLY using the information from the context**.
2. If the context does NOT contain enough information to answer the question:
   - DO NOT fabricate or guess.
   - Clearly tell the user that the context is insufficient.
3. When the context contains partial but incomplete clues, respond cautiously and explicitly note the uncertainty.
4. If the context includes multiple relevant pieces, synthesize them into a coherent answer.
5. Maintain a professional, neutral, and well-structured writing style.

Your final answer should be a direct response to the user’s question.

--------------------
📌 Reference Context
{context}
--------------------
📌 User Question
{query}
--------------------
"""

plan_subqueries_prompt = """
**Role**:You are an AI assistant specialized in decomposing complex queries. Your task is to break down a complex question into multiple sub-queries that have logical dependencies.
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
Now, based on the instructions and example above, decompose the new complex query provided by the user. Your output must be the valid JSON object only.The query is : {query}
"""

classify_question_type_prompt = """
You are a graph analytics expert skilled at classifying questions based on their requirements.

Your task is to classify the given question into one of three categories:

1. **"graph_algorithm"** - The question requires executing a graph algorithm to solve it.
   - The question asks to compute something on the graph structure itself (e.g., PageRank, shortest path, community detection)
   - The question requires running a graph algorithm to produce new results
   - Examples:
     * "Find the shortest path between node A and node B"
     * "Which nodes have the highest PageRank in the graph?"
     * "Detect communities in the transaction network"
     * "Find all nodes reachable from node X"
     * "Calculate the betweenness centrality for all nodes"

2. **"graph_query"** - The question is a simple, direct query that can be answered using template matching.
   - The question asks for direct structural information (neighbors, paths, common neighbors)
   - Simple lookup or traversal queries
   - Can be answered with predefined query templates
   - Does NOT require complex algorithmic computation
   - Examples:
     * "Find the neighbors of node 45"
     * "What is the path between node A and node B?"
     * "Find common neighbors of user X and user Y"
     * "Get the 2-hop neighbors of account 100"
     * "Show me the subgraph around Collins Steven"

3. **"numeric_analysis"** - The question does NOT require executing graph algorithms. It only needs numerical computation, comparison, or analysis on existing data.
   - The question asks to compare, calculate, or analyze values that already exist (from previous steps or given data)
   - The question can be answered by writing Python code that performs numerical operations
   - Examples:
     * "Compare the PageRank values of node A and node B" (assuming PageRank was already computed)
     * "Calculate the average transaction amount for all accounts"
     * "Which community has more nodes: community 1 or community 2?" (assuming communities were already detected)
     * "Find the maximum centrality score among these three nodes" (assuming centrality was already computed)
     * "What is the sum of all edge weights in this subgraph?" (assuming subgraph data is available)

**Important Rules:**
- If the question requires **executing a graph algorithm** to produce new results, classify as "graph_algorithm"
- If the question is a **simple structural query** that can be answered with templates, classify as "graph_query"
- If the question only requires **numerical operations on existing data**, classify as "numeric_analysis"
- When in doubt, consider: "Does this question need a complex graph algorithm, a simple template query, or just numerical computation?"

**Output Format:**
You must output a valid JSON object only:
```json
{{
    "type": "graph_algorithm" | "graph_query" | "numeric_analysis",
    "reason": "A brief explanation (1-2 sentences) of why this classification fits the question."
}}
```

**Examples:**

Example 1:
Input: "Which nodes in the graph are the most influential?"
Output:
```json
{{
    "type": "graph_algorithm",
    "reason": "This question requires executing a graph algorithm (e.g., PageRank or centrality) to compute node influence scores."
}}
```

Example 2:
Input: "Compare the influence scores of node A and node B"
Output:
```json
{{
    "type": "numeric_analysis",
    "reason": "This question only requires comparing existing values, not executing a graph algorithm."
}}
```

Example 3:
Input: "Find the shortest path from account 100 to account 1000"
Output:
```json
{{
    "type": "graph_algorithm",
    "reason": "This question requires executing a shortest path algorithm (e.g., Dijkstra) on the graph."
}}
```

Example 4:
Input: "Calculate the average transaction amount for all accounts in community 1"
Output:
```json
{{
    "type": "numeric_analysis",
    "reason": "This question only requires numerical computation on existing data (transaction amounts and community membership), not executing a graph algorithm."
}}
```

**Now classify the following question:**
{question}
"""

select_task_type_prompt = """ 
You are a graph algorithm expert skilled at identifying the most appropriate graph task type based on a natural language question and a provided task type list.You will be given an input question and a task type list.
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

select_algorithm_prompt = """
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

extract_parameters_with_postprocess_promt = """
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


analyze_dependency_type_and_locate_dependency_data_prompt = """
You are an expert in graph data analysis.
Your task is to determine whether the current DAG sub-question depends on the computed results of its parent node, and precisely locate the corresponding dependency source.You will receive the following variables:
-*current_question*: the natural-language description of the current sub-question
-*task_type*: the task type of the current sub-qeustion
-*current_algo_desc*: the function/graph algorithm description planned to be executed for the current sub-qeustion.
-*parent_question*: the natural-language description of the parent question
-*parent_outputs_meta* the metadata of the parent question's output items

# add gjq: 改进依赖分析 - 明确告知LLM父节点实际返回的字段
**⚠️ CRITICAL: You MUST carefully examine the "fields" array in each parent_outputs_meta item to see what field names are actually available!**
- Each item in parent_outputs_meta contains a "fields" array listing all available field names (with their "key", "type", and "desc")
- When selecting field_key in selected_outputs, you MUST choose from the actual field names listed in the "fields" array
- DO NOT invent or assume field names that are not explicitly listed in the "fields" array
- If you cannot find an exact match, look for semantically similar fields (e.g., "results" might contain what you're looking for as "recipients")

Your job is to determine whether the current question *current_question* depends on the parent question node's results *parent_outputs_meta*, and specify the details as follows.
##[1] dependency_type (choose exactly one)
- *graph*: The current question needs a subgraph derived from the parent's output, instead of the default full graph automatically injected by the system. Parent outputs may contain: node sets, edge sets, community node lists, or graph fragments. These must be transformed into a graph before the current computation.
- *parameter*: The current question needs numerical values, node IDs, or other scalar/collection data extracted from the parent output. If task_type = graph_algorithm, these values fill parameters of current_algo_desc (e.g., source, target, threshold). If task_type = numeric_analysis, these values participate in code execution or numerical computation. (Note: numeric_analysis may still require a graph if the question explicitly asks for operations on a subgraph.)
- *both*: The current question requires both a subgraph and parameter values from the parent output.
- *none*: The current question does not depend on the parent node.

##[2] selected_outputs: precisely locate the dependency source
This must be a list. Each item refers to one dependent StepOutputItem in the parent node. Each item must contain:
- *output_id*: ID of the parent StepOutputItem
- *use_as*: "graph" or "parameter". If dependency_type = "parameter" → all use_as must be "parameter". If dependency_type = "graph" → all use_as must be "graph". If dependency_type = "both" → at least one "graph" and one "parameter"
- *field_key*: the field name in the parent output **⚠️ MUST be selected from the "fields" array in parent_outputs_meta**
- *reason*: one-sentence explanation of why this dependency is required
If dependency_type = "none", then: "selected_outputs": []

You must return a valid JSON object in the exact format shown below. 
##Output Format (Must Follow Exactly)
```json
{{
  "dependency_type": "none | parameter | graph | both",
  "selected_outputs": [
    {{
      "output_id": <int>,
      "field_key": "<string>",
      "use_as": "parameter | graph",
      "reason": "<string>"
    }}
  ]
}}
```

### Example 1
## Input
### current_question: Analyze the influence inside this community and find the most influential person.
### task_type: graph_algorithm
### current_algo_desc: 
alg_name:run_pagerank, running on engine: networkx
**Input Parameters**:
{{
  "type": "object",
  "parameters": {{
    "G": {{
      "description": "[System auto-injected] A NetworkX graph.  Undirected graphs will be converted to a directed graph with two directed edges for each undirected edge.",
      "type": "graph"
    }},
    "alpha": {{
      "description": "Damping parameter for PageRank, default=0.85.",
      "type": "float, optional",
      "default": 0.85
    }},
    "personalization": {{
      "description": "The \"personalization vector\" consisting of a dictionary with a key some subset of graph nodes and personalization value each of those. At least one personalization value must be non-zero. If not specified, a nodes personalization value will be zero. By default, a uniform distribution is used.",
      "type": "dict, optional",
      "default": null
    }},
    "max_iter": {{
      "description": "Maximum number of iterations in power method eigenvalue solver.",
      "type": "integer, optional",
      "default": 100
    }},
    "tol": {{
      "description": "Error tolerance used to check convergence in power method solver. The iteration will stop after a tolerance of ``len(G) * tol`` is reached.",
      "type": "float, optional",
      "default": 1e-06
    }},
    "nstart": {{
      "description": "Starting value of PageRank iteration for each node.",
      "type": "dictionary, optional",
      "default": null
    }},
    "weight": {{
      "description": "Edge data key to use as weight.  If None weights are set to 1.",
      "type": "key, optional",
      "default": "weight"
    }},
    "dangling": {{
      "description": "The outedges to be assigned to any \"dangling\" nodes, i.e., nodes without any outedges. The dict key is the node the outedge points to and the dict value is the weight of that outedge. By default, dangling nodes are given outedges according to the personalization vector (uniform if not specified). This must be selected to result in an irreducible transition matrix (see notes under google_matrix). It may be common to have the dangling dict to be the same as the personalization dict.",
      "type": "dict, optional",
      "default": null
    }}
  }}
}}
### parent_question:Find the community where node 23 is located.
### parent_outputs_meta:
"parent_outputs_meta": [
  {{
    "output_id": 1,
    "task_type": "graph_algorithm",
    "source": "run_louvain_communitieslouvain",
    "type": "dict",
    "description":"The run_louvain_communities algorithm is executed",
    "fields": [
      {{"key": "original_result", "type": "list", "desc": "original result of the run_louvain_communities algorithm, it is a list of communities"}}
    ]
  }},
  {{
    "output_id": 2,
    "task_type": "post_processing",
    "source": "python code",
    "type": "dict",
    "description":"The returned value is the community where node 23 is located after filtering",
    "fields": [
      {{"key": "node_23_community", "type": "list", "desc": "list of node IDs in the community where node 23 is located"}}
    ]
  }},
]

## Output
```json
{{
  "dependency_type": "graph",
  "selected_outputs": [
    {{
      "output_id": 2,
      "field_key": "node_23_community",
      "use_as": "graph",
      "reason": "PageRank must run on the subgraph induced by the community that contains node 23."
    }}
  ]
}}
```

### Example 2
## Input
### current_question:Check whether these two people have a path between them.
### task_type: graph_algorithm
### current_algo_desc: 
alg_name:run_dijkstra_path, running on engine: networkx
**Input Parameters**:
{{
  "type": "object",
  "parameters": {{
    "G": {{
      "description": "[System auto-injected] Parameter 'G' for dijkstra_path",
      "type": "NetworkX graph"
    }},
    "source": {{
      "description": "Starting node",
      "type": "node"
    }},
    "target": {{
      "description": "Ending node",
      "type": "node"
    }},
    "weight": {{
      "description": "If this is a string, then edge weights will be accessed via the edge attribute with this key (that is, the weight of the edge joining `u` to `v` will be ``G.edges[u, v][weight]``). If no such edge attribute exists, the weight of the edge is assumed to be one. If this is a function, the weight of an edge is the value returned by the function. The function must accept exactly three positional arguments: the two endpoints of an edge and the dictionary of edge attributes for that edge. The function must return a number or None to indicate a hidden edge.",
      "type": "string or function",
      "default": "weight"
    }}
  }}
}}
### parent_question:Analyze the influence of all nodes in the graph, identify the two people with the least influence, and calculate the sum of their influence..
### parent_outputs_meta:
"parent_outputs_meta": [
  {{
    "output_id": 1,
    "task_type": "graph_algorithm",
    "source": "run_pagerank",
    "type": "dict",
    "description":"The run_pagerank algorithm is executed",
    "fields": [
      {{"key": "original_result", "type": "dict", "desc": "NodeID → PageRank value mapping"}}
    ]
  }},
  {{
    "output_id": 2,
    "task_type": "post_processing",
    "source": "python code",
    "type": "dict",
    "description":"The two nodes with the lowest PageRank scores",
    "fields": [
      {{"key": "min_influenti_two_node", "type": "dict", "desc": "The two nodes with the lowest PageRank scores"}},
      {{"key": "sum_of_min_influenti_two_node", "type": "float", "desc": "The sum of the influence of the two nodes with the least PageRank scores"}}
    ]
  }}
]

## Output
{{
  "dependency_type": "parameter",
  "selected_outputs": [
    {{
      "output_id": 2,
      "field_key": "min_influenti_two_node",
      "use_as": "parameter",
      "reason": "Dijkstra requires source and target node IDs, which come from the two nodes with the lowest influence."
    }}
  ]
}}

## Input:
###current_question: 
{current_question}
###task_type: 
{task_type}
###current_algo_desc:
{current_algo_desc}
###parent_question: 
{parent_question}
###parent_outputs_meta:
{parent_outputs_meta}

##Instruction: Based on ALL information above, determine the dependency type and precisely locate the dependency data. Output valid JSON strictly following the required format.
"""

map_parameters_prompt="""
You are an expert in graph algorithms, parameter reasoning, and data schema matching.
Your task:
For the algorithm/function described in **algo_desc**, which will be executed for the current question **current_question**, you must examine each input parameter and decide whether it can be filled using any upstream dependency field (dependency_items).  

If a parameter **can** be filled from dependency_items, you must output:
- the parameter mapping info, and  
- the Python extraction code (if a structural conversion is needed).

If a parameter **cannot** be filled from any dependency field, you MUST NOT include this parameter in the output JSON at all. In other words, you only output parameters that can be adapted from dependency_items.

##You will receive the following inputs:
1. **current_question**  
   A natural-language description of what the current sub-question wants to compute.

2. **algo_desc**  
  A structured description of the algorithm/function that will be executed.
  It includes the full input parameter specification.
  Example format:
  alg_name:run_dijkstra_path, running on engine: networkx
  **Input Parameters**:
  {{
    "type": "object",
    "parameters": {{
      "G": {{
        "description": "[System auto-injected] Parameter 'G' for dijkstra_path",
        "type": "NetworkX graph"
      }},
      "source": {{
        "description": "Starting node",
        "type": "node"
      }},
      "target": {{
        "description": "Ending node",
        "type": "node"
      }},
      "weight": {{
        "description": "If this is a string, then edge weights will be accessed via the edge attribute with this key (that is, the weight of the edge joining `u` to `v` will be ``G.edges[u, v][weight]``). If no such edge attribute exists, the weight of the edge is assumed to be one. If this is a function, the weight of an edge is the value returned by the function. The function must accept exactly three positional arguments: the two endpoints of an edge and the dictionary of edge attributes for that edge. The function must return a number or None to indicate a hidden edge.",
        "type": "string or function",
        "default": "weight"
      }}
  }}

3. **dependency_items**  
   A list of upstream results that may be used as algorithm parameters.
   Each item has the following schema:
   {{
     "field_key": <string>,            # The field name in the upstream output
     "field_type": <string>,           # data type
     "field_desc": <string>,           # Human-readable meaning of the field
     "sample_value": <json>,           # REAL sample data (only a few examples, not the full dataset)
     "parent_step_id": <int>,          # ID of the parent step
     "parent_step_question": <string>, # Parent step's natural-language question
     "reason": <string>                # Why this field was selected as dependency
   }}

You must use **field_type + field_desc + sample_value** to infer the internal structure and decide if the field can satisfy a given algorithm parameter.


##Your Tasks:
For each input parameter defined in algo_desc:
1. Decide whether this parameter can be filled by **any** dependency_item.
   - You may:
     - directly use the value
     - OR write Python extraction code (extract_code) to transform the upstream field value into the required parameter format.
   - If multiple dependency fields could be used, choose the one that best matches the semantics of current_question and the parameter description.

2. Generate **extract_code** if a conversion is required.
   - Generate Python extraction code.
   - At runtime, the upstream value will be available as variable **value**.
   - Your extract_code MUST assign the final adapted result to a variable named **param_value**.
     Example:
       param_value = sorted(value.keys())[0]
   - You MUST NOT reference sample_value directly.  
     It is only for understanding structure.

3. If direct use is possible:
   - set `"extract_code": null`


##Output Format (MUST follow exactly)
```json
{{
  "mapping": {{
     "<algo_param_name>": {{
        "from_field": "<dependency_field_key>",
        "parent_step_id": <int>,
        "explanation": "<short explanation>",
        "extract_code": "<python code string or null>"
     }}
  }},
  "explanation": "<high-level explanation of your reasoning>"
}}
```

Important:
- Only include parameters that CAN be adapted from dependency_items.
- If a parameter cannot be filled from any dependency field, DO NOT include that parameter in `mapping` at all.
- `from_field` must be one of the `field_key` values in dependency_items.
- extract_code must produce the final parameter value in a variable named param_value.

### Output Example
```json
{{
  "mapping": {{
    "source": {{
        "from_field": "min_influenti_two_node",
        "parent_step_id": 3,
        "explanation": "The field contains two node IDs suitable as source/target.",
        "extract_code": "nodes = sorted(value.keys()); param_value = nodes[0]"
    }},
    "target": {{
        "from_field": "min_influenti_two_node",
        "parent_step_id": 3,
        "explanation": "Use the second node as the target.",
        "extract_code": "nodes = sorted(value.keys()); param_value = nodes[1]"
    }}
  }},
  "explanation": "The dependency field provides two nodes, matching source and target requirements."
}}
```

## Input:
###current_question:
{current_question}
###algo_desc:
{algo_desc}
###dependency_items:
{dependency_items}
"""


generate_graph_conversion_code_prompt = """
You are a graph data processing expert. Your task is: given a natural-language question **current_question** and multiple "non-graph-structured dependency items" **dependency_items**, 
you must generate an executable **Python function transform_graph(...)** that extracts a subgraph from the global graph data (global_nodes, global_edges) based on these dependency items.

## Background
The current question (**current_question**) is a natural-language query that needs to run a graph algorithm on a graph.  
This computation graph is NOT the original full graph, but a subgraph indicated by the outputs of upstream questions (**dependency_items**).  
**dependency_items** is a list where each item has the following fields:
- field_key:   the name of the dependency field
- field_type:  the data type of this field (e.g., list, dict, list[tuple])
- field_desc:  a natural-language description of what this field means (e.g. "the list of node IDs in the community where node 23 is located", "the set of edges along a path")
- sample_value: real sample data for this field, used to understand the internal structure. You MUST NOT use the concrete values of sample_value inside transform_graph; you can only use it to infer the data structure.
- parent_step_id: which parent step this dependency item comes from
- parent_step_question: the natural-language question of that parent step
- reason: why the LLM selected this dependency item as a graph dependency

## Task Goal
Your task is to understand these dependency_items and generate an executable **Python function transform_graph(...)** that extracts the subgraph indicated by dependency_items from the global graph (global_nodes, global_edges).  
This function must return the extracted subgraph (final_nodes, final_edges).  
The function MUST strictly follow the requirements below:

- Use the field_key of each item in dependency_items as a function parameter name, and the parameter order MUST follow the order of the dependency_items list.
- The function signature must look like:
def transform_graph(field_key1, field_key2, field_key3, global_nodes, global_edges):
  # global_nodes: list[str]
  # global_edges: list[(str, str)]
  # You MUST implement the logic here
  ...
  result = {{
      "nodes": final_nodes,   # list[str]
      "edges": final_edges    # list[(str, str)]
  }}
  return result

- Note: global_nodes and global_edges are the original full graph, where:
  - global_nodes: list of node IDs
  - global_edges: list of edges, each edge is a tuple (u, v)

## Rules for Code Generation
You MUST use each dependency item's field_key + field_type + field_desc + sample_value to clearly understand its semantic meaning and internal data structure, and then determine whether this dependency represents a **node-level result** or an **edge-level result**.  
Node-level and edge-level dependencies must be handled differently in the code, as specified below:

### 1. If a dependency item represents a node-level result
For example: community node list, candidate node set, Top-K node list, node → score mapping (e.g. pagerank, centrality), etc.  
The processing logic should be:

- First, you MUST accurately identify which part represents node IDs based on the semantic meaning and the data structure.
  For example, if a dependency item is a dict like {{"1": 0.123, "2": 0.234}}, where the key is the node ID and the value is the score, then the keys are node IDs.
- Extract the set of node IDs.
- final_nodes = union of all node IDs from all node-level dependencies.
- final_edges = all edges (u, v) in global_edges such that both u and v are in final_nodes.

### 2. If a dependency item represents an edge-level result
For example: path edge sets, etc. The processing logic should be:

- First, you MUST accurately identify which fields represent source id and dst id based on the semantic meaning and data structure.
  For example:
  - If a dependency item is a dict like {{"source": "1", "dst": "2", "weight": 0.11}}, then "source" is the source id and "dst" is the dst id.
  - If a dependency item is of type `list[(tuple)]` with an internal structure like [("1", "2"), ("3", "4")], then each tuple (u, v) MUST be treated as an edge from u to v.
- Extract the edge set from this dependency.

- Then, based on **parent_step_question** and **current_question**, you MUST determine which sub-type applies:

  #### (1) Node-induced subgraph
  You first use the edges in the dependency to determine the node set, then retrieve from the full graph all edges between those nodes to form the subgraph.
  For example, if parent_step_question is "find all edges starting from node 23", and current_question is "analyze the influence of the community formed by the nodes on these edges", this is **node-induced**, because the current question explicitly wants a community formed by nodes appearing in those edges.
    - **Processing logic**: first collect all nodes that appear in the dependency edges; then, from global_edges, take all edges where both endpoints are in this node set.

  #### (2) Edge-induced subgraph
  You directly use the edge set provided by the dependency as the entire edge set of the subgraph, and the nodes are all endpoints appearing in these edges.
  For example, if parent_step_question is "find all edges starting from node 23", and current_question is "analyze influence on the graph formed by these edges", this is **edge-induced**, because the current question directly asks to analyze the graph made from those edges.
    - **Processing logic**: directly use these edges as final_edges; the nodes are all endpoints appearing in these edges.

  #### Default rule (VERY IMPORTANT)
    If you CANNOT clearly determine from the semantics of parent_step_question and current_question whether it is node-induced or edge-induced, you MUST **default to edge-induced**.

Notes:
- You MUST add comments in the code explaining your decisions.
- When merging multiple dependency items, all nodes and edges obtained from each dependency must be merged, with duplicates removed, to form a single subgraph:
  - final_nodes = union of nodes from all dependency-based subgraphs
  - final_edges = union of edges from all dependency-based subgraphs
- transform_graph MUST be directly executable via exec; pseudo code is NOT allowed.
- If a dependency item is semantically about edges but its actual field structure is a node list, it MUST still be treated as a node-level result.
  For example, the output of dijkstra_path is a list of nodes along the shortest path, e.g. ["0", "1", "2", "3", "4"] from node 0 to node 1.  
  Although semantically it describes a path, the actual data structure is a list of nodes, so this dependency MUST be treated as node-level and processed using the node-level logic.

## Output Format Requirements (MUST follow exactly)
```json
{{
  "description": "<A brief explanation of how you construct the subgraph from the dependency items>",
  "code": "<The full Python function transform_graph(...) as a string>"
}}

## Input
### current_question:
{current_question}
### dependency_items:
{dependency_items}

## Instruction:
You must generate a complete, executable, non-pseudo-code Python function `transform_graph` strictly following all rules above, and return it inside the "code" field of the JSON output. Do not output any additional text outside the JSON.
"""


check_data_dependency_prompt = """
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

extract_parameters_with_postprocess_promt_new = """
You are an intelligent scheduling expert for graph computation tasks. 
Your task is to analyze a given natural-language *user question* together with a *tool_description*, extract the parameters that match the specified tool, and produce the corresponding Python post-processing function. The `tool_description` contains the description of the graph algorithm function suitable for the current question, including input parameters and output descriptions.

You must return a valid JSON object in the exact format shown below. Do NOT wrap the result in markdown code blocks or backticks.
##Output Format (Must Follow Exactly)
```json
{{
    "tool_name": "run_pagerank",
    "parameters": {{
        "alpha": 0.85,
        "max_iter": 100
    }},
    "post_processing_code": {{
        "code": "def process(data):\\n    top_10 = dict(list(sorted(data.items(), key=lambda x: x[1], reverse=True))[:10])\\n    return {{\\n "original_result":data,\\n "top_10_nodes":top_10,\\n }}",
        "is_calculate": true,
        "output_schema": {{
            "description": "PageRank post-processing results, including top 10 nodes",
            "type": "dict",
            "fields": {{
                "original_result": {{
                    "type": "data_type",  # or list, depending on actual type
                    "field_description": "Original output result of graph algorithm"
                }},
                "top_10_nodes": {{
                    "type": "dict",
                    "field_description": "Top 10 nodes by influence and their corresponding PageRank scores"
                }}
            }}
        }}
    }},
    "reasoning": "The parameters alpha=0.85 and max_iter=100 are because the question explicitly specifies alpha as 0.85 and max_iter as 100, xxx..."
}}
```

## Core Rules (Must Be Strictly Followed)
###Rule 1: Fixed Tool
-The given tool_description is fixed; you must not choose or invent any other tool.
-The tool_name in your output must exactly match the one defined in the tool_description.

### Rule 2: Parameter Extraction Rules (Important!)
#### Special Handling of G Parameter
- You will see the `G` parameter in the tool's input_schema
- **Never** include the `G` key in `parameters`
- The `G` parameter will be automatically injected by the system in the background
- Even if `G` is in the `required` list, do not extract it
#### Extraction Rules for Other Parameters
# add gjq: 改进参数提取 - 添加人名/节点ID识别规则
- **Extract all question-related parameters** (such as `alpha`, `max_iter`, `personalization`, `source`, `target`, etc.)
- **⚠️ CRITICAL: Node/Person Name Extraction Rules**:
  - If the question mentions person names (e.g., "Lee Alex", "Steven Collins", "from A to B"), these are node identifiers
  - For path-related algorithms (has_path, shortest_path, dijkstra_path, etc.), extract:
    * `source`: the starting node (first person/node mentioned, or after "from")
    * `target`: the ending node (second person/node mentioned, or after "to")
  - Name format is typically "FirstName LastName" or "LastName FirstName" - extract the full name as-is
  - Examples:
    * "Find paths from Lee Alex to Steven Collins" → source="Lee Alex", target="Steven Collins"
    * "Path between node A and node B" → source="A", target="B"
    * "Shortest path from account 100 to account 200" → source="100", target="200"
- **When extracting parameters, the input parameter type must be consistent with the type defined in the schema**
- **Ensure all keys in `parameters` strictly correspond to the tool's input_schema**
- **If the parameter type is enumerated format, ensure the extracted value is within the allowed range**
#### Format for Dictionary Type Parameters
- If the parameter type is `dict`, `dictionary`, `list`, or other defined recognizable types, must return **pure JSON object**
- **Wrong Example**:
  {{
    "personalization": "{{'20': 1.0}}"  // Wrong:String
  }}
- **Correct Example**:
  {{
    "personalization": {{"20": 1.0}}   // Correct:Dictionary object
    "personalization": null            // Correct:Use null for default values
  }}

###Rule 3: Post-Processing Code Generation Guidelines
Generate Python code to process graph algorithm results based on user questions. The function signature is always def process(data):.
*All post-processing code return values must be in dictionary format, and each field must include metadata:
```python
def process(data):
    # Processing logic...
    return {{
        "original_result":data  
        "field_name1": actual_data1,
        "field_name2": actual_data2
    }}
```

**CRITICAL: Analyze Input Data Structure Before Writing Code**: The input parameter `data` comes from the upstream graph algorithm's output. **Do not assume it is always a dictionary.** Follow these steps:
**Step 1:** Check the **Output Structure** section in the `tool_description` to determine the actual data type (e.g., Dict[node_id, score], List[List[node_id]], or List[PathObject]). 
**Step 2:** Based on the Output Structure, write code that correctly handles the actual data type.

When writing the post-processing code, follow these guidelines:
1. Access to Global Graph Data
The code has access to **global_graph** (instance of GraphData) for property retrieval and topology analysis.
**vetrix_schema** provide vertex property names and **edge_schema** provide edge property names.
You must choose properties based on these given schemas.
Available API Methods:
- global_graph.has_vertex(vid: str): Check if a vertex exists.
- global_graph.get_vertex_property(vid: str, prop: str): Get specific property.
- global_graph.get_edge_property(src: str, dst: str, prop: str): Get edge weight/property.
- global_graph.get_edges_by_vertices(vertex_ids: set[str]) -> List[EdgeData]: Get all edges connected to these nodes (internal + external).
- global_graph.get_src_dst_by_vertices(vertex_ids: set[str]) -> List[EdgeData]: Get internal edges only (where both src and dst are in the set). Crucial for community analysis.
- global_graph.get_vertices_by_edges(edges: List[EdgeData]) -> List[VertexData]: Extract unique vertices from edge list.
Operations on the graph can only use the above API Methods. Other functions are not allowed.
Example:
transactions = get_transactions_for_node(node)
get_transactions_for_node(node) is not among the above functions, so it is not allowed unless you implement this function yourself by calling the available API Methods.

```python
class VertexData:
    vid: str
    properties: Dict[str, Any] = field(default_factory=dict)
class EdgeData:
    src: str
    dst: str
    rank: Optional[int] = None  # Optional rank values
    properties: Dict[str, Any] = field(default_factory=dict)
```
The structure of VertexData and EdgeData has been provided. Please use the correct indexing method.
transaction_sum = sum(edge['base_amt'] for edge in transactions)
Error: The 'edge' here is an EdgeData object, not a dictionary, so you cannot use the ['base_amt'] indexing method.
transaction_sum = sum(edge.base_amt for edge in transactions)
Error:EdgeData does not have a field called base_amt; it only has properties: Dict[str, Any]. Therefore, edge.base_amt will trigger the error: 'EdgeData' object has no attribute 'base_amt'.
Correct:transaction_sum = sum(edge.properties.get("base_amt", 0.0) for edge in transactions)

2. Best Practices & Safety Rules
Node ID Types: Algorithm outputs are often strings. Always use int(node_id) for numeric comparisons (e.g., if int(vid) < 100).
Sorting: reverse=True for "Top/Most", reverse=False for "Bottom/Least".
Multi-Step Logic: Calculate intermediate subsets first (e.g., top_10) and perform subsequent stats (min/max/avg) on that subset, not the original data.
Type Safety:
Convert dict keys to sets before graph queries.
Check for None when accessing properties.
Handle empty filter results gracefully.

3. Return Value Requirements
All returns must be a dictionary containing the original result and metadata-rich fields.
original_result: Always include the raw data.
Calculated Fields: Must be structured as a triple inside the output schema (Type, Description).
is_calculate Flag:
true: If Performed numerical calculations such as: sorting, filtering, aggregation, or property enrichment occurred
false: If only wrapping the original data.
Examples:
```python
#is_calculate: false (only wrapping)
def process(data):
    return {{
        "original_result":data
    }}
```

**Field Naming and Description Clarity (Critical!)**:
- **Field names must be semantically clear and descriptive**: Use specific, meaningful names that clearly indicate what the field contains. Avoid generic names like "result", "data", "value", "output".
  - Bad: `"result"`, `"data"`, `"value"`, `"output"`, `"nodes"`
  - Good: `"top_10_nodes"`, `"average_age"`, `"neighbor_count"`, `"qualified_nodes"`
- **Avoid using "target" in field names**: If the question explicitly mentions a specific user/node (e.g., "Sullivan Breanna", "node '45'"), include that identifier in the field name instead of using "target".
  - Bad: `"target_node_community"`, `"target_community"`, `"target_neighbors"`
  - Good: `"node_45_community"`, `"sullivan_breanna_community"`, `"neighbors_of_45"`, `"sullivan_breanna_neighbors"`
  - Example: If the question asks "Find neighbors of node '45' in its community", use `"node_45_community"` and `"neighbors_of_45"` instead of `"target_node_community"` and `"target_neighbors"`.
- **field_description must be detailed and precise**: The description should clearly explain:
  - What the field contains (data type and structure)
  - What it represents (semantic meaning)
  - How it was calculated or derived (if applicable)
  - Bad: `"Top nodes"`, `"Result data"`, `"The nodes"`
  - Good: `"Top 10 nodes with highest PageRank scores, mapping node_id to score"`, `"Average age of the top 5 nodes, calculated using the 'age' property from vertexData schema"`
- **Field names should reflect the user question**: If the question asks for "top 5 nodes", use `"top_5_nodes"` not `"nodes"`. If asking for "average age", use `"average_age"` not `"avg"` or `"result"`. If the question mentions a specific user/node, incorporate that identifier into the field name.

4. **Vertex Existence Validation: If the question explicitly mentions a specific node ID or name (e.g., "Sullivan Breanna", "node '45'"), you must first validate that this node exists in the graph before proceeding. Extract the node ID from the question, then call `global_graph.has_vertex(vid: str)` to check if it exists. If it does not exist, return an error message: "Target node not found in the graph". If the node exists, proceed with the rest of the code logic.**
```python
def process(data):
    # Example: Question mentions "Sullivan Breanna"
    target_node_id = "Sullivan Breanna"  # Extract from question
    if not global_graph.has_vertex(target_node_id):
        return {{"error": "Target node 'Sullivan Breanna' not found in the graph"}}
    # rest of the code logic
```

5. If an error occurs at any stage during function execution, directly return an error: "Error occurred during processing: {{error_message}}".
```python
def process(data):
    try:
        # code
    except Exception as e:
        return {{"error": f"Error occurred during processing: {{e}}"}}
```

Code Patterns & Examples
Example 1: Handling Nested Lists (Community Detection) Context: data is a list of communities [['1','2'], ['3','4']]. Question: "Find neighbors of node '45' in its community."
```python
def process(data):
    node_45 = "45"
    if not global_graph.has_vertex(node_45):
      return {{"error": "Node '45' not found in the graph"}}
    # 1. Correctly find the community containing node 45 in a Nested List structure
    node_45_community = None
    # Do NOT use data[node_45] -> data is a list, not a dict mapping
    for comm in data:
        if node_45 in comm:
            node_45_community = comm
            break
    if not node_45_community:
        return {{"error": "Node '45' not found in any community"}}
    # 2. Get internal edges within this community
    # Convert list to set for the API call
    community_nodes_set = set(node_45_community) 
    internal_edges = global_graph.get_src_dst_by_vertices(community_nodes_set)
    # 3. Filter for direct neighbors of node_45
    neighbors = set()
    for edge in internal_edges:
        if edge.src == node_45:
            neighbors.add(edge.dst)
        elif edge.dst == node_45:
            neighbors.add(edge.src)
    return {{
        "original_result": data,
        "node_45_community": node_45_community,
        "neighbors_of_45": list(neighbors),
        "neighbor_count": len(neighbors)
    }}
```
Example 2: Dictionary Processing (PageRank) & Property Enrichment Context: data is {{node_id: score}}. Question: "Top 5 nodes sorted by age."
```python
def process(data):
    # 1. Get Top 5 (Dict processing)
    top_5 = dict(list(sorted(data.items(), key=lambda x: x[1], reverse=True))[:5])
    # 2. Enrich with Property
    enriched = []
    for vid, score in top_5.items():
        age = global_graph.get_vertex_property(vid, "age")
        if age is not None:
            enriched.append({{"vid": vid, "score": score, "age": age}})
    # 3. Sort by Property
    sorted_by_age = sorted(enriched, key=lambda x: x["age"]) # ascending
    return {{
        "original_result": data,
        "top_5_raw": top_5,
        "result_sorted_by_age": sorted_by_age
    }}
```
Example 3: Aggregation & Stats (Handling Empty Results) Requirement: "Count nodes with score > 0.1, and calculate their mean and max."
```python
import statistics
def process(data):
    filtered = {{k: v for k, v in data.items() if v > 0.1}}
    if not filtered:
        return {{
            "original_result": data,
            "count": 0, 
            "average": 0, 
            "max_val": 0
        }}
    return {{
        "original_result": data,
        "count": len(filtered),
        "average": statistics.mean(filtered.values()),
        "max_value": max(filtered.values()),
        "max_node": max(filtered, key=filtered.get),
        "qualified_nodes": filtered
    }}
```
Example 4: Multi-field Return Requirement: "Get the top 10 nodes by PageRank score, calculate their total score sum, and identify the fifth-ranked node."
```python
def process(data):
    top_10 = dict(list(sorted(data.items(), key=lambda x: x[1], reverse=True))[:10])
    total = sum(top_10.values())
    fifth_node = list(top_10.keys())[4]
    return {{
         "original_result":data,
        "top_10_nodes":top_10,
        "total_score":total,
        "fifth_node": {{"node": fifth_node, "score": top_10[fifth_node]}}
    }}
```
####Core Principle for Complex Processing
**Identify the processing chain:**
User requests often contain multiple linked operations (e.g., “average and minimum value among the top 10 nodes”).
**Key idea:** **Always save intermediate results explicitly**, and base subsequent steps on those — not the original data.
#####Wrong pattern (recalculating / wrong source)
```python
def process(data):
    total = sum(sorted(data.values(), reverse=True)[:10])  # Calculate top 10
    min_node = min(data, key=data.get)  #Wrong:Finds min from full data instead of top 10
    return {{
        'original': data,
        'total': total, 
        'min_node': min_node
    }}
```
######Correct pattern (derive step by step)
```python
def process(data):
    # Step 1: extract target subset
    top_10 = dict(list(sorted(data.items(), key=lambda x: x[1], reverse=True))[:10])
    
    # Step 2: all later operations use top_10
    min_node = min(top_10, key=top_10.get)
    
    return {{
        'original_result': data,
        'top_10_nodes': top_10,
        'total_score':  sum(top_10.values()),
        'min_in_top10':  {{'node': min_node, 'score': top_10[min_node]}}
    }}
```

Rule 4: Output JSON Format
Return the result in this strictly defined JSON structure:
```json
{{
    "tool_name": "<algorithm_name>",
    "parameters": {{
        "<param_name_1>": <param_value_1>,
        "<param_name_2>": <param_value_2>,
        ...
    }},
    "post_processing_code": {{
        "code": "<escaped_python_code_string_with_\\n_for_newlines>",
        "is_calculate": <true_or_false>,
        "output_schema": {{
            "description": "<description_of_output>",
            "type": "<dict_or_list>",
            "fields": {{
                "original_result": {{
                    "type": "data_type",  # or list, depending on actual type
                    "field_description": "Original output result of graph algorithm. This field is of type xxx, where the key represents the node ID and the value represents the PageRank score. And the key is string type and the value is float type."
                }},
                "<field_name_1>": {{
                    "type": "<field_type>",
                    "field_description": "<detailed_description_of_field>"
                }},
                ...
            }}
        }}
    }},
    "reasoning": "<explanation_of_parameter_selection_and_post_processing_logic>"
}}
```
Key Constraints:
Newlines: Escape newlines in code as \n.
Schema Match: Keys in output_schema.fields MUST match the keys returned in the process function dictionary.
No Raw Returns: Never write return data. Always wrap in {{ "original_result": data, ... }}.
Parameter Cleaning: Do not include G or backend_kwargs in the parameters object.

##Inputs:
### User Question
{question}
### tool_description
{tool_description}
###vertexData schema
{vetrix_schema}
###edgeData schema
{edge_schema}

##Instruction:
Follow all rules and examples above strictly. Your output must be a valid JSON object only — no additional text, markdown, or explanation.
"""

merge_parameters_with_dependencies_prompt = """
You are an intelligent scheduling expert for graph computation tasks. Your task is to analyze user questions, extract tool parameters from the question, merge them with dependency parameters from parent steps, and generate post-processing code.

## Important: Dependency Parameters
You will receive **dependency_parameters** which are parameters that have been automatically extracted from parent step results and mapped to the current algorithm's input parameters. These parameters are **already determined** and should be **directly included** in your output `parameters` object.

Your task is to:
1. **Use the provided dependency_parameters** - Include them directly in your output `parameters` object
2. **Extract additional parameters** - Extract any other parameters from the user question that are not already covered by dependency_parameters
3. **Merge them together** - Combine dependency_parameters with newly extracted parameters
4. **Generate post-processing code** - Create the corresponding Python post-processing function.

You must return a valid JSON object in the exact format shown below. Do NOT wrap the result in markdown code blocks or backticks.

##Output Format (Must Follow Exactly)
```json
{{
    "tool_name": "run_pagerank",
    "parameters": {{
        "alpha": 0.85,
        "max_iter": 100,
        "personalization": {{"20": 1.0}}
    }},
    "post_processing_code": {{
        "code": "def process(data):\\n    top_10 = dict(list(sorted(data.items(), key=lambda x: x[1], reverse=True))[:10])\\n    return {{\\n "original_result":data,\\n "top_10_nodes":top_10,\\n }}",
        "is_calculate": true,
        "output_schema": {{
            "description": "PageRank post-processing results, including top 10 nodes",
            "type": "dict",
            "fields": {{
                "original_result": {{
                    "type": "data_type",
                    "field_description": "Original output result of graph algorithm"
                }},
                "top_10_nodes": {{
                    "type": "dict",
                    "field_description": "Top 10 nodes by influence and their corresponding PageRank scores"
                }}
            }}
        }}
    }},
    "reasoning": "The dependency_parameters provided personalization={{'20': 1.0}} from parent step. I extracted alpha=0.85 and max_iter=100 from the user question. The post_processing_code is to xxx"
}}
```

## Core Rules (Must Be Strictly Followed)

###Rule 1: Fixed Tool
-The given tool_description is fixed; you must not choose or invent any other tool.
-The tool_name in your output must exactly match the one defined in the tool_description.

### Rule 2: Parameter Merging Rules (Critical!)
#### Handling Dependency Parameters
- **Dependency parameters are pre-determined** - They come from parent step results and have been mapped to the current algorithm's input parameters
- **You MUST include all dependency_parameters** in your output `parameters` object
- **Do NOT modify dependency_parameters** - Use them as-is unless the user question explicitly requests a change
- **If a dependency parameter conflicts with a user-specified parameter**, prioritize the user's explicit request (but mention this in reasoning)

#### Parameter Extraction Rules
#### Special Handling of G Parameter
- You will see the `G` parameter in the tool's input_schema
- **Never** include the `G` key in `parameters`
- The `G` parameter will be automatically injected by the system in the background
- Even if `G` is in the `required` list, do not extract it

#### Extraction Rules for Other Parameters
- **Extract all question-related parameters** that are NOT already in dependency_parameters
- **When extracting parameters, the input parameter type must be consistent with the type defined in the schema**
- **Ensure all keys in `parameters` strictly correspond to the tool's input_schema**
- **If the parameter type is enumerated format, ensure the extracted value is within the allowed range**

#### Format for Dictionary Type Parameters
- If the parameter type is `dict`, `dictionary`, `list`, or other defined recognizable types, must return **pure JSON object**
- **Wrong Example**:
  {{
    "personalization": "{{'20': 1.0}}"  // Wrong:String
  }}
- **Correct Example**:
  {{
    "personalization": {{"20": 1.0}}   // Correct:Dictionary object
    "personalization": null            // Correct:Use null for default values
  }}

###Rule 3: Post-Processing Code Generation Guidelines
Generate Python code to process graph algorithm results based on user questions. The function signature is always def process(data):.
*All post-processing code return values must be in dictionary format, and each field must include metadata:
```python
def process(data):
    # Processing logic...
    return {{
        "original_result":data  
        "field_name1": actual_data1,
        "field_name2": actual_data2
    }}
```
CRITICAL: Analyze Input Data Structure Before Writing Code**: The input parameter `data` comes from the upstream graph algorithm's output. **Do not assume it is always a dictionary.** Follow these steps:
**Step 1:** Check the **Output Structure** section in the `tool_description` to determine the actual data type (e.g., Dict[node_id, score], List[List[node_id]], or List[PathObject]). 
**Step 2:** Based on the Output Structure, write code that correctly handles the actual data type.

When writing the post-processing code, follow these guidelines:
1. Access to Global Graph Data
The code has access to **global_graph** (instance of GraphData) for property retrieval and topology analysis.
**vetrix_schema** provide vertex property names and **edge_schema** provide edge property names.
You must choose properties based on these schemas.
Available API Methods:
- global_graph.has_vertex(vid: str): Check if a vertex exists.
- global_graph.get_vertex_property(vid: str, prop: str): Get specific property.
- global_graph.get_edge_property(src: str, dst: str, prop: str): Get edge weight/property.
- global_graph.get_edges_by_vertices(vertex_ids: set[str]) -> List[EdgeData]: Get all edges connected to these nodes (internal + external).
- global_graph.get_src_dst_by_vertices(vertex_ids: set[str]) -> List[EdgeData]: Get internal edges only (where both src and dst are in the set). Crucial for community analysis.
- global_graph.get_vertices_by_edges(edges: List[EdgeData]) -> List[VertexData]: Extract unique vertices from edge list.

```python
class VertexData:
    vid: str
    properties: Dict[str, Any] = field(default_factory=dict)
class EdgeData:
    src: str
    dst: str
    rank: Optional[int] = None  # Optional rank values
    properties: Dict[str, Any] = field(default_factory=dict)
```
The structure of VertexData and EdgeData has been provided. Please use the correct indexing method.
transaction_sum = sum(edge['base_amt'] for edge in transactions)
Error: The 'edge' here is an EdgeData object, not a dictionary, so you cannot use the ['base_amt'] indexing method.
transaction_sum = sum(edge.base_amt for edge in transactions)
Error:EdgeData does not have a field called base_amt; it only has properties: Dict[str, Any]. Therefore, edge.base_amt will trigger the error: 'EdgeData' object has no attribute 'base_amt'.
Correct:transaction_sum = sum(edge.properties.get("base_amt", 0.0) for edge in transactions)

2. Best Practices & Safety Rules
Node ID Types: Algorithm outputs are often strings. Always use int(node_id) for numeric comparisons (e.g., if int(vid) < 100).
Sorting: reverse=True for "Top/Most", reverse=False for "Bottom/Least".
Multi-Step Logic: Calculate intermediate subsets first (e.g., top_10) and perform subsequent stats (min/max/avg) on that subset, not the original data.
Type Safety:
Convert dict keys to sets before graph queries.
Check for None when accessing properties.
Handle empty filter results gracefully.

3. Return Value Requirements
All returns must be a dictionary containing the original result and metadata-rich fields.
original_result: Always include the raw data.
Calculated Fields: Must be structured as a triple inside the output schema (Type, Description).
is_calculate Flag:
true: If Performed numerical calculations such as: sorting, filtering, aggregation, or property enrichment occurred
false: If only wrapping the original data.

**Field Naming and Description Clarity (Critical!)**:
- **Field names must be semantically clear and descriptive**: Use specific, meaningful names that clearly indicate what the field contains. Avoid generic names like "result", "data", "value", "output".
  - Bad: `"result"`, `"data"`, `"value"`, `"output"`, `"nodes"`
  - Good: `"top_10_nodes"`, `"average_age"`, `"neighbor_count"`, `"qualified_nodes"`
- **Avoid using "target" in field names**: If the question explicitly mentions a specific user/node (e.g., "Sullivan Breanna", "node '45'"), include that identifier in the field name instead of using "target".
  - Bad: `"target_node_community"`, `"target_community"`, `"target_neighbors"`
  - Good: `"node_45_community"`, `"sullivan_breanna_community"`, `"neighbors_of_45"`, `"sullivan_breanna_neighbors"`
  - Example: If the question asks "Find neighbors of node '45' in its community", use `"node_45_community"` and `"neighbors_of_45"` instead of `"target_node_community"` and `"target_neighbors"`.
- **field_description must be detailed and precise**: The description should clearly explain:
  - What the field contains (data type and structure)
  - What it represents (semantic meaning)
  - How it was calculated or derived (if applicable)
  - Bad: `"Top nodes"`, `"Result data"`, `"The nodes"`
  - Good: `"Top 10 nodes with highest PageRank scores, mapping node_id to score"`, `"Average age of the top 5 nodes, calculated using the 'age' property from vertexData schema"`
- **Field names should reflect the user question**: If the question asks for "top 5 nodes", use `"top_5_nodes"` not `"nodes"`. If asking for "average age", use `"average_age"` not `"avg"` or `"result"`. If the question mentions a specific user/node, incorporate that identifier into the field name.

4. **Vertex Existence Validation: If the question explicitly mentions a specific node ID or name (e.g., "Sullivan Breanna", "node '45'"), you must first validate that this node exists in the graph before proceeding. Extract the node ID from the question, then call `global_graph.has_vertex(vid: str)` to check if it exists. If it does not exist, return an error message: "Target node not found in the graph". If the node exists, proceed with the rest of the code logic.**
```python
def process(data):
    # Example: Question mentions "Sullivan Breanna"
    target_node_id = "Sullivan Breanna"  # Extract from question
    if not global_graph.has_vertex(target_node_id):
        return {{"error": "Target node 'Sullivan Breanna' not found in the graph"}}
    # rest of the code logic
```

5. If an error occurs at any stage during function execution, directly return an error: "Error occurred during processing: {{error_message}}".
```python
def process(data):
    try:
        # code
    except Exception as e:
        return {{"error": f"Error occurred during processing: {{e}}"}}
```

Code Patterns & Examples
Example 1: Handling Nested Lists (Community Detection) Context: data is a list of communities [['1','2'], ['3','4']]. Requirement: "Find neighbors of node '45' in its community."
```python
def process(data):
    node_45 = "45"
    if not global_graph.has_vertex(node_45):
      return {{"error": "Node '45' not found in the graph"}}
    # 1. Correctly find the community containing node 45 in a Nested List structure
    node_45_community = None
    # Do NOT use data[node_45] -> data is a list, not a dict mapping
    for comm in data:
        if node_45 in comm:
            node_45_community = comm
            break
    if not node_45_community:
        return {{"error": "Node '45' not found in any community"}}
    # 2. Get internal edges within this community
    # Convert list to set for the API call
    community_nodes_set = set(node_45_community) 
    internal_edges = global_graph.get_src_dst_by_vertices(community_nodes_set)
    # 3. Filter for direct neighbors of node_45
    neighbors = set()
    for edge in internal_edges:
        if edge.src == node_45:
            neighbors.add(edge.dst)
        elif edge.dst == node_45:
            neighbors.add(edge.src)
    return {{
        "original_result": data,
        "node_45_community": node_45_community,
        "neighbors_of_45": list(neighbors),
        "neighbor_count": len(neighbors)
    }}
```
Example 2: Dictionary Processing (PageRank) & Property Enrichment Context: data is {{node_id: score}}. Requirement: "Top 5 nodes sorted by age."
```python
def process(data):
    # 1. Get Top 5 (Dict processing)
    top_5 = dict(list(sorted(data.items(), key=lambda x: x[1], reverse=True))[:5])
    # 2. Enrich with Property
    enriched = []
    for vid, score in top_5.items():
        age = global_graph.get_vertex_property(vid, "age")
        if age is not None:
            enriched.append({{"vid": vid, "score": score, "age": age}})
    # 3. Sort by Property
    sorted_by_age = sorted(enriched, key=lambda x: x["age"]) # ascending
    return {{
        "original_result": data,
        "top_5_raw": top_5,
        "result_sorted_by_age": sorted_by_age
    }}
```
Example 3: Aggregation & Stats (Handling Empty Results) Requirement: "Count nodes with score > 0.1, and calculate their mean and max."
```python
import statistics
def process(data):
    filtered = {{k: v for k, v in data.items() if v > 0.1}}
    if not filtered:
        return {{
            "original_result": data,
            "count": 0, 
            "average": 0, 
            "max_val": 0
        }}
    return {{
        "original_result": data,
        "count": len(filtered),
        "average": statistics.mean(filtered.values()),
        "max_value": max(filtered.values()),
        "max_node": max(filtered, key=filtered.get),
        "qualified_nodes": filtered
    }}
```
Example 4: Multi-field Return  Requirement: "Get the top 10 nodes by PageRank score, calculate their total score sum, and identify the fifth-ranked node."
```python
def process(data):
    top_10 = dict(list(sorted(data.items(), key=lambda x: x[1], reverse=True))[:10])
    total = sum(top_10.values())
    fifth_node = list(top_10.keys())[4]
    return {{
         "original_result":data,
        "top_10_nodes":top_10,
        "total_score":total,
        "fifth_node": {{"node": fifth_node, "score": top_10[fifth_node]}}
    }}
```
####Core Principle for Complex Processing
**Identify the processing chain:**
User requests often contain multiple linked operations (e.g., “average and minimum value among the top 10 nodes”).
**Key idea:** **Always save intermediate results explicitly**, and base subsequent steps on those — not the original data.
#####Wrong pattern (recalculating / wrong source)
```python
def process(data):
    total = sum(sorted(data.values(), reverse=True)[:10])  # Calculate top 10
    min_node = min(data, key=data.get)  #Wrong:Finds min from full data instead of top 10
    return {{
        'original': data,
        'total': total, 
        'min_node': min_node
    }}
```
######Correct pattern (derive step by step)
```python
def process(data):
    # Step 1: extract target subset
    top_10 = dict(list(sorted(data.items(), key=lambda x: x[1], reverse=True))[:10])
    
    # Step 2: all later operations use top_10
    min_node = min(top_10, key=top_10.get)
    
    return {{
        'original_result': data,
        'top_10_nodes': top_10,
        'total_score':  sum(top_10.values()),
        'min_in_top10':  {{'node': min_node, 'score': top_10[min_node]}}
    }}
```

Rule 4: Output JSON Format
Return the result in this strictly defined JSON structure:
```json
{{
    "tool_name": "<algorithm_name>",
    "parameters": {{
        "<param_name_1>": <param_value_1>,
        "<param_name_2>": <param_value_2>,
        ...
    }},
    "post_processing_code": {{
        "code": "<escaped_python_code_string_with_\\n_for_newlines>",
        "is_calculate": <true_or_false>,
        "output_schema": {{
            "description": "<description_of_output>",
            "type": "<dict_or_list>",
            "fields": {{
                "original_result": {{
                    "type": "data_type",  # or list, depending on actual type
                    "field_description": "Original output result of graph algorithm. This field is of type xxx, where the key represents the node ID and the value represents the PageRank score. And the key is string type and the value is float type."
                }},
                "<field_name_1>": {{
                    "type": "<field_type>",
                    "field_description": "<detailed_description_of_field>"
                }},
                ...
            }}
        }}
    }},
    "reasoning": "<explanation_of_parameter_selection_and_post_processing_logic>"
}}
```

Key Constraints:
Newlines: Escape newlines in code as \n.
Schema Match: Keys in output_schema.fields MUST match the keys returned in the process function dictionary.
No Raw Returns: Never write return data. Always wrap in {{ "original_result": data, ... }}.
Parameter Cleaning: Do not include G or backend_kwargs in the parameters object.

##Inputs:
### User Question
{question}
### tool_description
{tool_description}
### Dependency Parameters (Pre-determined from parent steps)
These parameters have been automatically extracted from parent step results and mapped to the current algorithm's input parameters. **You MUST include them in your output parameters object.**
```json
{dependency_parameters}
```
###vertexData schema
{vetrix_schema}
###edgeData schema
{edge_schema}

##Instruction:
1. **First, include all dependency_parameters in your output parameters object**
2. **Then, extract any additional parameters from the user question that are not in dependency_parameters**
3. **Merge them together** - The final parameters object should contain both dependency_parameters and newly extracted parameters
4. **Generate post-processing code** based on user questions
5. **Provide clear reasoning** explaining how you merged the parameters

Follow all rules and examples above strictly. Your output must be a valid JSON object only — no additional text, markdown, or explanation.
"""

generate_numeric_analysis_code_prompt = """
You are a professional Python data processing code generation expert.

Your task is to analyze a given natural-language **User Question** and generate a Python function to solve this problem. When generating this function, you need to use the upstream results **Dependency data items**. These results are computed from upstream questions and need to be passed as parameters to the function. **IMPORTANT**: 
- The function name must be `process`.
- The function parameter names must match the `field_key` of each item in **Dependency data items**, and each item should be passed as a function parameter, because solving the current question depends on these upstream result data.

## Input Format Example
### User Question
Find the top 5 nodes with highest PageRank scores and calculate their average age.
### Dependency data items
[
  {{
    "field_key": "node_pagerank",
    "field_type": "dict",
    "field_desc": "NodeID → PageRank value mapping",
    "value": {{
      "1": 0.0123,
      "2": 0.0234
    }}
  }}
]
### vertex_schema
{{
  "name": "str",
  "age": "int",
  "score": "float",
  "type": "str"
}}
### edge_schema
{{
  "weight": "float",
  "timestamp": "int",
  "type": "str"
}}

**Note**: 
- `dependency_data_items` is a list where each item contains `field_key`, `field_type`, `field_desc`, and `value` (sampled data, typically 1-2 examples).
- `vertex_schema` and `edge_schema` are dictionaries mapping property names to their Python type names (as strings).
- The `value` field in dependency items is sampled data (not the full dataset) used to understand the data structure.


You must return a valid JSON object in the exact format shown below. Do NOT wrap the result in markdown code blocks or backticks.
## Output Format (Must Follow Exactly)
```json
{{
    "numeric_analysis_code": 
    {{
        "code": "def process(node_pagerank):\\n    # node_pagerank: dict[node_id, score]\\n    top_5 = dict(list(sorted(node_pagerank.items(), key=lambda x: x[1], reverse=True))[:5])\\n    # Get ages for top 5 nodes using global_graph\\n    ages = []\\n    for node_id in top_5.keys():\\n        age = global_graph.get_vertex_property(node_id, 'age')\\n        if age is not None:\\n            ages.append(age)\\n    avg_age = sum(ages) / len(ages) if ages else None\\n    return {{\\n        \"top_5_nodes\": top_5,\\n        \"average_age\": avg_age\\n    }}",
        "output_schema": {{
           "description": "This code finds the top 5 nodes with highest PageRank scores and calculates their average age using the vertex 'age' property from the graph schema.",
            "type": "dict",
            "fields": {{
                "top_5_nodes": {{
                    "type": "dict",
                    "field_description": "Top 5 nodes with highest PageRank scores, mapping node_id to score"
                }},
                "average_age": {{
                    "type": "float",
                    "field_description": "Average age of the top 5 nodes, calculated using the 'age' property from vertexData schema"
                }}
            }}
        }}
    }},
    "reasoning": "The numeric analysis code logic is: extract top 5 nodes from PageRank results, then calculate their average age using the 'age' property from the graph schema."
}}
```

## Core Rules (Must Be Strictly Followed)
### Rule 1. Function Signature and Parameters
**CRITICAL**: 
- The function name must be `process`.
- **Dependency data items** represent the upstream results that the current question depends on. Each item contains a `field_key` that identifies the data field.
- **You MUST use each `field_key` as a function parameter name** (do NOT use a generic name like `data`), and **all items must be included as function parameters** because solving the current question depends on all these upstream results.
- The return values must be in **dictionary format**.

**Field Naming and Description Clarity (Critical!)**:
- **Field names must be semantically clear and descriptive**: Use specific, meaningful names that clearly indicate what the field contains. Avoid generic names like "result", "data", "value", "output".
  - Bad: `"result"`, `"data"`, `"value"`, `"output"`, `"nodes"`
  - Good: `"top_10_nodes"`, `"average_age"`, `"neighbor_count"`, `"qualified_nodes"`
- **Avoid using "target" in field names**: If the question explicitly mentions a specific user/node (e.g., "Sullivan Breanna", "node '45'"), include that identifier in the field name instead of using "target".
  - Bad: `"target_node_community"`, `"target_community"`, `"target_neighbors"`
  - Good: `"node_45_community"`, `"sullivan_breanna_community"`, `"neighbors_of_45"`, `"sullivan_breanna_neighbors"`
  - Example: If the question asks "Find neighbors of node '45' in its community", use `"node_45_community"` and `"neighbors_of_45"` instead of `"target_node_community"` and `"target_neighbors"`.
- **field_description must be detailed and precise**: The description should clearly explain:
  - What the field contains (data type and structure)
  - What it represents (semantic meaning)
  - How it was calculated or derived (if applicable)
  - Bad: `"Top nodes"`, `"Result data"`, `"The nodes"`
  - Good: `"Top 10 nodes with highest PageRank scores, mapping node_id to score"`, `"Average age of the top 5 nodes, calculated using the 'age' property from vertexData schema"`
- **Field names should reflect the user question**: If the question asks for "top 5 nodes", use `"top_5_nodes"` not `"nodes"`. If asking for "average age", use `"average_age"` not `"avg"` or `"result"`. If the question mentions a specific user/node, incorporate that identifier into the field name.

**Example**: If Dependency data items contains:
```json
[
  {{"field_key": "node_pagerank", ...}},
  {{"field_key": "community_nodes", ...}}
]
```
Then the function signature must be:
```python
def process(node_pagerank, community_nodes):
    # Processing logic using node_pagerank and community_nodes...
    return {{ 
        "field_name1": actual_data1,
        "field_name2": actual_data2
    }}
```

### Rule 2. **Access to Graph Data**
CRITICAL: You have access to a helper object **global_graph** for querying graph data. Prefer using these API methods over manual iteration.
**Analyze Input Data Structure Before Writing Code**: Before writing code, infer the structure of each dependency parameter (e.g., Dict[node_id, score], List[List[node_id]], or List[PathObject]) from the `field_type`, `field_desc` and `value` in Dependency data items.

When writing the numeric analysis code, follow these guidelines:
1. Access to Global Graph Data
The code has access to **global_graph** (instance of GraphData) for property retrieval and topology analysis.
**vertex_schema** provide vertex property names and **edge_schema** provide edge property names.
You must choose properties based on these schemas.
Available API Methods:
- global_graph.has_vertex(vid: str): Check if a vertex exists.
- global_graph.get_vertex_property(vid: str, prop: str): Get specific property.
- global_graph.get_edge_property(src: str, dst: str, prop: str): Get edge weight/property.
- global_graph.get_edges_by_vertices(vertex_ids: set[str]) -> List[EdgeData]: Get all edges connected to these nodes (internal + external).
- global_graph.get_src_dst_by_vertices(vertex_ids: set[str]) -> List[EdgeData]: Get internal edges only (where both src and dst are in the set). Crucial for community analysis.
- global_graph.get_vertices_by_edges(edges: List[EdgeData]) -> List[VertexData]: Extract unique vertices from edge list.

```python
class VertexData:
    vid: str
    properties: Dict[str, Any] = field(default_factory=dict)
class EdgeData:
    src: str
    dst: str
    rank: Optional[int] = None  # Optional rank values
    properties: Dict[str, Any] = field(default_factory=dict)
```
The structure of VertexData and EdgeData has been provided. Please use the correct indexing method.
transaction_sum = sum(edge['base_amt'] for edge in transactions)
Error: The 'edge' here is an EdgeData object, not a dictionary, so you cannot use the ['base_amt'] indexing method.
transaction_sum = sum(edge.base_amt for edge in transactions)
Error:EdgeData does not have a field called base_amt; it only has properties: Dict[str, Any]. Therefore, edge.base_amt will trigger the error: 'EdgeData' object has no attribute 'base_amt'.
Correct:transaction_sum = sum(edge.properties.get("base_amt", 0.0) for edge in transactions)

2. Best Practices & Safety Rules
- **Node ID Types**: The type of Node IDs are strings. Always use `int(node_id)` for numeric comparisons (e.g., `if int(vid) < 100`).
- **Sorting**: `reverse=True` for "Top/Most", `reverse=False` for "Bottom/Least".
- **Multi-Step Logic**: Calculate intermediate subsets first (e.g., `top_10`) and perform subsequent stats (min/max/avg) on that subset, not the original dependency data.
- **Type Safety**:
  - Convert dict keys to sets before graph queries.
  - Check for `None` when accessing properties.
  - Handle empty filter results gracefully.
- **Vertex Existence Validation**: If the question explicitly mentions a specific node ID or name (e.g., "Sullivan Breanna", "node '45'"), you must first validate that this node exists in the graph before proceeding. Extract the node ID from the question, then call `global_graph.has_vertex(vid: str)` to check if it exists. If it does not exist, return an error message: "Target node not found in the graph".
- **Error Handling**: If an error occurs at any stage during function execution, directly return an error: "Error occurred during processing: {{error_message}}".

**Example Usage (Graph Structure Computing)**
Example 1: Dictionary Processing (PageRank) & Property Enrichment  
Context: Dependency data item `node_pagerank` is `{{node_id: score}}`. Question: "Top 5 nodes sorted by age."
```python
def process(node_pagerank):
    # 1. Get Top 5 (Dict processing)
    top_5 = dict(list(sorted(node_pagerank.items(), key=lambda x: x[1], reverse=True))[:5])
    # 2. Enrich with Property
    enriched = []
    for vid, score in top_5.items():
        age = global_graph.get_vertex_property(vid, "age")
        if age is not None:
            enriched.append({{"vid": vid, "score": score, "age": age}})
    # 3. Sort by Property
    sorted_by_age = sorted(enriched, key=lambda x: x["age"])  # ascending
    return {{
        "top_5_raw": top_5,
        "result_sorted_by_age": sorted_by_age
    }}
```

Example 2: Handling Nested Lists (Community Detection)  
Context: Dependency data item `communities` is a list of communities `[['1','2'], ['3','4']]`. Question: "Find neighbors of node '45' in its community."
```python
def process(communities):
    node_45 = "45"
    if not global_graph.has_vertex(node_45):
      return {{"error": "Node '45' not found in the graph"}}
    # 1. Correctly find the community containing node 45 in a Nested List structure
    node_45_community = None
    # Do NOT use communities[node_45] -> communities is a list, not a dict mapping
    for comm in communities:
        if node_45 in comm:
            node_45_community = comm
            break
    if not node_45_community:
        return {{"error": "Node '45' not found in any community"}}
    # 2. Get internal edges within this community
    # Convert list to set for the API call
    community_nodes_set = set(node_45_community) 
    internal_edges = global_graph.get_src_dst_by_vertices(community_nodes_set)
    # 3. Filter for direct neighbors of node_45
    neighbors = set()
    for edge in internal_edges:
        if edge.src == node_45:
            neighbors.add(edge.dst)
        elif edge.dst == node_45:
            neighbors.add(edge.src)
    return {{
        "node_45_community": node_45_community,
        "neighbors_of_45": list(neighbors),
        "neighbor_count": len(neighbors)
    }}
```

**Standard Examples (Non-Graph Structure Computing)**
Example 1: Single Value Return
```python
def process(node_pagerank):
    max_node = max(node_pagerank, key=node_pagerank.get)
    return {{
        "max_influence_node": {{"node": max_node, "score": node_pagerank[max_node]}}
    }}
```

Example 2: Complex Statistics
```python
import statistics
def process(node_pagerank):
    filtered = {{k: v for k, v in node_pagerank.items() if v > 0.01}}
    return {{
        "high_influence_nodes": filtered,
        "count": len(filtered),
        "average_score": statistics.mean(filtered.values()) if filtered else 0
    }}
```

Example 3: Aggregation & Stats (Handling Empty Results)  
Requirement: "Count nodes with score > 0.1, and calculate their mean and max."
```python
import statistics
def process(node_pagerank):
    filtered = {{k: v for k, v in node_pagerank.items() if v > 0.1}}
    if not filtered:
        return {{
            "count": 0, 
            "average": 0, 
            "max_val": 0
        }}
    return {{
        "count": len(filtered),
        "average": statistics.mean(filtered.values()),
        "max_value": max(filtered.values()),
        "max_node": max(filtered, key=filtered.get),
        "qualified_nodes": filtered
    }}
```

**Important Notes**:
- Always check if property value is `None` before using it
- Node IDs in algorithm results are strings (e.g., "1", "30")
- Use these methods to enrich algorithm results with vertex/edge properties

Example 4: Vertex Existence Validation
```python
def process(node_pagerank):
    # Example: Question mentions "Sullivan Breanna"
    target_node_id = "Sullivan Breanna"  # Extract from question
    if not global_graph.has_vertex(target_node_id):
        return {{"error": "Target node 'Sullivan Breanna' not found in the graph"}}
    # rest of the code logic
```

Example 5: Error Handling
```python
def process(node_pagerank):
    try:
        # code
    except Exception as e:
        return {{"error": f"Error occurred during processing: {{e}}"}}
```

### Rule 3. Node ID Type Handling
Node IDs  are typically strings ("1", "30"), and must be converted before numerical comparison:
```python
# Right:
filtered = {{k: v for k, v in node_pagerank.items() if 1 <= int(k) <= 10}}
# Wrong:
filtered = {{k: v for k, v in node_pagerank.items() if 1 <= k <= 10}}  # String comparison logic error
```

### Rule 4. Core Principles for Complex Processing
Identify processing chains: User questions often include multiple related operations (e.g., "average and minimum of top 10 nodes")  
Key: Intermediate results must be explicitly saved, and subsequent operations are based on intermediate results, not the original dependency data.
**Wrong Pattern** (Repeated calculation/operating on wrong object):
```python
def process(node_pagerank):
    total = sum(sorted(node_pagerank.values(), reverse=True)[:10])  # Calculate Top 10
    min_node = min(node_pagerank, key=node_pagerank.get)  # Wrong: Finding minimum from all data (should find from Top 10)
    return {{'total': total, 'min_node': min_node}}
```
**Correct Pattern** (Extract subset first, then operate on subset):
```python
def process(node_pagerank):
    # Step 1: Extract target subset
    top_10 = dict(sorted(node_pagerank.items(), key=lambda x: x[1], reverse=True)[:10])
    # Step 2: All subsequent operations are based on top_10
    min_node = min(top_10, key=top_10.get)
    return {{
        'top_10_nodes': top_10,
        'total_score': sum(top_10.values()),
        'min_in_top10': {{"node": min_node, "score": top_10[min_node]}}
    }}
```

## Output Format (Must Follow Exactly)
```json
{{
    "numeric_analysis_code": {{
        "code": "def process(node_pagerank):\\n    top_10 = dict(list(sorted(node_pagerank.items(), key=lambda x: x[1], reverse=True))[:10])\\n    return {{\\n        \"top_10_nodes\": top_10,\\n        \"total_score\": sum(top_10.values())\\n    }}",
        "output_schema": {{
            "description": "Top 10 nodes by PageRank score and their total score",
            "type": "dict",
            "fields": {{
                "top_10_nodes": {{"type": "dict", "field_description": "Top 10 nodes and their PageRank scores"}},
                "total_score": {{"type": "float", "field_description": "Sum of top 10 PageRank scores"}}
            }}
        }}
    }},
    "reasoning": "Extract top 10 nodes from PageRank results and calculate total score"
}}
```
Output Format key Requirements:
- Newlines in code must be escaped as \\n
- The function name must be `process`

##Inputs:
### User Question
{question}
### Dependency data items
{dependency_data_items}
### vertex_schema
{vertex_schema}
### edge_schema
{edge_schema}

##Instructions:
Generate a Python function to solve the user question based on the dependency data items, vertexData schema, and edgeData schema.
"""


revise_subquery_plan_prompt = """
You are a graph workflow planning expert skilled at revising DAG-based subquery plans
based on a user's modification request. You will be given:
1. A current subquery plan in JSON format.
2. A natural language user request describing edits to the plan.

### Input Description
The current subquery plan consists of nodes, each containing:
- id: the unique node id (example: "q1")
- query: a natural language description of the step
- depends_on: a list of prerequisite node ids

The user request may include one or more of the following types of edits:
- Add a new step (node insertion)
- Delete a step (node removal)
- Modify an existing step's natural language query
- Change dependencies (parent or child relations)
- Express no required modification

### Revision Rules
1. Preserve original ids when modifying content.
2. Assign new ids only when new steps are added.
3. All dependencies must remain acyclic and logically correct.
4. Each node must contain exactly: id, query, depends_on.
5. If the user explicitly indicates no change, output the original plan unchanged.
6. The output must strictly follow the schema below without any explanation:
{{
    "subqueries": [
    {{
        "id": "qX",
        "query": "text",
        "depends_on": []
    }}
    ]
}}

### Example for Guidance
**Input Current Plan:**
{{
"subqueries": [
    {{"id": "q1", "query": "Check if user Anna is a high-risk user.", "depends_on": []}},
    {{"id": "q2", "query": "List all potential money laundering pathways around Anna.", "depends_on": ["q1"]}},
    {{"id": "q3", "query": "Estimate the amount of cash that may have been illegally transferred out in relation to Anna.", "depends_on": ["q2"]}},
    {{"id": "q4", "query": "Find the account with the largest transaction amount in the suspicious paths.", "depends_on": ["q2"]}}
]
}}

**User Request:**
Add a step between node 1 and node 2 to identify Anna's fraud community. Modify node 4 so it becomes: identify the account with the largest transaction amount within Anna’s community.

**Correct Output:**
{{
"subqueries": [
    {{"id": "q1", "query": "Check if user Anna is a high-risk user.", "depends_on": []}},
    {{"id": "q2", "query": "Identify the potential fraud community in which Anna resides to narrow the scope of subsequent risk monitoring.", "depends_on": ["q1"]}},
    {{"id": "q3", "query": "List all potential money laundering pathways within the high-risk community where Anna is located.", "depends_on": ["q2"]}},
    {{"id": "q4", "query": "Estimate the amount of cash that may have been illegally transferred out in relation to Anna.", "depends_on": ["q3"]}},
    {{"id": "q5", "query": "Identify the account with the largest transaction amount within Anna's fraud community.", "depends_on": ["q2"]}}
]
}}

### Now Execute
Based on the user instructions below, revise the current plan. Output must be valid JSON only:
Current Plan: {current_plan}
User Request: {user_request}
"""

# add gjq
# Prompts for natural language query engine parameter extraction

# add gjq - Node lookup extractor prompt
nl_query_node_lookup_prompt = """{schema_info}

## Query Type: node_lookup

### Function Description
Find a single node by unique key, or filter multiple nodes by property conditions.

### Parameter Description
**Required Parameters**:
- `label`: Node label (select from Schema)

**Optional Parameters (choose one of two modes)**:
1. **Single Node Mode**:
   - `key`: Property key (usually node_key)
   - `value`: Property value (e.g., person name)
   - `return_fields`: List of fields to return (optional)

2. **Multi-Node Filter Mode**:
   - `conditions`: Property condition dictionary (e.g., {{"country": "US", "state": "VT"}})
   - `return_fields`: List of fields to return (optional)

### Key Rules

1. **Mode Determination**:
   - If user provides unique identifier (e.g., person name, ID) → Use single node mode (key + value)
   - If user provides filter conditions (e.g., country, state) → Use multi-node filter mode (conditions)

2. **return_fields Extraction**:
   - When user says "return...fields" or lists specific fields, **must** extract return_fields
   - Field names must exactly match property names in Schema
   - Common mappings: account number→acct_id, account status→acct_stat, account open date→acct_open_date

3. **Range Condition Format**:
   - Simple equality: {{"property": value}}
   - Range condition: {{"property": [operator, value]}}
   - Operators: ">", "<", ">=", "<=", "!=", "IN", "CONTAINS", "STARTS WITH"
   - Example: {{"initial_deposit": [">", 500000]}}

4. **Boolean Format**:
   - Use lowercase in JSON: true / false (not True / False)

### Error Examples

❌ **Error 1**: Treating filter query as single node lookup
```json
// Question: "Query customers living in US with state VT"
// Wrong:
{{
  "params": {{
    "label": "Account",
    "key": "node_key",
    "value": "US VT"  // ❌ Wrong! This is not a unique identifier
  }}
}}

// Correct:
{{
  "params": {{
    "label": "Account",
    "conditions": {{"country": "US", "state": "VT"}}  // ✅ Use conditions
  }}
}}
```

❌ **Error 2**: Ignoring user-specified return fields
```json
// Question: "Query Collins Steven's account, return acct_id and acct_stat"
// Wrong:
{{
  "params": {{
    "label": "Account",
    "key": "node_key",
    "value": "Collins Steven"
    // ❌ Missing return_fields
  }}
}}

// Correct:
{{
  "params": {{
    "label": "Account",
    "key": "node_key",
    "value": "Collins Steven",
    "return_fields": ["acct_id", "acct_stat"]  // ✅ Add return_fields
  }}
}}
```

### User Question
{question}

### Task
Based on the above rules, extract parameters from the user question and return in JSON format:
{{
  "params": {{...}},
  "modifiers": {{...}}
}}
"""

# add gjq - Relationship filter extractor prompt
nl_query_relationship_filter_prompt = """{schema_info}

## Query Type: relationship_filter

### Function Description
Filter relationships based on relationship property conditions, return relationships and related node information that meet the conditions.

### Parameter Description
**Required Parameters**:
- `rel_type`: Relationship type (select from Schema)

**Optional Parameters**:
- `start_label`: Start node label
- `end_label`: End node label
- `rel_conditions`: Relationship property condition dictionary
- `return_fields`: List of fields to return
- `aggregate`: Aggregation function (COUNT/SUM/AVG, etc.)

### Key Rules

1. **rel_conditions Format**:
   - Simple equality: {{"property": value}}
   - Range condition: {{"property": [operator, value]}}
   - Operators: ">", "<", ">=", "<=", "!=", "IN", "CONTAINS", "STARTS WITH"
   - Example: {{"base_amt": [">", 400]}}

2. **Date Condition Handling**:
   - Use STARTS WITH operator for date matching
   - Example: {{"tran_timestamp": ["STARTS WITH", "2025-05-01"]}}

3. **Schema Validation Rules**:
   - **rel_conditions**: Can only contain relationship properties (e.g., tran_id, base_amt)
   - **where modifier**: Used for node properties (e.g., from.acct_id, to.acct_id)
   - Don't confuse relationship properties with node properties!

4. **return_fields Format**:
   - Relationship properties: Use property name directly (e.g., "tran_id", "base_amt")
   - Start node properties: Use "from." prefix (e.g., "from.acct_id")
   - End node properties: Use "to." prefix (e.g., "to.acct_id")

### Error Examples

❌ **Error 1**: Putting node properties in rel_conditions
```json
// Question: "Transactions with orig_acct 1111"
// Wrong:
{{
  "params": {{
    "rel_type": "TRANSFER",
    "rel_conditions": {{"orig_acct": 1111}}  // ❌ orig_acct is not a relationship property!
  }}
}}

// Correct:
{{
  "params": {{
    "rel_type": "TRANSFER",
    "start_label": "Account",
    "end_label": "Account"
  }},
  "modifiers": {{
    "where": "from.acct_id = 1111"  // ✅ Use where modifier
  }}
}}
```

❌ **Error 2**: Adding conditions not mentioned by user
```json
// Question: "Transaction type is TRANSFER"
// Wrong:
{{
  "params": {{
    "rel_type": "TRANSFER",
    "rel_conditions": {{"base_amt": [">", 400]}}  // ❌ User didn't mention amount condition!
  }}
}}

// Correct:
{{
  "params": {{
    "rel_type": "TRANSFER",
    "start_label": "Account",
    "end_label": "Account"
  }}
}}
```

### User Question
{question}

### Task
Based on the above rules, extract parameters from the user question and return in JSON format:
{{
  "params": {{...}},
  "modifiers": {{...}}
}}
"""

# add gjq
# Prompt for classifying query type in natural language query engine
nl_query_classify_type_prompt = """You are a Neo4j query assistant.
User question: {question}
Available query types and descriptions:
{query_templates}
⚠️ Key: Template Selection Guide (Must Read)

1. **node_lookup**: Find/filter nodes
   - Single node exact lookup: Provide unique identifier (e.g., person name, ID)
   - Multiple node conditional filtering: Provide property conditions (e.g., country=US, state=VT)
   - Keywords: find, query, return...customer/account
   - Returns: Node data or specified fields
   - ⚠️ Does not include aggregation functions (COUNT/SUM/AVG, etc.)

2. **relationship_filter**: Relationship property filtering (⭐ Focus: Returns raw data, not aggregation)
   - Filter relationships based on relationship properties (e.g., amount>400)
   - **Returns list of relationships that meet conditions** (raw data)
   - Keywords: find, list, return, query...transactions
   - Returns: Relationship data + related node information (e.g., transaction ID, amount, sender/receiver accounts)
   - ⚠️ **Judgment criteria**:
     * If question requires returning **specific transaction records/list** → relationship_filter
     * If question requires returning **statistical values** (count/total/average, etc.) → aggregation_query
   - Examples:
     * ✅ "Find transactions with amount greater than 400, return transaction ID and amount" → relationship_filter (returns list)
     * ❌ "Count transactions with amount greater than 400" → aggregation_query (returns value)

3. **aggregation_query**: Aggregation statistical query (⭐ Focus: **Only for GROUP BY aggregation**)
   - **⚠️ Key**: aggregation_query **only for grouped aggregation** (GROUP BY), not for global aggregation!
   - **Must include grouping**: `group_by_node` or `group_by_property`
   
   **⚠️ Conditions for aggregation_query (any one satisfied)**:
   
   **1️⃣ Return result format is 「value / table」**:
      - Count / times / total / average / maximum / minimum
      - Top N / sort / ranking
      - Y for each X (e.g., "transaction count for each account")
      - Return format: table/list (e.g., "Account A: 100 times, Account B: 80 times")
   
   **2️⃣ Question keywords matched**:
      - **Grouping keywords**: **"each", "every", "per", "by...group"**
      - **Aggregation keywords**: count | calculate | number | times | total | sum | average | maximum | minimum
      - **Ranking keywords**: Top | top N | sort | rank | percentage
      - **Existence query**: whether exists (based on count / exists)
   
   **3️⃣ Clearly 「global scan + grouping」**:
      - "each account / each branch / each status"
      - **No specific node ID given** (if there's a specific name/ID, should be neighbor_query + aggregate modifier)
   
   **⚠️ Judgment criteria (key)**:
     * ✅ If question contains "each", "every", "per" → aggregation_query (grouped aggregation)
     * ✅ If question requires "Top N", "ranking", "sort" + grouping → aggregation_query
     * ❌ If question contains specific name/ID + aggregation keywords → neighbor_query + aggregate modifier
     * ❌ If question is just "count...number", "calculate...total" (no "each") → relationship_filter + aggregate
   
   **Examples**:
     * ✅ "Count transaction times for each account" → aggregation_query (has "each")
     * ✅ "Calculate total outgoing transaction amount for each account" → aggregation_query (has "each")
     * ✅ "Return top 5 accounts with most transactions" → aggregation_query (Top N + grouping)
     * ✅ "Count number of accounts under each branch_id" → aggregation_query (each + grouping)
     * ❌ "Count transactions initiated by Collins Steven" → neighbor_query + aggregate (has specific name)
     * ❌ "Count total amount of TRANSFER type transactions" → relationship_filter + aggregate (no "each")
     * ❌ "Count transactions on 2025-05-01" → relationship_filter + aggregate (no "each")
     * ❌ "Count total amount of transactions with base_amt > 300" → relationship_filter + aggregate (no "each")
   
   **⚠️ Correct approach for global aggregation (no grouping)**:
   - "Count transactions initiated by Collins Steven" → neighbor_query + aggregate modifier
   - "Count transactions on 2025-05-01" → relationship_filter + aggregate
   - "Count total amount of transactions with base_amt > 300" → relationship_filter + aggregate + rel_conditions

4. **neighbor_query**: Neighbor query (⭐ Key distinction)
   - Query **specific node's** neighbors or N-hop relationships
   - ⚠️ **Key**: Question **must have clear starting node** (e.g., person name, ID, specific account)
   - Can return detailed information for each edge (hops=1 + return_fields)
   - Keywords:
     * Clear starting point: **someone's** neighbors, **some account's** counterparties, **Collins Steven's** transactions
     * Relationship description: directly transacted with, as sender account
   - Returns: Neighbor nodes + relationship information
   - Examples:
     * ✅ "Query all counterparty accounts that Steven Collins directly transacted with" → neighbor_query (has starting point: Steven Collins)
     * ✅ "Query all transactions where Collins Steven is the sender account" → neighbor_query (has starting point: Collins Steven)
     * ❌ "Query all TRANSFER type transactions" → relationship_filter (no clear starting point)
     * ❌ "Query transaction paths on 2025-05-01" → relationship_filter (no clear starting point, only time condition)
   - ⚠️ **Judgment criteria**: If question doesn't have specific name, ID or account identifier, it's not neighbor_query!

5. **path_query**: Path query
   - Query paths between two nodes
   - Keywords: path, how to reach, from A to B
   - Returns: Path information

6. **common_neighbor**: Common neighbors (supports relationship property filtering + aggregation ranking)
   - Query common neighbors of two **specific nodes**
   - ⚠️ Important: Both nodes must be specific node identifiers (e.g., person names, IDs), not conditions
   - Keywords: common, mutual, both...of, both people, simultaneously
   - Returns: Common neighbor nodes (optional: + properties of both relationships + aggregation statistics)
   - Supports three modes:
     * **Simple mode**: No relationship property filtering, e.g., "common neighbors of Collins Steven and Nunez Mitchell"
     * **Filter mode**: With relationship property filtering, e.g., "find common transaction neighbors of Steven Collins and Samantha Cook where transaction amounts are both greater than 400"
     * **Aggregation ranking mode**: Count transaction times of common neighbors and rank, e.g., "find recipients that both Steven Collins and Samantha Cook transferred to, give Top-5 ranking by common transfer count"
   - ⚠️ **Judgment criteria**:
     * If question contains "both people", "simultaneously", "common" + two specific names → common_neighbor
     * If question requires "rank by...", "Top-N" → common_neighbor + aggregate modifier
   - ❌ Wrong example: "both transacted with A and accounts with prior_sar_count=true" → This is not common_neighbor!
   - ✅ Correct examples:
     * "Common neighbors of Collins Steven and Nunez Mitchell" → common_neighbor (simple mode)
     * "Find common transaction neighbors of Steven Collins and Samantha Cook where transaction amounts are both greater than 400" → common_neighbor (filter mode)
     * "Find recipients that both Steven Collins and Samantha Cook transferred to, give Top-5 ranking by common transfer count" → common_neighbor (aggregation ranking mode)

7. **subgraph**: Subgraph extraction (supports three modes)
   - Extract subgraph (nodes + relationships)
   - Keywords: subgraph, extract, retrieve
   - Returns: Node set + relationship set
   - Supports three modes:
     * **Single center node mode**: N-hop subgraph centered on a node, e.g., "2-hop subgraph around Collins Steven"
     * **Relationship property filter mode**: Extract subgraph based on relationship property conditions, e.g., "extract subgraph of all transactions on 2025-05-01"
   - ⚠️ **Judgment criteria**:
     * Has specific node identifier (name, ID) → single center node mode
     * Has relationship property conditions (time, amount, etc.) → relationship property filter mode

8. **subgraph_by_nodes**: Multi-node subgraph extraction (⭐ Key distinction)
   - Extract subgraph based on node list (multiple specified nodes and their mutual relationships)
   - ⚠️ **Key**: Question **must contain multiple nodes** (e.g., node list [A,B,C])
   - Keywords:
     * Node list: [A,B,C], A, B, C
     * Mutual relationships: between each other, mutually, edges between
     * Contains: containing nodes, containing...and their
   - Returns: Specified nodes + relationships between them
   - Examples:
     * ✅ "Get subgraph containing node list [A,B,C] and all edges between them" → subgraph_by_nodes (has node list)
     * ✅ "Extract accounts A, B, C and transfer relationships between them" → subgraph_by_nodes (has multiple nodes)
     * ❌ "2-hop subgraph around Collins Steven" → subgraph (only one center node)
   - ⚠️ **Judgment criteria**: If question has multiple node identifiers (list, comma-separated, etc.), it's subgraph_by_nodes!

⚠️ Special note: aggregation_query vs relationship_filter
- "Count transaction times for each account" → aggregation_query (GROUP BY + COUNT)
- "Find transactions with amount greater than 400" → relationship_filter (WHERE filtering)
- "Calculate total amount for each account" → aggregation_query (GROUP BY + SUM)
- "List count of transactions with is_sar=False" → relationship_filter (global COUNT)

Please determine which query type the user question belongs to. Only return the type name (e.g., "node_lookup"), no other content.
"""

# add gjq
# Prompt for extracting parameters in natural language query engine
nl_query_extract_params_prompt = """{schema_info}
## Query Template Information
Query type: {query_type}
Description: {template_description}
Invocation method: {template_method}
Required parameters: {required_params}
Query-specific optional parameters: {optional_params}
## Universal Modifiers (Applicable to any query)
{query_modifiers}
## User Question
{question}
## Task
**Important: You must strictly reference the Schema information (node types, properties, relationship types, etc.) provided above to fill in the parameters.**
Based on the Schema information and query template requirements, extract parameter values from the user question and return them in JSON format.
**Key improvement: You must clearly indicate which universal modifiers need to be used!**
Requirements:
1. Required parameters must be filled
2. **label must be selected from the node types in the Schema** (refer to the "Node Types and Properties" section above)
3. **key selection rules (important)**：
   - **Check the property list for the label in the Schema**
   - If the Schema has a `node_key` property, **must use node_key first**
   - only consider other properties (like id, acct_id, etc.) if there is no node_key
4. **value/v1/v2 extraction rules (important)**:
   - **Extract based on the format of the key's corresponding Schema example values**
   - If key is node_key, and Schema example is name format, extract full name from question
   - Name format is typically "last_name first_name" (e.g., "Collins Steven", "Nunez Mitchell")
   - For path_query and common_neighbor, extract two names as v1 and v2
   - If question is "A to B" or "A and B", A is v1, B is v2
   - Do not extract unrelated values (e.g., base_amt, amount, transaction)
5. **rel_type selection rules**:
   - **must be selected from the "Relationship Types and Properties" section of the Schema**
   - Match relationship types based on keywords in the question (e.g., "transaction", "transfer")
   - If the question does not explicitly specify a relationship type, do not set it (use default all relationships)
6. **Universal modifier judgment rules (new, very important)**:
   **You must include a "modifiers" field in the JSON, listing the modifiers to be used!**
   a) **order_by and order_direction**:
      - Only used when the question explicitly mentions sorting needs like "sort", "order by", "largest to smallest", "smallest to largest", "largest", "smallest", etc.
      - If sorting is needed, must select the correct field name from the Schema
      - **Field name format (based on query type)**：
        * **neighbor_query**：
          - Relationship property: firstRel.property_name (e.g., firstRel.base_amt)
          - Node properties: `nbr.property_name` (e.g., `nbr.name`)
        * **common_neighbor**：
          - Relationship properties: `rA.property_name` or `rB.property_name` (e.g., `rA.base_amt`)
          - Node properties: `C.property_name` (e.g., `C.node_key`)
        * **path_query (path query)**:
          - Path length: hops
          - Node properties: need to use the node variables in the path
      - order_direction: "ASC" (ascending) or "DESC" (descending)
      - **If the question does not require sorting, do not include order_by in modifiers**
   b) **limit**：
      - Only used when the question explicitly mentions quantity limits (e.g., "first N", "at most N", "N items")
      - If the question contains "all", "entire", etc., **do not use limit**
      - If it's an aggregation query (just count/sum, etc.), **do not use limit**
      - **If the question does not have a quantity limit, do not include limit in modifiers**
   c) **where**：
      - Only used when the question explicitly mentions filtering conditions (e.g., "amount greater than 1000", "balance over 500")
      - **Condition format (varies by query type)**:
        * **neighbor_query**：
          - Node properties: `nbr.property_name operator value` (e.g., `nbr.balance > 1000`)
          - Relationship properties: `firstRel.property_name operator value` (e.g., `firstRel.base_amt > 500`)
        * **common_neighbor**：
          - Node properties: `C.property_name operator value` (e.g., `C.balance > 1000`)
          - Relationship properties: `rA.property_name operator value` or `rB.property_name operator value`
        * **path_query**:
          - Path length: `hops operator value` (e.g., `hops <= 3`)
      - **If the question does not have filtering conditions, do not include where in modifiers**
   d) **aggregate and aggregate_field**:
      - Only used when the question contains aggregate keywords (e.g., "count", "number of", "how many", "sum", "average")
      - aggregate 类型:count、sum、avg、max、min
      - **If the question is not an aggregate query, do not include aggregate in modifiers**
7. **JSON output format (important)**:
   Must contain two top-level fields:
   - "params": query parameters (required parameters + query-specific optional parameters)
   - "modifiers": universal modifiers to be used (only include needed modifiers)
Example Output 1 (Basic query - no modifiers):
{{{{
"params": {{{{
"label": "Account",
"key": "node_key",
"value": "Lee Alex"
}}}},
"modifiers": {{{{}}}}
}}}}
Example Output 2 (With sorting and limit):
{{{{
"params": {{{{
"label": "Account",
"key": "node_key",
"value": "Collins Steven",
"hops": 1
}}}},
"modifiers": {{{{
"order_by": "firstRel.base_amt",
"order_direction": "DESC",
"limit": 5
}}}}
}}}}
Example Output 3 (With filtering condition):
{{{{
"params": {{{{
"label": "Account",
"key": "node_key",
"value": "Collins Steven",
"hops": 1
}}}},
"modifiers": {{{{
"where": "firstRel.base_amt > 1000"
}}}}
}}}}
Example Output 4 (Aggregate query):
{{{{
"params": {{{{
"label": "Account",
"key": "node_key",
"value": "Collins Steven",
"hops": 1
}}}},
"modifiers": {{{{
"aggregate": "count"
}}}}
}}}}
Example Output 5 (Combined modifiers):
{{{{
"params": {{{{
"label": "Account",
"key": "node_key",
"value": "Collins Steven",
"hops": 1
}}}},
"modifiers": {{{{
"where": "firstRel.base_amt > 500",
"order_by": "firstRel.base_amt",
"order_direction": "DESC",
"limit": 3
}}}}
}}}}
Begin:
"""

# add gjq
# Prompt for validating Cypher statements in natural language query engine
nl_query_validate_cypher_prompt = """You are a Neo4j Cypher statement validation expert (Neo4j 3.5.25 version). Please verify if the following query meets the requirements.

{schema_info}

{template_info}

{template_cypher_example}

## User Question
{question}

## Query Type
{query_type}

## Extracted Parameters
{params}

## Parameter Description (for understanding query intent)
```
{cypher}
```

## ⚠️ CRITICAL: Common LLM Error Patterns and Correction Guide

### 🔴 Error 1: WHERE Condition Syntax Errors - Most basic but most common!
**Problem Description**: Basic syntax and logic errors when handling relationship or node property conditions.

**Common Errors**:
1. **Range Query Errors**:
   ```cypher
   // ❌ Wrong: Using dictionary syntax for range (this is a serious syntax error!)
   MATCH (a)-[r:TRANSFER]->(b)
   WHERE r.base_amt = {{"min": 300, "max": 500}}
   RETURN r
   
   // ✅ Correct: Use standard comparison operators
   MATCH (a)-[r:TRANSFER]->(b)
   WHERE r.base_amt >= 300 AND r.base_amt <= 500
   RETURN r
   ```

2. **Path Length Understanding Errors**:
   ```cypher
   // ❌ Wrong: User requires "neighbors exactly two hops away", but used *1..2 (one to two hops)
   MATCH (a:Account {{node_key: "Collins Steven"}})-[*1..2]-(b)
   RETURN DISTINCT b
   
   // ✅ Correct: Use *2 for exactly two hops
   MATCH (a:Account {{node_key: "Collins Steven"}})-[*2]-(b)
   RETURN DISTINCT b
   
   // 📝 Explanation:
   // *1..2 = one or two hops (includes direct neighbors)
   // *2 = exactly two hops (excludes direct neighbors)
   // *2..3 = two or three hops
   ```

3. **IN Operator Misuse**:
   ```cypher
   // ❌ Wrong: Directly assigning list to property
   MATCH (a)-[r:TRANSFER]->(b)
   WHERE r.tx_type = ["WIRE", "ACH"]
   RETURN r
   
   // ✅ Correct: Use IN operator
   MATCH (a)-[r:TRANSFER]->(b)
   WHERE r.tx_type IN ["WIRE", "ACH"]
   RETURN r
   ```

4. **Date Range Query Errors**:
   ```cypher
   // ❌ Wrong: Using dictionary syntax
   MATCH (a)-[r:TRANSFER]->(b)
   WHERE r.tran_timestamp = {{"start": "2025-05-01", "end": "2025-05-02"}}
   RETURN r
   
   // ✅ Correct: Use comparison operators
   MATCH (a)-[r:TRANSFER]->(b)
   WHERE r.tran_timestamp >= "2025-05-01" AND r.tran_timestamp < "2025-05-02"
   RETURN r
   ```

**Checklist**:
- [ ] Check if all WHERE conditions use dictionary syntax `= {{"min": x, "max": y}}`
- [ ] For range queries, must use `>= AND <=` or `BETWEEN` (Neo4j 3.5.25 supports)
- [ ] Check path length expressions:
  - "exactly N hops away" / "exactly N hops" → use `*N`
  - "within N hops" / "at most N hops" → use `*1..N`
  - "N to M hops" → use `*N..M`
- [ ] Check if IN operator is used correctly
- [ ] Check if date/time range queries use correct comparison operators

### 🔴 Error 2: Directionality Confusion - Most serious!
**Problem Description**: LLM tends to incorrectly "correct" queries with clear direction to undirected bidirectional queries `-[rel]-`, thinking bidirectional is "safer", but this violates the user's precise intent.

**Key Principles**:
1. **Default to directed relationships** `-[rel]->`, unless user explicitly indicates "bidirectional", "between", "mutual"
2. **Transfer/receipt direction must be precise**:
   - "send out" / "initiate" / "as sender account" / "transfer to..." → `-[rel]->` (from starting point)
   - "receive" / "accept" / "as receiver account" / "receive from..." → `<-[rel]-` (pointing to starting point)
   - "recipient" / "receiver" / "both transferred to" → must be arrow target `-[rel]->`
   - "payer" / "sender" / "both received from" → must be arrow source `<-[rel]-`

**Wrong Examples**:
```cypher
// ❌ Wrong: User says "recipients that both transferred to", intent is (A)->(C) and (B)->(C)
MATCH (a)-[rA]-(c), (b)-[rB]-(c)  // Wrong! Used bidirectional relationship
WHERE ...
RETURN c

// ✅ Correct: Must use directed relationship, arrow points to recipient
MATCH (a)-[rA]->(c), (b)-[rB]->(c)  // Correct! Arrow points to recipient
WHERE ...
RETURN c
```

```cypher
// ❌ Wrong: User says "accounts that received transfers from both", intent is (A)->(C) and (B)->(C)
MATCH (a)-[rA]-(c), (b)-[rB]-(c)  // Wrong! Used bidirectional relationship
WHERE ...
RETURN c

// ✅ Correct: Must use directed relationship, arrow points to receiver
MATCH (a)-[rA]->(c), (b)-[rB]->(c)  // Correct! Arrow points to receiver
WHERE ...
RETURN c
```

**Checklist**:
- [ ] Check direction keywords in user question: "send out", "receive", "recipient", "payer", "initiate", "accept", "transfer to..."
- [ ] If there's clear direction, Cypher must use `->` or `<-`, cannot use `-`
- [ ] "recipient" / "both transferred to" = arrow target node `-[rel]->`
- [ ] "payer" / "both received from" = arrow source `<-[rel]-`
- [ ] Only "between", "mutual", "bidirectional", "transacted with" (no clear direction) use `-[rel]-`

### 🔴 Error 3: Calculation and Statistics Errors
**Problem Description**: Frequent errors in aggregation and sorting.

**Common Errors**:
1. **MAX/MIN Misuse**:
   ```cypher
   // ❌ Wrong: Thinking MAX() can return the entire record with max value
   RETURN MAX(r.base_amt) AS max_record
   
   // ✅ Correct: Use ORDER BY + LIMIT 1
   RETURN r.base_amt, r.tran_id, a, b
   ORDER BY r.base_amt DESC
   LIMIT 1
   ```

2. **Aggregation Target Confusion**:
   ```cypher
   // ❌ Wrong: Confused "neighbor count" with "relationship count"
   MATCH (a)-[r]->(b)
   RETURN a, COUNT(r) AS neighbor_count  // Wrong! This is relationship count
   
   // ✅ Correct: Neighbor count should COUNT DISTINCT nodes
   MATCH (a)-[r]->(b)
   RETURN a, COUNT(DISTINCT b) AS neighbor_count
   ```

3. **Incomplete Multiple Aggregation Results**:
   ```cypher
   // ❌ Wrong: User requires "sum" and "max contributor", only returned sum
   MATCH (a)-[r]->(b)
   RETURN SUM(r.base_amt) AS total
   
   // ✅ Correct: Need to calculate separately or use WITH clause
   MATCH (a)-[r]->(b)
   WITH a, SUM(r.base_amt) AS total_per_a
   ORDER BY total_per_a DESC
   LIMIT 1
   RETURN a AS max_contributor, total_per_a AS max_amount
   ```

**Checklist**:
- [ ] If need "record with max/min value", use `ORDER BY ... LIMIT 1`, not `MAX()/MIN()`
- [ ] Distinguish "neighbor count" `COUNT(DISTINCT b)` from "relationship count" `COUNT(r)`
- [ ] If user requires multiple aggregation results, ensure all are returned

### 🔴 Error 4: Syntax Hallucination
**Problem Description**: LLM borrows from other languages (like SQL) or "invents" syntax that doesn't exist in Cypher.

**Common Hallucinations**:
1. **SQL-style SELECT Subqueries**:
   ```cypher
   // ❌ Wrong: Cypher doesn't support SELECT subqueries
   MATCH (a)
   WHERE a.id IN (SELECT b.id FROM Account b WHERE b.balance > 1000)
   RETURN a
   
   // ✅ Correct: Use WITH clause or direct MATCH
   MATCH (a:Account)
   WHERE a.balance > 1000
   RETURN a
   ```

2. **GROUP BY Clause**:
   ```cypher
   // ❌ Wrong: Neo4j 3.5.25 doesn't support explicit GROUP BY
   MATCH (a)-[r]->(b)
   RETURN a, SUM(r.base_amt) AS total
   GROUP BY a
   
   // ✅ Correct: Aggregation is implicit, no GROUP BY needed
   MATCH (a)-[r]->(b)
   RETURN a, SUM(r.base_amt) AS total
   ```

3. **JOIN Syntax**:
   ```cypher
   // ❌ Wrong: Cypher doesn't support JOIN
   MATCH (a) JOIN (b) ON a.id = b.ref_id
   
   // ✅ Correct: Use MATCH pattern matching
   MATCH (a), (b)
   WHERE a.id = b.ref_id
   ```

**Checklist**:
- [ ] Don't use `SELECT`, `FROM`, `JOIN`, `GROUP BY` and other SQL keywords
- [ ] Aggregation queries use aggregation functions directly in RETURN, no GROUP BY needed
- [ ] Subqueries use `WITH` clause, not `SELECT`

## Validation Requirements

### 1. Syntax Correctness (Neo4j 3.5.25)
- Check if Cypher syntax is correct
- Check if parentheses and quotes are matched
- Check if keywords are spelled correctly
- ⚠️ **Key**: If using relationship properties in RETURN (like rel.base_amt), must define relationship variable in MATCH (like [rel:TRANSFER])
- ⚠️ **Important**: Neo4j 3.5.25 **does not support** SQL-style `GROUP BY` syntax!
  - ❌ Wrong: `RETURN a, SUM(r.base_amt) AS total GROUP BY a`
  - ✅ Correct: `RETURN a, SUM(r.base_amt) AS total` (aggregation is implicit, no GROUP BY needed)
- ⚠️ **Important**: Correct syntax for aggregation queries:
  - When using aggregation functions (SUM, COUNT, AVG, etc.) in RETURN, all non-aggregated fields automatically become grouping keys
  - Example: `MATCH (a)-[r]->(b) RETURN a.name, COUNT(r) AS cnt ORDER BY cnt DESC`

### ⚠️ **CRITICAL: Multiple MATCH Statements and Semicolon Separation Errors**
- ❌ **Error 1: Multiple Independent MATCH Clauses**
  - Including multiple independent MATCH clauses in one query without using WITH to pass context
  - Wrong example:
    ```cypher
    MATCH (a:Account {{{{node_key: "Collins Steven"}}}}
    MATCH (b:Account {{{{node_key: "Cook Samantha"}}}}
    MATCH (a)-[rA:TRANSFER]->(c)
    MATCH (b)-[rB:TRANSFER]->(c)
    WHERE rA.base_amt > 400 AND rB.base_amt > 400
    RETURN c
    ```
  - ✅ Correct approach: Merge all MATCH into one query, or use WITH to pass context
    ```cypher
    MATCH (a:Account {{{{node_key: "Collins Steven"}}}}, (b:Account {{{{node_key: "Cook Samantha"}}}}
    MATCH (a)-[rA:TRANSFER]->(c), (b)-[rB:TRANSFER]->(c)
    WHERE rA.base_amt > 400 AND rB.base_amt > 400
    RETURN c
    ```

- ❌ **Error 2: Using Semicolons to Separate Multiple Queries**
  - Sending multiple query statements separated by semicolons `;` in one request
  - This is not allowed in most Cypher execution environments
  - Wrong example:
    ```cypher
    MATCH (a:Account) RETURN a LIMIT 10;
    MATCH (b:Account) RETURN b LIMIT 10;
    ```
  - ✅ Correct approach: Execute one query at a time, or use UNION to merge results
    ```cypher
    MATCH (a:Account) RETURN a LIMIT 10
    UNION
    MATCH (b:Account) RETURN b LIMIT 10
    ```

- **Check Points**:
  1. Count occurrences of MATCH keyword in query
  2. Check if there are semicolons `;` separating multiple statements
  3. If there are multiple MATCHes, check if WITH clause is used to connect them
  4. Ensure all MATCH clauses are logically coherent

### 2. Semantic Correctness
- Check if user question is correctly understood
- Check if all necessary conditions are included
- Check if important filtering conditions are missing

### 3. Special Checks (for different query types)

#### neighbor_query (Neighbor Query)
- ⚠️ **Key**: If need to return relationship properties (like rel.base_amt), must define relationship variable in MATCH
- Wrong example: `MATCH (a)-[:TRANSFER]->(b) RETURN rel.base_amt` ❌ (rel not defined)
- Correct example: `MATCH (a)-[rel:TRANSFER]->(b) RETURN rel.base_amt` ✅

#### common_neighbor (Common Neighbor Query)
If user question contains "and", "also", "simultaneously" keywords, indicating need to filter relationships:
- ✅ Correct: Use WHERE clause to filter both relationships
- ❌ Wrong: Only find common neighbors, no relationship property filtering

**Example**:
- Question: "Find common transaction neighbors of Steven Collins and Samantha Cook where transaction amounts must be greater than 400"
- Requirement: Must filter both relationships (Steven Collins → neighbor and Samantha Cook → neighbor) with base_amt > 400
- Correct Cypher should include:
  ```cypher
  WHERE rA.base_amt > 400 AND rB.base_amt > 400
  ```

#### relationship_filter (Relationship Filter Query)
- Check if relationship property filtering conditions are correctly applied
- Check if correct operators are used (>, <, =, STARTS WITH, etc.)

#### aggregation_query (Aggregation Query)
- Check if aggregation functions are correctly used (COUNT, SUM, AVG, etc.)
- Check if grouping is correctly done (GROUP BY)
- Check if aggregate_field is correct

### 4. Schema Validation (⚠️ Most Important)
- ⚠️ **Key**: Must use node labels and relationship types that actually exist in Schema
- ⚠️ **Key**: Must use property names that actually exist in Schema
- Check if node labels are in Schema's "Node Types and Properties"
- Check if relationship types are in Schema's "Relationship Types and Properties"
- Check if property names are in corresponding node/relationship property list
- ❌ Wrong example: Using non-existent relationship type `[:NEIGHBOR]` (not in Schema)
- ✅ Correct example: Using relationship type that exists in Schema, like `[:TRANSFER]`

### 5. Relationship Direction Validation (⚠️ Very Important)
- ⚠️ **CRITICAL**: Correct understanding of "between", "related to" and other bidirectional relationships
- **Problem Manifestation**: When user's natural language mentions "between A and B", "related to A" or "mutual" transactions,
  it usually means the relationship is bidirectional (A could be sender or receiver), but generated Cypher only considers unidirectional relationship
- **Wrong Examples**:
  * ❌ "transactions with Hernandez Alexis" → using `-[rel]->` (unidirectional)
  * ❌ "Steven Collins, Samantha Cook and all direct transactions between them" → using `-[rel]->` (unidirectional)
  * ❌ "suspicious transactions related to Steven Collins" → using `-[rel]->` (unidirectional)
- **Correct Approach**:
  * ✅ "transactions with Hernandez Alexis" → using `-[rel]-` (bidirectional)
  * ✅ "Steven Collins, Samantha Cook and all direct transactions between them" → using `-[rel]-` (bidirectional)
  * ✅ "suspicious transactions related to Steven Collins" → using `-[rel]-` (bidirectional)
- **Keyword Recognition**:
  * "between", "mutual", "each other", "mutually" → must use bidirectional `-[rel]-`
  * "related to...", "associated with...", "involving" → must use bidirectional `-[rel]-`
  * "occurred", "exists", "had" → must use bidirectional `-[rel]-`
  * "as sender account", "as receiver account", "from...to..." → use unidirectional `-[rel]->`
- **Check Points**:
  1. Check if user question contains bidirectional relationship keywords
  2. If yes, check if Cypher uses undirected matching pattern `-[rel]-`
  3. If using unidirectional arrow `-[rel]->` or `<-[rel]-`, mark as error
  4. Only questions with explicitly specified direction (like "from A to B", "A transfers to B") should use unidirectional arrow

### 6. Filter Condition Completeness Check (⚠️ Very Important)
- ⚠️ **CRITICAL**: When query requires relationships to occur "within" or "between" a specific group, filter conditions must be applied to both ends of the relationship
- **Problem Manifestation**: Only applied property filtering to one end of relationship, not the same requirement to the other end
- **Wrong Examples**:
  * ❌ "transfers between accounts with branch_id 1" → `MATCH (a {{{{branch_id: 1}}}}-[r]->(b)` (only filtered a)
  * ❌ "transactions between Pagetown customers" → `MATCH (a {{{{city: "Pagetown"}}}}-[r]->(b)` (only filtered a)
- **Correct Approach**:
  * ✅ "transfers between accounts with branch_id 1" → `MATCH (a {{{{branch_id: 1}}}}-[r]-(b {{{{branch_id: 1}}}}`
  * ✅ "transactions between Pagetown customers" → `MATCH (a {{{{city: "Pagetown"}}}}-[r]-(b {{{{city: "Pagetown"}}}}`
- **Keyword Recognition**:
  * "between", "among each other", "within", "mutually" → must apply same filter condition to both end nodes
  * "same", "identical", "both are" → must apply same filter condition to both end nodes
- **Check Points**:
  1. Check if user question contains "between", "mutual", "within" keywords
  2. Check if specific group properties are specified (like branch_id, city, country, etc.)
  3. If yes, check if Cypher applies same filter condition to both end nodes of relationship
  4. If only one end node is filtered, mark as error

### 7. Common Error Checks
- Are conditions explicitly mentioned by user missing
- Are non-existent property names used
- Are wrong node labels or relationship types used
- ⚠️ **Most Important**: If variables (like rA, rB, rel) are used in WHERE or RETURN, the variable must be defined in MATCH

### ⚠️ **CRITICAL: Relationship Variable Definition Check (Most Common Error!)**
**Problem Description**: Using relationship variables (like `rA.base_amt`) in WHERE or RETURN clause, but the variable is not defined in MATCH clause.

**Wrong Example**:
```cypher
// ❌ Wrong: rA and rB used in WHERE, but not defined in MATCH
MATCH (a:Account {{{{node_key: 'Collins Steven'}}}}-[:TRANSFER]->(c),
      (b:Account {{{{node_key: 'Cook Samantha'}}}}-[:TRANSFER]->(c)
WHERE rA.base_amt > 0 AND rB.base_amt > 0  // Wrong! rA and rB not defined
RETURN c
```

**Correct Approach**:
```cypher
// ✅ Correct: Define relationship variables in MATCH
MATCH (a:Account {{{{node_key: 'Collins Steven'}}}}-[rA:TRANSFER]->(c),
      (b:Account {{{{node_key: 'Cook Samantha'}}}}-[rB:TRANSFER]->(c)
WHERE rA.base_amt > 0 AND rB.base_amt > 0  // Correct! rA and rB defined
RETURN c
```

**Checklist**:
- [ ] Scan WHERE clause, find all used relationship variables (like `rA.xxx`, `rB.xxx`, `rel.xxx`)
- [ ] Scan RETURN clause, find all used relationship variables
- [ ] For each relationship variable, check if there's corresponding definition in MATCH clause (like `[rA:TRANSFER]`)
- [ ] If MATCH uses `[:TRANSFER]` instead of `[rA:TRANSFER]`, this is wrong!
- [ ] **Self-check**: Before generating corrected_cypher, check again if all variables are defined

## Output Format

Please return validation result in JSON format:

```json
{{{{
  "is_valid": true/false,
  "issues": [
    "Issue 1 description",
    "Issue 2 description"
  ],
  "suggestions": [
    "Suggestion 1",
    "Suggestion 2"
  ],
  "corrected_cypher": "Corrected Cypher statement (if correction needed)"
}}}}
```

**Notes**:
1. If parameter or logic issues are found, **must** provide corrected Cypher statement in `corrected_cypher`
2. Corrected Cypher must reference template information and examples above
3. For subgraph queries, ensure returning **all** 1-hop and 2-hop neighbors, don't miss any
4. ⚠️ **CRITICAL**: Before generating corrected_cypher, **must** perform self-check:
   - Check if all variables used in WHERE and RETURN are defined in MATCH
   - If using `rA.xxx`, MATCH must have `[rA:...]`
   - If using `rB.xxx`, MATCH must have `[rB:...]`
   - If using `rel.xxx`, MATCH must have `[rel:...]`
5. If parameters are completely correct and no correction needed, return:
```json
{{{{
  "is_valid": true,
  "issues": [],
  "suggestions": [],
  "corrected_cypher": null
}}}}
```
"""

# add gjq - Aggregation query extractor prompt
nl_query_aggregation_query_prompt = """{schema_info}

## Query Type: aggregation_query

### Function Description
Aggregation statistical query, supports grouping by node or property, calculates COUNT/SUM/AVG/MAX/MIN  etc.
**⚠️ Key**：aggregation_query **only for grouped aggregation**（GROUP BY），not for global aggregation！

### Parameter Description
**Required Parameters**：
- `aggregate_type`: Aggregation type（"COUNT", "SUM", "AVG", "MAX", "MIN"）
- `group_by_node`: Group by node（"start" or "end"）
- `node_label`: Node label
- `rel_type`: Relationship type
- `direction`: Direction（"out", "in", "both"）

**Optional Parameters**：
- `aggregate_field`: Aggregation field（SUM/AVG/MAX/MIN required when，COUNT not needed when）
- `return_fields`: Node fields to return
- `order_by`: Sort field（"count" or "total"）
- `order_direction`: Sort Direction（"ASC" or "DESC"）
- `limit`: Result count limit

### User Question
{question}

### Task
Based on the above rules, extract parameters from user question and return in JSON format：
{{
  "params": {{...}},
  "modifiers": {{...}}
}}
"""

# add gjq - Neighbor query extractor prompt
nl_query_neighbor_query_prompt = """{schema_info}

## Query Type: neighbor_query

### Function Description
Query neighbors or N-hop relationships of a specific node, supports returning detailed information for each edge。
**⚠️ Key**：Question **must have a clear starting node**（such as person name, ID, specific account）。

### Parameter Description
**Required Parameters**：
- `label`: Node label
- `key`: Property key（usually node_key）
- `value`: Property value（Identifier of starting node）

**Optional Parameters**：
- `hops`: Hops（default 1）
- `rel_type`: Relationship type
- `direction`: Direction（"out", "in", "both"）
- `return_fields`: List of fields to return

### User Question
{question}

### Task
Based on the above rules, extract parameters from user question and return in JSON format：
{{
  "params": {{...}},
  "modifiers": {{...}}
}}
"""

# add gjq - Path query extractor prompt
nl_query_path_query_prompt = """{schema_info}

## Query Type: path_query

### Function Description
Query paths between two nodes, supports complex filtering, sorting and calculation logic。

### Parameter Description
**Required Parameters**：
- `label`: Node label
- `key`: Property key（usually node_key）
- `v1`: Starting node value
- `v2`: Ending node value

**Optional Parameters**：
- `rel_type`: Relationship type
- `direction`: Direction（"out", "in", "both"）
- `min_hops`: minimumHops（default 1）
- `max_hops`: maximumHops（default 5）

### User Question
{question}

### Task
Based on the above rules, extract parameters from user question and return in JSON format：
{{
  "params": {{...}},
  "modifiers": {{...}}
}}
"""

# add gjq - Common neighbor extractor prompt
nl_query_common_neighbor_prompt = """{schema_info}

## Query Type: common_neighbor

### Function Description
Query common neighbors of two nodes，支持Relationship property filtering。

⚠️ **Important concept**：Common neighbor query uses AND logic, not OR logic！
- ✅ Correct：Find nodes **simultaneously** connected to both A and B（A→n AND B→n）
- ❌ Wrong：Find nodes connected to A **or** B（A→n OR B→n）

### Parameter Description
**Required Parameters**：
- `label`: Node label
- `key`: Property key（usually node_key）
- `v1`: First node value
- `v2`: Second node value

**Optional Parameters**：
- `rel_type`: Relationship type
- `direction`: Direction（"out", "in", "both"）
- `rel_conditions`: Relationship propertiesFilter conditions
- `return_fields`: List of fields to return

### User Question
{question}

### Task
Based on the above rules, extract parameters from user question and return in JSON format：
{{
  "params": {{...}},
  "modifiers": {{...}}
}}
"""

# add gjq - Subgraph extractor prompt
nl_query_subgraph_prompt = """{schema_info}

## Query Type: subgraph

### Function Description
Extract subgraph (nodes + relationships), supports two modes。

### Parameter Description
**Two Modes**：

1. **Single center node mode**：
   - `label`: Node label
   - `key`: Property key
   - `value`: Node value
   - `hops`: Hops
   - Optional：`rel_type`, `direction`, `limit_paths`

2. **Relationship propertiesFilter mode**：
   - `rel_type`: Relationship type
   - `rel_conditions`: Relationship property conditions
   - `start_label`: start node label
   - `end_label`: end node label
   - Optional：`limit`

### User Question
{question}

### Task
Based on the above rules, extract parameters from user question and return in JSON format：
{{
  "params": {{...}},
  "modifiers": {{...}}
}}
"""

# add gjq - Subgraph by nodes extractor prompt
nl_query_subgraph_by_nodes_prompt = """{schema_info}

## Query Type: subgraph_by_nodes

### Function Description
Extract subgraph based on node list (multiple specified nodes and their mutual relationships)。
**⚠️ Key**：Question中**must contain multiple nodes**（such as node list [A,B,C]）。

### Parameter Description
**Required Parameters**：
- `label`: Node label
- `key`: Property key（usually node_key）
- `values`: Node value list (array)

**Optional Parameters**：
- `include_internal`: whether to include internal edges (default true)
- `rel_type`: Relationship type
- `direction`: Direction

### User Question
{question}

### Task
Based on the above rules, extract parameters from user question and return in JSON format：
{{
  "params": {{...}},
  "modifiers": {{...}}
}}
"""
