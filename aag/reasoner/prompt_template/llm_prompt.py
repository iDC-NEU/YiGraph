
"""提示词1. 根据问题识别图算法"""
prompt_select_graph_algorithm_str_en = """
You are a graph algorithm expert who is good at judging the most appropriate graph algorithm based on natural language questions and relevant context retrieved from the graph algorithm paper knowledge base.
Now I will provide an **input question** and **retrieved context** related to the question. These contexts are a string list (List[str]) consisting of multiple paragraphs, each of which is a text fragment from a graph algorithm-related paper.
Based on the information from both, please infer and output the most appropriate graph algorithm name.

### Task:
1.Your task is to select the most appropriate graph algorithm from all possible candidates (e.g., PageRank, Dijkstra, Louvain, Connected Components, Betweenness Centrality, GCN, GAT) to solve the given problem.
2.If the context explicitly mentions that a certain algorithm is suitable for this type of problem, you should prioritize that algorithm.
3.If it is not clearly stated, please combine the problem type and context semantics to reason and select the most appropriate algorithm.
4.Only output the algorithm name, do not explain the reason, and do not output other content. Use the following format:

**Output:**
```json
{{
    "algorithm": # type: str
}}
```

### Example:
**Input question:**
In the Facebook social network, the user Anna was recently reported as a malicious account. We want to find out other accounts that may form a "malicious account gang" with her. Which accounts are in the same social connection group with her?

**Retrieved context:**
[
    "In a user relationship graph, each node represents a user and edges represent mutual (bidirectional) friendships. One commonly used method for identifying potential collaborative behavior groups is to find 'connected subgraphs' in the graph. The Connected Components algorithm can be used on undirected graphs to divide all mutually reachable nodes into separate components. It is well-suited for detecting tightly connected account clusters, and is widely used in malicious behavior tracing and abnormal account detection in social networks. By computing connected components, it is possible to quickly identify all accounts that belong to the same group as a known suspicious user, without traversing the entire graph structure. The algorithm is highly efficient and suitable for large-scale social networks.",
    "Connected Components is one of the fundamental algorithms in graph theory, mainly used to identify all sets of mutually reachable nodes in an undirected graph. Each connected component represents a subgraph in which there exists a path between any pair of nodes. This algorithm is widely applied in social network analysis, community detection, functional module identification in biological networks, and the discovery of isolated groups in large-scale graphs. Due to its computational efficiency, it is especially suitable for static and sparse graphs with known structure."
]

**Output:**
```json
{{
    "algorithm": "Connected Components"
}}
```

### Instruction:
**Please infer the most suitable graph algorithm based on the input question and the retrieved context provided below, and output your answer in the specified format:

**Input question:**
{input_question}

**Retrieved context:**
{context}

**Output:**

"""



prompt_select_graph_algorithm_str_zh = """
你是一位图算法专家，擅长根据自然语言问题和从图算法论文知识库中检索到的相关上下文，判断最合适的图算法。
现在我会提供一个**输入问题**和几段与该问题相关的**检索上下文**。这些上下文是一个由多个段落构成的字符串列表（List[str]），每段内容来自图算法相关论文的文本片段。
请你根据两者的信息，推理并输出一个最合适的图算法名称。

### 任务要求:
1.你的任务是从所有图算法中（如：PageRank、Dijkstra、Louvain、Connected Components、Betweenness Centrality、GCN、GAT等），选择一个最适合解决该问题的算法名称
2.如果上下文中明确指出某一算法适用于该类问题，优先选用该算法
3.如果没有明确指出，请结合问题类型和上下文语义进行推理，选择最合适的算法
4.只输出算法名称，不要解释理由、不输出其他内容。 输出如下格式：

**输出格式:**
```json
{{
    "algorithm": # type: str
}}
```

### 示例:
**输入问题:**
在 Facebook 社交网络中，用户Anna最近被举报为恶意账号，我们想找出可能与她组成“恶意账号团伙”的其他账号，有哪些账号与她在同一个社交连通群中？
**检索上下文:**
[
    "在用户关系图中，每个节点表示一个用户，边表示双向好友关系。为了识别潜在的协同行为群体，常用的方法之一是找到图中的“连通子图”。Connected Components 算法可用于无向图中划分出所有互相可达的节点集合，适合发现局部紧密联系的账号团伙，尤其在恶意行为溯源和社交异常检测中被广泛使用。通过计算连通分量，可以快速识别与某个已知异常用户处于同一个群体中的所有账号，无需遍历全图结构，计算效率高，适用于大规模社交网络。",
    "Connected Components 是图论中的基础算法之一，主要用于在无向图中识别所有互相可达的节点集合。每一个连通分量代表图中的一个子图，其中任意两个节点之间都存在路径连接。该算法广泛应用于社交网络分析、聚类检测、生物网络中的功能模块识别，以及大规模图中的孤立社群发现等任务。其计算效率高，适合用于结构已知的静态图和稀疏图中。"
]

**输出:**
```json
{{
    "algorithm": "Connected Components"
}}
```

### 指令:
**请根据下方提供的输入问题与检索上下文，推理出最合适的图算法，并以标准格式输出：

**输入问题:**
{input_question}

**检索上下文:**
{context}

**Output:**

"""




"""提示词2. 根据问题、图计算结果, 生成response"""







