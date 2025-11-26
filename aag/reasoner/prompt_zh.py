analyze_dependency_type_and_locate_dependency_data_prompt = f"""你是一名专业的图数据分析专家，负责判断 DAG 子问题是否依赖父节点的计算结果，并精确定位依赖数据的来源
你将收到以下变量：
- current_question:当前子问题的自然语言描述
- task_type:当前子问题的任务类型
- current_algo_desc:当前子问题计划执行的函数/图算法描述
- parent_question:父节点的问题描述
- parent_outputs_meta:父节点经过计算后输出的结果的元信息

根据上述提供的子问题的信息和父问题的信息，你的任务是判断当前子问题的计算是否依赖父节点的输出结果，并明确以下内容：
## [1] dependency_type（依赖类型），只能从以下四类中选择一种：
- "graph"      ：当前问题执行计算需要一张“由父节点数据转换得到的子图”，而不是系统默认注入的原始全图。父节点提供的数据可以是：节点集合、边集合、社区节点列表和图结构片段等。这些数据经过图提取模块转换后，才能作为当前计算的输入图。
- "parameter"  ：当前子问题需要从父节点输出中提取某些“值”或“标量/集合”，用于计算过程。如果 task_type = graph_algorithm，则这些参数用于填充 current_algo_desc 的输入参数（如 source、target、threshold 等）。task_type = numeric_analysis，则这些值作为变量参与数值计算或代码逻辑。
- "both"       ：当前子问题既依赖父节点问题的数据来构造子图，又依赖父节点问题的提供的数值/节点/参数；
- "none"       ：当前子问题不依赖父节点的结果 

注意：对于 task_type = numeric_analysis，如果当前问题只是在做数值计算（如“比较这两个人的影响力大小”“计算这些节点的平均得分”），通常属于 "parameter" 依赖；只有在问题明确要求“在某个子图上再次运行图算法/分析”时，才判定为 "graph" 或 "both"。

## [2] 精确选择依赖的数据源：selected_outputs
selected_outputs 是一个数组，每一项代表依赖父节点的某一个 StepOutputItem。每个依赖项必须包含：
- output_id:父节点中被依赖的 StepOutputItem 的编号
- use_as：“graph” 或 “parameter”，用于标注该依赖项的具体依赖方式。若 dependency_type 为 “parameter” 或 “graph”，则所有 use_as 必须与其一致；若 dependency_type 为 “both”，则 selected_outputs 中必须至少包含一个 use_as=“graph” 和一个 use_as=“parameter”
- field_key:父节点中计算结果的字段名
- reason（为什么当前子问题依赖该数据，一句话解释清楚）
如果 dependency_type = "none"，则 selected_outputs:[], 不要写任何说明，不要任何自然语言。

You must return a valid JSON object in the exact format shown below. 
##Output Format (Must Follow Exactly)
```json
{
  "dependency_type": "none | parameter | graph | both",
  "selected_outputs": [
    {
      "output_id": <int>,
      "field_key": "<string>",
      "use_as": "parameter | graph",
      "reason": "<string>"
    }
  ]
}
```

### 示例1
## Input
### current_question:
在这个社区内部分析影响力，找到影响力最高的人
### 当前任务类型（task_type）:
graph_algorithm
### 当前计划执行的函数或图算法描述（current_algo_desc）:
alg_name:run_pagerank, running on engine: networkx
**Input Parameters**:
```json
{
  "type": "object",
  "parameters": {
    "G": {
      "description": "[System auto-injected] A NetworkX graph.  Undirected graphs will be converted to a directed graph with two directed edges for each undirected edge.",
      "type": "graph"
    },
    "alpha": {
      "description": "Damping parameter for PageRank, default=0.85.",
      "type": "float, optional",
      "default": 0.85
    },
    "personalization": {
      "description": "The \"personalization vector\" consisting of a dictionary with a key some subset of graph nodes and personalization value each of those. At least one personalization value must be non-zero. If not specified, a nodes personalization value will be zero. By default, a uniform distribution is used.",
      "type": "dict, optional",
      "default": null
    },
    "max_iter": {
      "description": "Maximum number of iterations in power method eigenvalue solver.",
      "type": "integer, optional",
      "default": 100
    },
    "tol": {
      "description": "Error tolerance used to check convergence in power method solver. The iteration will stop after a tolerance of ``len(G) * tol`` is reached.",
      "type": "float, optional",
      "default": 1e-06
    },
    "nstart": {
      "description": "Starting value of PageRank iteration for each node.",
      "type": "dictionary, optional",
      "default": null
    },
    "weight": {
      "description": "Edge data key to use as weight.  If None weights are set to 1.",
      "type": "key, optional",
      "default": "weight"
    },
    "dangling": {
      "description": "The outedges to be assigned to any \"dangling\" nodes, i.e., nodes without any outedges. The dict key is the node the outedge points to and the dict value is the weight of that outedge. By default, dangling nodes are given outedges according to the personalization vector (uniform if not specified). This must be selected to result in an irreducible transition matrix (see notes under google_matrix). It may be common to have the dangling dict to be the same as the personalization dict.",
      "type": "dict, optional",
      "default": null
    }
  }
}
```
### 父节点问题（parent_question）:
找到节点23所在的社区
### 父节点输出元信息（parent_outputs_meta）:
"parent_outputs_meta": [
  {
    "output_id": 1,
    "task_type": "graph_algorithm",
    "source": "run_run_louvain_communitieslouvain",
    "type": "dict",
    "description":"执行的是run_louvain_communities算法",
    "fields": [
      {"key": "node23community", "type": "list", "desc": "节点23所在社区的节点id集合"}
    ]
  }
]
## Output
```json
{
  "dependency_type": "graph",
  "selected_outputs": [
    {
      "output_id": 1,
      "field_key": "node23community",
      "use_as": "graph",
      "reason": "xxxxx"
    }
  ]
}
```

### 示例2
## Input
### current_question:
找到这两个人是否有路径
### 当前任务类型（task_type）:
graph_algorithm
### 当前计划执行的函数或图算法描述（current_algo_desc）:
alg_name:run_dijkstra_path, running on engine: networkx
**Input Parameters**:
```json
{
  "type": "object",
  "parameters": {
    "G": {
      "description": "[System auto-injected] Parameter 'G' for dijkstra_path",
      "type": "NetworkX graph"
    },
    "source": {
      "description": "Starting node",
      "type": "node"
    },
    "target": {
      "description": "Ending node",
      "type": "node"
    },
    "weight": {
      "description": "If this is a string, then edge weights will be accessed via the edge attribute with this key (that is, the weight of the edge joining `u` to `v` will be ``G.edges[u, v][weight]``). If no such edge attribute exists, the weight of the edge is assumed to be one. If this is a function, the weight of an edge is the value returned by the function. The function must accept exactly three positional arguments: the two endpoints of an edge and the dictionary of edge attributes for that edge. The function must return a number or None to indicate a hidden edge.",
      "type": "string or function",
      "default": "weight"
    }
  },
  "required": [
    "source",
    "target"
  ]
}
```
### 父节点问题（parent_question）:
分析图中所有节点的影响力，找到最小影响力的两个人
### 父节点输出元信息（parent_outputs_meta）:
"parent_outputs_meta": [
  {
    "output_id": 1,
    "task_type": graph_algorithm,
    "source": "run_pagerank",
    "type": dict,
    "description":"执行的是run_pagerank算法",
    "fields": [
      {"key": "node_pagerank", "type": "dict", "desc": "节点ID → PageRank数值映射，不包含图结构"}
    ]
  },
    {
    "output_id": 2,
    "task_type": post_processing,
    "source": "python code",
    "type": dict,
    "description":"返回的是pagerank值最小的两个人",
    "fields": [
      {"key": "min_influenti_two_node", "type": "dict", "desc": "pagerank值最小的两个人"}
    ]
  }
]
## Output
{
  "dependency_type": "parameter",
  "selected_outputs": [
    {
      "output_id": 2,
      "field_key": "min_influenti_two_node",
      "use_as": "parameter",
      "reason": "xxxxx"
    }
  ]
}

## Input
### 当前子问题（current_question）:
{current_question}
### 当前任务类型（task_type）:
{task_type}
### 当前计划执行的函数或图算法描述（current_algo_desc）:
{current_algo_desc}
### 父节点问题（parent_question）:
{parent_question}
### 父节点输出元信息（parent_outputs_meta）:
{parent_outputs_meta}

##Instruction: 请基于上述全部信息，判断依赖类型并精确定位依赖数据，严格按照 JSON 输出格式作答。
"""



generate_graph_conversion_code_prompt="""
你是一名图数据处理专家，任务是给定一个自然语言问题描述**current_question** 和 多个“非图结构的数据依赖项” **dependency_items**，
生成一段可执行的**Python 函数 transform_graph(...)**用以在全图数据（global_nodes, global_edges）中根据这些数据依赖项提取一张子图数据。

##任务背景
当前问题（current_question）是一个需要在图结构上执行图算法分析的自然语言问题。 这张用于计算的图并不是原始全图，而是由上游问题的输出（dependency_items）所“指示”的部分。dependency_items 是一个列表，每一项由以下几个字段组成：
- field_key:   该依赖字段的名称
- field_type:  字段的数据类型（如 list, dict, list[tuple]）
- field_desc:  对该字段的自然语言含义说明（例如“节点23所在社区的节点列表”、“路径上的边集合”）
- sample_value: 该字段的真实样本数据，帮助理解字段的内部结构。你不能在 transform_graph 中使用 sample_value 的具体取值，只能用它推断字段结构。
- parent_step_id: 该依赖项来自哪个父节点
- parent_step_question: 父节点所对应的自然语言问题
- reason: LLM 判定为什么选中该依赖项作为图依赖

##任务目标
你的任务是理解这些dependency_items, 生成一段可执行的**Python 函数 transform_graph(...)**以实现从全图数据（global_nodes, global_edges）中提取**dependency_items**所指向的子图，该函数返回抽取子图（final_nodes, final_edges）。 该函数的要求如下，必须严格遵照下面的要求:
- 把dependency_items中每个项的field_key作为参数名，参数顺序必须按照 dependency_items 列表顺序展开。
- 函数格式如下：
def transform_graph(field_key1, field_key2, field_key3, global_nodes, global_edges):
  # global_nodes: list[str]
  # global_edges: list[(str, str)]
  # 你必须在此处填入代码逻辑
  ... 
  result = {{
      "nodes": final_nodes,   # list[str]
      "edges": final_edges    # list[(str, str)]
  }}
  return result

- 注意: global_nodes 和 global_edges 是原始全图，分别表示节点id集合 和 边集合。

### 生成代码的规则: 你需要根据每个数据依赖的field_key+field_type+field_desc+sample_value来清晰理它的语义含义和内部数据结构，进一步判定该数据依赖项表示节点级别的结果还是边级别的结果，对于节点级别结果和边级别结果在代码里采用不同的构造逻辑，具体如下：
### 1. 如果依赖项表示节点级别的结果,例如：社区节点列表、某些候选节点集合、Top-K 节点列表、节点 → 数值映射 （如 pagerank, centrality）等, 处理方式按照如下进行：
- 首先你必须根据语义和该依赖项的数据结构准确地定位哪个字段表示节点 ID, 例如一个数据依赖项的形式是dict，内部的数据结构是{"1":0.123, "2":0.234}，其中key表示顶点id，value表示得分，那么key存储的就是节点ID
- 获取节点ID集合
- final_nodes = union(所有节点ID)
- final_edges = 所有满足 (u,v) 都在 final_nodes 中的边

### 2. 如果依赖项表示边级别的结果,例如: 路径集合等吧，处理方式按照如下进行：
- 首先你必须根据语义和该依赖项的数据结构准确地定位那两个字段表示source id和dst id, 例如一个数据依赖项的形式是dict，内部的数据结构是{"source":"1", "dst":"2", "weight":0.11}，其中"source"表示source id，"dst"表示dst id。 例如一个数据依赖项的形式是list[(tuple)],内部的数据结构是[("1", "2"), ("3", "4")]，则视为 (source, dst)。
- 获取边集合
- 然后基于**parent_step_question** 和 **current_question**理解问题语义,来判断属于下面哪种子类型：
  #### (1) 点诱导类型（node-induced）: 根据依赖项提供的边先确定节点集合，再从全图中取出这些节点之间的所有边构成子图。例如 parent_step_question 是找到从节点23出发的所有边， current_question是把这些边上的点组成的社区进行影响力分析，这属于**点诱导**，因为current_question明确要求在这些边的点构成的社区进行分析。
    - 处理逻辑: 先取这些边中出现的所有节点，再从 global_edges 中找这些节点之间的所有边。
  #### (2) 边诱导子图（edge-induced）: 直接使用依赖项提供的边集合作为子图的全部边，节点为这些边所涉及的所有节点。例如 parent_step_question 是找到从节点23出发的所有边，current_question是在这些边组成的图上进行影响力分析，这属于**边诱导**，因为current_question明确直接在这些边构成的图上进行分析。
    - 处理逻辑: 直接使用这些边作为 final_edges，节点为出现在这些边上的所有节点。
  ### 默认规则（非常重要）
    如果无法从 parent_step_question 和 current_question 的语义明确判断是 node-induced 还是 edge-induced，则 **默认使用 edge-induced**。
注意：
-你必须在代码中注释你的判断。
-合并多个依赖项，所有依赖项转换得到的节点与边都必须合并，合并时去重复，返回一整张子图：
  - final_nodes = union(来自每个依赖项的节点)
  - final_edges = union(来自每个依赖项的边)。
- transform_graph 必须能单独被 exec 执行，不允许写伪代码。
- 如果一个依赖项的输出是一个从语义上是边，但是实际的字段结构是点集合，那么该依赖项也属于节点级别的结果。例如，dijkstra_path算法的输出结果是List of nodes in a shortest path, 从0号点到1号点的最短路径的表示实际是一个节点集合["0", "1", "2", "3", "4"], 那么该依赖项也属于节点级别，按照节点级别的逻辑处理。

## 输出格式要求
你最终的回答 **只能输出一个 JSON**：
```json
{{
  "description": "<简要的解释你是如何根据依赖项构造子图>",
  "code": "<完整的 Python 函数 transform_graph(...) 的代码字符串>"
}}
```
## Input
### 当前子问题:
{current_question}
### 数据依赖项:
{dependency_items}

##Instruction: 你必须根据上述所有规则生成一个完整、可执行、无伪代码、严格符合格式要求的 transform_graph Python 函数，并将其作为 JSON 中的 "code" 字段返回。不得输出多余文字。
"""