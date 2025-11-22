analyze_dependency_type_and_locate_dependency_data_prompt = """
You are an expert in graph data analysis.
Your task is to determine whether the current DAG sub-question depends on the computed results of its parent node, and precisely locate the corresponding dependency source.You will receive the following variables:
-*current_question*: the natural-language description of the current sub-question
-*task_type*: the task type of the current sub-qeustion
-*current_algo_desc*: the function/graph algorithm description planned to be executed for the current sub-qeustion.
-*parent_question*: the natural-language description of the parent question
-*parent_outputs_meta* the metadata of the parent question’s output items

Your job is to determine whether the current question depends on the parent question node’s results, and specify the details as follows.
##[1] dependency_type (choose exactly one)
- *graph*: The current question needs a subgraph derived from the parent’s output, instead of the default full graph automatically injected by the system. Parent outputs may contain: node sets, edge sets, community node lists, or graph fragments. These must be transformed into a graph before the current computation.
- *parameter*: The current question needs numerical values, node IDs, or other scalar/collection data extracted from the parent output. If task_type = graph_algorithm, these values fill parameters of current_algo_desc (e.g., source, target, threshold). If task_type = numeric_analysis, these values participate in code execution or numerical computation. (Note: numeric_analysis may still require a graph if the question explicitly asks for operations on a subgraph.)
- *both*: The current question requires both a subgraph and parameter values from the parent output.
- *none*: The current question does not depend on the parent node.

##[2] selected_outputs: precisely locate the dependency source
This must be a list. Each item refers to one dependent StepOutputItem in the parent node. Each item must contain:
- *output_id*: ID of the parent StepOutputItem
- *use_as*: "graph" or "parameter". If dependency_type = "parameter" → all use_as must be "parameter". If dependency_type = "graph" → all use_as must be "graph". If dependency_type = "both" → at least one "graph" and one "parameter"
- *field_key*: the field name in the parent output
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
      {{"key": "communities", "type": "list", "desc": "community list"}}
    ]
  }},
  {{
    "output_id": 2,
    "task_type": "post_processing",
    "source": "python code",
    "type": "dict",
    "description":"The returned value is the community where node 23 is located after filtering",
    "fields": [
      {{"key": "node23community", "type": "list", "desc": "list of node IDs in the community where node 23 is located"}}
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
      "field_key": "node23community",
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
      {{"key": "node_pagerank", "type": "dict", "desc": "NodeID → PageRank value mapping"}}
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
{{current_question}}
### dependency_items:
{{dependency_items}}

## Instruction:
You must generate a complete, executable, non-pseudo-code Python function `transform_graph` strictly following all rules above, and return it inside the "code" field of the JSON output. Do not output any additional text outside the JSON.
"""