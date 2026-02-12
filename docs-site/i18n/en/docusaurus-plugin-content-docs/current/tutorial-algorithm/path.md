---
sidebar_position: 2
---
# Path Operator Set


**Operator Category**: Path (Path and Reachability)

**Applicable Stages**: Route planning, dependency chain analysis, reachability determination, global distance evaluation, tour/traversal design

**Product Positioning**: Provides a unified path capability base for "how to take the shortest/most economical route", "whether there is a reachable path", "how to traverse all edges", and "what is the longest dependency chain/critical path in a DAG".


---
## 1. Operator Set Overview
The Path operator set targets various graph structures (traffic road networks, communication topologies, call dependencies, transaction links, knowledge citation networks, etc.) and covers three core types of problems:
1. **Single-source/Single-pair Shortest Path**: The shortest path/shortest distance from A to B (`shortest_path`, `dijkstra_path`, `dijkstra_path_length`, `bellman_ford_path`)
2. **All-pairs Shortest Path/Average Distance**: Shortest path distance matrix between any two nodes, network "average hop count/average distance" (`floyd_warshall`, `johnson`, `average_shortest_path_length`)
3. **Reachability and Structural Path**: Existence of a path, longest path in DAG (critical path), Euler path/circuit (traversing each edge exactly once) (`has_path`, `dag_longest_path`, `dag_longest_path_length`, `is_eulerian`, `eulerian_path`, `eulerian_circuit`)
---
## 2. Capability Classification and Selection Suggestions
| Objective | Recommended Operators | Application Scenarios |
|---|---|---|
| Shortest "path sequence" from A→B | `shortest_path` / `dijkstra_path` / `bellman_ford_path` | Need specific travel routes (node sequences) |
| Shortest "distance value" from A→B | `dijkstra_path_length` | Only need the shortest distance/cost |
| Negative weight edges may exist | `bellman_ford_path` (single-pair/single-source) or `johnson` (all-pairs) | E.g., subsidies, rebates, negative revenue, etc.; **no negative cycles allowed** |
| Shortest path distances between any two nodes (dense graph) | `floyd_warshall` | Moderate number of nodes (V), with many edges |
| Shortest path distances between any two nodes (sparse graph) | `johnson` | Large V, relatively sparse E; supports negative weights (no negative cycles) |
| Graph "average shortest path distance/average hop count" | `average_shortest_path_length` | Evaluate network efficiency, small-world property, etc. |
| Only determine "reachability" | `has_path` | Need Yes/No reachability result |
| Critical path/longest dependency chain of DAG | `dag_longest_path` / `dag_longest_path_length` | Scheduling, project management, dependency orchestration |
| Traverse each edge exactly once (open/closed) | `eulerian_path` / `eulerian_circuit` (+ first judge with `is_eulerian`) | Patrol inspection, road coverage, edge-by-edge link tracking |
---
## 3. General Input and Output Conventions
- **Input `G`**: NetworkX graph object (directed/undirected; some algorithms support multigraph; some support weights)
- **Weight Interpretation**:
  - Shortest path category: `weight` is usually interpreted as **distance/cost/time** (smaller is better)
  - DAG longest path category: `weight` is interpreted as **duration/revenue/cumulative cost** (larger is "longer")
- **Output**:
  - Path: Node sequence `list[node]`
  - Distance matrix/all-pairs shortest path: `dict[source][target] -> dist` or `dict[(source,target)] -> dist`
  - Euler path/circuit: Edge sequence iterator (or convertible to node sequence)
> Note: Many shortest path functions treat parallel edges as multiple candidate edges on **multigraph**; if edge keys need to be distinguished, use `keys=True` or explicitly process when reading edge attributes.
---
## 4. Detailed Operator Descriptions
### 4.1 shortest_path —— General Shortest Path (Single-pair / Single-source / All-source)
**Function Description**
Calculates the shortest path between two or more nodes in the graph (returns node sequence). Treated as an unweighted graph (each edge has a length of 1) when `weight=None`, otherwise uses `weight` as the edge weight.
**Typical Scenarios**
- GPS/Logistics: Shortest distance/shortest time-consuming route from A to B
- Citation/Dependency chain: Shortest link from the base node to the target node (minimum intermediate nodes)
- Network routing: Minimum latency/minimum cost transmission path
**Key Parameters (Tuning)**
- `source` / `target`: Specifying them can significantly reduce computation and output scale
- `weight`: None (unweighted) / field name / function
- `method`: `dijkstra` or `bellman-ford` (for weighted graphs)
**Complexity**
- Unweighted: Usually `O(V+E)` (BFS)
- Weighted (Dijkstra): Typically `O(E + V log V)` (heap optimization)
- Weighted (Bellman-Ford): Approximately `O(VE)`
**Directly Answerable Questions (Examples)**
- "Find the shortest path from A to B and output the node sequence."
- "What are the shortest paths from the starting point S to all reachable nodes?"
---
### 4.2 dijkstra_path —— Dijkstra Shortest Path (Returns Path Sequence)
**Function Description**
Calculates the shortest path (node sequence) from `source` to `target` in a graph with **non-negative edge weights**.
**Product Value**
- One of the most commonly used shortest path algorithms in engineering, with good performance and clear interpretability
- Supports weight field names or custom weight functions (edges can be hidden by returning `None`)
**Key Parameters (Tuning)**
- `weight`: Field name or function (commonly used: distance/time/cost)
**Complexity**
`O(E + V log V)` (typical implementation)
---
### 4.3 dijkstra_path_length —— Dijkstra Shortest Path Length (Returns Only Distance)
**Function Description**
Same as Dijkstra, but only outputs the shortest distance/cost value (does not return the specific node sequence).
**Typical Scenarios**
- Only care about the "minimum cost/minimum latency/shortest kilometer distance" without needing route details
**Complexity**
`O(E + V log V)`
---
### 4.4 bellman_ford_path —— Bellman-Ford Shortest Path (Supports Negative Weights)
**Function Description**
Calculates the shortest path (node sequence) from `source` to `target`, which can handle **negative weight edges** (but no negative weight cycles are allowed).
**Typical Scenarios**
- Cost models with subsidies/rebates leading to negative weights for some edges
- Risk control/Revenue: Edge weights can represent the transmission of "net cost/net revenue (take the opposite number)"
**Key Parameters (Tuning)**
- `weight`: Field name or function (default `"weight"`)
- `source` / `target`: Specify a single-pair path
**Complexity**
`O(VE)`
---
### 4.5 floyd_warshall —— Floyd-Warshall All-pairs Shortest Path (Distance Dictionary)
**Function Description**
Calculates the shortest path distances between **any two nodes** at once based on dynamic programming, returning `dist[u][v]`.
**Applicable Characteristics**
- More suitable for **dense graphs** or tasks that need to obtain a complete distance matrix at once
- High overhead when the number of nodes is large (cubic complexity)
**Key Parameters (Tuning)**
- `weight`: Edge weight field name (default `"weight"`)
**Complexity**
`O(V^3)`
---
### 4.6 johnson —— Johnson All-pairs Shortest Path (Sparse Graph Friendly, Supports Negative Weights)
**Function Description**
Calculates the shortest paths between any two nodes for **sparse graphs**. Can handle negative weight edges (no negative cycles), and is usually more suitable for large-scale sparse networks than Floyd-Warshall.
**Key Parameters (Tuning)**
- `weight`: Field name or custom function (can implement dynamic weights/rules)
**Complexity**
Approximately `O(VE + V^2 log V)` (common in engineering implementations)
---
### 4.7 average_shortest_path_length —— Average Shortest Path Length (Network Efficiency)
**Function Description**
Calculates the average value of the shortest path lengths of all reachable node pairs in the graph, used to measure the overall "connectivity efficiency/average hop count" of the network.
**Typical Scenarios**
- Social networks: Average number of intermediaries required
- Communication/Traffic networks: Average latency/average transfer times
- Complex network analysis: Small-world property, overall reachability
**Key Parameters (Tuning)**
- `weight`: None/field name/function (determines whether to use unweighted or weighted distance)
- `method`: `unweighted` / `dijkstra` / `bellman-ford` / `floyd-warshall`, etc.
**Complexity (Depends on method and graph structure)**
- Commonly `O(V*(V+E))` (multiple unweighted BFS) or `O(V*(E + V log V))` (multiple Dijkstra)
---
### 4.8 has_path —— Reachability Determination (Yes/No)
**Function Description**
Determines whether there is a path (reachability) between two nodes in the graph. Commonly uses DFS/BFS.
**Typical Scenarios**
- Quick verification of "Can A reach B?"
- Connectivity constraint check (whether the dependency chain is reachable, whether the road network is broken)
**Complexity**
`O(V+E)` (worst case)
---
### 4.9 dag_longest_path —— DAG Longest Path (Returns Node Sequence)
**Function Description**
Calculates the longest path in a **Directed Acyclic Graph (DAG)** (cumulative by edge weight, supports default weights). A topological order can be provided for acceleration or reuse.
**Typical Scenarios**
- Project management/Scheduling: Critical Path
- Workflow optimization: Longest dependency chain, maximum cumulative time-consuming chain
- Citation chain: Longest "sequential reading chain" from basic papers to cutting-edge reviews
**Key Parameters (Tuning)**
- `weight`: Edge weight field name (default `"weight"`)
- `default_weight`: Default weight for unweighted edges (default 1)
- `topo_order`: Externally provided topological order (algorithm calculates internally if None)
**Complexity**
`O(V+E)` (linear, suitable for large DAGs)
---
### 4.10 dag_longest_path_length —— DAG Longest Path Length (Returns Length Value)
**Function Description**
Similar to `dag_longest_path`, but only returns the cumulative weight/length of the longest path.
**Key Parameters (Tuning)**
- `weight`, `default_weight`
**Complexity**
`O(V+E)`
---
### 4.11 is_eulerian —— Existence of Eulerian Circuit
**Function Description**
Determines whether a graph is an Eulerian graph, i.e., whether there exists a circuit that **starts and ends at the same node** and **traverses each edge exactly once**.
**Key Judgment Points (Intuitive Understanding)**
- Undirected graph: All nodes have even degrees (and the relevant part is connected)
- Directed graph: In-degree = out-degree (and the relevant part is strongly connected/connected to meet conditions)
**Complexity**
`O(V+E)`
---
### 4.12 eulerian_path —— Eulerian Path (Open Trail)
**Function Description**
Returns a path that traverses each edge exactly once when conditions are met (no requirement for the start and end nodes to be the same).
Common existence conditions: An undirected graph has exactly 0 or 2 nodes with odd degrees; a directed graph satisfies the in-out degree difference constraints, etc.
**Typical Scenarios**
- Garbage collection/Patrol inspection: Expect to "travel each road once" without forcing a return to the starting point
- Circuit/Wiring: Cover all connections at once without repetition
**Complexity**
`O(E)` (constructive algorithm, nearly linear in common cases)
---
### 4.13 eulerian_circuit —— Eulerian Circuit (Closed Trail, Returns Edge Sequence Iterator)
**Function Description**
Returns an Eulerian circuit: starting from `source`, traversing each edge exactly once and returning to `source`. Supports directed/undirected graphs and multigraphs; can choose whether to output edge keys.
**Key Parameters (Tuning)**
- `source`: Specify the starting point; choose any if not specified
- `keys`: Whether to output `(u,v,k)` to distinguish parallel edges in multigraph
**Complexity**
`O(E)`
---
## 5. Summarized Examples of Directly Answerable Questions
- "What is the shortest path from A to B? Please output the node sequence."
- "What is the length of the shortest path from A to B (based on cost weight)?"
- "Does the graph have an Eulerian circuit? If yes, please give the traversal order of one circuit."
- "In the DAG, which nodes are included in the longest dependency chain (critical path)? What is the total duration?"
- "Please output the average shortest path length of the network to evaluate the overall connectivity efficiency."
- "Please calculate the shortest path distance matrix between any two nodes (all-pairs shortest path)."
---
## 6. Engineering Implementation Notes
1. **Unified weight semantics**: For shortest paths, smaller weights are better; for longest paths (DAG), larger weights are "longer". Clearly define fields/positive and negative signs when reusing the same graph.
2. **Selection for negative weight edges**: Dijkstra is not applicable if negative weight edges exist; choose Bellman-Ford (single-source/single-pair) or Johnson (all-pairs).
3. **Algorithm selection based on scale**:
   - Dense graph, moderate V: `floyd_warshall`
   - Sparse graph, large V: `johnson`
   - Single-pair: `dijkstra_path` / `bellman_ford_path`
4. **Pre-judgment for Eulerian algorithms**: In actual engineering, it is recommended to first use `is_eulerian` or check degree conditions before solving for `eulerian_path/circuit` to avoid exceptions and no-solution situations.
