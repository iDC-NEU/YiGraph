---
sidebar_position: 7
---
# Flow & Cut Operator Set


**Operator Category**: Flow & Cut (Network Flow and Cut)

**Description**: Used to solve problems such as "What is the maximum flow capacity?", "Where are the bottlenecks?", "What is the minimum cost to disconnect the network?", and "What is the minimum cut structure between any two nodes?". It is a core algorithm family for network capacity analysis, resource scheduling, and system resilience assessment.


---
## 1. Overview of the Operator Set
The Flow & Cut operator set mainly focuses on three core types of problems:
1. **Max Flow**
   - What is the maximum flow that can pass from the source node to the sink node?
   - Which edges/nodes will become bottlenecks first?
2. **Min Cut**
   - Which edges (or capacities) need to be cut at minimum to block connectivity?
   - Where is the "weakest point" of the network?
3. **Global Cut Structure**
   - What is the minimum cut between any two nodes?
   - Can all pairwise minimum cuts be compressed and represented by a tree?
---
## 2. Operator List
| Operator | Core Capability |
|---|---|
| `edmonds_karp` | Classic max flow algorithm based on BFS-augmented paths |
| `shortest_augmenting_path` | Shortest augmenting path max flow, suitable for medium-to-high density networks |
| `preflow_push` | Preflow push algorithm, suitable for high-concurrency/high-connection complex networks |
| `capacity_scaling` | Capacity-scaled minimum cost/max flow problems |
| `gomory_hu_tree` | Compressed representation of all-pairs minimum cut structure (cut tree) |
---
## 3. General Input and Output Conventions
### 3.1 Input
- **G**: Directed or undirected graph (some algorithms apply only to directed graphs)
- **capacity**: Edge capacity attribute name (default: `"capacity"`)
- **s / t**: Source and sink nodes (for max flow related algorithms)
- **demand / weight**: Used for minimum cost flow and capacity scaling problems
### 3.2 Output
- **Max flow algorithms**: Residual Network or flow value
- **Capacity/cost algorithms**: Optimal flow scheme + total cost
- **Gomory–Hu Tree**: An undirected tree encoding all pairwise minimum cuts
---
## 4. Detailed Operator Descriptions
### 4.1 edmonds_karp — Classic Max Flow Algorithm
#### Function Description
Based on the Ford–Fulkerson method, it always selects the **shortest (by number of edges) augmenting path** to gradually increase the flow from the source node `s` to the sink node `t`.
#### Applicable Scenarios
- Max flow calculation for small and medium-scale networks
- Flow analysis with strong teachability, debuggability and interpretability
- Scenarios that require output of complete augmenting paths and residual networks
#### Parameter Key Points
- **capacity**: Edge capacity attribute
- **cutoff**: Early termination threshold (stop when a certain flow is reached)
#### Principle and Complexity
- Uses BFS to find augmenting paths each time
- **Time Complexity**: `O(V · E²)`
#### Answerable Questions
- What is the maximum transport capacity from A to B?
- Which pipelines/links are fully utilized under the maximum flow?
### 4.2 shortest_augmenting_path — Shortest Augmenting Path Max Flow
#### Function Description
Uses distance labeling to prioritize the selection of the "minimum cost" augmenting path to improve efficiency.
#### Applicable Scenarios
- Medium to large-scale networks
- Max flow solution requiring higher speed than Edmonds–Karp
- Backbone networks of telecommunications, transportation and logistics
#### Parameter Key Points
- **two_phase**: Significantly accelerates unit capacity networks
- **cutoff**: Minimum acceptable flow threshold
#### Principle and Complexity
- Augmenting path search based on the shortest path idea
- **Complexity**: `O(V² · E)` (actual performance is usually better)
### 4.3 preflow_push — Preflow Push Algorithm
#### Function Description
Does not require flow conservation at intermediate nodes, allows "preflow", and gradually approaches the maximum flow through **Push / Relabel** operations.
#### Applicable Scenarios
- Extremely complex and highly connected large-scale networks
- High-concurrency and high-density graphs (e.g., data centers, chip routing)
- Max flow calculation with extremely high performance requirements
#### Parameter Key Points
- **global_relabel_freq**: Global relabeling frequency (an important performance tuning parameter)
- **value_only**: Only concerned with the max flow value, not the path
#### Principle and Complexity
- Local push + height labeling
- **Complexity**: `O(V² · E)` (often outperforms path-based algorithms in practice)
### 4.4 capacity_scaling — Capacity Scaling Flow Algorithm
#### Function Description
Gradually scales up the available capacity by capacity levels (Δ-scaling) to solve flow problems with **demand and cost constraints**.
#### Applicable Scenarios
- Supply and demand balance problems (multi-source and multi-sink)
- Cost-optimal transportation/scheduling problems
- Energy, logistics and cloud resource allocation
#### Parameter Key Points
- **demand**: Node demand attribute (negative for supply, positive for demand)
- **capacity**: Edge capacity attribute
- **weight**: Unit flow cost
#### Principle and Complexity
- Finds the optimal flow by relaxing capacity limits level by level
- **Complexity**: `O(E² log C)`
### 4.5 gomory_hu_tree — Gomory–Hu Cut Tree
#### Function Description
Represents the **minimum cut value between any two nodes** in the graph with a **tree**, supporting fast two-node cut/flow queries.
> The **minimum edge weight** on the path between any two nodes in the tree = the minimum cut of the two nodes in the original graph.
#### Applicable Scenarios
- Global network resilience analysis
- Finding the "weakest connection pair"
- Answering minimum cut/max flow queries between any two nodes quickly
#### Parameter Key Points
- **capacity**: Edge capacity attribute
- **flow_func**: Underlying max flow algorithm (`edmonds_karp` recommended for sparse graphs, `shortest_augmenting_path` for dense graphs)
#### Principle and Complexity
- Constructs max flow for `n-1` times
- **Complexity**: `O(V · MaxFlow)`
---
## 5. Selection Guide (How to Choose)
- **Calculating s → t max flow (small/medium graphs)**: `edmonds_karp`
- **Faster max flow calculation (larger graphs)**: `shortest_augmenting_path`
- **Extremely complex and high-concurrency networks**: `preflow_push`
- **With supply-demand + cost constraints**: `capacity_scaling`
- **One-time support for any two-node minimum cut queries**: `gomory_hu_tree`
---
## 6. Typical Answerable Questions
- "What is the maximum carrying capacity of this network from A to B?"
- "Which links are the bottlenecks of the system?"
- "Which connections need to be broken to disconnect the system at the minimum cost?"
- "For any two nodes, what is the minimum capacity to cut to isolate them?"
- "Which pair is the most vulnerable connection in the entire network?"
---