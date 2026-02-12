---
sidebar_position: 5
---
# Clustering & Community Operator Set


**Operator Category**: Clustering & Community (Clustering Coefficient, Community Detection, Transitivity and Cycle Structure)

**Applicable Stages**: Network Structure Insight, Circle/Gang Identification, Community Quality Evaluation, Closed Loop/Cycle Detection, Relationship Intensity Quantification

**Product Positioning**: Provide a unified operator capability base for answering "whether a network has tight small circles / how to divide people or entities into communities / how good is this division method / whether closed loops exist"


---
## 1. Operator Set Overview
The Clustering & Community operator set covers two major categories of capabilities:
1. **Local Intensity and Network Aggregation (Clustering / Transitivity)**
   - Are nodes/the entire network "clustered"?
   - Is the tendency for triangle closure strong?
   - Suitable for judging whether a network has small circles, gangs, and strong relationship groups from the perspective of "micro local structure".
2. **Community Detection & Evaluation**
   - Automatically find communities/camps/circles without labels.
   - Support different mechanisms: modularity maximization, label propagation, hierarchical splitting, k-clique percolation (overlapping communities).
   - Conduct quality scoring and validity verification for the obtained community division.

   In addition, this category also includes **cycle structure/cycle detection** capabilities (simple_cycles / cycle_basis), which are used to identify closed-loop patterns such as fund reflux, circular dependency, and feedback loops.
---
## 2. Operator Capability Classification
| Capability Type | Corresponding Operator | Function Description |
|---|---|---|
| Node Clustering Coefficient | `clustering` | Calculate the clustering coefficient of specified nodes or all nodes |
| Average Clustering Coefficient | `average_clustering` | Calculate the average clustering coefficient of the entire graph or a subset of nodes |
| Triangle Closure Count | `triangles` | Count the number of triangles a node participates in |
| Global Transitivity | `transitivity` | Calculate the overall degree of "a friend of a friend is also a friend" in the network |
| 4-Cycle Clustering | `square_clustering` | Calculate the local redundant structure tendency of nodes participating in 4-cycles |
| Community Detection - Greedy Modularity | `greedy_modularity_communities` | Clauset–Newman–Moore greedy merging to maximize modularity |
| Community Detection - Hierarchical Splitting | `girvan_newman` | Iteratively remove "the most critical edges" to obtain hierarchical community structures |
| Community Detection - Synchronous Label Propagation | `label_propagation_communities` | Form communities through majority label diffusion of neighbors |
| Community Detection - Asynchronous Label Propagation | `asyn_lpa_communities` | Asynchronous update label propagation (controllable weight and random seed) |
| Community Detection - k-clique | `k_clique_communities` | Overlapping community detection based on k-clique percolation |
| Community Detection - Louvain | `louvain_communities` | Multi-layer modularity optimization, suitable for fast community division of large graphs |
| Community Detection - Leiden | `leiden_communities` | Improved version of Louvain, more stable and ensures internal connectivity of communities |
| Cycle Structure - Simple Cycle Enumeration | `simple_cycles` | Enumerate all simple cycles in the graph (can set length upper limit) |
| Cycle Structure - Cycle Basis | `cycle_basis` | Basic cycle set of undirected graphs (cycle space basis) |
| Solution Quality - Modularity | `modularity` | Calculate modularity(Q) for a given partition |
| Solution Quality - Coverage/Performance | `partition_quality` | Calculate (coverage, performance) |
| Solution Validity | `is_partition` | Check whether the community list is a strict partition (non-overlapping and full node coverage) |
---
## 3. General Input and Output Conventions
- **Input `G`**: NetworkX Graph / DiGraph
- **Output Type**:
  - Indicator Type (Clustering/Transitivity/Modularity): `float`
  - Count Type (Triangles): `dict` or `int`
  - Community Result: `list[set]` / `iterable[set]` / `iterator[tuple[set]]`
  - Cycle Structure: `list[list[node]]` or iterator
  - Verification: `bool`
---
## 4. Detailed Operator Descriptions
### 4.1 Local Intensity and Network Aggregation
#### 1) clustering —— Node Clustering Coefficient
**Function Description**
Calculate the node clustering coefficient: measure the proportion of mutual connections between the neighbors of a node ("whether neighbors know each other").
**Product Value**
- Find central nodes of "tight friend circles"
- Candidate core points of gangs/tight groups
**Key Parameters**
- `nodes`: single node / multiple nodes / None (entire graph)
- `weight`: weighted clustering (edge weight represents relationship intensity)
**Complexity**: `O(V * d^2)` (d is the average degree)
---
#### 2) average_clustering —— Average Clustering Coefficient
**Function Description**
Return the average clustering coefficient of the entire graph or the specified node set, with a value ranging from 0 to 1.
**Product Value**
- Summarize the "overall clustering degree" in one sentence
- Used for network comparison, version comparison, and regional comparison
**Key Parameters**
- `nodes`: only view a specific subgroup
- `weight`: consider relationship intensity
- `count_zeros`: whether to include nodes with a clustering coefficient of 0 in the average
**Complexity**: `O(V * d^2)`
---
#### 3) triangles —— Triangle Count Statistics
**Function Description**
Count the number of triangles each node participates in, or return the number of triangles for the specified node.
**Product Value**
- More triangles mean the node is in a "closed small circle"
- Can be used as structural features in anti-fraud/gang identification
**Key Parameters**
- `nodes`: single node / multiple nodes / None (entire graph)
**Complexity**: `O(V * d^2)` or `O(E^{1.5})` (implementation-dependent)
---
#### 4) transitivity —— Global Transitivity
**Function Description**
Global indicator: the proportion of closed triplets (triangles) to all triplets, measuring the overall degree of "triangle closure".
**Product Value**
- Whether "a friend of a friend is more likely to be a friend" holds for the entire network
- Judge whether the network is more like a random network or a small-world network (tendency indicator)
**Complexity**: `O(V * d^2)`
---
#### 5) square_clustering —— 4-Cycle Clustering Coefficient
**Function Description**
Measure the tendency of nodes to participate in "4-cycle" structures, often used in bipartite graphs or alternative path redundancy analysis.
**Product Value**
- Discover redundant relationships of "connecting the same target through two different paths"
- Often more meaningful than triangles in bipartite graphs (e.g., user-commodity)
**Key Parameters**
- `nodes`: only calculate a subset of nodes
**Complexity**: `O(V * d^2)`
---
### 4.2 Community Detection
#### 6) greedy_modularity_communities —— Greedy Modularity Communities
**Function Description**
Maximize modularity through a greedy merging strategy, and output a list of communities (sorted by size).
**Applicable Characteristics**
- Fast speed and classic implementation
- Suitable for medium and large-scale undirected graphs, supports edge weights
**Key Parameters**
- `weight`: edge weight
- `resolution`: control community scale (&lt;1 for larger communities; &gt;1 for smaller communities)
- `cutoff` / `best_n`: control stop conditions and community quantity range
---
#### 7) girvan_newman —— Girvan–Newman Hierarchical Communities
**Function Description**
Iteratively remove "the most critical edges" (default to edges with the highest betweenness) to split the graph gradually and obtain a hierarchical community structure (from coarse to fine).
**Product Value**
- Suitable for highly interpretable "structural splitting"
- Can output multi-level solutions (suitable for small graphs/need for hierarchical structures)
**Key Parameters**
- `most_valuable_edge`: customize the scoring/selection method of "the most critical edge"
**Complexity**: Relatively high (`O(E^2 * V)`), not suitable for ultra-large graphs
---
#### 8) label_propagation_communities —— Synchronous Label Propagation
**Function Description**
Label diffusion based on majority voting of neighbors, no optimization objective required, usually fast in speed.
**Product Value**
- Fast rough clustering of ultra-large graphs
- Suitable for scenarios of "obtaining results first and then refining"
**Complexity**: `O(V + E)` per round (number of iterations depends on the graph)
---
#### 9) asyn_lpa_communities —— Asynchronous Label Propagation
**Function Description**
Similar to LPA, but adopts an asynchronous update order, allows reproducibility control through random seeds, and can use weights to affect label frequency.
**Key Parameters**
- `weight`: edge weight affects the "occurrence frequency" of neighbor labels
- `seed`: random state, affects result stability/reproducibility
---
#### 10) k_clique_communities —— k-clique Percolation (Overlapping Communities)
**Function Description**
Takes k-clique (complete subgraph) as the basic unit. Two k-cliques are considered adjacent if they share k-1 nodes, thus forming "percolation communities". Nodes can belong to multiple communities (overlapping).
**Product Value**
- Find "very tight" core circles
- Allow member overlap, consistent with real social/collaboration networks
**Key Parameters**
- `k`: minimum clique size (larger values mean stricter, smaller and tighter communities)
- `cliques`: precomputed clique list can be passed in (can significantly save repeated calculations)
**Complexity**: Affected by clique enumeration, usually exponential (suitable for small and medium graphs or local subgraphs)
---
#### 11) louvain_communities —— Louvain Communities
**Function Description**
Classic multi-layer modularity optimization method, fast speed, suitable for large-scale undirected graphs (supports weights).
**Key Parameters**
- `weight`: edge weight
- `resolution`: community scale
- `threshold` / `max_level` / `seed`: convergence threshold, number of layers, randomness control
**Characteristics**
- One of the "default first choices" commonly used in the industry
- Results may change with randomness (seed-controllable)
---
#### 12) leiden_communities —— Leiden Communities
**Function Description**
Improved version of Louvain, usually more stable, and emphasizes the internal connectivity of communities (avoids "bad communities that look like a group but are internally disconnected").
**Key Parameters**
- `weight` / `resolution`
- `max_level` / `seed`
**Characteristics**
- More stable quality, suitable for scenarios with rigorous requirements for community structure
---
### 4.3 Cycle Structure and Cycle Detection
#### 13) simple_cycles —— Simple Cycle Enumeration
**Function Description**
Enumerate all simple cycles in the graph (no repeated nodes, start point = end point). `length_bound` can be set to limit the cycle length.
**Product Value**
- Discover fund reflux and circular transactions
- Discover circular dependencies in software dependencies
- Discover feedback loops in biological metabolic/regulatory networks
**Key Parameters**
- `length_bound`: limit cycle length to avoid explosive output
**Complexity**: `O((V + E) * (C + 1))` (C is the number of cycles)
---
#### 14) cycle_basis —— Cycle Basis (Undirected Graph)
**Function Description**
Output a cycle basis for an undirected graph: a "basic independent cycle set" that can be combined to generate all cycles.
**Product Value**
- Circuit grid analysis (Kirchhoff's laws)
- Identification of basic closed blocks in road networks
- Decompose complex cycle structures into basic components
---
### 4.4 Community Result Evaluation and Verification
#### 15) modularity —— Modularity Score (Q Value)
**Function Description**
Calculate modularity for a given community partition: measure the degree of "more intra-group edges and fewer inter-group edges".
**Key Parameters**
- `communities`: must be a partition of nodes (mutually exclusive and fully covering)
- `weight` / `resolution`
**Output**: `float` (usually larger is better, but needs to be understood in combination with network type and resolution)
---
#### 16) partition_quality —— Coverage and Performance
**Function Description**
Output `(coverage, performance)`:
- coverage: proportion of intra-community edges (whether there are many intra-community edges)
- performance: proportion of intra-community edges + extra-community non-edges (how good the separation is)
---
#### 17) is_partition —— Partition Validity Verification
**Function Description**
Check whether the given community list is a strict partition:
- Whether it covers all nodes
- Whether there is no mutual overlap
**Note**
- For algorithms that allow overlapping communities (e.g., `k_clique_communities`), their output usually **does not satisfy** the partition definition, and `is_partition` should not be used for forced verification in this case.
---
## 5. Recommended Usage Guide (Selection Suggestions)
- **Want to first measure whether the network is "clustered"**: `average_clustering` + `transitivity`
- **Want to find local tight core points/gang features**: `clustering` + `triangles` (`square_clustering` if necessary)
- **Want to quickly obtain communities (priority for large graphs)**: `louvain_communities` or `leiden_communities`
- **Want more interpretable hierarchical splitting (small graphs)**: `girvan_newman`
- **Want extremely fast rough clustering (ultra-large graphs)**: `label_propagation_communities` / `asyn_lpa_communities`
- **Want to find overlapping, extremely tight small circles**: `k_clique_communities`
- **Want to score the partition**: `modularity` + `partition_quality` (verify with `is_partition` first)
- **Want to check closed loops/circular dependencies/fund reflux**: `simple_cycles` (add `length_bound` if necessary), use `cycle_basis` for undirected graphs
---
## 6. Typical Answerable Questions (Examples)
- "Is the entire network more like a loose random network or a small-world network? Give me an overall indicator."
- "Which nodes have the tightest friend circles? List the Top-20 clustering coefficients/triangle counts."
- "Automatically divide the network into several natural communities and output the members of each community."
- "I have two community division schemes, which one is better? Provide modularity and coverage/performance."
- "Is this community result a strict partition? Are there any missing nodes/duplicate assignments?"
- "Find all fund closed loops/circular dependency links (can limit cycle length)."
- "Find overlapping core small circles (communities connected by iron triangles)."
---
