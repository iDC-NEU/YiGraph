---
sidebar_position: 4
---
# Connectivity & Components Operator Set


**Operator Category**: Connectivity & Components (Connectivity, Connected Components and Cut Analysis)

**Applicable Stages**: Network structure health check, isolated island/partition identification, robustness assessment, key node/edge localization, fault/attack surface analysis

**Product Positioning**: Provides a unified capability base for answering "whether the network is connected / how many partitions it has / which nodes or edges will split the network when removed / the minimum number of cuts required to disconnect the network"


---
## 1. Operator Set Overview
The Connectivity & Components operator set focuses on connectivity analysis for **undirected and directed graphs**, covering three core types of problems:
1. **Connectivity and Connected Components**
   - Is an undirected graph fully connected? How many connected components does it have? (is_connected / connected_components / number_connected_components)
   - Is a directed graph strongly connected or weakly connected in a directional sense? (is_strongly_connected / strongly_connected_components / is_weakly_connected / weakly_connected_components)
2. **Network Robustness (Connectivity Metrics)**
   - What is the minimum number of **nodes** to remove to disconnect the network? (node_connectivity)
   - What is the minimum number of **edges** to remove to disconnect the network? (edge_connectivity)
3. **Cuts and Critical Structures (Cuts / Critical Elements)**
   - What is the minimum node cut/edge cut set? (minimum_node_cut / minimum_edge_cut)
   - Which nodes are articulation points? Which structures are the blocks decomposed by bridges? (articulation_points / bridge_components)
---
## 2. Operator Capability Classification
| Capability Type | Corresponding Operator | Function Description |
|---|---|---|
| Undirected graph global connectivity | `is_connected` | Judge whether an undirected graph is a single connected component |
| Undirected graph connected components | `connected_components` | Output the node set of each connected component |
| Undirected graph component count | `number_connected_components` | Return the number of connected components |
| Directed graph strong connectivity | `is_strongly_connected` | Whether every pair of nodes is mutually reachable |
| Directed graph strongly connected components | `strongly_connected_components` | Output SCC (Strongly Connected Component) partitioning |
| Directed graph weak connectivity | `is_weakly_connected` | Whether connected after ignoring edge directions |
| Directed graph weakly connected components | `weakly_connected_components` | Output WCC (Weakly Connected Component) partitioning |
| Robustness metric (node) | `node_connectivity` | Minimum number of nodes to remove to disconnect the graph (or s-t) |
| Robustness metric (edge) | `edge_connectivity` | Minimum number of edges to remove to disconnect the graph (or s-t) |
| Minimum node cut set | `minimum_node_cut` | Provide the minimum node set that disconnects the graph (or s-t) when removed |
| Minimum edge cut set | `minimum_edge_cut` | Provide the minimum edge set that disconnects the graph (or s-t) when removed |
| Critical nodes (articulation points) | `articulation_points` | Nodes in an undirected graph that increase the number of components when deleted |
| Bridge-connected component decomposition | `bridge_components` | 2-edge-connected components bounded by "bridges (cut edges)" |
---
## 3. General Input and Output Conventions
- **Input `G`**: NetworkX Graph / DiGraph (per algorithm requirements: some only support undirected/directed graphs)
- **Output**:
  - Judgment type: `bool`
  - Component type: `generator[set(node)]`
  - Connectivity metric: `int`
  - Minimum cut: `set(node)` or `set(edge)`
  - Articulation points: iterator (convertible to list)
  - bridge_components: 2-edge-connected component generator (each item is a node set)
> Notes:
> - "Strong connectivity/weak connectivity" is only meaningful for **directed graphs**.
> - "Articulation points/bridge-connected components" are usually for structural robustness analysis of **undirected graphs**.
---
## 4. Detailed Operator Descriptions
### 4.1 is_connected —— Undirected Graph Connectivity Check
**Function Description**
Judge whether there is a reachable path between any two nodes in an undirected graph.
**Product Value**
- The first health check for network health
- Quickly determine the existence of isolated islands/fractured areas
**Typical Scenarios**
- Whether there are completely isolated areas in the road network
- Whether the device interconnection topology forms a single network
- Whether there are completely isolated circles in the social network
**Applicability and Characteristics**
- Graph Type: Undirected graph
- Complexity: `O(V + E)`
---
### 4.2 connected_components —— Undirected Graph Connected Component Partitioning
**Function Description**
Output the node set of each connected component in an undirected graph (a component is a "mutually reachable subnetwork").
**Product Value**
- Identify the "operational/community/topological regions" that the network is divided into
- Provide partition boundaries for subsequent statistics, modeling and scheduling within components
**Typical Scenarios**
- Logistics station network: Divide into disconnected operational regions
- Device interconnection: Find isolated subnets
- Social network: Identify non-interacting circles
**Applicability and Characteristics**
- Graph Type: Undirected graph
- Complexity: `O(V + E)`
---
### 4.3 number_connected_components —— Connected Component Count
**Function Description**
Return the number of connected components in an undirected graph.
**Product Value**
- Quickly quantify the "fragmentation degree"
- Served as a network health KPI (more components usually mean poorer health/more fragmentation)
**Typical Scenarios**
- Evaluation of partition quantity after urban road network fracture
- Number of subnets after device network failure
- Number of isolated teams in a collaboration network
**Applicability and Characteristics**
- Graph Type: Undirected graph
- Complexity: `O(V + E)`
---
### 4.4 is_strongly_connected —— Directed Graph Strong Connectivity Check
**Function Description**
Judge whether a directed graph satisfies: mutual reachability between any two nodes (both u→v and v→u are reachable).
**Product Value**
- Judge whether the system forms a closed-loop structure (capable of circulation)
- Suitable for analyzing systems with "mutual calls/mutual jumps/capital circulation"
**Typical Scenarios**
- Service calls: Whether mutually reachable to form a unified module
- Page navigation: Whether any page can reach any other page and return
- Transaction flow: Whether a reflowable closed-loop network is formed
**Applicability and Characteristics**
- Graph Type: Directed graph
- Complexity: `O(V + E)`
---
### 4.5 strongly_connected_components —— Strongly Connected Components (SCC)
**Function Description**
Output the SCC partitioning of a directed graph: any two nodes within each SCC are mutually reachable.
**Product Value**
- Identify "cyclic modules/closed-loop groups"
- Commonly used for dependency analysis, dead loop troubleshooting and modular disassembly
**Typical Scenarios**
- Service dependency graph: Find modules with mutual dependent cycles
- Flow chart: Find cyclic step groups that can return to the starting point
- Transaction network: Identify small groups with capital circulation
**Applicability and Characteristics**
- Graph Type: Directed graph
- Complexity: `O(V + E)`
---
### 4.6 is_weakly_connected —— Directed Graph Weak Connectivity Check
**Function Description**
Judge connectivity when the directed graph is treated as an undirected graph (ignoring edge directions).
**Product Value**
- Judge whether "structurally a single network", even if not mutually reachable in direction
- Suitable for scenarios with "directions but still need to check overall fragmentation"
**Typical Scenarios**
- Email sending network: Ignore directions to check if the organization is connected as a whole
- Follower network: Ignore directions to check if users are split into multiple communities
- Page link network: Ignore directions to check if the site is divided into isolated islands
**Applicability and Characteristics**
- Graph Type: Directed graph
- Complexity: `O(V + E)`
---
### 4.7 weakly_connected_components —— Weakly Connected Components (WCC)
**Function Description**
Partition the directed graph into several mutually connected components after ignoring edge directions.
**Product Value**
- Identify "structural partitions" of directional networks
- Provide boundaries for further SCC, centrality and other analyses within partitions
**Typical Scenarios**
- Follower/email network: Identify disconnected community domains
- Service calls: Identify unrelated business domains
**Applicability and Characteristics**
- Graph Type: Directed graph
- Complexity: `O(V + E)`
---
### 4.8 node_connectivity —— Node Connectivity (Robustness Metric)
**Function Description**
Return the minimum number of **nodes** to delete to disconnect the graph; if `s, t` are given, return the minimum number of nodes to delete to disconnect s and t.
**Product Value**
- Quantify the network's "anti-node failure/attack" capability
- Evaluate the redundancy of key devices/key positions
**Typical Scenarios**
- Data center: Minimum number of device failures that disconnect the network
- Urban road network: Minimum number of road junctions to close that split the network
- Collaboration network: Minimum number of personnel departures that split the team
**Key Parameters**
- `s, t`: Calculate local (source-sink) connectivity
- `flow_func`: Maximum flow implementation selection (affects performance)
**Applicability and Characteristics**
- Graph Type: Directed/Undirected (implementation based on flow)
- Complexity: Flow based
---
### 4.9 edge_connectivity —— Edge Connectivity (Robustness Metric)
**Function Description**
Return the minimum number of **edges** to delete to disconnect the graph; if `s, t` are given, return the minimum number of edges to delete to disconnect s and t.
**Product Value**
- Quantify the network's "anti-link failure/attack" capability
- Used for link redundancy planning and reinforcement
**Typical Scenarios**
- Computer room links: Minimum number of link breaks that split the network
- Road network: Minimum number of roads to close that partition the network
- Logistics network: Minimum number of line interruptions that cause supply disruption
**Key Parameters**
- `flow_func`: Maximum flow implementation selection
- `cutoff`: Stop early when the threshold is reached (acceleration but only applicable for threshold judgment)
**Applicability and Characteristics**
- Graph Type: Directed/Undirected (implementation based on flow)
- Complexity: Flow based
---
### 4.10 minimum_node_cut —— Minimum Node Cut Set
**Function Description**
Output a minimum node set that disconnects the graph when deleted; or (given s, t) disconnects s and t when deleted.
**Product Value**
- Directly provide the "most vulnerable node set"
- Used for fault drill, attack surface assessment and reinforcement priority setting
**Typical Scenarios**
- Data center: Minimum set of devices whose simultaneous failure disconnects the network
- Transportation: Minimum set of road junctions whose closure splits the network
- Power grid: Minimum set of stations whose failure partitions the network
**Key Parameters**
- `s, t`: Calculate local (source-sink) cut
- `flow_func`: Maximum flow implementation
---
### 4.11 minimum_edge_cut —— Minimum Edge Cut Set
**Function Description**
Output a minimum edge set that disconnects the graph when deleted; or (given s, t) disconnects s and t when deleted.
**Product Value**
- Directly locate the "most vulnerable link set"
- Used for link reinforcement and backup line planning
**Typical Scenarios**
- Computer room: Minimum set of links whose disconnection splits the network
- Urban road network: Minimum set of roads whose closure partitions the network
- Logistics: Minimum set of lines whose interruption causes supply disruption
**Key Parameters**
- `s, t`: Local cut
- `flow_func`: Maximum flow implementation
---
### 4.12 articulation_points —— Articulation Points (Critical Nodes)
**Function Description**
In an undirected graph, a node is an articulation point if its deletion results in an increase in the number of connected components.
**Product Value**
- Identify "single point of failure" nodes (the most typical critical nodes)
- Complementary to node_connectivity/minimum cut: articulation points provide "the most obvious structural weak points"
**Typical Scenarios**
- Urban road network: Which road junctions' closure causes traffic fragmentation
- Data center: Which device failures cause network splitting
- Social network: Which user departures split the community
**Applicability and Characteristics**
- Graph Type: Undirected graph
- Complexity: `O(V + E)`
---
### 4.13 bridge_components —— Bridge-connected Components (2-edge-connected components)
**Function Description**
Find all 2-edge-connected components in an undirected graph: within the same component, there are at least two edge-disjoint alternative paths between any two nodes (more resistant to "single edge fracture"). The algorithm decomposes the graph by identifying bridges (cut edges).
**Product Value**
- Split the network into structural blocks with "stronger internal redundancy"
- Suitable for network partitioning, resilient module identification and hierarchical reinforcement
**Typical Scenarios**
- Road network: Regional blocks with more abundant internal alternative routes
- Data center: Device groups with higher internal link redundancy
- Collaboration network: Cooperation teams with more stable internal connections
**Applicability and Characteristics**
- Graph Type: Undirected graph (supports multigraph)
- Complexity: `O(V + E)`
---
## 5. Recommended Usage Guide (Practical Suggestions)
- **First answer "Is it a single network?"**:
  - Undirected: `is_connected`
  - Directed: `is_weakly_connected` (structural) / `is_strongly_connected` (directional)
- **Then check "How many partitions and which nodes belong to each?"**:
  - Undirected: `connected_components` + `number_connected_components`
  - Directed: `weakly_connected_components` / `strongly_connected_components`
- **Conduct "resilience and break point localization"**:
  - Metrics: `node_connectivity` / `edge_connectivity`
  - Solutions: `minimum_node_cut` / `minimum_edge_cut`
  - Intuitive single points: `articulation_points`
  - Structural blocks: `bridge_components`
---
## 6. Typical Directly Answerable Questions (Examples)
- "Is this undirected graph connected? If not, how many partitions is it divided into?"
- "Is the directed graph connected after ignoring directions? Is it strongly connected in the directional sense?"
- "Output all Strongly Connected Components (SCC) and sort them by size."
- "What is the minimum number of edges/nodes to disconnect this network?"
- "Provide a minimum node cut/minimum edge cut set."
- "List all articulation points for single point of failure troubleshooting."
- "Split the network into bridge components to identify regions with more robust internal structures."

---

# Clustering & Community Operator Set
**Operator Category**: Clustering & Community (Clustering Coefficient, Community Detection, Transitivity and Cycle Structure)
**Applicable Stages**: Network structure insight, circle/gang identification, community quality assessment, closed-loop/cycle detection, relationship tightness quantification
**Product Positioning**: Provides a unified operator capability base for answering "whether the network has tight small circles / how to partition people or entities into communities / how good is the partitioning / whether closed-loop cycles exist"
---
## 7. Operator Set Overview
The Clustering & Community operator set covers two major categories of capabilities:
1. **Local Tightness and Network Aggregation (Clustering / Transitivity)**
   - Are nodes/the entire network "clustered"?
   - Is the tendency for triangle closure strong?
   - Suitable for judging whether the network has small circles, gangs and strong relationship groups from the perspective of "micro local structure".
2. **Community Detection and Scheme Evaluation (Community Detection & Evaluation)**
   - Automatically find communities/camps/circles without labels.
   - Supports different mechanisms: modularity maximization, label propagation, hierarchical splitting, k-clique percolation (overlapping communities).
   - Conduct quality scoring and validity verification for the obtained community partitioning.

In addition, this category also includes **cycle structure/cycle detection** capabilities (simple_cycles / cycle_basis), used to identify closed-loop patterns such as capital reflow, cyclic dependencies and feedback loops.
---
## 8. Operator Capability Classification
| Capability Type | Corresponding Operator | Function Description |
|---|---|---|
| Node clustering coefficient | `clustering` | Calculate the clustering coefficient of specified nodes or all nodes |
| Average clustering coefficient | `average_clustering` | Calculate the average clustering coefficient of the entire graph or a subset of nodes |
| Triangle closure count | `triangles` | Count the number of triangles a node participates in |
| Global transitivity | `transitivity` | Calculate the overall degree of "a friend of a friend is also a friend" in the network |
| 4-cycle clustering | `square_clustering` | Calculate the local redundant structure tendency of nodes participating in 4-cycles |
| Community Detection - Greedy Modularity | `greedy_modularity_communities` | Clauset–Newman–Moore greedy merging to maximize modularity |
| Community Detection - Hierarchical Splitting | `girvan_newman` | Iteratively remove "the most critical edges" to obtain hierarchical community structures |
| Community Detection - Synchronous Label Propagation | `label_propagation_communities` | Form communities through majority label diffusion from neighbors |
| Community Detection - Asynchronous Label Propagation | `asyn_lpa_communities` | Asynchronous update label propagation (controllable weight and random seed) |
| Community Detection - k-clique | `k_clique_communities` | Overlapping community detection based on k-clique percolation |
| Community Detection - Louvain | `louvain_communities` | Multi-level modularity optimization, suitable for fast community partitioning of large graphs |
| Community Detection - Leiden | `leiden_communities` | Improved version of Louvain, more stable and guarantees internal community connectivity |
| Cycle Structure - Simple Cycle Enumeration | `simple_cycles` | Enumerate all simple cycles in the graph (can set length upper limit) |
| Cycle Structure - Cycle Basis | `cycle_basis` | Basic cycle set of an undirected graph (cycle space basis) |
| Scheme Quality - Modularity | `modularity` | Calculate modularity(Q) for a given partitioning |
| Scheme Quality - Coverage/Performance | `partition_quality` | Calculate (coverage, performance) |
| Scheme Validity | `is_partition` | Check whether the community list is a strict partition (non-overlapping and full coverage) |
---
## 9. General Input and Output Conventions
- **Input `G`**: NetworkX Graph / DiGraph
- **Output Type**:
  - Metric type (clustering/transitivity/modularity): `float`
  - Count type (triangles): `dict` or `int`
  - Community result: `list[set]` / `iterable[set]` / `iterator[tuple[set]]`
  - Cycle structure: `list[list[node]]` or iterator
  - Verification: `bool`
---
## 10. Detailed Operator Descriptions
### 10.1 Local Tightness and Aggregation
#### 10.1.1 clustering —— Node Clustering Coefficient
**Function Description**
Calculate the node clustering coefficient: measure the proportion of mutual connections among a node's neighbors ("whether neighbors know each other").
**Product Value**
- Find central nodes of "tight friend circles"
- Candidate core points of gangs/dense groups
**Key Parameters**
- `nodes`: Single node / multiple nodes / None (entire graph)
- `weight`: Weighted clustering (edge weight represents relationship strength)
**Complexity**: `O(V * d^2)` (d is average degree)
---
#### 10.1.2 average_clustering —— Average Clustering Coefficient
**Function Description**
Return the average clustering coefficient of the entire graph or a specified node set, with a value range of 0~1.
**Product Value**
- Summarize the "overall clustering degree" in one sentence
- Used for network comparison, version comparison and regional comparison
**Key Parameters**
- `nodes`: Only consider a certain subgroup
- `weight`: Consider relationship strength
- `count_zeros`: Whether to include nodes with a clustering coefficient of 0 in the average
**Complexity**: `O(V * d^2)`
---
#### 10.1.3 triangles —— Triangle Count Statistics
**Function Description**
Count the number of triangles each node participates in, or return the number of triangles for specified nodes.
**Product Value**
- More triangles mean the node is in a "closed small circle"
- Can be used as structural features in anti-fraud/gang identification
**Key Parameters**
- `nodes`: Single node / multiple nodes / None (entire graph)
**Complexity**: `O(V * d^2)` or `O(E^{1.5})` (implementation dependent)
---
#### 10.1.4 transitivity —— Global Transitivity
**Function Description**
Global metric: the proportion of closed triplets (triangles) to all triplets, measuring the overall degree of "triangle closure".
**Product Value**
- Whether "a friend of a friend is more likely to be a friend" holds for the entire network
- Judge whether the network is more like a random network or a small-world network (tendency metric)
**Complexity**: `O(V * d^2)`
---
#### 10.1.5 square_clustering —— 4-cycle Clustering Coefficient
**Function Description**
Measure the tendency of nodes to participate in "4-cycle" structures, commonly used in bipartite graphs or alternative path redundancy analysis.
**Product Value**
- Discover redundant relationships with "two different paths connecting the same target"
- Often more meaningful than triangles in bipartite graphs (e.g., user-commodity)
**Key Parameters**
- `nodes`: Only calculate for a subset of nodes
**Complexity**: `O(V * d^2)`
---
### 10.2 Community Detection
#### 10.2.1 greedy_modularity_communities —— Greedy Modularity Communities
**Function Description**
Maximize modularity through a greedy merging strategy, output the community list (sorted by size).
**Applicable Characteristics**
- Fast speed and classic implementation
- Suitable for medium and large-scale undirected graphs, supports edge weights
**Key Parameters**
- `weight`: Edge weight
- `resolution`: Control community scale (&lt;1 for larger communities; &gt;1 for smaller communities)
- `cutoff` / `best_n`: Control stop conditions and community quantity range
---
#### 10.2.2 girvan_newman —— Girvan–Newman Hierarchical Communities
**Function Description**
Iteratively remove "the most critical edges" (default with the largest edge betweenness) to split the graph gradually, obtaining a hierarchical community structure (from coarse to fine).
**Product Value**
- Suitable for interpretable "structural splitting"
- Can output multi-level schemes (suitable for small graphs/needing hierarchical structures)
**Key Parameters**
- `most_valuable_edge`: Custom scoring/selection method for "the most critical edge"
**Complexity**: Relatively high (`O(E^2 * V)`), not suitable for ultra-large graphs
---
#### 10.2.3 label_propagation_communities —— Synchronous Label Propagation
**Function Description**
Label diffusion based on majority voting of neighbors, no optimization objective required, usually fast.
**Product Value**
- Fast rough clustering for ultra-large graphs
- Suitable for scenarios of "generating results first then refining"
**Complexity**: `O(V + E)` per iteration (number of iterations depends on the graph)
---
#### 10.2.4 asyn_lpa_communities —— Asynchronous Label Propagation
**Function Description**
Similar to LPA, but adopts an asynchronous update order, allows reproducibility control through random seeds, and can use weights to influence label frequency.
**Key Parameters**
- `weight`: Edge weight affects the "occurrence frequency" of neighbor labels
- `seed`: Random state, affects result stability/reproducibility
---
#### 10.2.5 k_clique_communities —— k-clique Percolation (Overlapping Communities)
**Function Description**
Takes k-clique (complete subgraph) as the basic unit, two k-cliques are considered adjacent if they share k-1 nodes, thus forming "percolation communities". Nodes can belong to multiple communities (overlapping).
**Product Value**
- Find "very tight" core circles
- Allow member overlap, consistent with real social/collaboration networks
**Key Parameters**
- `k`: Minimum clique size (larger means stricter, smaller and tighter communities)
- `cliques`: Can pass in precomputed clique list (can significantly save repeated calculations)
**Complexity**: Affected by clique enumeration, usually exponential (suitable for small and medium graphs or local subgraphs)
---
#### 10.2.6 louvain_communities —— Louvain Communities
**Function Description**
Classic multi-level modularity optimization method, fast speed, suitable for large-scale undirected graphs (supports weights).
**Key Parameters**
- `weight`: Edge weight
- `resolution`: Community scale
- `threshold` / `max_level` / `seed`: Convergence threshold, number of levels, randomness control
**Characteristics**
- One of the "default first choices" commonly used in industry
- Results may change with randomness (seed controllable)
---
#### 10.2.7 leiden_communities —— Leiden Communities
**Function Description**
Improved version of Louvain, usually more stable, and emphasizes internal community connectivity (avoids "bad communities that look grouped but internally disconnected").
**Key Parameters**
- `weight` / `resolution`
- `max_level` / `seed`
**Characteristics**
- More stable quality, suitable for scenarios with strict requirements on community structure
---
### 10.3 Cycle Structure and Cycle Detection
#### 10.3.1 simple_cycles —— Simple Cycle Enumeration
**Function Description**
Enumerate all simple cycles in the graph (no repeated nodes, start point = end point). Can set `length_bound` to limit cycle length.
**Product Value**
- Discover capital reflow and cyclic transactions
- Discover cyclic dependencies in software dependencies
- Discover feedback loops in biological metabolic/regulatory networks
**Key Parameters**
- `length_bound`: Limit cycle length to avoid explosive output
**Complexity**: `O((V + E) * (C + 1))` (C is the number of cycles)
---
#### 10.3.2 cycle_basis —— Cycle Basis (Undirected Graph)
**Function Description**
Output a cycle basis for an undirected graph: a "basic independent cycle set" that can be combined to generate all cycles.
**Product Value**
- Circuit grid analysis (Kirchhoff's laws)
- Basic closed block identification in road networks
- Decompose complex cycle structures into basic components
---
### 10.4 Community Result Evaluation and Verification
#### 10.4.1 modularity —— Modularity Score (Q Value)
**Function Description**
Calculate modularity for a given community partitioning: measure the degree of "more intra-group edges and fewer inter-group edges".
**Key Parameters**
- `communities`: Must be a partition of nodes (mutually exclusive and full coverage)
- `weight` / `resolution`
**Output**: `float` (usually larger is better, but needs to be understood in combination with network type and resolution)
---
#### 10.4.2 partition_quality —— Coverage and Performance
**Function Description**
Output `(coverage, performance)`:
- coverage: Proportion of intra-community edges (whether there are many intra-community edges)
- performance: Proportion of intra-community edges + extra-community non-edges (how good the separation is)
---
#### 10.4.3 is_partition —— Partition Validity Verification
**Function Description**
Check whether a given community list is a strict partition:
- Whether it covers all nodes
- Whether mutually non-overlapping
**Note**
- For algorithms that allow overlapping communities (e.g., `k_clique_communities`), their output usually **does not satisfy** the partition definition, and `is_partition` should not be used for forced verification in this case.
---
## 11. Recommended Usage Guide (Selection Suggestions)
- **Want to first measure whether the network is "clustered"**: `average_clustering` + `transitivity`
- **Want to find local tight core points/gang features**: `clustering` + `triangles` ( `square_clustering` if necessary)
- **Want to quickly obtain communities (priority for large graphs)**: `louvain_communities` or `leiden_communities`
- **Want more interpretable hierarchical splitting (small graphs)**: `girvan_newman`
- **Want extremely fast rough clustering (ultra-large graphs)**: `label_propagation_communities` / `asyn_lpa_communities`
- **Want to find overlapping, extremely tight small circles**: `k_clique_communities`
- **Want to score the partitioning**: `modularity` + `partition_quality` (verify with `is_partition` first)
- **Want to check closed loops/cyclic dependencies/capital reflow**: `simple_cycles` (add `length_bound` if necessary), use `cycle_basis` for undirected graphs
---
## 12. Typical Directly Answerable Questions (Examples)
- "Is the overall network more like a loose random network or a small-world network? Give me an overall metric."
- "Which nodes have the tightest friend circles? List the Top-20 clustering coefficients/triangle counts."
- "Automatically partition the network into several natural communities and output the members of each community."
- "I have two community partitioning schemes, which one is better? Provide modularity and coverage/performance."
- "Is this community result a strict partition? Are there any missing/duplicate assignments?"
- "Find all capital closed loops/cyclic dependency links (can limit cycle length)."
- "Find overlapping core small circles (communities connected by iron triangles)."
---
