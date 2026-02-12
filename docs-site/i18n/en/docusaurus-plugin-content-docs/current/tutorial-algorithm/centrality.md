---
sidebar_position: 3
---
# Centrality and Key Operator Set


**Operator Category**: Centrality (Node/Edge Importance Measurement)

**Applicable Stages**: Key Node Identification, Influence Assessment, Bottleneck/Bridge Localization, Network Robustness Analysis

**Product Positioning**: Provide a unified centrality capability base for answering "Who is the most important / Who is the key intermediary / Which edge is the most critical / How to accelerate information diffusion"


---
## 1. Operator Set Overview
The Centrality operator set provides systematic characterization capabilities for **node and edge importance** for various relational networks (social networks, transaction networks, citation networks, communication networks, transportation networks, dependency networks, etc.), mainly covering six types of issues:
1. **Connection-scale Influence**: Who has the most connections and the most frequent direct interactions (e.g., Degree Centrality, In/Out-Degree Centrality)
2. **Shortest Path Bridges/Intermediaries**: Who/which edge most frequently appears on the shortest paths and acts as the "bridge" and "hub" of the network (e.g., Betweenness Centrality, Edge Betweenness, Load Centrality)
3. **Global Structural Authority**: Who is connected to "important others", and who is more authoritative under random walk/link voting (e.g., Eigenvector Centrality, Katz, PageRank, HITS)
4. **Distance-reachable Efficiency**: Who has a shorter average distance to others and higher reach efficiency (e.g., Closeness Centrality, Harmonic Centrality)
5. **Local Clique Structure Contribution**: Who participates in more dense substructures (e.g., Subgraph Centrality)
6. **Propagation Seed Selection**: Which nodes are more suitable as diffusion seeds (e.g., VoteRank)
---
## 2. Operator Capability Classification
| Capability Type | Corresponding Operator | Function Description |
|---|---|---|
| Connection Scale (Unweighted) | `degree_centrality` | Measure node influence by the number of connections (directed/undirected) |
| Connection Scale (Directed) | `in_degree_centrality`, `out_degree_centrality` | Measure the influence of "being pointed to/actively pointing to" respectively |
| Distance Efficiency | `closeness_centrality`, `harmonic_centrality` | Characterize reach efficiency based on the distance to other nodes; Harmonic Centrality is more robust to unreachable nodes |
| Shortest Path Bridge (Node) | `betweenness_centrality`, `load_centrality` | Measure the degree of a node acting as an intermediary/bearing traffic |
| Shortest Path Bridge (Edge) | `edge_betweenness_centrality` | Measure the contribution of key connecting edges to the global network connectivity |
| Global Authority/Influence (Spectral/Iterative) | `eigenvector_centrality`, `katz_centrality`, `pagerank`, `hits` | "Being connected to important nodes makes one more important"; or from the perspective of random walk/Hub-Authority |
| Clique Structure Contribution (Spectral) | `subgraph_centrality` | The degree of a node participating in small cliques/substructures |
| Propagation Seed Selection | `voterank` | Select diffusion seeds through a "voting-suppression" mechanism to avoid excessive aggregation of seeds |
---
## 3. General Input and Output Conventions
- **Input `G`**: NetworkX graph object (directed/undirected; some algorithms support weights, some do not)
- **Output**:
  - Node Centrality: `{node: score}` dictionary
  - Edge Centrality: `{(u,v): score}` dictionary
  - HITS: `(hubs_dict, authorities_dict)`
  - VoteRank: Node list sorted by influence
> Note: Weighted algorithms usually interpret weights as **distance** (for shortest paths) or **connection strength** (for spectral/voting algorithms); subject to the definition of specific operator parameters `weight`/`distance`.
---
## 4. Detailed Operator Descriptions
### 4.1 degree_centrality —— Degree Centrality (Connection Scale)
**Function Description**
Calculate the degree centrality of each node: the normalized result of the number of node connections in the global network scale, used to measure the "direct relationship coverage".
**Product Value**
- Intuitive and fast calculation, suitable for rapid screening of large graphs
- Identify "high-connection accounts/high-interaction users/high-dependency components", etc.
**Typical Scenarios**
- Social Networks: Find users with the most friends/most interactions  
- Transaction Networks: Find accounts with the most transaction counterparties  
- Dependency Networks: Find service modules with the most dependencies/being depended on  
**Applicability and Characteristics**
- Graph Type: Directed/Undirected (equivalent to total degree for directed graphs)
- Weight: Not supported (treated as unweighted)
- Complexity: `O(V + E)`
---
### 4.2 in_degree_centrality —— In-Degree Centrality (Being Followed/Credited)
**Function Description**
Calculate the degree to which each node is pointed to in a directed graph, reflecting the popularity or authority (at the connection level) of "being cited/followed/called".
**Product Value**
- Suitable for characterizing "passive influence": who is pointed to by many others
- Sensitive to the direction of citation/follow/call
**Typical Scenarios**
- Citation Networks: Papers with the most citations  
- Follower Networks: Accounts with the most followers  
- Call Dependency: Core services with the most calls  
**Applicability and Characteristics**
- Graph Type: Directed only
- Weight: Not supported
- Complexity: `O(V + E)`
---
### 4.3 out_degree_centrality —— Out-Degree Centrality (Active Diffusion/Initiative Call)
**Function Description**
Calculate the degree to which each node actively points to others in a directed graph, reflecting the activity or extroversion of "active connection/active propagation/active call".
**Product Value**
- Identify "broadcasters/active reachers"
- Complementary to in-degree centrality, easy to distinguish between "popular vs active"
**Typical Scenarios**
- Social Networks: Active accounts that follow many others  
- Citation Networks: Review papers that cite many literatures  
- Dependency Networks: Upstream entry services that call many downstream modules  
**Applicability and Characteristics**
- Graph Type: Directed only
- Weight: Not supported
- Complexity: `O(V + E)`
---
### 4.4 closeness_centrality —— Closeness Centrality (Shortest Average Distance)
**Function Description**
Calculate the reciprocal of the sum of the shortest path distances from each node to other nodes (Wasserman-Faust improvement optional). The smaller the distance, the more "central to the network" the node is.
**Product Value**
- Identify nodes that "reach the entire network faster": suitable for information distribution and resource scheduling
- Can be combined with edge distance (`distance`) to represent road network mileage/delay
**Typical Scenarios**
- Transportation Networks: Hub stations with the shortest average arrival time  
- Communication Networks: Relay nodes with the minimum average hop count/delay  
- Organizational Networks: Team members who can reach most people the fastest  
**Key Parameters (Tuning)**
- `u`: Calculate only a single node (for online query/local analysis)
- `distance`: Edge distance attribute key (weighted shortest path)
- `wf_improved`: Whether to scale by the reachable ratio (more friendly to disconnected graphs)
**Applicability and Characteristics**
- Graph Type: Directed/Undirected (reachable distance is calculated by direction for directed graphs)
- Weight: Supported (as distance)
- Complexity: `O(V*(V+E))` (usually a shortest path calculation for each node)
---
### 4.5 harmonic_centrality —— Harmonic Centrality (More Robust to Unreachability)
**Function Description**
Sum of the reciprocals of the distances to other nodes: `Σ 1/d(u,v)`. Unreachable nodes (infinite distance) contribute 0, so it is more stable than Closeness Centrality in disconnected graphs.
**Product Value**
- Can still give a usable ranking in networks with isolated islands/weak connectivity
- Balances global and local reach efficiency
**Typical Scenarios**
- Knowledge Graphs/Citation Networks: Identify cross-community high-reach nodes when multiple communities exist  
- Transportation/Communication Networks: Evaluate key nodes in scenarios of local disconnection or failure  
**Key Parameters (Tuning)**
- `nbunch`: Calculate only part of the nodes (improve performance)
- `sources`: Specify the source set for calculating the reciprocal of distance (calculate "centrality relative to a certain group")
- `distance`: Edge distance attribute key
**Applicability and Characteristics**
- Graph Type: Directed/Undirected
- Weight: Supported (as distance)
- Complexity: `O(V*(V+E))`
---
### 4.6 betweenness_centrality —— Betweenness Centrality (Key Intermediary/Bridge)
**Function Description**
Measure the degree to which a node lies on the shortest paths between other node pairs: the more shortest paths pass through the node, the more the node acts as a "bridge/gate/intermediary".
**Product Value**
- Locate structural holes and cross-community connectors
- Identify "single point of failure risk": deleting this node may significantly fragment the network
**Typical Scenarios**
- Social Networks: Cross-circle "relationship intermediaries"  
- Transaction Networks: Key transition accounts on the fund/transaction chain  
- Transportation Networks: Key hubs/mandatory intersections (weighted by distance/duration)  
**Key Parameters (Tuning)**
- `k`: Sampling approximation (the larger `k` is, the more accurate; the smaller, the faster), suitable for large graphs
- `weight`: Edge weight attribute key (as distance)
- `normalized`: Whether to normalize (facilitate cross-graph comparison)
- `endpoints`: Whether to count endpoints in the shortest path count
- `seed`: Random seed for sampling (reproduce experiments)
**Applicability and Characteristics**
- Graph Type: Directed/Undirected
- Weight: Supported (as distance)
- Complexity: Exact calculation is usually about `O(V*E)` (NetworkX implementation is based on Brandes); approximation depends on `k`
---
### 4.7 edge_betweenness_centrality —— Edge Betweenness Centrality (Key Connecting Edge)
**Function Description**
Measure the number of node pairs for which an edge lies on their shortest path. Edges with high edge betweenness are often "bridge edges" across communities and key connections near network cut edges.
**Product Value**
- Find key links where "disconnecting which edge has the greatest impact"
- Commonly used for community detection, vulnerability analysis and network reinforcement
**Typical Scenarios**
- Transportation Networks: The most critical road sections/bridges  
- Communication Networks: The most critical links/fiber optic sections  
- Supply Chains: Key transportation/supply connections across regions  
**Key Parameters (Tuning)**
- `k`: Sampling approximation (speed up for large graphs)
- `weight`: Edge weight attribute key (as distance)
- `normalized`: Whether to normalize
- `seed`: Random seed for sampling
**Applicability and Characteristics**
- Graph Type: Directed/Undirected
- Weight: Supported (as distance)
- Complexity: Usually about `O(V*E)` (approximation depends on `k`)
---
### 4.8 load_centrality —— Load Centrality (Traffic-Bearing Intermediary)
**Function Description**
Similar to Betweenness Centrality, it measures the bearing degree of a node in the sense of "load/traffic sharing" on the shortest path, and can be used for key node analysis that is closer to "flow volume".
**Product Value**
- More inclined to the interpretation of "network flow bearing", suitable for transportation/communication load assessment
- Can limit the path length (`cutoff`) to focus on local impact
**Typical Scenarios**
- Communication Networks: Nodes bearing the most relay load  
- Transportation Networks: Hubs bearing the most commuting flow  
- System Dependency: Gateway modules bearing the most forwarding/aggregation in the call chain  
**Key Parameters (Tuning)**
- `cutoff`: Only consider paths with length not exceeding cutoff (reduce calculation and focus on local)
- `weight`: Edge weight attribute key (as distance)
- `normalized`: Whether to normalize
**Applicability and Characteristics**
- Graph Type: Directed/Undirected
- Weight: Supported (as distance)
- Complexity: Usually about `O(V*E)`
---
### 4.9 eigenvector_centrality —— Eigenvector Centrality (Being Connected to Important Nodes Makes One More Important)
**Function Description**
Based on the principal eigenvector of the adjacency matrix, score nodes: being connected to high-score nodes will increase their own score, reflecting "mutual reinforcement of authority".
**Product Value**
- Can reflect "high-quality connections" better than Degree Centrality
- Suitable for finding core nodes in the core circle
**Typical Scenarios**
- Social Networks: Key influencers in the core community  
- Citation Networks: Papers connected/cited by high-impact papers  
- Organizational Networks: Members who collaborate closely with key positions  
**Key Parameters (Tuning)**
- `max_iter`: Maximum number of iterations (can be increased when non-convergent)
- `tol`: Convergence error threshold (the smaller, the more accurate but slower)
- `nstart`: Initial vector (all 1s is usually safe by default)
- `weight`: Edge weight (usually interpreted as connection strength in this algorithm)
**Applicability and Characteristics**
- Graph Type: Directed/Undirected (need to pay attention to the interpretation direction for directed graphs)
- Weight: Supported (connection strength)
- Complexity: `O(k*(V+E))` (k is the number of iterations)
---
### 4.10 katz_centrality —— Katz Centrality (Attenuated Accumulation Considering Multi-hop Influence)
**Function Description**
On the basis of Eigenvector Centrality, consider the contribution of all length paths and decrease by the attenuation factor `alpha`; introduce baseline influence with `beta` at the same time.
**Product Value**
- Balances direct and indirect influence, suitable for businesses where "influence propagates with attenuation along the link"
- Nodes with 0 in-degree can also be given a non-zero baseline (depending on beta)
**Typical Scenarios**
- Risk Control Conduction: Influence assessment of risk propagation with attenuation along the transaction chain  
- Citation Networks: Indirect influence of multi-hop citations  
- Organizational Networks: Indirect influence of cross-level collaboration  
**Key Parameters (Tuning)**
- `alpha`: Attenuation coefficient (the larger, the more attention to long-distance connections; convergence conditions must be met)
- `beta`: Neighborhood baseline (can be a constant or customized by node)
- `max_iter`, `tol`, `nstart`: Iterative solution control
- `normalized`: Whether to normalize
- `weight`: Connection strength weight
**Applicability and Characteristics**
- Graph Type: Directed/Undirected
- Weight: Supported (connection strength)
- Complexity: `O(k*(V+E))`
---
### 4.11 pagerank —— PageRank (Global Importance of Random Walk)
**Function Description**
Treat the graph as a random walk process: the importance of a node is accumulated from "pointing from important nodes", and the random jump is controlled by the damping coefficient `alpha`.
**Product Value**
- Suitable for stable ranking of large-scale networks: web pages, citations, transaction relationships, etc.
- Supports personalization (`personalization`) to achieve influence "centered on a certain group"
**Typical Scenarios**
- Link Networks: Authority ranking of pages/resources  
- Citation Networks: Influence ranking of papers/patents  
- Email/Communication Networks: Priority of key contacts  
**Key Parameters (Tuning)**
- `alpha`: Damping coefficient (0.85 is commonly used); the larger, the more dependent on the link structure
- `personalization`: Personalization vector (bias to a specific node set)
- `dangling`: Out-edge allocation strategy for dangling nodes (nodes with no out-edges)
- `max_iter`, `tol`, `nstart`: Iterative solution control
- `weight`: Edge weight (usually represents transition weight in PageRank)
**Applicability and Characteristics**
- Graph Type: Mainly directed; undirected graphs will be converted to bidirectional directed graphs
- Weight: Supported (transition weight)
- Complexity: `O(k*(V+E))`
---
### 4.12 hits —— HITS (Hub/Authority Dual Roles)
**Function Description**
Calculate two types of scores for each node:
- **Authority**: The degree of being pointed to by high Hubs (content authority)
- **Hub**: The degree of pointing to high Authorities (information hub)
**Product Value**
- Distinguish between "resource authority" and "aggregation entry" in "citation/link" scenarios
- More suitable for explaining "directory/navigation sites (Hub)" and "content sites (Authority)"
**Typical Scenarios**
- Site Links: Identification of navigation sites (Hub) and content sites (Authority)  
- Citation Networks: Review papers (Hub) and widely recognized key papers (Authority)  
- Dependency Networks: Call aggregation entries (Hub) and core base modules (Authority)  
**Key Parameters (Tuning)**
- `max_iter`, `tol`, `nstart`: Iterative solution control
- `normalized`: Whether to normalize by sum
**Applicability and Characteristics**
- Graph Type: Mainly directed (undirected is possible but with weak interpretation)
- Weight: Not supported (NetworkX HITS version is usually unweighted)
- Complexity: `O(k*(V+E))`
---
### 4.13 subgraph_centrality —— Subgraph Centrality (Small Clique Structure Participation)
**Function Description**
Measure the degree of a node participating in various closed loops/substructures (such as triangles, quadrangles, etc.) through spectral methods, emphasizing nodes "at the center of dense clique structures".
**Product Value**
- Identify key nodes of the "core small group"
- More sensitive to "collaboration/gang/close cooperation" structures
**Typical Scenarios**
- Social Networks: Core figures in close friend circles  
- Biological Networks: Genes/proteins participating in key functional modules  
- Collaboration Networks: Authors/teams participating in dense collaboration groups  
**Applicability and Characteristics**
- Graph Type: Usually used for undirected graphs
- Weight: Not supported
- Complexity: Related to spectral decomposition, commonly `O(V^3)` (more suitable for small and medium scales)
---
### 4.14 voterank —— VoteRank (Propagation Seed Selection)
**Function Description**
Iteratively select a set of influential seed nodes through a mechanism of "node voting + suppressing the voting ability of neighbors after being selected", making the seeds more dispersed and covering a larger area on the graph.
**Product Value**
- Less likely to cluster than simply selecting the nodes with the highest degree, suitable for seed selection for influence maximization
- Can output the top K candidate seeds (`number_of_nodes`)
**Typical Scenarios**
- Marketing Communication: Select more dispersed seed users for fission propagation  
- Security Alert: Select key nodes as priority monitoring points  
- Caching/Content Distribution: Select representative nodes for content delivery  
**Key Parameters (Tuning)**
- `number_of_nodes`: Number of seed nodes to return (retain only positive vote nodes as many as possible by default)
**Applicability and Characteristics**
- Graph Type: Directed/Undirected
- Weight: Not supported
- Complexity: Usually nearly linear `O(k*(V+E))` (k is the number of output nodes)
---
## 5. Recommended Usage Guide (Practical Suggestions)
- **Quickly find large accounts/active nodes**: `degree_centrality` / `in_degree_centrality` / `out_degree_centrality`
- **Find bridges and structural holes**: `betweenness_centrality` (nodes) + `edge_betweenness_centrality` (edges)
- **Find nodes "close to everyone"**: `closeness_centrality`; use `harmonic_centrality` first if the graph is disconnected
- **Find authority and core circles**: `pagerank` (global authority) / `eigenvector_centrality` (spectral authority) / `katz_centrality` (multi-hop attenuation)
- **Distinguish between entry and authoritative content**: `hits` (Hub vs Authority)
- **Find core nodes of close groups**: `subgraph_centrality`
- **Select propagation seeds**: `voterank`
---
## 6. Typical Answerable Questions (Examples)
- "Output the top 20 nodes with the highest betweenness centrality for identifying key intermediaries."
- "Calculate the edge betweenness centrality of all edges and output the top 50 key edges in descending order."
- "In the current email communication network, who is the contact with the highest PageRank?"
- "In a disconnected social graph, what is the list of the top 20 users with the highest harmonic centrality?"
- "Use VoteRank to select 30 diffusion seed nodes for marketing reach."
