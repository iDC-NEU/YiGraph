---
sidebar_position: 12
---
# Graph Query Operator Set


**Operator Category**: Graph Query

**Description**: Graph query tools for querying specific nodes, neighbors, subgraphs, and paths. It provides basic query capabilities for graph data such as node lookup, edge lookup, neighbor lookup, path lookup, subgraph extraction, and aggregate statistics. Suitable for interactive exploratory analysis, business retrieval, risk control investigation, knowledge graph relationship discovery, and visualized data extraction.
> Different from pure graph algorithms (shortest path, clustering, centrality, etc.): Graph Query is more focused on **data acquisition and filtering**. It is used to quickly locate the "objects/subgraphs to be analyzed" and then hand them over to upper-layer algorithms or business logic for processing.


---
## 1. Overview of the Operator Set
The Graph Query operator set covers 4 common query modes:
1. **Node Lookup**
   - Precisely query a single node (unique key: ID / account_id / name, etc.)
   - Filter a set of nodes by attribute conditions (age&gt;30, status=completed)
2. **Relationship/Edge Filter**
   - Filter edges by relationship type + attribute conditions (amount&gt;400, time range, duration, etc.)
   - Can directly return aggregate statistics (COUNT / SUM / AVG, etc.)
3. **Structural Traversal (Neighbors / Paths / Common Neighbors)**
   - Query 1~k hop neighbors with limited direction and relationship type
   - Query paths between two nodes (shortest or all paths, constrained by hops)
   - Query common neighbors (support filtering by relationship attributes)
4. **Subgraph Extraction**
   - Form an ego-network by expanding k hops from a central node
   - Or extract a "transaction subgraph/time window subgraph" by relationship attribute conditions
   - Or extract an induced subgraph from a given node list (only internal relationships are considered)
---
## 2. Operator List
| Operator | Core Capability |
|---|---|
| `node_lookup` | Precisely query nodes by label + unique key; or filter nodes by attribute conditions |
| `relationship_filter` | Filter edges by rel_type + attribute conditions; support aggregate statistics |
| `aggregation_query` | Group aggregate statistics (group by node/attribute, COUNT/SUM/AVG/…) |
| `neighbor_query` | k-hop neighbor traversal (can limit rel_type / direction, return edge fields) |
| `path_query` | Path query between two nodes (can limit relationship type, direction, min/max hops) |
| `common_neighbor` | Common neighbors of two nodes (can limit rel_type / direction / filter by edge attributes) |
| `subgraph` | Subgraph extraction: central node expansion mode / relationship filter mode |
| `subgraph_by_nodes` | Extract induced subgraph by node list (optional internal edges only) |
---
## 3. General Input and Output Conventions
### 3.1 Input
- **label / start_label / end_label**: Specify node types in a heterogeneous graph (e.g., `Account` / `Person` / `Paper`)
- **key / value / values**: Locate nodes by attributes (e.g., `account_id=ACC_12345`)
- **rel_type**: Relationship type (e.g., `TRANSFER` / `FRIEND` / `CITES`)
- **direction**: Direction (`OUTGOING` / `INCOMING` / `BOTH`)
- **conditions / rel_conditions**: Attribute filter conditions (support multi-condition combination and comparison operations)
- **return_fields**: List of returned fields (used to reduce transmission and avoid information loss)
- **hops / min_hops / max_hops / limit / limit_paths**: Control search depth and scale
### 3.2 Output
- `node_lookup`: Single node or node list
- `relationship_filter`: Relationship list or aggregate value
- `aggregation_query`: Group aggregate result list (group_key + aggregated_value)
- `neighbor_query`: Neighbor node/edge information or path structure (depending on return_fields and implementation)
- `path_query`: Path list (usually node sequence or node+edge structure)
- `common_neighbor`: Common neighbor node list (can be attached with fields)
- `subgraph / subgraph_by_nodes`: Subgraph (including nodes and relationships)
---
## 4. Detailed Operator Descriptions
### 4.1 node_lookup — Node Lookup (Precise Node Query / Conditional Node Filter)
#### Function Description
Supports two modes:
- **Single node precise query**: `label + key + value`
- **Multi-node conditional filter**: `label + conditions`
#### Parameter Key Points
- **return_fields**: It is recommended to only retrieve necessary fields to reduce IO overhead
- **conditions**: Support numerical comparison (`>`, `<`, `>=`, `<=`, `==`, `!=`) and string matching
#### Principle and Complexity
- Unique key node lookup: Positioning via index/hash, approximately `O(1)`
- Conditional filtering: Scanning is the main method without index, approximately `O(n)`
#### Answerable Questions
- Find the account node with `account_id = "ACC_12345"`
- List all users with `age > 30`
- Query all customers living in US with state `VT`
### 4.2 relationship_filter — Relationship Filter (Edge Filter + Aggregation Supported)
#### Function Description
Filter edges by **relationship type** (required), and can further specify:
- Start/end node label
- Relationship attribute conditions (amount, time, duration, tag, etc.)
- Aggregate statistics (COUNT/SUM/AVG/MAX/MIN)
#### Parameter Key Points
- **rel_conditions**: Support multi-condition combination (AND/OR logic is agreed by implementation)
- **aggregate**: It is recommended to enable when only statistical results are concerned to avoid returning massive edges
- **return_fields**: Return transaction fields you care about (amount, timestamp, is_sar…)
#### Principle and Complexity
- Scan and filter relationships of the specified type: `O(m)` (m is the number of relationships of the rel_type)
#### Answerable Questions
- Find all transactions with `amount > 400`
- List the count of transactions where `is_sar` is False
- Calculate the total amount of all outgoing transactions
### 4.3 aggregation_query — Group Aggregate Statistics
#### Function Description
Provides GROUP BY + aggregation capabilities for graph data, used for:
- Counting (COUNT)
- Summation/mean/extreme value (SUM/AVG/MAX/MIN)
Supports grouping by:
- **Node label** (per entity)
- **Attribute** (per category)
#### Parameter Key Points
- **aggregate_type**: COUNT / SUM / AVG / MAX / MIN (required)
- **aggregate_field**: Required for SUM/AVG/MAX/MIN
- **group_by_node / group_by_property**: Determine the statistical dimension
- **direction / rel_type**: Limit the scope of relationships involved in statistics
#### Principle and Complexity
- Traverse relevant nodes/edges and perform group aggregation: `O(n + m)` (related to the scale of data involved in statistics)
#### Answerable Questions
- Count the number of transactions per account
- Calculate the total amount of outgoing transactions per account
- Find the top 10 accounts with the most transactions
### 4.4 neighbor_query — Neighbor Query (k-hop)
#### Function Description
Perform **k-hop neighbor expansion** (BFS) starting from the specified node:
- 1-hop: Direct neighbors
- 2-hop: Friends of friends / Counterpart's counterparts in transactions
- 3-hop+: Larger scope of relationship circles (beware of scale explosion)
#### Parameter Key Points
- **hops**: Control expansion depth (default 1)
- **rel_type / direction**: Strongly recommended to constrain the scale
- **return_fields**: Return edge details (avoid information loss especially when hops=1)
#### Principle and Complexity
- BFS expansion, worst case approximately `O(d^k)` (d is the average degree, k is the number of hops)
#### Answerable Questions
- Query neighbors of Collins Steven
- Find all 2-hop neighbors of user Alice
- Find all accounts that Collins Steven has transferred money to
### 4.5 path_query — Path Query (How Two Nodes Are Connected)
#### Function Description
Query the connection paths between two nodes, which can be used for:
- Fund flow tracing
- Citation chain tracing
- Social relationship discovery
Supports restrictions on:
- Relationship type (only TRANSFER, etc.)
- Direction (OUTGOING/INCOMING/BOTH)
- Min/max hops (control search space)
#### Parameter Key Points
- **max_hops**: Strongly recommended to set to avoid full graph scale explosion
- **min_hops**: Used to exclude direct connections (for viewing "indirect relationships")
- **rel_type**: Limit to specific relationship types to improve semantic accuracy and performance
#### Principle and Complexity
- Shortest path: Typically `O(V+E)`
- Enumerate all paths: May be exponential (the denser the graph, the higher the risk)
#### Answerable Questions
- Find the path from Collins Steven to Nunez Mitchell
- Find all paths between two companies within 5 hops
- Trace the shortest supply chain path from supplier to customer
### 4.6 common_neighbor — Common Neighbor Query (Mutual Acquaintances / Shared Transaction Counterparts)
#### Function Description
Return the set of nodes connected to both v1 and v2 (intersection of neighbor sets), used for:
- Mutual associated objects (Mutual friends)
- Potential collusion/conspiracy detection (Shared transaction partners)
- Basic features for similarity and link prediction
#### Parameter Key Points
- **rel_conditions**: Can further filter the "edges involved in common neighbors" (e.g., amount&gt;400)
- **direction / rel_type**: Determine the business semantics of "common neighbors" (mutual incoming parties/mutual outgoing parties/mutual friends)
#### Principle and Complexity
- Take the intersection of neighbor sets: `O(d1 + d2)` (sum of the degrees of the two nodes)
#### Answerable Questions
- Identify mutual friends between user A and user B
- Find common transaction partners of two accounts
- Find common transaction neighbors where transaction amounts are all greater than 400
### 4.7 subgraph — Subgraph Extraction (Two Modes)
#### Function Description
Provides two extraction methods:
**Mode 1: Central Node Expansion (ego network)**
- Input: `label/key/value + hops (+ rel_type/direction)`
- Output: k-hop subgraph centered on the node (can be used for visualization and local analysis)
**Mode 2: Extraction by Relationship Filter (slice by edge filter)**
- Input: `rel_type + rel_conditions (+ start_label/end_label) + limit`
- Output: Subgraph consisting of all qualified relationships and their endpoints (e.g., "all transfer subgraphs in a certain month")
#### Parameter Key Points
- **limit_paths / limit**: Control the scale (especially in visualization/interactive scenarios)
- **rel_conditions**: Used for time window/amount window filtering
- **direction**: Particularly important in fund flow/citation chain scenarios
#### Principle and Complexity
- Mode 1: `O(d^k)` (grows with the number of hops)
- Mode 2: `O(m)` (m is the number of matching relationships)
#### Answerable Questions
- Extract subgraph around Collins Steven within 2 hops
- Extract subgraph of all transactions on 2025-05-01
- Extract subgraph of transactions with amounts between 300 and 500
### 4.8 subgraph_by_nodes — Extract Induced Subgraph by Node List
#### Function Description
Given a set of nodes (specified by `label + key + values`), extract the relationship subgraph between them:
- **include_internal=True (default)**: Only include internal edges between these nodes (most commonly used)
- **include_internal=False**: May include edges to external nodes (subject to implementation agreement)
#### Parameter Key Points
- **rel_type / direction**: It is recommended to specify in multi-relationship type scenarios
- **include_internal**: Used to control whether to "only view intra-group relationships"
#### Principle and Complexity
- Retrieve specified nodes + filter edges between them: `O(n + m)` (n is the number of nodes, m is the number of intra-group edges)
#### Answerable Questions
- Extract accounts A, B, C and their transfer relationships
- Analyze transaction network among 5 suspicious accounts
- Find all relationships among a set of companies
---
## 5. Selection Guide (How to Choose)
- **Query a single entity (by ID / account_id)**: `node_lookup` (key+value)
- **Filter a set of entities by attributes**: `node_lookup` (conditions)
- **Filter transaction/communication records by edge attributes**: `relationship_filter`
- **Directly generate statistical reports/rankings**: `aggregation_query`
- **View the local relationship circle of a node (k-hop)**: `neighbor_query`
- **Query how two nodes are associated (paths)**: `path_query` (be sure to set max_hops)
- **Query mutual friends/mutual counterparts/shared suppliers**: `common_neighbor`
- **Extract a subgraph for visualization/analysis**: `subgraph` (central expansion or relationship filter)
- **Specify a set of nodes to view intra-group relationships**: `subgraph_by_nodes`
---
## 6. Engineering Considerations and Common Pitfalls
1. **Path and k-hop queries are most prone to scale explosion**
   - It is recommended to always set: `max_hops` / `hops`、`rel_type`、`direction`、`limit/limit_paths`
2. **Prioritize aggregation (do not retrieve details unless necessary)**
   - As long as the result is a statistical value/TopK, prioritize using `relationship_filter(aggregate=...)` or `aggregation_query`
3. **return_fields is the key to performance**
   - Avoid returning all fields of the entire edge/node, especially for large graphs and multi-attribute graphs
4. **direction affects semantics**
   - In scenarios such as fund flow/citation chains, OUTGOING and INCOMING have completely different business meanings
5. **Unify the data types of conditional fields**
   - It is recommended to unify the timestamp/date format for time fields
   - Avoid storing numerical fields as strings leading to comparison errors
---
## 7. Typical Answerable Questions
- "Find the account information of account_id=ACC_12345"
- "Filter all transfers with amount&gt;400"
- "Top 10 accounts by the number of transfers per account"
- "Is there a fund path between A and B (&lt;=5 hop)?"
- "Who are the mutual friends of two people?"
- "Extract a 2-hop subgraph around a suspicious account for visual investigation"
- "Extract the internal transaction network among 5 specified accounts to judge whether a gang is formed"
---