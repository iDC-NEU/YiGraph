---
sidebar_position: 1
---

# Basics Operator Set

**Operator Category**: Basics (Graph Structure Validation and Basic Traversal)

**Applicable Stages**: Graph legality verification, structure diagnosis, dependency analysis, hierarchical/upstream-downstream analysis, traversal and sorting

**Product Positioning**: Provides basic operator capabilities for "Is this graph a legal structure / Does it have cycles / Can it be sorted / Who are the upstream/downstream / How to traverse"

---

## 1. Operator Set Overview

The Basics operator set focuses on **basic structural property judgment of graphs** and **basic traversal/sorting capabilities**, mainly answering the following questions:

1. **Structural Legality**  
   - Is it a tree / forest / DAG?
   - Does it contain cycles?

2. **Dependency and Hierarchical Relationships**  
   - Who are the upstream nodes of a node? (ancestors)
   - Who are the downstream nodes of a node? (descendants)

3. **Traversal and Structure Generation**  
   - How to expand layer by layer from a node? (BFS)
   - How to deeply explore a relationship chain? (DFS)

4. **Dependency Sorting Problems**  
   - Can it be topologically sorted?
   - What are the feasible execution orders?
   - How to stably sort among multiple feasible orders?

---

## 2. Operator Capability Classification

| Capability Type | Corresponding Operator | Function Description |
|---|---|---|
| Tree/Forest Determination | `is_tree`, `is_forest` | Determine if the graph satisfies tree or forest structure |
| DAG Determination | `is_directed_acyclic_graph` | Determine if the directed graph is acyclic |
| Basic Traversal (Breadth) | `bfs_tree` | Build BFS layered traversal tree |
| Basic Traversal (Depth) | `dfs_tree` | Build DFS depth traversal tree |
| Topological Sort | `topological_sort` | Output any legal topological order |
| Stable Topological Sort | `lexicographical_topological_sort` | Output topological order constrained by lexicographical order |
| Topological Full Enumeration | `all_topological_sorts` | Enumerate all legal topological orders |
| Upstream Analysis | `ancestors` | Query all ancestors of a node |
| Downstream Analysis | `descendants` | Query all successors of a node |

---

## 3. General Input/Output Conventions

- **Input `G`**: NetworkX Graph / DiGraph
- **Output Types**:
  - Determination: `bool`
  - Set: `set(node)`
  - Traversal: `NetworkX DiGraph` (spanning tree)
  - Sorting: `list` / `iterator`

---

## 4. Detailed Operator Description

### 4.1 is_tree —— Tree Structure Determination

**Function Description**  
Determine if an undirected graph satisfies the tree definition of "connected + acyclic".

**Product Value**
- Quickly verify if hierarchical structure is legal
- Prevent implicit cycles or multi-parent node issues

**Typical Scenarios**
- Organizational structure verification  
- Task dependency tree verification  
- Material/assembly hierarchy verification  

**Applicability and Characteristics**
- Graph Type: Undirected graph
- Complexity: `O(V + E)`

---

### 4.2 is_forest —— Forest Structure Determination

**Function Description**  
Determine if an undirected graph consists of multiple disconnected trees.

**Product Value**
- Verify if multiple subsystems are independent
- Check for cross-component cycles

**Typical Scenarios**
- Multi-department organizational structure  
- Multi-product line assembly relationships  
- Multiple independent fund chain analysis  

**Applicability and Characteristics**
- Graph Type: Undirected graph
- Complexity: `O(V + E)`

---

### 4.3 is_directed_acyclic_graph —— DAG Determination

**Function Description**  
Determine if a directed graph is a Directed Acyclic Graph (DAG).

**Product Value**
- Pre-verification for algorithms like topological sorting and critical path
- Identify circular dependency risks

**Typical Scenarios**
- Project task dependencies  
- Reference / transaction / call relationships  
- Approval and workflow systems  

**Applicability and Characteristics**
- Graph Type: Directed graph
- Complexity: `O(V + E)`

---

### 4.4 ancestors —— Upstream Node Query

**Function Description**  
Return the set of all upstream nodes that can reach the specified node.

**Product Value**
- Quickly locate "who influenced me"
- Used for traceability, responsibility chain, reference source analysis

**Typical Scenarios**
- Fund source tracing  
- Reference chain analysis  
- Management reporting chain query  

**Applicability and Characteristics**
- Graph Type: Directed graph
- Complexity: `O(V + E)`

---

### 4.5 descendants —— Downstream Node Query

**Function Description**  
Return the set of all downstream nodes reachable from the specified node.

**Product Value**
- Measure impact scope
- Used for propagation analysis, dependency impact assessment

**Typical Scenarios**
- Public opinion/information diffusion  
- Task delay impact assessment  
- Fund flow analysis  

**Applicability and Characteristics**
- Graph Type: Directed graph
- Complexity: `O(V + E)`

---

### 4.6 bfs_tree —— Breadth-First Search Tree

**Function Description**  
Build a BFS traversal tree in hierarchical order from the specified node.

**Product Value**
- Natural hierarchical structure
- Shortest hop relationship is interpretable

**Typical Scenarios**
- Social circle diffusion  
- Organizational hierarchy display  
- City/station layered reachability analysis  

**Key Parameters**
- `source`: Starting point
- `depth_limit`: Maximum number of layers

**Complexity**
- `O(V + E)`

---

### 4.7 dfs_tree —— Depth-First Search Tree

**Function Description**  
Build a DFS tree by going as deep as possible along a path from the specified node.

**Product Value**
- Suitable for path mining and chain analysis
- Quickly discover deep relationships

**Typical Scenarios**
- Call chain analysis  
- File/directory scanning  
- Deep investigation relationship chain  

**Complexity**
- `O(V + E)`

---

### 4.8 topological_sort —— Topological Sort

**Function Description**  
Generate any linear order that satisfies dependency constraints in a DAG.

**Product Value**
- Provide one solution for "executable order"
- Basic capability for scheduling, building, and process orchestration

**Typical Scenarios**
- Project task ordering  
- Build systems  
- Approval processes  

**Complexity**
- `O(V + E)`

---

### 4.9 lexicographical_topological_sort —— Lexicographical Topological Sort

**Function Description**  
Among all feasible topological orders, select the one with the smallest lexicographical order.

**Product Value**
- Stable and reproducible results
- Suitable for productized output and display

**Typical Scenarios**
- Compilation/build order  
- Page generation order  
- Approval number sorting  

**Complexity**
- `O(E + V log V)`

---

### 4.10 all_topological_sorts —— Full Topological Order Enumeration

**Function Description**  
Enumerate all possible topological sorting results in a DAG.

**Product Value**
- Explore all feasible solutions
- Used for solution enumeration and decision analysis

**Typical Scenarios**
- Project scheduling multi-solution analysis  
- Multiple approval path exploration  
- Multiple scheduling solution comparison  

**Notes**
- The number of results may grow exponentially
- Suitable for small-scale DAGs

---

## 5. Recommended Usage Guide

- **Verify structure first**: `is_directed_acyclic_graph` / `is_tree`
- **Query upstream/downstream relationships**: `ancestors` / `descendants`
- **Display hierarchical relationships**: `bfs_tree`
- **Deep path analysis**: `dfs_tree`
- **Provide execution order**: `topological_sort`
- **Need stable order**: `lexicographical_topological_sort`
- **Need all solutions**: `all_topological_sorts`

---

## 6. Typical Questions That Can Be Directly Answered

- "Is this dependency graph a DAG?"
- "Which downstream tasks will be affected if a certain task is delayed?"
- "List all upstream sources of a certain node."
- "Give a reasonable execution order."
- "How many legal execution plans are there in total?"
- "Give me an execution order with the smallest number priority."
