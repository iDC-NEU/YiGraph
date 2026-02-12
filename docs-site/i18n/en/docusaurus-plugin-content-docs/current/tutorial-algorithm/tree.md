---
sidebar_position: 6
---
# Tree & Spanning Tree Operator Set


**Operator Category**: Tree & Spanning Tree (Minimum/Maximum/Random Spanning Tree)

**Description**: Spanning tree-related algorithms for undirected graphs, used to construct a skeleton structure that "covers all nodes and is acyclic", and output spanning trees or spanning forests with different objectives (minimum cost / maximum revenue / random sampling).


---
## 1. Operator Set Overview
In graph analysis and network engineering, it is often necessary to "extract the skeleton" of complex networks:
- **Minimum Spanning Tree (MST)**: Minimize the total cost/distance/latency while ensuring connected coverage.
- **Maximum Spanning Tree (MaxST)**: Maximize the total revenue/bandwidth/similarity while ensuring connected coverage.
- **Random Spanning Tree (Random ST)**: Randomly sample spanning trees according to the distribution defined by weights, used for simulation, pressure testing, randomization strategies or generative content (such as mazes).
> Note: When the input graph `G` is **disconnected**, NetworkX will return a **spanning forest**, i.e., one tree for each connected component.
---
## 2. Operator List
- `minimum_spanning_tree`
- `maximum_spanning_tree`
- `random_spanning_tree`
---
## 3. General Input and Output Conventions
### 3.1 Input
- **G**: Undirected graph, supporting `multigraph`.
- **weight**: Edge weight attribute name (default `"weight"`, can be `None` for unweighted/equal weight in random spanning tree).
### 3.2 Output
- **minimum_spanning_tree / maximum_spanning_tree**: `NetworkX Graph` (spanning tree or spanning forest)
- **random_spanning_tree**: `nx.Graph` (a spanning tree sampled according to the weight distribution)
---
## 4. Detailed Operator Descriptions
### 4.1 minimum_spanning_tree —— Minimum Spanning Tree (MST)
#### Function Description
Select a set of edges in an undirected weighted graph such that:
1) All nodes are covered (a tree if connected, a forest if disconnected);
2) No cycles are included;
3) **The sum of total edge weights is minimized**.
#### Application Scenarios
- Network design: Connect all stations with the lowest construction cost/shortest cable length (fiber optic laying, water supply networks, power distribution cables).
- Route planning: Form a coverage skeleton with the lowest total latency/shortest total distance.
- Resource-optimal "full coverage connection scheme".
#### Parameters and Tuning Mechanisms
- **weight (str)**: Specify which edge attribute to use as the weight (e.g., `cost` / `distance` / `latency`).
  - Changing it will alter the evaluation criterion of "minimum", thus changing the output tree structure.
- **algorithm (str)**: `'kruskal' | 'prim' | 'boruvka'` (default `kruskal`)
  - **Kruskal**: Select edges step by step by edge weight sorting, suitable for sparse graphs;
  - **Prim**: Expand from a certain node, suitable for connected graphs and adjacency structures;
  - **Boruvka**: Parallel component merging, commonly used in some large graphs/engineering implementations.
- **ignore_nan (bool)**: Processing when encountering NaN weights
  - `False`: Throw an exception by default
  - `True`: Ignore the edge (equivalent to unavailable)
#### Principles
The algorithm merges multiple connected components by continuously selecting edges that "do not form cycles and are as cheap as possible" until all nodes are covered.
**Typical Complexity**: `O(E log V)` (common implementations of Kruskal/Prim)
#### Solvable Questions
- Calculate the MST of the weighted undirected graph and output the total weight and all edges.
- Output the complete edge list of the MST, sorted by weight from smallest to largest.
- Urban water supply/park fiber optics/logistics skeleton: How to cover all nodes with the lowest total cost?
### 4.2 maximum_spanning_tree —— Maximum Spanning Tree (MaxST)
#### Function Description
Contrary to MST: Select a set of edges in an undirected weighted graph such that all nodes are covered with no cycles, and **the sum of total edge weights is maximized**.
#### Application Scenarios
- Communication/backbone networks: Edge weights represent bandwidth or throughput, with the goal of maximizing the overall capacity skeleton.
- Correlation/similarity networks: Edge weights are similarity values, outputting a "maximum similarity skeleton" (often used for visualization and structure extraction).
- Trust/influence propagation: Edge weights represent trust intensity, extracting the maximum trusted propagation skeleton.
#### Parameters and Tuning Mechanisms
- **weight (str)**: Edge weight attribute name (default `"weight"`), determining the evaluation criterion of "maximization".
- **algorithm (str)**: `'kruskal' | 'prim' | 'boruvka'` (default `kruskal`)
  Different algorithms mainly affect performance and implementation paths, and the results should be equivalent under the same weight definition (multiple solutions may result from edges with the same weight).
- **ignore_nan (bool)**: Whether to ignore edges with NaN weights (same logic as MST).
#### Principles
Essentially, it is the spanning tree construction for the "maximization" objective, which can be regarded as performing MST after negating the edge weights, or directly using the corresponding maximum spanning tree version.
**Typical Complexity**: `O(E log V)`.
#### Solvable Questions
- Calculate the MaxST and output all edges and their weights.
- In a "bandwidth/trust/similarity" network, which edges are selected to form the maximized skeleton?
- Output the connection relationship matrix of the MaxST, sorted by weight from largest to smallest.
### 4.3 random_spanning_tree —— Random Spanning Tree Sampling
#### Function Description
**Randomly sample** a spanning tree from a given undirected (weighted) graph according to a specified probability distribution.
- When `weight` is specified, the probability of a spanning tree is related to edge weights;
- Suitable for generative structures, simulation testing, randomization strategies, avoiding fixed patterns, etc.
#### Application Scenarios
- Maze/map generation: Generate a random path skeleton that is "acyclic and fully connected".
- Network pressure testing: Randomly extract a temporary backbone structure from the real topology.
- Resilience simulation: Randomly sample multiple alternative skeletons to evaluate the occurrence frequency of critical edges.
#### Parameters and Tuning Mechanisms
- **weight (str | None)**: Edge weight attribute used to determine the sampling distribution
  - `None`: Equal weight sampling (closer to uniformly random spanning trees)
- **multiplicative (bool, default=True)**: Probability calculation method
  - `True`: Tree probability is proportional to the "product of edge weights" (favoring trees with strong edges on all paths)
  - `False`: Tree probability is proportional to the "sum of edge weights" (favoring overall larger trees while allowing local weak edges)
- **seed**: Random seed for reproducibility
#### Principles
The core idea is to sample spanning trees according to the distribution based on random processes (NetworkX implementation is usually related to random walk/spanning tree sampling ideas such as Wilson’s algorithm).
**Complexity**: Polynomial time (varies with implementation and graph scale).
#### Solvable Questions
- Randomly generate a spanning tree and output the edge list.
- Randomly sample 5 different spanning trees and output their connection relationships respectively.
- Maze design: Randomly generate a path structure that connects all rooms without cycles.
- Generate random alternative connection skeletons to avoid being identified with fixed patterns.
---
## 5. Selection Guide (Quick Selection)
- **Objective: Lowest cost/shortest distance/minimum latency**: Use `minimum_spanning_tree`
- **Objective: Maximum bandwidth/maximum trust/maximum similarity skeleton**: Use `maximum_spanning_tree`
- **Objective: Randomization, simulation, sampling, multiple alternative schemes**: Use `random_spanning_tree`
- **Input graph may be disconnected**: All three will output a **spanning forest** (one tree for each connected component)
---
## 6. Practical Suggestions (Common Pitfalls)
1. **Align edge weight criteria**:
   - Use MST for cost-related metrics; use MaxST for revenue/intensity-related metrics; do not reverse the meanings.
2. **Pay attention to keys in multigraph**:
   - NetworkX will retain multiple candidate edges in the case of multigraph, and the result may select edges with different keys.
3. **Process NaN weights in advance or enable ignore_nan**:
   - Otherwise, MST/MaxST may report an error directly.
4. **Multiple solutions caused by equal weights**:
   - Spanning trees are not unique, especially when there are many duplicate weights, different algorithms or traversal orders may give different but equivalent trees.
---