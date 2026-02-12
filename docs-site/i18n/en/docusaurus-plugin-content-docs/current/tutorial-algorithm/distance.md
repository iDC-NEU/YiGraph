---
sidebar_position: 11
---

# Distance & Measures Operator Set


**Operator Category**: Distance & Measures (Distance and Structural Metrics)

**Description**: Provides a set of global/semi-global graph metrics to measure network properties such as "span, compactness, central position, peripheral position, overall distance cost, and homophily connection tendency". Commonly used for network health checks, topology comparison, resilience assessment, propagation/tracking cost assessment, and structural preference analysis.


---

## 1. Operator Set Overview

The Distance & Measures operator set mainly covers two types of metrics:

### 1️⃣ Distance-based Global Structural Metrics
Built around **Shortest Path Distance**:
- **eccentricity**: "Distance to the farthest point" for a single node
- **radius**: Minimum eccentricity in the graph (worst distance from the optimal center)
- **diameter**: Maximum eccentricity in the graph (distance between the two farthest points in the network)
- **center**: Set of nodes with minimum eccentricity
- **periphery**: Set of nodes with eccentricity equal to the diameter
- **wiener_index**: Sum of all-pairs shortest path distances (overall "distance cost/compactness")

### 2️⃣ Structural Preference and Assortativity
Measures "whether connections tend to link similar nodes":
- **degree_assortativity_coefficient**: Degree assortativity (whether high-activity nodes prefer to connect with high-activity nodes)
- **attribute_assortativity_coefficient**: Attribute assortativity (whether nodes with the same label/type prefer to interconnect)

---

## 2. Operator List

| Operator | Core Output | Intuitive Meaning |
|---|---|---|
| `eccentricity` | dict[node→e] | Shortest path distance from node to farthest node |
| `radius` | int/float | Minimum eccentricity in the graph (worst distance from the best center) |
| `diameter` | int/float | Maximum eccentricity in the graph (distance between farthest two points) |
| `center` | list[nodes] | Set of center nodes with minimum eccentricity |
| `periphery` | list[nodes] | Set of peripheral nodes with eccentricity equal to diameter |
| `wiener_index` | number | Sum of all-pairs shortest path distances (overall distance cost) |
| `degree_assortativity_coefficient` | float r | Degree assortativity coefficient (-1~1) |
| `attribute_assortativity_coefficient` | float r | Attribute assortativity coefficient (-1~1) |

---

## 3. General Input/Output Conventions

### 3.1 Input

- **G**: NetworkX graph (supports directed/undirected, some optimizations only work for undirected graphs)
- **weight (optional)**: Edge weight field name / function / None
  - `None`: Each edge distance = 1 (unweighted graph)
  - string: Get distance from `G.edges[u, v][weight]`
  - function: Custom edge distance function (must return positive number)
  ⚠️ Note: Floating-point weights may introduce small rounding errors, prefer integer weights; weights should be positive (distance semantics).

- **e (optional)**: Pre-computed eccentricity dictionary (can be reused by center / radius / diameter / periphery)
- **usebounds (optional)**: Extrema bounding acceleration switch for undirected graphs (only effective when `e is None`)
- **sp (optional)**: Pre-computed shortest path distance dictionary (can be reused by eccentricity)
- **nodes (optional)**: Assortativity metrics can be limited to a subset of nodes
- **attribute (required)**: Node attribute key for attribute_assortativity_coefficient
- **x / y (optional)**: Degree type selection for directed graphs (degree assortativity, used to specify source/target using in or out)

### 3.2 Output

- Distance metrics: Numeric value (int/float), node list, or node→metric dictionary
- Assortativity: Single float `r` (usually in [-1, 1] range)

---

## 4. Detailed Operator Description

### 4.1 eccentricity —— Eccentricity (Node Farthest Distance)

#### Function Description
For node `v`, eccentricity is defined as:
> The maximum shortest path distance from `v` to all reachable nodes

By default, outputs an eccentricity dictionary for **all nodes**; can also query only a specific node via `v`.

#### Key Parameters
- **v (optional)**: Return eccentricity only for the specified node
- **sp (optional)**: Pass in pre-computed all-pairs shortest path lengths to avoid redundant computation
- **weight (optional)**: Calculate distance by weighted shortest path

#### Principle and Complexity
- Depends on shortest path computation (unweighted BFS / weighted Dijkstra, etc.)
- **Time Complexity**: Usually `O(V · (V + E))` (based on APSP/multi-source shortest path scale)

#### Applicable Scenarios
- Identify "most remote/hardest to reach" nodes (large eccentricity)
- Evaluate worst-case tracking distance for single-point monitoring entry

---

### 4.2 radius —— Graph Radius (Worst Distance from Best Center)

#### Function Description
Radius is defined as the minimum eccentricity in the graph:
```
radius(G) = min_v eccentricity(v)
```

It answers:
- "If we choose the optimal center point, what is the worst-case distance to the farthest point?"

#### Optimization Points
- Passing in **e** (pre-computed eccentricity) can be reused to accelerate multi-metric joint calculation
- Undirected graphs can try **usebounds=True** for acceleration (when e is not provided)

#### Complexity
- **Time Complexity**: `O(V · (V + E))` (usually same scale as eccentricity)

---

### 4.3 diameter —— Graph Diameter (Network Span Upper Bound)

#### Function Description
Diameter is defined as the maximum eccentricity in the graph:
```
diameter(G) = max_v eccentricity(v)
```

It answers:
- "What is the minimum shortest path distance between the two farthest points in the network?"

#### Optimization Points
Same as radius:
- Passing in **e** can accelerate
- Undirected graphs can try **usebounds=True**

#### Complexity
- **Time Complexity**: `O(V · (V + E))` (via APSP/multi-source shortest path)

---

### 4.4 center —— Graph Center (Minimum Eccentricity Node Set)

#### Function Description
Center nodes are those with minimum eccentricity (may be more than one):
```
center(G) = { v | eccentricity(v) == radius(G) }
```

Output: List of center nodes.

#### Typical Interpretation
- These points are most suitable as "global coordination center/monitoring entry/hub location" because they have the minimum distance to the farthest point.

---

### 4.5 periphery —— Graph Periphery (Maximum Eccentricity Node Set)

#### Function Description
Peripheral nodes are defined as nodes with eccentricity equal to the diameter:
```
periphery(G) = { v | eccentricity(v) == diameter(G) }
```

Output: List of peripheral nodes.

#### Typical Interpretation
- These points are often "most remote/hardest to cover" locations, suitable for identifying structural boundaries, isolated risk points, and coverage blind spots.

---

### 4.6 wiener_index —— Wiener Index (Total All-Pairs Distance Cost)

#### Function Description
The Wiener index is the sum of shortest path distances between **all node pairs**:
```
Wiener(G) = Σ(u<v) dist(u, v)
```
(Or accumulate all pairs according to NetworkX definition)

It measures:
- Whether the network is compact overall (smaller distance sum means more compact)
- The "distance cost" level for global propagation/tracking/collaboration

#### Key Parameters
- **weight (optional)**: If provided, accumulate by weighted distance

#### Complexity
- **Time Complexity**: `O(V · (V + E))` (essentially requires many shortest path distances)

#### Applicable Scenarios
- Global compactness comparison of different network structures
- Overall path cost assessment for transportation/supply chain/collaboration networks

---

### 4.7 degree_assortativity_coefficient —— Degree Assortativity Coefficient

#### Function Description
Measures whether "the degrees of nodes at both ends of connections" are correlated (a form of Pearson correlation coefficient):
- **r > 0**: Assortative (high-degree nodes prefer to connect with high-degree nodes)
- **r < 0**: Disassortative (high-degree nodes prefer to connect with low-degree nodes, showing hierarchical/hub-spoke structure)
- **r ≈ 0**: No obvious preference

#### Key Parameters
- For directed graphs, can specify:
  - **x**: Source end uses in/out-degree (default out)
  - **y**: Target end uses in/out-degree (default in)
- **weight (optional)**: Accumulate degree by edge weight (degree = sum of adjacent edge weights)
- **nodes (optional)**: Limit calculation to a subset of nodes (for group comparison)

#### Complexity
- **Time Complexity**: `O(E)`

#### Applicable Scenarios
- Determine if the network has obvious "core-periphery" structure (commonly negative assortativity)
- Determine if active accounts "cluster together" or "radiate outward"

---

### 4.8 attribute_assortativity_coefficient —— Attribute Assortativity Coefficient

#### Function Description
Measures "whether nodes with the same attribute are more likely to interconnect":
- **r > 0**: Homophilic connection obvious (same type clusters together)
- **r < 0**: Heterophilic connection obvious (cross-type connections stronger)
- **r ≈ 0**: Attribute has weak influence on connection preference

#### Key Parameters
- **attribute (required)**: Node attribute key (e.g., `risk_level` / `account_type` / `department`)
- **nodes (optional)**: Calculate on a specified subset of nodes (e.g., only look at a certain community/region)

#### Complexity
- **Time Complexity**: `O(E)`

#### Applicable Scenarios
- Whether risk level/account type shows "same type connects"
- Whether medical institution hierarchy/department attributes show "same level interconnects"
- Whether organizational structure is "mainly intra-department connections" or "mainly cross-department collaboration"

---

## 5. Selection Guide (How to Choose)

- **Want to know the maximum network span**: `diameter`
- **Want to know the worst distance from the best center**: `radius` + `center`
- **Want to find the most remote boundary nodes**: `periphery` (or look at maximum eccentricity)
- **Want to rank all nodes by "worst reachable distance"**: `eccentricity`
- **Want to see global compactness/distance cost**: `wiener_index`
- **Want to see if "high-activity clustering/core-periphery structure"**: `degree_assortativity_coefficient`
- **Want to see if "same type clustering/cross-type connection"**: `attribute_assortativity_coefficient`

---

## 6. Engineering and Usage Notes (Common Pitfalls)

1. **Connectivity Requirements**
   Metrics like diameter/radius/center/periphery/eccentricity are affected in "unreachable" situations.
   - Undirected graphs: If not connected, usually need to calculate separately for each connected component or take the largest connected component
   - Directed graphs: May need to consider strongly connected components or interpret results on reachable subgraphs

2. **Semantics of Weighted Distance**
   `weight` represents "distance/cost/time", should be positive; if your edge weight is "capacity/similarity/strength", don't use it directly as distance (need to convert first, such as taking reciprocal or negative logarithm, and ensure positive weight).

3. **Performance Recommendations**
   If you need `eccentricity + diameter + radius + center + periphery` in the same analysis:
   - **Calculate eccentricity (e) first**, then pass it to other metrics for reuse
   - Undirected graphs can try `usebounds=True` (when e is not pre-provided)

---
