---
sidebar_position: 8
---
# Link Prediction Operator Set


**Operator Category**: Link Prediction

**Description**: Used to predict "connections that do not yet exist but may appear" in a network, or evaluate the formation probability of potential edges. Widely applied in friend recommendations, relationship mining, anti-fraud, recommendation systems, research collaboration prediction, and other scenarios.


---

## 1. Operator Set Overview

The Link Prediction operator set mainly addresses the following types of problems:

- Which nodes are **more likely to form connections in the future**?
- In the current network structure, which "non-edges" are most worthy of attention?
- How to combine factors such as **common neighbors, node activity, community structure** to score and rank potential relationships?

Most algorithms in this operator set belong to **heuristic methods based on local structure**, with characteristics:
- No need for supervised learning and historical labels
- Computationally efficient with strong interpretability
- Very suitable as the first-layer capability for recommendation/risk control/exploratory analysis

---

## 2. Operator List and Capability Classification

### 2.1 Based on Common Neighbors

| Operator | Core Idea |
|---|---|
| `jaccard_coefficient` | Common neighbor ratio (intersection / union) |
| `adamic_adar_index` | Rare common neighbors have higher weight |
| `resource_allocation_index` | Allocate "resources" through low-degree nodes |
| `preferential_attachment` | High-degree nodes are more likely to form edges |

### 2.2 Community-aware

| Operator | Core Idea |
|---|---|
| `cn_soundarajan_hopcroft` | Common neighbors within the same community are more important |
| `ra_index_soundarajan_hopcroft` | Resource allocation under community constraints |
| `within_inter_cluster` | Distinguish "intra-community" and "inter-community" common neighbors |

---

## 3. General Input/Output Conventions

### 3.1 Input

- **G**: NetworkX undirected graph
- **ebunch (optional)**: Specify node pairs `(u, v)` for which to calculate scores
  - If `None`, calculate for all **currently non-existent edges**
- **community (some algorithms)**: Node attribute name representing community label

### 3.2 Output

- **iterator of (u, v, score)**
  - `u, v`: Node pair
  - `score`: Link prediction score (higher value means more likely to form connection)

---

## 4. Detailed Operator Description

### 4.1 jaccard_coefficient —— Jaccard Coefficient

#### Function Description
Measures the **overlap ratio** of two nodes' neighbor sets:

```
J(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
```

#### Applicable Scenarios
- "Common friends ratio" in friend recommendations
- Similarity assessment for projects/products/content
- Emphasizes **similarity ratio** rather than absolute quantity

#### Characteristics
- Simple, intuitive, easy to interpret
- Has some penalty for high-degree nodes (union becomes larger)

---

### 4.2 adamic_adar_index —— Adamic–Adar Index

#### Function Description
Weights common neighbors:
**The smaller the degree (more "rare") of a common neighbor, the greater its contribution**.

```
AA(u,v) = Σ(w ∈ CN(u,v)) 1/log(deg(w))
```

#### Applicable Scenarios
- "High-quality common friends" in social networks
- Hidden intermediary relationships in anti-fraud/AML
- "Strong signals within small circles" in collaboration networks

---

### 4.3 resource_allocation_index —— Resource Allocation Index (RA)

#### Function Description
Assumes common neighbors "allocate resources" to both end nodes,
**The smaller the degree, the more resources allocated**:

```
RA(u,v) = Σ(w ∈ CN(u,v)) 1/deg(w)
```

#### Applicable Scenarios
- Similar to Adamic–Adar, but with stronger penalty for high-degree nodes
- Suitable for finding **hidden, low-exposure but closely related** potential connections

---

### 4.4 preferential_attachment —— Preferential Attachment

#### Function Description
Assumes the network follows the "rich get richer" principle:

```
PA(u,v) = deg(u) × deg(v)
```

#### Applicable Scenarios
- In social platforms, new users are more likely to follow "big V"
- "Hub effect" in citation networks, web links, transaction networks

#### Characteristics
- Does not depend on common neighbors
- More biased towards **traffic/influence prediction** rather than similarity

---

### 4.5 cn_soundarajan_hopcroft —— Community-Aware Common Neighbors

#### Function Description
Introduces **community constraints** on top of regular common neighbors:
- Common neighbors within the same community contribute more
- Cross-community connections are naturally suppressed

#### Applicable Scenarios
- Internal recommendations within departments/circles/interest communities
- Prediction tasks for "same-circle relationship reinforcement"

---

### 4.6 ra_index_soundarajan_hopcroft —— Community-Aware RA Index

#### Function Description
Combines **resource allocation idea** with **community structure**:
- Low-degree common neighbor + same community → highest contribution

#### Applicable Scenarios
- Potential collaboration relationships within enterprises/teams
- Mining hidden strong relationships within communities

---

### 4.7 within_inter_cluster —— WIC (Within/Inter-Community)

#### Function Description
Compares two types of common neighbors:
- **Intra-community common neighbors**
- **Inter-community common neighbors**

Measures whether the relationship is "cohesive or bridging" through their ratio.

#### Applicable Scenarios
- Cross-department / cross-circle collaboration discovery
- Encouraging "breaking circles" connections vs strengthening internal connections
- Network exploration and structural diversity analysis

---

## 5. Selection Guide (How to Choose)

- **Want simple similarity**: `jaccard_coefficient`
- **Emphasize rare common neighbors**: `adamic_adar_index`
- **Stronger penalty for high-degree nodes**: `resource_allocation_index`
- **Focus on top/hub attraction**: `preferential_attachment`
- **Have community labels, do in-circle recommendations**: `cn_soundarajan_hopcroft`
- **Community + hidden strong relationships**: `ra_index_soundarajan_hopcroft`
- **Want to distinguish "cohesive vs cross-circle" relationships**: `within_inter_cluster`

---

## 6. Typical Questions That Can Be Directly Answered

- "Who am I most likely to know but haven't met yet?"
- "Which accounts have hidden relationships?"
- "Which cross-department employees are most likely to collaborate?"
- "Which products/content have potential strong associations?"
- "Does this network tend towards cohesive development or cross-circle connections?"

---
