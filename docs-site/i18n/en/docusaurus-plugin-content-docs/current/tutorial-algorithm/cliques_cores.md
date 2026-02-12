---
sidebar_position: 10
---
# Cliques & Cores Operator Set


**Operator Category**: Cliques & Cores  

**Description**: Used to discover "highly cohesive tight groups" and "stable core structures" in networks, characterize the deep structural strength and cohesion of networks, and serve as an important tool for community analysis, core member identification, and robustness analysis.


---
## 1. Operator Set Overview
The Cliques & Cores operator set mainly solves two types of structural problems:
### 1️⃣ Clique-related Problems
- Does there exist a set of nodes where **every pair is directly connected**?
- Which are the "non-extendable" maximal cliques?
- Considering weights, which clique has the highest "value"?
👉 Clique emphasizes **fully connected** structure with strict constraints, suitable for discovering **highly exclusive, strongly consistent small groups**.
---
### 2️⃣ Core / K-core-related Problems
- What is the stable structure remaining after continuously stripping off nodes with "too few connections"?
- Which nodes are in the "core layer" of the network and which are only peripheral?
- What is the deepest layer the network can be stripped to (maximum K value)?
👉 Core emphasizes **stability and hierarchy**, with looser constraints than clique, suitable for analyzing the **skeleton and resilience of large-scale networks**.
---
## 2. Operator List
| Operator | Core Capability |
|---|---|
| `find_cliques` | Enumerate all Maximal Cliques |
| `max_weight_clique` | Find the clique with the maximum sum of weights |
| `k_core` | Extract the core subgraph for a specified K |
| `core_number` | Calculate the core level of each node |
---
## 3. General Input and Output Conventions
### 3.1 Input
- **G**: NetworkX graph  
  - Clique algorithms require **undirected graphs**
  - Core algorithms support **directed / undirected / multigraphs**
- **nodes (optional)**: Specify nodes that must be included (find_cliques only)
- **weight (optional)**: Node weight field (max_weight_clique)
- **k (optional)**: Core order (k_core)
### 3.2 Output
- Clique algorithms:  
  - `find_cliques` → iterator[list[node]]  
  - `max_weight_clique` → clique node list + total weight
- Core algorithms:  
  - `core_number` → dict[node → core_index]  
  - `k_core` → subgraph (NetworkX graph)
---
## 4. Detailed Operator Descriptions
### 4.1 find_cliques —— Maximal Clique Enumeration
#### Function Description
Enumerate all **Maximal Cliques** in the graph:  
> A clique is called a "maximal clique" if no additional node can be added to it while maintaining full connectivity.
⚠️ Note:  
- **Maximal Clique ≠ Maximum Clique**  
- There may be many maximal cliques with significant differences in scale
#### Key Parameters
- **nodes (optional)**:  
  - Only return maximal cliques that "contain these nodes"  
  - An error will be reported directly if `nodes` itself is not a clique (to ensure semantic correctness)
#### Principle and Complexity
- Exhaustive search for all complete subgraphs
- **Time Complexity**: Exponential in the worst case  
  `O(3^(V/3))`
#### Applicable Scenarios
- "Acquaintance circles" in social networks
- "Fully interconnected small gangs" in fraud/criminal networks
- Discovery of protein complexes and fully collaborative teams
---
### 4.2 max_weight_clique —— Maximum Weight Clique
#### Function Description
Find the **clique with the maximum sum of node weights** among all cliques.
Different from `find_cliques`, it focuses on:
- Not necessarily the largest in scale
- But the one with the **"highest total value"**
#### Key Parameters
- **weight**:
  - Specify the node weight attribute
  - If set to `None`, all node weights default to 1 (degenerate to "maximum scale clique")
#### Principle and Complexity
- NP-hard problem
- **Time Complexity**: Exponential (closely related to graph scale)
#### Applicable Scenarios
- Identification of high-value "inner circles" (influence, revenue, weight)
- Personnel/portfolio selection: requiring full mutual compatibility and maximum total value
- Screening of investment portfolios, advertising channels, and gene modules
---
### 4.3 core_number —— Core Number Calculation
#### Function Description
Calculate the **Core Number** for each node:
> The core number of a node =  
> the k value of the **maximum k-core** that it can participate in.
Intuitive understanding:
- The larger the core number  
- The more the node is "in the deep core of the network"
#### Principle and Complexity
- Linear stripping algorithm based on degree decrement
- **Time Complexity**: `O(V + E)`
#### Applicable Scenarios
- Identification of core members / core devices / core papers
- Evaluation of network influence and propagation potential
- Network hierarchical structure analysis (core–periphery)
---
### 4.4 k_core —— K-Core Subgraph Extraction
#### Function Description
Extract a **k-core subgraph**:
> k-core =  
> the largest induced subgraph where all nodes have a degree ≥ k  
> (recursively remove nodes with degree < k until stable)
#### Key Parameters
- **k**:
  - If not specified → return the main core (maximum k-core)
- **core_number (optional)**:
  - Passing in pre-calculated core numbers can significantly accelerate the process
#### Principle and Complexity
- Recursively strip low-degree nodes
- **Time Complexity**: `O(V + E)`
#### Applicable Scenarios
- Network skeleton extraction
- Denoising (removal of edge/zombie nodes)
- Analysis of stable structures and network resilience
---
## 5. Clique vs Core: How to Choose?
| Dimension | Clique | Core |
|---|---|---|
| Connectivity Requirement | Fully connected | At least k neighbors |
| Strictness | Very strict | Relatively loose |
| Scale | Usually small | Can be very large |
| Complexity | Exponential | Linear |
| Typical Use Case | Small and strong "tight cliques" | Large-scale core skeleton |
**Rule of Thumb**:
- 👉 To find **small circles where "everyone knows each other"** → Clique  
- 👉 To find **core structures that "remain stable after continuous stripping"** → Core  
---
## 6. Quick Selection Guide
- **List all tight small groups**: `find_cliques`
- **Find the most valuable fully interconnected group**: `max_weight_clique`
- **Determine who is in the deepest core layer of the network**: `core_number`
- **Extract the network backbone / remove peripheral noise**: `k_core`
---
## 7. Typical Answerable Questions
- "What are the fully interconnected small circles in the network?"
- "Who are the core nodes that are the most difficult to strip off?"
- "What remains of the network after stripping off all peripheral nodes?"
- "Under the premise of full mutual compatibility, which set of nodes has the highest value?"
- "Is this person/device/paper a core member or a peripheral role?"
---
