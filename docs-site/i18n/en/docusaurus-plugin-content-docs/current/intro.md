---
sidebar_position: 1
title: Introduction
---

# YiGraph

## 1. Product Overview

**YiGraph is an end-to-end intelligent graph data analysis agent system** designed to help users quickly gain insights into key relationships from complex data.

YiGraph can automatically extract entities and relationships from various raw data sources such as logs, documents, and tables to build structured graph data. Users only need to describe business problems in **natural language**, and the system will automatically plan the analysis process, complete calculations, and generate **clear, interpretable, and traceable analysis reports**.

Internally, **large language models** are responsible for understanding user intent, breaking down analysis tasks, and organizing final outputs. The core technology supporting the reliability of analysis results is **AAG (Analytics-Augmented Generation)**.
AAG treats analytical computation as a core capability, invoking graph algorithms and graph systems at key stages to complete verifiable calculations, which are then interpreted and summarized by the model.

Therefore, YiGraph is not just a conversational AI that "answers questions", but an intelligent graph analysis agent that can transform business problems into **executable and reviewable analysis processes**.

---

## 2. What Problems Does YiGraph Solve

In many business scenarios, the real value behind data lies in "relationships":

- Connections between people in social networks
- Account and fund flows in financial transactions
- Associations between accounts, devices, and addresses in risk control
- Trading and transmission paths between enterprises in supply chains
- Entity and semantic relationships in knowledge graphs

These can all be abstracted as "graphs" (nodes + edges).
However, graph data is usually structurally complex, large-scale, and has a high barrier to entry: non-professional users find it difficult to accurately translate their business problems into "graph problems", let alone select algorithms, build graphs, run analyses, and interpret results.

**YiGraph's goal** is to make this simple:
Allow users to complete graph analysis as if "asking questions" — starting from natural language and obtaining trustworthy analysis conclusions and reports end-to-end.

---

## 3. Product Positioning and Core Value

- **Positioning**: Intelligent graph analysis agent platform for complex relational data (integrated graph analysis from data to conclusions)
- **Target Users**: Risk control/compliance/security analysis teams, data analysts, and business decision support teams (especially scenarios requiring "relational insights")
- **Core Value**: **Discover deep relationships in big data**
  - "Connect" data relationships scattered across different systems
  - Quickly discover hidden paths, groups, and risk transmission routes
  - Output interpretable and traceable results for easy review and implementation

---

## 4. Applicable Business Scenarios (Examples)

YiGraph can flexibly adapt to different industries and business needs, covering various complex relational data analysis scenarios, including but not limited to:

- **Financial anti-money laundering and suspicious transaction analysis**: Automatically build transaction networks from massive transaction flows to identify abnormal fund paths and suspicious transaction loops
- **E-commerce risk control and wool party identification**: Integrate multi-source data such as accounts, devices, and addresses to build graphs and discover organized fraud and associated malicious behavior
- **Enterprise association and risk investigation**: Build graphs through enterprise, equity, and transaction relationships to penetrate complex structures and identify potential compliance and operational risks
- **Park/city event analysis**: Unify access control, trajectory, and event data into graphs to restore personnel relationships and event evolution processes
- **Supply chain risk analysis**: Integrate enterprise and transaction data to build supply chain networks, locate hidden associated risks and transmission paths

---

## 5. Core Capabilities (AAG-Driven End-to-End Graph Analysis)

YiGraph's core capabilities come from **AAG (Analytics-Augmented Generation)**:
It combines "flexible semantic understanding" with "verifiable analysis processes" to complete end-to-end processing from natural language to structured conclusions.
AAG's goal is not to let large models "analyze everything themselves", but to have them invoke professional analysis modules (such as graph algorithms, graph systems, etc.) at appropriate times and organize the results into reports that users can understand and review.

AAG mainly includes three types of key capabilities:

### 5.1 Knowledge-Driven Task Planning
The system first understands what the user's question "wants to solve", then breaks it down into executable analysis steps, such as:
- What data fields and relationships are needed
- What kind of graph should be built (which entities, which relationships)
- What analysis methods and parameters should be used
- How analysis results should be interpreted and presented

> You don't need to understand graph algorithms; the system will translate "what I want to query" into "how to do the analysis".

### 5.2 Algorithm-Centric Reliable Execution
YiGraph will not let the model arbitrarily "write a piece of uncontrollable code and run it".
Instead, it centers on "verifiable algorithm modules" for invocation and combination, making each analysis step:
- **Reproducible** (same input yields stable and consistent output)
- **Traceable** (know which algorithms were used and which steps were executed)
- **More reliable** (key calculations are completed by professional modules rather than pure text reasoning)

#### Rich Graph Algorithm Library

YiGraph has built-in **93 professional graph algorithms**, covering **10 major categories**, providing powerful algorithm support for various graph analysis scenarios:

| Algorithm Category | Number of Algorithms | Core Capabilities |
|---------|---------|---------|
| **Basics** | 10 | Graph structure determination, basic traversal, topological sorting, dependency analysis |
| **Path** | 13 | Shortest path, Eulerian path, DAG longest path, reachability determination |
| **Centrality** | 14 | Key node identification, influence assessment, bridge location, seed selection |
| **Connectivity & Components** | 13 | Connected components, strongly connected components, cut vertices/edges, network robustness |
| **Clustering & Community** | 17 | Community detection, clustering coefficient, cycle detection, gang identification |
| **Tree & Spanning Tree** | 3 | Minimum/maximum spanning tree, network skeleton extraction |
| **Flow & Cut** | 5 | Maximum flow, minimum cut, capacity planning, bottleneck analysis |
| **Matching & Coloring** | 6 | Optimal matching, resource allocation, conflict detection |
| **Cliques & Cores** | 4 | Tight group discovery, core member identification, k-core analysis |
| **Distance & Measures** | 8 | Network diameter, center/periphery, assortativity coefficient, compactness analysis |

These algorithms cover the full process requirements from basic graph operations to advanced network analysis, and can support:
- **Relationship chain analysis**: Shortest path, dependency chain, fund flow tracking
- **Key node identification**: PageRank, betweenness centrality, influence assessment
- **Gang/community discovery**: Louvain, Leiden, label propagation, k-clique
- **Network robustness**: Connectivity analysis, cut vertices/edges, minimum cut
- **Capacity and flow**: Maximum flow, minimum cut, bottleneck identification
- **Structural insights**: Clustering coefficient, assortativity, network diameter

> For detailed algorithm descriptions and usage guides, please refer to the [Algorithm Documentation](./tutorial-algorithm/basic.md)

### 5.3 Task-Aware Graph Construction
YiGraph will not indiscriminately build all raw data into one large graph.
It will selectively extract and construct "entities and relationships relevant to the problem" based on current task needs, avoiding interference from irrelevant structures, and organize the graph into a form more suitable for execution, thereby improving efficiency and result quality.

---

## 6. Data Support and Usage Methods

### 6.1 Multiple Data Input Types
YiGraph supports various common data formats, adapting to different starting points from "existing graph data" to "building graphs from scratch":
- **Text/Logs**: Reports, audit materials, logs, event descriptions, etc.
- **Existing Relational Data/Knowledge Graphs**: Existing node-edge data or graph assets

### 6.2 Multiple Usage Methods (Suggested Naming)
To facilitate different users, YiGraph can provide multiple interaction and control granularities (names can be adjusted according to final product positioning):
- **Automatic Analysis Mode**: Input question → System automatically completes graph building, analysis, and report generation
- **Interactive Guidance Mode**: Key steps (such as data scope, entity types, analysis objectives) can be confirmed or adjusted by users
- **Expert Configuration Mode**: Advanced users can configure algorithms, parameters, and processes with finer granularity

---

## 7. Product Advantages Summary

- ✅ **End-to-End**: From raw data to graph construction, graph analysis, and report output, completed in one pipeline
- ✅ **Low Barrier**: Complete complex relational analysis by asking questions in natural language
- ✅ **Interpretable and Traceable**: Results not only "give conclusions" but also explain "how they were obtained"
- ✅ **More Reliable**: AAG ensures key calculations are completed by verifiable analysis modules, reducing "seemingly reasonable but unreviewable" answers

---

(Here you can place: product demo link / documentation entry / contact information)


📽 **Product Demo Video**
(Demo link can be placed here)

⭐️ **Welcome to Star / Follow AAG for the latest product progress and capability updates**

---

## 📞 Contact Us

Welcome to connect with us through the following channels:

<table style={{width: '100%', borderCollapse: 'collapse', border: 'none'}}>
  <tr>
    <td style={{textAlign: 'center', border: 'none', padding: '20px'}}>
      <h4>WeChat</h4>
      <img src={require('@site/static/img/wechat.png').default} alt="WeChat" width="180"/>
    </td>
    <td style={{textAlign: 'center', border: 'none', padding: '20px'}}>
      <h4>Xiaohongshu</h4>
      <img src={require('@site/static/img/redbook.png').default} alt="Xiaohongshu" width="180"/>
    </td>
    <td style={{textAlign: 'center', border: 'none', padding: '20px'}}>
      <h4>Twitter</h4>
      <img src={require('@site/static/img/twitter.png').default} alt="Twitter" width="180"/>
    </td>
  </tr>
</table>
