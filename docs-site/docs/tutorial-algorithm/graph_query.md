---
sidebar_position: 12
---
# Graph Query 图查询算子集

**算子类别**：Graph Query（图查询）  
**描述**：Graph query tools for querying specific nodes, neighbors, subgraphs, and paths。用于面向图数据做“查点 / 查边 / 查邻居 / 查路径 / 抽子图 / 做聚合统计”等基础查询能力，适合交互式探索分析、业务检索、风控排查、知识图谱关系发现与可视化取数。

> 与纯图算法（最短路、聚类、中心性等）不同：Graph Query 更偏 **数据获取与筛选**，用于把“要分析的对象/子图”快速定位出来，再交给上层算法或业务逻辑处理。

---

## 一、算子集概述

Graph Query 算子集覆盖 4 类常见查询模式：

1. **节点查询（Node Lookup）**  
   - 精确查一个节点（唯一键：ID / account_id / name 等）
   - 按属性条件筛选一批节点（年龄>30、状态=completed）

2. **关系/边查询（Relationship Filter）**  
   - 按关系类型 + 属性条件过滤边（金额>400、时间范围、时长等）
   - 可直接返回聚合统计（COUNT / SUM / AVG 等）

3. **结构遍历（Neighbors / Paths / Common Neighbors）**  
   - 查 1~k 跳邻居、限定方向与关系类型
   - 查两点之间的路径（最短或所有路径，受 hop 约束）
   - 查共同邻居（可带关系属性过滤）

4. **子图抽取（Subgraph）**  
   - 以中心节点扩展 k 跳形成 ego-network
   - 或按关系属性条件抽取“交易子图/时间窗子图”
   - 或按给定节点列表取诱导子图（只看内部关系）

---

## 二、算子列表

| 算子 | 核心能力 |
|---|---|
| `node_lookup` | 按 label + 唯一键精确查点；或按属性条件筛点 |
| `relationship_filter` | 按 rel_type + 属性条件筛边；可聚合统计 |
| `aggregation_query` | 分组聚合统计（按节点/属性分组，COUNT/SUM/AVG/…） |
| `neighbor_query` | k-hop 邻居遍历（可限定 rel_type / 方向，返回边字段） |
| `path_query` | 两点路径查询（可限定关系类型、方向、最小/最大跳数） |
| `common_neighbor` | 两点共同邻居（可限定 rel_type / 方向 / 边属性过滤） |
| `subgraph` | 子图抽取：中心点扩展模式 / 关系过滤模式 |
| `subgraph_by_nodes` | 按节点列表抽取诱导子图（可选仅内部边） |

---

## 三、通用输入输出约定

### 3.1 输入（Input）

- **label / start_label / end_label**：异构图中用于指定节点类型（如 `Account` / `Person` / `Paper`）
- **key / value / values**：用于按属性定位节点（如 `account_id=ACC_12345`）
- **rel_type**：关系类型（如 `TRANSFER` / `FRIEND` / `CITES`）
- **direction**：方向（`OUTGOING` / `INCOMING` / `BOTH`）
- **conditions / rel_conditions**：属性过滤条件（支持多条件组合与比较运算）
- **return_fields**：返回字段列表（用于减少传输、避免信息丢失）
- **hops / min_hops / max_hops / limit / limit_paths**：控制搜索深度与规模

### 3.2 输出（Output）

- `node_lookup`：单个 node 或 node 列表
- `relationship_filter`：relationship 列表或聚合值
- `aggregation_query`：分组聚合结果列表（group_key + aggregated_value）
- `neighbor_query`：邻居节点/边信息或路径结构（视 return_fields 与实现而定）
- `path_query`：路径列表（通常为节点序列或节点+边结构）
- `common_neighbor`：共同邻居节点列表（可附带字段）
- `subgraph / subgraph_by_nodes`：subgraph（包含 nodes 与 relationships）

---

## 四、算子详细说明

## 4.1 node_lookup —— 节点查询（精确查点 / 条件筛点）

### 功能说明
支持两种模式：
- **单点精确查询**：`label + key + value`
- **多点条件过滤**：`label + conditions`

### 参数要点
- **return_fields**：建议只取必要字段，降低 IO
- **conditions**：支持数值比较（`>`, `<`, `>=`, `<=`, `==`, `!=`）与字符串匹配

### 原理与复杂度
- 唯一键查点：走索引/哈希定位，近似 `O(1)`
- 条件过滤：无索引时扫描为主，近似 `O(n)`

### 可回答问题
- Find the account node with `account_id = "ACC_12345"`
- List all users with `age > 30`
- Query all customers living in US with state `VT`

---

## 4.2 relationship_filter —— 关系过滤（筛边 + 可聚合）

### 功能说明
按 **关系类型**（必填）过滤边，并可进一步指定：
- 起点/终点 label
- 关系属性条件（金额、时间、时长、标记等）
- 聚合统计（COUNT/SUM/AVG/MAX/MIN）

### 参数要点
- **rel_conditions**：支持多条件组合（AND/OR 逻辑由实现约定）
- **aggregate**：当只关心统计结果时建议开启，避免返回海量边
- **return_fields**：返回你关心的交易字段（amount、timestamp、is_sar…）

### 原理与复杂度
- 扫描指定类型的关系并过滤：`O(m)`（m 为该 rel_type 的关系数量）

### 可回答问题
- Find all transactions with `amount > 400`
- List the count of transactions where `is_sar` is False
- Calculate the total amount of all outgoing transactions

---

## 4.3 aggregation_query —— 分组聚合统计

### 功能说明
提供面向图数据的 GROUP BY + 聚合能力，用于：
- 计数（COUNT）
- 求和/均值/极值（SUM/AVG/MAX/MIN）

支持按：
- **节点 label 分组**（per entity）
- **属性分组**（per category）

### 参数要点
- **aggregate_type**：COUNT / SUM / AVG / MAX / MIN（必填）
- **aggregate_field**：SUM/AVG/MAX/MIN 时必填
- **group_by_node / group_by_property**：决定统计维度
- **direction / rel_type**：限制参与统计的关系范围

### 原理与复杂度
- 遍历相关节点/边并分组聚合：`O(n + m)`（与参与统计的数据规模有关）

### 可回答问题
- Count the number of transactions per account
- Calculate the total amount of outgoing transactions per account
- Find the top 10 accounts with the most transactions

---

## 4.4 neighbor_query —— 邻居查询（k-hop）

### 功能说明
从指定起点出发做 **k 跳邻居扩展**（BFS）：
- 1-hop：直接邻居
- 2-hop：朋友的朋友 / 交易对手的对手
- 3-hop+：更大范围的关系圈层（注意爆炸）

### 参数要点
- **hops**：控制扩展深度（默认 1）
- **rel_type / direction**：强烈建议用于约束规模
- **return_fields**：返回边详情（尤其 hops=1 时避免信息丢失）

### 原理与复杂度
- BFS 扩展，最坏近似 `O(d^k)`（d 为平均度，k 为 hops）

### 可回答问题
- Query neighbors of Collins Steven
- Find all 2-hop neighbors of user Alice
- Find all accounts that Collins Steven has transferred money to

---

## 4.5 path_query —— 路径查询（两点如何连起来）

### 功能说明
查询两个节点之间的连接路径，可用于：
- 资金链路追踪（Trace money flow）
- 引用链路追踪（Citation chain）
- 社交关系溯源（Connection discovery）

支持限制：
- 关系类型（只走 TRANSFER 等）
- 方向（OUTGOING/INCOMING/BOTH）
- 最小/最大跳数（控制搜索空间）

### 参数要点
- **max_hops**：强烈建议设置，避免全图爆炸
- **min_hops**：用于排除直接连接（想看“间接关系”）
- **rel_type**：限定特定关系类型，提升语义准确性与性能

### 原理与复杂度
- 最短路径：典型 `O(V+E)`
- 枚举所有路径：可能指数级（图越密越危险）

### 可回答问题
- Find the path from Collins Steven to Nunez Mitchell
- Find all paths between two companies within 5 hops
- Trace the shortest supply chain path from supplier to customer

---

## 4.6 common_neighbor —— 共同邻居查询（互相认识谁 / 共享交易对手）

### 功能说明
返回同时连接到 v1 与 v2 的节点集合（邻居集合交集），用于：
- 互相关联对象（Mutual friends）
- 潜在合谋/共谋检测（Shared transaction partners）
- 相似性与链接预测的基础特征

### 参数要点
- **rel_conditions**：可对“共同邻居所涉及的边”再筛一层（例如 amount>400）
- **direction / rel_type**：决定“共同邻居”的业务语义（共同转入方/共同转出方/共同朋友）

### 原理与复杂度
- 取邻居集合交集：`O(d1 + d2)`（两点度数之和）

### 可回答问题
- Identify mutual friends between user A and user B
- Find common transaction partners of two accounts
- Find common transaction neighbors where transaction amounts are all greater than 400

---

## 4.7 subgraph —— 子图抽取（两种模式）

### 功能说明
提供两种抽取方式：

**模式 1：中心节点扩展（ego network）**  
- 输入：`label/key/value + hops (+ rel_type/direction)`
- 输出：以该节点为中心的 k 跳子图（可用于可视化与局部分析）

**模式 2：按关系过滤抽取（slice by edge filter）**  
- 输入：`rel_type + rel_conditions (+ start_label/end_label) + limit`
- 输出：满足条件的所有关系及其端点组成的子图（如“某月所有转账子图”）

### 参数要点
- **limit_paths / limit**：控制规模（尤其可视化/交互场景）
- **rel_conditions**：用于时间窗/金额窗过滤
- **direction**：资金流/引用链场景下尤为重要

### 原理与复杂度
- 模式 1：`O(d^k)`（随 hops 增长）
- 模式 2：`O(m)`（m 为匹配的关系数量）

### 可回答问题
- Extract subgraph around Collins Steven within 2 hops
- Extract subgraph of all transactions on 2025-05-01
- Extract subgraph of transactions with amounts between 300 and 500

---

## 4.8 subgraph_by_nodes —— 按节点列表抽取诱导子图

### 功能说明
给定一组节点（通过 `label + key + values` 指定），抽取它们之间的关系子图：
- **include_internal=True（默认）**：仅包含这些节点之间的内部边（最常用）
- **include_internal=False**：可能包含到外部节点的边（具体以实现约定为准）

### 参数要点
- **rel_type / direction**：在多关系类型场景下建议指定
- **include_internal**：用于控制是否“只看组内关系”

### 原理与复杂度
- 取指定节点 + 过滤它们之间的边：`O(n + m)`（n 节点数，m 组内边数）

### 可回答问题
- Extract accounts A, B, C and their transfer relationships
- Analyze transaction network among 5 suspicious accounts
- Find all relationships among a set of companies

---

## 五、选型指南（怎么选）

- **查一个实体（按 ID / account_id）**：`node_lookup`（key+value）
- **按属性筛一批实体**：`node_lookup`（conditions）
- **按边属性筛交易/通信记录**：`relationship_filter`
- **直接做统计报表/排名**：`aggregation_query`
- **看某节点局部关系圈（k-hop）**：`neighbor_query`
- **查两点如何关联（路径）**：`path_query`（务必设 max_hops）
- **查共同好友/共同对手/共享供应商**：`common_neighbor`
- **抽一个可视化/分析子图**：`subgraph`（中心扩展或关系过滤）
- **指定一组节点看组内关系**：`subgraph_by_nodes`

---

## 六、工程注意事项与常见坑位

1. **路径与 k-hop 查询最容易爆炸**  
   - 建议始终设置：`max_hops` / `hops`、`rel_type`、`direction`、`limit/limit_paths`

2. **聚合优先（能不拉明细就不拉）**  
   - 只要结果是统计值/TopK，优先用 `relationship_filter(aggregate=...)` 或 `aggregation_query`

3. **return_fields 是性能关键**  
   - 避免返回整条边/整节点的所有字段，尤其是大图与多属性图

4. **direction 影响语义**  
   - 在资金流/引用链等场景，OUTGOING 与 INCOMING 的业务含义完全不同

5. **条件字段类型要统一**  
   - 时间字段建议统一 timestamp/date 格式  
   - 数值字段避免以字符串存储导致比较错误

---

## 七、可直接回答的典型问题

- “找 account_id=ACC_12345 的账户信息”
- “筛出所有金额>400 的转账”
- “每个账户的转账次数 Top10”
- "A 和 B 之间是否存在资金路径（&lt;=5 hop）？"
- “两个人的共同好友是谁？”
- “围绕某个可疑账户抽取 2 跳子图用于可视化排查”
- “抽取指定 5 个账户之间的内部交易网络，判断是否形成团伙”

---
