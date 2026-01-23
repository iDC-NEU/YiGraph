# 子图抽取功能使用指南

## 功能概述

本系统提供了两种子图抽取方法，可以解决不同场景的需求：

### 1. 单中心节点子图抽取 (`subgraph_extract`)
**适用场景**：以某个节点为中心，向外扩展N跳的子图

### 2. 多节点列表子图抽取 (`subgraph_extract_by_nodes`) ⭐ **新增**
**适用场景**：提取指定多个节点及其相互之间的关系

---

## 三种问题场景解决方案

### 场景1：交易转账网络
**问题**：提取账户 A、B、C 及其之间所有的相互转账记录，形成一个小型的资金互动网络

**解决方案**：使用 `subgraph_extract_by_nodes()` 方法

```python
from graph_query import Neo4jGraphClient, Neo4jConfig

# 连接数据库
config = Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", password="password")
client = Neo4jGraphClient(config)

# 提取账户 A、B、C 之间的转账关系
result = client.subgraph_extract_by_nodes(
    label="Account",
    key="node_key",
    values=["Collins Steven", "Nunez Mitchell", "Lee Alex"],
    rel_type="TRANSFER",      # 只关注转账关系
    direction="both",         # 双向（A->B 和 B->A）
    include_internal=False    # 不包含自己转给自己
)

print(f"节点数量: {result['node_count']}")
print(f"关系数量: {result['relationship_count']}")
print(f"节点列表: {result['nodes']}")
print(f"关系列表: {result['relationships']}")
```

**返回结果**：
```python
{
    "nodes": [节点A, 节点B, 节点C],
    "relationships": [A->B的转账, B->C的转账, ...],
    "node_count": 3,
    "relationship_count": 5
}
```

---

### 场景2：社交网络
**问题**：导出以"某大V"为中心，包括其关注者及关注者之间关系的局部网络拓扑

**解决方案**：使用 `subgraph_extract()` 方法（原有功能）

```python
# 以某大V为中心，抽取2跳邻居的子图
result = client.subgraph_extract(
    center=("User", "userId", "big_v_001"),
    hops=2,                   # 2跳邻居
    direction="both",         # 双向关系
    limit_paths=200           # 限制路径数量
)

print(f"节点数量: {len(result['nodes'])}")
print(f"关系数量: {len(result['relationships'])}")
```

**特点**：
- 自动扩展：从中心节点向外扩展指定跳数
- 包含间接关系：不仅包含大V的直接关注者，还包含关注者之间的关系
- 适合探索性分析：发现潜在的社区结构

---

### 场景3：指定节点子图
**问题**：获取包含节点列表 [A, B, C] 及其相互之间所有连边的子图

**解决方案**：使用 `subgraph_extract_by_nodes()` 方法

```python
# 提取指定节点及其相互关系（不限制关系类型）
result = client.subgraph_extract_by_nodes(
    label="Account",
    key="node_key",
    values=["Collins Steven", "Nunez Mitchell", "Lee Alex", "Smith John"],
    direction="both",         # 双向关系
    include_internal=True     # 包含自环（如果有）
    # rel_type=None           # 不限制关系类型
)

print(f"节点数量: {result['node_count']}")
print(f"关系数量: {result['relationship_count']}")

# 按关系类型统计
rel_types = {}
for rel in result['relationships']:
    rel_type = rel.get('type', 'UNKNOWN')
    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

for rel_type, count in rel_types.items():
    print(f"{rel_type}: {count} 条")
```

---

## 方法对比

| 特性 | `subgraph_extract` | `subgraph_extract_by_nodes` |
|------|-------------------|----------------------------|
| **输入** | 单个中心节点 | 多个节点列表 |
| **扩展方式** | 向外扩展N跳 | 只包含指定节点 |
| **关系范围** | 包含扩展路径上的所有节点和关系 | 只包含指定节点之间的关系 |
| **适用场景** | 探索性分析、社区发现 | 精确子图提取、关系验证 |
| **结果可控性** | 结果大小取决于跳数 | 结果大小取决于节点数量 |

---

## 参数说明

### `subgraph_extract_by_nodes()` 参数详解

```python
def subgraph_extract_by_nodes(
    label: str,                    # 节点标签（如 "Account", "User"）
    key: str,                      # 节点属性键（如 "node_key", "userId"）
    values: List[Any],             # 节点属性值列表（如 ["A", "B", "C"]）
    *,
    include_internal: bool = True, # 是否包含自环（A->A）
    rel_type: Optional[str] = None,# 关系类型（None=所有类型）
    direction: str = "both",       # 方向："out"=单向, "in"=反向, "both"=双向
    where: Optional[str] = None    # WHERE 过滤条件（如 "r.amount > 1000"）
) -> JsonDict
```

**返回值**：
```python
{
    "nodes": [节点对象列表],
    "relationships": [关系对象列表],
    "node_count": 节点数量,
    "relationship_count": 关系数量
}
```

---

## 自然语言查询支持

系统已集成到自然语言查询引擎，支持以下问法：

### 示例1：多节点子图
```
问题：提取账户 Collins Steven、Nunez Mitchell、Lee Alex 之间的转账关系
```

### 示例2：单中心子图
```
问题：获取 Collins Steven 周围2跳的子图
```

### 示例3：指定节点关系
```
问题：抽取包含 Collins Steven 和 Nunez Mitchell 及其相互关系的子图
```

---

## 性能建议

1. **节点数量控制**：
   - `subgraph_extract_by_nodes`: 建议节点数量 < 100
   - `subgraph_extract`: 建议跳数 ≤ 3

2. **关系类型过滤**：
   - 指定 `rel_type` 可以显著提升查询速度
   - 使用 `where` 条件进一步过滤结果

3. **方向选择**：
   - 如果只需要单向关系，使用 `direction="out"` 或 `"in"`
   - 双向查询 (`"both"`) 会返回更多结果

---

## 常见问题

### Q1: 如果某个节点不存在会怎样？
**A**: 系统会自动忽略不存在的节点，只返回存在的节点及其关系。

### Q2: 如何只获取节点之间的直接关系？
**A**: 使用 `subgraph_extract_by_nodes()` 方法，它只返回指定节点之间的直接关系。

### Q3: 如何获取包含间接关系的子图？
**A**: 使用 `subgraph_extract()` 方法，指定 `hops` 参数来控制扩展深度。

### Q4: 两种方法可以组合使用吗？
**A**: 可以。例如：
1. 先用 `subgraph_extract()` 获取中心节点周围的所有节点
2. 从结果中筛选感兴趣的节点
3. 再用 `subgraph_extract_by_nodes()` 精确提取这些节点之间的关系

---

## 测试代码

完整的测试代码请参考：[`test_subgraph_by_nodes.py`](test_subgraph_by_nodes.py)

运行测试：
```bash
cd AAG/aag/computing_engine/graph_query
python test_subgraph_by_nodes.py
```

---

## 总结

✅ **场景1（交易转账）**：使用 `subgraph_extract_by_nodes()` - **完全支持**

✅ **场景2（社交网络）**：使用 `subgraph_extract()` - **完全支持**

✅ **场景3（指定节点子图）**：使用 `subgraph_extract_by_nodes()` - **完全支持**

所有三种场景现在都可以通过相应的方法完美解决！
