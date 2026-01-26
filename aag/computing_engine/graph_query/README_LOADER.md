# Neo4j图数据加载器使用说明

## 概述

[`Neo4jGraphLoader`](load_data_into_neo4j.py:11) 是一个通用的Neo4j图数据加载器，支持根据schema配置动态加载图数据到Neo4j数据库。

## 主要特性

1. **动态node_key构建**：支持单个字段或多个字段组合作为节点的唯一标识
2. **灵活的属性加载**：
   - 如果`attribute_fields`为空列表，则加载CSV文件中的所有列
   - 如果`attribute_fields`指定了字段列表，则只加载指定的字段
3. **支持多种数据源**：
   - 直接传入schema字典
   - 从YAML文件读取schema
   - 从DatasetManager获取schema
4. **自动创建索引和约束**：为node_key和id_field自动创建索引以提升查询性能

## Schema配置格式

### Vertex配置

```python
{
    'path': '/path/to/vertices.csv',           # CSV文件路径
    'type': 'account',                          # 节点类型（会转换为Neo4j标签）
    'query_field': ['last_name', 'first_name'], # 用于构建node_key的字段（单个或列表）
    'id_field': 'acct_id',                      # ID字段
    'label_field': 'prior_sar_count',           # 标签字段（可选）
    'attribute_fields': []                      # 属性字段列表（空=加载所有列）
}
```

### Edge配置

```python
{
    'path': '/path/to/edges.csv',      # CSV文件路径
    'type': 'transfer',                 # 边类型（会转换为Neo4j关系类型）
    'source_field': 'orig_acct',        # 源节点字段
    'target_field': 'bene_acct',        # 目标节点字段
    'label_field': 'is_sar',            # 标签字段（可选）
    'weight_field': 'base_amt',         # 权重字段（可选）
    'attribute_fields': []              # 属性字段列表（空=加载所有列）
}
```

## 使用示例

### 示例1: 直接使用schema字典

```python
from aag.computing_engine.graph_query.load_data_into_neo4j import Neo4jGraphLoader

schema = {
    'vertex': [
        {
            'path': '/path/to/accounts.csv',
            'type': 'account',
            'query_field': ['last_name', 'first_name'],  # 多字段组合
            'id_field': 'acct_id',
            'label_field': 'prior_sar_count',
            'attribute_fields': []  # 加载所有列
        }
    ],
    'edge': [
        {
            'path': '/path/to/transactions.csv',
            'type': 'transfer',
            'source_field': 'orig_acct',
            'target_field': 'bene_acct',
            'label_field': 'is_sar',
            'attribute_fields': []  # 加载所有列
        }
    ]
}

loader = Neo4jGraphLoader(
    uri='bolt://localhost:7687',
    username='neo4j',
    password='password',
    schema=schema
)

success = loader.load_all_data(clear_existing=True)
```

### 示例2: 从YAML文件加载

```python
from aag.computing_engine.graph_query.load_data_into_neo4j import load_from_yaml

success = load_from_yaml(
    yaml_path='path/to/graph_schemas.yaml',
    uri='bolt://localhost:7687',
    username='neo4j',
    password='password',
    clear_existing=True
)
```

### 示例3: 在Scheduler中自动加载（已集成）

**重要：** 在 [`scheduler.py`](../../../engine/scheduler.py:175) 中，当调用 [`specific_analysis_dataset()`](../../../engine/scheduler.py:175) 方法时，会自动调用 [`_load_graph_to_neo4j()`](../../../engine/scheduler.py:173) 将图数据加载到Neo4j。

```python
# 在scheduler中使用（自动加载）
from aag.engine.scheduler import Scheduler
from aag.config.engine_config import EngineConfig

# 初始化scheduler
config = EngineConfig()
scheduler = Scheduler(config)

# 设置分析数据集 - 会自动加载到Neo4j
scheduler.specific_analysis_dataset("AMLSim1K", dtype="graph")

# 之后就可以直接使用图查询功能
result = await scheduler.execute("查询所有账户")
```

**工作流程：**
1. 调用 `specific_analysis_dataset("AMLSim1K")`
2. 从 `DatasetManager` 获取数据集配置
3. 从配置中提取 schema 信息（包含 vertex 和 edge 配置）
4. 从 `self.config.retrieval.database.neo4j` 获取 Neo4j 连接配置
5. 自动创建 `Neo4jGraphLoader` 并加载数据
6. 数据加载完成后，可以使用 `nl_query_engine` 进行图查询

**配置要求：**
确保在配置文件中启用了Neo4j：
```yaml
retrieval:
  database:
    neo4j:
      enabled: true
      uri: "bolt://localhost:7687"
      user: "neo4j"
      password: "password"
```

### 示例4: 只加载指定属性

```python
schema = {
    'vertex': [
        {
            'path': '/path/to/accounts.csv',
            'type': 'account',
            'query_field': ['last_name', 'first_name'],
            'id_field': 'acct_id',
            'attribute_fields': [  # 只加载这些字段
                'acct_id', 'first_name', 'last_name', 
                'prior_sar_count', 'type', 'initial_deposit'
            ]
        }
    ],
    'edge': [
        {
            'path': '/path/to/transactions.csv',
            'type': 'transfer',
            'source_field': 'orig_acct',
            'target_field': 'bene_acct',
            'attribute_fields': [  # 只加载这些字段
                'tran_id', 'base_amt', 'is_sar', 'tran_timestamp'
            ]
        }
    ]
}
```

### 示例5: 使用单个字段作为node_key

```python
schema = {
    'vertex': [
        {
            'path': '/path/to/accounts.csv',
            'type': 'account',
            'query_field': 'acct_id',  # 单个字段
            'id_field': 'acct_id',
            'attribute_fields': []
        }
    ],
    # ...
}
```

## query_field说明

`query_field` 用于构建节点的唯一标识 `node_key`：

- **单个字段**：`query_field: 'acct_id'` → `node_key = "12345"`
- **多个字段**：`query_field: ['last_name', 'first_name']` → `node_key = "Smith John"`

多个字段会用空格连接。这个`node_key`会作为Neo4j中节点的唯一标识，并创建唯一性约束。

## attribute_fields说明

`attribute_fields` 控制哪些字段会被加载为节点/边的属性：

- **空列表** `[]`：加载CSV文件中的所有列（除了用于匹配的字段）
- **指定列表** `['field1', 'field2']`：只加载指定的字段

对于节点，会自动排除`node_key`字段（因为已单独处理）。
对于边，会自动排除`source_field`和`target_field`（因为用于匹配节点）。

## 运行示例脚本

```bash
cd AAG_3/AAG
python -m aag.computing_engine.graph_query.load_data_example
```

然后根据提示选择要运行的示例（1-5）。

## 注意事项

1. **CSV文件路径**：确保schema中的path字段指向正确的CSV文件路径
2. **Neo4j连接**：确保Neo4j服务正在运行且连接信息正确
3. **数据清空**：`clear_existing=True`会删除数据库中的所有数据，请谨慎使用
4. **字段匹配**：边的`source_field`和`target_field`必须与节点的`id_field`对应
5. **批量处理**：默认批量大小为1000条，可以根据需要调整

## 与原AMLSimNeo4jLoader的区别

| 特性 | 原AMLSimNeo4jLoader | 新Neo4jGraphLoader |
|------|-------------------|-------------------|
| 数据源 | 硬编码文件路径 | 从schema动态读取 |
| node_key | 固定为"last_name first_name" | 根据query_field动态构建 |
| 属性加载 | 硬编码所有字段 | 根据attribute_fields动态加载 |
| 节点类型 | 固定为Account | 根据schema的type字段 |
| 边类型 | 固定为TRANSACTION | 根据schema的type字段 |
| 通用性 | 仅适用于AMLSim数据 | 适用于任何符合schema格式的图数据 |

## API参考

### Neo4jGraphLoader类

#### 构造函数
```python
Neo4jGraphLoader(uri, username, password, schema)
```

#### 主要方法

- [`load_all_data(clear_existing=False)`](load_data_into_neo4j.py:463): 加载所有数据
- [`load_vertices(vertex_config)`](load_data_into_neo4j.py:169): 加载顶点数据
- [`load_edges(edge_config, vertex_configs)`](load_data_into_neo4j.py:267): 加载边数据
- [`verify_data(vertex_configs, edge_configs)`](load_data_into_neo4j.py:371): 验证加载结果
- [`clear_database()`](load_data_into_neo4j.py:57): 清空数据库
- [`close_connection()`](load_data_into_neo4j.py:454): 关闭连接

### 辅助函数

- [`load_from_yaml(yaml_path, uri, username, password, clear_existing)`](load_data_into_neo4j.py:458): 从YAML文件加载数据

## 故障排查

### 连接失败
- 检查Neo4j服务是否运行
- 验证URI、用户名和密码是否正确
- 确认防火墙设置允许连接

### 数据加载失败
- 检查CSV文件路径是否正确
- 验证CSV文件格式是否正确
- 确认schema配置中的字段名与CSV列名匹配

### 节点匹配失败（边加载时）
- 确保边的`source_field`和`target_field`值在节点的`id_field`中存在
- 检查数据类型是否一致（字符串vs数字）
