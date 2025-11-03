# GraphProcessor - 图处理器

`GraphProcessor` 是一个用于处理图数据库提取的图数据并运行各种图算法的自定义类。

## 功能特性

### 1. 图数据转换
- 支持从边列表创建有向图和无向图
- 支持带权重的边
- 自动处理图数据库提取的图数据格式

### 2. 图算法支持
- **PageRank**: 计算节点的重要性分数
- **连通分量 (CC)**: 识别图中的连通组件
- **最短路径**: 计算两个节点间的最短路径
- **介数中心性**: 计算节点的介数中心性
- **接近中心性**: 计算节点的接近中心性
- **度中心性**: 计算节点的度中心性

## 安装依赖

确保已安装以下依赖：
```bash
pip install networkx
```

## 基本使用

### 1. 创建图处理器实例
```python
from graph_processor import GraphProcessor

processor = GraphProcessor()
```

### 2. 从边列表创建图
```python
# 基本边列表 (source, target)
edges = [
    ('A', 'B'),
    ('B', 'C'),
    ('C', 'D'),
    ('D', 'A')
]

# 创建有向图
graph = processor.create_graph_from_edges(edges, directed=True)

# 创建无向图
graph = processor.create_graph_from_edges(edges, directed=False)
```

### 3. 带权重的边
```python
# 带权重的边列表 (source, target, weight)
weighted_edges = [
    ('A', 'B', 2.0),
    ('B', 'C', 1.5),
    ('C', 'D', 3.0),
    ('D', 'A', 1.0)
]

graph = processor.create_graph_from_edges(weighted_edges, directed=True)
```

### 4. 运行图算法

#### PageRank算法
```python
pagerank_scores = processor.run_pagerank()
print(pagerank_scores)
# 输出: {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
```

#### 连通分量算法
```python
components = processor.run_connected_components()
print(components)
# 输出: [{'A', 'B', 'C', 'D'}]
```

#### 最短路径
```python
path = processor.run_shortest_path('A', 'D')
print(path)
# 输出: ['A', 'B', 'C', 'D']
```

#### 中心性算法
```python
# 介数中心性
betweenness = processor.run_betweenness_centrality()

# 接近中心性
closeness = processor.run_closeness_centrality()

# 度中心性
degree = processor.run_degree_centrality()
```

### 5. 通用算法运行器
```python
# 使用通用方法运行算法
result = processor.run_algorithm('pagerank')
result = processor.run_algorithm('cc')  # 连通分量
result = processor.run_algorithm('shortest_path', source='A', target='D')
```

### 6. 获取图信息
```python
info = processor.get_graph_info()
print(info)
# 输出: {'节点数': 4, '边数': 4, '图类型': '有向图', ...}
```

### 7. 执行计划功能
```python
# 方法1: 直接创建执行计划
plan = [
    {
        'algorithm': 'pagerank',
        'name': 'pagerank_scores',
        'params': {'alpha': 0.85}
    },
    {
        'algorithm': 'cc',
        'name': 'connected_components'
    },
    {
        'algorithm': 'betweenness',
        'name': 'betweenness_centrality'
    }
]

results = processor.execute_plan(plan)
print(results)

# 方法2: 使用create_execution_plan创建计划
algorithms = ['pagerank', 'cc', 'closeness']
algorithm_params = {
    'pagerank': {'alpha': 0.9},
    'cc': {}
}
result_names = ['pagerank_result', 'components', 'closeness_scores']

plan = processor.create_execution_plan(algorithms, algorithm_params, result_names)
results = processor.execute_plan(plan)
```

## 支持的算法

| 算法名称 | 方法名 | 描述 |
|---------|--------|------|
| PageRank | `run_pagerank()` | 计算节点重要性分数 |
| 连通分量 | `run_connected_components()` | 识别连通组件 |
| 最短路径 | `run_shortest_path(source, target)` | 计算最短路径 |
| 介数中心性 | `run_betweenness_centrality()` | 计算介数中心性 |
| 接近中心性 | `run_closeness_centrality()` | 计算接近中心性 |
| 度中心性 | `run_degree_centrality()` | 计算度中心性 |

## 执行计划功能

### 执行计划方法

| 方法名 | 描述 |
|--------|------|
| `execute_plan(plan)` | 根据执行计划按顺序执行算法 |
| `create_execution_plan(algorithms, algorithm_params, result_names)` | 创建执行计划 |

### 执行计划格式

每个执行计划步骤包含以下字段：
- `algorithm`: 算法名称（必需）
- `params`: 算法参数字典（可选）
- `name`: 结果名称（可选，默认为算法名称）

## 错误处理

类包含完善的错误处理机制：

```python
# 未初始化图时运行算法会抛出异常
try:
    processor.run_pagerank()
except ValueError as e:
    print(f"错误: {e}")

# 运行不存在的算法
try:
    processor.run_algorithm('invalid_algorithm')
except ValueError as e:
    print(f"错误: {e}")
```

## 示例

查看 `example_usage.py` 文件获取完整的使用示例。

## 测试

运行测试以确保功能正常：

```bash
python test_graph_processor.py
```

## 日志

类使用Python标准logging模块记录操作信息：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 注意事项

1. 在运行任何算法之前，必须先调用 `create_graph_from_edges()` 方法创建图
2. 对于有向图，连通分量算法使用强连通分量
3. 对于无向图，连通分量算法使用普通连通分量
4. 所有中心性算法的分数都归一化到 [0, 1] 区间
5. 最短路径算法在路径不存在时返回 `None`

## 扩展

可以通过继承 `GraphProcessor` 类来添加新的图算法：

```python
class ExtendedGraphProcessor(GraphProcessor):
    def run_custom_algorithm(self):
        # 实现自定义算法
        pass
``` 