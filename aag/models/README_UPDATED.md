# 更新后的GraphWorkflowDAG和Scheduler使用说明

本文档介绍了更新后的`GraphWorkflowDAG`和`Scheduler`的使用方法，重点是新实现的`build_dag_from_subquery_plan`方法。

## 主要更新

### 1. 简化的GraphWorkflowDAG

原有的复杂`GraphWorkflowDAG`已被简化版本替换，新版本专注于：

- ✅ **拓扑排序支持**：提供`topological_order()`方法
- ✅ **节点属性管理**：每个节点包含问题描述、图算法字段（可设为None）
- ✅ **依赖关系维护**：维护每个节点的入边(`parents_of`)和出边(`children_of`)
- ✅ **循环依赖检测**：自动检测并拒绝包含环的DAG
- ✅ **状态管理**：支持pending/running/success/failed状态

### 2. 完整的build_dag_from_subquery_plan实现

`Scheduler.build_dag_from_subquery_plan()`方法现已完整实现，满足所有要求。

## 使用方法

### 基本使用

```python
from graphllm.planner_and_scheduler.scheduler import Scheduler

# 创建调度器
scheduler = Scheduler()

# 子查询计划
subquery_plan = {
    "subqueries": [
        {
            "id": "q1",
            "query": "分析用户交易行为",
            "depends_on": []
        },
        {
            "id": "q2", 
            "query": "识别欺诈社区",
            "depends_on": ["q1"]
        },
        {
            "id": "q3",
            "query": "计算风险评分",
            "depends_on": ["q1", "q2"]
        }
    ]
}

# 构建DAG
dag = scheduler.build_dag_from_subquery_plan(subquery_plan)

# 获取拓扑序
topo_order = dag.topological_order()
print(f"执行顺序: {topo_order}")

# 查看节点信息
for step_id in topo_order:
    info = dag.get_step_info(step_id)
    print(f"步骤 {step_id}: {info['question']}")
    print(f"  父步骤: {info['parents']}")
    print(f"  子步骤: {info['children']}")
    print(f"  图算法: {info['graph_algorithm']}")  # 始终为None
```

### 数据结构

#### 输入格式

`subquery_plan`必须符合以下格式：

```python
{
    "subqueries": [
        {
            "id": "唯一标识符",           # 必需：子查询的唯一ID
            "query": "问题描述文本",      # 必需：具体的问题内容
            "depends_on": ["前置查询ID"]  # 可选：依赖的其他查询ID列表
        }
    ]
}
```

#### 节点属性

每个DAG节点(`WorkflowStep`)包含以下属性：

```python
@dataclass
class WorkflowStep:
    step_id: int                          # 步骤的唯一ID（自动分配）
    question: str                         # 该节点代表的子问题描述
    graph_algorithm: Optional[str] = None # 图算法字段（按要求设为None）
    status: str = "pending"               # 节点状态
    result: Any = None                    # 节点执行结果
```

## 核心方法

### GraphWorkflowDAG方法

- `topological_order() -> List[int]`: 返回拓扑排序后的步骤ID列表
- `parents_of(step_id: int) -> List[int]`: 获取节点的所有父节点
- `children_of(step_id: int) -> List[int]`: 获取节点的所有子节点
- `ready_steps() -> List[int]`: 获取当前可执行的步骤
- `get_step_info(step_id: int) -> Dict`: 获取步骤详细信息
- `add_step(question: str, graph_algorithm: Optional[str] = None) -> int`: 添加新步骤
- `add_dependency(parent_id: int, child_id: int)`: 添加依赖关系

### 状态管理

- `set_running(step_id: int)`: 设置为运行状态
- `set_success(step_id: int, output_data: Any = None)`: 设置为成功状态
- `set_failed(step_id: int, error: str)`: 设置为失败状态

## 错误处理

系统会自动检测以下错误：

1. **空计划**: 子查询列表为空
2. **格式错误**: 缺少必需字段（id、query）
3. **循环依赖**: 依赖关系形成环
4. **无效依赖**: 依赖的查询ID不存在
5. **自环**: 节点依赖自己

```python
try:
    dag = scheduler.build_dag_from_subquery_plan(plan)
except ValueError as e:
    print(f"DAG构建失败: {e}")
```

## 完整示例

### 金融欺诈检测流程

```python
fraud_detection_plan = {
    "subqueries": [
        {
            "id": "user_profile",
            "query": "获取用户基本档案信息",
            "depends_on": []
        },
        {
            "id": "transaction_analysis",
            "query": "分析交易模式", 
            "depends_on": ["user_profile"]
        },
        {
            "id": "network_analysis",
            "query": "构建交易网络图",
            "depends_on": ["user_profile"]
        },
        {
            "id": "risk_scoring",
            "query": "计算风险分数",
            "depends_on": ["transaction_analysis", "network_analysis"]
        },
        {
            "id": "final_report",
            "query": "生成检测报告",
            "depends_on": ["risk_scoring"]
        }
    ]
}

scheduler = Scheduler()
dag = scheduler.build_dag_from_subquery_plan(fraud_detection_plan)

# 输出拓扑序
print(f"执行顺序: {dag.topological_order()}")

# 查看详细信息
dag.print_dag_info()

# 导出为字典格式
dag_dict = dag.export_as_dict()
print(f"包含 {dag_dict['step_count']} 个步骤, {dag_dict['edge_count']} 条边")
```

## 与原版的区别

| 特性 | 原版 | 更新版 |
|------|------|--------|
| 复杂度 | 高（300+行） | 简化（200行左右） |
| 核心功能 | ✅ 拓扑排序 | ✅ 拓扑排序 |
| 节点属性 | 复杂的StepType枚举 | 简化的WorkflowStep |
| 依赖管理 | ✅ 入边/出边 | ✅ 入边/出边 |
| 环检测 | ✅ 支持 | ✅ 支持 |
| 缓存机制 | 支持 | 移除（简化） |
| 可视化 | DOT格式 | 保留DOT格式 |
| build_dag_from_subquery_plan | 未实现 | ✅ 完整实现 |

## 测试验证

运行测试脚本验证功能：

```bash
cd /home/wangzh/graphllm
python graphllm/test_updated_scheduler.py
```

测试包括：
- ✅ 基本DAG构建
- ✅ 复杂依赖关系
- ✅ 循环依赖检测
- ✅ 错误处理
- ✅ 真实场景应用

## 总结

更新后的实现满足了所有要求：

1. ✅ **支持拓扑排序**：`topological_order()`方法
2. ✅ **节点问题描述**：每个节点包含`question`字段
3. ✅ **图算法字段**：设置为`None`
4. ✅ **依赖关系维护**：完整的入边和出边管理
5. ✅ **完整实现**：`build_dag_from_subquery_plan`方法

代码更加简洁易懂，专注于核心功能，同时保持了与原有接口的兼容性。
