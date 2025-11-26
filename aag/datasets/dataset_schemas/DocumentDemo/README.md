# DocumentDemo 数据集示例

## 数据集说明

这是一个**文本类型数据集**的示例，展示了如何配置和管理文本数据，以及如何追踪文本到图的转换过程。

## 数据集结构

```
datasets/
├── dataset_schemas/
│   └── DocumentDemo/
│       ├── metadata.yaml          # 数据集元信息（状态、转换信息等）
│       ├── text_schemas.yaml      # 文本文件schema定义
│       └── README.md              # 本文件
│
└── data/
    └── DocumentDemo/
        └── text/                  # 文本文件目录
            ├── demo_document1.md  # 金融反洗钱知识文档
            ├── demo_document2.md  # 图数据分析技术文档
            └── demo_document3.md  # 知识图谱构建指南
```

## 文件说明

### 1. metadata.yaml

存储数据集的元信息，包括：

- **基本信息**：id、name、type、description
- **时间信息**：created_at、updated_at
- **数据内容标记**：has_text_data、has_graph_data、graph_source
- **文本文件信息**：文件列表、格式、大小等
- **转换信息**：转换状态、配置、统计信息
- **数据血缘**：数据来源和转换链

**关键字段说明：**

- `type: "text"`：表示这是文本类型数据集
- `has_graph_data: false`：表示尚未转换为图数据
- `graph_source: null`：表示图数据来源为空（尚未转换）
- `extraction_info.status: "pending"`：表示转换状态为待转换

### 2. text_schemas.yaml

定义文本文件的结构，每个文本文件对应一个数据集条目：

```yaml
datasets:
  - dataset_id: "dataset_002"
    name: "DocumentDemo_text"
    type: text
    schema:
      path: "../../data/DocumentDemo/text/demo_document1.md"
      format: "md"
      encoding: "utf-8"
```

## 与原生图数据集的对比

### 原生图数据集（AMLSim1K）

- `type: "graph"`：原生图数据
- `has_text_data: false`：没有文本数据
- `has_graph_data: true`：有图数据
- `graph_source: "native"`：原生图，不需要转换
- 数据文件在 `data/AMLSim1K/graph/` 目录下

### 文本数据集（DocumentDemo）

- `type: "text"`：文本数据
- `has_text_data: true`：有文本数据
- `has_graph_data: false`：尚未转换为图数据
- `graph_source: null`：尚未转换
- 数据文件在 `data/DocumentDemo/text/` 目录下

## 转换流程

当文本数据集转换为图数据后：

1. **更新 metadata.yaml**：
   - `has_graph_data: true`
   - `graph_source: "extracted"`
   - `extraction_info.status: "completed"`
   - 填充转换统计信息

2. **创建 graph_schemas.yaml**：
   - 定义转换后的图数据schema
   - 包含节点和边的定义

3. **创建图数据文件**：
   - 在 `data/DocumentDemo/graph/` 目录下
   - 包含 vertices.csv 和 edges.csv

## 使用示例

### 查询数据集信息

```python
from pathlib import Path
import yaml

# 读取 metadata.yaml
metadata_path = Path("datasets/dataset_schemas/DocumentDemo/metadata.yaml")
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = yaml.safe_load(f)

print(f"数据集名称: {metadata['name']}")
print(f"数据集类型: {metadata['type']}")
print(f"是否有图数据: {metadata['has_graph_data']}")
print(f"转换状态: {metadata['extraction_info']['status']}")
```

### 检查转换状态

```python
# 检查是否需要转换
if metadata['type'] == 'text' and not metadata['has_graph_data']:
    print("需要转换为图数据")
    print(f"当前状态: {metadata['extraction_info']['status']}")
```

## 注意事项

1. **路径使用相对路径**：schema文件中的路径使用相对路径，相对于schema文件本身
2. **编码统一**：文本文件统一使用 UTF-8 编码
3. **状态同步**：转换后需要同步更新 metadata.yaml 和创建 graph_schemas.yaml
4. **数据血缘**：metadata.yaml 中记录完整的数据血缘，便于追溯

