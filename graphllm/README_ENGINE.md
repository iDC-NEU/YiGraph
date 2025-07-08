# GraphLLM Engine

端到端的Graph Analysis + RAG + LLM框架，集成了知识图谱构建、多模态检索和大语言模型生成功能。

## 架构概览

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档输入      │    │   知识图谱      │    │   向量索引      │
│   (PDF/Text)    │───▶│   (NebulaGraph) │    │   (Milvus)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM生成       │◀───│   多模态检索     │◀───│   查询输入      │
│   (Ollama/      │    │   (Graph+Vector)│    │   (用户问题)    │
│    OpenAI)      │    └─────────────────┘    └─────────────────┘
└─────────────────┘
```

## 主要特性

- **多模态检索**: 结合图检索和向量检索的优势
- **灵活配置**: 支持多种LLM和嵌入模型
- **性能监控**: 内置性能指标收集和分析
- **易于扩展**: 模块化设计，便于添加新功能
- **多种运行模式**: 交互式、批处理、文档处理
- **配置文件支持**: 支持YAML配置文件

## 快速开始

### 1. 环境准备

确保已安装并启动必要的服务：

```bash
# 启动NebulaGraph
docker run -d --name nebula-graphd \
  -p 9669:9669 -p 19669:19669 \
  vesoft/nebula-graphd:v3.4.0

# 启动Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.3.3

# 启动Ollama (可选)
ollama serve
ollama pull llama3.1:70b
```

### 2. 基础使用

```python
from graphllm.graphllm_engine import GraphLLMEngine, create_engine_config

# 创建配置
config = create_engine_config(
    graph_space_name="my_space",
    vector_collection_name="my_collection",
    llm_model="llama3.1:70b",
    embedding_model="BAAI/bge-large-en-v1.5"
)

# 初始化Engine
engine = GraphLLMEngine(config)

# 处理文档
documents = [
    "Graph Neural Networks are powerful for graph-structured data.",
    "Knowledge graphs represent entities and relationships."
]
engine.process_documents(documents)

# 查询
result = engine.query("What are Graph Neural Networks?")
print(result['answer'])
```

### 3. 使用配置文件

```yaml
# config.yaml
database:
  graph:
    space_name: "my_space"
    server_ip: "127.0.0.1"
    server_port: "9669"
  
  vector:
    collection_name: "my_collection"
    dim: 1024

models:
  llm:
    type: "ollama"
    model_name: "llama3.1:70b"
    device: "cuda:0"
  
  embedding:
    model_name: "BAAI/bge-large-en-v1.5"
    device: "cuda:0"

rag:
  graph:
    k_hop: 2
    pruning: 30
    data_type: "qa"
  
  vector:
    k_similarity: 5
    data_type: "summary"

data_process:
  openai_api_key: "your-openai-api-key"
  extraction_model: "gpt-3.5-turbo"
```

```python
from graphllm.graphllm_engine import GraphLLMEngine, load_config_from_yaml

# 从配置文件加载
config = load_config_from_yaml("config.yaml")
engine = GraphLLMEngine(config)
```

### 4. 命令行使用

```bash
# 交互模式
python graphllm/main.py --mode interactive

# 使用配置文件
python graphllm/main.py --mode interactive --config config.yaml

# 批处理模式
python graphllm/main.py --mode batch \
  --questions "What is GNN?" "How does RAG work?"

# 文档处理模式
python graphllm/main.py --mode process \
  --input-file documents.txt \
  --output-file results.json
```

## 详细配置

### Engine配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `graph_space_name` | str | "graphllm_space" | 图数据库空间名称 |
| `vector_collection_name` | str | "graphllm_collection" | 向量数据库集合名称 |
| `llm_model` | str | "llama3.1:70b" | LLM模型名称 |
| `embedding_model` | str | "BAAI/bge-large-en-v1.5" | 嵌入模型名称 |
| `llm_type` | str | "ollama" | LLM类型 (ollama/openai) |
| `graph_k_hop` | int | 2 | 图遍历跳数 |
| `vector_k_similarity` | int | 5 | 向量相似度检索数量 |

### 支持的模型

#### LLM模型
- **Ollama**: llama3.1:70b, llama3.1:8b, gemma:7b, command-r:35b
- **OpenAI**: gpt-3.5-turbo, gpt-4, gpt-4-turbo

#### 嵌入模型
- **HuggingFace**: BAAI/bge-large-en-v1.5, BAAI/bge-base-en-v1.5, BAAI/bge-small-en-v1.5
- **OpenAI**: text-embedding-3-small, text-embedding-3-large

## 高级功能

### 1. 自定义检索策略

```python
# 仅使用图检索
result = engine.query(question, use_graph=True, use_vector=False)

# 仅使用向量检索
result = engine.query(question, use_graph=False, use_vector=True)

# 混合检索（默认）
result = engine.query(question, use_graph=True, use_vector=True)
```

### 2. 性能监控

```python
# 获取性能摘要
summary = engine.get_performance_summary()
print(f"平均检索时间: {summary['avg_retrieval_time']:.3f}s")
print(f"平均生成时间: {summary['avg_generation_time']:.3f}s")

# 清空性能指标
engine.clear_metrics()
```

### 3. 批量处理

```python
questions = [
    "What are Graph Neural Networks?",
    "How does knowledge graph work?",
    "Explain RAG architecture"
]

results = []
for question in questions:
    result = engine.query(question)
    results.append(result)
```

## 运行模式详解

### 1. 交互模式 (Interactive)

适合单次查询和调试：

```bash
python graphllm/main.py --mode interactive
```

支持的命令：
- `quit` / `exit` / `q`: 退出
- `stats`: 查看性能统计
- `clear`: 清空性能统计

### 2. 批处理模式 (Batch)

适合批量处理问题：

```bash
python graphllm/main.py --mode batch \
  --questions "Question 1" "Question 2" "Question 3" \
  --output-file results.json
```

### 3. 文档处理模式 (Process)

适合处理大量文档：

```bash
python graphllm/main.py --mode process \
  --input-file documents.txt \
  --output-dir ./processed_data
```

## 配置文件使用

### 1. 创建配置文件

复制 `config_template.yaml` 并根据需要修改：

```bash
cp graphllm/config_template.yaml my_config.yaml
```

### 2. 配置文件结构

```yaml
# 数据库配置
database:
  graph:
    space_name: "my_space"
    server_ip: "127.0.0.1"
    server_port: "9669"
    create: true
    verbose: false
  
  vector:
    collection_name: "my_collection"
    dim: 1024
    host: "localhost"
    port: 19530

# 模型配置
models:
  llm:
    type: "ollama"
    model_name: "llama3.1:70b"
    chunk_size: 512
    chunk_overlap: 20
    device: "cuda:0"
    timeout: 150000
    port: 11434
  
  embedding:
    model_name: "BAAI/bge-large-en-v1.5"
    batch_size: 20
    device: "cuda:0"

# RAG配置
rag:
  graph:
    k_hop: 2
    pruning: 30
    data_type: "qa"
    pruning_mode: "embedding_for_perentity"
  
  vector:
    k_similarity: 5
    data_type: "summary"

# 数据处理配置
data_process:
  openai_api_key: "your-openai-api-key"
  extraction_model: "gpt-3.5-turbo"
```

### 3. 使用配置文件

```python
from graphllm.graphllm_engine import GraphLLMEngine, load_config_from_yaml

# 加载配置
config = load_config_from_yaml("my_config.yaml")

# 初始化Engine
engine = GraphLLMEngine(config)

# 使用Engine
result = engine.query("What is GNN?")
print(result['answer'])
```

## 错误处理

Engine包含完善的错误处理机制：

```python
try:
    engine = GraphLLMEngine(config)
    result = engine.query("What is GNN?")
except Exception as e:
    print(f"处理失败: {e}")
    # 进行错误恢复或重试
```

常见错误及解决方案：

1. **数据库连接失败**
   - 检查NebulaGraph和Milvus服务状态
   - 验证连接参数

2. **模型加载失败**
   - 检查Ollama服务状态
   - 确认模型已下载
   - 验证GPU内存

3. **内存不足**
   - 减少batch_size
   - 使用更小的模型
   - 增加系统内存

## 性能优化

### 1. 硬件配置建议

- **GPU**: NVIDIA RTX 4090 或更高
- **内存**: 32GB+ RAM
- **存储**: SSD 用于向量索引

### 2. 参数调优

```python
# 优化检索性能
config = create_engine_config(
    graph_k_hop=1,  # 减少图遍历深度
    vector_k_similarity=3,  # 减少向量检索数量
    graph_pruning=20,  # 更严格的剪枝
    embed_batch_size=32  # 增加批处理大小
)
```

### 3. 缓存策略

- 启用模型缓存
- 使用向量索引缓存
- 实现查询结果缓存

## 扩展开发

### 1. 添加新的检索器

```python
class CustomRetriever:
    def retrieve(self, query: str):
        # 实现自定义检索逻辑
        pass

# 在Engine中集成
engine.custom_retriever = CustomRetriever()
```

### 2. 添加新的LLM

```python
# 在model_deploy/model_deployment.py中添加
class CustomLLMEnv:
    def __init__(self, model_name: str):
        # 初始化自定义LLM
        pass
    
    def generate_response(self, prompt: str):
        # 实现生成逻辑
        pass
```

### 3. 添加新的评估指标

```python
# 在Engine中添加自定义指标
engine.metrics["custom_metric"] = []

def custom_evaluation(result):
    # 实现自定义评估逻辑
    pass
```

## 故障排除

### 常见问题

1. **Ollama连接失败**
   ```bash
   # 检查Ollama服务
   curl http://localhost:11434/api/tags
   ```

2. **NebulaGraph连接失败**
   ```bash
   # 检查NebulaGraph状态
   docker logs nebula-graphd
   ```

3. **Milvus连接失败**
   ```bash
   # 检查Milvus状态
   docker logs milvus-standalone
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 在Engine中启用调试
config = create_engine_config(verbose=True)
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 讨论区: [Discussions] 