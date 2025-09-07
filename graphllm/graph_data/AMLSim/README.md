# AMLSim数据加载器

这个工具用于将AMLSim交易图数据加载到NebulaGraph数据库中。

## 功能特性

- 自动连接到NebulaGraph数据库
- 创建图空间和模式（Tag和Edge Type）
- 批量加载账户数据（顶点）
- 批量加载交易数据（边）
- 数据验证和错误处理
- 详细的日志记录

## 数据文件

- **账户数据**: `/home/chency/GraphLLM/graphllm/graph_data/AMLSim/1K/accounts.csv`
- **交易数据**: `/home/chency/GraphLLM/graphllm/graph_data/AMLSim/1K/transactions.csv`

## 图模式

### 顶点 (Account Tag)
- 顶点ID: `acct_id`
- 属性: 包含账户的所有字段（acct_id, dsply_nm, type, acct_stat等）

### 边 (Transaction Edge Type)
- 源顶点: `orig_acct`
- 目标顶点: `bene_acct`
- 属性: 包含交易的所有字段（tran_id, tx_type, base_amt, tran_timestamp等）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 确保NebulaGraph服务运行

确保NebulaGraph服务在localhost:9669上运行，默认用户名和密码为root/nebula。

### 2. 运行数据加载器

```bash
python load_data_into_nebulagraph.py
```

### 3. 自定义连接参数

如果需要修改连接参数，可以在代码中修改AMLSimDataLoader的初始化参数：

```python
loader = AMLSimDataLoader(
    host='your-nebulagraph-host',
    port=9669,
    username='your-username',
    password='your-password'
)
```

## 输出

程序会输出详细的日志信息，包括：
- 连接状态
- 空间创建状态
- 数据加载进度
- 最终的数据统计信息

## 错误处理

程序包含完善的错误处理机制：
- 连接失败时会重试
- 数据插入失败时会记录警告但继续处理
- 所有错误都会记录到日志中

## 注意事项

1. 确保NebulaGraph服务正在运行
2. 确保有足够的权限创建空间和模式
3. 数据文件路径必须正确
4. 建议在加载大量数据前先测试小数据集 