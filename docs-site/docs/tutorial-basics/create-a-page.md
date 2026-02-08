---
sidebar_position: 1
---

# AAG 安装与使用指南

本文档介绍了 **AAG**
项目的环境准备、安装步骤、配置方法以及基本使用方式。

------------------------------------------------------------------------

## 1. 准备环境

### 1.1 Python 版本要求

-   Python \>= **3.11**

请确认当前 Python 版本满足要求：

``` bash
python --version
# 或
python3 --version
```

### 1.2 使用 Conda 创建虚拟环境（推荐）

``` bash
conda create -n AAG python=3.11
conda activate AAG
```

------------------------------------------------------------------------

## 2. 获取源码并安装依赖

### 2.1 下载源码

``` bash
git clone https://github.com/superccy/AAG.git
cd AAG
```

### 2.2 安装依赖

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 3. 配置系统参数

### 3.1 配置推理与检索引擎

编辑配置文件：

``` text
config/engine_config.yaml
```

示例配置如下：

``` yaml
# 运行模式： interactive / batch
mode: interactive

# 推理模块配置
reasoner:
  llm:
    provider: "openai"   # 可选：ollama / openai
    openai:
      base_url: "https://your-api-endpoint/v1/"
      api_key: "your-api-key"
      model: "gpt-4o-mini"

# 检索模块配置
retrieval:
  database:
    graph:
      space_name: "AMLSim1K"
      server_ip: "127.0.0.1"
      server_port: "9669"
    vector:
      collection_name: "graphllm_collection"
      host: "localhost"
      port: 19530
  embedding:
    model_name: "BAAI/bge-large-en-v1.5"
    device: "cuda:2"
  rag:
    graph:
      k_hop: 2
    vector:
      k_similarity: 5
```

------------------------------------------------------------------------

### 3.2 配置数据集

编辑配置文件：

``` text
config/data_upload_config.yaml
```

示例配置如下：

``` yaml
datasets:
  - name: AMLSim1K
    type: graph
    schema:
      vertex:
        - type: account
          path: "/path/to/accounts.csv"
          format: csv
          id_field: acct_id
      edge:
        - type: transfer
          path: "/path/to/transactions.csv"
          format: csv
          source_field: orig_acct
          target_field: bene_acct
```

> 请将 `path` 修改为你本地真实的数据文件路径。

------------------------------------------------------------------------

## 4. 启动 AAG

在项目根目录下执行：

``` bash
# 启动交互模式
python aag/main.py
```

------------------------------------------------------------------------

## 5. 使用 AAG

-   启动成功后，根据终端提示在浏览器中访问对应地址
-   进入 Web 界面后，可进行交互式查询与分析
-   具体功能与操作方式请参考项目 README 或界面提示

------------------------------------------------------------------------

## 6. 常见问题建议

-   **GPU 设备不可用**：请确认 `embedding.device` 设置正确
-   **端口冲突**：检查图数据库与向量数据库服务是否已启动
-   **模型无法加载**：确认 API Key 与模型名称是否有效

------------------------------------------------------------------------

如需批量模式、更多模型配置或高级用法，请进一步查阅官方文档或源码注释。
