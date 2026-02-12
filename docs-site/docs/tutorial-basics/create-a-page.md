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

AAG 支持以下两种运行模式：

- **Web 交互模式（推荐）**  
  通过浏览器进行交互式分析，适合日常使用、演示与业务分析场景。

- **终端交互模式（Terminal）**  
  通过命令行进行交互，适合开发调试、快速验证与批量测试场景。

---

### 4.1 Web 交互模式

在项目根目录下执行以下命令启动 Web 服务：

```bash
python web/frontend/run.py
```
启动成功后，终端会输出可访问的服务地址。
请根据提示在浏览器中打开对应地址，即可进入 AAG 的 Web 界面。

在 Web 界面中，用户可以通过自然语言输入业务问题，系统将自动完成分析流程，并展示分析结果与报告。


### 4.2 终端交互模式（Terminal）

如果希望直接通过命令行与 AAG 进行交互，可在项目根目录下执行：

```bash
python aag/main.py
```

启动后，系统将进入终端交互模式。
用户可按照终端提示输入问题，AAG 将在命令行中完成分析并输出结果。

该模式主要用于开发调试、算法验证或快速测试场景。
------------------------------------------------------------------------

## 5. 使用 AAG

无论采用 Web 模式还是终端模式，AAG 的基本使用流程一致：

- 启动对应的运行模式

- 根据提示输入自然语言业务问题

- 系统自动完成任务理解、分析执行与结果生成

更多高级功能、参数说明与使用示例，请参考项目的 README 文档或界面中的操作提示。

------------------------------------------------------------------------

## 6. 常见问题建议

-   **GPU 设备不可用**：请确认 `embedding.device` 设置正确
-   **端口冲突**：检查图数据库与向量数据库服务是否已启动
-   **模型无法加载**：确认 API Key 与模型名称是否有效

------------------------------------------------------------------------

如需批量模式、更多模型配置或高级用法，请进一步查阅官方文档或源码注释。
