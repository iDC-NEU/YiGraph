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

### 1.3 Neo4j 安装与配置

AAG 需要使用 Neo4j 作为图数据库。本指南使用 **Neo4j 3.5.25** 版本。

#### 1.3.1 Java 版本要求

Neo4j 3.5.25 需要 Java 8 或 Java 11。请先检查 Java 版本：

``` bash
java -version
```

如果未安装 Java，请先安装对应版本。

#### 1.3.2 下载与解压 Neo4j

1. 从官网下载 Neo4j 3.5.25 安装包（通常是 `.tar.gz` 或 `.zip` 格式）
2. 解压安装包到指定位置：

**Linux/Mac 系统（.tar.gz 格式）：**
``` bash
tar -xzf neo4j-community-3.5.25-unix.tar.gz
cd neo4j-community-3.5.25
```

**Windows 系统（.zip 格式）：**
- 右键点击压缩包，选择"解压到当前文件夹"
- 或使用命令：`unzip neo4j-community-3.5.25-windows.zip`
- 进入解压后的目录

#### 1.3.3 配置 Neo4j

进入 `conf` 目录，编辑 `neo4j.conf` 配置文件：

``` bash
cd conf
```

在 `neo4j.conf` 中添加或修改以下配置：

``` properties
dbms.connectors.default_listen_address=0.0.0.0
dbms.connectors.default_advertised_address=localhost
dbms.connector.bolt.listen_address=0.0.0.0:7687
dbms.connector.http.listen_address=0.0.0.0:7474
dbms.connector.https.enabled=true
```

#### 1.3.4 启动与停止 Neo4j

进入 `bin` 目录，执行启动或停止命令：

**启动 Neo4j：**
``` bash
cd bin
./neo4j start
```

**停止 Neo4j：**
``` bash
./neo4j stop
```

启动 Neo4j 后，可以通过浏览器访问 `http://localhost:7474` 来验证安装是否成功。

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

> **重要提示：** 在启动 AAG 之前，请确保 Neo4j 数据库已经启动并正常运行。如果 Neo4j 未启动，AAG 将无法连接到图数据库。请参考 [1.3.4 启动与停止 Neo4j](#134-启动与停止-neo4j) 部分启动 Neo4j。

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

#### Web 界面使用说明

![AAG Web 界面](../../static/img/chat.png)

使用 AAG Web 界面进行分析的基本步骤如下：

1. **开始对话**：开启一个新对话或者从历史记录中选择一个现有对话。

2. **选择模式**：选择最适合您的模式。

3. **选择数据集**：会将您上传好的数据集列举出来。例如：DocumentDemo。

4. **输入您的请求**：在输入框中输入您的指令或问题。请尽可能清晰和具体。

5. **提交**：点击发送按钮。

6. **监控进度**：在主聊天区观察状态更新（运行中、规划中、分析中等）。

7. **查看结果**：处理完成后，结果将显示在主聊天区。然后您可以提出后续问题或开始新的请求。

#### 数据集管理

![数据集管理界面](../../static/img/dataset.png)

在 Web 界面中，您可以方便地管理数据集：

1. **创建数据集**：点击"创建"按钮。

2. **填写数据集信息**：
   - 输入数据集的名称
   - 选择数据集中文件的类型

3. **上传数据文件**：根据选择的文件类型上传相应的数据文件。

4. **保存数据集**：完成配置后保存，数据集将在对话中可供选择使用。

#### 文件管理

![文件管理界面](../../static/img/file.png)

在文件管理界面中，您可以对数据集中的文件进行管理和可视化：

1. **选择数据集**：从下拉列表中选择对应的数据集。

2. **上传文件**：向选定的数据集中上传文件。

3. **查看解析进度**：系统会显示文件的解析进度，实时反馈处理状态。

4. **可视化知识图谱**：文件解析完成后，点击"可视化"按钮，即可查看该数据集对应的知识图谱可视化展示。


### 4.2 终端交互模式（Terminal）

如果希望直接通过命令行与 AAG 进行交互，可在项目根目录下执行：

```bash
python aag/main.py
```
启动后，系统将进入终端交互模式。
用户可按照终端提示输入问题，AAG 将在命令行中完成分析并输出结果。

![终端交互模式](../../static/img/zhongduan.png)

#### 终端交互使用说明

使用终端交互模式的基本步骤：

1. **查看可用数据集**：通过指令查看系统中有哪些可用的数据集。

2. **选择数据集**：根据提示选择要使用的数据集。

3. **输入问题**：直接在终端中输入您的业务问题或分析需求。

4. **查看结果**：系统会在终端中实时显示分析过程和最终结果。

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
