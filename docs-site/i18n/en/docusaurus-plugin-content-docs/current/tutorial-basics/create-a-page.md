---
sidebar_position: 1
---

# AAG Installation and Usage Guide

This document introduces the environment preparation, installation steps, configuration methods, and basic usage of the **AAG** project.

------------------------------------------------------------------------

## 1. Environment Preparation

### 1.1 Python Version Requirements

-   Python &gt;= **3.11**

Please confirm that your current Python version meets the requirements:

``` bash
python --version
# or
python3 --version
```

### 1.2 Create Virtual Environment with Conda (Recommended)

``` bash
conda create -n AAG python=3.11
conda activate AAG
```

------------------------------------------------------------------------

## 2. Get Source Code and Install Dependencies

### 2.1 Download Source Code

``` bash
git clone https://github.com/superccy/AAG.git
cd AAG
```

### 2.2 Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 3. Configure System Parameters

### 3.1 Configure Inference and Retrieval Engine

Edit the configuration file:

``` text
config/engine_config.yaml
```

Example configuration:

``` yaml
# Running mode: interactive / batch
mode: interactive

# Reasoner module configuration
reasoner:
  llm:
    provider: "openai"   # Options: ollama / openai
    openai:
      base_url: "https://your-api-endpoint/v1/"
      api_key: "your-api-key"
      model: "gpt-4o-mini"

# Retrieval module configuration
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

### 3.2 Configure Dataset

Edit the configuration file:

``` text
config/data_upload_config.yaml
```

Example configuration:

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

> Please modify the `path` to your local actual data file path.

------------------------------------------------------------------------

## 4. Start AAG

AAG supports the following two running modes:

- **Web Interactive Mode (Recommended)**  
  Interactive analysis through a browser, suitable for daily use, demonstrations, and business analysis scenarios.

- **Terminal Interactive Mode**  
  Interaction through command line, suitable for development debugging, quick verification, and batch testing scenarios.

---

### 4.1 Web Interactive Mode

Execute the following command in the project root directory to start the web service:

```bash
python web/frontend/run.py
```
After successful startup, the terminal will output the accessible service address.
Please open the corresponding address in your browser according to the prompt to enter the AAG web interface.

In the web interface, users can input business questions in natural language, and the system will automatically complete the analysis process and display analysis results and reports.


### 4.2 Terminal Interactive Mode

If you want to interact with AAG directly through the command line, execute in the project root directory:

```bash
python aag/main.py
```

After startup, the system will enter terminal interactive mode.
Users can input questions according to terminal prompts, and AAG will complete the analysis and output results in the command line.

This mode is mainly used for development debugging, algorithm verification, or quick testing scenarios.
------------------------------------------------------------------------

## 5. Using AAG

Regardless of whether you use Web mode or Terminal mode, the basic usage process of AAG is consistent:

- Start the corresponding running mode

- Input natural language business questions according to prompts

- The system automatically completes task understanding, analysis execution, and result generation

For more advanced features, parameter descriptions, and usage examples, please refer to the project's README documentation or operation prompts in the interface.

------------------------------------------------------------------------

## 6. Common Issues and Suggestions

-   **GPU device unavailable**: Please confirm that `embedding.device` is set correctly
-   **Port conflict**: Check if graph database and vector database services are started
-   **Model cannot be loaded**: Confirm that API Key and model name are valid

------------------------------------------------------------------------

For batch mode, more model configurations, or advanced usage, please further consult the official documentation or source code comments.
