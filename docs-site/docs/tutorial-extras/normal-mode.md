---
sidebar_position: 1
---

# 案例1.normal-mode
## 第一步：安装环境
```
conda create -n AAG python=3.11
conda activate AAG
git clone https://github.com/superccy/AAG.git
cd AAG
pip install -r requirements.txt
```
## 第二步：配置你的api key
配置文件：
```
config/engine_config.yaml
```
配置内容：
```
reasoner:
  llm:
    provider: "openai"   # 可选：ollama / openai
    openai:
      base_url: "your-url"
      api_key: "your-api-key"
      model: "your-model"
```
## 第三步：配置数据文件
文件位于：
```
config/data_upload_config.yaml
```
这里以金融交易数据集AMLSim1K为例
```
datasets:
  - name: AMLSim1K
    type: graph
    schema:
      vertex:
        - type: account
          path: "/path/to/accounts.csv" #修改为本地数据路径
          format: csv
          id_field: acct_id
      edge:
        - type: transfer
          path: "/path/to/transactions.csv" #修改为本地数据路径
          format: csv
          source_field: orig_acct
          target_field: bene_acct
```
## 第四步：运行
在项目根目录下运行：
```
python aag/main.py
```
## 第五步：选择运行模式为:normal
初始运行时默认即为normal模式，也可使用指令显式切换：
```
mode normal
```
## 第六步：选择数据集
显示当前已配置完成数据集：
```
datasets
```
选择数据集（以AMLSim1K为例）：
```
use AMLSim1K
```
等待数据加载完成，初次加载可能需要较长时间
## 第七步：输入问题
这里以金融领域常见的洗钱行为检测为例：
```
最近发现用户 Anna 与一些“骡子账户”进行了频繁交易，需要围绕 Anna 检查是否存在洗钱行为：列出潜在洗钱路径、估算可能非法转出金额，并找出可疑路径中交易金额最大的账户。
```
等待易图运行产生分析报告

