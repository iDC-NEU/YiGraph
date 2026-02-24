---
sidebar_position: 2
---

# 案例2.interact-mode
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
## 第五步：选择运行模式为:interact
使用指令切换：
```
mode interact
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
示例问题：
```
推荐图中与 Hodges Mitchell 最有可能产生交易往来的潜在账户。
```
## 第八步：协同规划
输入问题后，易图会规划出可执行DAG，将计划展示给你，可以给出修改建议完善DAG，命令为：
```
modify 修改建议
```
示例：
```
modify 先找出与 Hodges Mitchell 联系紧密的账户群组,再在该群组中识别交易最活跃账户
```
确认计划无误后，开始分析
```
start
```
等待易图产生分析报告



