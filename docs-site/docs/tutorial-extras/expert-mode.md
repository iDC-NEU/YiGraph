---
sidebar_position: 3
---

# 案例3.expert-mode
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
## 第五步：选择运行模式为:expert
使用指令切换：
```
mode expert
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
## 第七步：输入指令
该模式需要你给出具体的执行步骤和图算法，建议先了解一下易图支持的各类图算法，易图会严格按照你的指令执行并产生分析报告。
示例问题：初步判断 Anna 所在资金强关联社区为高风险群体，但当前只能优先冻结 1 个账户。目标是最大程度打散高风险网络，尽可能切断可疑资金流转。
给易图的指令：
```
首先找到 Anna 所在的强连通分量，然后计算子图内各节点的关键指标（连接程度、中转作用、全局影响力等），最后逐个模拟删除候选账户，评估网络碎裂效果（删除节点后的连通块个数），选择冻结后“网络切断效果最大”的最优账户。
```
等待易图运行产生分析报告

