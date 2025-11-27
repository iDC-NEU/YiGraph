from datetime import datetime

MODEL_MAPPING = {
    "GPT 4": "qwen3-max",
    "Qwen 14B": "qwen3-max",
    "Qwen Plus": "qwen3-max"
}

# 知识库数据（内存模拟）
knowledge_bases = [
    {
        "id": 1,
        "名称": "NBA Match Records",
        "文件类型": "text",
        "文档个数": 5,
        "创建时间": datetime(2024, 9, 12, 10, 30).strftime("%Y-%m-%d %H:%M:%S"),
    },
    {
        "id": 2,
        "名称": "Social Network Graph",
        "文件类型": "graph",
        "文档个数": 3,
        "创建时间": datetime(2023, 6, 5, 14, 15).strftime("%Y-%m-%d %H:%M:%S"),
    },
    {
        "id": 3,
        "名称": "3",
        "文件类型": "text",
        "文档个数": 6,
        "创建时间": datetime(2022, 11, 20, 9, 0).strftime("%Y-%m-%d %H:%M:%S"),
    },
]

# 测试数据集
TEST_DATA = {
    "dag": [
        {
            "type": "thinking",
            "contentType": "text",
            "content": "Well, the user has raised a complex question. Now I will start identifying the sub-questions within it and determining the dependencies between them. I need to construct a thinking process in the form of a Directed Acyclic Graph and present it to the user..."
        },
        {
            "type": "result",
            "contentType": "dag",
            "content":  {
                            "nodes": [
                                {"id": "1", "label": "Check if user Anna is a high-risk user", "tasktype":"Graph Algorithm (Pagerank)"},
                                {"id": "2", "label": "List all potential money laundering pathways centered on Anna", "tasktype":"Graph Algorithm (DFS)"},
                                {"id": "3", "label": "Estimate the amount of cash that may have been illegally transferred out in relation to Anna", "tasktype":"Numerical Analysis (Python Code)"},
                                {"id": "4", "label": "Among the accounts involved in these suspicious pathways, identify the account with the largest transaction amount", "tasktype":"Numerical Analysis (Python Code)"}
                            ],
                            "edges": [
                                {"from": "1", "to": "2"},
                                {"from": "2", "to": "3"},
                                {"from": "2", "to": "4"},
                            ]
                        }
        }
    ],
    "dag_modification": [
        {
            "type": "thinking",
            "contentType": "text",
            "content": "I am currently adjusting the DAG structure according to your modification suggestions. You proposed modifying the DAG, and I will redesign the nodes and connection relationships based on your recommendations to better meet the requirements."
        },
        {
            "type": "result",
            "contentType": "dag",
            "content": {
                "nodes": [
                    {"id": "1", "label": "Check if user Anna is a high-risk user", "tasktype":"Graph Algorithm (Pagerank)"},
                    {"id": "2", "label": "Identify the potential fraud community in which Anna resides to narrow the scope of subsequent risk monitoring", "tasktype":"Graph Algorithm (Louvain)"},
                    {"id": "3", "label": "Anna List all potential money laundering pathways within the high-risk community where Anna is located", "tasktype":"Graph Algorithm (DFS)"},
                    {"id": "4", "label": "Estimate the amount of cash that may have been illegally transferred out in relation to Anna", "tasktype":"Graph Algorithm (Python Code)"},
                    {"id": "5", "label": "Identify the account with the largest transaction amount within Anna’s fraud community", "tasktype":"Numerical Analysis (Python Code)"},
                ],
                "edges": [
                    {"from": "1", "to": "2"},
                    {"from": "2", "to": "3"},
                    {"from": "2", "to": "5"},
                    {"from": "3", "to": "4"}
                ]
            }
        }
    ],
    "dag_confirmation": [
        {
            "type": "thinking",
            "contentType": "text",
            "content": "The user has confirmed that the mind map requires no revisions. Now I need to gradually execute the corresponding algorithms in accordance with the established steps of the Directed Acyclic Graph (DAG) to obtain the results and generate an analysis report. First of all..."
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "# Analysis Report"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "## 1. Anna Node Centrality Assessment (PageRank Results)"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "In the first step, the PageRank graph algorithm was applied to quantify the “importance” and “risk connectivity” of all account nodes within the entire transaction network.\nThe core implication of PageRank is: if an account frequently transacts with multiple high-risk accounts, or if it occupies a structurally prominent position within the transaction graph, its score will significantly increase—reflecting its potential influence and exposure risk within the overall network."
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "The results show that **Anna’s PageRank score ranks within the Top 2% of all customers**, significantly higher than typical retail clients.\nThis indicates that her account is heavily “relied upon” by other nodes, suggesting strong potential for risk propagation."
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "Based on the structural risk signals reflected by PageRank, **Anna is preliminarily assessed as a high-risk customer or a key monitoring target.**"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "## 2. Suspicious Path Identification (DFS Results)" 
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "In the second step, the DFS (Depth-First Search) graph algorithm was used to perform full path penetration analysis on fund flows associated with Anna.\nAfter executing DFS, the analysis identified **one structurally complete and logically closed suspicious path:**\n**Anna → Sean Hodges → Michelle Steele → Deborah Macdonald → Kyle Daniels → Anna**\nThis path exhibits a typical *closed-loop structure*, where funds originate from Anna, pass through multiple intermediary accounts, and ultimately return to Anna’s own account—forming a complete “outflow → transfer → return” chain.\nSuch configurations are classified as **high-risk patterns in AML analytics**, often associated with attempts to obscure transactional origins or disguise the true destination of funds."
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "## 3. Amount Assessment (Python Code Statistical Results)"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "According to the third step of the DAG, automated Python-based data processing was utilized to extract and aggregate all transaction amounts involved in the closed-loop path.\nStatistical results indicate:\n- **The total transaction amount within the path is 157,800 USD**\nThis value is relatively high for an ordinary retail customer, and given the closed-loop characteristics, the true source and purpose of these funds must be further verified."
        },
        {
            "type": "result",
            "contentType": "text",  
            "content": "## 4. Largest Transaction Accounts Identification (Python Code Statistical Results)"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "To identify potential key intermediary accounts within the path, automatically generated Python processing code was used to compute and rank the historical transaction volumes associated with all nodes in the chain.\nThe results show:\n- **Highest transaction volume account:** Deborah Macdonald (historical total 100,500 USD)\n- **Second highest:** Michelle Steele (historical total 100,100 USD)\nAccounts with such high transaction volumes often exhibit strong fund aggregation capabilities and may serve as “channel-type” or “hub-type” nodes within the network. These accounts should be prioritized for further investigation to determine whether their activity aligns with their business profile and whether they may be receiving or transferring funds on behalf of others."
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "## Summary"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "Combining PageRank centrality, DFS-identified suspicious paths, and transaction amount analysis, this assessment reveals multiple risk indicators in Anna’s transaction network:\n- **High structural risk**: presence of a complete closed-loop transaction path\n- **Abnormal fund return behavior**: funds ultimately flow back to Anna’s account\n- **Large transaction size**: loop transaction total reaches 157,800 USD\n- **Active intermediary nodes** : some accounts exhibit abnormally high historical transaction volumes\nTaken together, these factors suggest that Anna may be situated within a high-risk financial network environment."
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "## Recommendations"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "1. **Initiate Enhanced Due Diligence (EDD)**\n- Verify Anna’s income sources, economic activities, and transaction purposes.\n- Request relevant contracts, invoices, or fund-usage documentation."
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "2. **Strengthen monitoring of large-scale transactions**\n- Enhance monitoring of suspected key intermediary nodes such as Deborah Macdonald and Michelle Steele.\n- Apply stricter trigger rules for large-value transactions and frequent cyclic transfers."
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "These measures will help more effectively identify potential financial risks and strengthen overall monitoring. This analysis is based solely on transaction structure and algorithmic outputs and does not constitute a legal determination of customer behavior. **Final judgment** should be made jointly through KYC review, compliance analysis, and business investigations."
        }
    ]
}
