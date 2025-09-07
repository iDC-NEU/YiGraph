from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime
from graphllm.database.datatype import *

class GraphFeatureEncoder:
    """
    将 List[VertexData] 转换为 PyG 可用的节点特征矩阵 (torch.FloatTensor)
    支持三种模式：
    - attributes: 仅使用原始属性特征
    - random: 仅使用随机ID embedding
    - concat: 属性特征 + 随机embedding 拼接
    """
    def __init__(self, id_emb_dim: int = 32):
        self.id_emb_dim = id_emb_dim
        self.ohe_encoders: Dict[str, OneHotEncoder] = {}
        self.scaler = StandardScaler()
        self.id_embedding = None
    
    def set_id_emb_dim(self, id_emb_dim: int):
        self.id_emb_dim = id_emb_dim
        
    def _detect_type(self, value: Any) -> str:
        """根据单个属性值判断类型"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "nan"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, (int, float)):
            return "numeric"
        if isinstance(value, str):
            # 尝试识别日期
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
                try:
                    datetime.strptime(value, fmt)
                    return "date"
                except ValueError:
                    continue
            return "category"
        return "category"  # 默认类别型

    def _encode_attributes(self, vertices: List[VertexData]) -> np.ndarray:
        """将节点属性编码为numpy特征矩阵"""
        df = pd.DataFrame([v.properties for v in vertices])
        df.insert(0, "vid", [v.vid for v in vertices])

        # 检测列类型
        col_types = {}
        for col in df.columns:
            if col == "vid":
                continue
            sample_val = next((v for v in df[col].values if v is not None), None)
            col_types[col] = self._detect_type(sample_val)

        num_cols  = [c for c,t in col_types.items() if t=="numeric"]
        bool_cols = [c for c,t in col_types.items() if t=="bool"]
        cat_cols  = [c for c,t in col_types.items() if t=="category"]
        date_cols = [c for c,t in col_types.items() if t=="date"]

        # 数值型
        num_feat = df[num_cols].fillna(0).values if num_cols else np.empty((len(df),0))
        if num_feat.shape[1] > 0:
            num_feat = self.scaler.fit_transform(num_feat)

        # 布尔型
        bool_feat = df[bool_cols].astype(int).values if bool_cols else np.empty((len(df),0))

        # 类别型 → One-hot
        cat_feat_list = []
        for col in cat_cols:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            arr = enc.fit_transform(df[[col]].fillna("Unknown"))
            self.ohe_encoders[col] = enc
            cat_feat_list.append(arr)
        cat_feat = np.hstack(cat_feat_list) if cat_feat_list else np.empty((len(df),0))

        # 日期型 → 天数差
        today = pd.Timestamp.now()
        date_feat_list = []
        for col in date_cols:
            arr = pd.to_datetime(df[col], errors="coerce")
            days = (today - arr).dt.days.fillna(0).values.reshape(-1,1)
            date_feat_list.append(days)
        date_feat = np.hstack(date_feat_list) if date_feat_list else np.empty((len(df),0))

        # 拼接
        return np.hstack([x for x in [num_feat, bool_feat, cat_feat, date_feat] if x.shape[1]>0])

    def generate_random_embedding(self, num_nodes: int) -> torch.Tensor:
        """生成随机节点嵌入"""
        self.id_embedding = nn.Embedding(num_nodes, self.id_emb_dim)
        node_ids = torch.arange(num_nodes)
        return self.id_embedding(node_ids).detach()

    def fit_transform(self, vertices: List[VertexData], mode: str = "attributes") -> torch.Tensor:
        """
        将节点列表转为PyG可用特征矩阵
        mode:
          - "attributes": 仅属性特征
          - "random": 仅随机embedding
          - "concat": 属性特征 + 随机embedding
        """
        num_nodes = len(vertices)
        if num_nodes == 0:
            raise ValueError("输入的顶点列表为空")

        features_attr = self._encode_attributes(vertices) if mode in ["attributes","concat"] else np.empty((num_nodes,0))
        features_rand = self.generate_random_embedding(num_nodes).numpy() if mode in ["random","concat"] else np.empty((num_nodes,0))

        # 最终特征拼接
        features = np.hstack([x for x in [features_attr, features_rand] if x.shape[1]>0])
        return torch.tensor(features, dtype=torch.float32)