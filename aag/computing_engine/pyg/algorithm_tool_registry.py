"""
PyG algorithm-level tool registry.

每个工具 = backbone× task
工具名格式：run_{backbone}_{task}，例如 run_gcn_node_classification

"""

import logging
from typing import Callable, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from aag.computing_engine.networkx_server.dynamic_tool_registry import GenericToolOutput
from aag.computing_engine.pyg.dynamic_tool_registry import GraphDataConverter

logger = logging.getLogger(__name__)

# ============================================================================
# 配置表：backbone 和 task
# ============================================================================

BACKBONE_CONFIGS = {
    "gcn": {
        "conv_class": GCNConv,
        "conv_kwargs": {},
        "description": "GCN (Graph Convolutional Network) - 适合同质图，计算高效",
    },
    "gat": {
        "conv_class": GATConv,
        # concat=False：多头输出取均值而非拼接，保持输出维度统一
        "conv_kwargs": {"heads": 4, "concat": False},
        "description": "GAT (Graph Attention Network) - 通过注意力机制自适应聚合邻居",
    },
    "graphsage": {
        "conv_class": SAGEConv,
        "conv_kwargs": {},
        "description": "GraphSAGE - 邻居采样聚合，支持大图归纳学习",
    },
}

TASK_NAMES = ["node_classification", "link_prediction"]

TASK_DESCRIPTIONS = {
    "node_classification": (
        "使用 {backbone} backbone 进行节点分类。"
        "需要提供 node_labels 参数（节点 ID 到类别整数的映射，例如 {{\"node_1\": 0, \"node_2\": 1}}）。"
        "自动划分训练/测试集，返回测试准确率和每个已标注节点的预测类别。"
    ),
    "link_prediction": (
        "使用 {backbone} backbone 进行链接预测（预测图中未来可能出现的边）。"
        "无需外部标签，自动从现有边划分训练/测试集并采样负例。"
        "返回 AUC 评分和 Top-10 最可能出现的新边。"
    ),
}

# ============================================================================
# 多层 GNN backbone
# ============================================================================

class MultiLayerGNN(nn.Module):
    """
    通用多层 GNN backbone。

    层结构：
      第 1 层：conv(in_channels → hidden_channels) + ReLU + Dropout
      第 2..N-1 层：conv(hidden_channels → hidden_channels) + ReLU + Dropout
      第 N 层：conv(hidden_channels → hidden_channels)  # 最后一层不加激活，由 task head 决定
    """

    def __init__(
        self,
        conv_class,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        **conv_kwargs,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers 至少为 1"

        self.convs = nn.ModuleList()
        self.convs.append(conv_class(in_channels, hidden_channels, **conv_kwargs))
        for _ in range(num_layers - 1):
            self.convs.append(conv_class(hidden_channels, hidden_channels, **conv_kwargs))

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


# ============================================================================
# 工具执行函数
# ============================================================================

def _execute_node_classification(
    backbone_name: str,
    conv_class,
    conv_kwargs: dict,
    processor_getter: Callable,
    kwargs: dict,
) -> str:
    """节点分类执行逻辑：多层 GNN + Linear head + CrossEntropy loss"""
    algo_name = f"{backbone_name}_node_classification"
    try:
        processor = processor_getter()
        if processor.graph is None:
            return GenericToolOutput(
                algorithm=algo_name, success=False,
                error="Graph not initialized",
                summary="请先调用 run_initialize_graph",
            ).model_dump_json()

        # 参数
        num_layers      = kwargs["num_layers"]
        hidden_channels = kwargs["hidden_channels"]
        epochs          = kwargs["epochs"]
        lr              = kwargs["lr"]
        train_ratio     = kwargs["train_ratio"]
        node_features       = kwargs["node_features"]
        node_labels         = kwargs["node_labels"]
        node_labels_field   = kwargs.get("node_labels_field", "")

        # 从图节点属性自动读取标签
        if not node_labels and node_labels_field:
            raw = {}
            for nid, attrs in processor.graph.nodes(data=True):
                props = attrs.get("properties", attrs)
                if node_labels_field in props:
                    raw[str(nid)] = props[node_labels_field]
            if raw:
                # 尝试转 int；失败则 label encode（字符串类别）
                try:
                    node_labels = {k: int(float(v)) for k, v in raw.items()}
                except (ValueError, TypeError):
                    categories = sorted(set(raw.values()))
                    cat_map = {c: i for i, c in enumerate(categories)}
                    node_labels = {k: cat_map[v] for k, v in raw.items()}

        if not node_labels:
            return GenericToolOutput(
                algorithm=algo_name, success=False,
                error="node_labels is required",
                summary='节点分类需要 node_labels 参数，格式: {"node_id": class_int, ...}',
            ).model_dump_json()

        # 图数据转换
        feat_tensor = torch.tensor(node_features, dtype=torch.float) if node_features else None
        pyg_data = GraphDataConverter.networkx_to_pyg(processor.graph, feat_tensor)

        # 节点 ID → 索引映射
        node_ids = sorted(processor.graph.nodes())
        id_to_idx = {str(nid): idx for idx, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)

        # 构建标签 tensor（-1 表示无标签节点）
        y = torch.full((num_nodes,), -1, dtype=torch.long)
        for nid, label in node_labels.items():
            if str(nid) in id_to_idx:
                y[id_to_idx[str(nid)]] = int(label)

        labeled_indices = (y >= 0).nonzero(as_tuple=True)[0]
        if len(labeled_indices) < 2:
            return GenericToolOutput(
                algorithm=algo_name, success=False,
                error="Not enough labeled nodes (need >= 2)",
                summary="标注节点数量不足，至少需要 2 个",
            ).model_dump_json()

        # 训练 / 测试 mask
        perm = torch.randperm(len(labeled_indices))
        train_size = max(1, int(len(labeled_indices) * train_ratio))
        train_idx = labeled_indices[perm[:train_size]]
        test_idx  = labeled_indices[perm[train_size:]]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx]   = True

        # 构建模型
        num_classes = len(set(node_labels.values()))
        in_channels = pyg_data.x.size(1)

        backbone  = MultiLayerGNN(conv_class, in_channels, hidden_channels, num_layers, **conv_kwargs)
        head      = nn.Linear(hidden_channels, num_classes)
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(head.parameters()), lr=lr
        )

        # 训练循环
        backbone.train()
        head.train()
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            emb    = backbone(pyg_data.x, pyg_data.edge_index)
            logits = head(emb)
            loss   = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            loss_history.append(round(loss.item(), 4))

            if (epoch + 1) % 50 == 0:
                logger.info(f"[{algo_name}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # 评估
        backbone.eval()
        head.eval()
        with torch.no_grad():
            emb    = backbone(pyg_data.x, pyg_data.edge_index)
            logits = head(emb)
            pred   = logits.argmax(dim=1)

        test_acc = 0.0
        if test_mask.sum() > 0:
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        # 汇总结果：只返回有标签节点的预测
        labeled_mask = y >= 0
        predictions = {
            str(node_ids[i]): {"predicted": pred[i].item(), "true": y[i].item()}
            for i in range(num_nodes)
            if labeled_mask[i]
        }

        result = {
            "backbone":     backbone_name,
            "task":         "node_classification",
            "num_layers":   num_layers,
            "num_classes":  num_classes,
            "train_nodes":  int(train_mask.sum()),
            "test_nodes":   int(test_mask.sum()),
            "test_accuracy": round(test_acc, 4),
            "final_loss":   loss_history[-1] if loss_history else 0.0,
            "predictions":  predictions,
        }

        return GenericToolOutput(
            algorithm=algo_name, success=True, result=result,
            summary=f"{backbone_name.upper()} 节点分类完成，测试准确率: {test_acc:.2%}",
        ).model_dump_json()

    except Exception as e:
        logger.error(f"❌ {algo_name} error: {e}", exc_info=True)
        return GenericToolOutput(
            algorithm=algo_name, success=False,
            error=str(e), summary=f"{algo_name} 执行失败",
        ).model_dump_json()


def _execute_link_prediction(
    backbone_name: str,
    conv_class,
    conv_kwargs: dict,
    processor_getter: Callable,
    kwargs: dict,
) -> str:
    """链接预测执行逻辑：多层 GNN + 点积解码器 + BCE loss"""
    algo_name = f"{backbone_name}_link_prediction"
    try:
        processor = processor_getter()
        if processor.graph is None:
            return GenericToolOutput(
                algorithm=algo_name, success=False,
                error="Graph not initialized",
                summary="请先调用 run_initialize_graph",
            ).model_dump_json()

        # 参数
        num_layers      = kwargs["num_layers"]
        hidden_channels = kwargs["hidden_channels"]
        epochs          = kwargs["epochs"]
        lr              = kwargs["lr"]
        train_ratio     = kwargs["train_ratio"]
        node_features   = kwargs["node_features"]

        # 图数据转换
        feat_tensor = torch.tensor(node_features, dtype=torch.float) if node_features else None
        pyg_data = GraphDataConverter.networkx_to_pyg(processor.graph, feat_tensor)

        edge_index = pyg_data.edge_index
        num_edges  = edge_index.size(1)
        num_nodes  = pyg_data.num_nodes

        if num_edges < 4:
            return GenericToolOutput(
                algorithm=algo_name, success=False,
                error="Not enough edges (need >= 4)",
                summary="边数量不足，至少需要 4 条边",
            ).model_dump_json()

        # 划分训练 / 测试正样本边
        perm       = torch.randperm(num_edges)
        train_size = max(2, int(num_edges * train_ratio))

        train_edge_index    = edge_index[:, perm[:train_size]]
        test_pos_edge_index = edge_index[:, perm[train_size:]]

        # 已有边集合（用于负采样排除）
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        def _sample_neg_edges(n: int) -> torch.Tensor:
            """随机采样 n 条不存在的边"""
            neg = []
            attempts = 0
            while len(neg) < n and attempts < n * 20:
                src = torch.randint(0, num_nodes, (1,)).item()
                dst = torch.randint(0, num_nodes, (1,)).item()
                if src != dst and (src, dst) not in existing_edges:
                    neg.append([src, dst])
                attempts += 1
            if not neg:
                return torch.zeros((2, 0), dtype=torch.long)
            return torch.tensor(neg, dtype=torch.long).t()

        def _decode(z: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
            """点积解码器：节点对 embedding 内积"""
            return (z[ei[0]] * z[ei[1]]).sum(dim=1)

        # 构建模型
        in_channels = pyg_data.x.size(1)
        backbone    = MultiLayerGNN(conv_class, in_channels, hidden_channels, num_layers, **conv_kwargs)
        optimizer   = torch.optim.Adam(backbone.parameters(), lr=lr)

        # 训练循环
        backbone.train()
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            z        = backbone(pyg_data.x, train_edge_index)
            pos_pred = _decode(z, train_edge_index)
            neg_ei   = _sample_neg_edges(train_edge_index.size(1))
            neg_pred = _decode(z, neg_ei) if neg_ei.size(1) > 0 else torch.tensor([])

            preds  = torch.cat([pos_pred, neg_pred])
            labels = torch.cat([
                torch.ones(pos_pred.size(0)),
                torch.zeros(neg_pred.size(0)),
            ])
            loss = F.binary_cross_entropy_with_logits(preds, labels)
            loss.backward()
            optimizer.step()
            loss_history.append(round(loss.item(), 4))

            if (epoch + 1) % 50 == 0:
                logger.info(f"[{algo_name}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # 评估
        backbone.eval()
        with torch.no_grad():
            z = backbone(pyg_data.x, train_edge_index)

            pos_pred = _decode(z, test_pos_edge_index).sigmoid()
            neg_ei   = _sample_neg_edges(test_pos_edge_index.size(1))
            neg_pred = _decode(z, neg_ei).sigmoid() if neg_ei.size(1) > 0 else torch.tensor([])

        # AUC
        auc = 0.5
        if neg_pred.size(0) > 0:
            try:
                from sklearn.metrics import roc_auc_score
                all_preds  = torch.cat([pos_pred, neg_pred]).numpy()
                all_labels = torch.cat([
                    torch.ones(pos_pred.size(0)),
                    torch.zeros(neg_pred.size(0)),
                ]).numpy()
                auc = float(roc_auc_score(all_labels, all_preds))
            except Exception:
                pass

        # Top-10 预测新边（从负样本中取概率最高的）
        node_ids = sorted(processor.graph.nodes())
        top_links = []
        if neg_ei.size(1) > 0:
            probs = neg_pred.tolist()
            for i, prob in enumerate(probs):
                top_links.append({
                    "src": str(node_ids[neg_ei[0, i].item()]),
                    "dst": str(node_ids[neg_ei[1, i].item()]),
                    "probability": round(prob, 4),
                })
            top_links.sort(key=lambda x: x["probability"], reverse=True)
            top_links = top_links[:10]

        result = {
            "backbone":           backbone_name,
            "task":               "link_prediction",
            "num_layers":         num_layers,
            "train_edges":        int(train_size),
            "test_edges":         int(test_pos_edge_index.size(1)),
            "auc":                round(auc, 4),
            "final_loss":         loss_history[-1] if loss_history else 0.0,
            "top_predicted_links": top_links,
        }

        return GenericToolOutput(
            algorithm=algo_name, success=True, result=result,
            summary=f"{backbone_name.upper()} 链接预测完成，AUC: {auc:.4f}",
        ).model_dump_json()

    except Exception as e:
        logger.error(f"❌ {algo_name} error: {e}", exc_info=True)
        return GenericToolOutput(
            algorithm=algo_name, success=False,
            error=str(e), summary=f"{algo_name} 执行失败",
        ).model_dump_json()


# ============================================================================
# 工具工厂函数（闭包，不使用 exec）
# ============================================================================

def _create_algorithm_tool_function(
    backbone_name: str,
    task_name: str,
    conv_class,
    conv_kwargs: dict,
    processor_getter: Callable,
    post_processing_decorator: Callable,
) -> Callable:
    """
    用闭包捕获 backbone_name / task_name / conv_class / conv_kwargs，
    生成具体的工具函数。参数签名固定，不需要 exec。
    """

    def _tool_impl(
        num_layers: int = 2,
        hidden_channels: int = 64,
        epochs: int = 200,
        lr: float = 0.01,
        train_ratio: float = 0.8,
        node_features: Optional[List] = None,
        node_labels: Optional[Dict] = None,
        node_labels_field: str = "",
    ) -> str:
        kwargs = dict(
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            epochs=epochs,
            lr=lr,
            train_ratio=train_ratio,
            node_features=node_features,
            node_labels=node_labels,
            node_labels_field=node_labels_field,
        )
        if task_name == "node_classification":
            return _execute_node_classification(
                backbone_name, conv_class, conv_kwargs, processor_getter, kwargs
            )
        else:
            return _execute_link_prediction(
                backbone_name, conv_class, conv_kwargs, processor_getter, kwargs
            )

    return post_processing_decorator(_tool_impl)


# ============================================================================
# 注册入口
# ============================================================================

def register_pyg_algorithm_tools(
    mcp,
    processor_getter: Callable,
    post_processing_decorator: Callable,
) -> int:
    """
    动态注册所有 backbone × task 组合工具到 MCP 服务器。
    每次新增 backbone 只需在 BACKBONE_CONFIGS 里加一条；
    每次新增 task 只需在 TASK_NAMES 和对应执行函数里扩展。
    """
    logger.info("🚀 开始注册 PyG 算法级工具...")
    registered_count = 0

    for backbone_name, backbone_cfg in BACKBONE_CONFIGS.items():
        for task_name in TASK_NAMES:
            tool_name = f"run_{backbone_name}_{task_name}"
            try:
                conv_class  = backbone_cfg["conv_class"]
                conv_kwargs = backbone_cfg["conv_kwargs"]
                description = TASK_DESCRIPTIONS[task_name].format(
                    backbone=backbone_cfg["description"]
                )

                tool_func = _create_algorithm_tool_function(
                    backbone_name=backbone_name,
                    task_name=task_name,
                    conv_class=conv_class,
                    conv_kwargs=conv_kwargs,
                    processor_getter=processor_getter,
                    post_processing_decorator=post_processing_decorator,
                )
                tool_func.__doc__ = description

                mcp.tool(name=tool_name, description=description)(tool_func)

                registered_count += 1
                logger.info(f"✅ 已注册算法工具: {tool_name}")

            except Exception as e:
                logger.error(f"❌ 注册失败 '{tool_name}': {e}", exc_info=True)

    logger.info(f"📊 算法级工具注册统计: ✅ {registered_count} 成功")
    return registered_count
