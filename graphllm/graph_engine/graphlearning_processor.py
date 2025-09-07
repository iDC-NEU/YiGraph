import logging
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, HeteroData
from typing import List, Tuple, Dict, Any, Optional, Union
from graphllm.database.datatype import *
from graphllm.database.nebulagraph import *
from graphllm.graph_engine.graph_processor import GraphProcessor
from graphllm.graph_engine.utils.graph_feature_encoder import GraphFeatureEncoder

logger = logging.getLogger(__name__)

class GCN(torch.nn.Module):
    """
    GCN模型用于顶点分类
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第三层卷积
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GraphLearningProcessor(GraphProcessor):
    def __init__(self):
        super().__init__()
        self.graph = None
        self.vertices = None
        self.edges = None
        self.is_directed = False
        self.feature_encoder = GraphFeatureEncoder()


    def create_graph_from_edges(self, vertices: List[VertexData], edges: List[EdgeData],  hetero: bool = False):
        """
        从顶点和边列表创建 PyG 图

        Args:
            vertices: 顶点列表
            edges: 边列表
            hetero: 是否异构图（HeteroData）
        
        Returns:
            Data 或 HeteroData 对象
        """
        try:
            # Step 1: 生成节点标签（优先label -> prior_sar_count -> 默认0）
            y = []
            for vertex in vertices:
                if "label" in vertex.properties:
                    y.append(vertex.properties["label"])
                elif "prior_sar_count" in vertex.properties:
                    y.append(vertex.properties["prior_sar_count"])
                else:
                    raise ValueError(f"Vertex {vertex.vid} 缺少 label 或 prior_sar_count")
            y = torch.tensor(y, dtype=torch.long)  # 节点标签向量

            self.get_vertex_map(vertices)
            self.hetero = hetero

            if self.hetero:
                self.graph = HeteroData()
                # 异构节点特征示例
                node_features = self.feature_encoder.fit_transform(vertices, mode="random")
                # TODO: 根据业务逻辑把不同类型节点和边放入 HeteroData
                # data['node_type'].x = ...

            else:
                # 如果是同构图，则需要将节点分为不同的类型，使用pyg的同构图类型存储
                node_features = self.feature_encoder.fit_transform(vertices, "random")
                
                # 构建 edge_index
                src_list, dst_list = [], []
                for e in edges:
                    src_list.append(e.src)
                    dst_list.append(e.dst)

                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

                # Step 5: 构造 PyG Data
                self.graph = Data(
                    x=node_features,      # 节点特征
                    edge_index=edge_index, # 边索引
                    y=y                   # 节点标签
                )
            
            return self.graph

        except Exception as e:
            logger.error(f"创建图时发生错误: {e}")
            raise
        

    def get_vertex_map(self, vertices: List[VertexData]):
        """
        将 self.vertices 转换为以 vid 为键、VertexData 为值的字典。
        """
        self.vertices =  {vertex.vid: vertex for vertex in vertices}

    def get_graph_info(self) -> Dict[str, Any]:
        """
        获取图的基本信息
        
        Returns:
            包含图信息的字典
        """
        if self.graph is None:
            return {"error": "图未初始化"}

        info: Dict[str, Any] = {}

        # 基本信息
        info["num_nodes"] = self.graph.num_nodes
        info["num_edges"] = self.graph.num_edges
        info["num_node_features"] = self.graph.num_node_features
        info["has_edge_attr"] = hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None
        info["has_labels"] = hasattr(self.graph, 'y') and self.graph.y is not None

        # 打印第一个节点信息
        if self.graph.num_nodes > 0:
            info["first_node_feature"] = self.graph.x[0].tolist() if self.graph.x is not None else None
            if info["has_labels"]:
                info["first_node_label"] = self.graph.y[0].item()
        
        # 打印前几条边
        num_print_edges = min(10, self.graph.num_edges)
        edge_samples = []
        for i in range(num_print_edges):
            src, dst = self.graph.edge_index[0, i].item(), self.graph.edge_index[1, i].item()
            edge_info = {"src": src, "dst": dst}
            if info["has_edge_attr"]:
                edge_info["attr"] = self.graph.edge_attr[i].tolist()
            edge_samples.append(edge_info)
        info["sample_edges"] = edge_samples

        return info
    
    #用生成的图执行gcn计算， 下游任务是顶点分类
    def train_gcn_for_node_classification(self, 
                                        hidden_channels: int = 64,
                                        learning_rate: float = 0.01,
                                        weight_decay: float = 5e-4,
                                        epochs: int = 800,
                                        train_ratio: float = 0.8,
                                        val_ratio: float = 0.1,
                                        test_ratio: float = 0.1,
                                        device: str = 'gpu:0') -> Dict[str, Any]:
        """
        使用GCN进行顶点分类训练
        
        Args:
            hidden_channels: 隐藏层维度
            learning_rate: 学习率
            weight_decay: 权重衰减
            epochs: 训练轮数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            device: 设备类型 ('cpu' 或 'cuda')
        
        Returns:
            包含训练结果的字典
        """
        if self.graph is None:
            raise ValueError("图未初始化，请先调用 create_graph_from_edges")
        
        # 将图移动到指定设备
        device = torch.device(device)
        data = self.graph.to(device)
        
        # 获取图的属性
        num_features = data.num_node_features
        num_classes = data.y.max().item() + 1
        num_nodes = data.num_nodes
        
        # 创建GCN模型
        model = GCN(num_features, hidden_channels, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 划分数据集
        indices = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        # 训练循环
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 训练模式
            model.train()
            optimizer.zero_grad()
            
            # 前向传播
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[train_mask], data.y[train_mask])
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_pred = val_out[val_mask].argmax(dim=1)
                val_acc = (val_pred == data.y[val_mask]).float().mean().item()
                
                # 测试
                test_out = model(data.x, data.edge_index)
                test_pred = test_out[test_mask].argmax(dim=1)
                test_acc = (test_pred == data.y[test_mask]).float().mean().item()
            
            train_losses.append(loss.item())
            val_accuracies.append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # 打印进度
            if (epoch + 1) % 20 == 0:
                logger.info(f'Epoch {epoch+1:03d}: Loss: {loss:.4f}, '
                          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
                print(f'Epoch {epoch+1:03d}: Loss: {loss:.4f}, '
                          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        # 加载最佳模型进行最终测试
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            final_out = model(data.x, data.edge_index)
            final_pred = final_out[test_mask].argmax(dim=1)
            final_test_acc = (final_pred == data.y[test_mask]).float().mean().item()
        
        # 计算所有节点的预测结果
        all_pred = final_out.argmax(dim=1)
        
        # 返回结果
        results = {
            'model': model,
            'best_val_accuracy': best_val_acc,
            'final_test_accuracy': final_test_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'predictions': all_pred.cpu().numpy(),
            'true_labels': data.y.cpu().numpy(),
            'train_mask': train_mask.cpu().numpy(),
            'val_mask': val_mask.cpu().numpy(),
            'test_mask': test_mask.cpu().numpy(),
            'model_config': {
                'hidden_channels': hidden_channels,
                'num_features': num_features,
                'num_classes': num_classes,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'epochs': epochs
            }
        }
        
        logger.info(f"训练完成！最终测试准确率: {final_test_acc:.4f}")
        return results
    
    def predict_node_labels(self, model: torch.nn.Module, device: str = 'cpu') -> np.ndarray:
        """
        使用训练好的GCN模型预测节点标签
        
        Args:
            model: 训练好的GCN模型
            device: 设备类型
        
        Returns:
            预测的节点标签数组
        """
        if self.graph is None:
            raise ValueError("图未初始化，请先调用 create_graph_from_edges")
        
        device = torch.device(device)
        data = self.graph.to(device)
        model = model.to(device)
        
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
        
        return pred.cpu().numpy()
    
    def evaluate_node_classification(self, 
                                  predictions: np.ndarray, 
                                  true_labels: np.ndarray,
                                  mask: np.ndarray = None) -> Dict[str, float]:
        """
        评估顶点分类结果
        
        Args:
            predictions: 预测标签
            true_labels: 真实标签
            mask: 评估掩码（可选）
        
        Returns:
            包含评估指标的字典
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        
        if mask is not None:
            pred = predictions[mask]
            true = true_labels[mask]
        else:
            pred = predictions
            true = true_labels
        
        accuracy = accuracy_score(true, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # 生成详细分类报告
        report = classification_report(true, pred, output_dict=True)
        results['detailed_report'] = report
        
        return results

if __name__ == '__main__':
    testclient = NebulaGraphClient(space_name="AMLSim1K")
    vertices = testclient.get_all_vertices()
    edges = testclient.get_all_edges()
    print(f"vertices number: {len(vertices)}, edge number: {len(edges)}")
    graphprocessor = GraphLearningProcessor()
    graph=graphprocessor.create_graph_from_edges(vertices, edges)
    info=graphprocessor.get_graph_info()
    print(info)
    results=graphprocessor.train_gcn_for_node_classification()
    print(results)