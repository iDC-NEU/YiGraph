import logging
from typing import Dict, Optional
from aag.config.data_upload_config import *
from aag.expert_search_engine.database.datatype import VertexData, EdgeData
from aag.expert_search_engine.database.nebulagraph import  NebulaGraphClient

logger = logging.getLogger(__name__)

class GraphDataLoader:
    """
    图数据加载引擎：负责
    1. 读取 graph_schemas.yaml 文件（系统内已有数据集定义）
    2. 根据指定 DatasetConfig 加载数据文件
    3. 将数据写入图数据库（示例代码）
    4. 从数据库中统计节点/边数，更新 metadata 信息
    """

    def __init__(self, schema_path: str):
        """
        初始化图数据加载器。

        Args:
            schema_path (str): 存储所有图数据集 schema 的 YAML 文件路径。
        """
        self.schema_path = schema_path                         
        self.dataset_schemas: Dict[str, DatasetConfig] = {}     
        self.graphdb_client: Optional[NebulaGraphClient] = None                   
        self._load_yaml()                

        try:
            self.graphdb_client = NebulaGraphClient()
            logger.info("✅ 已初始化 NebulaGraphClient 实例")
        except Exception as e:
            logger.warning(f"⚠️ NebulaGraphClient 初始化失败：{e}")
            self.graphdb_client = None

        # logger.info(f"📚 GraphDataLoader 已初始化，共加载 {len(self.dataset_schemas)} 个图数据集定义")
                      

    
    def load_dataset(self, dataset_config: DatasetConfig):
        """
            加载单个图数据集。
            包括：
            1. 将节点、边文件写入图数据库；
            2. 统计节点和边数量；
            3. 更新 DatasetConfig 的 graph_store_info；
            4. 追加写回 graph_schemas.yaml。
        """
        # === Step 1: 将数据写入图数据库（示例代码） ===
        # TODO: 此处仅写示例逻辑，不实现物理数据库操作。
        # 假设 dataset_config.schema.vertex 和 edge 已经定义路径。
        print(f"[INFO] Loading dataset '{dataset_config.name}' into graph database...")


        # === Step 2: 模拟统计结果并更新 graph_store_info ===
        # TODO: 实际应从数据库查询，例如：MATCH (n) RETURN count(n)
        vertex_count = 1446  # 示例
        edge_count = 17512    # 示例

        # 构造 GraphMetadataConfig 并填充信息
        graph_store_info = GraphStoreInfoConfig(
            backend="nebula_graph",
            space_name=dataset_config.name,  # 默认使用数据集名作为 space 名
            vertex_count=vertex_count,
            edge_count=edge_count,
            status="success"
        )
        dataset_config.schema.graph_store_info = graph_store_info

        # === Step 3: 将更新后的数据集写回 YAML 注册表 ===
        # 补充代码：添加节点和边的属性字段， 如果 dataset_config的 vertextschemaconfig 或 edgeschemaconfig 里的attribute_fields 为空，对去对应文件的第一行，将第一行的filed 除了 在 vertextschemaconfig 或 edgeschemaconfig 里注册的都登记成 里的attribute_fields
        append_dataset_to_yaml(dataset_config, self.schema_path)
        print(f"[INFO] Dataset '{dataset_config.name}' registered in {self.schema_path}")

        # === Step 4: 同步内存对象 ===
        self.dataset_schemas[dataset_config.name] = dataset_config


    def _load_yaml(self):
        """读取 graph_schemas.yaml 文件"""
        try:
            self.dataset_schemas = read_datasets_map(self.schema_path)
            print(f"[INFO] Loaded {len(self.dataset_schemas)} datasets from {self.schema_path}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load dataset schemas from {self.schema_path}: {e}")

    def get_graph_content(self, dataset_config: DatasetConfig):
        """
        从图数据库中加载指定数据集的完整图结构。
        自动切换空间并返回标准化的顶点与边对象列表。
        """
        if not self.graphdb_client:
            raise RuntimeError("❌ GraphDB 客户端未初始化，请检查连接状态")

        space_name = dataset_config.schema.graph_store_info.space_name
        if not space_name:
            raise ValueError("❌ 数据集配置中未定义 space_name，无法加载图数据")

        # 切换图空间
        self.graphdb_client.use_space(space_name)

        # 获取图结构
        vertices, edges = self.graphdb_client.get_full_graph()

        # 返回 VertexData / EdgeData 列表
        return vertices, edges
