import logging
from typing import Dict, List, Optional
from aag.config.data_upload_config import *
from aag.expert_search_engine.database.datatype import VertexData, EdgeData
from aag.expert_search_engine.database.nebulagraph import  NebulaGraphClient
import pandas as pd

logger = logging.getLogger(__name__)

class GraphDataLoader:
    """
    Graph data loading engine: responsible for
    1. Reading graph_schemas.yaml files (existing dataset definitions in the system)
    2. Loading data files according to specified DatasetConfig
    3. Writing data to graph database (example code)
    4. Counting nodes/edges from database and updating graph_store_info
    """

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize graph data loader.

        Args:
            schema_path (str, optional): Path to graph schemas YAML file.
                                        If None, schemas will be loaded by DatasetManager.
        """
        self.schema_path = schema_path                         
        self.dataset_schemas: Dict[str, List[DatasetConfig]] = {}     
        self.graphdb_client: Optional[NebulaGraphClient] = None
        
        # Only load from schema_path if provided (for backward compatibility)
        if schema_path:
            self._load_yaml()                

        try:
            self.graphdb_client = NebulaGraphClient()
            logger.info("✅ Initialized NebulaGraphClient instance")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize NebulaGraphClient: {e}")
            self.graphdb_client = None
                      

    
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
        # Store as list to match new architecture (dataset_name -> List[DatasetConfig])
        self.dataset_schemas[dataset_config.name] = [dataset_config]


    def _load_yaml(self):
        """
        读取 graph_schemas.yaml 文件 (for backward compatibility)
        Converts Dict[str, DatasetConfig] to Dict[str, List[DatasetConfig]]
        """
        try:
            old_map = read_datasets_map(self.schema_path)
            # Convert to new format: Dict[str, List[DatasetConfig]]
            self.dataset_schemas = {name: [config] for name, config in old_map.items()}
            print(f"[INFO] Loaded {len(self.dataset_schemas)} datasets from {self.schema_path}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load dataset schemas from {self.schema_path}: {e}")
    

    def get_field_data(self, config, key):
        from dataclasses import fields

        result = []

        for f in fields(config):
            if key in f.name:
                value = getattr(config, f.name)

                if value is None:
                    continue  # 跳过 None

                if isinstance(value, list):
                    result += value
                else:
                    result.append(value)


        return list(set(result))


    def load_data_from_raw(self, data_config):

        all_dfs = []

        for config in data_config:
            all_fields = self.get_field_data(config, "field")
            logger.info(f"Reading ({all_fields}) fields from raw file")
            df = pd.read_csv(config.path, usecols=all_fields)

            # only for vetext data
            query_field = self.get_field_name(config, "query_field")
            if query_field and query_field not in df.columns:
                # print(f"{config.id_field=}")
                df[query_field] = df[config.query_field].astype(str).agg(" ".join, axis=1)

            all_dfs.append(df)

        merged_df = pd.concat(all_dfs, ignore_index=True)
        # print(merged_df.head())
        # print(f"{len(merged_df)=}")
        return merged_df


    def get_field_name(self, config, field):
        item = getattr(config, field, None)

        if item is None:
            return None

        if isinstance(item, list):
            return "_".join(item)

        if isinstance(item, str):
            return item

    
    def get_graph_content_from_raw(self, dataset_config: DatasetConfig):

        vertex_config = dataset_config.schema.vertex
        edge_config = dataset_config.schema.edge

        vertex_df = self.load_data_from_raw(vertex_config)
        edge_df = self.load_data_from_raw(edge_config)

        # process vertex: vid = query_name
        vertices = []
        query_name = self.get_field_name(vertex_config[0], "query_field")
        id_field = vertex_config[0].id_field
        for _, row in vertex_df.iterrows():
            vid = str(row[query_name])
            props = row.drop(query_name).to_dict()          
            vertices.append(VertexData(vid=vid, properties=props))
        

        # process edge: src => query_name, dst => query_name
        edges = []
        src_field = edge_config[0].source_field
        dst_field = edge_config[0].target_field
        id_to_query = dict(zip(vertex_df[id_field], vertex_df[query_name]))
        for _, row in edge_df.iterrows():
            src_raw = row[src_field]
            dst_raw = row[dst_field]

            try:
                src = str(id_to_query[src_raw])
                dst = str(id_to_query[dst_raw])
            except KeyError:
                raise ValueError(
                    f"Edge references unknown id: src={src_raw}, dst={dst_raw}"
                )

            # modify rank when need
            rank = _
            rank_field = None

            props = row.drop([src_field, dst_field] + ([rank_field] if rank_field else [])).to_dict()

            edges.append(EdgeData(src=src, dst=dst, rank=rank, properties=props))
        
        
        # print(f"{len(edges)=}")
        # print(edges[0])

        return vertices, edges
    
    
    
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
