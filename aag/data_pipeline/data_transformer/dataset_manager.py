from typing import Dict, List, Optional
from typing import Any, Tuple, Union
from dataclasses import asdict
from aag.data_pipeline.data_transformer.graph_loader import GraphDataLoader
from aag.data_pipeline.data_transformer.table_loader import TableDataLoader
from aag.data_pipeline.data_transformer.text_loader import TextDataLoader
from aag.config.data_upload_config import DataUploadConfig, DatasetConfig
from aag.utils.path_utils import GRAPH_SCHEMA_PATH, TABLE_SCHEMA_PATH, TEXT_SCHEMA_PATH


class DatasetManager:
    """统一管理数据集注册与加载的入口"""

    VALID_DATA_TYPES = {"graph", "table", "text"}

    def __init__(self, graph_schema_path: str = GRAPH_SCHEMA_PATH, 
                 table_schema_path: str = TABLE_SCHEMA_PATH, 
                 text_schema_path: str = TEXT_SCHEMA_PATH):
        """
        初始化数据集管理器
        
        Args:
            graph_schema_path: 图数据模式文件路径
            table_schema_path: 表格数据模式文件路径
            text_schema_path: 文本数据模式文件路径
        """
        self.graph_loader = GraphDataLoader(graph_schema_path)
        self.table_loader = TableDataLoader(table_schema_path)
        self.text_loader = TextDataLoader(text_schema_path)

        self.loaders = {
            "graph": self.graph_loader,
            "table": self.table_loader,
            "text": self.text_loader,
        }


    def load_dataset(self, data_upload_config: DataUploadConfig) -> None:
        """
        遍历 data_upload_config.datasets 并分发至对应加载器
        
        Args:
            data_upload_config: 用户上传数据配置对象
            
        Raises:
            ValueError: 不支持的数据集类型
        """
        for dataset in data_upload_config.datasets:
            dtype = (dataset.type or "").lower().strip()
            if dtype not in self.loaders:
                raise ValueError(
                    f"无效的数据类型 '{dtype}'。应为 {', '.join(self.VALID_DATA_TYPES)} 之一。"
                )
            self.loaders[dtype].load_dataset(dataset)


    def list_datasets(self, dtype: Optional[str] = None) -> Dict[str, List[str]]:
        """
        列出所有或指定类型的数据集名称
        
        Args:
            dtype: 数据类型，可为 "graph" / "table" / "text"。
                  如果未指定，则返回所有类型
                   
        Returns:
            结构为 {"graph": [...], "table": [...], "text": [...]} 的字典
            
        Raises:
            ValueError: 不支持的数据类型
        """
        dtype = (dtype or "").lower().strip()

        if not dtype:
            return {k: sorted(v.dataset_schemas.keys()) for k, v in self.loaders.items()}

        if dtype not in self.loaders:
            raise ValueError(
                f"无效的数据类型 '{dtype}'。应为 {', '.join(self.VALID_DATA_TYPES)} 之一。"
            )

        return {dtype: sorted(self.loaders[dtype].dataset_schemas.keys())}
    

    def get_dataset_content(
        self, dataset_config: DatasetConfig
    ) -> Union[Tuple[list, list], Any]:
        """
        根据数据类型自动加载数据内容。

        Returns:
            graph → (vertices, edges)
            table → pandas.DataFrame
            text → list[str]
        """
        dtype = dataset_config.type.lower()
        
        if dtype == "graph":
            return self.graph_loader.get_graph_content(dataset_config)
        
        elif dtype == "table":
            # ===== 表格数据逻辑 =====
            path = dataset_config.schema.path
            import pandas as pd
            return pd.read_csv(path)
        
        elif dtype == "text":
            # ===== 文本数据逻辑 =====
            path = dataset_config.schema.path
            encoding = getattr(dataset_config.schema, "encoding", "utf-8")
            with open(path, "r", encoding=encoding) as f:
                return f.readlines()
        
        else:
            raise ValueError(f"❌ 不支持的数据类型: {dtype}")

    def get_dataset_info(self, name: str, dtype: Optional[str] = None) -> Optional[DatasetConfig]:
        """
        获取某个数据集的详细信息（schema / metadata）
        
        Args:
            name: 数据集名称
            dtype: 数据类型，可为 "graph" / "table" / "text"。
                  若未指定，则在所有类型中搜索第一个匹配项
                   
        Returns:
            对应数据集配置对象；未找到则返回 None
            
        Raises:
            ValueError: 无效的数据类型
        """
        dtype = (dtype or "").lower().strip()
        
        if dtype and dtype not in self.loaders:
            raise ValueError(
                f"无效的数据类型 '{dtype}'。应为 {', '.join(self.VALID_DATA_TYPES)} 之一。"
            )

        # 指定类型：精准匹配
        if dtype in self.loaders:
            ds = self.loaders[dtype].dataset_schemas.get(name)
        else:
            # 未指定类型：模糊搜索
            ds = next(
                (loader.dataset_schemas[name]
                 for loader in self.loaders.values()
                 if name in loader.dataset_schemas),
                None
            )

        return ds