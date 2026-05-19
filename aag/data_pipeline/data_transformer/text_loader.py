"""
text_loader.py
----------------
文本数据加载引擎（TextDataLoader），未实现

负责：
1. 从 YAML 注册表读取文本数据集定义；
2. 加载文本文件（txt / json / markdown / docx 等）；
3. 转换为文档对象列表（list[dict]）；
"""

from typing import Any, Dict, List, Optional
import yaml
from aag.config.data_upload_config import *


class TextDataLoader:
    """
    文本数据加载引擎，用于加载非结构化文本类数据集。

    注意：核心方法（load_text_dataset 等）当前处于待实现状态。
    调用未实现的方法将抛出 NotImplementedError 而非静默返回空值。
    """

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize text data loader.

        Args:
            schema_path (str, optional): Path to text schemas YAML file.
                                        If None, schemas will be loaded by DatasetManager.
        """
        self.schema_path = schema_path
        self.dataset_schemas: Dict[str, List[DatasetConfig]] = {}

        # Only load from schema_path if provided (for backward compatibility)
        if schema_path:
            self._load_yaml()

    def _load_yaml(self):
        """
        读取 text_schemas.yaml 文件 (for backward compatibility)
        Converts Dict[str, DatasetConfig] to Dict[str, List[DatasetConfig]]
        Note: For text datasets, multiple files may share the same dataset name prefix.
        This method groups configs by dataset name.
        """
        try:
            old_map = read_datasets_map(self.schema_path)
            # Convert to new format: Dict[str, List[DatasetConfig]]
            # Group configs by dataset name (extract prefix before first underscore)
            grouped = {}
            for name, config in old_map.items():
                # Extract dataset name (prefix before first underscore, or full name if no underscore)
                dataset_name = name.split("_")[0] if "_" in name else name
                if dataset_name not in grouped:
                    grouped[dataset_name] = []
                grouped[dataset_name].append(config)
            self.dataset_schemas = grouped
            print(
                f"[INFO] Loaded {len(self.dataset_schemas)} datasets from {self.schema_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Failed to load dataset schemas from {self.schema_path}: {e}"
            )

    def _load_registry(self) -> Dict[str, Any]:
        """
        加载文本数据注册表（YAML 文件）。

        Returns:
            dict: 已解析的注册表内容。
        """
        raise NotImplementedError(
            "TextDataLoader._load_registry: YAML 注册表加载功能尚未实现"
        )

    # ==========================================================
    # 主功能接口
    # ==========================================================

    def list_datasets(self) -> List[str]:
        """
        列出当前注册表中所有文本数据集名称。
        """
        raise NotImplementedError(
            "TextDataLoader.list_datasets: 数据集列表功能尚未实现"
        )

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """
        获取指定文本数据集的详细配置信息。

        Args:
            name (str): 数据集名称。
        Returns:
            dict: 数据集定义信息。
        """
        raise NotImplementedError(
            "TextDataLoader.get_dataset_info: 数据集信息查询功能尚未实现"
        )

    def load_text_dataset(self, name: str) -> List[Dict[str, Any]]:
        """
        加载指定名称的文本数据集。

        Args:
            name (str): 数据集名称。
        Returns:
            list[dict]: 每条记录为 {"id": str, "text": str, "meta": dict}
        """
        raise NotImplementedError(
            "TextDataLoader.load_text_dataset: 文本数据集加载功能尚未实现。"
            "需要完成: 1. 查找定义 2. 根据格式读取 3. 解析为标准格式 4. 返回结果"
        )

    # ==========================================================
    # 可扩展方法
    # ==========================================================

    def preview(self, name: str, n: int = 3) -> Optional[List[str]]:
        """
        预览前 n 条文本内容。
        """
        raise NotImplementedError("TextDataLoader.preview: 文本预览功能尚未实现")

    def to_documents(self, name: str) -> List[Dict[str, Any]]:
        """
        转换为统一文档对象格式，用于后续向量化。
        """
        raise NotImplementedError(
            "TextDataLoader.to_documents: 文档转换功能尚未实现。"
            "需要完成: 1. 加载文本 2. 标准化结构 3. 返回文档列表"
        )

    def register_new_dataset(self, dataset_info: Dict[str, Any]) -> None:
        """
        将新的文本数据集注册到 YAML 文件中。

        Args:
            dataset_info (dict): 数据集配置项。
        """
        raise NotImplementedError(
            "TextDataLoader.register_new_dataset: 数据集注册功能尚未实现。"
            "需要完成: 1. 加载 YAML 2. 追加配置 3. 保存回文件"
        )

    def validate_schema(self, dataset_info: Dict[str, Any]) -> bool:
        """
        校验文本数据集的 schema 是否符合模板。
        """
        raise NotImplementedError(
            "TextDataLoader.validate_schema: Schema 校验功能尚未实现"
        )
