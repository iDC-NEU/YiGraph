from typing import Dict, List, Optional
from typing import Any, Tuple, Union
from pathlib import Path
import yaml
from aag.data_pipeline.data_transformer.graph_loader import GraphDataLoader
from aag.data_pipeline.data_transformer.table_loader import TableDataLoader
from aag.data_pipeline.data_transformer.text_loader import TextDataLoader
from aag.config.data_upload_config import DataUploadConfig, DatasetConfig, load_data_upload_config
from aag.utils.path_utils import  DATASETS_INDEX_PATH


class DatasetManager:
    """Unified dataset registration and loading entry point"""

    VALID_DATA_TYPES = {"graph", "table", "text"}

    def __init__(self, datasets_index_path: Optional[Path] = None):
        """
        Initialize dataset manager
        
        Args:
            datasets_index_path: Path to datasets.yaml index file. 
                                If None, uses default path from path_utils
        """
        if datasets_index_path is None:
            datasets_index_path = DATASETS_INDEX_PATH
        
        self.datasets_index_path = Path(datasets_index_path)
        self.datasets_schema_path = self.datasets_index_path.parent
        
        self.graph_loader = GraphDataLoader()
        self.table_loader = TableDataLoader()
        self.text_loader = TextDataLoader()

        self.loaders = {
            "graph": self.graph_loader,
            "table": self.table_loader,
            "text": self.text_loader,
        }
        
        # Cache dataset index for quick lookup
        self.datasets_index: Dict[str, Dict] = {}
        
        # Load all datasets from the new architecture
        self._load_all_datasets()


    def _load_all_datasets(self):
        """
        Load all datasets from the new architecture:
        1. Read datasets.yaml to get dataset list
        2. For each dataset, load corresponding schema files from dataset folder
        """
        if not self.datasets_index_path.exists():
            print(f"[WARNING] Datasets index file not found: {self.datasets_index_path}")
            return
        
        try:
            with open(self.datasets_index_path, 'r', encoding='utf-8') as f:
                index_data = yaml.safe_load(f) or {}
            
            datasets_list = index_data.get('datasets', [])
            
            for dataset_info in datasets_list:
                dataset_name = dataset_info.get('name')
                dataset_type = dataset_info.get('type', '').lower()
                data_path = dataset_info.get('data_path', dataset_name)
                
                if not dataset_name or dataset_type not in self.VALID_DATA_TYPES:
                    continue

                self.datasets_index[dataset_name] = dataset_info
                dataset_folder = self.datasets_schema_path / data_path
                
                # Load schema based on dataset type
                if dataset_type == "graph":
                    schema_file = dataset_folder / "graph_schemas.yaml"
                    if schema_file.exists():
                        self._load_graph_schemas(schema_file, dataset_name)
                
                elif dataset_type == "text":
                    schema_file = dataset_folder / "text_schemas.yaml"
                    if schema_file.exists():
                        self._load_text_schemas(schema_file, dataset_name)
                
                elif dataset_type == "table":
                    schema_file = dataset_folder / "table_schemas.yaml"
                    if schema_file.exists():
                        self._load_table_schemas(schema_file, dataset_name)
        
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load datasets from index: {e}")
    
    def _load_graph_schemas(self, schema_file: Path, dataset_name: str):
        """
        Load graph schemas from a dataset's graph_schemas.yaml
        
        For graph datasets, graph_schemas.yaml should contain only one entry.
        Store as dataset_name -> List[DatasetConfig] (list with single config)
        """
        try:
            schemas = load_data_upload_config(str(schema_file))
            if not schemas.datasets:
                print(f"[WARNING] No datasets found in {schema_file}")
                return
            
            # Graph schemas should have only one entry
            if len(schemas.datasets) > 1:
                print(f"[WARNING] Graph schema file {schema_file} contains {len(schemas.datasets)} entries, expected 1. Using first entry.")
            
            ds_config = schemas.datasets[0]
            # Update path to be absolute if it's relative
            self._resolve_paths(ds_config, schema_file.parent)
            # Store with dataset-level name as key, value is list with single config
            self.graph_loader.dataset_schemas[dataset_name] = [ds_config]
        except Exception as e:
            print(f"[WARNING] Failed to load graph schemas from {schema_file}: {e}")
    
    def _load_text_schemas(self, schema_file: Path, dataset_name: str):
        """
        Load text schemas from a dataset's text_schemas.yaml
        
        For text datasets, text_schemas.yaml may contain multiple file entries.
        Store as dataset_name -> List[DatasetConfig] (list of file configs)
        """
        try:
            schemas = load_data_upload_config(str(schema_file))
            if not schemas.datasets:
                print(f"[WARNING] No datasets found in {schema_file}")
                return
            
            # Collect all file-level configs for this dataset
            file_configs = []
            for ds_config in schemas.datasets:
                # Update path to be absolute if it's relative
                self._resolve_paths(ds_config, schema_file.parent)
                file_configs.append(ds_config)
            
            # Store with dataset-level name as key, value is list of file configs
            self.text_loader.dataset_schemas[dataset_name] = file_configs
        except Exception as e:
            print(f"[WARNING] Failed to load text schemas from {schema_file}: {e}")
    
    def _load_table_schemas(self, schema_file: Path, dataset_name: str):
        """
        Load table schemas from a dataset's table_schemas.yaml
        
        Store as dataset_name -> List[DatasetConfig] (list with single config)
        """
        try:
            schemas = load_data_upload_config(str(schema_file))
            if not schemas.datasets:
                print(f"[WARNING] No datasets found in {schema_file}")
                return
            
            # Collect all table configs for this dataset
            table_configs = []
            for ds_config in schemas.datasets:
                # Update path to be absolute if it's relative
                self._resolve_paths(ds_config, schema_file.parent)
                table_configs.append(ds_config)
            
            # Store with dataset-level name as key, value is list of configs
            self.table_loader.dataset_schemas[dataset_name] = table_configs
        except Exception as e:
            print(f"[WARNING] Failed to load table schemas from {schema_file}: {e}")
    
    def _resolve_paths(self, dataset_config: DatasetConfig, schema_dir: Path):
        """
        Resolve relative paths in dataset config to absolute paths
        
        Args:
            dataset_config: Dataset configuration object
            schema_dir: Directory where schema file is located
        """
        if dataset_config.type == "text":
            # For text, schema.path is relative to schema file
            if hasattr(dataset_config.schema, 'path'):
                rel_path = dataset_config.schema.path
                if not Path(rel_path).is_absolute():
                    abs_path = (schema_dir / rel_path).resolve()
                    dataset_config.schema.path = str(abs_path)
        
        elif dataset_config.type == "graph":
            # For graph, vertex and edge paths are relative to schema file
            if hasattr(dataset_config.schema, 'vertex'):
                for v in dataset_config.schema.vertex:
                    if hasattr(v, 'path') and not Path(v.path).is_absolute():
                        abs_path = (schema_dir / v.path).resolve()
                        v.path = str(abs_path)
            
            if hasattr(dataset_config.schema, 'edge'):
                for e in dataset_config.schema.edge:
                    if hasattr(e, 'path') and not Path(e.path).is_absolute():
                        abs_path = (schema_dir / e.path).resolve()
                        e.path = str(abs_path)
        
        elif dataset_config.type == "table":
            # For table, schema.path is relative to schema file
            if hasattr(dataset_config.schema, 'path'):
                rel_path = dataset_config.schema.path
                if not Path(rel_path).is_absolute():
                    abs_path = (schema_dir / rel_path).resolve()
                    dataset_config.schema.path = str(abs_path)

    def load_dataset(self, data_upload_config: DataUploadConfig) -> None:
        """
        Load datasets from DataUploadConfig and dispatch to corresponding loaders
        
        Args:
            data_upload_config: User uploaded data configuration object
            
        Raises:
            ValueError: Unsupported dataset type
        """

        # todo： 这里应该是要根据配置文件来加载数据集， 需要用户把真实数据放在data目录下，然后根据配置文件来填充 self.datasets_index 和 self.loaders
        for dataset in data_upload_config.datasets:
            dtype = (dataset.type or "").lower().strip()
            if dtype not in self.loaders:
                raise ValueError(
                    f"Invalid data type '{dtype}'. Should be one of {', '.join(self.VALID_DATA_TYPES)}."
                )
            self.loaders[dtype].load_dataset(dataset)


    def list_datasets(self, dtype: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all or specified type of dataset names (dataset-level only)
        
        Args:
            dtype: Data type, can be "graph" / "table" / "text".
                  If not specified, returns all types
                   
        Returns:
            Dictionary with structure {"graph": [...], "table": [...], "text": [...]}
            Only returns dataset-level names from datasets.yaml, not file-level names
            
        Raises:
            ValueError: Unsupported data type
        """
        dtype = (dtype or "").lower().strip()

        # Get dataset-level names from datasets.yaml index
        result = {"graph": [], "table": [], "text": []}
        
        for dataset_name, dataset_info in self.datasets_index.items():
            dataset_type = dataset_info.get('type', '').lower()
            if dataset_type in self.VALID_DATA_TYPES:
                result[dataset_type].append(dataset_name)
        
        if not dtype:
            return result

        if dtype not in self.VALID_DATA_TYPES:
            raise ValueError(
                f"Invalid data type '{dtype}'. Should be one of {', '.join(self.VALID_DATA_TYPES)}."
            )

        return {dtype: result[dtype]}
    

    def get_dataset_content(
        self, dataset_config: DatasetConfig
    ) -> Union[Tuple[list, list], Any]:
        """
        Automatically load data content based on data type.

        Returns:
            graph → (vertices, edges)
            table → pandas.DataFrame
            text → list[str]
        """
        dtype = dataset_config.type.lower()
        
        if dtype == "graph":
            return self.graph_loader.get_graph_content_from_raw(dataset_config)
        
        elif dtype == "table":
            path = dataset_config.schema.path
            import pandas as pd
            return pd.read_csv(path)
        
        elif dtype == "text":
            path = dataset_config.schema.path
            encoding = getattr(dataset_config.schema, "encoding", "utf-8")
            with open(path, "r", encoding=encoding) as f:
                return f.readlines()
        
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    def get_dataset_info(self, name: str, dtype: Optional[str] = None) -> Optional[List[DatasetConfig]]:
        """
        Get detailed information of a dataset (schema)
        
        Args:
            name: Dataset name (dataset-level)
            dtype: Data type, can be "graph" / "table" / "text".
                  If not specified, searches in all types and returns first match
                   
        Returns:
            - For graph/table: List[DatasetConfig] with single config
            - For text: List[DatasetConfig] with multiple file configs
            - None if not found
            
        Raises:
            ValueError: Invalid data type
        """
        dtype = (dtype or "").lower().strip()
        
        if dtype and dtype not in self.loaders:
            raise ValueError(
                f"Invalid data type '{dtype}'. Should be one of {', '.join(self.VALID_DATA_TYPES)}."
            )

        # Specified type: exact match
        if dtype in self.loaders:
            ds = self.loaders[dtype].dataset_schemas.get(name)
        else:
            # Unspecified type: fuzzy search
            ds = next(
                (loader.dataset_schemas[name]
                 for loader in self.loaders.values()
                 if name in loader.dataset_schemas),
                None
            )

        return ds
    
    def get_dataset_original_type(self, dataset_name: str) -> Optional[str]:
        """
        Get the original type of a dataset from datasets.yaml
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            "graph", "text", "table", or None if not found
        """
        if dataset_name in self.datasets_index:
            return self.datasets_index[dataset_name].get('type', '').lower()
        return None
    
    def get_converted_graph_dataset(self, dataset_name: str) -> Optional[DatasetConfig]:
        """
        Get the converted graph dataset config for a text dataset.
        Returns None if the dataset has not been converted to graph.
        
        This method combines the check and retrieval logic to avoid duplicate file loading.
        
        Args:
            dataset_name: Name of the text dataset
            
        Returns:
            DatasetConfig if converted graph exists, None otherwise
        """
        # Check if dataset exists in index
        if dataset_name not in self.datasets_index:
            return None
        
        dataset_info = self.datasets_index[dataset_name]
        original_type = dataset_info.get('type', '').lower()
        
        if original_type != 'text':
            return None

        data_path = dataset_info.get('data_path', dataset_name)
        dataset_folder = self.datasets_schema_path / data_path
        graph_schema_file = dataset_folder / "graph_schemas.yaml"
        
        if not graph_schema_file.exists():
            return None
        
        try:
            schemas = load_data_upload_config(str(graph_schema_file), validate_files=False)
            if schemas.datasets:
                graph_config = schemas.datasets[0]
                self._resolve_paths(graph_config, graph_schema_file.parent)
                return graph_config
        except Exception as e:
            print(f"[WARNING] Failed to load converted graph for {dataset_name}: {e}")
        
        return None