from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import List, Dict, Any, Union, Optional
import yaml
import os
from pathlib import Path

# ======== 各类型数据结构定义 ========
@dataclass
class VertexSchemaConfig:
    type: str
    path: str
    format: str
    query_field: str
    id_field: str
    label_field: Optional[str] = None
    attribute_fields: List[str] = field(default_factory=list)  # 节点属性字段列表

@dataclass
class EdgeSchemaConfig:
    type: str
    path: str
    format: str
    source_field: str
    target_field: str
    label_field: Optional[str] = None
    weight_field: Optional[str] = None
    rank_field: Optional[str] = None  # for multiple edges with the same source and target
    attribute_fields: List[str] = field(default_factory=list)  # 边属性字段列表

@dataclass
class GraphStructureConfig:
    directed: bool = True
    multigraph: bool = False
    weighted: bool = False
    heterogeneous: bool = False

@dataclass
class GraphStoreInfoConfig:
    backend: Optional[str] = None  # 图数据库类型，如 nebula_graph / neo4j
    space_name: Optional[str] = None  # 图数据库命名空间或图空间名
    vertex_count: Optional[int] = None
    edge_count: Optional[int] = None
    version: Optional[str] = None     # 数据版本号或导入批次
    status: Optional[str] = None      # 状态，如 "loaded" / "pending" / "failed"

@dataclass
class GraphSchemaConfig:
    vertex: List[VertexSchemaConfig]
    edge: List[EdgeSchemaConfig]
    graph_structure: GraphStructureConfig
    graph_store_info: GraphStoreInfoConfig
    


# ======== 表格型数据结构（示例，预留扩展） ========

@dataclass
class TableSchemaConfig:
    path: str
    format: str
    columns: List[str]
    primary_key: Optional[str] = None


# ======== 文本型数据结构（示例，预留扩展） ========
@dataclass
class TextSchemaConfig:
    path: str
    format: str
    encoding: str = "utf-8"


# ======== Dataset 配置结构 ========
@dataclass
class DatasetConfig:
    name: str
    type: str
    description: str
    schema: Union[GraphSchemaConfig, TableSchemaConfig, TextSchemaConfig]


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        dtype = data.get("type", "").lower()
        schema_data = data.get("schema", {})

        if dtype == "graph":
            schema_cls = GraphSchemaConfig
        elif dtype == "table":
            schema_cls = TableSchemaConfig
        elif dtype == "text":
            schema_cls = TextSchemaConfig
        else:
            raise ValueError(f"Unknown dataset type '{dtype}'")

        schema_obj = cls._from_dict_recursive(schema_cls, schema_data)
        return cls(
            name=data["name"],
            type=data["type"],
            description=data.get("description", ""),
            schema=schema_obj
        )

    @staticmethod
    def _from_dict_recursive(cls, data):
        if not is_dataclass(cls):
            return data
            
        kwargs = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if isinstance(value, dict):
                kwargs[f.name] = DatasetConfig._from_dict_recursive(f.type, value)
            elif isinstance(value, list):
                elem_type = getattr(f.type, "__args__", [Any])[0]
                kwargs[f.name] = [DatasetConfig._from_dict_recursive(elem_type, v) for v in value]
            else:
                kwargs[f.name] = value
        return cls(**kwargs)


@dataclass
class DataUploadConfig:
    datasets: List[DatasetConfig]
    
    def get_dataset(self, name: str) -> Optional[DatasetConfig]:
        """根据名称获取数据集配置"""
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None
    
    def list_datasets(self, dataset_type: Optional[str] = None) -> List[DatasetConfig]:
        """列出所有数据集，可选按类型过滤"""
        if dataset_type is None:
            return self.datasets
        return [d for d in self.datasets if d.type == dataset_type]
    
    def validate_dataset(self, dataset: DatasetConfig) -> List[str]:
        """验证数据集配置的有效性"""
        errors = []
        
        # 检查必需字段
        if not dataset.name:
            errors.append("数据集名称不能为空")
        
        if not dataset.type:
            errors.append("数据集类型不能为空")
        
        # 检查文件路径是否存在
        if dataset.type == "graph":
            for v in dataset.schema.vertex:
                if not os.path.exists(v.path):
                    errors.append(f"顶点文件不存在: {v.path}")
            for e in dataset.schema.edge:
                if not os.path.exists(e.path):
                    errors.append(f"边文件不存在: {e.path}")
        
        elif dataset.type == "table":
            if not os.path.exists(dataset.schema.path):
                errors.append(f"表格文件不存在: {dataset.schema.path}")
        
        elif dataset.type == "text":
            if not os.path.exists(dataset.schema.path):
                errors.append(f"文本文件不存在: {dataset.schema.path}")
        
        return errors


# ======== 动态加载函数 ========

def load_data_upload_config(yaml_path: str, validate_files: bool = False) -> DataUploadConfig:
    """
    从 data_upload_config.yaml 加载多类型数据注册配置（支持 graph/table/text）
    
    Args:
        yaml_path: YAML配置文件路径
        
    Returns:
        DataUploadConfig: 数据注册表配置对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
        ValueError: 配置格式错误
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"数据注册表配置文件不存在: {yaml_path}")
        
        # 加载YAML配置
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data or 'datasets' not in data:
            raise ValueError("配置文件格式错误：缺少 'datasets' 字段")
        
        datasets = []

        if not data.get("datasets"):
            return DataUploadConfig(datasets=[])

        for ds in data["datasets"]:
            dtype = ds.get("type", "").lower()

            # === 图数据（schema 内含 path/format、graph_structure、graph_store_info） ===
            if dtype == "graph":
                schema_block = ds.get("schema", {})
                vertex_items = schema_block.get("vertex", [])
                edge_items = schema_block.get("edge", [])

                # 组装 schema（字段名使用 *_field / attribute_fields）
                vertex_schema = [
                    VertexSchemaConfig(
                        type=v.get("type", ""),
                        path=v.get("original_path", ""),  #path
                        format=v.get("format", ""),
                        id_field=v.get("id_field", ""),
                        query_field=v.get("query_field", ""),
                        label_field=v.get("label_field"),
                        attribute_fields=v.get("attribute_fields", [])
                    ) for v in vertex_items
                ]
                edge_schema = [
                    EdgeSchemaConfig(
                        type=e.get("type", ""),
                        path=e.get("original_path", ""), #path
                        format=e.get("format", ""),
                        source_field=e.get("source_field", ""),
                        target_field=e.get("target_field", ""),
                        label_field=e.get("label_field"),
                        weight_field=e.get("weight_field"),
                        attribute_fields=e.get("attribute_fields", [])
                    ) for e in edge_items
                ]
                graph_structure = GraphStructureConfig(**schema_block.get("graph", {}))
                graph_store_info = GraphStoreInfoConfig(**schema_block.get("graph_store_info", {}))
                schema = GraphSchemaConfig(vertex=vertex_schema, edge=edge_schema, graph_structure=graph_structure, graph_store_info=graph_store_info)

            elif dtype == "table":
                s = ds.get("schema", {})
                schema = TableSchemaConfig(
                    path=s.get("path", ""),
                    format=s.get("format", ""),
                    columns=s.get("columns", []),
                    primary_key=s.get("primary_key")
                )

            elif dtype == "text":
                s = ds.get("schema", {})
                schema = TextSchemaConfig(
                    path=s.get("path", ""),
                    format=s.get("format", ""),
                    encoding=s.get("encoding", "utf-8")
                )

            else:
                raise ValueError(f"不支持的数据集类型: {dtype}")

            datasets.append(DatasetConfig(
                name=ds.get("name", ""),
                type=dtype,
                description=ds.get("description", ""),
                schema=schema
            ))

        # 创建数据注册表
        registry = DataUploadConfig(datasets=datasets)
        
        if validate_files:
            # 验证所有数据集
            for dataset in datasets:
                errors = registry.validate_dataset(dataset)
                if errors:
                    print(f"警告: 数据集 '{dataset.name}' 配置有问题:")
                    for error in errors:
                        print(f"  - {error}")
        
        return registry
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML解析错误: {e}")
    except Exception as e:
        raise ValueError(f"加载数据注册表失败: {e}")


def save_data_config(registry: DataUploadConfig, yaml_path: str) -> None:
    """
    保存数据注册表到YAML配置文件
    
    Args:
        registry: 数据注册表配置对象
        yaml_path: 配置文件路径
    """
    try:
        # 构建配置数据
        config_data = {
            'datasets': []
        }
        
        for dataset in registry.datasets:
            dataset_data = {
                'name': dataset.name,
                'type': dataset.type,
                'description': dataset.description,
                'schema': {}
            }
            
            # 根据数据类型构建 schema（按 data_upload_config.yaml 格式回写）
            if dataset.type == "graph":
                dataset_data['schema'] = {
                    'vertex': [asdict(vs) for vs in dataset.schema.vertex],
                    'edge': [asdict(es) for es in dataset.schema.edge],
                    'graph': asdict(dataset.schema.graph_structure),
                    'graph_store_info': asdict(dataset.schema.graph_store_info)
                }
            
            elif dataset.type == "table":
                dataset_data['schema'] = {
                    'path': dataset.schema.path,
                    'format': dataset.schema.format,
                    'columns': dataset.schema.columns,
                    'primary_key': dataset.schema.primary_key
                }
            
            elif dataset.type == "text":
                dataset_data['schema'] = {
                    'path': dataset.schema.path,
                    'format': dataset.schema.format,
                    'encoding': dataset.schema.encoding
                }
            
            config_data['datasets'].append(dataset_data)
        
        # 保存到文件
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"数据注册表已保存到: {yaml_path}")
        
    except Exception as e:
        raise ValueError(f"保存数据注册表失败: {e}")


def append_dataset_to_yaml(dataset: DatasetConfig, yaml_path: str) -> None:
    """
    追加一个数据集配置到指定 YAML 文件的 datasets 末尾。
    若文件不存在则报错，不创建。
    """
    try:
        # 读取现有 YAML（若不存在则报错）
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"目标 YAML 不存在: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "datasets" not in data or not isinstance(data["datasets"], list):
            data["datasets"] = []

        # 将 DatasetConfig 转为 YAML 结构字典（复用 save 的格式）
        ds_entry: Dict[str, Any] = {
            "name": dataset.name,
            "type": dataset.type,
            "description": dataset.description,
            "schema": {}
        }
        if dataset.type == "graph":
            ds_entry["schema"] = {
                "vertex": [asdict(vs) for vs in dataset.schema.vertex],
                "edge": [asdict(es) for es in dataset.schema.edge],
                "graph": asdict(dataset.schema.graph_structure),
                "graph_store_info": asdict(dataset.schema.graph_store_info),
            }
        elif dataset.type == "table":
            ds_entry["schema"] = {
                "path": dataset.schema.path,
                "format": dataset.schema.format,
                "columns": dataset.schema.columns,
                "primary_key": dataset.schema.primary_key,
            }
        elif dataset.type == "text":
            ds_entry["schema"] = {
                "path": dataset.schema.path,
                "format": dataset.schema.format,
                "encoding": dataset.schema.encoding,
            }
        else:
            raise ValueError(f"不支持的数据集类型: {dataset.type}")

        # 追加并写回
        data["datasets"].append(ds_entry)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
    except Exception as e:
        raise ValueError(f"追加数据集失败: {e}")


def read_datasets_map(yaml_path: str) -> Dict[str, DatasetConfig]:
    """
    读取 YAML 文件并返回 {name: DatasetConfig} 的映射。
    """
    cfg = load_data_upload_config(yaml_path)
    print(cfg)
    return {ds.name: ds for ds in cfg.datasets}


def get_dataset_info(dataset: DatasetConfig) -> Dict[str, Any]:
    """
    获取数据集的详细信息
    
    Args:
        dataset: 数据集配置
        
    Returns:
        Dict: 数据集信息字典
    """
    info = {
        'name': dataset.name,
        'type': dataset.type,
        'description': dataset.description
    }
    
    if dataset.type == "graph":
        info.update({
            'vertex_item_count': len(dataset.schema.vertex),
            'edge_item_count': len(dataset.schema.edge),
            'vertex_types': [v.type for v in dataset.schema.vertex],
            'edge_types': [e.type for e in dataset.schema.edge],
            'graph_properties': {
                'directed': dataset.schema.graph_structure.directed,
                'multigraph': dataset.schema.graph_structure.multigraph,
                'weighted': dataset.schema.graph_structure.weighted,
                'heterogeneous': dataset.schema.graph_structure.heterogeneous
            },
            'graph_store_info': asdict(dataset.schema.graph_store_info)
        })
    
    elif dataset.type == "table":
        info.update({
            'table_path': dataset.schema.path,
            'format': dataset.schema.format,
            'columns': dataset.schema.columns,
            'primary_key': dataset.schema.primary_key
        })
    
    elif dataset.type == "text":
        info.update({
            'file_path': dataset.schema.path,
            'format': dataset.schema.format,
            'encoding': dataset.schema.encoding
        })
    
    return info


if __name__ == "__main__":
    def test_data_registry():
        """测试数据注册表加载功能"""
        config_path = "/home/chency/GraphLLM/config/data_upload_config.yaml"
        
        try:
            print("=== 测试数据注册表加载 ===")
            registry = load_data_registry(config_path)
            
            print(f"成功加载数据注册表，包含 {len(registry.datasets)} 个数据集")
            
            # 列出所有数据集
            print("\n=== 数据集列表 ===")
            for dataset in registry.datasets:
                print(f"- {dataset.name} ({dataset.type})")
                print(f"  描述: {dataset.description}")
                
                if dataset.type == "graph":
                    print(f"  顶点文件: {len(dataset.schema.vertex)} 个")
                    print(f"  边文件: {len(dataset.schema.edge)} 个")
                    if dataset.schema.graph_structure:
                        print(
                            f"  图属性: 有向={dataset.schema.graph_structure.directed}, "
                            f"多图={dataset.schema.graph_structure.multigraph}, "
                            f"加权={dataset.schema.graph_structure.weighted}"
                        )
                elif dataset.type == "table":
                    print(f"  表格文件: {dataset.schema.path}")
                elif dataset.type == "text":
                    print(f"  文本文件: {dataset.schema.path}")
                print()
            
            # 测试获取特定数据集
            print("=== 测试获取特定数据集 ===")
            amlsim_dataset = registry.get_dataset("AMLSim1K")
            if amlsim_dataset:
                info = get_dataset_info(amlsim_dataset)
                print(f"数据集信息: {info}")
            else:
                print("未找到 AMLSim1K 数据集")
            
            # 按类型过滤
            print("\n=== 按类型过滤数据集 ===")
            graph_datasets = registry.list_datasets("graph")
            print(f"图数据集数量: {len(graph_datasets)}")
            for dataset in graph_datasets:
                print(f"  - {dataset.name}")
            
        except Exception as e:
            print(f"测试失败: {e}")
    
    test_data_registry()