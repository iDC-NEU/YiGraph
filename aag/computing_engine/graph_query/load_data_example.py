"""
示例：如何使用Neo4jGraphLoader从schema加载数据到Neo4j

这个脚本展示了多种使用方式：
1. 直接传入schema字典
2. 从YAML文件读取schema
3. 从DatasetManager获取schema（模拟scheduler中的使用）
4. 指定特定的属性字段
5. 使用单个字段作为query_field

注意：在scheduler.py中，当调用specific_analysis_dataset()时，
会自动调用_load_graph_to_neo4j()将图数据加载到Neo4j。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from aag.computing_engine.graph_query.load_data_into_neo4j import Neo4jGraphLoader, load_from_yaml
from aag.data_pipeline.data_transformer.dataset_manager import DatasetManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example1_direct_schema():
    """示例1: 直接使用schema字典"""
    logger.info("=" * 60)
    logger.info("示例1: 直接使用schema字典")
    logger.info("=" * 60)
    
    # 定义schema
    schema = {
        'vertex': [
            {
                'path': '/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/accounts.csv',
                'type': 'account',
                'query_field': ['last_name', 'first_name'],  # 多字段组合作为node_key
                'id_field': 'acct_id',
                'label_field': 'prior_sar_count',
                'attribute_fields': []  # 空列表 = 加载所有列
            }
        ],
        'edge': [
            {
                'path': '/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/transactions.csv',
                'type': 'transfer',
                'source_field': 'orig_acct',
                'target_field': 'bene_acct',
                'label_field': 'is_sar',
                'weight_field': None,
                'attribute_fields': []  # 空列表 = 加载所有列
            }
        ]
    }
    
    # 创建加载器
    loader = Neo4jGraphLoader(
        uri='bolt://202.199.13.67:7687',
        username='neo4j',
        password='12345678',
        schema=schema
    )
    
    # 加载数据
    success = loader.load_all_data(clear_existing=True)
    
    if success:
        logger.info("✅ 示例1执行成功！")
    else:
        logger.error("❌ 示例1执行失败！")
    
    return success

def example2_from_yaml():
    """示例2: 从YAML文件加载schema"""
    logger.info("=" * 60)
    logger.info("示例2: 从YAML文件加载schema")
    logger.info("=" * 60)
    
    yaml_path = 'AAG_3/AAG/aag/datasets/dataset_schemas/AMLSim1K/graph_schemas.yaml'
    
    success = load_from_yaml(
        yaml_path=yaml_path,
        uri='bolt://202.199.13.67:7687',
        username='neo4j',
        password='12345678',
        clear_existing=True
    )
    
    if success:
        logger.info("✅ 示例2执行成功！")
    else:
        logger.error("❌ 示例2执行失败！")
    
    return success

def example3_from_dataset_manager():
    """示例3: 从DatasetManager获取schema并加载"""
    logger.info("=" * 60)
    logger.info("示例3: 从DatasetManager获取schema并加载")
    logger.info("=" * 60)
    
    # 初始化DatasetManager
    dataset_manager = DatasetManager()
    
    # 获取数据集信息
    dataset_name = "AMLSim1K"
    dataset_configs = dataset_manager.get_dataset_info(dataset_name, dtype="graph")
    
    if not dataset_configs:
        logger.error(f"未找到数据集: {dataset_name}")
        return False
    
    # 获取第一个配置（对于图数据集通常只有一个）
    dataset_config = dataset_configs[0]
    
    # 从DatasetConfig中提取schema
    schema = dataset_config.schema
    
    logger.info(f"从DatasetManager获取到schema: {schema}")
    
    # 创建加载器
    loader = Neo4jGraphLoader(
        uri='bolt://202.199.13.67:7687',
        username='neo4j',
        password='12345678',
        schema=schema.__dict__ if hasattr(schema, '__dict__') else schema
    )
    
    # 加载数据
    success = loader.load_all_data(clear_existing=True)
    
    if success:
        logger.info("✅ 示例3执行成功！")
    else:
        logger.error("❌ 示例3执行失败！")
    
    return success

def example4_custom_attributes():
    """示例4: 指定特定的属性字段"""
    logger.info("=" * 60)
    logger.info("示例4: 指定特定的属性字段")
    logger.info("=" * 60)
    
    # 只加载指定的属性字段
    schema = {
        'vertex': [
            {
                'path': '/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/accounts.csv',
                'type': 'account',
                'query_field': ['last_name', 'first_name'],
                'id_field': 'acct_id',
                'label_field': 'prior_sar_count',
                'attribute_fields': [  # 只加载这些字段
                    'acct_id', 'first_name', 'last_name', 
                    'prior_sar_count', 'type', 'initial_deposit'
                ]
            }
        ],
        'edge': [
            {
                'path': '/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/transactions.csv',
                'type': 'transfer',
                'source_field': 'orig_acct',
                'target_field': 'bene_acct',
                'label_field': 'is_sar',
                'weight_field': 'base_amt',
                'attribute_fields': [  # 只加载这些字段
                    'tran_id', 'base_amt', 'is_sar', 'tran_timestamp'
                ]
            }
        ]
    }
    
    loader = Neo4jGraphLoader(
        uri='bolt://202.199.13.67:7687',
        username='neo4j',
        password='12345678',
        schema=schema
    )
    
    success = loader.load_all_data(clear_existing=True)
    
    if success:
        logger.info("✅ 示例4执行成功！")
    else:
        logger.error("❌ 示例4执行失败！")
    
    return success

def example5_single_query_field():
    """示例5: 使用单个字段作为query_field"""
    logger.info("=" * 60)
    logger.info("示例5: 使用单个字段作为query_field")
    logger.info("=" * 60)
    
    schema = {
        'vertex': [
            {
                'path': '/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/accounts.csv',
                'type': 'account',
                'query_field': 'acct_id',  # 单个字段
                'id_field': 'acct_id',
                'label_field': 'prior_sar_count',
                'attribute_fields': []
            }
        ],
        'edge': [
            {
                'path': '/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/transactions.csv',
                'type': 'transfer',
                'source_field': 'orig_acct',
                'target_field': 'bene_acct',
                'label_field': 'is_sar',
                'weight_field': None,
                'attribute_fields': []
            }
        ]
    }
    
    loader = Neo4jGraphLoader(
        uri='bolt://202.199.13.67:7687',
        username='neo4j',
        password='12345678',
        schema=schema
    )
    
    success = loader.load_all_data(clear_existing=True)
    
    if success:
        logger.info("✅ 示例5执行成功！")
    else:
        logger.error("❌ 示例5执行失败！")
    
    return success

if __name__ == "__main__":
    # 选择要运行的示例
    print("\n请选择要运行的示例:")
    print("1. 直接使用schema字典")
    print("2. 从YAML文件加载schema")
    print("3. 从DatasetManager获取schema")
    print("4. 指定特定的属性字段")
    print("5. 使用单个字段作为query_field")
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    if choice == "1":
        example1_direct_schema()
    elif choice == "2":
        example2_from_yaml()
    elif choice == "3":
        example3_from_dataset_manager()
    elif choice == "4":
        example4_custom_attributes()
    elif choice == "5":
        example5_single_query_field()
    else:
        print("无效的选项！")
        sys.exit(1)