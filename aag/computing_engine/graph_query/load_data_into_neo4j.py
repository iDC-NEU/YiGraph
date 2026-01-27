import pandas as pd
from neo4j import GraphDatabase
import logging
import sys
import time
from typing import Dict, List, Optional, Any, Union
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jGraphLoader:
    """
    通用的Neo4j图数据加载器，支持根据schema动态加载图数据
    """
    def __init__(self, 
                 uri: str = 'bolt://219.216.65.209:7687',
                 username: str = 'neo4j', 
                 password: str = '12345678',
                 schema: Optional[Dict[str, Any]] = None):
        """
        初始化Neo4j图数据加载器
        
        Args:
            uri: Neo4j连接URI，格式为 bolt://host:port 或 neo4j://host:port
            username: 用户名
            password: 密码
            schema: 图的schema信息，包含vertex和edge的配置
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.schema = schema
        
        self.connect_to_neo4j()
        
    def connect_to_neo4j(self):
        """连接到Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # 验证连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info("成功连接到Neo4j")
            return True
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            if self.driver:
                self.driver.close()
                self.driver = None
            raise Exception(f"无法连接到Neo4j数据库: {e}")
    
    def clear_database(self):
        """清空数据库（可选）"""
        if not self.driver:
            logger.error("数据库连接未建立，无法清空数据库")
            return False
        try:
            with self.driver.session() as session:
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("成功清空数据库")
                return True
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            return False
    
    def _build_node_key(self, row: pd.Series, query_fields: Union[str, List[str]]) -> str:
        """
        根据query_field构建node_key
        
        Args:
            row: DataFrame的一行数据
            query_fields: 单个字段名或字段名列表
            
        Returns:
            构建的node_key字符串
        """
        if isinstance(query_fields, str):
            query_fields = [query_fields]
        
        # 将多个字段值用空格连接
        values = [str(row[field]) for field in query_fields]
        return ' '.join(values)
    
    def _get_attribute_fields(self, df: pd.DataFrame, 
                             attribute_fields: Optional[List[str]], 
                             exclude_fields: Optional[List[str]] = None) -> List[str]:
        """
        获取需要加载的属性字段列表
        
        Args:
            df: DataFrame
            attribute_fields: schema中指定的属性字段列表，如果为空则使用所有列
            exclude_fields: 需要排除的字段列表
            
        Returns:
            属性字段列表
        """
        if attribute_fields and len(attribute_fields) > 0:
            # 如果指定了attribute_fields，使用指定的字段
            return [f for f in attribute_fields if f in df.columns]
        else:
            # 如果没有指定，使用所有列，但排除exclude_fields
            exclude_fields = exclude_fields or []
            return [col for col in df.columns if col not in exclude_fields]
    
    def create_constraints_and_indexes(self, vertex_configs: List[Dict[str, Any]]):
        """
        根据vertex配置创建约束和索引
        
        Args:
            vertex_configs: vertex配置列表
        """
        if not self.driver:
            logger.error("数据库连接未建立，无法创建约束和索引")
            return False
        try:
            with self.driver.session() as session:
                for vertex_config in vertex_configs:
                    vertex_type = vertex_config.get('type', 'Node')
                    # 将type首字母大写作为标签
                    label = vertex_type.capitalize()
                    
                    # 为node_key创建唯一性约束
                    try:
                        session.run(f"""
                            CREATE CONSTRAINT {label.lower()}_key_unique IF NOT EXISTS 
                            FOR (n:{label}) 
                            REQUIRE n.node_key IS UNIQUE
                        """)
                        logger.info(f"成功创建{label}.node_key唯一性约束")
                    except Exception as e:
                        logger.warning(f"创建约束时出现问题（可能已存在）: {e}")
                    
                    # 创建node_key索引
                    try:
                        session.run(f"CREATE INDEX {label.lower()}_key_index IF NOT EXISTS FOR (n:{label}) ON (n.node_key)")
                        logger.info(f"成功创建{label}.node_key索引")
                    except Exception as e:
                        logger.warning(f"创建索引时出现问题（可能已存在）: {e}")
                    
                    # 为id_field创建索引（如果存在）
                    id_field = vertex_config.get('id_field')
                    if id_field:
                        try:
                            session.run(f"CREATE INDEX {label.lower()}_id_index IF NOT EXISTS FOR (n:{label}) ON (n.{id_field})")
                            logger.info(f"成功创建{label}.{id_field}索引")
                        except Exception as e:
                            logger.warning(f"创建索引时出现问题（可能已存在）: {e}")
                
                return True
        except Exception as e:
            logger.error(f"创建约束和索引时发生错误: {e}")
            return False

    def load_vertices(self, vertex_config: Dict[str, Any]):
        """
        根据配置加载顶点数据
        
        Args:
            vertex_config: 顶点配置，包含以下字段：
                - path: CSV文件路径
                - type: 顶点类型
                - query_field: 用于构建node_key的字段（单个或列表）
                - id_field: ID字段
                - label_field: 标签字段（可选）
                - attribute_fields: 属性字段列表（如果为空则使用所有列）
        """
        if not self.driver:
            logger.error("数据库连接未建立，无法加载顶点数据")
            return False
        
        try:
            # 读取配置
            file_path = vertex_config.get('path')
            vertex_type = vertex_config.get('type', 'Node')
            query_field = vertex_config.get('query_field')
            id_field = vertex_config.get('id_field')
            label_field = vertex_config.get('label_field')
            attribute_fields = vertex_config.get('attribute_fields', [])
            
            # 将type首字母大写作为Neo4j标签
            neo4j_label = vertex_type.capitalize()
            
            # 读取数据
            df = pd.read_csv(file_path)
            logger.info(f"读取到 {len(df)} 条{vertex_type}数据")
            
            # 替换NaN值为None
            df = df.where(pd.notna(df), None)
            
            # 构建node_key
            if isinstance(query_field, list):
                df['node_key'] = df.apply(lambda row: self._build_node_key(row, query_field), axis=1)
            else:
                df['node_key'] = df[query_field].astype(str)
            
            # 确定需要加载的属性字段
            # 排除node_key（因为已经单独处理）
            exclude_fields = ['node_key']
            fields_to_load = self._get_attribute_fields(df, attribute_fields, exclude_fields)
            
            # 确保id_field和label_field在加载列表中
            if id_field and id_field not in fields_to_load:
                fields_to_load.append(id_field)
            if label_field and label_field not in fields_to_load:
                fields_to_load.append(label_field)
            
            logger.info(f"将加载以下属性字段: {fields_to_load}")
            
            batch_size = 1000
            total_vertices = len(df)
            success_count = 0
            fail_count = 0
            
            with self.driver.session() as session:
                for i in range(0, total_vertices, batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    # 只选择需要的列
                    columns_to_use = ['node_key'] + fields_to_load
                    batch_subset = batch[columns_to_use]
                    records = batch_subset.to_dict('records')
                    
                    # 动态构建SET子句
                    set_clauses = []
                    for field in fields_to_load:
                        set_clauses.append(f"n.{field} = record.{field}")
                    set_clause = ",\n                        ".join(set_clauses)
                    
                    # 构建Cypher查询
                    query = f"""
                    UNWIND $records AS record
                    MERGE (n:{neo4j_label} {{node_key: record.node_key}})
                    ON CREATE SET
                        {set_clause}
                    ON MATCH SET
                        {set_clause}
                    """
                    
                    try:
                        result = session.run(query, records=records)
                        summary = result.consume()
                        success_count += len(records)
                        logger.info(f"已处理 {min(i+batch_size, total_vertices)}/{total_vertices} 条{vertex_type}数据")
                    except Exception as e:
                        logger.error(f"批量插入{vertex_type}数据失败: {e}")
                        fail_count += len(records)
            
            logger.info(f"{vertex_type}数据加载完成，成功处理 {success_count} 条，失败 {fail_count} 条")
            return True
            
        except Exception as e:
            logger.error(f"加载{vertex_type}数据时发生错误: {e}")
            return False

    # add gjq: 修改边加载逻辑，通过id_field查找对应的node_key来匹配节点
    def load_edges(self, edge_config: Dict[str, Any], vertex_configs: List[Dict[str, Any]]):
        """
        根据配置加载边数据
        
        Args:
            edge_config: 边配置，包含以下字段：
                - path: CSV文件路径
                - type: 边类型
                - source_field: 源节点字段（对应vertex的id_field）
                - target_field: 目标节点字段（对应vertex的id_field）
                - label_field: 标签字段（可选）
                - weight_field: 权重字段（可选）
                - attribute_fields: 属性字段列表（如果为空则使用所有列）
            vertex_configs: 顶点配置列表，用于确定如何匹配节点
        """
        if not self.driver:
            logger.error("数据库连接未建立，无法加载边数据")
            return False
        
        try:
            # 读取配置
            file_path = edge_config.get('path')
            edge_type = edge_config.get('type', 'EDGE')
            source_field = edge_config.get('source_field')
            target_field = edge_config.get('target_field')
            label_field = edge_config.get('label_field')
            weight_field = edge_config.get('weight_field')
            attribute_fields = edge_config.get('attribute_fields', [])
            
            # 将type转换为大写作为Neo4j关系类型
            neo4j_rel_type = edge_type.upper()
            
            # add gjq: 获取顶点配置信息
            vertex_config = vertex_configs[0] if vertex_configs else {}
            vertex_label = vertex_config.get('type', 'Node').capitalize()
            vertex_id_field = vertex_config.get('id_field', 'id')
            vertex_query_field = vertex_config.get('query_field')
            vertex_path = vertex_config.get('path')
            
            # add gjq: 读取顶点数据，建立id_field到node_key的映射
            logger.info(f"正在读取顶点数据以建立ID到node_key的映射...")
            vertex_df = pd.read_csv(vertex_path)
            vertex_df = vertex_df.where(pd.notna(vertex_df), None)
            
            # add gjq: 构建node_key列
            if isinstance(vertex_query_field, list):
                vertex_df['node_key'] = vertex_df.apply(lambda row: self._build_node_key(row, vertex_query_field), axis=1)
            else:
                vertex_df['node_key'] = vertex_df[vertex_query_field].astype(str)
            
            # add gjq: 创建id到node_key的映射字典
            id_to_node_key = dict(zip(vertex_df[vertex_id_field], vertex_df['node_key']))
            logger.info(f"成功建立 {len(id_to_node_key)} 个ID到node_key的映射")
            
            # 读取边数据
            df = pd.read_csv(file_path)
            logger.info(f"读取到 {len(df)} 条{edge_type}数据")
            
            # 替换NaN值为None
            df = df.where(pd.notna(df), None)
            
            # add gjq: 将source_field和target_field的值转换为对应的node_key
            df['source_node_key'] = df[source_field].map(id_to_node_key)
            df['target_node_key'] = df[target_field].map(id_to_node_key)
            
            # add gjq: 检查是否有无法映射的节点
            missing_source = df['source_node_key'].isna().sum()
            missing_target = df['target_node_key'].isna().sum()
            if missing_source > 0 or missing_target > 0:
                logger.warning(f"发现 {missing_source} 个源节点和 {missing_target} 个目标节点无法映射到node_key，这些边将被跳过")
                # 过滤掉无法映射的边
                df = df.dropna(subset=['source_node_key', 'target_node_key'])
                logger.info(f"过滤后剩余 {len(df)} 条有效边数据")
            
            # 确定需要加载的属性字段
            # 排除source_field、target_field、source_node_key、target_node_key
            exclude_fields = [source_field, target_field, 'source_node_key', 'target_node_key']
            fields_to_load = self._get_attribute_fields(df, attribute_fields, exclude_fields)
            
            # 确保label_field和weight_field在加载列表中
            if label_field and label_field not in fields_to_load:
                fields_to_load.append(label_field)
            if weight_field and weight_field not in fields_to_load:
                fields_to_load.append(weight_field)
            
            logger.info(f"将加载以下边属性字段: {fields_to_load}")
            
            batch_size = 1000
            total_edges = len(df)
            success_count = 0
            fail_count = 0
            
            with self.driver.session() as session:
                for i in range(0, total_edges, batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    # add gjq: 选择需要的列，包括source_node_key和target_node_key
                    columns_to_use = ['source_node_key', 'target_node_key'] + fields_to_load
                    batch_subset = batch[columns_to_use]
                    records = batch_subset.to_dict('records')
                    
                    # 动态构建属性设置子句
                    if fields_to_load:
                        prop_clauses = []
                        for field in fields_to_load:
                            prop_clauses.append(f"{field}: record.{field}")
                        properties = "{" + ", ".join(prop_clauses) + "}"
                    else:
                        properties = ""
                    
                    # add gjq: 构建Cypher查询 - 通过node_key匹配节点
                    query = f"""
                    UNWIND $records AS record
                    MATCH (source:{vertex_label} {{node_key: record.source_node_key}})
                    MATCH (target:{vertex_label} {{node_key: record.target_node_key}})
                    CREATE (source)-[r:{neo4j_rel_type} {properties}]->(target)
                    """
                    
                    try:
                        result = session.run(query, records=records)
                        summary = result.consume()
                        batch_success = summary.counters.relationships_created
                        success_count += batch_success
                        logger.info(f"已处理 {min(i+batch_size, total_edges)}/{total_edges} 条{edge_type}数据，成功创建 {batch_success} 条关系")
                    except Exception as e:
                        logger.error(f"批量插入{edge_type}数据失败: {e}")
                        fail_count += len(records)
            
            logger.info(f"{edge_type}数据加载完成，成功插入 {success_count} 条关系，失败 {fail_count} 条")
            return True
            
        except Exception as e:
            logger.error(f"加载{edge_type}数据时发生错误: {e}")
            return False

    def verify_data(self, vertex_configs: List[Dict[str, Any]], edge_configs: List[Dict[str, Any]]):
        """验证数据加载结果"""
        if not self.driver:
            logger.error("数据库连接未建立，无法验证数据")
            return False
        try:
            with self.driver.session() as session:
                # 检查每种类型的节点数量
                for vertex_config in vertex_configs:
                    vertex_type = vertex_config.get('type', 'Node').capitalize()
                    result = session.run(f"MATCH (n:{vertex_type}) RETURN count(n) as count")
                    count = result.single()["count"]
                    logger.info(f"{vertex_type}节点数量: {count}")
                
                # 检查每种类型的关系数量
                for edge_config in edge_configs:
                    edge_type = edge_config.get('type', 'EDGE').upper()
                    result = session.run(f"MATCH ()-[r:{edge_type}]->() RETURN count(r) as count")
                    count = result.single()["count"]
                    logger.info(f"{edge_type}关系数量: {count}")
                
                # 显示示例节点
                if vertex_configs:
                    vertex_type = vertex_configs[0].get('type', 'Node').capitalize()
                    query_field = vertex_configs[0].get('query_field')
                    if isinstance(query_field, list):
                        field_names = ', '.join([f"n.{f}" for f in query_field])
                    else:
                        field_names = f"n.{query_field}"
                    
                    result = session.run(f"MATCH (n:{vertex_type}) RETURN n.node_key, {field_names} LIMIT 5")
                    logger.info(f"示例{vertex_type}节点的node_key:")
                    for record in result:
                        logger.info(f"  {dict(record)}")
                
                # 显示示例关系
                if edge_configs and vertex_configs:
                    edge_type = edge_configs[0].get('type', 'EDGE').upper()
                    vertex_type = vertex_configs[0].get('type', 'Node').capitalize()
                    result = session.run(f"""
                        MATCH (source:{vertex_type})-[r:{edge_type}]->(target:{vertex_type}) 
                        RETURN source.node_key as source_key, target.node_key as target_key
                        LIMIT 5
                    """)
                    logger.info(f"示例{edge_type}关系:")
                    for record in result:
                        logger.info(f"  {record['source_key']} -> {record['target_key']}")
            
            return True
            
        except Exception as e:
            logger.error(f"验证数据时发生错误: {e}")
            return False

    def close_connection(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("已关闭Neo4j连接")
    
    def load_all_data(self, clear_existing: bool = False):
        """
        加载所有数据的主流程
        
        Args:
            clear_existing: 是否清空现有数据（默认False）
        """
        if not self.schema:
            logger.error("未提供schema信息，无法加载数据")
            return False
        
        try:
            # 1. 可选：清空现有数据
            if clear_existing:
                if not self.clear_database():
                    logger.warning("清空数据库失败，继续执行...")
            
            # 获取vertex和edge配置
            vertex_configs = self.schema.get('vertex', [])
            edge_configs = self.schema.get('edge', [])
            
            if not vertex_configs:
                logger.error("schema中没有vertex配置")
                return False
            
            # 2. 创建约束和索引
            if not self.create_constraints_and_indexes(vertex_configs):
                logger.warning("创建约束和索引失败，继续执行...")
            
            # 3. 加载所有顶点
            for vertex_config in vertex_configs:
                if not self.load_vertices(vertex_config):
                    logger.error(f"加载顶点类型 {vertex_config.get('type')} 失败")
                    return False
            
            # 4. 加载所有边
            for edge_config in edge_configs:
                if not self.load_edges(edge_config, vertex_configs):
                    logger.error(f"加载边类型 {edge_config.get('type')} 失败")
                    return False
            
            # 5. 验证数据
            if not self.verify_data(vertex_configs, edge_configs):
                return False
            
            logger.info("所有数据加载完成！")
            return True
            
        except Exception as e:
            logger.error(f"加载数据时发生错误: {e}")
            return False
        finally:
            self.close_connection()

def load_from_yaml(yaml_path: str, 
                   uri: str = 'bolt://localhost:7687',
                   username: str = 'neo4j',
                   password: str = 'password',
                   clear_existing: bool = False) -> bool:
    """
    从YAML文件加载schema并导入数据到Neo4j
    
    Args:
        yaml_path: YAML schema文件路径
        uri: Neo4j连接URI
        username: Neo4j用户名
        password: Neo4j密码
        clear_existing: 是否清空现有数据
        
    Returns:
        是否成功
    """
    try:
        # 读取YAML文件
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 获取第一个数据集的schema
        datasets = config.get('datasets', [])
        if not datasets:
            logger.error("YAML文件中没有找到datasets")
            return False
        
        dataset = datasets[0]
        schema = dataset.get('schema')
        
        if not schema:
            logger.error("数据集中没有找到schema")
            return False
        
        logger.info(f"正在加载数据集: {dataset.get('name')}")
        
        # 创建加载器并加载数据
        loader = Neo4jGraphLoader(
            uri=uri,
            username=username,
            password=password,
            schema=schema
        )
        
        return loader.load_all_data(clear_existing=clear_existing)
        
    except Exception as e:
        logger.error(f"从YAML加载数据失败: {e}")
        return False

def main():
    """主函数 - 示例用法"""
    # 方式1: 直接使用schema字典
    schema = {
        'vertex': [
            {
                'path': '/home/gaojq/AAG/aag/datasets/graphs/transaction_amlsim/1K/accounts.csv',
                'type': 'account',
                'query_field': ['last_name', 'first_name'],  # 使用多个字段构建node_key
                'id_field': 'acct_id',
                'label_field': 'prior_sar_count',
                'attribute_fields': []  # 空列表表示加载所有列
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
                'attribute_fields': []  # 空列表表示加载所有列
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
        logger.info("数据加载成功完成！")
        sys.exit(0)
    else:
        logger.error("数据加载失败！")
        sys.exit(1)
    
    # 方式2: 从YAML文件加载
    # success = load_from_yaml(
    #     yaml_path='AAG_3/AAG/aag/datasets/dataset_schemas/AMLSim1K/graph_schemas.yaml',
    #     uri='bolt://202.199.13.67:7687',
    #     username='neo4j',
    #     password='12345678',
    #     clear_existing=True
    # )

if __name__ == "__main__":
    main()