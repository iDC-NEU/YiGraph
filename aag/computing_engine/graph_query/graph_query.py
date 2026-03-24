# /home/gaojq/AAG_duan/AAG/aag/computing_engine/graph_query/graph_query.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, DatabaseError, TransientError
import re
import time

JsonDict = Dict[str, Any]

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: Optional[str] = None


class Neo4jGraphClient:
    """
    Neo4j 3.5.25 兼容版查询模板库
    支持的查询类型：
    1. ID/唯一键查节点 - get_node_by_unique_key()
    2. 邻居查询 / n跳关系 - neighbors_n_hop()
    3. 公共邻居 - common_neighbors()
    4. 条件过滤查询 - filter_query()
    5. 子图抽取 - subgraph_extract()
    6. 聚合统计 - aggregate_stats()
    7. 两点之间的路径查询 - paths_between()
    """

    # Neo4j 保留字（部分）
    RESERVED_KEYWORDS = {
        'MATCH', 'RETURN', 'WHERE', 'CREATE', 'DELETE', 'SET', 
        'MERGE', 'WITH', 'UNWIND', 'CASE', 'WHEN', 'THEN', 
        'ELSE', 'END', 'ORDER', 'BY', 'SKIP', 'LIMIT', 'AS',
        'AND', 'OR', 'NOT', 'IN', 'IS', 'NULL', 'TRUE', 'FALSE'
    }

    def __init__(self, config: Neo4jConfig):
        """
        初始化 Neo4j 客户端
        
        Args:
            config: Neo4j 连接配置
        """
        self._driver = GraphDatabase.driver(
            config.uri, 
            auth=(config.user, config.password),
            max_connection_lifetime=3600,  # 连接最大存活时间 1小时
            max_connection_pool_size=50,    # 连接池大小
            connection_acquisition_timeout=60.0  # 获取连接超时
        )
        self._db = config.database

    def close(self) -> None:
        """关闭数据库连接"""
        if self._driver:
            self._driver.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()

    # ===== 核心执行器（带重试机制）=====
    def run(
        self,
        cypher: str,
        params: Optional[JsonDict] = None,
        *,
        read: bool = True,
        max_retries: int = 3,
        show_query: bool = True
    ) -> List[JsonDict]:
        """
        执行 Cypher 查询（带自动重试）
        
        Args:
            cypher: Cypher 查询语句
            params: 查询参数
            read: 是否为读操作（True=读，False=写）
            max_retries: 最大重试次数
            show_query: 是否打印填充参数后的查询
            
        Returns:
            查询结果列表
            
        Raises:
            RuntimeError: 查询失败
        """
        params = params or {}
        
        # 可选：打印填充参数后的查询（用于调试）
        if show_query:
            self._print_filled_query(cypher, params)
        
        # 重试逻辑
        for attempt in range(max_retries):
            try:
                with self._driver.session(database=self._db) as session:
                    if read:
                        return session.read_transaction(
                            lambda tx: [r.data() for r in tx.run(cypher, params)]
                        )
                    else:
                        return session.write_transaction(
                            lambda tx: [r.data() for r in tx.run(cypher, params)]
                        )
            
            except TransientError as e:
                # 临时性错误：重试
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Neo4j TransientError after {max_retries} retries: {e}\n"
                        f"Cypher: {cypher}\nParams: {params}"
                    ) from e
                
                wait_time = 2 ** attempt  # 指数退避
                print(f"⚠️  TransientError, retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            
            except (ClientError, DatabaseError) as e:
                # 客户端错误或数据库错误：不重试
                raise RuntimeError(
                    f"Neo4j Error: {e}\n"
                    f"Cypher: {cypher}\n"
                    f"Params: {params}"
                ) from e

    def _print_filled_query(self, cypher: str, params: Dict) -> None:
        """
        打印填充参数后的查询（用于调试）
        
        注意：这个方法只用于显示，实际执行时 Neo4j 驱动会正确处理参数
        """
        print("\n" + "="*80)
        print("📝 执行的 Cypher 查询语句（参数已填充）:")
        print("-"*80)
        
        filled_cypher = cypher
        if params:
            import json
            # 按参数名长度倒序排序，避免子串替换问题
            sorted_params = sorted(params.items(), key=lambda x: len(x[0]), reverse=True)
            
            for key, value in sorted_params:
                # ⚠️ CRITICAL: 跳过元组类型的参数（这些是 DSL 内部格式，不应该出现在最终 Cypher 中）
                # 元组格式如 (">", 500000) 应该已经在构建 WHERE 子句时被转换为 Cypher 操作符
                if isinstance(value, tuple):
                    # 这种情况不应该发生，如果发生了说明有 bug
                    print(f"⚠️ WARNING: 参数 ${key} 是元组格式 {value}，这不应该出现在最终查询中！")
                    continue
                
                # 根据值类型格式化
                if isinstance(value, str):
                    # 转义单引号
                    formatted_value = f"'{value.replace(chr(39), chr(39)+chr(39))}'"
                elif isinstance(value, (int, float)):
                    formatted_value = str(value)
                elif isinstance(value, bool):
                    # ⚠️ 修复1: 布尔值必须是小写 true/false（Neo4j 3.5.25 要求）
                    formatted_value = "true" if value else "false"
                elif value is None:
                    formatted_value = "null"
                elif isinstance(value, list):
                    # 列表格式（用于 IN 操作符）
                    formatted_value = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, dict):
                    # 字典格式（很少用）
                    formatted_value = json.dumps(value, ensure_ascii=False)
                else:
                    formatted_value = str(value)
                
                # 使用正则确保完整匹配（避免 $id 替换 $id2 的问题）
                filled_cypher = re.sub(
                    r'\$' + re.escape(key) + r'\b',  # \b 确保单词边界
                    formatted_value,
                    filled_cypher
                )
        
        print(filled_cypher)
        print("="*80 + "\n")

    # ===== Schema 获取 =====
    def get_schema(self) -> Dict:
        """
        获取图数据库 Schema 信息（增强版）
        
        Returns:
            {
                "node_labels": {
                    label: {
                        "properties": [prop_names],
                        "sample_values": {prop: sample_value}
                    }
                },
                "relationship_types": {
                    rel_type: {
                        "properties": [prop_names],
                        "sample_values": {prop: sample_value}
                    }
                },
                "patterns": [pattern_strings]
            }
        """
        schema = {
            "node_labels": {},
            "relationship_types": {},
            "patterns": []
        }
        
        # 1. 获取所有节点标签及其属性（包含示例值）
        labels_result = self.run("CALL db.labels() YIELD label RETURN label", show_query=False)
        valid_labels = [item["label"] for item in labels_result]
        
        for label in valid_labels:
            if not label or not self._is_valid_identifier(label):
                continue
            
            # 获取该标签的属性和示例值
            props_result = self.run(f"""
                MATCH (n:`{label}`)
                WITH n LIMIT 1
                UNWIND keys(n) AS key
                RETURN key, n[key] AS sample_value
                ORDER BY key
            """, show_query=False)
            
            if props_result:
                schema["node_labels"][label] = {
                    "properties": [p["key"] for p in props_result],
                    "sample_values": {p["key"]: p["sample_value"] for p in props_result}
                }
        
        # 2. 获取所有关系类型及其属性（包含示例值）
        rels_result = self.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
            show_query=False
        )
        valid_rels = [item["relationshipType"] for item in rels_result]
        
        for rel_type in valid_rels:
            if not rel_type or not self._is_valid_identifier(rel_type):
                continue
            
            # 获取该关系的属性和示例值
            props_result = self.run(f"""
                MATCH ()-[r:`{rel_type}`]->()
                WITH r LIMIT 1
                UNWIND keys(r) AS key
                RETURN key, r[key] AS sample_value
                ORDER BY key
            """, show_query=False)
            
            if props_result:
                schema["relationship_types"][rel_type] = {
                    "properties": [p["key"] for p in props_result],
                    "sample_values": {p["key"]: p["sample_value"] for p in props_result}
                }
        
        # 3. 获取关系模式
        patterns = self.run("""
            MATCH (a)-[r]->(b)
            WITH labels(a)[0] AS start_label,
                 type(r) AS rel_type,
                 labels(b)[0] AS end_label
            WHERE start_label IS NOT NULL
              AND rel_type IS NOT NULL
              AND end_label IS NOT NULL
            RETURN DISTINCT start_label, rel_type, end_label
            LIMIT 100
        """, show_query=False)
        
        schema["patterns"] = [
            f"({p['start_label']})-[:{p['rel_type']}]->({p['end_label']})"
            for p in patterns
        ]
        
        return schema

    # ===== 验证方法 =====
    @staticmethod
    def _is_valid_identifier(name: str) -> bool:
        """快速检查标识符是否有效（字母数字下划线）"""
        return bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name))

    @classmethod
    def _sanitize_label(cls, label: Optional[str]) -> str:
        """
        严格验证节点标签
        
        规则：
        - 必须以字母开头
        - 只能包含字母、数字、下划线
        - 长度不超过 255
        - 不能是保留字
        """
        if not label:
            return ""
        
        # 长度检查
        if len(label) > 255:
            raise ValueError(f"Label too long (max 255): {label}")
        
        # 格式检查
        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', label):
            raise ValueError(f"Invalid label format (must start with letter, contain only alphanumeric and underscore): {label}")
        
        # 保留字检查
        if label.upper() in cls.RESERVED_KEYWORDS:
            raise ValueError(f"Reserved keyword cannot be used as label: {label}")
        
        return label

    @classmethod
    def _sanitize_rel_type(cls, rel_type: Optional[str]) -> str:
        """
        严格验证关系类型
        
        规则：
        - 通常使用大写字母和下划线（如 FOLLOWS、HAS_FRIEND）
        - 也允许小写和驼峰（兼容性）
        - 长度不超过 255
        """
        if not rel_type:
            return ""
        
        if len(rel_type) > 255:
            raise ValueError(f"Relationship type too long (max 255): {rel_type}")
        
        # 格式检查（允许大小写字母、数字、下划线）
        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', rel_type):
            raise ValueError(f"Invalid relationship type format: {rel_type}")
        
        if rel_type.upper() in cls.RESERVED_KEYWORDS:
            raise ValueError(f"Reserved keyword cannot be used as relationship type: {rel_type}")
        
        return rel_type

    @staticmethod
    def _sanitize_property_key(key: str) -> str:
        """
        验证属性键
        
        规则：
        - 必须以字母开头
        - 只能包含字母、数字、下划线
        - 长度不超过 255
        """
        if not key:
            raise ValueError("Property key cannot be empty")
        
        if len(key) > 255:
            raise ValueError(f"Property key too long (max 255): {key}")
        
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', key):
            raise ValueError(f"Invalid property key format: {key}")
        
        return key

    # =========================================================
    # 1. 根据 ID / 唯一键查节点
    # =========================================================
    def get_node_by_internal_id(
        self,
        internal_id: int,
        *,
        return_props: bool = True
    ) -> Optional[JsonDict]:
        """
        使用 Neo4j 内部 id(n) 查询
        
        注意：内部 id 在导入/删除后可能变化，不建议作为业务主键
        
        Args:
            internal_id: Neo4j 内部节点 ID
            return_props: 是否只返回节点属性
            
        Returns:
            节点数据或 None
        """
        cypher = "MATCH (n) WHERE id(n) = $id RETURN n AS node"
        res = self.run(cypher, {"id": internal_id})
        
        if not res:
            return None
        return res[0]["node"] if return_props else res[0]

    def get_node_by_unique_key(
        self,
        label: str,
        key: str,
        value: Any,
        *,
        return_fields: Optional[List[str]] = None
    ) -> Optional[JsonDict]:
        """
        根据 label + 唯一键属性查询节点
        
        示例：
            # 返回整个节点
            get_node_by_unique_key("User", "userId", "u123")
            
            # 只返回指定字段
            get_node_by_unique_key("Account", "node_key", "Collins Steven",
                                  return_fields=["acct_id", "acct_stat", "acct_open_date"])
            
        Args:
            label: 节点标签
            key: 属性键（如 userId）
            value: 属性值（如 u123）
            return_fields: 要返回的字段列表（None=返回整个节点）
            
        Returns:
            节点数据或 None
        """
        label = self._sanitize_label(label)
        key = self._sanitize_property_key(key)
        
        # 构建RETURN子句
        if return_fields:
            # 返回指定字段
            return_parts = []
            for field in return_fields:
                field = self._sanitize_property_key(field)
                return_parts.append(f"n.`{field}` AS {field}")
            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            # 返回整个节点
            return_clause = "RETURN n AS node"
        
        cypher = f"MATCH (n:`{label}` {{`{key}`: $value}}) {return_clause} LIMIT 1"
        res = self.run(cypher, {"value": value})
        
        if not res:
            return None
        
        # 如果返回指定字段，直接返回结果字典；否则返回node
        if return_fields:
            return res[0]
        else:
            return res[0]["node"]

    # add gjq
    def filter_nodes_by_properties(
        self,
        label: str,
        conditions: Dict[str, Any],
        *,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order_direction: str = "ASC",
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        根据属性条件筛选多个节点（支持多条件AND组合）
        
        功能：
        - 支持多个属性条件的AND组合筛选
        - 支持指定返回字段（而非返回整个节点）
        - 支持排序和限制结果数量
        
        应用场景：
        1. 筛选查询：查找满足特定条件的所有节点
        2. 条件过滤：根据多个属性值组合筛选
        3. 字段投影：只返回需要的字段，减少数据传输
        
        示例：
            # 查询居住在US、州为VT的客户姓名和城市
            results = client.filter_nodes_by_properties(
                "Account",
                {"country": "US", "state": "VT"},
                return_fields=["last_name", "first_name", "city"]
            )
            
            # 查询账户币种是USD且账户状态为A的客户
            results = client.filter_nodes_by_properties(
                "Account",
                {"acct_rptng_crncy": "USD", "acct_stat": "A"},
                return_fields=["last_name", "first_name", "acct_id"],
                order_by="last_name",
                limit=10
            )
            
            # ⚠️ 范围条件查询：initial_deposit 大于 500000 的账户
            results = client.filter_nodes_by_properties(
                "Account",
                {"initial_deposit": (">", 500000)},
                return_fields=["acct_id", "initial_deposit"]
            )
            
            # 布尔值查询：prior_sar_count 为 true 的账户
            results = client.filter_nodes_by_properties(
                "Account",
                {"prior_sar_count": true},  # 注意：Python 中是 True，但会自动转换为 Neo4j 的 true
                return_fields=["acct_id"]
            )
        
        Args:
            label: 节点标签
            conditions: 属性条件字典，支持两种格式：
                      1. 简单等值：{"property": value}
                      2. 范围条件：{"property": (operator, value)}
                         operator 可以是: "=", ">", "<", ">=", "<=", "!=", "IN", "CONTAINS", "STARTS WITH"
                      示例：{"age": (">", 18), "country": "US"}
            return_fields: 要返回的字段列表（None=返回整个节点）
            order_by: 排序字段（如 "last_name"）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回数量（None=不限制）
            
        Returns:
            [
                {"last_name": "Smith", "first_name": "John", "city": "Burlington"},
                {"last_name": "Doe", "first_name": "Jane", "city": "Montpelier"},
                ...
            ]
            
        注意：
            - 这是多节点筛选查询，不是单节点精确查找
            - 所有条件使用AND逻辑组合
            - 如果需要OR逻辑，请使用filter_query方法
        """
        label = self._sanitize_label(label)
        
        if not conditions:
            raise ValueError("conditions cannot be empty")
        
        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")
        
        # 构建WHERE子句
        where_parts = []
        params = {}
        for i, (key, value) in enumerate(conditions.items()):
            key = self._sanitize_property_key(key)
            param_name = f"cond_{i}"
            
            # ⚠️ 修复2: 支持范围条件 (operator, value) 元组格式或 [operator, value] 数组格式
            if (isinstance(value, (tuple, list)) and len(value) == 2):
                operator, actual_value = value
                # 验证操作符
                valid_operators = ["=", ">", "<", ">=", "<=", "!=", "IN", "CONTAINS", "STARTS WITH"]
                if operator.upper() in ["IN", "CONTAINS"]:
                    where_parts.append(f"a.`{key}` {operator.upper()} ${param_name}")
                elif operator.upper() == "STARTS WITH":
                    where_parts.append(f"a.`{key}` STARTS WITH ${param_name}")
                elif operator in valid_operators:
                    where_parts.append(f"a.`{key}` {operator} ${param_name}")
                else:
                    raise ValueError(f"Invalid operator: {operator}")
                params[param_name] = actual_value
            else:
                # 简单等值条件
                where_parts.append(f"a.`{key}` = ${param_name}")
                params[param_name] = value
        
        where_clause = "WHERE " + " AND ".join(where_parts)
        
        # 构建RETURN子句
        if return_fields:
            # 返回指定字段
            return_parts = []
            for field in return_fields:
                field = self._sanitize_property_key(field)
                return_parts.append(f"a.`{field}` AS {field}")
            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            # 返回整个节点
            return_clause = "RETURN a AS node"
        
        # 可选的ORDER BY子句
        order_clause = ""
        if order_by:
            order_by = self._sanitize_property_key(order_by)
            order_clause = f"ORDER BY a.`{order_by}` {order_direction}"
        
        # 可选的LIMIT子句
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT $limit"
            params["limit"] = limit
        
        # 构建完整查询
        cypher = f"""
        MATCH (a:`{label}`)
        {where_clause}
        {return_clause}
        {order_clause}
        {limit_clause}
        """
        
        return self.run(cypher, params)

    # add gjq
    def filter_relationships(
        self,
        rel_type: str,
        *,
        start_label: Optional[str] = None,
        end_label: Optional[str] = None,
        rel_conditions: Optional[Dict[str, Any]] = None,
        return_fields: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        aggregate_field: Optional[str] = None,
        order_by: Optional[str] = None,
        order_direction: str = "ASC",
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        基于关系属性条件筛选关系（支持聚合统计）
        
        功能：
        - 根据关系属性条件筛选关系
        - 支持指定起点和终点节点类型
        - 支持返回指定字段（关系属性和节点属性）
        - 支持聚合统计（COUNT、SUM、AVG等）
        
        应用场景：
        1. 关系属性过滤：查找满足特定条件的关系
        2. 关系统计：统计满足条件的关系数量、金额总和等
        3. 关系分析：分析关系的分布、趋势等
        
        示例：
            # 查找交易金额大于400的交易
            results = client.filter_relationships(
                "TRANSFER",
                rel_conditions={"base_amt": (">", 400)},
                return_fields=["tran_id", "from.node_key", "to.node_key"]
            )
            
            # 统计is_sar为False的交易数量
            results = client.filter_relationships(
                "TRANSFER",
                rel_conditions={"is_sar": ("=", False)},
                aggregate="COUNT"
            )
            
            # 查找特定日期范围的交易
            results = client.filter_relationships(
                "TRANSFER",
                start_label="Account",
                end_label="Account",
                rel_conditions={"tran_timestamp": (">=", "2023-01-01")},
                return_fields=["tran_id", "base_amt", "from.acct_id", "to.acct_id"],
                order_by="base_amt",
                order_direction="DESC",
                limit=10
            )
        
        Args:
            rel_type: 关系类型
            start_label: 起点节点标签（可选）
            end_label: 终点节点标签（可选）
            rel_conditions: 关系属性条件字典，格式：{property: (operator, value)}
                           operator可以是: "=", ">", "<", ">=", "<=", "!=", "IN", "CONTAINS"
            return_fields: 要返回的字段列表，支持：
                          - 关系属性：直接写属性名，如 "tran_id", "base_amt"
                          - 起点节点属性：前缀"from."，如 "from.node_key", "from.acct_id"
                          - 终点节点属性：前缀"to."，如 "to.node_key", "to.acct_id"
            aggregate: 聚合类型（"COUNT", "SUM", "AVG", "MAX", "MIN"）
            aggregate_field: 聚合字段（当aggregate不是COUNT时需要）
            order_by: 排序字段
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回数量（None=不限制）
            
        Returns:
            如果是聚合查询，返回聚合结果：
            [{"aggregate_type": "count", "value": 100}]
            
            如果是普通查询，返回关系和节点信息：
            [
                {"tran_id": "T001", "from_account": "A001", "to_account": "A002"},
                ...
            ]
            
        注意：
            - 这是关系查询，不是节点查询
            - rel_conditions中的条件使用AND逻辑组合
            - 如果需要OR逻辑，请使用filter_query方法
        """
        rel_type = self._sanitize_rel_type(rel_type)
        
        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")
        
        # 构建节点模式
        start_pattern = f"(from:`{self._sanitize_label(start_label)}`)" if start_label else "(from)"
        end_pattern = f"(to:`{self._sanitize_label(end_label)}`)" if end_label else "(to)"
        
        # 构建WHERE子句
        where_parts = []
        params = {}
        
        if rel_conditions:
            for i, (key, condition) in enumerate(rel_conditions.items()):
                # 处理特殊的日期范围键名（如 tran_timestamp_start, tran_timestamp_end）
                if key.endswith("_start") or key.endswith("_end"):
                    # 提取实际的属性名
                    actual_key = key.rsplit("_", 1)[0]
                    actual_key = self._sanitize_property_key(actual_key)
                else:
                    actual_key = self._sanitize_property_key(key)
                
                if isinstance(condition, (tuple, list)) and len(condition) == 2:
                    operator, value = condition
                    param_name = f"rel_cond_{i}"
                    
                    if operator.upper() in ["IN", "CONTAINS"]:
                        where_parts.append(f"t.`{actual_key}` {operator.upper()} ${param_name}")
                    elif operator.upper() == "STARTS WITH":
                        where_parts.append(f"t.`{actual_key}` STARTS WITH ${param_name}")
                    else:
                        where_parts.append(f"t.`{actual_key}` {operator} ${param_name}")
                    
                    params[param_name] = value
                else:
                    # 简单等值条件
                    param_name = f"rel_cond_{i}"
                    where_parts.append(f"t.`{actual_key}` = ${param_name}")
                    params[param_name] = condition
        
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        
        # 聚合查询
        if aggregate:
            agg_type = aggregate.upper()
            if agg_type not in ["COUNT", "SUM", "AVG", "MAX", "MIN"]:
                raise ValueError(f"Invalid aggregate type: {aggregate}")
            
            if agg_type == "COUNT":
                agg_expr = "COUNT(t)"
            else:
                if not aggregate_field:
                    raise ValueError(f"aggregate_field is required for {agg_type}")
                agg_field = self._sanitize_property_key(aggregate_field)
                agg_expr = f"{agg_type}(t.`{agg_field}`)"
            
            cypher = f"""
            MATCH {start_pattern}-[t:`{rel_type}`]->{end_pattern}
            {where_clause}
            RETURN {agg_expr} AS value
            """
            
            result = self.run(cypher, params)
            if result:
                return [{"aggregate_type": agg_type.lower(), "value": result[0]["value"]}]
            return [{"aggregate_type": agg_type.lower(), "value": 0}]
        
        # 普通查询
        if return_fields:
            # 构建RETURN子句
            return_parts = []
            for field in return_fields:
                if field.startswith("from."):
                    # 起点节点属性
                    prop = self._sanitize_property_key(field[5:])
                    return_parts.append(f"from.`{prop}` AS {field.replace('.', '_')}")
                elif field.startswith("to."):
                    # 终点节点属性
                    prop = self._sanitize_property_key(field[3:])
                    return_parts.append(f"to.`{prop}` AS {field.replace('.', '_')}")
                elif field.startswith("rel."):
                    # 关系属性（带rel.前缀）
                    prop = self._sanitize_property_key(field[4:])
                    return_parts.append(f"t.`{prop}` AS {field.replace('.', '_')}")
                else:
                    # 关系属性（不带前缀，直接是属性名）
                    prop = self._sanitize_property_key(field)
                    return_parts.append(f"t.`{prop}` AS {prop}")
            
            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            # 返回整个关系和节点
            return_clause = "RETURN from, t AS relationship, to"
        
        # 可选的ORDER BY子句
        order_clause = ""
        if order_by:
            order_by = self._sanitize_property_key(order_by)
            order_clause = f"ORDER BY t.`{order_by}` {order_direction}"
        
        # 可选的LIMIT子句
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT $limit"
            params["limit"] = limit
        
        # 构建完整查询
        cypher = f"""
        MATCH {start_pattern}-[t:`{rel_type}`]->{end_pattern}
        {where_clause}
        {return_clause}
        {order_clause}
        {limit_clause}
        """
        
        return self.run(cypher, params)

    # add gjq
    def aggregation_query(
        self,
        aggregate_type: str,
        *,
        group_by_node: Optional[str] = None,
        group_by_property: Optional[str] = None,
        node_label: Optional[str] = None,
        rel_type: Optional[str] = None,
        direction: str = "out",
        aggregate_field: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        order_direction: str = "DESC",
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        聚合统计查询（支持按节点或属性分组）
        
        功能：
        - 支持按节点分组（如：每个账户）
        - 支持按属性分组（如：每个branch_id）
        - 支持多种聚合函数（COUNT、SUM、AVG、MAX、MIN）
        - 支持关系聚合和节点聚合
        - 支持排序和TOP-N查询
        
        应用场景：
        1. 统计每个账户的交易次数/金额
        2. 计算每个分支的账户数量
        3. 排名查询（TOP-N）
        
        示例：
            # 统计每个账户作为转出账户的交易次数，返回前5个
            results = client.aggregation_query(
                "COUNT",
                group_by_node="start",
                node_label="Account",
                rel_type="TRANSFER",
                direction="out",
                return_fields=["node_key"],
                order_by="count",
                order_direction="DESC",
                limit=5
            )
            
            # 计算每个账户的转出交易总金额
            results = client.aggregation_query(
                "SUM",
                group_by_node="start",
                node_label="Account",
                rel_type="TRANSFER",
                direction="out",
                aggregate_field="base_amt",
                return_fields=["last_name", "first_name"],
                order_by="total",
                order_direction="DESC"
            )
            
            # 统计每个branch_id下的账户数量
            results = client.aggregation_query(
                "COUNT",
                group_by_property="branch_id",
                node_label="Account"
            )
        
        Args:
            aggregate_type: 聚合类型（"COUNT", "SUM", "AVG", "MAX", "MIN"）
            group_by_node: 按节点分组（"start"=起点, "end"=终点, None=不按节点分组）
            group_by_property: 按属性分组（如 "branch_id"）
            node_label: 节点标签
            rel_type: 关系类型（可选，用于关系聚合）
            direction: 方向 ("out"=出边, "in"=入边, "both"=双向)
            aggregate_field: 聚合字段（当aggregate_type不是COUNT时需要）
            return_fields: 要返回的节点字段（如 ["last_name", "first_name"]）
            where: WHERE 过滤条件
            order_by: 排序字段（"count", "total", "avg"等）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回数量（None=不限制）
            
        Returns:
            [
                {"node_key": "A001", "count": 100},
                {"node_key": "A002", "count": 80},
                ...
            ]
        """
        agg_type = aggregate_type.upper()
        if agg_type not in ["COUNT", "SUM", "AVG", "MAX", "MIN"]:
            raise ValueError(f"Invalid aggregate type: {aggregate_type}")
        
        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")
        
        params = {}
        
        # 构建聚合表达式
        if agg_type == "COUNT":
            if rel_type:
                agg_expr = "COUNT(r) AS count"
                agg_alias = "count"
            else:
                agg_expr = "COUNT(n) AS count"
                agg_alias = "count"
        else:
            if not aggregate_field:
                raise ValueError(f"aggregate_field is required for {agg_type}")
            agg_field = self._sanitize_property_key(aggregate_field)
            if rel_type:
                agg_expr = f"{agg_type}(r.`{agg_field}`) AS total"
            else:
                agg_expr = f"{agg_type}(n.`{agg_field}`) AS total"
            agg_alias = "total"
        
        # 场景1：按节点属性分组（不涉及关系）
        if group_by_property and not rel_type:
            label = self._sanitize_label(node_label) if node_label else ""
            label_pattern = f":`{label}`" if label else ""
            prop = self._sanitize_property_key(group_by_property)
            
            where_clause = f"WHERE {where}" if where else ""
            limit_clause = "LIMIT $limit" if limit is not None else ""
            if limit is not None:
                params["limit"] = limit
            
            cypher = f"""
            MATCH (n{label_pattern})
            {where_clause}
            RETURN n.`{prop}` AS {prop}, {agg_expr}
            ORDER BY {order_by or agg_alias} {order_direction}
            {limit_clause}
            """
        
        # 场景2：按节点分组 + 关系聚合
        elif group_by_node and rel_type:
            label = self._sanitize_label(node_label) if node_label else ""
            label_pattern = f":`{label}`" if label else ""
            rt = self._sanitize_rel_type(rel_type)
            
            # 构建关系模式
            if direction == "out":
                if group_by_node == "start":
                    pattern = f"(n{label_pattern})-[r:`{rt}`]->()"
                else:
                    pattern = f"()-[r:`{rt}`]->(n{label_pattern})"
            elif direction == "in":
                if group_by_node == "start":
                    pattern = f"(n{label_pattern})<-[r:`{rt}`]-()"
                else:
                    pattern = f"()<-[r:`{rt}`]-(n{label_pattern})"
            else:
                pattern = f"(n{label_pattern})-[r:`{rt}`]-()"
            
            # 构建返回字段
            return_parts = []
            if return_fields:
                for field in return_fields:
                    field = self._sanitize_property_key(field)
                    return_parts.append(f"n.`{field}` AS {field}")
            else:
                return_parts.append("n.node_key AS node_key")
            
            return_clause = ", ".join(return_parts) + f", {agg_expr}"
            
            where_clause = f"WHERE {where}" if where else ""
            limit_clause = "LIMIT $limit" if limit is not None else ""
            if limit is not None:
                params["limit"] = limit
            
            cypher = f"""
            MATCH {pattern}
            {where_clause}
            RETURN {return_clause}
            ORDER BY {order_by or agg_alias} {order_direction}
            {limit_clause}
            """
        
        else:
            raise ValueError("Must specify either group_by_property or (group_by_node + rel_type)")
        
        return self.run(cypher, params)

    # =========================================================
    # 2. 邻居查询 / n跳关系
    # =========================================================
    # add gjq
    def neighbors_n_hop(
        self,
        label: str,
        key: str,
        value: Any,
        *,
        hops: int = 1,
        rel_type: Optional[str] = None,
        direction: str = "both",
        where: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order_direction: str = "ASC",
        limit: Optional[int] = None,
        return_distinct: bool = False,
        exclude_start: bool = False,
        return_path_length: bool = False
    ) -> List[JsonDict]:
        """
        查询节点的 N 跳邻居（支持通用修饰符和字段投影）
        
        功能：
        - 查询节点的N跳邻居
        - 支持指定返回字段（关系属性和邻居节点属性）
        - 支持WHERE条件过滤
        - 支持排序和限制结果数量
        - 支持去重和排除起始节点（多跳查询推荐）
        - 支持返回路径长度（按距离排序时需要）
        
        特别说明：
        - 当hops=1且需要返回每条边的详细信息时，建议使用return_fields指定字段
        - 对于"转出交易明细"等场景，应使用direction="out"确保方向正确
        - 多跳查询时建议设置 return_distinct=True 和 exclude_start=True
        
        示例：
            # 查询转出交易明细（每笔交易的转入账户和金额）
            results = client.neighbors_n_hop(
                "Account", "node_key", "Collins Steven",
                hops=1,
                rel_type="TRANSFER",
                direction="out",
                return_fields=["nbr.acct_id", "rel.base_amt", "rel.tran_id"]
            )
            
            # 查询二跳邻居（去重并排除起始节点）
            results = client.neighbors_n_hop(
                "Account", "node_key", "Lee Alex",
                hops=2,
                direction="both",
                return_distinct=True,
                exclude_start=True
            )
            
            # 按距离排序的多跳查询
            results = client.neighbors_n_hop(
                "Account", "node_key", "Collins Steven",
                hops=2,
                rel_type="TRANSFER",
                direction="out",
                return_distinct=True,
                exclude_start=True,
                return_path_length=True,
                order_by="path_length",
                order_direction="ASC"
            )
        
        Args:
            label: 起点节点标签
            key: 起点节点属性键
            value: 起点节点属性值
            hops: 跳数（1-10）
            rel_type: 关系类型（可选）
            direction: 方向 ("out"=出边, "in"=入边, "both"=双向)
            where: WHERE 过滤条件（如 "nbr.balance > 1000" 或 "firstRel.amount > 500"）
            return_fields: 要返回的字段列表，支持：
                          - 邻居节点属性：前缀"nbr."，如 "nbr.acct_id", "nbr.node_key"
                          - 关系属性：前缀"rel."，如 "rel.base_amt", "rel.tran_id"
                          - 路径长度："path_length"（当 return_path_length=True 时可用）
                          - 如果不指定，返回完整的neighbor/minHops/samplePath/rel
            order_by: 排序字段（如 "firstRel.base_amt" 或 "nbr.name" 或 "path_length"）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回数量（None=不限制）
            return_distinct: 是否去重（多跳查询时建议 True）
            exclude_start: 是否排除起始节点（多跳查询时建议 True）
            return_path_length: 是否返回路径长度（需要按距离排序时设为 True）
            
        Returns:
            如果指定return_fields，返回指定字段：
            [{"nbr_acct_id": "A001", "rel_base_amt": 1000, ...}, ...]
            
            如果不指定return_fields，返回完整信息：
            [{"neighbor": {...}, "minHops": 1, "samplePath": ..., "rel": {...}}, ...]
            
            如果 return_path_length=True，返回包含路径长度：
            [{"neighbor": {...}, "path_length": 1, ...}, ...]
        """
        label = self._sanitize_label(label)
        key = self._sanitize_property_key(key)
        
        if not (1 <= hops <= 10):
            raise ValueError("hops must be between 1 and 10")
        
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        
        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")
        
        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""
        
        # 特殊处理：hops=1且指定return_fields且不需要路径长度时，使用简化查询（避免路径聚合）
        if hops == 1 and return_fields and not return_path_length:
            # 构建单跳关系模式
            if direction == "out":
                pattern = f"(start)-[r{rel}]->(nbr)"
            elif direction == "in":
                pattern = f"(start)<-[r{rel}]-(nbr)"
            else:
                pattern = f"(start)-[r{rel}]-(nbr)"
            
            # 可选的 WHERE 子句
            where_parts = []
            if where:
                where_parts.append(where)
            if exclude_start:
                where_parts.append("nbr <> start")
            where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
            
            # 构建RETURN子句
            return_parts = []
            for field in return_fields:
                if field.startswith("nbr."):
                    # 邻居节点属性
                    prop = self._sanitize_property_key(field[4:])
                    return_parts.append(f"nbr.`{prop}` AS {field.replace('.', '_')}")
                elif field.startswith("rel."):
                    # 关系属性
                    prop = self._sanitize_property_key(field[4:])
                    return_parts.append(f"r.`{prop}` AS {field.replace('.', '_')}")
                else:
                    # 默认当作邻居节点属性
                    prop = self._sanitize_property_key(field)
                    return_parts.append(f"nbr.`{prop}` AS {prop}")
            
            return_clause = "RETURN " + (" DISTINCT " if return_distinct else " ") + ", ".join(return_parts)
            
            # 可选的 ORDER BY 子句
            order_clause = f"ORDER BY {order_by} {order_direction}" if order_by else ""
            
            # 可选的 LIMIT 子句
            limit_clause = "LIMIT $limit" if limit is not None else ""
            
            cypher = f"""
            MATCH (start:`{label}` {{`{key}`: $value}})
            MATCH {pattern}
            {where_clause}
            {return_clause}
            {order_clause}
            {limit_clause}
            """
        else:
            # 原有的多跳查询逻辑（使用路径聚合）
            # 构建路径模式
            if direction == "out":
                pattern = f"(start)-[r{rel}*1..{hops}]->(nbr)"
            elif direction == "in":
                pattern = f"(start)<-[r{rel}*1..{hops}]-(nbr)"
            else:
                pattern = f"(start)-[r{rel}*1..{hops}]-(nbr)"
            
            # 可选的 WHERE 子句
            where_parts = []
            if where:
                where_parts.append(where)
            if exclude_start:
                where_parts.append("nbr <> start")
            where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
            
            # 构建 RETURN 子句
            if return_path_length:
                # 返回路径长度
                distinct_clause = "DISTINCT" if return_distinct else ""
                return_clause = f"""
                RETURN {distinct_clause} nbr AS neighbor,
                       min(length(p)) AS path_length,
                       collect(p)[0] AS samplePath,
                       relationships(collect(p)[0])[0] AS rel
                """
            else:
                # 不返回路径长度
                distinct_clause = "DISTINCT" if return_distinct else ""
                return_clause = f"""
                RETURN {distinct_clause} nbr AS neighbor,
                       min(length(p)) AS minHops,
                       collect(p)[0] AS samplePath,
                       relationships(collect(p)[0])[0] AS rel
                """
            
            # 可选的 ORDER BY 子句
            if order_by:
                if order_by == "path_length" and return_path_length:
                    order_clause = f"ORDER BY path_length {order_direction}"
                else:
                    order_clause = f"ORDER BY {order_by} {order_direction}"
            else:
                order_clause = ""
            
            # 可选的 LIMIT 子句
            limit_clause = "LIMIT $limit" if limit is not None else ""
            
            cypher = f"""
            MATCH (start:`{label}` {{`{key}`: $value}})
            MATCH p = {pattern}
            WITH nbr, min(length(p)) AS minHops, collect(p)[0] AS samplePath, relationships(collect(p)[0]) AS rels
            WITH nbr, minHops, samplePath, rels[0] AS firstRel
            {where_clause}
            {return_clause}
            {order_clause}
            {limit_clause}
            """
        
        params = {"value": value}
        if limit is not None:
            params["limit"] = limit
        
        return self.run(cypher, params)

    # =========================================================
    # 3. 公共邻居
    # =========================================================
    def common_neighbors(
        self,
        a: Tuple[str, str, Any],
        b: Tuple[str, str, Any],
        *,
        rel_type: Optional[str] = None,
        direction: str = "both",
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        order_direction: str = "ASC",
        limit: Optional[int] = None,
        aggregate: bool = False
    ) -> List[JsonDict]:
        """
        查询两个节点的公共一跳邻居（支持通用修饰符 + 聚合排序）
        
        功能：
        - 查询两个节点的公共邻居
        - 支持按交易次数聚合排序（aggregate=True）
        
        Args:
            a: (label, key, value) 节点A
            b: (label, key, value) 节点B
            rel_type: 关系类型（可选）
            direction: 方向
            where: WHERE 过滤条件（如 "C.balance > 1000"）
            order_by: 排序字段（如 "C.name" 或 "rA.amount" 或 "count"）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回数量（None=不限制）
            aggregate: 是否启用聚合模式（按交易次数统计）
            
        Returns:
            如果 aggregate=False（默认）：
            [{"commonNeighbor": {...}, "relA": {...}, "relB": {...}}, ...]
            
            如果 aggregate=True：
            [{"commonNeighbor": {...}, "count": 2}, ...]
        """
        (la, ka, va) = a
        (lb, kb, vb) = b
        
        la = self._sanitize_label(la)
        lb = self._sanitize_label(lb)
        ka = self._sanitize_property_key(ka)
        kb = self._sanitize_property_key(kb)
        
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        
        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")
        
        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""
        
        # 构建路径模式
        if direction == "out":
            pat_a = f"(A)-[rA{rel}]->(C)"
            pat_b = f"(B)-[rB{rel}]->(C)"
        elif direction == "in":
            pat_a = f"(A)<-[rA{rel}]-(C)"
            pat_b = f"(B)<-[rB{rel}]-(C)"
        else:
            pat_a = f"(A)-[rA{rel}]-(C)"
            pat_b = f"(B)-[rB{rel}]-(C)"
        
        # 可选的 WHERE 子句
        where_clause = f"WHERE {where}" if where else ""
        
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        
        # 聚合模式：统计每个公共邻居的交易次数
        if aggregate:
            # 如果 order_by 是 "count"，使用聚合计数
            if order_by == "count":
                order_clause = f"ORDER BY count {order_direction}"
            else:
                order_clause = f"ORDER BY {order_by} {order_direction}" if order_by else "ORDER BY count DESC"
            
            cypher = f"""
            MATCH (A:`{la}` {{`{ka}`: $va}})
            MATCH (B:`{lb}` {{`{kb}`: $vb}})
            MATCH {pat_a}
            MATCH {pat_b}
            {where_clause}
            RETURN C AS commonNeighbor, COUNT(*) AS count
            {order_clause}
            {limit_clause}
            """
        else:
            # 普通模式：返回所有关系详情
            order_clause = f"ORDER BY {order_by} {order_direction}" if order_by else ""
            
            cypher = f"""
            MATCH (A:`{la}` {{`{ka}`: $va}})
            MATCH (B:`{lb}` {{`{kb}`: $vb}})
            MATCH {pat_a}
            MATCH {pat_b}
            {where_clause}
            RETURN C AS commonNeighbor, rA AS relA, rB AS relB
            {order_clause}
            {limit_clause}
            """
        
        params = {"va": va, "vb": vb}
        if limit is not None:
            params["limit"] = limit
        
        return self.run(cypher, params)

    def common_neighbors_with_rel_filter(
        self,
        a: Tuple[str, str, Any],
        b: Tuple[str, str, Any],
        *,
        rel_type: Optional[str] = None,
        direction: str = "both",
        rel_conditions: Optional[Dict[str, Any]] = None,
        neighbor_where: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order_direction: str = "ASC",
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        查询两个节点的公共邻居，支持对两条关系分别设置属性过滤条件
        
        功能：
        - 查询两个节点的公共一跳邻居
        - 支持对A→C和B→C的关系分别设置属性过滤条件
        - 支持对公共邻居节点C设置过滤条件
        - 支持指定返回字段
        
        应用场景：
        - 找出两个账户的共同交易对手，且交易金额都大于某个值
        - 找出两个用户的共同好友，且关系建立时间都在某个时间段内
        
        示例：
            # 找出Steven Collins和Samantha Cook的共同交易邻居，且交易金额都>400
            results = client.common_neighbors_with_rel_filter(
                a=("Account", "node_key", "Collins Steven"),
                b=("Account", "node_key", "Cook Samantha"),
                rel_type="TRANSFER",
                direction="both",
                rel_conditions={"base_amt": (">", 400)},
                return_fields=["C.node_key", "C.acct_id", "rA.base_amt", "rB.base_amt"]
            )
            
            # 找出两个账户的共同交易对手，且都是大额交易（>1000）且交易时间在2025年
            results = client.common_neighbors_with_rel_filter(
                a=("Account", "node_key", "Collins Steven"),
                b=("Account", "node_key", "Cook Samantha"),
                rel_type="TRANSFER",
                rel_conditions={
                    "base_amt": (">", 1000),
                    "tran_timestamp": (">=", "2025-01-01")
                },
                return_fields=["C.node_key", "rA.base_amt", "rA.tran_timestamp",
                              "rB.base_amt", "rB.tran_timestamp"]
            )
        
        Args:
            a: (label, key, value) 节点A
            b: (label, key, value) 节点B
            rel_type: 关系类型（可选）
            direction: 方向 ("out"=出边, "in"=入边, "both"=双向)
            rel_conditions: 关系属性条件字典，格式：{property: (operator, value)}
                           这些条件会同时应用到rA和rB两条关系上
                           operator可以是: "=", ">", "<", ">=", "<=", "!=", "IN", "CONTAINS"
            neighbor_where: 对公共邻居节点C的过滤条件（如 "C.balance > 1000"）
            return_fields: 要返回的字段列表，支持：
                          - 公共邻居属性：前缀"C."，如 "C.node_key", "C.acct_id"
                          - A→C关系属性：前缀"rA."，如 "rA.base_amt", "rA.tran_id"
                          - B→C关系属性：前缀"rB."，如 "rB.base_amt", "rB.tran_id"
                          - 如果不指定，返回完整的commonNeighbor/relA/relB
            order_by: 排序字段（如 "C.name" 或 "rA.base_amt"）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回数量（None=不限制）
            
        Returns:
            如果指定return_fields，返回指定字段：
            [{"C_node_key": "...", "rA_base_amt": 500, "rB_base_amt": 600}, ...]
            
            如果不指定return_fields，返回完整信息：
            [{"commonNeighbor": {...}, "relA": {...}, "relB": {...}}, ...]
            
        注意：
            - rel_conditions中的条件会同时应用到rA和rB（AND逻辑）
            - 如果需要对rA和rB设置不同的条件，请使用neighbor_where参数手动指定
        """
        (la, ka, va) = a
        (lb, kb, vb) = b
        
        la = self._sanitize_label(la)
        lb = self._sanitize_label(lb)
        ka = self._sanitize_property_key(ka)
        kb = self._sanitize_property_key(kb)
        
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        
        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")
        
        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""
        
        # 构建路径模式
        if direction == "out":
            pat_a = f"(A)-[rA{rel}]->(C)"
            pat_b = f"(B)-[rB{rel}]->(C)"
        elif direction == "in":
            pat_a = f"(A)<-[rA{rel}]-(C)"
            pat_b = f"(B)<-[rB{rel}]-(C)"
        else:
            pat_a = f"(A)-[rA{rel}]-(C)"
            pat_b = f"(B)-[rB{rel}]-(C)"
        
        # 构建WHERE子句
        where_parts = []
        params = {"va": va, "vb": vb}
        
        # 处理关系属性条件（同时应用到rA和rB）
        if rel_conditions:
            for i, (key, condition) in enumerate(rel_conditions.items()):
                key = self._sanitize_property_key(key)
                
                if isinstance(condition, (tuple, list)) and len(condition) == 2:
                    operator, value = condition
                    param_name_a = f"rel_cond_a_{i}"
                    param_name_b = f"rel_cond_b_{i}"
                    
                    if operator.upper() in ["IN", "CONTAINS"]:
                        where_parts.append(f"rA.`{key}` {operator.upper()} ${param_name_a}")
                        where_parts.append(f"rB.`{key}` {operator.upper()} ${param_name_b}")
                    elif operator.upper() == "STARTS WITH":
                        where_parts.append(f"rA.`{key}` STARTS WITH ${param_name_a}")
                        where_parts.append(f"rB.`{key}` STARTS WITH ${param_name_b}")
                    else:
                        where_parts.append(f"rA.`{key}` {operator} ${param_name_a}")
                        where_parts.append(f"rB.`{key}` {operator} ${param_name_b}")
                    
                    params[param_name_a] = value
                    params[param_name_b] = value
                else:
                    # 简单等值条件
                    param_name_a = f"rel_cond_a_{i}"
                    param_name_b = f"rel_cond_b_{i}"
                    where_parts.append(f"rA.`{key}` = ${param_name_a}")
                    where_parts.append(f"rB.`{key}` = ${param_name_b}")
                    params[param_name_a] = condition
                    params[param_name_b] = condition
        
        # 添加公共邻居节点的过滤条件
        if neighbor_where:
            where_parts.append(f"({neighbor_where})")
        
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        
        # 构建RETURN子句
        if return_fields:
            return_parts = []
            for field in return_fields:
                # 解析字段格式：C.node_key, rA.base_amt, rB.tran_id
                if "." in field:
                    prefix, prop = field.split(".", 1)
                    prop = self._sanitize_property_key(prop)
                    # 生成别名：C_node_key, rA_base_amt, rB_tran_id
                    alias = f"{prefix}_{prop}"
                    return_parts.append(f"{prefix}.`{prop}` AS {alias}")
                else:
                    # 如果没有前缀，默认是公共邻居的属性
                    prop = self._sanitize_property_key(field)
                    return_parts.append(f"C.`{prop}` AS {prop}")
            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            return_clause = "RETURN C AS commonNeighbor, rA AS relA, rB AS relB"
        
        # 可选的 ORDER BY 子句
        order_clause = f"ORDER BY {order_by} {order_direction}" if order_by else ""
        
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        if limit is not None:
            params["limit"] = limit
        
        cypher = f"""
        MATCH (A:`{la}` {{`{ka}`: $va}})
        MATCH (B:`{lb}` {{`{kb}`: $vb}})
        MATCH {pat_a}
        MATCH {pat_b}
        {where_clause}
        {return_clause}
        {order_clause}
        {limit_clause}
        """
        
        return self.run(cypher, params)

    # =========================================================
    # 4. 条件过滤查询
    # =========================================================
    def filter_query(
        self,
        start: Tuple[str, str, Any],
        *,
        rel_type: Optional[str] = None,
        node_label: Optional[str] = None,
        direction: str = "out",
        node_where: Optional[str] = None,
        rel_where: Optional[str] = None,
        params: Optional[JsonDict] = None,
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        从起点出发，按节点/关系条件过滤
        
        Args:
            start: (label, key, value) 起点节点
            rel_type: 关系类型（可选）
            node_label: 目标节点标签（可选）
            direction: 方向
            node_where: 节点过滤条件（如 "n.age >= $minAge"）
            rel_where: 关系过滤条件（如 "r.weight > $minW"）
            params: 额外参数
            limit: 最大返回数量（None=不限制）
            
        Returns:
            [{"start": {...}, "rel": {...}, "node": {...}}, ...]
            
        警告：
            node_where 和 rel_where 是字符串片段，需要自行确保安全性
        """
        (sl, sk, sv) = start
        sl = self._sanitize_label(sl)
        sk = self._sanitize_property_key(sk)
        
        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""
        
        tl = self._sanitize_label(node_label) if node_label else ""
        tlabel = f":`{tl}`" if tl else ""
        
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        
        # 构建路径模式
        if direction == "out":
            pat = f"(s)-[r{rel}]->(n{tlabel})"
        elif direction == "in":
            pat = f"(s)<-[r{rel}]-(n{tlabel})"
        else:
            pat = f"(s)-[r{rel}]-(n{tlabel})"
        
        # 构建 WHERE 子句
        where_parts = []
        if rel_where:
            where_parts.append(f"({rel_where})")
        if node_where:
            where_parts.append(f"({node_where})")
        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        
        cypher = f"""
        MATCH (s:`{sl}` {{`{sk}`: $sv}})
        MATCH {pat}
        {where_clause}
        RETURN s AS start, r AS rel, n AS node
        {limit_clause}
        """
        
        p = {"sv": sv}
        if limit is not None:
            p["limit"] = limit
        if params:
            p.update(params)
        
        return self.run(cypher, p)

    # =========================================================
    # 5. 子图抽取
    # =========================================================
    def subgraph_extract(
        self,
        center: Tuple[str, str, Any],
        *,
        hops: int = 2,
        rel_type: Optional[str] = None,
        direction: str = "both",
        where: Optional[str] = None,
        limit_paths: int = 200
    ) -> JsonDict:
        """
        抽取以某节点为中心的子图（支持通用修饰符）
        
        Args:
            center: (label, key, value) 中心节点
            hops: 半径（跳数）
            rel_type: 关系类型（可选）
            direction: 方向
            where: WHERE 过滤条件（如 "n.balance > 1000"）
            limit_paths: 最大路径数
            
        Returns:
            {"nodes": [...], "relationships": [...]}
        """
        (cl, ck, cv) = center
        cl = self._sanitize_label(cl)
        ck = self._sanitize_property_key(ck)
        
        if not (1 <= hops <= 5):
            raise ValueError("hops must be between 1 and 5")
        
        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""
        
        # 构建路径模式
        if direction == "out":
            pat = f"(c)-[r{rel}*1..{hops}]->(n)"
        elif direction == "in":
            pat = f"(c)<-[r{rel}*1..{hops}]-(n)"
        else:
            pat = f"(c)-[r{rel}*1..{hops}]-(n)"
        
        # 可选的 WHERE 子句
        where_clause = f"WHERE {where}" if where else ""
        
        cypher = f"""
        MATCH (c:`{cl}` {{`{ck}`: $cv}})
        MATCH p = {pat}
        {where_clause}
        WITH collect(p)[0..$limit_paths] AS ps
        UNWIND ps AS p
        UNWIND nodes(p) AS nn
        UNWIND relationships(p) AS rr
        WITH collect(DISTINCT nn) AS nodes, collect(DISTINCT rr) AS relationships
        RETURN nodes, relationships
        """
        
        res = self.run(cypher, {"cv": cv, "limit_paths": limit_paths})
        
        if res and res[0]:
            return {
                "nodes": res[0].get("nodes", []),
                "relationships": res[0].get("relationships", [])
            }
        
        return {"nodes": [], "relationships": []}

    def subgraph_extract_by_nodes(
        self,
        label: str,
        key: str,
        values: List[Any],
        *,
        include_internal: bool = True,
        rel_type: Optional[str] = None,
        direction: str = "both",
        where: Optional[str] = None
    ) -> JsonDict:
        """
        基于节点列表抽取子图（包含指定节点及其相互之间的关系）
        
        功能：
        - 提取指定节点列表中所有节点
        - 提取这些节点之间的所有关系
        - 可选择是否包含节点内部的关系（如 A->A）
        
        应用场景：
        1. 交易网络：提取账户 A、B、C 及其之间的转账记录
        2. 社交网络：提取指定用户群体及其相互关系
        3. 知识图谱：提取指定实体及其关联关系
        
        示例：
            # 提取账户 A、B、C 及其之间的转账关系
            subgraph = client.subgraph_extract_by_nodes(
                "Account",
                "node_key",
                ["Collins Steven", "Nunez Mitchell", "Lee Alex"],
                rel_type="TRANSFER",
                direction="both"
            )
            
            # 提取用户群体的社交关系
            subgraph = client.subgraph_extract_by_nodes(
                "User",
                "userId",
                ["u1", "u2", "u3", "u4"],
                rel_type="FOLLOWS",
                include_internal=False  # 不包含自环
            )
        
        Args:
            label: 节点标签
            key: 节点属性键
            values: 节点属性值列表（如 ["A", "B", "C"]）
            include_internal: 是否包含节点内部关系（如 A->A），默认 True
            rel_type: 关系类型（可选，None=任意类型）
            direction: 方向 ("out"=单向, "in"=反向, "both"=双向)
            where: WHERE 过滤条件（如 "r.amount > 1000"）
            
        Returns:
            {
                "nodes": [节点列表],
                "relationships": [关系列表],
                "node_count": 节点数量,
                "relationship_count": 关系数量
            }
            
        注意：
            - 只返回指定节点之间的关系，不会扩展到其他节点
            - 如果某个节点不存在，会在结果中忽略
            - 如果节点之间没有关系，relationships 为空列表
        """
        label = self._sanitize_label(label)
        key = self._sanitize_property_key(key)
        
        if not values or len(values) == 0:
            raise ValueError("values list cannot be empty")
        
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        
        # 构建关系模式
        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""
        
        # 构建路径模式
        if direction == "out":
            pattern = f"(n1)-[r{rel}]->(n2)"
        elif direction == "in":
            pattern = f"(n1)<-[r{rel}]-(n2)"
        else:
            pattern = f"(n1)-[r{rel}]-(n2)"
        
        # 构建 WHERE 子句
        where_parts = []
        
        # 节点必须在指定列表中
        where_parts.append("n1.`" + key + "` IN $values")
        where_parts.append("n2.`" + key + "` IN $values")
        
        # 是否排除自环
        if not include_internal:
            where_parts.append("n1 <> n2")
        
        # 用户自定义过滤条件
        if where:
            where_parts.append(f"({where})")
        
        where_clause = "WHERE " + " AND ".join(where_parts)
        
        # Cypher 查询
        cypher = f"""
        MATCH (n1:`{label}`)
        WHERE n1.`{key}` IN $values
        WITH collect(n1) AS allNodes
        
        MATCH {pattern}
        {where_clause}
        WITH allNodes, collect(DISTINCT r) AS allRels
        
        RETURN allNodes AS nodes,
               allRels AS relationships,
               size(allNodes) AS node_count,
               size(allRels) AS relationship_count
        """
        
        res = self.run(cypher, {"values": values})
        
        if res and res[0]:
            return {
                "nodes": res[0].get("nodes", []),
                "relationships": res[0].get("relationships", []),
                "node_count": res[0].get("node_count", 0),
                "relationship_count": res[0].get("relationship_count", 0)
            }
        
        return {
            "nodes": [],
            "relationships": [],
            "node_count": 0,
            "relationship_count": 0
        }

    def subgraph_extract_by_rel_filter(
        self,
        rel_type: str,
        rel_conditions: Dict[str, Any],
        *,
        start_label: Optional[str] = None,
        end_label: Optional[str] = None,
        direction: str = "both",
        limit: Optional[int] = None
    ) -> JsonDict:
        """
        基于关系属性条件抽取子图（提取满足条件的关系及相关节点）
        
        功能：
        - 根据关系属性条件筛选关系
        - 提取满足条件的关系及其起点和终点节点
        - 返回子图结构（节点 + 关系）
        
        应用场景：
        1. 时间范围：抽取某天/某时间段的所有交易子图
        2. 金额范围：抽取金额在指定范围内的交易子图
        3. 类型过滤：抽取特定类型的关系子图
        
        示例：
            # 抽取 2025-05-01 当天所有交易构成的子图
            subgraph = client.subgraph_extract_by_rel_filter(
                "TRANSFER",
                {"tran_timestamp": (">=", "2025-05-01"),
                 "tran_timestamp": ("<", "2025-05-02")}
            )
            
            # 抽取交易金额在 300 到 500 之间的交易子图
            subgraph = client.subgraph_extract_by_rel_filter(
                "TRANSFER",
                {"base_amt": (">=", 300), "base_amt": ("<=", 500)}
            )
            
            # 抽取可疑交易子图
            subgraph = client.subgraph_extract_by_rel_filter(
                "TRANSFER",
                {"is_sar": ("=", True)},
                start_label="Account",
                end_label="Account"
            )
        
        Args:
            rel_type: 关系类型
            rel_conditions: 关系属性条件字典，格式：{property: (operator, value)}
                           operator可以是: "=", ">", "<", ">=", "<=", "!=", "IN", "CONTAINS"
            start_label: 起点节点标签（可选）
            end_label: 终点节点标签（可选）
            direction: 方向 ("out"=单向, "in"=反向, "both"=双向)
            limit: 最大关系数量（None=不限制）
            
        Returns:
            {
                "nodes": [节点列表],
                "relationships": [关系列表],
                "node_count": 节点数量,
                "relationship_count": 关系数量
            }
            
        注意：
            - 返回的节点是满足条件的关系的起点和终点节点
            - 如果需要计算统计信息（如交易总数），可以从 relationship_count 获取
        """
        rel_type = self._sanitize_rel_type(rel_type)
        
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        
        # 构建节点模式
        start_pattern = f"(from:`{self._sanitize_label(start_label)}`)" if start_label else "(from)"
        end_pattern = f"(to:`{self._sanitize_label(end_label)}`)" if end_label else "(to)"
        
        # 构建关系模式
        if direction == "out":
            rel_pattern = f"{start_pattern}-[r:`{rel_type}`]->{end_pattern}"
        elif direction == "in":
            rel_pattern = f"{start_pattern}<-[r:`{rel_type}`]-{end_pattern}"
        else:
            rel_pattern = f"{start_pattern}-[r:`{rel_type}`]-{end_pattern}"
        
        # 构建WHERE子句
        where_parts = []
        params = {}
        
        if rel_conditions:
            for i, (key, condition) in enumerate(rel_conditions.items()):
                # 处理特殊的日期范围键名（如 tran_timestamp_start, tran_timestamp_end）
                if key.endswith("_start") or key.endswith("_end"):
                    # 提取实际的属性名
                    actual_key = key.rsplit("_", 1)[0]
                    actual_key = self._sanitize_property_key(actual_key)
                else:
                    actual_key = self._sanitize_property_key(key)
                
                if isinstance(condition, (tuple, list)) and len(condition) == 2:
                    operator, value = condition
                    param_name = f"rel_cond_{i}"
                    
                    if operator.upper() in ["IN", "CONTAINS"]:
                        where_parts.append(f"r.`{actual_key}` {operator.upper()} ${param_name}")
                    elif operator.upper() == "STARTS WITH":
                        where_parts.append(f"r.`{actual_key}` STARTS WITH ${param_name}")
                    else:
                        where_parts.append(f"r.`{actual_key}` {operator} ${param_name}")
                    
                    params[param_name] = value
                else:
                    # 简单等值条件
                    param_name = f"rel_cond_{i}"
                    where_parts.append(f"r.`{actual_key}` = ${param_name}")
                    params[param_name] = condition
        
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        if limit is not None:
            params["limit"] = limit
        
        # Cypher 查询
        cypher = f"""
        MATCH {rel_pattern}
        {where_clause}
        WITH collect(DISTINCT from) + collect(DISTINCT to) AS allNodes,
             collect(DISTINCT r) AS allRels
        {limit_clause}
        RETURN allNodes AS nodes,
               allRels AS relationships,
               size(allNodes) AS node_count,
               size(allRels) AS relationship_count
        """
        
        res = self.run(cypher, params)
        
        if res and res[0]:
            return {
                "nodes": res[0].get("nodes", []),
                "relationships": res[0].get("relationships", []),
                "node_count": res[0].get("node_count", 0),
                "relationship_count": res[0].get("relationship_count", 0)
            }
        
        return {
            "nodes": [],
            "relationships": [],
            "node_count": 0,
            "relationship_count": 0
        }

    # =========================================================
    # 6. 指定路径模式查询
    # =========================================================
    def match_path_pattern(
        self,
        *,
        pattern: str,
        where: Optional[str] = None,
        params: Optional[JsonDict] = None,
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        自定义路径模式查询
        
        示例:
            pattern = "(a:User {userId:$uid})-[:FOLLOWS]->(b:User)-[:POSTED]->(p:Post)"
            where   = "p.createdAt >= $since"
            
        Args:
            pattern: 路径模式
            where: WHERE 子句（可选）
            params: 参数
            limit: 最大返回数量（None=不限制）
            
        Returns:
            [{"path": ...}, ...]
            
        警告：
            pattern 和 where 是字符串片段，需要自行确保安全性
        """
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        
        cypher = f"""
        MATCH p = {pattern}
        {("WHERE " + where) if where else ""}
        RETURN p AS path
        {limit_clause}
        """
        
        p = {}
        if limit is not None:
            p["limit"] = limit
        if params:
            p.update(params)
        
        return self.run(cypher, p)

    # =========================================================
    # 7. 聚合统计类
    # =========================================================
    def aggregate_stats(
        self,
        label: str,
        *,
        group_by: Optional[str] = None,
        where: Optional[str] = None,
        params: Optional[JsonDict] = None,
        metrics: Optional[Sequence[str]] = None,
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        聚合统计查询
        
        示例：
            # 统计每个国家的用户数
            aggregate_stats("User", group_by="country")
            
            # 统计年龄>=18的用户，按城市分组
            aggregate_stats(
                "User",
                group_by="city",
                where="n.age >= $minAge",
                params={"minAge": 18}
            )
            
        Args:
            label: 节点标签
            group_by: 分组字段（可选）
            where: WHERE 子句（可选）
            params: 额外参数
            metrics: 聚合指标（默认 count(*)）
            limit: 最大返回数量（None=不限制）
            
        Returns:
            聚合结果列表
        """
        label = self._sanitize_label(label)
        metrics = list(metrics) if metrics else ["count(*) AS cnt"]
        
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        
        if group_by:
            group_by = self._sanitize_property_key(group_by)
            group_expr = f"n.`{group_by}` AS {group_by}"
            return_expr = ", ".join([group_expr] + list(metrics))
            
            cypher = f"""
            MATCH (n:`{label}`)
            {("WHERE " + where) if where else ""}
            WITH {return_expr}
            RETURN {return_expr}
            ORDER BY cnt DESC
            {limit_clause}
            """
        else:
            return_expr = ", ".join(metrics)
            
            cypher = f"""
            MATCH (n:`{label}`)
            {("WHERE " + where) if where else ""}
            RETURN {return_expr}
            {limit_clause}
            """
        
        p = {}
        if limit is not None:
            p["limit"] = limit
        if params:
            p.update(params)
        
        return self.run(cypher, p)
    # =========================================================
    # 两点间路径查询（完整版，不限定最短）
    # =========================================================
    def paths_between(
        self,
        a: Tuple[str, str, Any],
        b: Tuple[str, str, Any],
        *,
        rel_type: Optional[str] = None,
        direction: str = "both",
        min_hops: int = 1,
        max_hops: int = 5,
        where: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order_direction: str = "ASC",
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        查询两个节点之间的路径（支持通用修饰符 + 复杂过滤/排序/计算）
        
        功能：
        - 支持指定最小/最大跳数
        - 支持指定关系类型和方向
        - ⚠️ **支持复杂的 WHERE 过滤条件**（路径上的关系属性、节点属性）
        - ⚠️ **支持自定义返回字段**（可以计算路径总金额、最低金额等）
        - ⚠️ **支持自定义排序**（按金额、时间、路径长度等）
        - 可返回多条路径（不限定必须是最短）
        
        示例：
            # 例1：查询路径，按转账金额最大排序
            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                return_fields=["path", "hops", "maxAmount"],
                order_by="maxAmount",
                order_direction="DESC"
            )
            
            # 例4：查询路径，要求路径上包含 is_sar 的交易
            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                where="ANY(r IN relationships(p) WHERE r.is_sar = true)"
            )
            
            # 例8：计算每条路径的总金额
            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                return_fields=["path", "hops", "totalAmount"],
                order_by="totalAmount",
                order_direction="DESC"
            )
            
            # 例9：查询路径，要求所有交易金额都 < 1000
            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                where="ALL(r IN relationships(p) WHERE r.base_amt < 1000)"
            )
            
            # 例10：查询路径，要求不经过 bank 节点
            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                where="ALL(n IN nodes(p) WHERE n.bank_id <> 'bank')"
            )
        
        Args:
            a: (label, key, value) 起点节点
            b: (label, key, value) 终点节点
            rel_type: 关系类型（可选，None=任意类型）
            direction: 方向 ("out"=单向, "in"=反向, "both"=双向)
            min_hops: 最小跳数（默认1）
            max_hops: 最大跳数（默认5，建议不超过10）
            where: WHERE 过滤条件（支持复杂的路径过滤）
                   - 路径长度：hops <= 3
                   - 关系属性：ANY(r IN relationships(p) WHERE r.is_sar = true)
                   - 节点属性：ALL(n IN nodes(p) WHERE n.bank_id <> 'bank')
                   - 金额范围：ALL(r IN relationships(p) WHERE r.base_amt < 1000)
            return_fields: 要返回的字段列表（支持计算字段）
                          - 基础字段：path, hops, nodes, relationships
                          - 计算字段：
                            * totalAmount: REDUCE(s = 0, r IN relationships(p) | s + r.base_amt)
                            * maxAmount: REDUCE(m = 0, r IN relationships(p) | CASE WHEN r.base_amt > m THEN r.base_amt ELSE m END)
                            * minAmount: REDUCE(m = 999999, r IN relationships(p) | CASE WHEN r.base_amt < m THEN r.base_amt ELSE m END)
                            * avgAmount: REDUCE(s = 0, r IN relationships(p) | s + r.base_amt) / hops
            order_by: 排序字段（如 "hops", "totalAmount", "maxAmount"）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回路径数（None=不限制）
            
        Returns:
            [
                {
                    "path": <Path对象>,
                    "hops": 路径长度,
                    "nodes": [节点列表],
                    "relationships": [关系列表],
                    "totalAmount": 总金额（如果指定）,
                    "maxAmount": 最大金额（如果指定）,
                    ...
                },
                ...
            ]
            
        注意：
            - 默认按路径长度升序返回（短路径优先）
            - 如果两点不连通，返回空列表
            - max_hops 过大可能导致查询缓慢
            - WHERE 子句支持 Neo4j 的列表推导式（ANY, ALL, NONE, SINGLE）
        """
        (la, ka, va) = a
        (lb, kb, vb) = b
        
        # 参数验证
        la = self._sanitize_label(la)
        lb = self._sanitize_label(lb)
        ka = self._sanitize_property_key(ka)
        kb = self._sanitize_property_key(kb)
        
        if not (0 <= min_hops <= max_hops <= 10):
            raise ValueError("Invalid hop bounds: 0 <= min_hops <= max_hops <= 10")
        
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        
        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")
        
        # 构建关系模式
        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""
        
        # 构建路径模式
        if direction == "out":
            pattern = f"(A)-[r{rel}*{min_hops}..{max_hops}]->(B)"
        elif direction == "in":
            pattern = f"(A)<-[r{rel}*{min_hops}..{max_hops}]-(B)"
        else:
            pattern = f"(A)-[r{rel}*{min_hops}..{max_hops}]-(B)"
        
        # 可选的 WHERE 子句
        where_clause = f"WHERE {where}" if where else ""
        
        # 构建 RETURN 子句
        if return_fields:
            # 自定义返回字段
            return_parts = []
            for field in return_fields:
                field_lower = field.lower()
                
                # 基础字段
                if field_lower == "path":
                    return_parts.append("p AS path")
                elif field_lower == "hops":
                    return_parts.append("hops")
                elif field_lower == "nodes":
                    return_parts.append("pathNodes AS nodes")
                elif field_lower == "relationships":
                    return_parts.append("pathRels AS relationships")
                
                # 计算字段：总金额
                elif field_lower == "totalamount":
                    return_parts.append("REDUCE(s = 0, r IN pathRels | s + r.base_amt) AS totalAmount")
                
                # 计算字段：最大金额
                elif field_lower == "maxamount":
                    return_parts.append("REDUCE(m = 0, r IN pathRels | CASE WHEN r.base_amt > m THEN r.base_amt ELSE m END) AS maxAmount")
                
                # 计算字段：最小金额
                elif field_lower == "minamount":
                    return_parts.append("REDUCE(m = 999999, r IN pathRels | CASE WHEN r.base_amt < m THEN r.base_amt ELSE m END) AS minAmount")
                
                # 计算字段：平均金额
                elif field_lower == "avgamount":
                    return_parts.append("REDUCE(s = 0, r IN pathRels | s + r.base_amt) / hops AS avgAmount")
                
                # 其他字段：直接使用
                else:
                    return_parts.append(field)
            
            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            # 默认返回字段
            return_clause = """RETURN p AS path,
                hops,
                pathNodes AS nodes,
                pathRels AS relationships"""
        
        # 可选的 ORDER BY 子句（默认按 hops 升序）
        if order_by:
            order_clause = f"ORDER BY {order_by} {order_direction}"
        else:
            order_clause = "ORDER BY hops ASC"
        
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        
        # Cypher 查询
        cypher = f"""
        MATCH (A:`{la}` {{`{ka}`: $va}}), (B:`{lb}` {{`{kb}`: $vb}})
        MATCH p = {pattern}
        WITH p, length(p) AS hops, nodes(p) AS pathNodes, relationships(p) AS pathRels
        {where_clause}
        {return_clause}
        {order_clause}
        {limit_clause}
        """
        
        params = {"va": va, "vb": vb}
        if limit is not None:
            params["limit"] = limit
        
        return self.run(cypher, params)



# ==========================================
# 测试/演示代码
# ==========================================
if __name__ == "__main__":
    # 使用上下文管理器自动关闭连接
    with Neo4jGraphClient(
        Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
    ) as client:
        
        # 获取 Schema
        schema = client.get_schema()
        print("=== Schema 信息 ===")
        print(f"节点类型: {list(schema['node_labels'].keys())}")
        print(f"关系类型: {list(schema['relationship_types'].keys())}")
        print(f"模式样例: {schema['patterns'][:3]}\n")
        
        # 1. 唯一键查节点
        print("=== 测试1: 唯一键查节点 ===")
        user = client.get_node_by_unique_key("User", "userId", "u123")
        print(f"用户: {user}\n")
        
        # 2. N跳邻居
        print("=== 测试2: N跳邻居 ===")
        neighbors = client.neighbors_n_hop(
            "User", "userId", "u123",
            hops=2,
            rel_type="FOLLOWS",
            direction="out",
            limit=10
        )
        print(f"找到 {len(neighbors)} 个邻居\n")
        
        # 3. 公共邻居
        print("=== 测试3: 公共邻居 ===")
        common = client.common_neighbors(
            ("User", "userId", "u1"),
            ("User", "userId", "u2"),
            rel_type="FOLLOWS"
        )
        print(f"找到 {len(common)} 个公共邻居\n")
        
        # 4. 聚合统计
        print("=== 测试4: 聚合统计 ===")
        stats = client.aggregate_stats(
            "User",
            group_by="country",
            limit=5
        )
        print(f"统计结果: {stats}\n")