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
        """打印填充参数后的查询（用于调试）"""
        print("\n" + "="*80)
        print("📝 执行的 Cypher 查询语句（参数已填充）:")
        print("-"*80)
        
        filled_cypher = cypher
        if params:
            import json
            # 按参数名长度倒序排序，避免子串替换问题
            sorted_params = sorted(params.items(), key=lambda x: len(x[0]), reverse=True)
            
            for key, value in sorted_params:
                # 根据值类型格式化
                if isinstance(value, str):
                    # 转义单引号
                    formatted_value = f"'{value.replace(chr(39), chr(39)+chr(39))}'"
                elif isinstance(value, (int, float)):
                    formatted_value = str(value)
                elif isinstance(value, bool):
                    formatted_value = str(value).lower()
                elif value is None:
                    formatted_value = "null"
                elif isinstance(value, (list, dict)):
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
        value: Any
    ) -> Optional[JsonDict]:
        """
        根据 label + 唯一键属性查询节点
        
        示例：
            get_node_by_unique_key("User", "userId", "u123")
            
        Args:
            label: 节点标签
            key: 属性键（如 userId）
            value: 属性值（如 u123）
            
        Returns:
            节点数据或 None
        """
        label = self._sanitize_label(label)
        key = self._sanitize_property_key(key)
        
        cypher = f"MATCH (n:`{label}` {{`{key}`: $value}}) RETURN n AS node LIMIT 1"
        res = self.run(cypher, {"value": value})
        
        return res[0]["node"] if res else None

    # =========================================================
    # 2. 邻居查询 / n跳关系
    # =========================================================
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
        order_by: Optional[str] = None,
        order_direction: str = "ASC",
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        查询节点的 N 跳邻居（支持通用修饰符）
        
        Args:
            label: 起点节点标签
            key: 起点节点属性键
            value: 起点节点属性值
            hops: 跳数（1-10）
            rel_type: 关系类型（可选）
            direction: 方向 ("out"=出边, "in"=入边, "both"=双向)
            where: WHERE 过滤条件（如 "nbr.balance > 1000" 或 "firstRel.amount > 500"）
            order_by: 排序字段（如 "firstRel.base_amt" 或 "nbr.name"）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回数量（None=不限制）
            
        Returns:
            [{"neighbor": {...}, "minHops": 1, "samplePath": ..., "rel": {...}}, ...]
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
        
        # 构建路径模式
        if direction == "out":
            pattern = f"(start)-[r{rel}*1..{hops}]->(nbr)"
        elif direction == "in":
            pattern = f"(start)<-[r{rel}*1..{hops}]-(nbr)"
        else:
            pattern = f"(start)-[r{rel}*1..{hops}]-(nbr)"
        
        # 可选的 WHERE 子句
        where_clause = f"WHERE {where}" if where else ""
        
        # 可选的 ORDER BY 子句
        order_clause = f"ORDER BY {order_by} {order_direction}" if order_by else ""
        
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        
        cypher = f"""
        MATCH (start:`{label}` {{`{key}`: $value}})
        MATCH p = {pattern}
        WITH nbr, min(length(p)) AS minHops, collect(p)[0] AS samplePath, relationships(collect(p)[0]) AS rels
        WITH nbr, minHops, samplePath, rels[0] AS firstRel
        {where_clause}
        RETURN nbr AS neighbor, minHops, samplePath, firstRel AS rel
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
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        查询两个节点的公共一跳邻居（支持通用修饰符）
        
        Args:
            a: (label, key, value) 节点A
            b: (label, key, value) 节点B
            rel_type: 关系类型（可选）
            direction: 方向
            where: WHERE 过滤条件（如 "C.balance > 1000"）
            order_by: 排序字段（如 "C.name" 或 "rA.amount"）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回数量（None=不限制）
            
        Returns:
            [{"commonNeighbor": {...}, "relA": {...}, "relB": {...}}, ...]
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
        
        # 可选的 ORDER BY 子句
        order_clause = f"ORDER BY {order_by} {order_direction}" if order_by else ""
        
        # 可选的 LIMIT 子句
        limit_clause = "LIMIT $limit" if limit is not None else ""
        
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
        order_by: Optional[str] = None,
        order_direction: str = "ASC",
        limit: Optional[int] = None
    ) -> List[JsonDict]:
        """
        查询两个节点之间的路径（支持通用修饰符）
        
        功能：
        - 支持指定最小/最大跳数
        - 支持指定关系类型和方向
        - 支持 WHERE 过滤条件
        - 支持自定义排序（默认按路径长度升序）
        - 可返回多条路径（不限定必须是最短）
        
        示例：
            # 查询用户 u1 到 u2 之间的所有路径（最多3跳）
            paths = client.paths_between(
                ("User", "userId", "u1"),
                ("User", "userId", "u2"),
                max_hops=3,
                limit=5
            )
            
            # 查询通过 FOLLOWS 关系的路径，按路径长度降序
            paths = client.paths_between(
                ("User", "userId", "u1"),
                ("User", "userId", "u2"),
                rel_type="FOLLOWS",
                direction="out",
                min_hops=2,
                max_hops=4,
                order_by="hops",
                order_direction="DESC"
            )
        
        Args:
            a: (label, key, value) 起点节点
            b: (label, key, value) 终点节点
            rel_type: 关系类型（可选，None=任意类型）
            direction: 方向 ("out"=单向, "in"=反向, "both"=双向)
            min_hops: 最小跳数（默认1）
            max_hops: 最大跳数（默认5，建议不超过10）
            where: WHERE 过滤条件（如 "hops <= 3"）
            order_by: 排序字段（如 "hops"，默认按 hops ASC）
            order_direction: 排序方向 ("ASC"=升序, "DESC"=降序)
            limit: 最大返回路径数（None=不限制）
            
        Returns:
            [
                {
                    "path": <Path对象>,
                    "hops": 路径长度,
                    "nodes": [节点列表],
                    "relationships": [关系列表]
                },
                ...
            ]
            
        注意：
            - 默认按路径长度升序返回（短路径优先）
            - 如果两点不连通，返回空列表
            - max_hops 过大可能导致查询缓慢
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
        RETURN p AS path,
            hops,
            pathNodes AS nodes,
            pathRels AS relationships
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