# Legacy import path (reference): computing_engine/graph_query/graph_query.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, DatabaseError, TransientError
import os
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
    Neo4j 3.5.25-compatible query template library.
    Supported query types:
    1. Node lookup by ID/unique key - get_node_by_unique_key()
    2. Neighbor / N-hop relationships - neighbors_n_hop()
    3. Common neighbors - common_neighbors()
    4. Predicate-filtered queries - filter_query()
    5. Subgraph extraction - subgraph_extract()
    6. Aggregate statistics - aggregate_stats()
    7. Paths between two nodes - paths_between()
    """

    # Neo4j reserved keywords (partial list)
    RESERVED_KEYWORDS = {
        "MATCH",
        "RETURN",
        "WHERE",
        "CREATE",
        "DELETE",
        "SET",
        "MERGE",
        "WITH",
        "UNWIND",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "ORDER",
        "BY",
        "SKIP",
        "LIMIT",
        "AS",
        "AND",
        "OR",
        "NOT",
        "IN",
        "IS",
        "NULL",
        "TRUE",
        "FALSE",
    }

    def __init__(self, config: Neo4jConfig):
        """
        Initialize the Neo4j client.

        Args:
            config: Neo4j connection settings.
        """
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
            max_connection_lifetime=3600,  # Max connection lifetime: 1 hour
            max_connection_pool_size=50,  # Connection pool size
            connection_acquisition_timeout=60.0,  # Connection acquisition timeout (seconds)
        )
        self._db = config.database

    def close(self) -> None:
        """Close the database driver."""
        if self._driver:
            self._driver.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # ===== Core executor (with retries) =====
    def run(
        self,
        cypher: str,
        params: Optional[JsonDict] = None,
        *,
        read: bool = True,
        max_retries: int = 3,
        show_query: bool = True,
    ) -> List[JsonDict]:
        """
        Run a Cypher query with automatic retries on transient failures.

        Args:
            cypher: Cypher statement.
            params: Query parameters.
            read: True for read transactions, False for write.
            max_retries: Maximum retry attempts.
            show_query: If True, print the query with parameters substituted (debug).

        Returns:
            List of result rows as dicts.

        Raises:
            RuntimeError: Query failed after retries or on non-transient errors.
        """
        params = params or {}

        # Optionally print filled query (debug)
        if show_query:
            self._print_filled_query(cypher, params)

        # Retry loop
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
                # Transient error: retry
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Neo4j TransientError after {max_retries} retries: {e}\n"
                        f"Cypher: {cypher}\nParams: {params}"
                    ) from e

                wait_time = 2**attempt  # Exponential backoff
                print(
                    f"⚠️  TransientError, retrying in {wait_time}s ({attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)

            except (ClientError, DatabaseError) as e:
                # Client or database error: do not retry
                raise RuntimeError(
                    f"Neo4j Error: {e}\nCypher: {cypher}\nParams: {params}"
                ) from e

    def _print_filled_query(self, cypher: str, params: Dict) -> None:
        """
        Print the query with parameters substituted (for debugging only).

        Note: Display only; execution still uses parameterized queries via the driver.
        """
        print("\n" + "=" * 80)
        print("📝 执行的 Cypher 查询语句（参数已填充）:")
        print("-" * 80)

        filled_cypher = cypher
        if params:
            import json

            # Sort param names by length descending to avoid partial $id vs $id2 replacements
            sorted_params = sorted(
                params.items(), key=lambda x: len(x[0]), reverse=True
            )

            for key, value in sorted_params:
                # CRITICAL: Skip tuple params (DSL internal form; must not appear in final Cypher)
                # Tuple form like (">", 500000) should already be expanded in WHERE building
                if isinstance(value, tuple):
                    # Should not happen; indicates a bug if it does
                    print(
                        f"⚠️ WARNING: 参数 ${key} 是元组格式 {value}，这不应该出现在最终查询中！"
                    )
                    continue

                # Format value by type
                if isinstance(value, str):
                    # Escape single quotes
                    formatted_value = f"'{value.replace(chr(39), chr(39) + chr(39))}'"
                elif isinstance(value, (int, float)):
                    formatted_value = str(value)
                elif isinstance(value, bool):
                    # Fix 1: Booleans must be lowercase true/false (Neo4j 3.5.25)
                    formatted_value = "true" if value else "false"
                elif value is None:
                    formatted_value = "null"
                elif isinstance(value, list):
                    # List (e.g. for IN)
                    formatted_value = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, dict):
                    # Dict (rare)
                    formatted_value = json.dumps(value, ensure_ascii=False)
                else:
                    formatted_value = str(value)

                # Regex for full token match (avoid $id matching inside $id2)
                filled_cypher = re.sub(
                    r"\$" + re.escape(key) + r"\b",  # Word boundary
                    formatted_value,
                    filled_cypher,
                )

        print(filled_cypher)
        print("=" * 80 + "\n")

    # ===== Schema introspection =====
    def get_schema(self) -> Dict:
        """
        Fetch graph schema metadata (extended).

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
        schema = {"node_labels": {}, "relationship_types": {}, "patterns": []}

        # 1. All node labels and properties (with sample values)
        labels_result = self.run(
            "CALL db.labels() YIELD label RETURN label", show_query=False
        )
        valid_labels = [item["label"] for item in labels_result]

        for label in valid_labels:
            if not label or not self._is_valid_identifier(label):
                continue

            # Properties and sample values for this label
            props_result = self.run(
                f"""
                MATCH (n:`{label}`)
                WITH n LIMIT 1
                UNWIND keys(n) AS key
                RETURN key, n[key] AS sample_value
                ORDER BY key
            """,
                show_query=False,
            )

            if props_result:
                schema["node_labels"][label] = {
                    "properties": [p["key"] for p in props_result],
                    "sample_values": {
                        p["key"]: p["sample_value"] for p in props_result
                    },
                }

        # 2. All relationship types and properties (with sample values)
        rels_result = self.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
            show_query=False,
        )
        valid_rels = [item["relationshipType"] for item in rels_result]

        for rel_type in valid_rels:
            if not rel_type or not self._is_valid_identifier(rel_type):
                continue

            # Properties and sample values for this relationship type
            props_result = self.run(
                f"""
                MATCH ()-[r:`{rel_type}`]->()
                WITH r LIMIT 1
                UNWIND keys(r) AS key
                RETURN key, r[key] AS sample_value
                ORDER BY key
            """,
                show_query=False,
            )

            if props_result:
                schema["relationship_types"][rel_type] = {
                    "properties": [p["key"] for p in props_result],
                    "sample_values": {
                        p["key"]: p["sample_value"] for p in props_result
                    },
                }

        # 3. Relationship patterns (start_label, rel, end_label)
        patterns = self.run(
            """
            MATCH (a)-[r]->(b)
            WITH labels(a)[0] AS start_label,
                 type(r) AS rel_type,
                 labels(b)[0] AS end_label
            WHERE start_label IS NOT NULL
              AND rel_type IS NOT NULL
              AND end_label IS NOT NULL
            RETURN DISTINCT start_label, rel_type, end_label
            LIMIT 100
        """,
            show_query=False,
        )

        schema["patterns"] = [
            f"({p['start_label']})-[:{p['rel_type']}]->({p['end_label']})"
            for p in patterns
        ]

        return schema

    # ===== Validation helpers =====
    @staticmethod
    def _is_valid_identifier(name: str) -> bool:
        """Return True if name is a simple alphanumeric/underscore identifier."""
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))

    @classmethod
    def _sanitize_label(cls, label: Optional[str]) -> str:
        """
        Validate and return a safe node label.

        Rules:
        - Must start with a letter
        - Only letters, digits, underscore
        - Max length 255
        - Must not be a Cypher reserved word
        """
        if not label:
            return ""

        # Length check
        if len(label) > 255:
            raise ValueError(f"Label too long (max 255): {label}")

        # Format check
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", label):
            raise ValueError(
                f"Invalid label format (must start with letter, contain only alphanumeric and underscore): {label}"
            )

        # Reserved keyword check
        if label.upper() in cls.RESERVED_KEYWORDS:
            raise ValueError(f"Reserved keyword cannot be used as label: {label}")

        return label

    @classmethod
    def _sanitize_rel_type(cls, rel_type: Optional[str]) -> str:
        """
        Validate relationship type name.

        Rules:
        - Typically UPPER_SNAKE (e.g. FOLLOWS, HAS_FRIEND)
        - Lower/camel also allowed for compatibility
        - Max length 255
        """
        if not rel_type:
            return ""

        if len(rel_type) > 255:
            raise ValueError(f"Relationship type too long (max 255): {rel_type}")

        # Format: letter first, then alphanumeric/underscore
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", rel_type):
            raise ValueError(f"Invalid relationship type format: {rel_type}")

        if rel_type.upper() in cls.RESERVED_KEYWORDS:
            raise ValueError(
                f"Reserved keyword cannot be used as relationship type: {rel_type}"
            )

        return rel_type

    @staticmethod
    def _sanitize_property_key(key: str) -> str:
        """
        Validate a property key name.

        Rules:
        - Must start with a letter
        - Only letters, digits, underscore
        - Max length 255
        """
        if not key:
            raise ValueError("Property key cannot be empty")

        if len(key) > 255:
            raise ValueError(f"Property key too long (max 255): {key}")

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", key):
            raise ValueError(f"Invalid property key format: {key}")

        return key

    # =========================================================
    # 1. Lookup by internal ID / unique business key
    # =========================================================
    def get_node_by_internal_id(
        self, internal_id: int, *, return_props: bool = True
    ) -> Optional[JsonDict]:
        """
        Look up a node by Neo4j internal id(n).

        Note: Internal ids can change after import/delete; avoid as business keys.

        Args:
            internal_id: Neo4j internal node id.
            return_props: If True, return node properties map only.

        Returns:
            Node data dict or None.
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
        return_fields: Optional[List[str]] = None,
    ) -> Optional[JsonDict]:
        """
        Find a node by label and a unique key property.

        Examples:
            # Full node
            get_node_by_unique_key("User", "userId", "u123")

            # Projected fields only
            get_node_by_unique_key("Account", "node_key", "Collins Steven",
                                  return_fields=["acct_id", "acct_stat", "acct_open_date"])

        Args:
            label: Node label.
            key: Property name (e.g. userId).
            value: Property value (e.g. u123).
            return_fields: Fields to return (None = entire node).

        Returns:
            Node or field dict, or None.
        """
        label = self._sanitize_label(label)
        key = self._sanitize_property_key(key)

        # Build RETURN clause
        if return_fields:
            # Projected fields
            return_parts = []
            for field in return_fields:
                field = self._sanitize_property_key(field)
                return_parts.append(f"n.`{field}` AS {field}")
            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            # Whole node
            return_clause = "RETURN n AS node"

        cypher = f"MATCH (n:`{label}` {{`{key}`: $value}}) {return_clause} LIMIT 1"
        res = self.run(cypher, {"value": value})

        if not res:
            return None

        # Dict of fields vs. full node
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
        limit: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        Filter multiple nodes by property predicates (AND-combined).

        Behavior:
        - AND-combines multiple property conditions
        - Optional field projection instead of full nodes
        - Optional ORDER BY and LIMIT

        Typical uses:
        1. Find all nodes matching criteria
        2. Multi-attribute filtering
        3. Project only needed fields

        Examples:
            # Accounts in US, state VT — last/first name and city
            results = client.filter_nodes_by_properties(
                "Account",
                {"country": "US", "state": "VT"},
                return_fields=["last_name", "first_name", "city"]
            )

            # Accounts with USD reporting currency and status A
            results = client.filter_nodes_by_properties(
                "Account",
                {"acct_rptng_crncy": "USD", "acct_stat": "A"},
                return_fields=["last_name", "first_name", "acct_id"],
                order_by="last_name",
                limit=10
            )

            # Range: initial_deposit > 500000
            results = client.filter_nodes_by_properties(
                "Account",
                {"initial_deposit": (">", 500000)},
                return_fields=["acct_id", "initial_deposit"]
            )

            # Boolean: prior_sar_count is true in graph
            results = client.filter_nodes_by_properties(
                "Account",
                {"prior_sar_count": True},  # Note: Python True is sent as Neo4j true
                return_fields=["acct_id"]
            )

        Args:
            label: Node label.
            conditions: Dict of predicates:
                      1. Equality: {"property": value}
                      2. Range: {"property": (operator, value)}
                         operator: "=", ">", "<", ">=", "<=", "!=", "IN", "CONTAINS", "STARTS WITH"
                      Example: {"age": (">", 18), "country": "US"}
            return_fields: Fields to return (None = whole node).
            order_by: Sort property (e.g. "last_name").
            order_direction: "ASC" or "DESC".
            limit: Max rows (None = no limit).

        Returns:
            List of dict rows, e.g. [{"last_name": "Smith", ...}, ...]

        Notes:
            - Multi-node scan, not single-node key lookup.
            - Conditions are ANDed; for OR use filter_query or custom Cypher.
        """
        label = self._sanitize_label(label)

        if not conditions:
            raise ValueError("conditions cannot be empty")

        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")

        # Build WHERE clause
        where_parts = []
        params = {}
        for i, (key, value) in enumerate(conditions.items()):
            key = self._sanitize_property_key(key)
            param_name = f"cond_{i}"

            # Fix 2: (operator, value) tuple or [operator, value] list
            if isinstance(value, (tuple, list)) and len(value) == 2:
                operator, actual_value = value
                # Validate operator
                valid_operators = [
                    "=",
                    ">",
                    "<",
                    ">=",
                    "<=",
                    "!=",
                    "IN",
                    "CONTAINS",
                    "STARTS WITH",
                ]
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
                # Equality
                where_parts.append(f"a.`{key}` = ${param_name}")
                params[param_name] = value

        where_clause = "WHERE " + " AND ".join(where_parts)

        # Build RETURN clause
        if return_fields:
            # Projected fields
            return_parts = []
            for field in return_fields:
                field = self._sanitize_property_key(field)
                return_parts.append(f"a.`{field}` AS {field}")
            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            # Whole node
            return_clause = "RETURN a AS node"

        # Optional ORDER BY
        order_clause = ""
        if order_by:
            order_by = self._sanitize_property_key(order_by)
            order_clause = f"ORDER BY a.`{order_by}` {order_direction}"

        # Optional LIMIT
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT $limit"
            params["limit"] = limit

        # Full query
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
        limit: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        Filter relationships by relationship properties; optional aggregates.

        Behavior:
        - Filter by predicates on the relationship
        - Optional start/end node labels
        - Project relationship and endpoint fields
        - Optional COUNT / SUM / AVG / MAX / MIN

        Typical uses:
        1. Property-filtered relationship lists
        2. Counts and sums over matching edges
        3. Relationship-centric analytics

        Examples:
            # Transfers with base_amt > 400
            results = client.filter_relationships(
                "TRANSFER",
                rel_conditions={"base_amt": (">", 400)},
                return_fields=["tran_id", "from.node_key", "to.node_key"]
            )

            # Count transfers where is_sar is false
            results = client.filter_relationships(
                "TRANSFER",
                rel_conditions={"is_sar": ("=", False)},
                aggregate="COUNT"
            )

            # Transfers on/after a date
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
            rel_type: Relationship type.
            start_label: Optional start node label.
            end_label: Optional end node label.
            rel_conditions: {property: (operator, value)}; operators
                           "=", ">", "<", ">=", "<=", "!=", "IN", "CONTAINS", "STARTS WITH".
            return_fields: Projection list:
                          - Edge props: "tran_id", "base_amt"
                          - Start node: "from.node_key", "from.acct_id"
                          - End node: "to.node_key", "to.acct_id"
            aggregate: "COUNT", "SUM", "AVG", "MAX", or "MIN".
            aggregate_field: Required when aggregate is not COUNT.
            order_by: Sort field (relationship property).
            order_direction: "ASC" or "DESC".
            limit: Max rows (None = unlimited).

        Returns:
            Aggregate: [{"aggregate_type": "count", "value": 100}]
            Rows: [{"tran_id": "T001", "from_node_key": ...}, ...]

        Notes:
            - Relationship query, not node-only.
            - rel_conditions are ANDed; for OR use filter_query or custom Cypher.
        """
        rel_type = self._sanitize_rel_type(rel_type)

        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")

        # Node patterns
        start_pattern = (
            f"(from:`{self._sanitize_label(start_label)}`)" if start_label else "(from)"
        )
        end_pattern = (
            f"(to:`{self._sanitize_label(end_label)}`)" if end_label else "(to)"
        )

        # WHERE clause
        where_parts = []
        params = {}

        if rel_conditions:
            for i, (key, condition) in enumerate(rel_conditions.items()):
                # Keys like tran_timestamp_start / _end → strip suffix for real prop name
                if key.endswith("_start") or key.endswith("_end"):
                    # Actual property name
                    actual_key = key.rsplit("_", 1)[0]
                    actual_key = self._sanitize_property_key(actual_key)
                else:
                    actual_key = self._sanitize_property_key(key)

                if isinstance(condition, (tuple, list)) and len(condition) == 2:
                    operator, value = condition
                    param_name = f"rel_cond_{i}"

                    if operator.upper() in ["IN", "CONTAINS"]:
                        where_parts.append(
                            f"t.`{actual_key}` {operator.upper()} ${param_name}"
                        )
                    elif operator.upper() == "STARTS WITH":
                        where_parts.append(
                            f"t.`{actual_key}` STARTS WITH ${param_name}"
                        )
                    else:
                        where_parts.append(f"t.`{actual_key}` {operator} ${param_name}")

                    params[param_name] = value
                else:
                    # Equality
                    param_name = f"rel_cond_{i}"
                    where_parts.append(f"t.`{actual_key}` = ${param_name}")
                    params[param_name] = condition

        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        # Aggregate branch
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
                return [
                    {"aggregate_type": agg_type.lower(), "value": result[0]["value"]}
                ]
            return [{"aggregate_type": agg_type.lower(), "value": 0}]

        # Row-returning query
        if return_fields:
            # RETURN list
            return_parts = []
            for field in return_fields:
                if field.startswith("from."):
                    # Start node property
                    prop = self._sanitize_property_key(field[5:])
                    return_parts.append(f"from.`{prop}` AS {field.replace('.', '_')}")
                elif field.startswith("to."):
                    # End node property
                    prop = self._sanitize_property_key(field[3:])
                    return_parts.append(f"to.`{prop}` AS {field.replace('.', '_')}")
                elif field.startswith("rel."):
                    # Relationship property (rel. prefix)
                    prop = self._sanitize_property_key(field[4:])
                    return_parts.append(f"t.`{prop}` AS {field.replace('.', '_')}")
                else:
                    # Relationship property (bare name)
                    prop = self._sanitize_property_key(field)
                    return_parts.append(f"t.`{prop}` AS {prop}")

            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            # Full from / rel / to
            return_clause = "RETURN from, t AS relationship, to"

        # Optional ORDER BY
        order_clause = ""
        if order_by:
            order_by = self._sanitize_property_key(order_by)
            order_clause = f"ORDER BY t.`{order_by}` {order_direction}"

        # Optional LIMIT
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT $limit"
            params["limit"] = limit

        # Full Cypher
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
        limit: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        Aggregations with grouping by node or by a node property.

        Behavior:
        - Group by anchor node (start/end) along a relationship type
        - Or group by a single property on nodes (no relationship)
        - COUNT / SUM / AVG / MAX / MIN
        - ORDER BY and TOP-N

        Typical uses:
        1. Per-account transfer counts or sums
        2. Account counts per branch_id
        3. Ranked reports (TOP-N)

        Examples:
            # Outgoing TRANSFER count per Account (top 5)
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

            # Sum of outgoing base_amt per Account
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

            # Account count per branch_id
            results = client.aggregation_query(
                "COUNT",
                group_by_property="branch_id",
                node_label="Account"
            )

        Args:
            aggregate_type: "COUNT", "SUM", "AVG", "MAX", or "MIN".
            group_by_node: "start", "end", or None (with rel_type).
            group_by_property: Property name for property-only grouping.
            node_label: Node label filter.
            rel_type: Relationship type when grouping via edges.
            direction: "out", "in", or "both".
            aggregate_field: Required when type is not COUNT.
            return_fields: Extra node fields to return.
            where: Raw WHERE clause fragment (caller must sanitize).
            order_by: e.g. "count", "total".
            order_direction: "ASC" or "DESC".
            limit: Max groups returned.

        Returns:
            Rows like [{"node_key": "A001", "count": 100}, ...]
        """
        agg_type = aggregate_type.upper()
        if agg_type not in ["COUNT", "SUM", "AVG", "MAX", "MIN"]:
            raise ValueError(f"Invalid aggregate type: {aggregate_type}")

        if order_direction not in {"ASC", "DESC"}:
            raise ValueError("order_direction must be 'ASC' or 'DESC'")

        params = {}

        # Aggregate expression
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

        # Case 1: group by property on nodes only (no rel_type)
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

        # Case 2: group by node + relationship aggregation
        elif group_by_node and rel_type:
            label = self._sanitize_label(node_label) if node_label else ""
            label_pattern = f":`{label}`" if label else ""
            rt = self._sanitize_rel_type(rel_type)

            # Relationship pattern
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

            # RETURN parts
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
            raise ValueError(
                "Must specify either group_by_property or (group_by_node + rel_type)"
            )

        return self.run(cypher, params)

    # =========================================================
    # 2. Neighbors / N-hop reachability
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
        return_path_length: bool = False,
    ) -> List[JsonDict]:
        """
        N-hop neighbors from a start node (filters, projection, path stats).

        Behavior:
        - Reach neighbors within `hops` along optional rel_type and direction
        - Project neighbor vs. edge fields via return_fields
        - Optional WHERE, ORDER BY, LIMIT
        - DISTINCT and exclude_start for multi-hop
        - Optional shortest-hop length as path_length for distance ordering

        Notes:
        - For hops=1 and per-edge detail, prefer return_fields.
        - For outgoing transfer details use direction=\"out\".
        - Multi-hop: set return_distinct=True and exclude_start=True.

        Examples:
            # One-hop outgoing transfers — counterparty and amounts
            results = client.neighbors_n_hop(
                "Account", "node_key", "Collins Steven",
                hops=1,
                rel_type="TRANSFER",
                direction="out",
                return_fields=["nbr.acct_id", "rel.base_amt", "rel.tran_id"]
            )

            # Two-hop, distinct neighbors excluding source
            results = client.neighbors_n_hop(
                "Account", "node_key", "Lee Alex",
                hops=2,
                direction="both",
                return_distinct=True,
                exclude_start=True
            )

            # Multi-hop with distance ordering
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
            label: Start node label.
            key / value: Business key for start node.
            hops: 1–10.
            rel_type: Optional relationship type filter.
            direction: "out", "in", or "both".
            where: Predicate (e.g. "nbr.balance > 1000").
            return_fields: "nbr.*", "rel.*", or "path_length" when return_path_length.
            order_by: e.g. "firstRel.base_amt", "nbr.name", "path_length".
            order_direction: "ASC" or "DESC".
            limit: Max rows.
            return_distinct: Collapse duplicate endpoints.
            exclude_start: Drop the source node from results.
            return_path_length: Expose hop count as path_length.

        Returns:
            With return_fields: projected dicts.
            Without: neighbor, minHops, samplePath, rel (and relType).
            With return_path_length: path_length alias instead of minHops in naming above.
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

        # Fast path: single hop + projection + no path length → skip path aggregation
        if hops == 1 and return_fields and not return_path_length:
            # One-hop pattern
            if direction == "out":
                pattern = f"(start)-[r{rel}]->(nbr)"
            elif direction == "in":
                pattern = f"(start)<-[r{rel}]-(nbr)"
            else:
                pattern = f"(start)-[r{rel}]-(nbr)"

            # Optional WHERE
            where_parts = []
            if where:
                where_parts.append(where)
            if exclude_start:
                where_parts.append("nbr <> start")
            where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

            # RETURN
            return_parts = []
            for field in return_fields:
                if field.startswith("nbr."):
                    # Neighbor property
                    prop = self._sanitize_property_key(field[4:])
                    return_parts.append(f"nbr.`{prop}` AS {field.replace('.', '_')}")
                elif field.startswith("rel."):
                    # Relationship property
                    prop = self._sanitize_property_key(field[4:])
                    return_parts.append(f"r.`{prop}` AS {field.replace('.', '_')}")
                else:
                    # Treat bare name as neighbor property
                    prop = self._sanitize_property_key(field)
                    return_parts.append(f"nbr.`{prop}` AS {prop}")

            return_clause = (
                "RETURN "
                + (" DISTINCT " if return_distinct else " ")
                + ", ".join(return_parts)
            )

            # Optional ORDER BY
            order_clause = f"ORDER BY {order_by} {order_direction}" if order_by else ""

            # Optional LIMIT
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
            # Multi-hop via path aggregation
            # Variable-length path pattern
            if direction == "out":
                pattern = f"(start)-[r{rel}*1..{hops}]->(nbr)"
            elif direction == "in":
                pattern = f"(start)<-[r{rel}*1..{hops}]-(nbr)"
            else:
                pattern = f"(start)-[r{rel}*1..{hops}]-(nbr)"

            # Optional WHERE
            where_parts = []
            if where:
                where_parts.append(where)
            if exclude_start:
                where_parts.append("nbr <> start")
            where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

            # RETURN
            if return_path_length:
                # Include hop count as path_length
                distinct_clause = "DISTINCT" if return_distinct else ""
                return_clause = f"""
                RETURN {distinct_clause} nbr AS neighbor,
                       minHops AS path_length,
                       samplePath AS samplePath,
                       firstRel AS rel,
                       type(firstRel) AS relType
                """
            else:
                # minHops without renaming to path_length
                distinct_clause = "DISTINCT" if return_distinct else ""
                return_clause = f"""
                RETURN {distinct_clause} nbr AS neighbor,
                       minHops AS minHops,
                       samplePath AS samplePath,
                       firstRel AS rel,
                       type(firstRel) AS relType
                """

            # Optional ORDER BY
            if order_by:
                if order_by == "path_length" and return_path_length:
                    order_clause = f"ORDER BY path_length {order_direction}"
                else:
                    order_clause = f"ORDER BY {order_by} {order_direction}"
            else:
                order_clause = ""

            # Optional LIMIT
            limit_clause = "LIMIT $limit" if limit is not None else ""

            cypher = f"""
            MATCH (start:`{label}` {{`{key}`: $value}})
            MATCH p = {pattern}
            WITH start, nbr, min(length(p)) AS minHops, collect(p)[0] AS samplePath, relationships(collect(p)[0]) AS rels
            WITH start, nbr, minHops, samplePath, rels[0] AS firstRel
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
    # 3. Common neighbors
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
        aggregate: bool = False,
    ) -> List[JsonDict]:
        """
        One-hop common neighbors of two nodes; optional count-based ordering.

        Behavior:
        - Intersection of neighbors of A and B under the same rel pattern
        - aggregate=True: one row per common neighbor with COUNT(*)

        Args:
            a, b: (label, key, value) for each endpoint.
            rel_type: Optional relationship type.
            direction: "out", "in", or "both".
            where: Predicate on C (e.g. "C.balance > 1000").
            order_by: e.g. "C.name", "rA.amount", or "count".
            order_direction: "ASC" or "DESC".
            limit: Max rows.
            aggregate: If True, return per-neighbor counts.

        Returns:
            Default: [{"commonNeighbor": ..., "relA": ..., "relB": ...}, ...]
            aggregate: [{"commonNeighbor": ..., "count": n}, ...]
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

        # Path patterns A–C and B–C
        if direction == "out":
            pat_a = f"(A)-[rA{rel}]->(C)"
            pat_b = f"(B)-[rB{rel}]->(C)"
        elif direction == "in":
            pat_a = f"(A)<-[rA{rel}]-(C)"
            pat_b = f"(B)<-[rB{rel}]-(C)"
        else:
            pat_a = f"(A)-[rA{rel}]-(C)"
            pat_b = f"(B)-[rB{rel}]-(C)"

        # Optional WHERE
        where_clause = f"WHERE {where}" if where else ""

        # Optional LIMIT
        limit_clause = "LIMIT $limit" if limit is not None else ""

        # Aggregate: COUNT per common neighbor
        if aggregate:
            if order_by == "count":
                order_clause = f"ORDER BY count {order_direction}"
            else:
                order_clause = (
                    f"ORDER BY {order_by} {order_direction}"
                    if order_by
                    else "ORDER BY count DESC"
                )

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
            # Full relA / relB per pair
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
        limit: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        Common neighbors with identical predicates on edges A–C and B–C.

        Behavior:
        - One-hop common neighbor C
        - rel_conditions duplicated on rA and rB (AND)
        - Optional filter on C (neighbor_where)
        - Optional return_fields

        Use cases:
        - Shared counterparties with thresholds on both transfers
        - Shared ties with time windows on both sides

        Examples:
            # Shared TRANSFER neighbors; base_amt > 400 on both legs
            results = client.common_neighbors_with_rel_filter(
                a=("Account", "node_key", "Collins Steven"),
                b=("Account", "node_key", "Cook Samantha"),
                rel_type="TRANSFER",
                direction="both",
                rel_conditions={"base_amt": (">", 400)},
                return_fields=["C.node_key", "C.acct_id", "rA.base_amt", "rB.base_amt"]
            )

            # Large transfers (>1000) on or after 2025-01-01 on both legs
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
            a, b: (label, key, value) endpoints.
            rel_type: Optional type.
            direction: "out", "in", or "both".
            rel_conditions: {prop: (op, val)}; ANDed and applied to both rA and rB.
            neighbor_where: Raw predicate on C (caller sanitizes).
            return_fields: "C.*", "rA.*", "rB.*" or bare C property.
            order_by / order_direction / limit: sort and cap.

        Returns:
            Projected rows or full commonNeighbor / relA / relB.

        Notes:
            - Same predicates on rA and rB. For asymmetric filters, extend neighbor_where / Cypher.
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

        # A–C / B–C patterns
        if direction == "out":
            pat_a = f"(A)-[rA{rel}]->(C)"
            pat_b = f"(B)-[rB{rel}]->(C)"
        elif direction == "in":
            pat_a = f"(A)<-[rA{rel}]-(C)"
            pat_b = f"(B)<-[rB{rel}]-(C)"
        else:
            pat_a = f"(A)-[rA{rel}]-(C)"
            pat_b = f"(B)-[rB{rel}]-(C)"

        # WHERE
        where_parts = []
        params = {"va": va, "vb": vb}

        # Duplicate rel_conditions onto rA and rB
        if rel_conditions:
            for i, (key, condition) in enumerate(rel_conditions.items()):
                key = self._sanitize_property_key(key)

                if isinstance(condition, (tuple, list)) and len(condition) == 2:
                    operator, value = condition
                    param_name_a = f"rel_cond_a_{i}"
                    param_name_b = f"rel_cond_b_{i}"

                    if operator.upper() in ["IN", "CONTAINS"]:
                        where_parts.append(
                            f"rA.`{key}` {operator.upper()} ${param_name_a}"
                        )
                        where_parts.append(
                            f"rB.`{key}` {operator.upper()} ${param_name_b}"
                        )
                    elif operator.upper() == "STARTS WITH":
                        where_parts.append(f"rA.`{key}` STARTS WITH ${param_name_a}")
                        where_parts.append(f"rB.`{key}` STARTS WITH ${param_name_b}")
                    else:
                        where_parts.append(f"rA.`{key}` {operator} ${param_name_a}")
                        where_parts.append(f"rB.`{key}` {operator} ${param_name_b}")

                    params[param_name_a] = value
                    params[param_name_b] = value
                else:
                    # Equality on both edges
                    param_name_a = f"rel_cond_a_{i}"
                    param_name_b = f"rel_cond_b_{i}"
                    where_parts.append(f"rA.`{key}` = ${param_name_a}")
                    where_parts.append(f"rB.`{key}` = ${param_name_b}")
                    params[param_name_a] = condition
                    params[param_name_b] = condition

        # Predicate on C
        if neighbor_where:
            where_parts.append(f"({neighbor_where})")

        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        # RETURN
        if return_fields:
            return_parts = []
            for field in return_fields:
                # e.g. C.node_key, rA.base_amt
                if "." in field:
                    prefix, prop = field.split(".", 1)
                    prop = self._sanitize_property_key(prop)
                    # Alias: C_node_key, rA_base_amt
                    alias = f"{prefix}_{prop}"
                    return_parts.append(f"{prefix}.`{prop}` AS {alias}")
                else:
                    # Bare name → C.<prop>
                    prop = self._sanitize_property_key(field)
                    return_parts.append(f"C.`{prop}` AS {prop}")
            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            return_clause = "RETURN C AS commonNeighbor, rA AS relA, rB AS relB"

        # Optional ORDER BY
        order_clause = f"ORDER BY {order_by} {order_direction}" if order_by else ""

        # Optional LIMIT
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
    # 4. One-hop filter from a start node
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
        limit: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        One hop from `start` with optional rel/node predicates.

        Args:
            start: (label, key, value).
            rel_type / node_label / direction: pattern shape.
            node_where / rel_where: Raw Cypher fragments (caller must sanitize).
            params: Extra query parameters.
            limit: Max rows.

        Returns:
            [{"start": ..., "rel": ..., "node": ...}, ...]

        Warning:
            Injected strings must be trusted or validated by the caller.
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

        # s–(r)–n pattern
        if direction == "out":
            pat = f"(s)-[r{rel}]->(n{tlabel})"
        elif direction == "in":
            pat = f"(s)<-[r{rel}]-(n{tlabel})"
        else:
            pat = f"(s)-[r{rel}]-(n{tlabel})"

        # WHERE fragments
        where_parts = []
        if rel_where:
            where_parts.append(f"({rel_where})")
        if node_where:
            where_parts.append(f"({node_where})")
        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # Optional LIMIT
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
    # 5. Subgraph extraction
    # =========================================================
    def subgraph_extract(
        self,
        center: Tuple[str, str, Any],
        *,
        hops: int = 2,
        rel_type: Optional[str] = None,
        direction: str = "both",
        where: Optional[str] = None,
        limit_paths: int = 200,
    ) -> JsonDict:
        """
        Ego network around `center` up to `hops` (distinct nodes and rels from paths).

        Args:
            center: (label, key, value).
            hops: Radius (1–5).
            rel_type / direction: pattern constraints.
            where: Raw predicate on path elements.
            limit_paths: Cap on paths collected before unwind.

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

        # Variable-length from center
        if direction == "out":
            pat = f"(c)-[r{rel}*1..{hops}]->(n)"
        elif direction == "in":
            pat = f"(c)<-[r{rel}*1..{hops}]-(n)"
        else:
            pat = f"(c)-[r{rel}*1..{hops}]-(n)"

        # Optional WHERE
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
                "relationships": res[0].get("relationships", []),
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
        where: Optional[str] = None,
    ) -> JsonDict:
        """
        Induced subgraph on a set of nodes keyed by (label, key, values).

        Behavior:
        - Collect all matching nodes
        - Return edges whose endpoints are both in the key set
        - Optional exclusion of self-loops

        Examples:
            # Accounts and TRANSFER edges among three keys
            subgraph = client.subgraph_extract_by_nodes(
                "Account",
                "node_key",
                ["Collins Steven", "Nunez Mitchell", "Lee Alex"],
                rel_type="TRANSFER",
                direction="both"
            )

            # User ego slice without A->A
            subgraph = client.subgraph_extract_by_nodes(
                "User",
                "userId",
                ["u1", "u2", "u3", "u4"],
                rel_type="FOLLOWS",
                include_internal=False  # omit self-loops
            )

        Args:
            label / key / values: identify the node set.
            include_internal: If False, require n1 <> n2.
            rel_type: Optional type filter (None = any).
            direction: "out", "in", or "both".
            where: Extra predicate (caller sanitizes).

        Returns:
            nodes, relationships, node_count, relationship_count.

        Notes:
            - No expansion beyond the given key list.
            - Missing keys are omitted from the MATCH.
        """
        label = self._sanitize_label(label)
        key = self._sanitize_property_key(key)

        if not values or len(values) == 0:
            raise ValueError("values list cannot be empty")

        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")

        # Relationship type token
        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""

        # n1-(r)-n2 pattern
        if direction == "out":
            pattern = f"(n1)-[r{rel}]->(n2)"
        elif direction == "in":
            pattern = f"(n1)<-[r{rel}]-(n2)"
        else:
            pattern = f"(n1)-[r{rel}]-(n2)"

        # Both endpoints in values
        where_parts = []

        where_parts.append("n1.`" + key + "` IN $values")
        where_parts.append("n2.`" + key + "` IN $values")

        if not include_internal:
            where_parts.append("n1 <> n2")

        if where:
            where_parts.append(f"({where})")

        where_clause = "WHERE " + " AND ".join(where_parts)

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
                "relationship_count": res[0].get("relationship_count", 0),
            }

        return {
            "nodes": [],
            "relationships": [],
            "node_count": 0,
            "relationship_count": 0,
        }

    def subgraph_extract_by_rel_filter(
        self,
        rel_type: str,
        rel_conditions: Dict[str, Any],
        *,
        start_label: Optional[str] = None,
        end_label: Optional[str] = None,
        direction: str = "both",
        limit: Optional[int] = None,
    ) -> JsonDict:
        """
        Subgraph from relationships matching property predicates (endpoints + edges).

        Behavior:
        - MATCH rel_type with optional endpoint labels/direction
        - AND-combine rel_conditions
        - Return distinct endpoints and relationships (optionally capped)

        Examples:
            # Transfers on a single day (two bounds on same property — use distinct dict keys in practice)
            subgraph = client.subgraph_extract_by_rel_filter(
                "TRANSFER",
                {"tran_timestamp": (">=", "2025-05-01"),
                 "tran_timestamp": ("<", "2025-05-02")}
            )

            # base_amt in a band
            subgraph = client.subgraph_extract_by_rel_filter(
                "TRANSFER",
                {"base_amt": (">=", 300), "base_amt": ("<=", 500)}
            )

            # SAR-flagged Account-to-Account transfers
            subgraph = client.subgraph_extract_by_rel_filter(
                "TRANSFER",
                {"is_sar": ("=", True)},
                start_label="Account",
                end_label="Account"
            )

        Args:
            rel_type: Relationship type.
            rel_conditions: {prop: (op, val)}.
            start_label / end_label: Optional endpoint labels.
            direction: "out", "in", or "both".
            limit: Max relationships (applied in WITH).

        Returns:
            nodes, relationships, counts.

        Notes:
            - Nodes are union of from/to of matching edges.
            - relationship_count reflects collected edges after LIMIT.
        """
        rel_type = self._sanitize_rel_type(rel_type)

        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")

        # Endpoint patterns
        start_pattern = (
            f"(from:`{self._sanitize_label(start_label)}`)" if start_label else "(from)"
        )
        end_pattern = (
            f"(to:`{self._sanitize_label(end_label)}`)" if end_label else "(to)"
        )

        # Directional relationship pattern
        if direction == "out":
            rel_pattern = f"{start_pattern}-[r:`{rel_type}`]->{end_pattern}"
        elif direction == "in":
            rel_pattern = f"{start_pattern}<-[r:`{rel_type}`]-{end_pattern}"
        else:
            rel_pattern = f"{start_pattern}-[r:`{rel_type}`]-{end_pattern}"

        # WHERE on r.*
        where_parts = []
        params = {}

        if rel_conditions:
            for i, (key, condition) in enumerate(rel_conditions.items()):
                # *_start / *_end key suffix → real property name
                if key.endswith("_start") or key.endswith("_end"):
                    actual_key = key.rsplit("_", 1)[0]
                    actual_key = self._sanitize_property_key(actual_key)
                else:
                    actual_key = self._sanitize_property_key(key)

                if isinstance(condition, (tuple, list)) and len(condition) == 2:
                    operator, value = condition
                    param_name = f"rel_cond_{i}"

                    if operator.upper() in ["IN", "CONTAINS"]:
                        where_parts.append(
                            f"r.`{actual_key}` {operator.upper()} ${param_name}"
                        )
                    elif operator.upper() == "STARTS WITH":
                        where_parts.append(
                            f"r.`{actual_key}` STARTS WITH ${param_name}"
                        )
                    else:
                        where_parts.append(f"r.`{actual_key}` {operator} ${param_name}")

                    params[param_name] = value
                else:
                    # Equality
                    param_name = f"rel_cond_{i}"
                    where_parts.append(f"r.`{actual_key}` = ${param_name}")
                    params[param_name] = condition

        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        # Optional LIMIT
        limit_clause = "LIMIT $limit" if limit is not None else ""
        if limit is not None:
            params["limit"] = limit

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
                "relationship_count": res[0].get("relationship_count", 0),
            }

        return {
            "nodes": [],
            "relationships": [],
            "node_count": 0,
            "relationship_count": 0,
        }

    # =========================================================
    # 6. Ad-hoc path pattern MATCH
    # =========================================================
    def match_path_pattern(
        self,
        *,
        pattern: str,
        where: Optional[str] = None,
        params: Optional[JsonDict] = None,
        limit: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        Run MATCH p = <pattern> with optional WHERE (caller-built Cypher).

        Example:
            pattern = "(a:User {userId:$uid})-[:FOLLOWS]->(b:User)-[:POSTED]->(p:Post)"
            where   = "p.createdAt >= $since"

        Args:
            pattern: Path pattern string.
            where: Optional predicate (no leading WHERE keyword added if empty handling — see code).
            params: Driver parameters.
            limit: Optional row cap.

        Returns:
            [{"path": ...}, ...]

        Warning:
            Untrusted pattern/where strings are injection risk.
        """
        # Optional LIMIT
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
    # 7. Node-level aggregate stats
    # =========================================================
    def aggregate_stats(
        self,
        label: str,
        *,
        group_by: Optional[str] = None,
        where: Optional[str] = None,
        params: Optional[JsonDict] = None,
        metrics: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        Simple aggregation on nodes of a label (optional GROUP BY).

        Examples:
            aggregate_stats("User", group_by="country")
            aggregate_stats(
                "User",
                group_by="city",
                where="n.age >= $minAge",
                params={"minAge": 18}
            )

        Args:
            label: Node label.
            group_by: Property to group by.
            where: Raw WHERE on n.
            params: Extra parameters.
            metrics: Cypher metric expressions (default count(*) AS cnt).
            limit: Row cap.

        Returns:
            List of aggregation rows.
        """
        label = self._sanitize_label(label)
        metrics = list(metrics) if metrics else ["count(*) AS cnt"]

        # Optional LIMIT
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
    # Paths between two nodes (not necessarily shortest)
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
        limit: Optional[int] = None,
    ) -> List[JsonDict]:
        """
        Enumerate paths between A and B with hop bounds, filters, and derived metrics.

        Features:
        - Variable min/max hop count and optional rel_type/direction
        - Rich `where` on `p`, `pathRels`, `pathNodes` (e.g. ANY/ALL over path)
        - Built-in aliases: totalAmount / maxAmount / minAmount / avgAmount on r.base_amt
        - Multiple paths returned (not limited to shortest)

        Examples:
            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                return_fields=["path", "hops", "maxAmount"],
                order_by="maxAmount",
                order_direction="DESC"
            )

            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                where="ANY(r IN relationships(p) WHERE r.is_sar = true)"
            )

            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                return_fields=["path", "hops", "totalAmount"],
                order_by="totalAmount",
                order_direction="DESC"
            )

            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                where="ALL(r IN relationships(p) WHERE r.base_amt < 1000)"
            )

            paths = client.paths_between(
                ("Account", "node_key", "A"),
                ("Account", "node_key", "B"),
                rel_type="TRANSFER",
                where="ALL(n IN nodes(p) WHERE n.bank_id <> 'bank')"
            )

        Args:
            a, b: (label, key, value) endpoints.
            rel_type: Optional type (None = any).
            direction: "out", "in", or "both".
            min_hops / max_hops: Length bounds (max <= 10).
            where: Predicate after WITH ... pathRels (uses `p`, `hops`, etc.).
            return_fields: path, hops, nodes, relationships, or computed amount fields.
            order_by / order_direction / limit: sort and cap paths.

        Returns:
            Rows with path, hops, nodes, relationships, and optional aggregates.

        Notes:
            - Default ORDER BY hops ASC.
            - Empty list if no paths within bounds.
            - Large max_hops can be expensive.
        """
        (la, ka, va) = a
        (lb, kb, vb) = b

        # Validate identifiers and hop bounds
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

        rt = self._sanitize_rel_type(rel_type) if rel_type else ""
        rel = f":`{rt}`" if rt else ""

        # Variable-length pattern A … B
        if direction == "out":
            pattern = f"(A)-[r{rel}*{min_hops}..{max_hops}]->(B)"
        elif direction == "in":
            pattern = f"(A)<-[r{rel}*{min_hops}..{max_hops}]-(B)"
        else:
            pattern = f"(A)-[r{rel}*{min_hops}..{max_hops}]-(B)"

        where_clause = f"WHERE {where}" if where else ""

        if return_fields:
            # Custom RETURN list
            return_parts = []
            for field in return_fields:
                field_lower = field.lower()

                if field_lower == "path":
                    return_parts.append("p AS path")
                elif field_lower == "hops":
                    return_parts.append("hops")
                elif field_lower == "nodes":
                    return_parts.append("pathNodes AS nodes")
                elif field_lower == "relationships":
                    return_parts.append("pathRels AS relationships")

                elif field_lower == "totalamount":
                    return_parts.append(
                        "REDUCE(s = 0, r IN pathRels | s + r.base_amt) AS totalAmount"
                    )

                elif field_lower == "maxamount":
                    return_parts.append(
                        "REDUCE(m = 0, r IN pathRels | CASE WHEN r.base_amt > m THEN r.base_amt ELSE m END) AS maxAmount"
                    )

                elif field_lower == "minamount":
                    return_parts.append(
                        "REDUCE(m = 999999, r IN pathRels | CASE WHEN r.base_amt < m THEN r.base_amt ELSE m END) AS minAmount"
                    )

                elif field_lower == "avgamount":
                    return_parts.append(
                        "REDUCE(s = 0, r IN pathRels | s + r.base_amt) / hops AS avgAmount"
                    )

                else:
                    return_parts.append(field)

            return_clause = "RETURN " + ", ".join(return_parts)
        else:
            return_clause = """RETURN p AS path,
                hops,
                pathNodes AS nodes,
                pathRels AS relationships"""

        # ORDER BY (default hops ASC)
        if order_by:
            order_clause = f"ORDER BY {order_by} {order_direction}"
        else:
            order_clause = "ORDER BY hops ASC"

        limit_clause = "LIMIT $limit" if limit is not None else ""

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
# Demo / manual test entrypoint
# ==========================================
if __name__ == "__main__":
    # Context manager closes the driver
    with Neo4jGraphClient(
        Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password=os.getenv("NEO4J_PASSWORD", ""),
        )
    ) as client:
        # Schema introspection
        schema = client.get_schema()
        print("=== Schema 信息 ===")
        print(f"节点类型: {list(schema['node_labels'].keys())}")
        print(f"关系类型: {list(schema['relationship_types'].keys())}")
        print(f"模式样例: {schema['patterns'][:3]}\n")

        # 1. Lookup by unique key
        print("=== 测试1: 唯一键查节点 ===")
        user = client.get_node_by_unique_key("User", "userId", "u123")
        print(f"用户: {user}\n")

        # 2. N-hop neighbors
        print("=== 测试2: N跳邻居 ===")
        neighbors = client.neighbors_n_hop(
            "User",
            "userId",
            "u123",
            hops=2,
            rel_type="FOLLOWS",
            direction="out",
            limit=10,
        )
        print(f"找到 {len(neighbors)} 个邻居\n")

        # 3. Common neighbors
        print("=== 测试3: 公共邻居 ===")
        common = client.common_neighbors(
            ("User", "userId", "u1"), ("User", "userId", "u2"), rel_type="FOLLOWS"
        )
        print(f"找到 {len(common)} 个公共邻居\n")

        # 4. aggregate_stats
        print("=== 测试4: 聚合统计 ===")
        stats = client.aggregate_stats("User", group_by="country", limit=5)
        print(f"统计结果: {stats}\n")
