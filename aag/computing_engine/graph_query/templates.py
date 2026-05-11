QUERY_TEMPLATES = {
    "node_lookup": {
        "description": "Find a single node by unique key, or filter multiple nodes by property conditions",
        "method": "get_node_by_unique_key / filter_nodes_by_properties",
        "required_params": ["label"],
        "optional_params": ["key", "value", "conditions", "return_fields"],
        "example": "Find user Collins Steven or Query customers living in US with state VT",
        "note": "Supports two modes: 1) Single node exact lookup (requires key+value, uses get_node_by_unique_key) 2) Multiple node property filtering (requires conditions, uses filter_nodes_by_properties)"
    },
    "relationship_filter": {
        "description": "Filter relationships based on relationship property conditions, return relationships and related node information that meet the conditions",
        "method": "filter_relationships",
        "required_params": ["rel_type"],
        "optional_params": ["start_label", "end_label", "rel_conditions", "return_fields", "aggregate"],
        "example": "Find transactions with amount greater than 400 or List the count of transactions where is_sar is False",
        "note": "Used for relationship property filtering, relationship statistics, etc., not for node queries or path queries"
    },
    "aggregation_query": {
        "description": "Aggregation statistical query, supports grouping by node or property, calculates COUNT/SUM/AVG etc., used for ranking and TOP-N queries",
        "method": "aggregation_query",
        "required_params": ["aggregate_type"],
        "optional_params": ["group_by_node", "group_by_property", "node_label", "rel_type", "direction", "aggregate_field", "return_fields"],
        "example": "Count the number of transactions per account or Calculate the total amount of outgoing transactions per account or Count the number of accounts under each branch_id",
        "note": "Used for GROUP BY + aggregation function scenarios, the difference from relationship_filter is: aggregation_query returns grouped aggregation results, relationship_filter returns raw data or global aggregation"
    },
    "neighbor_query": {
        "description": "Query neighbors or N-hop relationships of a node, supports returning detailed information for each edge",
        "method": "neighbors_n_hop",
        "required_params": ["label", "key", "value"],
        "optional_params": ["hops", "rel_type", "direction", "return_fields"],
        "example": "Query neighbors of Collins Steven or Query all transaction details where Collins Steven is the source account",
        "note": "When hops=1 and detailed information for each edge is needed, use return_fields to specify fields to avoid data loss from path aggregation"
    },
    "path_query": {
        "description": "Query paths between two nodes",
        "method": "paths_between",
        "required_params": ["label", "key", "v1", "v2"],
        "optional_params": ["rel_type", "direction", "min_hops", "max_hops"],
        "example": "Path from Collins Steven to Nunez Mitchell"
    },
    "common_neighbor": {
        "description": "Query common neighbors of two nodes, supports relationship property filtering",
        "method": "common_neighbors / common_neighbors_with_rel_filter",
        "required_params": ["label", "key", "v1", "v2"],
        "optional_params": ["rel_type", "direction", "rel_conditions", "return_fields"],
        "example": "Common neighbors of Collins Steven and Nunez Mitchell or Find common transaction neighbors of Steven Collins and Samantha Cook where transaction amounts are all greater than 400",
        "note": "Supports two modes: 1) Simple common neighbor query (no rel_conditions, uses common_neighbors) 2) With relationship property filtering (has rel_conditions, uses common_neighbors_with_rel_filter)"
    },
    "subgraph": {
        "description": "Extract subgraph (supports three modes)",
        "method": "subgraph_extract / subgraph_extract_by_rel_filter",
        "required_params": [],  # Varies by mode
        "optional_params": ["label", "key", "value", "hops", "rel_type", "direction", "limit_paths", "rel_conditions", "start_label", "end_label", "limit"],
        "example": "Subgraph around Collins Steven within 2 hops or Extract subgraph of all transactions on 2025-05-01 or Extract subgraph of transactions with amounts between 300 and 500",
        "note": "Supports three modes: 1) Single center node (requires label+key+value, uses subgraph_extract) 2) Relationship property filtering (requires rel_type+rel_conditions, uses subgraph_extract_by_rel_filter)"
    },
    "subgraph_by_nodes": {
        "description": "Extract subgraph based on node list (multiple specified nodes and their mutual relationships)",
        "method": "subgraph_extract_by_nodes",
        "required_params": ["label", "key", "values"],
        "optional_params": ["include_internal", "rel_type", "direction"],
        "example": "Extract accounts A, B, C and their transfer relationships"
    }
}

# ==========================================
# 通用修饰符定义
# ==========================================
QUERY_MODIFIERS = {
    "order_by": {
        "description": "Sort field (relationship property or node property)",
        "format": "firstRel.property_name or nbr.property_name",
        "example": "firstRel.base_amt, nbr.name"
    },
    "order_direction": {
        "description": "Sort direction",
        "values": ["ASC", "DESC"],
        "default": "ASC"
    },
    "limit": {
        "description": "Limit on number of results returned",
        "type": "integer",
        "example": 10
    },
    "where": {
        "description": "Filter conditions (node or relationship properties)",
        "format": "n.property operator value or r.property operator value",
        "example": "n.balance > 1000, r.amount > 500"
    },
    "aggregate": {
        "description": "Aggregation function",
        "values": ["COUNT", "SUM", "AVG", "MAX", "MIN"],
        "note": "Returns statistical results rather than raw data after applying aggregation"
    },
    "aggregate_field": {
        "description": "Aggregation field (required only when aggregate is not COUNT)",
        "format": "rel.property_name or neighbor.property_name",
        "example": "rel.base_amt"
    }
}
