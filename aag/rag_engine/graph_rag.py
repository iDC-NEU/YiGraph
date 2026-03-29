import time
import re
import difflib
import os
import tempfile
import yaml
# import QueryBundle
from llama_index.core import QueryBundle
from llama_index.core.utils import print_text
from llama_index.core import Settings

from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import *
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from typing import Optional, Dict, List, Any
from aag.utils.pruning import simple_pruning
from aag.rag_engine.rag import RAG
from aag.rag_engine.graph_query.graph_query import Neo4jConfig, Neo4jGraphClient
from aag.expert_search_engine.database.milvus import MilvusDB
from aag.config.engine_config import RetrievalConfig


class _GraphQueryEngineWrapper:
    """Keep old GraphRAG access pattern while using pure synthesizer."""

    def __init__(self, response_synthesizer):
        self._response_synthesizer = response_synthesizer

    def query(self, question):
        raise NotImplementedError(
            "Graph query-by-text retriever is not available in Neo4j adapter mode. "
            "Use retrieve() + generation() pipeline instead."
        )


class Neo4jGraphRAGAdapter:
    """Adapter that provides GraphRAG-compatible APIs on top of Neo4jGraphClient."""

    def __init__(self, config: RetrievalConfig):
        neo4j_cfg = getattr(config.database, "neo4j", {}) or {}

        host = neo4j_cfg.get("host", neo4j_cfg.get("server_ip", "127.0.0.1"))
        port = neo4j_cfg.get("port", neo4j_cfg.get("server_port", 7687))
        uri = neo4j_cfg.get("uri", f"bolt://{host}:{port}")
        user = neo4j_cfg.get("user", neo4j_cfg.get("username", "neo4j"))
        password = neo4j_cfg.get("password", "neo4j")
        database = neo4j_cfg.get("database", None)

        self.node_label = neo4j_cfg.get("node_label", "Account")
        self.entity_key = neo4j_cfg.get("entity_key", "node_key")
        self.relationship_type = neo4j_cfg.get("relationship_type", None)
        self.relationship_key = neo4j_cfg.get("relationship_key", "relationship")
        self.space_name = neo4j_cfg.get("space_name") or database or "neo4j"
        self.entity_limit = int(neo4j_cfg.get("entity_limit", 50000))

        self.client = Neo4jGraphClient(
            Neo4jConfig(uri=uri, user=user, password=password, database=database)
        )
        print(
            "[GraphRAG][Neo4j] config "
            f"label={self.node_label}, entity_key={self.entity_key}, "
            f"relationship_type={self.relationship_type}, database={database}"
        )
        self.entities = self._load_entities()

    def set_retriever(self, **kwargs):
        return None

    def get_space_name(self):
        return self.space_name

    def _load_entities(self) -> List[str]:
        key = self.entity_key
        cypher = (
            f"MATCH (n:`{self.node_label}`) "
            f"WHERE n.`{key}` IS NOT NULL "
            f"RETURN DISTINCT n.`{key}` AS entity "
            f"LIMIT $limit"
        )
        rows = self.client.run(cypher, {"limit": self.entity_limit}, show_query=False)
        entities = []
        for row in rows:
            entity = row.get("entity")
            if entity is None:
                continue
            entities.append(str(entity))
        return sorted(list(set(entities)))

    def _safe_node_name(self, node: Any) -> str:
        try:
            if hasattr(node, "get"):
                name = node.get(self.entity_key)
                if name is None:
                    name = node.get("name")
                if name is not None:
                    return str(name)
            if hasattr(node, "__getitem__"):
                if self.entity_key in node:
                    return str(node[self.entity_key])
                if "name" in node:
                    return str(node["name"])
        except Exception:
            pass
        return str(getattr(node, "id", node))

    def _safe_rel_name(self, rel: Any, default_rel_name: Optional[str] = None) -> str:
        rel_type = getattr(rel, "type", "")
        if callable(rel_type):
            try:
                rel_type = rel_type()
            except Exception:
                rel_type = ""
        try:
            if hasattr(rel, "get"):
                rel_name = rel.get(self.relationship_key)
                if rel_name is not None:
                    return str(rel_name)
                rel_name = rel.get("type")
                if rel_name is not None:
                    return str(rel_name)
            if hasattr(rel, "__getitem__") and self.relationship_key in rel:
                return str(rel[self.relationship_key])
            if hasattr(rel, "__getitem__") and "type" in rel:
                return str(rel["type"])
        except Exception:
            pass
        if isinstance(rel, dict):
            rel_name = rel.get(self.relationship_key) or rel.get("type")
            if rel_name:
                return str(rel_name)
        # Neo4j Relationship fallback: parse type from repr string.
        rel_text = str(rel)
        matched = re.search(r"type='([^']+)'", rel_text)
        if matched:
            return matched.group(1)
        if rel_type:
            return str(rel_type)
        if default_rel_name:
            return str(default_rel_name)
        if self.relationship_type:
            return str(self.relationship_type)
        return "RELATED_TO"

    def _path_to_sequence(self, path: Any, default_rel_name: Optional[str] = None) -> str:
        if path is None:
            return ""
        nodes = list(getattr(path, "nodes", []))
        rels = list(getattr(path, "relationships", []))
        if not nodes:
            return ""
        if not rels:
            return self._safe_node_name(nodes[0])

        sequence_parts = [self._safe_node_name(nodes[0])]
        for idx, rel in enumerate(rels):
            if idx + 1 >= len(nodes):
                break
            next_node = nodes[idx + 1]
            rel_name = self._safe_rel_name(rel, default_rel_name=default_rel_name)
            # Keep subgraph sequence undirected in GraphRAG context.
            arrow = f" -[{rel_name}]- "
            sequence_parts.append(arrow)
            sequence_parts.append(self._safe_node_name(next_node))
        return "".join(sequence_parts)

    def get_rel_map(self, entities: List[str], depth: int = 2, limit: int = 30):
        # Neo4jGraphClient.neighbors_n_hop requires hops in [1, 10].
        try:
            depth = int(depth)
        except Exception:
            depth = 2
        depth = max(1, min(depth, 10))

        rel_map = {}
        for entity in entities:
            query_kwargs = dict(
                hops=depth,
                rel_type=self.relationship_type,
                direction="both",
                limit=limit if limit and limit < 1000000 else None,
                return_distinct=True,
                exclude_start=True,
                return_path_length=True,
            )
            try:
                rows = self.client.neighbors_n_hop(
                    self.node_label,
                    self.entity_key,
                    entity,
                    **query_kwargs,
                )
                # If rel_type filtering yields no rows, retry without rel_type.
                if not rows and self.relationship_type:
                    fallback_kwargs = dict(query_kwargs)
                    fallback_kwargs["rel_type"] = None
                    rows = self.client.neighbors_n_hop(
                        self.node_label,
                        self.entity_key,
                        entity,
                        **fallback_kwargs,
                    )
                    print(
                        f"[GraphRAG][Neo4j] no rows with rel_type={self.relationship_type}, "
                        f"fallback rel_type=None for entity={entity}"
                    )
            except Exception as e:
                print(
                    f"[GraphRAG][Neo4j] neighbors_n_hop failed "
                    f"(entity={entity}, depth={depth}, rel_type={self.relationship_type}): {e}"
                )
                # Last retry without rel_type to avoid silent empty-context failures.
                try:
                    retry_kwargs = dict(query_kwargs)
                    retry_kwargs["rel_type"] = None
                    rows = self.client.neighbors_n_hop(
                        self.node_label,
                        self.entity_key,
                        entity,
                        **retry_kwargs,
                    )
                    print(
                        f"[GraphRAG][Neo4j] retry without rel_type succeeded for entity={entity}, "
                        f"rows={len(rows) if rows else 0}"
                    )
                except Exception as retry_e:
                    print(
                        f"[GraphRAG][Neo4j] retry without rel_type failed "
                        f"(entity={entity}): {retry_e}"
                    )
                    rel_map[entity] = []
                    continue

            sequences = []
            for row in rows:
                rel_type_hint = row.get("relType")
                seq = self._path_to_sequence(
                    row.get("samplePath"),
                    default_rel_name=str(rel_type_hint) if rel_type_hint else None,
                )
                if not seq:
                    neighbor = row.get("neighbor")
                    rel = row.get("rel")
                    if neighbor:
                        rel_name = self._safe_rel_name(
                            rel,
                            default_rel_name=str(rel_type_hint) if rel_type_hint else None,
                        ) if rel is not None else (str(rel_type_hint) if rel_type_hint else "RELATED_TO")
                        seq = f"{entity} -[{rel_name}]- {self._safe_node_name(neighbor)}"
                if seq:
                    sequences.append(seq)
            rel_map[entity] = list(dict.fromkeys(sequences))
        return rel_map

    def clean_rel_map(self, rel_map: Dict[str, List[str]]):
        clean_map = {}
        for entity, sequences in (rel_map or {}).items():
            clean_sequences = [seq.strip() for seq in sequences if isinstance(seq, str) and seq.strip()]
            clean_map[str(entity)] = clean_sequences
        return clean_map

    def get_knowledge_sequence(self, rel_map):
        if not rel_map:
            return []
        return [seq for sequences in rel_map.values() for seq in sequences]

    def _build_nodes(self, knowledge_sequence: List[str], rel_map: Optional[Dict[Any, Any]] = None) -> List[NodeWithScore]:
        if not knowledge_sequence:
            return []
        context_string = (
            "The following are knowledge sequence in the form of graph relation paths:\n"
            "`subject -[predicate]- object`\n"
            + "\n".join(knowledge_sequence)
        )
        rel_node_info = {
            "kg_rel_map": rel_map if rel_map is not None else {},
            "kg_rel_text": knowledge_sequence,
            "kg_backend": "neo4j",
        }
        metadata_keys = ["kg_rel_map", "kg_rel_text", "kg_backend"]
        node = NodeWithScore(
            node=TextNode(
                text=context_string,
                score=1.0,
                metadata=rel_node_info,
                excluded_embed_metadata_keys=metadata_keys,
                excluded_llm_metadata_keys=metadata_keys,
            )
        )
        return [node]

    def get_all_edges(self):
        cypher = f"""
            MATCH (n1)-[e]->(n2)
            RETURN n1.`{self.entity_key}` AS head,
                   coalesce(e.`{self.relationship_key}`, type(e)) AS relation,
                   n2.`{self.entity_key}` AS tail
        """
        rows = self.client.run(cypher, show_query=False)
        triplets = []
        for row in rows:
            head = row.get("head")
            relation = row.get("relation")
            tail = row.get("tail")
            if head is None or tail is None:
                continue
            triplets.append((str(head), str(relation), str(tail)))
        return triplets


class GraphRAG(RAG):

    def __init__(
        self,
        config: RetrievalConfig,
        llm_env_ = None,
        pruning=30,
        data_type: str = 'qa',
        pruning_mode='embedding_for_perentity'  # embedding_for_perentity or embedding
    ) -> None:
        super().__init__()

        self.config = config

        self._init_graph_database()
        self._init_embedding_model()
        self._configure_pruning_runtime()

        self.graph_k_hop = self.config.rag.graph.get("k_hop", 2)

        self.graph_rag_retriever = self.graph_db.set_retriever(
            graph_traversal_depth=self.graph_k_hop, llm_env=llm_env_)

        self.graph_entities = self.graph_db.entities or []
        self.graph_entities_lc = [str(e).strip().lower() for e in self.graph_entities]
        self.graph_entity_lookup = {}
        for entity in self.graph_entities:
            key = self._normalize_entity_text(entity)
            if key and key not in self.graph_entity_lookup:
                self.graph_entity_lookup[key] = str(entity)
        self.entity_match_top_k = self.config.rag.graph.get("entity_match_top_k", 12)
        self._init_entity_retriever()
        if data_type.lower() == 'qa':
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT
            )
            self.kg_query_engine = _GraphQueryEngineWrapper(response_synthesizer)
        elif data_type.lower() == 'summary':
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.TREE_SUMMARIZE
            )
            self.kg_query_engine = _GraphQueryEngineWrapper(response_synthesizer)
            print("sumamry engine")
        else:
            raise ValueError("Unsupported data type. Use 'qa' or'summary'.")
        self.limit = 1000000
        self.depth = self.graph_k_hop
        self.pruning = pruning
        self.pruning_mode = pruning_mode


        print(f"✓ Graph RAG initialized with k_hop={self.graph_k_hop}")

    
    
    def _init_graph_database(self):
        """初始化图数据库连接"""
        try:
            self.graph_db = Neo4jGraphRAGAdapter(self.config)
            print(f"✓ Graph database initialized (Neo4j): {self.graph_db.get_space_name()}")
        except Exception as e:
            print(f"✗ Graph database initialization failed: {e}")
            raise

    def _init_embedding_model(self):
        """Initialize embedding model so MilvusDB can embed query/entities."""
        self.llm_embed_name = self.config.embedding.get("model_name")
        self.embed_batch_size = self.config.embedding.get("batch_size", 32)
        self.device = self.config.embedding.get("device", "cpu")
        self.chunk_size = self.config.embedding.get("chunk_size", 1024)
        self.chunk_overlap = self.config.embedding.get("chunk_overlap", 50)

        if not self.llm_embed_name or not isinstance(self.llm_embed_name, str):
            raise ValueError("The embedding.model_name configuration is missing or invalid")

        name_lower = self.llm_embed_name.lower()
        is_openai = name_lower.startswith("text-embedding") or "openai" in name_lower

        if is_openai:
            Settings.embed_model = OpenAIEmbedding(
                model=self.llm_embed_name,
                embed_batch_size=self.embed_batch_size,
            )
        else:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.llm_embed_name,
                embed_batch_size=self.embed_batch_size,
                device=self.device,
            )
        self.dim = self._infer_dim(Settings.embed_model)
        Settings.chunk_size = int(self.chunk_size)
        Settings.chunk_overlap = int(self.chunk_overlap)

    def _infer_dim(self, emb) -> int:
        for attr in ("_model", "model"):
            m = getattr(emb, attr, None)
            if m is not None:
                try:
                    return int(m.get_sentence_embedding_dimension())
                except Exception:
                    pass
        try:
            vec = emb.get_query_embedding("__dim_probe__")
        except Exception:
            vec = emb.get_text_embedding("__dim_probe__")
        return len(vec)

    def _configure_pruning_runtime(self):
        """
        Keep pruning embedding aligned with GraphRAG embedding model.
        This mirrors VectorRAG behavior without forcing cache directory changes.
        """
        graph_cfg = self.config.rag.graph or {}
        pruning_embed_model = graph_cfg.get("pruning_embed_model") or self.llm_embed_name
        pruning_embed_batch_size = graph_cfg.get("pruning_embed_batch_size", self.embed_batch_size)
        pruning_embed_device = graph_cfg.get("pruning_embed_device", self.device)

        os.environ["AAG_PRUNING_EMBED_MODEL"] = str(pruning_embed_model)
        os.environ["AAG_PRUNING_EMBED_BATCH_SIZE"] = str(pruning_embed_batch_size)
        os.environ["AAG_PRUNING_EMBED_DEVICE"] = str(pruning_embed_device)

        print(
            "[GraphRAG] pruning runtime configured: "
            f"model={os.getenv('AAG_PRUNING_EMBED_MODEL')}, "
            f"device={os.getenv('AAG_PRUNING_EMBED_DEVICE')}"
        )

    def _build_entity_yaml(self, collection_name: str) -> str:
        cache_dir = os.path.join(tempfile.gettempdir(), "aag_graphrag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", collection_name)
        yaml_path = os.path.join(cache_dir, f"{safe_name}.yaml")

        knowledge = []
        for idx, entity in enumerate(self.graph_entities):
            knowledge.append({
                "id": f"entity_{idx}",
                "title": entity,
                "algorithm_id": "entity",
                "type": "entity",
                "url": "",
                "content": entity,
            })
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"knowledge": knowledge}, f, allow_unicode=True, sort_keys=False)
        return yaml_path

    def _init_entity_retriever(self):
        vector_cfg = self.config.database.vector
        collection_name = f"{self.graph_db.get_space_name()}_entities"
        self.entity_retriever = MilvusDB(
            collection_name=collection_name,
            host=vector_cfg.get("host", "127.0.0.1"),
            port=vector_cfg.get("port", 19530),
            dim=self.dim,
            top_k=self.entity_match_top_k,
            overwrite=False,
            verbose=False,
        )
        self.entity_retriever.process_data([self._build_entity_yaml(collection_name)])
        print(f"✓ Graph entity retriever initialized with Milvus: {collection_name}")

    def _search_entities_with_milvus(self, query_text: str, top_k: int) -> List[str]:
        if self.entity_retriever is None or not query_text:
            return []
        old_top_k = getattr(self.entity_retriever, "top_k", self.entity_match_top_k)
        self.entity_retriever.top_k = max(1, int(top_k))
        try:
            nodes = self.entity_retriever.search(query_text)
        finally:
            self.entity_retriever.top_k = old_top_k

        entities = []
        seen = set()
        for node in nodes or []:
            candidates = []
            text = str(getattr(node, "text", "")).strip()
            if text:
                candidates.append(text)
            metadata = getattr(node, "metadata", {}) or {}
            title = metadata.get("title")
            if title:
                candidates.append(str(title).strip())
            for c in candidates:
                key = c.lower()
                if not c or key in seen:
                    continue
                entities.append(c)
                seen.add(key)
        return entities

    def _log_entity_flow(self, extracted_entities, similar_entities, selected_entities):
        print(f"[GraphRAG] extracted entities: {extracted_entities}")
        print(f"[GraphRAG] similar entities (embedding search): {similar_entities}")
        print(f"[GraphRAG] selected entities (after filter): {selected_entities}")

    def _log_subgraph(self, rel_map, stage_name: str, max_entities: int = 3, max_paths: int = 5):
        print(f"[GraphRAG] {stage_name} subgraph entities: {len(rel_map)}")
        for idx, (entity, paths) in enumerate((rel_map or {}).items()):
            if idx >= max_entities:
                break
            show_paths = paths[:max_paths]
            print(f"[GraphRAG]   entity={entity}, paths={len(paths)}, preview={show_paths}")

    @staticmethod
    def _normalize_entity_text(text: str) -> str:
        return str(text).strip().lower()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9_]+", str(text).lower()) if t]

    def _canonicalize_entity(self, entity: str) -> str:
        text = str(entity).strip()
        if not text:
            return ""
        key = self._normalize_entity_text(text)
        if key in self.graph_entity_lookup:
            return self.graph_entity_lookup[key]
        # Keep original text when no canonical match is found.
        return text

    def _canonicalize_entities(self, entities: List[str]) -> List[str]:
        canonical = []
        seen = set()
        for entity in entities or []:
            c = self._canonicalize_entity(entity)
            if not c:
                continue
            key = self._normalize_entity_text(c)
            if key in seen:
                continue
            seen.add(key)
            canonical.append(c)
        return canonical

    def _extract_query_entities(self, query_str: str, max_keywords: int = 3) -> List[str]:
        query_lc = self._normalize_entity_text(query_str)
        # 1) Prefer exact mentions of known graph entities in question text.
        exact_mentions = [
            entity for entity, entity_lc in zip(self.graph_entities, self.graph_entities_lc)
            if entity_lc and entity_lc in query_lc
        ]
        if exact_mentions:
            return exact_mentions[:max_keywords]

        # 2) Fallback to regex phrases (capitalized words / alnum tokens).
        phrases = re.findall(r"[A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+){0,2}", query_str or "")
        phrases = [p.strip() for p in phrases if p.strip()]
        # dedupe preserve order
        seen = set()
        deduped = []
        for p in phrases:
            key = self._normalize_entity_text(p)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
            if len(deduped) >= max_keywords:
                break
        return deduped

    def _expand_similar_entities(self, query_str: str, query_entities: List[str], limit: int = 3) -> List[str]:
        if not self.graph_entities:
            return []

        # VectorRAG-style semantic retrieval on the query text.
        candidates = []
        candidates.extend(
            self._search_entities_with_milvus(query_str, top_k=max(limit * 2, self.entity_match_top_k))
        )
        for q_entity in query_entities or []:
            candidates.extend(
                self._search_entities_with_milvus(q_entity, top_k=max(limit, self.entity_match_top_k // 2))
            )

        # remove duplicates while preserving order
        selected = []
        seen = set()
        for entity in candidates:
            key = self._normalize_entity_text(entity)
            if key in seen:
                continue
            selected.append(entity)
            seen.add(key)
            if len(selected) >= max(1, int(limit)):
                break
        return selected

    def _filter_entities_with_openai(self, query_str: str, entities: List[str]) -> List[str]:
        if not entities:
            return []
        # Local deterministic filter: prioritize explicit mentions in query.
        query_lc = self._normalize_entity_text(query_str)
        mentioned = [e for e in entities if self._normalize_entity_text(e) in query_lc]
        not_mentioned = [e for e in entities if e not in mentioned]
        return mentioned + not_mentioned

    def _resolve_entities(self, query_str: str, entity_num: int = 3, seed_entities: Optional[List[str]] = None):
        extracted_entities = seed_entities or self._extract_query_entities(query_str, max_keywords=3)
        extracted_entities = [str(e).strip() for e in extracted_entities if str(e).strip()]
        similar_entities = self._expand_similar_entities(query_str, extracted_entities, limit=entity_num)
        similar_entities = [str(e).strip() for e in similar_entities if str(e).strip()]
        selected_entities = self._filter_entities_with_openai(query_str, similar_entities)
        selected_entities = [str(e).strip() for e in selected_entities if str(e).strip()]

        extracted_entities = self._canonicalize_entities(extracted_entities)
        similar_entities = self._canonicalize_entities(similar_entities)
        selected_entities = self._canonicalize_entities(selected_entities)
        if not selected_entities:
            selected_entities = similar_entities or extracted_entities
        return extracted_entities, similar_entities, selected_entities
    
    
    def _retrieve_from_entities(self, query_str: str, entities: List[str], query_id: int):
        if len(entities) == 0:
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': "",
                'retrieve_results': {},
            }
            return [], retrieve_information, None

        time_graph_query = time.time()
        rel_map = self.graph_db.get_rel_map(entities=entities, depth=self.depth, limit=self.limit)
        self._log_subgraph(rel_map, stage_name="raw")
        self.time_info["time_graph_query"] = time.time() - time_graph_query

        time_clean_graph_query = -time.time()
        clean_rel_map = self.graph_db.clean_rel_map(rel_map)
        self._log_subgraph(clean_rel_map, stage_name="cleaned")
        knowledge_sequence = self.graph_db.get_knowledge_sequence(clean_rel_map)
        time_clean_graph_query += time.time()
        self.time_info["time_clean_graph_query"] = time_clean_graph_query

        if len(knowledge_sequence) == 0:
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': entities,
                'retrieve_results': {},
            }
            return [], retrieve_information, None

        time_graph_postprocess = -time.time()
        filter_rel_map = None
        if self.pruning_mode == 'embedding':
            nodes, pruning_knowledge_sequence = self.limit_rel_by_embedding_similarity(
                query_str, knowledge_sequence
            )
            retrieve_results = pruning_knowledge_sequence
        elif self.pruning_mode == 'embedding_for_perentity':
            all_pruning_knowledge_sequence = []
            filter_rel_map = {}
            for entity, sequences in clean_rel_map.items():
                _, per_entity_pruning = self.limit_rel_by_embedding_similarity(
                    query_str, sequences
                )
                filter_rel_map[entity] = per_entity_pruning
                all_pruning_knowledge_sequence.extend(per_entity_pruning)
            self._log_subgraph(filter_rel_map, stage_name="pruned")
            pruning_knowledge_dict = {"pruning": all_pruning_knowledge_sequence}
            nodes = self.graph_db._build_nodes(all_pruning_knowledge_sequence, pruning_knowledge_dict)
            retrieve_results = filter_rel_map
        else:
            nodes = self.graph_db._build_nodes(knowledge_sequence, rel_map)
            retrieve_results = knowledge_sequence

        time_graph_postprocess += time.time()
        self.time_info["time_graph_postprocess"] = time_graph_postprocess
        self.time_info["time_retrieve"] = (
            self.time_info["time_extract_entities"] +
            self.time_info["time_graph_query"] +
            self.time_info["time_graph_postprocess"]
        )

        retrieve_information = {
            'id': query_id,
            'query': query_str,
            'entities': entities,
            'retrieve_results': retrieve_results,
        }
        return nodes, retrieve_information, filter_rel_map

    def retrieve(self, query_str: str, query_id: Optional[int] = -1, entity_num=3):
        self.clear_time_info()
        time_extract_entities = time.time()
        extracted_entities, similary_entities, entities = self._resolve_entities(
            query_str=query_str, entity_num=entity_num
        )
        print(f"similary_entities:{similary_entities}")
        print(f"filter_entities:{entities}")
        self._log_entity_flow(extracted_entities, similary_entities, entities)
        self.time_info["time_extract_entities"] = time.time() - time_extract_entities

        nodes, retrieve_information, _ = self._retrieve_from_entities(query_str, entities, query_id)
        return nodes, retrieve_information

    def retrieve_deduplication(self, query_str: str, query_id: Optional[int] = -1, entity_num=3):
        # Keep interface compatibility; use standard retrieval for now.
        nodes, retrieve_information = self.retrieve(query_str, query_id, entity_num)
        deduplication_nodes = None
        deduplication_rate = 0.0
        return nodes, retrieve_information, deduplication_nodes, deduplication_rate

    def retrieve_based_entity(self, query_str: str, entities, query_id: Optional[int] = -1, entity_num=10):
        self.clear_time_info()
        time_extract_entities = time.time()
        extracted_entities, similary_entities, selected_entities = self._resolve_entities(
            query_str=query_str, entity_num=entity_num, seed_entities=entities
        )
        print(f"similary_entities:{similary_entities}")
        print(f"filter_entities:{selected_entities}")
        self._log_entity_flow(extracted_entities, similary_entities, selected_entities)
        self.time_info["time_extract_entities"] = time.time() - time_extract_entities

        nodes, retrieve_information, _ = self._retrieve_from_entities(
            query_str, selected_entities, query_id
        )
        return nodes, retrieve_information

    def build_knowledge_sequence(self, rel_map):
        knowledge_sequence = []
        for key, values in rel_map.items():
            for value in values:
                per_sentence = str(key) + " " + str(value)
                knowledge_sequence.append(per_sentence)
        return knowledge_sequence

    def limit_rel_by_numbers(self, knowledge_sequence, limit=5):
        return knowledge_sequence[:limit]

    def limit_rel_by_embedding_similarity(self, question, knowledge_sequence):
        # time_kg_pruning = -time.time()
        time_kg_pruning = -time.time()
        sorted_all_rel_scores = simple_pruning(question,
                                               knowledge_sequence,
                                               topk=self.pruning)
        time_kg_pruning += time.time()
        print(f'kg pruning time: {time_kg_pruning:.3f}')

        pruning_knowledge_sequence = [rel for rel, _ in sorted_all_rel_scores]
        pruning_knowledge_dict = {"pruning": pruning_knowledge_sequence}

        nodes = self.graph_db._build_nodes(pruning_knowledge_sequence,
                                           pruning_knowledge_dict)
        return nodes, pruning_knowledge_sequence

    def generation(self, query_str, nodes):
        time_llm = -time.time()
        response = self.kg_query_engine._response_synthesizer.synthesize(
            query=QueryBundle(query_str),
            nodes=nodes,
        )
        time_llm += time.time()
        self.time_info["time_generation"] = time_llm
        # 打印 time_info["time_generation"]
        print(f"time_info['time_generation']: {self.time_info['time_generation']:.3f}")
        return response

    def query(self, question):
        nodes, _ = self.retrieve(question)
        if not nodes:
            return "No information was retrieved, LLM cannot generate a response"
        return self.generation(question, nodes)
    
    def get_all_edges(self):
        """
                Retrieve all edges from Neo4j, representing the edge table of the entire graph.
            Returns:
                List[Tuple[str, str, str]]: Each element is a tuple of (head, relation, tail).
        """
        return self.graph_db.get_all_edges()
