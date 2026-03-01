import time
from llama_index.core import QueryBundle
from llama_index.core.utils import print_text
from llama_index.core.response_synthesizers.type import *

from typing import Optional
from aag.expert_search_engine.database.milvus import *
from aag.expert_search_engine.database.entitiesdb import *
from aag.rag_engine.rag import RAG
from aag.config.engine_config import RetrievalConfig
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


class VectorRAG(RAG):

    def __init__(
        self,
        config: RetrievalConfig,
        data_type: str = 'summary'
    ) -> None:
        super().__init__()

        self.config = config

        self.llm_embed_name = self.config.embedding.get("model_name")
        self.embed_batch_size = self.config.embedding.get("batch_size", 32)
        self.device = self.config.embedding.get("device", "cpu")
        self.chunk_size = self.config.embedding.get("chunk_size", 1024)
        self.chunk_overlap = self.config.embedding.get("chunk_overlap", 50)

        # set collection name in ininialize function
        self.collection_name = None
        # self.collection_name = self.config.database.vector.get("collection_name", "test")


        _rag = getattr(self.config, "rag", None) or {}
        self.vector_k_similarity = self.config.rag.vector.get("k_similarity", 5)

        self.host = self.config.database.vector.get("host", "localhost")
        self.port = self.config.database.vector.get("port", 19530)

        # self.llm_env_ = llm_env_
        # if data_type.lower() == 'qa':
        #     self.vector_query_engine = RetrieverQueryEngine.from_args(
        #         self.vector_rag_retriever)
        # elif data_type.lower() == 'summary':
        #     self.vector_query_engine = RetrieverQueryEngine.from_args(
        #         self.vector_rag_retriever, response_mode=ResponseMode.TREE_SUMMARIZE)
        # else:
        #     raise ValueError("Unsupported data type. Use 'qa' or'summary'.")

        self._init_embedding_model()

        

        # prepare data
        self._initialized = False

        print(f"✓ VectorRAG initialized with vector_k_similarity={self.vector_k_similarity}")




    def initialize(self, db_name: str, file_paths: List[str]):
        
        self._init_vector_database(db_name)

        self.vector_db.process_data(file_paths)

        self._initialized = True


    def _init_vector_database(self, db_name: str):

        self.collection_name = db_name

        try:
            self.vector_db = MilvusDB(
                collection_name=self.collection_name,
                top_k=self.vector_k_similarity,
                dim=self.dim,
                host=self.host,
                port=self.port,
            )
            print(f"✓ Vector database initialized: {self.collection_name}")
        except Exception as e:
            print(f"✗ Vector database initialization failed: {e}")
            raise


    def retrieve_deduplication(self):
        raise NotImplementedError        


    def _init_embedding_model(self):
        """初始化嵌入模型(Settings.embed_model/chunk配置)。"""
        try:
            

            if not self.llm_embed_name or not isinstance(self.llm_embed_name, str):
                raise ValueError("embedding.model_name 配置缺失或非法")

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
            self.dim  = self._infer_dim(Settings.embed_model)
            Settings.chunk_size = int(self.chunk_size)
            Settings.chunk_overlap = int(self.chunk_overlap)

            print(
                f"✓ Embedding model initialized: {self.llm_embed_name} | dime={self.dim}, batch={self.embed_batch_size}, device={self.device}, "
                f"chunk_size={Settings.chunk_size}, chunk_overlap={Settings.chunk_overlap}"
            )
        except Exception as e:
            print(f"✗ Embedding model initialization failed: {e}")
            raise

    
    def _infer_dim(self, emb) -> int:
        """
        优先零开销方式（HF 直接读维度）；否则做一次最小嵌入推断长度。
        对 OpenAI 会触发一次 API 调用；对本地 HF 只是一次前向。
        """
        # 1) HuggingFaceEmbedding：拿到底层 sentence-transformers 模型直接读维度
        for attr in ("_model", "model"):
            m = getattr(emb, attr, None)
            if m is not None:
                try:
                    return int(m.get_sentence_embedding_dimension())
                except Exception:
                    pass

        # 2) 通用兜底：做一次极小样本嵌入，取向量长度
        try:
            vec = emb.get_query_embedding("__dim_probe__")
        except Exception:
            vec = emb.get_text_embedding("__dim_probe__")
        return len(vec)
    

    
    def build_index(self, file_path: str):
        self.vector_rag_retriever = self.vector_db.build_index(file_path)

    def retrieve(self, query_str: str, query_id: Optional[int] = -1):
        time_query = -time.time()
        
        node_with_scores = self.vector_db.search(query_str)
        
        time_query += time.time()
        
        self.time_info['time_query'] = time_query

        # print(f"{type(node_with_scores)=}")
        # node_with_scores = self.vector_query_engine.retrieve(QueryBundle(query_str))
        if len(node_with_scores) == 0:
            response = "No information was retrieved, LLM cannot generate a response"
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'retrieve_results': [],
            }
            return [], retrieve_information

        self.time_info["time_retrieve"] = self.time_info['time_query']

        retrieve_results = []
        retrieved_context = []
        for node in node_with_scores:
            retrieve_results.append({
                "node_score": node.get_score(),
                "node_text": node.text,
            })
            retrieved_context.append(node.text)
        retrieve_information = {
            'id': query_id,
            'query': query_str,
            'retrieve_results': retrieve_results,
        }
        return retrieved_context, retrieve_information


    def retrieve_delay(self, query_str: str, query_id: Optional[int] = -1):
        str_or_query_bundle = QueryBundle(query_str)

        time_embeding = -time.time()
        if self.vector_rag_retriever._vector_store.is_embedding_query:
            if str_or_query_bundle.embedding is None and len(str_or_query_bundle.embedding_strs) > 0:
                str_or_query_bundle.embedding = (
                    self.vector_rag_retriever._embed_model.get_agg_embedding_from_queries(
                        str_or_query_bundle.embedding_strs
                    )
                )
        # embedding = self.llm_env_.embed_model.get_text_embedding(query_str)
        time_embeding += time.time()
        self.time_info['time_embeding'] = time_embeding

        time_query = -time.time()
        node_with_scores = self.vector_rag_retriever._get_node_with_embedding(
            str_or_query_bundle)
        # node_with_scores = self.vector_db.retrieve_nodes(query_str, str_or_query_bundle.embedding)
        time_query += time.time()
        self.time_info['time_query'] = time_query

        print(node_with_scores)
        # node_with_scores = self.vector_query_engine.retrieve(QueryBundle(query_str))
        if len(node_with_scores) == 0:
            response = "No information was retrieved, LLM cannot generate a response"
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'retrieve_results': [],
            }
            return [], retrieve_information

        self.time_info["time_retrieve"] = self.time_info['time_embeding'] + \
            self.time_info['time_query']

        retrieve_results = []
        for node in node_with_scores:
            retrieve_results.append({
                "node_score": node.get_score(),
                "node_text": node.text,
            })
        retrieve_information = {
            'id': query_id,
            'query': query_str,
            'retrieve_results': retrieve_results,
        }
        return node_with_scores, retrieve_information

    def generation(self, query_str, nodes):
        time_llm = -time.time()
        response = self.vector_query_engine._response_synthesizer.synthesize(
            query=QueryBundle(query_str),
            nodes=nodes,
        )
        time_llm += time.time()
        self.time_info["time_generation"] = time_llm
        return response

    def query(self, question):
        return self.vector_query_engine.query(question)