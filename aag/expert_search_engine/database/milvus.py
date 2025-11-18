from pymilvus import connections, Collection
from llama_index.vector_stores.milvus import MilvusVectorStore
import time
import yaml
import os
from typing import (
    # Optional, Dict,
    List)
from llama_index.core.utils import print_text
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings
from pymilvus import (
    connections,
    # utility,
    # FieldSchema,
    # CollectionSchema,
    DataType,
    Collection,
    #Milvus,
    MilvusClient,
    # Connections,
)
from llama_index.core.schema import NodeWithScore, QueryBundle

from llama_index.core.retrievers import VectorIndexRetriever

fmt = "\n=== {:30} ===\n"


class myMilvus(MilvusClient):

    def __init__(self, host='127.0.0.1', port='19530', **kwargs):
        super().__init__(host=host, port=port, **kwargs)

    def show_all_collections(self):
        ret = self.list_collections()
        print(f'=== all collections name: {ret}')

    def show_collections_stats(self, collection_name):
        ret = self.get_collection_stats(collection_name)
        print(f'=== stat of {collection_name}: {ret}')

    def show_collections_schema(self, collection_name):
        ret = self.describe_collection(collection_name)
        print(f'=== schema of {collection_name}: {ret}')

    # def drop(self, collection_name):
    #     ret = self.drop_collection(collection_name)
    #     print(f'=== clear collection {collection_name}: {ret}')

    # def exist(self, collection_name):
    #     return self.has_collection(collection_name)

    def get_vector_count(self, collection_name):
        ret = self.get_collection_stats(collection_name)
        return ret['row_count']


class MilvusClientTool:

    def __init__(
        self,
        server_ip='127.0.0.1',
        server_port='19530',
    ):
        self.client = MilvusClient(uri=f"http://{server_ip}:{server_port}")

    def show_all_collections(self):
        print(fmt.format('all collections name'))
        ret = self.client.list_collections()
        print(ret)

    def show_collections_stats(self):
        print(fmt.format(f'stat of {self.collection_name}'))
        ret = self.client.get_collection_stats(self.collection_name)
        print(ret)

    def show_collections_schema(self):
        print(fmt.format(f'schema of {self.collection_name}'))
        ret = self.client.describe_collection(self.collection_name)
        print(ret)

    def clear(self, collection_name):
        print(fmt.format(f'clear collection {collection_name}'))
        self.client.drop_collection(collection_name)


class MilvusDB:

    def __init__(self,
                 collection_name,
                 dim,
                 overwrite=False,
                 server_ip='127.0.0.1',
                 server_port='19530',
                 log_file='./database/milvus.log',
                 store=False,
                 verbose=False,
                 metric='COSINE',
                 retriever=False):
        self.collection_name = collection_name
        self.dim = dim
        self.overwrite = overwrite
        self.server_ip = server_ip
        self.server_port = server_port
        self.log_file = log_file

        # self.client = Milvus(server_ip, server_port)
        self.client = MilvusClient(uri=f"http://{server_ip}:{server_port}")

        self.store = None
        self.storage_context = None
        self.db = None
        self.verbose = verbose
        self.metric = metric

        self.index = None
        self.retriever = None

        connections.connect("default", host="localhost", port="19530")

        if store:
            self.init_store()

        if retriever:
            index = self.get_vector_index()
            self.retriever = VectorIndexRetriever(index=index)

    def get_storage_context(self):
        return self.storage_context

    def init_store(self):
        self.store = MilvusVectorStore(dim=self.dim,
                                       collection_name=self.collection_name,
                                       overwrite=self.overwrite)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.store)

    def get_vector_index(self):
        if not self.index:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.store)
        return self.index

    def show_all_collections(self):
        print(fmt.format('all collections name'))
        ret = self.client.list_collections()
        print(ret)

    def show_collections_stats(self):
        print(fmt.format(f'stat of {self.collection_name}'))
        ret = self.client.get_collection_stats(self.collection_name)
        # ret = self.client.num_entities(self.collection_name)
        print(ret)

    def show_collections_schema(self):
        print(fmt.format(f'schema of {self.collection_name}'))
        ret = self.client.describe_collection(self.collection_name)
        print(ret)

    # def insert(self, embedding):
    #     self.client.insert(self.collection_name, embedding)

    def insert(self, entities):
        time_s = time.time()
        # self.client.insert(self.collection_name, entities)
        self.db.insert(entities)
        time_e = time.time()
        if self.verbose:
            print(f'insert cost {time_e - time_s}')

        self.db.load()

    def load(self):
        if not self.db:
            self.db = Collection(self.collection_name)
            self.db.load()

    def search(self, embedding, limit=3, min_distance=0.5):
        # self.db.load()
        # time_s = time.time()
        # self.db.flush()
        # time_e = time.time()
        # print(f'time cost {time_e - time_s:.3f}')
        search_params = {
            "metric_type": self.metric,
            "params": {
                "nprobe": 10
            },
        }
        start_time = time.time()
        result = self.db.search(embedding, "vec", search_params, limit=limit)
        end_time = time.time()

        # logging.info(f'embedding {embedding}, search cost {end_time-start_time:.3f}')

        # print(type(result), type(result[0]), type(result[0][0]))

        distance = [hit.distance for hits in result for hit in hits]
        pk = [hit.pk for hits in result for hit in hits]

        if self.verbose:
            print(f'search cost {end_time-start_time:.3f}')

        # for hits in result:
        #     print(hits)
        #     for hit in hits:
        #         print(f"hit: {hit}, random field: {hit.entity.get('distance')}, {hit.distance}")

        return pk, distance

    def clear(self):
        print(fmt.format(f'clear collection {self.collection_name}'))
        ret = self.client.drop_collection(self.collection_name)
        return ret

    def create(self, consistency_level="Session"):

        # connections.connect("default", host="localhost", port="19530")

        if self.overwrite and self.collection_name in self.client.list_collections(
        ):
            self.client.drop_collection(self.collection_name)

        # fields = [
        #     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        #     # FieldSchema(name="random", dtype=DataType.DOUBLE),
        #     FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        # ]

        fields = {
            "fields": [{
                "name": "pk",
                "type": DataType.INT64,
                "is_primary": True
            }, {
                "name": "vec",
                "type": DataType.FLOAT_VECTOR,
                "params": {
                    "dim": self.dim
                }
            }],
            "auto_id":
            False
        }

        self.client.create_collection(self.collection_name,
                                      fields,
                                      consistency_level=consistency_level
                                      # Strong, Bounded, Eventually, Session
                                      )

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": self.metric,
            "params": {
                "nlist": 128
            },
        }
        # self.db.create_index("embeddings", index)

        # schema = CollectionSchema(fields, "customize schema")

        # self.db = Collection(self.collection_name, schema)

        self.client.create_index(self.collection_name, 'vec', index)

        # print(Connections().list_connections)

        # print('has', self.client.has_collection(self.collection_name))

        # self.db = Collection(self.collection_name, consistency_level=consistency_level)
        self.db = Collection(self.collection_name)
        self.db.load()

    def process_docs(self, documents, data_dir='./storage_vector', cache=True):
        # TODO: use bge to generate vector

        # filter documents
        # filter_documents = [doc for doc in documents if not is_file_processed(self.log_file, doc.id_)]
        # print(f'filter {len(documents) - len(filter_documents)} documents, last {len(filter_documents)} documents.')
        # documents = filter_documents

        # if len(documents) == 0:
        #     return

        # vec_index = VectorStoreIndex.from_documents(
        #     documents,
        #     storage_context=self.storage_context,
        #     show_progress=True
        # )

        # for doc in documents:
        #     append_log(self.log_file, doc.id_)

        # return vec_index
        index_loaded = False
        if cache:
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=data_dir, vector_store=self.store)
                vector_index = load_index_from_storage(
                    storage_context=storage_context,
                    # service_context=service_context,
                    show_progress=True,
                )
                index_loaded = True
                print(f"vector index load from {data_dir}.")
                return vector_index
            except Exception:
                index_loaded = False

        if not index_loaded:
            vector_index = VectorStoreIndex.from_documents(
                documents=documents,
                # service_context=service_context,
                show_progress=True,
                storage_context=self.storage_context,
            )

        if cache:
            vector_index.storage_context.persist(persist_dir=data_dir)
            print(f"vector index store to {data_dir}.")
        return vector_index

    def set_retriever(self, retriever):
        self.retriever = retriever

    def set_retriever_with_similarity_topk(self, similarity_top_k=2):
        self.retriever = VectorIndexRetriever(
            index=self.get_vector_index(),
            similarity_top_k=similarity_top_k
        )
        return self.retriever

    def retrieve_nodes(self, query, embedding) -> List[NodeWithScore]:
        assert self.retriever, 'please use set_retriever() to init retriever!'

        query_bundle = QueryBundle(query_str=query, embedding=embedding)
        print(query_bundle.query_str)
        # query_bundle.query_str = query
        # query_bundle.embedding = embedding
        nodes = self.retriever._retrieve(query_bundle=query_bundle)
        return nodes


class MilvusDB2:
    """
    MilvusDB2 - 基于LlamaIndex的Milvus向量数据库封装类
    支持自动构建索引、文档加载和语义搜索功能
    """
    
    def __init__(self, 
                 collection_name: str,
                 dim: int = 1024,
                 host: str = '127.0.0.1',
                 port: str = '19530',
                 index_params: dict = None,
                 search_params: dict = None,
                 metric: str = 'COSINE',
                 verbose: bool = False):
        """
        初始化MilvusDB2
        
        Args:
            collection_name: 集合名称
            dim: 向量维度
            server_ip: Milvus服务器IP
            server_port: Milvus服务器端口
            index_params: 索引参数
            search_params: 搜索参数
            metric: 距离度量方式
            verbose: 是否显示详细信息
        """
        self.collection_name = collection_name
        self.dim = dim
        self.server_ip = host
        self.server_port = port
        self.metric = metric
        self.verbose = verbose
        self.overwrite = True
        
        # 设置默认的索引参数
        if index_params is None:
            self.index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": metric,
                "params": {"nlist": 128}
            }
        else:
            self.index_params = index_params
            
        # 设置默认的搜索参数
        if search_params is None:
            self.search_params = {
                "metric_type": metric,
                "params": {"nprobe": 10}
            }
        else:
            self.search_params = search_params
        
        # 初始化组件
        self.client = None
        self.store = None
        self.storage_context = None
        self.index = None
        self.retriever = None
        
        # 连接数据库
        self._connect_database()

        # create new database when not exist
        # print(f"{self.collection_name=}")
        if self.collection_name not in self.client.list_collections() or True:
            self.create()
            print(f"create new vector database {self.collection_name}")
        
        # 初始化向量存储
        self._init_vector_store()
    
    
    def create(self, consistency_level="Session"):

        # connections.connect("default", host="localhost", port="19530")

        if self.overwrite and self.collection_name in self.client.list_collections(
        ):
            self.client.drop_collection(self.collection_name)

        # 1. Create schema
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )


        # 2. Add fields to schema
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        # schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)


        index_params = self.client.prepare_index_params()

        # 4. Add indexes

        index_params.add_index(
            field_name="vec", 
            index_type="AUTOINDEX",
            metric_type=self.metric
        )


        # 5. Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )


        self.db = Collection(self.collection_name)
        self.db.load()
    
    
    def _connect_database(self):
        """连接Milvus数据库"""
        try:
            # 连接Milvus
            connections.connect(
                alias="default", 
                host=self.server_ip, 
                port=self.server_port
            )
            
            # 创建Milvus客户端
            self.client = MilvusClient(uri=f"http://{self.server_ip}:{self.server_port}")
            
            if self.verbose:
                print(f"Successfully connected to Milvus at {self.server_ip}:{self.server_port}")
                
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise
    
    def _init_vector_store(self):
        """初始化MilvusVectorStore"""
        try:
            self.store = MilvusVectorStore(
                dim=self.dim,
                collection_name=self.collection_name,
                index_params=self.index_params,
                search_params=self.search_params,
                overwrite=False  # 不覆盖现有集合
            )
            
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.store
            )
            
            if self.verbose:
                print(f"Successfully initialized MilvusVectorStore for collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Failed to initialize MilvusVectorStore: {e}")
            raise
    
    def build_index(self, file_path: str):
        """
        构建索引
        
        Args:
            file_path: 文件路径，支持YAML配置文件
            
        Returns:
            VectorStoreIndex: 构建的向量索引
        """
        try:
            # 检查集合是否为空
            if self._is_collection_empty():
                if self.verbose:
                    print(f"Collection {self.collection_name} is empty, building index from files...")
                
                # 从文件加载文档并构建索引
                documents = self._load_documents_from_file(file_path)
                
                if not documents:
                    print("No documents found to build index")
                    return None
                
                # 构建向量索引
                self.index = VectorStoreIndex.from_documents(
                    documents=documents,
                    storage_context=self.storage_context,
                    show_progress=True
                )
                
                if self.verbose:
                    print(f"Successfully built index with {len(documents)} documents")
                    
            else:
                if self.verbose:
                    print(f"Collection {self.collection_name} is not empty, loading existing index...")
                
                # 从现有向量存储加载索引
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.store
                )
                
                if self.verbose:
                    print("Successfully loaded existing index")
            
            return self.index
            
        except Exception as e:
            print(f"Failed to build index: {e}")
            raise
    
    def _is_collection_empty(self) -> bool:
        """检查集合是否为空"""
        try:
            if self.collection_name in self.client.list_collections():
                stats = self.client.get_collection_stats(self.collection_name)
                return stats.get('row_count', 0) == 0
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error checking collection status: {e}")
            return True
    
    def _load_documents_from_file(self, file_path: str):
        """
        从文件加载文档
        
        Args:
            file_path: 文件路径，支持YAML配置文件
            
        Returns:
            List[Document]: 加载的文档列表
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        
        # 检查文件扩展名
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # TODO(sanzo): support different file formats
        if file_extension in ['.yaml', '.yml']:
            # YAML文件
            return self._load_from_yaml_config(file_path)
        elif file_extension in ['.md', '.txt']:
            return self._load_from_raw_text(file_path)
        else:
            # 其他文件类型，暂时忽略
            print(f"File type {file_extension} is not supported yet. Only YAML files are supported.")
            return []
    
    def _load_from_yaml_config(self, yaml_file: str):
        """从YAML配置文件加载文档"""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            documents = []
            
            # 处理knowledge格式的YAML文件
            if 'knowledge' in config:
                for item in config['knowledge']:
                    # 从每个knowledge item创建Document对象
                    doc = Document(
                        text=item.get('content', ''),
                        metadata={
                            'id': item.get('id', ''),
                            'title': item.get('title', ''),
                            'algorithm_id': item.get('algorithm_id', ''),
                            'type': item.get('type', ''),
                            'url': item.get('url', '')
                        }
                    )
                    documents.append(doc)
                    
                    if self.verbose:
                        print(f"Loaded document: {item.get('id', 'Unknown')} - {item.get('title', 'No title')}")
            
            if self.verbose:
                print(f"Loaded {len(documents)} documents from YAML config")
                
            return documents
            
        except Exception as e:
            print(f"Error loading from YAML config: {e}")
            return []
    
    def _load_from_raw_text(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        return [Document(text=text)]
    
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0, candidate_algorithms: List[str] = None):
        """
        搜索功能 - 使用LlamaIndex实现先过滤再搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            candidate_algorithms: 候选算法ID列表，用于过滤搜索结果。如果为None，则进行全局搜索
            
        Returns:
            List[NodeWithScore]: 搜索结果
        """
        try:
            if not self.index:
                print("Index not built yet. Please call build_index() first.")
                return []
            
            # 创建检索器
            if not self.retriever:
                self.retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=top_k
                )
            
            # 如果有候选算法过滤条件，使用LlamaIndex的元数据过滤
            if candidate_algorithms is not None:
                return self._search_with_llamaindex_filter(query, top_k, similarity_threshold, candidate_algorithms)
            
            # 否则进行全局搜索
            query_bundle = QueryBundle(query_str=query)
            nodes = self.retriever.retrieve(query_bundle)
            
            # 过滤相似度阈值
            if similarity_threshold > 0:
                nodes = [node for node in nodes if node.score >= similarity_threshold]
            
            if self.verbose:
                print(f"Found {len(nodes)} results for query: {query}")
                for i, node in enumerate(nodes):
                    algorithm_id = node.metadata.get('algorithm_id', 'Unknown')
                    print(f"Result {i+1} (score: {node.score:.4f}, algorithm: {algorithm_id}): {node.text[:100]}...")
            
            return nodes
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def _search_with_llamaindex_filter(self, query: str, top_k: int, similarity_threshold: float, candidate_algorithms: List[str]):
        """
        使用LlamaIndex实现过滤搜索 - 先过滤再搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            candidate_algorithms: 候选算法ID列表
            
        Returns:
            List[NodeWithScore]: 搜索结果
        """
        try:
            
            if self.verbose:
                print(f"Using LlamaIndex filter for algorithms: {candidate_algorithms}")
            
            # 创建元数据过滤器
            # 使用 "in" 操作符来匹配候选算法列表中的任何一个
            metadata_filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="algorithm_id",
                        value=candidate_algorithms,
                        operator="in"
                    )
                ]
            )
            
            # 创建带过滤器的检索器
            filtered_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
                filters=metadata_filters
            )
            
            # 执行过滤搜索
            query_bundle = QueryBundle(query_str=query)
            nodes = filtered_retriever.retrieve(query_bundle)
            
            # 过滤相似度阈值
            if similarity_threshold > 0:
                nodes = [node for node in nodes if node.score >= similarity_threshold]
            
            if self.verbose:
                print(f"Found {len(nodes)} results for query: {query}")
                for i, node in enumerate(nodes):
                    algorithm_id = node.metadata.get('algorithm_id', 'Unknown')
                    print(f"Result {i+1} (score: {node.score:.4f}, algorithm: {algorithm_id}): {node.text[:100]}...")
            
            return nodes
            
        except Exception as e:
            print(f"Error during LlamaIndex filtered search: {e}")
            # 如果LlamaIndex过滤失败，回退到后过滤方式
            return self._search_with_post_filter(query, top_k, similarity_threshold, candidate_algorithms)
    
    def _search_with_post_filter(self, query: str, top_k: int, similarity_threshold: float, candidate_algorithms: List[str]):
        """
        后过滤搜索 - 先搜索再过滤（备用方案）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            candidate_algorithms: 候选算法ID列表
            
        Returns:
            List[NodeWithScore]: 搜索结果
        """
        try:
            if self.verbose:
                print(f"Using post-filter approach for algorithms: {candidate_algorithms}")
            
            # 先进行全局搜索，获取更多结果
            expanded_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k * 3  # 获取更多结果以便过滤
            )
            
            query_bundle = QueryBundle(query_str=query)
            nodes = expanded_retriever.retrieve(query_bundle)
            
            # 应用算法过滤
            filtered_nodes = []
            for node in nodes:
                node_algorithm_id = node.metadata.get('algorithm_id', '')
                if node_algorithm_id in candidate_algorithms:
                    filtered_nodes.append(node)
            
            # 限制结果数量
            filtered_nodes = filtered_nodes[:top_k]
            
            # 过滤相似度阈值
            if similarity_threshold > 0:
                filtered_nodes = [node for node in filtered_nodes if node.score >= similarity_threshold]
            
            if self.verbose:
                print(f"Found {len(filtered_nodes)} results for query: {query}")
                for i, node in enumerate(filtered_nodes):
                    algorithm_id = node.metadata.get('algorithm_id', 'Unknown')
                    print(f"Result {i+1} (score: {node.score:.4f}, algorithm: {algorithm_id}): {node.text[:100]}...")
            
            return filtered_nodes
            
        except Exception as e:
            print(f"Error during post-filter search: {e}")
            return []
    
    def get_collection_stats(self):
        """获取集合统计信息"""
        try:
            if self.collection_name in self.client.list_collections():
                stats = self.client.get_collection_stats(self.collection_name)
                return stats
            return {"row_count": 0}
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"row_count": 0}
    
    def show_collection_info(self):
        """显示集合信息"""
        print(f"\n=== Collection: {self.collection_name} ===")
        stats = self.get_collection_stats()
        print(f"Row count: {stats.get('row_count', 0)}")
        print(f"Dimension: {self.dim}")
        print(f"Metric: {self.metric}")
        print(f"Index params: {self.index_params}")
        print(f"Search params: {self.search_params}")



def test_milvusdb2():
    """测试MilvusDB2类的功能"""
    print("=== Testing MilvusDB2 ===")
    
    # 创建MilvusDB2实例
    db2 = MilvusDB2(
        collection_name="test_collection",
        dim=1024,
        verbose=True
    )
    
    # 显示集合信息
    db2.show_collection_info()
    
    # 构建索引（如果集合为空）
    # 示例：从YAML配置加载文档
    # index = db2.build_index("/home/chency/GraphLLM/aag/knowledge_base/knowledge.yaml")
    
    # 搜索示例
    # results = db2.search("your query here", top_k=5)
    # for result in results:
    #     print(f"Score: {result.score}, Text: {result.text[:100]}...")
    
    print("MilvusDB2 test completed!")


def test_db(db_name):
    vector_db = MilvusDB('cache', 4, overwrite=True, metric='COSINE')

    vector_db.create()
    print("create done!")

    print(f'schema {vector_db.db.schema}')

    vector_db.show_collections_stats()
    pk, dis = vector_db.search([[1, 1, 1, 1]], limit=10)
    print(f'pk {pk}, dis {dis}')

    vector_db.insert([[1], [[1, 1, 1, 1]]])
    vector_db.insert([[2], [[2, 2, 2, 2]]])
    vector_db.insert([[3], [[3, 3, 3, 3]]])
    pk, dis = vector_db.search([[1, 1, 1, 1]], limit=10)
    print(f'pk {pk}, dis {dis}')

    vector_db.insert([[5, 6, 7, 8],
                      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8,
                                                                  8]]])
    pk, dis = vector_db.search([[7, 7, 7, 7]], limit=10)
    print(f'pk {pk}, dis {dis}')

    vector_db.show_collections_stats()


# def test_retrieve_nodes(db_name):
#     from utils.llm_env import OllamaEnv
#     OllamaEnv(llm_mode_name='llama2:7b', timeout=200)

#     vector_db = MilvusDB(db_name, 1024, overwrite=False, store=True)
#     vector_db.show_collections_stats()

#     question = "Tell me about Lebron James?"

#     embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

#     embedding = embed_model.get_text_embedding(question)

#     # vector_index = vector_db.get_vector_index()
#     # vector_retriever = VectorIndexRetriever(index=vector_index)
#     # vector_db.set_retriever(vector_retriever)

#     nodes = vector_db.retrieve_nodes(question, embedding)

#     print(f'nodes:\n{nodes}')
#     for node in nodes:
#         print_text(f'{node.text}\n', color='yellow')


def create_datebase(db_name, dim=1024):
    vector_db = MilvusDB(db_name, dim, overwrite=True, store=True)
    vector_db.show_collections_stats()
    vector_db.show_all_collections()


if __name__ == '__main__':

    # test_db('cache')

    db_name = 'rgb'
    dim = 1024

    # # create vector database
    # # create_datebase(db_name='crag_small')

    vector_db = MilvusDB("rgb", dim,
                         overwrite=False, store=True)
    vector_db.show_all_collections()
    # vector_db.clear()
    # vector_db.show_all_collections()

    # test_retrieve_nodes(db_name)

    # 连接到 Milvus 服务
    connections.connect(alias="rgb",
                        host="127.0.0.1", port="19530")

    # 指定集合名称
    collection_name = "postdragonball"

    # 获取集合对象
    collection = Collection(name=collection_name)

    # 获取集合中的元素个数
    element_count = collection.num_entities

    print(f"Collection '{collection_name}' contains {element_count} entities.")
