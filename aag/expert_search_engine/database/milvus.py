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
from llama_index.vector_stores.milvus import MilvusVectorStore

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



class MilvusDB:
    """
    MilvusDB2 - 基于LlamaIndex的Milvus向量数据库封装类
    支持自动构建索引、文档加载和语义搜索功能
    """
    
    def __init__(self, 
                 collection_name: str,
                 dim: int = 1024,
                 host: str = '127.0.0.1',
                 port: str = '19530',
                 consistency_level="Bounded",
                 overwrite: bool = False,
                 top_k: int = 3,
                 verbose: bool = False,
                 ):
        """
        初始化MilvusDB2
        
        Args:
            collection_name: 集合名称
            dim: 向量维度
            server_ip: Milvus服务器IP
            server_port: Milvus服务器端口
            metric: 距离度量方式
            verbose: 是否显示详细信息
        """
        self.collection_name = collection_name
        self.consistency_level = consistency_level
        self.dim = dim
        self.server_ip = host
        self.server_port = port
        self.verbose = verbose
        self.overwrite = overwrite
        self.top_k = top_k
        
        self.index = None
        self.retriever = None
        self.uri = f"http://{self.server_ip}:{self.server_port}"

        self.store = MilvusVectorStore(uri=self.uri, dim=dim, overwrite=self.overwrite, collection_name=self.collection_name, consistency_level=self.consistency_level)

        # print(f"{vector_store.client.get_collection_stats(collection_name='test_demo')=}")
        self.storage_context = StorageContext.from_defaults(vector_store=self.store)
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.store)
        self.retriever = self.index.as_retriever()

    
    def process_data(self, file_paths: List[str]):

        if self.collection_name not in self.store.client.list_collections() \
            or self.overwrite or self.get_collection_stats()['row_count'] == 0:

            documents = []

            if not isinstance(file_paths, list):
                file_paths = [file_paths]

            for file_path in file_paths:
                documents.extend(self._load_documents_from_file(file_path))

            if not documents:
                print("No documents found to build index")
                return None
            
            # 构建向量索引
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                show_progress=True
            )

            self.store.client.flush(self.collection_name)

            self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

            if self.verbose:
                print(f"Successfully built index with {len(documents)} documents")
        
    
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
    
    
    def search(self, query: str, similarity_threshold: float = 0.0, candidate_algorithms: List[str] = None):
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

            # 如果有候选算法过滤条件，使用LlamaIndex的元数据过滤
            if candidate_algorithms is not None:
                return self._search_with_llamaindex_filter(query, self.top_k, similarity_threshold, candidate_algorithms)
            
            nodes = self.retriever.retrieve(query)

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
        return self.store.client.get_collection_stats(self.collection_name)


    def show_collection_info(self):
        """显示集合信息"""
        print(f"\n=== Collection: {self.collection_name} ===")
        stats = self.get_collection_stats()
        print(f"Row count: {stats.get('row_count', 0)}")
        print(f"Dimension: {self.dim}")
        print(f"Metric: {self.metric}")
        print(f"Index params: {self.index_params}")
        print(f"Search params: {self.search_params}")


def test_db(db_name):
    db = MilvusDB(db_name, 1024)
    text = "hello aag! " * 100
    documents = [Document(text=text)] * 2048

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        embed_batch_size=1024,
        device="cuda:0",
    )


    vector_store = MilvusVectorStore(uri="http://127.0.0.1:19530", dim=1024, overwrite=False, collection_name='test_demo', consistency_level='Strong')

    print(f"{vector_store.client.get_collection_stats(collection_name='test_demo')=}")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

    index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store
                )

    retriever = index.as_retriever()

    nodes = retriever.retrieve("summary this paragraph!")

    print(nodes)


if __name__ == '__main__':

    test_db('test_demo')

