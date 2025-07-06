from pymilvus import connections, Collection
from llama_index.vector_stores.milvus import MilvusVectorStore
import time
from typing import (
    # Optional, Dict,
    List)
from llama_index.core.utils import print_text
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import (
    VectorStoreIndex,
    # SimpleDirectoryReader,
    # Document,
    StorageContext,
    load_index_from_storage,
)

from pymilvus import (
    connections,
    # utility,
    # FieldSchema,
    # CollectionSchema,
    DataType,
    Collection,
    # Milvus,
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
        self.client = Milvus(server_ip, server_port)

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
