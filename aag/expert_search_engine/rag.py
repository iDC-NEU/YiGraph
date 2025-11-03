import time
import re
import spacy
import os
import logging
# import QueryBundle
from llama_index.core import QueryBundle
from llama_index.core.utils import print_text

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever, KnowledgeGraphRAGRetriever

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import (
    Settings,
    # callback_manager_from_settings_or_context,
    # llm_from_settings_or_context,
)
from llama_index.core.response_synthesizers.type import *
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from typing import List
from typing import Optional
from abc import ABC, abstractmethod
from aag.expert_search_engine.data_process.openai_extractor.gpt_extract_triplets import OpenAIExtractor
from aag.expert_search_engine.database.milvus import *
from aag.expert_search_engine.database.nebulagraph import *
from aag.expert_search_engine.database.entitiesdb import *
from aag.reasoner.model_deployment import Reasoner
from aag.utils.pruning import simple_pruning
from aag.utils.retrieval_metric import parse_paths_to_triples, parse_paths_to_triples_by_sentence
from aag.config.engine_config import RetrievalConfig
from aag.utils.file_operation import read_yaml
from aag.utils.path_utils import TASK_TYPES_PATH, ALGORITHMS_PATH, KNOWLEDGE_PATH



logger = logging.getLogger(__name__)


class RAG(ABC):

    def __init__(self):
        # def __init__(self, db:Union[NebulaDB,MilvusDB], questions:list[str], llm_env:LLMEnv):
        # self.db = db
        # questions = questions
        # self.llm_env = llm_env
        self.time_info = {}

    def get_time_info(self):
        return self.time_info

    def clear_time_info(self):
        self.time_info.clear()

    def preprocess(self, question):
        pass

    def postprocess(self, question):
        pass

    @abstractmethod
    def retrieve(self, query_str: str, query_id: Optional[int] = -1):
        pass

    @abstractmethod
    def retrieve_deduplication(self, query_str: str, query_id: Optional[int] = -1):
        pass
    
    @abstractmethod
    def generation(self, query_str, nodes):
        pass

class GraphRAG(RAG):

    def __init__(
        self,
        graph_db: NebulaDB,
        graph_k_hop = 2,
        llm_env_ = None,
        pruning=30,
        data_type: str = 'qa',
        pruning_mode='embedding_for_perentity'  # embedding_for_perentity or embedding
    ) -> None:
        super().__init__()
        self.graph_db = graph_db
        self.graph_rag_retriever = self.graph_db.set_retriever(
            graph_traversal_depth=graph_k_hop, llm_env=llm_env_)

        db_entities_name = f"{self.graph_db.get_space_name()}_entities"
        self.entities_db: EntitiesDB = EntitiesDB(db_name=db_entities_name,
                                                  entities=graph_db.entities,
                                                  overwrite=False)
        if data_type.lower() == 'qa':
            self.kg_query_engine = RetrieverQueryEngine.from_args(
                self.graph_rag_retriever)
        elif data_type.lower() == 'summary':
            self.kg_query_engine = RetrieverQueryEngine.from_args(
                self.graph_rag_retriever, response_mode=ResponseMode.TREE_SUMMARIZE)
            print("sumamry engine")
        else:
            raise ValueError("Unsupported data type. Use 'qa' or'summary'.")
        self.limit = 1000000
        self.depth = graph_k_hop
        self.pruning = pruning
        self.pruning_mode = pruning_mode

    def retrieve(self, query_str: str, query_id: Optional[int] = -1, entity_num=3):
        self.clear_time_info()

        """Get entities from query string."""
        time_extract_entities = time.time()
        # method 1. get entities by local model
        # entities = self.graph_rag_retriever._get_entities(query_str)
        #
        # method 2. get entities by openai
        entities = self.entities_db.get_query_entities(
            query_str, max_keywords=3)
        entities = [entity.capitalize() for entity in entities]
        print(entities)
        q_entity_embeddings = self.entities_db.get_embedding(entities)
        #
        # method 3. get entities by query
        # query_embedding = self.entities_db.get_embedding(query_str)

        # 基于相似度比较实体和查询实体
        ids, distances = self.entities_db.search(
            q_entity_embeddings, limit=entity_num)
        similary_entities = list(
            set([self.entities_db.id2entity[id] for id in ids]))
        print(f"similary_entities:{similary_entities}")

        # filter by openai
        similary_entities = self.entities_db.get_filter_keyword_from_question_by_openai(
            query_str, similary_entities)
        similary_entities = [similary_entity.capitalize()
                             for similary_entity in similary_entities]
        print(f"filter_entities:{similary_entities}")

        entities = similary_entities
        time_extract_entities = time.time() - time_extract_entities

        self.time_info["time_extract_entities"] = time_extract_entities

        if len(entities) == 0:
            response = "No entity , LLM cannot generate a response"
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': "",
                'retrieve_results': {},
            }
            return [], retrieve_information

        time_graph_query = time.time()

        rel_map = self.graph_db.get_rel_map(entities=entities,
                                            depth=self.depth,
                                            limit=self.limit)
        time_graph_query = time.time() - time_graph_query
        self.time_info["time_graph_query"] = time_graph_query

        time_clean_graph_query = -time.time()
        clean_rel_map = self.graph_db.clean_rel_map(rel_map)
        knowledge_sequence = self.graph_db.get_knowledge_sequence(
            clean_rel_map)
        time_clean_graph_query += time.time()
        self.time_info["time_clean_graph_query"] = time_clean_graph_query

        if len(knowledge_sequence) == 0:
            response = "No information was retrieved, LLM cannot generate a response"
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': "",
                'retrieve_results': {},
            }
            return [], retrieve_information

        # 后处理
        time_graph_postprocess = -time.time()
        if self.pruning_mode == 'embedding':   # 对所有三元组一起剪枝
            nodes, pruning_knowledge_sequence = self.limit_rel_by_embedding_similarity(
                query_str, knowledge_sequence)
        elif self.pruning_mode == 'embedding_for_perentity':    # 对每个实体的子图分别剪枝
            all_pruning_knowledge_sequence = []
            filter_rel_map = {}
            for entity, sequences in clean_rel_map.items():
                _, pruning_knowledge_sequence = self.limit_rel_by_embedding_similarity(
                    query_str, sequences)
                filter_rel_map[entity] = pruning_knowledge_sequence
                all_pruning_knowledge_sequence.extend(
                    pruning_knowledge_sequence)
            pruning_knowledge_dict = {
                "pruning": all_pruning_knowledge_sequence}
            print(f"all_pruning_knowledge_sequence:{all_pruning_knowledge_sequence}")
            nodes = self.graph_db._build_nodes(all_pruning_knowledge_sequence,
                                               pruning_knowledge_dict)
        else:
            nodes = self.graph_db._build_nodes(knowledge_sequence, rel_map)
            pruning_knowledge_sequence = knowledge_sequence
        time_graph_postprocess += time.time()
        self.time_info["time_graph_postprocess"] = time_graph_postprocess
        self.time_info["time_retrieve"] = self.time_info["time_extract_entities"] + \
            self.time_info["time_graph_query"] + \
            self.time_info["time_graph_postprocess"]

        if self.pruning_mode == 'embedding_for_perentity':
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': entities,
                'retrieve_results': filter_rel_map
            }
        else:
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': entities,
                'retrieve_results': pruning_knowledge_sequence
            }
        # print(f"nodes length:{len(nodes)} ")
        # print_text(nodes[0].get_content())

        return nodes, retrieve_information

    def retrieve_deduplication(self, query_str: str, query_id: Optional[int] = -1, entity_num=3):
        self.clear_time_info()
        deduplication_nodes  = None
        deduplication_rate = 0
        """Get entities from query string."""
        time_extract_entities = time.time()
        # method 1. get entities by local model
        # entities = self.graph_rag_retriever._get_entities(query_str)
        #
        # method 2. get entities by openai
        entities = self.entities_db.get_query_entities(
            query_str, max_keywords=3)
        entities = [entity.capitalize() for entity in entities]
        print(entities)
        q_entity_embeddings = self.entities_db.get_embedding(entities)
        #
        # method 3. get entities by query
        # query_embedding = self.entities_db.get_embedding(query_str)

        # 基于相似度比较实体和查询实体
        ids, distances = self.entities_db.search(
            q_entity_embeddings, limit=entity_num)
        similary_entities = list(
            set([self.entities_db.id2entity[id] for id in ids]))
        print(f"similary_entities:{similary_entities}")

        # filter by openai
        similary_entities = self.entities_db.get_filter_keyword_from_question_by_openai(
            query_str, similary_entities)
        similary_entities = [similary_entity.capitalize()
                             for similary_entity in similary_entities]
        print(f"filter_entities:{similary_entities}")

        entities = similary_entities
        time_extract_entities = time.time() - time_extract_entities

        self.time_info["time_extract_entities"] = time_extract_entities

        if len(entities) == 0:
            response = "No entity , LLM cannot generate a response"
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': "",
                'retrieve_results': {},
            }
            return [], retrieve_information

        time_graph_query = time.time()

        rel_map = self.graph_db.get_rel_map(entities=entities,
                                            depth=self.depth,
                                            limit=self.limit)
        time_graph_query = time.time() - time_graph_query
        self.time_info["time_graph_query"] = time_graph_query

        time_clean_graph_query = -time.time()
        clean_rel_map = self.graph_db.clean_rel_map(rel_map)
        knowledge_sequence = self.graph_db.get_knowledge_sequence(
            clean_rel_map)
        time_clean_graph_query += time.time()
        self.time_info["time_clean_graph_query"] = time_clean_graph_query

        if len(knowledge_sequence) == 0:
            response = "No information was retrieved, LLM cannot generate a response"
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': "",
                'retrieve_results': {},
            }
            return [], retrieve_information

        # 后处理
        time_graph_postprocess = -time.time()
        if self.pruning_mode == 'embedding':   # 对所有三元组一起剪枝
            nodes, pruning_knowledge_sequence = self.limit_rel_by_embedding_similarity(
                query_str, knowledge_sequence)
        elif self.pruning_mode == 'embedding_for_perentity':    # 对每个实体的子图分别剪枝
            all_pruning_knowledge_sequence = []
            filter_rel_map = {}
            for entity, sequences in clean_rel_map.items():
                _, pruning_knowledge_sequence = self.limit_rel_by_embedding_similarity(
                    query_str, sequences)
                filter_rel_map[entity] = pruning_knowledge_sequence
                all_pruning_knowledge_sequence.extend(
                    pruning_knowledge_sequence)
            pruning_knowledge_dict = {
                "pruning": all_pruning_knowledge_sequence}# 将去重后的三元组集合转换回列表
            print(f"before:{all_pruning_knowledge_sequence}")
            nodes = self.graph_db._build_nodes(all_pruning_knowledge_sequence,
                                               pruning_knowledge_dict)
            # 补充代码：需要对all_pruning_knowledge_sequence进行去重, 里面每个元素是一个path，参考下面的函数把path转成三元组集合，去重后转成list
            def path_to_triplets(path):
                # 将路径转换为三元组集合
                # 调用 neutronrag/utils/retrieval_metric.py 中的 parse_paths_to_triples
                triplets = parse_paths_to_triples_by_sentence([path])
                return triplets

            # 将所有路径转换为三元组并去重
            all_triplets = list()
            for path in all_pruning_knowledge_sequence:
                triplets = path_to_triplets(path)
                all_triplets.extend(triplets)
            total_triplets = len(all_triplets)
            all_triplets = list(set(all_triplets))
            deduplication_triplets = len(all_triplets)
            deduplication_rate = 1- (deduplication_triplets / total_triplets)
            print(f"deduplication_rate:{deduplication_rate}")
            all_pruning_knowledge_sequence_new = all_triplets
            pruning_knowledge_dict_new = {
                "pruning": all_pruning_knowledge_sequence_new}# 将去重后的三元组集合转换回列表
            print(f"after:{all_pruning_knowledge_sequence_new}")
            deduplication_nodes = self.graph_db._build_nodes(all_pruning_knowledge_sequence_new,
                                               pruning_knowledge_dict_new)
        else:
            nodes = self.graph_db._build_nodes(knowledge_sequence, rel_map)
            pruning_knowledge_sequence = knowledge_sequence
        time_graph_postprocess += time.time()
        self.time_info["time_graph_postprocess"] = time_graph_postprocess
        self.time_info["time_retrieve"] = self.time_info["time_extract_entities"] + \
            self.time_info["time_graph_query"] + \
            self.time_info["time_graph_postprocess"]

        if self.pruning_mode == 'embedding_for_perentity':
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': entities,
                'retrieve_results': filter_rel_map
            }
        else:
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': entities,
                'retrieve_results': pruning_knowledge_sequence
            }
        # print(f"nodes length:{len(nodes)} ")
        # print_text(nodes[0].get_content())

        return nodes, retrieve_information, deduplication_nodes, deduplication_rate


    def retrieve_based_entity(self, query_str: str, entities, query_id: Optional[int] = -1, entity_num=10):
        self.clear_time_info()

        """Get entities from query string."""
        time_extract_entities = time.time()
        # method 1. get entities by local model
        # entities = self.graph_rag_retriever._get_entities(query_str)
        #
        # method 2. get entities by openai
        # entities = self.entities_db.get_query_entities(
        #     query_str, max_keywords=3)
        entities = [entity.capitalize() for entity in entities]
        print(entities)
        q_entity_embeddings = self.entities_db.get_embedding(entities)
        #
        # method 3. get entities by query
        # query_embedding = self.entities_db.get_embedding(query_str)

        # 基于相似度比较实体和查询实体
        ids, distances = self.entities_db.search(
            q_entity_embeddings, limit=entity_num)
        similary_entities = list(
            set([self.entities_db.id2entity[id] for id in ids]))
        print(f"similary_entities:{similary_entities}")

        # filter by openai
        similary_entities = self.entities_db.get_filter_keyword_from_question_by_openai(
            query_str, similary_entities)
        similary_entities = [similary_entity.capitalize()
                             for similary_entity in similary_entities]
        print(f"filter_entities:{similary_entities}")

        entities = similary_entities
        time_extract_entities = time.time() - time_extract_entities

        self.time_info["time_extract_entities"] = time_extract_entities

        if len(entities) == 0:
            response = "No entity , LLM cannot generate a response"
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': "",
                'retrieve_results': {},
            }
            return [], retrieve_information

        time_graph_query = time.time()

        rel_map = self.graph_db.get_rel_map(entities=entities,
                                            depth=self.depth,
                                            limit=self.limit)
        time_graph_query = time.time() - time_graph_query
        self.time_info["time_graph_query"] = time_graph_query

        time_clean_graph_query = -time.time()
        clean_rel_map = self.graph_db.clean_rel_map(rel_map)
        knowledge_sequence = self.graph_db.get_knowledge_sequence(
            clean_rel_map)
        time_clean_graph_query += time.time()
        self.time_info["time_clean_graph_query"] = time_clean_graph_query

        if len(knowledge_sequence) == 0:
            response = "No information was retrieved, LLM cannot generate a response"
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': "",
                'retrieve_results': {},
            }
            return [], retrieve_information

        # 后处理
        time_graph_postprocess = -time.time()
        if self.pruning_mode == 'embedding':   # 对所有三元组一起剪枝
            nodes, pruning_knowledge_sequence = self.limit_rel_by_embedding_similarity(
                query_str, knowledge_sequence)
        elif self.pruning_mode == 'embedding_for_perentity':    # 对每个实体的子图分别剪枝
            all_pruning_knowledge_sequence = []
            filter_rel_map = {}
            for entity, sequences in clean_rel_map.items():
                _, pruning_knowledge_sequence = self.limit_rel_by_embedding_similarity(
                    query_str, sequences)
                filter_rel_map[entity] = pruning_knowledge_sequence
                all_pruning_knowledge_sequence.extend(
                    pruning_knowledge_sequence)
            pruning_knowledge_dict = {
                "pruning": all_pruning_knowledge_sequence}
            nodes = self.graph_db._build_nodes(all_pruning_knowledge_sequence,
                                               pruning_knowledge_dict)
        else:
            nodes = self.graph_db._build_nodes(knowledge_sequence, rel_map)
            pruning_knowledge_sequence = knowledge_sequence
        time_graph_postprocess += time.time()
        self.time_info["time_graph_postprocess"] = time_graph_postprocess
        self.time_info["time_retrieve"] = self.time_info["time_extract_entities"] + \
            self.time_info["time_graph_query"] + \
            self.time_info["time_graph_postprocess"]

        if self.pruning_mode == 'embedding_for_perentity':
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': entities,
                'retrieve_results': filter_rel_map
            }
        else:
            retrieve_information = {
                'id': query_id,
                'query': query_str,
                'entities': entities,
                'retrieve_results': pruning_knowledge_sequence
            }
        # print(f"nodes length:{len(nodes)} ")
        # print_text(nodes[0].get_content())

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
        return self.kg_query_engine.query(question)
    
    def query_with_entity(self, entities):  #TODO
        """
          entities 是一个list类型的参数实体列表，
          算法：遍历每个entities，从nebulagraph查询每个实体的子图， 并把将这些子图转成边表形式 edges: List[Tuple]
        """
        pass

    def get_all_edges(self):
        """
            从nebula中获取所有边,整张图的边表
            Returns:
                List[Tuple[str, str, str]]: 每个元素为(head, relation, tail)
        """
        triplets = []
        result = self.graph_db.client.session.execute(
            f'use {self.graph_db.space_name}; MATCH (n1)-[e]->(n2) RETURN n1, e, n2;'
        )
        if result.row_size() > 0:
            for row in result.rows():
                values = row.values
                head = ''
                relation = ''
                tail = ''
                for value in values:
                    if value.field == 9:  # Vertex
                        vertex = value.get_vVal()
                        if not head:
                            head = vertex.vid.get_sVal().decode('utf-8')
                        else:
                            tail = vertex.vid.get_sVal().decode('utf-8')
                    elif value.field == 10:  # Edge
                        edge = value.get_eVal()
                        relation = edge.props.get(b'relationship').get_sVal().decode('utf-8')
                triplets.append((head, relation, tail))
        return triplets


class VectorRAG_delay(RAG):

    def __init__(
        self,
        vector_db: MilvusDB2,
        vector_k_similarity: int,
        llm_env_ = None,
        data_type: str = 'summary'
    ) -> None:
        super().__init__()
        
        self.vector_db =  vector_db
        self.vector_rag_retriever = vector_db.set_retriever_with_similarity_topk(
            similarity_top_k=vector_k_similarity)
        self.llm_env_ = llm_env_
        # if data_type.lower() == 'qa':
        #     self.vector_query_engine = RetrieverQueryEngine.from_args(
        #         self.vector_rag_retriever)
        # elif data_type.lower() == 'summary':
        #     self.vector_query_engine = RetrieverQueryEngine.from_args(
        #         self.vector_rag_retriever, response_mode=ResponseMode.TREE_SUMMARIZE)
        # else:
        #     raise ValueError("Unsupported data type. Use 'qa' or'summary'.")

    def retrieve(self, query_str: str, query_id: Optional[int] = -1):
        """检索相关的图算法论文"""
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
        node_with_scores = self.vector_rag_retriever._get_nodes_with_embeddings(
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
        node_with_scores = self.vector_rag_retriever._get_nodes_with_embeddings(
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



class VectorRAG(RAG):

    def __init__(
        self,
        vector_db: MilvusDB2,
        vector_k_similarity: int,
        llm_env_ = None,
        data_type: str = 'summary'
    ) -> None:
        super().__init__()
        
        self.vector_db =  vector_db
        self.llm_env_ = llm_env_
        # if data_type.lower() == 'qa':
        #     self.vector_query_engine = RetrieverQueryEngine.from_args(
        #         self.vector_rag_retriever)
        # elif data_type.lower() == 'summary':
        #     self.vector_query_engine = RetrieverQueryEngine.from_args(
        #         self.vector_rag_retriever, response_mode=ResponseMode.TREE_SUMMARIZE)
        # else:
        #     raise ValueError("Unsupported data type. Use 'qa' or'summary'.")

    def build_index(self, file_path: str):
        self.vector_rag_retriever = self.vector_db.build_index(file_path)

    def retrieve(self, query_str: str, query_id: Optional[int] = -1):
        """检索相关的图算法论文"""
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
        node_with_scores = self.vector_rag_retriever._get_nodes_with_embeddings(
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
        node_with_scores = self.vector_rag_retriever._get_nodes_with_embeddings(
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



class RAG_Engine:

    def __init__(
            self,
            vector_db,
            vector_k_similarity,
            graph_db,
            vector_rag_llm_env_ = None,
            graph_rag_llm_env_ = None,
    ):
        self.vector_rag = VectorRAG(vector_db, vector_k_similarity, llm_env_=vector_rag_llm_env_)
        self.graph_rag = GraphRAG(graph_db, llm_env_=graph_rag_llm_env_)

    def _retrieve_graph_papers(self, query_str: str, query_id: Optional[int] = -1):
        # TODO: 根据用户query，从vectorrag里检索相关的图算法论文，返回list[str1, str2, ...]
        pass        

    def _retrieve_graph_data(self, query_str: str, query_id: Optional[int] = -1):
        # TODO: 根据用户query，从graphrag里检索相关的图数据，返回形式unknown
        pass

    def retrieve(self, query_str: str, query_id: Optional[int] = -1):
        # TODO: 根据用户query，从vectorrag里检索相关的图算法论文，返回list[str1, str2, ...]
        papers = self._retrieve_graph_papers(query_str, query_id)
        graph_data = self._retrieve_graph_data(query_str, query_id)
        return papers, graph_data


class ExpertSearchEngine:

    def __init__(
            self,
            config: RetrievalConfig,
            vector_rag_llm_env_ = None,
            graph_rag_llm_env_ = None,
    ):
        if config is None:
            raise ValueError("ExpertSearchEngine requires a valid RetrievalConfig with retrieval settings")
        self.config = config
        self.graph_db: Optional[NebulaDB] = None
        self.vector_db: Optional[MilvusDB2] = None

        self.graph_rag: Optional[GraphRAG] = None
        self.vector_rag: Optional[VectorRAG] = None 

        self.task_index: Dict[Any, Any] = {}
        self.algo_index: Dict[Any, Any] = {}

        self.vector_rag_llm_env_ = vector_rag_llm_env_
        self.graph_rag_llm_env_ =  graph_rag_llm_env_    
        
        self._initialize_components()


    def _initialize_components(self):
        """初始化 ExpertSearchEngine 各个组件"""
        print("Initializing ExpertSearchEngine components...")

        self._init_embedding_model()

        # self._init_graph_database()

        # self._init_vector_database()
        
        # self._init_graph_rag()

        # self._init_vector_rag()

         # 构造  task 指向 alg 的分层知识库 函数
        self._build_hierarchical_knowledge_base()
        

    def _init_embedding_model(self):
        """初始化嵌入模型(Settings.embed_model/chunk配置)。"""
        try:
            llm_embed_name = self.config.embedding.get("model_name")
            embed_batch_size = self.config.embedding.get("batch_size", 32)
            device = self.config.embedding.get("device", "cpu")
            chunk_size = self.config.embedding.get("chunk_size", 1024)
            chunk_overlap = self.config.embedding.get("chunk_overlap", 50)

            if not llm_embed_name or not isinstance(llm_embed_name, str):
                raise ValueError("embedding.model_name 配置缺失或非法")

            name_lower = llm_embed_name.lower()
            is_openai = name_lower.startswith("text-embedding") or "openai" in name_lower

            if is_openai:
                Settings.embed_model = OpenAIEmbedding(
                    model=llm_embed_name,
                    embed_batch_size=embed_batch_size,
                )
            else:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=llm_embed_name,
                    embed_batch_size=embed_batch_size,
                    device=device,
                )
            self.dim  = self._infer_dim(Settings.embed_model)
            Settings.chunk_size = int(chunk_size)
            Settings.chunk_overlap = int(chunk_overlap)

            print(
                f"✓ Embedding model initialized: {llm_embed_name} | dime={self.dim}, batch={embed_batch_size}, device={device}, "
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

    def _init_graph_database(self):
        """初始化图数据库连接"""
        try:
            # 初始化图数据库
            self.graph_db = NebulaDB(
                space_name=self.config.database.graph.get("space_name", "null_space"),
                server_ip=self.config.database.graph.get("server_ip", "127.0.0.1"),
                server_port=self.config.database.graph.get("server_port", "9669"),
                create=self.config.database.graph.get("create", False),
                verbose=self.config.database.graph.get("verbose", False)
            )
            print(f"✓ Graph database initialized: {self.config.database.graph.get('space_name', 'null_space')}")
        except Exception as e:
            print(f"✗ Graph database initialization failed: {e}")
            raise
    
    def _init_vector_database(self):
        """初始化向量数据库连接"""
        try:
            # 初始化向量数据库
            self.vector_db = MilvusDB2(
                collection_name=self.config.database.vector.get("collection_name", "knowledge"),
                dim=self.dim,
                host=self.config.database.vector.get("host", "localhost"),
                port=self.config.database.vector.get("port", 19530)
            )
            print(f"✓ Vector database initialized: {self.config.database.vector.get('collection_name', 'null_collection')}")
        except Exception as e:
            print(f"✗ Vector database initialization failed: {e}")
            raise

    def _init_graph_rag(self):
        try:
            graph_k_hop = self.config.rag.graph.get("k_hop", 2)
            self.graph_rag = GraphRAG(self.graph_db, graph_k_hop, llm_env_=self.graph_rag_llm_env_)
            print(f"✓ Graph RAG initialized with k_hop={graph_k_hop}")
        except Exception as e:
            print(f"✗ Graph RAG initialization failed: {e}")
            raise
    
    def _init_vector_rag(self):
        try:
            vector_k_similarity = self.config.rag.vector.get("k_similarity", 5)
            self.vector_rag = VectorRAG(self.vector_db, vector_k_similarity, llm_env_=self.vector_rag_llm_env_)
            print(f"✓ Vector RAG initialized with k_similarity={vector_k_similarity}")
        except Exception as e:
            print(f"✗ Vector RAG initialization failed: {e}")
            raise

    def _build_hierarchical_knowledge_base(self):
        """构造分层知识库连接"""
        try:
            
            self._construct_task_to_alg_knowledge_base()
            # self.vector_rag.build_index(KNOWLEDGE_PATH)

            print(f"✓ hierarchical knowledge base initialized")
        except Exception as e:
            print(f"✗ hierarchical knowledge base initialization failed: {e}")
            raise

    def _build_index_from_yaml(self, file_path: str, item_name: str) -> Dict[Any, Any]:
        """通用索引构建函数"""
        items = read_yaml(file_path)
        index = {}
        for obj in items:
            if not isinstance(obj, dict):
                logger.warning(f"跳过非字典格式的{item_name}: {obj}")
                continue
            obj_id = obj.get("id")
            if not obj_id:
                logger.warning(f"跳过缺少id的{item_name}: {obj}")
                continue
            index[obj_id] = obj
        return index

    def _construct_task_to_alg_knowledge_base(self):
        """构造 task 指向 alg 的分层知识库"""
        try:                      
            # 构建索引字典
            self.task_index = self._build_index_from_yaml(TASK_TYPES_PATH, "TASK_TYPE")
            self.algo_index = self._build_index_from_yaml(ALGORITHMS_PATH, "ALGORITHMS")
            logger.info(f"构建完成: 任务={len(self.task_index)} 算法={len(self.algo_index)}")
            
            # 验证任务类型中的算法是否都存在
            missing = [
                f"任务 {tid} 中的算法 {aid}"
                for tid, task in self.task_index.items()
                for aid in task.get("algorithm", [])
                if aid not in self.algo_index
            ]
            if missing:
                logger.warning("以下算法引用缺失:")
                for m in missing:
                    logger.warning(f"  - {m}")

        except Exception as e:
            print(f"构建知识库索引时发生错误: {e}")
            # 设置空索引，避免后续访问出错
            self.task_index = {}
            self.algo_index = {}
            raise
    
    #todo(chaoyi): 相似度计算，获取最相关的任务类型
    def retrieve_task_type(self, query_str: str) -> List[Dict[Any, Any]]:
        # 遍历self.task_index, 获取每个task_type的task_type和description, 返回一个list[{"task_type": task_type, "description": description}]
        task_type_list = []
        for task_type in self.task_index.values():
            task_type_list.append({
                "id": task_type.get("id", ""),
                "task_type": task_type.get("task_type", ""),
                "description": task_type.get("description", "")
            })
        return task_type_list

    #todo(chaoyi): 相似度计算，获取最相关的算法
    def retrieve_algorithm(self, query_str: str, task_type_id: str) -> List[Dict[Any, Any]]:
        # 从 self.task_index 中获取 task_type_id 对应的 task_type, 然后获取这个task_type 里的algorithm， 遍历algorithm， 从self.algo_index 中获取每个algorithm的algorithm和description, 返回一个list[{"id": id, "algorithm": algorithm, "description": description}]
        algorithm_list = []
        selected_algorithm_ids = self.task_index.get(task_type_id, {}).get("algorithm", [])
        for algorithm_id in selected_algorithm_ids:
            algorithm_list.append({
                "id": algorithm_id,
                "description_principle": self.algo_index[algorithm_id].get("description").get("principle"),
                "description_meaning": self.algo_index[algorithm_id].get("description").get("meaning"),
            })
        return algorithm_list
        


    def _retrieve_graph_papers(self, query_str: str, query_id: Optional[int] = -1):
        # TODO: 根据用户query，从vectorrag里检索相关的图算法论文，返回list[str1, str2, ...]
        pass        

    def _retrieve_graph_data(self, query_str: str, query_id: Optional[int] = -1):
        # TODO: 根据用户query，从graphrag里检索相关的图数据，返回形式unknown
        pass

    def retrieve(self, query_str: str, query_id: Optional[int] = -1):
        # TODO: 根据用户query，从vectorrag里检索相关的图算法论文，返回list[str1, str2, ...]
        papers = self._retrieve_graph_papers(query_str, query_id)
        graph_data = self._retrieve_graph_data(query_str, query_id)
        return papers, graph_data


if __name__ == "__main__":
    qeustion_list = [
        "Did the Sporting News article report a higher batting average for Jung Hoo Lee in 2022 than Yardbarker reported for Juan Soto in the year referenced?",
        "Did the article from Cnbc | World Business News Leader on \"Nike's Latin America and Asia Pacific unit\" and the article from TechCrunch on \"Simply Homes\" both report an increase in their respective company's revenues?",
        "Between the Sporting News report on Tyreek Hill's chances of achieving 2,000-plus receiving yards before December 5, 2023, and the CBSSports.com report on Tyreek Hill's required average yards per game to reach his goal of 2,000 receiving yards, was there a change in the reporting of Tyreek Hill's progress towards his season goal?",
        "Does the TechCrunch article on Meta's GDPR compliance concerns suggest a different legal issue than the TechCrunch article on Meta's responsibility for teen social media monitoring, and does it also differ from the TechCrunch article on Meta's moderation bias affecting Palestinian voices?",
        "After Jerome Powell's aggressive interest rate hikes mentioned by 'Fortune' on October 6th, 2023, did 'Business Line' report on October 14th, 2023, suggest that central bankers' stance on interest rates was consistent or inconsistent with Powell's approach as reported by 'Fortune'?",
        "Has the reporting on the involvement of individuals in their respective football teams by Sporting News remained consistent between the article discussing Cameron Carter-Vickers' debut for Celtic after a hamstring injury (published at '2023-10-04T22:42:00+00:00') and the article detailing Daniel Garnero's debut as the new permanent manager of the Paraguay national football team 30 minutes before kickoff (published at '2023-10-12T23:22:00+00:00')?",
    ]
