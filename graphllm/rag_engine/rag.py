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

from typing import List
from typing import Optional
from database.milvus import *
from database.nebulagraph import *
from database.entitiesdb import *
from utils.pruning import simple_pruning
from utils.retrieval_metric import parse_paths_to_triples, parse_paths_to_triples_by_sentence
import time
import spacy
from abc import ABC, abstractmethod
from dataprocess.openai_extractor.gpt_extract_triplets import OpenAIExtractor
import re


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
        graph_k_hop,
        llm_env_,
        graph_db: NebulaDB,
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


class VectorRAG(RAG):

    def __init__(
        self,
        vector_k_similarity,
        llm_env_,
        vector_db: MilvusDB,
        data_type: str = 'summary'
    ) -> None:
        super().__init__()
        self.vector_db = vector_db
        self.vector_rag_retriever = vector_db.set_retriever_with_similarity_topk(
            similarity_top_k=vector_k_similarity)
        if data_type.lower() == 'qa':
            self.vector_query_engine = RetrieverQueryEngine.from_args(
                self.vector_rag_retriever)
        elif data_type.lower() == 'summary':
            self.vector_query_engine = RetrieverQueryEngine.from_args(
                self.vector_rag_retriever, response_mode=ResponseMode.TREE_SUMMARIZE)
        else:
            raise ValueError("Unsupported data type. Use 'qa' or'summary'.")
        self.llm_env_ = llm_env_

    def retrieve(self, query_str: str, query_id: Optional[int] = -1):
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

    def retrieve_deduplication(self, query_str: str, query_id: Optional[int] = -1):
        return ""


if __name__ == "__main__":
    qeustion_list = [
        "Did the Sporting News article report a higher batting average for Jung Hoo Lee in 2022 than Yardbarker reported for Juan Soto in the year referenced?",
        "Did the article from Cnbc | World Business News Leader on \"Nike's Latin America and Asia Pacific unit\" and the article from TechCrunch on \"Simply Homes\" both report an increase in their respective company's revenues?",
        "Between the Sporting News report on Tyreek Hill's chances of achieving 2,000-plus receiving yards before December 5, 2023, and the CBSSports.com report on Tyreek Hill's required average yards per game to reach his goal of 2,000 receiving yards, was there a change in the reporting of Tyreek Hill's progress towards his season goal?",
        "Does the TechCrunch article on Meta's GDPR compliance concerns suggest a different legal issue than the TechCrunch article on Meta's responsibility for teen social media monitoring, and does it also differ from the TechCrunch article on Meta's moderation bias affecting Palestinian voices?",
        "After Jerome Powell's aggressive interest rate hikes mentioned by 'Fortune' on October 6th, 2023, did 'Business Line' report on October 14th, 2023, suggest that central bankers' stance on interest rates was consistent or inconsistent with Powell's approach as reported by 'Fortune'?",
        "Has the reporting on the involvement of individuals in their respective football teams by Sporting News remained consistent between the article discussing Cameron Carter-Vickers' debut for Celtic after a hamstring injury (published at '2023-10-04T22:42:00+00:00') and the article detailing Daniel Garnero's debut as the new permanent manager of the Paraguay national football team 30 minutes before kickoff (published at '2023-10-12T23:22:00+00:00')?",
    ]
