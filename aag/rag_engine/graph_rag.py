import time
# import QueryBundle
from llama_index.core import QueryBundle
from llama_index.core.utils import print_text

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.type import *

from typing import Optional
from aag.expert_search_engine.database.nebulagraph import *
from aag.expert_search_engine.database.entitiesdb import *
from aag.utils.pruning import simple_pruning
from aag.rag_engine.rag import RAG
from aag.config.engine_config import RetrievalConfig



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

        self.graph_k_hop = self.config.rag.graph.get("k_hop", 2)

        self.graph_rag_retriever = self.graph_db.set_retriever(
            graph_traversal_depth=self.graph_k_hop, llm_env=llm_env_)

        db_entities_name = f"{self.graph_db.get_space_name()}_entities"
        self.entities_db: EntitiesDB = EntitiesDB(db_name=db_entities_name,
                                                  entities=self.graph_db.entities,
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
        self.depth = self.graph_k_hop
        self.pruning = pruning
        self.pruning_mode = pruning_mode


        print(f"✓ Graph RAG initialized with k_hop={self.graph_k_hop}")

    
    
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
