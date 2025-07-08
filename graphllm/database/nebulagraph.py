import os
# from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core import StorageContext
from llama_index.core import KnowledgeGraphIndex
# from llama_index.core.graph_stores.nebula import NebulaGraphStore
from llama_index.legacy.graph_stores.nebulagraph import NebulaGraphStore

# from nebula3.common.ttypes import ErrorCode
# from nebula3.gclient.net import Connection
# from nebula3.gclient.net.SessionPool import SessionPool
# from nebula3.Config import SessionPoolConfig
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.data.ResultSet import ResultSet
# from nebula3.common import *
import json

# import sys
# sys.path.append('.')
# sys.path.append('..')

from graphllm.utils.FormatResp import print_resp
from graphllm.utils.file_operation import *
import time
# from database.utils import *

# from utils.openai_env import service_context
from llama_index.core import load_index_from_storage
from llama_index.core.utils import print_text
import re
# from utils.utils import create_dir
from llama_index.core.schema import (
    # BaseNode,
    # MetadataMode,
    NodeWithScore,
    # QueryBundle,
    TextNode,
)

from llama_index.core.retrievers import (
    KnowledgeGraphRAGRetriever, )

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from graphllm.utils.extract_subgraph import filter_pr_rels
from graphllm.utils.pruning import simple_pruning

fmt = "\n=== {:30} ===\n"
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


class NebulaClient:

    def __init__(self):
        config = Config()
        config.max_connection_pool_size = 10

        self.connection_pool = ConnectionPool()
        ok = self.connection_pool.init([('127.0.0.1', 9669)], config)
        assert ok

        self.session = self.connection_pool.get_session('root', 'nebula')

    def __del__(self):
        if self.connection_pool:
            self.connection_pool.close()
        if self.session:
            self.session.release()

    def create_space(self, space_name):
        self.session.execute(
            f'CREATE SPACE IF NOT EXISTS {space_name}(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);'
        )
        time.sleep(10)
        self.session.execute(
            f'USE {space_name}; CREATE TAG IF NOT EXISTS entity(name string);')
        self.session.execute(
            f'USE {space_name}; CREATE EDGE IF NOT EXISTS relationship(relationship string);'
        )
        self.session.execute(
            f'USE {space_name}; CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));'
        )
        time.sleep(10)

    def drop_space(self, space_name):
        if not isinstance(space_name, list):
            space_name = [space_name]
        for space in space_name:
            self.session.execute(f'drop space {space}')

    def info(self, space_name):
        result = self.session.execute(
            f'use {space_name}; submit job stats; show stats;')
        print(result)
        print_resp(result)

    def count_edges(self, space_name):
        result = self.session.execute(
            f'use {space_name}; MATCH (m)-[e]->(n) RETURN COUNT(*);')
        print_resp(result)

    def show_space(self):
        result = self.session.execute('SHOW SPACES;')
        print_resp(result)

    def show_edges(self, space_name, limits):
        result = self.session.execute(
            f'use {space_name}; MATCH ()-[e]->() RETURN e LIMIT {limits};')
        print_resp(result)

    def clear(self, space_name):
        query = f'CLEAR SPACE {space_name};'
        self.session.execute(query)

    def get_triplets(self, space_name):
        result = self.session.execute(
            f'use {space_name}; MATCH (n1)-[e]->(n2) RETURN n1, e, n2;')

    def show_triplets(self, space_name, file_path=None):
        result = self.session.execute(
            f'use {space_name}; MATCH (n1)-[e]->(n2) RETURN n1, e, n2;')

        # print(result, len(result))
        # assert False

        if not file_path:
            file_path = space_name + '_triplets.txt'

        json_path = file_path + '.json'
        all_triples = []

        with open(file_path, 'w', encoding='utf-8') as file:
            if result.row_size() > 0:
                for row in result.rows():
                    values = row.values

                    head = ''
                    relation = ''
                    tail = ''

                    for value in values:
                        if value.field == 9:  # 对应 Vertex
                            vertex = value.get_vVal()
                            if not head:
                                head = vertex.vid.get_sVal().decode('utf-8')
                            else:
                                tail = vertex.vid.get_sVal().decode('utf-8')

                        elif value.field == 10:  # 对应 Edge
                            edge = value.get_eVal()
                            relation = edge.props.get(
                                b'relationship').get_sVal().decode('utf-8')

                    triplet = [head, relation, tail]
                    all_triples.append(triplet)

                    file.write(f"{triplet}\n")
                print(
                    f'triplet write to {file_path}, tot {len(result.rows())} triplets.'
                )
            else:
                print('No data found.')
                file.write('No data found.\n')

        all_triples = set(tuple(triplet) for triplet in all_triples)
        print(
            f'after filter the sample triplet last {len(all_triples)} triplets.'
        )

        all_triples = [list(triplet) for triplet in all_triples]
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(all_triples, file, ensure_ascii=False, indent=4)
            print(f'save {len(all_triples)} triples to {json_path}.')


class NebulaDB:

    def __init__(self,
                 space_name,
                 log_file='./database/nebula.log',
                 server_ip='127.0.0.1',
                 server_port='9669',
                 create=False,
                 verbose=False):
        #  verbose=False, retriever=False, llm_env=None):
        self.log_file = log_file
        self.server_ip = server_ip
        self.server_port = server_port

        os.environ["NEBULA_USER"] = "root"
        os.environ["NEBULA_PASSWORD"] = "nebula"  # default is "nebula"
        os.environ["NEBULA_ADDRESS"] = f"{self.server_ip}:{self.server_port}"

        self.space_name = space_name
        self.edge_types = ['relationship']
        self.rel_prop_names = ['relationship']
        self.tags = ['entity']
        self.client = NebulaClient()
        self.verbose = verbose
        self.store: NebulaGraphStore = None

        try:
            self.store, self.storage_context = self.init_nebula_store()
        except Exception:
            print(
                f'please use NebulaClient().create() to create space {self.space_name}!!!\n\n\n'
            )
        self.graph_schema = self.store.get_schema(refresh=None)

        self.retriever = None
        self.entities = self.get_all_entities()

    def __del__(self):
        del self.client

    def init_nebula_store(self):
        nebula_store = NebulaGraphStore(
            space_name=self.space_name,
            edge_types=self.edge_types,
            rel_prop_names=self.rel_prop_names,
            tags=self.tags,
        )
        storage_context = StorageContext.from_defaults(
            graph_store=nebula_store)
        return nebula_store, storage_context

    def upsert_triplet(self, triplet: Tuple[str, str, str]):
        self.store.upsert_triplet(*triplet)

    def get_storage_context(self):
        return self.storage_context

    def get_space_name(self):
        return self.space_name

    def get_index(self):
        return load_index_from_storage(self.storage_context)

    def get_triplets(self):
        return self.client.get_triplets(self.space_name)

    def get_all_entities(self, triplets_file=""):
        entities_file = f'/home/chency/NeutronRAG/external_corpus/all_processed_corpus/entity_data/{self.space_name}_entities.json'

        if file_exist(entities_file):
            print(f"load entities from {entities_file}")
            entities = read_json(entities_file)
        else:
            # TODO: need specific triplets_file
            # triplets_file = "f'xxx/{self.space_name}_triplets.json'"
            triplets_file = triplets_file

            if file_exist(triplets_file):
                all_triplets = read_triplets_from_json(triplets_file)
                all_triplets = list(set(all_triplets))
            else:
                all_triplets = self.get_triplets()

            left_entities = [triplet[0] for triplet in all_triplets]
            right_entities = [triplet[2] for triplet in all_triplets]
            entities = sorted(list(set(left_entities + right_entities)))

            # assert len(entities - entities1) == 0 and len(entities1 - entities) == 0
            save_response(entities, entities_file)

            # print(f'triplets: {len(all_triplets)}, entities: {len(entities)}')
        print(f'entities: {len(entities)}')

        return set(entities)

    def process_docs(self,
                     documents,
                     triplets_per_chunk=10,
                     include_embeddings=True,
                     data_dir='./storage_graph',
                     extract_fn=None,
                     cache=True):

        # TODO: use rebel to extract the kg elements.

        # filter documents
        # filter_documents = [doc for doc in documents if not is_file_processed(self.log_file, doc.id_)]
        # print(f'filter {len(documents) - len(filter_documents)} documents, last {len(filter_documents)} documents.')
        # documents = filter_documents

        # if len(documents) == 0:
        #     return

        # kg_index = KnowledgeGraphIndex.from_documents(
        #     documents,
        #     storage_context=self.storage_context,
        #     max_triplets_per_chunk=triplets_per_chunk,
        #     space_name=self.space_name,
        #     edge_types=self.edge_types,
        #     rel_prop_names=self.rel_prop_names,
        #     tags=self.tags,
        #     include_embeddings=True,
        #     show_progress=True,
        # )
        print(os.getcwd(), data_dir)
        index_loaded = False
        if cache:
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=data_dir, graph_store=self.store)
                kg_index = load_index_from_storage(
                    storage_context=storage_context,
                    # service_context=service_context,
                    max_triplets_per_chunk=triplets_per_chunk,
                    space_name=self.space_name,
                    edge_types=self.edge_types,
                    rel_prop_names=self.rel_prop_names,
                    tags=self.tags,
                    verbose=True,
                    show_progress=True,
                )
                index_loaded = True
                print(f"graph index load from {data_dir}.")
                return kg_index
            except Exception:
                index_loaded = False

        if not index_loaded:
            kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                kg_triplet_extract_fn=extract_fn,
                # service_context=service_context,
                max_triplets_per_chunk=triplets_per_chunk,
                space_name=self.space_name,
                edge_types=self.edge_types,
                rel_prop_names=self.rel_prop_names,
                tags=self.tags,
                include_embeddings=True,
                show_progress=True,
            )
        if cache:
            kg_index.storage_context.persist(persist_dir=data_dir)
            print(f"kg index store to {data_dir}.")
        # for doc in documents:
        #     append_log(self.log_file, doc.id_)
        return kg_index

    def get_rel_map(self, entities, depth=2, limit=30):
        rel_map: Optional[Dict] = self.store.get_rel_map(entities,
                                                         depth=depth,
                                                         limit=limit)
        return rel_map

    def set_retriever(self, graph_traversal_depth=2, llm_env=None, limit=30, max_entities=5, max_synonyms=0):
        self.retriever = KnowledgeGraphRAGRetriever(
            storage_context=self.storage_context,
            graph_traversal_depth=graph_traversal_depth,
            max_entities=max_entities,
            max_synonyms=max_synonyms,
            retriever_mode='keyword',
            verbose=False,
            entity_extract_template=llm_env.keyword_extract_prompt_template,
            synonym_expand_template=llm_env.synonym_expand_prompt_template,
            # clean_kg_sequences_fn=self.clean_kg_sequences,
            max_knowledge_sequence=limit)
        return self.retriever

    def get_entities(self, query_str: str) -> List[str]:
        """Get entities from query string."""
        return self.retriever._get_entities(query_str)

    def _get_knowledge_sequence(
            self,
            entities: List[str]) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        return self.retriever._get_knowledge_sequence(entities)

    def _build_nodes(
            self,
            knowledge_sequence: List[str],
            rel_map: Optional[Dict[Any, Any]] = None) -> List[NodeWithScore]:

        return self.retriever._build_nodes(knowledge_sequence, rel_map)

    def get_knowledge_sequence(self, rel_map):
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend([
                str(rel_obj) for rel_objs in rel_map.values()
                for rel_obj in rel_objs
            ])
        else:
            print("> No knowledge sequence extracted from entities.")
            return []
        return knowledge_sequence

    def clean_sequence(self,
                       sequence,
                       name_pattern=r'(?<=\{name: )([^{}]+)(?=\})',
                       edge_pattern=r'(?<=\{relationship: )([^{}]+)(?=\})'):
        '''
        kg result: 'James{name: James} -[relationship:{relationship: Joined}]-> Michael jordan{name: Michael jordan}'

        clean the kg result above to James -Joined-> Michael jordan
        '''
        names = re.findall(name_pattern, sequence)
        edges = re.findall(edge_pattern, sequence)
        assert len(names) == sequence.count('{name:')
        assert len(edges) == sequence.count('{relationship:')
        for name in names:
            sequence = sequence.replace(f'{{name: {name}}}', '')
        for edge in edges:
            sequence = sequence.replace(
                f'[relationship:{{relationship: {edge}}}]', f'{edge}')
        return sequence

    def clean_kg_sequences(self, knowledge_sequence):
        clean_knowledge_sequence = [
            self.clean_sequence(seq) for seq in knowledge_sequence
        ]
        return clean_knowledge_sequence

    def clean_rel_map(self, rel_map):
        name_pattern = r'(?<=\{name: )([^{}]+)(?=\})'
        clean_rel_map = {}
        for entity, sequences in rel_map.items():
            name = re.findall(name_pattern, entity)[0]
            clean_ent = entity.replace(f'{{name: {name}}}', '')
            clean_seq = [self.clean_sequence(seq) for seq in sequences]
            clean_rel_map[clean_ent] = clean_seq
        return clean_rel_map

    def build_nodes(self,
                    rel_map,
                    knowledge_sequence,
                    depth=2) -> List[NodeWithScore]:
        """Build nodes from knowledge sequence."""
        new_line_char = "\n"
        context_string = (
            f"The following are knowledge sequence in max depth"
            f" {depth} "
            f"in the form of directed graph like:\n"
            f"`subject -[predicate]->, object, <-[predicate_next_hop]-,"
            f" object_next_hop ...`"
            f" extracted based on key entities as subject:\n"
            f"{new_line_char.join(knowledge_sequence)}")
        if self.verbose:
            print_text(f"Graph RAG context:\n{context_string}\n", color="blue")

        rel_node_info = {
            "kg_rel_map": rel_map,
            "kg_rel_text": knowledge_sequence,
        }
        metadata_keys = ["kg_rel_map", "kg_rel_text"]

        if self.graph_schema != "":
            rel_node_info["kg_schema"] = {"schema": self.graph_schema}
            metadata_keys.append("kg_schema")
        node = NodeWithScore(node=TextNode(
            text=context_string,
            score=1.0,
            metadata=rel_node_info,
            excluded_embed_metadata_keys=metadata_keys,
            excluded_llm_metadata_keys=metadata_keys,
        ))
        return [node]

    def drop(self):
        self.client.drop_space(self.space_name)

    def info(self):
        self.client.info(self.space_name)

    def count_edges(self):
        self.client.count_edges(self.space_name)

    def show_edges(self, limits=10):
        self.client.show_edges(self.space_name, limits)

    def clear(self):
        self.client.clear(self.space_name)

    def show_space(self):
        self.client.show_space()

    def show_triplets(self, file_path=None):
        self.client.show_triplets(self.space_name, file_path)

    def execute(self, query):
        result = self.store.execute(query)
        return result

    def two_hop_parse_triplets(self, query):
        # 定义正则表达式模式
        two_hop_pattern1 = re.compile(r'(.+?) <-(.+?)- (.+?) -(.+?)-> (.+)')
        two_hop_pattern2 = re.compile(r'(.+?) <-(.+?)- (.+?) <-(.+?)- (.+)')
        two_hop_pattern3 = re.compile(r'(.+?) -(.+?)-> (.+?) -(.+?)-> (.+)')
        two_hop_pattern4 = re.compile(r'(.+?) -(.+?)-> (.+?) <-(.+?)- (.+)')

        one_hop_pattern5 = re.compile(r'(.+?) -(.+?)-> (.+)')
        one_hop_pattern6 = re.compile(r'(.+?) <-(.+?)- (.+)')

        match = two_hop_pattern1.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1),
                    (entity2, relation2, entity3)]

        match = two_hop_pattern2.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1),
                    (entity3, relation2, entity2)]

        match = two_hop_pattern3.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2),
                    (entity2, relation2, entity3)]

        match = two_hop_pattern4.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2),
                    (entity3, relation2, entity2)]

        match = one_hop_pattern5.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity1, relation1, entity2)]

        match = one_hop_pattern6.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity2, relation1, entity1)]

        assert False, query

    def two_hop_parse_multi_triplets(self, queries):
        triplets = []
        rel_to_entities = {}
        for query in queries:
            query_triplets = self.two_hop_parse_triplets(query)
            triplets += query_triplets
            if query not in rel_to_entities:
                rel_to_entities[query] = set()
            for triplet in query_triplets:
                rel_to_entities[query].add(triplet[0])
                rel_to_entities[query].add(triplet[2])

        # print(len(rel_to_entities), len(triplets))
        # for x in triplets:
        #     print(type(x), x)
        print(len(rel_to_entities), len(triplets), len(set(triplets)))
        return triplets, rel_to_entities


def test_show_triplets(db: NebulaDB):
    # NameList = ['llama27btest','llama213btest','llama270btest','chatgpttest','rebeltest','vicuna7btest','vicuna33btest','vicuna13btest']
    db.show_triplets()


def test_rel_map(db: NebulaDB):
    entities = ['Lebron james', 'James']
    entities = ['Lebron james']
    entities = ['Lakers', 'James']
    entities = ['James', 'Lakers']
    rel_map = db.get_rel_map(entities=entities, depth=2, limit=30)
    print_text(f"rel_map: {rel_map}\n", color='yellow')
    print_text(f"rel_map has {len(rel_map)} entities, {rel_map.keys()}\n",
               color='red')


def test_clean(db: NebulaDB):
    entities = ['James', 'Lakers']
    rel_map = db.get_rel_map(entities=entities, depth=2, limit=30)
    print_text(f"rel_map: {rel_map}\n", color='yellow')

    knowledge_sequence = db.get_knowledge_sequence(rel_map)
    print_text(f"\nknowledge_sequence: {knowledge_sequence}\n", color='yellow')

    clean_knowledge_sequence = db.clean_kg_sequences(knowledge_sequence)
    print_text(f"\nclean_knowledge_sequence: {clean_knowledge_sequence}\n",
               color='yellow')

    clean_rel_map = db.clean_rel_map(rel_map)
    print_text(f"\nclean_rel_map: {clean_rel_map}\n", color='yellow')


def test_build_nodes(db: NebulaDB):
    entities = ['Lakers']
    rel_map = db.get_rel_map(entities=entities, depth=2, limit=30)
    rel_map1 = db.clean_rel_map(rel_map)
    kg_seq1 = [seq for _, seqs in rel_map1.items() for seq in seqs]
    print_text(f"\nkg_seq1: {kg_seq1}\n", color='yellow')

    entities = ['James']
    rel_map = db.get_rel_map(entities=entities, depth=2, limit=30)
    rel_map2 = db.clean_rel_map(rel_map)
    kg_seq2 = db.get_knowledge_sequence(rel_map)
    kg_seq2 = db.clean_kg_sequences(kg_seq2)
    print_text(f"\nkg_seq2: {kg_seq2}\n", color='yellow')

    assert len(set(rel_map1.keys()) & set(rel_map2.keys())) == 0
    all_rel_map = {**rel_map1, **rel_map2}
    all_kg_seq = kg_seq1 + kg_seq2

    print(f'all_rel_map: {len(all_rel_map)}')
    print(f'all_kg_seq: {len(all_kg_seq)}')
    # print_text(f"\nall_rel_map: {all_rel_map}\n", color='yellow')
    # print_text(f"\nall_kg_seq: {all_kg_seq}\n", color='yellow')

    nodes = db.build_nodes(all_rel_map, all_kg_seq)
    print_text(f"\nnodes: {nodes}\n", color='blue')


def test_parse_2_hop_rel(db: NebulaDB):

    # if not questions:
    #     questions = """Elon musk <-Led by- Team -Play-> Next time
    # Elon musk <-Reports to- Linda yaccarino -Is-> The new ceo of twitter
    # Shaw <-Found work for- Lee <-Followed- Shaw's mother
    # Shaw <-Found work for- Lee -Is-> Harry higgs
    # Shaw <-Found work for- Lee -Will be-> Star
    # Shaw -Dropped-> George -Has-> No vacation days left
    # Shaw -Agreed to-> Marriage -Took place-> Covent garden
    # Shaw -Agreed to-> Marriage <-Beautiful music- Mitchell
    # Shaw -Was tempted by-> Fascism <-Suffered under- Pinocchio
    # """

    entities = ['Twitter', 'Elon musk']
    rel_map = db.get_rel_map(entities=entities, depth=2, limit=30)
    print_text(f"\nrel_map: {rel_map}\n", color='green')
    clean_map = db.clean_rel_map(rel_map)
    print_text(f"\nclean_map: {clean_map}\n", color='yellow')
    knowledge_sequence = db.get_knowledge_sequence(clean_map)
    print_text(f"\knowledge_sequence: {knowledge_sequence}\n", color='blue')
    # for keyword, rels in clean_map.items():
    #     print('###', keyword)
    #     for rel in rels:
    #         print(rel)
    #         result = db.two_hop_parse_triplets(rel)
    #         if result:
    #             print(result)
    #         else:
    #             print(f"No match found for query: {rel}")

    #     triplets, rel_to_entities = db.two_hop_parse_multi_triplets(rels)
    #     for triplet in triplets:
    #         print(triplet)
    #     print(len(triplets))


def query_time(
    nebula_db: NebulaDB,
    entities,
    limit=30,
):
    # all_time = []
    for en in entities:
        # start_time = time.time()
        rel_map = nebula_db.get_rel_map(en, limit=limit)
        for k, v in rel_map.items():
            print('############', k)
            for rel in v:
                print(rel)
        print()

        clean_rel_map = nebula_db.clean_rel_map(rel_map)
        # print(clean_rel_map)

        for k, v in clean_rel_map.items():
            print('############', k)
            for rel in v:
                print(rel)

        # knowledge = nebula_db.clean_kg_sequences()
        # end_time = time.time()
        # all_time.append(end_time - start_time)
        # print(f'{en} rel_map:', ret)
        # print(len(ret.keys()), len(ret.values()[0]))
        # print(f"{en} cost {end_time-start_time:.3f}")

    # print('all query time: ', [f'{x:.2f}' for x in all_time])
    # print('avg:', np.average(all_time))
    # print('sum:', np.sum(all_time))


def test_query_time(nebula_db: NebulaDB):

    #     ret = nebula_db.execute("""
    # # MATCH (talent:entity {name: 'Talent'})-[:relationship*2]-(neighbor)
    # # RETURN DISTINCT neighbor
    # # LIMIT 30

    # MATCH (sub:entity {name: 'Talent'})-[r*1..2]-(neighbor)
    # RETURN DISTINCT r
    # LIMIT 30
    #                             """)

    #     print(ret)

    # q1 = "Which company won Yahoo Finance's 2022 'Company of the Year' Award?"
    # q1_entity = [
    #     'Honoree', 'Yahoo', 'Award', 'Year', 'Recognition', 'Google',
    #     'Organization', 'Company', 'Company of the Year Award', 'Finance',
    #     'Yahoo Finance', 'Financial', '2022', 'Winner', 'Award Winner'
    # ]

    q1_entity = [
        'Top two contestants', 'Finale', "America's Got Talent", 'Talent',
        'Top', 'Contestants', 'Season 17'
    ]
    q1_entity = [
        'Top two contestants', 'Finale', "America's Got Talent", 'Top',
        'Contestants', 'Season 17', 'Talent'
    ]
    query_time(nebula_db, [q1_entity], limit=30)

    # q2 = "How much did Elon Musk bought Twitter?"
    # q2_entity = [
    #     'Twitter', 'Musk', 'Elon', 'Acquisition', 'media', 'expense',
    #     'platform', 'financial', 'tech giant', 'Elon Musk',
    #     'Social media platform', 'Purchase', 'buyout', 'Social', 'transaction',
    #     'entrepreneur', 'financial transaction', 'tech', 'giant', 'Cost'
    # ]
    # query_time(nebula_db, [q2_entity], limit=30)


def test_simple_pruning(nebula_db: NebulaDB):
    # q1_entity = [
    #     'Top two contestants', 'Finale', "America's Got Talent", 'Top',
    #     'Contestants', 'Season 17', 'Talent'
    # ]

    q1_entity = [
        'Top two contestants', 'Finale', "America's Got Talent", 'Talent',
        'Top', 'Contestants', 'Season 17'
    ]

    question = "Who were the top two contestants in the America's Got Talent Season 17 finale?"

    # q1_entity = ['Wallis', 'Susie', 'Susie wallis', 'Film', 'Actress', 'Movie star', 'Motion picture', 'Flick', 'Movie', 'Searches', 'Star', 'Susie searches', 'Picture', 'Motion']

    # question = "Who stars as Susie Wallis in Susie Searches?"

    # # ['Kiersey Clemons']

    # q1_entity = ['New', 'Who', 'Doctor who', 'Companion', 'Doctor', 'Character']
    # question = "Who is the Doctor's new companion in Doctor Who?"

    # question =  "Who will be playing the role of Billy Batson in Shazam! Fury of the Gods?"

    # q1_entity = ['Shazam', 'Divine beings', 'Cosmic', 'Actor', 'Deities', 'Batson', 'Entities', 'Supreme', 'Fury', 'Beings', 'Gods: deities', 'Billy', 'Divine', 'Gods', 'Billy batson', 'Supreme beings', 'Casting', 'Cosmic entities', 'Fury of the gods']

    rel_map = nebula_db.get_rel_map(q1_entity, limit=3000)

    # for k, v in rel_map.items():
    #     print('############', k)
    #     for rel in v:
    #         print(rel)
    # print()

    clean_rel_map = nebula_db.clean_rel_map(rel_map)

    simple_pruning(question, clean_rel_map, topk=30)


def test_ppr(db: NebulaDB):
    entities = ['Twitter', 'Elon musk']
    question = 'What relation between Twitter and Elon musk'
    # rel_map = db.get_rel_map(entities=entities, depth=2, limit=30)
    rel_map = db.get_rel_map(entities=entities, depth=2, limit=30000)
    clean_map = db.clean_rel_map(rel_map)
    print_text(f"\nclean_map: {clean_map}\n", color='yellow')

    all_rels = []
    for rels in clean_map.values():
        all_rels += rels
    print('relations:', len(all_rels))

    triplets, rel_to_entities = db.two_hop_parse_multi_triplets(all_rels)

    filter_rels, filter_triplets = filter_pr_rels(question,
                                                  entities,
                                                  triplets,
                                                  rel_to_entities,
                                                  max_ent=5)

    # for rel in filter_rels:
    #     print(rel)
    # print('filter_rels', len(filter_rels))

    # for rel in filter_triplets:
    #     print(rel)
    # print('filter_triplets', len(filter_triplets))

    return filter_rels, filter_triplets


if __name__ == '__main__':

    # space_name = 'integrationrgb'
    client = NebulaClient()
    client.show_space()
    # client.info(space_name)
    # client.count_edges(space_name)
    # client.show_edges(space_name, 5)
    # client_1 = NebulaDB("integrationrgb")

    # # create space
    # client = NebulaClient()
    # client.create_space('crag_small')
    # client.show_space()
    # exit(0)

    # drop space
    # client.drop_space('crudrag')
    client.show_space()
    # client.clear('rgb')

    space_name = 'multihop_ccy'
    # space_name = 'kelm_1m'
    # space_name = 'rgb_llama2_70b'
    # space_name = 'newrgb'
    db = NebulaDB(space_name)

    # db.show_edges()
    db.info()
    db.count_edges()
    db.show_space()

    space_name = 'multihop_ccy'
    # client.drop_space('hotpotqa')
    # client.show_space()
    # db.show_space()

    # test_parse_2_hop_rel(db)
    # test_rel_map(db)
    # test_clean(db)
    # test_build_nodes(db)
    # test_show_triplets(db)
    # test_query_time(db)
    # test_ppr(db)
    # test_simple_pruning(db)
