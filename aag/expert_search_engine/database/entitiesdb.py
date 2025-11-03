from aag.reasoner.model_deployment import EmbeddingEnv
from aag.expert_search_engine.database.milvus import myMilvus, MilvusDB
from aag.utils.timer import Timer
from aag.expert_search_engine.data_process.openai_extractor.gpt_extract_triplets import OpenAIExtractor
from tqdm import tqdm


class EntitiesDB:

    def __init__(self,
                 db_name,
                 entities,
                 embed_name="BAAI/bge-large-en-v1.5",
                 overwrite=True,
                 step=100,
                 device='cuda:0',
                 verbose=False):
        self.openai_extractor = OpenAIExtractor()
        self.embed_model = EmbeddingEnv(embed_name=embed_name, device=device)

        self.entities = sorted(list(entities))
        self.db_name = db_name

        self.id2entity = {i: entity for i, entity in enumerate(self.entities)}

        self.milvus_client = myMilvus()
        self.timer = Timer(verbose=verbose)

        create_new_db = True

        if self.milvus_client.has_collection(db_name):
            print(f"exist {self.milvus_client.has_collection(db_name)}")
            print(f"count {self.milvus_client.get_vector_count(db_name)}")
            print(f"entities {len(entities)}")

        if entities and self.milvus_client.has_collection(
                db_name) and self.milvus_client.get_vector_count(
                    db_name) == len(entities):
            create_new_db = False
            print(f"{db_name} is existing!")

        overwrite = overwrite or create_new_db

        if overwrite:
            assert entities, 'need specify the entities when create new vector database.'

        self.db = MilvusDB(db_name,
                           1024,
                           overwrite=overwrite,
                           metric='COSINE',
                           verbose=False)
        if overwrite:
            # Strong, Bounded, Eventually, Session
            self.db.create(consistency_level="Strong")
            self.generate_embedding_and_insert(step=step)

        self.db.load()

    # def generate_embedding_and_insert(self):
    #     print(
    #         f'start generate emebedding for {self.db_name} and insert to database...'
    #     )
    #     step = 150
    #     # time.sleep(0.5)
    #     n_entities = len(self.entities)
    #     for i in tqdm(range(0, n_entities, step),
    #                   f'insert vector to {self.db_name}'):
    #         start_idx = i
    #         end_idx = min(n_entities, i + step)
    #         # print(start_idx, end_idx)
    #         # print(start_idx, end_idx)
    #         embeddings = self.get_embedding(self.entities[start_idx:end_idx])
    #         ids = list(range(start_idx, end_idx))
    #         self.insert(ids, embeddings)
    #         assert len(ids) == len(embeddings)
    #         # print(ids)
    #         # if i % (step *  10) == 0:
    #         #     print(f'{get_date_now()} insert {len(ids)} vectors')
    def generate_embedding_and_insert(self, step=150, start_num=0):
        print(
            f'start generate emebedding for {self.db_name} and insert to database...'
        )
        # time.sleep(0.5)
        n_entities = len(self.entities)
        for i in tqdm(range(0, n_entities, step),
                      f'insert vector to {self.db_name}'):
            start_idx = i
            end_idx = min(n_entities, i + step)
            # print(start_idx, end_idx)
            # print(start_idx, end_idx)
            embeddings = self.get_embedding(self.entities[start_idx:end_idx])
            ids = list(range(start_idx + start_num, end_idx + start_num))
            self.insert(ids, embeddings)
            assert len(ids) == len(embeddings)
            # print(ids)
            # if i % (step *  10) == 0:
            #     print(f'{get_date_now()} insert {len(ids)} vectors')

    def get_embedding(self, query):
        with self.timer.timing('embedding_one_query'):
            if isinstance(query, list):
                ret = self.embed_model.get_embeddings(query)
            else:
                ret = self.embed_model.get_embedding(query)
        return ret

    def search(self, query_embedding, limit=3):
        assert isinstance(query_embedding, list)
        if not isinstance(query_embedding[0], list):
            query_embedding = [query_embedding]

        with self.timer.timing('search'):
            ids, distances = self.db.search(query_embedding, limit=limit)
        return (ids, distances)

    def insert(self, id, query_embedding):
        if not isinstance(id, list):
            id = [id]
            query_embedding = [query_embedding]

        with self.timer.timing('search'):
            self.db.insert([id, query_embedding])

    def get_query_entities(self, question_str, max_keywords=2):
        entities = self.openai_extractor.get_keyword_from_question(
            question_str, max_keywords=max_keywords)
        return entities

    def get_filter_keyword_from_question_by_openai(self, question_str, entities):
        entities = self.openai_extractor.get_filter_keyword_from_question(
            question_str, entities)
        return entities
