import logging
# import QueryBundle
from llama_index.core.utils import print_text
from llama_index.core.response_synthesizers.type import *
from typing import List
from typing import Optional
from abc import ABC, abstractmethod
from aag.expert_search_engine.database.milvus import *
from aag.expert_search_engine.database.nebulagraph import *
from aag.expert_search_engine.database.entitiesdb import *
from aag.reasoner.model_deployment import Reasoner

from aag.rag_engine.vector_rag import VectorRAG
from aag.rag_engine.graph_rag import GraphRAG


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


