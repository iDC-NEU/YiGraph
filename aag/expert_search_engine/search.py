import logging
from llama_index.core.utils import print_text

from llama_index.core.response_synthesizers.type import *
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from typing import List
from typing import Optional
from aag.expert_search_engine.database.milvus import *
from aag.expert_search_engine.database.nebulagraph import *
from aag.expert_search_engine.database.entitiesdb import *
from aag.reasoner.model_deployment import Reasoner
from aag.config.engine_config import RetrievalConfig
from aag.utils.file_operation import read_yaml
from aag.utils.path_utils import TASK_TYPES_PATH, ALGORITHMS_PATH, KNOWLEDGE_PATH

from aag.rag_engine.vector_rag import VectorRAG
from aag.rag_engine.graph_rag import GraphRAG

logger = logging.getLogger(__name__)



class RAG_Engine:

    def __init__(
            self,
            vector_db,
            vector_k_similarity,
            graph_db,
            vector_rag_llm_env_ = None,
            graph_rag_llm_env_ = None,
    ):


        from aag.rag_engine.vector_rag import VectorRAG
        from aag.rag_engine.graph_rag import GraphRAG

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

        
        # self.graph_rag = GraphRAG(config=config)
        # self.vector_rag = VectorRAG(config=config)
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

        # self._init_embedding_model()

        # self._init_graph_database()

        # self._init_vector_database()
        
        # self._init_graph_rag()

        # self._init_vector_rag()

         # 构造  task 指向 alg 的分层知识库 函数
        self._build_hierarchical_knowledge_base()
        

    

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
                "description": task_type.get("description", ""),
                "algorithm": task_type.get("algorithm", []),
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
                "Application_scenario": self.algo_index[algorithm_id].get("Application_scenario"),
                "Deployment_method": self.algo_index[algorithm_id].get("Deployment_method"),
                "Principles": self.algo_index[algorithm_id].get("Principles"),
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
