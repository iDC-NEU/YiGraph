
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.ingestion import run_transformations
from llama_index.core import (
    Document,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from graphllm.model_deploy.model_deployment import OllamaEnv
from typing import Literal
# from dataprocess.kg_extract import llm_kg_extract
from graphllm.utils.pruning import *
import os
import re
import math

"""
    mrr, hit  比较检索的上下文和真实的上下文质量
"""


def to_lower_nested_list(nested_list):
    """递归地将嵌套列表中的所有字符串转换为小写。"""
    if isinstance(nested_list, list):
        return [to_lower_nested_list(item) for item in nested_list]
    elif isinstance(nested_list, str):
        return nested_list.lower()
    else:
        return nested_list


def flatten_nested_list(nested_list):
    """将嵌套列表展平成一维列表。"""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list


def get_triples(target_texts):
    transformations = Settings.transformations
    if isinstance(target_texts, list):
        docs = [Document(text=t) for t in target_texts]
    else:
        docs = [Document(text=target_texts)]
    nodes = run_transformations(
        docs,  # type: ignore
        transformations,
        show_progress=True,
    )
    llm_env = OllamaEnv(llm_mode_name="llama2:70b",
                        embed_batch_size=10)
    all_triplets = llm_kg_extract(
        llm_env,
        nodes,
        triplets_per_chunk=10,
    )
    return set(all_triplets)


def two_hop_parse_triplets(query):
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


def two_hop_parse_multi_triplets(queries):
    triplets = []
    rel_to_entities = {}
    for query in queries:
        query_triplets = two_hop_parse_triplets(query)
        triplets += query_triplets
    return set(triplets)


def parse_paths_to_triples(retrieved_text):
    """
    Parse a complex list of paths into triples (entity1, relation, entity2).

    Parameters:
        paths (list): Each element is a path string, such as:
                    "76th edition of the citrus bowl <-Was the- Citrus bowl 2022 -Was played on-> January 1, 2022"

    Returns:
        list: A list of parsed triples.
    """
    triples = []
    for path in retrieved_text:
        parts = re.split(r'(\-.*?->|\<-.*?\-)', path)
        i = 0
        while i < len(parts):
            if '<-' in parts[i]:
                predicate = parts[i].replace('<-', '').replace('-', '')
                subject = parts[i + 1].strip()
                object_ = parts[i - 1].strip()
                triples.append([subject, predicate, object_])
                i += 2
            elif '->' in parts[i]:
                predicate = parts[i].replace('->', '').replace('-', '')
                subject = parts[i - 1].strip()
                object_ = parts[i + 1].strip()
                triples.append([subject, predicate, object_])
                i += 2
            else:
                i += 1
    return triples



def parse_paths_to_triples_by_sentence(retrieved_text):
    """
    Parse a complex list of paths into triples (entity1, relation, entity2).

    Parameters:
        paths (list): Each element is a path string, such as:
                    "76th edition of the citrus bowl <-Was the- Citrus bowl 2022 -Was played on-> January 1, 2022"

    Returns:
        list: A list of parsed triples.
    """
    triples = []
    for path in retrieved_text:
        parts = re.split(r'(\-.*?->|\<-.*?\-)', path)
        i = 0
        while i < len(parts):
            if '<-' in parts[i]:
                predicate = parts[i].replace('<-', '').replace('-', '')
                subject = parts[i + 1].strip()
                object_ = parts[i - 1].strip()
                # 把 predicated 和 subject 和 object 组合成一个句子:  "subject-predicate-> object_和 object 组合成一个句子"
                triple_str = f"{subject}-{predicate}->{object_}"
                triples.append(triple_str)
                i += 2
            elif '->' in parts[i]:
                predicate = parts[i].replace('->', '').replace('-', '')
                subject = parts[i - 1].strip()
                object_ = parts[i + 1].strip()
                triple_str = f"{subject}-{predicate}->{object_}"
                triples.append(triple_str)
                i += 2
            else:
                i += 1
    return triples


def parse_sentence(retrieved_text):
    triples = []
    for sentence in retrieved_text:
        parts = sentence.split(' ')
        i = 0
        while i < len(parts):
            if '<-' in parts[i]:
                predicate = parts[i].replace('<-', '').replace('-', '')
                subject = parts[i + 1]
                object_ = parts[i - 1]
                triples.append((subject, predicate, object_))
                i += 2
            elif '->' in parts[i]:
                predicate = parts[i].replace('->', '').replace('-', '')
                subject = parts[i - 1]
                object_ = parts[i + 1]
                triples.append((subject, predicate, object_))
                i += 2
            else:
                i += 1
    return set(triples)


class RetrievalEvaluator:

    def __init__(self,
                 ) -> None:
        pass

    def evaluate_graph_precision(self, retrieved_context, ground_truth_context):
        retrieved_triplets = parse_paths_to_triples(retrieved_context)
        retrieved_triplets = self._remove_duplicates(retrieved_triplets)

        is_evidences_triplets = self._is_evidences_triplets(
            ground_truth_context)
        # is_evidences_triplets = True

        len_ground_truth_triples = len(ground_truth_context)

        if not is_evidences_triplets:
            ground_truth_context_parse = []
            for each_target_triplets_set in ground_truth_context:
                for target_triple in each_target_triplets_set:
                    if isinstance(target_triple, list) and len(target_triple) == 3:
                        ground_truth_context_parse.append(target_triple)
                    elif isinstance(target_triple, str):
                        parsed_triples = parse_paths_to_triples(
                            [target_triple])
                        ground_truth_context_parse.extend(parsed_triples)
                    else:
                        print(f"unknow target type: {target_triple}")
            ground_truth_context = ground_truth_context_parse

        ground_truth_context = self._remove_duplicates(ground_truth_context)

        # Calculate the numerator
        numerator = sum(
            (1 if retrieved_triplets[i]
             in ground_truth_context else 0) * self.weight(i + 1)
            for i in range(len(retrieved_triplets))
        )

        # Calculate the denominator
        denominator = sum(self.weight(t)
                          for t in range(1, len_ground_truth_triples + 1))

        # Return retrieval precision
        return numerator / denominator if denominator != 0 else 0

    def evaluation_precision(self, retrieved_context, ground_truth_context):
        """
        Calculate the retrieval precision metric.

        Parameters:
            retrieved_context (list): List of retrieved documents, ranked from 1 onward.
            ground_truth_context (set): Set of ground truth relevant documents.

        Returns:
            float: The calculated retrieval precision.
        """

        # Calculate the numerator
        numerator = sum(
            (1 if retrieved_context[i]
             in ground_truth_context else 0) * self.weight(i + 1)
            for i in range(len(retrieved_context))
        )

        # Calculate the denominator
        denominator = sum(self.weight(t)
                          for t in range(1, len(ground_truth_context) + 1))

        # Return retrieval precision
        return numerator / denominator if denominator != 0 else 0

    # Define the weight function w(ri) = 1 / log(ri + 1) for each rank
    def weight(self, rank):
        return 1 / math.log(rank + 1, 2)   # Using log base 2

    def _evaluate_vector_recall(self, retrieved_context, ground_truth_context):
        """
        Calculate recall using the vector-based approach.

        Parameters:
            retrieved_context (list): List of retrieved documents.
            ground_truth_context (set): Set of ground truth relevant documents.

        Returns:
            float: The calculated recall for vector mode.
        """
        # Calculate the number of relevant retrieved documents (Ri in T)
        relevant_retrieved = sum(
            1 for item in retrieved_context if item in ground_truth_context)

        # Calculate the recall by dividing by the size of the ground truth set
        recall = relevant_retrieved / \
            len(ground_truth_context) if ground_truth_context else 0
        return recall

    def _is_hit(self, gt_item, retrieved_triplets):
        """
        Check if a single retrieval item matches a groundtruth item.
        - If gt_item is a string (path), parse it into triples and check if all triples are in ret_item.
        - If gt_item is a regular list, check if it matches ret_item directly.
        """
        if isinstance(gt_item, str):  # path
            parsed_triples = parse_paths_to_triples([gt_item])
            return all(triple in retrieved_triplets for triple in parsed_triples)
        return gt_item in retrieved_triplets

    def _remove_duplicates(self, data):
        """
        Remove duplicate lists from a list of lists while preserving the order.

        Parameters:
            data (list): A list of lists, where duplicates will be removed.

        Returns:
            list: A new list with duplicates removed, preserving the original order.
        """
        unique_data = []
        for item in data:
            if item not in unique_data:
                unique_data.append(item)
        return unique_data

    def _evaluate_graph_recall(self, retrieved_context, ground_truth_context):
        """
        Calculate recall using the graph-based approach.

        Parameters:
            retrieved_context (list): List of retrieved graph nodes/edges.
            ground_truth_context (set): Set of ground truth graph nodes/edges.

        Returns:
            float: The calculated recall for graph mode.
        """
        retrieved_triplets = parse_paths_to_triples(retrieved_context)
        retrieved_triplets = self._remove_duplicates(retrieved_triplets)

        is_evidences_triplets = self._is_evidences_triplets(
            ground_truth_context)
        # is_evidences_triplets = True

        hits = 0
        if is_evidences_triplets:  # evidences triplets
            ground_truth_context = self._remove_duplicates(
                ground_truth_context)
            for each_target_triple in ground_truth_context:
                if self._is_hit(each_target_triple, retrieved_triplets):
                    hits += 1
        else:  # merged triplets
            for each_target_triplets_set in ground_truth_context:
                for target_triple in each_target_triplets_set:
                    if self._is_hit(target_triple, retrieved_triplets):
                        hits += 1
                        break
        recall = hits / \
            len(ground_truth_context) if ground_truth_context else 0
        return recall

    def evaluation_recall(self, retrieved_context, ground_truth_context, mode=Literal["vector", "graph"]):
        """
        Calculate the retrieval recall metric based on the specified mode.

        Parameters:
            retrieved_context (list): List of retrieved documents.
            ground_truth_context (set): Set of ground truth relevant documents.
            mode (str): The mode of evaluation, either "vector" or "graph". Defaults to "vector".

        Returns:
            float: The calculated retrieval recall.
        """
        if mode == "vector":
            return self._evaluate_vector_recall(retrieved_context, ground_truth_context)
        elif mode == "graph":
            return self._evaluate_graph_recall(retrieved_context, ground_truth_context)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _evaluate_vector_relevance(self, retrieved_context, ground_truth_context):
        """
        Calculate relevance using the vector-based approach

        Parameters:
            retrieved_context (list): List of retrieved documents.
            ground_truth_context (set): Set of ground truth relevant documents.

        Returns:
            float: The calculated relevance for vector mode.
        """
        # Calculate the number of relevant retrieved documents (Ri in T)
        relevant_retrieved = sum(
            1 for item in retrieved_context if item in ground_truth_context)

        # Calculate the relevance by dividing by the size of the relevant_retrieved set
        relevance = relevant_retrieved / \
            len(retrieved_context) if retrieved_context else 0
        return relevance

    def _is_evidences_triplets(self, ground_truth_context):
        pattern = r"(\-.*?->|\<-.*?\-)"
        for item in ground_truth_context:
            if not isinstance(item, list) or len(item) != 3:
                return False
            for element in item:
                if isinstance(element, list) or re.search(pattern, element):
                    return False
        return True

    # note 检索的结果没有去重
    def _evaluate_graph_relevance(self, retrieved_context, ground_truth_context):
        """
        Calculate relevance using the graph-based approach.

        Parameters:
            retrieved_context (list): List of retrieved documents (or graph nodes/edges).
            ground_truth_context (set): Set of ground truth relevant documents (or graph nodes/edges).

        Returns:
            float: The calculated relevance for graph mode.
        """

        retrieved_triplets = parse_paths_to_triples(retrieved_context)
        # retrieved_triplets = list(set(retrieved_triplets))

        is_evidences_triplets = self._is_evidences_triplets(
            ground_truth_context)
        # is_evidences_triplets = True

        if not is_evidences_triplets:
            ground_truth_context_parse = []
            for each_target_triplets_set in ground_truth_context:
                for target_triple in each_target_triplets_set:
                    if isinstance(target_triple, list) and len(target_triple) == 3:
                        ground_truth_context_parse.append(target_triple)
                    elif isinstance(target_triple, str):
                        parsed_triples = parse_paths_to_triples(
                            [target_triple])
                        ground_truth_context_parse.extend(parsed_triples)
                    else:
                        print(f"unknow target type: {target_triple}")
            ground_truth_context = ground_truth_context_parse

        ground_truth_context = self._remove_duplicates(ground_truth_context)

        hits = 0
        for each_retrieved_triple in retrieved_triplets:
            if each_retrieved_triple in ground_truth_context:
                hits += 1
        relevance = hits / len(retrieved_triplets) if retrieved_triplets else 0
        return relevance

    def evaluation_relevance(self, retrieved_context, ground_truth_context, mode=Literal["vector", "graph"]):
        """
        Calculate the retrieval relevance metric based on the specified mode.

        Parameters:
            retrieved_context (list): List of retrieved documents.
            ground_truth_context (set): Set of ground truth relevant documents.
            mode (str): The mode of evaluation, either "vector" or "graph". Defaults to "vector".

        Returns:
            float: The calculated retrieval relevance.
        """
        if mode == "vector":
            return self._evaluate_vector_relevance(retrieved_context, ground_truth_context)
        elif mode == "graph":
            return self._evaluate_graph_relevance(retrieved_context, ground_truth_context)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def hit_rate_vector_rag(self, retrieved_texts, target_texts):
        """
            recall
            Calculate Hit Rate at k.

            :param retrieved_texts: A list of lists, where each sublist contains the predicted items for a query
            :param target_texts: A list of lists, where each sublist contains the actual relevant items for a query
            :param k: The number of top predicted items to consider for each query
            :return: Hit Rate at k
        """
        hits = 0
        retrieved_texts = to_lower_nested_list(retrieved_texts)
        target_texts = to_lower_nested_list(target_texts)

        # retrieved_texts = flatten_nested_list(retrieved_texts)
        # target_texts = flatten_nested_list(target_texts)
        # for pred, truth in zip(retrieved_texts, target_texts):
        #     # Check if any of the top-k predictions is in the ground truth
        #     # print(f"pred[:k]:{pred[:k]}")
        #     if any(item in truth for item in pred):
        #         hits += 1
        # 计算命中次数并记录命中的文本
        hits = 0
        hit_texts = []

        for pred in retrieved_texts:
            if pred in target_texts:
                hits += 1
                hit_texts.append(pred)

        total_target = len(target_texts)
        if total_target == 0:
            return 0, []
        hit_rate = round(hits / total_target, 2) if total_target > 0 else 0.0
        print(f"Total text length: {total_target}")
        print(f"Hits: {hit_rate}")
        print(f"Hit Texts: {hit_texts}")
        return hit_rate, hit_texts

    def precision_rate_vector_rag(self, retrieved_texts, target_texts):
        hits = 0
        retrieved_texts = to_lower_nested_list(retrieved_texts)
        target_texts = to_lower_nested_list(target_texts)

        hits = 0
        hit_texts = []

        for pred in retrieved_texts:
            if pred in target_texts:
                hits += 1
                hit_texts.append(pred)

        total_retrieval = len(retrieved_texts)
        if total_retrieval == 0:
            return 0, []
        precision_rate = round(hits / total_retrieval,
                               2) if total_retrieval > 0 else 0.0
        print(f"Total Queries: {total_retrieval}")
        print(f"precision: {precision_rate}")
        print(f"Hit Texts: {hit_texts}")

        return precision_rate, hit_texts

    # 计算相似性，或者挖三元组

    def recall_rate_graph_rag(self, graph_retrieved_texts, target_texts):
        target_text_triples = get_triples(target_texts)
        retrieved_text_triples = two_hop_parse_multi_triplets(
            graph_retrieved_texts)
        match_count = 0

        for triple1 in retrieved_text_triples:
            for triple2 in target_text_triples:
                similarity = cosine_similarity_np(np.array(get_text_embedding(triple1)).reshape(
                    1, -1), np.array(get_text_embedding(triple2)).reshape(1, -1))
                if similarity > 0.8:
                    match_count += 1
                    print(f"match item:{triple1}, {triple2}")
                    break  # 如果找到了一个匹配项，可以跳过该 triple2

        precision = round(match_count / len(retrieved_text_triples),
                          2) if retrieved_text_triples else 0.0
        # recall = round(match_count / len(target_text_triples), 2) if target_text_triples else 0.0
        return precision

    def cohere_rerank_metric(self):
        os.environ["COHERE_API_KEY"]
        pass


if __name__ == "__main__":
    path = [
        "Victoria general hospital -Is the hospital for-> Y. mendoza",
        "Victoria general hospital -Is the hospital for-> Y. mendoza <-Is the historian for- Mother",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Was admitted on-> 10th, march",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Was born in-> Victoria",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Resides in-> Victoria",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Recorded on-> 10th, march",
        "Victoria general hospital -Is the hospital for-> Y. mendoza <-Is the physician for- Dr. john smith",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Is-> Male",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Head and organs are observed as-> Normal",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Has-> Normal cbc",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Is categorized as-> Newborn",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Resides at-> 32, riverside street, victoria",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Is-> 8 hours old",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Has chief complaint of-> Limited mouth opening",
        "Victoria general hospital -Is the hospital for-> Y. mendoza <-Accompany- No other noticeable symptoms",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Pulse is recorded as-> 140 bpm",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Neck is confirmed as-> Supple",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Has-> No disease history",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Is-> Stable after initial management",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Temperature is recorded as-> 36.8°c",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Diagnosis basis is based on-> Limited mouth opening",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Demonstrates-> Normal urination and defecation",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Limbs showed-> Normal movement",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Has chief complaint of-> Crying",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Shows-> Reduced sleep",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Presents with symptoms of-> Crying",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Presents with symptoms of-> Limited mouth opening",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Diagnosis basis is based on-> Crying",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Diagnosis basis is based on-> History of perinatal period",
        "Victoria general hospital -Is the hospital for-> Y. mendoza -Has-> No surgery or trauma history"
    ]
    test = [
        ['Jonas vingegaard', 'Is', 'The overall winner of the 2022 tour de france'],
        ['The overall winner of the 2022 tour de france', 'Is', 'Jonas vingegaard']
    ]
    print(RetrievalEvaluator()._remove_duplicates)
    # result = parse_paths_to_triples(path)
    # for triple in result:
    #     print(triple)
    #     print("\n")
    # predictions = [
    #     ["iTem2", "1234"],
    # ]

# ground_truths = [
#     ["item2", "23445", "454555", "1234"],
#     ["iTem2", "1234"],
#     ["item10"],
#     ["item12"],
#     ["item16"]
# ]

# predictions1 = [
#     "iTem2", "1234",

# ]

# ground_truths1 = [
#     "item2 iuid u", "23445", "454555", "1234",
#     "item5",
#     "item10",
#     "item12",
#     "item16"
# ]

# # RetrievalEvaluator().hit_rate_vector_rag(predictions, ground_truths)
# # print("Hit Rate at  {:.2f}".format(hit_rate_at_k))

# graph_retrieval = {"Nephron": [["CONSISTS_OF", "Tubule"], ["IS", "Basic functional unit"], ["CONSISTS_OF", "Collecting duct"], ["CONSISTS_OF", "Blood filter"]], "Kidney": [["ELIMINATES", "Metabolic waste"]], "Renal": [["TRANSCRIPTION", "Factors"]], "Kidneys": [["INCUBATED", "30% sucrose"]], "Nephrons": [["FUSE", "Cloaca"]], "Findings": [["BROADLY_AGREE", "Results"], [
#     "NOTEWORTHY", "First published report"], ["BROADLY_AGREE", "Results", "SHOWED", "Figure 7b"], ["BROADLY_AGREE", "Results", "SUPPORTED", "Finding"], ["BROADLY_AGREE", "Results", "OBTAINED", "Necessarily"], ["BROADLY_AGREE", "Results", "DIFFER", "Qualitatively"], ["BROADLY_AGREE", "Results", "MAY_BE_USED", "Risk stratification"], ["BROADLY_AGREE", "Results", "HAVE_TO_BE", "Correctly interpreted"]]}

# ground_truths1 = [
#     ["Nephron CONSISTS_OF Tubule"],
#     "item5",
#     "item10",
#     "item12",
#     "item16"
# ]

# # print(RetrievalEvaluator().recall_rate_graph_rag(graph_retrieval, ground_truths))
