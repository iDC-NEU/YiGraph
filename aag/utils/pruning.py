import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_EMBED_MODEL = None


def _get_embed_model():
    """Lazily initialize the HuggingFace embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    model_name = os.getenv("AAG_PRUNING_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    batch_size = int(os.getenv("AAG_PRUNING_EMBED_BATCH_SIZE", "10"))
    device = os.getenv("AAG_PRUNING_EMBED_DEVICE", "cuda:0")

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=batch_size,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001 - propagate friendly message
        logger.error(
            "无法初始化 HuggingFaceEmbedding(model=%s, device=%s): %s",
            model_name,
            device,
            exc,
        )
        raise

    return _EMBED_MODEL


class RelItem:
    def __init__(self, key, values):
        self.key = key
        self.values = values


def get_text_embedding(text):
    embedding = _get_embed_model()._get_text_embedding(text)
    return embedding


def get_text_embeddings(texts, step=400):
    all_embeddings = []
    n_text = len(texts)
    # time_s = time.time()
    for start in range(0, n_text, step):
        input_texts = texts[start:min(start + step, n_text)]
        # print('process', len(input_texts))
        embeddings = _get_embed_model()._get_text_embeddings(input_texts)

        # print('start', start)
        all_embeddings += embeddings
    # time_e = time.time()
    # print(f'get_text_embeddings cost ({n_text}) {time_e - time_s:.3f}')
    # print(len(all_embeddings))
    return all_embeddings


def cosine_similarity_np(
    embeddings1,
    embeddings2,
) -> float:
    product = np.dot(embeddings1, embeddings2.T)

    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    norm_product = np.dot(norm1, norm2.T)
    cosine_similarities = product / norm_product

    return cosine_similarities


def simple_pruning(question, knowledge_sequence, topk=30):

    # rel_map = nebula_db.get_rel_map(q1_entity, limit=3000)
    # clean_rel_map = nebula_db.clean_rel_map(rel_map)

    all_rel_scores = []

    question_embed = np.array(get_text_embedding(question)).reshape(1, -1)

    # for ent, rels in clean_rel_map.items():
    rel_embeddings = get_text_embeddings(knowledge_sequence)
    if len(rel_embeddings) == 1:
        rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
    else:
        rel_embeddings = np.array(rel_embeddings)
    # time_s = time.time()
    # similarity = cosine_similarity(question_embed, rel_embeddings)
    similarity = cosine_similarity_np(question_embed, rel_embeddings)[0]
    # time_e = time.time()
    # print(similarity.shape)
    # print(similarity[:10])
    # print(f'similarity time cost {time_e - time_s:.3f}')

    all_rel_scores += [
        (rel, score)
        for rel, score in zip(knowledge_sequence, similarity.tolist())
    ]
    # print(len(all_rel_scores))

    sorted_all_rel_scores = sorted(all_rel_scores,
                                   key=lambda x: x[1],
                                   reverse=True)

    # print('\n###########################')
    # for rel, score in sorted_all_rel_scores[:]:
    #     if 'Kristy sellar' in rel or 'Mayyas' in rel:
    #         print(rel, f'{score:.3f}')

    # print('\n###########################')
    # for rel, score in sorted_all_rel_scores[:topk]:
    #     print(rel, f'{score:.3f}')

    # print('\n###########################')
    # for rel, score in sorted_all_rel_scores[:topk]:
    #     print(rel)

    return sorted_all_rel_scores[:topk]


def simple_pruning_neo4j(question, rel_map, topk=30):

    all_rel_scores = []
    question_embed = np.array(get_text_embedding(question)).reshape(1, -1)
    time1 = time.time()
    ''''组织成句子'''
    key_value_rel = []
    sentences = []
    for key, values in rel_map.items():
        for value in values:
            sentence = str(key) + ' '.join(value)
            key_value_rel.append(RelItem(key, value))
            sentences.append(sentence)
    time1 = time.time() - time1
    time2 = time.time()
    rel_embeddings = get_text_embeddings(sentences)
    if len(rel_embeddings) == 1:
        rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
    else:
        rel_embeddings = np.array(rel_embeddings)
    # time_s = time.time()
    # similarity = cosine_similarity(question_embed, rel_embeddings)
    similarity = cosine_similarity_np(question_embed, rel_embeddings)[0]
    # time_e = time.time()
    # print(similarity.shape)
    # print(similarity[:10])
    # print(f'similarity time cost {time_e - time_s:.3f}')
    time2 = time.time() - time2
    all_rel_scores += [
        (rel, score)
        for rel, score in zip(key_value_rel, similarity.tolist())
    ]
    # print(len(all_rel_scores))

    sorted_all_rel_scores = sorted(all_rel_scores,
                                   key=lambda x: x[1],
                                   reverse=True)

    # print('\n###########################')
    # for rel, score in sorted_all_rel_scores[:]:
    #     if 'Kristy sellar' in rel or 'Mayyas' in rel:
    #         print(rel, f'{score:.3f}')

    # print('\n###########################')
    # for rel, score in sorted_all_rel_scores[:topk]:
    #     print(rel, f'{score:.3f}')

    # print('\n###########################')
    # for rel, score in sorted_all_rel_scores[:topk]:
    #     print(rel)
    time3 = time.time()
    sort_rel_map = {}
    get_top_rel = sorted_all_rel_scores[:topk]
    for rel, score in get_top_rel:
        if rel.key in sort_rel_map:
            sort_rel_map[rel.key].extend([rel.values])
        else:
            sort_rel_map[rel.key] = [rel.values]
    time3 = time.time() - time3
    print(f"time1:{time1:.3f}")
    print(f"time2:{time2}")
    print(f"time3:{time3}")

    return sort_rel_map


def limit_rel_by_numbers(rel_map, limit=1):
    for key, values in rel_map.items():
        if len(values) > limit:
            rel_map[key] = values[:limit]
    return rel_map


if __name__ == "__main__":
    rel_map = {}
    rel_map["C"] = [["IS", "Third goalkeeper"], ["IS", "Third goalkeeper"]]
    rel_map["Call"] = [["POSSIBLE", "Third goalkeeper"], ["WILL_CONTAIN", "Third goalkeeper"], ["WILL_CONTAIN", "Forward-looking statements", "INCLUDED_HEREIN", "Made only as of the date hereof"], ["WILL_CONTAIN", "Forward-looking statements", "OFTEN_ADDRESS", "Expected future business and financial performance"], ["WILL_CONTAIN", "Forward-looking statements", "INVOLVE", "Risks and uncertainties"], ["WILL_CONTAIN",
                                                                                                                                                                                                                                                                                                                                                                                                                   "Forward-looking statements", "EXPRESSED_OR_IMPLIED", "Mandiant's actual results"], ["WILL_CONTAIN", "Forward-looking statements", "ARE", "Only predictions"], ["WILL_CONTAIN", "Forward-looking statements", "DIFFER_MATERIALLY_FROM", "Actual outcomes and results"], ["WILL_CONTAIN", "Forward-looking statements", "ARE_INCLUDED_UNDER", "Risk factors caption in irobot's most recent annual and quarterly reports filed with the sec"]]
    rel_map["Carole"] = [["HAS_CAREER_IN", "Third goalkeeper"], ["HAS_CAREER_IN", "Tapestry", "PENNED", "Album"], ["HAS_CAREER_IN", "Tapestry", "IS", "Store"], ["HAS_CAREER_IN", "Tapestry", "HAS", "Years"], ["HAS_CAREER_IN", "Tapestry", "HAS",
                                                                                                                                                                                                                "Career"], ["VISITED", "Store"], ["VISITED", "Store", "CLOSED", "June 2015"], ["VISITED", "Store", "REOPENED", "September"], ["VISITED", "Store", "SELLING", "Iphones"], ["EXPLORED", "Career"], ["EXPLORED", "Career", "IS", "Through years"]]
    question = "what is Third goalkeeper"
    # test_time = time.time()
    # reuslt = simple_pruning(question, rel_map, topk=5)
    # test_time1 = time.time() - test_time
    # print(reuslt)
    # print(test_time1)
    print(limit_rel_by_numbers(rel_map))
