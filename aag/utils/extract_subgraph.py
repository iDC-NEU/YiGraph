from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.utils import print_text

MAX_FACTS = 5000000
MAX_ITER = 20
RESTART = 0.8
MAX_ENT = 500
NOTFOUNDSCORE = 0.
EXPONENT = 2.
MAX_SEEDS = 1
DECOMPOSE_PPV = True
SEED_WEIGHTING = True
RELATION_WEIGHTING = True
FOLLOW_NONCVT = True
USEANSWER = False


def personalized_pagerank(seed, W):
    """Return the PPR vector for the given seed and adjacency matrix.

    Args:
        seed: A sparse matrix of size E x 1.
        W: A sparse matrix of size E x E whose rows sum to one.

    Returns:
        ppr: A vector of size E.
    """
    restart_prob = RESTART
    r = restart_prob * seed
    s_ovr = np.copy(r)
    for i in range(MAX_ITER):
        r_new = (1. - restart_prob) * (W.transpose().dot(r))
        s_ovr = s_ovr + r_new
        delta = abs(r_new.sum())
        if delta < 1e-5: 
            break
        r = r_new
    return np.squeeze(s_ovr)


def get_subgraph(entities, multigraph_W, max_ent=30):
    """Get subgraph describing a neighbourhood around given entities."""
    seed = np.zeros((multigraph_W.shape[0], 1))
    if not SEED_WEIGHTING:
        seed[entities] = 1. / len(set(entities))
    else:
        seed[entities] = np.expand_dims(np.arange(len(entities), 0, -1),
                                        axis=1)
        seed = seed / seed.sum()
    ppr = personalized_pagerank(seed, multigraph_W)

    sorted_idx = np.argsort(ppr)[::-1]
    # print('sorted_idx', sorted_idx)
    # print('ppr', ppr[sorted_idx])

    extracted_ents = sorted_idx[:max_ent]
    extracted_scores = ppr[sorted_idx[:max_ent]]

    # check if any ppr values are nearly zero
    zero_idx = np.where(ppr[extracted_ents] < 1e-6)[0]
    # print('zero_idx', zero_idx)
    if zero_idx.shape[0] > 0:
        extracted_ents = extracted_ents[:zero_idx[0]]
        extracted_scores = extracted_scores[:zero_idx[0]]

    submat = multigraph_W[extracted_ents, :]
    submat = submat[:, extracted_ents]
    row_idx, col_idx = submat.nonzero()
    print('subgraph edges:', len(row_idx))

    return extracted_ents, extracted_scores


def filter_pr_rels(question, entities, triplets, rel_to_entities, max_ent=30):
    # rel_map = db.get_rel_map(entities=entities, depth=2, limit=30)
    # print_text(f"\nclean_map: {clean_map}\n", color='yellow')

    # all_rels = []
    # for rels in clean_map.values():
    #     all_rels += rels
    # print('relations:', len(all_rels))

    # triplets, rel_to_entities = db.two_hop_parse_multi_query(all_rels)

    triplets = list(set(triplets))
    print('triplets:', len(triplets))

    entity_to_idx = {}
    idx_to_entity = {}
    entity_set = set()
    num_entities = 0

    rel_to_idx = {}
    idx_to_rel = {}
    rel_set = set()
    num_rel = 0

    # map entity and relation to a new number
    for triplet in triplets:
        ent1, rel, ent2 = triplet
        if ent1 not in entity_set:
            entity_set.add(ent1)
            entity_to_idx[ent1] = num_entities
            idx_to_entity[num_entities] = ent1
            num_entities += 1

        if ent2 not in entity_set:
            entity_set.add(ent2)
            entity_to_idx[ent2] = num_entities
            idx_to_entity[num_entities] = ent2
            num_entities += 1

        if rel not in rel_set:
            rel_set.add(rel)
            rel_to_idx[rel] = num_rel
            idx_to_rel[num_rel] = rel
            num_rel += 1

    # print('entity_set', len(entity_set), num_entities, len(entity_to_idx), len(idx_to_entity))
    # print('rel_set', len(rel_set), num_rel, len(rel_to_idx), len(idx_to_rel))

    # print('entity_set', entity_set, '\n')
    # print('rel_set', rel_set, '\n')

    entities = [ent for ent in entities if ent in entity_set]
    entities_id = [entity_to_idx[ent] for ent in entities]

    print('entities:', entities)

    all_row_ones, all_col_ones = [], []
    all_edge_value = []

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    question_embedding = embed_model.get_text_embedding(question)

    # construct a graph
    for triplet in triplets:
        ent1, rel, ent2 = triplet
        rel_embeddings = embed_model.get_text_embedding(rel)
        score = np.dot(question_embedding,
                       rel_embeddings) / (np.linalg.norm(question_embedding) *
                                          np.linalg.norm(rel_embeddings))
        score = np.power(score, EXPONENT)

        all_row_ones.append(entity_to_idx[ent1])
        all_col_ones.append(entity_to_idx[ent2])
        all_edge_value.append(score)

        all_row_ones.append(entity_to_idx[ent2])
        all_col_ones.append(entity_to_idx[ent1])
        all_edge_value.append(score)

    adj_mat = csr_matrix((np.array(all_edge_value),
                          (np.array(all_row_ones), np.array(all_col_ones))),
                         shape=(num_entities, num_entities))
    adj_mat = normalize(adj_mat, norm="l1", axis=1)
    extracted_ents, extracted_scores = get_subgraph(entities_id,
                                                    adj_mat,
                                                    max_ent=max_ent)
    extracted_ents = set(idx_to_entity[ent] for ent in extracted_ents)

    print('extracted_ents', extracted_ents)

    # print('extracted_ents', extracted_ents)
    filter_rels = []
    for rel, ents in rel_to_entities.items():
        if all([ent in extracted_ents for ent in ents]):
            filter_rels.append(rel)

    filter_triplets = []
    for triplet in triplets:
        # print(triplet)
        if triplet[0] in entities and triplet[2] in entities:
            filter_triplets.append(triplet)

    for rel in filter_rels:
        print(rel)
    print('filter_rels', len(filter_rels))
    print('extracted_ents', extracted_ents)

    for rel in filter_triplets:
        print(rel)
    print('filter_triplets', len(filter_triplets))

    print('triplets_num', len(filter_triplets), len(triplets))
    # print(filter_triplets)
    # print('filter_triplets', len(filter_triplets))
    return filter_rels, filter_triplets
