import os
import json


def write_list_to_json_file(file_path, data_list):
    # 检查并创建文件夹（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f'save {len(data_list)} response to {file_path}.')


def file_exist(path):
    return os.path.exists(path)


def save_response(all_response, json_file):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(all_response, file, ensure_ascii=False, indent=4)
        print(f'save {len(all_response)} response to {json_file}.')


def read_json(file_path: str):
    assert file_exist(file_path), file_path
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def read_triplets_from_json(file_path: str):
    assert file_exist(file_path), file_path
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    triplets = []
    triplets.extend(
        triple for item in data for triple_list in item["context_triplets"].values() for triple in triple_list)
    return triplets


if __name__ == '__main__':
    test_file = "/home/chency/NeutronRAG/external_corpus/all_processed_corpus/single_hop/reasoning/single_object/rgb_qa_triplets.json"
    triplets = read_triplets_from_json(test_file)
    # print(len(triplets))
    left_entities = [triplet[0] for triplet in triplets]
    right_entities = [triplet[2] for triplet in triplets]
    relation = [triplet[1] for triplet in triplets]
    for en in left_entities:
        if not isinstance(en, str):
            print(en)
    # print(left_entities[:5])
    # print(len(list(set(left_entities))))
    # print(len(right_entities))
    # print(len(left_entities + right_entities))
    # entities = list(set(left_entities + right_entities))
    entities = sorted(list(set(left_entities + right_entities)))
    print(len(entities))
    relation = list(set(relation))
    print(len(relation))

    test_log_file = "external_corpus/all_processed_corpus/single_hop/reasoning/single_object/logs/build_kg-2024-11-16_18-25-17.json"
    triplets = read_json(test_log_file)
    tt = []
    for item in triplets:
        tt.extend(item["triplets"])

    # print(len(tt))
    left_entities_t = [triplet[0] for triplet in tt]
    right_entities_t = [triplet[2] for triplet in tt]
    relation = [triplet[1] for triplet in tt]
    for en in left_entities_t:
        if not isinstance(en, str):
            print(en)
    entities = sorted(list(set(left_entities_t + right_entities_t)))
    print(len(entities))
    relation = list(set(relation))
    print(len(relation))
