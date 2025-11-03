import os
import json

def get_rgb_or_multihop_question_list(question_data_dir):
    question_list = []
    with open(question_data_dir, 'r') as f:
        for line in f:
            question_list.append(json.loads(line))
    return question_list


def save_response(all_response, json_file):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(all_response, file, ensure_ascii=False, indent=4)
        print(f'save {len(all_response)} response to {json_file}.')


if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    exist_file_path = os.path.join(home_dir,
                                'NeutronRAG/neutronrag/dataprocess/utils/external_corpus/demo_data.json')
    store_file_path = os.path.join(home_dir,
                                'NeutronRAG/neutronrag/dataprocess/utils/external_corpus/demo_data_t.json')
    
    get_reuslt = get_rgb_or_multihop_question_list(exist_file_path)
    save_response(get_reuslt, store_file_path)