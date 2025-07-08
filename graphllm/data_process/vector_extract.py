import os
import argparse
from graphllm.database.milvus import MilvusDB
from graphllm.model_deploy.model_deployment import OllamaEnv
import json

from llama_index.core import (
    VectorStoreIndex, )

from llama_index.core.schema import Document

from typing import Optional, List


def concat_strings_in_list(input_list):
    # 检查输入列表是否是一个list
    if not isinstance(input_list, list):
        raise ValueError("The input must be a list.")

    # 如果列表中的item是字符串，则返回原列表
    if all(isinstance(item, str) for item in input_list):
        return input_list

    # 如果列表中的item是列表，解析双重列表
    elif all(isinstance(item, list) for item in input_list):
        result = [element for sublist in input_list for element in sublist]
        return result

    else:
        raise ValueError(
            "The items in the list must either all be strings or all be lists.")


class MultihopReader():
    """MultihopReader reader.

    Reads JSON documents with options to help suss out relationships between nodes.

    """

    def __init__(
        self,
        is_jsonl: Optional[bool] = False,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.is_jsonl = is_jsonl

    def load_data(self, input_file: str) -> List[Document]:
        """Load data from the input file."""

        documents = []
        with open(input_file, 'r') as file:
            load_data = json.load(file)
        for data in load_data:
            metadata = {"title": data['title'], "published_at": data['published_at'],
                        "source": data['source'], "url": data["url"]}
            # documents.append(Document(text=data['body'], metadata=metadata))
            documents.append(Document(text=data['body']))
        return documents


def read_rgb_data(rgb_data_path):
    # read rgb data
    with open(rgb_data_path, 'r') as f:
        data_set = json.load(f)

    text_list = []

    # for instance in data_set:
    #     text_list.append(instance['context']['positive'])
    #     text_list.append(instance['context']['negative'])
    for instance in data_set:
        for item in instance["context"].values():
            text_list.append(item)

    # print(text_list)
    text_list = list(set(text_list))

    alpha_count = sum([len(x) for x in text_list])
    word_count = sum([len(x.split(' ')) for x in text_list])
    print(
        f'text_list: {len(text_list)}, alpha: {alpha_count}, word_count: {word_count}'
    )

    # build do1cument for llamaindex
    rgb_docs = [Document(text=t) for t in text_list]
    print(f'all_rgb_docs: {len(rgb_docs)}')
    return rgb_docs


def read_rgb_only_positive_data(rgb_data_path):
    # read rgb data
    with open(rgb_data_path, 'r') as f:
        data_set = json.load(f)

    text_list = []

    # for instance in data_set:
    #     text_list.append(instance['context']['positive'])
    #     text_list.append(instance['context']['negative'])
    for instance in data_set:
         text_list.append(instance['context']['positive'])

    # print(text_list)
    text_list = list(set(text_list))

    alpha_count = sum([len(x) for x in text_list])
    word_count = sum([len(x.split(' ')) for x in text_list])
    print(
        f'text_list: {len(text_list)}, alpha: {alpha_count}, word_count: {word_count}'
    )

    # build do1cument for llamaindex
    rgb_docs = [Document(text=t) for t in text_list]
    print(f'all_rgb_docs: {len(rgb_docs)}')
    return rgb_docs

def read_integrationrgb_data(rgb_data_path):
    with open(rgb_data_path, 'r') as f:
        data_set = json.load(f)
    text_list = []

    for instance in data_set:
        # for item in instance["context"].values():
        #     text_list.append(item)
        text_list.append(instance['context']['positive'])

    text_list = list(set(text_list))

    alpha_count = sum([len(x) for x in text_list])
    word_count = sum([len(x.split(' ')) for x in text_list])
    print(
        f'text_list: {len(text_list)}, alpha: {alpha_count}, word_count: {word_count}'
    )

    # build do1cument for llamaindex
    rgb_docs = [Document(text=t) for t in text_list]
    print(f'all_rgb_docs: {len(rgb_docs)}')
    return rgb_docs


def read_integrationrgb_dense_positive_data(rgb_data_path):
    with open(rgb_data_path, 'r') as f:
        data_set = json.load(f)
    text_list = []

    for instance in data_set:
        # for item in instance["context"].values():
        #     text_list.append(item)
        text_list.append(instance['context']['positive1'])
        text_list.append(instance['context']['positive2'])
    text_list = list(set(text_list))

    alpha_count = sum([len(x) for x in text_list])
    word_count = sum([len(x.split(' ')) for x in text_list])
    print(
        f'text_list: {len(text_list)}, alpha: {alpha_count}, word_count: {word_count}'
    )

    # build do1cument for llamaindex
    rgb_docs = [Document(text=t) for t in text_list]
    print(f'all_rgb_docs: {len(rgb_docs)}')
    return rgb_docs


def process_milvus_data(docs, db_name):
    print(db_name)
    milvus_db = MilvusDB(db_name, 1024, overwrite=True, store=True)

    storage_context = milvus_db.get_storage_context()

    VectorStoreIndex.from_documents(
        documents=docs,
        show_progress=True,
        storage_context=storage_context,
    )
    milvus_db.show_all_collections()


def get_crag_task1_docs(data_file):
    instances = []
    with open(data_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'search_results' in data:
                search_results = data['search_results']
                for result in search_results:
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        instances.append(snippet)

    alpha_count = sum([len(x) for x in instances])
    word_count = sum([len(x.split(' ')) for x in instances])
    print(
        f'text_list: {len(instances)}, alpha: {alpha_count}, word_count: {word_count}'
    )

    # build document for llamaindex
    crag_task1_docs = [Document(text=t) for t in instances]
    return crag_task1_docs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--chunk_size', type=int,
                        required=True, help='chunk size')

    parser.add_argument('--data',
                        type=str,
                        required=True,
                        choices=['report', 'multihop', 'arxiv', 'rgb', 'dragonball', 'integrationrgb', 'crudrag', 'hotpotqa', 'raldata', 'rgb_dense_only_positive', 'integrationrgb_dense_positive'])

    parser.add_argument('--llm',
                        choices=[
                            'chatgpt', 'llama2:7b', 'llama2:13b', 'llama2:70b',
                            'vicuna:7b', 'vicuna:13b', 'vicuna:33b'
                        ],
                        type=str,
                        help='llm, e.g. openai,llama2:7b,',
                        required=True)
    # parser.add_argument('--db',
    #                     type=str,
    #                     required=True,
    #                     help='database name, e.g. test.')

    args = parser.parse_args()
    print(args)

    llm_env = OllamaEnv(llm_mode_name=args.llm, chunk_size=args.chunk_size,
                        embed_batch_size=10)

    home_dir = os.path.expanduser("~")

    if args.data == 'multihop':
        db_name = 'multihop'
        multihop_data_path = os.path.join(
            home_dir, "NeutronRAG/external_corpus/all_processed_corpus/multi_hop/summarization/multi_object/multihop.json")
        docs = read_integrationrgb_data(multihop_data_path)
    elif args.data == 'rgb':
        db_name = 'rgb'
        rgb_data_path = os.path.join(
            home_dir, "NeutronRAG/external_corpus/all_processed_corpus/single_hop/reasoning/single_object/rgb_qa_triplets.json")
        docs = read_rgb_data(rgb_data_path)
    elif args.data == 'integrationrgb':
        db_name = 'integrationrgb'
        integrationrgb_data_path = os.path.join(
            home_dir, "NeutronRAG/external_corpus/all_processed_corpus/multi_hop/reasoning/multi_object/integrationrgb_qa_triplets.json")
        docs = read_integrationrgb_data(integrationrgb_data_path)
    elif args.data == 'integrationrgb_dense_positive':
        db_name = 'integrationrgb_dense_positive'
        integrationrgb_data_path = os.path.join(
            home_dir, "NeutronRAG/external_corpus/all_processed_corpus/multi_hop/reasoning/multi_object/integrationrgb_qa_triplets.json")
        docs = read_integrationrgb_dense_positive_data(integrationrgb_data_path)
    elif args.data == 'dragonball':
        # db_name = 'dragonball'
        # integrationrgb_data_path = os.path.join(
        #     home_dir, "NeutronRAG/external_corpus/all_processed_corpus/single_hop/summarization/single_object/dragonball_qa_triplets.json")
        db_name = 'postdragonball'
        integrationrgb_data_path = os.path.join(
            home_dir, "NeutronRAG/external_corpus/all_processed_corpus/single_hop/summarization/single_object/dragonball_qa_triplets_positive.json")
        docs = read_integrationrgb_data(integrationrgb_data_path)
    elif args.data == 'crudrag':  # chunk_overlap=350,
        db_name = 'crudrag'
        crugrag_data_path = os.path.join(
            home_dir, "/home/chency/NeutronRAG/external_corpus/all_processed_corpus/multi_hop/summarization/single_object/crud_rag.json")
        docs = read_integrationrgb_data(crugrag_data_path)
    elif args.data == 'hotpotqa':
        db_name = 'newhotpotqa'
        hotpotqa_data_path = os.path.join(
            home_dir, "/home/chency/NeutronRAG/external_corpus/all_processed_corpus/multi_hop/reasoning/single_object/hotpotqa_vectorrag.json")
        docs = read_integrationrgb_data(hotpotqa_data_path)
    elif args.data == 'rgb_dense_only_positive':
        db_name = 'rgb_dense_only_positive'
        rgb_data_path = os.path.join(
            home_dir, "NeutronRAG/external_corpus/all_processed_corpus/single_hop/reasoning/single_object/rgb_qa_triplets.json")
        docs = read_rgb_only_positive_data(rgb_data_path)
    
    else:
        raise ValueError(
            "Error: data parameter is not valid. It must be either'report' or'multihop'.")

    print(f'document length:{len(docs)}')

    process_milvus_data(docs, db_name)

    # process_neo4j_data(rgb_docs)
    print("finished processing")
