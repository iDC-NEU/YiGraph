"""
    三个模型，llm, 嵌入模型（for vector）, 挖掘实体模型

    两个函数，在线部署， 本地部署

    三个模型是在线的

"""

import numpy as np
import os
import openai
import json
from typing import Literal, List
from llama_index.core import Settings
from llama_index.core.utils import print_text
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from prompt_template.llm_prompt import prompt_select_graph_algorithm_str_en, prompt_select_graph_algorithm_str_zh

from model_deploy.prompt import (
    command_keyword_extract_prompt_template,
    command_synonym_expand_prompt_template,
    gemma_keyword_extract_prompt_template,
    gemma_synonym_expand_prompt_template,
)


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(),
                      re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()

def get_pasrse_output(output_str, field=Literal["algorithm"]):
    retry = 3
    while retry > 0:
        try:
            output_data = json.loads(extract_json_str(output_str))
            assert field in output_data
            if field == "algorithm":
                algorithm = output_data[field]
                return algorithm
            else:
                raise ValueError(f"Field {field} is not supported")
        except json.JSONDecodeError as e:
            retry -= 1
            if retry == 0:
                return {"error": "JSONDecodeError", "message": str(e), "output": output_str}


class ModelDeploy:

    def __init__(self,
                 chunk_size=512,
                 chunk_overlap=20):
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

    def online_model_deployment(self):
        os.environ["OPENAI_API_KEY"] = "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"
        Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.node_parser = SentenceSplitter(
            chunk_size=512, chunk_overlap=20)
        self.llm = Settings.llm

    def generate_response(self, query: str):
        response = self.llm.complete(query)
        return response

    def local_model_deployment(self):
        # self.llm=LlamaCPP(
        #     # You can pass in the URL to a GGML model to download it automatically
        #     model_url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q2_K.gguf",
        #     # optionally, you can set the path to a pre-downloaded model instead of model_url
        #     model_path=None,
        #     temperature=0.1,
        #     max_new_tokens=256,
        #     # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        #     context_window=3900,
        #     # kwargs to pass to __call__()
        #     generate_kwargs={},
        #     # kwargs to pass to __init__()
        #     # set to at least 1 to use GPU
        #     model_kwargs={"n_gpu_layers": 1},
        #     # transform inputs into Llama2 format
        #     messages_to_prompt=messages_to_prompt,
        #     completion_to_prompt=completion_to_prompt,
        #     verbose=True,
        # )

        Settings.embed_model = resolve_embed_model(
            "local:BAAI/bge-small-en-v1.5")

    # TODO: 指定把模型放在cpu 还是 gpu


EMBEDD_DIMS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    # "text-embedding-ada-002": 1536,
}


class EmbeddingEnv:

    def __init__(self,
                 embed_name="BAAI/bge-large-en-v1.5",
                 embed_batch_size=20,
                 device="cuda:0"):
        self.embed_name = embed_name
        self.embed_batch_size = embed_batch_size

        assert embed_name in EMBEDD_DIMS
        self.dim = EMBEDD_DIMS[embed_name]

        if 'BAAI' in embed_name:
            print(f"use huggingface embedding {embed_name}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=embed_name,
                embed_batch_size=embed_batch_size,
                device=device)
        else:
            print(f"use openai embedding {embed_name}")
            self.embed_model = OpenAIEmbedding(
                model=embed_name, embed_batch_size=embed_batch_size)

        Settings.embed_model = self.embed_model
        print_text(
            f"EmbeddingEnv: embed_name {embed_name}, embed_batch_size {self.embed_batch_size}, dim {self.dim}\n",
            color='red')

    def __str__(self):
        return f"{self.embed_name} {self.embed_batch_size}"

    def get_embedding(self, query_str):
        embedding = self.embed_model._get_text_embedding(query_str)
        return embedding

    def get_embeddings(self, query_str_list):
        embeddings = self.embed_model._get_text_embeddings(query_str_list)
        return embeddings

    def calculate_similarity(self, query1, query2):
        # Cosine similarity
        embedding1 = self.embed_model.get_embedding(query1)
        embedding2 = self.embed_model.get_embedding(query2)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = round(dot_product / (norm1 * norm2), 6)
        return similarity

    # def parallel_generate_embedding(self,
    #                                 entities,
    #                                 step,
    #                                 devices=["cuda:0", "cuda:2"]):
    #     nproc = len(devices)
    #     job_len = len(entities) // nproc
    #     jobs = [
    #         entities[i:i + job_len] for i in range(0, len(entities), job_len)
    #     ]

    #     # multi processing
    #     for i, job in enumerate(range(jobs)):
    #         save_path = './embedding_{i}.json'
    #         process_one_job(job, step, save_path)


class OllamaEnv:

    def __init__(self,
                 llm_mode_name="llama3.1:70b",
                 llm_embed_name="BAAI/bge-large-en-v1.5",
                 chunk_size=512,
                 chunk_overlap=20,
                 embed_batch_size=20,
                 device='cuda:2',
                 timeout=150000,
                 port=11434,
                 verbose=False):

        # base_url = "http://localhost:11434"
        base_url = f"http://localhost:{port}"
        Settings.llm = Ollama(model=llm_mode_name, request_timeout=timeout,
                              temperature=0.0, base_url=base_url)  # , device=device
        self.verbose = verbose

        if 'llama' in llm_mode_name:
            self.keyword_extract_prompt_template = command_keyword_extract_prompt_template
            self.synonym_expand_prompt_template = command_synonym_expand_prompt_template
        elif 'gemma' in llm_mode_name and 'instruct' in llm_mode_name:
            self.keyword_extract_prompt_template = gemma_keyword_extract_prompt_template
            self.synonym_expand_prompt_template = gemma_synonym_expand_prompt_template
        elif 'command' in llm_mode_name:
            self.keyword_extract_prompt_template = command_keyword_extract_prompt_template
            self.synonym_expand_prompt_template = command_synonym_expand_prompt_template
        else:
            self.keyword_extract_prompt_template = None
            self.synonym_expand_prompt_template = None

        if 'BAAI' in llm_embed_name:
            print(f"use huggingface embedding {llm_embed_name}")
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=llm_embed_name,
                embed_batch_size=embed_batch_size,
                device=device)
        else:
            print(f"use openai embedding {llm_embed_name}")
            Settings.embed_model = OpenAIEmbedding(
                model=llm_embed_name, embed_batch_size=embed_batch_size)

        if llm_embed_name not in EMBEDD_DIMS.keys():
            raise NotImplementedError('embed model not support!')

        self.llm = Settings.llm
        self.embed_model = Settings.embed_model
        self.dim = EMBEDD_DIMS[llm_embed_name]
        # print(f"using {self.dim}")
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        print(
            f"llm_mode_name: {llm_mode_name}, llm_embed_name: {llm_embed_name}, chunk_size: {Settings.chunk_size}, chunk_overlap: {chunk_overlap}")

    def complete(self, prompt, info=""):
        response = self.llm.complete(prompt)
        print_text(f'{prompt}\n', color='yellow')
        print_text(f'{response.text}\n', color='green')

        generate_time, load_time, prefill_time, decode_time, prompt_len, generate_len = self.parse_response_time(
            response)

        # if self.logger is not None:
        #     self.logger.add(f'{info}generate_time', generate_time)
        #     self.logger.add(f'{info}load_time', load_time)
        #     self.logger.add(f'{info}prefill_time', prefill_time)
        #     self.logger.add(f'{info}decode_time', decode_time)
        #     self.logger.add(f'{info}prompt_len', prompt_len)
        #     self.logger.add(f'{info}generate_len', generate_len)
        if self.verbose:
            print_text(
                f"generate_time {generate_time:.3f}s, load_time {load_time:.3f}s, prefill_time {prefill_time:.3f}s, decode_time {decode_time:.3f}s, prompt_len {prompt_len}, generate_len {generate_len}\n",
                color='red')
            
        return response.text

    def parse_response_time(self, response, verbose=False):
        # total_duration: time spent generating the response
        total_time = response.raw['total_duration'] / 1e9

        # load_duration: time spent in nanoseconds loading the model
        load_time = response.raw['load_duration'] / 1e9

        # (prefill): prompt_eval_duration: time spent in nanoseconds evaluating the prompt
        prefill_time = response.raw['prompt_eval_duration'] / 1e9

        # (generation): eval_duration: time in nanoseconds spent generating the response
        decode_time = response.raw['eval_duration'] / 1e9

        # prompt_eval_count: number of tokens in the prompt
        prompt_len = response.raw[
            'prompt_eval_count'] if 'prompt_eval_count' in response.raw else 0

        # eval_count: number of tokens in the response
        generate_len = response.raw['eval_count']

        # print(f'total_duration: {total_duration:.3f}ms, load_duration: {load_duration:.3f}ms, prompt_eval_duration: {prompt_eval_duration:.3f}ms, eval_duration: {eval_duration:.3f}ms')
        if verbose:
            print_text(
                f'total_duration: {total_time:.3f}ms, load, prompt, eval: ({load_time:.3f}, {prefill_time:.3f}, {decode_time:.3f})ms\n',
                color='red')
            print_text(
                f'prompt_eval_count: {prompt_len} tokens, eval_count: {generate_len} tokens\n',
                color='red')

            # print(total_duration, prompt_eval_duration + eval_duration + load_time)

        return (total_time, load_time, prefill_time, decode_time, prompt_len,
                generate_len)

    def generate_response(self, query: str):
        response = self.llm.complete(query)
        return response
    
    def get_graph_algorithm(self, question, context, language="en"):
        try:
            prompt_mapping = {
                "en": prompt_select_graph_algorithm_str_en,
                "zh": prompt_select_graph_algorithm_str_zh,
            }
            prompt_extract_triplest = prompt_mapping.get(language.lower())
            if not prompt_extract_triplest:
                raise NotImplementedError(
                    f"Language '{language}' is not supported.")
            full_prompt = prompt_extract_triplest.format(
                input_question=question,
                context=context
            )
            response = self.complete(full_prompt, info="graph_algorithm_selection")
            algorithm = get_pasrse_output(response.strip(), field="algorithm")
            return algorithm     
        except Exception as e:
            print(f"Error in Ollama graph algorithm selection: {e}")
            return None
        
    def get_question_entity(self, question, language="en"):  #TODO: 需要补充一个函数，从问题中确定需要查询的实体, 如果查询实体为空，则返回none（表示要查询全图）
        """从问题中确定需要查询的实体, 如果查询实体为空，则返回none（表示要查询全图）"""
        """返回类型是list，表示需要查询的实体"""
        return []

    def get_quetion_response(self, question, graph_result, language="en"):  #TODO: 需要补充一个函数
        """根据问题和图算法结果，生成响应"""
        pass

class OpenAIEnv:

    def __init__(self, 
                 base_url,
                 api_key,
                 model_name):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model_name

        openai.api_key = self.api_key
        openai.base_url = self.base_url
        self.client = openai

    def get_graph_algorithm(self, question, context, language="en"):
        try:
            prompt_mapping = {
                "en": prompt_select_graph_algorithm_str_en,
                "zh": prompt_select_graph_algorithm_str_zh,
            }
            prompt_extract_triplest = prompt_mapping.get(language.lower())
            if not prompt_extract_triplest:
                raise NotImplementedError(
                    f"Language '{language}' is not supported.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_extract_triplest.format(
                    context=context)}]
            )
            response = response.choices[0].message.content
            return get_pasrse_output(response.strip(), field="algorithm")
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None

    def get_question_entity(self, question, context, language="en"):  #TODO: 需要补充一个函数，从问题中确定需要查询的实体, 如果查询实体为空，则返回none（表示要查询全图）
        """从问题中确定需要查询的实体, 如果查询实体为空，则返回none（表示要查询全图）"""
        """返回类型是list，表示需要查询的实体"""
        return []

    def get_quetion_response(self, question, graph_result, language="en"):  #TODO: 需要补充一个函数
        """根据问题和图算法结果，生成响应"""
        pass


if __name__ == '__main__':

    llm_env = OllamaEnv()
    print(llm_env.complete("山西的省会城市？"))
