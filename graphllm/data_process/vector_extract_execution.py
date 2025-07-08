
import subprocess
import os


def main():
    # 创建日志目录
    log_dir = "./log/milvus_process_data_log/"
    os.makedirs(log_dir, exist_ok=True)

    # 定义要执行的命令
    # command1 = "nohup python  vector_extract.py --chunk_size 4096 --llm llama2:70b --data multihop > ./log/milvus_process_data_log/process_multihop_vector.log 2>&1 &"
    # command2 = "nohup python  vector_extract.py --chunk_size 1024 --llm llama2:70b --data rgb > ./log/milvus_process_data_log/process_rgb_vector_12_03.log 2>&1 &"
    # command3 = "nohup python  vector_extract.py --chunk_size 4096 --llm llama2:70b --data dragonball > ./log/milvus_process_data_log/process_dragonball_vector_03_30.log 2>&1 &"
    # command4 = "nohup python  vector_extract.py --chunk_size 1024 --llm llama2:70b --data integrationrgb > ./log/milvus_process_data_log/process_integrationrgb_vector_12_03.log 2>&1 &"
    # command5 = "nohup python  vector_extract.py --chunk_size 512  --llm llama2:70b --data crudrag > ./log/milvus_process_data_log/process_crudrag_vector.log 2>&1 &"
    # command6 = "nohup python  vector_extract.py --chunk_size 4096  --llm llama2:70b --data hotpotqa > ./log/milvus_process_data_log/process_hotpotqa_vector_for_new_vrag.log 2>&1 &"
    # command7 = "nohup python  vector_extract.py --chunk_size 1024 --llm llama2:70b --data rgb_dense_only_positive > ./log/milvus_process_data_log/process_rgb_dense_only_positive_vector_06_15.log 2>&1 &"
    command8 = "nohup python  vector_extract.py --chunk_size 1024 --llm llama2:70b --data integrationrgb_dense_positive > ./log/milvus_process_data_log/process_integrationrgb_dense_positive_06_16.log 2>&1 &"
    # command = "python ../external_corpus/process_rgb_data.py --chunk_size 512 --llm llama2:70b"
    # 使用 subprocess 模块执行命令
    subprocess.call(command8, shell=True)
    # subprocess.call(command1, shell=True)


if __name__ == "__main__":
    main()
