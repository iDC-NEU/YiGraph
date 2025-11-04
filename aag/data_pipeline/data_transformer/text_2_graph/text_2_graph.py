import re
from typing import Dict, List
from tqdm import tqdm

from aag.utils.file_operation import file_exist
from aag.utils.parse_json import extract_json_from_response
from aag.reasoner.model_deployment import OllamaEnv

prompt_template_str = """
#### Process
**Identify Named Entities**: Extract entities based on the given entity types, ensuring they appear in the order they are mentioned in the text.
**Establish Triplets**: Form triples with reference to the provided predicates, again in the order they appear in the text.

Your final response should follow this format:

**Output:**
```json
{{
    "entities": # type: Dict
    {{
        "Entity Type": ["entity_name"]
    }},
    "triplets": # type: List
    [
        ["subject", "predicate", "object"]
    ]
}}
```

### Example:
**Entity Types:**
ORGANIZATION
COMPANY
CITY
STATE
COUNTRY
OTHER
PERSON
YEAR
MONTH
DAY
OTHER
QUANTITY
EVENT

**Predicates:**
FOUNDED_BY
HEADQUARTERED_IN
OPERATES_IN
OWNED_BY
ACQUIRED_BY
HAS_EMPLOYEE_COUNT
GENERATED_REVENUE
LISTED_ON
INCORPORATED
HAS_DIVISION
ALIAS
ANNOUNCED
HAS_QUANTITY
AS_OF


**Input:**
Walmart Inc. (formerly Wal-Mart Stores, Inc.) is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas.[10] The company was founded by brothers Sam and James "Bud" Walton in nearby Rogers, Arkansas in 1962 and incorporated under Delaware General Corporation Law on October 31, 1969. It also owns and operates Sam's Club retail warehouses.[11][12]

As of October 31, 2022, Walmart has 10,586 stores and clubs in 24 countries, operating under 46 different names.[2][3][4] The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, and as Flipkart Wholesale in India.

**Output:**
```json
{{
"entities": {{
    "COMPANY": ["Walmart Inc.", "Sam's Club", "Flipkart Wholesale"],
    "PERSON": ["Sam Walton", "James 'Bud' Walton"],
    "COUNTRY": ["United States", "Canada", "Mexico", "Central America", "India"],
    "CITY": ["Bentonville", "Rogers"],
    "STATE": ["Arkansas"],
    "DATE": ["1962", "October 31, 1969", "October 31, 2022"],
    "ORGANIZATION": ["Delaware General Corporation Law"]
}},
"triplets": [
    ["Walmart Inc.", "FOUNDED_BY", "Sam Walton"],
    ["Walmart Inc.", "FOUNDED_BY", "James 'Bud' Walton"],
    ["Walmart Inc.", "HEADQUARTERED_IN", "Bentonville, Arkansas"],
    ["Walmart Inc.", "FOUNDED_IN", "1962"],
    ["Walmart Inc.", "INCORPORATED", "October 31, 1969"],
    ["Sam Walton", "FOUNDED", "Walmart Inc."],
    ["James \"Bud\" Walton", "CO-FOUNDED", "Walmart Inc."],
    ["Walmart Inc.", "OWNS", "Sam's Club"],
    ["Flipkart Wholesale", "OWNED_BY", "Walmart Inc."],
    ["Walmart Inc.", "OPERATES_IN", "United States"],
    ["Walmart Inc.", "OPERATES_IN", "Canada"],
    ["Walmart Inc.", "OPERATES_IN", "Mexico"],
    ["Walmart Inc.", "OPERATES_IN", "Central America"],
    ["Walmart Inc.", "OPERATES_IN", "India"]
]
}}
```

### Task:
Your task is to perform Named Entity Recognition (NER) and knowledge graph triplet extraction on the text that follows below.

**Input:**
{context}

**Output:**
"""


prompt_entity_type = """
#### Task
You are asked to determine the type of a given entity. Only provide the entity type as the output. Do not include any extra information or explanations.

**Entity Types:**
ORGANIZATION
COMPANY
CITY
STATE
COUNTRY
PERSON
YEAR
MONTH
DAY
EVENT
QUANTITY
OTHER

**Reference Text:**
{text_context}

**Input:**
{entity}

**Output:**
"""


class Text2Graph:
    def __init__(self, text_file: str, llm_name: str, chunk_size: int = 512):
        """
        初始化文本到图转换器
        
        Args:
            text_file: 输入的文本文件路径
        """
        self.text_file = text_file

        if not file_exist(self.text_file):
            raise FileNotFoundError(f"文本文件未找到: {self.text_file}")
        
        with open(self.text_file, 'r', encoding='utf-8') as f:
            sentences = re.split(r'(?<=[。！？\n])', f.read().strip())  # 按句子切分
        self.text_chunks = []
        current_chunk = ''

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                self.text_chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            self.text_chunks.append(current_chunk)
        
        print(f"文本已加载，共 {len(self.text_chunks)} 个块, 每块约 {chunk_size} 字符。")

        self.llm = OllamaEnv(llm_mode_name = llm_name)

    def extract_graph(self, MAX_RETRIES = 5):
        """
        从文本数据中提取知识图谱
        
        Returns:
            三元组列表
        """
        print("开始从文本中提取知识图谱...")
        
        triplets = []
        entities = []

        for idx, chunk in enumerate(tqdm(self.text_chunks)):
            
            for attempt in range(1, MAX_RETRIES + 1):
                response = self.llm.complete(prompt=prompt_template_str.format(context=chunk))
                # 判空
                if not response:
                    print(f"⚠️ 块 {idx} 第 {attempt} 次尝试未获得响应")
                    continue
                # 尝试解析 JSON
                response_parsed = extract_json_from_response(response)
                if isinstance(response_parsed, str):
                    print(f"⚠️ 块 {idx} 第 {attempt} 次响应解析失败")
                    continue

                # 成功获取合法响应，跳出重试循环
                response = response_parsed
                break
            else:
                # 5 次都不成功，跳过该块
                print(f"❌ 块 {idx} 超过 {MAX_RETRIES} 次尝试仍失败，跳过")
                continue
            
            # 注释掉是应为提取的实体和三元组不一样，有可能对应不上，我们只用提取到的三元组就OK了
            # entity_label = {}
            # if isinstance(response["entities"], List):
            #     for entities in response["entities"]:
            #         for entity_type, names in entities.items():
            #             for name in names:
            #                 if not isinstance(name, str) or not isinstance(
            #                     entity_type, str
            #                 ):
            #                     continue
            #                 entity_label[name.capitalize()] = entity_type.capitalize()
            # else:
            #     assert isinstance(response["entities"], Dict)
            #     for entity_type, names in response["entities"].items():
            #         for name in names:
            #             if not isinstance(name, str) or not isinstance(entity_type, str):
            #                 continue
            #             entity_label[name.capitalize()] = entity_type.capitalize()

            new_triplets = []

            for triplet in response.get("triplets", []):
                # 1️⃣ 必须是列表/元组且长度为3
                if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                    print(f"⚠️ 无效三元组，跳过: {triplet}")
                    continue
                
                # 2️⃣ 遍历每个元素，处理字符串大小写
                processed_triplet = []
                for phrase in triplet:
                    if isinstance(phrase, str) and phrase.strip():
                        processed_triplet.append(phrase.strip().capitalize())
                    else:
                        processed_triplet.append(phrase)  # 保留原样（可能是数字或空）
                
                new_triplets.append(processed_triplet)

            triplets.extend(new_triplets)

        print(f"提取完成，共获得 {len(triplets)} 个三元组。")
        return triplets

    def save_triplet(self, triplets: List[List[str]]):
        import os, csv
        dir_path = os.path.dirname(self.text_file)
        base_name = os.path.splitext(os.path.basename(self.text_file))[0]
        entities_csv_path = os.path.join(dir_path, f"{base_name}_accounts.csv")
        triplets_csv_path = os.path.join(dir_path, f"{base_name}_transactions.csv")
        
        valid_triplets = []
        for t in triplets:
            if len(t) != 3:
                print(f"⚠️ 无效三元组，跳过: {t}")
                continue
            head, rel, tail = t
            if not head or not tail or not rel:
                print(f"⚠️ 三元组元素为空，跳过: {t}")
                continue
            valid_triplets.append([head, rel, tail])

        if not valid_triplets:
            print("⚠️ 没有合法三元组，退出。")
            return

        entities = {}
        next_id = 1
        for head, rel, tail in valid_triplets:
            for ent in [head, tail]:
                if ent not in entities:
                    entities[ent] = next_id
                    next_id += 1

        with open(entities_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["acct_id", "dsply_nm"])
            for ent, eid in entities.items():
                writer.writerow([eid, ent])

        with open(triplets_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tran_id", "orig_acct", "bene_acct"])
            for head, rel, tail in valid_triplets:
                writer.writerow([entities[head], entities[tail], rel])

        print(f"✅ 实体 CSV 保存到: {entities_csv_path}")
        print(f"✅ 边 CSV 保存到: {triplets_csv_path}")

        schema_path = os.path.join(dir_path, f"{base_name}_graph_schemas.yaml")
        graph_name = f"{base_name}_Graph"

        schema_dict = {
            "datasets": [
                {
                    "name": graph_name,
                    "description": f"{graph_name} graph generated from {self.text_file}",
                    "type": "graph",
                    "schema": {
                        "vertex": [
                            {
                                "attribute_fields": [],
                                "format": "csv",
                                "id_field": "acct_id",
                                "label_field": "dsply_nm",
                                "path": entities_csv_path,
                                "type": "account"
                            }
                        ],
                        "edge": [
                            {
                                "attribute_fields": [],
                                "format": "csv",
                                "label_field": "bene_acct",
                                "path": triplets_csv_path,
                                "source_field": "tran_id",
                                "target_field": "orig_acct",
                                "type": "transfer",
                                "weight_field": None
                            }
                        ],
                        "graph": {
                            "directed": "true",
                            "heterogeneous": "false",
                            "multigraph": "false",
                            "weighted": "false"
                        },
                        "graph_store_info": {
                            "backend": "nebula_graph",
                            "edge_count": len(valid_triplets),
                            "space_name": graph_name,
                            "status": "success",
                            "version": "null",
                            "vertex_count": len(entities)
                        }
                    }
                }
            ]
        }
        import yaml
        with open(schema_path, "w", encoding="utf-8") as f:
            yaml.dump(schema_dict, f, sort_keys=False, allow_unicode=True)

        print(f"✅ Graph schema YAML: {schema_path}")

    def extract_graph_by_openie(self):
        print("开始使用OpenIE从文本中提取知识图谱...")
        try:
            from openie import StanfordOpenIE
        except ImportError:
            import subprocess, sys
            package_name = "stanford-openie"
            print(f"⚙️ 检测到未安装依赖 '{package_name}'，正在自动安装...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"✅ 成功安装 {package_name}")
                from openie import StanfordOpenIE  # 再次导入
            except subprocess.CalledProcessError as e:
                print(f"❌ 安装 {package_name} 失败，请手动安装：pip install {package_name}")
                raise e
        
        triplets = []
        for idx, chunk in enumerate(tqdm(self.text_chunks)):
            properties = {
                'openie.affinity_probability_cap': 2 / 3,
            }
            with StanfordOpenIE(properties=properties) as client:
                for triple in client.annotate(chunk):
                    head = triple['subject'].strip().capitalize()
                    relation = triple['relation'].strip().capitalize()
                    tail = triple['object'].strip().capitalize()
                    triplet = [head, relation, tail]
                    triplets.append(triplet)
        print(f"提取完成，共获得 {len(triplets)} 个三元组。")
        print(triplets)
        assert False, "stop here"
        return triplets

    def extract_graph_and_entity_by_LLM(self, MAX_RETRIES = 5):
        """
        从文本数据中提取三元组与实体类型

        Returns:
            triplets: 合法三元组列表
            entity2id: 实体 -> ID
            entity2type: 实体 -> 类型
        """
        print(f"开始使用 LLM 从文本中提取知识图谱...")
        triplets: List[List[str]] = []
        entity2type: Dict[str, str] = {}

        for idx, chunk in enumerate(tqdm(self.text_chunks)):
            for attempt in range(1, MAX_RETRIES + 1):
                response = self.llm.complete(prompt=prompt_template_str.format(context=chunk))
                if not response:
                    print(f"⚠️ 块 {idx} 第 {attempt} 次无响应")
                    continue

                response_parsed = extract_json_from_response(response)
                if not isinstance(response_parsed, dict):
                    print(f"⚠️ 块 {idx} 第 {attempt} 次 JSON 解析失败")
                    continue
                response = response_parsed
                break
            else:
                print(f"❌ 块 {idx} 超过 {MAX_RETRIES} 次尝试失败，跳过")
                continue

            # ======================
            # 1️⃣ 解析 entities 部分
            # ======================
            entities_from_response = {}
            if isinstance(response.get("entities"), dict):
                for entity_type, names in response["entities"].items():
                    for name in names:
                        if isinstance(name, str) and name.strip():
                            entities_from_response[name.strip()] = entity_type.strip().capitalize()

            # ======================
            # 2️⃣ 解析 triplets 部分
            # ======================
            new_triplets = []
            for triplet in response.get("triplets", []):
                if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                    print(f"⚠️ 无效三元组，跳过: {triplet}")
                    continue
                # 去掉空字符串，只保留非空字符串和数字/其他类型
                clean_triplet = []
                for x in triplet:
                    if isinstance(x, str):
                        x_clean = x.strip()
                        if x_clean:  # 非空字符串才保留
                            clean_triplet.append(x_clean.capitalize())
                    else:
                        clean_triplet.append(x)  # 数字或其他类型保留原样

                # 长度必须为3才保留
                if len(clean_triplet) == 3:
                    new_triplets.append(clean_triplet)
                else:
                    print(f"⚠️ 三元组长度不足3，跳过: {triplet}")

            triplets.extend(new_triplets)

            # ======================
            # 3️⃣ 更新实体类型映射
            # ======================
            for head, rel, tail in new_triplets:
                for ent in [head, tail]:
                    if ent not in entity2type:
                        if ent in entities_from_response:
                            entity2type[ent] = entities_from_response[ent]
                        else:
                            # 调用大模型补全实体类型
                            entity_type = self._ask_entity_type(ent, chunk, MAX_RETRIES)
                            entity2type[ent] = entity_type

        # ======================
        # 4️⃣ 生成实体-ID 映射
        # ======================
        entity2id = {ent: i + 1 for i, ent in enumerate(entity2type.keys())}

        print(f"✅ 提取完成，共 {len(triplets)} 个三元组，{len(entity2id)} 个实体")
        return triplets, entity2id, entity2type

    def _ask_entity_type(self, entity: str, chunk: str, MAX_RETRIES = 5) -> str:
        """向 LLM 查询实体类型"""
        prompt = prompt_entity_type.format(entity=entity, text_context=chunk)
        for attempt in range(1, MAX_RETRIES + 1):
            response = self.llm.complete(prompt=prompt)
            if response and isinstance(response, str):
                return response.strip().capitalize()
        print(f"⚠️ 无法确定实体类型：{entity}，默认设为 Unknown")
        return "Unknown"
    
    def save_graph_with_entity(self, triplets: List[List[str]], entity2id: Dict[str, int], entity2type: Dict[str, str]):
        import os, csv, yaml

        dir_path = os.path.dirname(self.text_file)
        base_name = os.path.splitext(os.path.basename(self.text_file))[0]

        entities_csv_path = os.path.join(dir_path, f"{base_name}_accounts.csv")
        triplets_csv_path = os.path.join(dir_path, f"{base_name}_transactions.csv")
        schema_path = os.path.join(dir_path, f"{base_name}_graph_schemas.yaml")
        graph_name = f"{base_name}_Graph"

        # -------------------------
        # 1️⃣ 保存实体 CSV（增加 type 列）
        # -------------------------
        with open(entities_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["acct_id", "dsply_nm", "type"])
            for ent, eid in entity2id.items():
                ent_type = entity2type.get(ent, "OTHER")
                writer.writerow([eid, ent, ent_type])
        print(f"✅ 实体 CSV 保存到: {entities_csv_path}")

        # -------------------------
        # 2️⃣ 保存三元组 CSV
        # -------------------------
        with open(triplets_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tran_id", "orig_acct", "bene_acct"])
            for head, rel, tail in triplets:
                writer.writerow([entity2id[head], entity2id[tail], rel])
        print(f"✅ 边 CSV 保存到: {triplets_csv_path}")

        # -------------------------
        # 3️⃣ 保存 Graph schema YAML
        # -------------------------
        schema_dict = {
            "datasets": [
                {
                    "name": graph_name,
                    "description": f"{graph_name} graph generated from {self.text_file}",
                    "type": "graph",
                    "schema": {
                        "vertex": [
                            {
                                "attribute_fields": ["type"],
                                "format": "csv",
                                "id_field": "acct_id",
                                "label_field": "dsply_nm",
                                "path": entities_csv_path,
                                "type": "account"
                            }
                        ],
                        "edge": [
                            {
                                "attribute_fields": ["bene_acct"],
                                "format": "csv",
                                "label_field": "bene_acct",
                                "path": triplets_csv_path,
                                "source_field": "tran_id",
                                "target_field": "orig_acct",
                                "type": "transfer",
                                "weight_field": None
                            }
                        ],
                        "graph": {
                            "directed": "true",
                            "heterogeneous": "false",
                            "multigraph": "false",
                            "weighted": "false"
                        },
                        "graph_store_info": {
                            "backend": "nebula_graph",
                            "edge_count": len(triplets),
                            "space_name": graph_name,
                            "status": "success",
                            "version": "null",
                            "vertex_count": len(entity2id)
                        }
                    }
                }
            ]
        }

        with open(schema_path, "w", encoding="utf-8") as f:
            yaml.dump(schema_dict, f, sort_keys=False, allow_unicode=True)
        print(f"✅ Graph schema YAML: {schema_path}")


if __name__ == '__main__':
    text_2_graph = Text2Graph(
        text_file='./aag/data_pipeline/data_transformer/text_2_graph/example.txt',
        llm_name='llama3:8b',
        chunk_size=512
    )
    # text_2_graph.save_triplet(text_2_graph.extract_graph())
    triplets, entity2id, entity2type =  text_2_graph.extract_graph_and_entity_by_LLM()
    text_2_graph.save_graph_with_entity(triplets, entity2id, entity2type)



"""
执行代码：python -m aag.data_pipeline.data_transformer.text_2_graph.text_2_graph


git命令示例：
git stash push -m "backup before pull"
git pull origin main

LLM output:

Here is the output:

```
{
"entities": {
    "ORGANIZATION": ["LSU", "Purdue"],
    "DATE": ["December 4, 2022"],
    "COUNTRY": ["United States"]
},
"triplets": [
    ["LSU", "PLAYED_AGAINST", "Purdue"],
    ["Quad Wilson", "PICKED_OFF", "Jack Albers"],
    ["Quad Wilson", "RETURNED", "99 yards"],
    ["Quad Wilson", "SCORED_TOUCHDOWN", "63-7"],
    ["2022-23 season", "ENDED_WITH", "Citrus Bowl victory"]
]
}
```

Let me know if you have any questions or need further clarification!

"""
