# api.py
import json
import asyncio
import websockets
from tqdm import tqdm
import os
import yaml
import datetime
import logging
logger = logging.getLogger(__name__)
from aag.data_pipeline.data_transformer.text_2_graph.text_2_graph import Text2Graph

class DocumentAPIServer:
    def __init__(self, dataset_folder: str = "../../aag/datasets",):
        """
        dataset_folder: 存放知识库 YAML 文件的目录
        """
        self.datasets = {}
        self.dataset_folder = dataset_folder
        self.dataset_schema_path = os.path.join(self.dataset_folder, "dataset_schemas/datasets.yaml")
        self.load_existing_datasets()

    def load_existing_datasets(self):
        """
        加载指定数据集目录下的 graph_schema.yaml 和 text_schema.yaml 文件。
        """
        with open(self.dataset_schema_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        if not yaml_data or "datasets" not in yaml_data:
            raise ValueError(f"YAML 中不包含 datasets 字段: {self.graph_schema_file}")

        self.datasets = {}
        for dataset in yaml_data.get("datasets", []):
            key = dataset.get("name")
            if key:
                self.datasets[key] = dataset

    async def create_dataset(self, websocket, message):
        """
        创建一个新的数据集目录。
        message: {
            "db_name": "example_db"
        }
        """
        name = message.get("name")
        type = message.get("type")
        if not name:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": "缺少 db_name 参数"
            }, ensure_ascii=False))
            return
        if name in self.datasets:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"Dataset '{name}' already exists."
            }, ensure_ascii=False))
            return

        try:
            # 创建数据集目录
            dataset_path = os.path.join(self.dataset_folder, f"dataset_schemas/{name}")
            os.makedirs(dataset_path, exist_ok=True)        
            # 创建数据目录
            dataset_path = os.path.join(self.dataset_folder, f"data/{name}/{type}")
            os.makedirs(dataset_path, exist_ok=True)

            new_dataset = {
                "name": name,
                "type": type,
                "description": f"{name} documents.",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "active",
                "data_path": name   
            }

            output_file = []
            for key in self.datasets:
                output_file.append(self.datasets[key])

            output_file.append(new_dataset)
            final_schema = {"datasets": output_file}

            with open(self.dataset_schema_path, "w", encoding="utf-8") as f:
                yaml.dump(final_schema, f, sort_keys=False, allow_unicode=True)

            original_schema_file = os.path.join(self.dataset_folder, f"dataset_schemas/{name}/{type}_schemas.yaml")
            with open(original_schema_file, "w", encoding="utf-8") as f:
                f.write("datasets:\n")

            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "db_name": name,
                        "message": f"Dataset '{name}' created successfully."
                    }
                }
            }, ensure_ascii=False))
        except Exception as e:
            print("Error creating dataset:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"create error {str(e)}"
            }, ensure_ascii=False))
    
    def load_each_dataset(self, db_name: str):
        self.each_dataset_schema_file = os.path.join(self.dataset_folder, f"dataset_schemas/{db_name}/{self.datasets[db_name]['type']}_schemas.yaml")
        with open(self.each_dataset_schema_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        if not yaml_data or "datasets" not in yaml_data:
            raise ValueError(f"YAML 中不包含 datasets 字段: {self.graph_schema_file}")
        
        datasets_list = yaml_data.get("datasets")
        if not isinstance(datasets_list, list):
            datasets_list = []

        self.each_dataset = {}
        for dataset in datasets_list:
            key = dataset.get("name")
            if key:
                self.each_dataset[key] = dataset

    async def upload_file(self, websocket, message):
        """
        模拟文件上传接口，实际应用中应处理文件流。
        message: {
            "file_name": "example.pdf",
            "ds_name": "dataset1"
        }
        """
        file_name = message.get("file_name")
        ds_name = message.get("ds_name")
        if not file_name or not ds_name:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": "缺少 file_name 或 ds_name 参数"
            }, ensure_ascii=False))
            return
        
        dataset_schema = self.datasets.get(ds_name)
        if dataset_schema is None:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"Dataset '{ds_name}' does not exist."
            }, ensure_ascii=False))
            return
        
        self.load_each_dataset(ds_name)

        if file_name in self.each_dataset:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"File '{file_name}' already exists in dataset '{ds_name}'."
            }, ensure_ascii=False))
            return

        if dataset_schema["type"] == "text":
            await self.upload_text_file(websocket, file_name, ds_name)
        elif dataset_schema["type"] == "graph":
            await self.upload_graph_file(message, websocket, file_name, ds_name)
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"Unsupported dataset type for '{ds_name}'."
            }, ensure_ascii=False))
            return

    async def upload_text_file(self, websocket, file_name, ds_name):
        file_path = os.path.join(self.dataset_folder, f"data/{ds_name}/text/{file_name}")
        md_folder = os.path.join(self.dataset_folder, f"data/{ds_name}/process_text/")
        os.makedirs(md_folder, exist_ok=True)
        md_file_path = os.path.join(self.dataset_folder, f"data/{ds_name}/process_text/{file_name}.md")

        ext = os.path.splitext(file_name)[1].lower()
        error_msg = None

        try:
            if ext == ".doc":
                error_msg = f"Unsupported file type: {file_name}. Please convert .doc files to .docx first."
            else:
                # 1️⃣ 转成 Markdown
                from markitdown import MarkItDown
                md = MarkItDown()
                result = md.convert(file_path)
                markdown_text = result.text_content
                # 2️⃣ 保存 Markdown 文件
                with open(md_file_path, "w", encoding="utf-8") as f:
                    f.write(markdown_text)
                
                new_schema = {
                    "name": file_name,
                    "description": f"{file_name} doc",
                    "type": "markdown",
                    "graph_status": "pending",
                    "parsing_rate": 0,
                    "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "schema": {
                        "path": md_file_path,
                        "original_path": file_path,
                        "format": "md"
                    },
                }
                
                output_file = []
                for key in self.each_dataset:
                    output_file.append(self.each_dataset[key])

                output_file.append(new_schema)
                final_schema = {"datasets": output_file}

                with open(self.each_dataset_schema_file, "w", encoding="utf-8") as f:
                    yaml.dump(final_schema, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            # 捕获不可预期的异常
            error_msg = str(e)

        # 返回状态和错误信息
        if error_msg:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"上传失败 {str(error_msg)}"
            }, ensure_ascii=False))

    async def upload_graph_file(self, message, websocket, file_name, ds_name):
        """
        用户上传的是一个 KG 文件，而不是自然语言文本。
        解析用户上传的 edge.csv / vertex.csv, 并创建图结构。
        """

        graph_name = file_name

        vertex_file_name = message.get("vertex_file_name", None)
        vertex_id_field = message.get("vertex_id_field", None)

        edge_source_field = message.get("source_field")
        edge_target_field = message.get("target_field")
        import ast
        edge_relation_field_str = message.get("relation_field", None)
        if edge_relation_field_str is None:
            edge_relation_field = None
        else:
            edge_relation_field = ast.literal_eval(edge_relation_field_str)
        
        vertex_attribute_field_srt = message.get("vertex_attribute_field", None)
        vertex_name_field_str = message.get("vertex_name_field", vertex_id_field)

        is_directed = message.get("is_directed", True)

        try:

            # ----------------------------------------------------
            # 1. 先读取边文件
            # ----------------------------------------------------
            edge_file_path = os.path.join(self.dataset_folder, f"data/{ds_name}/graph/{file_name}")
            edges = []
            all_node_names = set()

            import csv
            with open(edge_file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # 检查所有 relation 列是否存在
                for field in [edge_source_field, edge_target_field]:
                    if field not in reader.fieldnames:
                        raise ValueError(f"Edge file missing required column: {field}")

                for row in reader:
                    src = row[edge_source_field]
                    dst = row[edge_target_field]

                    if src == "" or dst == "":
                        continue
                    
                    # 读取多个 relation 列
                    if edge_relation_field is None:
                        rel = ""
                    else:
                        for field in edge_relation_field:
                            if field not in reader.fieldnames:
                                raise ValueError(f"Edge file missing required column: {field}")
                        rel_values = [row[field] for field in edge_relation_field]

                        # 你可以选择把多个字段拼起来（常见做法）
                        rel = ",".join(rel_values)

                    edges.append((src, rel, dst))
                    all_node_names.add(src)
                    all_node_names.add(dst)

            # ----------------------------------------------------
            # 2. 读取实体文件
            # ----------------------------------------------------
            id2name = {}

            if vertex_file_name:
                # ========== 用户提供顶点文件 ==========
                vertex_file_path = os.path.join(self.dataset_folder, f"data/{ds_name}/graph/{vertex_file_name}")
                if not os.path.exists(vertex_file_path):
                    raise FileNotFoundError(f"Vertex file does not exist: {vertex_file_path}")

                if not vertex_id_field or not vertex_name_field_str:
                    raise ValueError("vertex_id_field and vertex_name_field must be provided when vertex_file_path exists.")

                # 1. 解析用户传入的 attribute 列（字符串形式）
                vertex_attribute_fields = []
                if vertex_attribute_field_srt:
                    vertex_attribute_fields = ast.literal_eval(vertex_attribute_field_srt)  # -> list

                # 2. 规范化 name 列为 list（支持单列名或字符串形式的列表）
                if isinstance(vertex_name_field_str, str) and vertex_name_field_str.strip().startswith("["):
                    vertex_name_fields = ast.literal_eval(vertex_name_field_str)
                elif isinstance(vertex_name_field_str, str):
                    vertex_name_fields = [vertex_name_field_str]
                else:
                    vertex_name_fields = list(vertex_name_field_str)  # 已经是 list 的情况

                # 3. 读取 CSV 并在必要时自动选择属性列
                vertex2attributes = {}  # vid -> {attr_name: attr_value} 或 vid -> 拼接字符串
                id2name = {}

                with open(vertex_file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    # 基础列存在性检查
                    for field in [vertex_id_field] + vertex_name_fields:
                        if field not in reader.fieldnames:
                            raise ValueError(f"Vertex file missing required column: {field}")

                    # 如果没有显式给出属性列，就把 CSV 中除 id 和 name 列外的所有列当作属性列
                    if not vertex_attribute_fields:
                        vertex_attribute_fields = [c for c in reader.fieldnames if c not in ([vertex_id_field] + vertex_name_fields)]

                    # 可选：再次检查属性列是否都存在
                    for field in vertex_attribute_fields:
                        if field not in reader.fieldnames:
                            raise ValueError(f"Vertex file missing attribute column: {field}")

                    for row in reader:
                        vid = row[vertex_id_field]
                        # 如果 name 是多列，拼接成一个名字字符串（用逗号或你想要的分隔符）
                        name_parts = [row[nf] for nf in vertex_name_fields]
                        name = ",".join([p for p in name_parts if p])  # 过滤空字段
                        id2name[vid] = name

                        # 读取属性列并拼接（也可以保留为 dict）
                        attr_values = [row[f] for f in vertex_attribute_fields]
                        # 过滤空值再用逗号拼接，若要保留空值可去掉过滤
                        attr_joined = ",".join([v for v in attr_values if v])
                        vertex2attributes[vid] = attr_joined

                # 检查：边文件中的 name 是否在顶点文件里
                for id in all_node_names:
                    if id not in id2name:
                        raise ValueError(
                            f"Node '{name}' in edge file not found in vertex file. "
                            f"Edge file must use names consistent with vertex file."
                        )

            
            # ============================================================
            # 3. 存储到 process_graph 目录
            # ============================================================
            process_dir = os.path.join(self.dataset_folder, f"data/{ds_name}/process_graph")
            os.makedirs(process_dir, exist_ok=True)

            # ------------------------------------------------------------
            # 保存 edge 文件
            # ------------------------------------------------------------
            edge_out_path = os.path.join(process_dir, f"{file_name}_edges.csv")
            with open(edge_out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["src", "dst", "relation"])
                
                for src, rel, dst in edges:
                    writer.writerow([src, dst, rel])

            # ------------------------------------------------------------
            # 保存 vertex 文件（只有用户提供了实体文件才写）
            # ------------------------------------------------------------
            if vertex_file_name:
                vertex_out_path = os.path.join(process_dir, f"{file_name}_vertices.csv")
                with open(vertex_out_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "name", "attributes"])

                    for vid in id2name:
                        name = id2name[vid]
                        attrs = vertex2attributes.get(vid, "")
                        writer.writerow([vid, name, attrs])

            if os.path.exists(self.each_dataset_schema_file):
                with open(self.each_dataset_schema_file, "r", encoding="utf-8") as f:
                    existing_schema = yaml.safe_load(f) or {}
            else:
                existing_schema = {}

            # 确保 'datasets' 字段存在
            if "datasets" not in existing_schema or existing_schema["datasets"] is None:
                existing_schema["datasets"] = []

            # ------------------------------------------------------------
            # 2. 构造当前 dataset 的 schema
            # ------------------------------------------------------------
            edge_schema_entry = {
                "attribute_fields": edge_relation_field_str,  # 或 edge_relation_field 列表
                "format": "csv",
                "label_field": None,
                "path": edge_out_path,
                "original_path": edge_file_path,
                "source_field": edge_source_field,
                "target_field": edge_target_field,
                "type": "edge",
                "weight_field": None
            }

            vertex_schema_entry = None
            if vertex_file_name:
                vertex_schema_entry = {
                    "attribute_fields": vertex_attribute_field_srt,  # 用户传入或自动解析的列
                    "format": "csv",
                    "query_field": None,  # 用户名字字段，可能是多列
                    "id_field": vertex_id_field,
                    "label_field": None,
                    "path": vertex_out_path,
                    "original_path": vertex_file_path,
                    "type": "vertex"
                }

            dataset_entry = {
                "description": f"{graph_name} processed graph.",
                "name": graph_name,
                "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "graph_status": "completed",
                "schema": {
                    "edge": [edge_schema_entry],
                    "graph": {
                        "directed": is_directed,
                        "heterogeneous": False,
                        "multigraph": True,
                        "weighted": False
                    },
                    "graph_store_info": {},
                    "vertex": [vertex_schema_entry] if vertex_schema_entry else []
                },
                "type": "graph"
            }

            # ------------------------------------------------------------
            # 3. 追加到 existing schema
            # ------------------------------------------------------------
            # 如果已有相同 name 的 dataset，先删除再追加
            existing_schema["datasets"] = [
                d for d in existing_schema["datasets"] if d.get("name") != ds_name
            ]
            existing_schema["datasets"].append(dataset_entry)

            # ------------------------------------------------------------
            # 4. 写回 YAML
            # ------------------------------------------------------------
            with open(self.each_dataset_schema_file, "w", encoding="utf-8") as f:
                yaml.dump(existing_schema, f, sort_keys=False, allow_unicode=True)
            
            self.each_dataset[ds_name] = dataset_entry

            # 构建前端要求的数据
            response_data = {
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "三元组个数": len(edges),
                        "创建时间": self.each_dataset[ds_name]["create_time"],
                    }
                }
            }

            await websocket.send(json.dumps(response_data, ensure_ascii=False))

        except Exception as e:
            print("Error creating knowledge base:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"添加失败 {str(e)}"
            }, ensure_ascii=False))

    async def delete_file(self, websocket, message):
        """
        删除知识库图及对应的文件
        message: {
            "file_name": "example"
            "ds_name": "debug_fzb"
        }
        """
        file_name = message.get("file_name")
        ds_name = message.get("ds_name")
        if not file_name:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": "缺少 graph_name 参数"
            }, ensure_ascii=False))
            return

        self.load_each_dataset(ds_name)

        if not file_name in self.each_dataset:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"{file_name} 不存在"
            }, ensure_ascii=False))
            return

        try:
            if self.each_dataset[file_name]["type"] == "graph" or self.each_dataset[file_name].get("graph_status", "") == "completed":
                self.load_each_dataset_graph_schema(ds_name)
                schema = self.graph_schema[file_name].get("schema", {})
                # 删除顶点文件
                for v in schema.get("vertex", []):
                    path = v.get("path")
                    if path and os.path.exists(path):
                        os.remove(path)
                    if self.each_dataset[file_name]["type"] == "graph":
                        original_vertex_path = v.get("original_path")
                        if original_vertex_path and os.path.exists(original_vertex_path):
                            os.remove(original_vertex_path)
                # 删除边文件
                for e in schema.get("edge", []):
                    path = e.get("path")
                    if path and os.path.exists(path):
                        os.remove(path)
                    if self.each_dataset[file_name]["type"] == "graph":
                        original_edge_path = e.get("original_path")
                        if original_edge_path and os.path.exists(original_edge_path):
                            os.remove(original_edge_path)
                # 删除内存中的记录
                del self.graph_schema[file_name]

                # 更新 YAML 文件：先读全部，再去掉这个 graph
                if os.path.exists(self.graph_schema_file):
                    with open(self.graph_schema_file, "r", encoding="utf-8") as f:
                        all_data = yaml.safe_load(f) or {}

                    datasets = all_data.get("datasets", [])
                    all_data["datasets"] = [d for d in datasets if d.get("name") != file_name]

                    with open(self.graph_schema_file, "w", encoding="utf-8") as f:
                        yaml.dump(all_data, f, sort_keys=False, allow_unicode=True)
            if self.datasets[ds_name]["type"] == "text":
                schema = self.each_dataset[file_name].get("schema", {})
                path = schema.get("path")
                if path and os.path.exists(path):
                    os.remove(path)
                original_path = schema.get("original_path")
                if original_path and os.path.exists(original_path):
                    os.remove(original_path)
                # 删除内存中的记录
                del self.each_dataset[file_name]
                # 更新 YAML 文件：先读全部，再去掉这个文本文件
                if os.path.exists(self.each_dataset_schema_file):
                    with open(self.each_dataset_schema_file, "r", encoding="utf-8") as f:
                        all_data = yaml.safe_load(f) or {}

                    datasets = all_data.get("datasets", [])
                    all_data["datasets"] = [d for d in datasets if d.get("name") != file_name]

                    with open(self.each_dataset_schema_file, "w", encoding="utf-8") as f:
                        yaml.dump(all_data, f, sort_keys=False, allow_unicode=True)
                        
            # 成功返回
            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "id": file_name,
                        "名称": file_name,
                        "删除时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            }, ensure_ascii=False))

        except Exception as e:
            print("删除知识库失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"删除失败：{str(e)}"
            }, ensure_ascii=False))
      
    async def parsing_text_file(self, websocket, message):
        file_name = message["file_name"]
        ds_name = message["ds_name"]

        self.load_each_dataset(ds_name)

        if not file_name in self.each_dataset:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"file '{file_name}' don't exist."
            }, ensure_ascii=False))
            return
        if self.each_dataset[file_name].get("graph_status", "") != "pending":
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"file '{file_name}' already parse."
            }, ensure_ascii=False))
            return
        
        try:
            self.each_dataset[file_name]["graph_status"] = "parsing"
            output_file = []
            for key in self.each_dataset:
                output_file.append(self.each_dataset[key])
            final_schema = {"datasets": output_file}
            with open(self.each_dataset_schema_file, "w", encoding="utf-8") as f:
                yaml.dump(final_schema, f, sort_keys=False, allow_unicode=True)

            await websocket.send(json.dumps({"type": "status", "message": "Starting Text2Graph..."}))
            file_path = self.each_dataset[file_name]["schema"]["path"]
            graph_name = file_name
            text_2_graph = Text2Graph(
                file_path=file_path,
                graph_name=graph_name,
                llm_name='llama3:8b',
                chunk_size=512
            )

            await websocket.send(json.dumps({"type": "status", "message": "Extracting triplets and entities..."}))

            triplets, entity2id, entity2type = text_2_graph.extract_graph_and_entity_by_LLM(self.each_dataset, file_name, self.each_dataset_schema_file)

            await websocket.send(json.dumps({"type": "status", "message": "Saving graph..."}))

            graph_schema_file = os.path.join(self.dataset_folder, f"dataset_schemas/{ds_name}/graph_schemas.yaml")
            if not os.path.exists(graph_schema_file):
                # 创建空的 graph_schemas.yaml 文件
                with open(graph_schema_file, "w", encoding="utf-8") as f:
                    f.write("datasets:\n")

            new_schema = text_2_graph.save_graph_with_entity(
                triplets, entity2id, entity2type, graph_schema_file
            )

            self.each_dataset[file_name]["graph_status"] = "completed"
            output_file = []
            for key in self.each_dataset:
                output_file.append(self.each_dataset[key])
            final_schema = {"datasets": output_file}
            with open(self.each_dataset_schema_file, "w", encoding="utf-8") as f:
                yaml.dump(final_schema, f, sort_keys=False, allow_unicode=True)

            response_data = {
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "id": len(self.each_dataset),
                        "名称": graph_name,
                        "三元组个数": len(triplets),
                        "创建时间": new_schema["create_time"],
                    }
                }
            }

            await websocket.send(json.dumps(response_data, ensure_ascii=False))

        except Exception as e:
            print("Error creating knowledge base:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "file_name": file_name,
                "ds_name": ds_name,
                "content": f"解析失败 {str(e)}"
            }, ensure_ascii=False))

    def load_each_dataset_graph_schema(self, ds_name: str):
        self.graph_schema_file = os.path.join(self.dataset_folder, f"dataset_schemas/{ds_name}/graph_schemas.yaml")
        if not os.path.exists(self.graph_schema_file):
            raise FileNotFoundError(f"Graph schema file does not exist: {self.graph_schema_file}")
        
        with open(self.graph_schema_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        if not yaml_data or "datasets" not in yaml_data:
            raise ValueError(f"YAML 中不包含 datasets 字段: {self.graph_schema_file}")
        
        datasets_list = yaml_data.get("datasets")
        if not isinstance(datasets_list, list):
            datasets_list = []

        self.graph_schema = {}
        for dataset in datasets_list:
            key = dataset.get("name")
            if key:
                self.graph_schema[key] = dataset
    
    def read_graph_triplets(self, graph_name: str):
        import csv
        dataset = self.graph_schema[graph_name]
        schema = dataset.get("schema", {})

        # 获取顶点 CSV 文件路径和 ID -> 名称映射
        vertex_path = schema["vertex"][0]["path"]
        id_field = schema["vertex"][0]["id_field"]
        label_field = schema["vertex"][0]["label_field"]

        id_to_name = {}
        with open(vertex_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_to_name[row[id_field]] = row[label_field]

        # 获取边 CSV 文件路径
        edge_path = schema["edge"][0]["path"]
        source_field = schema["edge"][0]["source_field"]
        target_field = schema["edge"][0]["target_field"]
        label_field_edge = schema["edge"][0].get("label_field", "type")

        triplets = []
        with open(edge_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            total = sum(1 for _ in open(edge_path, "r", encoding="utf-8")) - 1  # 不算 header
            f.seek(0)
            next(reader)  # skip header
            count = 0
            for row in reader:
                src_name = id_to_name.get(row[source_field], row[source_field])
                tgt_name = id_to_name.get(row[target_field], row[target_field])
                label = row.get(label_field_edge, "")
                triplets.append([src_name, label, tgt_name])
        return triplets

    async def get_triplets_for_each_graph(self, websocket, message):
        """
        异步返回指定知识库的三元组给前端
        message: {"file_name": "example.txt",
                  "ds_name": "debug_fzb"}
        """
        ds_name = message["ds_name"]
        graph_name = message["file_name"]
        if not graph_name:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": "缺少 graph_name 参数"
            }, ensure_ascii=False))
            return
        self.load_each_dataset(ds_name)
        if not graph_name in self.each_dataset:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"file '{graph_name}' don't exist."
            }, ensure_ascii=False))
            return
        if self.each_dataset[graph_name].get("graph_status", "") != "completed":
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"file '{graph_name}' not parse yet."
            }, ensure_ascii=False))
            return
        self.load_each_dataset_graph_schema(ds_name)


        if not graph_name in self.graph_schema:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"file '{graph_name}' don't exist graph."
            }, ensure_ascii=False))
            return
        
        try:
            triplets = self.read_graph_triplets(graph_name)

            # 完成后返回数据
            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": triplets
                }
            }, ensure_ascii=False))

        except Exception as e:
            print("获取三元组失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"获取三元组失败: {str(e)}"
            }, ensure_ascii=False))
    
    async def get_triplets_for_each_graph_for_graph_dataset(self, websocket, message):
        """
        异步返回指定知识库的三元组给前端
        message: {"file_name": "example.txt",
                  "ds_name": "debug_fzb"}
        """
        ds_name = message["ds_name"]
        try:
            self.load_each_dataset(ds_name)
            for key in self.each_dataset:
                graph_name = key
            if not graph_name in self.each_dataset:
                await websocket.send(json.dumps({
                    "type": "error",
                    "contentType": "text",
                    "content": f"file '{graph_name}' don't exist."
                }, ensure_ascii=False))
                return

            graph_schema = self.each_dataset[graph_name].get("schema", {})

            # -----------------------
            # 读取顶点文件（可选）
            # -----------------------
            import csv
            vertex_schema = graph_schema.get("vertex")
            if vertex_schema and len(vertex_schema) > 0:
                vertex_path = vertex_schema[0]["path"]
                vertex_id_field = "id"
                vertex_name_field = "name"
                vertex_attributes_field = "attributes"

                id2name = {}
                name2attribute = {}
                with open(vertex_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        vid = row[vertex_id_field]
                        name = row[vertex_name_field]
                        attrs = row[vertex_attributes_field]

                        id2name[vid] = name
                        name2attribute[name] = attrs
            else:
                id2name = None
                name2attribute = None

            # -----------------------
            # 读取边文件
            # -----------------------
            edge_schema = graph_schema.get("edge")[0]  # 假设每个 graph 至少有一个边文件
            edge_path = edge_schema["path"]
            edge_src_field = "src"
            edge_dst_field = "dst"
            edge_rel_field = "relation"

            edges = []
            with open(edge_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = row[edge_src_field]
                    dst = row[edge_dst_field]
                    rel = row[edge_rel_field]

                    # 如果顶点文件存在，替换 id -> name
                    if id2name:
                        src = id2name.get(src, src)
                        dst = id2name.get(dst, dst)
                    edges.append((src, rel, dst))

            # -----------------------
            # 返回结果
            # -----------------------
            data = {
                "vertices": name2attribute,
                "edges": edges,
                "is_directed": self.each_dataset[graph_name]["schema"]["graph"]["directed"]
            }

            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": data
                }
            }, ensure_ascii=False))
        except Exception as e:
            print("获取三元组失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"获取三元组失败: {str(e)}"
            }, ensure_ascii=False))
    
    async def get_overall_triplets(self, websocket, message):
        """
        异步返回指定知识库的所有三元组给前端
        message: {"ds_name": "debug_fzb"}
        """
        ds_name = message["ds_name"]
        if not ds_name:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": "缺少 ds_name 参数"
            }, ensure_ascii=False))
            return
        self.load_each_dataset(ds_name)
        if self.each_dataset == {}:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"ds '{ds_name}' 没有文件."
            }, ensure_ascii=False))
            return
        self.load_each_dataset_graph_schema(ds_name)

        try:
            overall_triplets = []
            for graph_name in self.graph_schema:
                triplets = self.read_graph_triplets(graph_name)
                overall_triplets.extend(triplets)

            # 完成后返回数据
            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": overall_triplets
                }
            }, ensure_ascii=False))

        except Exception as e:
            print("获取三元组失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"获取三元组失败: {str(e)}"
            }, ensure_ascii=False))
    
    async def delete_dataset(self, websocket, message):
        """
        删除知识库图及对应的文件
        message: {
            "ds_name": "example"
        }
        """
        ds_name = message.get("ds_name")
        if not ds_name:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": "缺少 dataset name 参数"
            }, ensure_ascii=False))
            return

        if ds_name not in self.datasets:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"图 {ds_name} 不存在"
            }, ensure_ascii=False))
            return

        try:
            schema_floder = os.path.join(self.dataset_folder, f"dataset_schemas/{ds_name}")
            data_floder = os.path.join(self.dataset_folder, f"data/{ds_name}")
            if os.path.exists(schema_floder):
                import shutil
                shutil.rmtree(schema_floder)
            
            if os.path.exists(data_floder):
                import shutil
                shutil.rmtree(data_floder)
            
            # 更新 YAML 文件：先读全部，再去掉这个 dataset
            if os.path.exists(self.dataset_schema_path):
                with open(self.dataset_schema_path, "r", encoding="utf-8") as f:
                    all_data = yaml.safe_load(f) or {}

                datasets = all_data.get("datasets", [])
                all_data["datasets"] = [d for d in datasets if d.get("name") != ds_name]

                with open(self.dataset_schema_path, "w", encoding="utf-8") as f:
                    yaml.dump(all_data, f, sort_keys=False, allow_unicode=True)

            # 成功返回
            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "id": ds_name,
                        "名称": ds_name,
                        "删除时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            }, ensure_ascii=False))

        except Exception as e:
            print("删除知识库失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"删除失败：{str(e)}"
            }, ensure_ascii=False))
    
    async def get_parsing_status(self, websocket, message):
        ds_name = message["ds_name"]
        try:
            self.load_each_dataset(ds_name)

            if self.each_dataset == {}:
                await websocket.send(json.dumps({
                    "type": "error",
                    "contentType": "text",
                    "content": f"ds '{ds_name}' 没有文件."
                }, ensure_ascii=False))
                return
            detailed_parsing_status = {}
            ack = True
            for file_name in self.each_dataset:
                status = []
                status.append(self.each_dataset[file_name]["graph_status"])
                status.append(self.each_dataset[file_name].get("parsing_rate", 0))
                if status[0] != "completed":
                    ack = False
                detailed_parsing_status[file_name] = status

            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "total_parsing_status": ack,
                        "parsing_status": detailed_parsing_status
                    }
                }
            }, ensure_ascii=False))

        except Exception as e:
            print("获取解析状态失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"获取解析状态失败: {str(e)}"
            }, ensure_ascii=False))

    async def get_dataset_schema(self, websocket, message):
        ds_name = message["ds_name"]
        try:
            self.load_each_dataset(ds_name)

            result = []
            id = 0
            for key in self.each_dataset:
                result.append({})
                result[id]["id"] = id+1
                result[id]["name"] = key
                result[id]["type"] = self.each_dataset[key]["type"]
                result[id]["uploadDate"] = self.each_dataset[key]["create_time"]
                result[id]["graph_status"] = self.each_dataset[key]["graph_status"]
                if self.datasets[ds_name]["type"] == "text":
                    result[id]["parsing_rate"] = self.each_dataset[key]["parsing_rate"]
                    result[id]["size"] = os.path.getsize(self.each_dataset[key]["schema"]["original_path"]) 
                elif self.datasets[ds_name]["type"] == "graph":
                    result[id]["edge_size"] = os.path.getsize(self.each_dataset[key]["schema"]["edge"][0]["original_path"]) 
                    if len(self.each_dataset[key]["schema"]["vertex"]) > 0:
                        result[id]["vertex_size"] = os.path.getsize(self.each_dataset[key]["schema"]["vertex"][0]["original_path"]) 
                        result[id]["vertex_file"] = self.each_dataset[key]["schema"]["vertex"][0]["original_path"]
                id += 1

            await websocket.send(json.dumps({
            "type": "data",
            "contentType": "json",
            "content": {
                "success": True,
                "data": result
            }
        }, ensure_ascii=False))
        except Exception as e:
            print("获取数据集 schema 失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"获取数据集 schema 失败: {str(e)}"
            }, ensure_ascii=False))

    async def get_datasets(self, websocket):
        try:
            self.datasets = {}
            self.load_existing_datasets()

            result = []
            id = 0
            for key in self.datasets:
                result.append({})
                self.load_each_dataset(key)
                result[id]["id"] = id+1
                result[id]["名称"] = key
                result[id]["文件类型"] = self.datasets[key]["type"]
                result[id]["创建时间"] = self.datasets[key]["created_at"]
                result[id]["文档个数"] = len(self.each_dataset)
                id += 1

            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": result
                }
            }, ensure_ascii=False))
        except Exception as e:
            print("获取数据集列表失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"获取数据集列表失败: {str(e)}"
            }, ensure_ascii=False))
    
    async def schema_refine(self, websocket, message):
        file_name = message["file_name"]
        ds_name = message["ds_name"]
        try:
            self.load_each_dataset(ds_name)

            if not file_name in self.each_dataset:
                await websocket.send(json.dumps({
                    "type": "error",
                    "contentType": "text",
                    "content": f"file '{file_name}' don't exist."
                }, ensure_ascii=False))
                return

            if self.each_dataset[file_name].get("graph_status", "") != "parsing":
                await websocket.send(json.dumps({
                    "type": "error",
                    "contentType": "text",
                    "content": f"file '{file_name}' not parsing status."
                }, ensure_ascii=False))
                return
            
            self.each_dataset[file_name]["graph_status"] = "pending"
            self.each_dataset[file_name]["parsing_rate"] = 0
            output_file = []
            for key in self.each_dataset:
                output_file.append(self.each_dataset[key])
            final_schema = {"datasets": output_file}
            with open(self.each_dataset_schema_file, "w", encoding="utf-8") as f:
                yaml.dump(final_schema, f, sort_keys=False, allow_unicode=True)

        except Exception as e:
            print("重置解析状态失败:", e)
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"重置解析状态失败: {str(e)}"
            }, ensure_ascii=False))
            
    
    async def handler(self, websocket):
        # async for msg in websocket:
        try:
            data = json.loads(websocket.message)
            action = data.get("action")

            if action == "create_dataset":
                await self.create_dataset(websocket, data)
            elif action == "upload_file":
                await self.upload_file(websocket, data)
            elif action == "parsing_file":
                await self.parsing_text_file(websocket, data)
            elif action == "get_file_triplets":
                await self.get_triplets_for_each_graph(websocket, data)
            elif action == "get_overall_triplets":
                await self.get_overall_triplets(websocket, data)
            elif action == "get_file_triplets_from_graph_dataset":
                await self.get_triplets_for_each_graph_for_graph_dataset(websocket, data)
            elif action == "get_parsing_status":
                await self.get_parsing_status(websocket, data)
            elif action == "get_dataset_schema":
                await self.get_dataset_schema(websocket, data)
            elif action == "delete_file":
                await self.delete_file(websocket, data)
            elif action == "delete_dataset":
                await self.delete_dataset(websocket, data)
            elif action == "get_datasets":
                await self.get_datasets(websocket)
            elif action == "schema_refine":
                await self.schema_refine(websocket, data)

            else:
                await websocket.send(json.dumps({"type": "error", "message": "Unknown action."}))
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON."}))

server_Test = DocumentAPIServer()

async def main():
    server = DocumentAPIServer()

    class DummySocket:
        """模拟前端 websocket"""
        async def send(self, msg):
            print("[TEST OUTPUT]", msg)

        def __init__(self):
            # 模拟前端发送两条消息：创建和删除
            # self.message = json.dumps({"action": "create_dataset", 
            #                 "name": "fzb_debug_graph",
            #                 "type": "graph"})
            # self.message = json.dumps({"action": "upload_file", 
            #                 "file_name": "example.doc",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "upload_file", 
            #                 "file_name": "paper4.pdf",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "upload_file", 
            #                 "file_name": "example.txt",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "upload_file", 
            #                 "file_name": "example.docx_transactions.csv",
            #                 "source_field": "orig_acct",
            #                 "target_field": "bene_acct",
            #                 # "relation_field": "['tx_type', 'base_amt']",
            #                 "vertex_file_name": "example.docx_entities.csv",
            #                 "vertex_id_field": "acct_id",
            #                 "vertex_name_field": "['first_name', 'last_name']",
            #                 "ds_name": "fzb_debug_graph"})
            # self.message = json.dumps({"action": "parsing_file", 
            #                 "file_name": "example.txt",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "parsing_file", 
            #                 "file_name": "paper4.pdf",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "parsing_file", 
            #                 "file_name": "Test11.txt",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "get_file_triplets", 
            #                 "file_name": "example.txt",
            #                 "ds_name": "fzb_debug"})
            self.message = json.dumps({"action": "get_file_triplets", 
                            "file_name": "paper4.pdf",
                            "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "get_file_triplets_from_graph_dataset", 
            #                 # "file_name": "example.docx_transactions.csv",
            #                 "ds_name": "fzb_debug_graph"})
            # self.message = json.dumps({"action": "get_overall_triplets", 
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "get_parsing_status", 
            #                 "ds_name": "fzb_debug"})         
            # self.message = json.dumps({"action": "get_dataset_schema", 
            #                 "ds_name": "fzb_debug"})    
            # self.message = json.dumps({"action": "delete_file", 
            #                 "file_name": "example.docx",
            #                 "ds_name": "fzb_debug"})     
            # self.message = json.dumps({"action": "delete_file", 
            #                 "file_name": "example.txt",
            #                 "ds_name": "fzb_debug"})    
            # self.message = json.dumps({"action": "delete_dataset", 
            #                 "ds_name": "fzb_debug"})      
            # self.message = json.dumps({"action": "get_datasets"})
            # self.message = json.dumps({"action": "schema_refine", 
            #                 "file_name": "paper.pdf",
            #                 "ds_name": "fzb_debug"})

            

    dummy_ws = DummySocket()
    await server.handler(dummy_ws)


if __name__ == "__main__":
    asyncio.run(main())
