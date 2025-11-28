# api.py
import json
import asyncio
from tqdm import tqdm
import os
import yaml
import datetime
from aag.utils.path_utils import GRAPH_SCHEMA_PATH, TEXT_SCHEMA_PATH
from aag.data_pipeline.data_transformer.text_2_graph.text_2_graph import Text2Graph

class DocumentAPIServer:
    def __init__(self, graph_schema_file: str = GRAPH_SCHEMA_PATH,
                 text_schema_file: str = TEXT_SCHEMA_PATH):
        """
        kb_folder: 存放知识库 YAML 文件的目录
        """
        self.user_data = {}
        self.graph_schema_file = graph_schema_file
        self.text_schema_file = text_schema_file
        self.load_existing_user_data()

    def load_existing_user_data(self):
        """
        传入一个 YAML 文件路径，读取其中的 datasets 并加载到 user_data 中。
        """
        if not os.path.exists(self.graph_schema_file):
            raise FileNotFoundError(f"指定的知识库 YAML 不存在: {self.graph_schema_file}")

        with open(self.graph_schema_file, "r", encoding="utf-8") as f:
            kb_data = yaml.safe_load(f)

        if not kb_data or "datasets" not in kb_data:
            raise ValueError(f"YAML 中不包含 datasets 字段: {self.graph_schema_file}")

        for ds in kb_data["datasets"]:
            graph_name = ds.get("name")
            if not graph_name:
                print(f"{self.graph_schema_file} 中存在一个 dataset 没有 name 字段，已跳过")
                continue
            self.user_data[graph_name] = ds
            print(f"已加载图: {graph_name}")

    async def create_user_data(self, websocket, message):
        file_path = message["file_path"]
        graph_name = message["graph_name"]

        # 如果知识库已存在
        if graph_name in self.user_data:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"Knowledge base '{graph_name}' already exists."
            }, ensure_ascii=False))
            return
        
        try:
            await websocket.send(json.dumps({"type": "status", "message": "Starting Text2Graph..."}))

            text_2_graph = Text2Graph(
                file_path=file_path,
                graph_name=graph_name,
                llm_name='llama3:8b',
                chunk_size=512
            )

            await websocket.send(json.dumps({"type": "status", "message": "Extracting triplets and entities..."}))

            triplets, entity2id, entity2type = text_2_graph.extract_graph_and_entity_by_LLM()

            await websocket.send(json.dumps({"type": "status", "message": "Saving graph..."}))

            new_schema = text_2_graph.save_graph_with_entity(
                triplets, entity2id, entity2type, self.graph_schema_file, self.text_schema_file
            )

            self.user_data[graph_name] = new_schema

            # 构建前端要求的数据
            response_data = {
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "id": len(self.user_data),  # 你可按自己逻辑修改id生成方式
                        "名称": graph_name,
                        "文档个数": len(triplets),
                        "创建时间": new_schema["create time"],
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

    async def delete_user_data(self, websocket, message):
        """
        删除知识库图及对应的文件
        message: {
            "graph_name": "example"
        }
        """
        graph_name = message.get("graph_name")
        if not graph_name:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": "缺少 graph_name 参数"
            }, ensure_ascii=False))
            return

        if graph_name not in self.user_data:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"图 {graph_name} 不存在"
            }, ensure_ascii=False))
            return

        try:
            schema = self.user_data[graph_name].get("schema", {})

            # 删除顶点文件
            for v in schema.get("vertex", []):
                path = v.get("path")
                if path and os.path.exists(path):
                    os.remove(path)

            # 删除边文件
            for e in schema.get("edge", []):
                path = e.get("path")
                if path and os.path.exists(path):
                    os.remove(path)

            # 删除内存中的记录
            del self.user_data[graph_name]

            # 更新 YAML 文件：先读全部，再去掉这个 graph
            if os.path.exists(self.graph_schema_file):
                with open(self.graph_schema_file, "r", encoding="utf-8") as f:
                    all_data = yaml.safe_load(f) or {}

                datasets = all_data.get("datasets", [])
                all_data["datasets"] = [d for d in datasets if d.get("name") != graph_name]

                with open(self.graph_schema_file, "w", encoding="utf-8") as f:
                    yaml.dump(all_data, f, sort_keys=False, allow_unicode=True)
            
            # 更新 text_schema.yaml 文件，去掉这个文本知识库
            if os.path.exists(self.text_schema_file):
                with open(self.text_schema_file, "r", encoding="utf-8") as f:
                    all_data = yaml.safe_load(f) or {}

                datasets = all_data.get("datasets", [])
                # 找到要删除的文本文件路径
                for d in datasets:
                    if d.get("name") == graph_name:
                        path = d.get("schema", {}).get("path")
                        if path and os.path.exists(path):
                            os.remove(path)
                all_data["datasets"] = [d for d in datasets if d.get("name") != graph_name]

                with open(self.text_schema_file, "w", encoding="utf-8") as f:
                    yaml.dump(all_data, f, sort_keys=False, allow_unicode=True)

            # 成功返回
            await websocket.send(json.dumps({
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "id": graph_name,
                        "名称": graph_name,
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
    
    async def get_triplets(self, websocket, message):
        """
        异步返回指定知识库的三元组给前端
        message: {"graph_name": "example"}
        """
        import csv
        graph_name = message.get("graph_name")
        if not graph_name:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": "缺少 graph_name 参数"
            }, ensure_ascii=False))
            return

        if graph_name not in self.user_data:
            await websocket.send(json.dumps({
                "type": "error",
                "contentType": "text",
                "content": f"知识库 {graph_name} 不存在"
            }, ensure_ascii=False))
            return

        try:
            dataset = self.user_data[graph_name]
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
    
    async def create_kb_from_kg(self, websocket, message):
        """
        用户上传的是一个 KG 文件，而不是自然语言文本。
        解析用户上传的 edge.csv / vertex.csv, 并创建图结构。
        """

        graph_name = message.get("graph_name")

        vertex_file_path = message.get("vertex_file", None)  # 可选
        vertex_id_field = message.get("vertex_id_field", None)
        vertex_name_field = message.get("vertex_name_field", None)

        edge_file_path = message.get("edge_file")
        edge_source_field = message.get("source_field")
        edge_target_field = message.get("target_field")
        edge_relation_field = message.get("relation_field")
        weight_col = message.get("weight_field", None)

        try:
            if graph_name in self.user_data:
                await websocket.send(json.dumps({
                    "type": "error",
                    "contentType": "text",
                    "content": f"Knowledge base '{graph_name}' already exists."
                }, ensure_ascii=False))
                return

            # ----------- 参数检查 ------------
            if not os.path.exists(edge_file_path):
                raise FileNotFoundError(f"Edge file does not exist: {edge_file_path}")

            # ----------------------------------------------------
            # 1. 先读取边文件
            # ----------------------------------------------------
            edges = []
            all_node_names = set()

            import csv

            with open(edge_file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # 检查字段是否存在
                for field in [edge_source_field, edge_target_field, edge_relation_field]:
                    if field not in reader.fieldnames:
                        raise ValueError(f"Edge file missing required column: {field}")

                for row in reader:
                    src = row[edge_source_field]
                    dst = row[edge_target_field]
                    rel = row[edge_relation_field]

                    if src == "" or dst == "":
                        continue

                    edges.append((src, rel, dst))  # 暂时使用名字形式
                    all_node_names.add(src)
                    all_node_names.add(dst)

            # ----------------------------------------------------
            # 2. 构建 name2id
            # ----------------------------------------------------
            name2id = {}

            if vertex_file_path:
                # ========== 用户提供顶点文件 ==========
                if not os.path.exists(vertex_file_path):
                    raise FileNotFoundError(f"Vertex file does not exist: {vertex_file_path}")

                if not vertex_id_field or not vertex_name_field:
                    raise ValueError("vertex_id_field and vertex_name_field must be provided when vertex_file_path exists.")

                with open(vertex_file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    # 检查字段
                    for field in [vertex_id_field, vertex_name_field]:
                        if field not in reader.fieldnames:
                            raise ValueError(f"Vertex file missing column: {field}")

                    for row in reader:
                        vid = row[vertex_id_field]
                        name = row[vertex_name_field]

                        name2id[name] = vid

                # 检查：边文件中的 name 是否在顶点文件里
                for name in all_node_names:
                    if name not in name2id:
                        raise ValueError(
                            f"Node '{name}' in edge file not found in vertex file. "
                            f"Edge file must use names consistent with vertex file."
                        )

            else:
                # ========== 用户没有提供顶点文件 → 自动构建 ==========
                sorted_nodes = sorted(list(all_node_names))
                name2id = {n: i for i, n in enumerate(sorted_nodes)}

            # ----------------------------------------------------
            # 3. 生成最终三元组 triplets (id-id-rel)
            # ----------------------------------------------------
            triplets = []
            for src_name, rel, dst_name in edges:
                triplets.append((src_name, rel, dst_name))

            text_2_graph = Text2Graph(edge_file_path, 
                                graph_name, 
                                llm_name='llama3:8b',
                                chunk_size=512,
                                input_file_format="graph")
            print(triplets)
            print(name2id)
            entity2type = {}
            print(1)
            for entity in name2id:
                entity_type = text_2_graph._ask_entity_type(entity, "Unknow", MAX_RETRIES = 5)
                print(entity_type)
                entity2type["entity"] = entity_type
                print(2)
            new_schema = text_2_graph.save_graph_with_entity(
                    triplets, name2id, entity2type, self.graph_schema_file, self.text_schema_file
                )

            self.user_data[graph_name] = new_schema

            # 构建前端要求的数据
            response_data = {
                "type": "data",
                "contentType": "json",
                "content": {
                    "success": True,
                    "data": {
                        "id": len(self.user_data),  # 你可按自己逻辑修改id生成方式
                        "名称": graph_name,
                        "三元组个数": len(triplets),
                        "创建时间": new_schema["create time"],
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
    
    async def handler(self, websocket):
        async for msg in websocket:
            try:
                data = json.loads(msg)
                action = data.get("action")
                if action == "create_kb":
                    await self.create_user_data(websocket, data)
                elif action == "delete_kb":
                    await self.delete_user_data(websocket, data)
                elif action == "get_triplets":
                    await self.get_triplets(websocket, data)
                elif action == "upload_graph":
                    await self.create_kb_from_kg(websocket, data)
                else:
                    await websocket.send(json.dumps({"type": "error", "message": "Unknown action."}))
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON."}))

server_Test = DocumentAPIServer()

class DummySocket:
        """模拟前端 websocket"""
        def __init__(self,msg1):
            self.msgs = msg1
        async def send(self, msg):
            print("[TEST OUTPUT]", msg)

        async def __aiter__(self):
            # 模拟前端发送两条消息：创建和删除
            # self.msgs = [
            #     # json.dumps({"action": "create_kb", 
            #     #             "file_path": "./aag/data_pipeline/data_transformer/text_2_graph/debug_file/example/example.pdf", 
            #     #             "graph_name": "example",
            #     #             "db_name": "debug_file"}),
                            
            #     # json.dumps({"action": "delete_kb", 
            #     #             "graph_name": "example2",
            #     #             "db_name": "debug_file"}),

            #     # json.dumps({"action": "get_triplets", 
            #     #             "graph_name": "example",
            #     #             "db_name": "debug_file"}),

            #     # json.dumps({"action": "upload_graph", 
            #     #             "graph_name": "example2",

            #     #             "vertex_file": "./aag/data_pipeline/data_transformer/text_2_graph/debug_file/example2/entity.csv",
            #     #             "vertex_id_field": "acct_id",
            #     #             "vertex_name_field": "dsply_nm",
                            
            #     #             "edge_file": "./aag/data_pipeline/data_transformer/text_2_graph/debug_file/example2/example2.csv",
            #     #             "source_field": "src_name",
            #     #             "target_field": "dst_name",
            #     #             "relation_field": "relation_field",
            #     #             "weight_field": "weight_field",
            #     #             }),
            # ]
            for m in self.msgs:
                yield m


async def main():
    server = DocumentAPIServer()

    class DummySocket:
        """模拟前端 websocket"""
        async def send(self, msg):
            print("[TEST OUTPUT]", msg)

        async def __aiter__(self):
            # 模拟前端发送两条消息：创建和删除
            self.msgs = [
                # json.dumps({"action": "create_kb", 
                #             "file_path": "./aag/data_pipeline/data_transformer/text_2_graph/debug_file/example/example.pdf", 
                #             "graph_name": "example",
                #             "db_name": "debug_file"}),
                            
                # json.dumps({"action": "delete_kb", 
                #             "graph_name": "example2",
                #             "db_name": "debug_file"}),

                # json.dumps({"action": "get_triplets", 
                #             "graph_name": "example",
                #             "db_name": "debug_file"}),

                # json.dumps({"action": "upload_graph", 
                #             "graph_name": "example2",

                #             "vertex_file": "./aag/data_pipeline/data_transformer/text_2_graph/debug_file/example2/entity.csv",
                #             "vertex_id_field": "acct_id",
                #             "vertex_name_field": "dsply_nm",
                            
                #             "edge_file": "./aag/data_pipeline/data_transformer/text_2_graph/debug_file/example2/example2.csv",
                #             "source_field": "src_name",
                #             "target_field": "dst_name",
                #             "relation_field": "relation_field",
                #             "weight_field": "weight_field",
                #             }),
            ]
            for m in self.msgs:
                yield m

    dummy_ws = DummySocket()
    await server.handler(dummy_ws)


if __name__ == "__main__":
    asyncio.run(main())
