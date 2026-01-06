import json

class DummySocket:
        """模拟前端 websocket"""
        async def send(self, msg):
            print("[TEST OUTPUT]", msg)
            self.returnmsg = msg

        def __init__(self,msg11):
            self.message = msg11
            # 模拟前端发送两条消息：创建和删除
            # self.message = json.dumps({"action": "create_dataset", 
            #                 "name": "fzb_debug",
            #                 "type": "text"})
            # self.message = json.dumps({"action": "upload_file", 
            #                 "file_name": "example.txt",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "parsing_file", 
            #                 "file_name": "example.txt",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "get_file_triplets", 
            #                 "file_name": "example.txt",
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "get_overall_triplets", 
            #                 "ds_name": "fzb_debug"})
            # self.message = json.dumps({"action": "get_parsing_status", 
            #                 "ds_name": "fzb_debug"})         
            # self.message = json.dumps({"action": "get_dataset_schema", 
            #                 "ds_name": "fzb_debug"})    
            # self.message = json.dumps({"action": "delete_file", 
            #                 "file_name": "example.txt",
            #                 "ds_name": "fzb_debug"})      
            # self.message = json.dumps({"action": "delete_dataset", 
            #                 "ds_name": "fzb_debug"})      
            #self.message = json.dumps({"action": "get_datasets"})
