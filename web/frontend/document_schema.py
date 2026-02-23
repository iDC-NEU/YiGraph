
import os
import json
from datetime import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))


KNOWLEDGE_BASES_DATA_FILE = os.path.join(current_dir, "knowledge_bases_data.json")


FILE_STORAGE_BASE_PATH = os.path.join(current_dir, "debug", "files")

def get_knowledge_base_name(kb_id):
    """根据知识库ID获取知识库名称"""
    knowledge_bases = load_knowledge_bases()
    for kb in knowledge_bases:
        if kb["id"] == int(kb_id):
            return kb.get("name") or kb.get("名称")
    return f"kb_{kb_id}"  # 默认名称

def load_knowledge_bases():
    """加载所有知识库数据"""
    if not os.path.exists(KNOWLEDGE_BASES_DATA_FILE):
        # 如果文件不存在
        initial_data = {
            "knowledge_bases": [],
        }
        with open(KNOWLEDGE_BASES_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
        return initial_data["knowledge_bases"]
    
    try:
        with open(KNOWLEDGE_BASES_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("knowledge_bases", [])
    except Exception as e:
        print(f"加载知识库数据失败: {e}")
        return []

def save_knowledge_bases(knowledge_bases):
    """保存所有知识库数据"""
    data = {
        "knowledge_bases": knowledge_bases,
    }
    try:
        with open(KNOWLEDGE_BASES_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存知识库数据失败: {e}")
        return False

def count_files_in_knowledge_base(kb_id):
    """
    统计指定知识库文件夹中的文件个数

    """
    # 获取知识库名称
    kb_name = get_knowledge_base_name(kb_id)
    
    # 构建知识库对应的文件夹路径
    kb_folder_path = os.path.join(FILE_STORAGE_BASE_PATH, kb_name)
    
    # 检查文件夹是否存在
    if not os.path.exists(kb_folder_path) or not os.path.isdir(kb_folder_path):
        return 0
    
    try:
 
        file_count = 0
        for item in os.listdir(kb_folder_path):
            item_path = os.path.join(kb_folder_path, item)
            if os.path.isdir(item_path):
                expected_file = os.path.join(item_path, item)
                if os.path.isfile(expected_file):
                    file_count += 1
        
        return file_count
    except Exception as e:
        print(f"统计知识库 {kb_id} 文件个数失败: {e}")
        return 0

def update_all_knowledge_bases_file_count():
    """
    更新所有知识库的文件个数并保存到JSON
    """
    knowledge_bases = load_knowledge_bases()
    
    # 更新每个知识库的文件个数
    for kb in knowledge_bases:
        kb_id = kb["id"]
        actual_file_count = count_files_in_knowledge_base(kb_id)
        kb["文件个数"] = actual_file_count
    
    # 保存到JSON
    return save_knowledge_bases(knowledge_bases)

def create_knowledge_base(name, file_type):
    """创建新的知识库"""
    knowledge_bases = load_knowledge_bases()
    
    # 生成新ID
    new_id = max([kb["id"] for kb in knowledge_bases]) + 1 if knowledge_bases else 1
    
    kb_folder_path = os.path.join(FILE_STORAGE_BASE_PATH, name)
    os.makedirs(kb_folder_path, exist_ok=True)
    
    new_kb = {
        "id": new_id,
        "name": name,
        "file_type": file_type,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_count": 0,
    }
    
    knowledge_bases.append(new_kb)
    
    if save_knowledge_bases(knowledge_bases):
        return new_kb
    else:
        return None

def delete_knowledge_base(kb_id):
    """删除知识库"""
    knowledge_bases = load_knowledge_bases()
    
    # 找到对应的知识库并获取名称
    kb_to_delete = None
    for kb in knowledge_bases:
        if kb["id"] == kb_id:
            kb_to_delete = kb
            break
    
    if kb_to_delete:
        # 从列表中删除
        knowledge_bases = [kb for kb in knowledge_bases if kb["id"] != kb_id]
        
        # 删除对应的文件夹 - 使用知识库名称而不是ID
        kb_folder_path = os.path.join(FILE_STORAGE_BASE_PATH, kb_to_delete.get("name") or kb_to_delete.get("名称"))
        try:
            import shutil
            if os.path.exists(kb_folder_path):
                shutil.rmtree(kb_folder_path)
        except Exception as e:
            print(f"删除知识库文件夹失败: {e}")
        
        if save_knowledge_bases(knowledge_bases):
            return True
    
    return False