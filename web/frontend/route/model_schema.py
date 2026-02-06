import json
import os
import logging
from datetime import datetime


DATA_FILE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models.json")

logger = logging.getLogger(__name__)

def _init_data_file():
    """初始化数据文件"""
    if not os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception as e:
            logger.error(f"初始化模型数据文件失败: {e}")

def load_models():
    """加载所有模型"""
    _init_data_file()
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取模型数据失败: {e}")
        return []

def save_models(models):
    """保存模型列表到文件"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(models, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"保存模型数据失败: {e}")
        return False

def create_model(name, base_url, api_key):
    """创建新模型"""
    models = load_models()
    
    # 生成新 ID (如果列表为空则为1，否则为最大ID+1)
    new_id = 1
    if models:
        new_id = max(m.get('id', 0) for m in models) + 1
        
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_model = {
        "id": new_id,
        "name": name,
        "base_url": base_url,
        "api_key": api_key, # 注意：实际生产环境中 API Key 应当加密存储
        "created_at": current_time
    }
    
    models.append(new_model)
    
    if save_models(models):
        return new_model
    return None

def delete_model(model_id):
    """删除模型"""
    models = load_models()
    
    # 过滤掉要删除的 ID
    new_models = [m for m in models if m.get('id') != model_id]
    
    if len(new_models) < len(models):
        save_models(new_models)
        return True
    return False

def get_model_by_id(model_id):
    """根据 ID 获取模型详情"""
    models = load_models()
    for m in models:
        if m.get('id') == model_id:
            return m
    return None