import json
import os
import base64
import logging
from datetime import datetime


DATA_FILE = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models.json"
)

logger = logging.getLogger(__name__)

# 前端模型名称到后端实际模型标识的映射
# 统一来源：config.py 中的 MODEL_MAPPING 已迁移至此，请从此处导入。
MODEL_MAPPING = {
    "GPT 4": "qwen3-max",
    "Qwen 14B": "qwen3-max",
    "Qwen Plus": "qwen3-max",
}


def _init_data_file():
    """初始化数据文件"""
    if not os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            logger.error(f"初始化模型数据文件失败: {e}")


def _obfuscate_api_key(api_key: str) -> str:
    """
    对 API Key 进行简单混淆（Base64 编码）后存储。
    ⚠️ 注意：这是混淆（obfuscation）而非加密（encryption），
    Base64 可被轻易解码。生产环境应使用密钥管理服务（如 HashiCorp Vault、AWS KMS）。
    """
    if not api_key:
        return api_key
    return base64.b64encode(api_key.encode("utf-8")).decode("utf-8")


def _deobfuscate_api_key(encoded_key: str) -> str:
    """反向解码混淆后的 API Key。"""
    if not encoded_key:
        return encoded_key
    try:
        return base64.b64decode(encoded_key.encode("utf-8")).decode("utf-8")
    except Exception:
        # 兼容已存储的未混淆明文 Key
        return encoded_key


def load_models():
    """加载所有模型（自动解码 API Key）"""
    _init_data_file()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            models = json.load(f)
        # 解码存储的 API Key（兼容明文和编码两种格式）
        for model in models:
            if "api_key" in model:
                model["api_key"] = _deobfuscate_api_key(model["api_key"])
        return models
    except Exception as e:
        logger.error(f"读取模型数据失败: {e}")
        return []


def save_models(models):
    """保存模型列表到文件"""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
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
        new_id = max(m.get("id", 0) for m in models) + 1

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_model = {
        "id": new_id,
        "name": name,
        "base_url": base_url,
        "api_key": _obfuscate_api_key(api_key),  # 混淆存储；生产环境应使用 KMS 加密
        "created_at": current_time,
    }

    models.append(new_model)

    if save_models(models):
        return new_model
    return None


def delete_model(model_id):
    """删除模型"""
    models = load_models()

    # 过滤掉要删除的 ID
    new_models = [m for m in models if m.get("id") != model_id]

    if len(new_models) < len(models):
        save_models(new_models)
        return True
    return False


def get_model_by_id(model_id):
    """根据 ID 获取模型详情"""
    models = load_models()
    for m in models:
        if m.get("id") == model_id:
            return m
    return None
