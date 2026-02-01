import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__) 
bp = Blueprint("chat", __name__)

@bp.route("/api/models", methods=["GET"])
def get_models():
    try:
        logger.info("收到模型查询请求")
        models = [
            {"id": 1, "name": "GPT 4", "description": "OpenAI的GPT-4模型"},
            {"id": 2, "name": "Qwen 14B", "description": "通义千问14B参数模型"},
            {"id": 3, "name": "Qwen Plus", "description": "通义千问增强版模型"},
            {"id": 4, "name": "Llama 3", "description": "Meta的Llama 3开源模型"}
        ]
        return jsonify({
            "success": True,
            "data": models,
            "count": len(models)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


