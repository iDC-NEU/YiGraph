from flask import Blueprint, request, jsonify
import logging
import requests  # 引入 requests 库用于发送 HTTP 请求
from .model_schema import (
    load_models,
    create_model,
    delete_model,
    get_model_by_id
)

# 创建蓝图
bp = Blueprint('models', __name__, url_prefix='/api')

# 配置日志
logger = logging.getLogger(__name__)

def verify_model_access(base_url, api_key, model_name):
    """
    核心验证函数：尝试调用一下模型，看是否能连通
    """
    try:
        # 1. 规范化 URL：去掉末尾的斜杠
        clean_url = base_url.rstrip('/')
        
        # 这里假设用户填写的是 Base URL (如 https://api.openai.com/v1)
        
        # 2. 构造测试请求
        # 我们发送一个极小的对话请求来验证 Key、URL 和 模型名称 是否都正确
        target_url = f"{clean_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1  # 只要生成 1 个 token，节省成本且速度快
        }

        # 3. 发送请求 (设置 10 秒超时，防止卡死)
        response = requests.post(target_url, json=payload, headers=headers, timeout=10)

        # 4. 检查状态码
        if response.status_code == 200:
            return True, "验证通过"
        elif response.status_code == 401:
            return False, "API Key 无效或未授权 (401)"
        elif response.status_code == 404:
            return False, f"找不到该模型或 URL 路径错误 (404)。请检查 Base URL 是否正确，通常应以 /v1 结尾。"
        else:
            # 尝试解析错误信息
            try:
                err_msg = response.json().get('error', {}).get('message', response.text)
            except:
                err_msg = response.text
            return False, f"模型连接失败 (HTTP {response.status_code}): {err_msg}"

    except requests.exceptions.ConnectionError:
        return False, "无法连接到服务器 (Connection Error)。请检查 Base URL 是否可访问。"
    except requests.exceptions.Timeout:
        return False, "请求超时 (Timeout)。服务器响应太慢。"
    except Exception as e:
        return False, f"验证过程发生未知错误: {str(e)}"


# 获取模型列表 (保持不变)
@bp.route("/models", methods=["GET"])
def get_models():
    try:
        models = load_models()
        return jsonify({"success": True, "data": models, "count": len(models)})
    except Exception as e:
        logger.error(f"获取模型列表失败：{str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# 创建新模型 (已添加校验逻辑)
@bp.route("/models", methods=["POST"])
def create_model_route():
    """创建新模型"""
    try:
        data = request.get_json()
        logger.info(f"收到创建模型请求: {data}")

        # 1. 基础字段校验
        name = data.get("name")
        base_url = data.get("base_url")
        api_key = data.get("api_key")

        if not name or not base_url or not api_key:
            return jsonify({
                "success": False,
                "error": "模型名称、Base URL 和 API Key 均为必填项"
            }), 400

        # ==========================================
        # 2. 新增：后端连通性校验 (关键步骤)
        # ==========================================
        logger.info(f"正在验证模型连通性: {name} @ {base_url}")
        is_valid, error_message = verify_model_access(base_url, api_key, name)
        
        if not is_valid:
            logger.warning(f"模型验证失败: {error_message}")
            return jsonify({
                "success": False,
                "error": "模型验证失败",
                "message": error_message  # 把具体的错误原因返回给前端
            }), 400

        # 3. 校验通过，写入文件
        new_model = create_model(name=name, base_url=base_url, api_key=api_key)

        if new_model:
            logger.info(f"成功创建并验证模型: {new_model['name']}")
            return jsonify({
                "success": True,
                "message": "模型验证通过并创建成功",
                "data": new_model
            })
        else:
            return jsonify({"success": False, "error": "写入数据库失败"}), 500
            
    except Exception as e:
        logger.error(f"创建模型失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "系统内部错误",
            "message": str(e)
        }), 500


# 删除模型 (保持不变)
@bp.route("/models/<int:model_id>", methods=["DELETE"])
def delete_model_route(model_id):
    try:
        success = delete_model(model_id)
        if success:
            return jsonify({"success": True, "message": "成功删除模型"})
        else:
            return jsonify({"success": False, "error": "未找到指定的模型"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500