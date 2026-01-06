from flask import Blueprint, request, jsonify
import logging
from document_schema import (
    load_knowledge_bases, 
    create_knowledge_base, 
    delete_knowledge_base,
    count_files_in_knowledge_base,
    update_all_knowledge_bases_file_count
)

# 创建蓝图
bp = Blueprint('documents', __name__, url_prefix='/api')

# 配置日志
logger = logging.getLogger(__name__)

# 获取知识库列表
@bp.route("/knowledge_bases", methods=["GET"])
def get_knowledge_bases():
    """获取知识库列表"""
    try:
        logger.info("收到知识库查询请求")
        
        # 更新所有知识库的文件数量
        update_all_knowledge_bases_file_count()
        
        # 直接从文件加载数据
        knowledge_bases = load_knowledge_bases()
        
        return jsonify({
            "success": True,
            "data": knowledge_bases,
            "count": len(knowledge_bases)
        })
        
    except Exception as e:
        logger.error(f"获取知识库列表失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "获取知识库列表失败",
            "message": str(e)
        }), 500

# 创建知识库
@bp.route("/knowledge_bases", methods=["POST"])
def create_knowledge_base_route():
    """创建知识库"""
    try:
        data = request.get_json()
        logger.info(f"收到创建知识库请求: {data}")

        if not data or not data.get("名称"):
            return jsonify({
                "success": False,
                "error": "知识库名称不能为空"
            }), 400

        file_type = data.get("文件类型", "text")
        if file_type not in ["text", "graph"]:
            file_type = "text"

        # 使用管理器创建知识库
        new_kb = create_knowledge_base(
            name=data.get("名称"),
            file_type=file_type
        )

        if new_kb:
            logger.info(f"成功创建知识库: {new_kb['名称']}, 文件类型: {file_type}")
            return jsonify({
                "success": True,
                "data": new_kb
            })
        else:
            return jsonify({
                "success": False,
                "error": "创建知识库失败"
            }), 500
            
    except Exception as e:
        logger.error(f"创建知识库失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "创建知识库失败",
            "message": str(e)
        }), 500

# 删除知识库
@bp.route("/knowledge_bases/<int:kb_id>", methods=["DELETE"])
def delete_knowledge_base_route(kb_id):
    """删除知识库"""
    try:
        logger.info(f"收到删除知识库请求，ID: {kb_id}")
        
        success = delete_knowledge_base(kb_id)
        
        if success:
            logger.info(f"成功删除知识库 ID: {kb_id}")
            return jsonify({
                "success": True,
                "message": f"成功删除知识库"
            })
        else:
            logger.warning(f"未找到指定的知识库 ID: {kb_id}")
            return jsonify({
                "success": False,
                "error": "未找到指定的知识库"
            }), 404
            
    except Exception as e:
        logger.error(f"删除知识库失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "删除知识库失败",
            "message": str(e)
        }), 500

# 获取知识库文件个数（实时统计，不更新JSON）
@bp.route("/knowledge_bases/<int:kb_id>/file_count", methods=["GET"])
def get_knowledge_base_file_count(kb_id):
    """获取指定知识库的实时文件个数（不更新JSON）"""
    try:
        file_count = count_files_in_knowledge_base(kb_id)
        
        return jsonify({
            "success": True,
            "data": {
                "kb_id": kb_id,
                "file_count": file_count
            }
        })
        
    except Exception as e:
        logger.error(f"获取知识库文件个数失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "获取文件个数失败",
            "message": str(e)
        }), 500

# 更新所有知识库文件个数
@bp.route("/knowledge_bases/update_all_file_counts", methods=["POST"])
def update_all_file_counts_route():
    """更新所有知识库的文件个数"""
    try:
        success = update_all_knowledge_bases_file_count()
        
        if success:
            # 重新加载更新后的知识库数据
            knowledge_bases = load_knowledge_bases()
            
            return jsonify({
                "success": True,
                "message": "成功更新所有知识库文件个数",
                "data": knowledge_bases
            })
        else:
            return jsonify({
                "success": False,
                "error": "更新文件个数失败"
            }), 500
            
    except Exception as e:
        logger.error(f"更新所有文件个数失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "更新所有文件个数失败",
            "message": str(e)
        }), 500