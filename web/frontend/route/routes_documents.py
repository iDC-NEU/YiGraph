from flask import Blueprint, request, jsonify
import logging
import sys
import json
import asyncio
from document_schema import (
    load_knowledge_bases,
    create_knowledge_base,
    delete_knowledge_base,
    count_files_in_knowledge_base,
    update_all_knowledge_bases_file_count,
)
from DummySocket import DummySocket
sys.path.append("../../")
from aag.api.DocumentAPI import server_Test

# 创建蓝图
bp = Blueprint('documents', __name__, url_prefix='/api')

# 配置日志
logger = logging.getLogger(__name__)

def load_knowledge_bases():
    a23 = DummySocket(json.dumps({"action": "get_datasets"}))
    asyncio.run(server_Test.handler(a23))
    knowledge_bases_with_count = a23.returnmsg
    logger.info("#### %s",knowledge_bases_with_count)
    logger.info("#### %s",type(knowledge_bases_with_count))
    print(jsonify({
        'success': True,
        'data': knowledge_bases_with_count
    }))
    basescy = json.loads(knowledge_bases_with_count)
    logger.info("#### %s",basescy)
    logger.info("#### %s",type(basescy))
    return basescy["content"]["data"]


def _parse_ws_result(raw_msg: str):
    """统一解析 DummySocket 返回的 JSON 字符串"""
    try:
        payload = json.loads(raw_msg)
    except Exception as e:
        logger.error("解析 DocumentAPI 返回值失败: %s | 原始: %s", e, raw_msg)
        return None

    # DocumentAPI 返回格式可能是 {"type":"data","contentType":"json","content":{...}}
    if isinstance(payload, dict):
        if "content" in payload:
            return payload["content"]
        # 也可能直接是最终数据
        return payload
    return None

def get_knowledge_base_name(kb_id):
    """根据知识库ID获取知识库名称"""
    try:
        kb_id = int(kb_id)  
        knowledge_bases = load_knowledge_bases() 
        for kb in knowledge_bases:
            if kb["id"] == kb_id:
                return kb["名称"]
                logger.info(f"名称获取成功：{kb}")
        return f"kb_{kb_id}"  
    except Exception as e:
        logger.error(f"获取知识库名称错误: {str(e)}")
        return f"kb_{kb_id}"

# 获取知识库列表
@bp.route("/knowledge_bases", methods=["GET"])
def get_knowledge_bases():
    """获取知识库列表"""
    try:
        logger.info("收到知识库查询请求")
        
        a1 = DummySocket(json.dumps({"action": "get_datasets"}))
        asyncio.run(server_Test.handler(a1))
        gkb = a1.returnmsg
        knowledge_bases = json.loads(gkb)
        logger.info("####这是子涵页面里信息 %s",knowledge_bases)
        logger.info("####这是子涵页面里信息类型 %s",type(knowledge_bases))
        for kb in knowledge_bases["content"]["data"]:
            if kb["文件类型"] == "graph":
                if kb["文档个数"] == 1:
                    a6 = DummySocket(json.dumps({"action": "get_dataset_schema","ds_name":kb["名称"]}))
                    asyncio.run(server_Test.handler(a6))
                    gkb6 = a6.returnmsg
                    gkb6_json = json.loads(gkb6)
                    logger.info("####判断信息类型 %s",gkb6_json)
                    logger.info("####判断信息类型 %s",type(gkb6_json))
                    if gkb6_json["content"]["data"][0].get("vertex_file",None) is not None:
                        kb["文档个数"] = 2
        return jsonify({
            "success": True,
            "data": knowledge_bases["content"]["data"],
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
        # new_kb = create_knowledge_base(
        #     name=data.get("名称"),
        #     file_type=file_type
        # )

        a2 = DummySocket(json.dumps({"action": "create_dataset","name":data.get("名称"),"type":data.get("文件类型", "text")}))
        asyncio.run(server_Test.handler(a2))
        result = _parse_ws_result(getattr(a2, "returnmsg", "{}"))

        if result and result.get("success"):
            # DocumentAPI 返回 content.data 里包含 db_name/message
            return jsonify({
                "success": True,
                "data": result.get("data", {})
            })
        else:
            err = None
            if result:
                err = result.get("error") or result.get("message")
            return jsonify({
                "success": False,
                "error": err or "创建知识库失败"
            }), 500
            
    except Exception as e:
        logger.error(f"创建知识库失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "创建知识库失败",
            "message": str(e)
        }), 500

@bp.route("/knowledge_bases/<int:kb_id>", methods=["DELETE"])
def delete_knowledge_base_route(kb_id):
    """删除知识库"""
    try:
        logger.info(f"收到删除知识库请求，ID: {kb_id}")
        
        # 获取要删除的知识库名称，用于后续调用 
        #knowledge_bases = load_knowledge_bases()
        kb_to_delete = get_knowledge_base_name(kb_id)
        # for kb in knowledge_bases:
        #     if kb.get("id") == kb_id:
        #         kb_to_delete = kb
        #         break
        
        if not kb_to_delete:
            logger.warning(f"未找到指定的知识库 ID: {kb_id}")
            return jsonify({
                "success": False,
                "error": "未找到指定的知识库"
            }), 404
        
        a3 = DummySocket(json.dumps({"action": "delete_dataset","ds_name":kb_to_delete}))
        asyncio.run(server_Test.handler(a3))
        result = _parse_ws_result(getattr(a3, "returnmsg", "{}"))

        if result and result.get("success"):
            return jsonify({
                "success": True,
                "message": result.get("message") or "成功删除知识库"
            })
        else:
            err = None
            if result:
                err = result.get("error") or result.get("message")
            logger.warning(f"未找到指定的知识库 ID: {kb_id}")
            return jsonify({
                "success": False,
                "error": err or "未找到指定的知识库"
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