from datetime import datetime
import os
import json
import logging
import time
import random
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from flask_cors import CORS

# 配置日志（方便调试API调用过程）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(
    __name__,
    template_folder="app",  
    static_folder="static"        
)
CORS(app)  # 解决跨域问题

MODEL_MAPPING = {
    "GPT 4": "qwen3-max",    
    "Qwen 14B": "qwen3-max",  
    "Qwen Plus": "qwen3-max"          
}
# 知识库数据存储
knowledge_bases = [
    {
        "id": 1,
        "名称": "AI Knowledge Base",
        "文档个数": 128,
        "创建时间": datetime(2024, 9, 12, 10, 30).strftime("%Y-%m-%d %H:%M:%S"),
    },
    {
        "id": 2,
        "名称": "Frontend Development Docs",
        "文档个数": 86,
        "创建时间": datetime(2023, 6, 5, 14, 15).strftime("%Y-%m-%d %H:%M:%S"),
    },
    {
        "id": 3,
        "名称": "Company Policies & Regulations",
        "文档个数": 54,
        "创建时间": datetime(2022, 11, 20, 9, 0).strftime("%Y-%m-%d %H:%M:%S"),
    },
]

# 测试数据集
TEST_DATA = {
    "dag": [
        {
            "type": "thinking",
            "contentType": "text",
            "content": "用户想要一个JavaScript函数来在网页上显示随机名言。这需要几个组件：HTML结构来显示名言，JavaScript数组存储名言，以及函数来随机选择和显示名言。"
        },
        {
            "type": "result",
            "contentType": "dag",
            "content":  {
                            "nodes": [
                                {"id": "1", "label": "用户问题：分析文档"},
                                {"id": "2", "label": "提取关键词"},
                                {"id": "3", "label": "检索知识库"},
                                {"id": "4", "label": "生成回答"},
                                {"id": "5", "label": "JavaScript代码示例"}
                            ],
                            "edges": [
                                {"from": "1", "to": "2"},
                                {"from": "2", "to": "3"},
                                {"from": "3", "to": "1"},
                                {"from": "3", "to": "4"},
                                {"from": "3", "to": "5"}
                            ]
                        }
        }
    ],
    "dag_confirmation": [
        {
            "type": "thinking",
            "contentType": "text",
            "content": "用户确认了DAG结构正确，现在需要基于该DAG生成详细回答。首先我需要回顾DAG中的各个节点和流程。"
        },
        {
            "type": "thinking",
            "contentType": "text",
            "content": "DAG显示了从用户问题到提取关键词，再到检索知识库，最后生成回答的完整流程。我需要按照这个逻辑展开详细说明。"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "根据您确认的DAG结构，以下是详细的处理流程说明："
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "1. **用户问题分析（节点A）**：系统首先对用户输入的问题进行语义分析，确定问题类型和核心需求。"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "2. **关键词提取（节点B）**：从分析后的问题中提取关键信息和术语，为后续知识库检索做准备。"
        },
        {
            "type": "result",
            "contentType": "code",
            "content": {
                "language": "python",
                "code": "def extract_keywords(text):\n    # 使用NLP工具提取关键词\n    import jieba.analyse\n    keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=True)\n    return [(word, weight) for word, weight in keywords]"
            }
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "3. **知识库检索（节点C）**：基于提取的关键词在知识库中进行精确匹配和模糊搜索，获取相关文档。"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "4. **生成回答（节点D）**：结合检索到的知识和AI模型，生成准确、简洁的自然语言回答。"
        },
        {
            "type": "result",
            "contentType": "code",
            "content": {
                "language": "python",
                "code": "def categorize_goals(goals):\n    categories = {\n        'personal': [],\n        'family': [],\n        'professional': []\n    }\n    for goal in goals:\n        if any(word in goal.lower() for word in ['health', 'fitness', 'learn', 'read']):\n            categories['personal'].append(goal)\n        elif any(word in goal.lower() for word in ['family', 'spouse', 'children', 'home']):\n            categories['family'].append(goal)\n        elif any(word in goal.lower() for word in ['career', 'work', 'skill', 'project']):\n            categories['professional'].append(goal)\n    return categories"
            }
        }
    ]
}

########################这一部分集中渲染页面#################################
@app.route("/")
def index():
    """根路由：返回聊天页面"""
    return render_template("template-chatbot-s2-convo.html")  

@app.route('/overview')
def overview():
    return render_template('overview.html')

@app.route('/documents')
def documents():
    return render_template('documents.html')

@app.route('/manage_dataset')
def manage_dataset_page():
    return render_template('manage_dataset.html')


@app.route("/api/chat", methods=["GET"])
def chat():
    """聊天接口：接收用户消息，返回模拟的思考过程和最终结果"""
    # 1. 获取前端传递的参数
    try:
        user_message = request.args.get("message", "").strip()
        selected_model = request.args.get("model", "")
        dag_confirm = request.args.get("dag_confirm", "").strip()   
    except Exception as e:
        logger.error(f"解析请求参数失败：{str(e)}")
        return Response(
            json.dumps({"error": "请求格式错误，请检查参数"}),
            mimetype="application/json",
            status=400
        )

    # 2. 验证参数合法性
    if not user_message and not dag_confirm:
        return Response(
            json.dumps({"error": "消息内容不能为空"}),
            mimetype="application/json",
            status=400
        )
    
    # 3. 选择测试数据
    if dag_confirm == "yes":
        test_data_key = "dag_confirmation"
    elif "dag" in user_message.lower() or "javascript" in user_message.lower():
        test_data_key = "dag"
    elif "test" in user_message.lower():
        test_data_key = "dag"
    else:
        test_data_key = random.choice(list(TEST_DATA.keys()))
    
    test_data = TEST_DATA[test_data_key]
    
    logger.info(f"开始处理请求：模型={selected_model}，消息={user_message[:20]}...，dag_confirm={dag_confirm}，使用测试数据={test_data_key}")

    # 4. 定义流式响应生成函数
    @stream_with_context
    def generate_stream():
        try:
            # 模拟思考过程
            for item in test_data:
                # 模拟处理时间
                time.sleep(0.6)
                
                # 返回数据
                yield f"data: {json.dumps(item)}\n\n"
                logger.debug(f"返回流式数据：{item}")
            
            # 发送结束信号
            yield "event: end\ndata: Stream completed\n\n"
            logger.info("流式响应完成")

        except Exception as e:
            # 捕获所有异常并返回给前端
            error_msg = f"处理失败：{str(e)}"
            logger.error(error_msg)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    # 5. 返回流式响应（指定SSE格式）
    return Response(generate_stream(), mimetype="text/event-stream")

########################API路由#################################
@app.route("/api/models", methods=["GET"])
def get_models():
    """获取可用模型列表"""
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
        logger.error(f"获取模型列表失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "获取模型列表失败",
            "message": str(e)
        }), 500

@app.route("/api/knowledge_bases", methods=["GET"])
def get_knowledge_bases():
    """获取知识库列表 - 简化版本"""
    try:
        logger.info("收到知识库查询请求")
        
        # 直接返回数据，不进行任何复杂处理
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

@app.route("/api/knowledge_bases/<int:kb_id>", methods=["DELETE"])
def delete_knowledge_base(kb_id):
    """删除知识库"""
    try:
        global knowledge_bases
        logger.info(f"收到删除知识库请求，ID: {kb_id}")
        
        # 找到要删除的知识库索引
        original_count = len(knowledge_bases)
        knowledge_bases = [kb for kb in knowledge_bases if kb["id"] != kb_id]
        
        if len(knowledge_bases) < original_count:
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

@app.route("/api/knowledge_bases", methods=["POST"])
def create_knowledge_base():
    """创建知识库"""
    try:
        data = request.get_json()
        logger.info(f"收到创建知识库请求: {data}")
        
        if not data or not data.get("名称"):
            return jsonify({
                "success": False,
                "error": "知识库名称不能为空"
            }), 400
            
        # 生成新ID
        new_id = max([kb["id"] for kb in knowledge_bases]) + 1 if knowledge_bases else 1
        
        new_kb = {
            "id": new_id,
            "名称": data.get("名称"),
            "文档个数": 0,
            "创建时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        knowledge_bases.append(new_kb)
        
        logger.info(f"成功创建知识库: {new_kb['名称']}")
        return jsonify({
            "success": True,
            "data": new_kb
        })
        
    except Exception as e:
        logger.error(f"创建知识库失败：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "创建知识库失败",
            "message": str(e)
        }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )