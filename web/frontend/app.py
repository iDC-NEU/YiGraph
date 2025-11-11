from datetime import datetime
import os
import json
import logging
import time
import random
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量（从.env文件读取配置）
load_dotenv()

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

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "DASHSCOPE_API_KEY"),  
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

MODEL_MAPPING = {
    "GPT 4": "qwen3-max",    
    "Qwen 14B": "qwen3-max",  
    "Qwen Plus": "qwen3-max"          
}
# 知识库数据存储
knowledge_bases = [
    {
        "id": 1,
        "名称": "人工智能知识库",
        "文档个数": 128,
        "创建时间": datetime(2024, 9, 12, 10, 30).strftime("%Y-%m-%d %H:%M:%S"),
    },
    {
        "id": 2,
        "名称": "前端开发文档库",
        "文档个数": 86,
        "创建时间": datetime(2023, 6, 5, 14, 15).strftime("%Y-%m-%d %H:%M:%S"),
    },
    {
        "id": 3,
        "名称": "公司政策与规章",
        "文档个数": 54,
        "创建时间": datetime(2022, 11, 20, 9, 0).strftime("%Y-%m-%d %H:%M:%S"),
    },
]

# 测试数据集
TEST_DATA = {
    "new_year_resolutions": [
        {
            "type": "thinking",
            "contentType": "text",
            "content": "用户要求提供新年计划建议，需要分为个人、家庭和职业三个类别。我需要为每个类别提供具体、可行的建议。"
        },
        {
            "type": "thinking", 
            "contentType": "text",
            "content": "对于个人目标，应该关注健康、学习和个人成长。家庭目标应该关注关系维护和共同活动。职业目标应该关注技能提升和职业发展。"
        },
        {
            "type": "thinking",
            "contentType": "code",
            "content": {
                "language": "python",
                "code": "def categorize_goals(goals):\n    categories = {\n        'personal': [],\n        'family': [],\n        'professional': []\n    }\n    for goal in goals:\n        if any(word in goal.lower() for word in ['health', 'fitness', 'learn', 'read']):\n            categories['personal'].append(goal)\n        elif any(word in goal.lower() for word in ['family', 'spouse', 'children', 'home']):\n            categories['family'].append(goal)\n        elif any(word in goal.lower() for word in ['career', 'work', 'skill', 'project']):\n            categories['professional'].append(goal)\n    return categories"
            }
        },
        {
            "type": "result",
            "contentType": "text", 
            "content": "以下是分三个类别的新年计划建议：\n\n"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "**个人目标：**\n1. 建立每日运动习惯，每周至少锻炼3次\n2. 阅读12本书，每月1本，涵盖不同领域\n3. 学习一项新技能，如编程、烹饪或乐器\n4. 每天冥想10分钟，提升心理健康\n5. 建立健康的睡眠习惯，保证7-8小时睡眠\n\n"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "**家庭目标：**\n1. 每周安排一次家庭聚餐，增进感情\n2. 每月一次家庭户外活动，如徒步、野餐\n3. 为家庭成员庆祝生日和重要纪念日\n4. 建立家庭储蓄计划，为共同目标努力\n5. 每周至少2小时的无电子设备家庭时间\n\n"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "**职业目标：**\n1. 完成至少2个专业认证或课程\n2. 建立职业发展计划，设定季度里程碑\n3. 扩展专业网络，每月认识1-2位行业同行\n4. 提升演讲和沟通技能，主动承担更多责任\n5. 学习数据分析或AI相关技能，保持竞争力"
        }
    ],
    "random_quote_js": [
        {
            "type": "thinking",
            "contentType": "text",
            "content": "用户想要一个JavaScript函数来在网页上显示随机名言。这需要几个组件：HTML结构来显示名言，JavaScript数组存储名言，以及函数来随机选择和显示名言。"
        },
        {
            "type": "thinking",
            "contentType": "code",
            "content": {
                "language": "javascript",
                "code": "// 初步设计\nconst quotes = [\n  { text: '生活就像一盒巧克力', author: '阿甘正传' },\n  { text: '成功是1%的灵感加99%的汗水', author: '爱迪生' }\n];\n\nfunction getRandomQuote() {\n  const randomIndex = Math.floor(Math.random() * quotes.length);\n  return quotes[randomIndex];\n}"
            }
        },
        {
            "type": "thinking",
            "contentType": "text",
            "content": "还需要考虑DOM操作和事件处理，以及CSS样式让显示效果更好。应该提供完整的HTML、CSS和JavaScript代码示例。"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "以下是完整的实现代码，可以在网页上显示随机名言：\n\n"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "\n\n这个示例包含了：\n1. 响应式的HTML结构\n2. 漂亮的CSS样式，包括渐变背景和动画效果\n3. JavaScript逻辑，包含10条预定义的名言\n4. 自动加载初始名言的功能\n5. 平滑的过渡动画效果\n\n您可以直接复制这段代码到HTML文件中，用浏览器打开即可使用。如需添加更多名言，只需在`quotes`数组中添加新的对象即可。"
        }
    ],
    "sentiment_analysis": [
        {
            "type": "thinking", 
            "contentType": "text",
            "content": "用户想要一个Python脚本来分析文本情感。这需要使用自然语言处理库，如NLTK或TextBlob。我会使用TextBlob，因为它更简单且易于使用。"
        },
        {
            "type": "thinking",
            "contentType": "text",
            "content": "需要添加错误处理、批量分析功能，以及从文件读取文本的能力。还应该提供可视化功能来展示结果。"
        },
        {
            "type": "result",
            "contentType": "text",
            "content": "以下是完整的Python文本情感分析脚本：\n\n"
        },
        {
            "type": "result",
            "contentType": "code",
            "content": {
                "language": "python",
                "code": "# 初步设计\nfrom textblob import TextBlob\n\ndef analyze_sentiment(text):\n    analysis = TextBlob(text)\n    polarity = analysis.sentiment.polarity\n    \n    if polarity > 0.1:\n        return '正面', polarity\n    elif polarity < -0.1:\n        return '负面', polarity\n    else:\n        return '中性', polarity"
            }
        },
        {
            "type": "result", 
            "contentType": "text",
            "content": "\n\n### 使用说明：\n\n1. **安装依赖**：\n```bash\npip install textblob matplotlib pandas numpy\npython -m textblob.download_corpora\n```\n\n2. **功能特点**：\n- 单文本实时分析\n- 批量文件分析（支持TXT、CSV、JSON、Excel）\n- 情感分布可视化\n- 情感极性趋势分析\n- 详细的统计报告\n- 结果导出功能\n\n3. **运行方式**：\n```bash\npython sentiment_analyzer.py\n```\n\n这个脚本提供了完整的文本情感分析功能，适合用于社交媒体监控、客户反馈分析、市场调研等场景。"
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
def manage_dataset():
    return render_template('manage_dataset.html')


@app.route("/api/chat", methods=["GET"])
def chat():
    """聊天接口：接收用户消息，返回模拟的思考过程和最终结果"""
    # 1. 获取前端传递的参数
    try:
        user_message = request.args.get("message", "").strip()
        selected_model = request.args.get("model", "")
    except Exception as e:
        logger.error(f"解析请求参数失败：{str(e)}")
        return Response(
            json.dumps({"error": "请求格式错误，请检查参数"}),
            mimetype="application/json",
            status=400
        )

    # 2. 验证参数合法性
    if not user_message:
        return Response(
            json.dumps({"error": "消息内容不能为空"}),
            mimetype="application/json",
            status=400
        )
    
    # 3. 选择测试数据
    if "new year" in user_message.lower() or "resolution" in user_message.lower():
        test_data_key = "new_year_resolutions"
    elif "quote" in user_message.lower() or "javascript" in user_message.lower():
        test_data_key = "random_quote_js"
    elif "sentiment" in user_message.lower() or "analysis" in user_message.lower() or "python" in user_message.lower():
        test_data_key = "sentiment_analysis"
    else:
        test_data_key = random.choice(list(TEST_DATA.keys()))
    
    test_data = TEST_DATA[test_data_key]
    
    logger.info(f"开始处理请求：模型={selected_model}，消息={user_message[:20]}...，使用测试数据={test_data_key}")

    # 4. 定义流式响应生成函数
    @stream_with_context
    def generate_stream():
        try:
            # 模拟思考过程
            for item in test_data:
                # 模拟处理时间
                time.sleep(0.5)
                
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

def test_qwen_api():
    try:
        print("测试Qwen API连接...")
        completion = client.chat.completions.create(
            model="qwen3-max",
            messages=[{"role": "user", "content": "你好"}],
            stream=False
        )
        print("API测试成功！响应：", completion.choices[0].message.content)
    except Exception as e:
        print("API测试失败：", str(e))
        print("将使用模拟数据进行测试")

########################API路由#################################
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
    #test_qwen_api()
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
