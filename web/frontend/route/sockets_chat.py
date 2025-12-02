import asyncio
import sys
import os
from . import socketio
import logging
from flask_socketio import emit

# 添加项目路径以导入AAG服务
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from aag.api.services.chat_service import ChatService

logger = logging.getLogger(__name__)

import re

def split_into_sentences(text: str, max_len: int = 80):
    if not text:
        return [text]

    # 第一步：尝试用标点分句（中英文都支持）
    # (?<=[。！？\n])          中文标点或换行后
    # (?<=[.!?]\s)            英文句号/问号/感叹号 + 空格后
    sentences = re.split(r'(?<=[。！？\n])|(?<=[.!?]\s)', text)
    
    # 过滤空字符串并去除首尾空白
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 如果分不出句子（比如全是英文没有空格后标点，或纯长文本），就按段落分
    if len(sentences) <= 1:
        # 按空行分段，再按单换行分（避免段落被强行合并）
        paragraphs = [p.strip() for p in text.replace('\n\n', '\n').split('\n') if p.strip()]
        if len(paragraphs) > 1:
            sentences = paragraphs

    # 第二步：对每个句子，如果太长就按单词边界智能切分
    def smart_split_long_sentence(sentence: str, max_len: int):
        if len(sentence) <= max_len:
            return [sentence]
        
        result = []
        start = 0
        text_len = len(sentence)
        
        while start < text_len:
            end = start + max_len
            
            # 如果已经到结尾，直接加入
            if end >= text_len:
                result.append(sentence[start:].strip())
                break
            
            # 从 end 往前找第一个空格或中文标点，作为切割点
            split_pos = -1
            for i in range(end, max( start, end - 20), -1):  # 只往前看最多20个字符，效率高
                if sentence[i] in ' \t\n。！？,!?':
                    split_pos = i
                    break
            
            # 如果实在找不到空格（如全是中文或超长无空格英文），就按 max_len 
            if split_pos == -1:
                split_pos = end
            
            chunk = sentence[start:split_pos + 1].strip()
            if chunk:
                result.append(chunk)
            start = split_pos + 1
            
            # 防止死循环（极端情况）
            if start >= text_len:
                break
                
        return result

    final = []
    for sent in sentences:
        if len(sent) > max_len:
            final.extend(smart_split_long_sentence(sent, max_len))
        else:
            final.append(sent)
    
    return final if final else [text.strip()]


@socketio.on('chat_request')
def handle_chat_request(data):
    """WebSocket聊天处理器：接收用户消息，推送流式结果"""
    # 1. 获取参数
    try:
        user_message = data.get("message", "").strip()
        selected_model = data.get("model", "")
        dag_confirm = data.get("dag_confirm", "").strip()
        is_dag_modification = str(data.get("is_dag_modification", "false")).lower() == "true"
        dag_id = data.get("dag_id", "")
        modifications = data.get("modifications", "")
        expert_mode = data.get("expert_mode", False)
        dataset = data.get("dataset", "AMLSim1K")  # 数据集名称，默认使用AMLSim1K
        
    except Exception as e:
        logger.error(f"解析参数失败：{str(e)}")
        emit('chat_response', {"error": "请求格式错误，请检查参数"})
        return

    # 2. 验证
    if not user_message and not dag_confirm and not is_dag_modification:
        emit('chat_response', {"error": "消息内容不能为空"})
        return

    # 3. 使用ChatService处理请求
    try:
        chat_service = ChatService()
        
        # 确保使用默认数据集AMLSim1K（如果前端没有指定）
        if not dataset:
            dataset = "AMLSim1K"
        
        # 定义回调函数用于发送流式数据
        def send_response(data_chunk):
            """发送响应数据到前端"""
            emit('chat_response', data_chunk)
        
        # 确定模式
        mode = "expert" if expert_mode else "normal"
        
        logger.info(f"WS处理请求：模型={selected_model}，数据集={dataset}，消息={user_message[:20]}...，ExpertMode={expert_mode}，Mode={mode}")
        
        # 处理不同类型的请求
        if dag_confirm == "yes":
            # DAG确认，开始执行分析
            # 先选择数据集
            engine = chat_service.engine_service.get_engine()
            engine.specific_dataset(dataset)
            logger.info(f"DAG确认，开始执行专家模式分析")
            result = asyncio.run(chat_service.start_expert_analysis())
            
            if result.get("success"):
                result_text = result.get("result", "")
                # 分段发送文本结果
                paragraphs = [p.strip() for p in result_text.split('\n') if p.strip()]
                for i, para in enumerate(paragraphs):
                    send_response({
                        'type': 'result',
                        'contentType': 'text',
                        'content': para
                    })
                    socketio.sleep(0.5 if i < len(paragraphs)-1 else 0.3)
            else:
                send_response({
                    'error': result.get("error", "分析执行失败")
                })
                
        elif is_dag_modification or (dag_confirm == "no" and modifications):
            # DAG修改请求
            # 先选择数据集
            engine = chat_service.engine_service.get_engine()
            engine.specific_dataset(dataset)
            modification_request = modifications or user_message
            logger.info(f"收到DAG修改请求：{modification_request}")
            
            result = asyncio.run(chat_service.process_dag_modification(modification_request))
            
            if result.get("success"):
                # 转换DAG格式并发送
                dag_content = chat_service._convert_dag_to_frontend_format(result)
                send_response({
                    'type': 'result',
                    'contentType': 'dag',
                    'content': dag_content
                })
            else:
                send_response({
                    'error': result.get("error", "DAG修改失败")
                })
                
        else:
            # 普通聊天请求 - 流式处理
            logger.info(f"处理普通聊天请求，模式={mode}")
            
            # 使用流式处理（内部会发送stream_end信号）
            # dataset参数已确保有默认值AMLSim1K
            asyncio.run(chat_service.process_streaming_chat(
                message=user_message,
                model=selected_model,
                dataset=dataset,
                mode=mode,
                expert_mode=expert_mode,
                callback=send_response
            ))
        
        # 对于非流式处理的请求，需要手动发送结束信号
        if dag_confirm == "yes" or (is_dag_modification or (dag_confirm == "no" and modifications)):
            send_response({'type': 'stream_end'})
        
        logger.info("WS响应完成")
        
    except Exception as e:
        error_msg = f"处理失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        emit('chat_response', {'error': error_msg})
