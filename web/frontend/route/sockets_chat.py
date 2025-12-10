import json
import time
from . import socketio
import os
import json
import logging
import random
from flask_cors import CORS
from flask_socketio import emit


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

def smart_split_markdown(text: str, max_len: int = 80):
    """
    Markdown 感知的智能切分函数。
    优先保护代码块和关键语法不被拆分，实现流式输出的“打字机”效果。
    """
    if not text:
        return []

    # --- 第一层：保护代码块 ---
    # 使用正则将文本拆分为：[普通文本, 代码块, 普通文本, 代码块...]
    # (```[\s\S]*?```) 匹配多行代码块
    # (`[^`\n]+`) 匹配行内代码（简单的防止在变量名中间切分）
    parts = re.split(r'(```[\s\S]*?```|`[^`\n]+`)', text)

    atoms = []
    
    for part in parts:
        if not part:
            continue
            
        # 1. 如果是代码块 (以 ` 开头)，视为一个整体原子
        if part.startswith('`'):
            # 如果代码块特别长（超过 max_len），为了流式体验，我们只能按换行符强行切分
            # 否则前端会卡顿很久才一次性渲染出来
            if len(part) > max_len and '\n' in part:
                code_lines = part.split('\n')
                # 保留换行符，重组原子
                for idx, line in enumerate(code_lines):
                    suffix = '\n' if idx < len(code_lines) - 1 else ''
                    atoms.append(line + suffix)
            else:
                atoms.append(part)
        
        # 2. 如果是普通文本，进行细粒度切分
        else:
            # 先按段落/换行切分 (Markdown中换行很重要)
            # 这一步能有效保护列表项 (如 "1. xxx") 不被横腰截断
            lines = part.split('\n')
            for i, line in enumerate(lines):
                # 恢复换行符（最后一行除外）
                suffix = '\n' if i < len(lines) - 1 else ''
                full_line = line + suffix
                
                if not line.strip():
                    atoms.append(full_line)
                    continue

                # 在行内按句子标点切分
                # pattern解释: 
                # ([。！？]) -> 中文标点
                # ([.!?]\s) -> 英文标点+空格 (Lookbehind不方便捕获，这里直接切分再拼回去)
                sub_parts = re.split(r'([。！？]|(?<=[.!?])\s)', line)
                
                current_sent = ""
                for sub in sub_parts:
                    current_sent += sub
                    # 如果积累到了标点，或者本来就是短片段，就作为一个原子存入
                    if sub in ['。', '！', '？'] or (sub.strip() == '' and len(current_sent) > 0):
                        atoms.append(current_sent)
                        current_sent = ""
                    # 英文句号的情况处理 (正则切分后空格可能在下一个片段)
                    elif len(current_sent) > 0 and current_sent[-1] in '.!?':
                         # 这里稍微激进一点，遇到英文标点暂不切，等后面的空格，或者直接切
                         # 为了简化，这里不做过度复杂的英文NLP，直接作为原子
                         pass
                
                if current_sent:
                    atoms.append(current_sent)
                
                # 别忘了补上换行符（作为单独的原子或附在最后一个原子后）
                if suffix:
                    if atoms:
                        atoms[-1] += suffix
                    else:
                        atoms.append(suffix)

    # --- 第二层：组装 (Re-assemble) ---
    # 将原子拼凑成长度不超过 max_len 的块
    final_chunks = []
    current_chunk = ""

    for atom in atoms:
        # 如果加入当前原子会超出限制，且当前块不为空 -> 发送当前块
        if len(current_chunk) + len(atom) > max_len:
            if current_chunk:
                final_chunks.append(current_chunk)
                current_chunk = ""
            
            # 如果单个原子本身就巨长（比如超长的一行文本），强行加入（或考虑再次强切，这里选择保留完整性）
            current_chunk = atom
        else:
            current_chunk += atom

    if current_chunk:
        final_chunks.append(current_chunk)

    return final_chunks


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
    except Exception as e:
        logger.error(f"解析参数失败：{str(e)}")
        emit('chat_response', {"error": "请求格式错误，请检查参数"})
        return

    # 2. 验证
    if not user_message and not dag_confirm and not is_dag_modification:
        emit('chat_response', {"error": "消息内容不能为空"})
        return
    
    """

        这里是写死的读取config 模板
        3. 选择测试数据

    """

    if dag_confirm == "yes":
        test_data_key = "dag_new"
    elif not expert_mode:
        test_data_key = "normal"
    elif dag_confirm == "yes":
        test_data_key = "dag_confirmation"
    elif is_dag_modification or (dag_confirm == "no" and modifications):
        logger.info(f"收到DAG修改请求：{user_message[:50]}...")
        test_data_key = "dag_modification"
    elif "dag" in user_message.lower() or "javascript" in user_message.lower():
        test_data_key = "dag"
    elif "test" in user_message.lower():
        test_data_key = "dag"
    else:
        test_data_key = random.choice(list(TEST_DATA.keys()))

    logger.info(f"WS处理请求：模型={selected_model}，消息={user_message[:20]}...，ExpertMode={expert_mode}，KEY={test_data_key}")

    # 4. 流式推送数据
    try:
        thinking_done = False
        
        for item in test_data:
            if item["type"] == "thinking":
                content = item["content"]
                sentences = split_into_sentences(content)
                for i, sentence in enumerate(sentences):
                    clean_sentence = sentence.strip()
                    if not clean_sentence:
                        continue
                    
                    # 发送思考片段
                    emit('chat_response', {
                        'type': 'thinking',
                        'contentType': 'text',
                        'content': clean_sentence
                    })
                    # 非阻塞休眠
                    socketio.sleep(1 if i < len(sentences)-1 else 0.8)

                if not thinking_done:
                    thinking_done = True

            elif item["type"] == "result":
                if thinking_done:
                    thinking_done = False
                
                if item["contentType"] == "text":
                    paragraphs = [p.strip() for p in item["content"].split('\n') if p.strip()]
                    for i, para in enumerate(paragraphs):
                        emit('chat_response', {
                            'type': 'result',
                            'contentType': 'text',
                            'content': para
                        })
                        socketio.sleep(2 if i < len(paragraphs)-1 else 1.8)
                
                elif item["contentType"] == "dag":
                    emit('chat_response', {
                        'type': 'result',
                        'contentType': 'dag',
                        'content': item['content']
                    })
        
        # 5. 发送结束信号
        emit('chat_response', {'type': 'stream_end'})
        logger.info("WS响应完成")

    except Exception as e:
        error_msg = f"处理失败：{str(e)}"
        logger.error(error_msg)
        emit('chat_response', {'error': error_msg})