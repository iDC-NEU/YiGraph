import sys
import os
import re
import logging
import asyncio
from flask_socketio import emit
from . import socketio
# Add project path to import AAG services
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from aag.api.async_runtime import get_background_loop, get_chat_service


logger = logging.getLogger(__name__)


def split_into_sentences(text: str, max_len: int = 80):
    if not text:
        return [text]

    # Step 1: split by punctuation (CJK and Western)
    # (?<=[。！？\n])  after CJK punctuation or newline
    # (?<=[.!?]\s)     after period/question/exclamation + space
    sentences = re.split(r'(?<=[。！？\n])|(?<=[.!?]\s)', text)
    
    # Filter empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If no sentence boundaries found (e.g. long run-on text), split by paragraphs
    if len(sentences) <= 1:
        # Split by blank lines, then by single newlines
        paragraphs = [p.strip() for p in text.replace('\n\n', '\n').split('\n') if p.strip()]
        if len(paragraphs) > 1:
            sentences = paragraphs

    # Step 2: for each sentence, if too long, split at word/punctuation boundaries
    def smart_split_long_sentence(sentence: str, max_len: int):
        if len(sentence) <= max_len:
            return [sentence]
        
        result = []
        start = 0
        text_len = len(sentence)
        
        while start < text_len:
            end = start + max_len
            
            # Already at end, append remainder
            if end >= text_len:
                result.append(sentence[start:].strip())
                break
            
            # Look backward from end for first space or CJK punctuation as split point
            split_pos = -1
            for i in range(end, max(start, end - 20), -1):
                if sentence[i] in ' \t\n。！？,!?':
                    split_pos = i
                    break
            
            # No break character found, force split at max_len
            if split_pos == -1:
                split_pos = end
            
            chunk = sentence[start:split_pos + 1].strip()
            if chunk:
                result.append(chunk)
            start = split_pos + 1
            
            # Guard against infinite loop
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
    Markdown-aware smart split. Preserves code blocks and key syntax;
    produces chunks suitable for streaming "typewriter" output.
    """
    if not text:
        return []

    # --- Layer 1: preserve code blocks ---
    # Split into [plain, code block, plain, code block, ...]
    # (```[\s\S]*?```) multi-line code, (`[^`\n]+`) inline code
    parts = re.split(r'(```[\s\S]*?```|`[^`\n]+`)', text)

    atoms = []
    
    for part in parts:
        if not part:
            continue
            
        # 1. Code block (starts with `): treat as single atom
        if part.startswith('`'):
            # Very long block: split by newlines for streaming
            if len(part) > max_len and '\n' in part:
                code_lines = part.split('\n')
                for idx, line in enumerate(code_lines):
                    suffix = '\n' if idx < len(code_lines) - 1 else ''
                    atoms.append(line + suffix)
            else:
                atoms.append(part)
        
        # 2. Plain text: fine-grained split
        else:
            # Split by paragraph/newline first (important in Markdown)
            lines = part.split('\n')
            for i, line in enumerate(lines):
                suffix = '\n' if i < len(lines) - 1 else ''
                full_line = line + suffix
                
                if not line.strip():
                    atoms.append(full_line)
                    continue

                # Split by sentence punctuation within line
                sub_parts = re.split(r'([。！？]|(?<=[.!?])\s)', line)
                
                current_sent = ""
                for sub in sub_parts:
                    current_sent += sub
                    if sub in ['。', '！', '？'] or (sub.strip() == '' and len(current_sent) > 0):
                        atoms.append(current_sent)
                        current_sent = ""
                    elif len(current_sent) > 0 and current_sent[-1] in '.!?':
                         pass
                
                if current_sent:
                    atoms.append(current_sent)
                
                if suffix:
                    if atoms:
                        atoms[-1] += suffix
                    else:
                        atoms.append(suffix)

    # --- Layer 2: reassemble into chunks <= max_len ---
    final_chunks = []
    current_chunk = ""

    for atom in atoms:
        if len(current_chunk) + len(atom) > max_len:
            if current_chunk:
                final_chunks.append(current_chunk)
                current_chunk = ""
            current_chunk = atom
        else:
            current_chunk += atom

    if current_chunk:
        final_chunks.append(current_chunk)

    return final_chunks


@socketio.on('chat_request')
def handle_chat_request(data):
    """WebSocket chat handler: receive user message, push streaming results."""
    # 1. Parse parameters
    try:
        user_message = str(data.get("message") or "").strip()
        selected_model = str(data.get("model") or "")
        dag_confirm = str(data.get("dag_confirm") or "").strip()
        is_dag_modification = str(data.get("is_dag_modification", "false")).lower() == "true"
        dag_id = str(data.get("dag_id") or "")
        modifications = data.get("modifications", "")  # may be str or other
        expert_mode = data.get("expert_mode", False)
        dataset = str(data.get("dataset") or "").strip()  # dataset name from frontend
        _dtype = data.get("dataset_type") or data.get("file_type")
        dataset_type = str(_dtype).strip() if _dtype else None  # "text" | "graph" | None
        custom_mode = data.get("mode")
    except Exception as e:
        logger.error(f"Failed to parse parameters: {e}")
        emit('chat_response', {"error": "Invalid request format. Please check parameters."})
        return

    # 2. Validation
    if not user_message and not dag_confirm and not is_dag_modification:
        emit('chat_response', {"error": "Message content cannot be empty."})
        return

    if not dataset:
        emit('chat_response', {"error": "Dataset is empty. Please specify a dataset first."})
        return

    try:
        chat_service = get_chat_service()
        loop = get_background_loop()

        # Callback to send streaming data (called from background event loop thread)
        def send_response(data_chunk):
            """Send response data to frontend."""
            socketio.emit('chat_response', data_chunk)

        # Determine mode
        if custom_mode == "interact":
            mode = "interact"
        else:
            mode = "expert" if expert_mode else "normal"
        logger.info(f"WS request: model={selected_model}, dataset={dataset}, message={user_message[:20]}..., expertMode={expert_mode}, mode={mode}")
        print(f"WS request: model={selected_model}, dataset={dataset}, message={user_message[:20]}..., expertMode={expert_mode}, mode={mode}")

        async def process_request():
            try:
                if dag_confirm == "yes":
                    engine = chat_service.engine_service.get_engine()
                    engine.specific_dataset(dataset, dataset_type)
                    logger.info("DAG confirmed, starting expert mode analysis")
                    result = await chat_service.start_expert_analysis()

                    if result.get("success"):
                        result_text = result.get("result", "")
                        paragraphs = [p.strip() for p in result_text.split('\n') if p.strip()]
                        for i, para in enumerate(paragraphs):
                            send_response({
                                'type': 'result',
                                'contentType': 'text',
                                'content': para
                            })
                            await asyncio.sleep(0.5 if i < len(paragraphs)-1 else 0.3)
                    else:
                        send_response({
                            'error': result.get("error", "Analysis execution failed.")
                        })
                    send_response({'type': 'stream_end'})
                    return

                if is_dag_modification or (dag_confirm == "no" and modifications):
                    engine = chat_service.engine_service.get_engine()
                    engine.specific_dataset(dataset, dataset_type)
                    modification_request = modifications or user_message
                    logger.info(f"DAG modification request received: {modification_request}")

                    result = await chat_service.process_dag_modification(modification_request)

                    if result.get("success"):
                        dag_content = chat_service._convert_dag_to_frontend_format(result)
                        send_response({
                            'type': 'result',
                            'contentType': 'dag',
                            'content': dag_content
                        })
                    else:
                        send_response({
                            'error': result.get("error", "DAG modification failed.")
                        })
                    send_response({'type': 'stream_end'})
                    return

                # Normal chat request — streaming (stream_end is sent internally)
                logger.info(f"Processing normal chat request, mode={mode}")
                await chat_service.process_streaming_chat(
                    message=user_message,
                    model=selected_model,
                    dataset=dataset,
                    dataset_type=dataset_type,
                    mode=mode,
                    expert_mode=expert_mode,
                    callback=send_response
                )
            except Exception as exc:
                logger.error(f"Background processing failed: {exc}", exc_info=True)
                send_response({'type': 'result', 'contentType': 'text', 'content': CHAT_FRIENDLY_ERROR_MSG})
                send_response({'type': 'stream_end'})

        future = asyncio.run_coroutine_threadsafe(process_request(), loop)
        future.result()

        logger.info("WS response completed")

    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        emit('chat_response', {'type': 'result', 'contentType': 'text', 'content': CHAT_FRIENDLY_ERROR_MSG})
        emit('chat_response', {'type': 'stream_end'})
