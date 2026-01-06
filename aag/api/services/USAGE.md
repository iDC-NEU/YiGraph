# AAG API Services 使用指南

## 概述

AAG API Services 提供了统一的服务层，封装了后端核心功能，供前端路由文件使用。

## 服务列表

- **EngineService**: 引擎管理服务（单例模式）
- **ChatService**: 聊天服务
- **DatasetService**: 数据集管理服务
- **ModelService**: 模型服务

## 使用示例

### 1. 在 sockets_chat.py 中使用 ChatService

```python
# web/frontend/route/sockets_chat.py

import asyncio
from aag.api.services.chat_service import ChatService

@socketio.on('chat_request')
def handle_chat_request(data):
    """WebSocket聊天处理器"""
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

    # 3. 使用ChatService处理请求
    chat_service = ChatService()
    
    # 确定模式
    mode = "expert" if expert_mode else "normal"
    
    # 定义回调函数用于发送流式数据
    def send_response(data_chunk):
        emit('chat_response', data_chunk)
    
    # 处理不同类型的请求
    try:
        if dag_confirm == "yes":
            # DAG确认，开始执行分析
            result = asyncio.run(chat_service.start_expert_analysis())
            send_response({
                "type": "result",
                "contentType": "text",
                "content": result.get("result", "")
            })
        elif is_dag_modification or (dag_confirm == "no" and modifications):
            # DAG修改请求
            modification_request = modifications or user_message
            result = asyncio.run(chat_service.process_dag_modification(modification_request))
            if result.get("success"):
                dag_content = chat_service._convert_dag_to_frontend_format(result)
                send_response({
                    "type": "result",
                    "contentType": "dag",
                    "content": dag_content
                })
            else:
                send_response({"error": result.get("error", "修改失败")})
        else:
            # 普通聊天请求 - 流式处理
            asyncio.run(chat_service.process_streaming_chat(
                message=user_message,
                model=selected_model,
                dataset=None,  # 可以从data中获取
                mode=mode,
                expert_mode=expert_mode,
                callback=send_response
            ))
        
        # 发送结束信号
        send_response({"type": "stream_end"})
        
    except Exception as e:
        error_msg = f"处理失败：{str(e)}"
        logger.error(error_msg)
        emit('chat_response', {'error': error_msg})
```

### 2. 在 routes_chat.py 中使用 ModelService

```python
# web/frontend/route/routes_chat.py

from flask import Blueprint, jsonify
from aag.api.services.model_service import ModelService

bp = Blueprint("chat", __name__)

@bp.route("/api/models", methods=["GET"])
def get_models():
    try:
        model_service = ModelService()
        models = model_service.get_models()
        return jsonify({
            "success": True,
            "data": models,
            "count": len(models)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

### 3. 在 routes_manage_dataset.py 中使用 DatasetService

```python
# web/frontend/route/routes_manage_dataset.py

import asyncio
from flask import Blueprint, jsonify, request
from aag.api.services.dataset_service import DatasetService

bp = Blueprint("manage", __name__)

@bp.route('/api/parse_control', methods=['GET'])
def api_parse_control():
    """HTTP API: 文件解析控制"""
    try:
        kb_id = request.args.get('kb_id')
        file_name = request.args.get('file_name')
        action = request.args.get('action')
        
        if not kb_id or not file_name or not action:
            return jsonify({
                'success': False,
                'error': 'Missing kb_id, file_name or action'
            }), 400
        
        # 获取知识库名称
        kb_name = get_knowledge_base_name(kb_id)
        
        # 构建文件路径
        kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
        file_dir = os.path.join(kb_dir, file_name)
        file_path = os.path.join(file_dir, file_name)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'File not found: {file_path}'
            }), 404
        
        name_without_ext = file_name.rsplit('.', 1)[0]
        
        # 使用DatasetService创建知识库
        dataset_service = DatasetService()
        result = asyncio.run(dataset_service.create_knowledge_base(
            file_path=file_path,
            graph_name=name_without_ext,
            db_name=kb_name
        ))
        
        if result.get("success"):
            return jsonify({
                'success': True,
                'kb_id': kb_id,
                'file_name': file_name,
                'action': action,
                'message': f'解析请求已接收: {action} {file_name}'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get("error", "创建知识库失败")
            }), 500
        
    except Exception as e:
        logger.error(f"解析控制处理错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/api/delete_file', methods=['GET'])
def api_delete_file():
    """HTTP API: 删除文件"""
    try:
        kb_id = request.args.get('kb_id')
        file_name = request.args.get('file_name')
        
        if not kb_id or not file_name:
            return jsonify({
                'success': False,
                'error': 'Missing kb_id or file_name'
            }), 400
        
        kb_name = get_knowledge_base_name(kb_id)
        name_without_ext = file_name.rsplit('.', 1)[0]
        
        # 使用DatasetService删除知识库
        dataset_service = DatasetService()
        result = asyncio.run(dataset_service.delete_knowledge_base(
            graph_name=name_without_ext,
            db_name=kb_name
        ))
        
        if result.get("success"):
            # 删除文件
            kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
            file_dir = os.path.join(kb_dir, file_name)
            if os.path.exists(file_dir):
                import shutil
                shutil.rmtree(file_dir)
            
            return jsonify({
                'success': True,
                'kb_id': kb_id,
                'file_name': file_name,
                'message': f'File {file_name} deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get("error", "删除知识库失败")
            }), 500
        
    except Exception as e:
        logger.error(f"删除文件处理错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

## 注意事项

1. **异步处理**: 所有服务方法都是异步的，需要使用 `asyncio.run()` 或在异步函数中调用
2. **引擎初始化**: EngineService 使用单例模式，首次调用会自动初始化引擎
3. **错误处理**: 所有服务方法都返回包含 `success` 字段的字典，需要检查该字段
4. **流式响应**: ChatService 的 `process_streaming_chat` 方法需要提供回调函数来发送数据

## 服务详细说明

### EngineService

- `get_instance()`: 获取单例实例
- `get_engine()`: 获取或初始化AAG引擎
- `shutdown()`: 关闭引擎
- `is_initialized()`: 检查引擎是否已初始化

### ChatService

- `process_message()`: 处理用户消息（同步接口）
- `process_streaming_chat()`: 流式处理聊天（用于WebSocket）
- `process_dag_modification()`: 处理DAG修改请求
- `start_expert_analysis()`: 开始专家模式分析

### DatasetService

- `create_knowledge_base()`: 创建知识库（从文本文件）
- `delete_knowledge_base()`: 删除知识库
- `get_triplets()`: 获取知识库的三元组
- `create_kb_from_graph()`: 从图文件创建知识库

### ModelService

- `get_models()`: 获取可用模型列表
- `get_model_by_id()`: 根据ID获取模型信息
- `get_model_by_name()`: 根据名称获取模型信息
- `get_model_config()`: 获取模型配置

