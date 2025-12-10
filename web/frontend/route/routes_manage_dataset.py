import logging
from datetime import datetime
import sys
import os
import json
import asyncio
import base64
import yaml

from flask import Blueprint, jsonify, request
from flask_socketio import emit, join_room, leave_room
sys.path.append("../../")
from aag.api.DocumentAPI import server_Test,DummySocket
from document_schema import (
    load_knowledge_bases, 
    create_knowledge_base, 
    delete_knowledge_base,
    count_files_in_knowledge_base,
    update_all_knowledge_bases_file_count
)
#from document_schema import load_knowledge_bases

from .config import knowledge_bases

current_dir = os.path.dirname(os.path.abspath(__file__))
# 在文件开头添加YAML文件路径
GRAPH_SCHEMAS_PATH = os.path.join(
    current_dir, 
    "../../../aag/data_pipeline/data_transformer/dataset_schemas/graph_schemas.yaml"
)
TEXT_SCHEMAS_PATH = os.path.join(
    current_dir,
    "../../../aag/data_pipeline/data_transformer/dataset_schemas/text_schemas.yaml"
)

logger = logging.getLogger(__name__)

bp = Blueprint("manage", __name__)

# 确保上传目录存在 - 修改为前一个目录下的debug/files
UPLOAD_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "debug", "files")
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)

def load_yaml_file(file_path):
    """加载YAML文件"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return None
    except Exception as e:
        logger.error(f"加载YAML文件失败 {file_path}: {str(e)}")
        return None

def check_file_parsed_status(file_name, dataset_type):
    """
    检查文件是否已解析
    返回: True(已解析)/False(未解析)
    """
    try:
        # 从文件名提取基础名称（去掉扩展名）
        base_name = file_name.split('.')[0]
        
        if dataset_type == 'graph':
            # 检查graph_schemas.yaml
            if os.path.exists(TEXT_SCHEMAS_PATH):
                print(f"路径存在: {TEXT_SCHEMAS_PATH}")
            else:
                print(f"路径不存在: {TEXT_SCHEMAS_PATH}")
            graph_data = load_yaml_file(GRAPH_SCHEMAS_PATH)
            logger.info(f"这里可以")
            logger.info(GRAPH_SCHEMAS_PATH)
            if graph_data in graph_data:
                logger.info(graph_data)
                logger.info(f"找到yaml文件")
                for dataset in graph_data['datasets']:
                    logger.info(dataset.get('name'))
                    if dataset.get('name') == base_name:
                        
                        logger.info(f"文件 {file_name} 在graph_schemas.yaml中找到，解析状态: 已解析")
                        return True
        
        elif dataset_type == 'text':
            # 检查text_schemas.yaml
            if os.path.exists(TEXT_SCHEMAS_PATH):
                logger.info(f"路径存在: {TEXT_SCHEMAS_PATH}")
            else:
                logger.info(f"路径不存在: {TEXT_SCHEMAS_PATH}")
            text_data = load_yaml_file(GRAPH_SCHEMAS_PATH)
            logger.info(f"这里可以")
            logger.info(TEXT_SCHEMAS_PATH)
            if True:
                logger.info(text_data)
                logger.info(f"找到yaml文件")
                for dataset in text_data['datasets']:
                    logger.info("#########",dataset.get('name'))
                    if dataset.get('name') == base_name:
                        logger.info(f"文件 {file_name} 在text_schemas.yaml中找到，解析状态: 已解析")
                        return True
        
        logger.info(f"文件 {file_name} 未在YAML文件中找到，解析状态: 未解析")
        return False
        
    except Exception as e:
        logger.error(f"检查文件解析状态失败 {file_name}: {str(e)}")
        return False

def get_all_files_parsed_status(kb_id, files):
    """
    获取数据集所有文件的解析状态
    返回: {
        'all_parsed': True/False,
        'file_status': {
            'file1.txt': True,
            'file2.pdf': False,
            ...
        }
    }
    """
    try:
        # 获取数据集类型
        dataset_type = get_dataset_type_from_json(kb_id)
        
        file_status = {}
        all_parsed = True
        
        for file_name in files:
            is_parsed = check_file_parsed_status(file_name, dataset_type)
            file_status[file_name] = is_parsed
            if not is_parsed:
                all_parsed = False
        logger.info({
            'all_parsed': all_parsed,
            'file_status': file_status,
            'dataset_type': dataset_type
        })
        return {
            'all_parsed': all_parsed,
            'file_status': file_status,
            'dataset_type': dataset_type
        }
        
    except Exception as e:
        logger.error(f"获取文件解析状态失败: {str(e)}")
        return {
            'all_parsed': False,
            'file_status': {},
            'dataset_type': 'unknown'
        }

# 添加新的API端点来检查解析状态
@bp.route('/api/check_parsing_status', methods=['POST'])
def api_check_parsing_status():
    """检查数据集所有文件的解析状态"""
    try:
        data = request.get_json()
        kb_id = data.get('kb_id')
        files = data.get('files', [])
        
        if not kb_id:
            return jsonify({
                'success': False,
                'error': 'Missing kb_id'
            }), 400
        
        logger.info(f"检查解析状态请求: 知识库 {kb_id}, 文件数 {len(files)}")
        
        # 获取解析状态
        parsing_status = get_all_files_parsed_status(kb_id, files)
        
        return jsonify({
            'success': True,
            'kb_id': kb_id,
            'all_parsed': parsing_status['all_parsed'],
            'file_status': parsing_status['file_status'],
            'dataset_type': parsing_status['dataset_type'],
            'message': f"解析状态: {len(files)}个文件中{sum(parsing_status['file_status'].values())}个已解析"
        })
        
    except Exception as e:
        logger.error(f"检查解析状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def get_knowledge_base_name(kb_id):
    """根据知识库ID获取知识库名称"""
    try:
        kb_id = int(kb_id)  
        knowledge_bases = load_knowledge_bases() 
        for kb in knowledge_bases:
            if kb["id"] == kb_id:
                return kb["名称"]
        return f"kb_{kb_id}"  
    except Exception as e:
        logger.error(f"获取知识库名称错误: {str(e)}")
        return f"kb_{kb_id}"

def get_files_count_for_knowledge_base(kb_id):
    """获取指定知识库的文件数量"""
    try:
        kb_name = get_knowledge_base_name(kb_id)
        kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
        
        if not os.path.exists(kb_dir):
            return 0
            
        # 计算目录数量（每个文件一个目录）
        file_count = 0
        for item in os.listdir(kb_dir):
            item_path = os.path.join(kb_dir, item)
            if os.path.isdir(item_path):
                # 检查目录中是否有与目录同名的文件
                expected_file = os.path.join(item_path, item)
                if os.path.isfile(expected_file):
                    file_count += 1
                    
        return file_count
    except Exception as e:
        logger.error(f"获取知识库 {kb_id} 文件数量错误: {str(e)}")
        return 0

def get_files_for_knowledge_base(kb_id):
    """获取指定知识库的所有文件信息"""
    try:
        kb_name = get_knowledge_base_name(kb_id)
        kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
        
        if not os.path.exists(kb_dir):
            return []
            
        files = []
        for item in os.listdir(kb_dir):
            item_path = os.path.join(kb_dir, item)
            if os.path.isdir(item_path):
                # 检查目录中是否有与目录同名的文件
                expected_file = os.path.join(item_path, item)
                if os.path.isfile(expected_file):
                    file_stat = os.stat(expected_file)
                    files.append({
                        'name': item,
                        'size': file_stat.st_size,
                        'uploadDate': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'path': expected_file,
                        'parsed': False  # 可以根据需要添加解析状态文件来跟踪
                    })
                    
        return files
    except Exception as e:
        logger.error(f"获取知识库 {kb_id} 文件列表错误: {str(e)}")
        return []

def get_dataset_type_from_json(kb_id):
    """从knowledge_bases_data.json文件中获取知识库类型"""
    try:
        # 构建knowledge_bases_data.json文件的路径
        json_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_bases_data.json")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 查找匹配的知识库
        for kb in data.get("knowledge_bases", []):
            if str(kb["id"]) == str(kb_id):
                return kb["文件类型"]  # 返回 "text" 或 "graph"
        
        # 如果没有找到，返回默认类型
        return "text"
    except Exception as e:
        logger.error(f"从JSON文件获取数据集类型失败: {str(e)}")
        return "text"  # 出错时返回默认类型

@bp.route('/api/knowledge_bases/<kb_id>/files', methods=['GET'])
def get_knowledge_base_files(kb_id):
    update_all_knowledge_bases_file_count()
    """获取指定知识库的文件列表"""
    try:
        files = get_files_for_knowledge_base(kb_id)
        return jsonify({
            'success': True,
            'data': files
        })
    except Exception as e:
        logger.error(f"获取知识库文件列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



def register_socket_events(socketio):
    """注册WebSocket事件处理"""
    #@socketio.on('delete_file', namespace='/manage')
    #def handle_delete_file(data):
    #     """处理文件删除请求"""
    #     try:
    #         kb_id = data.get('kb_id')
    #         file_name = data.get('file_name')
        
    #         if not kb_id or not file_name:
    #             emit('delete_file_error', {
    #                 'error': 'Missing kb_id or file_name'
    #             })
    #             return
        
    #         logger.info(f"删除文件请求: 知识库 {kb_id}, 文件 {file_name}")
        
    #         # 获取知识库名称
    #         kb_name = get_knowledge_base_name(kb_id)
        
    #         # 构建文件目录路径 - 根据您的描述，路径结构应该是: UPLOAD_BASE_DIR/知识库名称/文件名称/
    #         kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
    #         file_dir = os.path.join(kb_dir, file_name)  # 要删除的文件夹
        
    #         logger.info(f"尝试删除文件夹: {file_dir}")
        
    #         if not os.path.exists(file_dir):
    #             logger.warning(f"文件夹不存在: {file_dir}")
    #             emit('delete_file_error', {
    #                 'error': f'File directory not found: {file_dir}'
    #             })
    #             return
        
    #     # 检查文件夹内是否有对应的文件
    #         expected_file_path = os.path.join(file_dir, file_name)
    #         if not os.path.exists(expected_file_path):
    #             logger.warning(f"文件不存在: {expected_file_path}")
        
    #     # 删除整个文件夹及其内容
    #         try:
    #             import shutil
    #             shutil.rmtree(file_dir)
    #             logger.info(f"成功删除文件夹: {file_dir}")
            
    #         # 发送删除完成消息
    #             emit('delete_file_complete', {
    #                 'success': True,
    #                 'kb_id': kb_id,
    #                 'file_name': file_name,
    #                 'message': f'File {file_name} deleted successfully'
    #             })
            
    #         except Exception as e:
    #             logger.error(f"删除文件夹失败: {str(e)}")
    #             emit('delete_file_error', {
    #                 'error': f'Failed to delete directory: {str(e)}'
    #             })
            
    #     except Exception as e:
    #         logger.error(f"删除文件处理错误: {str(e)}")
    #         emit('delete_file_error', {
    #             'error': str(e)
    #         }) 

    @socketio.on('connect', namespace='/manage')
    def handle_connect():
        logger.info("客户端连接到文件管理WebSocket")
        emit('connection_status', {'status': 'connected'})
    
    @socketio.on('disconnect', namespace='/manage')
    def handle_disconnect():
        logger.info("客户端断开文件管理WebSocket连接")
    
    @socketio.on('file_upload_start', namespace='/manage')
    def handle_file_upload_start(data):
        """开始文件上传"""
        try:
            kb_id = data.get('kb_id')
            file_name = data.get('file_name')
            file_size = data.get('file_size')
            file_type = data.get('file_type')
            graph_info = data.get('graph_info')  # 图数据的额外信息
            
            logger.info(f"开始上传文件: {file_name}, 大小: {file_size}, 知识库: {kb_id}")
            
            # 验证图数据的必填字段
            if file_type == 'graph' and graph_info:
                required_fields = ['graph_name', 'edge_filename', 'edge_source_field', 'edge_target_field']
                missing_fields = [field for field in required_fields if not graph_info.get(field)]
                if missing_fields:
                    emit('upload_error', {
                        'file_name': file_name,
                        'error': f'图数据缺少必填字段: {", ".join(missing_fields)}'
                    })
                    return
            
            # 获取知识库名称
            kb_name = get_knowledge_base_name(kb_id)
            
            # 创建目录结构: UPLOAD_BASE_DIR/知识库名称/文件名称/
            kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
            file_dir = os.path.join(kb_dir, file_name)  # 为每个文件创建单独目录
            os.makedirs(file_dir, exist_ok=True)
            
            file_path = os.path.join(file_dir, file_name)
            
            # 发送确认消息
            emit('upload_ready', {
                'file_name': file_name,
                'file_path': file_path
            })
            
        except Exception as e:
            logger.error(f"文件上传开始错误: {str(e)}")
            emit('upload_error', {
                'file_name': data.get('file_name', 'unknown'),
                'error': str(e)
            })
    
    @socketio.on('file_chunk', namespace='/manage')
    def handle_file_chunk(data):
        """处理文件分块上传"""
        try:
            file_name = data.get('file_name')
            chunk_data = data.get('chunk_data')
            chunk_index = data.get('chunk_index')
            total_chunks = data.get('total_chunks')
            kb_id = data.get('kb_id')
            
            # 获取知识库名称
            kb_name = get_knowledge_base_name(kb_id)
            
            # 创建目录结构: UPLOAD_BASE_DIR/知识库名称/文件名称/
            kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
            file_dir = os.path.join(kb_dir, file_name)
            os.makedirs(file_dir, exist_ok=True)
            
            file_path = os.path.join(file_dir, file_name)
            
            # 解码base64数据并写入文件
            if isinstance(chunk_data, str):
                # 如果是base64字符串，需要解码
                chunk_bytes = base64.b64decode(chunk_data)
            else:
                chunk_bytes = chunk_data
            
            # 写入文件块
            with open(file_path, 'ab' if chunk_index > 0 else 'wb') as f:
                f.write(chunk_bytes)
            
            # 计算进度
            progress = int((chunk_index + 1) / total_chunks * 100)
            
            emit('upload_progress', {
                'file_name': file_name,
                'progress': progress,
                'chunk_index': chunk_index,
                'total_chunks': total_chunks
            })
            
            # 如果是最后一个块，完成上传
            if chunk_index == total_chunks - 1:
                file_size = os.path.getsize(file_path)
                emit('upload_complete', {
                    'file_name': file_name,
                    'file_path': file_path,
                    'file_size': file_size,
                    'upload_time': datetime.now().isoformat()
                })
                logger.info(f"文件上传完成: {file_name}, 大小: {file_size} bytes, 路径: {file_path}")
                
        except Exception as e:
            logger.error(f"文件块上传错误: {str(e)}")
            emit('upload_error', {
                'file_name': data.get('file_name', 'unknown'),
                'error': str(e)
            })
    
    @socketio.on('file_upload_complete', namespace='/manage')
    def handle_upload_complete(data):
        """处理整个上传完成（所有文件）"""
        try:
            kb_id = data.get('kb_id')
            file_count = data.get('file_count')
            
            logger.info(f"知识库 {kb_id} 上传完成，共 {file_count} 个文件")
            
            emit('all_uploads_complete', {
                'kb_id': kb_id,
                'file_count': file_count,
                'message': f'成功上传 {file_count} 个文件'
            })
            
        except Exception as e:
            logger.error(f"上传完成处理错误: {str(e)}")
            emit('upload_error', {
                'file_name': 'all_files',
                'error': str(e)
            })

    # @socketio.on('parse_control', namespace='/manage')
    # def handle_parse_control(data):
    #     """处理文件解析控制请求 - 简化版"""
    #     try:
    #         kb_id = data.get('kb_id')
    #         file_name = data.get('file_name')
    #         action = data.get('action')  # 'start' 或 'pause'
            
    #         if not kb_id or not file_name or not action:
    #             emit('parse_error', {
    #                 'error': 'Missing kb_id, file_name or action'
    #             })
    #             return
            
    #         logger.info(f"解析控制请求: 知识库 {kb_id}, 文件 {file_name}, 操作 {action}")
            
    #         # 获取知识库名称
    #         kb_name = get_knowledge_base_name(kb_id)
            
    #         # 构建文件路径
    #         kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
    #         file_dir = os.path.join(kb_dir, file_name)
    #         file_path = os.path.join(file_dir, file_name)
            
    #         if not os.path.exists(file_path):
    #             emit('parse_error', {
    #                 'file_name': file_name,
    #                 'error': f'File not found: {file_path}'
    #             })
    #             return
            
    #         # ============================================
    #         # 在这里添加您的解析逻辑
    #         # 参数说明:
    #         # - kb_id: 知识库ID
    #         # - kb_name: 知识库名称  
    #         # - file_name: 文件名
    #         # - file_path: 文件完整路径
    #         # - action: 'start' 或 'pause'
    #         # ============================================
            
    #         # 示例：打印接收到的信息
    #         logger.info(f"=== 解析请求信息 ===")
    #         logger.info(f"知识库ID: {kb_id}")
    #         logger.info(f"知识库名称: {kb_name}")
    #         logger.info(f"文件名: {file_name}")
    #         logger.info(f"文件路径: {file_path}")
    #         logger.info(f"操作类型: {action}")
    #         logger.info(f"===================")
            
    #         # 这里可以调用您的 DocumentAPI 或其他处理逻辑
    #         # 例如：
    #         # from your_module import your_parse_function
    #         # your_parse_function(kb_id, kb_name, file_name, file_path, action)
            
    #         # 发送确认消息回前端
    #         emit('parse_status', {
    #             'kb_id': kb_id,
    #             'file_name': file_name,
    #             'action': action,
    #             'message': f'解析请求已接收: {action} {file_name}'
    #         })
            
    #     except Exception as e:
    #         logger.error(f"解析控制处理错误: {str(e)}")
    #         emit('parse_error', {
    #             'file_name': data.get('file_name', 'unknown'),
    #             'error': str(e)
    #         })


# API路由保持不变
@bp.route('/api/knowledge_bases1', methods=['GET'])
def get_knowledge_bases1():
    update_all_knowledge_bases_file_count()
    """获取知识库列表（从JSON文件读取文件个数）"""
    try:
        # 构建knowledge_bases_data.json文件的路径
        json_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_bases_data.json")
        logger.info('############',json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        knowledge_bases_with_count = []
        for kb in data.get("knowledge_bases", []):
            kb_with_count = kb.copy()
            # 确保字段名一致
            if "文件个数" in kb_with_count:
                kb_with_count["文档个数"] = kb_with_count["文件个数"]
            elif "文档个数" in kb_with_count:
                kb_with_count["文件个数"] = kb_with_count["文档个数"]
            else:
                kb_with_count["文档个数"] = 0
                kb_with_count["文件个数"] = 0
                
            knowledge_bases_with_count.append(kb_with_count)
            
        return jsonify({
            'success': True,
            'data': knowledge_bases_with_count
        })
    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/api/dataset_type', methods=['GET'])
def get_dataset_type():
    """获取数据集类型 - 从JSON文件动态获取"""
    kb_id = request.args.get('kb_id')
    
    if not kb_id:
        return jsonify({
            'success': False,
            'error': 'Missing kb_id parameter'
        }), 400
    
    try:
        # 从JSON文件获取真实的数据集类型
        dataset_type = get_dataset_type_from_json(kb_id)
        
        return jsonify({
            'success': True,
            'dataset_type': dataset_type
        })
    except Exception as e:
        logger.error(f"获取数据集类型失败: {str(e)}")
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
        
        logger.info(f"=== HTTP删除文件请求 ===")
        logger.info(f"知识库ID: {kb_id}")
        logger.info(f"文件名: {file_name}")
        
        if not kb_id or not file_name:
            return jsonify({
                'success': False,
                'error': 'Missing kb_id or file_name'
            }), 400
        
        # 获取知识库名称
        kb_name = get_knowledge_base_name(kb_id)
        
        # 构建文件目录路径
        kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
        file_dir = os.path.join(kb_dir, file_name)
        
        logger.info(f"尝试删除文件夹: {file_dir}")
        
        if not os.path.exists(file_dir):
            logger.warning(f"文件夹不存在: {file_dir}")
            return jsonify({
                'success': False,
                'error': f'File directory not found: {file_dir}'
            }), 404
        
        # 检查文件夹内是否有对应的文件
        expected_file_path = os.path.join(file_dir, file_name)
        if not os.path.exists(expected_file_path):
            logger.warning(f"文件不存在: {expected_file_path}")
        
        # 删除整个文件夹及其内容
        try:
            import shutil
            shutil.rmtree(file_dir)
            name_without_ext = file_name.rsplit('.', 1)[0]
            socketserver2 = DummySocket([json.dumps(
            {"action":"delete_kb",
            "graph_name":name_without_ext,
            "db_name":kb_name}

        )])
            asyncio.run(server_Test.handler(socketserver2))
            logger.info(f"成功删除文件夹: {file_dir}")
            
            return jsonify({
                'success': True,
                'kb_id': kb_id,
                'file_name': file_name,
                'message': f'File {file_name} deleted successfully'
            })
            
        except Exception as e:
            logger.error(f"删除文件夹失败: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to delete directory: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"删除文件处理错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/api/parse_control', methods=['GET'])
def api_parse_control():
    """HTTP API: 文件解析控制"""
    try:
        kb_id = request.args.get('kb_id')
        file_name = request.args.get('file_name')
        action = request.args.get('action')  # 'start' 或 'pause'
        
        logger.info(f"=== HTTP解析控制请求 ===")
        logger.info(f"知识库ID: {kb_id}")
        logger.info(f"文件名: {file_name}")
        logger.info(f"操作类型: {action}")
        
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
        print(name_without_ext)
        #a1=(i for i in )

        socketserver = DummySocket([json.dumps(
            {"action":"create_kb",
            "file_path":file_path,
            "graph_name":name_without_ext,
            "db_name":kb_name}

        )])
        asyncio.run(server_Test.handler(socketserver))

        # ============================================
        # 在这里添加您的解析逻辑
        # 参数说明:
        # - kb_id: 知识库ID
        # - kb_name: 知识库名称  
        # - file_name: 文件名
        # - file_path: 文件完整路径
        # - action: 'start' 或 'pause'
        # ============================================
        
        # 示例：打印接收到的信息
        logger.info(f"=== 解析请求信息 ===")
        logger.info(f"知识库ID: {kb_id}")
        logger.info(f"知识库名称: {kb_name}")
        logger.info(f"文件名: {file_name}")
        logger.info(f"文件路径: {file_path}")
        logger.info(f"操作类型: {action}")
        logger.info(f"===================")
        
        # 这里可以调用您的 DocumentAPI 或其他处理逻辑
        # 例如：
        # from your_module import your_parse_function
        # your_parse_function(kb_id, kb_name, file_name, file_path, action)
        
        # 返回成功响应
        return jsonify({
            'success': True,
            'kb_id': kb_id,
            'file_name': file_name,
            'action': action,
            'message': f'解析请求已接收: {action} {file_name}'
        })
        
    except Exception as e:
        logger.error(f"解析控制处理错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/api/generate_graph', methods=['POST'])
def api_generate_graph():
    """HTTP API: 生成知识图谱"""
    try:
        data = request.get_json()
        kb_id = data.get('kb_id')
        kb_name = data.get('kb_name')
        files = data.get('files', [])
        
        logger.info(f"=== 生成图谱请求 ===")
        logger.info(f"知识库ID: {kb_id}")
        logger.info(f"知识库名称: {kb_name}")
        logger.info(f"文件列表: {files}")
        graph_name1 = files[0].split('.')[0]
        logger.info(graph_name1)
        if not kb_id:
            return jsonify({
                'success': False,
                'error': 'Missing kb_id'
            }), 400
        
        # 首先检查所有文件是否已解析
        parsing_status = get_all_files_parsed_status(kb_id, files)
        if not parsing_status['all_parsed']:
            unparsed_files = [f for f, status in parsing_status['file_status'].items() if not status]
            return jsonify({
                'success': False,
                'error': f'以下文件尚未解析: {", ".join(unparsed_files)}',
                'unparsed_files': unparsed_files
            }), 400
        
        # 获取知识库名称
        kb_real_name = get_knowledge_base_name(kb_id)
        
        # ============================================
        # 在这里调用您的三元组生成逻辑
        # ============================================
        
        logger.info(f"=== 图谱生成请求信息 ===")
        logger.info(f"知识库ID: {kb_id}")
        logger.info(f"知识库显示名称: {kb_name}")
        logger.info(f"知识库实际名称: {kb_real_name}")
        logger.info(f"文件数量: {len(files)}")
        for file in files:
            logger.info(f"  - {file}")
        logger.info(f"===================")
        
        # ============================================
        # 这里是您需要实现的三元组生成逻辑
        # 请替换下面的示例代码
        # ============================================
        
       
         # 示例三元组数据 - 请替换为您的实际逻辑
        msg12 = [json.dumps({"action": "get_triplets", 
                    "graph_name": graph_name1,
                    "db_name": "debug_file"})]
        a33 = DummySocket(msg12)
        asyncio.run(server_Test.handler(a33))
        triplets = server_Test.triplets111
        # ============================================
        # 三元组生成逻辑结束
        # ============================================
        
        # 返回成功响应
        return jsonify({
            'success': True,
            'kb_id': kb_id,
            'kb_name': kb_name,
            'file_count': len(files),
            'triplets': triplets,
            'message': f'成功生成知识图谱，包含 {len(triplets)} 个三元组'
        })
        
    except Exception as e:
        logger.error(f"生成图谱处理错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

#async def main():
    #from aag.api.DocumentAPI import server_Test



#asyncio.run(main())
