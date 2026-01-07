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
from DummySocket import DummySocket
sys.path.append("../../")
from aag.api.DocumentAPI import server_Test

class MessageCollectingSocket:
    """包装DummySocket，收集所有消息，优先返回error消息"""
    def __init__(self, message):
        self.socket = DummySocket(message)
        self.messages = []
        self.has_error = False
        self.error_message = None
        
    async def send(self, msg):
        # 调用原始socket的send方法
        await self.socket.send(msg)
        # 收集消息
        self.messages.append(msg)
        # 如果是error消息，记录下来
        try:
            parsed = json.loads(msg)
            if parsed.get("type") == "error":
                self.has_error = True
                self.error_message = msg
        except:
            pass
            
    @property
    def returnmsg(self):
        # 如果有错误，返回错误消息；否则返回最后一个消息
        if self.has_error and self.error_message:
            return self.error_message
        elif self.messages:
            return self.messages[-1]
        else:
            return self.socket.returnmsg if hasattr(self.socket, 'returnmsg') else None
    
    def __getattr__(self, name):
        # 代理其他属性到原始socket
        return getattr(self.socket, name)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

bp = Blueprint("manage", __name__)

# 确保上传目录存在 - 修改为前一个目录下的debug/files
UPLOAD_BASE_DIR = "./../../../aag/datasets"
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)


# 添加新的API端点来检查解析状态
@bp.route('/api/check_parsing_status', methods=['POST'])
def api_check_parsing_status():
    logger.info("检查解析状态请求#############")
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
        kb_name = get_knowledge_base_name(kb_id)
        kb_type = get_dataset_type_from_back(kb_id)
        # 获取解析状态
        a24 = DummySocket(json.dumps({"action": "get_parsing_status","ds_name":kb_name}))
        asyncio.run(server_Test.handler(a24))
        acps = a24.returnmsg
        acpss = json.loads(acps)
        logger.info("##### %s",acpss)
        if kb_type == 'graph':
            a25 = DummySocket(json.dumps({"action": "get_dataset_schema","ds_name":kb_name}))
            asyncio.run(server_Test.handler(a25))
            acps5 = a25.returnmsg
            acpss5 = json.loads(acps5)
            if acpss5["content"]["data"][0].get("vertex_file",None) is not None:
                acpss["content"]["data"]["parsing_status"][acpss5["content"]["data"][0]["vertex_file"].split("/")[-1]] = "completed"
        logger.info("###解析状态检查###%s",acpss)
        return jsonify({
            'success': True,
            'kb_id': kb_id,
            'all_parsed': acpss["content"]["data"]["total_parsing_status"],
            'file_status': acpss["content"]["data"]["parsing_status"],
            #'dataset_type': parsing_status['dataset_type'],
            #'message': f"解析状态: {len(files)}个文件中{sum(parsing_status['file_status'].values())}个已解析"
        })
        
    except Exception as e:
        logger.error(f"检查解析状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
    return basescy["content"]["data"]
    
def format_string_to_list(input_str):
    """
    将逗号分隔的字符串转换为列表格式的字符串
    
    参数:
    input_str: 逗号分隔的字符串，例如 "name,age"
    
    返回:
    字符串格式的列表表示，例如 "['name','age']"
    """
    # 分割字符串并去除每个元素两端的空格
    items = [item.strip() for item in input_str.split(',') if item.strip()]
    
    # 构建格式化的字符串
    formatted_items = [f"'{item}'" for item in items]
    result = f"[{','.join(formatted_items)}]"
    
    return result


def get_knowledge_base_name(kb_id):
    """根据知识库ID获取知识库名称"""
    try:
        # 兼容 "kb_6" 这类前缀
        if isinstance(kb_id, str) and kb_id.startswith("kb_"):
            kb_id = kb_id[3:]
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

def get_dataset_type_from_back(kb_id):
    """根据知识库ID获取知识库类型"""
    try:
        if isinstance(kb_id, str) and kb_id.startswith("kb_"):
            kb_id = kb_id[3:]
        kb_id = int(kb_id)  
        knowledge_bases = load_knowledge_bases() 
        for kb in knowledge_bases:
            if kb["id"] == kb_id:
                return kb["文件类型"]
                logger.info(f"文件类型获取成功：{kb}")
        return f"kb_{kb_id}"  
    except Exception as e:
        logger.error(f"获取文件类型错误: {str(e)}")
        return f"kb_{kb_id}"

def get_files_for_knowledge_base(kb_id):
    """获取指定知识库的所有文件信息"""
    try:
        kb_name = get_knowledge_base_name(kb_id)
        logger.info("查找文本文档调用")
        a23 = DummySocket(json.dumps({"action": "get_dataset_schema","ds_name": kb_name}))
        asyncio.run(server_Test.handler(a23))
        logger.info("####### %s",a23.returnmsg)
        regffkb = json.loads(a23.returnmsg)
        reregffkb = regffkb["content"]["data"]
        logger.info("查找文本文档返回 %s",reregffkb)
        logger.info("查找文本文档返回类型 %s",type(reregffkb))
        if reregffkb[0]["type"] == "graph":
            reregffkb[0]["size"] = reregffkb[0]["edge_size"]
            if reregffkb[0].get("vertex_file",None) is not None:
                reregffkb.append( {
                    "id" : 2,
                    "type" : "graph",
                    "name" : reregffkb[0]["vertex_file"].split("/")[-1],
                    "size" : reregffkb[0]["vertex_size"],
                    "uploadDate" : reregffkb[0]["uploadDate"],
                    "graph_status" : "completed",
                })
            else:
                pass
        logger.info("查找文本文档返回修改之后 %s",reregffkb)
        #return files
        return reregffkb
    except Exception as e:
        logger.error(f"获取知识库 {kb_id} 文件列表错误: {str(e)}")
        return []

@bp.route('/api/knowledge_bases/<kb_id>/files', methods=['GET'])
def get_knowledge_base_files(kb_id):
    """获取指定知识库的文件列表"""
    logger.info("收到查询文件的请求 %s", get_knowledge_base_name(kb_id))
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

@bp.route('/api/preview_file', methods=['GET'])
def api_preview_file():
    """获取文件内容用于预览"""
    try:
        kb_id = request.args.get('kb_id')
        file_name = request.args.get('file_name')
        
        logger.info(f"预览文件请求: kb_id={kb_id}, file_name={file_name}")
        
        if not kb_id or not file_name:
            return jsonify({
                'success': False,
                'error': 'Missing kb_id or file_name'
            }), 400
        
        # 获取知识库名称
        kb_name = get_knowledge_base_name(kb_id)
        kb_type = get_dataset_type_from_back(kb_id)
        
        # 获取文件路径
        file_path = None
        if kb_type == 'text':
            # 文本数据集：从schema中获取original_path
            server_Test.load_each_dataset(kb_name)
            if file_name in server_Test.each_dataset:
                file_path = server_Test.each_dataset[file_name]["schema"].get("original_path")
        elif kb_type == 'graph':
            # 图数据集：从schema中获取original_path
            server_Test.load_each_dataset(kb_name)
            
            # 获取dataset_folder的绝对路径
            # server_Test.dataset_folder 是相对于 DocumentAPI.py 的路径
            # DocumentAPI.py 在 aag/api/ 目录下
            if not os.path.isabs(server_Test.dataset_folder):
                # 获取 DocumentAPI.py 的目录
                import inspect
                from aag.api.DocumentAPI import DocumentAPIServer
                api_file = inspect.getfile(DocumentAPIServer)
                api_dir = os.path.dirname(os.path.abspath(api_file))
                dataset_folder_abs = os.path.normpath(os.path.join(api_dir, server_Test.dataset_folder))
                logger.info(f"解析dataset_folder: {server_Test.dataset_folder} (相对路径)")
                logger.info(f"  - api_dir: {api_dir}")
                logger.info(f"  - dataset_folder_abs: {dataset_folder_abs}")
            else:
                dataset_folder_abs = server_Test.dataset_folder
                logger.info(f"使用绝对路径dataset_folder: {dataset_folder_abs}")
            
            # 图数据集可能有多个文件，需要遍历查找
            for dataset_name, dataset_info in server_Test.each_dataset.items():
                schema = dataset_info.get("schema", {})
                
                # 辅助函数：解析路径
                def resolve_file_path(original_path, target_file_name):
                    """解析文件路径"""
                    if not original_path:
                        return None
                    
                    # 检查文件名是否匹配
                    path_file_name = os.path.basename(original_path)
                    logger.info(f"文件名匹配检查: target={target_file_name}, path_file={path_file_name}, original_path={original_path}")
                    if target_file_name != path_file_name and target_file_name not in original_path:
                        logger.info(f"文件名不匹配，跳过")
                        return None
                    logger.info(f"文件名匹配，继续解析路径")
                    
                    # 解析路径：从original_path中提取data/之后的部分
                    if not os.path.isabs(original_path):
                        # original_path格式：../../aag/datasets/data/{kb_name}/graph/{file_name}
                        # 提取 data/ 之后的部分
                        if "data/" in original_path:
                            data_index = original_path.find("data/")
                            rel_path = original_path[data_index:]  # data/{kb_name}/graph/{file_name}
                            resolved_path = os.path.normpath(os.path.join(dataset_folder_abs, rel_path))
                            logger.info(f"路径解析: original_path={original_path}, rel_path={rel_path}, resolved_path={resolved_path}, exists={os.path.exists(resolved_path)}")
                        else:
                            # 如果没有data/，直接使用标准路径
                            resolved_path = os.path.join(dataset_folder_abs, "data", kb_name, "graph", target_file_name)
                            logger.info(f"使用标准路径: resolved_path={resolved_path}, exists={os.path.exists(resolved_path)}")
                    else:
                        resolved_path = original_path
                        logger.info(f"使用绝对路径: resolved_path={resolved_path}, exists={os.path.exists(resolved_path)}")
                    
                    # 检查路径是否存在
                    if os.path.exists(resolved_path):
                        logger.info(f"✓ 找到文件: {resolved_path}")
                        return resolved_path
                    else:
                        logger.warning(f"✗ 路径不存在: {resolved_path}")
                        logger.warning(f"  - original_path: {original_path}")
                        logger.warning(f"  - dataset_folder_abs: {dataset_folder_abs}")
                        logger.warning(f"  - target_file_name: {target_file_name}")
                        # 尝试列出目录内容以帮助调试
                        graph_dir = os.path.join(dataset_folder_abs, "data", kb_name, "graph")
                        if os.path.exists(graph_dir):
                            files_in_dir = os.listdir(graph_dir)
                            logger.warning(f"  - graph目录下的文件: {files_in_dir}")
                    return None
                
                # 先检查edge文件
                if "edge" in schema and len(schema["edge"]) > 0:
                    edge_original_path = schema["edge"][0].get("original_path", "")
                    logger.info(f"检查edge文件: file_name={file_name}, edge_original_path={edge_original_path}")
                    resolved_path = resolve_file_path(edge_original_path, file_name)
                    if resolved_path:
                        file_path = resolved_path
                        break
                
                # 再检查vertex文件（即使edge文件匹配但路径不存在，也要继续检查）
                if "vertex" in schema and len(schema["vertex"]) > 0:
                    vertex_original_path = schema["vertex"][0].get("original_path", "")
                    logger.info(f"检查vertex文件: file_name={file_name}, vertex_original_path={vertex_original_path}")
                    resolved_path = resolve_file_path(vertex_original_path, file_name)
                    if resolved_path:
                        file_path = resolved_path
                        break
        
        if not file_path or not os.path.exists(file_path):
            # 添加详细的错误信息用于调试
            logger.error(f"文件未找到: kb_id={kb_id}, kb_name={kb_name}, file_name={file_name}, file_path={file_path}")
            if kb_type == 'graph' and 'dataset_folder_abs' in locals():
                logger.error(f"dataset_folder_abs: {dataset_folder_abs}")
                # 列出可能的路径
                possible_path = os.path.join(dataset_folder_abs, "data", kb_name, "graph", file_name)
                logger.error(f"尝试的标准路径: {possible_path}, 存在: {os.path.exists(possible_path)}")
                # 列出graph目录下的所有文件
                graph_dir = os.path.join(dataset_folder_abs, "data", kb_name, "graph")
                if os.path.exists(graph_dir):
                    files_in_dir = os.listdir(graph_dir)
                    logger.error(f"graph目录下的文件: {files_in_dir}")
            return jsonify({
                'success': False,
                'error': f'File {file_name} not found. Path: {file_path}'
            }), 404
        
        # 获取文件扩展名
        file_extension = file_name.split('.').pop().lower()
        
        # 根据文件类型读取内容
        content = None
        content_type = 'text'
        
        if file_extension == 'csv':
            # CSV文件：读取为表格数据
            import csv
            try:
                rows = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # 获取表头
                    headers = reader.fieldnames
                    # 读取前100行（避免数据过大）
                    for i, row in enumerate(reader):
                        if i >= 100:
                            break
                        rows.append(row)
                
                content = {
                    'headers': headers,
                    'rows': rows,
                    'total_rows': sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1,  # 减去header行
                    'displayed_rows': len(rows)
                }
                content_type = 'csv'
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试其他编码
                try:
                    rows = []
                    with open(file_path, 'r', encoding='gbk') as f:
                        reader = csv.DictReader(f)
                        headers = reader.fieldnames
                        for i, row in enumerate(reader):
                            if i >= 100:
                                break
                            rows.append(row)
                    content = {
                        'headers': headers,
                        'rows': rows,
                        'total_rows': sum(1 for _ in open(file_path, 'r', encoding='gbk')) - 1,
                        'displayed_rows': len(rows)
                    }
                    content_type = 'csv'
                except Exception as e:
                    logger.error(f"读取CSV文件失败: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': f'Failed to read CSV file: {str(e)}'
                    }), 500
            except Exception as e:
                logger.error(f"读取CSV文件失败: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Failed to read CSV file: {str(e)}'
                }), 500
        elif file_extension == 'txt':
            # 直接读取文本文件
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试其他编码
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                except:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
        elif file_extension == 'pdf':
            # PDF文件：返回base64编码的内容，前端使用PDF.js显示
            try:
                with open(file_path, 'rb') as f:
                    pdf_content = f.read()
                    content = base64.b64encode(pdf_content).decode('utf-8')
                    content_type = 'pdf'
            except Exception as e:
                logger.error(f"读取PDF文件失败: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Failed to read PDF file: {str(e)}'
                }), 500
        elif file_extension == 'docx':
            # DOCX文档：使用mammoth转换为HTML格式，保持原有格式
            try:
                import mammoth
                with open(file_path, 'rb') as f:
                    result = mammoth.convert_to_html(f)
                    content = result.value
                    content_type = 'html'
                    # 如果有警告，记录但不影响显示
                    if result.messages:
                        logger.warning(f"DOCX转换警告: {result.messages}")
            except ImportError:
                # 如果mammoth未安装，尝试使用markitdown
                try:
                    from markitdown import MarkItDown
                    md = MarkItDown()
                    result = md.convert(file_path)
                    # markitdown可能返回markdown，需要转换为HTML
                    import markdown
                    if hasattr(result, 'html_content') and result.html_content:
                        content = result.html_content
                    else:
                        # 将markdown转换为HTML
                        content = markdown.markdown(result.text_content, extensions=['tables', 'fenced_code'])
                    content_type = 'html'
                except Exception as e:
                    logger.error(f"转换DOCX文档失败: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': f'Failed to convert DOCX document: {str(e)}. Please install mammoth: pip install mammoth'
                    }), 500
            except Exception as e:
                logger.error(f"转换DOCX文档失败: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Failed to convert DOCX document: {str(e)}'
                }), 500
        elif file_extension == 'doc':
            # DOC文档（旧格式）：尝试转换为文本或提示用户
            try:
                # 尝试使用markitdown转换
                from markitdown import MarkItDown
                md = MarkItDown()
                result = md.convert(file_path)
                # 将markdown转换为HTML
                import markdown
                if hasattr(result, 'html_content') and result.html_content:
                    content = result.html_content
                else:
                    content = markdown.markdown(result.text_content, extensions=['tables', 'fenced_code'])
                content_type = 'html'
            except Exception as e:
                logger.error(f"转换DOC文档失败: {str(e)}")
                # DOC格式较老，可能无法完美转换，返回文本内容
                return jsonify({
                    'success': False,
                    'error': f'DOC格式文件预览受限，建议转换为DOCX格式。转换错误: {str(e)}'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported file type: {file_extension}'
            }), 400
        
        return jsonify({
            'success': True,
            'content': content,
            'content_type': content_type,
            'file_name': file_name,
            'file_path': file_path
        })
        
    except Exception as e:
        logger.error(f"预览文件处理错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/api/upload_file', methods=['POST'])
def api_upload_file():
    """HTTP API: 文件上传"""
    # 用于跟踪已保存的文件，以便在发生错误时清理
    saved_files = []
    saved_file_path = None

    def is_duplicate_error(msg: str) -> bool:
        """检测后端返回的重复文件错误，避免误删已存在文件"""
        if not msg:
            return False
        lower_msg = str(msg).lower()
        return "already exists" in lower_msg or "已存在" in lower_msg
    try:
        # 检查是否为批量上传（图数据集两个文件的情况）
        is_batch_upload = request.form.get('is_batch_upload') == 'true'
        kb_id = request.form.get('kb_id')
        file_type = request.form.get('file_type')
        graph_info_json = request.form.get('graph_info')
        
        if not kb_id:
            return jsonify({
                'success': False,
                'error': 'Missing kb_id'
            }), 400
        
        logger.info("这是表单填写的数据%s和该数据的类型%s",graph_info_json, type(graph_info_json))
        
        # 获取知识库名称
        kb_name = get_knowledge_base_name(kb_id)
        
        # 创建目录结构
        file_dir = os.path.join("..","..", "aag", "datasets", "data" ,kb_name, file_type)
        os.makedirs(file_dir, exist_ok=True)
        
        if is_batch_upload and file_type == 'graph':
            # 批量上传模式：图数据集上传两个文件
            files = request.files.getlist('files')
            if not files or len(files) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Missing files'
                }), 400
            
            logger.info(f"=== HTTP批量文件上传请求 ===")
            logger.info(f"知识库ID: {kb_id}")
            logger.info(f"文件数量: {len(files)}")
            logger.info(f"文件类型: {file_type}")
            
            # 先保存所有文件
            for file in files:
                file_path = os.path.join(file_dir, file.filename)
                file.save(file_path)
                file_size = os.path.getsize(file_path)
                saved_files.append({
                    'filename': file.filename,
                    'path': file_path,
                    'size': file_size
                })
                logger.info(f"文件保存成功: {file_path}, 大小: {file_size} bytes")
            
            # 所有文件保存完成后，再调用DocumentAPI
            logger.info("开始修改schema")
            try:
                graph_info = json.loads(graph_info_json)
            except (json.JSONDecodeError, TypeError) as e:
                # JSON解析失败，删除已保存的文件
                for saved_file in saved_files:
                    if os.path.exists(saved_file['path']):
                        try:
                            os.remove(saved_file['path'])
                            logger.info(f"JSON解析失败：已删除文件: {saved_file['path']}")
                        except Exception as del_e:
                            logger.error(f"删除文件失败 {saved_file['path']}: {str(del_e)}")
                return jsonify({
                    'success': False,
                    'error': f'Invalid graph_info JSON: {str(e)}'
                }), 400
            logger.info("这是转换之后的info %s 和类型 %s",graph_info, type(graph_info))
            graph_name = graph_info.get("graphName")
            # 提取顶点文件相关字段
            vertex_file_name = graph_info["vertexSchema"].get("fileName", "")  # 顶点文件名（可选）
            vertex_id_field = graph_info["vertexSchema"].get("idField", "")  # 顶点ID字段（如果有顶点文件则必填）
            vertex_name_field = graph_info["vertexSchema"].get("nameField", "")
            vertex_name_field = vertex_id_field if not vertex_name_field else vertex_name_field # 顶点名称字段（如果有顶点文件则必填）
            
            # 提取边文件相关字段
            edge_file_name = graph_info["edgeSchema"]["fileName"]  # 边文件名（必填）
            edge_source_field = graph_info["edgeSchema"]["sourceField"]  # 源节点字段（必填）
            edge_target_field = graph_info["edgeSchema"]["targetField"]  # 目标节点字段（必填）
            edge_relation_field = graph_info["edgeSchema"].get("relationField", "")  # 关系字段（必填）
            weight_col = graph_info["edgeSchema"].get("weightField", "")  # 权重字段（可选）
            is_directed = graph_info["graphProperties"].get("isDirected", True)  # 是否为有向图

            # 验证必填字段
            if not edge_file_name or not edge_source_field or not edge_target_field:
                # 删除已保存的文件
                for saved_file in saved_files:
                    if os.path.exists(saved_file['path']):
                        try:
                            os.remove(saved_file['path'])
                            logger.info(f"验证失败：已删除文件: {saved_file['path']}")
                        except Exception as del_e:
                            logger.error(f"删除文件失败 {saved_file['path']}: {str(del_e)}")
                return jsonify({
                    'success': False,
                    'error': 'Edge file name, source field, target field, and are required for Graph dataset'
                }), 400
            
            # 如果有顶点文件，验证顶点文件相关字段
            if vertex_file_name and (not vertex_id_field or not vertex_name_field):
                # 删除已保存的文件
                for saved_file in saved_files:
                    if os.path.exists(saved_file['path']):
                        try:
                            os.remove(saved_file['path'])
                            logger.info(f"验证失败：已删除文件: {saved_file['path']}")
                        except Exception as del_e:
                            logger.error(f"删除文件失败 {saved_file['path']}: {str(del_e)}")
                return jsonify({
                    'success': False,
                    'error': 'If vertex file is provided, vertex ID field is required'
                }), 400
            
            vertex_name_field = format_string_to_list(vertex_name_field)
            edge_relation_field = format_string_to_list(edge_relation_field)
            logger.info("这是前端返回的结果%s和类型%s", graph_info, type(graph_info))
            msg111 = json.dumps({
                "action": "upload_file", 
                "file_name": edge_file_name, 
                "ds_name": kb_name,
                "vertex_file_name": vertex_file_name if vertex_file_name else None,  # 只有提供了文件名才发送
                "vertex_id_field": vertex_id_field if vertex_file_name else None,
                "vertex_name_field": vertex_name_field if vertex_file_name else None,
                "source_field": edge_source_field, 
                "target_field": edge_target_field,
                "relation_field": edge_relation_field,  # 使用 relation_field 而不是 edge_relation_field
                "is_directed": is_directed
                #"weight_field": weight_col if weight_col else None
            })
            logger.info("这是准备发给DocumentAPI的消息%s", msg111)
            a25 = MessageCollectingSocket(msg111)
            asyncio.run(server_Test.handler(a25))
            acps = a25.returnmsg
            logger.info("这是DocumentAPI返回的结果%s", acps)
            logger.info("收集到的所有消息: %s", a25.messages)
                # 如果有错误，优先使用错误消息
            if a25.has_error:
                try:
                    acps = json.loads(a25.error_message)
                    error_content = acps.get("content", "上传失败")
                except:
                    error_content = "上传失败，请检查文件格式和字段配置"
                # 非重复错误才清理文件，重复文件保持
                if not is_duplicate_error(error_content):
                    for saved_file in saved_files:
                        if os.path.exists(saved_file['path']):
                            try:
                                os.remove(saved_file['path'])
                                logger.info(f"已删除文件: {saved_file['path']}")
                            except Exception as e:
                                logger.error(f"删除文件失败 {saved_file['path']}: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': error_content
                }), 400
            acps = json.loads(acps)
            if acps["type"] == "error":
                error_content = acps.get("content", "上传失败")
                # 非重复错误才清理文件
                if not is_duplicate_error(error_content):
                    for saved_file in saved_files:
                        if os.path.exists(saved_file['path']):
                            try:
                                os.remove(saved_file['path'])
                                logger.info(f"已删除文件: {saved_file['path']}")
                            except Exception as e:
                                logger.error(f"删除文件失败 {saved_file['path']}: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': error_content
                }), 400
            
            # 返回所有文件的信息
            files_info = []
            for saved_file in saved_files:
                files_info.append({
                    'file_name': saved_file['filename'],
                    'file_path': saved_file['path'],
                    'file_size': saved_file['size'],
                    'upload_time': datetime.now().isoformat()
                })
            
            return jsonify({
                'success': True,
                'kb_id': kb_id,
                'files': files_info,
                'message': f'Files uploaded successfully'
            })
        else:
            # 单个文件上传模式（原有逻辑）
            file = request.files.get('file')
            if not file:
                return jsonify({
                    'success': False,
                    'error': 'Missing file'
                }), 400
            
            logger.info(f"=== HTTP文件上传请求 ===")
            logger.info(f"知识库ID: {kb_id}")
            logger.info(f"文件名: {file.filename}")
            logger.info(f"文件类型: {file_type}")
            
            # 保存文件
            file_path = os.path.join(file_dir, file.filename)
            file.save(file_path)
            saved_file_path = file_path  # 记录已保存的文件路径，用于错误清理
            
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            
            logger.info(f"文件保存成功: {file_path}, 大小: {file_size} bytes")
            logger.info("开始修改schema")
            if file_type == 'text':
                # 使用可捕获错误信息的socket
                a25 = MessageCollectingSocket(json.dumps({
                    "action": "upload_file",
                    "file_name": file.filename,
                    "ds_name": kb_name
                }))
                asyncio.run(server_Test.handler(a25))
                acps = a25.returnmsg
                logger.info("这是DocumentAPI返回的结果%s", acps)
                logger.info("收集到的所有消息: %s", a25.messages)
                # 如果有错误，优先使用错误消息
                if a25.has_error:
                    try:
                        acps = json.loads(a25.error_message)
                        error_content = acps.get("content", "上传失败")
                    except:
                        error_content = "上传失败，请检查文件格式和内容"
                # 非重复错误才删除文件；重复错误保留（覆盖的文件）
                    if not is_duplicate_error(error_content):
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                logger.info(f"已删除文件: {file_path}")
                            except Exception as e:
                                logger.error(f"删除文件失败 {file_path}: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': error_content
                    }), 400
                # 尝试解析返回结果并检查错误类型
                try:
                    acps_json = json.loads(acps)
                except Exception:
                    acps_json = {}
                if isinstance(acps_json, dict) and acps_json.get("type") == "error":
                    error_content = acps_json.get("content", "上传失败")
                    if not is_duplicate_error(error_content):
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                logger.info(f"已删除文件: {file_path}")
                            except Exception as e:
                                logger.error(f"删除文件失败 {file_path}: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': error_content
                    }), 400
            else:
                # 如果图数据上传，还需要传输配置文件
                try:
                    graph_info = json.loads(graph_info_json)
                except (json.JSONDecodeError, TypeError) as e:
                    # JSON解析失败，删除已保存的文件
                    if saved_file_path and os.path.exists(saved_file_path):
                        try:
                            os.remove(saved_file_path)
                            logger.info(f"JSON解析失败：已删除文件: {saved_file_path}")
                        except Exception as del_e:
                            logger.error(f"删除文件失败 {saved_file_path}: {str(del_e)}")
                    return jsonify({
                        'success': False,
                        'error': f'Invalid graph_info JSON: {str(e)}'
                    }), 400
                logger.info("这是转换之后的info %s 和类型 %s",graph_info, type(graph_info))
                graph_name = graph_info.get("graphName")
                # 提取顶点文件相关字段
                vertex_file_name = graph_info["vertexSchema"].get("fileName", "")  # 顶点文件名（可选）
                vertex_id_field = graph_info["vertexSchema"].get("idField", "")  # 顶点ID字段（如果有顶点文件则必填）
                vertex_name_field = graph_info["vertexSchema"].get("nameField", "")  # 顶点名称字段（如果有顶点文件则必填）
                
                # 提取边文件相关字段
                edge_file_name = graph_info["edgeSchema"]["fileName"]  # 边文件名（必填）
                edge_source_field = graph_info["edgeSchema"]["sourceField"]  # 源节点字段（必填）
                edge_target_field = graph_info["edgeSchema"]["targetField"]  # 目标节点字段（必填）
                edge_relation_field = graph_info["edgeSchema"].get("relationField", "")  # 关系字段（必填）
                weight_col = graph_info["edgeSchema"].get("weightField", "")  # 权重字段（可选）
                is_directed = graph_info["graphProperties"].get("isDirected", True)  # 是否为有向图

                # 验证必填字段
                if not edge_file_name or not edge_source_field or not edge_target_field:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({
                        'success': False,
                        'error': 'Edge file name, source field, target field, and are required for Graph dataset'
                    }), 400
                
                # 如果有顶点文件，验证顶点文件相关字段
                if vertex_file_name and (not vertex_id_field or not vertex_name_field):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({
                        'success': False,
                        'error': 'If vertex file is provided, vertex ID field and name field are required'
                    }), 400
                vertex_name_field = format_string_to_list(vertex_name_field)
                edge_relation_field = format_string_to_list(edge_relation_field)
                logger.info("这是前端返回的结果%s和类型%s", graph_info, type(graph_info))
                msg111 = json.dumps({
                    "action": "upload_file", 
                    "file_name": edge_file_name, 
                    "ds_name": kb_name,
                    "vertex_file_name": vertex_file_name if vertex_file_name else None,  # 只有提供了文件名才发送
                    "vertex_id_field": vertex_id_field if vertex_file_name else None,
                    "vertex_name_field": vertex_name_field if vertex_file_name else None,
                    "source_field": edge_source_field, 
                    "target_field": edge_target_field,
                    "relation_field": edge_relation_field,  # 使用 relation_field 而不是 edge_relation_field
                    "is_directed": is_directed,
                    #"weight_field": weight_col if weight_col else None
                })
                logger.info("这是准备发给DocumentAPI的消息%s", msg111)
                a25 = MessageCollectingSocket(msg111)
                asyncio.run(server_Test.handler(a25))
                acps = a25.returnmsg
                logger.info("这是DocumentAPI返回的结果%s", acps)
                logger.info("收集到的所有消息: %s", a25.messages)
                # 如果有错误，优先使用错误消息
                if a25.has_error:
                    try:
                        acps = json.loads(a25.error_message)
                        error_content = acps.get("content", "上传失败")
                    except:
                        error_content = "上传失败，请检查文件格式和字段配置"
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"已删除文件: {file_path}")
                        except Exception as e:
                            logger.error(f"删除文件失败 {file_path}: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': error_content
                    }), 400
                acps = json.loads(acps)
                if acps["type"] == "error":
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"已删除文件: {file_path}")
                        except Exception as e:
                            logger.error(f"删除文件失败 {file_path}: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': acps.get("content", "上传失败")
                    }), 400
            
            return jsonify({
                'success': True,
                'kb_id': kb_id,
                'file_name': file.filename,
                'file_path': file_path,
                'file_size': file_size,
                'upload_time': datetime.now().isoformat(),
                'message': f'File {file.filename} uploaded successfully'
            })
        
    except Exception as e:
        logger.error(f"文件上传处理错误: {str(e)}")
        # 清理已保存的文件
        # 批量上传的情况
        if saved_files:
            for saved_file in saved_files:
                if os.path.exists(saved_file['path']):
                    try:
                        os.remove(saved_file['path'])
                        logger.info(f"异常处理：已删除文件: {saved_file['path']}")
                    except Exception as del_e:
                        logger.error(f"异常处理：删除文件失败 {saved_file['path']}: {str(del_e)}")
        # 单个文件上传的情况
        elif saved_file_path and os.path.exists(saved_file_path):
            try:
                os.remove(saved_file_path)
                logger.info(f"异常处理：已删除文件: {saved_file_path}")
            except Exception as del_e:
                logger.error(f"异常处理：删除文件失败 {saved_file_path}: {str(del_e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def process_graph_file(kb_name, file_name, graph_info):
    """处理图数据文件"""
    try:
        # 构建文件路径
        file_dir = os.path.join(UPLOAD_BASE_DIR, kb_name, file_name)
        file_path = os.path.join(file_dir, file_name)
        
        # 这里调用你的图数据处理逻辑
        logger.info(f"处理图数据文件: {file_name} in dataset {kb_name}")
        
        # 示例：调用DocumentAPI处理图数据
        # 请根据你的实际需求实现这个函数
        # from your_module import process_graph_data
        # process_graph_data(kb_name, file_path, graph_info)
        
    except Exception as e:
        logger.error(f"处理图数据文件失败: {str(e)}")


def process_text_file(kb_name, file_name):
    """处理文本数据文件"""
    try:
        # 构建文件路径
        file_dir = os.path.join(UPLOAD_BASE_DIR, kb_name, file_name)
        file_path = os.path.join(file_dir, file_name)
        
        # 这里调用你的文本处理逻辑
        logger.info(f"处理文本文件: {file_name} in dataset {kb_name}")
        
        # 示例：调用DocumentAPI处理文本数据
        # 请根据你的实际需求实现这个函数
        # from your_module import process_text_data
        # process_text_data(kb_name, file_path)
        
    except Exception as e:
        logger.error(f"处理文本文件失败: {str(e)}")

# API路由保持不变
@bp.route('/api/knowledge_bases1', methods=['GET'])
def get_knowledge_bases1():
    """获取数据集列表（从JSON文件读取文件个数）"""
    try:
        a1 = DummySocket(json.dumps({"action": "get_datasets"}))
        asyncio.run(server_Test.handler(a1))
        gkb = a1.returnmsg
        knowledge_bases = json.loads(gkb)
        logger.info("####这是子涵页面里信息 %s",knowledge_bases)
        logger.info("####这是子涵页面里信息类型 %s",type(knowledge_bases))
        for kb in knowledge_bases["content"]["data"]:
            if kb.get("文件类型") == "graph" and kb.get("文档个数") == 1:
                try:
                    a6 = DummySocket(json.dumps({"action": "get_dataset_schema","ds_name":kb["名称"]}))
                    asyncio.run(server_Test.handler(a6))
                    gkb6 = a6.returnmsg
                    gkb6_json = json.loads(gkb6)
                    logger.info("####判断信息类型 %s",gkb6_json)
                    logger.info("####判断信息类型 %s",type(gkb6_json))
                    data_list = gkb6_json.get("content", {}).get("data", [])
                    if isinstance(data_list, list) and data_list and data_list[0].get("vertex_file",None) is not None:
                        kb["文档个数"] = 2
                except Exception as inner_e:
                    logger.warning(f"统计图文件数时出错，已忽略: {inner_e}")
        logger.info("这是return给网页的信息%s",len(knowledge_bases["content"]["data"]))
        return jsonify({
            "success": True,
            "data": knowledge_bases["content"]["data"],
            "count": len(knowledge_bases["content"]["data"])
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
        # kb_id 必须可转换为数字，否则直接返回 400，避免返回伪类型
        raw_kb_id = kb_id
        if isinstance(kb_id, str) and kb_id.startswith("kb_"):
            kb_id = kb_id[3:]
        kb_id_int = int(kb_id)
        
        # 从JSON文件获取真实的数据集类型
        dataset_type = get_dataset_type_from_back(kb_id_int)
        
        return jsonify({
            'success': True,
            'dataset_type': dataset_type
        })
    except Exception as e:
        logger.error(f"获取数据集类型失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Invalid kb_id: {raw_kb_id}'
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
        kb_type = get_dataset_type_from_back(kb_id)
        if kb_type == 'graph':
            a26 = DummySocket(json.dumps({"action": "get_dataset_schema","ds_name": kb_name}))
            asyncio.run(server_Test.handler(a26))
            acps = a26.returnmsg
            acps = json.loads(acps)
            file_name = acps["content"]["data"][0]["name"]
        try:
            logger.info("###### %s %s", file_name, kb_name)
            logger.info("删除可以进来这里")
            socketserver2 = DummySocket(json.dumps(
                {"action":"delete_file",
                "file_name":file_name,
                "ds_name":kb_name}
            ))
            logger.info("这是真删除成功了")
            asyncio.run(server_Test.handler(socketserver2))
        #     
        #     logger.info(f"成功删除文件夹: {file_dir}")
            
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
        
        # # 构建文件路径
        # kb_dir = os.path.join(UPLOAD_BASE_DIR, kb_name)
        # file_dir = os.path.join(kb_dir, file_name)
        # file_path = os.path.join(file_dir, file_name)
        # if not os.path.exists(file_path):
        #     return jsonify({
        #         'success': False,
        #         'error': f'File not found: {file_path}'
        #     }), 404
        # 使用 MessageCollectingSocket 捕获 DocumentAPI 返回的 error 消息
        socketserver = MessageCollectingSocket(json.dumps(
            {
                "action": "parsing_file",
                "file_name": file_name,
                "ds_name": kb_name
            }
        ))
        asyncio.run(server_Test.handler(socketserver))

        # 优先使用收集到的 error 消息
        raw_msg = socketserver.returnmsg
        try:
            parsed_msg = json.loads(raw_msg)
        except Exception:
            parsed_msg = {}
        logger.info("这是解析后的消息%s",parsed_msg)
        # 如果 DocumentAPI 返回的是 error，则把详细信息透传给前端
        if isinstance(parsed_msg, dict) and parsed_msg.get("type") == "error":
            a6 = DummySocket(json.dumps({"action": "schema_refine","ds_name": kb_name,"file_name": file_name}))
            asyncio.run(server_Test.handler(a6))
            #acps = a6.returnmsg
            #acps = json.loads(acps)
            #logger.info("这是schema_refine返回的结果%s",acps)
            # content 中已经包含“添加失败 ❌ ... 未找到 JSON 内容，原始响应: ...”
            error_content = parsed_msg.get("content") or "解析失败，请检查文件内容"
            logger.error(f"在数据集{kb_name}中解析文件{file_name}失败: {error_content}")
            return jsonify({
                "success": False,
                "error": error_content
            }), 400

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
        #logger.info(f"文件路径: {file_path}")
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
        dataset_type = get_dataset_type_from_back(kb_id)
        logger.info(f"=== 生成图谱请求 ===")
        logger.info(f"知识库ID: {kb_id}")
        logger.info(f"知识库名称: {kb_name}")
        logger.info(f"文件列表: {files}")
        logger.info(f"数据集类型: {dataset_type}")
        graph_name1 = files[0].split('.')[0]
        logger.info(graph_name1)
        if not kb_id:
            return jsonify({
                'success': False,
                'error': 'Missing kb_id'
            }), 400
        
        # 首先检查所有文件是否已解析
        # #parsing_status = get_all_files_parsed_status(kb_id, files)
        # if not parsing_status['all_parsed']:
        #     unparsed_files = [f for f, status in parsing_status['file_status'].items() if not status]
        #     return jsonify({
        #         'success': False,
        #         'error': f'以下文件尚未解析: {", ".join(unparsed_files)}',
        #         'unparsed_files': unparsed_files
        #     }), 400
        
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
        if dataset_type == 'text':
            is_directed = True
            msg12 = json.dumps({"action": "get_overall_triplets", 
                    #"graph_name": graph_name1,
                    "ds_name": kb_name})
            a33 = DummySocket(msg12)
            asyncio.run(server_Test.handler(a33))
            agg = a33.returnmsg
            agg1 = json.loads(agg)
            logger.info("####这里都可以的 %s", agg)
            triplets = agg1["content"]["data"]
            if len(triplets) > 500:
                triplets = triplets[:500]
        elif dataset_type == 'graph':
            msg12 = json.dumps({"action": "get_file_triplets_from_graph_dataset", 
                    "file_name": files[0],
                    "ds_name": kb_name})
            a33 = DummySocket(msg12)
            asyncio.run(server_Test.handler(a33))
            agg = a33.returnmsg
            agg1 = json.loads(agg)
            logger.info("####这里都可以的 %s", agg1)
            triplets = agg1["content"]["data"]["edges"]
            is_directed = agg1["content"]["data"].get("is_directed",True)
            if len(triplets) > 500:
                triplets = triplets[:500]
         # 示例三元组数据 - 请替换为您的实际逻辑
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
            'is_directed': is_directed,
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
