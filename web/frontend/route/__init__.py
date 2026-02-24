import logging
import os
import sys
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from aag.api.async_runtime import start_async_runtime, stop_async_runtime
# 日志配置（全局一次即可）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

socketio = SocketIO(cors_allowed_origins="*")


def create_app():
    # 当前文件目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "..", "app"),
        static_folder=os.path.join(base_dir, "..", "static"),
    )

    CORS(app)

    # ====== 注册各个蓝图（路由模块） ======
    from .routes_pages import bp as pages_bp      
    from .routes_health import bp as health_bp
    from .routes_documents import bp as documents_bp
    from .routes_manage_dataset import bp as manage_bp
    from .routes_models import bp as models_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(manage_bp)
    app.register_blueprint(models_bp)

    # ====== 初始化 SocketIO ======
    socketio.init_app(app)
 
    # ====== 注册 WebSocket 事件处理 ======
    try:
        from . import sockets_chat  
        logger.info("WebSocket 事件模块 sockets_chat 已加载")
    except ImportError as e:
        logger.warning(f"导入 sockets_chat 失败: {e}")

# ====== 注册文件管理 WebSocket 事件 ======
    try:
        from .routes_manage_dataset import register_socket_events
        register_socket_events(socketio)
        logger.info("文件管理 WebSocket 事件已注册")
    except Exception as e:
        logger.error(f"注册文件管理 WebSocket 事件失败: {str(e)}")

    return app
