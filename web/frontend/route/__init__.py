import logging
import os
import atexit
import sys

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aag.api.async_runtime import start_async_runtime, stop_async_runtime


# Global logging configuration (set once)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

socketio = SocketIO(cors_allowed_origins="*")


def create_app():
    # Current file directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "..", "app"),
        static_folder=os.path.join(base_dir, "..", "static"),
    )

    CORS(app)

    # ====== Register blueprints (route modules) ======
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

    # ====== Initialize SocketIO and async runtime ======
    socketio.init_app(app)
    start_async_runtime()
    atexit.register(stop_async_runtime)
 
    # ====== Register WebSocket event handlers ======
    try:
        from . import sockets_chat  
        logger.info("WebSocket event module sockets_chat loaded")
    except ImportError as e:
        logger.warning(f"Failed to import sockets_chat: {e}")

    return app