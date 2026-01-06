import logging
from datetime import datetime

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

bp = Blueprint("health", __name__)


@bp.route("/api/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })
