import logging
from flask import Blueprint, render_template

logger = logging.getLogger(__name__)

bp = Blueprint("pages", __name__)


@bp.route("/")
def index():
    """根路由：返回聊天页面"""
    return render_template("template-chatbot-s2-convo.html")


@bp.route("/overview")
def overview():
    return render_template("overview.html")


@bp.route("/documents")
def documents():
    return render_template("documents.html")


@bp.route("/manage_dataset")
def manage_dataset_page():
    return render_template("manage_dataset.html")

@bp.route("/model_manager")
def model_manager_page():
    return render_template("model-manager.html")