"""
模型服务
提供模型列表和配置信息
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelService:
    """
    模型服务 - 提供模型列表和配置
    """
    
    def __init__(self):
        """初始化模型服务"""
        self._models = [
            {"id": 1, "name": "GPT 4", "description": "OpenAI的GPT-4模型"},
            {"id": 2, "name": "Qwen 14B", "description": "通义千问14B参数模型"},
            {"id": 3, "name": "Qwen Plus", "description": "通义千问增强版模型"},
            {"id": 4, "name": "Llama 3", "description": "Meta的Llama 3开源模型"}
        ]
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        获取可用模型列表
        
        Returns:
            模型列表
        """
        return self._models.copy()
    
    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID获取模型信息
        
        Args:
            model_id: 模型ID
        
        Returns:
            模型信息字典，如果不存在返回None
        """
        for model in self._models:
            if model.get("id") == model_id:
                return model.copy()
        return None
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        根据名称获取模型信息
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型信息字典，如果不存在返回None
        """
        for model in self._models:
            if model.get("name") == model_name:
                return model.copy()
        return None
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型配置
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型配置字典
        """
        model = self.get_model_by_name(model_name)
        if model:
            return {
                "name": model["name"],
                "description": model["description"],
                # 可以从配置文件或数据库读取更多配置
                "config": {}
            }
        return {}
    
    def add_model(self, model: Dict[str, Any]) -> bool:
        """
        添加新模型（动态添加）
        
        Args:
            model: 模型信息字典，必须包含 id, name, description
        
        Returns:
            是否添加成功
        """
        try:
            # 检查是否已存在
            if self.get_model_by_id(model.get("id")):
                logger.warning(f"模型ID {model.get('id')} 已存在")
                return False
            
            self._models.append(model)
            logger.info(f"已添加模型: {model.get('name')}")
            return True
        except Exception as e:
            logger.error(f"添加模型失败: {str(e)}")
            return False
    
    def remove_model(self, model_id: int) -> bool:
        """
        移除模型
        
        Args:
            model_id: 模型ID
        
        Returns:
            是否移除成功
        """
        try:
            original_count = len(self._models)
            self._models = [m for m in self._models if m.get("id") != model_id]
            
            if len(self._models) < original_count:
                logger.info(f"已移除模型ID: {model_id}")
                return True
            else:
                logger.warning(f"模型ID {model_id} 不存在")
                return False
        except Exception as e:
            logger.error(f"移除模型失败: {str(e)}")
            return False

