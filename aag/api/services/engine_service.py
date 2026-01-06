"""
引擎管理服务
负责管理AAG引擎实例，提供单例模式
"""

import logging
import sys
import os
from typing import Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from aag.engine.aag_engine import AAGEngine
from aag.config.engine_config import load_config_from_yaml
from aag.utils.path_utils import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


class EngineService:
    """
    AAG引擎管理服务 - 单例模式
    负责管理引擎实例的生命周期
    """
    
    _instance: Optional['EngineService'] = None
    _engine: Optional[AAGEngine] = None
    _initialized: bool = False
    
    def __init__(self):
        """私有构造函数，通过 get_instance() 获取实例"""
        if EngineService._instance is not None:
            raise RuntimeError("EngineService is a singleton. Use get_instance() to get the instance.")
        self._config_path = DEFAULT_CONFIG_PATH
    
    @classmethod
    def get_instance(cls) -> 'EngineService':
        """
        获取引擎服务单例
        
        Returns:
            EngineService实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_engine(self) -> AAGEngine:
        """
        获取或初始化AAG引擎
        
        Returns:
            AAGEngine实例
        """
        if self._engine is None or not self._initialized:
            try:
                logger.info(f"正在初始化AAG引擎，配置文件: {self._config_path}")
                
                # 检查配置文件是否存在
                if not self._config_path.exists():
                    raise FileNotFoundError(f"配置文件不存在: {self._config_path}")
                
                # 从配置文件加载配置
                config = load_config_from_yaml(str(self._config_path))
                
                # 初始化引擎
                self._engine = AAGEngine(config)
                self._initialized = True
                
                logger.info("AAG引擎初始化完成")
            except Exception as e:
                logger.error(f"AAG引擎初始化失败: {str(e)}")
                raise
        
        return self._engine
    
    def set_config_path(self, config_path: str):
        """
        设置配置文件路径（在初始化前调用）
        
        Args:
            config_path: 配置文件路径
        """
        from pathlib import Path
        self._config_path = Path(config_path)
        # 如果已经初始化，需要重新初始化
        if self._initialized:
            logger.warning("配置文件路径已更改，需要重新初始化引擎")
            self._initialized = False
            self._engine = None
    
    async def shutdown(self):
        """
        关闭引擎，释放资源
        """
        if self._engine is not None:
            try:
                logger.info("正在关闭AAG引擎...")
                await self._engine.shutdown()
                self._engine = None
                self._initialized = False
                logger.info("AAG引擎已关闭")
            except Exception as e:
                logger.error(f"关闭AAG引擎时出错: {str(e)}")
                raise
    
    def is_initialized(self) -> bool:
        """
        检查引擎是否已初始化
        
        Returns:
            True if initialized, False otherwise
        """
        return self._initialized and self._engine is not None
    
    def reset(self):
        """
        重置引擎（用于测试或重新初始化）
        """
        self._engine = None
        self._initialized = False
        logger.info("引擎服务已重置")

