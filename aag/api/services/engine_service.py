"""
Engine management service: manages AAG engine instance(s) with a singleton.
"""

import logging
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from aag.engine.aag_engine import AAGEngine
from aag.config.engine_config import load_config_from_yaml
from aag.utils.path_utils import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


class EngineService:
    """
    AAG engine management service (singleton); manages engine lifecycle.
    """
    
    _instance: Optional['EngineService'] = None
    _engine: Optional[AAGEngine] = None
    _initialized: bool = False
    
    def __init__(self):
        """Private constructor; use get_instance() to obtain the instance."""
        if EngineService._instance is not None:
            raise RuntimeError("EngineService is a singleton. Use get_instance() to get the instance.")
        self._config_path = DEFAULT_CONFIG_PATH
    
    @classmethod
    def get_instance(cls) -> 'EngineService':
        """Return the engine service singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_engine(self) -> AAGEngine:
        """Get or initialize the AAG engine."""
        if self._engine is None or not self._initialized:
            try:
                logger.info(f"Initializing AAG engine; config: {self._config_path}")
                if not self._config_path.exists():
                    raise FileNotFoundError(f"Config file not found: {self._config_path}")
                config = load_config_from_yaml(str(self._config_path))
                self._engine = AAGEngine(config)
                self._initialized = True
                logger.info("AAG engine initialized")
            except Exception as e:
                logger.error(f"AAG engine initialization failed: {str(e)}")
                raise
        
        return self._engine
    
    def set_config_path(self, config_path: str):
        """Set config file path (call before initializing)."""
        from pathlib import Path
        self._config_path = Path(config_path)
        if self._initialized:
            logger.warning("Config path changed; engine will need re-initialization")
            self._initialized = False
            self._engine = None
    
    async def shutdown(self):
        """Shut down the engine and release resources."""
        if self._engine is not None:
            try:
                logger.info("Shutting down AAG engine...")
                await self._engine.shutdown()
                self._engine = None
                self._initialized = False
                logger.info("AAG engine shut down")
            except Exception as e:
                logger.error(f"Error shutting down AAG engine: {str(e)}")
                raise
    
    def is_initialized(self) -> bool:
        """Return True if the engine is initialized."""
        return self._initialized and self._engine is not None
    
    def reset(self):
        """Reset the engine (for tests or re-initialization)."""
        self._engine = None
        self._initialized = False
        logger.info("Engine service reset")

