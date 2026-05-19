"""测试 EngineService 单例 —— 线程安全与生命周期。

注意：EngineService.get_instance() 不需要参数（与原规范不同），
内部通过 set_config_path() 设置配置文件路径。
"""

import threading
from unittest.mock import patch

import pytest
from aag.api.services.engine_service import EngineService


class TestEngineServiceSingleton:
    """覆盖单例模式的核心属性：唯一实例、线程安全。"""

    def test_singleton_same_instance(self):
        """两次调用 get_instance() 应返回同一对象。"""
        # 由于类变量 _instance 可能被先前测试污染，先重置
        EngineService._instance = None
        EngineService._lock = threading.Lock()

        instance1 = EngineService.get_instance()
        instance2 = EngineService.get_instance()
        assert instance1 is instance2

    def test_thread_safety(self):
        """10 线程并发获取单例，所有线程必须返回同一实例。"""
        EngineService._instance = None
        EngineService._lock = threading.Lock()

        instances = []
        lock = threading.Lock()

        def get_instance():
            inst = EngineService.get_instance()
            with lock:
                instances.append(inst)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有线程应返回同一实例
        first = instances[0]
        for inst in instances[1:]:
            assert inst is first

    def test_direct_constructor_raises(self):
        """直接调用 EngineService() 构造器在单例已存在时应抛出 RuntimeError。"""
        EngineService._instance = None
        EngineService._lock = threading.Lock()

        # 正常获取单例
        EngineService.get_instance()
        # 再次直接构造应失败
        with pytest.raises(RuntimeError, match="singleton"):
            EngineService()

    def test_reset_clears_state(self):
        """reset() 应清除引擎引用并将 _initialized 置为 False。"""
        EngineService._instance = None
        EngineService._lock = threading.Lock()

        svc = EngineService.get_instance()
        # 直接设置内部状态模拟"已初始化"
        svc._engine = object()  # 用通用对象代替真实 AAGEngine
        svc._initialized = True

        svc.reset()
        assert svc._engine is None
        assert svc._initialized is False
        assert svc.is_initialized() is False

    def test_is_initialized_returns_false_when_none(self):
        """未设置 engine 时 is_initialized() 必须返回 False。"""
        EngineService._instance = None
        EngineService._lock = threading.Lock()

        svc = EngineService.get_instance()
        svc._engine = None
        svc._initialized = False
        assert svc.is_initialized() is False
