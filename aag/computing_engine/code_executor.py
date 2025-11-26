"""
Dynamic Code Executor
支持自动安装依赖的动态代码执行器
"""

import logging
import importlib
import subprocess
import sys
import re
from typing import Any, Optional
from aag.expert_search_engine.database.datatype import GraphData

logger = logging.getLogger(__name__)


class DynamicCodeExecutor:
    """支持自动安装依赖的动态代码执行器"""
    
    def __init__(self, timeout: int = 120, auto_install: bool = True):
        self.timeout = timeout
        self.auto_install = auto_install
        self.installed_packages = set()
    
    def extract_imports(self, code: str) -> list:
        """提取代码中的 import 语句"""
        imports = []
        pattern1 = r'^\s*import\s+([a-zA-Z0-9_]+)'
        pattern2 = r'^\s*from\s+([a-zA-Z0-9_]+)\s+import'
        
        for line in code.split('\n'):
            match1 = re.match(pattern1, line)
            match2 = re.match(pattern2, line)
            if match1:
                imports.append(match1.group(1))
            elif match2:
                imports.append(match2.group(1))
        
        return imports
    
    def install_package(self, package_name: str) -> bool:
        """安装 Python 包"""
        if package_name in self.installed_packages:
            logger.info(f"📦 {package_name} 已安装，跳过")
            return True
        
        # 包名映射
        package_mapping = {
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow',
            'yaml': 'pyyaml',
        }
        
        install_name = package_mapping.get(package_name, package_name)
        
        try:
            logger.info(f"📥 正在安装 {install_name}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', install_name],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {install_name} 安装成功")
                self.installed_packages.add(package_name)
                return True
            else:
                logger.error(f"❌ {install_name} 安装失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 安装 {install_name} 时出错: {e}")
            return False
    
    def check_and_install_dependencies(self, code: str) -> bool:
        """检查并安装代码依赖"""
        if not self.auto_install:
            return True
        
        imports = self.extract_imports(code)
        logger.info(f"🔍 检测到导入: {imports}")
        
        # 标准库列表（无需安装）
        stdlib = {'os', 'sys', 're', 'json', 'math', 'datetime', 
                  'time', 'random', 'itertools', 'functools', 
                  'collections', 'typing', 'pathlib', 'copy', 
                  'statistics', 'decimal', 'fractions'}
        
        for package in imports:
            if package in stdlib:
                continue
            
            try:
                importlib.import_module(package)
                logger.info(f"✅ {package} 已存在")
            except ImportError:
                logger.warning(f"⚠️  {package} 未安装，尝试安装...")
                if not self.install_package(package):
                    return False
        
        return True
    
    def execute(self, code: str, data: Any, global_graph: Optional[GraphData] = None) -> Any:
        """在独立命名空间中执行代码"""
        # 检查并安装依赖
        if not self.check_and_install_dependencies(code):
            raise RuntimeError("依赖安装失败")
        
        # 创建独立命名空间
        namespace = {
            "data": data,
            "global_graph": global_graph,  
            "__builtins__": __builtins__,  # 保留内置函数
        }
        
        try:
            # 执行代码
            exec(code, namespace)
            
            # 检查 process 函数
            if "process" not in namespace:
                raise ValueError("后处理代码必须定义 'process(data)' 函数")
            
            process_func = namespace["process"]
            
            if not callable(process_func):
                raise ValueError("'process' 必须是一个函数")
            
            # 执行处理函数
            result = process_func(data)
            logger.info(f"✅ 后处理执行成功")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 后处理失败: {e}", exc_info=True)
            raise RuntimeError(f"后处理代码执行错误: {e}")

