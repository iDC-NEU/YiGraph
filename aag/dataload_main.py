import os
import sys
import datetime

# 确保可从项目根目录导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aag.config.data_upload_config import load_data_upload_config
from aag.data_pipeline.data_transformer.dataset_manager import DatasetManager
from aag.utils.path_utils import DEFAULT_DATA_UPLOAD_CONFIG_PATH

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main() -> int:
    """用户上传数据加载入口。读取配置并调度各类型数据加载器。"""
    try:
        # 配置文件路径（固定为用户提供路径）
        config_path = DEFAULT_DATA_UPLOAD_CONFIG_PATH
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"用户数据上传配置未找到: {config_path}")

        print(f"{CYAN}{'=' * 74}{RESET}")
        print(f"{BOLD} 📦  数据上传加载器 (Data Upload Loader){RESET}")
        print(f"{CYAN}{'=' * 74}{RESET}")
        print(f" 配置文件: {config_path}")
        print(f" 启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 74)

        # 读取上传配置
        cfg = load_data_upload_config(config_path)
        if not cfg.datasets:
            print(f"{YELLOW}⚠ 未在配置中找到任何数据集条目{RESET}")
            return 0

        # 初始化 DatasetManager（使用默认 schema 路径）
        manager = DatasetManager()

        # 列出数据集摘要
        print(f"{BOLD}发现 {len(cfg.datasets)} 个数据集:{RESET}")
        by_type = {}
        for ds in cfg.datasets:
            by_type.setdefault(ds.type, []).append(ds.name)
        for t, names in by_type.items():
            print(f" - {t}: {', '.join(sorted(names))}")
        print("-" * 74)

        # 实际加载
        print("🚚 正在加载数据集...")
        manager.load_dataset(cfg)
        print(f"{GREEN}✓ 数据加载完成{RESET}")
        print(f"{CYAN}{'=' * 74}{RESET}")

        # 输出已有的 datasets（按类型分组）
        existing = manager.list_datasets()
        print(f"{BOLD}现有已注册数据集:{RESET}")
        for t, names in existing.items():
            print(f" - {t} ({len(names)}): {', '.join(names)}")

        return 0

    except Exception as e:
        print(f"{YELLOW}✗ 加载失败: {e}{RESET}")
        return 1


if __name__ == "__main__":
    exit(main())
