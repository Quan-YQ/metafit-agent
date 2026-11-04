"""将测试临时文件放在项目本地，兼容受限环境。"""
from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    if config.option.basetemp is None:
        # 每次使用未存在的新目录，避免 pytest 清理任何用户已有目录。
        parent = Path(__file__).resolve().parents[1] / ".tmp" / "pytest"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid4().hex)
