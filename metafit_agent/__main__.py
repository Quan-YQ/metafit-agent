"""本地启动入口：只验证 R0 工程与配置可运行。"""

import logging
import os
import sys


def main() -> int:
    level = os.environ.get("METAFIT_LOG_LEVEL", "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        print("配置错误：METAFIT_LOG_LEVEL 必须是 DEBUG、INFO、WARNING 或 ERROR。", file=sys.stderr)
        return 2

    logging.basicConfig(level=level)
    print(f"MetaFit Agent R0 工程入口已启动（日志级别：{level}）。")
    print("本入口检查工程配置；启动本机模拟接口：python -m metafit_agent.simulation_api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
