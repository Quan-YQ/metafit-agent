"""从本地 JSON 文件演示 R1 校验，不提供网络服务或持久化。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .training_events import validate_training_event


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("non-standard JSON constant")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验一条模拟训练事件")
    parser.add_argument("path", type=Path, help="UTF-8 JSON 文件路径")
    args = parser.parse_args()
    try:
        data = json.loads(args.path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (OSError, UnicodeError, ValueError):
        print("无法读取有效的 UTF-8 JSON 文件。", file=sys.stderr)
        return 2

    result = validate_training_event(data)
    if result.event is None:
        print(json.dumps({"ok": False, "errors": [asdict(error) for error in result.errors]}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"ok": True, "event": result.event.to_dict(),
                      "timestamp_utc": result.event.timestamp_utc}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
