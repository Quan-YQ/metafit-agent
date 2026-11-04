"""离线检查保存的模拟模型文本，回放 R2 结构解析，不重新请求模型。"""

import argparse
import json
from pathlib import Path
import sys

from .decision import _parse_candidate
from .demo_decision import _load_json, _utf8_output


def replay(record: dict) -> dict:
    if not isinstance(record, dict) or not isinstance(record.get("data"), dict):
        raise ValueError("invalid record")
    data = record["data"]
    if data.get("schema") == "r2-simulation-record-1":
        samples = [{"scenario": "single", "model_content_unreviewed": data.get("model_content_unreviewed")}]
    elif data.get("schema") == "r2-simulation-evaluation-1" and isinstance(data.get("evaluation"), dict):
        samples = data["evaluation"].get("scenarios")
        if not isinstance(samples, list):
            raise ValueError("invalid evaluation")
    else:
        raise ValueError("unsupported record schema")
    checks = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("invalid sample")
        content = sample.get("model_content_unreviewed")
        if content is None:
            checks.append({"scenario": sample.get("scenario"), "status": "no_model_content"})
            continue
        candidate, problem = _parse_candidate(content)
        checks.append({"scenario": sample.get("scenario"), "status": "structure_valid" if candidate else "structure_invalid",
                       "error_code": problem})
    return {"replay": "local_structure_only", "checks": checks,
            "notice": "不调用模型，不重放网络或安全流程；过滤可能改变记录，结构有效不代表证据或结论正确。"}


def main() -> int:
    parser = argparse.ArgumentParser(description="离线回放已保存模型内容的结构校验")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        result = replay(_load_json(args.path))
    except (OSError, UnicodeError, ValueError, TypeError):
        print("无法读取受支持的回放文件。", file=sys.stderr)
        return 2
    _utf8_output()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if any(item["status"] == "structure_invalid" for item in result["checks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
