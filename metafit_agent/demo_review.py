"""R3 离线模拟演示，无真实模型、网络、持久化或投递。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .decision import UNKNOWN_REASON, decide_simulated_event
from .decision_review import review_outcome
from .demo_decision import _load_json, _utf8_output
from .model_response import ModelResponse


class DemoModel:
    def __init__(self, action: str):
        self.action = action

    def complete(self, facts):
        status = next(f.value for f in facts if f.name == "discomfort_status")
        completed = next((f for f in facts if f.name == "completed_reps"), None)
        reason = UNKNOWN_REASON if status == "unknown" else "已自报无不适，当前状态和报告时效未确认。"
        message = None if self.action == "silent" else (f"本组完成了{completed.value}次。" if completed else "模拟反馈。")
        evidence = [] if self.action == "silent" else [completed.fact_id if completed else facts[0].fact_id]
        content = json.dumps({"suggested_action": self.action, "message": message, "reason": reason,
                              "evidence": evidence}, ensure_ascii=False)
        return ModelResponse(content, "mock", "mock", None, None, 0.0, "stop")


def main() -> int:
    _utf8_output()
    parser = argparse.ArgumentParser(description="R3 离线模拟：不调用真实模型")
    parser.add_argument("event", type=Path, help="模拟训练事件 JSON")
    parser.add_argument("--action", choices=("silent", "notify", "defer"), default="notify")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dnd-on", dest="dnd", action="store_const", const=True)
    group.add_argument("--dnd-off", dest="dnd", action="store_const", const=False)
    parser.set_defaults(dnd=None)
    args = parser.parse_args()
    try:
        event = _load_json(args.event)
    except (OSError, ValueError, RecursionError):
        print(json.dumps({"status": "error", "code": "invalid_input_file"}, ensure_ascii=False))
        return 1
    source = decide_simulated_event(event, DemoModel(args.action))
    reviewed = review_outcome(source, do_not_disturb=args.dnd)
    output = {"status": reviewed.status, "stage": reviewed.stage, "code": reviewed.code,
              "effective_action": reviewed.effective_action, "rule_code": reviewed.rule_code,
              "reference_check": reviewed.reference_check, "warnings": reviewed.warnings,
              "notice": reviewed.notice + " 独立 R3 不读取会话安全状态。"}
    if reviewed.status == "reviewed":
        output["model_candidate_unverified"] = asdict(source.candidate)
    if reviewed.status == "safety_pending":
        output["safety_message"] = source.safety_message
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return {"reviewed": 0, "stopped": 2, "safety_pending": 3}[reviewed.status]


if __name__ == "__main__":
    raise SystemExit(main())
