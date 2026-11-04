"""R4 模拟上下文与 R2/R3 的离线演示，不联网或保存。"""
import json
from pathlib import Path

from .decision import decide_simulated_event
from .decision_review import review_outcome
from .demo_decision import _DemoModel, _load_json, _utf8_output


def main() -> int:
    _utf8_output()
    root = Path(__file__).resolve().parents[1]
    try:
        event = _load_json(root / "examples" / "set_completed.json")
        context = _load_json(root / "examples" / "r4_context.json")
    except (OSError, ValueError, RecursionError):
        print(json.dumps({"status": "error", "code": "invalid_example"}))
        return 1
    outcome = decide_simulated_event(event, _DemoModel(), context)
    reviewed = review_outcome(outcome, do_not_disturb=False)
    build = outcome.context_build
    output = {"status": reviewed.status, "code": reviewed.code, "effective_action": reviewed.effective_action,
              "notice": "仅限模拟；未验证真实来源、权限、文本正确性或安全性。"}
    if reviewed.status == "safety_pending":
        output["safety_message"] = outcome.safety_message
    if build:
        output["facts"] = [{"fact_id": f.fact_id, "name": f.name, "value": f.value, "kind": f.kind, "origin": f.origin} for f in build.facts]
        output["missing_categories"] = build.missing_categories
        output["discomfort_time_relation"] = build.discomfort_time_relation
        output["history_selection"] = {"input_count": build.history_input_count,
                                       "selected_count": build.history_selected_count,
                                       "limit": build.history_limit, "window_days": build.history_window_days}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return {"reviewed": 0, "stopped": 2, "safety_pending": 3}[reviewed.status]


if __name__ == "__main__":
    raise SystemExit(main())
