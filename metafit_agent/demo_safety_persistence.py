"""R5 安全状态离线演示；确认答案为程序固定的模拟输入。"""
import argparse
import json
from pathlib import Path
import sys

from .model_response import ModelResponse
from .safety_coordinator import DurableSafetyCoordinator
from .safety_store import SafetySQLiteStore


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="R5 模拟安全状态演示")
    parser.add_argument("--db", type=Path, default=root / "outputs" / "r5_simulation" / "safety_demo.db")
    parser.add_argument("--scenario", choices=("pending", "report", "confirm"), default="pending")
    args = parser.parse_args()
    event = json.loads((root / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    event["event_id"] = "r5_unit4_" + args.scenario
    event["discomfort"] = {"status": "reported" if args.scenario == "report" else "none", "source": "self_report"}
    c = DurableSafetyCoordinator(SafetySQLiteStore(args.db))
    registered = c.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="event_clock")
    confirmation = None
    if args.scenario == "confirm" and registered.status == "registered":
        common = {"now": "2026-11-03T08:00:00Z", "clock_id": "event_clock"}
        c.apply(event["session_id"], "open_question", question_id="demo_confirmation", kind="safety_confirmation",
                deadline="2026-11-03T08:02:00Z", **common)
        c.apply(event["session_id"], "present_question", question_id="demo_confirmation", **common)
        confirmation = c.apply(event["session_id"], "answer_question", question_id="demo_confirmation", answer="none", **common)

    class Model:
        calls = 0
        def complete(self, facts):
            self.calls += 1
            content = json.dumps({"suggested_action": "notify", "message": "本组完成了10次。",
                                  "reason": "模拟状态仅供参考。", "evidence": ["fact_3"]}, ensure_ascii=False)
            return ModelResponse(content, "mock", "mock", None, None, 0, "stop")

    model = Model()
    result = c.decide(event["session_id"], event, model, do_not_disturb=False)
    print(json.dumps({"registration": registered.status, "confirmation": confirmation.status if confirmation else None,
                      "status": result.status, "code": result.code, "event_saved": result.event_saved,
                      "pending_message": result.pending_message,
                      "safety_state_durability": result.safety_state_durability, "model_calls": model.calls,
                      "warnings": result.warnings,
                      "notice": "全部为模拟；自动确认不代表真实用户自报，不构成安全判断。"}, ensure_ascii=False, indent=2))
    return 0 if result.status in {"reviewed", "safety_pending"} or result.code in {"pending_confirmation", "duplicate_event"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
