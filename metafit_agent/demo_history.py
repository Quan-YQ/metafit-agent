"""R5 决策历史演示，只使用固定模拟输入和模型。"""
import argparse
import json
from pathlib import Path
import sys

from .decision_store import DecisionSQLiteStore
from .model_response import ModelResponse
from .recorded_coordinator import RecordedSimulationCoordinator


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="R5 模拟决策历史演示")
    parser.add_argument("--db", type=Path, default=root / "outputs" / "r5_simulation" / "decision_history.db")
    args = parser.parse_args()
    event = json.loads((root / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    event["event_id"] = "r5_unit2_demo"
    event["discomfort"] = {"status": "none", "source": "self_report"}
    store = DecisionSQLiteStore(args.db)
    c = RecordedSimulationCoordinator(store)
    c.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="event_clock")

    class Model:
        calls = 0
        def complete(self, facts):
            self.calls += 1
            content = json.dumps({"suggested_action": "notify", "message": "本组完成了10次。",
                                  "reason": "模拟状态仅供参考。", "evidence": ["fact_3"]}, ensure_ascii=False)
            return ModelResponse(content, "mock", "mock", None, None, 0, "stop")

    model = Model()
    result = c.decide(event["session_id"], event, model, do_not_disturb=False)
    print(json.dumps({"status": result.status, "code": result.code, "event_saved": result.event_saved,
                      "has_text": result.has_text, "decided_at": result.decided_at,
                      "model_calls": model.calls, "warnings": result.warnings,
                      "notice": "仅为模拟历史；原文未验证，不能用于自动投递。"}, ensure_ascii=False, indent=2))
    return 0 if result.status == "reviewed" or result.code == "duplicate_event" else 2


if __name__ == "__main__":
    raise SystemExit(main())
