"""R5 模拟事件持久化演示；仅固定模拟模型，不请求网络。"""
import argparse
import json
from pathlib import Path
import sys

from .event_store import SQLiteEventStore, StorageError
from .model_response import ModelResponse
from .stored_coordinator import StoredSimulationCoordinator


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="R5 本地模拟数据库演示")
    parser.add_argument("--db", type=Path, default=root / "outputs" / "r5_simulation" / "demo_events.db")
    parser.add_argument("--scenario", choices=("normal", "safety"), default="normal")
    args = parser.parse_args()
    try:
        event = json.loads((root / "examples" / "set_completed.json").read_text(encoding="utf-8"))
        store = SQLiteEventStore(args.db)
    except (OSError, ValueError, StorageError):
        print(json.dumps({"status": "error", "code": "invalid_demo_input"}, ensure_ascii=False))
        return 1
    event["event_id"] = "r5_demo_" + args.scenario
    event["discomfort"] = {"status": "reported" if args.scenario == "safety" else "none", "source": "self_report"}
    coordinator = StoredSimulationCoordinator(store)
    coordinator.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="event_clock")
    class Model:
        calls = 0
        def complete(self, facts):
            self.calls += 1
            content = json.dumps({"suggested_action": "notify", "message": "本组完成了10次。",
                                  "reason": "模拟自报仅供参考。", "evidence": ["fact_3"]}, ensure_ascii=False)
            return ModelResponse(content, "mock", "mock", None, None, 0, "stop")
    model = Model()
    result = coordinator.decide(event["session_id"], event, model, do_not_disturb=False)
    output = {"status": result.status, "code": result.code, "stage": result.stage,
              "event_saved": result.event_saved, "storage_status": result.storage_status,
              "decision_state": result.stored_event.decision_state if result.stored_event else None,
              "model_calls": model.calls, "warnings": result.warnings,
              "notice": "仅限模拟；数据库明文，未实现真实鉴权、会话安全持久化或 App 投递。"}
    if result.safety_message:
        output["safety_message"] = result.safety_message
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if result.status in ("reviewed", "safety_pending") or result.code == "duplicate_event" else 2


if __name__ == "__main__":
    raise SystemExit(main())
