"""R5 模拟引用快照演示；不调用网络、不返回事实值。"""
import argparse
import json
from pathlib import Path
import sys

from .model_response import ModelResponse
from .snapshot_coordinator import SnapshotSimulationCoordinator
from .snapshot_store import SnapshotSQLiteStore


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="R5 模拟引用快照演示")
    parser.add_argument("--db", type=Path, default=root / "outputs" / "r5_simulation" / "snapshot_history.db")
    args = parser.parse_args()
    event = json.loads((root / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    event["event_id"] = "r5_unit3_demo"
    event["discomfort"] = {"status": "none", "source": "self_report"}
    store = SnapshotSQLiteStore(args.db)
    c = SnapshotSimulationCoordinator(store)
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
    row = None
    if result.event_saved is True:
        rows = store.query(event["user_id"], session_id=event["session_id"], event_id=event["event_id"], limit=1)
        row = rows[0] if rows else None
    print(json.dumps({"status": result.status, "code": result.code, "model_calls": model.calls,
                      "has_snapshot": result.has_snapshot,
                      "snapshot_state": row["snapshot_state"] if row else None,
                      "facts": row["facts"] if row else [],
                      "notice": "仅供模拟排查；快照和指纹不能证明事实真实或文字安全。"}, ensure_ascii=False, indent=2))
    return 0 if result.status == "reviewed" or result.code == "duplicate_event" else 2


if __name__ == "__main__":
    raise SystemExit(main())
