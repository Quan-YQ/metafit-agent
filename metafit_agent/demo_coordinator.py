"""R4 第三单元离线演示；不连接真实模型、数据库或 App。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .coordinator import SimulationCoordinator
from .model_response import ModelResponse


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="单进程模拟协调，无网络")
    parser.add_argument("--scenario", choices=("normal", "before", "during"), default="during")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    event = json.loads((root / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    event["discomfort"] = {"status": "none", "source": "self_report"}
    session_id = event["session_id"]
    coordinator = SimulationCoordinator()
    coordinator.register(user_id=event["user_id"], session_id=session_id, clock_id="event_clock")
    def report():
        return coordinator.apply(session_id, "report_discomfort", now=event["timestamp"], clock_id="event_clock")
    if args.scenario == "before":
        report()
    class MockModel:
        calls = 0
        def complete(self, facts):
            self.calls += 1
            if args.scenario == "during":
                report()
            content = json.dumps({"suggested_action": "notify", "message": "本组完成了10次。",
                                  "reason": "模拟自报状态仅供参考。", "evidence": ["fact_3"]}, ensure_ascii=False)
            return ModelResponse(content, "mock", "mock", None, None, 0.0, "stop")
    model = MockModel()
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    output = asdict(result)
    output["mock_model_calls"] = model.calls
    output["scenario"] = args.scenario
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if result.status in ("reviewed", "safety_pending") else 2


if __name__ == "__main__":
    raise SystemExit(main())
