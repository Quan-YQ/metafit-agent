"""R4 第二单元的离线模拟；不连接模型、数据库或 App。"""
import argparse
import json
import sys

from .interaction_state import (create_session, mark_training_started, open_question, present_question,
                                report_discomfort, skip_question)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="模拟状态交互，不会发送消息")
    parser.add_argument("--scenario", choices=("skip", "safety", "expiry"), default="skip")
    args = parser.parse_args()
    clock = "event_clock"
    t0, t1, t2 = "2026-11-02T10:00:00+08:00", "2026-11-02T10:00:30+08:00", "2026-11-02T10:02:01+08:00"
    deadline = "2026-11-02T10:02:00+08:00"
    state = create_session("simulation_session", clock)
    steps = []
    def record(name, result):
        item = {"step": name, "status": result.status, "code": result.code, "state_changed": result.state_changed}
        if result.question_text is not None:
            item["question_text"] = result.question_text
        if result.safety_message is not None:
            item["safety_message"] = result.safety_message
        steps.append(item)
        return result.session
    if args.scenario in ("skip", "safety"):
        state = record("create_pre", open_question(state, question_id="pre", kind="pre_training", now=t0,
                                                   deadline=deadline, clock_id=clock))
        state = record("present_pre", present_question(state, question_id="pre", now=t0, clock_id=clock))
        if args.scenario == "skip":
            state = record("skip_pre", skip_question(state, question_id="pre", now=t1, clock_id=clock))
            state = record("training_started", mark_training_started(state, now=t1, clock_id=clock))
            state = record("try_followup", open_question(state, question_id="f1", kind="followup", now=t1,
                    deadline=deadline, clock_id=clock, trigger_reason="user_requested", trigger_reference="trigger1"))
        else:
            state = record("discomfort", report_discomfort(state, now=t1, clock_id=clock))
            state = record("try_present_again", present_question(state, question_id="pre", now=t1, clock_id=clock))
    else:
        state = record("training_started", mark_training_started(state, now=t0, clock_id=clock))
        state = record("create_followup", open_question(state, question_id="f1", kind="followup", now=t1,
                deadline=deadline, clock_id=clock, trigger_reason="user_requested", trigger_reference="trigger1"))
        state = record("present_after_deadline", present_question(state, question_id="f1", now=t2, clock_id=clock))
    print(json.dumps({"scenario": args.scenario, "steps": steps, "question_statuses": [q.status for q in state.questions],
                      "notice": "仅是内存模拟，未发送消息、连接 App、持久化或调用模型。"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
