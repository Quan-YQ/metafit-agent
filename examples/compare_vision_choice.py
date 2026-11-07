"""同一虚构事件：有/无模型图像观察时的候选对照；不投递。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafit_agent.decision import decide_simulated_event
from metafit_agent.decision_review import review_outcome
from research_motion_choice import DeepSeekChoiceModel, OfflineChoiceModel, RESEARCH_SYSTEM, run_choice
from trace_motion_agent import DeepSeekObserver, ScriptedObserver
from metafit_agent.model_response import ModelResponse


PAIRED_SYSTEM = RESEARCH_SYSTEM.replace(
    "若建议 notify 或 defer，evidence 必须引用该观察的 fact_id。",
    "若存在该观察且建议 notify 或 defer，evidence 必须引用该观察的 fact_id。")


class OfflineNoObservation:
    mode = "offline_text_only"
    def __init__(self):
        self.calls = 0
        self.response = None
    def complete(self, facts):
        self.calls += 1
        self.response = ModelResponse(json.dumps({"suggested_action": "silent", "message": None,
                                                   "reason": "模拟文本对照", "evidence": []}, ensure_ascii=False),
                                      self.mode, None, None, None, 0, "stop")
        return self.response


def run_pair(event, sample, observer, with_observation, without_observation):
    visual = run_choice(event, sample, observer, with_observation)
    result = {"sample": sample, "visual_result": visual,
              "text_only": {"status": "skipped"}, "action_comparison": "not_compared",
              "may_display": False, "delivery": False,
              "notice": "两组同一模拟事件和同一选择提示；图像组多一步视觉观察。结果仅对比候选，不证明质量、安全或真实设备关联。"}
    if visual["r3_status"] != "reviewed":
        return result
    outcome = decide_simulated_event(event, without_observation)
    review = review_outcome(outcome, do_not_disturb=False)
    response = getattr(without_observation, "response", None)
    budget = getattr(without_observation, "budget", None)
    result["text_only"] = {
        "status": outcome.status, "error": outcome.error_code, "review": review.status,
        "reference_check": review.reference_check,
        "candidate_action": outcome.candidate.suggested_action if outcome.candidate else None,
        "candidate_evidence": list(outcome.candidate.evidence) if outcome.candidate else [],
        "requests_started": without_observation.calls if without_observation.mode == "live_deepseek_choice" else 0,
        "usage": ({"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
                  if response is not None and response.usage else None),
        "elapsed_ms": round(response.elapsed_ms, 2) if response else None,
        "estimated_yuan": str(budget.estimated_spent_yuan) if budget else None,
    }
    if review.status == "reviewed":
        result["action_comparison"] = ("same_action" if visual["candidate_action"] == outcome.candidate.suggested_action
                                       else "different_action")
        base_fact_ids = {fact.fact_id for fact in outcome.facts}
        result["visual_observation_cited"] = any(fact_id not in base_fact_ids for fact_id in visual["candidate_evidence"])
    return result


def main():
    parser = argparse.ArgumentParser(description="对照有无公开骨架观察的研究候选")
    parser.add_argument("--live", action="store_true", help="最多一次图片及两次文本请求，可能收费")
    parser.add_argument("--sample", choices=("round_a", "squat_1"), default="round_a")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    event = json.loads(Path(__file__).with_name("set_completed.json").read_text(encoding="utf-8"))
    event["exercise_id"] = "research_motion"
    for field in ("planned_reps", "completed_reps", "weight_kg_per_hand", "metrics", "set_index", "rest_seconds"):
        event.pop(field, None)
    event["discomfort"] = {"status": "none", "source": "self_report"}
    if args.live:
        observer = DeepSeekObserver()
        with_model = DeepSeekChoiceModel(system_prompt=PAIRED_SYSTEM)
        without_model = DeepSeekChoiceModel(system_prompt=PAIRED_SYSTEM)
    else:
        observer = ScriptedObserver("up" if args.sample == "round_a" else "down")
        with_model, without_model = OfflineChoiceModel(), OfflineNoObservation()
    result = run_pair(event, args.sample, observer, with_model, without_model)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["action_comparison"] != "not_compared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
