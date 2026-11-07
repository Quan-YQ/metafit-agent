"""模拟事件与模型观察的绑定、引用及 R3 检查；结果不可展示。"""

import argparse
from dataclasses import replace
import hashlib
import hmac
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafit_agent.context_builder import fact_list_digest
from metafit_agent.decision import ModelCallError, UNKNOWN_REASON, decide_simulated_event
from metafit_agent.decision_review import review_outcome
from metafit_agent.demo_decision import make_key_provider
from metafit_agent.facts import DecisionScope, Fact
from metafit_agent.model_response import ModelResponse
from check_motion_geometry import SAMPLES, analyze_bytes
from check_vision_observation import PROMPT
from render_motion_sample import parse_bvh, render_frames
from trace_motion_agent import DeepSeekObserver, ObservationGate, ScriptedObserver


def digest_json(data):
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def observation_id(observation):
    return "obs_" + digest_json({k: v for k, v in observation.items() if k != "observation_id"})[:16]


def create_binding(scope, facts, observation):
    """绑定仅证明本地模拟内容一致，不认证来源或真实身份。"""
    return {
        "schema": "simulated-motion-binding-1",
        "event_facts_digest": fact_list_digest(scope, facts),
        "observation_id": observation["observation_id"],
        "source_sha256": observation["source_sha256"],
        "image_sha256": observation["image_sha256"],
        "prompt_sha256": observation["prompt_sha256"],
        "frame_order": observation["frame_order"],
        "model_reported_hip_change": observation["model_hip_change"],
    }


def verify_binding(binding, binding_sha256, scope, facts, observation, motion_fact, source_bytes, source_name):
    """对照事件事实、样本快照、图像、观察与引用；非认证。"""
    try:
        if not hmac.compare_digest(binding_sha256, digest_json(binding)):
            return False
        if observation["observation_id"] != observation_id(observation):
            return False
        if observation["source_file"] != source_name or observation["comparison"] != "matches_reference":
            return False
        if observation["frame_order"] not in ("original", "reversed"):
            return False
        reverse = observation["frame_order"] == "reversed"
        geo = analyze_bytes(source_bytes, source_name, reverse=reverse)
        joints, frames = parse_bvh(source_bytes.decode("utf-8"))
        image = render_frames(joints, frames, reverse=reverse)
        if (observation["source_sha256"] != geo["source_sha256"]
                or observation["image_sha256"] != hashlib.sha256(image).hexdigest()
                or observation["prompt_sha256"] != hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
                or observation["reference"] != geo["hip_endpoint_reference"]
                or observation["model_hip_change"] != geo["hip_endpoint_reference"]["direction"]):
            return False
        if binding != create_binding(scope, facts, observation):
            return False
        expected_id = f"fact_{max(int(f.fact_id[5:]) for f in facts) + 1}"
        return (isinstance(motion_fact, Fact) and motion_fact.fact_id == expected_id
                and motion_fact.name == "simulated_model_reported_hip_change"
                and motion_fact.value == observation["model_hip_change"]
                and (motion_fact.user_id, motion_fact.session_id, motion_fact.event_id)
                == (scope.user_id, scope.session_id, scope.event_id)
                and motion_fact.origin == "event" and motion_fact.kind == "simulated_motion_observation"
                and motion_fact.field == "shadow_observation:" + observation["observation_id"])
    except (KeyError, TypeError, ValueError, UnicodeError, IndexError, StopIteration, OverflowError):
        return False


class BoundSilentDecision:
    """只在研究沙盒构造引用，不把视觉输出变成正式可信事实。"""

    def __init__(self, gate):
        self.gate = gate
        self.binding = None
        self.binding_sha256 = None
        self.motion_fact = None

    def complete(self, facts):
        if not facts or any(f.origin != "event" for f in facts):
            raise ModelCallError("unsupported_research_scope")
        first = facts[0]
        scope = DecisionScope(first.user_id, first.session_id, first.event_id)
        if any((f.user_id, f.session_id, f.event_id) != (scope.user_id, scope.session_id, scope.event_id) for f in facts):
            raise ModelCallError("unsupported_research_scope")
        observation = self.gate.observation
        self.binding = create_binding(scope, facts, observation)
        self.binding_sha256 = digest_json(self.binding)
        next_id = f"fact_{max(int(f.fact_id[5:]) for f in facts) + 1}"
        self.motion_fact = Fact(next_id, "simulated_model_reported_hip_change", observation["model_hip_change"],
                                scope.user_id, scope.session_id, scope.event_id,
                                "shadow_observation:" + observation["observation_id"],
                                "event", "simulated_motion_observation")
        unknown = any(f.name == "discomfort_status" and f.value == "unknown" for f in facts)
        reason = UNKNOWN_REASON if unknown else "仅记录模型报告的模拟观察，尚未验证真实动作。"
        payload = {"suggested_action": "silent", "message": None, "reason": reason, "evidence": [next_id]}
        return ModelResponse(json.dumps(payload, ensure_ascii=False), "bound_offline_decision", None, None, None, 0.0, "stop")


def run_bound(event, sample="round_a", observer=None, *, reverse=False):
    if sample not in ("squat_1", "round_a"):
        raise ValueError("unsupported research sample")
    chosen = observer if observer is not None else ScriptedObserver()
    path = SAMPLES[sample]
    gate = ObservationGate(path, chosen, reverse=reverse)
    decision = BoundSilentDecision(gate)
    gate.decision_model = decision
    outcome = decide_simulated_event(event, gate)
    binding_status = "not_checked"
    if outcome.status == "candidate":
        try:
            valid = (outcome.scope is not None and outcome.context_build is not None
                     and outcome.context_build.fact_list_digest == fact_list_digest(outcome.scope, outcome.facts)
                     and verify_binding(decision.binding, decision.binding_sha256, outcome.scope, outcome.facts,
                                        gate.observation, decision.motion_fact, path.read_bytes(), path.name)
                     and outcome.candidate.evidence == (decision.motion_fact.fact_id,))
        except (OSError, ValueError, TypeError, AttributeError):
            valid = False
        if valid:
            binding_status = "matched_simulation"
            # R3 演示需要研究事实；丢弃未覆盖它的原上下文摘要，禁止误用到存储链。
            outcome = replace(outcome, facts=outcome.facts + (decision.motion_fact,), context_build=None)
        else:
            binding_status = "rejected"
            outcome = replace(outcome, status="error", candidate=None, error_code="simulated_binding_mismatch",
                              model_response=None, context_build=None)
    review = review_outcome(outcome, do_not_disturb=False)
    return {
        "simulation": True, "mode": chosen.mode, "sample": sample,
        "binding_status": binding_status, "binding_sha256": decision.binding_sha256,
        "binding_association": "synthetic_not_real_world_proof",
        "observation": gate.observation, "observation_attempts": chosen.calls,
        "model_usage": chosen.usage, "model_elapsed_ms": chosen.elapsed_ms,
        "decision_model": "offline_scripted_silent", "decision_attempts": gate.decision_calls,
        "cited_fact": ({"fact_id": decision.motion_fact.fact_id, "name": decision.motion_fact.name,
                        "value": decision.motion_fact.value, "kind": decision.motion_fact.kind}
                       if binding_status == "matched_simulation" else None),
        "r2_status": outcome.status, "r2_error": outcome.error_code,
        "r3_status": review.status, "reference_check": review.reference_check,
        "do_not_disturb": {"value": False, "source": "simulation_default"},
        "effective_action": review.effective_action, "may_display": False,
        "notice": "仅研究模拟：观察是模型报告而非动作真相；绑定不认证真实来源。"
                  "研究事实由独立适配器加入 R3，不进入正式事件、存储、R5 或 App。",
    }


def main():
    parser = argparse.ArgumentParser(description="模拟动作观察绑定及 R3 引用检查")
    parser.add_argument("--sample", choices=("round_a", "squat_1"), default="round_a")
    parser.add_argument("--live", action="store_true", help="发起一次 DeepSeek 观察请求，可能产生费用")
    parser.add_argument("--prompt-key", action="store_true", help="在终端隐藏输入密钥")
    args = parser.parse_args()
    if args.prompt_key and not args.live:
        parser.error("--prompt-key 需要 --live")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        event = json.loads(Path(__file__).with_name("set_completed.json").read_text(encoding="utf-8"))
        event["exercise_id"] = "squat"
        for name in ("planned_reps", "completed_reps", "weight_kg_per_hand", "metrics", "set_index", "rest_seconds"):
            event.pop(name, None)
        event["discomfort"] = {"status": "none", "source": "self_report"}
        if args.live:
            print("最多发送一次公开骨架图请求，可能产生费用；失败不自动重试。")
            observer = DeepSeekObserver(make_key_provider() if args.prompt_key else None)
        else:
            observer = ScriptedObserver("up" if args.sample == "round_a" else "down")
        result = run_bound(event, args.sample, observer)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["r3_status"] == "reviewed" else 1
    except (OSError, ValueError, json.JSONDecodeError):
        print("研究演示未完成。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
