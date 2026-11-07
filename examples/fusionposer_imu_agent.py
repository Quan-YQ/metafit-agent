"""公开 IMU→预测骨架→视觉观察→模拟候选的隔离研究流程。"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafit_agent.context_builder import fact_list_digest
from metafit_agent.decision import ModelCallError, _parse_candidate, decide_simulated_event
from metafit_agent.decision_review import review_outcome
from metafit_agent.demo_decision import make_key_provider
from metafit_agent.facts import DecisionScope, Fact
from metafit_agent.model_response import ModelResponse
from check_vision_observation import PROMPT, compare
from fusionposer_research import experiment
from research_motion_choice import DeepSeekChoiceModel, OfflineChoiceModel
from trace_motion_agent import DeepSeekObserver, ScriptedObserver


def digest(data):
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IMUGate:
    """仅用于公开样本的研究接合；不为真实身份或动作质量背书。"""
    def __init__(self, observer, decision):
        self.observer, self.decision = observer, decision
        self.report = None
        self.observation = None
        self.motion_fact = None
        self.binding = None
        self.binding_sha256 = None
        self.precheck = "not_checked"

    def _verify(self, facts, scope):
        try:
            fresh, image, _ = experiment()
            observation = self.observation
            if (not observation or observation["comparison"] != "matches_reference"
                    or observation["source_x_sha256"] != fresh["x_sha256"]
                    or observation["source_y_sha256"] != fresh["y_sha256"]
                    or observation["selected_original_rows"] != fresh["selected_original_rows"]
                    or observation["image_sha256"] != hashlib.sha256(image).hexdigest()
                    or observation["prompt_sha256"] != hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
                    or observation["reference"] != fresh["predicted_hip_reference"]
                    or observation["model_hip_change"] != fresh["predicted_hip_reference"]["direction"]):
                return False
            expected_id = "imu_obs_" + digest({k: v for k, v in observation.items() if k != "observation_id"})[:16]
            if observation["observation_id"] != expected_id:
                return False
            expected = {"schema": "public-imu-simulated-binding-1", "event_facts_digest": fact_list_digest(scope, facts),
                        "observation_id": expected_id, "source_x_sha256": fresh["x_sha256"],
                        "source_y_sha256": fresh["y_sha256"], "image_sha256": observation["image_sha256"]}
            fact = self.motion_fact
            expected_fact_id = f"fact_{max(int(f.fact_id[5:]) for f in facts) + 1}"
            return (self.binding == expected and self.binding_sha256 == digest(expected)
                    and isinstance(fact, Fact) and fact.fact_id == expected_fact_id
                    and fact.name == "simulated_model_reported_hip_change" and fact.value == observation["model_hip_change"]
                    and (fact.user_id, fact.session_id, fact.event_id) == (scope.user_id, scope.session_id, scope.event_id)
                    and fact.origin == "event" and fact.kind == "simulated_motion_observation"
                    and fact.field == "shadow_observation:" + expected_id)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, StopIteration, OverflowError):
            return False

    def complete(self, facts):
        if not facts or any(f.origin != "event" for f in facts):
            raise ModelCallError("unsupported_research_scope")
        first = facts[0]
        scope = DecisionScope(first.user_id, first.session_id, first.event_id)
        if any((f.user_id, f.session_id, f.event_id) != (scope.user_id, scope.session_id, scope.event_id) for f in facts):
            raise ModelCallError("unsupported_research_scope")
        try:
            self.report, image, _ = experiment()
        except (OSError, ValueError, TypeError, IndexError, OverflowError):
            raise ModelCallError("imu_research_data_invalid") from None
        if not isinstance(image, bytes) or not image.startswith(b"\x89PNG\r\n\x1a\n") or len(image) > 2 * 1024 * 1024:
            raise ModelCallError("imu_predicted_image_invalid")
        reference = self.report["predicted_hip_reference"]
        self.observation = {
            "schema": "imu-prediction-observation-research-1", "source_x_sha256": self.report["x_sha256"],
            "source_y_sha256": self.report["y_sha256"], "selected_original_rows": self.report["selected_original_rows"],
            "image_sha256": hashlib.sha256(image).hexdigest(), "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "observer_mode": self.observer.mode, "requested_model": self.observer.requested_model,
            "reference": reference, "model_hip_change": None, "comparison": "not_completed",
        }
        try:
            response = self.observer.observe(image)
            checked = compare(response, reference)
        except ValueError:
            self.observation["comparison"] = "invalid_output"
            raise ModelCallError("observation_invalid_output") from None
        self.observation["model_hip_change"] = checked["model_observation"]["hip_change"]
        self.observation["comparison"] = checked["comparison"]
        self.observation["observation_id"] = "imu_obs_" + digest(self.observation)[:16]
        if checked["comparison"] != "matches_reference":
            raise ModelCallError("observation_not_matched")
        next_id = f"fact_{max(int(f.fact_id[5:]) for f in facts) + 1}"
        self.motion_fact = Fact(next_id, "simulated_model_reported_hip_change", self.observation["model_hip_change"],
                                scope.user_id, scope.session_id, scope.event_id,
                                "shadow_observation:" + self.observation["observation_id"], "event", "simulated_motion_observation")
        self.binding = {"schema": "public-imu-simulated-binding-1", "event_facts_digest": fact_list_digest(scope, facts),
                        "observation_id": self.observation["observation_id"], "source_x_sha256": self.report["x_sha256"],
                        "source_y_sha256": self.report["y_sha256"], "image_sha256": self.observation["image_sha256"]}
        self.binding_sha256 = digest(self.binding)
        if not self._verify(facts, scope):
            self.precheck = "rejected"
            raise ModelCallError("binding_precheck_failed")
        self.precheck = "passed"
        result = self.decision.complete(facts + (self.motion_fact,))
        if not isinstance(result, ModelResponse):
            raise ModelCallError("invalid_research_decision")
        _, error = _parse_candidate(result.content)
        if error:
            raise ModelCallError(error)
        return result


def run_pipeline(event, observer=None, decision=None):
    chosen_observer = observer if observer is not None else ScriptedObserver("no_clear_change")
    chosen_decision = decision if decision is not None else OfflineChoiceModel()
    gate = IMUGate(chosen_observer, chosen_decision)
    outcome = decide_simulated_event(event, gate)
    binding_status = "not_checked"
    if outcome.status == "candidate":
        try:
            allowed = {f.fact_id for f in outcome.facts} | {gate.motion_fact.fact_id}
            evidence = outcome.candidate.evidence
            reference_ok = (all(item in allowed for item in evidence)
                            and (outcome.candidate.suggested_action == "silent" or gate.motion_fact.fact_id in evidence))
            valid = (outcome.scope is not None and outcome.context_build is not None
                     and outcome.context_build.fact_list_digest == fact_list_digest(outcome.scope, outcome.facts)
                     and gate._verify(outcome.facts, outcome.scope) and reference_ok)
        except (ValueError, TypeError, AttributeError):
            valid = False
        if valid:
            binding_status = "matched_simulation"
            outcome = replace(outcome, facts=outcome.facts + (gate.motion_fact,), context_build=None)
        else:
            binding_status = "rejected"
            outcome = replace(outcome, status="error", candidate=None, error_code="binding_or_evidence_rejected",
                              model_response=None, context_build=None)
    reviewed = review_outcome(outcome, do_not_disturb=False)
    response = getattr(chosen_decision, "response", None)
    usage = ({"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
             if response is not None and response.usage else None)
    budget = getattr(chosen_decision, "budget", None)
    return {
        "simulation": True, "source_kind": "public_real_imu_and_head_height", "skeleton_origin": "predicted_by_single_sequence_ridge",
        "reference_origin": "public_optical_motion_capture", "event_image_association": "synthetic_not_real_world_proof",
        "prediction_research": gate.report, "visual_observation": gate.observation,
        "visual_requests_started": chosen_observer.calls if chosen_observer.mode == "live_deepseek_observation" else 0,
        "visual_usage": chosen_observer.usage, "visual_elapsed_ms": chosen_observer.elapsed_ms,
        "binding_precheck": gate.precheck, "binding_status": binding_status, "binding_sha256": gate.binding_sha256,
        "decision_mode": chosen_decision.mode,
        "decision_requests_started": chosen_decision.calls if chosen_decision.mode == "live_deepseek_choice" else 0,
        "decision_usage": usage, "decision_elapsed_ms": round(response.elapsed_ms, 2) if response else None,
        "decision_reason_adjustment": getattr(chosen_decision, "reason_adjustment", None),
        "decision_estimated_yuan": str(budget.estimated_spent_yuan) if budget else None,
        "decision_budget_uncertain": budget.uncertain if budget else None,
        "r2_status": outcome.status, "r2_error": outcome.error_code, "r3_status": reviewed.status,
        "reference_check": reviewed.reference_check,
        "candidate_action": outcome.candidate.suggested_action if outcome.candidate else None,
        "candidate_evidence": list(outcome.candidate.evidence) if outcome.candidate else [],
        "do_not_disturb": {"value": False, "source": "simulation_default"},
        "effective_action": reviewed.effective_action, "may_display": False, "delivery": False,
        "notice": "仅单序列研究：视觉几何参考来自预测骨架而非人体真值；输出不是训练安全判断。"
                  "图片与虚构事件没有经身份或设备认证；非 R5、非 App，候选不可投递。"
                  "文本费用估算不含视觉请求，也不是总账单上限。",
    }


def main():
    parser = argparse.ArgumentParser(description="公开 IMU 数据的预测骨架与 Agent 研究实验")
    parser.add_argument("--live", action="store_true", help="最多两次 DeepSeek 请求，可能收费")
    parser.add_argument("--prompt-key", action="store_true", help="在终端隐藏输入密钥")
    args = parser.parse_args()
    if args.prompt_key and not args.live:
        parser.error("--prompt-key 需要 --live")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        event = json.loads(Path(__file__).with_name("set_completed.json").read_text(encoding="utf-8"))
        event["exercise_id"] = "public_imu_research"
        for field in ("planned_reps", "completed_reps", "weight_kg_per_hand", "metrics", "set_index", "rest_seconds"):
            event.pop(field, None)
        event["discomfort"] = {"status": "none", "source": "self_report"}
        if args.live:
            print("最多两次请求：发送公开 IMU 预测骨架图、模拟决策事实；可能收费，不自动重试。")
            provider = make_key_provider() if args.prompt_key else None
            observer, decision = DeepSeekObserver(provider), DeepSeekChoiceModel(provider)
        else:
            observer, decision = ScriptedObserver("no_clear_change"), OfflineChoiceModel()
        result = run_pipeline(event, observer, decision)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["r3_status"] == "reviewed" else 1
    except (OSError, ValueError, json.JSONDecodeError):
        print("IMU 研究流程未完成。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
