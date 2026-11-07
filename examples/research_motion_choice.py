"""研究级动作观察与候选选择：仅模拟、不投递，允许模型选择三种动作。"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafit_agent.context_builder import fact_list_digest
from metafit_agent.decision import ModelCallError, UNKNOWN_REASON, _parse_candidate, decide_simulated_event
from metafit_agent.decision_review import review_outcome
from metafit_agent.deepseek_client import ENDPOINT, MODEL
from metafit_agent.demo_decision import make_key_provider
from metafit_agent.facts import DecisionScope
from metafit_agent.model_client import CallBudget, MAX_OUTPUT_TOKENS, SYSTEM_PROMPT, _NoRedirect, invoke, prepare
from metafit_agent.model_response import ModelResponse
from bound_motion_agent import BoundSilentDecision, verify_binding
from check_motion_geometry import SAMPLES
from check_vision_observation import PROMPT
from trace_motion_agent import DeepSeekObserver, ObservationGate, ScriptedObserver


RESEARCH_SYSTEM = (SYSTEM_PROMPT + "\n"
                   "simulated_model_reported_hip_change 表示另一个模型对模拟图片的观察，可能出错，"
                   "不是已确认的动作事实，不代表动作标准、安全或健康状况。"
                   "这是不可投递的研究候选；若建议 notify 或 defer，evidence 必须引用该观察的 fact_id。"
                   "不得给出动作是否合格、是否安全、疼痛原因或训练调整建议。"
                   "当 discomfort_status 为 none，reason 必须明确说明模拟数据记录了自报无不适，"
                   "并说明时效和当前状态未确认；不得使用仅适用于 unknown 的固定原因。")

NONE_REASON = "模拟数据记录为自报无不适，时效及当前状态未确认。"


class UserReportedReplay:
    """复用用户提供的 round_a 观察；历史提供方响应未经本进程独立认证。"""
    mode = "user_reported_previous_observation"
    requested_model = "deepseek-flash"
    source_sha256 = "34b974256d6e755995e2d430db2224ade65934384912bc676cdfb86bd253d9bc"
    image_sha256 = "40edff67755080be4e7651fe7f9cbbc418276004d2af10154e7f09f260bd2fc9"
    prompt_sha256 = "6887863d5f80e760f400fb64386aa0b5830ae0bacb4b1ef52baee6482607f553"
    prior_usage = {"prompt_tokens": 482, "completion_tokens": 32}

    def __init__(self, path):
        self.path = path
        self.calls = 0
        self.usage = None
        self.elapsed_ms = None

    def observe(self, image):
        if self.calls:
            raise ModelCallError("replay_call_limit")
        self.calls += 1
        try:
            source_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError:
            raise ModelCallError("replay_source_unavailable") from None
        if (self.path.name != "cmu_23_14_sample.bvh" or source_hash != self.source_sha256
                or hashlib.sha256(image).hexdigest() != self.image_sha256
                or hashlib.sha256(PROMPT.encode("utf-8")).hexdigest() != self.prompt_sha256):
            raise ModelCallError("replay_input_mismatch")
        return json.dumps({"hip_change": "up", "explanation": "用户提供的上次模拟观察记录"}, ensure_ascii=False)


class OfflineChoiceModel:
    """测试替身；根据预设规则产生结构化候选，不代表真实模型。"""
    mode = "offline_scripted_choice"

    def __init__(self):
        self.calls = 0
        self.response = None

    def complete(self, facts):
        self.calls += 1
        observation = next(f for f in facts if f.name == "simulated_model_reported_hip_change")
        unknown = any(f.name == "discomfort_status" and f.value == "unknown" for f in facts)
        action = "notify" if observation.value == "up" and not unknown else "silent"
        body = {"suggested_action": action,
                "message": "模拟模型报告图中髋部上升。" if action == "notify" else None,
                "reason": UNKNOWN_REASON if unknown else "仅根据模拟观察构造研究候选。",
                "evidence": [observation.fact_id]}
        self.response = ModelResponse(json.dumps(body, ensure_ascii=False), "offline_scripted_choice",
                                      None, None, None, 0.0, "stop")
        return self.response


class DeepSeekChoiceModel:
    """仅发送经过白名单构建的模拟事实；最多一次文本决策请求。"""
    mode = "live_deepseek_choice"

    def __init__(self, key_provider=None, *, system_prompt=RESEARCH_SYSTEM):
        self.key_provider = key_provider
        self.system_prompt = system_prompt
        self.budget = CallBudget(max_calls=1)
        self.response = None
        self.reason_adjustment = None

    @property
    def calls(self):
        return self.budget.calls

    def complete(self, facts):
        safe = [{"fact_id": fact.fact_id, "name": fact.name, "value": fact.value} for fact in facts]
        body = {"model": MODEL, "messages": [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps({"facts": safe}, ensure_ascii=False)},
        ], "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"},
            "max_tokens": MAX_OUTPUT_TOKENS, "stream": False}
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        prepare(payload, self.budget)
        key = (self.key_provider() if self.key_provider is not None else os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise ModelCallError("missing_api_key")
        self.response = invoke(ENDPOINT, payload, key, self.budget, MODEL,
                               build_opener=lambda: request.build_opener(_NoRedirect()))
        # 只处理已验证结构的一个已知歧义句；保留原始响应、动作、消息和引用。
        # unknown 或其他问题仍交给 R2 原规则拦截；这不是自然语言真实性检查。
        candidate, error = _parse_candidate(self.response.content)
        statuses = [fact.value for fact in facts if fact.name == "discomfort_status" and fact.origin == "event"]
        if error is None and candidate is not None and statuses == ["none"] and candidate.reason == UNKNOWN_REASON:
            body = {"suggested_action": candidate.suggested_action, "message": candidate.message,
                    "reason": NONE_REASON, "evidence": list(candidate.evidence)}
            self.reason_adjustment = "none_ambiguous_reason_normalized"
            return replace(self.response, content=json.dumps(body, ensure_ascii=False))
        return self.response


class BoundChoiceDecision(BoundSilentDecision):
    """复用模拟绑定；将模型报告而非动作真相作为研究事实。"""

    def __init__(self, gate, choice_model):
        super().__init__(gate)
        self.choice_model = choice_model
        self.precheck_status = "not_checked"

    def complete(self, facts):
        # 父类仅用于构造绑定和研究事实；其固定候选不会被使用。
        super().complete(facts)
        first = facts[0]
        scope = DecisionScope(first.user_id, first.session_id, first.event_id)
        try:
            valid = verify_binding(self.binding, self.binding_sha256, scope, facts, self.gate.observation,
                                   self.motion_fact, self.gate.path.read_bytes(), self.gate.path.name)
        except (OSError, ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            self.precheck_status = "rejected"
            raise ModelCallError("binding_precheck_failed")
        self.precheck_status = "passed"
        return self.choice_model.complete(facts + (self.motion_fact,))


def run_choice(event, sample="round_a", observer=None, choice_model=None):
    if sample not in ("squat_1", "round_a"):
        raise ValueError("unsupported research sample")
    chosen_observer = observer if observer is not None else ScriptedObserver("up" if sample == "round_a" else "down")
    chosen_decision = choice_model if choice_model is not None else OfflineChoiceModel()
    path = SAMPLES[sample]
    gate = ObservationGate(path, chosen_observer, allow_non_silent=True)
    binding = BoundChoiceDecision(gate, chosen_decision)
    gate.decision_model = binding
    outcome = decide_simulated_event(event, gate)
    binding_status = "not_checked"
    if outcome.status == "candidate":
        try:
            evidence = outcome.candidate.evidence
            allowed = {f.fact_id for f in outcome.facts} | {binding.motion_fact.fact_id}
            reference_ok = (all(item in allowed for item in evidence)
                            and (outcome.candidate.suggested_action == "silent" or binding.motion_fact.fact_id in evidence))
            valid = (outcome.scope is not None and outcome.context_build is not None
                     and outcome.context_build.fact_list_digest == fact_list_digest(outcome.scope, outcome.facts)
                     and verify_binding(binding.binding, binding.binding_sha256, outcome.scope, outcome.facts,
                                        gate.observation, binding.motion_fact, path.read_bytes(), path.name)
                     and reference_ok)
        except (OSError, ValueError, TypeError, AttributeError):
            valid = False
        if valid:
            binding_status = "matched_simulation"
            # 这个研究事实不进入原上下文、持久化或 App。
            outcome = replace(outcome, facts=outcome.facts + (binding.motion_fact,), context_build=None)
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
        "simulation": True, "sample": sample, "event_image_association": "synthetic_not_real_world_proof",
        "observation_mode": chosen_observer.mode, "observation": gate.observation,
        "observation_calls": chosen_observer.calls,
        "visual_requests_started": chosen_observer.calls if chosen_observer.mode == "live_deepseek_observation" else 0,
        "visual_usage": chosen_observer.usage,
        "visual_replay_origin": ("user_reported_not_independently_verified"
                                 if chosen_observer.mode == "user_reported_previous_observation" else None),
        "prior_reported_visual_usage": (dict(chosen_observer.prior_usage)
                                        if getattr(chosen_observer, "prior_usage", None) is not None else None),
        "visual_elapsed_ms": chosen_observer.elapsed_ms,
        "binding_precheck": binding.precheck_status, "binding_status": binding_status, "binding_sha256": binding.binding_sha256,
        "decision_mode": chosen_decision.mode, "decision_calls": chosen_decision.calls,
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
        "notice": "研究候选不可投递；观察是模型报告，引用检查不验证自然语言或动作安全。"
                  "图片和虚构事件不证明真实关联；不是 R5 或 App 流程。"
                  "文本决策费用仅为估算，不包含图片请求费用，也不是账单上限。",
    }


def main():
    parser = argparse.ArgumentParser(description="模拟动作观察驱动 Agent 候选选择；不可投递")
    parser.add_argument("--sample", choices=("round_a", "squat_1"), default="round_a")
    parser.add_argument("--reuse-observation", action="store_true", help="复用用户提供的 round_a 上次观察记录")
    parser.add_argument("--live", action="store_true", help="最多发起两次 DeepSeek 请求，可能收费")
    parser.add_argument("--prompt-key", action="store_true", help="需要时隐藏输入密钥")
    args = parser.parse_args()
    if args.prompt_key and not args.live:
        parser.error("--prompt-key 需要 --live")
    if args.reuse_observation and args.sample != "round_a":
        parser.error("已有观察记录只对应 round_a")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        event = json.loads(Path(__file__).with_name("set_completed.json").read_text(encoding="utf-8"))
        event["exercise_id"] = "squat"
        for field in ("planned_reps", "completed_reps", "weight_kg_per_hand", "metrics", "set_index", "rest_seconds"):
            event.pop(field, None)
        event["discomfort"] = {"status": "none", "source": "self_report"}
        if args.live:
            print("最多一次文本选择请求；复用用户提供的上次观察记录，不重新看图。可能收费，失败不自动重试。"
                  if args.reuse_observation else
                  "最多两次请求：一次图片观察和一次文本选择；可能收费，失败不自动重试。")
            provider = make_key_provider() if args.prompt_key else None
            observer = UserReportedReplay(SAMPLES[args.sample]) if args.reuse_observation else DeepSeekObserver(provider)
            decision = DeepSeekChoiceModel(provider)
        else:
            observer = UserReportedReplay(SAMPLES[args.sample]) if args.reuse_observation else ScriptedObserver("up" if args.sample == "round_a" else "down")
            decision = OfflineChoiceModel()
        result = run_choice(event, args.sample, observer, decision)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["r3_status"] == "reviewed" else 1
    except (OSError, ValueError, json.JSONDecodeError):
        print("研究演示未完成。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
