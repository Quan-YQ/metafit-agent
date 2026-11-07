"""观察驱动候选选择：研究隔离、不适优先、引用范围和费用边界。"""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from metafit_agent.decision import ModelCallError
from metafit_agent.model_response import ModelResponse, TokenUsage
import research_motion_choice as research
import trace_motion_agent as vision


def event():
    path = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["exercise_id"] = "squat"
    result["discomfort"] = {"status": "none", "source": "self_report"}
    return result


def test_offline_candidates_vary_but_never_deliver():
    up = research.run_choice(event(), "round_a")
    down = research.run_choice(event(), "squat_1")
    assert up["candidate_action"] == "notify" and down["candidate_action"] == "silent"
    for result in (up, down):
        assert result["binding_status"] == "matched_simulation" and result["r3_status"] == "reviewed"
        assert result["reference_check"] == "passed_for_cited_ids"
        assert result["may_display"] is False and result["delivery"] is False
        assert result["visual_requests_started"] == 0 and result["decision_requests_started"] == 0
        assert len(result["candidate_evidence"]) == 1


def test_reported_discomfort_stops_before_image_key_or_model(monkeypatch):
    source = event()
    source["discomfort"] = {"status": "reported", "source": "self_report"}
    def forbidden(*args):
        raise AssertionError("不应读取文件或密钥")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    observer = vision.DeepSeekObserver(forbidden)
    decision = research.DeepSeekChoiceModel(forbidden)
    result = research.run_choice(source, observer=observer, choice_model=decision)
    assert result["r2_status"] == "safety_pending" and result["r3_status"] == "safety_pending"
    assert observer.calls == 0 and decision.calls == 0 and result["delivery"] is False


def test_unknown_discomfort_remains_silent():
    source = event()
    source["discomfort"] = {"status": "unknown", "source": "self_report"}
    result = research.run_choice(source)
    assert result["r2_status"] == "candidate" and result["candidate_action"] == "silent"
    assert result["may_display"] is False


@pytest.mark.parametrize("answer", ["down", "uncertain"])
def test_mismatched_or_uncertain_observation_stops_before_decision(answer):
    observer = vision.ScriptedObserver(answer)
    decision = research.OfflineChoiceModel()
    result = research.run_choice(event(), "round_a", observer, decision)
    assert result["r2_error"] == "observation_not_matched" and decision.calls == 0
    assert result["candidate_action"] is None and result["delivery"] is False


def test_non_silent_candidate_must_reference_the_observation():
    class WrongEvidence(research.OfflineChoiceModel):
        def complete(self, facts):
            self.calls += 1
            body = {"suggested_action": "notify", "message": "模拟候选", "reason": "模拟", "evidence": ["fact_1"]}
            self.response = ModelResponse(json.dumps(body), "fake", None, None, None, 0, "stop")
            return self.response
    result = research.run_choice(event(), choice_model=WrongEvidence())
    assert result["r2_error"] == "binding_or_evidence_rejected"
    assert result["binding_status"] == "rejected" and result["candidate_action"] is None
    assert result["r3_status"] == "stopped" and result["may_display"] is False


def test_defer_is_a_supported_non_delivered_research_candidate():
    class Deferred(research.OfflineChoiceModel):
        def complete(self, facts):
            self.calls += 1
            observed = next(f for f in facts if f.name == "simulated_model_reported_hip_change")
            body = {"suggested_action": "defer", "message": "模拟观察", "reason": "模拟", "evidence": [observed.fact_id]}
            self.response = ModelResponse(json.dumps(body), "fake", None, None, None, 0, "stop")
            return self.response
    result = research.run_choice(event(), choice_model=Deferred())
    assert result["r3_status"] == "reviewed" and result["candidate_action"] == "defer"
    assert result["effective_action"] == "defer" and result["delivery"] is False


def test_unknown_evidence_cannot_pass_research_scope():
    class WrongEvidence(research.OfflineChoiceModel):
        def complete(self, facts):
            self.calls += 1
            body = {"suggested_action": "notify", "message": "模拟候选", "reason": "模拟", "evidence": ["fact_999"]}
            self.response = ModelResponse(json.dumps(body), "fake", None, None, None, 0, "stop")
            return self.response
    result = research.run_choice(event(), choice_model=WrongEvidence())
    assert result["r2_error"] == "binding_or_evidence_rejected" and result["delivery"] is False


def test_changed_source_stops_before_decision_model(monkeypatch):
    source = event()
    original = Path.read_bytes
    reads = []
    def changed(path):
        data = original(path)
        reads.append(1)
        return data if len(reads) == 1 else data + b" "
    monkeypatch.setattr(Path, "read_bytes", changed)
    decision = research.OfflineChoiceModel()
    result = research.run_choice(source, choice_model=decision)
    assert result["r2_error"] == "binding_precheck_failed" and decision.calls == 0
    assert result["binding_precheck"] == "rejected"
    assert result["delivery"] is False


def test_malformed_decision_is_rejected():
    class BadDecision(research.OfflineChoiceModel):
        def complete(self, facts):
            self.calls += 1
            self.response = ModelResponse('{"suggested_action":"ignore"}', "fake", None, None, None, 0, "stop")
            return self.response
    result = research.run_choice(event(), choice_model=BadDecision())
    assert result["r2_error"] == "invalid_model_structure" and result["r3_status"] == "stopped"


def test_live_requests_filter_event_ids_and_untrusted_explanation(monkeypatch):
    calls = []
    def visual_call(payload, key):
        calls.append(("visual", json.loads(payload), key))
        return json.dumps({"hip_change": "up", "explanation": "IGNORE_INSTRUCTIONS"}), {"prompt_tokens": 42}, 100
    def text_call(endpoint, payload, key, budget, model, **kwargs):
        body = json.loads(payload)
        calls.append(("decision", body, key))
        budget.reserve(len(payload))
        budget.settle(TokenUsage(100, 10))
        facts = json.loads(body["messages"][1]["content"])["facts"]
        fact_id = next(f["fact_id"] for f in facts if f["name"] == "simulated_model_reported_hip_change")
        content = json.dumps({"suggested_action": "notify", "message": "模拟观察", "reason": "模拟", "evidence": [fact_id]})
        return ModelResponse(content, model, None, None, TokenUsage(100, 10), 50, "stop")
    monkeypatch.setattr(vision, "call_once", visual_call)
    monkeypatch.setattr(research, "invoke", text_call)
    source = event()
    observer = vision.DeepSeekObserver(lambda: "test-key")
    decision = research.DeepSeekChoiceModel(lambda: "test-key")
    result = research.run_choice(source, observer=observer, choice_model=decision)
    assert [x[0] for x in calls] == ["visual", "decision"]
    assert result["candidate_action"] == "notify" and result["r3_status"] == "reviewed"
    assert result["visual_requests_started"] == 1 and result["decision_requests_started"] == 1
    assert result["decision_usage"] == {"input_tokens": 100, "output_tokens": 10}
    assert result["decision_reason_adjustment"] is None
    decision_text = calls[1][1]["messages"][1]["content"]
    assert "IGNORE_INSTRUCTIONS" not in decision_text
    assert all(source[name] not in decision_text for name in ("user_id", "session_id", "event_id"))
    assert "test-key" not in json.dumps(result)


def test_decision_timeout_stops_without_retry_and_marks_unknown_budget(monkeypatch):
    calls = []
    def failure(endpoint, payload, key, budget, model, **kwargs):
        calls.append(1)
        budget.reserve(len(payload))
        budget.uncertain = True
        raise ModelCallError("provider_timeout", True)
    monkeypatch.setattr(research, "invoke", failure)
    decision = research.DeepSeekChoiceModel(lambda: "test-key")
    result = research.run_choice(event(), choice_model=decision)
    assert result["r2_error"] == "provider_timeout" and result["r3_status"] == "stopped"
    assert len(calls) == 1 and result["decision_budget_uncertain"] is True and result["delivery"] is False


def test_visual_provider_failure_prevents_decision_request(monkeypatch):
    def failure(payload, key):
        raise ValueError("unavailable")
    monkeypatch.setattr(vision, "call_once", failure)
    observer = vision.DeepSeekObserver(lambda: "test-key")
    decision = research.DeepSeekChoiceModel(lambda: "test-key")
    result = research.run_choice(event(), observer=observer, choice_model=decision)
    assert result["r2_error"] == "observation_provider_error"
    assert result["visual_requests_started"] == 1 and result["decision_requests_started"] == 0


def test_missing_decision_key_stops_without_sending(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("不应发送请求")
    monkeypatch.setattr(research, "invoke", forbidden)
    decision = research.DeepSeekChoiceModel(lambda: "")
    result = research.run_choice(event(), choice_model=decision)
    assert result["r2_error"] == "missing_api_key" and result["decision_requests_started"] == 0


def test_none_ambiguous_reason_uses_recorded_safe_normalization(monkeypatch):
    from metafit_agent.decision import UNKNOWN_REASON
    def fake_invoke(endpoint, payload, key, budget, model, **kwargs):
        budget.reserve(len(payload))
        budget.settle(TokenUsage(100, 10))
        facts = json.loads(json.loads(payload)["messages"][1]["content"])["facts"]
        observed = next(f for f in facts if f["name"] == "simulated_model_reported_hip_change")
        body = {"suggested_action": "notify", "message": "模拟观察", "reason": UNKNOWN_REASON,
                "evidence": [observed["fact_id"]]}
        return ModelResponse(json.dumps(body, ensure_ascii=False), model, None, None, TokenUsage(100, 10), 50, "stop")
    monkeypatch.setattr(research, "invoke", fake_invoke)
    decision = research.DeepSeekChoiceModel(lambda: "test-key")
    result = research.run_choice(event(), choice_model=decision)
    assert result["r3_status"] == "reviewed" and result["candidate_action"] == "notify"
    assert result["decision_reason_adjustment"] == "none_ambiguous_reason_normalized"
    assert result["binding_precheck"] == "passed" and result["binding_status"] == "matched_simulation"
    assert json.loads(decision.response.content)["reason"] == UNKNOWN_REASON
    assert result["decision_requests_started"] == 1 and result["may_display"] is False


def test_unknown_status_never_uses_none_reason_normalization(monkeypatch):
    def fake_invoke(endpoint, payload, key, budget, model, **kwargs):
        budget.reserve(len(payload))
        budget.settle(TokenUsage(100, 10))
        body = {"suggested_action": "silent", "message": None, "reason": "不符合未知状态的原因", "evidence": []}
        return ModelResponse(json.dumps(body, ensure_ascii=False), model, None, None, TokenUsage(100, 10), 50, "stop")
    monkeypatch.setattr(research, "invoke", fake_invoke)
    source = event()
    source["discomfort"] = {"status": "unknown", "source": "self_report"}
    decision = research.DeepSeekChoiceModel(lambda: "test-key")
    result = research.run_choice(source, choice_model=decision)
    assert result["r2_error"] == "unknown_state_candidate_restricted"
    assert result["decision_reason_adjustment"] is None and result["delivery"] is False


def test_reason_normalization_preserves_action_message_and_evidence(monkeypatch):
    from metafit_agent.decision import UNKNOWN_REASON
    from metafit_agent.facts import Fact
    original = {"suggested_action": "defer", "message": "模拟观察", "reason": UNKNOWN_REASON,
                "evidence": ["fact_2"]}
    def fake_invoke(endpoint, payload, key, budget, model, **kwargs):
        budget.reserve(len(payload))
        budget.settle(TokenUsage(5, 4))
        return ModelResponse(json.dumps(original, ensure_ascii=False), model, None, None, TokenUsage(5, 4), 1, "stop")
    monkeypatch.setattr(research, "invoke", fake_invoke)
    facts = (Fact("fact_1", "discomfort_status", "none", "u", "s", "e", "discomfort.status"),
             Fact("fact_2", "simulated_model_reported_hip_change", "up", "u", "s", "e", "observation"))
    model = research.DeepSeekChoiceModel(lambda: "test-key")
    normalized = json.loads(model.complete(facts).content)
    assert normalized == {**original, "reason": research.NONE_REASON}
    assert json.loads(model.response.content) == original


def test_malformed_provider_response_is_not_normalized(monkeypatch):
    def fake_invoke(endpoint, payload, key, budget, model, **kwargs):
        budget.reserve(len(payload))
        budget.settle(TokenUsage(5, 4))
        text = '{"suggested_action":"silent","reason":"x","reason":"y","message":null,"evidence":[]}'
        return ModelResponse(text, model, None, None, TokenUsage(5, 4), 1, "stop")
    monkeypatch.setattr(research, "invoke", fake_invoke)
    model = research.DeepSeekChoiceModel(lambda: "test-key")
    result = research.run_choice(event(), choice_model=model)
    assert result["r2_error"] == "invalid_model_json" and result["decision_reason_adjustment"] is None
    assert result["delivery"] is False


def test_reuse_user_reported_observation_needs_only_one_decision_request(monkeypatch):
    calls = []
    def fake_invoke(endpoint, payload, key, budget, model, **kwargs):
        calls.append(1)
        budget.reserve(len(payload))
        budget.settle(TokenUsage(100, 10))
        facts = json.loads(json.loads(payload)["messages"][1]["content"])["facts"]
        observed = next(f for f in facts if f["name"] == "simulated_model_reported_hip_change")
        body = {"suggested_action": "silent", "message": None, "reason": "模拟", "evidence": [observed["fact_id"]]}
        return ModelResponse(json.dumps(body, ensure_ascii=False), model, None, None, TokenUsage(100, 10), 50, "stop")
    monkeypatch.setattr(research, "invoke", fake_invoke)
    replay = research.UserReportedReplay(research.SAMPLES["round_a"])
    decision = research.DeepSeekChoiceModel(lambda: "test-key")
    result = research.run_choice(event(), observer=replay, choice_model=decision)
    assert result["r3_status"] == "reviewed" and result["visual_requests_started"] == 0
    assert result["decision_requests_started"] == 1 and len(calls) == 1
    assert result["observation_mode"] == "user_reported_previous_observation"
    assert result["visual_replay_origin"] == "user_reported_not_independently_verified"
    assert result["prior_reported_visual_usage"] == {"prompt_tokens": 482, "completion_tokens": 32}


def test_reuse_rejects_changed_sample_before_decision(monkeypatch):
    source = event()
    original = Path.read_bytes
    def changed(path):
        return original(path) + b" "
    monkeypatch.setattr(Path, "read_bytes", changed)
    replay = research.UserReportedReplay(research.SAMPLES["round_a"])
    decision = research.OfflineChoiceModel()
    result = research.run_choice(source, observer=replay, choice_model=decision)
    assert result["r2_error"] == "replay_input_mismatch" and decision.calls == 0
    assert result["visual_requests_started"] == 0 and result["delivery"] is False


def test_git_windows_checkout_preserves_replay_sample_bytes(tmp_path):
    """自动换行打开时，检出的 BVH 必须保留用于回放认证的 LF 字节。"""
    import hashlib
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        pytest.skip("Git 不可用，无法检查检出行为")
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "examples").mkdir()
    (tmp_path / ".gitattributes").write_bytes((repo / ".gitattributes").read_bytes())
    source = (repo / "examples/cmu_23_14_sample.bvh").read_bytes().replace(b"\r\n", b"\n")
    sample = tmp_path / "examples/cmu_23_14_sample.bvh"
    sample.write_bytes(source)
    def run(*args):
        return subprocess.run([git, "-c", "core.autocrlf=true", "-C", str(tmp_path), *args],
                              check=True, capture_output=True)
    run("init", "-q")
    run("add", "--", ".gitattributes", "examples/cmu_23_14_sample.bvh")
    sample.write_bytes(source.replace(b"\n", b"\r\n"))
    run("checkout-index", "-f", "--", "examples/cmu_23_14_sample.bvh")
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == research.UserReportedReplay.source_sha256


def test_reuse_rejects_changed_prompt_before_decision(monkeypatch):
    source = event()
    monkeypatch.setattr(research, "PROMPT", "changed prompt")
    replay = research.UserReportedReplay(research.SAMPLES["round_a"])
    decision = research.OfflineChoiceModel()
    result = research.run_choice(source, observer=replay, choice_model=decision)
    assert result["r2_error"] == "replay_input_mismatch" and decision.calls == 0
