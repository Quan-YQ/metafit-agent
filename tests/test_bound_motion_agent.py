"""模拟绑定、事实引用及不适优先；不证明真实身份或动作安全。"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from metafit_agent.decision import decide_simulated_event
import bound_motion_agent as bound
import trace_motion_agent as visual


def event():
    path = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["exercise_id"] = "squat"
    result["discomfort"] = {"status": "none", "source": "self_report"}
    return result


def construct():
    path = bound.SAMPLES["round_a"]
    gate = bound.ObservationGate(path, visual.ScriptedObserver("up"))
    decision = bound.BoundSilentDecision(gate)
    gate.decision_model = decision
    outcome = decide_simulated_event(event(), gate)
    assert outcome.status == "candidate"
    return path, gate, decision, outcome


def test_simulated_observation_is_bound_and_cited_in_silent_r3_result():
    result = bound.run_bound(event())
    assert result["binding_status"] == "matched_simulation"
    assert result["binding_association"] == "synthetic_not_real_world_proof"
    assert result["cited_fact"]["name"] == "simulated_model_reported_hip_change"
    assert result["cited_fact"]["value"] == "up"
    assert result["r2_status"] == "candidate" and result["r3_status"] == "reviewed"
    assert result["reference_check"] == "passed_for_cited_ids"
    assert result["effective_action"] == "silent" and result["may_display"] is False


def test_binding_rejects_changed_event_sample_observation_or_reference():
    path, gate, decision, outcome = construct()
    raw = path.read_bytes()
    def check(facts=outcome.facts, scope=outcome.scope, obs=gate.observation,
              motion=decision.motion_fact, data=raw, seal=decision.binding_sha256):
        return bound.verify_binding(decision.binding, seal, scope, facts, obs, motion, data, path.name)
    assert check()
    changed_facts = (replace(outcome.facts[0], event_id="other_event"),) + outcome.facts[1:]
    assert not check(facts=changed_facts)
    assert not check(scope=replace(outcome.scope, event_id="other_event"))
    assert not check(motion=replace(decision.motion_fact, fact_id="fact_999"))
    assert not check(motion=replace(decision.motion_fact, user_id="other_user"))
    assert not check(motion=replace(decision.motion_fact, value="down"))
    assert not check(data=raw + b" ")
    assert not check(obs={**gate.observation, "model_hip_change": "down"})
    assert not check(seal="0" * 64)


def test_source_changed_after_model_observation_is_rejected(monkeypatch):
    original = Path.read_bytes
    count = []
    def changed(path):
        value = original(path)
        count.append(1)
        return value if len(count) == 1 else value + b" "
    monkeypatch.setattr(Path, "read_bytes", changed)
    result = bound.run_bound(event())
    assert len(count) == 2 and result["binding_status"] == "rejected"
    assert result["r2_error"] == "simulated_binding_mismatch" and result["r3_status"] == "stopped"
    assert result["cited_fact"] is None and result["may_display"] is False


@pytest.mark.parametrize("answer", ["down", "uncertain"])
def test_unreliable_observation_never_reaches_research_decision(answer):
    observer = visual.ScriptedObserver(answer)
    result = bound.run_bound(event(), observer=observer)
    assert result["r2_error"] == "observation_not_matched"
    assert result["decision_attempts"] == 0 and result["cited_fact"] is None
    assert result["binding_status"] == "not_checked" and result["may_display"] is False


def test_discomfort_and_invalid_event_stop_before_image_or_network(monkeypatch):
    def forbidden(*args):
        raise AssertionError("不应读取图片或密钥")
    source = event()
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    observer = visual.DeepSeekObserver(forbidden)
    source["discomfort"] = {"status": "reported", "source": "self_report"}
    result = bound.run_bound(source, observer=observer)
    assert result["r2_status"] == "safety_pending" and observer.calls == 0
    assert result["observation"] is None and result["cited_fact"] is None
    source.pop("user_id")
    source.pop("discomfort")
    result = bound.run_bound(source, observer=observer)
    assert result["r2_error"] == "invalid_event" and observer.calls == 0


def test_unknown_discomfort_remains_silent_and_reference_scoped():
    source = event()
    source["discomfort"] = {"status": "unknown", "source": "self_report"}
    result = bound.run_bound(source)
    assert result["r2_status"] == "candidate" and result["reference_check"] == "passed_for_cited_ids"
    assert result["effective_action"] == "silent" and result["may_display"] is False


def test_live_transport_is_single_call_and_does_not_send_event_ids(monkeypatch):
    captured = []
    def fake_call(payload, key):
        captured.append((json.loads(payload), key))
        response = json.dumps({"hip_change": "up", "explanation": "模拟服务"})
        return response, {"prompt_tokens": 40, "completion_tokens": 10}, 123
    monkeypatch.setattr(visual, "call_once", fake_call)
    observer = visual.DeepSeekObserver(lambda: "test-key")
    source = event()
    result = bound.run_bound(source, observer=observer)
    assert result["binding_status"] == "matched_simulation" and observer.calls == 1
    assert result["model_usage"] == {"prompt_tokens": 40, "completion_tokens": 10}
    request, key = captured[0]
    assert key == "test-key" and request["model"] == "deepseek-flash"
    text = request["messages"][0]["content"][0]["text"]
    assert all(source[name] not in text for name in ("user_id", "session_id", "event_id"))
    assert "test-key" not in json.dumps(result)


def test_remote_failure_has_one_attempt_and_unknown_usage(monkeypatch):
    calls = []
    def fail(payload, key):
        calls.append(1)
        raise ValueError("provider failed")
    monkeypatch.setattr(visual, "call_once", fail)
    observer = visual.DeepSeekObserver(lambda: "test-key")
    result = bound.run_bound(event(), observer=observer)
    assert result["r2_error"] == "observation_provider_error" and len(calls) == 1
    assert result["model_usage"] is None and result["decision_attempts"] == 0
