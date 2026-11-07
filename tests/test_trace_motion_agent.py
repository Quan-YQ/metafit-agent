"""可追踪动作观察的研究门控：离线替身不理解图片。"""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from metafit_agent.model_response import ModelResponse
import trace_motion_agent as experiment


def event():
    path = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["discomfort"] = {"status": "none", "source": "self_report"}
    return result


def test_matching_observation_reaches_existing_silent_decision_with_provenance():
    captured = []
    class Decision:
        def complete(self, facts):
            captured.extend(facts)
            content = json.dumps({"suggested_action": "silent", "message": None,
                                  "reason": "离线模拟", "evidence": []})
            return ModelResponse(content, "test", None, None, None, 0, "stop")
    result = experiment.run_trace(event(), observer=experiment.ScriptedObserver("up"), decision_model=Decision())
    observation = result["observation"]
    assert observation["comparison"] == "matches_reference" and observation["model_hip_change"] == "up"
    assert all(len(observation[key]) == 64 for key in ("source_sha256", "image_sha256", "prompt_sha256"))
    assert observation["observation_id"].startswith("obs_")
    assert captured and all("motion" not in fact.name for fact in captured)
    assert result["decision_model_attempts"] == 1 and result["r3_status"] == "reviewed"
    assert result["decision_observation_id"] == observation["observation_id"]
    assert result["effective_action"] == "silent" and result["may_display"] is False
    assert result["event_image_association"] == "not_verified"


@pytest.mark.parametrize("prediction,comparison", [("down", "differs_from_reference"), ("uncertain", "inconclusive")])
def test_mismatch_or_uncertainty_stops_before_decision(prediction, comparison):
    class Forbidden:
        def complete(self, facts):
            raise AssertionError("不应执行候选决策")
    result = experiment.run_trace(event(), observer=experiment.ScriptedObserver(prediction), decision_model=Forbidden())
    assert result["observation"]["comparison"] == comparison
    assert result["r2_error"] == "observation_not_matched" and result["decision_model_attempts"] == 0
    assert result["decision_observation_id"] is None
    assert result["r3_status"] == "stopped" and result["may_display"] is False


def test_discomfort_and_invalid_event_stop_before_reading_image_or_key(monkeypatch):
    def forbidden(*args):
        raise AssertionError("不应读取图像或访问密钥")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    source = event()
    observer = experiment.DeepSeekObserver(forbidden)
    source["discomfort"] = {"status": "reported", "source": "self_report"}
    result = experiment.run_trace(source, observer=observer)
    assert result["r2_status"] == "safety_pending" and result["observation"] is None and observer.calls == 0
    source.pop("user_id")
    source.pop("discomfort")
    result = experiment.run_trace(source, observer=observer)
    assert result["r2_error"] == "invalid_event" and observer.calls == 0


def test_malformed_observation_fails_closed_with_no_decision():
    class BadObserver(experiment.ScriptedObserver):
        def observe(self, image):
            self.calls += 1
            return '{"hip_change":"up","hip_change":"down","explanation":"x"}'
    result = experiment.run_trace(event(), observer=BadObserver())
    assert result["r2_error"] == "observation_invalid_output"
    assert result["observation"]["comparison"] == "invalid_output" and result["decision_model_attempts"] == 0


def test_unknown_discomfort_stays_silent_after_matching_observation():
    source = event()
    source["discomfort"] = {"status": "unknown", "source": "self_report"}
    result = experiment.run_trace(source, observer=experiment.ScriptedObserver("up"))
    assert result["r2_status"] == "candidate" and result["effective_action"] == "silent"
    assert result["may_display"] is False
    assert result["do_not_disturb"] == {"value": False, "source": "simulation_default"}


def test_non_silent_candidate_is_blocked():
    class Decision:
        def complete(self, facts):
            content = json.dumps({"suggested_action": "notify", "message": "模拟反馈",
                                  "reason": "模拟", "evidence": [facts[0].fact_id]})
            return ModelResponse(content, "test", None, None, None, 0, "stop")
    result = experiment.run_trace(event(), observer=experiment.ScriptedObserver("up"), decision_model=Decision())
    assert result["r2_error"] == "shadow_non_silent_blocked" and result["may_display"] is False


def test_same_source_snapshot_is_used_for_reference_and_image(monkeypatch):
    original = Path.read_bytes
    reads = []
    def once(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", once)
    result = experiment.run_trace(event())
    assert result["r3_status"] == "reviewed" and len(reads) == 1


def test_reverse_changes_image_and_observation_identity():
    forward = experiment.run_trace(event(), observer=experiment.ScriptedObserver("up"))
    reverse = experiment.run_trace(event(), observer=experiment.ScriptedObserver("down"), reverse=True)
    first, last = forward["observation"], reverse["observation"]
    assert first["source_sha256"] == last["source_sha256"]
    assert first["image_sha256"] != last["image_sha256"]
    assert first["observation_id"] != last["observation_id"]


def test_live_observer_uses_one_image_request_and_keeps_usage(monkeypatch):
    captured = []
    def fake_call(payload, key):
        captured.append((json.loads(payload), key))
        return json.dumps({"hip_change": "up", "explanation": "模拟服务"}), {"prompt_tokens": 20}, 100
    monkeypatch.setattr(experiment, "call_once", fake_call)
    observer = experiment.DeepSeekObserver(lambda: "test-key")
    result = experiment.run_trace(event(), observer=observer)
    assert result["r3_status"] == "reviewed" and result["model_usage"] == {"prompt_tokens": 20}
    assert len(captured) == 1 and captured[0][0]["messages"][0]["content"][0]["text"] == experiment.PROMPT
    assert "test-key" not in json.dumps(result)


def test_provider_failure_stops_and_preserves_attempt_count(monkeypatch):
    calls = []
    def fake_call(payload, key):
        calls.append(1)
        raise ValueError("failed")
    monkeypatch.setattr(experiment, "call_once", fake_call)
    observer = experiment.DeepSeekObserver(lambda: "test-key")
    result = experiment.run_trace(event(), observer=observer)
    assert result["r2_error"] == "observation_provider_error" and len(calls) == 1
    assert result["observation_attempts"] == 1 and result["model_usage"] is None
