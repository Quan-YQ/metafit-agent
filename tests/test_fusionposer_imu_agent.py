"""公开 IMU 研究链的边界；所有网络请求均为替身。"""
import base64
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import fusionposer_imu_agent as pipeline
import research_motion_choice as choice
import trace_motion_agent as vision
from metafit_agent.decision import ModelCallError
from metafit_agent.model_response import ModelResponse, TokenUsage


def event():
    data = json.loads((Path(__file__).resolve().parents[1] / "examples/set_completed.json").read_text(encoding="utf-8"))
    data["discomfort"] = {"status": "none", "source": "self_report"}
    return data


@pytest.fixture
def fake_experiment(monkeypatch):
    image = b"\x89PNG\r\n\x1a\nPREDICTED"
    report = {"x_sha256": "x", "y_sha256": "y", "selected_original_rows": [1, 2, 3, 4, 5, 6],
              "predicted_hip_reference": {"direction": "no_clear_change"}}
    monkeypatch.setattr(pipeline, "experiment", lambda: (dict(report), image, b"OPTICAL_REFERENCE"))
    return report, image


def test_offline_pipeline_is_bound_and_never_delivers(fake_experiment):
    result = pipeline.run_pipeline(event())
    assert result["binding_precheck"] == "passed"
    assert result["binding_status"] == "matched_simulation"
    assert result["r3_status"] == "reviewed"
    assert result["may_display"] is False and result["delivery"] is False
    assert result["visual_requests_started"] == result["decision_requests_started"] == 0


@pytest.mark.parametrize("invalid", [False, True])
def test_discomfort_or_invalid_event_stops_before_reading_and_calling(monkeypatch, invalid):
    def forbidden():
        raise AssertionError("must not read sensor data")
    monkeypatch.setattr(pipeline, "experiment", forbidden)
    data = event()
    if invalid:
        data["schema_version"] = "invalid"
    else:
        data["discomfort"] = {"status": "reported", "source": "self_report"}
    observer, decision = vision.ScriptedObserver(), choice.OfflineChoiceModel()
    result = pipeline.run_pipeline(data, observer, decision)
    assert result["r2_status"] == ("error" if invalid else "safety_pending")
    assert observer.calls == decision.calls == 0 and result["delivery"] is False


@pytest.mark.parametrize("answer", ["up", "uncertain"])
def test_mismatch_or_uncertainty_stops_before_decision(fake_experiment, answer):
    decision = choice.OfflineChoiceModel()
    result = pipeline.run_pipeline(event(), vision.ScriptedObserver(answer), decision)
    assert result["r2_error"] == "observation_not_matched"
    assert decision.calls == 0 and result["delivery"] is False


def test_changed_data_fails_binding_before_decision(fake_experiment, monkeypatch):
    report, image = fake_experiment
    reads = []
    def changed():
        reads.append(1)
        return ({**report, "x_sha256": "changed"} if len(reads) > 1 else dict(report), image, b"reference")
    monkeypatch.setattr(pipeline, "experiment", changed)
    decision = choice.OfflineChoiceModel()
    result = pipeline.run_pipeline(event(), decision=decision)
    assert result["r2_error"] == "binding_precheck_failed" and decision.calls == 0
    assert result["binding_precheck"] == "rejected"


def test_non_silent_candidate_must_cite_observation(fake_experiment):
    class BadEvidence(choice.OfflineChoiceModel):
        def complete(self, facts):
            self.calls += 1
            self.response = ModelResponse(json.dumps({"suggested_action": "notify", "message": "simulation",
                                                      "reason": "simulation", "evidence": ["fact_999"]}),
                                          "fake", None, None, None, 0, "stop")
            return self.response
    result = pipeline.run_pipeline(event(), decision=BadEvidence())
    assert result["r2_error"] == "binding_or_evidence_rejected"
    assert result["r3_status"] == "stopped" and result["delivery"] is False


def test_mocked_live_sends_only_prediction_and_filtered_facts(fake_experiment, monkeypatch):
    calls = []
    def visual_call(payload, key):
        calls.append(("visual", json.loads(payload)))
        return json.dumps({"hip_change": "no_clear_change", "explanation": "UNTRUSTED_TEXT"}), {"prompt_tokens": 1}, 1
    def text_call(endpoint, payload, key, budget, model, **kwargs):
        body = json.loads(payload)
        calls.append(("decision", body))
        budget.reserve(len(payload))
        budget.settle(TokenUsage(3, 2))
        facts = json.loads(body["messages"][1]["content"])["facts"]
        observed = next(f for f in facts if f["name"] == "simulated_model_reported_hip_change")
        content = json.dumps({"suggested_action": "silent", "message": None, "reason": "simulation", "evidence": [observed["fact_id"]]})
        return ModelResponse(content, model, None, None, TokenUsage(3, 2), 1, "stop")
    monkeypatch.setattr(vision, "call_once", visual_call)
    monkeypatch.setattr(choice, "invoke", text_call)
    source = event()
    result = pipeline.run_pipeline(source, vision.DeepSeekObserver(lambda: "test-key"), choice.DeepSeekChoiceModel(lambda: "test-key"))
    assert [item[0] for item in calls] == ["visual", "decision"]
    serialized_image_request = json.dumps(calls[0][1])
    assert base64.b64encode(fake_experiment[1]).decode() in serialized_image_request
    assert base64.b64encode(b"OPTICAL_REFERENCE").decode() not in serialized_image_request
    decision_text = calls[1][1]["messages"][1]["content"]
    assert "UNTRUSTED_TEXT" not in decision_text
    assert all(source[name] not in decision_text for name in ("user_id", "session_id", "event_id"))
    assert "test-key" not in json.dumps(result)
    assert result["r3_status"] == "reviewed" and result["delivery"] is False


def test_provider_failure_does_not_retry_or_call_decision(fake_experiment, monkeypatch):
    calls = []
    def fail(payload, key):
        calls.append(1)
        raise ValueError("network failure")
    monkeypatch.setattr(vision, "call_once", fail)
    decision = choice.DeepSeekChoiceModel(lambda: "test-key")
    result = pipeline.run_pipeline(event(), vision.DeepSeekObserver(lambda: "test-key"), decision)
    assert calls == [1] and decision.calls == 0
    assert result["r2_error"] == "observation_provider_error" and result["delivery"] is False
