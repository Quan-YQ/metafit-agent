"""骨架图实验接入 R2/R3 的离线边界测试。"""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from metafit_agent.model_response import ModelResponse
from metafit_agent.decision import ModelCallError
import deepseek_motion_shadow as provider
import motion_shadow_agent as shadow
from motion_shadow_agent import run_shadow


def event():
    result = json.loads((Path(__file__).resolve().parents[1] / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    result["discomfort"] = {"status": "none", "source": "self_report"}
    return result


def test_normal_simulation_delivers_image_and_facts_without_display():
    class ObservingStandIn:
        def complete(self, facts, image):
            assert image.startswith(b"\x89PNG\r\n\x1a\n")
            assert any(f.name == "completed_reps" for f in facts)
            body = json.dumps({"suggested_action": "silent", "message": None,
                               "reason": "研究替身返回", "evidence": []})
            return ModelResponse(body, "test", None, None, None, 0.0, "stop")
    result = run_shadow(event(), "round_b", ObservingStandIn())
    assert result["visual_model_attempts"] == 1 and len(result["image_sha256"]) == 64
    assert result["r2_status"] == "candidate" and result["r3_status"] == "reviewed"
    assert result["effective_action"] == "silent" and result["may_display"] is False


def test_discomfort_and_invalid_event_never_render_or_invoke():
    def forbidden(_):
        raise AssertionError("不应处理图像")
    discomfort = event()
    discomfort["discomfort"] = {"status": "reported", "source": "self_report"}
    result = run_shadow(discomfort, "round_a", renderer=forbidden)
    assert result["r2_status"] == "safety_pending" and result["visual_model_attempts"] == 0
    invalid = event()
    invalid.pop("user_id")
    result = run_shadow(invalid, "round_a", renderer=forbidden)
    assert result["r2_error"] == "invalid_event" and result["visual_model_attempts"] == 0


def test_non_silent_output_is_blocked_even_if_structurally_valid():
    class UnsafeStandIn:
        def complete(self, facts, image):
            body = json.dumps({"suggested_action": "notify", "message": "模拟反馈",
                               "reason": "模拟", "evidence": [facts[0].fact_id]})
            return ModelResponse(body, "test", None, None, None, 0.0, "stop")
    result = run_shadow(event(), "round_b", UnsafeStandIn())
    assert result["r2_error"] == "shadow_non_silent_blocked"
    assert result["r3_status"] == "stopped" and result["may_display"] is False


def test_image_failure_prevents_model_call():
    class FailIfCalled:
        def complete(self, facts, image):
            raise AssertionError("不应调用模型")
    result = run_shadow(event(), "round_b", FailIfCalled(), renderer=lambda _: b"bad")
    assert result["r2_error"] == "invalid_motion_image" and result["visual_model_attempts"] == 0


def test_live_shadow_builds_image_request_and_filters_identifiers(monkeypatch):
    captured = []
    def fake_call(payload, key):
        captured.append((json.loads(payload), key))
        body = json.dumps({"suggested_action": "silent", "message": None,
                           "reason": "模拟检查", "evidence": []})
        return body, {"prompt_tokens": 421, "completion_tokens": 30}, 123.0
    monkeypatch.setattr(provider, "call_once", fake_call)
    model = provider.DeepSeekShadowModel(lambda: "test-key")
    source = event()
    result = run_shadow(source, "round_a", model)
    assert result["r2_status"] == "candidate" and result["r3_status"] == "reviewed"
    assert result["may_display"] is False and result["provider_requests_started"] == 1
    assert result["model_usage"] == {"prompt_tokens": 421, "completion_tokens": 30}
    body, key = captured[0]
    assert key == "test-key" and body["model"] == "deepseek-flash"
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["response_format"] == {"type": "json_object"}
    text = body["messages"][1]["content"][0]["text"]
    for name in ("user_id", "event_id", "session_id"):
        assert source[name] not in text
    with pytest.raises(ModelCallError, match="shadow_call_limit"):
        model.complete((), b"\x89PNG\r\n\x1a\n")
    assert len(captured) == 1


def test_live_mode_with_discomfort_does_not_request_key_or_send(monkeypatch):
    def fail(*args):
        raise AssertionError("不应读取密钥或发送请求")
    monkeypatch.setattr(provider, "call_once", fail)
    model = provider.DeepSeekShadowModel(fail)
    source = event()
    source["discomfort"] = {"status": "reported", "source": "self_report"}
    result = run_shadow(source, "round_b", model)
    assert result["r2_status"] == "safety_pending" and model.calls == 0
    assert result["image_sha256"] is None


def test_live_rejected_candidate_keeps_usage_and_never_retries(monkeypatch):
    def fake_call(payload, key):
        body = json.dumps({"suggested_action": "notify", "message": "模拟反馈",
                           "reason": "模拟", "evidence": ["fact_1"]})
        return body, {"prompt_tokens": 10, "completion_tokens": 8}, 200.0
    monkeypatch.setattr(provider, "call_once", fake_call)
    model = provider.DeepSeekShadowModel(lambda: "test-key")
    result = run_shadow(event(), "round_a", model)
    assert result["r2_error"] == "shadow_non_silent_blocked" and result["provider_requests_started"] == 1
    assert result["model_usage"] == {"prompt_tokens": 10, "completion_tokens": 8}


def test_live_provider_error_is_uncertain_and_not_retried(monkeypatch):
    calls = []
    def fail(payload, key):
        calls.append(1)
        raise ValueError("不回显的远端异常")
    monkeypatch.setattr(provider, "call_once", fail)
    model = provider.DeepSeekShadowModel(lambda: "test-key")
    result = run_shadow(event(), "round_b", model)
    assert result["r2_error"] == "shadow_provider_error" and len(calls) == 1
    assert result["provider_requests_started"] == 1 and result["model_usage"] is None


def test_cli_uses_hypothetical_squat_event_without_unrelated_facts(monkeypatch, capsys):
    captured = []
    def fake_run(event_data, sample, model):
        captured.append((event_data, sample))
        return {"r3_status": "reviewed"}
    monkeypatch.setattr(shadow, "run_shadow", fake_run)
    monkeypatch.setattr(sys, "argv", ["motion_shadow_agent.py"])
    assert shadow.main() == 0
    current, sample = captured[0]
    assert sample == "round_a" and current["exercise_id"] == "squat"
    assert all(name not in current for name in ("planned_reps", "completed_reps", "weight_kg_per_hand", "metrics"))


def test_live_provider_error_is_uncertain_and_not_retried(monkeypatch):
    calls = []
    def fail(payload, key):
        calls.append(1)
        raise ValueError("不回显的远端异常")
    monkeypatch.setattr(provider, "call_once", fail)
    model = provider.DeepSeekShadowModel(lambda: "test-key")
    result = run_shadow(event(), "round_b", model)
    assert result["r2_error"] == "shadow_provider_error" and len(calls) == 1
    assert result["provider_requests_started"] == 1 and result["model_usage"] is None
