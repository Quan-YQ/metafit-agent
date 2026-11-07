"""智谱 GLM 客户端的离线测试：不需要密钥或费用，也不访问网络。"""

from decimal import Decimal
import json
import sys
from urllib.error import HTTPError, URLError

import pytest

from metafit_agent.decision import UNKNOWN_REASON, decide_simulated_event
from metafit_agent.glm_client import ENDPOINT, GLMModel, KEY_ENV, MAX_OUTPUT_TOKENS
from metafit_agent.model_response import ModelResponse


@pytest.fixture
def event():
    return {"schema_version": "0.1", "event_id": "private_evt", "event_type": "set_completed",
            "user_id": "private_user", "session_id": "private_session", "set_id": "set_1",
            "exercise_id": "squat", "timestamp": "2026-11-02T18:30:00+08:00", "source": "simulation",
            "planned_reps": 12, "completed_reps": 10, "set_index": 2,
            "discomfort": {"status": "unknown", "source": "self_report"}}


class _Response:
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, limit):
        return self.body[:limit]


class _Opener:
    def __init__(self, response):
        self.response = response
        self.requests = []
    def open(self, req, timeout):
        self.requests.append((req, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return _Response(self.response)


def _provider_response(content, finish_reason="stop"):
    return json.dumps({"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}).encode()


def candidate():
    return json.dumps({"suggested_action": "silent", "message": None,
                       "reason": UNKNOWN_REASON, "evidence": []}, ensure_ascii=False)


def _wire(monkeypatch, opener, key="test-only-secret"):
    monkeypatch.setenv(KEY_ENV, key)
    monkeypatch.setattr("metafit_agent.glm_client.request.build_opener", lambda *_: opener)


def test_glm_request_is_openai_compatible_and_sends_only_allowlisted_facts(event, monkeypatch):
    opener = _Opener(_provider_response(candidate()))
    _wire(monkeypatch, opener)
    outcome = decide_simulated_event(event, GLMModel())
    assert outcome.status == "candidate" and len(opener.requests) == 1
    req, timeout = opener.requests[0]
    body = json.loads(req.data.decode())
    assert req.full_url == ENDPOINT and timeout == 120
    assert req.get_header("Authorization") == "Bearer test-only-secret"
    assert body["model"] == "glm-5.3-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == MAX_OUTPUT_TOKENS == 8192 and body["stream"] is False
    # GLM-5.3-Flash 强制深度思考，thinking.disabled 会被服务端拒绝。
    assert "thinking" not in body
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    sent = req.data.decode()
    for private in ("private_user", "private_evt", "private_session", "test-only-secret"):
        assert private not in sent


def test_glm_model_name_can_be_overridden(event, monkeypatch):
    opener = _Opener(_provider_response(candidate()))
    _wire(monkeypatch, opener)
    assert decide_simulated_event(event, GLMModel(model="glm-5.3-flashx")).status == "candidate"
    body = json.loads(opener.requests[0][0].data.decode())
    assert body["model"] == "glm-5.3-flashx"


def test_glm_missing_key_does_not_make_network_request(event, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    result = decide_simulated_event(event, GLMModel())
    assert result.status == "error" and result.error_code == "missing_api_key"


def test_explicit_glm_prompt_key_takes_priority(event, monkeypatch):
    opener = _Opener(_provider_response(candidate()))
    monkeypatch.setenv(KEY_ENV, "old-test-key")
    monkeypatch.setattr("metafit_agent.glm_client.request.build_opener", lambda *_: opener)
    result = decide_simulated_event(event, GLMModel(key_provider=lambda: "new-test-key"))
    assert result.status == "candidate"
    assert opener.requests[0][0].get_header("Authorization") == "Bearer new-test-key"


def test_glm_usage_settles_into_shared_budget(event, monkeypatch):
    body = json.dumps({"id": "request-1", "model": "glm-5.3-flash",
                       "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                       "choices": [{"finish_reason": "stop", "message": {"content": candidate()}}]}).encode()
    _wire(monkeypatch, _Opener(body))
    model = GLMModel()
    result = decide_simulated_event(event, model)
    assert result.model_response.request_id == "request-1"
    assert result.model_response.returned_model == "glm-5.3-flash"
    assert result.model_response.usage.input_tokens == 100
    assert result.model_response.usage.output_tokens == 20
    assert model.budget.estimated_spent_yuan == Decimal("0.00036")


def test_glm_budget_reserves_room_for_thinking_output(event, monkeypatch):
    from metafit_agent.model_client import CallBudget
    opener = _Opener(_provider_response(candidate()))
    _wire(monkeypatch, opener)
    result = decide_simulated_event(event, GLMModel(budget=CallBudget(limit_yuan=Decimal("0.01"))))
    assert result.error_code == "budget_exceeded" and not opener.requests


@pytest.mark.parametrize("status,code,retryable", [
    (400, "invalid_provider_request", False), (401, "authentication_failed", False),
    (402, "insufficient_balance", False), (403, "permission_denied", False),
    (404, "provider_not_found", False), (408, "provider_timeout", True),
    (422, "invalid_provider_request", False), (429, "rate_limited", True),
    (500, "provider_unavailable", True), (302, "redirect_blocked", False),
])
def test_glm_http_errors_are_classified_without_body_or_retry(event, monkeypatch, status, code, retryable):
    opener = _Opener(HTTPError(ENDPOINT, status, "private details", {}, None))
    _wire(monkeypatch, opener)
    result = decide_simulated_event(event, GLMModel())
    assert result.error_code == code and result.retryable is retryable
    assert len(opener.requests) == 1 and "private details" not in str(result)


@pytest.mark.parametrize("exc", [TimeoutError("secret detail"), URLError(TimeoutError("secret detail"))])
def test_glm_timeout_is_distinct_and_never_retried(event, monkeypatch, exc):
    opener = _Opener(exc)
    _wire(monkeypatch, opener)
    result = decide_simulated_event(event, GLMModel())
    assert result.error_code == "provider_timeout" and result.retryable is True
    assert len(opener.requests) == 1 and "secret detail" not in str(result)


@pytest.mark.parametrize("body,expected", [
    (_provider_response(" "), "empty_model_output"),
    (_provider_response("{}", "length"), "incomplete_model_output"),
    (b"not json", "invalid_provider_response"),
    (b"x" * 131073, "response_too_large"),
    (URLError("private error details"), "provider_connection_error"),
], ids=["blank", "truncated", "bad_json", "oversized", "connection"])
def test_glm_provider_failures_are_sanitized(event, monkeypatch, body, expected):
    opener = _Opener(body)
    _wire(monkeypatch, opener)
    result = decide_simulated_event(event, GLMModel())
    assert result.status == "error" and result.error_code == expected
    assert len(opener.requests) == 1
    assert "test-only-secret" not in str(result)


def test_sensitive_glm_response_is_not_exposed(event, monkeypatch):
    key = "sk-testonlysecret123"
    opener = _Opener(_provider_response(candidate() + key))
    _wire(monkeypatch, opener, key=key)
    result = decide_simulated_event(event, GLMModel())
    assert result.error_code == "sensitive_model_output"
    assert result.model_response is None and key not in str(result)


def test_glm_evaluation_reports_budget_and_three_model_calls(event, monkeypatch):
    from metafit_agent.evaluate_decision import run_evaluation
    body = json.dumps({"id": "request-1", "model": "glm-5.3-flash",
                       "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                       "choices": [{"finish_reason": "stop", "message": {"content": candidate()}}]}).encode()
    _wire(monkeypatch, _Opener(body))
    context = {"source": "simulation", "context_id": "ctx_01"}
    result = run_evaluation(GLMModel(), event, context, "glm")
    assert len(result["scenarios"]) == 5
    assert result["scenarios"][0]["outcome"]["status"] == "safety_pending"
    assert result["model_calls"] == 3
    assert result["budget"]["calls"] == 3
    assert result["budget"]["estimated_spent_yuan"] == "0.00108"
    assert result["mode"] == "glm"


def test_evaluation_main_routes_provider_glm(monkeypatch):
    from metafit_agent import evaluate_decision
    from metafit_agent.model_client import CallBudget
    captured = {}

    class FakeGLM:
        def __init__(self, budget=None, key_provider=None):
            captured["budget"] = budget
            self.budget = budget

    monkeypatch.setattr(evaluate_decision, "GLMModel", FakeGLM)
    monkeypatch.setattr(evaluate_decision, "run_evaluation",
                        lambda model, base, context, mode, prior_coverage=None: {"mode": mode, "scenarios": []})
    monkeypatch.setattr(sys, "argv", ["evaluate_decision", "--live", "--provider", "glm"])
    assert evaluate_decision.main() == 0
    assert isinstance(captured["budget"], CallBudget)


def test_demo_decision_live_glm_uses_glm_model(event, monkeypatch, capsys):
    from metafit_agent import demo_decision
    from metafit_agent.model_client import CallBudget

    class FakeGLM:
        def __init__(self, key_provider=None):
            self.budget = CallBudget()

        def complete(self, facts):
            return ModelResponse(candidate(), "glm-5.3-flash", "glm-5.3-flash", None, None, 0.0, "stop")

    monkeypatch.setattr(demo_decision, "GLMModel", FakeGLM)
    monkeypatch.setattr(demo_decision, "_load_json", lambda _: event)
    monkeypatch.setattr(sys, "argv", ["demo_decision", "unused.json", "--live", "--provider", "glm"])
    assert demo_decision.main() == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "glm"
