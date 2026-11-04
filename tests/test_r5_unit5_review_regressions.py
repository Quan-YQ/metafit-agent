"""回归测试：第五单元评审发现的契约与安全边界。"""
from copy import deepcopy
import http.client
import json
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from test_r5_simulation_api import (CONTRACT, check, create, event_data, question,
                                    request, service, submit, validate)
from metafit_agent.simulation_api import (MAX_BODY, MAX_CONCURRENT, TIMEOUT_SECONDS,
                                          CONFIRMATION_SECONDS, PUBLIC_CODES)
from metafit_agent.event_store import StorageError


def test_created_contract_example_matches_actual(service):
    example = CONTRACT["components"]["responses"]["Created"]["content"]["application/json"]["examples"]["default"]["value"]
    status, actual = create(service)
    assert status == 201 and actual == example


def test_record_state_is_consistent_for_same_event(service):
    create(service)
    status, submitted = submit(service, event_data())
    assert status == 200
    status, queried = check(service, "GET", "/sim/v0/sessions/sim_session_001/events/sim_event_001", "/sim/v0/sessions/{session_id}/events/{event_id}")
    assert status == 200 and submitted["record_state"] == queried["record_state"] == "decided"


@pytest.mark.parametrize("mutation", ["user", "session", "envelope"])
def test_unattributed_report_keeps_warning_without_changing_session(service, mutation):
    api, _ = service
    create(service)
    data = event_data()
    data["event"]["discomfort"]["status"] = "reported"
    if mutation == "user": data["event"]["user_id"] = "other_user"
    if mutation == "session": data["event"]["session_id"] = "other_session"
    if mutation == "envelope": del data["do_not_disturb"]
    status, body = request(service[1], "POST", "/sim/v0/sessions/sim_session_001/events", data)
    assert status == 400 and body["required_action"] == "report_discomfort"
    assert body["display"]["kind"] == "safety_unknown" and body["event_saved"] is False
    sample = CONTRACT["components"]["responses"]["BadRequest"]["content"]["application/json"]["examples"]["safety_report_unattributed"]["value"]
    assert body == sample
    assert not api.coordinator.view("sim_session_001").safety_reported and api.model.calls == 0
    status, reported = question(service, "answer", answer="reported")
    assert status == 200 and reported["status"] == "safety_pending"


def test_skip_unpresented_confirmation_closes_and_stays_closed(service):
    create(service)
    question(service, "open")
    status, result = question(service, "skip")
    assert status == 200 and result["confirmation_state"] == "closed"
    assert not result["can_request_confirmation"]
    assert question(service, "skip")[1]["confirmation_state"] == "closed"
    assert question(service, "open", qid="other")[1]["confirmation_state"] == "closed"
    answer = question(service, "answer", answer="none")[1]
    assert answer["confirmation_state"] == "closed" and answer["code"] == "confirmation_closed"


@pytest.mark.parametrize("method,path", [("PUT", "/sim/v0/meta"), ("HEAD", "/sim/v0/meta"),
    ("OPTIONS", "/sim/v0/meta"), ("DELETE", "/sim/v0/meta"),
    ("POST", "/sim/v0/meta"), ("GET", "/sim/v0/sessions")])
def test_unsupported_method_is_json_405(service, method, path):
    conn = http.client.HTTPConnection("127.0.0.1", service[1].server_port, timeout=8)
    conn.request(method, path)
    response = conn.getresponse()
    payload = response.read()
    assert response.status == 405 and response.getheader("Content-Type").startswith("application/json")
    policy = CONTRACT["x-http-errors"]["unsupported_method"]
    assert response.status == policy["status"] and policy["code"] == "method_not_allowed"
    assert response.getheader("Allow")
    if method == "HEAD": assert payload == b""
    else: assert json.loads(payload)["code"] == "method_not_allowed"
    conn.close()


def test_too_long_url_is_json_414(service):
    status, body = request(service[1], "GET", "/sim/v0/" + "a" * 70000)
    assert status == 414 and body["code"] == "uri_too_long"
    policy = CONTRACT["x-http-errors"]["request_uri_too_long"]
    assert status == policy["status"] and body["code"] == policy["code"]


def test_unknown_path_matches_contract_policy(service):
    status, body = request(service[1], "GET", "/sim/v0/not-a-route")
    policy = CONTRACT["x-http-errors"]["unknown_path"]
    assert status == policy["status"] and body["code"] == policy["code"]


def test_contract_constraints_and_meta_are_meaningful(service):
    props = CONTRACT["components"]["schemas"]["Response"]["properties"]
    assert "enum" in props["status"] and "enum" in props["record_state"]
    assert all(props[name].get("description") for name in ("code", "event_status", "event_code", "warnings"))
    status, body = check(service, "GET", "/sim/v0/meta", "/sim/v0/meta")
    assert status == 200 and body["limits"] == {"max_body_bytes": MAX_BODY, "events_per_request": 1,
        "max_concurrent": MAX_CONCURRENT, "read_timeout_seconds": TIMEOUT_SECONDS, "confirmation_seconds": CONFIRMATION_SECONDS}
    example = CONTRACT["components"]["responses"]["Meta"]["content"]["application/json"]["examples"]["default"]["value"]
    assert body == example
    assert set(props["code"]["enum"]) == PUBLIC_CODES | {None}
    for field in ("status", "code", "record_state"):
        sample = deepcopy(body)
        sample[field] = "not_a_declared_value"
        with pytest.raises(AssertionError): validate(CONTRACT["components"]["schemas"]["Response"], sample)


def test_question_conflict_includes_current_snapshot(service):
    create(service)
    question(service, "open")
    status, body = question(service, "answer", answer="none")
    assert status == 409 and body["code"] == "question_conflict"
    assert body["confirmation_state"] == "open" and body["question_id"] == "q1" and body["question_presented"] is False
    example = deepcopy(CONTRACT["components"]["responses"]["QuestionConflict"]["content"]["application/json"]["examples"]["question_conflict"]["value"])
    example["display"]["expires_at"] = body["display"]["expires_at"]
    assert body == example


def test_report_between_question_failure_and_response_takes_priority(service, monkeypatch):
    api, _ = service
    create(service)
    question(service, "open")
    original = api.coordinator.apply
    def interleaved(sid, operation, **kwargs):
        result = original(sid, operation, **kwargs)
        if operation == "answer_question" and kwargs.get("answer") == "none":
            original(sid, "report_discomfort", now=kwargs["now"], clock_id=kwargs["clock_id"])
        return result
    monkeypatch.setattr(api.coordinator, "apply", interleaved)
    status, body = question(service, "answer", answer="none")
    assert status == 200 and body["status"] == "safety_pending" and body["required_action"] == "report_discomfort"


def test_expire_only_target_safety_question():
    from metafit_agent.interaction_state import Question, create_session, expire_safety_confirmation
    from dataclasses import replace
    start = datetime(2026, 11, 3, 8, 0, tzinfo=timezone.utc)
    session = create_session("session", "clock")
    questions = (Question("safety", "safety_confirmation", start, start + timedelta(seconds=1)),
                 Question("other", "followup", start, start + timedelta(seconds=1)))
    session = replace(session, questions=questions)
    result = expire_safety_confirmation(session, question_id="safety", now="2026-11-03T08:00:02Z", clock_id="clock")
    assert result.status == "expired" and result.session.questions[0].presented is False
    assert result.session.questions[1] == questions[1]


def test_request_filters_and_body_size_boundary(service):
    server = service[1]
    def custom(headers, body=b"{}"):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request("POST", "/sim/v0/sessions", body=body, headers=headers)
        response = conn.getresponse(); result = json.loads(response.read()); conn.close()
        return response.status, result
    assert custom({"Host": "localhost", "Content-Type": "application/json"})[0] == 400
    assert custom({"Origin": "http://example.invalid", "Content-Type": "application/json"})[0] == 400
    assert custom({"Content-Type": "text/plain"})[0] == 400
    data = b'{"user_id":"body_user","session_id":"body_session"}'
    data += b" " * (MAX_BODY - len(data))
    assert custom({"Content-Type": "application/json"}, data)[0] == 201


def test_actual_safety_state_unknown_has_warning_and_no_candidate(service, monkeypatch):
    api, _ = service
    def fail(*args, **kwargs): raise StorageError("storage_failed")
    monkeypatch.setattr(api.store, "open_session", fail)
    status, body = create(service)
    assert status == 503 and body["code"] == "safety_state_unknown"
    assert body["confirmation_state"] == "unknown" and body["display"]["kind"] == "safety_unknown"
    assert body["candidate"]["may_display"] is False


def test_partial_body_read_timeout_returns_invalid_request(service, monkeypatch):
    import metafit_agent.simulation_api as module
    monkeypatch.setattr(module, "TIMEOUT_SECONDS", 0.2)
    server = service[1]
    client = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
    try:
        client.sendall((f"POST /sim/v0/sessions HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                        "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{").encode("ascii"))
        raw = client.recv(8192)
        assert b" 400 " in raw and b"application/json" in raw
    finally:
        client.close()


def test_real_slow_requests_fill_concurrency_limit(service):
    server = service[1]
    clients = []
    try:
        for _ in range(MAX_CONCURRENT):
            client = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
            client.sendall((f"POST /sim/v0/sessions HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                            "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n").encode("ascii"))
            clients.append(client)
        deadline = time.monotonic() + 4
        while server.slots._value != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.slots._value == 0
        status, body = request(server, "GET", "/sim/v0/meta")
        assert status == 503 and body["code"] == "server_busy"
    finally:
        for client in clients: client.close()
