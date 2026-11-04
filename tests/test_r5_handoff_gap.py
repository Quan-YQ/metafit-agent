"""未知会话中的不适信号：提示但绝不错误归属或保存。"""
from copy import deepcopy

import pytest

from test_r5_simulation_api import CONTRACT, create, event_data, request, service


@pytest.mark.parametrize("route,payload", [
    ("events", {"event": {"discomfort": {"status": "reported"}}, "do_not_disturb": False}),
    ("events", {"event": {"discomfort": {"status": "reported"}}}),
    ("questions", {"operation": "answer", "question_id": "q1", "answer": "reported"}),
    ("questions", {"operation": "answer", "answer": "reported"}),
])
def test_unknown_session_report_gives_safety_notice_without_attribution(service, route, payload):
    api, server = service
    create(service)
    before = api.coordinator.confirmation_view("sim_session_001")
    status, body = request(server, "POST", f"/sim/v0/sessions/ghost/{route}", payload)
    assert status == 400 and body["code"] == "safety_report_unattributed"
    assert body == CONTRACT["components"]["responses"]["BadRequest"]["content"]["application/json"]["examples"]["safety_report_unattributed"]["value"]
    assert body["required_action"] == "report_discomfort" and body["display"]["kind"] == "safety_unknown"
    assert body["event_saved"] is False and body["candidate"]["may_display"] is False
    assert api.coordinator.confirmation_view("ghost") is None
    assert api.coordinator.confirmation_view("sim_session_001") == before
    assert api.model.calls == 0


@pytest.mark.parametrize("route,payload", [
    ("events", {"event": {"discomfort": {"status": "none"}}, "do_not_disturb": False}),
    ("questions", {"operation": "answer", "question_id": "q1", "answer": "none"}),
    ("questions", {"operation": "open", "question_id": "q1", "answer": "reported"}),
])
def test_unknown_session_non_report_stays_404(service, route, payload):
    status, body = request(service[1], "POST", f"/sim/v0/sessions/ghost/{route}", payload)
    assert status == 404 and body["code"] == "state_missing"


def test_known_session_report_behavior_unchanged(service):
    create(service)
    data = event_data()
    data["event"]["discomfort"]["status"] = "reported"
    status, body = request(service[1], "POST", "/sim/v0/sessions/sim_session_001/events", data)
    assert status == 200 and body["status"] == "safety_pending"


def test_invalidated_session_report_is_not_written(service):
    api, server = service
    create(service)
    api.coordinator.invalidate("sim_session_001")
    data = event_data()
    data["event"]["discomfort"]["status"] = "reported"
    status, body = request(server, "POST", "/sim/v0/sessions/sim_session_001/events", data)
    assert status == 400 and body["code"] == "safety_report_unattributed"
    assert body["event_saved"] is False and not api.coordinator.view("sim_session_001").safety_reported


def test_invalid_session_identifier_report_still_provides_notice(service):
    status, body = request(service[1], "POST", "/sim/v0/sessions/bad.id/questions", {"operation": "answer", "answer": "reported"})
    assert status == 400 and body["code"] == "safety_report_unattributed" and body["event_saved"] is False


@pytest.mark.parametrize("payload", [
    {"operation": "answer", "answer": "reported"},
    {"operation": "answer", "question_id": "bad.id", "answer": "reported"},
    {"operation": "answer", "question_id": "q1", "kind": "safety_confirmation", "answer": "reported"},
    {"operation": "answer", "question_id": "q1", "answer": "reported", "extra": True},
])
def test_malformed_report_for_known_session_gives_notice_without_recording(service, payload):
    api, server = service
    create(service)
    before = api.coordinator.confirmation_view("sim_session_001")
    status, body = request(server, "POST", "/sim/v0/sessions/sim_session_001/questions", payload)
    assert status == 400 and body["code"] == "safety_report_unattributed"
    assert body["required_action"] == "report_discomfort" and body["event_saved"] is False
    assert body["display"]["kind"] == "safety_unknown"
    assert api.coordinator.confirmation_view("sim_session_001") == before
    status, saved = request(server, "POST", "/sim/v0/sessions/sim_session_001/questions", {"operation": "answer", "question_id": "q1", "answer": "reported"})
    assert status == 200 and saved["status"] == "safety_pending"


@pytest.mark.parametrize("payload", [
    {"operation": ["answer"], "question_id": "q1", "answer": "reported"},
    {"operation": "answer", "question_id": "q1", "answer": ["none"]},
])
def test_non_reported_bad_types_are_plain_invalid_request(service, payload):
    create(service)
    status, body = request(service[1], "POST", "/sim/v0/sessions/sim_session_001/questions", payload)
    assert status == 400 and body["code"] == "invalid_request"
