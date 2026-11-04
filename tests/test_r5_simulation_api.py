"""本地 HTTP 和静态契约的离线验收；不访问真实模型或用户数据。"""
from copy import deepcopy
import http.client
import json
from pathlib import Path
from datetime import timedelta
from datetime import datetime
import re
import os
import subprocess
import sys
import threading

import pytest

from metafit_agent.event_store import StorageError
from metafit_agent.safety_store import SafetySQLiteStore
from metafit_agent.simulation_api import LocalServer, SimulationAPI, MAX_BODY, _now

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "docs" / "openapi.r5-draft.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def service(tmp_path):
    api = SimulationAPI(SafetySQLiteStore(tmp_path / "simulation.db"))
    server = LocalServer(0, api)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield api, server
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def validate(schema, value):
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return validate(CONTRACT["components"]["schemas"][name], value)
    if "oneOf" in schema:
        matches = 0
        for alternative in schema["oneOf"]:
            try:
                validate(alternative, value)
                matches += 1
            except AssertionError:
                pass
        assert matches == 1
        return
    if "const" in schema:
        assert type(value) is type(schema["const"]) and value == schema["const"]
    if "enum" in schema:
        assert value in schema["enum"]
    if "pattern" in schema and isinstance(value, str):
        assert re.search(schema["pattern"], value)
    if schema.get("format") == "date-time" and isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            assert parsed.tzinfo is not None and not value.endswith("-00:00")
        except ValueError as exc:
            raise AssertionError("invalid date-time") from exc
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    checks = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
              "string": lambda x: isinstance(x, str), "boolean": lambda x: type(x) is bool,
              "integer": lambda x: type(x) is int, "number": lambda x: type(x) in (int, float), "null": lambda x: x is None}
    if types:
        assert any(checks[t](value) for t in types), (types, value)
    if isinstance(value, dict):
        assert set(schema.get("required", [])) <= set(value)
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            assert set(value) <= set(properties), set(value) - set(properties)
        for key in set(value) & set(properties):
            validate(properties[key], value[key])
    if isinstance(value, list) and "items" in schema:
        for item in value:
            validate(schema["items"], item)
    if type(value) in (int, float):
        if "minimum" in schema: assert value >= schema["minimum"]
        if "maximum" in schema: assert value <= schema["maximum"]


def request(server, method, path, data=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=8)
    body = raw if raw is not None else (json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    value = json.loads(response.read())
    conn.close()
    return response.status, value


def check(service, method, path, template, data=None, raw=None):
    status, value = request(service[1], method, path, data, raw)
    declared = CONTRACT["paths"][template][method.lower()]["responses"]
    assert str(status) in declared, (template, status, value)
    response_name = declared[str(status)]["$ref"].split("/")[-1]
    schema = CONTRACT["components"]["responses"][response_name]["content"]["application/json"]["schema"]
    validate(schema, value)
    return status, value


def create(service, sid="sim_session_001"):
    return check(service, "POST", "/sim/v0/sessions", "/sim/v0/sessions", {"user_id": "sim_user_001", "session_id": sid})


def question(service, op, sid="sim_session_001", answer=None, qid="q1"):
    data = {"operation": op, "question_id": qid}
    if op == "open": data["kind"] = "safety_confirmation"
    if op == "answer": data["answer"] = answer
    return check(service, "POST", f"/sim/v0/sessions/{sid}/questions", "/sim/v0/sessions/{session_id}/questions", data)


def event_data(sid="sim_session_001", eid="sim_event_001"):
    event = json.loads((ROOT / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    event.update(session_id=sid, event_id=eid, discomfort={"status": "none", "source": "self_report"})
    return {"event": event, "do_not_disturb": False}


def submit(service, data, sid="sim_session_001"):
    return check(service, "POST", f"/sim/v0/sessions/{sid}/events", "/sim/v0/sessions/{session_id}/events", data)


def test_full_flow_contract_dedup_and_conflict(service):
    api, server = service
    assert server.server_address[0] == "127.0.0.1"
    status, meta = check(service, "GET", "/sim/v0/meta", "/sim/v0/meta")
    assert status == 200 and meta["meta"] == {"contract_version": "0.3.0-draft", "simulation": True}
    status, body = create(service)
    assert status == 201 and body["confirmation_state"] == "not_asked" and body["can_request_confirmation"]
    status, body = check(service, "GET", "/sim/v0/sessions/sim_session_001", "/sim/v0/sessions/{session_id}")
    assert body["required_action"] == "confirm"
    opened = question(service, "open")[1]
    assert opened["confirmation_state"] == "open" and opened["question_id"] == "q1" and opened["question_presented"] is False
    shown = question(service, "present")[1]
    assert shown["display"]["kind"] == "question" and shown["question_presented"] is True
    assert question(service, "answer", answer="none")[1]["confirmation_state"] == "confirmed"
    data = event_data()
    status, body = submit(service, data)
    assert status == 200 and body["status"] == "reviewed" and body["event_saved"] is True
    assert body["candidate"] == {"text": None, "may_display": False} and api.model.calls == 1
    status, body = check(service, "GET", "/sim/v0/sessions/sim_session_001/events/sim_event_001", "/sim/v0/sessions/{session_id}/events/{event_id}")
    assert status == 200 and body["event_saved"] is True
    status, body = submit(service, data)
    assert status == 200 and body["code"] == "duplicate_event" and api.model.calls == 1
    conflict = deepcopy(data)
    conflict["event"]["completed_reps"] = 7
    assert submit(service, conflict)[0] == 409


def test_pending_event_and_confirmation_close(service):
    api, _ = service
    create(service)
    data = event_data()
    status, body = submit(service, data)
    assert status == 200 and body["code"] == "pending_confirmation" and body["event_saved"] is True
    assert api.model.calls == 0
    assert submit(service, data)[1]["code"] == "duplicate_event"
    question(service, "open")
    question(service, "present")
    status, body = question(service, "skip")
    assert status == 200 and body["confirmation_state"] == "closed" and not body["can_request_confirmation"]
    assert question(service, "open", qid="q2")[1]["confirmation_state"] == "closed"


def test_storage_fault_never_hides_reported_safety(service, monkeypatch):
    api, _ = service
    create(service)
    def fail(*args, **kwargs): raise StorageError("storage_failed")
    monkeypatch.setattr(api.store, "report", fail)
    monkeypatch.setattr(api.store, "claim", fail)
    data = event_data()
    data["event"]["discomfort"]["status"] = "reported"
    status, body = submit(service, data)
    assert status == 200 and body["status"] == "safety_pending"
    assert body["safety_state_durability"] == "volatile" and body["display"]["kind"] == "safety_pending"
    assert body["candidate"]["may_display"] is False and api.model.calls == 0


def test_storage_fault_without_report_is_503_with_safety_notice(service, monkeypatch):
    api, _ = service
    create(service)
    def fail(*args, **kwargs): raise StorageError("storage_failed")
    monkeypatch.setattr(api.store, "claim", fail)
    status, body = submit(service, event_data())
    assert status == 503 and body["display"]["kind"] == "safety_unknown" and body["display"]["text"]
    assert body["candidate"]["text"] is None and api.model.calls == 0


def test_errors_limits_and_missing_resources(service):
    api, server = service
    status, body = check(service, "POST", "/sim/v0/sessions", "/sim/v0/sessions", {"user_id": "bad"})
    assert status == 400 and body["code"] == "invalid_request"
    assert create(service)[0] == 201
    assert create(service)[0] == 409
    assert check(service, "GET", "/sim/v0/sessions/absent", "/sim/v0/sessions/{session_id}")[0] == 404
    assert check(service, "GET", "/sim/v0/sessions/sim_session_001/events/missing", "/sim/v0/sessions/{session_id}/events/{event_id}")[0] == 404
    assert check(service, "POST", "/sim/v0/sessions", "/sim/v0/sessions", raw=b" " * (MAX_BODY + 1))[0] == 413
    for _ in range(4): assert server.slots.acquire(timeout=5)
    try:
        status, body = check(service, "GET", "/sim/v0/meta", "/sim/v0/meta")
        assert status == 503 and body["display"]["kind"] == "safety_unknown"
    finally:
        for _ in range(4): server.slots.release()


def test_static_contract_examples_follow_declared_schema():
    assert CONTRACT["openapi"] == "3.1.0" and CONTRACT["info"]["version"] == "0.3.0-draft"
    for response in CONTRACT["components"]["responses"].values():
        if "content" not in response:
            assert response.get("headers")
            continue
        content = response["content"]["application/json"]
        for sample in content["examples"].values():
            validate(content["schema"], sample["value"])
    fixtures = json.loads((ROOT / "examples" / "r5_api_fixtures.json").read_text(encoding="utf-8"))
    for fixture in fixtures:
        validate(CONTRACT["components"]["schemas"]["Response"], fixture["response"])


def test_valid_request_fixtures_follow_contract():
    schemas = CONTRACT["components"]["schemas"]
    validate(schemas["CreateSession"], {"user_id": "sim_user_001", "session_id": "sim_session_001"})
    validate(schemas["SubmitEvent"], event_data())
    for value in [
        {"operation": "open", "question_id": "q1", "kind": "safety_confirmation"},
        {"operation": "present", "question_id": "q1"},
        {"operation": "answer", "question_id": "q1", "answer": "none"},
        {"operation": "skip", "question_id": "q1"},
    ]:
        validate(schemas["Question"], value)
    with pytest.raises(AssertionError):
        validate(schemas["Question"], {"operation": "open", "question_id": "q1"})
    fixtures = json.loads((ROOT / "examples" / "r5_api_fixtures.json").read_text(encoding="utf-8"))
    for fixture in fixtures:
        request_data = fixture["request"]
        method, path = request_data["method"].lower(), request_data["path"]
        matches = []
        for template, operations in CONTRACT["paths"].items():
            pattern = re.sub(r"\{[^}]+\}", r"[^/]+", template)
            if re.fullmatch(pattern, path) and method in operations:
                matches.append(operations[method])
        assert len(matches) == 1, fixture["name"]
        operation = matches[0]
        assert str(fixture["http_status"]) in operation["responses"]
        assert fixture["expected_code"] == fixture["response"]["code"]
        assert fixture["preconditions"] and fixture["expect_error"] == (fixture["http_status"] >= 400)
        if not fixture["expect_error"] and request_data["json"] is not None:
            schema = operation["requestBody"]["content"]["application/json"]["schema"]
            validate(schema, request_data["json"])


def test_restart_and_previously_pending_event_are_not_replayed(service):
    api, server = service
    create(service)
    data = event_data()
    submit(service, data)
    question(service, "open")
    question(service, "present")
    question(service, "answer", answer="none")
    status, historical = check(service, "GET", "/sim/v0/sessions/sim_session_001/events/sim_event_001", "/sim/v0/sessions/{session_id}/events/{event_id}")
    assert status == 200 and historical["event_code"] == "pending_confirmation"
    assert historical["confirmation_state"] == "confirmed" and api.model.calls == 0
    server.api = SimulationAPI(SafetySQLiteStore(api.store.path))
    status, restored = create((server.api, server))
    assert status == 201 and restored["confirmation_state"] == "not_asked"
    assert submit((server.api, server), data)[1]["code"] == "duplicate_event"
    assert server.api.model.calls == 0


def test_expired_confirmation_closes_without_presenting(service, monkeypatch):
    import metafit_agent.simulation_api as module
    create(service)
    now = _now()
    monkeypatch.setattr(module, "_now", lambda: now)
    question(service, "open")
    monkeypatch.setattr(module, "_now", lambda: now + timedelta(seconds=121))
    status, body = check(service, "GET", "/sim/v0/sessions/sim_session_001", "/sim/v0/sessions/{session_id}")
    assert status == 200 and body["confirmation_state"] == "closed" and not body["can_request_confirmation"]
    assert question(service, "open", qid="q2")[1]["confirmation_state"] == "closed"


def test_reported_session_restores_as_safety_pending(service):
    api, server = service
    create(service)
    data = event_data()
    data["event"]["discomfort"]["status"] = "reported"
    assert submit(service, data)[1]["status"] == "safety_pending"
    server.api = SimulationAPI(SafetySQLiteStore(api.store.path))
    status, body = create((server.api, server))
    assert status == 200 and body["status"] == "safety_pending" and body["required_action"] == "report_discomfort"


def test_concurrent_report_discards_candidate(service):
    from metafit_agent.simulation_api import OfflineModel
    api, _ = service
    create(service)
    question(service, "open")
    question(service, "present")
    question(service, "answer", answer="none")
    entered, release = threading.Event(), threading.Event()
    class SlowModel(OfflineModel):
        def complete(self, facts):
            entered.set()
            assert release.wait(5)
            return super().complete(facts)
    api.model = SlowModel()
    result = []
    worker = threading.Thread(target=lambda: result.append(submit(service, event_data())), daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        status, body = question(service, "answer", answer="reported")
        assert status == 200 and body["status"] == "safety_pending"
    finally:
        release.set()
        worker.join(timeout=8)
    assert result and result[0][1]["status"] == "safety_pending"
    assert result[0][1]["candidate"] == {"text": None, "may_display": False}


def test_example_client_as_subprocess(service):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(ROOT / "examples/r5_api_client.py"), "--port", str(service[1].server_port)],
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert "duplicate_event" in proc.stdout


def test_every_declared_path_and_status_is_exercised(service):
    _, server = service
    observed = set()
    def invoke(method, path, template, data=None, raw=None):
        status, value = check(service, method, path, template, data, raw)
        observed.add((template, method.lower(), str(status)))
        return status, value
    sid = "sim_session_001"
    prefix = f"/sim/v0/sessions/{sid}"
    endpoints = [
        ("GET", "/sim/v0/meta", "/sim/v0/meta"),
        ("POST", "/sim/v0/sessions", "/sim/v0/sessions"),
        ("GET", prefix, "/sim/v0/sessions/{session_id}"),
        ("POST", prefix + "/questions", "/sim/v0/sessions/{session_id}/questions"),
        ("POST", prefix + "/events", "/sim/v0/sessions/{session_id}/events"),
        ("GET", prefix + "/events/sim_event_001", "/sim/v0/sessions/{session_id}/events/{event_id}"),
    ]
    invoke("GET", endpoints[0][1], endpoints[0][2])
    invoke("POST", endpoints[0][1], endpoints[0][2])
    invoke("GET", endpoints[1][1], endpoints[1][2])
    invoke("POST", endpoints[1][1], endpoints[1][2], {"user_id": "sim_user_001", "session_id": sid})
    invoke("POST", endpoints[1][1], endpoints[1][2], {"user_id": "sim_user_001", "session_id": sid})
    invoke("GET", prefix, endpoints[2][2])
    invoke("GET", "/sim/v0/sessions/missing", endpoints[2][2])
    invoke("POST", prefix + "/questions", endpoints[3][2], {"operation": "open", "question_id": "q1", "kind": "safety_confirmation"})
    invoke("POST", prefix + "/questions", endpoints[3][2], {"operation": "answer", "question_id": "q1", "answer": "none"})
    invoke("POST", "/sim/v0/sessions/missing/questions", endpoints[3][2], {"operation": "open", "question_id": "q1", "kind": "safety_confirmation"})
    data = event_data()
    invoke("POST", prefix + "/events", endpoints[4][2], data)
    other = deepcopy(data); other["event"]["completed_reps"] = 4
    invoke("POST", prefix + "/events", endpoints[4][2], other)
    invoke("POST", "/sim/v0/sessions/missing/events", endpoints[4][2], data)
    invoke("GET", prefix + "/events/sim_event_001", endpoints[5][2])
    invoke("GET", prefix + "/events/missing", endpoints[5][2])
    for method, path, template in endpoints:
        if method == "POST":
            invoke(method, path, template, {})
            invoke(method, path, template, raw=b" " * (MAX_BODY + 1))
        else:
            invoke(method, path + "?unexpected=1", template)
        acquired = 0
        try:
            for _ in range(4):
                assert server.slots.acquire(timeout=5)
                acquired += 1
            invoke(method, path, template, {} if method == "POST" else None)
        finally:
            for _ in range(acquired): server.slots.release()
    for method, path, template in [("POST", "/sim/v0/meta", "/sim/v0/meta"),
                                   ("GET", "/sim/v0/sessions", "/sim/v0/sessions")]:
        acquired = 0
        try:
            for _ in range(4):
                assert server.slots.acquire(timeout=5)
                acquired += 1
            invoke(method, path, template, {} if method == "POST" else None)
        finally:
            for _ in range(acquired): server.slots.release()
    # A previously reported persisted session yields 200, rather than 201, on restoration.
    service[0].coordinator.apply(sid, "report_discomfort", now="2026-11-03T00:00:00Z", clock_id="api_clock")
    server.api = SimulationAPI(SafetySQLiteStore(service[0].store.path))
    invoke("POST", endpoints[1][1], endpoints[1][2], {"user_id": "sim_user_001", "session_id": sid})
    covered = {(template, method.lower()) for method, _, template in endpoints} | {
        ("/sim/v0/meta", "post"), ("/sim/v0/sessions", "get")}
    expected = {(path, method, code) for path, operations in CONTRACT["paths"].items()
                for method, operation in operations.items() if (path, method) in covered for code in operation["responses"]}
    assert observed == expected, ("missing", expected - observed, "extra", observed - expected)
