"""0.2-draft 复审：故障示例、满载 HEAD 与 405 契约。"""
import http.client
import json
import socket
import time
from pathlib import Path

import pytest

from metafit_agent.event_store import StorageError
from metafit_agent.simulation_api import MAX_CONCURRENT
from test_r5_simulation_api import CONTRACT, create, event_data, request, service, submit

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {item["name"]: item for item in json.loads((ROOT / "examples/r5_api_fixtures.json").read_text(encoding="utf-8"))}
EXAMPLES = CONTRACT["components"]["responses"]["Unavailable"]["content"]["application/json"]["examples"]


def fail(*args, **kwargs):
    raise StorageError("storage_failed")


def test_safety_state_unknown_example_and_fixture_match_http(service, monkeypatch):
    api, _ = service
    monkeypatch.setattr(api.store, "open_session", fail)
    status, actual = create(service)
    assert status == 503
    assert actual == EXAMPLES["safety_state_unknown"]["value"] == FIXTURES["safety_state_unknown"]["response"]
    assert actual["confirmation_state"] == "unknown" and actual["safety_state_durability"] == "volatile"
    assert actual["warnings"] == ["storage_failed"]


def test_unknown_snapshot_after_failed_creation_has_own_context(service, monkeypatch):
    api, server = service
    monkeypatch.setattr(api.store, "open_session", fail)
    assert create(service)[0] == 503
    status, actual = request(server, "GET", "/sim/v0/sessions/sim_session_001")
    assert status == 503 and actual["confirmation_state"] == "unknown"
    assert actual["safety_state_durability"] == "volatile" and actual["warnings"] == []
    assert actual != EXAMPLES["safety_state_unknown"]["value"]


def test_storage_failed_example_and_fixture_match_http(service, monkeypatch):
    api, _ = service
    create(service)
    monkeypatch.setattr(api.store, "claim", fail)
    status, actual = submit(service, event_data())
    assert status == 503
    assert actual == EXAMPLES["storage_failed"]["value"] == FIXTURES["storage_failed"]["response"]
    assert actual["warnings"] == ["storage_failed"]


def test_reported_volatile_fixture_matches_http(service, monkeypatch):
    api, _ = service
    create(service)
    monkeypatch.setattr(api.store, "report", fail)
    monkeypatch.setattr(api.store, "claim", fail)
    data = event_data()
    data["event"]["discomfort"]["status"] = "reported"
    status, actual = submit(service, data)
    example = CONTRACT["components"]["responses"]["OK"]["content"]["application/json"]["examples"]["reported_volatile"]["value"]
    assert status == 200 and actual == example == FIXTURES["reported_volatile"]["response"]
    assert actual["warnings"] == ["time_unvalidated", "storage_failed"]


def test_server_busy_example_fixture_and_head_wire_body(service):
    server = service[1]
    acquired = 0
    try:
        for _ in range(MAX_CONCURRENT):
            assert server.slots.acquire(timeout=5)
            acquired += 1
        status, body = request(server, "GET", "/sim/v0/meta")
        assert status == 503 and body == EXAMPLES["server_busy"]["value"] == FIXTURES["server_busy"]["response"]
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request("GET", "/sim/v0/meta")
        resp = conn.getresponse(); payload = json.loads(resp.read())
        assert resp.status == 503 and resp.getheader("Cache-Control") == "no-store" and payload == body
        conn.close()
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as client:
            client.sendall(f"HEAD /sim/v0/meta HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n\r\n".encode("ascii"))
            chunks = []
            while True:
                data = client.recv(8192)
                if not data: break
                chunks.append(data)
        head, separator, payload = b"".join(chunks).partition(b"\r\n\r\n")
        assert separator and b" 503 " in head and payload == b""
        assert b"Cache-Control: no-store" in head and b"Content-Type: application/json" in head
    finally:
        for _ in range(acquired): server.slots.release()


@pytest.mark.parametrize("finish_method", [True, False])
def test_busy_partial_head_never_emits_body(service, finish_method):
    server = service[1]
    acquired = 0
    try:
        for _ in range(MAX_CONCURRENT):
            assert server.slots.acquire(timeout=5)
            acquired += 1
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as client:
            client.sendall(b"H")
            if finish_method:
                time.sleep(0.03)
                client.sendall(f"EAD /sim/v0/meta HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n\r\n".encode("ascii"))
            chunks = []
            while True:
                data = client.recv(8192)
                if not data: break
                chunks.append(data)
        head, separator, payload = b"".join(chunks).partition(b"\r\n\r\n")
        assert separator and b" 503 " in head and payload == b""
        if not finish_method:
            assert b"Content-Length: 0" in head
    finally:
        for _ in range(acquired): server.slots.release()


@pytest.mark.parametrize("template,path", [
    ("/sim/v0/meta", "/sim/v0/meta"),
    ("/sim/v0/sessions", "/sim/v0/sessions"),
    ("/sim/v0/sessions/{session_id}", "/sim/v0/sessions/s1"),
    ("/sim/v0/sessions/{session_id}/questions", "/sim/v0/sessions/s1/questions"),
    ("/sim/v0/sessions/{session_id}/events", "/sim/v0/sessions/s1/events"),
    ("/sim/v0/sessions/{session_id}/events/{event_id}", "/sim/v0/sessions/s1/events/e1"),
])
def test_all_unsupported_methods_have_declared_405(service, template, path):
    allowed = {"/sim/v0/meta": "get", "/sim/v0/sessions": "post", "/sim/v0/sessions/{session_id}": "get",
               "/sim/v0/sessions/{session_id}/questions": "post", "/sim/v0/sessions/{session_id}/events": "post",
               "/sim/v0/sessions/{session_id}/events/{event_id}": "get"}[template]
    methods = ("get", "post", "put", "head", "options", "delete", "patch", "trace")
    for method in methods:
        if method == allowed: continue
        operation = CONTRACT["paths"][template][method]
        assert "503" in operation["responses"]
        ref = operation["responses"]["405"]["$ref"].split("/")[-1]
        assert "Allow" in CONTRACT["components"]["responses"][ref]["headers"]
        conn = http.client.HTTPConnection("127.0.0.1", service[1].server_port, timeout=5)
        conn.request(method.upper(), path)
        response = conn.getresponse(); data = response.read()
        assert response.status == 405 and response.getheader("Allow") == allowed.upper()
        assert response.getheader("Content-Type").startswith("application/json")
        if method == "head": assert data == b""
        else: assert json.loads(data)["code"] == "method_not_allowed"
        conn.close()
