"""doctor 自检、版本门禁与演示库清理边界。"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from examples import r5_api_doctor as doctor
from test_r5_simulation_api import service


def test_supported_versions_are_explicit():
    assert doctor.compatible("0.3.0-draft") and doctor.compatible("0.3.1-draft")
    assert not doctor.compatible("0.2.2-draft") and not doctor.compatible("0.4.0-draft")
    assert not doctor.compatible("0.3.0") and not doctor.compatible(None)


def test_doctor_self_started_json_and_confirmed_reset(tmp_path, monkeypatch, capsys):
    path = tmp_path / "doctor"
    monkeypatch.setattr(doctor, "DOCTOR_DIR", path)
    assert doctor.main(["--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["overall"] == "PASS" and output["simulation"] is True
    assert {x["check"] for x in output["checks"]} >= {"meta", "version", "event_saved", "duplicate", "database"}
    database = path / "doctor.db"
    assert database.is_file()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    monkeypatch.setattr("builtins.input", lambda: "no")
    assert doctor.main(["--reset", "--json"]) == 1
    refused = json.loads(capsys.readouterr().out)
    assert refused["overall"] == "FAIL" and hashlib.sha256(database.read_bytes()).hexdigest() == before
    monkeypatch.setattr("builtins.input", lambda: "RESET")
    assert doctor.main(["--reset", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["overall"] == "PASS" and database.is_file()


def test_reset_refuses_unmarked_or_unknown_files(tmp_path, monkeypatch, capsys):
    path = tmp_path / "doctor"
    path.mkdir()
    sentinel = path / "personal.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(doctor, "DOCTOR_DIR", path)
    monkeypatch.setattr("builtins.input", lambda: "RESET")
    assert doctor.main(["--reset", "--json"]) == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert json.loads(capsys.readouterr().out)["overall"] == "FAIL"
    (path / ".metafit-doctor").write_text(doctor.MARKER_TEXT, encoding="utf-8")
    assert doctor.main(["--reset", "--json"]) == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert json.loads(capsys.readouterr().out)["overall"] == "FAIL"


def test_doctor_rejects_hardlinked_database(tmp_path, monkeypatch, capsys):
    path = tmp_path / "doctor"
    path.mkdir()
    (path / ".metafit-doctor").write_text(doctor.MARKER_TEXT, encoding="utf-8")
    source = tmp_path / "unrelated.db"
    source.write_bytes(b"do-not-touch")
    try:
        os.link(source, path / "doctor.db")
    except OSError:
        pytest.skip("系统不允许创建硬链接")
    monkeypatch.setattr(doctor, "DOCTOR_DIR", path)
    assert doctor.main(["--json"]) == 1
    assert source.read_bytes() == b"do-not-touch"
    assert json.loads(capsys.readouterr().out)["overall"] == "FAIL"


def test_connect_mode_and_forbidden_reset(service, capsys):
    port = service[1].server_port
    assert doctor.main(["--connect", str(port), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["overall"] == "PASS"
    assert "无法验证" in next(x["detail"] for x in output["checks"] if x["check"] == "database")
    assert doctor.main(["--connect", str(port), "--reset", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["overall"] == "FAIL"


def test_version_mismatch_is_actionable(monkeypatch, capsys):
    def fake_call(*args, **kwargs):
        return 200, {"meta": {"simulation": True, "contract_version": "0.4.0-draft"}}
    monkeypatch.setattr(doctor, "_call", fake_call)
    assert doctor.main(["--connect", "12345", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    failed = next(x for x in result["checks"] if x["status"] == "FAIL")
    assert "0.3.x" in failed["detail"] and "0.4.0-draft" in failed["detail"] and failed["next_step"]


def test_doctor_refuses_directory_outside_project(tmp_path, monkeypatch, capsys):
    path = doctor.ROOT.parent / "doctor-outside-project-should-not-be-created"
    monkeypatch.setattr(doctor, "DOCTOR_DIR", path)
    assert doctor.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["overall"] == "FAIL" and "项目内" in result["checks"][-1]["detail"]


def test_doctor_refuses_service_claiming_real_model(monkeypatch, capsys):
    calls = []
    def fake_call(port, method, path, body=None):
        calls.append((method, path))
        return 200, {"meta": {"simulation": True, "contract_version": "0.3.0-draft"},
                     "features": {"safety_confirmation": True, "event_deduplication": True,
                                  "real_model": True, "delivery": False},
                     "limits": {"max_body_bytes": 65536, "events_per_request": 1, "max_concurrent": 4,
                                "read_timeout_seconds": 5, "confirmation_seconds": 120}}
    monkeypatch.setattr(doctor, "_call", fake_call)
    assert doctor.main(["--connect", "12345", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["overall"] == "FAIL" and calls == [("GET", "/meta")]


def test_doctor_does_not_treat_numeric_flags_as_booleans(monkeypatch, capsys):
    def fake_call(port, method, path, body=None):
        return 200, {"meta": {"simulation": True, "contract_version": "0.3.0-draft"},
                     "features": {"safety_confirmation": 1, "event_deduplication": True,
                                  "real_model": False, "delivery": False},
                     "limits": {"max_body_bytes": 65536, "events_per_request": 1, "max_concurrent": 4,
                                "read_timeout_seconds": 5, "confirmation_seconds": 120}}
    monkeypatch.setattr(doctor, "_call", fake_call)
    assert doctor.main(["--connect", "12345", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["overall"] == "FAIL"


def test_stale_lock_is_not_automatically_removed_by_reset(tmp_path, monkeypatch, capsys):
    path = tmp_path / "doctor"
    path.mkdir()
    (path / ".metafit-doctor").write_text(doctor.MARKER_TEXT, encoding="utf-8")
    lock = path / ".doctor.lock"
    lock.write_text("unknown owner", encoding="utf-8")
    database = path / "doctor.db"
    database.write_bytes(b"preserve")
    monkeypatch.setattr(doctor, "DOCTOR_DIR", path)
    monkeypatch.setattr("builtins.input", lambda: "RESET")
    assert doctor.main(["--reset", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["overall"] == "FAIL" and lock.read_text(encoding="utf-8") == "unknown owner"
    assert database.read_bytes() == b"preserve"


def test_json_cli_is_utf8_and_has_chinese_diagnostics(service):
    script = doctor.ROOT / "examples/r5_api_doctor.py"
    proc = subprocess.run([sys.executable, str(script), "--connect", str(service[1].server_port), "--json"],
                          capture_output=True, timeout=20)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    text = proc.stdout.decode("utf-8")
    assert json.loads(text)["overall"] == "PASS" and "模拟" in text and "\ufffd" not in text
