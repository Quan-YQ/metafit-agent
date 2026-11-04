import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Event, Thread

import pytest

from metafit_agent.decision_store import DecisionSQLiteStore
from metafit_agent.event_store import SQLiteEventStore, StorageError
from metafit_agent.model_response import ModelResponse
from metafit_agent.recorded_coordinator import RecordedSimulationCoordinator


@pytest.fixture
def event():
    data = json.loads((Path(__file__).resolve().parents[1] / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    data["discomfort"] = {"status": "none", "source": "self_report"}
    return data


class Model:
    def __init__(self, evidence="fact_3"):
        self.calls = 0
        self.evidence = evidence

    def complete(self, facts):
        self.calls += 1
        text = json.dumps({"suggested_action": "notify", "message": "模拟反馈文字。",
                           "reason": "不得保存的内部说明", "evidence": [self.evidence]}, ensure_ascii=False)
        return ModelResponse(text, "mock", "mock", None, None, 0, "stop")


def coordinator(store, event):
    c = RecordedSimulationCoordinator(store)
    assert c.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="clock").status == "registered"
    return c


def test_archive_default_redaction_reopen_duplicate_and_join(tmp_path, event):
    path = tmp_path / "history.db"
    store = DecisionSQLiteStore(path)
    model = Model()
    first = coordinator(store, event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert first.status == "reviewed" and first.has_text and first.decided_at and model.calls == 1
    reopened = DecisionSQLiteStore(path)
    row = reopened.query(event["user_id"], session_id=event["session_id"])[0]
    assert "message" not in row and "reason" not in row and row["has_text"] is True
    assert row["text_verification"] == "unverified" and row["reference_check"] == "passed_for_cited_ids"
    assert row["record_state"] == "decided" and row["delivery_eligibility"] == "not_eligible"
    assert row["not_delivered_reason"] == "not_delivered" and row["evidence"] == ["fact_3"]
    text = reopened.query(event["user_id"], event_id=event["event_id"], include_text=True)[0]
    assert text["message"] == "模拟反馈文字。" and "reason" not in text
    again_model = Model()
    duplicate = coordinator(reopened, event).decide(event["session_id"], event, again_model, do_not_disturb=False)
    assert duplicate.code == "duplicate_event" and duplicate.has_text and duplicate.candidate is None and again_model.calls == 0
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM events JOIN decisions USING(user_id,session_id,event_id)").fetchone()[0] == 1
        assert "不得保存的内部说明" not in repr(conn.execute("SELECT * FROM decisions").fetchall())
    script = "from metafit_agent.decision_store import DecisionSQLiteStore; import sys,json; print(json.dumps(DecisionSQLiteStore(sys.argv[1]).query(sys.argv[2])))"
    run = subprocess.run([sys.executable, "-c", script, str(path), event["user_id"]], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(run.stdout)[0]["has_text"] is True


def test_rule_rejection_no_message_and_failure_attempt(tmp_path, event):
    path = tmp_path / "history.db"
    store = DecisionSQLiteStore(path)
    result = coordinator(store, event).decide(event["session_id"], event, Model("fact_999"), do_not_disturb=False)
    assert result.status == "stopped" and result.code == "evidence_unknown_id"
    row = store.query(event["user_id"], include_text=True)[0]
    assert row["message"] is None and not row["has_text"] and row["code"] == "evidence_unknown_id"
    assert row["not_delivered_reason"] == "rule_stopped"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT failure_code FROM attempts").fetchall() == [("decision_stopped",)]
    assert store.query_attempts(event["user_id"], event["session_id"], event["event_id"])[0]["failure_code"] == "decision_stopped"


def test_safety_during_archive_records_discard_reason_and_does_not_block(tmp_path, event):
    entered, release = Event(), Event()
    class SlowStore(DecisionSQLiteStore):
        def finish_with_decision(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().finish_with_decision(*args, **kwargs)
    store = SlowStore(tmp_path / "history.db")
    c = coordinator(store, event)
    results = []
    worker = Thread(target=lambda: results.append(c.decide(event["session_id"], event, Model(), do_not_disturb=False)))
    worker.start()
    try:
        assert entered.wait(3)
        assert c.apply(event["session_id"], "report_discomfort", now=None, clock_id="clock").status == "safety_pending"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and results[0].status == "safety_pending" and results[0].candidate is None
    row = store.query(event["user_id"], include_text=True)[0]
    assert row["not_delivered_reason"] == "safety_discarded" and row["message"] == "模拟反馈文字。"


def test_archive_failure_has_no_decision_and_no_normal_candidate(tmp_path, event):
    class FailArchive(DecisionSQLiteStore):
        def finish_with_decision(self, *args, **kwargs):
            raise StorageError("storage_failed")
    path = tmp_path / "history.db"
    store = FailArchive(path)
    result = coordinator(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    assert result.code == "storage_failed" and result.candidate is None and result.event_saved is True
    row = store.query(event["user_id"])[0]
    assert row["decision_state"] == "reserved" and row["status"] is None and not row["has_text"]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM decisions").fetchone()[0] == 0
        assert conn.execute("SELECT failure_code FROM attempts").fetchone()[0] == "storage_failed"


def test_reserved_legacy_upgrade_time_filter_and_delete(tmp_path, event):
    path = tmp_path / "history.db"
    old = SQLiteEventStore(path)
    old.claim(event)
    store = DecisionSQLiteStore(path)
    with pytest.raises(StorageError):
        store.query(event["user_id"])
    # 明确发起写入后升级，保留旧事件及预占状态。
    store.claim(event)
    rows = store.query(event["user_id"], start="2000-01-01T00:00:00Z", end="2100-01-01T00:00:00Z")
    assert len(rows) == 1 and rows[0]["decision_state"] == "reserved" and rows[0]["status"] is None
    assert rows[0]["record_state"] == "pending" and rows[0]["delivery_eligibility"] == "not_eligible"
    with pytest.raises(StorageError, match="event_in_progress_or_unresolved"):
        store.delete_event(event["user_id"], event["session_id"], event["event_id"])
    assert SQLiteEventStore(path).get(event["user_id"], event["session_id"], event["event_id"]) is not None
    assert store.delete_event(event["user_id"], event["session_id"], event["event_id"], allow_unresolved=True)
    assert store.query(event["user_id"]) == []


def test_delete_terminal_event_without_affecting_other(tmp_path, event):
    store = DecisionSQLiteStore(tmp_path / "history.db")
    coordinator(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    other = dict(event, event_id="other")
    coordinator(store, other).decide(other["session_id"], other, Model(), do_not_disturb=False)
    assert store.delete_event(event["user_id"], event["session_id"], event["event_id"])
    assert [row["event_id"] for row in store.query(event["user_id"])] == ["other"]
    assert not store.delete_event(event["user_id"], event["session_id"], event["event_id"])


def test_dnd_is_not_treated_as_delivery(tmp_path, event):
    store = DecisionSQLiteStore(tmp_path / "history.db")
    result = coordinator(store, event).decide(event["session_id"], event, Model(), do_not_disturb=True)
    row = store.query(event["user_id"])[0]
    assert result.effective_action == "defer" and row["rule_code"] == "dnd_notify_to_defer"
    assert row["not_delivered_reason"] == "dnd_deferred"


def test_storage_and_attempt_failure_are_reported_without_model(tmp_path, event):
    class FailStore(DecisionSQLiteStore):
        def claim(self, data):
            raise StorageError("storage_failed")
        def record_failure(self, *args):
            raise StorageError("storage_failed")
    store = FailStore(tmp_path / "history.db")
    model = Model()
    result = coordinator(store, event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.code == "storage_failed" and result.event_saved is None and model.calls == 0
    assert "failure_record_unavailable" in result.warnings
    assert not store.path.exists()


def test_failed_claim_is_recorded_when_database_can_still_be_used(tmp_path, event):
    class FailClaim(DecisionSQLiteStore):
        def claim(self, data):
            raise StorageError("storage_failed")
    store = FailClaim(tmp_path / "history.db")
    model = Model()
    result = coordinator(store, event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.code == "storage_failed" and model.calls == 0
    assert store.query(event["user_id"]) == []
    assert store.query_attempts(event["user_id"], event["session_id"], event["event_id"])[0]["failure_code"] == "storage_failed"
    assert store.delete_event(event["user_id"], event["session_id"], event["event_id"])
    assert store.query_attempts(event["user_id"], event["session_id"], event["event_id"]) == []


def test_failed_reason_update_never_returns_candidate(tmp_path, event):
    class FailUpdate(DecisionSQLiteStore):
        def update_not_delivered(self, *args):
            raise StorageError("storage_failed")
    store = FailUpdate(tmp_path / "history.db")
    result = coordinator(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    assert result.code == "storage_failed" and result.candidate is None and result.candidate_discarded
    row = store.query(event["user_id"])[0]
    assert row["not_delivered_reason"] == "safety_check_pending"
    assert row["record_state"] == "pending" and row["delivery_eligibility"] == "not_eligible"


def test_safety_and_failed_reason_update_preserves_warning(tmp_path, event):
    class FailUpdate(DecisionSQLiteStore):
        def update_not_delivered(self, *args):
            raise StorageError("storage_failed")
    store = FailUpdate(tmp_path / "history.db")
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    model = Model()
    result = coordinator(store, event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.safety_message and model.calls == 0
    assert "decision_record_update_unconfirmed" in result.warnings
    assert store.query(event["user_id"])[0]["not_delivered_reason"] == "safety_discarded"


def test_safety_while_summary_query_runs_is_not_lost(tmp_path, event):
    entered, release = Event(), Event()
    class SlowQuery(DecisionSQLiteStore):
        def query(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().query(*args, **kwargs)
    store = SlowQuery(tmp_path / "history.db")
    c = coordinator(store, event)
    results = []
    worker = Thread(target=lambda: results.append(c.decide(event["session_id"], event, Model(), do_not_disturb=False)))
    worker.start()
    try:
        assert entered.wait(3)
        assert c.apply(event["session_id"], "report_discomfort", now=None, clock_id="clock").status == "safety_pending"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and results[0].status == "safety_pending" and results[0].candidate is None
    assert DecisionSQLiteStore(store.path).query(event["user_id"])[0]["not_delivered_reason"] == "safety_discarded"


def test_failed_late_safety_update_reports_unconfirmed(tmp_path, event):
    entered, release = Event(), Event()
    class FailingLateUpdate(DecisionSQLiteStore):
        def query(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().query(*args, **kwargs)
        def update_not_delivered(self, user, session, event_id, reason):
            if reason == "safety_discarded":
                raise StorageError("storage_failed")
            return super().update_not_delivered(user, session, event_id, reason)
    store = FailingLateUpdate(tmp_path / "history.db")
    c = coordinator(store, event)
    results = []
    worker = Thread(target=lambda: results.append(c.decide(event["session_id"], event, Model(), do_not_disturb=False)))
    worker.start()
    try:
        assert entered.wait(3)
        c.apply(event["session_id"], "report_discomfort", now=None, clock_id="clock")
    finally:
        release.set()
        worker.join(5)
    result = results[0]
    assert result.status == "safety_pending" and result.candidate is None
    assert result.storage_status == "saved_state_unconfirmed" and "decision_record_update_unconfirmed" in result.warnings
    assert store.query_attempts(event["user_id"], event["session_id"], event["event_id"])[0]["failure_code"] == "decision_update_failed"


def test_delete_removes_related_attempts(tmp_path, event):
    store = DecisionSQLiteStore(tmp_path / "history.db")
    coordinator(store, event).decide(event["session_id"], event, Model("fact_999"), do_not_disturb=False)
    assert store.query_attempts(event["user_id"], event["session_id"], event["event_id"])
    assert store.delete_event(event["user_id"], event["session_id"], event["event_id"])
    assert store.query_attempts(event["user_id"], event["session_id"], event["event_id"]) == []


def test_query_does_not_create_database(tmp_path):
    path = tmp_path / "absent.db"
    with pytest.raises(StorageError):
        DecisionSQLiteStore(path).query("user")
    assert not path.exists()


def test_mismatched_event_never_leaks_session_safety(tmp_path, event):
    store = DecisionSQLiteStore(tmp_path / "history.db")
    c = coordinator(store, event)
    assert c.apply(event["session_id"], "report_discomfort", now=None, clock_id="clock").status == "safety_pending"
    other = dict(event, user_id="someone_else")
    result = c.decide(event["session_id"], other, Model(), do_not_disturb=False)
    assert result.status == "stopped" and result.code == "scope_mismatch" and result.safety_message is None
    assert not store.path.exists()


def test_incompatible_event_store_fails_before_any_event_or_model_call(tmp_path):
    path = tmp_path / "history.db"
    with pytest.raises(TypeError, match="decision store missing required methods: finish_with_decision"):
        RecordedSimulationCoordinator(SQLiteEventStore(path))
    assert not path.exists()


def test_delete_user_preserves_other_user_and_reports_unattributed_attempt(tmp_path, event):
    store = DecisionSQLiteStore(tmp_path / "history.db")
    coordinator(store, event).decide(event["session_id"], event, Model("fact_999"), do_not_disturb=False)
    other = dict(event, user_id="other_user", event_id="other_event")
    coordinator(store, other).decide(other["session_id"], other, Model("fact_999"), do_not_disturb=False)
    store.record_failure(event["user_id"], event["session_id"], "unsaved", "a" * 64, "storage_failed")
    result = store.delete_user(event["user_id"])
    assert result == {"events_deleted": 1, "decisions_deleted": 1, "attempts_deleted": 1,
                      "unattributed_attempts_present": True, "complete_in_this_database": False}
    assert store.query(event["user_id"]) == []
    assert len(store.query(other["user_id"])) == 1
    assert len(store.query_attempts(other["user_id"], other["session_id"], other["event_id"])) == 1
    assert len(store.query_attempts(event["user_id"], event["session_id"], "unsaved")) == 1
    assert store.delete_event(event["user_id"], event["session_id"], "unsaved")
    assert store.delete_user(event["user_id"])["complete_in_this_database"] is True


def test_delete_user_pending_rolls_back_all_then_explicit_override(tmp_path, event):
    store = DecisionSQLiteStore(tmp_path / "history.db")
    coordinator(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    pending = dict(event, event_id="pending")
    store.claim(pending)
    with pytest.raises(StorageError, match="event_in_progress_or_unresolved") as exc:
        store.delete_user(event["user_id"])
    assert exc.value.unresolved_count == 1
    without_ids = store.inspect_user_deletion(event["user_id"])
    assert without_ids["unresolved_count"] == 1 and without_ids["unresolved"] == []
    inspection = store.inspect_user_deletion(event["user_id"], include_ids=True)
    assert inspection["event_count"] == 2 and inspection["unresolved_count"] == 1
    assert inspection["unresolved"] == [{"session_id": event["session_id"], "event_id": "pending"}]
    assert len(store.query(event["user_id"])) == 2
    result = store.delete_user(event["user_id"], allow_unresolved=True)
    assert result["events_deleted"] == 2 and result["decisions_deleted"] == 1
    assert store.query(event["user_id"]) == []


def test_prune_attempts_removes_only_older_attempts(tmp_path, event):
    path = tmp_path / "history.db"
    store = DecisionSQLiteStore(path)
    coordinator(store, event).decide(event["session_id"], event, Model("fact_999"), do_not_disturb=False)
    store.record_failure(event["user_id"], event["session_id"], event["event_id"], "a" * 64, "storage_failed")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE attempts SET attempted_at='2000-01-01T00:00:00.000000Z' WHERE failure_code='decision_stopped'")
        conn.execute("UPDATE attempts SET attempted_at='2100-01-01T00:00:00.000000Z' WHERE failure_code='storage_failed'")
    assert store.prune_attempts(before="2050-01-01T00:00:00Z") == 1
    attempts = store.query_attempts(event["user_id"], event["session_id"], event["event_id"])
    assert [attempt["failure_code"] for attempt in attempts] == ["storage_failed"]
    assert len(store.query(event["user_id"])) == 1
    with pytest.raises(ValueError):
        store.prune_attempts(before="2050-01-01T00:00:00")


def test_terminal_old_event_without_decision_is_explicit(tmp_path, event):
    path = tmp_path / "history.db"
    old = SQLiteEventStore(path)
    old.claim(event)
    old.finish(event["user_id"], event["session_id"], event["event_id"], state="finished")
    store = DecisionSQLiteStore(path)
    store.claim(event)
    row = store.query(event["user_id"])[0]
    assert row["record_state"] == "no_decision" and row["status"] is None


def test_inconsistent_reserved_and_decision_is_not_claimed_complete(tmp_path, event):
    path = tmp_path / "history.db"
    store = DecisionSQLiteStore(path)
    coordinator(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE events SET decision_state='reserved'")
    row = store.query(event["user_id"])[0]
    assert row["record_state"] == "inconsistent" and row["delivery_eligibility"] == "not_eligible"
