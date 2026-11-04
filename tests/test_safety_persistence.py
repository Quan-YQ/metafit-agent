import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Event, Thread

import pytest

from metafit_agent.decision_store import DecisionSQLiteStore
from metafit_agent.event_store import StorageError
from metafit_agent.model_response import ModelResponse
from metafit_agent.safety_coordinator import DurableSafetyCoordinator
from metafit_agent.safety_store import SafetySQLiteStore
from metafit_agent.snapshot_store import SnapshotSQLiteStore


TIME = "2026-11-03T08:00:00Z"
DEADLINE = "2026-11-03T08:02:00Z"


@pytest.fixture
def event():
    e = json.loads((Path(__file__).resolve().parents[1] / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    e["discomfort"] = {"status": "none", "source": "self_report"}
    return e


class Model:
    def __init__(self): self.calls = 0
    def complete(self, facts):
        self.calls += 1
        content = json.dumps({"suggested_action": "notify", "message": "本组完成了10次。",
                              "reason": "模拟状态仅供参考。", "evidence": ["fact_3"]}, ensure_ascii=False)
        return ModelResponse(content, "mock", "mock", None, None, 0, "stop")


def setup(store, event, *, new=True, ready=True):
    c = DurableSafetyCoordinator(store)
    result = c.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="clock", new_session=new)
    assert result.status == "registered"
    if new and ready:
        assert confirm(c, event["session_id"], "initial_confirm").status == "answered"
    return c


def confirm(c, session_id, question="confirm_1"):
    opened = c.apply(session_id, "open_question", question_id=question, kind="safety_confirmation",
                     now=TIME, deadline=DEADLINE, clock_id="clock")
    assert opened.status == "created"
    shown = c.apply(session_id, "present_question", question_id=question, now=TIME, clock_id="clock")
    assert shown.status == "presented" and len(shown.question_text) <= 60
    return c.apply(session_id, "answer_question", question_id=question, answer="none", now=TIME, clock_id="clock")


def test_report_survives_real_process_restart_without_model(tmp_path, event):
    path = tmp_path / "safety.db"
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json,sys
from metafit_agent.safety_store import SafetySQLiteStore
from metafit_agent.safety_coordinator import DurableSafetyCoordinator
e=json.loads(sys.stdin.read()); c=DurableSafetyCoordinator(SafetySQLiteStore(sys.argv[1]))
r=c.register(user_id=e['user_id'],session_id=e['session_id'],clock_id='clock',new_session=sys.argv[2]=='write')
class M:
 calls=0
 def complete(self,facts): self.calls+=1; raise AssertionError('must not call')
m=M()
if sys.argv[2]=='write':
 out=c.apply(e['session_id'],'report_discomfort',now='2026-11-03T08:00:00Z',clock_id='clock')
else:
 out=c.decide(e['session_id'],e,m,do_not_disturb=False)
print(json.dumps({'status':out.status,'durability':out.safety_state_durability,'calls':m.calls}))
'''
    results = []
    for mode in ("write", "read"):
        proc = subprocess.run([sys.executable, "-c", script, str(path), mode], cwd=root,
                              input=json.dumps(event), text=True, encoding="utf-8", capture_output=True, check=True, timeout=15)
        results.append(json.loads(proc.stdout))
    assert results == [{"status": "safety_pending", "durability": "durable", "calls": 0}] * 2
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT state,reported_at FROM safety_sessions").fetchone() == ("reported", "2026-11-03T08:00:00+00:00")
        assert conn.execute("SELECT action FROM safety_log").fetchall() == [("create",), ("report",), ("restore",)]
    reopened = DurableSafetyCoordinator(SafetySQLiteStore(path))
    restored = reopened.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="clock")
    assert restored.status == "safety_pending" and restored.safety_message and restored.safety_state_durability == "durable"
    assert "restored_source_unverified" in restored.warnings


def test_restart_pending_can_be_confirmed_but_each_restart_needs_reconfirmation(tmp_path, event):
    path = tmp_path / "safety.db"
    first = setup(SafetySQLiteStore(path), event)
    assert first.view(event["session_id"]).pending_confirmation is False
    c = setup(SafetySQLiteStore(path), event, new=False)
    model = Model()
    blocked = c.decide(event["session_id"], event, model, do_not_disturb=False)
    assert blocked.code == "pending_confirmation" and blocked.event_saved is True and model.calls == 0
    history = c._event_store.query(event["user_id"], session_id=event["session_id"], event_id=event["event_id"])[0]
    assert history["code"] == "pending_confirmation" and history["not_delivered_reason"] == "pending_confirmation"
    assert history["not_delivered_reason_source"] == "stored"
    result = confirm(c, event["session_id"])
    assert result.status == "answered" and result.safety_state_durability == "durable"
    assert not c.view(event["session_id"]).pending_confirmation
    assert c.decide(event["session_id"], event, model, do_not_disturb=False).code == "duplicate_event" and model.calls == 0
    subsequent = dict(event, event_id="after_confirmation")
    assert c.decide(event["session_id"], subsequent, model, do_not_disturb=False).status == "reviewed" and model.calls == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT confirmed_at FROM safety_sessions").fetchone()[0] is not None
        assert len(conn.execute("SELECT action FROM safety_log WHERE action='confirm'").fetchall()) == 2
    again = setup(SafetySQLiteStore(path), event, new=False)
    assert again.view(event["session_id"]).pending_confirmation


def test_fresh_session_stays_pending_until_explicit_confirmation(tmp_path, event):
    c = setup(SafetySQLiteStore(tmp_path / "safety.db"), event, ready=False)
    assert c.view(event["session_id"]).pending_confirmation
    model = Model()
    pending = c.decide(event["session_id"], event, model, do_not_disturb=False)
    assert pending.code == "pending_confirmation" and pending.event_saved is True and model.calls == 0
    assert pending.pending_message and "确认" in pending.pending_message
    assert confirm(c, event["session_id"]).status == "answered"
    subsequent = dict(event, event_id="after_confirmation")
    assert c.decide(event["session_id"], subsequent, model, do_not_disturb=False).status == "reviewed"


def test_old_pending_history_is_interpreted_without_changing_database(tmp_path, event):
    path = tmp_path / "safety.db"
    c = setup(SafetySQLiteStore(path), event, ready=False)
    assert c.decide(event["session_id"], event, Model(), do_not_disturb=False).code == "pending_confirmation"
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE decisions SET not_delivered_reason='decision_stopped' WHERE event_id=?", (event["event_id"],))
    history = SafetySQLiteStore(path).query(event["user_id"], event_id=event["event_id"])[0]
    assert history["not_delivered_reason"] == "pending_confirmation" and history["not_delivered_reason_source"] == "legacy_code"
    assert DecisionSQLiteStore(path).query(event["user_id"], event_id=event["event_id"])[0]["not_delivered_reason_source"] == "legacy_code"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT not_delivered_reason FROM decisions WHERE event_id=?", (event["event_id"],)).fetchone()[0] == "decision_stopped"


def test_invalid_confirmation_time_never_unblocks(tmp_path, event):
    c = setup(SafetySQLiteStore(tmp_path / "safety.db"), event, ready=False)
    sid = event["session_id"]
    result = c.apply(sid, "open_question", question_id="q", kind="safety_confirmation",
                     now=None, deadline=DEADLINE, clock_id="clock")
    assert result.status == "error" and c.view(sid).pending_confirmation
    model = Model()
    assert c.decide(sid, event, model, do_not_disturb=False).code == "pending_confirmation" and model.calls == 0


def test_safety_simulation_never_connects_to_network(tmp_path, event, monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: pytest.fail("network access"))
    c = setup(SafetySQLiteStore(tmp_path / "safety.db"), event)
    assert c.decide(event["session_id"], event, Model(), do_not_disturb=False).status == "reviewed"


def test_failed_report_stays_safe_retries_and_preserves_first_time(tmp_path, event):
    class FailReport(SafetySQLiteStore):
        fail = True
        def report(self, *args, **kwargs):
            if self.fail: raise StorageError("storage_failed")
            return super().report(*args, **kwargs)
    store = FailReport(tmp_path / "safety.db")
    c = setup(store, event)
    report = c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")
    assert report.status == "safety_pending" and report.safety_state_durability == "volatile"
    assert "storage_failed" in report.warnings
    before_retry = Model()
    assert c.decide(event["session_id"], event, before_retry, do_not_disturb=False).safety_state_durability == "volatile"
    assert before_retry.calls == 0
    store.fail = False
    model = Model()
    safe = c.decide(event["session_id"], event, model, do_not_disturb=False)
    assert safe.status == "safety_pending" and safe.safety_state_durability == "durable" and model.calls == 0
    c.apply(event["session_id"], "report_discomfort", now=DEADLINE, clock_id="clock")
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT reported_at FROM safety_sessions").fetchone()[0] == "2026-11-03T08:00:00+00:00"
        assert conn.execute("SELECT count(*) FROM safety_log WHERE action='report'").fetchone()[0] == 1


def test_reported_event_with_safety_write_failure_never_calls_model(tmp_path, event):
    class FailReport(SafetySQLiteStore):
        def report(self, *args, **kwargs): raise StorageError("storage_failed")
    store = FailReport(tmp_path / "safety.db")
    c = setup(store, event)
    reported = dict(event, discomfort={"status": "reported", "source": "self_report"})
    model = Model()
    result = c.decide(event["session_id"], reported, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.safety_message and result.safety_source == "event"
    assert result.safety_state_durability == "volatile" and "storage_failed" in result.warnings
    assert result.event_saved is True and model.calls == 0


def test_failed_report_then_restart_is_pending_not_unsafe(tmp_path, event):
    class FailReport(SafetySQLiteStore):
        def report(self, *args, **kwargs): raise StorageError("storage_failed")
    path = tmp_path / "safety.db"
    c = setup(FailReport(path), event)
    assert c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock").safety_state_durability == "volatile"
    restarted = setup(SafetySQLiteStore(path), event, new=False)
    model = Model()
    result = restarted.decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.code == "pending_confirmation" and model.calls == 0


def test_unwritten_report_across_real_process_restart_fails_closed(tmp_path, event):
    path = tmp_path / "safety.db"
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json,sys
from metafit_agent.event_store import StorageError
from metafit_agent.safety_store import SafetySQLiteStore
from metafit_agent.safety_coordinator import DurableSafetyCoordinator
e=json.loads(sys.stdin.read())
class FailReport(SafetySQLiteStore):
 def report(self,*a,**k): raise StorageError('storage_failed')
s=FailReport(sys.argv[1]) if sys.argv[2]=='write' else SafetySQLiteStore(sys.argv[1])
c=DurableSafetyCoordinator(s); c.register(user_id=e['user_id'],session_id=e['session_id'],clock_id='clock',new_session=sys.argv[2]=='write')
class M:
 calls=0
 def complete(self,facts): self.calls+=1; raise AssertionError('must not call')
m=M()
if sys.argv[2]=='write': out=c.apply(e['session_id'],'report_discomfort',now=None,clock_id='clock')
else: out=c.decide(e['session_id'],e,m,do_not_disturb=False)
print(json.dumps({'status':out.status,'code':out.code,'durability':out.safety_state_durability,'calls':m.calls}))
'''
    outputs = []
    for mode in ("write", "read"):
        proc = subprocess.run([sys.executable, "-c", script, str(path), mode], cwd=root,
                              input=json.dumps(event), text=True, encoding="utf-8", capture_output=True, check=True, timeout=15)
        outputs.append(json.loads(proc.stdout))
    assert outputs[0]["status"] == "safety_pending" and outputs[0]["durability"] == "volatile"
    assert outputs[1]["status"] == "stopped" and outputs[1]["code"] == "pending_confirmation" and outputs[1]["calls"] == 0


def test_pending_unknown_or_skipped_answer_never_unlocks(tmp_path, event):
    store = SafetySQLiteStore(tmp_path / "safety.db")
    c = setup(store, event, ready=False)
    sid = event["session_id"]
    assert c.view(sid).pending_confirmation
    assert c.apply(sid, "open_question", question_id="q", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock").status == "created"
    assert c.apply(sid, "present_question", question_id="q", now=TIME, clock_id="clock").status == "presented"
    assert c.apply(sid, "answer_question", question_id="q", answer="unknown", now=TIME, clock_id="clock").status == "answered"
    assert c.view(sid).pending_confirmation
    assert c.apply(sid, "open_question", question_id="q2", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock").code == "confirmation_closed"
    model = Model()
    assert c.decide(sid, event, model, do_not_disturb=False).code == "pending_confirmation" and model.calls == 0


@pytest.mark.parametrize("terminal", ["skip", "expire"])
def test_confirmation_skip_or_expiry_is_not_reasked_after_restart(tmp_path, event, terminal):
    path = tmp_path / "safety.db"
    c = setup(SafetySQLiteStore(path), event, ready=False)
    sid = event["session_id"]
    c.apply(sid, "open_question", question_id="q", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock")
    if terminal == "skip":
        c.apply(sid, "present_question", question_id="q", now=TIME, clock_id="clock")
        outcome = c.apply(sid, "skip_question", question_id="q", now=TIME, clock_id="clock")
        assert outcome.status == "skipped"
    else:
        outcome = c.apply(sid, "present_question", question_id="q", now="2026-11-03T08:03:00Z", clock_id="clock")
        assert outcome.status == "expired" and outcome.question_text is None
    assert outcome.safety_state_durability == "durable"
    restarted = setup(SafetySQLiteStore(path), event, new=False)
    assert restarted.view(sid).pending_confirmation
    retry = restarted.apply(sid, "open_question", question_id="q2", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock")
    assert retry.code == "confirmation_closed" and "新会话" in retry.pending_message


def test_skipped_confirmation_explains_how_to_continue_in_new_session(tmp_path, event):
    store = SafetySQLiteStore(tmp_path / "safety.db")
    c = setup(store, event, ready=False)
    sid = event["session_id"]
    c.apply(sid, "open_question", question_id="q", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock")
    c.apply(sid, "present_question", question_id="q", now=TIME, clock_id="clock")
    skipped = c.apply(sid, "skip_question", question_id="q", now=TIME, clock_id="clock")
    assert skipped.status == "skipped" and "新会话" in skipped.pending_message
    new = c.register(user_id=event["user_id"], session_id="new_session", clock_id="clock", new_session=True)
    assert new.status == "registered" and new.view.pending_confirmation
    assert confirm(c, "new_session").status == "answered"
    next_event = dict(event, session_id="new_session", event_id="new_session_event")
    model = Model()
    assert c.decide("new_session", next_event, model, do_not_disturb=False).status == "reviewed" and model.calls == 1


def test_skipped_confirmation_recovers_after_initial_store_failure(tmp_path, event):
    class FlakyOpen(SafetySQLiteStore):
        fail = True
        def open_session(self, *args, **kwargs):
            if self.fail: raise StorageError("storage_failed")
            return super().open_session(*args, **kwargs)
    store = FlakyOpen(tmp_path / "safety.db")
    c = setup(store, event, new=False)
    sid = event["session_id"]
    c.apply(sid, "open_question", question_id="q", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock")
    c.apply(sid, "present_question", question_id="q", now=TIME, clock_id="clock")
    skipped = c.apply(sid, "skip_question", question_id="q", now=TIME, clock_id="clock")
    assert skipped.status == "skipped" and skipped.safety_state_durability == "volatile"
    store.fail = False
    c.apply(sid, "mark_training_started", now=TIME, clock_id="clock")
    assert c.view(sid).safety_state_durability == "durable"
    restarted = setup(SafetySQLiteStore(store.path), event, new=False)
    retry = restarted.apply(sid, "open_question", question_id="q2", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock")
    assert retry.code == "confirmation_closed"


def test_failed_confirmation_stays_pending_and_can_retry(tmp_path, event):
    class FailConfirm(SafetySQLiteStore):
        fail = True
        def confirm(self, *args, **kwargs):
            if self.fail: raise StorageError("storage_failed")
            return super().confirm(*args, **kwargs)
    store = FailConfirm(tmp_path / "safety.db")
    c = setup(store, event, ready=False)
    sid = event["session_id"]
    assert confirm(c, sid).code == "pending_confirmation"
    assert c.view(sid).pending_confirmation and c.view(sid).safety_state_durability == "volatile"
    store.fail = False
    retried = c.apply(sid, "answer_question", question_id="confirm_1", answer="none", now=TIME, clock_id="clock")
    assert retried.status == "answered" and retried.safety_state_durability == "durable"
    assert not c.view(sid).pending_confirmation


def test_reported_session_cannot_be_cleared_by_none_confirmation(tmp_path, event):
    c = setup(SafetySQLiteStore(tmp_path / "safety.db"), event)
    c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")
    result = c.apply(event["session_id"], "answer_question", question_id="initial_confirm", answer="none", now=TIME, clock_id="clock")
    assert result.status == "safety_pending" and c.view(event["session_id"]).safety_reported


def test_prior_user_report_makes_new_session_pending(tmp_path, event):
    store = SafetySQLiteStore(tmp_path / "safety.db")
    c = setup(store, event)
    c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")
    assert c.register(user_id=event["user_id"], session_id="new_session", clock_id="clock", new_session=True).view.pending_confirmation
    next_event = dict(event, session_id="new_session", event_id="new_event")
    model = Model()
    assert c.decide("new_session", next_event, model, do_not_disturb=False).code == "pending_confirmation"
    assert model.calls == 0
    assert confirm(c, "new_session").status == "answered"
    assert c.decide("new_session", next_event, model, do_not_disturb=False).code == "duplicate_event"
    next_event["event_id"] = "new_event_after_confirmation"
    assert c.decide("new_session", next_event, model, do_not_disturb=False).status == "reviewed"
    assert c.view(event["session_id"]).safety_reported
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT last_discomfort_at FROM safety_users").fetchone()[0] == "2026-11-03T08:00:00+00:00"


def test_report_in_other_session_discards_inflight_feedback(tmp_path, event):
    store = SafetySQLiteStore(tmp_path / "safety.db")
    c = setup(store, event)
    assert c.register(user_id=event["user_id"], session_id="second", clock_id="clock", new_session=True).status == "registered"
    assert confirm(c, "second", "initial_second").status == "answered"
    other = dict(event, session_id="second", event_id="second_event")
    entered, release = Event(), Event()
    class Blocking(Model):
        def complete(self, facts):
            entered.set()
            assert release.wait(5)
            return super().complete(facts)
    results = []
    thread = Thread(target=lambda: results.append(c.decide("second", other, Blocking(), do_not_disturb=False)))
    thread.start()
    try:
        assert entered.wait(3)
        assert c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock").status == "safety_pending"
    finally:
        release.set()
        thread.join(8)
    assert not thread.is_alive() and results[0].status == "stopped" and results[0].candidate is None
    assert c.view("second").pending_confirmation


def test_user_report_during_event_storage_prevents_model_start(tmp_path, event):
    entered, release = Event(), Event()
    class SlowClaim(SafetySQLiteStore):
        def claim(self, data):
            entered.set()
            assert release.wait(5)
            return super().claim(data)
    store = SlowClaim(tmp_path / "safety.db")
    c = setup(store, event)
    assert c.register(user_id=event["user_id"], session_id="second", clock_id="clock", new_session=True).status == "registered"
    assert confirm(c, "second", "initial_second").status == "answered"
    other, model, results = dict(event, session_id="second", event_id="second_event"), Model(), []
    thread = Thread(target=lambda: results.append(c.decide("second", other, model, do_not_disturb=False)))
    thread.start()
    try:
        assert entered.wait(3)
        assert c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock").status == "safety_pending"
    finally:
        release.set()
        thread.join(8)
    assert not thread.is_alive() and results[0].code == "pending_confirmation" and model.calls == 0
    history = store.query(event["user_id"], session_id="second", event_id="second_event")[0]
    assert history["code"] == "pending_confirmation" and history["not_delivered_reason"] == "pending_confirmation"


def test_user_report_during_snapshot_write_marks_history_discarded(tmp_path, event):
    entered, release, reporting = Event(), Event(), Event()
    class SlowSnapshot(SafetySQLiteStore):
        def _insert_decision_extras(self, conn, key, result):
            if key[2] == "second_event":
                entered.set()
                assert release.wait(5)
            return super()._insert_decision_extras(conn, key, result)
        def report(self, *args, **kwargs):
            reporting.set()
            return super().report(*args, **kwargs)
    store = SlowSnapshot(tmp_path / "safety.db")
    c = setup(store, event)
    c.register(user_id=event["user_id"], session_id="second", clock_id="clock", new_session=True)
    assert confirm(c, "second", "initial_second").status == "answered"
    other, results = dict(event, session_id="second", event_id="second_event"), []
    worker = Thread(target=lambda: results.append(c.decide("second", other, Model(), do_not_disturb=False)))
    worker.start()
    report_worker = None
    try:
        assert entered.wait(3)
        report_worker = Thread(target=lambda: c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock"))
        report_worker.start()
        assert reporting.wait(3)
    finally:
        release.set()
        worker.join(8)
        if report_worker: report_worker.join(8)
    assert not worker.is_alive() and report_worker is not None and not report_worker.is_alive()
    assert results[0].code == "pending_confirmation" and results[0].candidate is None
    row = store.query(event["user_id"], session_id="second", event_id="second_event")[0]
    assert row["not_delivered_reason"] == "safety_discarded"


def test_report_during_pending_event_save_remains_safety_pending(tmp_path, event):
    entered, release = Event(), Event()
    class SlowClaim(SafetySQLiteStore):
        def claim(self, data):
            entered.set()
            assert release.wait(5)
            return super().claim(data)
    store = SlowClaim(tmp_path / "safety.db")
    c = setup(store, event, ready=False)
    model, results = Model(), []
    thread = Thread(target=lambda: results.append(c.decide(event["session_id"], event, model, do_not_disturb=False)))
    thread.start()
    try:
        assert entered.wait(3)
        assert c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock").status == "safety_pending"
    finally:
        release.set()
        thread.join(8)
    assert not thread.is_alive() and results[0].status == "safety_pending" and results[0].event_saved is True
    assert model.calls == 0


def test_new_session_cannot_bypass_user_report_still_being_written(tmp_path, event):
    entered, release = Event(), Event()
    class SlowReport(SafetySQLiteStore):
        def report(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().report(*args, **kwargs)
    store = SlowReport(tmp_path / "safety.db")
    c = setup(store, event)
    results = []
    thread = Thread(target=lambda: results.append(c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")))
    thread.start()
    try:
        assert entered.wait(3)
        created = c.register(user_id=event["user_id"], session_id="new_session", clock_id="clock", new_session=True)
        assert created.view.pending_confirmation and created.safety_state_durability == "volatile"
        # 不等待持久化调用即可验证当前实例内的待确认状态。
    finally:
        release.set()
        thread.join(8)
    assert not thread.is_alive() and results[0].safety_state_durability == "durable"
    model = Model()
    new_event = dict(event, session_id="new_session", event_id="new_event")
    assert c.decide("new_session", new_event, model, do_not_disturb=False).code == "pending_confirmation"
    assert model.calls == 0


def test_later_report_during_confirmation_cannot_clear_pending(tmp_path, event):
    entered, release = Event(), Event()
    class SlowConfirm(SafetySQLiteStore):
        enabled = False
        def confirm(self, *args, **kwargs):
            if self.enabled:
                entered.set()
                assert release.wait(5)
            return super().confirm(*args, **kwargs)
    store = SlowConfirm(tmp_path / "safety.db")
    c = setup(store, event)
    c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")
    assert c.register(user_id=event["user_id"], session_id="second", clock_id="clock", new_session=True).view.pending_confirmation
    c.apply("second", "open_question", question_id="confirm_1", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock")
    c.apply("second", "present_question", question_id="confirm_1", now=TIME, clock_id="clock")
    store.enabled = True
    results = []
    thread = Thread(target=lambda: results.append(c.apply("second", "answer_question", question_id="confirm_1", answer="none", now=TIME, clock_id="clock")))
    thread.start()
    try:
        assert entered.wait(3)
        c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")
    finally:
        release.set()
        thread.join(8)
    assert not thread.is_alive() and results[0].code == "pending_confirmation"
    assert c.view("second").pending_confirmation
    model = Model()
    other = dict(event, session_id="second", event_id="second_event")
    assert c.decide("second", other, model, do_not_disturb=False).code == "pending_confirmation" and model.calls == 0
    store.enabled = False
    assert confirm(c, "second", "confirm_new_report").status == "answered"


def test_same_session_report_during_confirmation_returns_safety_prompt(tmp_path, event):
    entered, release = Event(), Event()
    class SlowConfirm(SafetySQLiteStore):
        def confirm(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().confirm(*args, **kwargs)
    store = SlowConfirm(tmp_path / "safety.db")
    c = setup(store, event, ready=False)
    sid = event["session_id"]
    c.apply(sid, "open_question", question_id="q", kind="safety_confirmation", now=TIME, deadline=DEADLINE, clock_id="clock")
    c.apply(sid, "present_question", question_id="q", now=TIME, clock_id="clock")
    outputs = []
    worker = Thread(target=lambda: outputs.append(c.apply(sid, "answer_question", question_id="q", answer="none", now=TIME, clock_id="clock")))
    worker.start()
    try:
        assert entered.wait(3)
        reported = c.apply(sid, "report_discomfort", now=TIME, clock_id="clock")
        assert reported.status == "safety_pending" and reported.safety_state_durability == "durable"
    finally:
        release.set()
        worker.join(8)
    assert not worker.is_alive() and outputs[0].status == "safety_pending" and outputs[0].safety_message
    assert c.view(sid).safety_reported


def test_corrupt_missing_and_unavailable_are_pending(tmp_path, event):
    path = tmp_path / "safety.db"
    setup(SafetySQLiteStore(path), event)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE safety_sessions SET state='corrupt'")
    c = setup(SafetySQLiteStore(path), event, new=False)
    model = Model()
    assert c.view(event["session_id"]).pending_confirmation
    assert c.decide(event["session_id"], event, model, do_not_disturb=False).code == "pending_confirmation"
    assert model.calls == 0
    broken = tmp_path / "unreadable.db"
    broken.write_bytes(b"not a database")
    unavailable = setup(SafetySQLiteStore(broken), dict(event, session_id="unavailable"), new=False)
    other = dict(event, session_id="unavailable")
    assert unavailable.decide("unavailable", other, Model(), do_not_disturb=False).code == "pending_confirmation"
    assert unavailable.apply("unavailable", "report_discomfort", now=None, clock_id="clock").status == "safety_pending"
    other["discomfort"] = {"status": "reported"}
    safe = unavailable.decide("unavailable", other, Model(), do_not_disturb=False)
    assert safe.status == "safety_pending" and safe.safety_state_durability == "volatile" and safe.safety_message


def test_known_report_with_corrupt_user_generation_remains_safe(tmp_path, event):
    path = tmp_path / "safety.db"
    c = setup(SafetySQLiteStore(path), event)
    c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE safety_users SET report_generation=-1")
    restarted = DurableSafetyCoordinator(SafetySQLiteStore(path))
    restored = restarted.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="clock")
    assert restored.status == "safety_pending" and restored.safety_state_durability == "volatile"
    model = Model()
    safe = restarted.decide(event["session_id"], event, model, do_not_disturb=False)
    assert safe.status == "safety_pending" and safe.safety_state_durability == "volatile" and model.calls == 0


def test_recovered_existing_report_keeps_persisted_first_timestamp(tmp_path, event):
    path = tmp_path / "safety.db"
    first = setup(SafetySQLiteStore(path), event)
    first.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")
    class FlakyOpen(SafetySQLiteStore):
        def open_session(self, *args, **kwargs): raise StorageError("storage_failed")
    c = setup(FlakyOpen(path), event, new=False)
    result = c.apply(event["session_id"], "report_discomfort", now=DEADLINE, clock_id="clock")
    assert result.status == "safety_pending" and result.safety_state_durability == "durable"
    assert c.view(event["session_id"]).safety_reported_at.isoformat() == "2026-11-03T08:00:00+00:00"


def test_explicit_event_report_without_source_is_safe_and_scope_isolated(tmp_path, event):
    c = setup(SafetySQLiteStore(tmp_path / "safety.db"), event)
    mismatch = dict(event, user_id="someone_else", discomfort={"status": "reported"})
    stopped = c.decide(event["session_id"], mismatch, Model(), do_not_disturb=False)
    assert stopped.code == "scope_mismatch" and stopped.safety_message is None
    bad = dict(event, discomfort={"status": "reported"})
    model = Model()
    safe = c.decide(event["session_id"], bad, model, do_not_disturb=False)
    assert safe.status == "safety_pending" and safe.safety_source == "event" and model.calls == 0
    assert "discomfort_source_unverified" in safe.warnings and safe.safety_state_durability == "durable"


def test_delete_invalidates_and_waits_for_inflight_operations(tmp_path, event):
    path = tmp_path / "safety.db"
    store = SafetySQLiteStore(path)
    c = setup(store, event)
    entered, release = Event(), Event()
    class Blocking(Model):
        def complete(self, facts):
            entered.set()
            assert release.wait(5)
            return super().complete(facts)
    results = []
    thread = Thread(target=lambda: results.append(c.decide(event["session_id"], event, Blocking(), do_not_disturb=False)))
    thread.start()
    try:
        assert entered.wait(3)
        assert c.delete_user(event["user_id"])["code"] == "operations_in_progress"
    finally:
        release.set()
        thread.join(8)
    assert not thread.is_alive() and results[0].code == "session_invalidated" and results[0].candidate is None
    deleted = c.delete_user(event["user_id"], allow_unresolved=True)
    assert deleted["status"] == "deleted"
    assert c.delete_user(event["user_id"])["already_deleted"] is True
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM safety_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM safety_log").fetchone()[0] == 0
    restarted = setup(SafetySQLiteStore(path), event, new=False)
    assert restarted.view(event["session_id"]).pending_confirmation


def test_concurrent_delete_does_not_resurrect_safety_state(tmp_path, event):
    entered, release = Event(), Event()
    class SlowDelete(SafetySQLiteStore):
        def invalidate_session(self, *args):
            entered.set()
            assert release.wait(5)
            return super().invalidate_session(*args)
    store = SlowDelete(tmp_path / "safety.db")
    c = setup(store, event)
    outputs = []
    thread = Thread(target=lambda: outputs.append(c.delete_user(event["user_id"])))
    thread.start()
    try:
        assert entered.wait(3)
        assert c.delete_user(event["user_id"])["code"] == "deletion_in_progress"
    finally:
        release.set()
        thread.join(8)
    assert not thread.is_alive() and outputs[0]["status"] == "deleted"
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM safety_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM safety_log").fetchone()[0] == 0


def test_delete_during_registration_never_reports_session_ready(tmp_path, event):
    entered, release = Event(), Event()
    class SlowOpen(SafetySQLiteStore):
        def open_session(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().open_session(*args, **kwargs)
    store = SlowOpen(tmp_path / "safety.db")
    c = DurableSafetyCoordinator(store)
    outputs = []
    worker = Thread(target=lambda: outputs.append(c.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="clock")))
    worker.start()
    try:
        assert entered.wait(3)
        assert c.delete_user(event["user_id"])["code"] == "operations_in_progress"
    finally:
        release.set()
        worker.join(8)
    assert not worker.is_alive() and outputs[0].code == "session_invalidated"
    assert c.delete_user(event["user_id"])["status"] == "deleted"


def test_legacy_delete_also_removes_v5_safety_records(tmp_path, event):
    path = tmp_path / "safety.db"
    c = setup(SafetySQLiteStore(path), event)
    c.apply(event["session_id"], "report_discomfort", now=TIME, clock_id="clock")
    removed = DecisionSQLiteStore(path).delete_user(event["user_id"])
    assert removed["safety_sessions_deleted"] == 1 and removed["safety_logs_deleted"] >= 1


def test_invalidate_persists_and_log_can_be_queried(tmp_path, event):
    path = tmp_path / "safety.db"
    store = SafetySQLiteStore(path)
    c = setup(store, event)
    result = c.invalidate(event["session_id"])
    assert result.status == "invalidated" and result.safety_state_durability == "durable"
    logs = store.list_safety_log(event["user_id"], session_id=event["session_id"])
    assert logs[0]["action"] == "invalidate" and any(item["action"] == "confirm" for item in logs)
    reopened = DurableSafetyCoordinator(SafetySQLiteStore(path))
    assert reopened.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="clock").code == "session_invalidated"


def test_v4_to_v5_migration_failure_rolls_back(tmp_path, event):
    path = tmp_path / "history.db"
    previous = SnapshotSQLiteStore(path)
    previous.claim(event)
    class Proxy:
        def __init__(self, conn): self.conn = conn
        def execute(self, sql, *args):
            if sql.startswith("CREATE TABLE safety_users"):
                raise sqlite3.OperationalError("injected")
            return self.conn.execute(sql, *args)
        def __getattr__(self, name): return getattr(self.conn, name)
    class BrokenUpgrade(SafetySQLiteStore):
        def _connect(self): return Proxy(super()._connect())
    with pytest.raises(StorageError, match="storage_failed"):
        BrokenUpgrade(path).open_session(event["user_id"], event["session_id"])
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='safety_sessions'").fetchone()[0] == 0
    assert SnapshotSQLiteStore(path).get(event["user_id"], event["session_id"], event["event_id"]) is not None
