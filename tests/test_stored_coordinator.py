"""R5 协调器与文件存储的端到端模拟测试。"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

import pytest

from metafit_agent.event_store import SQLiteEventStore, StorageError
from metafit_agent.model_response import ModelResponse
from metafit_agent.stored_coordinator import StoredSimulationCoordinator


CLOCK = "event_clock"
TIME = "2026-11-02T18:30:00+08:00"


@pytest.fixture
def event():
    path = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["discomfort"] = {"status": "none", "source": "self_report"}
    return data


class Model:
    def __init__(self):
        self.calls = 0

    def complete(self, facts):
        self.calls += 1
        content = json.dumps({"suggested_action": "notify", "message": "本组完成了10次。",
                              "reason": "模拟状态仅供参考。", "evidence": ["fact_3"]}, ensure_ascii=False)
        return ModelResponse(content, "mock", "mock", None, None, 0, "stop")


def setup(store, event):
    coordinator = StoredSimulationCoordinator(store)
    assert coordinator.register(user_id=event["user_id"], session_id=event["session_id"], clock_id=CLOCK).status == "registered"
    return coordinator


def test_duplicate_after_new_coordinator_and_connection_never_calls_model(tmp_path, event):
    path = tmp_path / "events.db"
    first_model = Model()
    first = setup(SQLiteEventStore(path), event).decide(event["session_id"], event, first_model, do_not_disturb=False)
    assert first.status == "reviewed" and first.event_saved is True and first_model.calls == 1
    retry_model = Model()
    retry = setup(SQLiteEventStore(path), event).decide(event["session_id"], event, retry_model, do_not_disturb=False)
    assert retry.status == "stopped" and retry.code == "duplicate_event" and retry.stage == "storage"
    assert retry_model.calls == 0 and retry.stored_event.event_dict()["event_id"] == event["event_id"]


def test_previously_reserved_event_never_automatically_retries(tmp_path, event):
    path = tmp_path / "events.db"
    assert SQLiteEventStore(path).claim(event).record.decision_state == "reserved"
    model = Model()
    result = setup(SQLiteEventStore(path), event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.code == "duplicate_event" and result.stored_event.decision_state == "reserved"
    assert result.event_saved is True and model.calls == 0


def test_identity_conflict_does_not_overwrite_or_call_model(tmp_path, event):
    store = SQLiteEventStore(tmp_path / "events.db")
    assert store.claim(event).status == "inserted"
    changed = deepcopy(event)
    changed["completed_reps"] = 11
    model = Model()
    result = setup(store, event).decide(event["session_id"], changed, model, do_not_disturb=False)
    assert (result.status, result.code, result.stage, result.event_saved) == ("stopped", "event_conflict", "storage", False)
    assert model.calls == 0 and store.get(event["user_id"], event["session_id"], event["event_id"]).event_dict()["completed_reps"] == 10


def test_safety_still_returns_when_event_identity_conflicts(tmp_path, event):
    store = SQLiteEventStore(tmp_path / "events.db")
    assert store.claim(event).status == "inserted"
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    model = Model()
    result = setup(store, event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.event_saved is False and model.calls == 0
    assert "event_identity_conflict" in result.warnings


class FailingStore:
    def claim(self, event_data):
        raise StorageError("storage_failed")

    def finish(self, user_id, session_id, event_id, *, state):
        raise StorageError("storage_failed")


def test_storage_failure_stops_without_model(event):
    model = Model()
    result = setup(FailingStore(), event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert (result.status, result.code, result.stage, result.event_saved) == ("stopped", "storage_failed", "storage", None)
    assert "storage_failed" in result.warnings and model.calls == 0


def test_storage_failure_never_suppresses_safety(event):
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    model = Model()
    result = setup(FailingStore(), event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.safety_message and result.safety_source == "event"
    assert result.event_saved is None and "storage_failed" in result.warnings and model.calls == 0


def test_valid_safety_event_is_saved_without_calling_model(tmp_path, event):
    store = SQLiteEventStore(tmp_path / "events.db")
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    model = Model()
    result = setup(store, event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.event_saved is True and model.calls == 0
    assert result.stored_event.decision_state == "safety_only"
    assert store.get(event["user_id"], event["session_id"], event["event_id"]).decision_state == "safety_only"


def test_safety_demo_shows_confirmed_marker(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, "-m", "metafit_agent.demo_storage", "--scenario", "safety",
                          "--db", str(tmp_path / "demo.db")], cwd=root,
                         capture_output=True, text=True, encoding="utf-8", check=True, timeout=10)
    output = json.loads(run.stdout)
    assert (output["status"], output["decision_state"], output["model_calls"]) == ("safety_pending", "safety_only", 0)


def test_invalid_safety_event_is_not_stored_but_safety_is_returned(tmp_path, event):
    store = SQLiteEventStore(tmp_path / "events.db")
    event["discomfort"] = {"status": "reported"}
    model = Model()
    result = setup(store, event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.event_saved is False and model.calls == 0
    assert "event_not_saved_invalid" in result.warnings and "discomfort_source_unverified" in result.warnings
    assert store.get(event["user_id"], event["session_id"], event["event_id"]) is None


def test_corrupted_database_stops_normal_but_not_safety(tmp_path, event):
    path = tmp_path / "corrupt.db"
    path.write_bytes(b"not a sqlite database")
    store = SQLiteEventStore(path)
    model = Model()
    coordinator = setup(store, event)
    normal = coordinator.decide(event["session_id"], event, model, do_not_disturb=False)
    assert normal.code == "storage_failed" and normal.stage == "storage" and model.calls == 0
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    safety = coordinator.decide(event["session_id"], event, model, do_not_disturb=False)
    assert safety.status == "safety_pending" and "storage_failed" in safety.warnings and model.calls == 0


def test_storage_wait_does_not_hold_session_lock_and_safety_prevents_model(tmp_path, event):
    entered, release = Event(), Event()
    delegate = SQLiteEventStore(tmp_path / "events.db")
    class BlockingStore:
        def claim(self, data):
            entered.set()
            if not release.wait(5):
                raise StorageError("storage_failed")
            return delegate.claim(data)
        def finish(self, user_id, session_id, event_id, *, state):
            return delegate.finish(user_id, session_id, event_id, state=state)
    coordinator = setup(BlockingStore(), event)
    model, results = Model(), []
    worker = Thread(target=lambda: results.append(coordinator.decide(event["session_id"], event, model, do_not_disturb=False)), daemon=True)
    worker.start()
    try:
        assert entered.wait(3)
        safety_results = []
        safety = Thread(target=lambda: safety_results.append(coordinator.apply(event["session_id"], "report_discomfort", now=TIME, clock_id=CLOCK)), daemon=True)
        safety.start()
        safety.join(2)
        assert not safety.is_alive() and safety_results[0].status == "safety_pending"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and results[0].status == "safety_pending" and model.calls == 0
    assert results[0].event_saved is True and delegate.get(event["user_id"], event["session_id"], event["event_id"]).decision_state == "safety_only"


def test_storage_failure_during_concurrent_discomfort_still_returns_safety(event):
    entered, release = Event(), Event()
    class BlockingFailure:
        def claim(self, data):
            entered.set()
            if not release.wait(5):
                raise StorageError("storage_failed")
            raise StorageError("storage_failed")
        def finish(self, user_id, session_id, event_id, *, state):
            raise StorageError("storage_failed")
    coordinator = setup(BlockingFailure(), event)
    model, results = Model(), []
    worker = Thread(target=lambda: results.append(coordinator.decide(event["session_id"], event, model, do_not_disturb=False)), daemon=True)
    worker.start()
    try:
        assert entered.wait(3)
        assert coordinator.apply(event["session_id"], "report_discomfort", now=TIME, clock_id=CLOCK).status == "safety_pending"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and results[0].status == "safety_pending" and model.calls == 0
    assert results[0].event_saved is None and "storage_failed" in results[0].warnings


def test_marker_write_failure_discards_ordinary_candidate(tmp_path, event):
    delegate = SQLiteEventStore(tmp_path / "events.db")
    class FailFinish:
        def claim(self, data): return delegate.claim(data)
        def finish(self, user_id, session_id, event_id, *, state): raise StorageError("storage_failed")
    model = Model()
    result = setup(FailFinish(), event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert (result.status, result.code, result.stage) == ("stopped", "storage_failed", "storage")
    assert result.event_saved is True and result.storage_status == "saved_state_unconfirmed"
    assert result.candidate is None and result.candidate_discarded and model.calls == 1
    assert "decision_state_update_failed" in result.warnings


def test_marker_write_failure_does_not_suppress_safety(tmp_path, event):
    delegate = SQLiteEventStore(tmp_path / "events.db")
    class FailFinish:
        def claim(self, data): return delegate.claim(data)
        def finish(self, user_id, session_id, event_id, *, state): raise StorageError("storage_failed")
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    model = Model()
    result = setup(FailFinish(), event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.event_saved is True and model.calls == 0
    assert result.storage_status == "saved_state_unconfirmed" and "decision_state_update_failed" in result.warnings
    assert result.stored_event is None
    assert delegate.get(event["user_id"], event["session_id"], event["event_id"]).decision_state == "reserved"


def test_marker_failure_after_model_reports_discomfort_still_returns_safety(tmp_path, event):
    delegate = SQLiteEventStore(tmp_path / "events.db")
    class FailFinish:
        def claim(self, data): return delegate.claim(data)
        def finish(self, user_id, session_id, event_id, *, state): raise StorageError("storage_failed")
    coordinator = setup(FailFinish(), event)
    class ReportsDuringModel(Model):
        def complete(self, facts):
            assert coordinator.apply(event["session_id"], "report_discomfort", now=TIME, clock_id=CLOCK).status == "safety_pending"
            return super().complete(facts)
    model = ReportsDuringModel()
    result = coordinator.decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.candidate_discarded and result.candidate is None
    assert "decision_state_update_failed" in result.warnings and result.event_saved is True and model.calls == 1


def test_directory_permission_failure_stops_normal_but_not_safety(tmp_path, event, monkeypatch):
    blocked_parent = tmp_path / "blocked"
    original = Path.mkdir
    def reject_parent(self, *args, **kwargs):
        if self == blocked_parent:
            raise PermissionError("private filesystem details")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", reject_parent)
    store = SQLiteEventStore(blocked_parent / "events.db")
    model = Model()
    coordinator = setup(store, event)
    stopped = coordinator.decide(event["session_id"], event, model, do_not_disturb=False)
    assert (stopped.status, stopped.code, stopped.stage) == ("stopped", "storage_failed", "storage")
    assert model.calls == 0 and "private filesystem details" not in repr(stopped)
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    safety = coordinator.decide(event["session_id"], event, model, do_not_disturb=False)
    assert safety.status == "safety_pending" and "storage_failed" in safety.warnings and model.calls == 0


def test_unknown_claim_outcome_does_not_infer_failure_or_retry_model(tmp_path, event):
    delegate = SQLiteEventStore(tmp_path / "events.db")
    class CommitThenLoseReply:
        def claim(self, data):
            delegate.claim(data)
            raise StorageError("storage_failed")
        def finish(self, user_id, session_id, event_id, *, state):
            return delegate.finish(user_id, session_id, event_id, state=state)
    model = Model()
    result = setup(CommitThenLoseReply(), event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.code == "storage_failed" and result.event_saved is None and model.calls == 0
    assert delegate.get(event["user_id"], event["session_id"], event["event_id"]).decision_state == "reserved"
    retry_model = Model()
    retry = setup(delegate, event).decide(event["session_id"], event, retry_model, do_not_disturb=False)
    assert retry.code == "duplicate_event" and retry_model.calls == 0


def test_safety_during_finish_discards_ordinary_result(tmp_path, event):
    entered, release = Event(), Event()
    delegate = SQLiteEventStore(tmp_path / "events.db")
    class BlockingFinish:
        def claim(self, data): return delegate.claim(data)
        def finish(self, user_id, session_id, event_id, *, state):
            entered.set()
            if not release.wait(5):
                raise StorageError("storage_failed")
            return delegate.finish(user_id, session_id, event_id, state=state)
    coordinator = setup(BlockingFinish(), event)
    model, results = Model(), []
    worker = Thread(target=lambda: results.append(coordinator.decide(event["session_id"], event, model, do_not_disturb=False)), daemon=True)
    worker.start()
    try:
        assert entered.wait(3)
        safety_results = []
        safety = Thread(target=lambda: safety_results.append(coordinator.apply(event["session_id"], "report_discomfort", now=TIME, clock_id=CLOCK)), daemon=True)
        safety.start()
        safety.join(2)
        assert not safety.is_alive() and safety_results[0].status == "safety_pending"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and model.calls == 1
    assert results[0].status == "safety_pending" and results[0].candidate_discarded
    assert results[0].candidate is None and results[0].event_saved is True


def test_simultaneous_duplicate_only_one_model_call(tmp_path, event):
    entered, release = Event(), Event()
    store = SQLiteEventStore(tmp_path / "events.db")
    coordinator = setup(store, event)
    class BlockingModel(Model):
        def complete(self, facts):
            entered.set()
            if not release.wait(5):
                raise AssertionError("model wait timeout")
            return super().complete(facts)
    first_model, second_model, results = BlockingModel(), Model(), []
    worker = Thread(target=lambda: results.append(coordinator.decide(event["session_id"], event, first_model, do_not_disturb=False)), daemon=True)
    worker.start()
    try:
        assert entered.wait(3)
        duplicate = coordinator.decide(event["session_id"], event, second_model, do_not_disturb=False)
        assert duplicate.code == "duplicate_event" and second_model.calls == 0
        assert duplicate.stored_event.decision_state == "reserved"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and first_model.calls == 1 and results[0].status == "reviewed"


def test_repeat_across_two_real_processes_does_not_call_model_again(tmp_path, event):
    path = tmp_path / "process.db"
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json, sys
from metafit_agent.event_store import SQLiteEventStore
from metafit_agent.stored_coordinator import StoredSimulationCoordinator
from metafit_agent.model_response import ModelResponse
from metafit_agent.training_events import validate_training_event
event=json.loads(sys.stdin.read())
store=SQLiteEventStore(sys.argv[1])
c=StoredSimulationCoordinator(store)
c.register(user_id=event['user_id'],session_id=event['session_id'],clock_id='clock')
class Model:
    calls=0
    def complete(self,facts):
        self.calls+=1
        raw=json.dumps({'suggested_action':'notify','message':'ok','reason':'ok','evidence':['fact_1']})
        return ModelResponse(raw,'mock','mock',None,None,0,'stop')
model=Model()
result=c.decide(event['session_id'],event,model,do_not_disturb=False)
record=store.get(event['user_id'],event['session_id'],event['event_id'])
print(json.dumps({'status':result.status,'code':result.code,'calls':model.calls,
                  'valid':validate_training_event(record.event_dict()).event is not None,
                  'state':record.decision_state}))
'''
    outputs = []
    for _ in range(2):
        run = subprocess.run([sys.executable, "-c", script, str(path)], cwd=root,
                             input=json.dumps(event, ensure_ascii=True), capture_output=True,
                             text=True, encoding="utf-8", check=True, timeout=10)
        outputs.append(json.loads(run.stdout))
    assert outputs[0] == {"status": "reviewed", "code": None, "calls": 1, "valid": True, "state": "finished"}
    assert outputs[1] == {"status": "stopped", "code": "duplicate_event", "calls": 0, "valid": True, "state": "finished"}
