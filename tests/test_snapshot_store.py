import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from dataclasses import replace
from threading import Barrier, Event, Thread

import pytest

from metafit_agent.context_builder import build_context, fact_list_digest
from metafit_agent.coordinator import CoordinatedResult
from metafit_agent.decision import CandidateDecision
from metafit_agent.decision_store import DecisionSQLiteStore
from metafit_agent.event_store import SQLiteEventStore, StorageError
from metafit_agent.model_response import ModelResponse
from metafit_agent.decision import ModelCallError
from metafit_agent.recorded_coordinator import RecordedSimulationCoordinator
from metafit_agent.snapshot_coordinator import SnapshotSimulationCoordinator
from metafit_agent.snapshot_store import SnapshotCapture, SnapshotSQLiteStore
from metafit_agent.training_events import validate_training_event


@pytest.fixture
def event():
    data = json.loads((Path(__file__).resolve().parents[1] / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    data["discomfort"] = {"status": "none", "source": "self_report"}
    return data


class Model:
    def __init__(self, evidence=("fact_6",), action="notify"):
        self.evidence, self.action, self.calls = evidence, action, 0

    def complete(self, facts):
        self.calls += 1
        message = None if self.action == "silent" else "模拟反馈。"
        payload = json.dumps({"suggested_action": self.action, "message": message,
                              "reason": "模拟原因。", "evidence": self.evidence}, ensure_ascii=False)
        return ModelResponse(payload, "mock", "mock", None, None, 0, "stop")


def setup(store, event, cls=SnapshotSimulationCoordinator):
    c = cls(store)
    assert c.register(user_id=event["user_id"], session_id=event["session_id"], clock_id="clock").status == "registered"
    return c


def test_snapshot_is_actual_cited_fact_and_private_by_default(tmp_path, event):
    path = tmp_path / "history.db"
    store = SnapshotSQLiteStore(path)
    model = Model(("fact_6", "fact_1", "fact_3"))
    result = setup(store, event).decide(event["session_id"], event, model, do_not_disturb=False)
    assert result.status == "reviewed" and result.has_snapshot is True and model.calls == 1
    row = store.query(event["user_id"])[0]
    assert row["snapshot_state"] == "recorded" and row["cited_fact_count"] == 3
    assert row["visible_fact_count"] == len(build_context(validate_training_event(event).event).facts)
    assert row["fact_list_digest"] == build_context(validate_training_event(event).event).fact_list_digest
    assert [f["fact_id"] for f in row["facts"]] == ["fact_6", "fact_1", "fact_3"]
    assert all("value" not in fact for fact in row["facts"]) and "message" not in row
    values = store.query(event["user_id"], include_values=True)[0]
    assert "message" not in values
    assert values["facts"][0]["name"] == "weight_kg_per_hand" and values["facts"][0]["value"] == 8
    assert type(values["facts"][0]["value"]) is int
    assert type(values["facts"][1]["value"]) is str and values["facts"][1]["value"] == "dumbbell_fly"
    assert store.count_fact_references(event["user_id"], "weight_kg_per_hand") == 1
    with sqlite3.connect(path) as conn:
        snapshots = [row[0] for row in conn.execute("SELECT snapshot_json FROM decision_facts ORDER BY ordinal")]
    for raw in snapshots:
        assert all(key not in raw for key in ('"user_id"', '"session_id"', '"event_id"'))
        assert all(event[key] not in raw for key in ("user_id", "session_id", "event_id"))
    duplicate_model = Model()
    duplicate = setup(SnapshotSQLiteStore(path), event).decide(event["session_id"], event, duplicate_model, do_not_disturb=False)
    assert duplicate.code == "duplicate_event" and duplicate.has_snapshot and duplicate_model.calls == 0
    script = "from metafit_agent.snapshot_store import SnapshotSQLiteStore; import sys,json; print(json.dumps(SnapshotSQLiteStore(sys.argv[1]).query(sys.argv[2],include_values=True)))"
    run = subprocess.run([sys.executable, "-c", script, str(path), event["user_id"]], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(run.stdout)[0]["facts"][0]["value"] == 8


def test_no_references_and_rejected_are_distinct(tmp_path, event):
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    silent = setup(store, event).decide(event["session_id"], event, Model((), "silent"), do_not_disturb=False)
    rejected_event = dict(event, event_id="reject")
    reject = setup(store, rejected_event).decide(event["session_id"], rejected_event, Model(("fact_999",)), do_not_disturb=False)
    assert silent.status == "reviewed" and silent.has_snapshot is False
    assert reject.status == "stopped" and reject.code == "evidence_unknown_id"
    by_id = {row["event_id"]: row for row in store.query(event["user_id"])}
    assert by_id[event["event_id"]]["snapshot_state"] == "no_references"
    assert by_id[event["event_id"]]["cited_fact_count"] == 0
    assert by_id["reject"]["snapshot_state"] == "check_failed" and by_id["reject"]["facts"] == []
    assert by_id["reject"]["cited_fact_count"] == 1


def test_json_preserves_numeric_and_string_eight_in_synthetic_capture(tmp_path, event):
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    store.claim(event)
    built = build_context(validate_training_event(event).event)
    # 存储层类型测试使用明确标识的合成事实，不代表模型真实输入或产品字段。
    facts = (replace(built.facts[0], value=8), replace(built.facts[1], value="8"))
    evidence = (facts[0].fact_id, facts[1].fact_id)
    capture = SnapshotCapture(facts, fact_list_digest(built.scope, facts), evidence,
                              "reviewed", None, built.scope, True)
    candidate = CandidateDecision("notify", "模拟。", "模拟原因。", evidence)
    result = CoordinatedResult("reviewed", candidate=candidate)
    store.finish_with_snapshot(event["user_id"], event["session_id"], event["event_id"],
                               state="finished", result=result, capture=capture)
    values = store.query(event["user_id"], include_values=True)[0]["facts"]
    assert values[0]["value"] == 8 and type(values[0]["value"]) is int
    assert values[1]["value"] == "8" and type(values[1]["value"]) is str


def test_v2_old_history_remains_not_recorded_after_upgrade(tmp_path, event):
    path = tmp_path / "history.db"
    old = DecisionSQLiteStore(path)
    result = setup(old, event, RecordedSimulationCoordinator).decide(event["session_id"], event, Model(), do_not_disturb=False)
    assert result.status == "reviewed"
    new = SnapshotSQLiteStore(path)
    assert new.query(event["user_id"])[0]["snapshot_state"] == "not_recorded"
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert new.claim(event).status == "duplicate"  # 首次写入入口触发 v2 -> v3 升级
    row = new.query(event["user_id"])[0]
    assert row["snapshot_state"] == "not_recorded" and row["facts"] == [] and row["visible_fact_count"] is None
    assert SQLiteEventStore(path).get(event["user_id"], event["session_id"], event["event_id"]) is not None
    assert DecisionSQLiteStore(path).query(event["user_id"])[0]["status"] == "reviewed"


def test_v3_snapshot_remains_readable_and_digest_unknown_after_v4_upgrade(tmp_path, event):
    path = tmp_path / "v3.db"
    old = DecisionSQLiteStore(path)
    assert setup(old, event, RecordedSimulationCoordinator).decide(event["session_id"], event, Model(), do_not_disturb=False).status == "reviewed"
    built = build_context(validate_training_event(event).event)
    fact = next(f for f in built.facts if f.fact_id == "fact_6")
    snapshot = {"fact_id": fact.fact_id, "name": fact.name, "value": fact.value,
                "kind": fact.kind, "origin": fact.origin, "field": fact.field}
    key = (event["user_id"], event["session_id"], event["event_id"])
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE decision_fact_meta (user_id TEXT,session_id TEXT,event_id TEXT,snapshot_state TEXT,visible_fact_count INTEGER,cited_fact_count INTEGER,fact_list_digest TEXT,PRIMARY KEY(user_id,session_id,event_id))")
        conn.execute("CREATE TABLE decision_facts (user_id TEXT,session_id TEXT,event_id TEXT,ordinal INTEGER,snapshot_json TEXT,PRIMARY KEY(user_id,session_id,event_id,ordinal))")
        conn.execute("INSERT INTO decision_fact_meta VALUES (?,?,?,?,?,?,?)", (*key, "recorded", len(built.facts), 1, built.fact_list_digest))
        conn.execute("INSERT INTO decision_facts VALUES (?,?,?,?,?)", (*key, 0, json.dumps(snapshot)))
        conn.execute("PRAGMA user_version=3")
    store = SnapshotSQLiteStore(path)
    before = store.query(event["user_id"], include_values=True)[0]
    assert before["facts_digest"] is None and before["facts"][0]["value"] == 8
    assert store.claim(event).status == "duplicate"
    after = store.query(event["user_id"], include_values=True)[0]
    assert after["facts_digest"] is None and after["fact_list_digest"] == before["fact_list_digest"]
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4


def test_scope_snapshot_excludes_context_identifier_and_prior_ids(tmp_path, event):
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    prior = dict(event, event_id="secret_prior_event", session_id="secret_prior_session", timestamp="2026-11-01T18:30:00+08:00")
    context = {"source": "simulation", "context_id": "secret_context", "history": [prior]}
    # 读取实际事实编号，不依赖对编号的历史含义作假设。
    built = build_context(validate_training_event(event).event, context)
    history_id = next(f.fact_id for f in built.facts if f.name == "history_completed_reps")
    result = setup(store, event).decide(event["session_id"], event, Model((history_id,)), context=context, do_not_disturb=False)
    assert result.status == "reviewed"
    fact = store.query(event["user_id"], include_values=True)[0]["facts"][0]
    assert fact["name"] == "history_completed_reps" and fact["origin"] == "history"
    with sqlite3.connect(store.path) as conn:
        raw = conn.execute("SELECT snapshot_json FROM decision_facts").fetchone()[0]
    assert all(value not in raw for value in ("secret_prior_event", "secret_prior_session", "secret_context"))


def test_context_origin_removes_context_identifier(tmp_path, event):
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    context = {"source": "simulation", "context_id": "private_context_id", "goal": "general_fitness"}
    built = build_context(validate_training_event(event).event, context)
    goal_id = next(f.fact_id for f in built.facts if f.name == "goal")
    result = setup(store, event).decide(event["session_id"], event, Model((goal_id,)), context=context, do_not_disturb=False)
    assert result.status == "reviewed"
    assert store.query(event["user_id"])[0]["facts"][0]["origin"] == "simulated_context"
    with sqlite3.connect(store.path) as conn:
        assert "private_context_id" not in conn.execute("SELECT snapshot_json FROM decision_facts").fetchone()[0]


def test_failed_snapshot_write_rolls_back_decision_and_keeps_safety(tmp_path, event):
    class Broken(SnapshotSQLiteStore):
        def _insert_decision_extras(self, conn, key, result):
            raise StorageError("storage_failed")
    store = Broken(tmp_path / "history.db")
    result = setup(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    assert result.code == "storage_failed" and result.candidate is None
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM decisions").fetchone()[0] == 0
        assert conn.execute("SELECT decision_state FROM events").fetchone()[0] == "reserved"
    reported = dict(event, event_id="safety", discomfort={"status": "reported", "source": "self_report"})
    safe = setup(store, reported).decide(event["session_id"], reported, Model(), do_not_disturb=False)
    assert safe.status == "safety_pending" and safe.safety_message and "decision_state_update_failed" in safe.warnings


def test_defensive_subset_check_rolls_back(tmp_path, event):
    class BadCapture(SnapshotSimulationCoordinator):
        def _after_review(self, outcome, reviewed):
            super()._after_review(outcome, reviewed)
            capture = self._review_capture.get()
            self._review_capture.set(replace(capture, facts=()))
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    result = setup(store, event, BadCapture).decide(event["session_id"], event, Model(), do_not_disturb=False)
    assert result.code == "storage_failed" and result.candidate is None
    row = store.query(event["user_id"])[0]
    assert row["decision_state"] == "reserved" and row["snapshot_state"] == "not_recorded"


def test_changed_in_memory_fact_cannot_be_recorded_under_old_digest(tmp_path, event):
    class ChangedFact(SnapshotSimulationCoordinator):
        def _after_review(self, outcome, reviewed):
            super()._after_review(outcome, reviewed)
            capture = self._review_capture.get()
            changed = (replace(capture.facts[0], value="changed"), *capture.facts[1:])
            self._review_capture.set(replace(capture, facts=changed))
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    result = setup(store, event, ChangedFact).decide(event["session_id"], event, Model(), do_not_disturb=False)
    assert result.code == "storage_failed" and result.candidate is None
    assert store.query(event["user_id"])[0]["snapshot_state"] == "not_recorded"


def test_query_isolates_snapshot_reference_inconsistency(tmp_path, event):
    path = tmp_path / "history.db"
    store = SnapshotSQLiteStore(path)
    setup(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    with sqlite3.connect(path) as conn:
        raw = conn.execute("SELECT snapshot_json FROM decision_facts").fetchone()[0]
        fact = json.loads(raw)
        fact["fact_id"] = "fact_999"
        conn.execute("UPDATE decision_facts SET snapshot_json=?", (json.dumps(fact),))
    row = store.query(event["user_id"], include_values=True)[0]
    assert row["snapshot_state"] == "snapshot_invalid" and row["snapshot_warning"] == "snapshot_invalid"
    assert row["facts"] == [] and row["has_snapshot"] is None


def test_discomfort_during_snapshot_write_keeps_historical_snapshot(tmp_path, event):
    entered, release = Event(), Event()
    class Blocking(SnapshotSQLiteStore):
        def _insert_decision_extras(self, conn, key, result):
            entered.set()
            assert release.wait(5)
            return super()._insert_decision_extras(conn, key, result)
    store = Blocking(tmp_path / "history.db")
    c = setup(store, event)
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
    row = store.query(event["user_id"])[0]
    assert row["snapshot_state"] == "recorded" and row["not_delivered_reason"] == "safety_discarded"


def test_delete_event_and_user_cascade_snapshots(tmp_path, event):
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    setup(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    other = dict(event, event_id="other")
    setup(store, other).decide(other["session_id"], other, Model(), do_not_disturb=False)
    other_user = dict(event, user_id="other_user", event_id="other_user_event")
    setup(store, other_user).decide(other_user["session_id"], other_user, Model(), do_not_disturb=False)
    assert store.delete_event(event["user_id"], event["session_id"], event["event_id"])
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM decision_facts").fetchone()[0] == 2
    assert store.delete_user(event["user_id"])["events_deleted"] == 1
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM decision_fact_meta").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM decision_facts").fetchone()[0] == 1
    assert store.query(other_user["user_id"])[0]["snapshot_state"] == "recorded"


def test_parallel_decisions_capture_their_own_model_facts(tmp_path, event):
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    c = setup(store, event)
    barrier = Barrier(2, timeout=5)
    class ParallelModel(Model):
        def complete(self, facts):
            barrier.wait()
            return super().complete(facts)
    second = dict(event, event_id="parallel_second")
    models = (ParallelModel(("fact_1",)), ParallelModel(("fact_6",)))
    results = []
    workers = [Thread(target=lambda data=data, model=model: results.append(c.decide(data["session_id"], data, model, do_not_disturb=False)))
               for data, model in zip((event, second), models)]
    for worker in workers: worker.start()
    for worker in workers: worker.join(10)
    assert all(not worker.is_alive() for worker in workers)
    assert len(results) == 2 and all(result.status == "reviewed" for result in results)
    rows = {row["event_id"]: row for row in store.query(event["user_id"])}
    assert rows[event["event_id"]]["facts"][0]["fact_id"] == "fact_1"
    assert rows[second["event_id"]]["facts"][0]["fact_id"] == "fact_6"


def test_simulation_does_not_connect_to_network(tmp_path, event, monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: pytest.fail("network access"))
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    assert setup(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False).status == "reviewed"


def test_facts_digest_compares_model_input_without_event_identity(tmp_path, event):
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    c = setup(store, event)
    second = dict(event, event_id="different_event")
    third = dict(event, event_id="changed", completed_reps=9)
    for item in (event, second, third):
        assert c.decide(item["session_id"], item, Model(), do_not_disturb=False).status == "reviewed"
    rows = {row["event_id"]: row for row in store.query(event["user_id"])}
    a, b, changed = rows[event["event_id"]], rows[second["event_id"]], rows[third["event_id"]]
    assert a["fact_list_digest"] != b["fact_list_digest"]
    assert a["facts_digest"] == b["facts_digest"] != changed["facts_digest"]
    assert a["facts_digest_version"] == 1


def test_corrupt_snapshot_only_quarantines_one_row_and_hides_values(tmp_path, event):
    path = tmp_path / "history.db"
    store = SnapshotSQLiteStore(path)
    c = setup(store, event)
    good = dict(event, event_id="good")
    for item in (event, good):
        assert c.decide(item["session_id"], item, Model(), do_not_disturb=False).status == "reviewed"
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE decision_facts SET snapshot_json=? WHERE event_id=?", ('{"bad":', event["event_id"]))
    rows = {row["event_id"]: row for row in store.query(event["user_id"], include_values=True)}
    damaged = rows[event["event_id"]]
    assert damaged["snapshot_state"] == "snapshot_invalid" and damaged["no_snapshot_reason"] == "invalid"
    assert damaged["facts"] == [] and damaged["has_snapshot"] is None and damaged["facts_digest"] is None
    assert rows["good"]["facts"][0]["value"] == 8 and rows["good"]["snapshot_state"] == "recorded"


def test_invalid_snapshot_after_write_stops_candidate(tmp_path, event):
    class CorruptOnRead(SnapshotSQLiteStore):
        def _decorate_query(self, conn, rows, include_values, include_names):
            output = super()._decorate_query(conn, rows, include_values, include_names)
            for row in output: row["snapshot_state"] = "snapshot_invalid"
            return output
    store = CorruptOnRead(tmp_path / "history.db")
    result = setup(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False)
    assert result.status == "stopped" and result.code == "snapshot_invalid" and result.candidate is None
    assert result.stage == "storage" and "snapshot_invalid" in result.warnings


def test_invalid_snapshot_during_query_never_suppresses_new_safety(tmp_path, event):
    class CorruptOnRead(SnapshotSQLiteStore):
        def _decorate_query(self, conn, rows, include_values, include_names):
            output = super()._decorate_query(conn, rows, include_values, include_names)
            self.report_safety()
            for row in output: row["snapshot_state"] = "snapshot_invalid"
            return output
    store = CorruptOnRead(tmp_path / "history.db")
    c = setup(store, event)
    store.report_safety = lambda: c.apply(event["session_id"], "report_discomfort", now=None, clock_id="clock")
    result = c.decide(event["session_id"], event, Model(), do_not_disturb=False)
    assert result.status == "safety_pending" and result.candidate is None and result.safety_message
    assert "snapshot_invalid" in result.warnings


def test_hide_names_and_explain_missing_snapshot(tmp_path, event):
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    c = setup(store, event)
    assert c.decide(event["session_id"], event, Model(), do_not_disturb=False).status == "reviewed"
    row = store.query(event["user_id"], include_names=False)[0]
    assert all("name" not in fact and "field" not in fact and "value" not in fact for fact in row["facts"])
    assert row["snapshot_state"] == "recorded" and row["no_snapshot_reason"] is None
    pending = dict(event, event_id="pending")
    store.claim(pending)
    rejected = dict(event, event_id="reject")
    c.decide(rejected["session_id"], rejected, Model(("fact_999",)), do_not_disturb=False)
    class BrokenModel:
        def complete(self, facts): raise ModelCallError("test_error")
    failure = dict(event, event_id="failure")
    c.decide(failure["session_id"], failure, BrokenModel(), do_not_disturb=False)
    rows = {item["event_id"]: item for item in store.query(event["user_id"])}
    assert rows["pending"]["no_snapshot_reason"] == "unprocessed"
    assert rows["reject"]["no_snapshot_reason"] == "r3_rejected"
    assert rows["failure"]["no_snapshot_reason"] == "r2_error"
    safety = dict(event, event_id="safety", discomfort={"status": "reported", "source": "self_report"})
    setup(store, safety).decide(safety["session_id"], safety, Model(), do_not_disturb=False)
    assert store.query(event["user_id"], event_id="safety")[0]["no_snapshot_reason"] == "safety"


def test_count_by_kind_origin_and_query_batches(tmp_path, event, monkeypatch):
    import metafit_agent.decision_store as module
    store = SnapshotSQLiteStore(tmp_path / "history.db")
    c = setup(store, event)
    for i in range(25):
        item = dict(event, event_id=f"batch_{i}")
        assert c.decide(item["session_id"], item, Model(), do_not_disturb=False).status == "reviewed"
    assert store.count_fact_references(event["user_id"], kind="current_event", origin="event") == 25
    assert store.count_fact_references(event["user_id"], kind="training_history") == 0
    statements = []
    original = module.sqlite3.connect
    def traced(*args, **kwargs):
        conn = original(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn
    monkeypatch.setattr(module.sqlite3, "connect", traced)
    assert len(store.query(event["user_id"])) == 25
    snapshot_selects = [s for s in statements if s.startswith("SELECT ") and (" FROM decision_fact_meta WHERE " in s or " FROM decision_facts WHERE " in s)]
    assert len(snapshot_selects) == 2


def test_v3_to_v4_migration_failure_is_transactional(tmp_path, event):
    path = tmp_path / "migration.db"
    store = SnapshotSQLiteStore(path)
    assert setup(store, event).decide(event["session_id"], event, Model(), do_not_disturb=False).status == "reviewed"
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE decision_fact_meta DROP COLUMN facts_digest_version")
        conn.execute("ALTER TABLE decision_fact_meta DROP COLUMN facts_digest")
        conn.execute("PRAGMA user_version=3")
    class Proxy:
        def __init__(self, conn): self.conn = conn
        def execute(self, sql, *args):
            if sql.startswith("ALTER TABLE decision_fact_meta ADD COLUMN facts_digest_version"):
                raise sqlite3.OperationalError("injected")
            return self.conn.execute(sql, *args)
        def __getattr__(self, name): return getattr(self.conn, name)
    class BrokenUpgrade(SnapshotSQLiteStore):
        def _connect(self): return Proxy(super()._connect())
    with pytest.raises(StorageError, match="storage_failed"):
        BrokenUpgrade(path).claim(event)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        names = [row[1] for row in conn.execute("PRAGMA table_info(decision_fact_meta)")]
        assert "facts_digest" not in names and "facts_digest_version" not in names
    assert SnapshotSQLiteStore(path).query(event["user_id"])[0]["snapshot_state"] == "recorded"


def test_v2_upgrade_failure_rolls_back_without_overwriting_old_records(tmp_path, event):
    path = tmp_path / "old.db"
    old = DecisionSQLiteStore(path)
    setup(old, event, RecordedSimulationCoordinator).decide(event["session_id"], event, Model(), do_not_disturb=False)
    class ConnectionProxy:
        def __init__(self, conn): self.conn = conn
        def execute(self, sql, *args):
            if sql.startswith("CREATE TABLE decision_facts"):
                raise sqlite3.OperationalError("injected migration failure")
            return self.conn.execute(sql, *args)
        def __getattr__(self, name): return getattr(self.conn, name)
    class BrokenMigration(SnapshotSQLiteStore):
        def _connect(self):
            return ConnectionProxy(super()._connect())
    with pytest.raises(StorageError, match="storage_failed"):
        BrokenMigration(path).claim(event)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM decisions").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='decision_fact_meta'").fetchone()[0] == 0
    assert DecisionSQLiteStore(path).query(event["user_id"])[0]["status"] == "reviewed"
