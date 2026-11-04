"""R5 的 SQLite 模拟事件持久化契约，使用文件数据库。"""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from metafit_agent.event_store import SQLiteEventStore, StorageError
from metafit_agent.training_events import validate_training_event


@pytest.fixture
def event():
    path = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_saved_event_survives_new_connection_and_revalidates(tmp_path, event):
    path = tmp_path / "events.db"
    first = SQLiteEventStore(path)
    claimed = first.claim(event)
    assert claimed.status == "inserted" and claimed.record.decision_state == "reserved"
    reopened = SQLiteEventStore(path)
    loaded = reopened.get(event["user_id"], event["session_id"], event["event_id"])
    assert loaded.event_dict() == validate_training_event(event).event.to_dict()
    assert validate_training_event(loaded.event_dict()).event is not None


def test_v1_canonical_example_stays_compatible(tmp_path, event):
    # 固定样例的指纹是存储兼容性哨兵；改动需先设计旧库读取或迁移。
    record = SQLiteEventStore(tmp_path / "events.db").claim(event).record
    assert record.content_hash == "00e52824a45e805fe1241782dea1ba1ac1384a78d64b34811dd9b0451155c649"


def test_duplicate_normalizes_key_order_and_conflict_preserves_original(tmp_path, event):
    store = SQLiteEventStore(tmp_path / "events.db")
    first = store.claim(event)
    reordered = dict(reversed(list(event.items())))
    duplicate = store.claim(reordered)
    assert duplicate.status == "duplicate" and duplicate.record.content_hash == first.record.content_hash
    changed = deepcopy(event)
    changed["completed_reps"] = 99
    assert store.claim(changed).status == "conflict"
    assert store.get(event["user_id"], event["session_id"], event["event_id"]).event_dict()["completed_reps"] == 10


def test_null_and_missing_are_distinct_events_under_same_identity(tmp_path, event):
    store = SQLiteEventStore(tmp_path / "events.db")
    event.pop("perceived_effort")
    assert store.claim(event).status == "inserted"
    changed = deepcopy(event)
    changed["perceived_effort"] = None
    assert store.claim(changed).status == "conflict"


def test_cross_session_same_event_id_has_warning_for_same_simulated_user(tmp_path, event):
    store = SQLiteEventStore(tmp_path / "events.db")
    assert store.claim(event).status == "inserted"
    second = deepcopy(event)
    second["session_id"] = "another_session"
    result = store.claim(second)
    assert result.status == "inserted" and "event_id_reused_across_sessions" in result.warnings


def test_finish_distinguishes_completed_attempt_from_reserved(tmp_path, event):
    store = SQLiteEventStore(tmp_path / "events.db")
    assert store.claim(event).record.decision_state == "reserved"
    completed = store.finish(event["user_id"], event["session_id"], event["event_id"], state="finished")
    assert completed.decision_state == "finished"
    assert SQLiteEventStore(tmp_path / "events.db").claim(event).record.decision_state == "finished"


def test_file_database_required():
    with pytest.raises(StorageError, match="file_database_required"):
        SQLiteEventStore(":memory:")


def test_corrupt_file_and_unwritable_directory_return_controlled_errors(tmp_path, event):
    corrupted = tmp_path / "corrupt.db"
    corrupted.write_bytes(b"not sqlite")
    with pytest.raises(StorageError, match="storage_failed"):
        SQLiteEventStore(corrupted).claim(event)
    not_directory = tmp_path / "file"
    not_directory.write_text("x", encoding="utf-8")
    with pytest.raises(StorageError, match="storage_failed"):
        SQLiteEventStore(not_directory / "events.db").claim(event)


def test_unknown_event_and_store_schema_versions_are_rejected(tmp_path, event):
    event_path = tmp_path / "event.db"
    store = SQLiteEventStore(event_path)
    store.claim(event)
    with sqlite3.connect(event_path) as conn:
        conn.execute("UPDATE events SET schema_version='999'")
    with pytest.raises(StorageError, match="unsupported_schema_version"):
        store.get(event["user_id"], event["session_id"], event["event_id"])
    store_path = tmp_path / "store.db"
    SQLiteEventStore(store_path).claim(event)
    with sqlite3.connect(store_path) as conn:
        conn.execute("PRAGMA user_version=999")
    with pytest.raises(StorageError, match="unsupported_store_version"):
        SQLiteEventStore(store_path).get(event["user_id"], event["session_id"], event["event_id"])


def test_unrelated_version_one_database_is_not_modified(tmp_path, event):
    path = tmp_path / "unrelated.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE unrelated (value TEXT)")
        conn.execute("PRAGMA user_version=1")
    with pytest.raises(StorageError, match="unsupported_store_version"):
        SQLiteEventStore(path).claim(event)
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"unrelated"}


def test_database_with_only_user_view_is_not_treated_as_empty(tmp_path, event):
    path = tmp_path / "view_only.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE VIEW unrelated AS SELECT 1 AS value")
    with pytest.raises(StorageError, match="unsupported_store_version"):
        SQLiteEventStore(path).claim(event)
    with sqlite3.connect(path) as conn:
        objects = {(row[0], row[1]) for row in conn.execute("SELECT type,name FROM sqlite_master WHERE substr(name,1,7)<>'sqlite_'")}
        assert objects == {("view", "unrelated")}


def test_unexpected_schema_change_is_rejected_by_existing_store(tmp_path, event):
    path = tmp_path / "events.db"
    store = SQLiteEventStore(path)
    store.claim(event)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE VIEW unexpected AS SELECT 1 AS value")
    with pytest.raises(StorageError, match="unsupported_store_version"):
        store.get(event["user_id"], event["session_id"], event["event_id"])
