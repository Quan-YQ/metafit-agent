"""R5 模拟事件 SQLite 文件存储；不负责鉴权、加密或真实安全状态。"""
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Protocol

from .training_events import validate_training_event

_APPLICATION_ID = 0x4D465431
_V1_OBJECTS = {("table", "events")}
_V2_OBJECTS = _V1_OBJECTS | {("table", "decisions"), ("table", "attempts")}
_V3_OBJECTS = _V2_OBJECTS | {("table", "decision_fact_meta"), ("table", "decision_facts")}
_V4_OBJECTS = _V3_OBJECTS
_V5_OBJECTS = _V4_OBJECTS | {("table", "safety_sessions"), ("table", "safety_users"), ("table", "safety_log")}


class StorageError(Exception):
    """只暴露受控错误码，不回显路径、事件正文或 SQLite 原始错误。"""
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class StoredEvent:
    user_id: str
    session_id: str
    event_id: str
    schema_version: str
    event_json: str = field(repr=False)
    content_hash: str
    stored_at: str
    decision_state: str

    def event_dict(self) -> dict:
        return json.loads(self.event_json)


@dataclass(frozen=True, slots=True)
class ClaimResult:
    status: str
    record: StoredEvent | None
    warnings: tuple[str, ...] = ()


class EventStore(Protocol):
    def claim(self, event_data: Any) -> ClaimResult: ...
    def finish(self, user_id: str, session_id: str, event_id: str, *, state: str) -> StoredEvent: ...


def _canonical(event: Any) -> tuple[str, str]:
    try:
        payload = json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        raise StorageError("event_encoding_error") from None
    return payload, digest


class SQLiteEventStore:
    """每次操作使用独立连接；重复数据比较规范化内容并原子预占决策机会。"""

    def __init__(self, path: str | Path):
        raw = str(path)
        if raw == ":memory:" or raw.startswith("file:"):
            raise StorageError("file_database_required")
        self.path = Path(path)
        self._init_lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as conn:
                version, app_id, objects = self._identity(conn)
                new = version == 0 and app_id == 0 and not objects
                existing = app_id == _APPLICATION_ID and ((version == 1 and objects == _V1_OBJECTS) or (version == 2 and objects == _V2_OBJECTS) or (version == 3 and objects == _V3_OBJECTS) or (version == 4 and objects == _V4_OBJECTS) or (version == 5 and objects == _V5_OBJECTS))
                if not (new or existing):
                    raise StorageError("unsupported_store_version")
                if conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
                    raise StorageError("storage_failed")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("BEGIN IMMEDIATE")
                version, app_id, objects = self._identity(conn)
                new = version == 0 and app_id == 0 and not objects
                existing = app_id == _APPLICATION_ID and ((version == 1 and objects == _V1_OBJECTS) or (version == 2 and objects == _V2_OBJECTS) or (version == 3 and objects == _V3_OBJECTS) or (version == 4 and objects == _V4_OBJECTS) or (version == 5 and objects == _V5_OBJECTS))
                if not (new or existing):
                    raise StorageError("unsupported_store_version")
                conn.execute("""CREATE TABLE IF NOT EXISTS events (
                    user_id TEXT NOT NULL, session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL, event_json TEXT NOT NULL, content_hash TEXT NOT NULL,
                    stored_at TEXT NOT NULL, decision_state TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id, event_id)
                )""")
                conn.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                if new:
                    conn.execute("PRAGMA user_version=1")
                conn.commit()
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=3.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=3000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @staticmethod
    def _identity(conn: sqlite3.Connection) -> tuple[int, int, set[tuple[str, str]]]:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        app_id = conn.execute("PRAGMA application_id").fetchone()[0]
        rows = conn.execute("SELECT type,name FROM sqlite_master WHERE substr(name,1,7)<>'sqlite_'").fetchall()
        return version, app_id, {(row[0], row[1]) for row in rows}

    @classmethod
    def _verify_identity(cls, conn: sqlite3.Connection) -> None:
        if cls._identity(conn) not in ((1, _APPLICATION_ID, _V1_OBJECTS), (2, _APPLICATION_ID, _V2_OBJECTS), (3, _APPLICATION_ID, _V3_OBJECTS), (4, _APPLICATION_ID, _V4_OBJECTS), (5, _APPLICATION_ID, _V5_OBJECTS)):
            raise StorageError("unsupported_store_version")

    @staticmethod
    def _record(row: tuple | None) -> StoredEvent | None:
        if row is None:
            return None
        record = StoredEvent(*row)
        if record.schema_version != "0.1":
            raise StorageError("unsupported_schema_version")
        try:
            data = record.event_dict()
            validated = validate_training_event(data)
            if validated.event is None:
                raise StorageError("stored_event_invalid")
            canonical, digest = _canonical(validated.event)
            if (canonical != record.event_json or digest != record.content_hash or
                (validated.event.user_id, validated.event.session_id, validated.event.event_id) !=
                (record.user_id, record.session_id, record.event_id) or
                record.decision_state not in {"reserved", "finished", "safety_only"}):
                raise StorageError("stored_event_invalid")
        except (json.JSONDecodeError, TypeError, ValueError, UnicodeError, RecursionError):
            raise StorageError("stored_event_invalid") from None
        return record

    @staticmethod
    def _select(conn: sqlite3.Connection, user_id: str, session_id: str, event_id: str) -> tuple | None:
        return conn.execute("""SELECT user_id, session_id, event_id, schema_version, event_json,
                            content_hash, stored_at, decision_state FROM events
                            WHERE user_id=? AND session_id=? AND event_id=?""", (user_id, session_id, event_id)).fetchone()

    def get(self, user_id: str, session_id: str, event_id: str) -> StoredEvent | None:
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                self._verify_identity(conn)
                return self._record(self._select(conn, user_id, session_id, event_id))
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def claim(self, event_data: Any) -> ClaimResult:
        validated = validate_training_event(event_data)
        if validated.event is None:
            raise StorageError("invalid_event")
        event = validated.event
        payload, digest = _canonical(event)
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                previous = self._record(self._select(conn, event.user_id, event.session_id, event.event_id))
                reused = conn.execute("SELECT 1 FROM events WHERE user_id=? AND event_id=? AND session_id<>? LIMIT 1",
                                      (event.user_id, event.event_id, event.session_id)).fetchone() is not None
                warnings = ("event_id_reused_across_sessions",) if reused else ()
                if previous is not None:
                    if previous.event_json != payload or previous.content_hash != digest:
                        conn.commit()
                        return ClaimResult("conflict", None, warnings)
                    conn.commit()
                    return ClaimResult("duplicate", previous, warnings)
                stored_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
                conn.execute("""INSERT INTO events (user_id, session_id, event_id, schema_version, event_json,
                             content_hash, stored_at, decision_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                             (event.user_id, event.session_id, event.event_id, event.schema_version,
                              payload, digest, stored_at, "reserved"))
                record = self._record(self._select(conn, event.user_id, event.session_id, event.event_id))
                conn.commit()
                return ClaimResult("inserted", record, warnings)
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def finish(self, user_id: str, session_id: str, event_id: str, *, state: str) -> StoredEvent:
        if state not in {"finished", "safety_only"}:
            raise StorageError("invalid_decision_state")
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                current = self._record(self._select(conn, user_id, session_id, event_id))
                if current is None or current.decision_state != "reserved":
                    raise StorageError("invalid_decision_state")
                conn.execute("UPDATE events SET decision_state=? WHERE user_id=? AND session_id=? AND event_id=?",
                             (state, user_id, session_id, event_id))
                result = self._record(self._select(conn, user_id, session_id, event_id))
                conn.commit()
                return result
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None
