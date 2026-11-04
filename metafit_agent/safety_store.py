"""单数据库模拟安全状态；不是身份验证、医疗判断或可靠跨设备存储。"""
from contextlib import closing
from dataclasses import dataclass
import sqlite3

from .decision_store import _now
from .event_store import StorageError, _APPLICATION_ID, _V4_OBJECTS, _V5_OBJECTS
from .interaction_state import _timestamp
from .snapshot_store import SnapshotSQLiteStore


@dataclass(frozen=True, slots=True)
class SafetyRecord:
    state: str
    user_generation: int
    reported_at: str | None = None
    recorded_at: str | None = None
    confirmed_at: str | None = None
    warning: str | None = None
    confirmation_closed: bool = False


class SafetySQLiteStore(SnapshotSQLiteStore):
    """单实例模拟；append-only 仅约束本 API，删除用户时日志也删除。"""

    def _initialize(self):
        super()._initialize()
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                identity = self._identity(conn)
                if identity == (4, _APPLICATION_ID, _V4_OBJECTS):
                    conn.execute("""CREATE TABLE safety_sessions (
                        user_id TEXT NOT NULL, session_id TEXT NOT NULL UNIQUE,
                        state TEXT NOT NULL, reported_at TEXT, recorded_at TEXT, confirmed_at TEXT,
                        PRIMARY KEY(user_id,session_id))""")
                    conn.execute("""CREATE TABLE safety_users (
                        user_id TEXT PRIMARY KEY, report_generation INTEGER NOT NULL,
                        last_discomfort_at TEXT, last_recorded_at TEXT NOT NULL)""")
                    conn.execute("""CREATE TABLE safety_log (
                        user_id TEXT NOT NULL, session_id TEXT NOT NULL, action TEXT NOT NULL,
                        generation INTEGER NOT NULL, recorded_at TEXT NOT NULL)""")
                    conn.execute("PRAGMA user_version=5")
                elif identity != (5, _APPLICATION_ID, _V5_OBJECTS):
                    raise StorageError("unsupported_store_version")
                conn.commit()
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    @staticmethod
    def _generation(conn, user_id):
        row = conn.execute("SELECT report_generation FROM safety_users WHERE user_id=?", (user_id,)).fetchone()
        if row is None:
            return 0
        if type(row[0]) is not int or row[0] < 1:
            raise StorageError("safety_state_invalid")
        return row[0]

    @staticmethod
    def _log(conn, user_id, session_id, action, generation):
        conn.execute("INSERT INTO safety_log VALUES (?,?,?,?,?)", (user_id, session_id, action, generation, _now()))

    def open_session(self, user_id, session_id, *, new_session=False):
        """缺失和未报告会话均待确认；新会话标识不能证明用户没有不适。"""
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                row = conn.execute("SELECT user_id,state,reported_at,recorded_at,confirmed_at FROM safety_sessions WHERE session_id=?", (session_id,)).fetchone()
                if row is not None and row[0] != user_id:
                    raise StorageError("scope_mismatch")
                owners = conn.execute("SELECT DISTINCT user_id FROM events WHERE session_id=?", (session_id,)).fetchall()
                if any(owner[0] != user_id for owner in owners):
                    raise StorageError("scope_mismatch")
                try:
                    gen = self._generation(conn, user_id)
                except StorageError:
                    if row is not None and row[1] == "reported":
                        return SafetyRecord("reported", 0, row[2] if _timestamp(row[2]) else None, row[3], warning="safety_state_invalid")
                    raise
                if row is None:
                    state = "pending"
                    conn.execute("INSERT INTO safety_sessions VALUES (?,?,?,?,?,?)", (user_id, session_id, state, None, None, None))
                    self._log(conn, user_id, session_id, "create" if new_session else "restore", gen)
                    conn.commit()
                    return SafetyRecord(state, gen)
                _, state, reported_at, recorded_at, confirmed_at = row
                if state not in {"normal", "pending", "pending_closed", "reported", "invalidated"}:
                    raise StorageError("safety_state_invalid")
                if state == "reported" and gen == 0:
                    return SafetyRecord("reported", 0, reported_at if _timestamp(reported_at) else None, recorded_at,
                                        warning="safety_state_invalid")
                if state in {"normal", "pending", "pending_closed"} and (reported_at is not None or recorded_at is not None):
                    raise StorageError("safety_state_invalid")
                warning = None
                if reported_at is not None and _timestamp(reported_at) is None:
                    reported_at, warning = None, "time_unvalidated"
                if state == "reported" and (recorded_at is None or _timestamp(recorded_at) is None):
                    warning = "safety_state_invalid"
                if state in {"normal", "pending", "pending_closed"} and confirmed_at is not None and _timestamp(confirmed_at) is None:
                    raise StorageError("safety_state_invalid")
                if state == "normal":
                    state = "pending"  # 每次重新打开都需明确确认，不把旧记录当实时状态。
                    conn.execute("UPDATE safety_sessions SET state='pending' WHERE user_id=? AND session_id=?", (user_id, session_id))
                self._log(conn, user_id, session_id, "restore", gen)
                conn.commit()
                return SafetyRecord("pending" if state == "pending_closed" else state, gen, reported_at, recorded_at,
                                    confirmed_at, warning, state == "pending_closed")
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def report(self, user_id, session_id, reported_at):
        """首次报告单向写入；重复报告不覆盖首次时间。"""
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                row = conn.execute("SELECT user_id,state,reported_at,recorded_at FROM safety_sessions WHERE session_id=?", (session_id,)).fetchone()
                if row is not None and row[0] != user_id:
                    raise StorageError("scope_mismatch")
                if row is not None and row[1] not in {"normal", "pending", "pending_closed", "reported", "invalidated"}:
                    raise StorageError("safety_state_invalid")
                if row is not None and row[1] == "invalidated":
                    raise StorageError("session_invalidated")
                gen = self._generation(conn, user_id)
                if row is not None and row[1] == "reported":
                    if gen == 0 or row[3] is None or _timestamp(row[3]) is None:
                        raise StorageError("safety_state_invalid")
                    conn.commit()
                    warning = "time_unvalidated" if row[2] is not None and _timestamp(row[2]) is None else None
                    return SafetyRecord("reported", gen, row[2] if warning is None else None, row[3], warning=warning)
                now = _now()
                safe_at = reported_at if _timestamp(reported_at) is not None else None
                gen += 1
                conn.execute("""INSERT INTO safety_users VALUES (?,?,?,?)
                             ON CONFLICT(user_id) DO UPDATE SET report_generation=excluded.report_generation,
                             last_discomfort_at=excluded.last_discomfort_at,last_recorded_at=excluded.last_recorded_at""", (user_id, gen, safe_at, now))
                conn.execute("""INSERT INTO safety_sessions VALUES (?,?,?,?,?,NULL)
                             ON CONFLICT(user_id,session_id) DO UPDATE SET state='reported',reported_at=excluded.reported_at,
                             recorded_at=excluded.recorded_at,confirmed_at=NULL""", (user_id, session_id, "reported", safe_at, now))
                self._log(conn, user_id, session_id, "report", gen)
                conn.commit()
                return SafetyRecord("reported", gen, safe_at, now)
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def confirm(self, user_id, session_id, expected_generation):
        """仅允许明确回答 none，且不得覆盖已报告的不适。"""
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                row = conn.execute("SELECT state FROM safety_sessions WHERE user_id=? AND session_id=?", (user_id, session_id)).fetchone()
                if row is None or row[0] not in {"normal", "pending"}:
                    raise StorageError("safety_state_invalid")
                gen = self._generation(conn, user_id)
                if gen != expected_generation:
                    raise StorageError("safety_generation_changed")
                now = _now()
                conn.execute("UPDATE safety_sessions SET state='normal',confirmed_at=? WHERE user_id=? AND session_id=?", (now, user_id, session_id))
                self._log(conn, user_id, session_id, "confirm", gen)
                conn.commit()
                return SafetyRecord("normal", gen, confirmed_at=now)
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def close_confirmation(self, user_id, session_id):
        """跳过、过期或回答未知后关闭本会话的确认问题，不解锁普通反馈。"""
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                row = conn.execute("SELECT state FROM safety_sessions WHERE user_id=? AND session_id=?", (user_id, session_id)).fetchone()
                if row is None or row[0] not in {"pending", "pending_closed"}:
                    raise StorageError("safety_state_changed")
                gen = self._generation(conn, user_id)
                if row[0] == "pending":
                    conn.execute("UPDATE safety_sessions SET state='pending_closed' WHERE user_id=? AND session_id=?", (user_id, session_id))
                    self._log(conn, user_id, session_id, "confirmation_closed", gen)
                conn.commit()
                return SafetyRecord("pending", gen, confirmation_closed=True)
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def invalidate_session(self, user_id, session_id):
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                row = conn.execute("SELECT user_id,state FROM safety_sessions WHERE session_id=?", (session_id,)).fetchone()
                if row is not None and row[0] != user_id:
                    raise StorageError("scope_mismatch")
                if row is None:
                    conn.execute("INSERT INTO safety_sessions VALUES (?,?,?,?,?,?)", (user_id, session_id, "invalidated", None, None, None))
                    self._log(conn, user_id, session_id, "invalidate", self._generation(conn, user_id))
                elif row[1] != "invalidated":
                    conn.execute("UPDATE safety_sessions SET state='invalidated' WHERE user_id=? AND session_id=?", (user_id, session_id))
                    self._log(conn, user_id, session_id, "invalidate", self._generation(conn, user_id))
                conn.commit()
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def list_safety_log(self, user_id, *, session_id=None, limit=100):
        """只读列出模拟状态记录；日志不能证明真实性或完整性。"""
        if (not isinstance(user_id, str) or not user_id.strip() or
            (session_id is not None and (not isinstance(session_id, str) or not session_id.strip())) or
            type(limit) is not int or not 1 <= limit <= 500):
            raise ValueError("invalid log query")
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3.0)) as conn:
                if self._identity(conn) != (5, _APPLICATION_ID, _V5_OBJECTS):
                    raise StorageError("unsupported_store_version")
                conn.execute("PRAGMA query_only=ON")
                cond = "user_id=?" if session_id is None else "user_id=? AND session_id=?"
                params = (user_id, limit) if session_id is None else (user_id, session_id, limit)
                return [{"session_id": row[0], "action": row[1], "generation": row[2], "recorded_at": row[3]}
                        for row in conn.execute("SELECT session_id,action,generation,recorded_at FROM safety_log WHERE " + cond + " ORDER BY rowid DESC LIMIT ?", params)]
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

