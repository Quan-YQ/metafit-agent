"""R5 第二单元：仅用于模拟数据的本地历史，不构成投递或鉴权。"""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import sqlite3

from .event_store import SQLiteEventStore, StorageError, _APPLICATION_ID, _V1_OBJECTS, _V2_OBJECTS, _V3_OBJECTS, _V4_OBJECTS, _V5_OBJECTS


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("time must be ISO 8601 with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    except (ValueError, OverflowError):
        raise ValueError("time must be ISO 8601 with timezone") from None


def _key_hash(user_id, session_id, event_id):
    return hashlib.sha256(json.dumps([user_id, session_id, event_id], ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class DecisionSQLiteStore(SQLiteEventStore):
    """显式使用本类时才升级模拟库；v1 事件保持原样。"""

    def _connect(self):
        conn = super()._connect()
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _insert_decision_extras(self, conn, key, result):
        """供本地扩展在同一个事务写入附加记录。"""
        pass

    def _decorate_query(self, conn, rows, include_values, include_names):
        return rows

    def _delete_user_extras(self, conn, user_id):
        if self._identity(conn) == (5, _APPLICATION_ID, _V5_OBJECTS):
            logs = conn.execute("DELETE FROM safety_log WHERE user_id=?", (user_id,)).rowcount
            sessions = conn.execute("DELETE FROM safety_sessions WHERE user_id=?", (user_id,)).rowcount
            users = conn.execute("DELETE FROM safety_users WHERE user_id=?", (user_id,)).rowcount
            return {"safety_sessions_deleted": sessions, "safety_users_deleted": users, "safety_logs_deleted": logs}
        return {}

    def _initialize(self):
        super()._initialize()
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                identity = self._identity(conn)
                if identity == (1, _APPLICATION_ID, _V1_OBJECTS):
                    conn.execute("""CREATE TABLE decisions (
                        user_id TEXT NOT NULL, session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                        status TEXT NOT NULL, code TEXT, stage TEXT, effective_action TEXT,
                        rule_code TEXT, reference_check TEXT NOT NULL, evidence_json TEXT NOT NULL,
                        message TEXT, text_verification TEXT NOT NULL, decided_at TEXT NOT NULL,
                        not_delivered_reason TEXT NOT NULL,
                        PRIMARY KEY (user_id, session_id, event_id),
                        FOREIGN KEY (user_id, session_id, event_id) REFERENCES events (user_id, session_id, event_id)
                    )""")
                    conn.execute("""CREATE TABLE attempts (
                        key_hash TEXT NOT NULL, content_hash TEXT NOT NULL, attempted_at TEXT NOT NULL, failure_code TEXT NOT NULL
                    )""")
                    conn.execute("PRAGMA user_version=2")
                elif identity not in ((2, _APPLICATION_ID, _V2_OBJECTS), (3, _APPLICATION_ID, _V3_OBJECTS), (4, _APPLICATION_ID, _V4_OBJECTS), (5, _APPLICATION_ID, _V5_OBJECTS)):
                    raise StorageError("unsupported_store_version")
                conn.commit()
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def finish_with_decision(self, user_id, session_id, event_id, *, state, result):
        if state not in {"finished", "safety_only"}:
            raise StorageError("invalid_decision_state")
        approved = result is not None and result.status == "reviewed" and result.candidate is not None
        candidate = result.candidate if approved else None
        message = candidate.message if candidate else None
        evidence = candidate.evidence if candidate else ()
        status = result.status if result is not None else "safety_pending"
        code = result.code if result is not None else None
        stage = result.stage if result is not None else "coordinator"
        action = result.effective_action if approved else None
        rule = result.rule_code if result is not None else None
        ref = result.reference_check if result is not None else "not_checked"
        reason = ("safety_discarded" if status == "safety_pending" else
                  "session_invalidated" if code in ("session_invalidated", "state_missing") else
                  "pending_confirmation" if code == "pending_confirmation" else
                  ("rule_stopped" if stage == "r3" else "decision_stopped") if status == "stopped" else
                  "dnd_deferred" if rule == "dnd_notify_to_defer" else "safety_check_pending")
        key = (user_id, session_id, event_id)
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                current = self._record(self._select(conn, *key))
                if current is None or current.decision_state != "reserved":
                    raise StorageError("invalid_decision_state")
                conn.execute("""INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                             (*key, status, code, stage, action, rule, ref,
                              json.dumps(evidence, ensure_ascii=True), message,
                              "unverified" if message is not None else "not_applicable", _now(), reason))
                self._insert_decision_extras(conn, key, result)
                conn.execute("UPDATE events SET decision_state=? WHERE user_id=? AND session_id=? AND event_id=?",
                             (state, *key))
                if status == "stopped":
                    conn.execute("INSERT INTO attempts VALUES (?,?,?,?)", (_key_hash(*key), current.content_hash, _now(), "decision_stopped"))
                updated = self._record(self._select(conn, *key))
                conn.commit()
                return updated
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def update_not_delivered(self, user_id, session_id, event_id, reason):
        if reason not in {"safety_discarded", "rule_stopped", "decision_stopped", "dnd_deferred", "session_invalidated", "not_delivered", "pending_confirmation"}:
            raise ValueError("invalid reason")
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                cur = conn.execute("UPDATE decisions SET not_delivered_reason=? WHERE user_id=? AND session_id=? AND event_id=?",
                                   (reason, user_id, session_id, event_id))
                if cur.rowcount != 1:
                    raise StorageError("decision_missing")
                conn.commit()
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def record_failure(self, user_id, session_id, event_id, content_hash, failure_code):
        if not isinstance(content_hash, str) or len(content_hash) != 64 or any(c not in "0123456789abcdef" for c in content_hash):
            raise ValueError("invalid digest")
        if failure_code not in {"storage_failed", "event_conflict", "decision_stopped", "decision_update_failed"}:
            raise ValueError("invalid failure code")
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                conn.execute("INSERT INTO attempts VALUES (?,?,?,?)", (_key_hash(user_id, session_id, event_id), content_hash, _now(), failure_code))
                conn.commit()
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def query(self, user_id, *, session_id=None, event_id=None, start=None, end=None, limit=100, include_text=False, include_values=False, include_names=True):
        if not isinstance(user_id, str) or not user_id.strip() or (session_id is not None and (not isinstance(session_id, str) or not session_id.strip())):
            raise ValueError("user and session must be nonempty")
        if type(limit) is not int or not 1 <= limit <= 500 or type(include_text) is not bool or type(include_values) is not bool or type(include_names) is not bool:
            raise ValueError("invalid query options")
        if event_id is not None and (not isinstance(event_id, str) or not event_id.strip()):
            raise ValueError("event must be nonempty")
        start = _utc(start) if start is not None else None
        end = _utc(end) if end is not None else None
        if start and end and start >= end:
            raise ValueError("start must precede end")
        filters, args = ["e.user_id=?"], [user_id]
        if session_id is not None:
            filters.append("e.session_id=?")
            args.append(session_id)
        if event_id is not None:
            filters.append("e.event_id=?")
            args.append(event_id)
        # 以事件本机保存时间筛选，包含 reserved 和旧版没有记录的事件。
        if start:
            filters.append("e.stored_at>=?")
            args.append(start)
        if end:
            filters.append("e.stored_at<?")
            args.append(end)
        args.append(limit)
        text_column = ",d.message" if include_text else ""
        sql = """SELECT e.user_id,e.session_id,e.event_id,e.stored_at,e.decision_state,
                 d.status,d.code,d.stage,d.effective_action,d.rule_code,d.reference_check,
                 d.evidence_json,d.text_verification,d.decided_at,d.not_delivered_reason,
                 CASE WHEN d.message IS NULL THEN 0 ELSE 1 END AS has_text""" + text_column + """
                 FROM events e LEFT JOIN decisions d USING(user_id,session_id,event_id) WHERE """ + " AND ".join(filters) + " ORDER BY e.stored_at DESC,e.event_id DESC LIMIT ?"
        try:
            # 查询不创建文件、不自动升级旧库；打开失败交由调用方处理。
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3.0)) as conn:
                if self._identity(conn) not in ((2, _APPLICATION_ID, _V2_OBJECTS), (3, _APPLICATION_ID, _V3_OBJECTS), (4, _APPLICATION_ID, _V4_OBJECTS), (5, _APPLICATION_ID, _V5_OBJECTS)):
                    raise StorageError("unsupported_store_version")
                conn.execute("PRAGMA query_only=ON")
                conn.execute("BEGIN")
                conn.row_factory = sqlite3.Row
                rows = []
                for row in conn.execute(sql, args):
                    item = dict(row)
                    item["has_text"] = bool(item["has_text"])
                    legacy_pending = (item["status"] == "stopped" and item["stage"] == "coordinator" and
                                      item["code"] == "pending_confirmation" and item["not_delivered_reason"] == "decision_stopped")
                    if legacy_pending:
                        item["not_delivered_reason"] = "pending_confirmation"
                    item["not_delivered_reason_source"] = ("legacy_code" if legacy_pending else
                                                           "stored" if item["not_delivered_reason"] is not None else "not_available")
                    evidence_json = item.pop("evidence_json")
                    item["evidence"] = json.loads(evidence_json) if evidence_json else []
                    item["record_state"] = (
                        "inconsistent" if item["decision_state"] == "reserved" and item["status"] is not None else
                        "pending" if item["decision_state"] == "reserved" or item["not_delivered_reason"] == "safety_check_pending" else
                        "no_decision" if item["status"] is None else "decided"
                    )
                    # 仅表示本模块不允许自动投递，不能代替产品接口的访问控制。
                    item["delivery_eligibility"] = "not_eligible"
                    rows.append(item)
                return self._decorate_query(conn, rows, include_values, include_names)
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def query_attempts(self, user_id, session_id, event_id, *, limit=100):
        if any(not isinstance(x, str) or not x.strip() for x in (user_id, session_id, event_id)) or type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("invalid identity or limit")
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3.0)) as conn:
                if self._identity(conn) not in ((2, _APPLICATION_ID, _V2_OBJECTS), (3, _APPLICATION_ID, _V3_OBJECTS), (4, _APPLICATION_ID, _V4_OBJECTS), (5, _APPLICATION_ID, _V5_OBJECTS)):
                    raise StorageError("unsupported_store_version")
                conn.execute("PRAGMA query_only=ON")
                return [{"content_hash": row[0], "attempted_at": row[1], "failure_code": row[2]}
                        for row in conn.execute("SELECT content_hash, attempted_at, failure_code FROM attempts WHERE key_hash=? ORDER BY attempted_at DESC LIMIT ?",
                                                (_key_hash(user_id, session_id, event_id), limit))]
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def delete_event(self, user_id, session_id, event_id, *, allow_unresolved=False):
        """删除事件和关联记录；未完成状态只允许调用方显式指定。"""
        if type(allow_unresolved) is not bool:
            raise ValueError("allow_unresolved must be bool")
        if not self.path.exists():
            return False
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                key = (user_id, session_id, event_id)
                record = self._record(self._select(conn, *key))
                if record is None:
                    removed = conn.execute("DELETE FROM attempts WHERE key_hash=?", (_key_hash(*key),)).rowcount
                    conn.commit()
                    return bool(removed)
                if record.decision_state == "reserved" and not allow_unresolved:
                    raise StorageError("event_in_progress_or_unresolved")
                pending = conn.execute("SELECT not_delivered_reason FROM decisions WHERE user_id=? AND session_id=? AND event_id=?", key).fetchone()
                if pending is not None and pending[0] == "safety_check_pending" and not allow_unresolved:
                    raise StorageError("event_in_progress_or_unresolved")
                conn.execute("DELETE FROM decisions WHERE user_id=? AND session_id=? AND event_id=?", key)
                conn.execute("DELETE FROM attempts WHERE key_hash=?", (_key_hash(*key),))
                conn.execute("DELETE FROM events WHERE user_id=? AND session_id=? AND event_id=?", key)
                conn.commit()
                return True
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def delete_user(self, user_id, *, allow_unresolved=False):
        """删除可归属于该用户的模拟事件；无法归属的孤立尝试明确标记为未知。"""
        if not isinstance(user_id, str) or not user_id.strip() or type(allow_unresolved) is not bool:
            raise ValueError("invalid user or allow_unresolved")
        if not self.path.exists():
            return {"events_deleted": 0, "decisions_deleted": 0, "attempts_deleted": 0,
                    "unattributed_attempts_present": False, "complete_in_this_database": True}
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                # attempts 仅存组合键哈希；先建立当前全部事件的哈希集合，识别无法归属的尝试。
                conn.execute("CREATE TEMP TABLE known_attempt_keys (key_hash TEXT PRIMARY KEY)")
                all_keys = conn.execute("SELECT user_id, session_id, event_id FROM events").fetchall()
                conn.executemany("INSERT OR IGNORE INTO known_attempt_keys VALUES (?)",
                                 [(_key_hash(*key),) for key in all_keys])
                orphan = conn.execute("""SELECT EXISTS(SELECT 1 FROM attempts a WHERE NOT EXISTS
                                      (SELECT 1 FROM known_attempt_keys k WHERE k.key_hash=a.key_hash))""").fetchone()[0] == 1
                keys = [key for key in all_keys if key[0] == user_id]
                unresolved_count = 0
                for key in keys:
                    record = self._record(self._select(conn, *key))
                    pending = conn.execute("SELECT not_delivered_reason FROM decisions WHERE user_id=? AND session_id=? AND event_id=?", key).fetchone()
                    if record.decision_state == "reserved" or (pending is not None and pending[0] == "safety_check_pending"):
                        unresolved_count += 1
                if unresolved_count and not allow_unresolved:
                    error = StorageError("event_in_progress_or_unresolved")
                    error.unresolved_count = unresolved_count
                    raise error
                attempts_deleted = 0
                for key in keys:
                    attempts_deleted += conn.execute("DELETE FROM attempts WHERE key_hash=?", (_key_hash(*key),)).rowcount
                decisions_deleted = conn.execute("DELETE FROM decisions WHERE user_id=?", (user_id,)).rowcount
                events_deleted = conn.execute("DELETE FROM events WHERE user_id=?", (user_id,)).rowcount
                extra = self._delete_user_extras(conn, user_id)
                conn.commit()
                return {"events_deleted": events_deleted, "decisions_deleted": decisions_deleted,
                        "attempts_deleted": attempts_deleted, "unattributed_attempts_present": orphan,
                        "complete_in_this_database": not orphan, **extra}
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def inspect_user_deletion(self, user_id, *, include_ids=False, limit=100):
        """只读检查待删除的模拟事件；结果可能因并发写入而过时。"""
        if (not isinstance(user_id, str) or not user_id.strip() or type(include_ids) is not bool or
            type(limit) is not int or not 1 <= limit <= 500):
            raise ValueError("invalid user or options")
        if not self.path.exists():
            return {"event_count": 0, "unresolved_count": 0, "unresolved": [], "truncated": False}
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3.0)) as conn:
                version, app_id, objects = self._identity(conn)
                if (version, app_id, objects) not in ((1, _APPLICATION_ID, _V1_OBJECTS), (2, _APPLICATION_ID, _V2_OBJECTS),
                                                       (3, _APPLICATION_ID, _V3_OBJECTS), (4, _APPLICATION_ID, _V4_OBJECTS), (5, _APPLICATION_ID, _V5_OBJECTS)):
                    raise StorageError("unsupported_store_version")
                conn.execute("PRAGMA query_only=ON")
                conn.execute("BEGIN")
                join = "" if version == 1 else " LEFT JOIN decisions d USING(user_id,session_id,event_id)"
                pending = "e.decision_state='reserved'" if version == 1 else "(e.decision_state='reserved' OR d.not_delivered_reason='safety_check_pending')"
                total, count = conn.execute("SELECT count(*),coalesce(sum(CASE WHEN " + pending + " THEN 1 ELSE 0 END),0) FROM events e" + join + " WHERE e.user_id=?", (user_id,)).fetchone()
                unresolved = []
                if include_ids:
                    unresolved = [{"session_id": row[0], "event_id": row[1]} for row in conn.execute(
                        "SELECT e.session_id,e.event_id FROM events e" + join + " WHERE e.user_id=? AND " + pending + " ORDER BY e.session_id,e.event_id LIMIT ?", (user_id, limit))]
                return {"event_count": total, "unresolved_count": count, "unresolved": unresolved,
                        "truncated": include_ids and count > limit}
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def prune_attempts(self, *, before):
        """按本机记录时间删除全库较早的模拟失败尝试，不影响事件或决策。"""
        cutoff = _utc(before)
        if not self.path.exists():
            return 0
        try:
            self._ensure_initialized()
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._verify_identity(conn)
                removed = conn.execute("DELETE FROM attempts WHERE attempted_at<?", (cutoff,)).rowcount
                conn.commit()
                return removed
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None
