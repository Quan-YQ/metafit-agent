"""模拟引用快照，只用于本机历史排查，不是事实认证或投递依据。"""
from contextlib import closing
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
import re
import sqlite3

from .decision_store import DecisionSQLiteStore
from .event_store import StorageError, _APPLICATION_ID, _V2_OBJECTS, _V3_OBJECTS, _V4_OBJECTS, _V5_OBJECTS
from .context_builder import fact_list_digest
from .facts import DecisionScope, Fact


def facts_digest(facts):
    """比较 DeepSeek 输入的有序 fact_id/name/value，不包含范围身份。"""
    payload = {"version": 1, "facts": [{"fact_id": f.fact_id, "name": f.name, "value": f.value} for f in facts]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SnapshotCapture:
    """只在调用栈内短暂保留真实模型输入与检查结果。"""
    facts: tuple[Fact, ...]
    digest: str
    evidence: tuple[str, ...]
    review_status: str
    review_stage: str | None
    scope: DecisionScope | None
    consistent: bool


class SnapshotSQLiteStore(DecisionSQLiteStore):
    """与决策同事务保存模拟引用；快照值默认不输出，不可用来自动投递。"""

    def __init__(self, path):
        super().__init__(path)
        self._capture = ContextVar(f"snapshot_capture_{id(self)}", default=None)

    def _initialize(self):
        super()._initialize()
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                identity = self._identity(conn)
                if identity == (2, _APPLICATION_ID, _V2_OBJECTS):
                    conn.execute("""CREATE TABLE decision_fact_meta (
                        user_id TEXT NOT NULL, session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                        snapshot_state TEXT NOT NULL, visible_fact_count INTEGER,
                        cited_fact_count INTEGER, fact_list_digest TEXT,
                        PRIMARY KEY (user_id, session_id, event_id),
                        FOREIGN KEY (user_id, session_id, event_id) REFERENCES decisions (user_id, session_id, event_id) ON DELETE CASCADE
                    )""")
                    conn.execute("""CREATE TABLE decision_facts (
                        user_id TEXT NOT NULL, session_id TEXT NOT NULL, event_id TEXT NOT NULL,
                        ordinal INTEGER NOT NULL, snapshot_json TEXT NOT NULL,
                        PRIMARY KEY (user_id, session_id, event_id, ordinal),
                        FOREIGN KEY (user_id, session_id, event_id) REFERENCES decisions (user_id, session_id, event_id) ON DELETE CASCADE
                    )""")
                    conn.execute("PRAGMA user_version=3")
                    identity = (3, _APPLICATION_ID, _V3_OBJECTS)
                if identity == (3, _APPLICATION_ID, _V3_OBJECTS):
                    conn.execute("ALTER TABLE decision_fact_meta ADD COLUMN facts_digest TEXT")
                    conn.execute("ALTER TABLE decision_fact_meta ADD COLUMN facts_digest_version INTEGER")
                    conn.execute("PRAGMA user_version=4")
                elif identity not in ((4, _APPLICATION_ID, _V4_OBJECTS), (5, _APPLICATION_ID, _V5_OBJECTS)):
                    raise StorageError("unsupported_store_version")
                conn.commit()
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None

    def finish_with_snapshot(self, user_id, session_id, event_id, *, state, result, capture):
        token = self._capture.set(capture)
        try:
            return self.finish_with_decision(user_id, session_id, event_id, state=state, result=result)
        finally:
            self._capture.reset(token)

    def _insert_decision_extras(self, conn, key, result):
        capture = self._capture.get()
        # 旧入口或安全拦截不冒充有快照；旧记录也不回填猜测值。
        state, visible, cited, digest, content_digest, snapshots = "not_recorded", None, None, None, None, []
        if capture is not None and result is not None:
            if not isinstance(capture, SnapshotCapture) or not isinstance(capture.facts, tuple) or not isinstance(capture.evidence, tuple):
                raise StorageError("snapshot_invalid")
            if (not capture.consistent or not isinstance(capture.scope, DecisionScope) or
                (capture.scope.user_id, capture.scope.session_id, capture.scope.event_id) != key or
                not isinstance(capture.digest, str) or not re.fullmatch(r"[0-9a-f]{64}", capture.digest)):
                raise StorageError("snapshot_invalid")
            try:
                if fact_list_digest(capture.scope, capture.facts) != capture.digest:
                    raise StorageError("snapshot_invalid")
            except (AttributeError, TypeError, ValueError, UnicodeError, RecursionError):
                raise StorageError("snapshot_invalid") from None
            visible, cited, digest = len(capture.facts), len(capture.evidence), capture.digest
            try:
                content_digest = facts_digest(capture.facts)
            except (AttributeError, TypeError, ValueError, UnicodeError, RecursionError):
                raise StorageError("snapshot_invalid") from None
            if result.status == "reviewed" and capture.review_status == "reviewed":
                if result.candidate is None or result.candidate.evidence != capture.evidence:
                    raise StorageError("snapshot_invalid")
                index = {f.fact_id: f for f in capture.facts if isinstance(f, Fact)}
                if len(index) != len(capture.facts) or len(set(capture.evidence)) != len(capture.evidence):
                    raise StorageError("snapshot_invalid")
                if any(fact_id not in index for fact_id in capture.evidence):
                    raise StorageError("snapshot_invalid")
                for fact_id in capture.evidence:
                    fact = index[fact_id]
                    origin = "simulated_context" if fact.origin.startswith("simulated_context:") else fact.origin
                    if origin not in {"event", "history", "simulated_context"}:
                        raise StorageError("snapshot_invalid")
                    snapshot = {"fact_id": fact.fact_id, "name": fact.name, "value": fact.value,
                                "kind": fact.kind, "origin": origin, "field": fact.field}
                    try:
                        snapshots.append(json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":"), allow_nan=False))
                    except (TypeError, ValueError, UnicodeError, RecursionError):
                        raise StorageError("snapshot_invalid") from None
                state = "recorded" if snapshots else "no_references"
            elif result.status == "stopped" and capture.review_status == "stopped" and capture.review_stage == "r3":
                state = "check_failed"
            else:
                visible, cited, digest, content_digest = None, None, None, None
        conn.execute("INSERT INTO decision_fact_meta VALUES (?,?,?,?,?,?,?,?,?)",
                     (*key, state, visible, cited, digest, content_digest, 1 if content_digest is not None else None))
        conn.executemany("INSERT INTO decision_facts VALUES (?,?,?,?,?)",
                         [(*key, i, value) for i, value in enumerate(snapshots)])

    def _decorate_query(self, conn, rows, include_values, include_names):
        version = self._identity(conn)[0]
        metadata, facts_by_key = {}, {}
        keys = [(row["user_id"], row["session_id"], row["event_id"]) for row in rows]
        if version >= 3:
            # 200 个组合键使用 600 个绑定参数，避免大量逐行查询和旧 SQLite 参数限制。
            for offset in range(0, len(keys), 200):
                chunk = keys[offset:offset + 200]
                where = " OR ".join("(user_id=? AND session_id=? AND event_id=?)" for _ in chunk)
                params = [part for key in chunk for part in key]
                digest_columns = ",facts_digest,facts_digest_version" if version >= 4 else ",NULL,NULL"
                for item in conn.execute("SELECT user_id,session_id,event_id,snapshot_state,visible_fact_count,cited_fact_count,fact_list_digest" + digest_columns + " FROM decision_fact_meta WHERE " + where, params):
                    metadata[tuple(item[:3])] = tuple(item[3:])
                for item in conn.execute("SELECT user_id,session_id,event_id,ordinal,snapshot_json FROM decision_facts WHERE " + where + " ORDER BY ordinal", params):
                    facts_by_key.setdefault(tuple(item[:3]), []).append((item[3], item[4]))
        for row, key in zip(rows, keys):
            meta = metadata.get(key)
            row["snapshot_state"] = meta[0] if meta else "not_recorded"
            row["visible_fact_count"] = meta[1] if meta else None
            row["cited_fact_count"] = meta[2] if meta else None
            row["fact_list_digest"] = meta[3] if meta else None
            row["facts_digest"] = meta[4] if meta else None
            row["facts_digest_version"] = meta[5] if meta else None
            row["has_snapshot"] = row["snapshot_state"] == "recorded"
            row["facts"] = []
            row["snapshot_warning"] = None
            row["no_snapshot_reason"] = None
            try:
                ids = []
                for ordinal, raw in facts_by_key.get(key, []):
                    if ordinal != len(ids) or not isinstance(raw, str) or len(raw) > 32768:
                        raise StorageError("snapshot_invalid")
                    fact = json.loads(raw)
                    if not isinstance(fact, dict) or set(fact) != {"fact_id", "name", "value", "kind", "origin", "field"}:
                        raise StorageError("snapshot_invalid")
                    if (any(not isinstance(fact[k], str) for k in ("fact_id", "name", "kind", "origin", "field")) or
                        fact["origin"] not in {"event", "history", "simulated_context"} or
                        type(fact["value"]) not in {str, int, float} or
                        (type(fact["value"]) is float and not math.isfinite(fact["value"]))):
                        raise StorageError("snapshot_invalid")
                    ids.append(fact["fact_id"])
                    if not include_values:
                        fact.pop("value")
                    if not include_names:
                        fact.pop("name")
                        fact.pop("field")
                    row["facts"].append(fact)
                state = row["snapshot_state"]
                if state not in {"recorded", "no_references", "not_recorded", "check_failed"}:
                    raise StorageError("snapshot_invalid")
                if state in {"recorded", "no_references", "check_failed"}:
                    if (type(row["visible_fact_count"]) is not int or row["visible_fact_count"] < 0 or
                        type(row["cited_fact_count"]) is not int or row["cited_fact_count"] < 0 or
                        not isinstance(row["fact_list_digest"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["fact_list_digest"])):
                        raise StorageError("snapshot_invalid")
                elif any(row[name] is not None for name in ("visible_fact_count", "cited_fact_count", "fact_list_digest")):
                    raise StorageError("snapshot_invalid")
                fd, fd_version = row["facts_digest"], row["facts_digest_version"]
                if ((fd is None) != (fd_version is None) or
                    (state == "not_recorded" and fd is not None) or
                    (fd is not None and (not isinstance(fd, str) or not re.fullmatch(r"[0-9a-f]{64}", fd) or type(fd_version) is not int or fd_version != 1))):
                    raise StorageError("snapshot_invalid")
                if state == "recorded":
                    if (not ids or len(set(ids)) != len(ids) or ids != row["evidence"] or
                        len(ids) != row["cited_fact_count"] or row["visible_fact_count"] < len(ids)):
                        raise StorageError("snapshot_invalid")
                elif ids or (state == "no_references" and (row["evidence"] or row["cited_fact_count"] != 0)):
                    raise StorageError("snapshot_invalid")
                if state == "check_failed":
                    row["no_snapshot_reason"] = "r3_rejected"
                elif state == "not_recorded":
                    row["no_snapshot_reason"] = ("unprocessed" if row["decision_state"] == "reserved" else
                        "legacy_or_other_entry" if meta is None else
                        "safety" if row["status"] == "safety_pending" else
                        "r2_error" if row["stage"] == "r2" else "not_applicable")
            except (StorageError, ValueError, TypeError, RecursionError):
                row.update(snapshot_state="snapshot_invalid", snapshot_warning="snapshot_invalid",
                           no_snapshot_reason="invalid", has_snapshot=None, facts=[], visible_fact_count=None,
                           cited_fact_count=None, fact_list_digest=None, facts_digest=None, facts_digest_version=None)
        return rows

    def count_fact_references(self, user_id, name=None, *, kind=None, origin=None):
        """按名称、类别和来源筛选本机模拟引用；不返回值，不用于授权。"""
        if (not isinstance(user_id, str) or not user_id.strip() or
            all(value is None for value in (name, kind, origin)) or
            any(value is not None and (not isinstance(value, str) or not value.strip()) for value in (name, kind, origin))):
            raise ValueError("invalid user or fact filters")
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3.0)) as conn:
                identity = self._identity(conn)
                if identity == (2, _APPLICATION_ID, _V2_OBJECTS):
                    return 0
                if identity not in ((3, _APPLICATION_ID, _V3_OBJECTS), (4, _APPLICATION_ID, _V4_OBJECTS), (5, _APPLICATION_ID, _V5_OBJECTS)):
                    raise StorageError("unsupported_store_version")
                conn.execute("PRAGMA query_only=ON")
                conditions, params = ["user_id=?"], [user_id]
                for field, value in (("name", name), ("kind", kind), ("origin", origin)):
                    if value is not None:
                        conditions.append("json_extract(snapshot_json,'$." + field + "')=?")
                        params.append(value)
                return conn.execute("SELECT count(*) FROM decision_facts WHERE " + " AND ".join(conditions), params).fetchone()[0]
        except StorageError:
            raise
        except (sqlite3.Error, OSError, ValueError):
            raise StorageError("storage_failed") from None
