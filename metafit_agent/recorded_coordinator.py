"""R5 离线模拟记录入口；结果不能作为投递凭证。"""
from dataclasses import replace
from copy import deepcopy
from typing import Protocol

from .coordinator import _merge_warnings
from .event_store import StorageError, _canonical
from .stored_coordinator import StoredSimulationCoordinator
from .training_events import validate_training_event


class DecisionStore(Protocol):
    """记录协调器需要的存储能力；不验证实现的行为或线程安全。"""
    def claim(self, event_data): ...
    def finish_with_decision(self, user_id, session_id, event_id, *, state, result): ...
    def update_not_delivered(self, user_id, session_id, event_id, reason): ...
    def record_failure(self, user_id, session_id, event_id, content_hash, failure_code): ...
    def query(self, user_id, *, session_id=None, event_id=None, limit=100): ...


class RecordedSimulationCoordinator(StoredSimulationCoordinator):
    def __init__(self, event_store: DecisionStore):
        required = ("claim", "finish_with_decision", "update_not_delivered", "record_failure", "query")
        missing = [name for name in required if not callable(getattr(event_store, name, None))]
        if missing:
            raise TypeError("decision store missing required methods: " + ", ".join(missing))
        super().__init__(event_store)

    def _finish_record(self, event, state, result=None):
        return self._event_store.finish_with_decision(event.user_id, event.session_id, event.event_id,
                                                       state=state, result=result)

    def _on_record_query(self, result, row):
        return replace(result, has_text=row["has_text"], decided_at=row["decided_at"],
                       has_snapshot=row.get("has_snapshot"))

    def decide(self, session_id, event, model, *, context=None, do_not_disturb=None):
        snapshot = deepcopy(event)
        result = super().decide(session_id, snapshot, model, context=context, do_not_disturb=do_not_disturb)
        if result.storage_status == "not_attempted":
            return result
        valid = validate_training_event(snapshot).event
        if valid is None:
            return result
        key = (valid.user_id, valid.session_id, valid.event_id)
        # 只对本次插入的记录写最终核对结果，不能重写重复请求的历史。
        if result.storage_status == "saved":
            reason = ("safety_discarded" if result.status == "safety_pending" else
                      "session_invalidated" if result.code in {"session_invalidated", "state_missing"} else
                      "pending_confirmation" if result.code == "pending_confirmation" else
                      ("rule_stopped" if result.stage == "r3" else "decision_stopped") if result.status == "stopped" else
                      "dnd_deferred" if result.rule_code == "dnd_notify_to_defer" else "not_delivered")
            try:
                self._event_store.update_not_delivered(*key, reason)
            except StorageError:
                warnings = _merge_warnings(result.warnings, ("decision_record_update_unconfirmed",))
                if result.status == "safety_pending":
                    result = replace(result, warnings=warnings, storage_status="saved_state_unconfirmed")
                else:
                    result = self._storage_stop("storage_failed", version=result.checked_at_version,
                                                epoch=result.checked_safety_epoch,
                                                discarded=result.candidate is not None or result.candidate_discarded,
                                                event_saved=True, storage_status="saved_state_unconfirmed", warnings=warnings)
            # 更新期间仍允许不适登记；复核后先修正历史，再返回安全结果。
            with self._lock:
                entry = self._entries.get(session_id) if isinstance(session_id, str) else None
                if entry is None:
                    final_reason = "session_invalidated"
                    final = replace(self._stop("state_missing", discarded=result.candidate is not None or result.candidate_discarded),
                                    warnings=result.warnings)
                elif not entry.active:
                    final_reason = "session_invalidated"
                    final = replace(self._stop("session_invalidated", entry.version, entry.safety_epoch,
                                               result.candidate is not None or result.candidate_discarded), warnings=result.warnings)
                elif entry.state.safety_reported:
                    final_reason = "safety_discarded"
                    final = self._safety(result.safety_source or "session", entry.version, entry.safety_epoch,
                                         result.candidate is not None or result.candidate_discarded,
                                         _merge_warnings(entry.safety_warnings, result.warnings))
                else:
                    final_reason = reason
                    final = result
            if final_reason != reason:
                marker_status = result.storage_status
                try:
                    self._event_store.update_not_delivered(*key, final_reason)
                except StorageError:
                    final = replace(final, warnings=_merge_warnings(final.warnings, ("decision_record_update_unconfirmed",)))
                    marker_status = "saved_state_unconfirmed"
                result = replace(final, event_saved=True, storage_status=marker_status, stored_event=result.stored_event)
        if result.code in {"storage_failed", "event_conflict"} or result.storage_status == "saved_state_unconfirmed":
            try:
                _, digest = _canonical(valid)
                self._event_store.record_failure(*key, digest, "event_conflict" if result.code == "event_conflict" else "storage_failed")
            except StorageError:
                result = replace(result, warnings=_merge_warnings(result.warnings, ("failure_record_unavailable",)))
        if result.event_saved is True and result.storage_status in {"saved", "existing"}:
            try:
                rows = self._event_store.query(valid.user_id, session_id=valid.session_id, event_id=valid.event_id, limit=1)
                row = rows[0] if rows else None
                if row is not None:
                    result = self._on_record_query(result, row)
            except StorageError:
                result = replace(result, warnings=_merge_warnings(result.warnings, ("decision_query_unavailable",)))
        # 最后一次读写之后再核对，防止查询期间登记不适后仍返回普通候选。
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            if entry is None or not entry.active:
                reason = "session_invalidated"
                final = replace(self._stop("state_missing" if entry is None else "session_invalidated",
                                           entry.version if entry else None, entry.safety_epoch if entry else None,
                                           result.candidate is not None or result.candidate_discarded), warnings=result.warnings)
            elif entry.state.safety_reported:
                reason = "safety_discarded"
                final = self._safety(result.safety_source or "session", entry.version, entry.safety_epoch,
                                     result.candidate is not None or result.candidate_discarded,
                                     _merge_warnings(entry.safety_warnings, result.warnings))
            else:
                return result
        if result.storage_status == "saved":
            try:
                self._event_store.update_not_delivered(*key, reason)
            except StorageError:
                final = replace(final, warnings=_merge_warnings(final.warnings, ("decision_record_update_unconfirmed",)))
                result = replace(result, storage_status="saved_state_unconfirmed")
                try:
                    _, digest = _canonical(valid)
                    self._event_store.record_failure(*key, digest, "decision_update_failed")
                except StorageError:
                    final = replace(final, warnings=_merge_warnings(final.warnings, ("failure_record_unavailable",)))
        return replace(final, event_saved=result.event_saved, storage_status=result.storage_status,
                       stored_event=result.stored_event, has_text=result.has_text, decided_at=result.decided_at,
                       has_snapshot=result.has_snapshot)
