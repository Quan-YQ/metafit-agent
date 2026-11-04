"""R5 模拟保存协调入口；旧的独立协调入口不自动写入数据库。"""
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .coordinator import SimulationCoordinator, _explicit_mismatch, _merge_warnings
from .event_store import ClaimResult, EventStore, StorageError
from .training_events import validate_training_event


class _CountedModel:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def complete(self, facts):
        self.calls += 1
        return self.model.complete(facts)


class StoredSimulationCoordinator(SimulationCoordinator):
    """保存只在锁外进行；内存会话安全状态仍不会跨进程持久化。"""

    def __init__(self, event_store: EventStore):
        super().__init__()
        self._event_store = event_store

    def _finish_record(self, event, state, result=None):
        return self._event_store.finish(event.user_id, event.session_id, event.event_id, state=state)

    def _storage_stop(self, code, *, version=None, epoch=None, discarded=False,
                      event_saved=None, storage_status="unknown", record=None, warnings=()):
        result = self._stop(code, version, epoch, discarded)
        return replace(result, stage="storage", event_saved=event_saved, storage_status=storage_status,
                       stored_event=record, warnings=_merge_warnings(result.warnings, warnings))

    def _safety_after_storage(self, session_id, initial_source, *, event_saved, storage_status,
                              warnings=(), discarded=False, record=None):
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            if entry is None:
                return replace(self._stop("state_missing", discarded=discarded), event_saved=event_saved,
                               storage_status=storage_status, warnings=warnings)
            if not entry.active:
                return replace(self._stop("session_invalidated", entry.version, entry.safety_epoch, discarded),
                               event_saved=event_saved, storage_status=storage_status, warnings=warnings)
            if not entry.state.safety_reported:
                return self._storage_stop("safety_state_unavailable", version=entry.version,
                                          epoch=entry.safety_epoch, discarded=discarded,
                                          event_saved=event_saved, storage_status=storage_status, warnings=warnings)
            source = initial_source if initial_source else "session"
            result = self._safety(source, entry.version, entry.safety_epoch, discarded,
                                  _merge_warnings(entry.safety_warnings, warnings))
            return replace(result, event_saved=event_saved, storage_status=storage_status, stored_event=record)

    def decide(self, session_id: object, event: Any, model: Any, *, context: Any = None,
               do_not_disturb: object = None):
        event_snapshot = deepcopy(event)
        context_snapshot = deepcopy(context)
        # 先锁定并更新安全状态，再释放锁进行所有数据库 I/O。
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            if entry is None:
                return replace(self._stop("state_missing"), storage_status="not_attempted")
            if not entry.active:
                return replace(self._stop("session_invalidated", entry.version, entry.safety_epoch),
                               storage_status="not_attempted")
            if _explicit_mismatch(event_snapshot, entry):
                return replace(self._stop("scope_mismatch"), storage_status="not_attempted")
            initial_source = self._observe_event_locked(entry, event_snapshot)

        validated = validate_training_event(event_snapshot)
        claim = None
        storage_warnings = ()
        if validated.event is None:
            event_saved = False
            storage_status = "invalid_event"
            storage_warnings = ("event_not_saved_invalid",)
        else:
            try:
                claim = self._event_store.claim(event_snapshot)
                if not isinstance(claim, ClaimResult) or claim.status not in {"inserted", "duplicate", "conflict"}:
                    raise StorageError("storage_failed")
                event_saved = claim.status != "conflict"
                storage_status = {"inserted": "saved", "duplicate": "existing", "conflict": "conflict"}[claim.status]
                storage_warnings = claim.warnings + (("event_identity_conflict",) if claim.status == "conflict" else ())
            except StorageError:
                event_saved = None
                storage_status = "unknown"
                storage_warnings = ("storage_failed",)

        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            if entry is None:
                return replace(self._stop("state_missing"), event_saved=event_saved, storage_status=storage_status,
                               warnings=storage_warnings)
            if not entry.active:
                return replace(self._stop("session_invalidated", entry.version, entry.safety_epoch),
                               event_saved=event_saved, storage_status=storage_status, warnings=storage_warnings)
            safety_now = entry.state.safety_reported

        if safety_now:
            safety_record = claim.record if claim is not None and claim.status == "duplicate" else None
            if claim is not None and claim.status == "inserted":
                try:
                    safety_record = self._finish_record(validated.event, "safety_only")
                except StorageError:
                    storage_warnings = _merge_warnings(storage_warnings, ("decision_state_update_failed",))
                    storage_status = "saved_state_unconfirmed"
            return self._safety_after_storage(session_id, initial_source, event_saved=event_saved,
                                              storage_status=storage_status, warnings=storage_warnings,
                                              record=safety_record)
        if storage_status in {"unknown", "conflict", "existing"}:
            # 最终返回前在锁内再次读取安全状态；与不适登记串行化。
            with self._lock:
                entry = self._entries.get(session_id) if isinstance(session_id, str) else None
                if entry is None:
                    return replace(self._stop("state_missing"), event_saved=event_saved,
                                   storage_status=storage_status, warnings=storage_warnings)
                if not entry.active:
                    return replace(self._stop("session_invalidated", entry.version, entry.safety_epoch),
                                   event_saved=event_saved, storage_status=storage_status, warnings=storage_warnings)
                if entry.state.safety_reported:
                    safe = self._safety(initial_source or "session", entry.version, entry.safety_epoch,
                                        warnings=_merge_warnings(entry.safety_warnings, storage_warnings))
                    return replace(safe, event_saved=event_saved, storage_status=storage_status)
                code = {"unknown": "storage_failed", "conflict": "event_conflict", "existing": "duplicate_event"}[storage_status]
                return self._storage_stop(code, version=entry.version, epoch=entry.safety_epoch,
                                          event_saved=event_saved, storage_status=storage_status,
                                          record=claim.record if storage_status == "existing" else None,
                                          warnings=storage_warnings)

        counted = _CountedModel(model)
        result = super().decide(session_id, event_snapshot, counted, context=context_snapshot,
                                do_not_disturb=do_not_disturb)
        finish_failed = False
        stored_record = claim.record if claim is not None else None
        if claim is not None and claim.status == "inserted":
            marker = "safety_only" if result.status == "safety_pending" and counted.calls == 0 else "finished"
            try:
                stored_record = self._finish_record(validated.event, marker, result)
            except StorageError:
                finish_failed = True
                storage_status = "saved_state_unconfirmed"
                storage_warnings = _merge_warnings(storage_warnings, ("decision_state_update_failed",))

        # 完成标记也可能耗时；最后重新检查不适与失效，不在锁内执行 I/O。
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            discarded = result.candidate is not None or result.candidate_discarded
            if entry is None:
                return replace(self._stop("state_missing", discarded=discarded), event_saved=event_saved,
                               storage_status=storage_status, warnings=storage_warnings)
            if not entry.active:
                return replace(self._stop("session_invalidated", entry.version, entry.safety_epoch, discarded),
                               event_saved=event_saved, storage_status=storage_status, warnings=storage_warnings)
            if entry.state.safety_reported:
                source = initial_source if initial_source else (result.safety_source or "session")
                safe = self._safety(source, entry.version, entry.safety_epoch, discarded,
                                    _merge_warnings(entry.safety_warnings, result.warnings, storage_warnings))
                return replace(safe, event_saved=event_saved, storage_status=storage_status)
            if finish_failed:
                return self._storage_stop("storage_failed", version=entry.version, epoch=entry.safety_epoch,
                                          discarded=discarded, event_saved=event_saved,
                                          storage_status=storage_status, warnings=storage_warnings)
            return replace(result, event_saved=event_saved, storage_status=storage_status,
                           stored_event=stored_record, warnings=_merge_warnings(result.warnings, storage_warnings))
