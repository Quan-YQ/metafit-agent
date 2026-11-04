"""单协调器模拟安全状态；不支持真实身份、跨进程或实际投递。"""
from dataclasses import dataclass, replace
from copy import deepcopy
from .coordinator import SessionOperation, _explicit_mismatch, _merge_warnings
from .event_store import ClaimResult, StorageError, _canonical
from .interaction_state import _timestamp
from .safety_signals import inspect_discomfort_signal
from .snapshot_coordinator import SnapshotSimulationCoordinator
from .training_events import validate_training_event


_PENDING_MESSAGE = "请先确认目前是否有不适，再获取普通反馈。"
_CLOSED_MESSAGE = "本会话的确认已结束；需要继续时，请主动开始新会话。"


@dataclass(slots=True)
class _SafetyMeta:
    pending: bool = True
    durability: str = "volatile"
    dirty_report: bool = False
    warning: str | None = None
    confirmation_closed: bool = False
    dirty_close: bool = False


def _pending_message(meta):
    return _CLOSED_MESSAGE if meta.confirmation_closed else _PENDING_MESSAGE


class DurableSafetyCoordinator(SnapshotSimulationCoordinator):
    """单实例模拟；普通状态每次重启需重新通过交互确认。"""

    def __init__(self, event_store):
        super().__init__(event_store)
        for method in ("open_session", "report", "confirm", "close_confirmation", "invalidate_session", "delete_user"):
            if not callable(getattr(event_store, method, None)):
                raise TypeError("safety store missing required method: " + method)
        self._safety_meta: dict[str, _SafetyMeta] = {}
        self._user_generation: dict[str, int] = {}
        self._user_revision: dict[str, int] = {}
        self._operations: dict[str, int] = {}
        self._deleting: set[str] = set()
        self._deletion_writing: set[str] = set()
        self._delete_results: dict[str, dict] = {}

    def _enter(self, session_id):
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            if entry is None or not entry.active or entry.user_id in self._deleting:
                return None
            user = entry.user_id
            self._operations[user] = self._operations.get(user, 0) + 1
            return user

    def _leave(self, user):
        with self._lock:
            self._operations[user] -= 1

    def _propagate_report_locked(self, user_id, reported_session):
        self._user_revision[user_id] = self._user_revision.get(user_id, 0) + 1
        for sid, entry in self._entries.items():
            if entry.user_id == user_id and sid != reported_session and entry.active:
                meta = self._safety_meta.get(sid)
                if meta is not None:
                    meta.pending = True
                    meta.durability = "volatile"
                    entry.state = replace(entry.state, questions=tuple(
                        replace(question, status="cancelled") if (question.status == "pending" or
                            (question.kind == "safety_confirmation" and question.status == "answered" and question.answer == "none")) else question
                        for question in entry.state.questions))
                    entry.safety_epoch += 1
                    entry.version += 1

    def view(self, session_id):
        with self._lock:
            view = super().view(session_id)
            meta = self._safety_meta.get(session_id) if isinstance(session_id, str) else None
            return replace(view, pending_confirmation=meta.pending, safety_state_durability=meta.durability) if view and meta else view

    def confirmation_view(self, session_id):
        """取得本实例的确认状态快照；不打开问题，也不代表真实授权。"""
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            meta = self._safety_meta.get(session_id) if isinstance(session_id, str) else None
            if entry is None or meta is None:
                return None
            questions = [q for q in entry.state.questions if q.kind == "safety_confirmation" and q.status != "cancelled"]
            question = questions[-1] if questions else None
            if entry.state.safety_reported:
                state = "reported"
            elif not entry.active or meta.durability != "durable":
                state = "unknown"
            elif meta.confirmation_closed:
                state = "closed"
            elif not meta.pending:
                state = "confirmed"
            elif question is not None:
                state = "open"
            else:
                state = "not_asked"
            return {"view": self.view(session_id), "confirmation_state": state,
                    "question_id": question.question_id if question and question.status == "pending" else None,
                    "question_presented": question.presented if question and question.status == "pending" else None,
                    "deadline": question.deadline if question and question.status == "pending" else None}

    def register(self, *, user_id, session_id, clock_id, new_session=False):
        if type(new_session) is not bool:
            return SessionOperation("stopped", "invalid_session")
        with self._lock:
            if isinstance(user_id, str) and user_id.strip() in self._deleting:
                return SessionOperation("stopped", "user_deleting")
            result = super().register(user_id=user_id, session_id=session_id, clock_id=clock_id)
            if result.status != "registered":
                return result
            user = result.view.user_id
            self._safety_meta[session_id] = _SafetyMeta()
            self._operations[user] = self._operations.get(user, 0) + 1
        try:
            try:
                saved = self._event_store.open_session(user, session_id, new_session=new_session)
            except StorageError as exc:
                with self._lock:
                    meta = self._safety_meta[session_id]
                    meta.warning = exc.code
                    if user in self._deleting or not self._entries[session_id].active:
                        return SessionOperation("stopped", "session_invalidated", safety_state_durability="volatile")
                    if exc.code == "scope_mismatch":
                        self._entries[session_id].active = False
                        return SessionOperation("stopped", "scope_mismatch", safety_state_durability="volatile")
                    return SessionOperation("registered", "pending_confirmation", self.view(session_id),
                                            warnings=(exc.code,), safety_state_durability="volatile", pending_message=_PENDING_MESSAGE)
            with self._lock:
                entry, meta = self._entries[session_id], self._safety_meta[session_id]
                if user in self._deleting or not entry.active:
                    return SessionOperation("stopped", "session_invalidated", self.view(session_id), safety_state_durability="volatile")
                previous_gen = self._user_generation.get(user, 0)
                self._user_generation[user] = max(previous_gen, saved.user_generation)
                if saved.user_generation > previous_gen:
                    self._propagate_report_locked(user, session_id)
                    if saved.warning is None and not any(self._safety_meta[other].dirty_report for other, e in self._entries.items() if e.user_id == user):
                        for other, e in self._entries.items():
                            other_meta = self._safety_meta[other]
                            if e.user_id == user and other_meta.warning is None and not other_meta.dirty_report:
                                other_meta.durability = "durable"
                if saved.state == "invalidated":
                    entry.active = False
                    return SessionOperation("stopped", "session_invalidated", self.view(session_id), safety_state_durability="durable")
                if saved.state == "reported":
                    entry.state = replace(entry.state, safety_reported=True, safety_reported_at=_timestamp(saved.reported_at))
                    entry.safety_epoch += 1
                    entry.safety_warnings = _merge_warnings(entry.safety_warnings, ("restored_source_unverified",))
                    meta.pending = True
                elif not entry.state.safety_reported and not meta.dirty_report:
                    local_report = any(e.user_id == user and e.state.safety_reported for e in self._entries.values())
                    meta.pending = saved.state != "normal" or local_report
                meta.confirmation_closed = meta.confirmation_closed or saved.confirmation_closed
                # 加载期间可能已报告不适，不能把未确认的内存状态当作已持久化。
                user_has_dirty_report = any(self._safety_meta[other].dirty_report for other, e in self._entries.items() if e.user_id == user)
                meta.durability = "volatile" if user_has_dirty_report or meta.dirty_report or saved.warning == "safety_state_invalid" else "durable"
                meta.dirty_report = meta.dirty_report or saved.warning == "safety_state_invalid" and entry.state.safety_reported
                meta.warning = (meta.warning or saved.warning) if meta.dirty_report or meta.dirty_close else saved.warning
                if saved.warning:
                    entry.safety_warnings = _merge_warnings(entry.safety_warnings, (saved.warning,))
                if entry.state.safety_reported:
                    safe = self._safety("session", entry.version, entry.safety_epoch, warnings=entry.safety_warnings)
                    return SessionOperation("safety_pending", view=self.view(session_id), safety_message=safe.safety_message,
                                            warnings=safe.warnings, safety_state_durability=meta.durability)
                return SessionOperation("registered", "pending_confirmation" if meta.pending and not entry.state.safety_reported else None,
                                        self.view(session_id), warnings=(saved.warning,) if saved.warning else (),
                                        safety_state_durability=meta.durability, pending_message=_pending_message(meta) if meta.pending else None)
        finally:
            self._leave(user)

    def _flush_report(self, user, sid):
        with self._lock:
            entry, meta = self._entries.get(sid), self._safety_meta.get(sid)
            if entry is None or meta is None or not meta.dirty_report:
                return
            reported = entry.state.safety_reported_at
            reported_at = reported.isoformat() if reported is not None else None
        try:
            saved = self._event_store.report(user, sid, reported_at)
        except StorageError as exc:
            with self._lock:
                meta.durability, meta.warning = "volatile", exc.code
            return
        with self._lock:
            meta.dirty_report = False
            meta.durability, meta.warning = "durable", saved.warning
            # 库中可能已有更早报告（例如恢复时一度无法读取），以首次已持久化时间为准。
            first_at = _timestamp(saved.reported_at)
            if entry.state.safety_reported_at != first_at:
                entry.state = replace(entry.state, safety_reported_at=first_at)
                entry.version += 1
            if saved.warning:
                entry.safety_warnings = _merge_warnings(entry.safety_warnings, (saved.warning,))
            self._user_generation[user] = max(self._user_generation.get(user, 0), saved.user_generation)
            if not any(self._safety_meta[other].dirty_report for other, e in self._entries.items() if e.user_id == user):
                for other, e in self._entries.items():
                    other_meta = self._safety_meta[other]
                    if e.user_id == user and not other_meta.dirty_report and other_meta.warning is None:
                        other_meta.durability = "durable"

    def _retry_user(self, user):
        with self._lock:
            dirty = [sid for sid, entry in self._entries.items() if entry.user_id == user and self._safety_meta[sid].dirty_report]
        for sid in dirty:
            self._flush_report(user, sid)
        with self._lock:
            closed = [sid for sid, entry in self._entries.items() if entry.user_id == user and self._safety_meta[sid].dirty_close]
        for sid in closed:
            self._flush_closed(user, sid)

    def _flush_closed(self, user, sid):
        with self._lock:
            entry, meta = self._entries[sid], self._safety_meta[sid]
            if not meta.dirty_close:
                return
            if entry.state.safety_reported:
                meta.dirty_close = False
                return
        try:
            self._event_store.close_confirmation(user, sid)
        except StorageError as exc:
            if exc.code == "safety_state_changed":
                try:
                    saved = self._event_store.open_session(user, sid)
                    if saved.state == "reported":
                        with self._lock:
                            if not entry.state.safety_reported:
                                entry.state = replace(entry.state, safety_reported=True, safety_reported_at=_timestamp(saved.reported_at))
                                entry.safety_epoch += 1
                                entry.version += 1
                                entry.safety_warnings = _merge_warnings(entry.safety_warnings, ("restored_source_unverified",))
                                self._propagate_report_locked(user, sid)
                            meta.dirty_close = False
                            meta.dirty_report = saved.warning == "safety_state_invalid"
                            meta.durability = "volatile" if meta.dirty_report else "durable"
                            meta.warning = saved.warning
                            self._user_generation[user] = max(self._user_generation.get(user, 0), saved.user_generation)
                        return
                    if saved.state == "invalidated":
                        with self._lock:
                            entry.active = False
                            meta.dirty_close = False
                            meta.durability = "durable"
                        return
                    self._event_store.close_confirmation(user, sid)
                except StorageError as retry_error:
                    with self._lock:
                        meta.durability, meta.warning = "volatile", retry_error.code
                    return
            else:
                with self._lock:
                    meta.durability, meta.warning = "volatile", exc.code
                return
        with self._lock:
            meta.dirty_close = False
            if not any(self._safety_meta[other].dirty_report for other, e in self._entries.items() if e.user_id == user):
                meta.durability, meta.warning = "durable", None

    def _pre_model_check(self, session_id, entry):
        meta = self._safety_meta[session_id]
        if meta.pending or meta.durability != "durable":
            return replace(self._stop("pending_confirmation", entry.version, entry.safety_epoch),
                           safety_state_durability=meta.durability, pending_message=_pending_message(meta))
        return None

    def apply(self, session_id, operation, *, expected_version=None, **kwargs):
        user = self._enter(session_id)
        if user is None:
            return SessionOperation("stopped", "session_invalidated" if self.view(session_id) else "state_missing")
        try:
            is_report = operation == "report_discomfort" or (operation == "answer_question" and kwargs.get("answer") == "reported")
            if not is_report:
                self._retry_user(user)
            with self._lock:
                answer_revision = self._user_revision.get(user, 0)
                meta = self._safety_meta[session_id]
                if operation == "open_question" and meta.pending and kwargs.get("kind") != "safety_confirmation":
                    return SessionOperation("stopped", "pending_confirmation", self.view(session_id), safety_state_durability=meta.durability,
                                            pending_message=_pending_message(meta))
                if operation == "open_question" and not meta.pending and kwargs.get("kind") == "safety_confirmation":
                    return SessionOperation("stopped", "confirmation_not_needed", self.view(session_id), safety_state_durability=meta.durability)
                if operation == "open_question" and kwargs.get("kind") == "safety_confirmation" and meta.confirmation_closed:
                    return SessionOperation("stopped", "confirmation_closed", self.view(session_id), safety_state_durability=meta.durability,
                                            pending_message=_CLOSED_MESSAGE)
            if is_report:
                with self._lock:
                    result = super().apply(session_id, operation, expected_version=expected_version, **kwargs)
                    if result.status == "safety_pending":
                        meta = self._safety_meta[session_id]
                        meta.pending, meta.dirty_report, meta.durability = True, True, "volatile"
                        self._propagate_report_locked(user, session_id)
                if result.status == "safety_pending":
                    self._flush_report(user, session_id)
                    with self._lock:
                        warning = (meta.warning,) if meta.warning else ()
                        return replace(result, view=self.view(session_id), warnings=_merge_warnings(result.warnings, warning),
                                       safety_state_durability=meta.durability)
                return result
            result = super().apply(session_id, operation, expected_version=expected_version, **kwargs)
            with self._lock:
                entry, meta = self._entries[session_id], self._safety_meta[session_id]
                closed_now = meta.pending and not meta.confirmation_closed and any(
                    q.kind == "safety_confirmation" and (q.status in {"skipped", "expired"} or
                    (q.status == "answered" and q.answer == "unknown")) for q in entry.state.questions)
                if closed_now:
                    meta.confirmation_closed, meta.dirty_close, meta.durability = True, True, "volatile"
            if closed_now:
                self._flush_closed(user, session_id)
                with self._lock:
                    if not entry.active:
                        return SessionOperation("stopped", "session_invalidated", self.view(session_id), safety_state_durability=meta.durability)
                    if entry.state.safety_reported:
                        safe = self._safety("session", entry.version, entry.safety_epoch, warnings=entry.safety_warnings)
                        return SessionOperation("safety_pending", view=self.view(session_id), safety_message=safe.safety_message,
                                                warnings=_merge_warnings(safe.warnings, (meta.warning,) if meta.warning else ()),
                                                safety_state_durability=meta.durability)
                    return replace(result, view=self.view(session_id), warnings=_merge_warnings(result.warnings, (meta.warning,) if meta.warning else ()),
                                   safety_state_durability=meta.durability, pending_message=_pending_message(meta))
            if operation == "answer_question" and kwargs.get("answer") == "none" and result.status == "answered":
                with self._lock:
                    entry, meta = self._entries[session_id], self._safety_meta[session_id]
                    q = next((q for q in entry.state.questions if q.question_id == kwargs.get("question_id")), None)
                    needs_confirmation = meta.pending and q is not None and q.kind == "safety_confirmation"
                    rev = answer_revision
                    dirty = any(self._safety_meta[sid].dirty_report for sid, e in self._entries.items() if e.user_id == user)
                    changed = self._user_revision.get(user, 0) != rev
                    gen = self._user_generation.get(user, 0)
                    volatile = meta.durability == "volatile"
                if needs_confirmation:
                    try:
                        if dirty or changed:
                            raise StorageError("safety_state_unconfirmed")
                        if volatile:
                            saved = self._event_store.open_session(user, session_id)
                            if saved.state == "reported":
                                with self._lock:
                                    entry.state = replace(entry.state, safety_reported=True, safety_reported_at=_timestamp(saved.reported_at))
                                    entry.safety_epoch += 1
                                    entry.version += 1
                                    entry.safety_warnings = _merge_warnings(entry.safety_warnings, ("restored_source_unverified",))
                                    meta.pending = True
                                    meta.dirty_report = saved.warning == "safety_state_invalid"
                                    meta.durability = "volatile" if meta.dirty_report else "durable"
                                    meta.warning = saved.warning
                                    self._user_generation[user] = max(self._user_generation.get(user, 0), saved.user_generation)
                                    self._propagate_report_locked(user, session_id)
                                    safe = self._safety("session", entry.version, entry.safety_epoch,
                                                        warnings=_merge_warnings(entry.safety_warnings, (saved.warning,) if saved.warning else ()))
                                    return SessionOperation("safety_pending", view=self.view(session_id), safety_message=safe.safety_message,
                                                            warnings=safe.warnings, safety_state_durability=meta.durability)
                            if saved.state == "invalidated":
                                raise StorageError("session_invalidated")
                            gen = saved.user_generation
                        saved = self._event_store.confirm(user, session_id, gen)
                    except StorageError as exc:
                        with self._lock:
                            meta.pending, meta.warning = True, exc.code
                            if entry.state.safety_reported:
                                safe = self._safety("session", entry.version, entry.safety_epoch,
                                                    warnings=_merge_warnings(entry.safety_warnings, (exc.code,)))
                                return SessionOperation("safety_pending", view=self.view(session_id), safety_message=safe.safety_message,
                                                        warnings=safe.warnings, safety_state_durability=meta.durability)
                            if not entry.active:
                                meta.durability = "volatile"
                                return SessionOperation("stopped", "session_invalidated", self.view(session_id),
                                                        warnings=(exc.code,), safety_state_durability=meta.durability)
                            meta.durability = "volatile"
                            return SessionOperation("stopped", "pending_confirmation", self.view(session_id),
                                                    warnings=(exc.code,), safety_state_durability="volatile", pending_message=_pending_message(meta))
                    with self._lock:
                        if (entry.state.safety_reported or not entry.active or
                            self._user_revision.get(user, 0) != rev or user in self._deleting):
                            meta.pending = True
                            if entry.state.safety_reported:
                                safe = self._safety("session", entry.version, entry.safety_epoch, warnings=entry.safety_warnings)
                                return SessionOperation("safety_pending", view=self.view(session_id), safety_message=safe.safety_message,
                                                        warnings=safe.warnings, safety_state_durability=meta.durability)
                            return SessionOperation("stopped", "pending_confirmation", self.view(session_id), safety_state_durability=meta.durability,
                                                    pending_message=_pending_message(meta))
                        meta.pending, meta.durability, meta.warning = False, "durable", None
                        self._user_generation[user] = max(self._user_generation.get(user, 0), saved.user_generation)
            with self._lock:
                return replace(result, view=self.view(session_id), safety_state_durability=self._safety_meta[session_id].durability)
        finally:
            self._leave(user)

    def decide(self, session_id, event, model, *, context=None, do_not_disturb=None):
        event = deepcopy(event)
        user = self._enter(session_id)
        if user is None:
            return self._stop("session_invalidated" if self.view(session_id) else "state_missing")
        try:
            with self._lock:
                entry, meta = self._entries[session_id], self._safety_meta[session_id]
                if _explicit_mismatch(event, entry):
                    return replace(self._stop("scope_mismatch"), safety_state_durability=meta.durability)
                prior = entry.state.safety_reported
                reported, _ = inspect_discomfort_signal(event)
                clock = entry.state.clock_id
            if reported:
                report = event.get("discomfort", {})
                self.apply(session_id, "report_discomfort", now=report.get("reported_at"), clock_id=clock)
            else:
                self._retry_user(user)
            with self._lock:
                entry, meta = self._entries[session_id], self._safety_meta[session_id]
                if not entry.active:
                    return replace(self._stop("session_invalidated", entry.version, entry.safety_epoch), safety_state_durability=meta.durability)
                if meta.pending and not entry.state.safety_reported:
                    save_pending = True
                else:
                    save_pending = False
            if save_pending:
                return self._save_pending_event(session_id, event)
            result = super().decide(session_id, event, model, context=context, do_not_disturb=do_not_disturb)
            with self._lock:
                entry, meta = self._entries[session_id], self._safety_meta[session_id]
                if entry.active and not entry.state.safety_reported and not meta.pending:
                    return replace(result, safety_state_durability=meta.durability)
                interrupted = "session_invalidated" if not entry.active else "safety_discarded"
                needs_update = (result.event_saved is True and result.storage_status == "saved" and
                                (result.candidate is not None or result.candidate_discarded))
            warnings, storage = result.warnings, result.storage_status
            if needs_update:
                valid = validate_training_event(event).event
                if valid is not None:
                    try:
                        self._event_store.update_not_delivered(valid.user_id, valid.session_id, valid.event_id, interrupted)
                    except StorageError:
                        warnings = _merge_warnings(warnings, ("decision_record_update_unconfirmed",))
                        storage = "saved_state_unconfirmed"
            with self._lock:
                entry, meta = self._entries[session_id], self._safety_meta[session_id]
                discarded = result.candidate is not None or result.candidate_discarded
                if not entry.active:
                    return replace(self._stop("session_invalidated", entry.version, entry.safety_epoch, discarded),
                                   event_saved=result.event_saved, storage_status=storage,
                                   safety_state_durability=meta.durability, warnings=warnings)
                if entry.state.safety_reported:
                    source = ("both" if prior else "event") if reported else (result.safety_source or "session")
                    safe = self._safety(source, entry.version, entry.safety_epoch, discarded,
                                        _merge_warnings(entry.safety_warnings, warnings, (meta.warning,) if meta.warning else ()))
                    return replace(safe, event_saved=result.event_saved, storage_status=storage,
                                   has_snapshot=result.has_snapshot, safety_state_durability=meta.durability)
                return replace(self._stop("pending_confirmation", entry.version, entry.safety_epoch, discarded),
                               event_saved=result.event_saved, storage_status=storage,
                               safety_state_durability=meta.durability, warnings=warnings, pending_message=_pending_message(meta))
        finally:
            self._leave(user)

    def _save_pending_event(self, session_id, event):
        """合法事件先保留；等待确认的事件不会自动重跑模型。"""
        validated = validate_training_event(event)
        saved, storage, record, warnings = False, "invalid_event", None, ("event_not_saved_invalid",)
        if validated.event is not None:
            try:
                claim = self._event_store.claim(event)
                if not isinstance(claim, ClaimResult) or claim.status not in {"inserted", "duplicate", "conflict"}:
                    raise StorageError("storage_failed")
                saved = claim.status != "conflict"
                storage = {"inserted": "saved", "duplicate": "existing", "conflict": "conflict"}[claim.status]
                warnings = _merge_warnings(claim.warnings, ("event_identity_conflict",) if claim.status == "conflict" else ())
                record = claim.record
                if claim.status == "inserted":
                    try:
                        record = self._finish_record(validated.event, "finished", self._stop("pending_confirmation"))
                    except StorageError:
                        storage = "saved_state_unconfirmed"
                        warnings = _merge_warnings(warnings, ("decision_state_update_failed",))
            except StorageError:
                saved, storage, warnings = None, "unknown", ("storage_failed",)
            if storage in {"unknown", "conflict", "saved_state_unconfirmed"}:
                try:
                    _, digest = _canonical(validated.event)
                    self._event_store.record_failure(validated.event.user_id, validated.event.session_id, validated.event.event_id,
                                                     digest, "event_conflict" if storage == "conflict" else "storage_failed")
                except StorageError:
                    warnings = _merge_warnings(warnings, ("failure_record_unavailable",))
        with self._lock:
            entry, meta = self._entries[session_id], self._safety_meta[session_id]
            if not entry.active:
                result = self._stop("session_invalidated", entry.version, entry.safety_epoch)
            elif entry.state.safety_reported:
                result = self._safety("session", entry.version, entry.safety_epoch,
                                      warnings=_merge_warnings(entry.safety_warnings, warnings))
            else:
                result = self._stop("pending_confirmation", entry.version, entry.safety_epoch)
            return replace(result, event_saved=saved, storage_status=storage, stored_event=record,
                           warnings=_merge_warnings(result.warnings, warnings), safety_state_durability=meta.durability,
                           pending_message=_pending_message(meta) if result.code == "pending_confirmation" else None)

    def invalidate(self, session_id, *, expected_version=None):
        user = self._enter(session_id)
        if user is None:
            return SessionOperation("stopped", "session_invalidated" if self.view(session_id) else "state_missing")
        try:
            result = super().invalidate(session_id, expected_version=expected_version)
            if result.status != "invalidated":
                return result
            try:
                self._event_store.invalidate_session(user, session_id)
                durability, warnings = "durable", ()
            except StorageError as exc:
                durability, warnings = "volatile", (exc.code,)
            with self._lock:
                self._safety_meta[session_id].durability = durability
                self._safety_meta[session_id].pending = True
                return replace(result, view=self.view(session_id), warnings=warnings, safety_state_durability=durability)
        finally:
            self._leave(user)

    def delete_user(self, user_id, *, allow_unresolved=False):
        """先失效所有本实例会话；繁忙时不删除，待操作结束后可重试。"""
        if not isinstance(user_id, str) or not user_id.strip() or type(allow_unresolved) is not bool:
            raise ValueError("invalid user or allow_unresolved")
        user_id = user_id.strip()
        with self._lock:
            if user_id in self._delete_results:
                return {**self._delete_results[user_id], "already_deleted": True}
            if user_id in self._deletion_writing:
                return {"status": "stopped", "code": "deletion_in_progress"}
            self._deleting.add(user_id)
            sessions = [sid for sid, entry in self._entries.items() if entry.user_id == user_id]
            for sid in sessions:
                super().invalidate(sid)
                self._safety_meta[sid].pending = True
            if self._operations.get(user_id, 0):
                return {"status": "stopped", "code": "operations_in_progress"}
            self._deletion_writing.add(user_id)
        try:
            for sid in sessions:
                self._event_store.invalidate_session(user_id, sid)
            result = self._event_store.delete_user(user_id, allow_unresolved=allow_unresolved)
            output = {"status": "deleted", **result}
            with self._lock:
                self._delete_results[user_id] = output
            return output
        except StorageError as exc:
            return {"status": "stopped", "code": exc.code, "unresolved_count": getattr(exc, "unresolved_count", None)}
        finally:
            with self._lock:
                self._deletion_writing.discard(user_id)
