"""本机模拟引用快照入口；不得把历史快照当作授权、事实证明或投递依据。"""
from contextvars import ContextVar
from dataclasses import replace

from .coordinator import _merge_warnings
from .recorded_coordinator import RecordedSimulationCoordinator
from .snapshot_store import SnapshotCapture


class SnapshotSimulationCoordinator(RecordedSimulationCoordinator):
    def __init__(self, event_store):
        super().__init__(event_store)
        if not callable(getattr(event_store, "finish_with_snapshot", None)):
            raise TypeError("snapshot store missing required method: finish_with_snapshot")
        self._review_capture = ContextVar(f"review_capture_{id(self)}", default=None)

    def decide(self, session_id, event, model, *, context=None, do_not_disturb=None):
        token = self._review_capture.set(None)
        try:
            return super().decide(session_id, event, model, context=context, do_not_disturb=do_not_disturb)
        finally:
            self._review_capture.reset(token)

    def _after_review(self, outcome, reviewed):
        # 仅在本次确有候选及构建上下文时观察 R3 的实际输入，禁止事后重新构造。
        if outcome.candidate is None:
            return
        built, scope = outcome.context_build, outcome.scope
        consistent = (built is not None and scope is not None and built.facts == outcome.facts and built.scope == scope)
        digest = built.fact_list_digest if built is not None else ""
        self._review_capture.set(SnapshotCapture(outcome.facts, digest, outcome.candidate.evidence,
                                                 reviewed.status, reviewed.stage, scope, consistent))

    def _finish_record(self, event, state, result=None):
        capture = self._review_capture.get() if result is not None else None
        return self._event_store.finish_with_snapshot(event.user_id, event.session_id, event.event_id,
                                                       state=state, result=result, capture=capture)

    def _on_record_query(self, result, row):
        result = super()._on_record_query(result, row)
        if row.get("snapshot_state") != "snapshot_invalid":
            return result
        warnings = _merge_warnings(result.warnings, ("snapshot_invalid",))
        if result.status != "reviewed":
            return replace(result, warnings=warnings, has_snapshot=None)
        return self._storage_stop("snapshot_invalid", version=result.checked_at_version,
                                  epoch=result.checked_safety_epoch, discarded=result.candidate is not None,
                                  event_saved=result.event_saved, storage_status="saved_state_unconfirmed",
                                  warnings=warnings)
