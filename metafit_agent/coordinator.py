"""R4 第三单元：单进程模拟协调；不是鉴权、存储或安全投递系统。"""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from inspect import signature
import threading
from typing import Any

from . import interaction_state as interactions
from .decision import CandidateDecision, DecisionOutcome, decide_simulated_event
from .decision_review import review_outcome
from .safety_signals import inspect_discomfort_signal
from .training_events import validate_training_event


@dataclass(frozen=True, slots=True)
class SessionView:
    session_id: str
    user_id: str
    version: int
    safety_epoch: int
    safety_reported: bool
    safety_reported_at: datetime | None
    active: bool
    safety_state_durability: str | None = None
    pending_confirmation: bool = False


@dataclass(frozen=True, slots=True)
class SessionOperation:
    status: str
    code: str | None = None
    view: SessionView | None = None
    question_text: str | None = None
    safety_message: str | None = None
    warnings: tuple[str, ...] = ()
    safety_state_durability: str | None = None
    pending_message: str | None = None


@dataclass(frozen=True, slots=True)
class CoordinatedResult:
    status: str
    code: str | None = None
    effective_action: str | None = None
    candidate: CandidateDecision | None = None
    safety_source: str | None = None
    safety_message: str | None = None
    checked_at_version: int | None = None
    checked_safety_epoch: int | None = None
    candidate_discarded: bool = False
    notice: str = "仅供单进程模拟；未验证真实身份、自然语言安全或实际投递。"
    stage: str | None = None
    warnings: tuple[str, ...] = ()
    event_saved: bool | None = None
    storage_status: str = "not_configured"
    stored_event: Any = None
    rule_code: str | None = None
    reference_check: str = "not_checked"
    has_text: bool | None = None
    decided_at: str | None = None
    has_snapshot: bool | None = None
    safety_state_durability: str | None = None
    pending_message: str | None = None


@dataclass(slots=True)
class _Entry:
    user_id: str
    state: interactions.InteractionSession
    version: int = 0
    safety_epoch: int = 0
    active: bool = True
    safety_warnings: tuple[str, ...] = ()


_OPERATIONS = {
    "open_question": interactions.open_question,
    "present_question": interactions.present_question,
    "answer_question": interactions.answer_question,
    "skip_question": interactions.skip_question,
    "expire_safety_confirmation": interactions.expire_safety_confirmation,
    "mark_training_started": interactions.mark_training_started,
    "report_discomfort": interactions.report_discomfort,
}


def _view(entry: _Entry) -> SessionView:
    return SessionView(entry.state.session_id, entry.user_id, entry.version, entry.safety_epoch,
                       entry.state.safety_reported, entry.state.safety_reported_at, entry.active)


def _explicit_mismatch(event: Any, entry: _Entry) -> bool:
    if not isinstance(event, dict):
        return False
    for key, expected in (("session_id", entry.state.session_id), ("user_id", entry.user_id)):
        value = event.get(key)
        if isinstance(value, str) and value.strip() and value.strip() != expected:
            return True
    return False


def _merge_warnings(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _safety_source(event: bool, session: bool) -> str | None:
    return "both" if event and session else "event" if event else "session" if session else None


class SimulationCoordinator:
    """每实例自有注册表和锁；直接调用旧 R2/R3 入口仍会绕过本协调器。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}

    def _after_review(self, outcome, reviewed):
        """记录入口可观察本次 R3 检查，不改变模型或 R3 接口。"""
        pass

    def _pre_model_check(self, session_id, entry):
        """持锁调用；派生协调器可在模型启动前拒绝。"""
        return None

    def register(self, *, user_id: object, session_id: object, clock_id: object) -> SessionOperation:
        if not isinstance(user_id, str) or not user_id.strip():
            return SessionOperation("stopped", "invalid_user")
        try:
            state = interactions.create_session(session_id, clock_id)
        except ValueError:
            return SessionOperation("stopped", "invalid_session")
        with self._lock:
            if state.session_id in self._entries:
                return SessionOperation("stopped", "session_exists")
            entry = _Entry(user_id.strip(), state)
            self._entries[state.session_id] = entry
            return SessionOperation("registered", view=_view(entry))

    def view(self, session_id: object) -> SessionView | None:
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            return _view(entry) if entry is not None else None

    def apply(self, session_id: object, operation: str, *, expected_version: object = None, **kwargs) -> SessionOperation:
        """只接收操作参数，不接受外部状态对象；报告不适不因旧版本丢弃。"""
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            if entry is None:
                return SessionOperation("stopped", "state_missing")
            if not entry.active:
                return SessionOperation("stopped", "session_invalidated", view=_view(entry))
            fn = _OPERATIONS.get(operation) if isinstance(operation, str) else None
            if fn is None:
                return SessionOperation("stopped", "invalid_operation", view=_view(entry))
            is_safety = operation == "report_discomfort" or (operation == "answer_question" and kwargs.get("answer") == "reported")
            if entry.state.safety_reported and not is_safety:
                return SessionOperation("safety_pending", view=_view(entry), safety_message=interactions.SAFETY_MESSAGE,
                                        warnings=entry.safety_warnings)
            if not is_safety:
                try:
                    signature(fn).bind(entry.state, **kwargs)
                except TypeError:
                    return SessionOperation("stopped", "invalid_arguments", view=_view(entry))
            version_bad = expected_version is not None and (type(expected_version) is not int or expected_version != entry.version)
            if version_bad and not is_safety:
                return SessionOperation("stopped", "version_conflict", view=_view(entry))
            # 明确报告不适时，缺失问题 ID、时间或多余字段均不能使报告丢失。
            result = (interactions.report_discomfort(entry.state, now=kwargs.get("now"), clock_id=kwargs.get("clock_id"))
                      if is_safety else fn(entry.state, **kwargs))
            was_reported = entry.state.safety_reported
            allowed = {"now", "clock_id"} if operation == "report_discomfort" else {"question_id", "answer", "now", "clock_id"}
            extras = is_safety and bool(set(kwargs) - allowed)
            warnings = (result.warnings + (("stale_version_ignored_for_safety",) if version_bad and is_safety else ())
                        + (("extra_safety_arguments",) if extras else ()))
            safety_warnings = (_merge_warnings(entry.safety_warnings, result.warnings,
                                               ("extra_safety_arguments",) if extras else ())
                               if is_safety else entry.safety_warnings)
            if result.session != entry.state or safety_warnings != entry.safety_warnings:
                entry.state = result.session
                entry.safety_warnings = safety_warnings
                entry.version += 1
                if not was_reported and entry.state.safety_reported:
                    entry.safety_epoch += 1
            if is_safety:
                warnings = _merge_warnings(entry.safety_warnings, warnings)
            return SessionOperation(result.status, result.code, _view(entry), result.question_text,
                                    result.safety_message, warnings)

    def invalidate(self, session_id: object, *, expected_version: object = None) -> SessionOperation:
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            if entry is None:
                return SessionOperation("stopped", "state_missing")
            if expected_version is not None and (type(expected_version) is not int or expected_version != entry.version):
                return SessionOperation("stopped", "version_conflict", view=_view(entry))
            if entry.active:
                entry.active = False
                entry.version += 1
                entry.safety_epoch += 1
            return SessionOperation("invalidated", view=_view(entry))

    @staticmethod
    def _stop(code: str, version: int | None = None, epoch: int | None = None,
              discarded: bool = False) -> CoordinatedResult:
        reviewed = review_outcome(DecisionOutcome("error", error_code=code))
        return CoordinatedResult("stopped", code=reviewed.code, checked_at_version=version,
                                 checked_safety_epoch=epoch, candidate_discarded=discarded,
                                 stage="coordinator", warnings=reviewed.warnings)

    @staticmethod
    def _safety(source: str, version: int, epoch: int, discarded: bool = False,
                warnings: tuple[str, ...] = ()) -> CoordinatedResult:
        reviewed = review_outcome(DecisionOutcome("safety_pending", safety_message=interactions.SAFETY_MESSAGE,
                                                   warnings=warnings))
        return CoordinatedResult(reviewed.status, safety_source=source, safety_message=reviewed.source.safety_message,
                                 checked_at_version=version, checked_safety_epoch=epoch, candidate_discarded=discarded,
                                 stage="coordinator", warnings=reviewed.warnings)

    @staticmethod
    def _observe_event_locked(entry: _Entry, event_snapshot: Any) -> str | None:
        """只在持有会话锁时聚合事件不适；供离线协调与存储协调共用。"""
        event_reported, source_valid = inspect_discomfort_signal(event_snapshot)
        prior_session_reported = entry.state.safety_reported
        if event_reported:
            report = event_snapshot["discomfort"]
            marked = interactions.report_discomfort(entry.state, now=report.get("reported_at"), clock_id=entry.state.clock_id)
            input_warnings = () if validate_training_event(event_snapshot).event is not None else ("input_unvalidated",)
            source_warnings = () if source_valid else ("discomfort_source_unverified",)
            safety_warnings = _merge_warnings(entry.safety_warnings, marked.warnings, input_warnings, source_warnings)
            if marked.session != entry.state or safety_warnings != entry.safety_warnings:
                entry.state = marked.session
                entry.safety_warnings = safety_warnings
                entry.version += 1
                if not prior_session_reported:
                    entry.safety_epoch += 1
        return _safety_source(event_reported, prior_session_reported)

    def decide(self, session_id: object, event: Any, model: Any, *, context: Any = None,
               do_not_disturb: object = None) -> CoordinatedResult:
        """所有正常业务返回经过 R3，最终状态只代表本进程检查瞬间。"""
        event_snapshot = deepcopy(event)
        context_snapshot = deepcopy(context)
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            if entry is None:
                return self._stop("state_missing")
            if not entry.active:
                return self._stop("session_invalidated", entry.version, entry.safety_epoch)
            if _explicit_mismatch(event_snapshot, entry):
                return self._stop("scope_mismatch")
            source = self._observe_event_locked(entry, event_snapshot)
            if source:
                return self._safety(source, entry.version, entry.safety_epoch, warnings=entry.safety_warnings)
            stopped = self._pre_model_check(session_id, entry)
            if stopped is not None:
                return stopped
            initial_epoch = entry.safety_epoch
        # 模型调用期间不持锁；其他执行流可立即记录不适。
        outcome = decide_simulated_event(event_snapshot, model, context_snapshot)
        reviewed = review_outcome(outcome, do_not_disturb=do_not_disturb)
        self._after_review(outcome, reviewed)
        with self._lock:
            entry = self._entries.get(session_id) if isinstance(session_id, str) else None
            discarded = outcome.candidate is not None
            if entry is None:
                return self._stop("state_missing", discarded=discarded)
            if not entry.active:
                return self._stop("session_invalidated", entry.version, entry.safety_epoch, discarded)
            if entry.state.safety_reported:
                return self._safety("session", entry.version, entry.safety_epoch, discarded, entry.safety_warnings)
            if entry.safety_epoch != initial_epoch:
                return self._stop("safety_state_changed", entry.version, entry.safety_epoch, discarded)
            if reviewed.status == "safety_pending":
                marked = interactions.report_discomfort(entry.state, now=None, clock_id=entry.state.clock_id)
                safety_warnings = _merge_warnings(entry.safety_warnings, marked.warnings, reviewed.warnings)
                if marked.session != entry.state or safety_warnings != entry.safety_warnings:
                    entry.state = marked.session
                    entry.safety_warnings = safety_warnings
                    entry.version += 1
                    entry.safety_epoch += 1
                return self._safety("event", entry.version, entry.safety_epoch, discarded, entry.safety_warnings)
            if reviewed.status != "reviewed":
                return CoordinatedResult("stopped", code=reviewed.code, checked_at_version=entry.version,
                                         checked_safety_epoch=entry.safety_epoch, candidate_discarded=discarded,
                                         stage=reviewed.stage, warnings=reviewed.warnings,
                                         rule_code=reviewed.rule_code, reference_check=reviewed.reference_check)
            return CoordinatedResult("reviewed", effective_action=reviewed.effective_action,
                                     candidate=outcome.candidate, checked_at_version=entry.version,
                                     checked_safety_epoch=entry.safety_epoch, warnings=reviewed.warnings,
                                     rule_code=reviewed.rule_code, reference_check=reviewed.reference_check)
