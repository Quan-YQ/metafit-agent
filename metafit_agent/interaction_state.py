"""R4 第二单元：确定性的模拟交互状态；不调用模型、网络或存储。"""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import wraps
import re


SAFETY_MESSAGE = "你报告了不适，请先停止当前动作；如症状明显、持续或加重，请联系医疗专业人员。"
TEMPLATES = {"pre_training": "今天身体有不适吗？", "followup": "需要继续讨论这个问题吗？", "safety_confirmation": "目前有不适吗？"}
FOLLOWUP_REASONS = frozenset({"user_requested", "user_reported_uncertainty"})
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z")


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    kind: str
    asked_at: datetime
    deadline: datetime
    trigger_reason: str | None = None
    trigger_reference: str | None = None
    status: str = "pending"
    presented: bool = False
    answer: str | None = None


@dataclass(frozen=True, slots=True)
class InteractionSession:
    session_id: str
    clock_id: str
    questions: tuple[Question, ...] = ()
    last_time: datetime | None = None
    training_started_at: datetime | None = None
    pretraining_closed: bool = False
    followup_used: bool = False
    pretraining_skipped: bool = False
    safety_reported: bool = False
    safety_reported_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class InteractionResult:
    session: InteractionSession
    status: str
    code: str | None = None
    question_text: str | None = None
    safety_message: str | None = None
    warnings: tuple[str, ...] = ()
    state_changed: bool = False


def _mark_state_changes(operation):
    """所有返回状态均可能包含过期等处理，统一标记并交还调用方。"""
    @wraps(operation)
    def wrapped(session, *args, **kwargs):
        result = operation(session, *args, **kwargs)
        return replace(result, state_changed=result.session != session)
    return wrapped


def create_session(session_id: str, clock_id: str) -> InteractionSession:
    """调用方提供模拟会话和时钟来源标识；标识不证明真实身份或同步。"""
    if not isinstance(session_id, str) or not _ID.fullmatch(session_id) or not isinstance(clock_id, str) or not _ID.fullmatch(clock_id):
        raise ValueError("invalid_session_or_clock")
    return InteractionSession(session_id, clock_id)


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not _TIME.fullmatch(value) or value.endswith("-00:00"):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _checked_time(session: InteractionSession, now: object, clock_id: object) -> datetime | None:
    time = _timestamp(now)
    if clock_id != session.clock_id or time is None or (session.last_time is not None and time < session.last_time):
        return None
    return time


def _advance(session: InteractionSession, time: datetime) -> InteractionSession:
    questions = tuple(replace(q, status="expired") if q.status == "pending" and
                      (time > q.deadline or (q.kind == "pre_training" and session.training_started_at is not None
                                             and time >= session.training_started_at)) else q
                      for q in session.questions)
    return replace(session, questions=questions, last_time=time)


def _safe(session: InteractionSession, warnings: tuple[str, ...] = ()) -> InteractionResult:
    return InteractionResult(session, "safety_pending", safety_message=SAFETY_MESSAGE, warnings=warnings)


def _index(session: InteractionSession, question_id: object) -> int | None:
    return next((i for i, q in enumerate(session.questions) if q.question_id == question_id), None)


def _put(session: InteractionSession, index: int, question: Question) -> InteractionSession:
    items = list(session.questions)
    items[index] = question
    return replace(session, questions=tuple(items))


@_mark_state_changes
def report_discomfort(session: InteractionSession, *, now: object = None, clock_id: object = None) -> InteractionResult:
    """安全报告不因坏时间被丢弃；普通操作不可解除标记。"""
    time = _checked_time(session, now, clock_id)
    questions = tuple(replace(q, status="cancelled") if q.status == "pending" else q for q in session.questions)
    updated = replace(session, safety_reported=True, questions=questions,
                      last_time=time if time is not None else session.last_time,
                      safety_reported_at=(session.safety_reported_at if session.safety_reported else time))
    return _safe(updated, () if time is not None else ("time_unvalidated",))


@_mark_state_changes
def mark_training_started(session: InteractionSession, *, now: object, clock_id: object) -> InteractionResult:
    """记录调用方观察到的训练开始；不控制训练本身。"""
    time = _checked_time(session, now, clock_id)
    if time is None:
        # 已收到开始信号但时间不可核实时，不能继续放出可能过期的训练前问题。
        questions = tuple(replace(q, status="cancelled") if q.kind == "pre_training" and q.status == "pending" else q
                          for q in session.questions)
        closed = replace(session, questions=questions, pretraining_closed=True)
        return _safe(closed, ("time_unvalidated",)) if closed.safety_reported else InteractionResult(closed, "closed", "time_unvalidated")
    if session.training_started_at is not None:
        return _safe(session) if session.safety_reported else InteractionResult(session, "training_started")
    updated = _advance(replace(session, training_started_at=time, pretraining_closed=True), time)
    return _safe(updated) if updated.safety_reported else InteractionResult(updated, "training_started")


@_mark_state_changes
def open_question(session: InteractionSession, *, question_id: object, kind: object, now: object,
                  deadline: object, clock_id: object, trigger_reason: object = None,
                  trigger_reference: object = None) -> InteractionResult:
    if session.safety_reported:
        return _safe(session)
    time = _checked_time(session, now, clock_id)
    end = _timestamp(deadline)
    if time is None or end is None:
        return InteractionResult(session, "error", "invalid_time")
    if not isinstance(question_id, str) or not _ID.fullmatch(question_id) or not isinstance(kind, str) or kind not in TEMPLATES:
        return InteractionResult(session, "error", "invalid_question")
    if kind == "followup":
        if not isinstance(trigger_reason, str) or trigger_reason not in FOLLOWUP_REASONS or not isinstance(trigger_reference, str) or not _ID.fullmatch(trigger_reference):
            return InteractionResult(session, "error", "invalid_trigger")
    elif trigger_reason is not None or trigger_reference is not None:
        return InteractionResult(session, "error", "invalid_trigger")
    current = _advance(session, time)
    existing_index = _index(current, question_id)
    if existing_index is not None:
        q = current.questions[existing_index]
        if (q.kind, q.deadline, q.trigger_reason, q.trigger_reference) != (kind, end, trigger_reason, trigger_reference):
            return InteractionResult(current, "error", "conflicting_question_id")
        return InteractionResult(current, "duplicate", "question_already_created")
    if end < time:
        return InteractionResult(current, "error", "invalid_time")
    if kind == "pre_training" and (current.pretraining_closed or current.training_started_at is not None or any(q.kind == kind for q in current.questions)):
        return InteractionResult(current, "blocked", "pretraining_unavailable")
    if kind == "followup" and (current.training_started_at is None or current.pretraining_skipped or current.followup_used):
        return InteractionResult(current, "blocked", "followup_unavailable")
    if kind == "safety_confirmation" and any(q.kind == kind and q.status != "cancelled" for q in current.questions):
        return InteractionResult(current, "blocked", "safety_confirmation_unavailable")
    if any(q.status == "pending" for q in current.questions):
        return InteractionResult(current, "blocked", "question_already_pending")
    q = Question(question_id, kind, time, end, trigger_reason, trigger_reference)
    updated = replace(current, questions=current.questions + (q,), followup_used=current.followup_used or kind == "followup")
    return InteractionResult(updated, "created")


@_mark_state_changes
def present_question(session: InteractionSession, *, question_id: object, now: object, clock_id: object) -> InteractionResult:
    """只在有效时间内返回一次问题文本；不表示已向 App 投递。"""
    if session.safety_reported:
        return _safe(session)
    time = _checked_time(session, now, clock_id)
    if time is None:
        return InteractionResult(session, "error", "invalid_time")
    current = _advance(session, time)
    index = _index(current, question_id)
    if index is None:
        return InteractionResult(current, "error", "unknown_question")
    q = current.questions[index]
    if q.status != "pending":
        return InteractionResult(current, q.status)
    if q.presented:
        return InteractionResult(current, "duplicate", "question_already_presented")
    current = _put(current, index, replace(q, presented=True))
    return InteractionResult(current, "presented", question_text=TEMPLATES[q.kind])


@_mark_state_changes
def answer_question(session: InteractionSession, *, question_id: object, answer: object,
                    now: object, clock_id: object) -> InteractionResult:
    if answer == "reported":
        return report_discomfort(session, now=now, clock_id=clock_id)
    if session.safety_reported:
        return _safe(session)
    time = _checked_time(session, now, clock_id)
    if time is None:
        return InteractionResult(session, "error", "invalid_time")
    current = _advance(session, time)
    index = _index(current, question_id)
    if index is None:
        return InteractionResult(current, "error", "unknown_question")
    q = current.questions[index]
    allowed = ("none", "unknown") if q.kind in {"pre_training", "safety_confirmation"} else ("yes", "no", "unknown")
    if answer not in allowed:
        return InteractionResult(current, "error", "invalid_answer")
    if q.status == "answered":
        return InteractionResult(current, "answered") if q.answer == answer else InteractionResult(current, "error", "conflicting_answer")
    if q.status == "skipped":
        return InteractionResult(current, "error", "question_skipped")
    if q.status != "pending":
        return InteractionResult(current, q.status)
    if not q.presented:
        return InteractionResult(current, "blocked", "question_not_presented")
    return InteractionResult(_put(current, index, replace(q, status="answered", answer=answer)), "answered")


@_mark_state_changes
def expire_safety_confirmation(session: InteractionSession, *, question_id: object, now: object, clock_id: object) -> InteractionResult:
    """只过期指定的安全确认，不将查询误记为展示或推进其他问题。"""
    if session.safety_reported:
        return _safe(session)
    time = _checked_time(session, now, clock_id)
    if time is None:
        return InteractionResult(session, "error", "invalid_time")
    index = _index(session, question_id)
    if index is None:
        return InteractionResult(session, "error", "unknown_question")
    question = session.questions[index]
    if question.kind != "safety_confirmation":
        return InteractionResult(session, "error", "invalid_question")
    if question.status == "pending" and time > question.deadline:
        updated = _put(replace(session, last_time=time), index, replace(question, status="expired"))
        return InteractionResult(updated, "expired")
    return InteractionResult(session, question.status)


@_mark_state_changes
def skip_question(session: InteractionSession, *, question_id: object, now: object, clock_id: object) -> InteractionResult:
    if session.safety_reported:
        return _safe(session)
    time = _checked_time(session, now, clock_id)
    if time is None:
        return InteractionResult(session, "error", "invalid_time")
    current = _advance(session, time)
    index = _index(current, question_id)
    if index is None:
        return InteractionResult(current, "error", "unknown_question")
    q = current.questions[index]
    if q.status == "answered":
        return InteractionResult(current, "error", "already_answered")
    if q.status != "pending":
        return InteractionResult(current, q.status)
    if not q.presented and q.kind != "safety_confirmation":
        return InteractionResult(current, "blocked", "question_not_presented")
    updated = _put(current, index, replace(q, status="skipped"))
    if q.kind == "pre_training":
        updated = replace(updated, pretraining_skipped=True)
    return InteractionResult(updated, "skipped")
