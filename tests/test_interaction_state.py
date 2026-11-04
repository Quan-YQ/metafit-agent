"""R4 第二单元：模拟交互状态、时间与安全边界。"""
import ast
from pathlib import Path

import pytest

from metafit_agent.interaction_state import (
    SAFETY_MESSAGE, TEMPLATES, answer_question, create_session, mark_training_started,
    open_question, present_question, report_discomfort, skip_question,
)


CLOCK = "event_clock"
T0 = "2026-11-02T10:00:00+08:00"
T1 = "2026-11-02T10:00:10+08:00"
T2 = "2026-11-02T10:01:00+08:00"
T3 = "2026-11-02T10:01:01+08:00"


@pytest.fixture
def session():
    return create_session("session_1", CLOCK)


def pre(session, question_id="pre", now=T0, deadline=T2):
    return open_question(session, question_id=question_id, kind="pre_training", now=now,
                         deadline=deadline, clock_id=CLOCK)


def follow(session, question_id="follow", now=T1, deadline=T2, reason="user_requested", reference="trigger_1"):
    return open_question(session, question_id=question_id, kind="followup", now=now, deadline=deadline,
                         clock_id=CLOCK, trigger_reason=reason, trigger_reference=reference)


def present(session, question_id="pre", now=T0):
    return present_question(session, question_id=question_id, now=now, clock_id=CLOCK)


def test_default_silent_and_single_short_template():
    session = create_session("session_1", CLOCK)
    assert session.questions == () and not session.followup_used
    assert all(0 < len(text) <= 60 and text.count("？") == 1 for text in TEMPLATES.values())
    result = pre(session)
    assert result.status == "created" and result.question_text is None
    shown = present(result.session)
    assert shown.status == "presented" and shown.question_text == TEMPLATES["pre_training"]
    again = present(shown.session)
    assert again.status == "duplicate" and again.question_text is None


def test_skip_pretraining_is_remembered_and_prevents_followup(session):
    state = present(pre(session).session).session
    skipped = skip_question(state, question_id="pre", now=T0, clock_id=CLOCK)
    repeated = skip_question(skipped.session, question_id="pre", now=T0, clock_id=CLOCK)
    assert skipped.status == repeated.status == "skipped" and repeated.session == skipped.session
    started = mark_training_started(repeated.session, now=T1, clock_id=CLOCK)
    result = follow(started.session)
    assert result.status == "blocked" and result.code == "followup_unavailable"


def test_safety_cancels_all_pending_and_cannot_be_reversed(session):
    state = present(pre(session).session).session
    safety = report_discomfort(state, now=None, clock_id=None)
    assert safety.status == "safety_pending" and safety.safety_message == SAFETY_MESSAGE
    assert safety.warnings == ("time_unvalidated",) and safety.session.questions[0].status == "cancelled"
    assert safety.session.safety_reported
    result = answer_question(safety.session, question_id="pre", answer="none", now=T1, clock_id=CLOCK)
    assert result.status == "safety_pending" and result.session.safety_reported and result.question_text is None
    assert follow(result.session).status == "safety_pending"


def test_reported_answer_takes_priority_over_invalid_time_and_unknown_question(session):
    result = answer_question(session, question_id="unknown", answer="reported", now="bad", clock_id="bad")
    assert result.status == "safety_pending" and result.session.safety_reported
    assert result.warnings == ("time_unvalidated",)


@pytest.mark.parametrize("time,expected", [(T2, "presented"), (T3, "expired")])
def test_deadline_is_inclusive_and_expiry_has_no_text(session, time, expected):
    state = pre(session).session
    result = present(state, now=time)
    assert result.status == expected
    if expected == "expired":
        assert result.question_text is None and result.session.questions[0].status == "expired"
        retry = present(result.session, now=time)
        assert retry.status == "expired" and retry.question_text is None


def test_training_start_expires_pre_question_and_prevents_late_pre_question(session):
    state = pre(session).session
    started = mark_training_started(state, now=T1, clock_id=CLOCK)
    assert started.session.questions[0].status == "expired"
    assert present(started.session, now=T1).status == "expired"
    late = pre(started.session, question_id="new_pre", now=T1)
    assert late.status == "blocked" and late.code == "pretraining_unavailable"


@pytest.mark.parametrize("started_at,clock", [(T0, CLOCK), (None, CLOCK), (T1, "other")])
def test_invalid_or_late_start_signal_closes_pretraining_without_display(session, started_at, clock):
    state = pre(session, now=T1).session
    result = mark_training_started(state, now=started_at, clock_id=clock)
    assert result.status == "closed" and result.code == "time_unvalidated" and result.state_changed
    assert result.session.pretraining_closed and result.session.questions[0].status == "cancelled"
    assert present(result.session, now=T1).status == "cancelled"
    assert pre(result.session, question_id="new", now=T1).code == "pretraining_unavailable"


def test_followup_requires_trigger_and_expiry_consumes_allowance(session):
    state = mark_training_started(session, now=T0, clock_id=CLOCK).session
    assert follow(state, reason=None).code == "invalid_trigger"
    assert follow(state, reason="timer").code == "invalid_trigger"
    opened = follow(state)
    assert opened.status == "created" and opened.session.followup_used
    expired = present(opened.session, question_id="follow", now=T3)
    assert expired.status == "expired" and expired.question_text is None
    new = follow(expired.session, question_id="second", now=T3, deadline="2026-11-02T10:02:00+08:00")
    assert new.code == "followup_unavailable"


def test_idempotence_and_conflicting_replay(session):
    opened = pre(session)
    duplicate = pre(opened.session)
    assert duplicate.status == "duplicate" and duplicate.question_text is None and len(duplicate.session.questions) == 1
    conflict = pre(duplicate.session, deadline="2026-11-02T10:02:00+08:00")
    assert conflict.code == "conflicting_question_id"
    state = present(duplicate.session).session
    answered = answer_question(state, question_id="pre", answer="none", now=T1, clock_id=CLOCK)
    again = answer_question(answered.session, question_id="pre", answer="none", now=T1, clock_id=CLOCK)
    assert answered.status == again.status == "answered" and again.session == answered.session
    conflict = answer_question(again.session, question_id="pre", answer="unknown", now=T1, clock_id=CLOCK)
    assert conflict.status == "error" and conflict.code == "conflicting_answer"
    assert skip_question(again.session, question_id="pre", now=T1, clock_id=CLOCK).code == "already_answered"


@pytest.mark.parametrize("now,clock", [(None, CLOCK), ("bad", CLOCK), ("2026-11-02T10:00:00", CLOCK),
                                       ("2026-11-02T10:00:00-00:00", CLOCK), (T0, "other")])
def test_missing_invalid_time_or_other_clock_never_shows_question(session, now, clock):
    state = pre(session).session
    result = present_question(state, question_id="pre", now=now, clock_id=clock)
    assert result.status == "error" and result.code == "invalid_time" and result.question_text is None
    assert result.session == state


def test_time_cannot_go_backwards_and_deadline_must_be_valid(session):
    state = pre(session, now=T1).session
    assert present(state, now=T0).code == "invalid_time"
    bad = pre(session, deadline=None)
    assert bad.status == "error" and bad.code == "invalid_time" and not bad.session.questions


@pytest.mark.parametrize("operation", ["present", "answer", "skip", "open"])
def test_error_can_still_advance_expiration_and_must_return_latest_state(session, operation):
    state = pre(session).session
    if operation == "present":
        result = present_question(state, question_id="missing", now=T3, clock_id=CLOCK)
    elif operation == "answer":
        result = answer_question(state, question_id="missing", answer="none", now=T3, clock_id=CLOCK)
    elif operation == "skip":
        result = skip_question(state, question_id="missing", now=T3, clock_id=CLOCK)
    else:
        result = open_question(state, question_id="pre", kind="pre_training", now=T3,
                               deadline="2026-11-02T10:02:00+08:00", clock_id=CLOCK)
    assert result.status == "error" and result.state_changed
    assert result.session.questions[0].status == "expired"
    assert present(result.session, now=T3).status == "expired"


def test_state_changed_is_false_for_unchanged_error_and_idempotent_replay(session):
    state = pre(session).session
    invalid = present_question(state, question_id="pre", now=None, clock_id=CLOCK)
    assert invalid.status == "error" and not invalid.state_changed and invalid.session == state
    shown = present(state)
    assert shown.state_changed
    repeated = present(shown.session)
    assert repeated.status == "duplicate" and not repeated.state_changed


def test_only_one_question_pending_and_followup_independent_of_pre_counter(session):
    state = pre(session).session
    duplicate_kind = pre(state, question_id="different")
    assert duplicate_kind.code == "pretraining_unavailable"
    shown = present(state).session
    answered = answer_question(shown, question_id="pre", answer="none", now=T0, clock_id=CLOCK)
    started = mark_training_started(answered.session, now=T1, clock_id=CLOCK)
    opened = follow(started.session)
    assert opened.status == "created" and opened.session.followup_used
    assert len(opened.session.questions) == 2


def test_expired_and_cancelled_cannot_be_revived(session):
    expired = present(pre(session).session, now=T3)
    replay = open_question(expired.session, question_id="pre", kind="pre_training", now=T3, deadline=T2, clock_id=CLOCK)
    assert replay.status == "duplicate" and replay.session.questions[0].status == "expired"
    result = answer_question(replay.session, question_id="pre", answer="none", now=T3, clock_id=CLOCK)
    assert result.status == "expired" and result.question_text is None
    cancelled = report_discomfort(pre(session).session, now=T0, clock_id=CLOCK)
    assert present(cancelled.session).status == "safety_pending"
    assert cancelled.session.questions[0].status == "cancelled"


def test_invalid_input_does_not_crash(session):
    assert open_question(session, question_id="q", kind=[], now=T0, deadline=T2, clock_id=CLOCK).code == "invalid_question"
    state = mark_training_started(session, now=T0, clock_id=CLOCK).session
    assert follow(state, reason=[]).code == "invalid_trigger"


def test_deterministic_no_provider_or_database_imports(session):
    assert pre(session) == pre(session)
    path = Path(__file__).resolve().parents[1] / "metafit_agent" / "interaction_state.py"
    imports = [n.module if isinstance(n, ast.ImportFrom) else n.names[0].name
               for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(name and any(blocked in name for blocked in ("deepseek", "sqlite", "database")) for name in imports)


def test_safety_copy_matches_existing_r2_wording():
    from metafit_agent.decision import SAFETY_MESSAGE as R2_SAFETY_MESSAGE
    assert SAFETY_MESSAGE == R2_SAFETY_MESSAGE
