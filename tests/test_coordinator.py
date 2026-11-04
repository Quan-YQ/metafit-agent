"""单进程模拟协调、安全优先与并发接缝；无真实模型调用。"""
from copy import deepcopy
import json
from pathlib import Path
from threading import Thread

import pytest

import metafit_agent.coordinator as module
from metafit_agent.coordinator import SimulationCoordinator
from metafit_agent.model_response import ModelResponse


TIME = "2026-11-02T18:30:00+08:00"
CLOCK = "event_clock"


@pytest.fixture
def event():
    path = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["discomfort"] = {"status": "none", "source": "self_report"}
    return data


@pytest.fixture
def setup(event):
    coordinator = SimulationCoordinator()
    assert coordinator.register(user_id=event["user_id"], session_id=event["session_id"], clock_id=CLOCK).status == "registered"
    return coordinator, event["session_id"]


class Model:
    def __init__(self, during=None):
        self.calls = 0
        self.during = during

    def complete(self, facts):
        self.calls += 1
        if self.during:
            self.during()
        content = json.dumps({"suggested_action": "notify", "message": "本组完成了10次。",
                              "reason": "模拟自报状态仅供参考。", "evidence": ["fact_3"]}, ensure_ascii=False)
        return ModelResponse(content, "mock", "mock", None, None, 0, "stop")


def report(coordinator, session_id, **kwargs):
    return coordinator.apply(session_id, "report_discomfort", now=TIME, clock_id=CLOCK, **kwargs)


def run_in_other_thread(fn):
    outcome = []
    def invoke():
        try:
            outcome.append(fn())
        except BaseException as exc:
            outcome.append(exc)
    thread = Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(2)
    assert not thread.is_alive(), "模型调用期间不应锁住状态更新"
    assert outcome and not isinstance(outcome[0], BaseException)
    return outcome[0]


def test_normal_path_goes_through_r3(setup, event, monkeypatch):
    coordinator, session_id = setup
    seen = []
    real = module.review_outcome
    def spy(outcome, **kwargs):
        seen.append(outcome.status)
        return real(outcome, **kwargs)
    monkeypatch.setattr(module, "review_outcome", spy)
    model = Model()
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert result.status == "reviewed" and result.effective_action == "notify"
    assert result.candidate.suggested_action == "notify" and model.calls == 1
    assert seen == ["candidate"] and result.checked_at_version == 0 and result.checked_safety_epoch == 0


@pytest.mark.parametrize("scenario,expected", [("event", "event"), ("session", "session"), ("both", "both")])
def test_safety_source_is_aggregated_and_model_not_called(setup, event, scenario, expected):
    coordinator, session_id = setup
    if scenario in ("session", "both"):
        assert report(coordinator, session_id).status == "safety_pending"
    if scenario in ("event", "both"):
        event["discomfort"] = {"status": "reported", "source": "self_report", "reported_at": TIME}
    model = Model()
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.safety_source == expected
    assert model.calls == 0 and result.candidate is None and result.effective_action is None
    assert coordinator.view(session_id).safety_reported
    if scenario == "event":
        next_result = coordinator.decide(session_id, {**event, "discomfort": {"status": "none", "source": "self_report"}}, model)
        assert next_result.safety_source == "session" and model.calls == 0


def test_invalid_event_with_session_discomfort_still_safety(setup, event):
    coordinator, session_id = setup
    report(coordinator, session_id)
    event.pop("event_id")
    model = Model()
    result = coordinator.decide(session_id, event, model)
    assert result.status == "safety_pending" and result.safety_source == "session" and model.calls == 0


def test_event_reported_even_if_other_event_fields_invalid_latches_safety(setup, event):
    coordinator, session_id = setup
    event.pop("event_id")
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    model = Model()
    assert coordinator.decide(session_id, event, model).safety_source == "event"
    assert coordinator.view(session_id).safety_reported and model.calls == 0


def test_whitespace_in_safety_status_cannot_bypass_latch(setup, event):
    coordinator, session_id = setup
    event["discomfort"] = {"status": " reported ", "source": " self_report "}
    result = coordinator.decide(session_id, event, Model())
    assert result.status == "safety_pending" and coordinator.view(session_id).safety_reported


@pytest.mark.parametrize("source", ["missing", None, "sensor", 123])
def test_reported_status_with_unverified_source_preserves_safety_and_warning(setup, event, source):
    coordinator, session_id = setup
    report = {"status": "reported"}
    if source != "missing":
        report["source"] = source
    event["discomfort"] = report
    model = Model()
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.safety_source == "event"
    assert result.stage == "coordinator" and result.safety_message and model.calls == 0
    assert "discomfort_source_unverified" in result.warnings and "input_unvalidated" in result.warnings
    assert "time_unvalidated" in result.warnings and coordinator.view(session_id).safety_reported
    next_event = deepcopy(event)
    next_event["discomfort"] = {"status": "none", "source": "self_report"}
    followup = coordinator.decide(session_id, next_event, model, do_not_disturb=False)
    assert followup.status == "safety_pending" and followup.safety_source == "session"
    assert "discomfort_source_unverified" in followup.warnings and model.calls == 0


def test_invalid_source_report_still_respects_scope_mismatch(setup, event):
    coordinator, session_id = setup
    event["session_id"] = "other"
    event["discomfort"] = {"status": "reported"}
    result = coordinator.decide(session_id, event, Model())
    assert (result.status, result.code, result.stage, result.safety_message) == ("stopped", "scope_mismatch", "coordinator", None)
    assert not coordinator.view(session_id).safety_reported


def test_coordinator_result_distinguishes_stages_and_retains_warnings(setup, event):
    coordinator, session_id = setup
    assert coordinator.decide("missing", event, Model()).stage == "coordinator"
    invalid = deepcopy(event)
    invalid.pop("event_id")
    r2_error = coordinator.decide(session_id, invalid, Model(), do_not_disturb=False)
    assert (r2_error.status, r2_error.code, r2_error.stage) == ("stopped", "invalid_event", "r2")
    r3_error = coordinator.decide(session_id, event, Model(), do_not_disturb=None)
    assert (r3_error.status, r3_error.code, r3_error.stage) == ("stopped", "dnd_state_missing", "r3")
    unmapped = deepcopy(event)
    unmapped["exercise_id"] = "unknown_exercise"
    reviewed = coordinator.decide(session_id, unmapped, Model(), do_not_disturb=False)
    assert reviewed.status == "reviewed" and reviewed.stage is None and "exercise_unmapped" in reviewed.warnings
    stopped = coordinator.decide(session_id, unmapped, Model(), do_not_disturb=None)
    assert stopped.stage == "r3" and "exercise_unmapped" in stopped.warnings


def test_session_safety_warning_survives_later_decisions(setup, event):
    coordinator, session_id = setup
    first = coordinator.apply(session_id, "report_discomfort", now=None, clock_id=CLOCK)
    assert "time_unvalidated" in first.warnings
    result = coordinator.decide(session_id, event, Model(), do_not_disturb=False)
    assert result.status == "safety_pending" and result.stage == "coordinator"
    assert "time_unvalidated" in result.warnings
    operation = coordinator.apply(session_id, "skip_question", question_id="x", now=TIME, clock_id=CLOCK)
    assert operation.status == "safety_pending" and "time_unvalidated" in operation.warnings


def test_concurrent_safety_time_warning_survives_final_recheck(setup, event):
    coordinator, session_id = setup
    model = Model(lambda: run_in_other_thread(lambda: coordinator.apply(session_id, "report_discomfort", now=None, clock_id=CLOCK)))
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert result.status == "safety_pending" and result.candidate_discarded
    assert result.stage == "coordinator" and "time_unvalidated" in result.warnings


@pytest.mark.parametrize("field", ["session_id", "user_id"])
def test_explicit_scope_mismatch_wins_over_existing_and_event_safety(setup, event, field):
    coordinator, session_id = setup
    report(coordinator, session_id)
    event[field] = "other"
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    model = Model()
    result = coordinator.decide(session_id, event, model)
    assert (result.status, result.code, result.safety_message, model.calls) == ("stopped", "scope_mismatch", None, 0)


def test_missing_and_invalidated_state_stop_before_model(event):
    coordinator = SimulationCoordinator()
    model = Model()
    assert coordinator.decide("missing", event, model).code == "state_missing"
    coordinator.register(user_id=event["user_id"], session_id=event["session_id"], clock_id=CLOCK)
    coordinator.invalidate(event["session_id"])
    result = coordinator.decide(event["session_id"], event, model)
    assert result.code == "session_invalidated" and result.candidate is None and model.calls == 0


def test_state_changes_to_safety_during_model_call_discards_candidate(setup, event):
    coordinator, session_id = setup
    model = Model(lambda: run_in_other_thread(lambda: report(coordinator, session_id)))
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert model.calls == 1 and result.status == "safety_pending" and result.safety_source == "session"
    assert result.candidate_discarded and result.candidate is None and result.effective_action is None
    assert result.checked_at_version == 1 and result.checked_safety_epoch == 1


def test_event_report_during_other_model_call_latches_and_discards(setup, event):
    coordinator, session_id = setup
    safety_event = {**event, "discomfort": {"status": "reported", "source": "self_report"}}
    nested_model = Model()
    model = Model(lambda: run_in_other_thread(lambda: coordinator.decide(session_id, safety_event, nested_model)))
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert nested_model.calls == 0 and coordinator.view(session_id).safety_reported
    assert result.status == "safety_pending" and result.candidate_discarded and result.safety_source == "session"


def test_safety_report_after_r3_but_before_final_check_discards_candidate(setup, event, monkeypatch):
    coordinator, session_id = setup
    real = module.review_outcome
    seen = []
    def review_then_report(outcome, **kwargs):
        result = real(outcome, **kwargs)
        seen.append(outcome.status)
        if outcome.status == "candidate":
            run_in_other_thread(lambda: report(coordinator, session_id))
        return result
    monkeypatch.setattr(module, "review_outcome", review_then_report)
    result = coordinator.decide(session_id, event, Model(), do_not_disturb=False)
    assert seen == ["candidate", "safety_pending"]
    assert result.status == "safety_pending" and result.candidate_discarded and result.candidate is None


def test_checked_version_does_not_prove_future_delivery_safety(setup, event):
    coordinator, session_id = setup
    result = coordinator.decide(session_id, event, Model(), do_not_disturb=False)
    assert result.status == "reviewed" and result.checked_at_version == 0
    assert report(coordinator, session_id).view.version == 1
    assert coordinator.view(session_id).safety_reported


def test_ordinary_state_change_during_model_call_does_not_discard(setup, event):
    coordinator, session_id = setup
    model = Model(lambda: run_in_other_thread(lambda: coordinator.apply(session_id, "mark_training_started", now=TIME, clock_id=CLOCK)))
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert result.status == "reviewed" and result.candidate is not None and not result.candidate_discarded
    assert result.checked_at_version == 1 and result.checked_safety_epoch == 0


@pytest.mark.parametrize("change", ["answer", "expiry"])
def test_answer_or_expiry_during_model_call_only_changes_version(setup, event, change):
    coordinator, session_id = setup
    opened = coordinator.apply(session_id, "open_question", question_id="pre", kind="pre_training",
                               now=TIME, deadline=TIME, clock_id=CLOCK)
    assert opened.status == "created"
    if change == "answer":
        assert coordinator.apply(session_id, "present_question", question_id="pre", now=TIME, clock_id=CLOCK).status == "presented"
        action = lambda: coordinator.apply(session_id, "answer_question", question_id="pre", answer="none", now=TIME, clock_id=CLOCK)
    else:
        action = lambda: coordinator.apply(session_id, "present_question", question_id="pre",
                                           now="2026-11-02T18:30:01+08:00", clock_id=CLOCK)
    before = coordinator.view(session_id)
    result = coordinator.decide(session_id, event, Model(lambda: run_in_other_thread(action)), do_not_disturb=False)
    assert result.status == "reviewed" and result.checked_at_version > before.version
    assert result.checked_safety_epoch == 0 and not result.candidate_discarded


def test_invalidation_during_model_call_discards_candidate(setup, event):
    coordinator, session_id = setup
    model = Model(lambda: run_in_other_thread(lambda: coordinator.invalidate(session_id)))
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert result.status == "stopped" and result.code == "session_invalidated"
    assert result.candidate_discarded and result.candidate is None


def test_old_version_rejected_but_safety_report_not_lost(setup):
    coordinator, session_id = setup
    assert coordinator.apply(session_id, "mark_training_started", now=TIME, clock_id=CLOCK, expected_version=0).view.version == 1
    stale = coordinator.apply(session_id, "skip_question", question_id="x", now=TIME, clock_id=CLOCK, expected_version=0)
    assert stale.code == "version_conflict" and stale.view.version == 1 and not stale.view.safety_reported
    safety = report(coordinator, session_id, expected_version=0)
    assert safety.status == "safety_pending" and safety.view.safety_reported
    assert "stale_version_ignored_for_safety" in safety.warnings and safety.view.safety_epoch == 1
    blocked = coordinator.apply(session_id, "skip_question", question_id="x", now=TIME, clock_id=CLOCK, expected_version=0)
    assert blocked.status == "safety_pending" and blocked.view.safety_reported


@pytest.mark.parametrize("operation,arguments", [
    ("answer_question", {"answer": "reported"}),
    ("report_discomfort", {"unexpected": "ignored"}),
])
def test_reported_discomfort_not_lost_if_ordinary_arguments_missing_or_extra(setup, operation, arguments):
    coordinator, session_id = setup
    result = coordinator.apply(session_id, operation, **arguments)
    assert result.status == "safety_pending" and result.view.safety_reported
    assert "time_unvalidated" in result.warnings
    if "unexpected" in arguments:
        assert "extra_safety_arguments" in result.warnings


def test_first_report_time_is_never_guessed_or_replaced(setup):
    coordinator, session_id = setup
    first = coordinator.apply(session_id, "report_discomfort", now=None, clock_id=CLOCK)
    assert first.view.safety_reported and first.view.safety_reported_at is None
    later = report(coordinator, session_id)
    assert later.view.safety_reported_at is None and later.view.safety_epoch == 1


def test_valid_first_report_time_is_recorded_and_not_rewritten(setup):
    coordinator, session_id = setup
    first = report(coordinator, session_id)
    assert first.view.safety_reported_at.isoformat() == "2026-11-02T10:30:00+00:00"
    later = coordinator.apply(session_id, "report_discomfort", now="2026-11-02T18:31:00+08:00", clock_id=CLOCK)
    assert later.view.safety_reported_at == first.view.safety_reported_at and later.view.safety_epoch == 1


def test_business_stops_and_safety_always_reach_r3(setup, event, monkeypatch):
    coordinator, session_id = setup
    seen = []
    real = module.review_outcome
    def spy(outcome, **kwargs):
        seen.append(outcome.status)
        return real(outcome, **kwargs)
    monkeypatch.setattr(module, "review_outcome", spy)
    assert coordinator.decide("missing", event, Model()).status == "stopped"
    mismatch = {**event, "session_id": "other"}
    assert coordinator.decide(session_id, mismatch, Model()).code == "scope_mismatch"
    report(coordinator, session_id)
    assert coordinator.decide(session_id, event, Model()).status == "safety_pending"
    assert seen == ["error", "error", "safety_pending"]


def test_r3_rejection_does_not_expose_candidate(setup, event):
    coordinator, session_id = setup
    model = Model()
    result = coordinator.decide(session_id, event, model, do_not_disturb=None)
    assert result.status == "stopped" and result.code == "dnd_state_missing"
    assert result.candidate_discarded and result.candidate is None and result.effective_action is None


def test_instances_are_isolated_and_duplicate_registration_cannot_reset_safety(setup, event):
    coordinator, session_id = setup
    report(coordinator, session_id)
    assert coordinator.register(user_id=event["user_id"], session_id=session_id, clock_id=CLOCK).code == "session_exists"
    assert coordinator.view(session_id).safety_reported
    assert SimulationCoordinator().view(session_id) is None


def test_model_mutating_callers_event_after_snapshot_does_not_change_current_decision(setup, event):
    coordinator, session_id = setup
    original = deepcopy(event)
    model = Model(lambda: event.update(user_id="another_user", session_id="different_session"))
    result = coordinator.decide(session_id, event, model, do_not_disturb=False)
    assert result.status == "reviewed" and original["session_id"] == session_id
    assert event["session_id"] == "different_session"


def test_mock_flow_does_not_open_network_or_import_storage(setup, event, monkeypatch):
    import ast
    import metafit_agent.deepseek_client as provider
    monkeypatch.setattr(provider.request, "build_opener", lambda *_: pytest.fail("network not allowed"))
    coordinator, session_id = setup
    assert coordinator.decide(session_id, event, Model(), do_not_disturb=False).status == "reviewed"
    source = (Path(__file__).resolve().parents[1] / "metafit_agent" / "coordinator.py").read_text(encoding="utf-8")
    imports = [n.module if isinstance(n, ast.ImportFrom) else n.names[0].name
               for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(name and any(blocked in name for blocked in ("sqlite", "database", "mysql")) for name in imports)
