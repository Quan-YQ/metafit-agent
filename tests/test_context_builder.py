"""R4 模拟输入、缺失语义、历史边界和 R2/R3 接缝。"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from metafit_agent.context_builder import ContextError, MetricConcept, MetricDefinition, build_context, metric_definition_is_usable
from metafit_agent.decision import decide_simulated_event
from metafit_agent.decision_review import review_outcome
from metafit_agent.model_response import ModelResponse
from metafit_agent.training_events import validate_training_event


@pytest.fixture
def event():
    path = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    return json.loads(path.read_text(encoding="utf-8"))


def prior(event, event_id, timestamp, **changes):
    item = deepcopy(event)
    item.update(event_id=event_id, set_id="set_" + event_id, timestamp=timestamp, **changes)
    return item


def context(**items):
    return {"source": "simulation", "context_id": "ctx", **items}


def build(event, ctx=None):
    validated = validate_training_event(event)
    assert validated.event is not None
    return build_context(validated.event, ctx)


def test_all_five_context_kinds_and_missing_information(event):
    history = [prior(event, "old_1", "2026-11-01T12:00:00+08:00", session_id="old_session")]
    ctx = context(profile={"goal": "general_fitness"}, history=history,
                  long_term_profile={"weekly_session_goal": 3}, daily_state={"energy_level": "medium"})
    result = build(event, ctx)
    assert {f.kind for f in result.facts} == {"profile", "training_history", "long_term_profile", "daily_state", "current_event"}
    assert result.missing_categories == () and result.scope.allowed_history == (("old_1", "old_session"),)
    empty = build(event)
    assert set(empty.missing_categories) == {"profile", "training_history", "long_term_profile", "daily_state"}
    assert all(f.name not in {"goal", "history_completed_reps", "weekly_session_goal", "energy_level"} for f in empty.facts)


def test_legacy_context_remains_compatible_and_zero_is_not_missing(event):
    result = build(event, context(goal="general_fitness", previous_completed_reps=0))
    fact = next(f for f in result.facts if f.name == "previous_completed_reps")
    assert fact.value == 0 and fact.kind == "training_history" and fact.origin == "simulated_context:ctx"


def test_missing_discomfort_preserves_r2_unknown_safety_contract(event):
    event.pop("discomfort")
    result = build(event)
    assert next(f.value for f in result.facts if f.name == "discomfort_status") == "unknown"


@pytest.mark.parametrize("reported_at,expected", [
    (None, "unknown"), ("2026-11-02T18:00:00+08:00", "before_or_at_event"),
    ("2026-11-02T19:00:00+08:00", "after_event"),
])
def test_discomfort_time_relation_is_chronology_not_freshness(event, reported_at, expected):
    event["discomfort"] = {"status": "none", "source": "self_report", "reported_at": reported_at}
    result = build(event)
    assert result.discomfort_time_relation == expected
    assert all(f.name != "reported_at" and f.name != "discomfort_time_relation" for f in result.facts)


def test_selection_is_same_user_same_exercise_within_seven_days_and_three_most_recent(event):
    entries = [prior(event, "h1", "2026-11-01T12:00:00+08:00", session_id="s1"),
               prior(event, "h2", "2026-10-31T12:00:00+08:00", session_id="s2"),
               prior(event, "h3", "2026-10-30T12:00:00+08:00"),
               prior(event, "h4", "2026-10-29T12:00:00+08:00"),
               prior(event, "old", "2026-10-01T12:00:00+08:00"),
               prior(event, "future", "2026-11-03T12:00:00+08:00"),
               prior(event, "other_move", "2026-11-01T12:00:00+08:00", exercise_id="squat")]
    result = build(event, context(history=entries))
    assert result.history_input_count == 7 and result.history_selected_count == 3
    assert result.scope.allowed_history == (("h1", "s1"), ("h2", "s2"), ("h3", event["session_id"]))
    assert [f.event_id for f in result.facts if f.origin == "history"] == ["h1", "h2", "h3"]


@pytest.mark.parametrize("history_kind", ["cross_user", "duplicate", "current", "not_simulated", "too_many"])
def test_invalid_or_cross_user_history_is_rejected(event, history_kind):
    item = prior(event, "old_1", "2026-11-01T12:00:00+08:00")
    if history_kind == "cross_user": item["user_id"] = "another_user"
    if history_kind == "current": item["event_id"] = event["event_id"]
    if history_kind == "not_simulated": item["source"] = "real"
    history = [item, item] if history_kind == "duplicate" else [item] * (51 if history_kind == "too_many" else 1)
    with pytest.raises(ContextError, match="invalid_simulated_context"):
        build(event, context(history=history))


def test_oversized_history_repetition_count_is_rejected(event):
    item = prior(event, "old_1", "2026-11-01T12:00:00+08:00", completed_reps=2 ** 129)
    with pytest.raises(ContextError, match="input_value_too_large"):
        build(event, context(history=[item]))


@pytest.mark.parametrize("ctx", [
    context(profile={"goal": "general_fitness"}, goal="general_fitness"),
    context(previous_completed_reps=1, history=[]), context(profile=None),
    context(daily_state={"discomfort": "none"}), context(daily_state={"energy_level": None}),
    context(long_term_profile={"weekly_session_goal": True}), context(long_term_profile={"weekly_session_goal": 15}),
])
def test_invalid_context_does_not_call_model(event, ctx):
    class NeverModel:
        def complete(self, facts): pytest.fail("invalid context must not reach model")
    assert decide_simulated_event(event, NeverModel(), ctx).error_code == "invalid_simulated_context"


def test_metric_definitions_are_inert_and_values_not_in_fact_list(event):
    concept = MetricConcept("range", "range", "未核实", "模拟样例")
    definition = MetricDefinition("range", concept.concept, "squat", "%", "未知", "模拟样例", "未知", "未校准")
    assert concept.verification == "unverified" and not metric_definition_is_usable(definition)
    assert not metric_definition_is_usable(replace(definition, verification="verified"))
    result = build(event)
    assert -18 not in [f.value for f in result.facts]
    assert next(f.value for f in result.facts if f.name == "uninterpreted_metric_count") == 1


def test_digest_and_selection_are_deterministic_and_change_with_fact(event):
    ctx = context(profile={"goal": "general_fitness"})
    first, second = build(event, ctx), build(event, deepcopy(ctx))
    assert first == second and len(first.fact_list_digest) == 64
    changed = deepcopy(event)
    changed["completed_reps"] = 11
    assert first.fact_list_digest != build(changed, ctx).fact_list_digest


def test_history_fact_passes_r3_and_mismatches_fail_closed(event, monkeypatch):
    monkeypatch.setattr("metafit_agent.deepseek_client.request.build_opener", lambda *_: pytest.fail("network"))
    history = [prior(event, "old_1", "2026-11-01T12:00:00+08:00", session_id="old_session")]
    class Model:
        def complete(self, facts):
            historical = next(f for f in facts if f.origin == "history")
            content = json.dumps({"suggested_action": "silent", "message": None,
                                  "reason": "不适状态未知，仅依据当前模拟事实。", "evidence": [historical.fact_id]}, ensure_ascii=False)
            return ModelResponse(content, "mock", "mock", None, None, 0, "stop")
    outcome = decide_simulated_event(event, Model(), context(history=history))
    assert review_outcome(outcome, do_not_disturb=False).status == "reviewed"
    fact = next(f for f in outcome.facts if f.origin == "history")
    for changed in (replace(fact, user_id="wrong"), replace(fact, event_id="other"),
                    replace(fact, session_id="other"), replace(fact, kind="current_event")):
        tampered = replace(outcome, facts=tuple(changed if f is fact else f for f in outcome.facts))
        assert review_outcome(tampered, do_not_disturb=False).code == "evidence_out_of_scope"


def test_context_builder_does_not_import_provider():
    source = (Path(__file__).resolve().parents[1] / "metafit_agent" / "context_builder.py").read_text(encoding="utf-8")
    assert "deepseek_client" not in source


@pytest.mark.parametrize("report,expected", [
    ("absent", "unknown"), (None, "unknown"),
    ({"status": "unknown", "source": "self_report"}, "unknown"),
    ({"status": "none", "source": "self_report"}, "none"),
])
def test_r2_and_r4_agree_on_discomfort_state(event, report, expected):
    if report == "absent":
        event.pop("discomfort")
    else:
        event["discomfort"] = report
    class Model:
        def complete(self, facts):
            assert next(f.value for f in facts if f.name == "discomfort_status") == expected
            reason = "不适状态未知，仅依据当前模拟事实。" if expected == "unknown" else "一次自报无不适，当前状态未确认。"
            content = json.dumps({"suggested_action": "silent", "message": None,
                                  "reason": reason, "evidence": []}, ensure_ascii=False)
            return ModelResponse(content, "mock", "mock", None, None, 0, "stop")
    result = decide_simulated_event(event, Model())
    assert result.status == "candidate"
    assert next(f.value for f in result.context_build.facts if f.name == "discomfort_status") == expected


def test_demo_context_safety_status_has_distinct_exit_code(event, monkeypatch, capsys):
    from metafit_agent import demo_context
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    monkeypatch.setattr(demo_context, "_load_json", lambda path: event if path.name == "set_completed.json" else context())
    assert demo_context.main() == 3
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "safety_pending" and output["safety_message"]
