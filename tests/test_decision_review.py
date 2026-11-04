"""R3 离线契约：只验证模拟引用及规则，不验证模型自由文字。"""

from dataclasses import replace
import json
import subprocess
import sys
from pathlib import Path

import pytest

from metafit_agent.decision import CandidateDecision, DecisionOutcome, DecisionScope, Fact, decide_simulated_event
from metafit_agent.decision_review import review_outcome
from metafit_agent.model_response import ModelResponse


@pytest.fixture
def outcome():
    scope = DecisionScope("user", "session", "event", ("event", "simulated_context:ctx"))
    facts = (Fact("fact_1", "completed_reps", 10, "user", "session", "event", "completed_reps"),
             Fact("fact_2", "goal", "general_fitness", "user", "session", "event", "goal", "simulated_context:ctx"))
    candidate = CandidateDecision("notify", "自由文本未经语义验证", "模型理由", ("fact_1", "fact_2"))
    return DecisionOutcome("candidate", candidate=candidate, facts=facts, warnings=("exercise_unmapped",), scope=scope)


@pytest.mark.parametrize("action,dnd,expected,rule", [
    ("notify", True, "defer", "dnd_notify_to_defer"), ("notify", False, "notify", None),
    ("defer", True, "defer", None), ("defer", False, "defer", None),
    ("silent", True, "silent", None), ("silent", False, "silent", None),
])
def test_action_and_dnd(outcome, action, dnd, expected, rule):
    candidate = replace(outcome.candidate, suggested_action=action, message=None if action == "silent" else "自由文本")
    updated = replace(outcome, candidate=candidate)
    result = review_outcome(updated, do_not_disturb=dnd)
    assert (result.status, result.effective_action, result.rule_code) == ("reviewed", expected, rule)
    assert result.source is updated and result.source.candidate is candidate
    assert result.warnings == ("exercise_unmapped",) and result.reference_check == "passed_for_cited_ids"
    assert "不验证" in result.notice


@pytest.mark.parametrize("action", ["silent", "notify", "defer"])
@pytest.mark.parametrize("dnd,code", [(None, "dnd_state_missing"), ("false", "dnd_state_invalid"), (0, "dnd_state_invalid")])
def test_missing_or_invalid_dnd_stops_every_ordinary_action(outcome, action, dnd, code):
    candidate = replace(outcome.candidate, suggested_action=action, message=None if action == "silent" else "text")
    result = review_outcome(replace(outcome, candidate=candidate), do_not_disturb=dnd)
    assert (result.status, result.code, result.effective_action) == ("stopped", code, None)


@pytest.mark.parametrize("dnd", [None, True, False, "invalid"])
def test_safety_wins_and_preserves_warning(dnd):
    source = DecisionOutcome("safety_pending", safety_message="fixed", warnings=("input_unvalidated",))
    result = review_outcome(source, do_not_disturb=dnd)
    assert result.status == "safety_pending" and result.source.safety_message == "fixed"
    assert result.effective_action is None and result.warnings == ("input_unvalidated",)


def test_upstream_error_and_unknown_status_are_distinct(outcome):
    error = replace(outcome, status="error", candidate=None, error_code="provider_timeout")
    result = review_outcome(error, do_not_disturb=False)
    assert (result.status, result.stage, result.code) == ("stopped", "r2", "provider_timeout")
    assert review_outcome(replace(outcome, status="future"), do_not_disturb=False).code == "unrecognized_r2_status"


def test_r2_error_without_code_is_reported(outcome):
    source = replace(outcome, status="error", candidate=None, error_code=None)
    result = review_outcome(source, do_not_disturb=False)
    assert (result.status, result.stage, result.code) == ("stopped", "r2", "r2_error_without_code")


@pytest.mark.parametrize("facts", [None, [], ("not-a-fact",), (Fact("bad", "x", 1, "user", "session", "event", "x"),)])
def test_invalid_fact_list_is_reported(outcome, facts):
    assert review_outcome(replace(outcome, facts=facts), do_not_disturb=False).code == "invalid_fact_list"


@pytest.mark.parametrize("scope", [
    DecisionScope("", "session", "event"), DecisionScope("user", " ", "event"),
    DecisionScope("user", "session", "event", ()), DecisionScope("user", "session", "event", ["event"]),
    DecisionScope("user", "session", "event", ("event", "event")),
    DecisionScope("user", "session", "event", ("simulated_context:ctx",)),
    DecisionScope("user", "session", "event", ("event", "other")),
    DecisionScope("user", "session", "event", ("event", 123)),
])
def test_malformed_scope_is_rejected(outcome, scope):
    assert review_outcome(replace(outcome, scope=scope), do_not_disturb=False).code == "invalid_scope"


@pytest.mark.parametrize("dnd", [None, "invalid", False])
def test_invalid_reference_reported_even_with_missing_or_invalid_dnd(outcome, dnd):
    source = replace(outcome, candidate=replace(outcome.candidate, evidence=("fact_1", "fact_99")))
    result = review_outcome(source, do_not_disturb=dnd)
    assert (result.status, result.code, result.effective_action) == ("stopped", "evidence_unknown_id", None)


def test_missing_dnd_after_valid_reference_keeps_reference_check(outcome):
    result = review_outcome(outcome)
    assert (result.status, result.code, result.reference_check) == (
        "stopped", "dnd_state_missing", "passed_for_cited_ids")


@pytest.mark.parametrize("evidence", [("fact_99",), ("fact_1", "fact_99")])
def test_unknown_or_partially_unknown_references_stop_whole_candidate(outcome, evidence):
    result = review_outcome(replace(outcome, candidate=replace(outcome.candidate, evidence=evidence)), do_not_disturb=False)
    assert (result.status, result.code, result.effective_action) == ("stopped", "evidence_unknown_id", None)


def test_empty_facts_referenced_and_duplicate_fact_ids_stop(outcome):
    assert review_outcome(replace(outcome, facts=()), do_not_disturb=False).code == "evidence_unknown_id"
    assert review_outcome(replace(outcome, facts=(outcome.facts[0], outcome.facts[0])), do_not_disturb=False).code == "duplicate_fact_id"


@pytest.mark.parametrize("field,value", [("user_id", "other"), ("session_id", "other"), ("event_id", "other")])
def test_mismatched_simulated_scope_stops(outcome, field, value):
    fact = replace(outcome.facts[0], **{field: value})
    assert review_outcome(replace(outcome, facts=(fact, outcome.facts[1])), do_not_disturb=False).code == "evidence_out_of_scope"


def test_unknown_origin_and_missing_scope_stop(outcome):
    fact = replace(outcome.facts[0], origin="simulated_context:other")
    assert review_outcome(replace(outcome, facts=(fact, outcome.facts[1])), do_not_disturb=False).code == "evidence_origin_not_allowed"
    assert review_outcome(replace(outcome, scope=None), do_not_disturb=False).code == "invalid_scope"


def test_silent_can_have_no_references_and_duplicate_candidate_ids_are_rejected(outcome):
    silent = replace(outcome.candidate, suggested_action="silent", message=None, evidence=())
    assert review_outcome(replace(outcome, candidate=silent), do_not_disturb=False).status == "reviewed"
    duplicate = replace(outcome.candidate, evidence=("fact_1", "fact_1"))
    assert review_outcome(replace(outcome, candidate=duplicate), do_not_disturb=False).code == "invalid_candidate"


def test_deterministic_and_does_not_mutate(outcome):
    original = repr(outcome)
    assert review_outcome(outcome, do_not_disturb=True) == review_outcome(outcome, do_not_disturb=True)
    assert repr(outcome) == original


def test_real_r2_result_connects_to_r3_once_without_network(monkeypatch):
    sample = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    event = json.loads(sample.read_text(encoding="utf-8"))
    monkeypatch.setattr("metafit_agent.deepseek_client.request.build_opener", lambda *_: pytest.fail("network"))
    class Model:
        calls = 0
        def complete(self, facts):
            self.calls += 1
            content = json.dumps({"suggested_action": "notify", "message": "本组完成了10次。",
                                  "reason": "不适状态未知，仅依据当前模拟事实。", "evidence": ["fact_1"]}, ensure_ascii=False)
            return ModelResponse(content, "mock", "mock", None, None, 0.0, "stop")
    model = Model()
    source = decide_simulated_event(event, model)
    result = review_outcome(source, do_not_disturb=True)
    assert model.calls == 1 and result.effective_action == "defer" and source.scope is not None


def test_reported_safety_bypasses_model_and_missing_dnd():
    sample = Path(__file__).resolve().parents[1] / "examples" / "set_completed.json"
    event = json.loads(sample.read_text(encoding="utf-8"))
    event["discomfort"] = {"status": "reported", "source": "self_report"}
    class NoModel:
        def complete(self, facts):
            pytest.fail("safety path must not call the model")
    source = decide_simulated_event(event, NoModel())
    result = review_outcome(source)
    assert result.status == "safety_pending" and result.effective_action is None
    assert result.source.safety_message


def test_demo_exit_codes():
    root = Path(__file__).resolve().parents[1]
    base = [sys.executable, "-m", "metafit_agent.demo_review", "examples/set_completed.json"]
    successful = subprocess.run(base + ["--dnd-on"], cwd=root, capture_output=True, text=True, encoding="utf-8")
    missing = subprocess.run(base, cwd=root, capture_output=True, text=True, encoding="utf-8")
    assert successful.returncode == 0 and json.loads(successful.stdout)["effective_action"] == "defer"
    assert missing.returncode == 2 and json.loads(missing.stdout)["code"] == "dnd_state_missing"
