"""R3：仅对模拟候选进行确定性的引用和普通反馈规则检查。"""

from dataclasses import dataclass
import json
import re

from .decision import CandidateDecision, DecisionOutcome, DecisionScope, Fact, _parse_candidate
from .context_builder import HISTORY_LIMIT


NOTICE = "仅供模拟；引用检查不验证自然语言真实性、安全性、真实身份或权限。"


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    status: str
    source: DecisionOutcome
    effective_action: str | None = None
    stage: str | None = None
    code: str | None = None
    rule_code: str | None = None
    reference_check: str = "not_checked"
    warnings: tuple[str, ...] = ()
    notice: str = NOTICE


def _candidate_is_valid(candidate: object) -> bool:
    if not isinstance(candidate, CandidateDecision) or not isinstance(candidate.evidence, tuple):
        return False
    try:
        raw = json.dumps({"suggested_action": candidate.suggested_action, "message": candidate.message,
                          "reason": candidate.reason, "evidence": candidate.evidence}, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return False
    parsed, error = _parse_candidate(raw)
    return error is None and parsed == candidate


def _scope_is_valid(scope: object) -> bool:
    if not isinstance(scope, DecisionScope):
        return False
    if not all(isinstance(value, str) and value.strip() for value in (scope.user_id, scope.session_id, scope.event_id)):
        return False
    if not isinstance(scope.allowed_origins, tuple) or not scope.allowed_origins or not all(isinstance(x, str) for x in scope.allowed_origins):
        return False
    if len(scope.allowed_origins) != len(set(scope.allowed_origins)):
        return False
    if not all(origin in ("event", "history") or re.fullmatch(r"simulated_context:[A-Za-z0-9_-]{1,64}", origin)
               for origin in scope.allowed_origins) or "event" not in scope.allowed_origins:
        return False
    history = scope.allowed_history
    if not isinstance(history, tuple) or len(history) > HISTORY_LIMIT:
        return False
    if any(not isinstance(pair, tuple) or len(pair) != 2 or any(not isinstance(v, str) or not v.strip() for v in pair)
           for pair in history):
        return False
    if len({pair[0] for pair in history}) != len(history) or any(pair[0] == scope.event_id for pair in history):
        return False
    return bool(history) == ("history" in scope.allowed_origins)


def review_outcome(outcome: DecisionOutcome, *, do_not_disturb: object = None) -> ReviewOutcome:
    """纯内存检查，不联网、不调用模型；输入必须来自受控模拟流程。"""
    def stop(stage: str, code: str, reference_check: str = "not_checked") -> ReviewOutcome:
        return ReviewOutcome("stopped", outcome, stage=stage, code=code, reference_check=reference_check,
                             warnings=outcome.warnings)

    if outcome.status == "safety_pending":
        return ReviewOutcome("safety_pending", outcome, warnings=outcome.warnings)
    if outcome.status == "error":
        return stop("r2", outcome.error_code or "r2_error_without_code")
    if outcome.status != "candidate":
        return stop("r3", "unrecognized_r2_status")
    candidate = outcome.candidate
    if not _candidate_is_valid(candidate):
        return stop("r3", "invalid_candidate")
    scope = outcome.scope
    if not _scope_is_valid(scope):
        return stop("r3", "invalid_scope")
    if not isinstance(outcome.facts, tuple) or any(not isinstance(fact, Fact) or not isinstance(fact.fact_id, str)
                                                   or not re.fullmatch(r"fact_[1-9][0-9]*", fact.fact_id)
                                                   for fact in outcome.facts):
        return stop("r3", "invalid_fact_list")
    index = {fact.fact_id: fact for fact in outcome.facts}
    if len(index) != len(outcome.facts):
        return stop("r3", "duplicate_fact_id")
    for fact_id in candidate.evidence:
        fact = index.get(fact_id)
        if fact is None:
            return stop("r3", "evidence_unknown_id")
        if fact.user_id != scope.user_id:
            return stop("r3", "evidence_out_of_scope")
        if fact.origin not in scope.allowed_origins:
            return stop("r3", "evidence_origin_not_allowed")
        if fact.origin == "history":
            if (fact.event_id, fact.session_id) not in scope.allowed_history or fact.kind != "training_history":
                return stop("r3", "evidence_out_of_scope")
        elif (fact.event_id, fact.session_id) != (scope.event_id, scope.session_id):
            return stop("r3", "evidence_out_of_scope")
    if do_not_disturb is None:
        return stop("r3", "dnd_state_missing", "passed_for_cited_ids")
    if type(do_not_disturb) is not bool:
        return stop("r3", "dnd_state_invalid", "passed_for_cited_ids")
    action = candidate.suggested_action
    rule = None
    if do_not_disturb and action == "notify":
        action, rule = "defer", "dnd_notify_to_defer"
    return ReviewOutcome("reviewed", outcome, effective_action=action, rule_code=rule,
                         reference_check="passed_for_cited_ids", warnings=outcome.warnings)
