"""R2：针对模拟训练事件生成尚未通过 R3 验证的候选决策。"""

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol

from .model_response import ModelResponse
from .facts import DecisionScope, Fact
from .training_events import DiscomfortReport, validate_training_event
from .context_builder import ContextBuild, ContextError, build_context
from .safety_signals import inspect_discomfort_signal


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    suggested_action: str
    message: str | None
    reason: str
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """候选决策不是最终决策；本模块不持久化数据。"""

    status: str
    candidate: CandidateDecision | None = None
    facts: tuple[Fact, ...] = ()
    error_code: str | None = None
    retryable: bool | None = None
    safety_message: str | None = None
    warnings: tuple[str, ...] = ()
    model_response: ModelResponse | None = None
    scope: DecisionScope | None = None
    context_build: ContextBuild | None = None


class ModelCallError(Exception):
    """向调用方传递不包含密钥、用户数据或供应商原始响应的错误代码。"""

    def __init__(self, code: str, retryable: bool | None = False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class DecisionModel(Protocol):
    def complete(self, facts: tuple[Fact, ...]) -> ModelResponse: ...


SAFETY_MESSAGE = "你报告了不适，请先停止当前动作；如症状明显、持续或加重，请联系医疗专业人员。"
UNKNOWN_REASON = "不适状态未知，仅依据当前模拟事实。"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError("invalid JSON constant")


def _parse_candidate(text: str) -> tuple[CandidateDecision | None, str | None]:
    if not isinstance(text, str) or not text.strip() or len(text) > 32768:
        return None, "invalid_model_json"
    try:
        data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_bad_constant)
    except (ValueError, RecursionError):
        return None, "invalid_model_json"
    if not isinstance(data, dict) or set(data) != {"suggested_action", "message", "reason", "evidence"}:
        return None, "invalid_model_structure"
    action, message, reason, evidence = (data[k] for k in ("suggested_action", "message", "reason", "evidence"))
    if action not in ("silent", "notify", "defer"):
        return None, "invalid_model_structure"
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        return None, "invalid_model_structure"
    if action == "silent":
        if message is not None:
            return None, "invalid_model_structure"
    elif not isinstance(message, str) or not message.strip() or len(message) > 500:
        return None, "invalid_model_structure"
    if not isinstance(evidence, list) or len(evidence) > 32:
        return None, "invalid_model_structure"
    if any(not isinstance(item, str) or not re.fullmatch(r"fact_[1-9][0-9]*", item) for item in evidence):
        return None, "invalid_model_structure"
    if len(evidence) != len(set(evidence)) or (action != "silent" and not evidence):
        return None, "invalid_model_structure"
    return CandidateDecision(action, message, reason, tuple(evidence)), None


def decide_simulated_event(data: Any, model: DecisionModel,
                           context: dict[str, Any] | None = None) -> DecisionOutcome:
    """R1 验证 → 不适拦截 → 最小事实 → 一次模型调用 → 候选结构检查。"""
    validated = validate_training_event(data)
    if validated.event is None:
        reported, source_valid = inspect_discomfort_signal(data)
        if reported:
            warnings = ("input_unvalidated",) + (() if source_valid else ("discomfort_source_unverified",))
            return DecisionOutcome("safety_pending", safety_message=SAFETY_MESSAGE, warnings=warnings)
        return DecisionOutcome("error", error_code="invalid_event")
    event = validated.event
    if isinstance(event.discomfort, DiscomfortReport) and event.discomfort.status == "reported":
        return DecisionOutcome("safety_pending", safety_message=SAFETY_MESSAGE)
    try:
        built = build_context(event, context)
    except ContextError as exc:
        return DecisionOutcome("error", error_code=exc.code, retryable=False)
    facts, warnings, scope = built.facts, built.warnings, built.scope
    try:
        response = model.complete(facts)
    except ModelCallError as exc:
        return DecisionOutcome("error", facts=facts, error_code=exc.code, retryable=exc.retryable, warnings=warnings, scope=scope, context_build=built)
    candidate, parse_error = _parse_candidate(response.content)
    if candidate is None:
        return DecisionOutcome("error", facts=facts, error_code=parse_error, retryable=False,
                               warnings=warnings, model_response=response, scope=scope, context_build=built)
    unknown_discomfort = not isinstance(event.discomfort, DiscomfortReport) or event.discomfort.status == "unknown"
    # 拦截已在评估中观察到的模糊固定句；这不是通用自然语言事实核验。
    if not unknown_discomfort and candidate.reason == UNKNOWN_REASON:
        return DecisionOutcome("error", facts=facts, error_code="none_state_ambiguous_reason",
                               retryable=False, warnings=warnings, model_response=response, scope=scope, context_build=built)
    if unknown_discomfort:
        if candidate.reason != UNKNOWN_REASON:
            return DecisionOutcome("error", facts=facts, error_code="unknown_state_candidate_restricted",
                                   retryable=False, warnings=warnings, model_response=response, scope=scope, context_build=built)
        if candidate.suggested_action != "silent":
            completed = event.completed_reps
            allowed = f"本组完成了{completed}次。" if type(completed) is int else None
            if allowed is None or candidate.message != allowed:
                return DecisionOutcome("error", facts=facts, error_code="unknown_state_candidate_restricted",
                                       retryable=False, warnings=warnings, model_response=response, scope=scope, context_build=built)
    return DecisionOutcome("candidate", candidate=candidate, facts=facts, warnings=warnings, model_response=response, scope=scope, context_build=built)
