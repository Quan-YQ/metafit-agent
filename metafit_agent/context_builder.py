"""R4：受控模拟上下文唯一事实构建入口；不联网、不解释未确认指标。"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import re
from typing import Any

from .facts import DecisionScope, Fact
from .training_events import MISSING, DiscomfortReport, TrainingEvent, validate_training_event

EXERCISES = {"squat": "squat", "深蹲": "squat", "dumbbell_fly": "dumbbell_fly", "哑铃飞鸟": "dumbbell_fly"}
GOALS = {"technique_familiarization", "general_fitness"}
KINDS = ("profile", "training_history", "long_term_profile", "daily_state", "current_event")
# 工程性模拟筛选限制，不是训练或医学建议；参考时间为当前模拟事件时间。
HISTORY_LIMIT = 3
HISTORY_WINDOW_DAYS = 7
HISTORY_INPUT_LIMIT = 50


class ContextError(ValueError):
    """不回显输入值的受控上下文错误。"""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ContextBuild:
    facts: tuple[Fact, ...]
    scope: DecisionScope
    warnings: tuple[str, ...]
    missing_categories: tuple[str, ...]
    history_input_count: int
    history_selected_count: int
    fact_list_digest: str
    discomfort_time_relation: str
    history_window_days: int = HISTORY_WINDOW_DAYS
    history_limit: int = HISTORY_LIMIT


@dataclass(frozen=True, slots=True)
class MetricConcept:
    metric_id: str
    concept: str
    description: str
    source: str
    verification: str = "unverified"


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    metric_id: str
    concept: str
    exercise: str
    unit: str
    measurement_target: str
    source: str
    reliability: str
    limitation: str
    verification: str = "unverified"


def metric_definition_is_usable(definition: MetricDefinition) -> bool:
    """本单元无经专业确认的定义，所有定义均不得作为模型事实。"""
    return False


def _object(value: Any, allowed: set[str]) -> dict:
    if not isinstance(value, dict) or any(not isinstance(k, str) or k not in allowed for k in value):
        raise ContextError("invalid_simulated_context")
    return value


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def fact_list_digest(scope: DecisionScope, facts: tuple[Fact, ...]) -> str:
    """版本 1 本机比较用摘要，不是认证、匿名化或跨版本凭证。"""
    digest_input = {"version": 1, "scope": [scope.user_id, scope.session_id, scope.event_id,
                    scope.allowed_origins, scope.allowed_history],
                    "facts": [[f.fact_id, f.name, f.value, f.user_id, f.session_id, f.event_id,
                               f.field, f.origin, f.kind] for f in facts]}
    encoded = json.dumps(digest_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_context(event: TrainingEvent, context: Any = None) -> ContextBuild:
    """输入为 R1 已验证事件及明确标记的模拟上下文；本函数不验证真实数据来源。"""
    if not isinstance(event, TrainingEvent):
        raise ContextError("invalid_event")
    if context is None:
        context = {}
    else:
        context = _object(context, {"source", "context_id", "goal", "previous_completed_reps", "profile",
                                    "history", "long_term_profile", "daily_state"})
        if context.get("source") != "simulation" or not isinstance(context.get("context_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", context["context_id"]):
            raise ContextError("invalid_simulated_context")
    profile = _object(context["profile"], {"goal"}) if "profile" in context else {}
    long_term = _object(context["long_term_profile"], {"weekly_session_goal"}) if "long_term_profile" in context else {}
    daily = _object(context["daily_state"], {"energy_level"}) if "daily_state" in context else {}
    if "goal" in context and "goal" in profile:
        raise ContextError("invalid_simulated_context")
    goal = context.get("goal", profile.get("goal"))
    if ("goal" in context or "goal" in profile) and (not isinstance(goal, str) or goal not in GOALS):
        raise ContextError("invalid_simulated_context")
    previous = context.get("previous_completed_reps")
    if "previous_completed_reps" in context and (type(previous) is not int or not 0 <= previous <= 10000):
        raise ContextError("invalid_simulated_context")
    if "previous_completed_reps" in context and "history" in context:
        raise ContextError("invalid_simulated_context")
    if "weekly_session_goal" in long_term and (type(long_term["weekly_session_goal"]) is not int or not 0 <= long_term["weekly_session_goal"] <= 14):
        raise ContextError("invalid_simulated_context")
    if "energy_level" in daily and daily["energy_level"] not in ("low", "medium", "high"):
        raise ContextError("invalid_simulated_context")
    raw_history = context.get("history", [])
    if not isinstance(raw_history, list) or len(raw_history) > HISTORY_INPUT_LIMIT:
        raise ContextError("invalid_simulated_context")
    history = []
    seen = {event.event_id}
    now = _instant(event.timestamp_utc)
    exercise = EXERCISES.get(event.exercise_id)
    for raw in raw_history:
        validated = validate_training_event(raw)
        prior = validated.event
        if prior is None or prior.user_id != event.user_id or prior.event_id in seen:
            raise ContextError("invalid_simulated_context")
        seen.add(prior.event_id)
        when = _instant(prior.timestamp_utc)
        if exercise is not None and EXERCISES.get(prior.exercise_id) == exercise and when < now and now - when <= timedelta(days=HISTORY_WINDOW_DAYS):
            history.append((when, prior.event_id, prior))
    history.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in history[:HISTORY_LIMIT]]
    origins = ("event",) if not context else ("event", "simulated_context:" + context["context_id"])
    if selected:
        origins += ("history",)
    scope = DecisionScope(event.user_id, event.session_id, event.event_id, origins,
                          tuple((item.event_id, item.session_id) for item in selected))
    entries = []
    warnings = []
    def add(name, value, field, origin="event", kind="current_event", source_event=event):
        entries.append((name, value, field, origin, kind, source_event))
    if exercise is None:
        warnings.append("exercise_unmapped")
    else:
        add("exercise_id", exercise, "exercise_id")
    for name in ("planned_reps", "completed_reps", "set_index", "rest_seconds", "weight_kg_per_hand"):
        value = getattr(event, name)
        if value is not MISSING and value is not None:
            if type(value) is int and value.bit_length() > 128:
                raise ContextError("input_value_too_large")
            add(name, value, name)
    # 缺失派生 unknown 是 R2 已有安全规则，不是声称用户报告了 unknown。
    status = event.discomfort.status if isinstance(event.discomfort, DiscomfortReport) else "unknown"
    add("discomfort_status", status, "discomfort.status")
    report_time = event.discomfort.reported_at if isinstance(event.discomfort, DiscomfortReport) else None
    time_relation = "unknown" if not isinstance(report_time, str) else (
        "before_or_at_event" if _instant(report_time) <= now else "after_event")
    if event.metrics is not MISSING and event.metrics is not None and event.metrics:
        add("uninterpreted_metric_count", len(event.metrics), "uninterpreted_metric_count")
    context_origin = "simulated_context:" + context["context_id"] if context else "event"
    if goal is not None:
        add("goal", goal, "goal", context_origin, "profile")
    if previous is not None:
        add("previous_completed_reps", previous, "previous_completed_reps", context_origin, "training_history")
    for prior in selected:
        if prior.completed_reps is not MISSING and prior.completed_reps is not None:
            if prior.completed_reps.bit_length() > 128:
                raise ContextError("input_value_too_large")
            add("history_completed_reps", prior.completed_reps, "completed_reps", "history", "training_history", prior)
    if "weekly_session_goal" in long_term:
        add("weekly_session_goal", long_term["weekly_session_goal"], "weekly_session_goal", context_origin, "long_term_profile")
    if "energy_level" in daily:
        add("energy_level", daily["energy_level"], "energy_level", context_origin, "daily_state")
    facts = tuple(Fact(f"fact_{i}", name, value, source.user_id, source.session_id, source.event_id, field, origin, kind)
                  for i, (name, value, field, origin, kind, source) in enumerate(entries, 1))
    present = {fact.kind for fact in facts}
    missing = tuple(kind for kind in KINDS if kind not in present)
    # 仅供本机比较，未加盐摘要可能泄露低熵信息，不是认证或匿名化。
    digest = fact_list_digest(scope, facts)
    return ContextBuild(facts, scope, tuple(warnings), missing, len(raw_history), len(selected), digest, time_relation)
