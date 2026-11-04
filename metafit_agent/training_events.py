"""R1：模拟训练事件的数据结构、输入校验及规范化。

本模块只处理内存中的数据，不调用模型、不持久化，也不提供网络接口。
"""

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from enum import Enum
import json
import math
import re
from typing import Any


class Missing(Enum):
    """字段未提供，与明确提供 None（JSON null）区分。"""

    VALUE = "missing"


MISSING = Missing.VALUE
Number = int | float


@dataclass(frozen=True, slots=True)
class ValidationError:
    """一项输入错误；不包含用户提供的字段值。"""

    path: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExtensionPayload:
    """扩展对象的不可变 JSON 快照，不赋予其中数据业务含义。"""

    json_text: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.json_text)


@dataclass(frozen=True, slots=True)
class DiscomfortReport:
    status: str
    source: str
    reported_at: str | None | Missing = MISSING


@dataclass(frozen=True, slots=True)
class Metric:
    metric_id: str
    value: Number
    unit: str
    source: str
    observation_id: str
    confidence: Number | None | Missing = MISSING
    quality: str | None | Missing = MISSING
    extensions: ExtensionPayload | None | Missing = MISSING


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    schema_version: str
    event_id: str
    event_type: str
    user_id: str
    session_id: str
    set_id: str
    exercise_id: str
    timestamp: str
    source: str
    set_index: int | None | Missing = MISSING
    rest_seconds: Number | None | Missing = MISSING
    discomfort: DiscomfortReport | None | Missing = MISSING
    planned_reps: int | None | Missing = MISSING
    completed_reps: int | None | Missing = MISSING
    weight_kg_per_hand: Number | None | Missing = MISSING
    perceived_effort: Number | None | Missing = MISSING
    metrics: tuple[Metric, ...] | None | Missing = MISSING
    extensions: ExtensionPayload | None | Missing = MISSING

    @property
    def timestamp_utc(self) -> str:
        """UTC 供比较；原始 timestamp 保留，精度不超过微秒。"""
        return _utc_timestamp(self.timestamp)

    def to_dict(self, *, include_derived: bool = False) -> dict[str, Any]:
        """转为字典；默认可重新验证，可选输出派生的 UTC 时间。"""
        result: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is MISSING:
                continue
            if field.name == "metrics" and value is not None:
                result[field.name] = [_metric_to_dict(metric) for metric in value]
            elif field.name == "discomfort" and value is not None:
                result[field.name] = _discomfort_to_dict(value)
            elif field.name == "extensions" and value is not None:
                result[field.name] = value.to_dict()
            else:
                result[field.name] = value
        if include_derived:
            result["timestamp_utc"] = self.timestamp_utc
        return result


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """成功时返回事件；失败时只返回错误，避免使用部分有效的数据。"""

    event: TrainingEvent | None
    errors: tuple[ValidationError, ...]


def _metric_to_dict(metric: Metric) -> dict[str, Any]:
    result = {}
    for field in fields(metric):
        value = getattr(metric, field.name)
        if value is MISSING:
            continue
        result[field.name] = value.to_dict() if isinstance(value, ExtensionPayload) else value
    return result


def _discomfort_to_dict(report: DiscomfortReport) -> dict[str, Any]:
    return {field.name: value for field in fields(report)
            if (value := getattr(report, field.name)) is not MISSING}


_EVENT_FIELDS = {field.name for field in fields(TrainingEvent)}
_METRIC_FIELDS = {field.name for field in fields(Metric)}
_DISCOMFORT_FIELDS = {field.name for field in fields(DiscomfortReport)}
_REQUIRED_EVENT_STRINGS = (
    "schema_version", "event_id", "event_type", "user_id", "session_id", "set_id",
    "exercise_id", "timestamp", "source",
)
_REQUIRED_METRIC_STRINGS = ("metric_id", "unit", "source", "observation_id")
_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-](?P<offset_hour>\d{2}):(?P<offset_minute>\d{2}))$"
)
_SAFE_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _error(errors: list[ValidationError], path: str, code: str, message: str) -> None:
    errors.append(ValidationError(path, code, message))


def _check_unknown(data: dict[str, Any], allowed: set[str], path: str,
                   errors: list[ValidationError]) -> None:
    for key in data:
        if key not in allowed:
            safe_key = key if isinstance(key, str) and _SAFE_FIELD_NAME.fullmatch(key) else "[未显示字段名]"
            field_path = f"{path}.{safe_key}" if path != "$" else f"$.{safe_key}"
            _error(errors, field_path, "unknown_field", "未识别的核心字段；请检查版本和字段名。")


def _utc_timestamp(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _check_timestamp(timestamp: str, path: str, errors: list[ValidationError]) -> None:
    try:
        match = _TIMESTAMP.fullmatch(timestamp)
        if not match or timestamp.endswith("-00:00"):
            raise ValueError
        if match.group("offset_hour") is not None and (
            int(match.group("offset_hour")) > 23 or int(match.group("offset_minute")) > 59
        ):
            raise ValueError
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError
        parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        _error(errors, path, "invalid_timestamp", "时间须包含已知时区，小数秒最多 6 位，且可转换为 UTC。")


def _extension(data: dict[str, Any], path: str,
               errors: list[ValidationError]) -> ExtensionPayload | None | Missing:
    if "extensions" not in data:
        return MISSING
    value = data["extensions"]
    if value is None:
        return None
    if not isinstance(value, dict):
        _error(errors, path, "invalid_extensions", "扩展内容必须是 JSON 对象或 null。")
        return None
    try:
        # JSON 对象键必须是字符串，避免 JSON 序列化隐式转换键。
        stack = [(value, 0)]
        seen_count = 0
        while stack:
            item, depth = stack.pop()
            seen_count += 1
            if depth > 8 or seen_count > 1000:
                raise ValueError
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise ValueError
                stack.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                stack.extend((child, depth + 1) for child in item)
            elif item is not None and type(item) not in (str, int, float, bool):
                raise ValueError
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 16384:
            raise ValueError
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        _error(errors, path, "invalid_extensions", "扩展内容须为有限、大小和嵌套受限的 JSON 对象。")
        return None
    return ExtensionPayload(encoded)


def _required_string(data: dict[str, Any], name: str, path: str,
                     errors: list[ValidationError]) -> str | None:
    field_path = f"{path}.{name}" if path else name
    if name not in data:
        _error(errors, field_path, "required", "缺少必填字段。")
        return None
    value = data[name]
    if not isinstance(value, str) or not value.strip():
        _error(errors, field_path, "invalid_string", "必须是非空字符串。")
        return None
    return value.strip()


def _optional_string(data: dict[str, Any], name: str, path: str,
                     errors: list[ValidationError]) -> str | None | Missing:
    if name not in data:
        return MISSING
    if data[name] is None:
        return None
    return _required_string(data, name, path, errors)


def _number(data: dict[str, Any], name: str, path: str,
            errors: list[ValidationError], *, required: bool = False,
            integer: bool = False, nonnegative: bool = False) -> Number | None | Missing:
    field_path = f"{path}.{name}" if path else name
    if name not in data:
        if required:
            _error(errors, field_path, "required", "缺少必填字段。")
        return MISSING
    value = data[name]
    if value is None and not required:
        return None
    valid_type = type(value) is int or (not integer and type(value) is float)
    if not valid_type:
        _error(errors, field_path, "invalid_number", "必须是整数。" if integer else "必须是数值。")
        return None
    if type(value) is float and not math.isfinite(value):
        _error(errors, field_path, "non_finite", "数值必须是有限值。")
        return None
    if nonnegative and value < 0:
        _error(errors, field_path, "negative", "数值不能为负。")
        return None
    return value


def _parse_metric(data: Any, index: int, errors: list[ValidationError]) -> Metric | None:
    path = f"metrics[{index}]"
    if not isinstance(data, dict):
        _error(errors, path, "invalid_object", "指标必须是对象。")
        return None
    start = len(errors)
    _check_unknown(data, _METRIC_FIELDS, path, errors)
    values = {name: _required_string(data, name, path, errors) for name in _REQUIRED_METRIC_STRINGS}
    if values["source"] is not None and values["source"] != "simulation":
        _error(errors, f"{path}.source", "unsupported_source", "R1 的指标必须标记为 simulation。")
    values["value"] = _number(data, "value", path, errors, required=True)
    values["confidence"] = _number(data, "confidence", path, errors)
    confidence = values["confidence"]
    if type(confidence) in (int, float) and not 0 <= confidence <= 1:
        _error(errors, f"{path}.confidence", "out_of_range", "可信度须位于 0 到 1 之间。")
    values["quality"] = _optional_string(data, "quality", path, errors)
    values["extensions"] = _extension(data, f"{path}.extensions", errors)
    return Metric(**values) if len(errors) == start else None


def _parse_discomfort(data: dict[str, Any], errors: list[ValidationError]) -> DiscomfortReport | None | Missing:
    if "discomfort" not in data:
        return MISSING
    report = data["discomfort"]
    if report is None:
        return None
    if not isinstance(report, dict):
        _error(errors, "discomfort", "invalid_object", "不适自报必须是对象或 null。")
        return None
    start = len(errors)
    _check_unknown(report, _DISCOMFORT_FIELDS, "discomfort", errors)
    status = _required_string(report, "status", "discomfort", errors)
    source = _required_string(report, "source", "discomfort", errors)
    reported_at = _optional_string(report, "reported_at", "discomfort", errors)
    if status is not None and status not in {"reported", "none", "unknown"}:
        _error(errors, "discomfort.status", "invalid_status", "状态必须是 reported、none 或 unknown。")
    if source is not None and source != "self_report":
        _error(errors, "discomfort.source", "invalid_source", "不适状态来源必须是 self_report。")
    if isinstance(reported_at, str):
        reported_at = report["reported_at"]
        _check_timestamp(reported_at, "discomfort.reported_at", errors)
    if len(errors) != start:
        return None
    return DiscomfortReport(status=status, source=source, reported_at=reported_at)


def validate_training_event(data: Any) -> ValidationResult:
    """检查一条事件，收集可确定错误；通过后返回规范化数据模型。"""
    errors: list[ValidationError] = []
    if not isinstance(data, dict):
        _error(errors, "$", "invalid_object", "训练事件必须是对象。")
        return ValidationResult(None, tuple(errors))

    _check_unknown(data, _EVENT_FIELDS, "$", errors)
    values: dict[str, Any] = {
        name: _required_string(data, name, "", errors) for name in _REQUIRED_EVENT_STRINGS
    }
    if values["schema_version"] is not None and values["schema_version"] != "0.1":
        _error(errors, "schema_version", "unsupported_version", "R1 仅支持内部草案版本 0.1。")
    if values["event_type"] is not None and values["event_type"] != "set_completed":
        _error(errors, "event_type", "unsupported_event", "R1 仅支持 set_completed 事件。")
    if values["source"] is not None and values["source"] != "simulation":
        _error(errors, "source", "unsupported_source", "R1 只接受标记为 simulation 的模拟事件。")

    timestamp = values["timestamp"]
    if timestamp is not None:
        timestamp = data["timestamp"]
        values["timestamp"] = timestamp
        _check_timestamp(timestamp, "timestamp", errors)

    values["set_index"] = _number(data, "set_index", "", errors, integer=True, nonnegative=True)
    if type(values["set_index"]) is int and values["set_index"] == 0:
        _error(errors, "set_index", "out_of_range", "组序从 1 开始。")
    values["rest_seconds"] = _number(data, "rest_seconds", "", errors, nonnegative=True)
    values["discomfort"] = _parse_discomfort(data, errors)
    values["extensions"] = _extension(data, "extensions", errors)

    for name in ("planned_reps", "completed_reps"):
        values[name] = _number(data, name, "", errors, integer=True, nonnegative=True)
    values["weight_kg_per_hand"] = _number(data, "weight_kg_per_hand", "", errors, nonnegative=True)
    values["perceived_effort"] = _number(data, "perceived_effort", "", errors)

    if "metrics" not in data:
        values["metrics"] = MISSING
    elif data["metrics"] is None:
        values["metrics"] = None
    elif not isinstance(data["metrics"], list):
        _error(errors, "metrics", "invalid_list", "metrics 必须是数组或 null。")
        values["metrics"] = None
    else:
        parsed_metrics: list[Metric] = []
        seen: set[str] = set()
        for index, item in enumerate(data["metrics"]):
            metric = _parse_metric(item, index, errors)
            observation_id = item.get("observation_id") if isinstance(item, dict) else None
            if isinstance(observation_id, str) and observation_id.strip():
                observation_id = observation_id.strip()
                if observation_id in seen:
                    _error(errors, f"metrics[{index}].observation_id", "duplicate_observation", "同一事件内的观测 ID 重复。")
                seen.add(observation_id)
            if metric is not None:
                parsed_metrics.append(metric)
        values["metrics"] = tuple(parsed_metrics)

    if errors:
        return ValidationResult(None, tuple(errors))
    return ValidationResult(TrainingEvent(**values), ())
