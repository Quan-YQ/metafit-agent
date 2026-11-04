"""模型输出元信息；字段缺失时保持未知，不推断服务商的计费结果。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    requested_model: str
    returned_model: str | None
    request_id: str | None
    usage: TokenUsage | None
    elapsed_ms: float
    finish_reason: str | None
