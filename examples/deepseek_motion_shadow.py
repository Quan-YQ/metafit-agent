"""一次性视觉 R2 研究模型；只供固定公开样本的 shadow 演示。"""

import base64
import json
import os

from metafit_agent.decision import ModelCallError
from metafit_agent.deepseek_client import MODEL
from metafit_agent.model_client import SYSTEM_PROMPT
from metafit_agent.model_response import ModelResponse, TokenUsage
from check_vision import call_once


MAX_TEXT_BYTES = 16 * 1024
SHADOW_SYSTEM = SYSTEM_PROMPT + "\n本次是影子研究：无论图片内容如何，必须输出 silent，message 为 null，evidence 为空；不向用户提供动作指导。"


class DeepSeekShadowModel:
    """每个实例最多发起一次请求；无自动重试，不代表账单上限。"""
    mode = "live_deepseek_shadow"

    def __init__(self, key_provider=None):
        self.key_provider = key_provider
        self.calls = 0
        self.last_usage = None
        self.elapsed_ms = None

    def complete(self, facts, image):
        if self.calls:
            raise ModelCallError("shadow_call_limit")
        safe = [{"fact_id": fact.fact_id, "name": fact.name, "value": fact.value} for fact in facts]
        text = json.dumps({"facts": safe}, ensure_ascii=False, allow_nan=False)
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ModelCallError("input_too_large")
        if not isinstance(image, bytes) or not image.startswith(b"\x89PNG\r\n\x1a\n") or len(image) > 2 * 1024 * 1024:
            raise ModelCallError("invalid_motion_image")
        encoded = base64.b64encode(image).decode("ascii")
        body = {"model": MODEL, "messages": [
            {"role": "system", "content": SHADOW_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": text},
                                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}]},
        ], "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"},
            "max_tokens": 512, "stream": False}
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        key = (self.key_provider() if self.key_provider is not None else os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise ModelCallError("missing_api_key")
        self.calls += 1
        try:
            content, usage, elapsed = call_once(payload, key)
        except ValueError:
            raise ModelCallError("shadow_provider_error") from None
        self.last_usage = usage or None
        self.elapsed_ms = elapsed
        token_usage = None
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            token_usage = TokenUsage(usage["prompt_tokens"], usage["completion_tokens"])
        return ModelResponse(content, MODEL, None, None, token_usage, elapsed, "stop")
