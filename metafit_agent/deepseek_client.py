"""DeepSeek 官方 Chat Completions 单次调用，当前仅发送模拟文本。

预算、HTTP 错误码与响应解析在 model_client.py 中与智谱客户端共用，
本模块只保留 DeepSeek 特有的端点和请求体（含关闭深度思考）。
"""

import json
import os
from typing import Callable
from urllib import request

from .decision import Fact, ModelCallError
from .model_client import (CallBudget, MAX_OUTPUT_TOKENS, MAX_REQUEST_BYTES,
                           SYSTEM_PROMPT, _NoRedirect, invoke, prepare)
from .model_response import ModelResponse


ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"


class DeepSeekModel:
    """只使用本次进程的密钥；无自动重试，失败后停止同预算继续调用。"""

    def __init__(self, budget: CallBudget | None = None, key_provider: Callable[[], str] | None = None):
        self.budget = budget if budget is not None else CallBudget()
        self.key_provider = key_provider

    def complete(self, facts: tuple[Fact, ...]) -> ModelResponse:
        safe_facts = [{"fact_id": fact.fact_id, "name": fact.name, "value": fact.value} for fact in facts]
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"facts": safe_facts}, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        prepare(payload_bytes, self.budget)
        # 显式选择隐藏输入时使用本次输入，避免误用环境中的旧密钥。
        api_key = (self.key_provider() if self.key_provider is not None
                   else os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not api_key:
            raise ModelCallError("missing_api_key")
        return invoke(ENDPOINT, payload_bytes, api_key, self.budget, MODEL,
                      build_opener=lambda: request.build_opener(_NoRedirect()))
