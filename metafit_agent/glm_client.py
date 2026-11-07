"""智谱 GLM Chat Completions 单次调用（OpenAI 兼容端点）。

预算、HTTP 错误码与响应解析在 model_client.py 中与 DeepSeek 客户端共用。
与 DeepSeek 的差异：
- 端点与密钥（ZHIPUAI_API_KEY）不同；
- 智谱文档明确 GLM-5.3 / GLM-5.3-Flash 不支持 thinking.type=disabled（请求会报错），
  因此请求体不发送 thinking 字段；思考内容位于 message.reasoning_content，
  最终 JSON 仍位于 message.content；
- 强制思考会占用输出额度，max_tokens 提高到 4096，避免 JSON 被思考链截断。
"""

import json
import os
from typing import Callable
from urllib import request

from .decision import Fact, ModelCallError
from .model_client import CallBudget, SYSTEM_PROMPT, _NoRedirect, invoke, prepare
from .model_response import ModelResponse


ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-5.3-flash"
KEY_ENV = "ZHIPUAI_API_KEY"
MAX_OUTPUT_TOKENS = 8192
TIMEOUT_SECONDS = 120


class GLMModel:
    """只使用本次进程的密钥；无自动重试，失败后停止同预算继续调用。"""

    def __init__(self, budget: CallBudget | None = None, key_provider: Callable[[], str] | None = None,
                 model: str = MODEL):
        self.budget = budget if budget is not None else CallBudget()
        self.key_provider = key_provider
        self.model = model

    def complete(self, facts: tuple[Fact, ...]) -> ModelResponse:
        safe_facts = [{"fact_id": fact.fact_id, "name": fact.name, "value": fact.value} for fact in facts]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"facts": safe_facts}, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        prepare(payload_bytes, self.budget, MAX_OUTPUT_TOKENS)
        # 显式选择隐藏输入时使用本次输入，避免误用环境中的旧密钥。
        api_key = (self.key_provider() if self.key_provider is not None
                   else os.environ.get(KEY_ENV, "")).strip()
        if not api_key:
            raise ModelCallError("missing_api_key")
        return invoke(ENDPOINT, payload_bytes, api_key, self.budget, self.model,
                      timeout=TIMEOUT_SECONDS,
                      build_opener=lambda: request.build_opener(_NoRedirect()))
