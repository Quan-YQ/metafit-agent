"""R2 演示，默认离线；仅 --live 允许向 DeepSeek 发起请求。"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
import sys
import warnings

from .decision import UNKNOWN_REASON, DecisionOutcome, Fact, ModelCallError, decide_simulated_event
from .deepseek_client import DeepSeekModel
from .demo_training_event import _invalid_constant, _unique_object
from .model_response import ModelResponse


class _DemoModel:
    def complete(self, facts: tuple[Fact, ...]) -> ModelResponse:
        status = next(fact.value for fact in facts if fact.name == "discomfort_status")
        reason = "已自报无不适，当前状态和报告时效未确认。" if status == "none" else UNKNOWN_REASON
        content = json.dumps({"suggested_action": "silent", "message": None,
                              "reason": reason, "evidence": []}, ensure_ascii=False)
        return ModelResponse(content, "mock", "mock", None, None, 0.0, "stop")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def make_key_provider():
    """仅在确需网络调用时提示一次；无法隐藏输入时中止。"""
    cache: list[str] = []

    def prompt() -> str:
        if cache:
            return cache[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("请输入 DeepSeek API Key（输入不会显示）：")
        except getpass.GetPassWarning:
            raise ModelCallError("key_prompt_unavailable") from None
        except (EOFError, KeyboardInterrupt):
            raise ModelCallError("key_prompt_cancelled") from None
        cache.append(key)
        return key
    return prompt


def outcome_dict(result: DecisionOutcome, mode: str) -> dict:
    output = {"status": result.status, "error_code": result.error_code, "retryable": result.retryable,
              "mode": mode, "warnings": result.warnings,
              "notice": "仅供模拟测试；独立 R2 不读取会话安全状态，候选未经 R3 检查，不可用于真实训练。"}
    if result.safety_message is not None:
        output["safety_message"] = result.safety_message
    if result.model_response is not None:
        response = result.model_response
        output["model_metadata"] = {
            "requested_model": response.requested_model,
            "returned_model": response.returned_model,
            "request_id": response.request_id,
            "usage": asdict(response.usage) if response.usage is not None else None,
            "elapsed_ms": round(response.elapsed_ms, 2),
            "finish_reason": response.finish_reason,
        }
    if result.facts:
        output["available_facts"] = [{"fact_id": fact.fact_id, "name": fact.name, "value": fact.value,
                                      "origin_kind": ("simulation_context" if fact.origin.startswith("simulated_context:")
                                                      else "history" if fact.origin == "history" else "event")}
                                     for fact in result.facts]
    if result.candidate is not None:
        output["candidate"] = asdict(result.candidate)
        output["evidence_verified"] = False
    return output


def _sanitize_record(value):
    """过滤常见密钥、邮箱、手机号；不能保证识别所有个人信息。"""
    if isinstance(value, str):
        value = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED_KEY]", value, flags=re.IGNORECASE)
        value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", value)
        return re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[REDACTED_PHONE]", value)
    if isinstance(value, dict):
        return {key: _sanitize_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_record(item) for item in value]
    return value


def record_results(payload: dict) -> Path:
    """记录经有限过滤的模拟结果；文件不可覆盖。"""
    directory = Path(__file__).resolve().parents[1] / "outputs" / "r2_records"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"r2_{stamp}.json"
    with path.open("x", encoding="utf-8") as handle:
        record = {"sanitization_notice": "仅过滤常见密钥、邮箱和手机号，不能保证完全脱敏；过滤会改变回放文本。",
                  "data": _sanitize_record(payload)}
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def _utf8_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="演示模拟事件的未审核候选决策；默认不联网")
    parser.add_argument("path", type=Path, help="UTF-8 模拟事件 JSON 文件")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="离线模拟模型（默认）")
    mode.add_argument("--live", action="store_true", help="显式调用 DeepSeek，可能产生费用")
    parser.add_argument("--context", type=Path, help="可选的模拟上下文 JSON 文件")
    parser.add_argument("--prompt-key", action="store_true", help="需要时隐藏输入 API Key，仅本进程使用")
    parser.add_argument("--record", action="store_true", help="显式保存脱敏模拟结果供回放")
    args = parser.parse_args()
    if (args.prompt_key or args.record) and not args.live:
        parser.error("--prompt-key 和 --record 需要与 --live 一起使用")
    try:
        data = _load_json(args.path)
        context = _load_json(args.context) if args.context else None
    except (OSError, UnicodeError, ValueError):
        print("无法读取有效的 UTF-8 JSON 文件。", file=sys.stderr)
        return 2
    if args.live:
        print("真实调用可能产生费用；仅发送经过筛选的模拟事实。", file=sys.stderr)
    model = DeepSeekModel(key_provider=make_key_provider() if args.prompt_key else None) if args.live else _DemoModel()
    result = decide_simulated_event(data, model, context)
    output = outcome_dict(result, "deepseek" if args.live else "mock")
    if args.live:
        output["budget"] = {"limit_yuan": str(model.budget.limit_yuan), "calls": model.budget.calls,
                            "estimated_spent_yuan": str(model.budget.estimated_spent_yuan),
                            "uncertain": model.budget.uncertain,
                            "notice": "基于高峰价格与估算的进程内控制，并非供应商账单硬上限。"}
    if args.record:
        record = {"schema": "r2-simulation-record-1", "output": output,
                  "model_content_unreviewed": result.model_response.content if result.model_response else None}
        try:
            output["record_path"] = str(record_results(record))
        except (OSError, ValueError):
            output["record_error"] = "record_write_failed"
    _utf8_output()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    # 3 表示业务安全状态，不表示技术异常；1 才表示本次流程失败。
    return 0 if result.status == "candidate" else 3 if result.status == "safety_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
