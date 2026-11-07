"""单次 DeepSeek 调用的延迟检查；只发送模拟事实，需设置 DEEPSEEK_API_KEY。

用法（在项目根目录）：
    py -3.12 examples/check_deepseek_latency.py
    py -3.12 examples/check_deepseek_latency.py --repeat 3

只发一次真实请求，用于对比 DeepSeek（关闭深度思考）与 GLM（强制思考）的耗时。
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafit_agent.decision import decide_simulated_event
from metafit_agent.deepseek_client import DeepSeekModel


def main() -> int:
    parser = argparse.ArgumentParser(description="单次 DeepSeek 调用的延迟检查；只发送模拟数据")
    parser.add_argument("--repeat", type=int, default=1, help="调用次数，默认 1，最多 5")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 5:
        parser.error("--repeat 必须在 1 到 5 之间")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass
    root = Path(__file__).resolve().parents[1]
    event = json.loads((root / "examples" / "set_completed.json").read_text(encoding="utf-8"))
    context = json.loads((root / "examples" / "simulated_context.json").read_text(encoding="utf-8"))
    # 用普通一组（自报无不适）做延迟基准，避免 unknown 规则拦截候选。
    event["discomfort"] = {"status": "none", "source": "self_report"}
    model = DeepSeekModel()
    result = None
    for index in range(args.repeat):
        started = time.monotonic()
        result = decide_simulated_event(event, model, context)
        wall_ms = round((time.monotonic() - started) * 1000)
        print(f"第 {index + 1} 次：{wall_ms} ms | 状态={result.status} | 错误码={result.error_code or '无'}")
        if result.candidate is not None:
            print(f"  suggested_action={result.candidate.suggested_action}")
            print(f"  message={result.candidate.message}")
            print(f"  reason={result.candidate.reason}")
        if result.model_response is not None:
            usage = result.model_response.usage
            print(f"  usage: input={usage.input_tokens if usage else '?'} output={usage.output_tokens if usage else '?'}")
    print(f"预算：调用 {model.budget.calls} 次，估算 {model.budget.estimated_spent_yuan} 元，uncertain={model.budget.uncertain}")
    return 0 if result is not None and result.status in {"candidate", "safety_pending"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
