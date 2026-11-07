"""单图观察对照：默认离线；真实请求最多一次，仅作研究实验。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafit_agent.decision import ModelCallError, _unique_object
from metafit_agent.demo_decision import make_key_provider
from check_motion_geometry import SAMPLES, analyze
from check_vision import build_payload, call_once
from render_motion_sample import render


PROMPT = ("图中六列从左到右是同一动作的时间顺序，上下两排是同一时刻的两个视角。"
          "只比较第六列与第一列的髋部圆点在图中的垂直高度。图中所有帧的绘图比例相同。"
          "如果末帧髋部升高至少整张骨架竖直范围的 5%，回答 up；下降至少 5%，回答 down；"
          "变化小于 5%，回答 no_clear_change；看不清或无法确定则回答 uncertain。"
          "只输出 JSON 对象，字段只有 hip_change 和 explanation；explanation 用一句中文说明可见依据。"
          "不要推断训练安全或给出建议。")
VALUES = {"up", "down", "no_clear_change", "uncertain"}


def compare(response: str, reference: dict) -> dict:
    if not isinstance(response, str) or len(response) > 32768:
        raise ValueError("模型观察长度无效")
    try:
        value = json.loads(response, object_pairs_hook=_unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant")))
    except (ValueError, RecursionError):
        raise ValueError("模型没有返回有效的 JSON") from None
    if (not isinstance(value, dict) or not {"hip_change", "explanation"}.issubset(value)
            or not isinstance(value["hip_change"], str) or value["hip_change"] not in VALUES or not isinstance(value["explanation"], str)
            or not value["explanation"].strip() or len(value["explanation"]) > 500):
        raise ValueError("模型观察格式不符合实验要求")
    comparison = ("inconclusive" if value["hip_change"] == "uncertain" else
                  "matches_reference" if value["hip_change"] == reference["direction"] else "differs_from_reference")
    # 只保留已验证字段。额外字段不传播到决策、报告或日志。
    safe_observation = {key: value[key] for key in ("hip_change", "explanation")}
    return {"model_observation": safe_observation, "reference": reference, "comparison": comparison,
            "notice": "对照仅检验约定的图像几何观察，阈值不是专业标准；不代表训练质量或安全性。"}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="比较模型观察与公开骨架数据的可重复计算")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", choices=sorted(SAMPLES), help="选择一个公开样本，默认 round_a")
    source.add_argument("--controls", action="store_true", help="测试下降与近静止两个对照样本")
    parser.add_argument("--reverse", action="store_true", help="反转所选样本的帧顺序")
    parser.add_argument("--live", action="store_true", help="发送真实图片请求，可能收费")
    parser.add_argument("--prompt-key", action="store_true", help="在终端隐藏输入密钥")
    args = parser.parse_args()
    if args.prompt_key and not args.live:
        parser.error("--prompt-key 需要 --live")
    if args.controls and args.reverse:
        parser.error("--reverse 仅用于单个样本")
    try:
        names = ("squat_1", "round_c") if args.controls else (args.sample or "round_a",)
        cases = [(name, analyze(SAMPLES[name], reverse=args.reverse)["hip_endpoint_reference"],
                  render(SAMPLES[name], reverse=args.reverse)) for name in names]
        if not args.live:
            checks = [{"sample": name, "reference": reference} for name, reference, _ in cases]
            output = checks[0] if len(checks) == 1 else {"cases": checks}
            output["network"] = False
            output["frame_order"] = "reversed" if args.reverse else "original"
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0
        from os import environ
        key = (make_key_provider()() if args.prompt_key else environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise ValueError("未找到 DeepSeek 密钥")
        print(f"将依次发送最多 {len(cases)} 次公开样本图片请求，可能产生费用；遇到错误停止，不自动重试。")
        for name, reference, image in cases:
            payload = json.loads(build_payload(image, PROMPT))
            payload["response_format"] = {"type": "json_object"}
            print(f"正在检查样本 {name}……", flush=True)
            response, usage, elapsed = call_once(json.dumps(payload, ensure_ascii=False).encode("utf-8"), key)
            try:
                output = compare(response, reference)
            except ValueError:
                print(json.dumps({"status": "invalid_model_output", "sample": name,
                                  "frame_order": "reversed" if args.reverse else "original",
                                  "usage": usage or None, "elapsed_ms": round(elapsed, 2),
                                  "notice": "模型输出格式不合要求；已隐藏原文，不自动重试。"}, ensure_ascii=False, indent=2))
                return 1
            output.update({"sample": name, "frame_order": "reversed" if args.reverse else "original",
                           "usage": usage or None, "elapsed_ms": round(elapsed, 2)})
            print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, ModelCallError):
        print("实验未完成；如已发送请求，费用以服务商账单为准，程序没有自动重试。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
