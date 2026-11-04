"""以五个预置模拟场景评估 R2；默认离线，真实调用由 --live 明确启用。"""

import argparse
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import time

from .decision import _parse_candidate, decide_simulated_event
from .deepseek_client import CallBudget, DeepSeekModel
from .demo_decision import _DemoModel, _load_json, _utf8_output, make_key_provider, outcome_dict, record_results
from .model_response import ModelResponse


def run_evaluation(model, base: dict, context: dict, mode: str,
                   prior_coverage: dict | None = None) -> dict:
    scenarios = []
    reported = deepcopy(base)
    reported["discomfort"] = {"status": "reported", "source": "self_report"}
    invalid = deepcopy(base)
    invalid.pop("user_id", None)
    normal = deepcopy(base)
    normal["discomfort"] = {"status": "none", "source": "self_report"}
    with_context = deepcopy(base)
    with_context["discomfort"] = {"status": "unknown", "source": "self_report"}
    sparse = deepcopy(base)
    sparse.pop("completed_reps", None)
    sparse.pop("planned_reps", None)
    scenarios.extend([
        ("reported_discomfort", reported, None),
        ("invalid_input", invalid, None),
        ("ordinary_simulation", normal, None),
        ("unknown_with_simulated_history", with_context, context),
        ("insufficient_information", sparse, None),
    ])

    records = []
    model_attempts = 0
    structured = 0
    latencies = []
    stopped = False
    for name, event, ctx in scenarios:
        if name == "unknown_with_simulated_history" and prior_coverage is not None:
            records.append({"scenario": name, "status": "previously_covered", "model_called": False,
                            "prior_result": prior_coverage})
            continue
        if stopped:
            records.append({"scenario": name, "status": "skipped", "reason": "budget_or_connection_uncertain"})
            continue
        before = model.budget.calls if mode == "deepseek" else None
        started = time.monotonic()
        result = decide_simulated_event(event, model, ctx)
        wall_ms = round((time.monotonic() - started) * 1000, 2)
        after = model.budget.calls if mode == "deepseek" else None
        called = (after > before) if mode == "deepseek" else name not in {"reported_discomfort", "invalid_input"}
        if called:
            model_attempts += 1
            latencies.append(wall_ms)
            if result.candidate is not None or result.error_code in {
                "unknown_state_candidate_restricted", "none_state_ambiguous_reason"
            }:
                structured += 1
        record = {"scenario": name, "model_called": called, "elapsed_ms": wall_ms,
                  "outcome": outcome_dict(result, mode),
                  "model_content_unreviewed": result.model_response.content if result.model_response else None}
        records.append(record)
        if mode == "deepseek" and (model.budget.uncertain or result.error_code in {
            "budget_exceeded", "budget_unknown", "call_limit_reached", "missing_api_key",
            "key_prompt_cancelled", "key_prompt_unavailable",
        }):
            stopped = True

    output = {"mode": mode, "scenarios": records, "model_calls": model_attempts,
              "structured_output_successes": structured,
              "structured_output_rate": structured / model_attempts if model_attempts else None,
              "elapsed_ms_for_model_calls": latencies,
              "notice": "结构合规率不代表判断正确、安全或最终费用；候选未通过 R3 验证。"}
    if mode == "deepseek":
        output["budget"] = {"limit_yuan": str(model.budget.limit_yuan), "max_calls": model.budget.max_calls,
                            "calls": model.budget.calls, "estimated_spent_yuan": str(model.budget.estimated_spent_yuan),
                            "uncertain": model.budget.uncertain,
                            "notice": "进程内估算控制，不是供应商账单硬上限。"}
    return output


def validate_prior_success(record: dict, base: dict, context: dict) -> tuple[dict, Decimal]:
    """检查本地记录是否覆盖相同的模拟场景；不把历史未知费用记为零。"""
    if not isinstance(record, dict) or not isinstance(record.get("data"), dict):
        raise ValueError("invalid prior record")
    data = record["data"]
    output = data.get("output")
    if data.get("schema") != "r2-simulation-record-1" or not isinstance(output, dict):
        raise ValueError("unsupported prior record")
    if output.get("status") != "candidate" or output.get("mode") != "deepseek":
        raise ValueError("prior call was not a successful real candidate")
    metadata = output.get("model_metadata")
    if not isinstance(metadata, dict) or metadata.get("requested_model") != "deepseek-flash":
        raise ValueError("prior model metadata absent")
    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("prior usage absent")
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    if type(input_tokens) is not int or type(output_tokens) is not int or input_tokens < 0 or output_tokens < 0:
        raise ValueError("invalid prior usage")
    saved_candidate, problem = _parse_candidate(data.get("model_content_unreviewed"))
    if saved_candidate is None or problem is not None:
        raise ValueError("prior structure invalid")
    candidate = output.get("candidate")
    expected_candidate = {"suggested_action": saved_candidate.suggested_action,
                          "message": saved_candidate.message, "reason": saved_candidate.reason,
                          "evidence": list(saved_candidate.evidence)}
    if candidate != expected_candidate:
        raise ValueError("prior candidate mismatch")
    class _RecordedModel:
        def complete(self, facts):
            return ModelResponse(data["model_content_unreviewed"], "replay", "replay", None, None, 0.0, "stop")

    expected = decide_simulated_event(base, _RecordedModel(), context)
    if expected.status != "candidate":
        raise ValueError("current simulated scenario invalid")
    expected_facts = outcome_dict(expected, "mock").get("available_facts")
    if output.get("available_facts") != expected_facts:
        raise ValueError("prior and current simulated facts differ")
    cost = (Decimal(input_tokens) * 2 + Decimal(output_tokens) * 8) / 1000000
    return ({"status": "candidate", "action": saved_candidate.suggested_action,
             "requested_model": metadata.get("requested_model"), "returned_model": metadata.get("returned_model"),
             "usage": usage, "elapsed_ms": metadata.get("elapsed_ms"),
             "evidence_verified": False, "estimated_peak_yuan": str(cost)}, cost)


def main() -> int:
    parser = argparse.ArgumentParser(description="五个模拟场景的 R2 评估；默认离线")
    parser.add_argument("--live", action="store_true", help="显式调用 DeepSeek，可能收费")
    parser.add_argument("--prompt-key", action="store_true", help="需要时隐藏输入密钥")
    parser.add_argument("--record", action="store_true", help="显式保存脱敏模拟回放记录")
    parser.add_argument("--prior-calls", type=int, help="本轮前已发起的请求次数；重试前填写已核实数值")
    parser.add_argument("--prior-spend-yuan", help="本轮前已核实的累计费用（元），需与 --prior-calls 同时提供")
    parser.add_argument("--resume-record", type=Path, help="已成功的相同模拟场景记录；复用其结果")
    parser.add_argument("--accept-prior-unknown-cost", action="store_true", help="明确接受历史失败请求费用未知")
    args = parser.parse_args()
    if (args.prompt_key or args.record) and not args.live:
        parser.error("--prompt-key 和 --record 需要 --live")
    if args.accept_prior_unknown_cost and (not args.live or not args.resume_record or args.prior_calls is None):
        parser.error("接受未知历史费用时需要 --live、--resume-record 与 --prior-calls")
    if args.resume_record and not args.accept_prior_unknown_cost:
        parser.error("复用记录且历史费用未知时，必须显式提供 --accept-prior-unknown-cost")
    if args.accept_prior_unknown_cost and args.prior_spend_yuan is not None:
        parser.error("历史费用未知时不能填写已核实的 --prior-spend-yuan")
    if not args.accept_prior_unknown_cost and (args.prior_calls is None) != (args.prior_spend_yuan is None):
        parser.error("--prior-calls 与 --prior-spend-yuan 必须同时提供")
    if args.prior_calls is not None and not args.live:
        parser.error("既往请求次数和费用仅用于 --live")
    try:
        prior_spend = Decimal(args.prior_spend_yuan) if args.prior_spend_yuan is not None else Decimal("0")
    except InvalidOperation:
        parser.error("既往费用必须为数字")
    if not prior_spend.is_finite() or not 0 <= prior_spend <= 1:
        parser.error("既往费用必须是 0 到 1 元之间的有限数值")
    prior_calls = args.prior_calls if args.prior_calls is not None else 0
    if not 0 <= prior_calls <= 5:
        parser.error("既往请求次数必须在 0 到 5 之间")
    root = Path(__file__).resolve().parents[1]
    base = _load_json(root / "examples" / "set_completed.json")
    context = _load_json(root / "examples" / "simulated_context.json")
    prior_coverage = None
    if args.resume_record:
        if prior_calls < 1:
            parser.error("复用成功记录时此前请求次数必须至少为 1")
        try:
            prior_coverage, prior_spend = validate_prior_success(_load_json(args.resume_record), base, context)
        except (OSError, UnicodeError, ValueError, TypeError):
            parser.error("既有记录无效，或与当前模拟场景不一致；未发送请求")
    _utf8_output()
    if args.live:
        remaining = max(0, 5 - prior_calls)
        planned = min(remaining, 2 if prior_coverage is not None else 3)
        print(f"历史请求计入 {prior_calls} 次，本次最多发起 {planned} 次请求；进程内估算预算 1 元。")
        print("仅发送筛选后的模拟事实；进程内估算不是服务商账单硬上限。")
        if args.accept_prior_unknown_cost:
            print("注意：历史失败请求费用未知；只计入已有成功记录的用量估算，不能保证累计账单不超过 1 元。")
    budget = CallBudget(calls=prior_calls, estimated_spent_yuan=prior_spend)
    model = DeepSeekModel(budget=budget, key_provider=make_key_provider() if args.prompt_key else None) if args.live else _DemoModel()
    result = run_evaluation(model, base, context, "deepseek" if args.live else "mock", prior_coverage)
    if args.resume_record:
        result["prior_coverage"] = {"record": str(args.resume_record), "previous_requests": prior_calls,
                                    "successful_record": prior_coverage,
                                    "other_previous_costs_unknown": True,
                                    "notice": "未知历史费用不在累计估算中，不能保证总账单上限。"}
    if args.record:
        try:
            result["record_path"] = str(record_results({"schema": "r2-simulation-evaluation-1", "evaluation": result}))
        except (OSError, ValueError):
            result["record_error"] = "record_write_failed"
    _utf8_output()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(item.get("status") == "skipped" or item.get("outcome", {}).get("status") == "error"
           and item["scenario"] != "invalid_input" for item in result["scenarios"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
