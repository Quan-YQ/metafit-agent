"""R2/R3 的离线图像数据流实验：固定样本、模拟模型、结果不可展示。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafit_agent.decision import ModelCallError, UNKNOWN_REASON, _parse_candidate, decide_simulated_event
from metafit_agent.decision_review import review_outcome
from metafit_agent.model_response import ModelResponse
from check_motion_geometry import SAMPLES
from render_motion_sample import render


class OfflineVisualStandIn:
    """模型替身只检查数据流并返回 silent；不执行图片理解。"""
    mode = "offline_visual_stand_in"

    def complete(self, facts, image):
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ModelCallError("invalid_motion_image")
        unknown = any(f.name == "discomfort_status" and f.value == "unknown" for f in facts)
        result = {"suggested_action": "silent", "message": None,
                  "reason": UNKNOWN_REASON if unknown else "模拟数据流检查，不生成反馈。", "evidence": []}
        return ModelResponse(json.dumps(result, ensure_ascii=False), "offline_visual_stand_in", None,
                             None, None, 0.0, "stop")


class ShadowAdapter:
    """只在 R2 允许调用后加载图片；拒绝所有非 silent 的候选。"""

    def __init__(self, sample, model, renderer=render):
        self.sample, self.model, self.renderer = sample, model, renderer
        self.calls = 0
        self.image_sha256 = None

    def complete(self, facts):
        try:
            image = self.renderer(self.sample)
        except (OSError, ValueError, IndexError, StopIteration, OverflowError):
            raise ModelCallError("motion_image_unavailable") from None
        if not isinstance(image, bytes) or not image.startswith(b"\x89PNG\r\n\x1a\n") or len(image) > 2 * 1024 * 1024:
            raise ModelCallError("invalid_motion_image")
        self.image_sha256 = hashlib.sha256(image).hexdigest()
        self.calls += 1
        response = self.model.complete(facts, image)
        if not isinstance(response, ModelResponse):
            raise ModelCallError("invalid_shadow_response")
        candidate, error = _parse_candidate(response.content)
        if error or candidate is None:
            raise ModelCallError("invalid_shadow_response")
        if candidate.suggested_action != "silent":
            raise ModelCallError("shadow_non_silent_blocked")
        return response


def run_shadow(event, sample, model=None, renderer=render):
    if sample not in SAMPLES:
        raise ValueError("unknown sample")
    selected_model = model if model is not None else OfflineVisualStandIn()
    adapter = ShadowAdapter(SAMPLES[sample], selected_model, renderer)
    outcome = decide_simulated_event(event, adapter)
    review = review_outcome(outcome, do_not_disturb=False)
    return {
        "simulation": True,
        "mode": getattr(selected_model, "mode", "injected_test_model"),
        "sample": sample,
        "visual_model_attempts": adapter.calls,
        "provider_requests_started": getattr(selected_model, "calls", None),
        "image_sha256": adapter.image_sha256,
        "r2_status": outcome.status,
        "r2_error": outcome.error_code,
        "r3_status": review.status,
        "effective_action": review.effective_action,
        "may_display": False,
        "model_usage": getattr(selected_model, "last_usage", None),
        "model_elapsed_ms": getattr(selected_model, "elapsed_ms", None),
        "notice": "影子研究只允许 silent；引用检查不验证图像描述。R2/R3 非 R5 会话安全协调，候选不可展示。",
    }


def main():
    parser = argparse.ArgumentParser(description="离线模拟骨架图经过 R2/R3 的数据流")
    parser.add_argument("--sample", choices=("squat_1", "round_a"), default="round_a")
    parser.add_argument("--live", action="store_true", help="使用 DeepSeek 发送一次真实请求，可能收费")
    parser.add_argument("--prompt-key", action="store_true", help="需要时隐藏输入 DeepSeek 密钥")
    args = parser.parse_args()
    if args.prompt_key and not args.live:
        parser.error("--prompt-key 需要 --live")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        event_path = Path(__file__).with_name("set_completed.json")
        event = json.loads(event_path.read_text(encoding="utf-8"))
        # 模拟事件与公开图像仅用于研究组合，不声称片段对应真实训练组。
        event["exercise_id"] = "squat"
        for field in ("planned_reps", "completed_reps", "weight_kg_per_hand", "metrics", "set_index", "rest_seconds"):
            event.pop(field, None)
        event["discomfort"] = {"status": "none", "source": "self_report"}
        model = None
        if args.live:
            from deepseek_motion_shadow import DeepSeekShadowModel
            from metafit_agent.demo_decision import make_key_provider
            print("真实调用最多一次，发送公开骨架图和模拟事实，可能产生费用；费用以服务商账单为准。")
            model = DeepSeekShadowModel(make_key_provider() if args.prompt_key else None)
        result = run_shadow(event, args.sample, model=model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["r3_status"] == "reviewed" else 1
    except (OSError, ValueError, json.JSONDecodeError):
        print("离线模拟未完成。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
