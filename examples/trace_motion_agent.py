"""公开动作样本的可追踪影子实验；只允许模拟和不可投递的 silent。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafit_agent.decision import ModelCallError, _parse_candidate, decide_simulated_event
from metafit_agent.decision_review import review_outcome
from metafit_agent.deepseek_client import MODEL
from metafit_agent.demo_decision import _DemoModel, make_key_provider
from metafit_agent.model_response import ModelResponse
from check_motion_geometry import SAMPLES, analyze_bytes
from check_vision import build_payload, call_once
from check_vision_observation import PROMPT, VALUES, compare
from render_motion_sample import parse_bvh, render_frames


class ScriptedObserver:
    """人为指定答案，供离线验证数据流；不理解图片。"""
    mode = "offline_scripted_observation"
    requested_model = None

    def __init__(self, answer="up"):
        if answer not in VALUES:
            raise ValueError("invalid scripted answer")
        self.answer = answer
        self.calls = 0
        self.usage = None
        self.elapsed_ms = None

    def observe(self, image):
        self.calls += 1
        return json.dumps({"hip_change": self.answer, "explanation": "人工指定的离线模拟回答"})


class DeepSeekObserver:
    """一次真实图片观察；无自动重试，失败可能已产生费用。"""
    mode = "live_deepseek_observation"
    requested_model = MODEL

    def __init__(self, key_provider=None):
        self.key_provider = key_provider
        self.calls = 0
        self.usage = None
        self.elapsed_ms = None

    def observe(self, image):
        if self.calls:
            raise ModelCallError("observation_call_limit")
        payload = json.loads(build_payload(image, PROMPT))
        payload["response_format"] = {"type": "json_object"}
        key = (self.key_provider() if self.key_provider is not None else os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise ModelCallError("missing_api_key")
        self.calls += 1
        try:
            content, usage, elapsed = call_once(json.dumps(payload, ensure_ascii=False).encode("utf-8"), key)
        except ValueError:
            raise ModelCallError("observation_provider_error") from None
        self.usage = usage or None
        self.elapsed_ms = round(elapsed, 2)
        return content


class ObservationGate:
    """观察是独立研究材料，不被伪装成 R2 事实；不一致或不确定时停止。"""

    def __init__(self, path, observer, decision_model=None, reverse=False, allow_non_silent=False):
        self.path, self.observer, self.reverse = path, observer, reverse
        self.decision_model = decision_model if decision_model is not None else _DemoModel()
        self.allow_non_silent = allow_non_silent
        self.observation = None
        self.decision_calls = 0

    def complete(self, facts):
        try:
            # 从同一份快照计算参考值和生成图片，避免两次读取产生错配。
            raw = self.path.read_bytes()
            geometry = analyze_bytes(raw, self.path.name, reverse=self.reverse)
            joints, frames = parse_bvh(raw.decode("utf-8"))
            image = render_frames(joints, frames, reverse=self.reverse)
        except (OSError, UnicodeError, ValueError, IndexError, StopIteration, OverflowError):
            raise ModelCallError("observation_source_invalid") from None
        if not image.startswith(b"\x89PNG\r\n\x1a\n") or len(image) > 2 * 1024 * 1024:
            raise ModelCallError("observation_image_invalid")
        reference = geometry["hip_endpoint_reference"]
        self.observation = {
            "schema": "motion-observation-shadow-1", "source_file": geometry["source_file"],
            "source_sha256": geometry["source_sha256"], "frame_order": geometry["frame_order"],
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "observer_mode": self.observer.mode, "requested_model": self.observer.requested_model, "reference": reference,
            "model_hip_change": None, "comparison": "not_completed",
        }
        try:
            response = self.observer.observe(image)
            checked = compare(response, reference)
            self.observation["model_hip_change"] = checked["model_observation"]["hip_change"]
            self.observation["comparison"] = checked["comparison"]
        except ValueError:
            self.observation["comparison"] = "invalid_output"
            raise ModelCallError("observation_invalid_output") from None
        # 只记录标签和摘要；不把未验证的文字、图片或几何指标塞进 R2 事实。
        canonical = json.dumps(self.observation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.observation["observation_id"] = "obs_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        if self.observation["comparison"] != "matches_reference":
            raise ModelCallError("observation_not_matched")
        self.decision_calls += 1
        decision = self.decision_model.complete(facts)
        if not isinstance(decision, ModelResponse):
            raise ModelCallError("shadow_invalid_decision")
        candidate, error = _parse_candidate(getattr(decision, "content", None))
        if error or candidate is None:
            raise ModelCallError(error or "shadow_invalid_decision")
        if candidate.suggested_action != "silent" and not self.allow_non_silent:
            raise ModelCallError("shadow_non_silent_blocked")
        return decision


def run_trace(event, sample="round_a", observer=None, decision_model=None, *, reverse=False):
    if sample not in SAMPLES:
        raise ValueError("unknown sample")
    chosen = observer if observer is not None else ScriptedObserver()
    gate = ObservationGate(SAMPLES[sample], chosen, decision_model, reverse)
    outcome = decide_simulated_event(event, gate)
    review = review_outcome(outcome, do_not_disturb=False)
    return {
        "simulation": True, "mode": chosen.mode, "sample": sample,
        "event_image_association": "not_verified", "observation": gate.observation,
        "observation_attempts": chosen.calls, "model_usage": chosen.usage,
        "model_elapsed_ms": chosen.elapsed_ms, "decision_model_attempts": gate.decision_calls,
        "decision_observation_id": (gate.observation.get("observation_id")
                                    if gate.decision_calls and gate.observation else None),
        "r2_status": outcome.status, "r2_error": outcome.error_code, "r3_status": review.status,
        "do_not_disturb": {"value": False, "source": "simulation_default"},
        "effective_action": review.effective_action, "may_display": False,
        "notice": "研究门控仅比较一个几何方向；摘要不是认证，图片与模拟事件不证明存在真实关联。"
                  "观察不是 R2 事实，检查通过不证明动作安全或模型整体可靠；非 R5 流程。",
    }


def main():
    parser = argparse.ArgumentParser(description="公开动作观察与 R2/R3 的可追踪影子演示")
    parser.add_argument("--sample", choices=("squat_1", "round_a"), default="round_a")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--live", action="store_true", help="发送一次公开图片请求，可能收费")
    parser.add_argument("--prompt-key", action="store_true", help="需要时隐藏输入密钥")
    parser.add_argument("--scripted-observation", choices=sorted(VALUES), help="指定离线模拟观察")
    args = parser.parse_args()
    if args.prompt_key and not args.live:
        parser.error("--prompt-key 需要 --live")
    if args.scripted_observation and args.live:
        parser.error("真实调用不能指定模拟观察")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        event_path = Path(__file__).with_name("set_completed.json")
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["exercise_id"] = "squat"
        for field in ("planned_reps", "completed_reps", "weight_kg_per_hand", "metrics", "set_index", "rest_seconds"):
            event.pop(field, None)
        event["discomfort"] = {"status": "none", "source": "self_report"}
        if args.live:
            print("最多发送一次公开样本图片请求，可能收费；不自动重试。")
            observer = DeepSeekObserver(make_key_provider() if args.prompt_key else None)
        else:
            observer = ScriptedObserver(args.scripted_observation or "up")
        result = run_trace(event, args.sample, observer, reverse=args.reverse)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["r3_status"] == "reviewed" else 1
    except (OSError, ValueError, json.JSONDecodeError):
        print("影子实验未完成。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
