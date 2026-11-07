"""独立读图探针：默认不联网；--live 才发送一张模拟 PNG，可能产生费用。"""

import argparse
import base64
import getpass
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafit_agent.model_client import _NoRedirect
from metafit_agent.deepseek_client import ENDPOINT, MODEL
from make_vision_sample import make_image
from render_motion_sample import ROUND_SAMPLES, WALK_SAMPLE, render as render_motion


MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 131072
PROMPT = ("这是一张人工绘制的骨架示意图，不是真人。请分别描述左侧和右侧人物的可见姿态，"
          "并比较它们的髋部和膝部弯曲情况。仅描述图中可见内容；看不清的地方请说无法判断，"
          "不要给出健康判断或训练建议。")
MOTION_PROMPT = ("这是一段公开动作捕捉数据转换的骨架图。每一列代表同一时刻，"
                 "每列上、下两张图是同一姿态的两个不同视角，六列从左到右按时间排列。"
                 "请描述整段中躯干、髋部和膝部可见的变化，以及最可能的动作类型。"
                 "仅依据画面；不确定或看不清时请说明，不要推断疼痛、动作是否安全，"
                 "不要给出训练建议。")


def prepare_image(path: Path) -> bytes:
    with path.open("rb") as image:
        data = image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("图片超过 2 MiB 限制")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("图片不是 PNG 格式")
    return data


def build_payload(image: bytes, prompt: str = PROMPT) -> bytes:
    encoded = base64.b64encode(image).decode("ascii")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
        ]}],
        "thinking": {"type": "disabled"},
        "max_tokens": 512,
        "stream": False,
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def call_once(payload: bytes, key: str, opener=None) -> tuple[str, object, float]:
    """仅调用一次，不跟随重定向，不输出密钥、图片或服务商错误正文。"""
    req = request.Request(ENDPOINT, data=payload, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with (opener or request.build_opener(_NoRedirect())).open(req, timeout=60) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("响应过大")
        body = json.loads(raw.decode("utf-8"))
        choice = body["choices"][0]
        text = choice["message"]["content"]
        if choice.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
            raise ValueError("模型没有返回完整回答")
        if key in text or re.search(r"sk-[A-Za-z0-9_-]{8,}", text, re.IGNORECASE):
            raise ValueError("回答包含敏感内容，已隐藏")
        usage = body.get("usage")
        safe_usage = ({name: usage.get(name) for name in ("prompt_tokens", "completion_tokens")
                       if type(usage.get(name)) is int and usage[name] >= 0}
                      if isinstance(usage, dict) else {})
        return text, safe_usage, (time.monotonic() - started) * 1000
    except error.HTTPError as exc:
        raise ValueError(f"服务商返回 HTTP {exc.code}；请检查密钥、余额及模型权限") from None
    except (error.URLError, OSError, TimeoutError, socket.timeout):
        raise ValueError("无法连接服务商或请求超时；本次没有自动重试") from None
    except (KeyError, IndexError, TypeError, UnicodeError, json.JSONDecodeError):
        raise ValueError("服务商响应格式异常") from None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="骨架图读图探针，默认只进行本地检查")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--image", type=Path, help="自备的模拟 PNG 图片路径；默认使用内置示意图")
    sources.add_argument("--motion-sample", action="store_true", help="使用公开动作数据生成的骨架图")
    sources.add_argument("--walk-sample", action="store_true", help="使用公开对照动作数据生成的骨架图")
    sources.add_argument("--final-round", action="store_true", help="依次测试三个公开动作样本")
    parser.add_argument("--live", action="store_true", help="发送真实 API 请求（可能产生费用）")
    parser.add_argument("--ask-key", action="store_true", help="在终端隐藏输入本次 API 密钥")
    args = parser.parse_args()
    try:
        if args.final_round:
            samples = [(label, render_motion(path)) for label, path in zip("ABC", ROUND_SAMPLES)]
        elif args.walk_sample:
            samples = [("单张", render_motion(WALK_SAMPLE))]
        elif args.motion_sample:
            samples = [("单张", render_motion())]
        else:
            samples = [("单张", prepare_image(args.image) if args.image else make_image())]
        prompt = MOTION_PROMPT if args.motion_sample or args.walk_sample or args.final_round else PROMPT
        payloads = [(label, build_payload(image, prompt)) for label, image in samples]
        print(f"骨架图片已就绪：{len(samples)} 张；模型：{MODEL}")
        if not args.live:
            print(f"检查完成：未联网。添加 --live 将依次发送 {len(samples)} 次请求。")
            return 0
        key = (getpass.getpass("请输入 DeepSeek API 密钥（不会显示）：") if args.ask_key
               else os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            print("未找到密钥：请设置 DEEPSEEK_API_KEY，或使用 --ask-key。")
            return 1
        for label, payload in payloads:
            print(f"\n正在测试样本 {label}……", flush=True)
            answer, usage, elapsed_ms = call_once(payload, key)
            print(f"样本 {label} 模型回答：\n{answer}\n\n耗时：{elapsed_ms:.0f} ms；token 用量：{usage or '未提供'}")
        print("注意：这只是读图实验，回答未经专业验证，不能用于训练指导。")
        return 0
    except (OSError, ValueError) as exc:
        print(f"实验未完成：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
