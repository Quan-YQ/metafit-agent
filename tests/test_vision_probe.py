"""骨架图探针的离线测试，绝不连接真实服务。"""

import base64
import importlib.util
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
spec = importlib.util.spec_from_file_location("vision_probe_test_module", EXAMPLES / "check_vision.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.data[:limit]


class Opener:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def open(self, req, timeout):
        self.requests.append((req, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return Response(self.response)


def test_payload_contains_generated_image_in_user_message():
    image = probe.make_image()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    body = json.loads(probe.build_payload(image))
    assert body["model"] == "deepseek-flash"
    assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
    text, picture = body["messages"][0]["content"]
    assert text["type"] == "text" and "无法判断" in text["text"]
    assert base64.b64decode(picture["image_url"]["url"].split(",", 1)[1]) == image


def test_live_call_once_reads_answer_and_token_counts_without_retry():
    body = {"choices": [{"finish_reason": "stop", "message": {"content": "左侧较直，右侧弯曲。"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    opener = Opener(json.dumps(body).encode())
    answer, usage, elapsed = probe.call_once(probe.build_payload(probe.make_image()), "test-secret", opener)
    assert answer == "左侧较直，右侧弯曲。"
    assert usage == {"prompt_tokens": 100, "completion_tokens": 20}
    assert len(opener.requests) == 1 and elapsed >= 0
    assert opener.requests[0][0].get_header("Authorization") == "Bearer test-secret"


def test_http_failure_does_not_expose_body_or_retry():
    opener = Opener(HTTPError(probe.ENDPOINT, 401, "secret details", {}, None))
    with pytest.raises(ValueError, match="HTTP 401") as exc:
        probe.call_once(b"{}", "test-secret", opener)
    assert "secret details" not in str(exc.value) and len(opener.requests) == 1


def test_incomplete_or_sensitive_answer_is_rejected():
    for finish, answer in [("length", "incomplete"), ("stop", "test-secret")]:
        body = {"choices": [{"finish_reason": finish, "message": {"content": answer}}]}
        with pytest.raises(ValueError):
            probe.call_once(b"{}", "test-secret", Opener(json.dumps(body).encode()))


def test_dry_run_never_calls_network(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision.py"])
    monkeypatch.setattr(probe, "call_once", lambda *args: pytest.fail("unexpected network call"))
    assert probe.main() == 0
    assert "未联网" in capsys.readouterr().out


def test_motion_sample_uses_neutral_sequence_prompt():
    body = json.loads(probe.build_payload(probe.render_motion(), probe.MOTION_PROMPT))
    prompt = body["messages"][0]["content"][0]["text"]
    assert "从左到右按时间排列" in prompt
    assert "深蹲" not in prompt


def test_walk_sample_uses_same_neutral_prompt_without_network(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision.py", "--walk-sample"])
    monkeypatch.setattr(probe, "call_once", lambda *args: pytest.fail("unexpected network call"))
    assert probe.main() == 0
    assert "未联网" in capsys.readouterr().out
    body = json.loads(probe.build_payload(probe.render_motion(probe.WALK_SAMPLE), probe.MOTION_PROMPT))
    prompt = body["messages"][0]["content"][0]["text"]
    assert prompt == probe.MOTION_PROMPT and "走路" not in prompt and "下蹲" not in prompt


def test_final_round_sends_three_individual_images_with_same_prompt(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision.py", "--final-round", "--live", "--ask-key"])
    monkeypatch.setattr(probe.getpass, "getpass", lambda _: "test-key")
    requests = []
    def fake_call(payload, key):
        requests.append((json.loads(payload), key))
        return "示例回答", {}, 123
    monkeypatch.setattr(probe, "call_once", fake_call)
    assert probe.main() == 0
    assert len(requests) == 3
    assert all(req[0]["messages"][0]["content"][0]["text"] == probe.MOTION_PROMPT for req in requests)
    assert len({req[0]["messages"][0]["content"][1]["image_url"]["url"] for req in requests}) == 3
    output = capsys.readouterr().out
    assert all(f"样本 {label} 模型回答" in output for label in "ABC")
    assert "test-key" not in output


def test_final_round_stops_on_first_request_failure(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision.py", "--final-round", "--live"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    calls = []
    def fake_call(payload, key):
        calls.append(payload)
        raise ValueError("模拟失败")
    monkeypatch.setattr(probe, "call_once", fake_call)
    assert probe.main() == 1 and len(calls) == 1
