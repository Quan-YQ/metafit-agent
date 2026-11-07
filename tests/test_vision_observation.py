"""模型观察与骨架计算的离线对照测试。"""

import json
from pathlib import Path
import sys

import pytest


def test_extra_model_fields_are_discarded_without_affecting_observation():
    import json
    import check_vision_observation as observation
    response = json.dumps({"hip_change": "up", "explanation": "visible", "other": "untrusted"})
    checked = observation.compare(response, {"direction": "up"})
    assert checked["model_observation"] == {"hip_change": "up", "explanation": "visible"}
    assert checked["comparison"] == "matches_reference"
    with pytest.raises(ValueError):
        observation.compare(response + " " * 32769, {"direction": "up"})

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import check_vision_observation as observation
from check_motion_geometry import SAMPLES, analyze


def test_reference_is_consistent_with_three_motion_samples():
    assert analyze(SAMPLES["squat_1"])["hip_endpoint_reference"]["direction"] == "down"
    assert analyze(SAMPLES["round_a"])["hip_endpoint_reference"]["direction"] == "up"
    assert analyze(SAMPLES["round_c"])["hip_endpoint_reference"]["direction"] == "no_clear_change"


def test_reversed_sequence_inverts_reference_without_changing_source():
    forward = analyze(SAMPLES["round_a"])
    backward = analyze(SAMPLES["round_a"], reverse=True)
    assert forward["source_sha256"] == backward["source_sha256"]
    assert forward["hip_endpoint_reference"]["direction"] == "up"
    assert backward["hip_endpoint_reference"]["direction"] == "down"
    assert backward["hip_endpoint_reference"]["change_over_skeleton_y_span"] == pytest.approx(
        -forward["hip_endpoint_reference"]["change_over_skeleton_y_span"])


@pytest.mark.parametrize("prediction,expected", [("up", "matches_reference"), ("down", "differs_from_reference"),
                                                  ("uncertain", "inconclusive")])
def test_comparison_does_not_count_uncertain_as_correct(prediction, expected):
    response = json.dumps({"hip_change": prediction, "explanation": "依据图像观察"})
    assert observation.compare(response, {"direction": "up"})["comparison"] == expected


@pytest.mark.parametrize("response", ['{"hip_change":"up","hip_change":"down","explanation":"x"}',
                                       '{"hip_change":"unknown","explanation":"x"}',
                                       '{"hip_change":[],"explanation":"x"}',
                                       '{"hip_change":"up","explanation":""}', 'not json'])
def test_bad_model_output_is_rejected(response):
    with pytest.raises(ValueError):
        observation.compare(response, {"direction": "up"})


def test_offline_mode_never_calls_provider(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision_observation.py", "--sample", "round_a"])
    monkeypatch.setattr(observation, "call_once", lambda *args: pytest.fail("unexpected network"))
    assert observation.main() == 0
    assert json.loads(capsys.readouterr().out)["network"] is False


def test_live_request_is_one_image_and_preserves_uncertainty(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision_observation.py", "--sample", "round_a", "--live"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    calls = []
    def fake_call(payload, key):
        calls.append((json.loads(payload), key))
        return json.dumps({"hip_change": "uncertain", "explanation": "看不清"}), {"prompt_tokens": 20}, 200
    monkeypatch.setattr(observation, "call_once", fake_call)
    assert observation.main() == 0 and len(calls) == 1
    body, key = calls[0]
    assert key == "test-key" and body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    output = capsys.readouterr().out
    assert "inconclusive" in output and "test-key" not in output


def test_live_provider_error_stops_without_retry(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision_observation.py", "--live"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    calls = []
    def fake_call(payload, key):
        calls.append(1)
        raise ValueError("error")
    monkeypatch.setattr(observation, "call_once", fake_call)
    assert observation.main() == 1 and len(calls) == 1


def test_invalid_live_output_keeps_returned_token_usage(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision_observation.py", "--live"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(observation, "call_once", lambda *args: ("bad", {"prompt_tokens": 42}, 20))
    assert observation.main() == 1
    output = capsys.readouterr().out
    assert "invalid_model_output" in output and '"prompt_tokens": 42' in output


def test_controls_use_same_prompt_in_two_separate_requests(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision_observation.py", "--controls", "--live"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    calls = []
    def fake_call(payload, key):
        calls.append(json.loads(payload))
        value = "down" if len(calls) == 1 else "no_clear_change"
        return json.dumps({"hip_change": value, "explanation": "依据图像"}), {}, 100
    monkeypatch.setattr(observation, "call_once", fake_call)
    assert observation.main() == 0 and len(calls) == 2
    prompts = [call["messages"][0]["content"][0]["text"] for call in calls]
    images = [call["messages"][0]["content"][1]["image_url"]["url"] for call in calls]
    assert prompts[0] == prompts[1] and images[0] != images[1]
    assert capsys.readouterr().out.count("matches_reference") == 2


def test_controls_stop_after_first_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_vision_observation.py", "--controls", "--live"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    calls = []
    def fake_call(payload, key):
        calls.append(1)
        raise ValueError("unavailable")
    monkeypatch.setattr(observation, "call_once", fake_call)
    assert observation.main() == 1 and len(calls) == 1


def test_reversed_live_request_uses_same_question_and_reversed_image(monkeypatch, capsys):
    import base64
    monkeypatch.setattr(sys, "argv", ["check_vision_observation.py", "--sample", "round_a", "--reverse", "--live"])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    captured = []
    def fake_call(payload, key):
        captured.append(json.loads(payload))
        return json.dumps({"hip_change": "down", "explanation": "图像观察"}), {}, 100
    monkeypatch.setattr(observation, "call_once", fake_call)
    assert observation.main() == 0 and len(captured) == 1
    parts = captured[0]["messages"][0]["content"]
    assert parts[0]["text"] == observation.PROMPT
    actual = base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1])
    assert actual == observation.render(SAMPLES["round_a"], reverse=True)
    assert actual != observation.render(SAMPLES["round_a"])
    assert '"frame_order": "reversed"' in capsys.readouterr().out
