"""对照同一模拟事件时保持安全拦截和不投递。"""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import compare_vision_choice as compare
from research_motion_choice import OfflineChoiceModel
from trace_motion_agent import ScriptedObserver


def event():
    data = json.loads((Path(__file__).resolve().parents[1] / "examples/set_completed.json").read_text(encoding="utf-8"))
    data["discomfort"] = {"status": "none", "source": "self_report"}
    return data


def test_offline_pair_can_measure_different_action_without_delivery():
    result = compare.run_pair(event(), "round_a", ScriptedObserver("up"), OfflineChoiceModel(), compare.OfflineNoObservation())
    assert result["visual_result"]["candidate_action"] == "notify"
    assert result["text_only"]["candidate_action"] == "silent"
    assert result["action_comparison"] == "different_action"
    assert result["visual_observation_cited"] is True
    assert result["delivery"] is False and result["may_display"] is False


@pytest.mark.parametrize("reported,answer", [(False, "uncertain"), (True, "up")])
def test_observation_failure_or_reported_discomfort_stops_both_arms(reported, answer):
    source = event()
    if reported:
        source["discomfort"] = {"status": "reported", "source": "self_report"}
    baseline = compare.OfflineNoObservation()
    result = compare.run_pair(source, "round_a", ScriptedObserver(answer), OfflineChoiceModel(), baseline)
    assert baseline.calls == 0 and result["action_comparison"] == "not_compared"
    assert result["delivery"] is False


def test_paired_prompt_does_not_require_missing_observation():
    assert "若存在该观察" in compare.PAIRED_SYSTEM
    assert "若建议 notify 或 defer，evidence 必须引用该观察" not in compare.PAIRED_SYSTEM
