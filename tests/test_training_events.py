"""从 R1 的输入输出契约出发，检查典型情况与边界。"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from metafit_agent.training_events import MISSING, validate_training_event
from metafit_agent.demo_training_event import _invalid_constant, _unique_object


@pytest.fixture
def valid_event():
    return {
        "schema_version": "0.1", "event_id": "evt_1", "event_type": "set_completed", "user_id": "user_1",
        "session_id": "session_1", "set_id": "set_1", "exercise_id": "squat",
        "timestamp": "2026-11-02T18:30:00+08:00", "source": "simulation",
    }


@pytest.fixture
def metric():
    return {"metric_id": "range_change", "value": -18, "unit": "%", "source": "simulation", "observation_id": "obs_1"}


@pytest.mark.parametrize("exercise", ["squat", "dumbbell_fly", "custom_exercise"])
def test_multiple_exercises_use_same_event_workflow(valid_event, exercise):
    valid_event["exercise_id"] = exercise
    result = validate_training_event(valid_event)
    assert result.errors == ()
    assert result.event.to_dict()["exercise_id"] == exercise


def test_missing_null_and_zero_stay_distinct(valid_event, metric):
    valid_event.update(completed_reps=0, perceived_effort=None, metrics=[metric])
    result = validate_training_event(valid_event)
    assert not result.errors
    assert result.event.planned_reps is MISSING
    normalized = result.event.to_dict()
    assert "planned_reps" not in normalized
    assert normalized["completed_reps"] == 0
    assert normalized["perceived_effort"] is None
    assert "confidence" not in normalized["metrics"][0]


@pytest.mark.parametrize("metrics", [None, []])
def test_null_and_empty_metrics_remain_distinct(valid_event, metrics):
    valid_event["metrics"] = metrics
    result = validate_training_event(valid_event)
    assert not result.errors
    assert result.event.to_dict()["metrics"] == metrics


def test_sample_is_normalized_without_mutating_the_input(valid_event, metric):
    valid_event.update(event_id="  evt_1  ", metrics=[metric])
    original = copy.deepcopy(valid_event)
    result = validate_training_event(valid_event)
    assert not result.errors
    assert result.event.to_dict()["event_id"] == "evt_1"
    assert result.event.to_dict()["metrics"][0]["source"] == "simulation"
    assert valid_event == original


def test_collects_multiple_errors_across_event_and_metrics(valid_event, metric):
    valid_event.pop("user_id")
    valid_event.update(timestamp="not-a-time", completed_reps=-1)
    valid_event["metrics"] = [{**metric, "unit": "", "value": True}, "not-an-object"]
    result = validate_training_event(valid_event)
    assert result.event is None
    assert {error.path for error in result.errors} == {
        "user_id", "timestamp", "completed_reps", "metrics[0].unit",
        "metrics[0].value", "metrics[1]",
    }
    assert all("not-a-time" not in error.message for error in result.errors)


@pytest.mark.parametrize("field", ["schema_version", "event_id", "event_type", "user_id", "session_id", "set_id", "exercise_id", "timestamp", "source"])
def test_each_essential_field_is_required(valid_event, field):
    valid_event.pop(field)
    result = validate_training_event(valid_event)
    assert result.event is None
    assert any(e.path == field and e.code == "required" for e in result.errors)


@pytest.mark.parametrize("timestamp", [
    "2026-11-02T18:30:00", "2026-02-30T18:30:00Z", "2026-11-02 18:30:00+08:00",
    "2026-11-02T18:30:00+24:00", "2026-11-02T18:30:00+08:60",
    "2026-11-02T18:30:00-00:00", "2026-11-02T18:30:00.1234567Z",
    "0001-01-01T00:00:00+23:00",
])
def test_invalid_or_timezone_free_timestamps_are_rejected(valid_event, timestamp):
    valid_event["timestamp"] = timestamp
    assert any(e.path == "timestamp" for e in validate_training_event(valid_event).errors)


@pytest.mark.parametrize("timestamp", ["2026-11-02T10:30:00Z", "2026-11-02T18:30:00.123+08:00"])
def test_valid_timezone_aware_timestamps_are_accepted(valid_event, timestamp):
    valid_event["timestamp"] = timestamp
    assert not validate_training_event(valid_event).errors


@pytest.mark.parametrize("bad_value", [True, "3", None, float("nan"), float("inf"), float("-inf")])
def test_metric_value_must_be_finite_numeric_and_present(valid_event, metric, bad_value):
    metric["value"] = bad_value
    valid_event["metrics"] = [metric]
    assert any(e.path == "metrics[0].value" for e in validate_training_event(valid_event).errors)


@pytest.mark.parametrize("field,value", [
    ("planned_reps", True), ("completed_reps", 1.5), ("planned_reps", -1),
    ("weight_kg_per_hand", -0.5), ("perceived_effort", float("nan")),
])
def test_invalid_optional_numeric_values_are_rejected(valid_event, field, value):
    valid_event[field] = value
    assert any(e.path == field for e in validate_training_event(valid_event).errors)


def test_safety_source_and_event_type_are_explicit(valid_event):
    valid_event.update(source="sensor", event_type="rep_completed")
    result = validate_training_event(valid_event)
    assert {e.code for e in result.errors} >= {"unsupported_source", "unsupported_event"}


def test_observation_ids_cannot_repeat_even_if_other_metric_fields_are_invalid(valid_event, metric):
    valid_event["metrics"] = [
        {**metric, "unit": "", "observation_id": "obs_1"},
        {**metric, "observation_id": "obs_1"},
    ]
    result = validate_training_event(valid_event)
    assert {e.code for e in result.errors} >= {"invalid_string", "duplicate_observation"}


def test_unknown_fields_report_safe_name_without_echoing_values(valid_event):
    valid_event["client_version"] = "private value"
    result = validate_training_event(valid_event)
    assert result.errors[0].code == "unknown_field"
    assert result.errors[0].path == "$.client_version"
    assert "private value" not in str(result.errors)


def test_unsafe_unknown_field_name_is_not_echoed(valid_event):
    valid_event["untrusted\nprivate value"] = 1
    result = validate_training_event(valid_event)
    assert result.errors[0].code == "unknown_field"
    assert "private value" not in str(result.errors)


@pytest.mark.parametrize("data", [None, [], "hello", 5])
def test_non_object_root_returns_structured_error(data):
    result = validate_training_event(data)
    assert result.event is None
    assert result.errors[0].code == "invalid_object"


def test_demo_runs_real_sample_and_reports_valid_json():
    project = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    run = subprocess.run(
        [sys.executable, "-m", "metafit_agent.demo_training_event", "examples/set_completed.json"],
        cwd=project, env=env, capture_output=True, encoding="utf-8", check=False,
    )
    assert run.returncode == 0
    output = json.loads(run.stdout)
    assert output["ok"] is True
    assert output["event"]["source"] == "simulation"
    assert output["event"]["metrics"][0]["source"] == "simulation"


@pytest.mark.parametrize("content", ['{"id": 1, "id": 2}', '{"value": NaN}'])
def test_demo_json_parser_rejects_duplicate_keys_and_nonstandard_constants(content):
    with pytest.raises(ValueError):
        json.loads(content, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


@pytest.mark.parametrize("version", ["1.0", "0.2", "v0.1"])
def test_unrecognized_versions_are_not_processed(valid_event, version):
    valid_event["schema_version"] = version
    assert any(e.code == "unsupported_version" for e in validate_training_event(valid_event).errors)


@pytest.mark.parametrize("observation_id", [None, "", "   "])
def test_metric_requires_a_nonempty_observation_id(valid_event, metric, observation_id):
    metric["observation_id"] = observation_id
    valid_event["metrics"] = [metric]
    assert any(e.path == "metrics[0].observation_id" for e in validate_training_event(valid_event).errors)


def test_metric_rejects_missing_observation_id(valid_event, metric):
    metric.pop("observation_id")
    valid_event["metrics"] = [metric]
    assert any(e.code == "required" and e.path == "metrics[0].observation_id"
               for e in validate_training_event(valid_event).errors)


@pytest.mark.parametrize("confidence", [0, 0.45, 1, None])
def test_valid_confidence_bounds_and_null(valid_event, metric, confidence):
    metric["confidence"] = confidence
    valid_event["metrics"] = [metric]
    assert not validate_training_event(valid_event).errors


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 9999, True, "0.5"])
def test_invalid_confidence_is_rejected(valid_event, metric, confidence):
    metric["confidence"] = confidence
    valid_event["metrics"] = [metric]
    assert any(e.path == "metrics[0].confidence" for e in validate_training_event(valid_event).errors)


def test_simulated_metric_cannot_claim_sensor_source(valid_event, metric):
    metric["source"] = "sensor"
    valid_event["metrics"] = [metric]
    assert any(e.path == "metrics[0].source" and e.code == "unsupported_source"
               for e in validate_training_event(valid_event).errors)


@pytest.mark.parametrize("status", ["reported", "none", "unknown"])
def test_discomfort_report_is_preserved_without_treating_it_as_a_safety_decision(valid_event, status):
    valid_event["discomfort"] = {"status": status, "source": "self_report", "reported_at": None}
    result = validate_training_event(valid_event)
    assert not result.errors
    assert result.event.to_dict()["discomfort"] == valid_event["discomfort"]


def test_absent_null_and_explicit_unknown_discomfort_stay_distinct(valid_event):
    absent = validate_training_event(valid_event).event.to_dict()
    valid_event["discomfort"] = None
    null = validate_training_event(valid_event).event.to_dict()
    valid_event["discomfort"] = {"status": "unknown", "source": "self_report"}
    unknown = validate_training_event(valid_event).event.to_dict()
    assert "discomfort" not in absent
    assert null["discomfort"] is None
    assert unknown["discomfort"]["status"] == "unknown"


def test_reported_at_preserves_valid_original_and_rejects_surrounding_spaces(valid_event):
    timestamp = "2026-11-02T10:30:00Z"
    valid_event["discomfort"] = {"status": "none", "source": "self_report", "reported_at": timestamp}
    result = validate_training_event(valid_event)
    assert result.event.to_dict()["discomfort"]["reported_at"] == timestamp
    valid_event["discomfort"]["reported_at"] = f" {timestamp} "
    assert any(error.path == "discomfort.reported_at" for error in validate_training_event(valid_event).errors)


@pytest.mark.parametrize("report", [
    {"source": "self_report"}, {"status": "reported"},
    {"status": "painful", "source": "self_report"},
    {"status": "none", "source": "sensor"},
    {"status": "none", "source": "self_report", "reported_at": "2026-11-02"},
])
def test_invalid_discomfort_data_is_rejected(valid_event, report):
    valid_event["discomfort"] = report
    assert validate_training_event(valid_event).event is None


@pytest.mark.parametrize("field,value", [("set_index", 0), ("set_index", -1),
                                        ("set_index", 1.5), ("set_index", True),
                                        ("rest_seconds", -1), ("rest_seconds", float("inf"))])
def test_invalid_group_context_is_rejected(valid_event, field, value):
    valid_event[field] = value
    assert any(e.path == field for e in validate_training_event(valid_event).errors)


def test_valid_group_context_and_current_unscaled_effort_are_preserved(valid_event):
    valid_event.update(set_index=2, rest_seconds=90.5, perceived_effort=7)
    result = validate_training_event(valid_event)
    assert not result.errors
    assert result.event.to_dict()["rest_seconds"] == 90.5
    assert result.event.to_dict()["perceived_effort"] == 7


def test_equivalent_timestamps_normalize_to_utc_without_losing_originals(valid_event):
    first = validate_training_event(valid_event).event
    valid_event["timestamp"] = "2026-11-02T10:30:00Z"
    second = validate_training_event(valid_event).event
    assert first.timestamp == "2026-11-02T18:30:00+08:00"
    assert second.timestamp == "2026-11-02T10:30:00Z"
    assert first.timestamp_utc == second.timestamp_utc == "2026-11-02T10:30:00.000000Z"
    assert "timestamp_utc" not in first.to_dict()
    assert first.to_dict(include_derived=True)["timestamp_utc"] == first.timestamp_utc
    assert not validate_training_event(first.to_dict()).errors


def test_microsecond_precision_is_preserved(valid_event):
    valid_event["timestamp"] = "2026-11-02T18:30:00.123456+08:00"
    event = validate_training_event(valid_event).event
    assert event.timestamp_utc == "2026-11-02T10:30:00.123456Z"


def test_extensions_are_preserved_as_an_isolated_json_snapshot(valid_event, metric):
    valid_event["extensions"] = {"client_version": "0.1", "extra": [1, {"x": True}]}
    metric["extensions"] = {"algorithm": "simulated"}
    valid_event["metrics"] = [metric]
    event = validate_training_event(valid_event).event
    valid_event["extensions"]["extra"][1]["x"] = False
    result = event.to_dict()
    assert result["extensions"]["extra"][1]["x"] is True
    assert result["metrics"][0]["extensions"] == {"algorithm": "simulated"}
    result["extensions"]["extra"].append("changed")
    assert len(event.to_dict()["extensions"]["extra"]) == 2
    assert not validate_training_event(event.to_dict()).errors


@pytest.mark.parametrize("extension", [[], 12, {"value": float("nan")}, {1: "integer key"},
                                        {"oversized": "x" * 17000}])
def test_invalid_extensions_are_rejected(valid_event, extension):
    valid_event["extensions"] = extension
    assert any(e.path == "extensions" and e.code == "invalid_extensions"
               for e in validate_training_event(valid_event).errors)


def test_invalid_nested_safety_and_unknown_fields_report_multiple_paths(valid_event, metric):
    valid_event["discomfort"] = {"status": "reported", "source": "sensor", "pain_level": 3}
    metric["confidence"] = 2
    metric["unexpected_unit_hint"] = "unused value"
    valid_event["metrics"] = [metric]
    result = validate_training_event(valid_event)
    assert {e.path for e in result.errors} >= {
        "discomfort.source", "discomfort.pain_level", "metrics[0].confidence",
        "metrics[0].unexpected_unit_hint",
    }
    assert "unused value" not in str(result.errors)
