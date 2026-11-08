"""核对误差公式、字段诊断与按录制隔离的参数选择。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import fusionposer_diagnostics as diagnostic
from fusionposer_research import mean_joint_error


def test_mean_joint_error_is_mean_of_3d_distances():
    target = [[0.0] * 60, [0.0] * 60]
    predictions = [[3.0, 4.0, 0.0] * 20, [0.0, 0.0, 12.0] * 20]
    assert mean_joint_error(predictions, target) == pytest.approx(8.5)
    with pytest.raises(ValueError):
        mean_joint_error([[0.0] * 59], [[0.0] * 59])


def test_heldout_target_does_not_influence_parameter_selection():
    def recording(multiplier):
        x = [[float(i)] + [0.0] * 42 for i in range(5)]
        y = [[multiplier * float(i)] * 60 for i in range(5)]
        return x, y
    records = {name: recording(n) for name, n in zip(diagnostic.RECORDINGS, (1.0, 2.0, 3.0))}
    first = diagnostic.evaluate_fold(records, "test_2", alphas=(1.0, 20.0))
    altered = dict(records)
    altered["test_2"] = (records["test_2"][0], [[1000.0] * 60 for _ in range(5)])
    second = diagnostic.evaluate_fold(altered, "test_2", alphas=(1.0, 20.0))
    assert first["validation_recording"] == "test_1"
    assert first["selection_train_recording"] == "train"
    assert first["selected_alpha"] == second["selected_alpha"]
    assert first["validation_errors_by_alpha"] == second["validation_errors_by_alpha"]
    assert first["test_error"] != second["test_error"]


def test_layout_diagnostic_does_not_treat_unit_norm_as_proof():
    class Matrix:
        rows = 2
        def row(self, index):
            return [0.0] * 60 + [10.0, 10.0] + [0.0, 0.0, 0.0, 1.0] * 21
    pairs = {name: (None, Matrix()) for name in diagnostic.RECORDINGS}
    result = diagnostic.layout_diagnostics(pairs)
    assert all(item["quaternion_unit_norm_fraction_by_offset"]["62"] == 1 for item in result["recordings"])
    assert all(item["quaternion_unit_norm_fraction_by_offset"]["60"] < 1 for item in result["recordings"])
    assert "不能证明" in result["notice"]
