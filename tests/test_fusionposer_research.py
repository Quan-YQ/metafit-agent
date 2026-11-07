"""公开 IMU 样本的受限读取、时序测试与预测数据流。"""

import math
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import fusionposer_research as research


def make_npy(rows, *, descriptor="<f8", fortran=False):
    header = repr({"descr": descriptor, "fortran_order": fortran, "shape": (len(rows), len(rows[0]))}).encode("latin1") + b"\n"
    values = [value for row in rows for value in row]
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<" + "d" * len(values), *values)


def test_safe_npy_reader_decodes_and_bounds_checks():
    matrix = research.NpyMatrix(make_npy([[1.0, 2.0], [3.0, 4.0]]), 2)
    assert (matrix.rows, matrix.columns) == (2, 2)
    assert list(matrix.row(1)) == [3.0, 4.0]
    with pytest.raises(IndexError):
        matrix.row(-1)
    with pytest.raises(IndexError):
        matrix.row(2)


@pytest.mark.parametrize("raw", [b"not numpy", make_npy([[1.0]], descriptor="|O"),
                                make_npy([[1.0]], fortran=True), make_npy([[float('nan')]]),
                                make_npy([[1.0]])[:-1]])
def test_npy_rejects_objects_unsafe_layout_nonfinite_and_truncation(raw):
    with pytest.raises(ValueError):
        research.NpyMatrix(raw, 1)


def test_ridge_learns_known_relation_without_external_dependencies():
    inputs = [[i / 10, float(i % 3)] for i in range(60)]
    targets = [[2 * row[0] - 3 * row[1] + 4] for row in inputs]
    model = research.train_ridge(inputs, targets, alpha=0.00001)
    assert research.predict(model, [1.5, 2])[0] == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError):
        research.predict(model, [float("nan"), 1])


def test_research_sample_is_time_split_and_outperforms_constant_on_this_sequence():
    if not research.X_FILE.exists() or not research.Y_FILE.exists():
        pytest.skip("公开研究样本尚未下载到本机")
    report, predicted, reference = research.experiment()
    assert (report["rows"], report["train_rows"], report["gap_rows"], report["test_rows"]) == (685, 411, 30, 244)
    assert report["selected_original_rows"][0] == 441 and report["selected_original_rows"][-1] == 684
    assert math.isfinite(report["test_mean_joint_error_cm"])
    assert report["test_mean_joint_error_cm"] < report["constant_pose_baseline_error_cm"]
    assert predicted.startswith(b"\x89PNG\r\n\x1a\n") and reference.startswith(b"\x89PNG\r\n\x1a\n")
    assert predicted != reference


def test_prediction_does_not_read_held_out_reference_positions(monkeypatch):
    class FakeMatrix:
        rows = 100
        def __init__(self, is_x, alter_test=False):
            self.is_x, self.alter_test = is_x, alter_test
        def row(self, index):
            if self.is_x:
                value = [0.0] * 430
                value[research.WINDOW_INDEX * 43] = 170 + index / 100
                return value
            value = [float((j % 3) * 10 + j // 3) for j in range(60)] + [0.0] * 86
            if self.alter_test and index >= 90:
                value[0] += 100
            return value
    monkeypatch.setattr(research, "load_pair", lambda: (FakeMatrix(True), FakeMatrix(False)))
    first, predicted_a, reference_a = research.experiment()
    monkeypatch.setattr(research, "load_pair", lambda: (FakeMatrix(True), FakeMatrix(False, True)))
    second, predicted_b, reference_b = research.experiment()
    assert predicted_a == predicted_b and reference_a != reference_b
    assert first["test_mean_joint_error_cm"] != second["test_mean_joint_error_cm"]
