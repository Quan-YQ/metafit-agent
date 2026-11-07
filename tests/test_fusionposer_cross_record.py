"""跨录制划分的基本可靠性，不依赖公开数据文件。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import fusionposer_cross_record as experiment


class Matrix:
    def __init__(self, marker, count=3, y=False):
        self.marker, self.rows, self.y = marker, count, y

    def row(self, i):
        return [self.marker] * (146 if self.y else 430)


def test_each_record_is_excluded_from_its_training_fold(monkeypatch):
    trained = []
    def train(inputs, labels):
        trained.append({row[0] for row in inputs})
        return 1
    monkeypatch.setattr(experiment, "train_ridge", train)
    monkeypatch.setattr(experiment, "predict", lambda model, features: [0] * 60)
    records = [(str(i), (Matrix(i), Matrix(i, y=True))) for i in (1, 2, 3)]
    result = experiment.leave_one_record_out(records)
    assert trained == [{2, 3}, {1, 3}, {1, 2}]
    assert [r["train_frames"] for r in result] == [6, 6, 6]
    assert [r["frames"] for r in result] == [3, 3, 3]


def test_rejects_duplicate_names_and_unpaired_training():
    pair = (Matrix(1), Matrix(1, y=True))
    with pytest.raises(ValueError):
        experiment.leave_one_record_out([("a", pair), ("a", pair)])
    with pytest.raises(ValueError):
        experiment.fit([(Matrix(1), Matrix(1, count=2, y=True))])


def test_changed_data_is_rejected_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "SAMPLES", {"test": ("1", "0" * 64, "0" * 64)})
    (tmp_path / "motion_1_x.npz").write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="摘要"):
        experiment.load_recording("test", tmp_path)
