"""留出录制不得参与参数选择；下载只接受上游固定摘要。"""
import io
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import fusionposer_holdout as holdout


def test_validation_fold_never_trains_on_its_own_recording(monkeypatch):
    calls = []
    def fit(inputs, targets, alpha):
        calls.append((set(row[0] for row in inputs), alpha))
        return alpha
    monkeypatch.setattr(holdout, "train_ridge", fit)
    monkeypatch.setattr(holdout, "predict", lambda model, x: [model] * 60)
    monkeypatch.setattr(holdout, "mean_joint_error", lambda pred, target: abs(pred[0][0] - target[0][0]))
    records = {name: ([[float(i)]], [[float(i)] * 60]) for i, name in enumerate(holdout.RECORDINGS, 1)}
    chosen, scores = holdout.select_alpha(records)
    assert len(calls) == len(holdout.ALPHAS) * len(holdout.RECORDINGS)
    for i, (train_markers, alpha) in enumerate(calls):
        expected_validation = i % 3 + 1
        assert expected_validation not in train_markers
        assert len(train_markers) == 2
    assert chosen in holdout.ALPHAS and all(len(scores[a]) == 3 for a in holdout.ALPHAS)


def test_holdout_is_loaded_only_after_selection_and_refit(monkeypatch):
    events = []
    class Matrix:
        rows = 1
        def __init__(self, columns):
            self.columns = columns
        def row(self, index):
            return [0.0] * self.columns
    monkeypatch.setattr(holdout, "load_recording", lambda name, directory: (Matrix(430), Matrix(146)))
    def select(records):
        events.append("select")
        return 1.0, {a: [0.0] * 3 for a in holdout.ALPHAS}
    def train(inputs, targets, alpha):
        events.append("refit")
        return None
    def load(directory):
        events.append("holdout")
        return {"x": b"x", "y": b"y"}
    monkeypatch.setattr(holdout, "select_alpha", select)
    monkeypatch.setattr(holdout, "train_ridge", train)
    monkeypatch.setattr(holdout, "get_heldout_bytes", load)
    monkeypatch.setattr(holdout, "NpyMatrix", lambda raw, columns: Matrix(columns))
    monkeypatch.setattr(holdout, "predict", lambda model, x: [0.0] * 60)
    report = holdout.run()
    assert events == ["select", "refit", "holdout"]
    assert report["heldout_error"] == 0 and report["network"] is False


def test_invalid_download_is_never_saved(tmp_path, monkeypatch):
    good = b"public sample"
    monkeypatch.setattr(holdout, "HOLDOUT", {"x": ("x.npz", len(good), holdout.git_blob_sha(good), 1)})
    with pytest.raises(ValueError):
        holdout.get_heldout_bytes(tmp_path, download=True, opener=lambda *args, **kwargs: io.BytesIO(b"wrong"))
    assert not (tmp_path / "x.npz").exists()


def test_existing_changed_sample_is_not_overwritten(tmp_path, monkeypatch):
    good = b"public sample"
    monkeypatch.setattr(holdout, "HOLDOUT", {"x": ("x.npz", len(good), holdout.git_blob_sha(good), 1)})
    path = tmp_path / "x.npz"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError):
        holdout.get_heldout_bytes(tmp_path, download=True, opener=lambda *args, **kwargs: pytest.fail("unexpected network"))
    assert path.read_bytes() == b"wrong"


def test_download_counts_only_actual_network_calls(tmp_path, monkeypatch):
    good = b"public sample"
    monkeypatch.setattr(holdout, "HOLDOUT", {"x": ("x.npz", len(good), holdout.git_blob_sha(good), 1)})
    first = {"requests_started": 0}
    holdout.get_heldout_bytes(tmp_path, download=True, opener=lambda *a, **k: io.BytesIO(good), stats=first)
    second = {"requests_started": 0}
    holdout.get_heldout_bytes(tmp_path, download=True, opener=lambda *a, **k: pytest.fail("unexpected network"), stats=second)
    assert first["requests_started"] == 1 and second["requests_started"] == 0
