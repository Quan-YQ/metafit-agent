"""离线几何核对：检验数值和边界，不判断动作质量。"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import check_motion_geometry as geometry
from render_motion_sample import Joint


def test_right_angle_and_degenerate_knee():
    assert geometry._angle((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(90)
    assert geometry._angle((0, 0, 0), (0, 0, 0), (0, 1, 0)) is None


def test_walk_case_exposes_model_claim_conflict():
    result = geometry.analyze(geometry.SAMPLES["round_b"])
    frames = result["frames"]
    assert result["frame_count"] == 6 and len(result["source_sha256"]) == 64
    # 这组数据的最后一帧髋部高于首帧，躯干线接近竖直；不替代真实人体测量。
    assert frames[-1]["hip_y_change_from_first_source_units"] > 0
    assert max(f["torso_line_angle_from_positive_y_degrees"] for f in frames) < 5
    assert "不作为 Agent 事实" in result["notice"]


def test_still_and_rising_samples_show_distinct_height_changes():
    rising = geometry.analyze(geometry.SAMPLES["round_a"])["frames"]
    standing = geometry.analyze(geometry.SAMPLES["round_c"])["frames"]
    assert rising[-1]["hip_y_change_from_first_source_units"] > 5
    assert max(abs(f["hip_y_change_from_first_source_units"]) for f in standing) < 0.1


def test_invalid_missing_and_nonfinite_inputs_fail_closed(monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"sample")
    monkeypatch.setattr(geometry, "parse_bvh", lambda _: ([Joint("Hips", None, (0, 0, 0), [])], [[0.0]]))
    with pytest.raises(ValueError):
        geometry.analyze(Path("unused"))
    joints = [Joint(name, None, (0, 0, 0), []) for name in geometry.REQUIRED]
    monkeypatch.setattr(geometry, "parse_bvh", lambda _: (joints, [[float("nan")]]))
    with pytest.raises(ValueError):
        geometry.analyze(Path("unused"))
