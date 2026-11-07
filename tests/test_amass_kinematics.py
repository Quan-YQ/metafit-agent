"""不使用受许可限制的数据或模型，测试轴角与关节树的数学基础。"""
import math
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from amass_kinematics import forward_kinematics


def test_identity_preserves_rest_pose_and_applies_translation():
    result = forward_kinematics([(1, 2, 3), (1, 3, 3), (2, 3, 3)], [-1, 0, 1],
                                [(0, 0, 0)] * 3, (4, 5, 6))
    assert result == [(5, 7, 9), (5, 8, 9), (6, 8, 9)]


def test_root_and_child_rotations_affect_descendants():
    points = forward_kinematics([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [-1, 0, 1],
                                [(0, 0, math.pi / 2), (0, 0, math.pi / 2), (0, 0, 0)])
    assert points[1] == pytest.approx((0, 1, 0), abs=1e-10)
    assert points[2] == pytest.approx((-1, 1, 0), abs=1e-10)


@pytest.mark.parametrize("parents,angles", [([-1, 2], [(0, 0, 0)] * 2),
                                             ([-1, 0], [(0, 0, 0)]),
                                             ([-1, 0], [(0, 0, 0), (float('nan'), 0, 0)])])
def test_invalid_joint_tree_or_nonfinite_angles_fail(parents, angles):
    with pytest.raises(ValueError):
        forward_kinematics([(0, 0, 0), (1, 0, 0)], parents, angles)
