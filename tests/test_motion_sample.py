"""动作骨架计算的离线校验。"""

import math
from pathlib import Path
import struct
import sys
import zlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from render_motion_sample import Joint, ROUND_SAMPLES, WALK_SAMPLE, load_bvh, render, world_positions


def test_parent_rotation_moves_child_and_preserves_bone_length():
    joints = [Joint("root", None, (0, 0, 0), ["Zrotation"]),
              Joint("child", 0, (2, 0, 0), [])]
    root, child = world_positions(joints, [90])
    assert root == pytest.approx((0, 0, 0))
    assert child == pytest.approx((0, 2, 0))


def test_rotation_channels_follow_bvh_order():
    joints = [Joint("root", None, (0, 0, 0), ["Zrotation", "Yrotation"]),
              Joint("child", 0, (1, 0, 0), [])]
    _, child = world_positions(joints, [90, 90])
    assert child == pytest.approx((0, 0, -1), abs=1e-9)


def test_real_sample_produces_finite_varying_poses_and_png():
    joints, frames = load_bvh()
    assert len(frames) == 6 and len(joints) > 20
    poses = [world_positions(joints, frame) for frame in frames]
    assert all(math.isfinite(value) for pose in poses for point in pose for value in point)
    assert poses[0] != poses[-1]
    for index, joint in enumerate(joints):
        if joint.parent is not None:
            expected = math.dist((0, 0, 0), joint.offset)
            for pose in poses:
                assert math.dist(pose[index], pose[joint.parent]) == pytest.approx(expected, abs=1e-8)
    assert render().startswith(b"\x89PNG\r\n\x1a\n")


def test_walk_sample_has_six_real_frames_and_a_distinct_image():
    joints, frames = load_bvh(WALK_SAMPLE)
    assert len(frames) == 6
    poses = [world_positions(joints, frame) for frame in frames]
    assert poses[0] != poses[-1]
    assert render(WALK_SAMPLE) != render()


def test_round_samples_are_six_frame_distinct_sequences():
    images = []
    for path in ROUND_SAMPLES:
        joints, frames = load_bvh(path)
        assert len(frames) == 6
        assert world_positions(joints, frames[0]) != world_positions(joints, frames[-1])
        images.append(render(path))
    assert len(set(images)) == 3


def test_reverse_renders_exactly_the_same_frames_in_reverse_order():
    def rows(png):
        pos, chunks = 8, []
        while pos < len(png):
            size = struct.unpack(">I", png[pos:pos + 4])[0]
            kind = png[pos + 4:pos + 8]
            if kind == b"IDAT":
                chunks.append(png[pos + 8:pos + 8 + size])
            pos += size + 12
        data = zlib.decompress(b"".join(chunks))
        stride = 1200 * 3 + 1
        return [data[y * stride + 1:(y + 1) * stride] for y in range(460)]
    forward = rows(render(ROUND_SAMPLES[0]))
    backward = rows(render(ROUND_SAMPLES[0], reverse=True))
    for line, reversed_line in zip(forward, backward):
        for column in range(6):
            assert line[column * 600:(column + 1) * 600] == reversed_line[(5 - column) * 600:(6 - column) * 600]
