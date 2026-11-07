"""仅用于公开样本的几何核对；不作为训练、医疗或 Agent 决策依据。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from render_motion_sample import SAMPLE, WALK_SAMPLE, ROUND_SAMPLES, parse_bvh, world_positions


SAMPLES = {
    "squat_1": SAMPLE,
    "walk_1": WALK_SAMPLE,
    "round_a": ROUND_SAMPLES[0],
    "round_b": ROUND_SAMPLES[1],
    "round_c": ROUND_SAMPLES[2],
}
REQUIRED = ("Hips", "LeftArm", "RightArm", "LeftUpLeg", "LeftLeg", "LeftFoot",
            "RightUpLeg", "RightLeg", "RightFoot")
NOTICE = ("公开动捕样本的几何检查，单位和角度取决于数据与关节定义；"
          "不是人体实际测量、动作质量或安全判断，不作为 Agent 事实。")


def _angle(a, b, c):
    first = tuple(a[i] - b[i] for i in range(3))
    second = tuple(c[i] - b[i] for i in range(3))
    lengths = math.hypot(*first) * math.hypot(*second)
    if lengths <= 1e-12:
        return None
    cosine = sum(x * y for x, y in zip(first, second)) / lengths
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def analyze(path: Path, *, reverse: bool = False) -> dict:
    return analyze_bytes(path.read_bytes(), path.name, reverse=reverse)


def analyze_bytes(raw: bytes, source_name: str, *, reverse: bool = False) -> dict:
    joints, frames = parse_bvh(raw.decode("utf-8"))
    if reverse:
        frames = list(reversed(frames))
    names = [joint.name for joint in joints]
    if any(names.count(name) != 1 for name in REQUIRED):
        raise ValueError("缺少必要关节或关节名称重复")
    if (not frames or any(not math.isfinite(value) for frame in frames for value in frame)
            or any(not math.isfinite(value) for joint in joints for value in joint.offset)):
        raise ValueError("动作数据为空或含非有限数值")
    index = {name: names.index(name) for name in REQUIRED}
    poses = [world_positions(joints, frame) for frame in frames]
    if any(not math.isfinite(value) for pose in poses for point in pose for value in point):
        raise ValueError("计算所得坐标无效")
    start_height = poses[0][index["Hips"]][1]
    vertical_span = max(point[1] for pose in poses for point in pose) - min(point[1] for pose in poses for point in pose)
    if not math.isfinite(vertical_span) or vertical_span <= 1e-12:
        raise ValueError("骨架竖直范围无效")
    hip_change_ratio = (poses[-1][index["Hips"]][1] - start_height) / vertical_span
    reference = "up" if hip_change_ratio >= 0.05 else "down" if hip_change_ratio <= -0.05 else "no_clear_change"
    output = []
    for number, pose in enumerate(poses, 1):
        hip = pose[index["Hips"]]
        left_shoulder, right_shoulder = pose[index["LeftArm"]], pose[index["RightArm"]]
        middle = tuple((left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3))
        vector = tuple(middle[i] - hip[i] for i in range(3))
        tilt = (math.degrees(math.atan2(math.hypot(vector[0], vector[2]), vector[1]))
                if math.hypot(*vector) > 1e-12 else None)
        def knee(side):
            angle = _angle(*(pose[index[side + part]] for part in ("UpLeg", "Leg", "Foot")))
            return round(angle, 2) if angle is not None else None
        output.append({
            "frame": number,
            "hip_y_change_from_first_source_units": round(hip[1] - start_height, 4),
            "torso_line_angle_from_positive_y_degrees": round(tilt, 2) if tilt is not None else None,
            "left_knee_joint_line_angle_degrees": knee("Left"),
            "right_knee_joint_line_angle_degrees": knee("Right"),
        })
    return {"schema": "motion-geometry-research-1", "source_file": source_name,
            "source_sha256": hashlib.sha256(raw).hexdigest(), "frame_order": "reversed" if reverse else "original",
            "coordinate_convention": "BVH Y 轴向上；髋部变化相对首帧，单位沿用源数据",
            "hip_endpoint_reference": {"direction": reference, "change_over_skeleton_y_span": round(hip_change_ratio, 4),
                                       "threshold": 0.05, "note": "工程试验阈值，不是专业标准或训练质量标签。"},
            "frame_count": len(output), "frames": output, "notice": NOTICE}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="离线核对公开样本的骨架几何变化")
    parser.add_argument("--sample", choices=sorted(SAMPLES), default="round_b")
    args = parser.parse_args()
    try:
        print(json.dumps(analyze(SAMPLES[args.sample]), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, IndexError, StopIteration, OverflowError):
        print("无法计算：样本不存在或格式无效。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
