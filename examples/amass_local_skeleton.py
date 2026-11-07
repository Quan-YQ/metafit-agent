"""仅本地研究：AMASS + 已授权 SMPL+H 模型转换为 6 帧线框，不联网。"""
import argparse
import io
import json
import math
from pathlib import Path
import sys
import tarfile
import zipfile

import numpy as np

from amass_kinematics import forward_kinematics
from render_motion_sample import Joint, render_frames


ROOT = Path(__file__).resolve().parents[1]
MOTION_ARCHIVE = ROOT / "data/amass/ACCAD.tar.bz2"
MODEL_ARCHIVE = ROOT / "data/body_models/smplh.tar.xz"
DEFAULT_OUTPUT = ROOT / "data/amass/local_output/accad_skeleton.png"
MAX_MEMBER = 256 * 1024 * 1024
BODY_NAMES = ("Pelvis", "LeftHip", "RightHip", "Spine1", "LeftKnee", "RightKnee", "Spine2", "LeftAnkle",
              "RightAnkle", "Spine3", "LeftFoot", "RightFoot", "Neck", "LeftCollar", "RightCollar", "Head",
              "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow", "LeftWrist", "RightWrist")


def read_npz(raw, fields):
    """只读取数值数组；限制解压总量，绝不加载 pickle。"""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        items = archive.infolist()
        if len(items) > 32 or sum(item.file_size for item in items) > 512 * 1024 * 1024:
            raise ValueError("NPZ 解压大小超限")
        names = [item.filename for item in items]
        if len(set(names)) != len(names):
            raise ValueError("NPZ 包含重复成员")
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        if not set(fields).issubset(loaded.files):
            raise ValueError("缺少必需字段")
        return {name: loaded[name] for name in fields}


def _member_bytes(archive, member):
    if not member.isfile() or member.size <= 0 or member.size > MAX_MEMBER:
        raise ValueError("压缩包成员无效或超限")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError("无法读取压缩包成员")
    raw = stream.read(MAX_MEMBER + 1)
    if len(raw) != member.size:
        raise ValueError("压缩包成员长度不匹配")
    return raw


def load_motion(path=MOTION_ARCHIVE, index=0):
    if type(index) is not int or index < 0:
        raise ValueError("动作序号无效")
    with tarfile.open(path, mode="r|bz2") as archive:
        seen = 0
        for member in archive:
            if member.isfile() and member.name.lower().endswith(".npz"):
                if seen == index:
                    return read_npz(_member_bytes(archive, member), ("poses", "trans", "betas", "gender", "mocap_framerate"))
                seen += 1
    raise ValueError("动作序号超出数据范围")


def load_model(gender, path=MODEL_ARCHIVE):
    if gender not in ("male", "female", "neutral"):
        raise ValueError("模型性别标签无法识别")
    target = gender + "/model.npz"
    with tarfile.open(path, mode="r|xz") as archive:
        for member in archive:
            if member.name.replace("\\", "/").lstrip("./") == target and member.isfile():
                return read_npz(_member_bytes(archive, member), ("v_template", "shapedirs", "J_regressor", "kintree_table"))
    raise ValueError("未找到相应模型")


def skeleton_frames(motion, model):
    poses, trans, betas = (motion[k] for k in ("poses", "trans", "betas"))
    v, directions, regressor, tree = (model[k] for k in ("v_template", "shapedirs", "J_regressor", "kintree_table"))
    if (poses.ndim != 2 or poses.shape[0] < 6 or poses.shape[1] != 156
            or trans.shape != (len(poses), 3) or betas.ndim != 1 or betas.shape[0] < 16
            or v.ndim != 2 or v.shape[1] != 3 or directions.shape != (len(v), 3, 16)
            or regressor.shape != (52, len(v)) or tree.shape != (2, 52)):
        raise ValueError("动作或模型维度不匹配")
    if any(not np.isfinite(a).all() for a in (poses, trans, betas, v, directions, regressor)):
        raise ValueError("动作或模型包含非有限值")
    ids = [int(x) for x in tree[1]]
    if ids != list(range(52)):
        raise ValueError("模型关节编号不符合预期")
    parents = [-1] + [int(x) for x in tree[0, 1:]]
    if any(parent < 0 or parent >= i for i, parent in enumerate(parents) if i):
        raise ValueError("模型关节树无效")
    rest = regressor @ (v + np.einsum("vck,k->vc", directions, betas[:16]))
    chosen = [round(i * (len(poses) - 1) / 5) for i in range(6)]
    joints = [Joint(BODY_NAMES[i], None if i == 0 else parents[i], (0, 0, 0),
                    ["Xposition", "Yposition", "Zposition"]) for i in range(22)]
    frames = []
    for idx in chosen:
        xyz = forward_kinematics(rest.tolist(), parents, poses[idx].reshape(52, 3).tolist(), trans[idx].tolist())[:22]
        # AMASS 世界坐标按常用 Z-up 显示；米转换为厘米以适配已有绘图尺度。
        points = [(x * 100, z * 100, -y * 100) for x, y, z in xyz]
        frame = []
        for i, point in enumerate(points):
            parent = joints[i].parent
            origin = points[parent] if parent is not None else (0, 0, 0)
            frame.extend(point[k] - origin[k] for k in range(3))
        frames.append(frame)
    return joints, frames, chosen


def main():
    parser = argparse.ArgumentParser(description="本地生成 AMASS 骨架，绝不联网")
    parser.add_argument("--index", type=int, default=0, help="数据包内动作序号，从 0 开始")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        motion = load_motion(index=args.index)
        gender = str(motion["gender"].item()).lower()
        model = load_model(gender)
        joints, frames, chosen = skeleton_frames(motion, model)
        fps = float(motion["mocap_framerate"].item())
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("采样频率无效")
        image = render_frames(joints, frames, draw_root=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(image)
        print(json.dumps({"status": "generated_locally", "frames_total": len(motion["poses"]),
                          "selected_frame_indices": chosen, "fps": fps, "joint_count": len(joints),
                          "output": str(args.output), "network": False,
                          "notice": "Z-up 显示假设待核对；未含 DMPL 和蒙皮。仅本地研究，不构成安全判断。"}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, tarfile.TarError) as exc:
        print("本地骨架生成未完成：", type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
