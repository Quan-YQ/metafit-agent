"""FusionPoser 单序列 IMU→骨架研究基线；不适用于实际训练指导。"""

import argparse
from array import array
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from render_motion_sample import Joint, render_frames, world_positions


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fusionposer"
X_FILE = DATA / "motion_0625180826_x.npz"
Y_FILE = DATA / "motion_0625180826_y.npz"
X_SHA = "59a1e0bce5ad3157137b43b7f41e3ca05d366272e93519976e5f8762233cebc9"
Y_SHA = "4dc960fcb05a83d3ec7b6997f134c0686ed638603ec1b15e6a4c4b23e4940ac6"
OUTPUT = ROOT / "outputs" / "fusionposer"
ALPHA = 20.0
GAP = 30
WINDOW_INDEX = 6  # 论文 X_t 的第 7 个时间点为 t；数据本身已预处理。
JOINT_NAMES = ("Root", "Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
               "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot",
               "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase")
PARENTS = (None, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)


class NpyMatrix:
    """只接受受限的 NPY v1.0、二维、little-endian float64，绝不反序列化 pickle。"""
    def __init__(self, raw: bytes, expected_columns: int):
        if len(raw) < 10 or len(raw) > 32 * 1024 * 1024 or raw[:8] != b"\x93NUMPY\x01\x00":
            raise ValueError("不支持的数据格式或文件大小")
        header_size = struct.unpack_from("<H", raw, 8)[0]
        if not 1 <= header_size <= 4096 or 10 + header_size > len(raw):
            raise ValueError("数据头长度无效")
        try:
            header = ast.literal_eval(raw[10:10 + header_size].decode("latin1").strip())
        except (SyntaxError, ValueError, UnicodeError):
            raise ValueError("数据头无效") from None
        if (not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}
                or header["descr"] != "<f8" or header["fortran_order"] is not False):
            raise ValueError("不支持的数据类型或布局")
        shape = header["shape"]
        if (not isinstance(shape, tuple) or len(shape) != 2 or type(shape[0]) is not int
                or not 1 <= shape[0] <= 100000 or type(shape[1]) is not int or shape[1] != expected_columns):
            raise ValueError("数据维度无效")
        payload = raw[10 + header_size:]
        if len(payload) != shape[0] * shape[1] * 8:
            raise ValueError("数据长度不匹配")
        values = array("d")
        if values.itemsize != 8:
            raise ValueError("平台不支持 float64")
        values.frombytes(payload)
        if sys.byteorder != "little":
            values.byteswap()
        if any(not math.isfinite(v) for v in values):
            raise ValueError("数据包含非有限值")
        self.rows, self.columns, self.values = shape[0], shape[1], values

    def row(self, index: int):
        if type(index) is not int or not 0 <= index < self.rows:
            raise IndexError("帧索引越界")
        offset = index * self.columns
        return self.values[offset:offset + self.columns]


def load_pair(x_path: Path = X_FILE, y_path: Path = Y_FILE):
    x_raw, y_raw = x_path.read_bytes(), y_path.read_bytes()
    if hashlib.sha256(x_raw).hexdigest() != X_SHA or hashlib.sha256(y_raw).hexdigest() != Y_SHA:
        raise ValueError("样本摘要不匹配，拒绝使用未核对的数据")
    x, y = NpyMatrix(x_raw, 430), NpyMatrix(y_raw, 146)
    if x.rows != y.rows or x.rows < 100:
        raise ValueError("传感器与参考样本无法按行配对")
    return x, y


def solve_linear(matrix, targets):
    """带部分选主元的线性方程组；矩阵奇异时失败。"""
    n = len(matrix)
    if not n or len(targets) != n:
        raise ValueError("线性系统维度无效")
    width = len(targets[0])
    a = [list(matrix[i]) + list(targets[i]) for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) < 1e-12:
            raise ValueError("线性系统不可解")
        a[column], a[pivot] = a[pivot], a[column]
        scale = a[column][column]
        for j in range(column, n + width):
            a[column][j] /= scale
        for row in range(n):
            if row == column:
                continue
            scale = a[row][column]
            if scale:
                for j in range(column, n + width):
                    a[row][j] -= scale * a[column][j]
    return [row[n:] for row in a]


def train_ridge(inputs, targets, alpha=ALPHA):
    if not inputs or len(inputs) != len(targets) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("训练数据无效")
    feature_count, target_count, n = len(inputs[0]), len(targets[0]), len(inputs)
    if (feature_count < 1 or target_count < 1 or any(len(row) != feature_count for row in inputs)
            or any(len(row) != target_count for row in targets)
            or any(not math.isfinite(v) for row in inputs for v in row)
            or any(not math.isfinite(v) for row in targets for v in row)):
        raise ValueError("训练数据维度或数值无效")
    means = [sum(row[j] for row in inputs) / n for j in range(feature_count)]
    scales = [math.sqrt(sum((row[j] - means[j]) ** 2 for row in inputs) / n) for j in range(feature_count)]
    scales = [value if value > 1e-10 else 1.0 for value in scales]
    n_features = feature_count + 1
    gram = [[0.0] * n_features for _ in range(n_features)]
    cross = [[0.0] * target_count for _ in range(n_features)]
    for x, y in zip(inputs, targets):
        z = [1.0] + [(x[j] - means[j]) / scales[j] for j in range(feature_count)]
        for i in range(n_features):
            for j in range(n_features):
                gram[i][j] += z[i] * z[j]
            for k in range(target_count):
                cross[i][k] += z[i] * y[k]
    for i in range(1, n_features):
        gram[i][i] += alpha
    return means, scales, solve_linear(gram, cross)


def predict(model, features):
    means, scales, weights = model
    if len(features) != len(means) or any(not math.isfinite(v) for v in features):
        raise ValueError("传感器输入无效")
    z = [1.0] + [(features[i] - means[i]) / scales[i] for i in range(len(means))]
    return [sum(z[i] * weights[i][j] for i in range(len(z))) for j in range(len(weights[0]))]


def feature_row(row):
    start = WINDOW_INDEX * 43
    return list(row[start:start + 43])


def mean_joint_error(predictions, targets):
    if not predictions or len(predictions) != len(targets):
        raise ValueError("评估数据无效")
    total = 0.0
    for pred, target in zip(predictions, targets):
        if len(pred) != 60 or len(target) != 60:
            raise ValueError("关节坐标维度无效")
        for k in range(0, 60, 3):
            total += math.dist(pred[k:k + 3], target[k:k + 3])
    return total / (len(predictions) * 20)


def make_render_frames(positions, head_heights):
    """20 个相对根节点关节加根节点；用头部高度估计竖直平移。"""
    joints = [Joint(name, PARENTS[i], (0, 0, 0), ["Xposition", "Yposition", "Zposition"])
              for i, name in enumerate(JOINT_NAMES)]
    frames = []
    for row, head_height in zip(positions, head_heights):
        if len(row) != 60 or not math.isfinite(head_height) or any(not math.isfinite(v) for v in row):
            raise ValueError("关节坐标无效")
        # p5 是头部；其余 p2..p21 顺序依据论文图 4，坐标以根为中心。
        root_y = head_height - row[3 * 3 + 1]
        absolute = [(0.0, root_y, 0.0)] + [(row[i], row[i + 1] + root_y, row[i + 2]) for i in range(0, 60, 3)]
        frame = []
        for i, point in enumerate(absolute):
            parent = PARENTS[i]
            reference = absolute[parent] if parent is not None else (0.0, 0.0, 0.0)
            frame.extend(point[k] - reference[k] for k in range(3))
        frames.append(frame)
    if len(frames) != 6:
        raise ValueError("绘图需要六帧")
    return joints, frames


def predicted_hip_reference(joints, frames):
    """只比较预测骨架的几何变化；不是人体安全或动作质量标签。"""
    poses = [world_positions(joints, frame) for frame in frames]
    span = max(point[1] for pose in poses for point in pose) - min(point[1] for pose in poses for point in pose)
    if not math.isfinite(span) or span <= 1e-12:
        raise ValueError("预测骨架竖直范围无效")
    ratio = (poses[-1][0][1] - poses[0][0][1]) / span
    return {"direction": "up" if ratio >= 0.05 else "down" if ratio <= -0.05 else "no_clear_change",
            "change_over_skeleton_y_span": round(ratio, 4), "threshold": 0.05,
            "note": "来自预测骨架的工程对照，不是人体真值、训练质量或安全标签。"}


def experiment():
    x, y = load_pair()
    train_end = int(x.rows * 0.6)
    test_start = train_end + GAP
    if x.rows - test_start < 6:
        raise ValueError("测试样本不足")
    train_x = [feature_row(x.row(i)) for i in range(train_end)]
    train_y = [list(y.row(i)[:60]) for i in range(train_end)]
    test_x = [feature_row(x.row(i)) for i in range(test_start, x.rows)]
    test_y = [list(y.row(i)[:60]) for i in range(test_start, x.rows)]
    model = train_ridge(train_x, train_y)
    predictions = [predict(model, row) for row in test_x]
    mean_pose = [sum(row[j] for row in train_y) / len(train_y) for j in range(60)]
    chosen = [round(i * (len(test_x) - 1) / 5) for i in range(6)]
    predicted_positions = [predictions[i] for i in chosen]
    reference_positions = [test_y[i] for i in chosen]
    head_heights = [test_x[i][0] for i in chosen]
    joints, pred_frames = make_render_frames(predicted_positions, head_heights)
    _, ref_frames = make_render_frames(reference_positions, head_heights)
    report = {
        "schema": "fusionposer-single-sequence-research-1",
        "data": "FusionPoser real: six Xsens IMUs and OptiTrack reference",
        "x_sha256": X_SHA, "y_sha256": Y_SHA, "rows": x.rows,
        "train_rows": train_end, "gap_rows": GAP, "test_rows": len(test_x),
        "features": "43 维：当前窗口位置假设为第 7 帧，含头部高度、6 个 IMU 加速度和方向；数据已预处理",
        "predictor": "岭回归，alpha=20；单序列前段训练、后段测试",
        "test_mean_joint_error_cm": round(mean_joint_error(predictions, test_y), 3),
        "constant_pose_baseline_error_cm": round(mean_joint_error([mean_pose] * len(test_y), test_y), 3),
        "selected_original_rows": [test_start + i for i in chosen],
        "predicted_hip_reference": predicted_hip_reference(joints, pred_frames),
        "notice": "公开数据单序列实验，需头部高度；坐标布局依据论文与样本结构推断。"
                  "非跨人验证、非真实设备接入、非临床或训练质量判断。",
    }
    return report, render_frames(joints, pred_frames, draw_root=True), render_frames(joints, ref_frames, draw_root=True)


def main():
    parser = argparse.ArgumentParser(description="运行公开 IMU 样本的离线预测基线")
    parser.add_argument("--save-images", action="store_true", help="将预测和参考骨架图保存到 outputs/fusionposer")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        report, predicted, reference = experiment()
        if args.save_images:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            # 不覆盖已有研究结果。
            for name, image in (("imu_predicted.png", predicted), ("mocap_reference.png", reference)):
                with (OUTPUT / name).open("xb") as handle:
                    handle.write(image)
            report["images"] = [str(OUTPUT / "imu_predicted.png"), str(OUTPUT / "mocap_reference.png")]
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, IndexError, OverflowError) as exc:
        print(f"实验未完成：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
