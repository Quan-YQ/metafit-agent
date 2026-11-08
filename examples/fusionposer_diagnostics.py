"""FusionPoser 离线数据诊断与分录制嵌套验证；不用于训练指导。"""
import json
import math
import statistics
import sys

from fusionposer_cross_record import load_recording
from fusionposer_research import PARENTS, feature_row, mean_joint_error, predict, train_ridge


RECORDINGS = ("train", "test_1", "test_2")
ALPHAS = (1.0, 20.0, 200.0, 2000.0)


def percentile(values, q):
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError("分位数输入无效")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def features_and_targets(pair):
    x, y = pair
    if x.rows < 1 or x.rows != y.rows:
        raise ValueError("数据行无法配对")
    return ([feature_row(x.row(i)) for i in range(x.rows)],
            [list(y.row(i)[:60]) for i in range(y.rows)])


def data_diagnostics(records):
    training_features, _ = records["train"]
    head = [row[0] for row in training_features]
    lower, upper = min(head), max(head)
    output = []
    for name in RECORDINGS:
        features, targets = records[name]
        heights = [row[0] for row in features]
        output.append({"recording": name, "frames": len(features),
                       "head_height_median": round(statistics.median(heights), 3),
                       "head_height_p01_p99": [round(percentile(heights, q), 3) for q in (.01, .99)],
                       "outside_train_head_range_percent": round(100 * sum(v < lower or v > upper for v in heights) / len(heights), 2)})
    return {"training_head_range": [round(lower, 3), round(upper, 3)], "recordings": output,
            "notice": "数值遵循数据文件原始尺度；头部高度解释依据论文，具体字段仍须作者确认。"}


def layout_diagnostics(pairs):
    output = []
    for name in RECORDINGS:
        _, y = pairs[name]
        quaternion_fraction = {}
        for offset in (60, 62):
            count = 0
            for i in range(y.rows):
                row = y.row(i)
                for start in range(offset, offset + 84, 4):
                    norm = math.sqrt(sum(row[j] ** 2 for j in range(start, start + 4)))
                    count += abs(norm - 1) < .01
            quaternion_fraction[str(offset)] = round(count / (y.rows * 21), 4)
        bone_lengths = [[] for _ in range(20)]
        for i in range(y.rows):
            row = y.row(i)
            points = [(0, 0, 0)] + [tuple(row[j:j + 3]) for j in range(0, 60, 3)]
            for j in range(1, 21):
                bone_lengths[j - 1].append(math.dist(points[j], points[PARENTS[j]]))
        deviations = [statistics.pstdev(lengths) for lengths in bone_lengths]
        output.append({"recording": name, "quaternion_unit_norm_fraction_by_offset": quaternion_fraction,
                       "median_bone_length_std": round(statistics.median(deviations), 8),
                       "max_bone_length_std": round(max(deviations), 8)})
    return {"recordings": output,
            "notice": "单位四元数与骨长稳定只是字段布局的支持证据，不能证明字段名称、身份或标签绝对正确。"}


def combined(records, names):
    inputs, targets = [], []
    for name in names:
        x, y = records[name]
        inputs.extend(x)
        targets.extend(y)
    return inputs, targets


def evaluate_fold(records, holdout, alphas=ALPHAS):
    if holdout not in RECORDINGS or not alphas or any(not math.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("验证参数无效")
    other = [name for name in RECORDINGS if name != holdout]
    train_name, validation_name = other
    train_x, train_y = records[train_name]
    valid_x, valid_y = records[validation_name]
    scores = []
    for alpha in alphas:
        model = train_ridge(train_x, train_y, alpha=alpha)
        error = mean_joint_error([predict(model, row) for row in valid_x], valid_y)
        scores.append((error, alpha))
    selected = min(scores)[1]
    train_x, train_y = combined(records, other)
    model = train_ridge(train_x, train_y, alpha=selected)
    baseline = [sum(row[j] for row in train_y) / len(train_y) for j in range(60)]
    test_x, test_y = records[holdout]
    error = mean_joint_error([predict(model, row) for row in test_x], test_y)
    base_error = mean_joint_error([baseline] * len(test_y), test_y)
    return {"test_recording": holdout, "selection_train_recording": train_name,
            "validation_recording": validation_name,
            "validation_errors_by_alpha": {str(a): round(e, 3) for e, a in scores},
            "selected_alpha": selected, "refit_train_frames": len(train_x), "test_frames": len(test_x),
            "test_error": round(error, 3), "baseline_error": round(base_error, 3)}


def run():
    pairs = {name: load_recording(name) for name in RECORDINGS}
    records = {name: features_and_targets(pair) for name, pair in pairs.items()}
    return {"schema": "fusionposer-diagnostics-1", "network": False,
            "data_diagnostics": data_diagnostics(records), "layout_diagnostics": layout_diagnostics(pairs),
            "nested_recording_validation": [evaluate_fold(records, name) for name in RECORDINGS],
            "notice": "小样本探索：此前已查看这些录制，不是未接触的独立最终测试；身份未知，不用于训练指导。"}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, IndexError, OverflowError):
        print("诊断未完成：请检查公开样本与摘要。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
