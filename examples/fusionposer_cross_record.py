"""跨录制研究对照：一段训练，另外两段只用于测试；不联网。"""
import hashlib
import json
from pathlib import Path
import sys

from fusionposer_research import NpyMatrix, feature_row, mean_joint_error, predict, train_ridge


DATA = Path(__file__).resolve().parents[1] / "data/fusionposer"
SAMPLES = {
    "train": ("0625180826", "59a1e0bce5ad3157137b43b7f41e3ca05d366272e93519976e5f8762233cebc9", "4dc960fcb05a83d3ec7b6997f134c0686ed638603ec1b15e6a4c4b23e4940ac6"),
    "test_1": ("0625181142", "e16d71c0fa22fa68416a48ee3e33bc827f740a6d2133dbb508f05fb69776ba2c", "0b9e10b696618c650f538d3d43149151eb7750f716ac3591c1155264c376e392"),
    "test_2": ("0625181240", "72f875584c346705e5d1ef4830644741c853b49f3246920797601b255ec46cc8", "244e8787ebfd18599433ecc463d3acbe634416fc20de6b775ee3675db17ae3b7"),
}


def load_recording(name, data_dir=DATA):
    if name not in SAMPLES:
        raise ValueError("未知研究样本")
    identifier, x_hash, y_hash = SAMPLES[name]
    matrices = []
    for suffix, digest, columns in (("x", x_hash, 430), ("y", y_hash, 146)):
        raw = (data_dir / f"motion_{identifier}_{suffix}.npz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("数据摘要不符")
        matrices.append(NpyMatrix(raw, columns))
    if matrices[0].rows != matrices[1].rows or matrices[0].rows < 100:
        raise ValueError("数据行无法配对")
    return tuple(matrices)


def fit(train_pairs):
    if not train_pairs or any(x.rows < 1 or x.rows != y.rows for x, y in train_pairs):
        raise ValueError("训练数据行无法配对")
    inputs = [feature_row(x.row(i)) for x, y in train_pairs for i in range(x.rows)]
    labels = [list(y.row(i)[:60]) for x, y in train_pairs for i in range(y.rows)]
    if not labels or len(inputs) != len(labels):
        raise ValueError("训练数据无效")
    model = train_ridge(inputs, labels)
    baseline = [sum(row[j] for row in labels) / len(labels) for j in range(60)]
    return model, baseline


def score(model, baseline, test_pairs):
    results = []
    for name, (test_x, test_y) in test_pairs:
        if not test_x.rows or test_x.rows != test_y.rows:
            raise ValueError("测试数据无效")
        predictions = [predict(model, feature_row(test_x.row(i))) for i in range(test_x.rows)]
        targets = [list(test_y.row(i)[:60]) for i in range(test_y.rows)]
        results.append({"recording": name, "frames": test_x.rows,
                        "prediction_error_cm": round(mean_joint_error(predictions, targets), 3),
                        "constant_pose_baseline_cm": round(mean_joint_error([baseline] * len(targets), targets), 3)})
    return results


def evaluate(train_pair, test_pairs):
    model, baseline = fit([train_pair])
    return {"train_frames": train_pair[0].rows, "tests": score(model, baseline, test_pairs),
            "notice": "训练与测试为不同录制，受试者身份未知；坐标布局按论文及样本推断，不能代表跨人、临床或动作安全性能。"}


def leave_one_record_out(records):
    if len(records) < 2 or len({name for name, _ in records}) != len(records):
        raise ValueError("至少需要两个名称不同的录制")
    results = []
    for holdout, test in records:
        training = [pair for name, pair in records if name != holdout]
        model, baseline = fit(training)
        result = score(model, baseline, [(holdout, test)])[0]
        result["train_frames"] = sum(x.rows for x, _ in training)
        results.append(result)
    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        train = load_recording("train")
        tests = [(name, load_recording(name)) for name in ("test_1", "test_2")]
        report = evaluate(train, tests)
        report["leave_one_record_out"] = leave_one_record_out([("train", train)] + tests)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, IndexError, OverflowError):
        print("跨录制实验未完成：请检查研究样本和摘要。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
