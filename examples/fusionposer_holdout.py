"""固定参数选择流程及单段留出录制：独立研究，不用于训练指导。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

from fusionposer_cross_record import DATA, load_recording
from fusionposer_diagnostics import ALPHAS, RECORDINGS, combined, features_and_targets
from fusionposer_research import NpyMatrix, mean_joint_error, predict, train_ridge


HOLDOUT = {
    "x": ("motion_0625174249_x.npz", 4286320, "bfdd51d7df7dd208ed0f1584193bb1f1da68b7e2", 430),
    "y": ("motion_0625174249_y.npz", 1455408, "8fea642ccccb40a0ea24e4e4c29fdd828ef57c39", 146),
}
BASE = "https://raw.githubusercontent.com/LuzyCat/FusionPoser/e8ba6f89b2f6cc031b37c55dd72fc674192df86f/Dataset/real/"


def git_blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def get_heldout_bytes(directory=DATA, *, download=False, opener=urlopen, stats=None):
    """只接受固定的公开样本与上游 Git 摘要；不覆盖文件，不自动重试。"""
    if download:
        directory.mkdir(parents=True, exist_ok=True)
    blobs = {}
    for role, (filename, size, blob_sha, _) in HOLDOUT.items():
        path = directory / filename
        if path.exists():
            if path.stat().st_size != size:
                raise ValueError("已有留出样本大小不符")
            raw = path.read_bytes()
        elif download:
            if stats is not None:
                stats["requests_started"] += 1
            with opener(BASE + filename, timeout=60) as response:
                raw = response.read(size + 1)
            if len(raw) != size or git_blob_sha(raw) != blob_sha:
                raise ValueError("留出样本下载校验失败")
            with path.open("xb") as handle:
                handle.write(raw)
        else:
            raise FileNotFoundError("尚未准备留出样本")
        if len(raw) != size or git_blob_sha(raw) != blob_sha:
            raise ValueError("留出样本摘要不符")
        blobs[role] = raw
    return blobs


def select_alpha(records):
    """只在三段开发录制上轮流验证；每段权重相同。"""
    if set(records) != set(RECORDINGS):
        raise ValueError("开发录制不完整")
    result = {}
    for alpha in ALPHAS:
        errors = []
        for validation in RECORDINGS:
            train_names = [name for name in RECORDINGS if name != validation]
            x, y = combined(records, train_names)
            val_x, val_y = records[validation]
            model = train_ridge(x, y, alpha=alpha)
            errors.append(mean_joint_error([predict(model, row) for row in val_x], val_y))
        result[alpha] = errors
    selected = min(ALPHAS, key=lambda a: (sum(result[a]) / len(result[a]), a))
    return selected, result


def run(directory=DATA):
    records = {name: features_and_targets(load_recording(name, directory)) for name in RECORDINGS}
    selected, validation = select_alpha(records)
    x, y = combined(records, RECORDINGS)
    model = train_ridge(x, y, alpha=selected)
    baseline = [sum(row[j] for row in y) / len(y) for j in range(60)]
    # 在完成参数选择与训练后才加载留出录制，防止测试数据参与选择。
    raw = get_heldout_bytes(directory)
    pair = (NpyMatrix(raw["x"], HOLDOUT["x"][3]), NpyMatrix(raw["y"], HOLDOUT["y"][3]))
    test_x, test_y = features_and_targets(pair)
    error = mean_joint_error([predict(model, row) for row in test_x], test_y)
    base_error = mean_joint_error([baseline] * len(test_y), test_y)
    return {"schema": "fusionposer-fixed-holdout-1", "network": False,
            "development_recordings": list(RECORDINGS), "development_frames": len(x),
            "validation_errors_by_alpha": {str(a): [round(e, 3) for e in validation[a]] for a in ALPHAS},
            "selected_alpha": selected, "heldout_recording": "0625174249", "heldout_frames": len(test_x),
            "heldout_x_sha256": hashlib.sha256(raw["x"]).hexdigest(),
            "heldout_y_sha256": hashlib.sha256(raw["y"]).hexdigest(),
            "heldout_error": round(error, 3), "heldout_baseline_error": round(base_error, 3),
            "notice": "在原有三段上选择参数后仅评估这一段；非跨人或完整可靠性评估，字段布局仍待确认。"}


def main():
    parser = argparse.ArgumentParser(description="公开 IMU 留出录制的离线研究")
    parser.add_argument("--download-heldout", action="store_true", help="下载固定的公开留出样本；失败不重试")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        stats = {"requests_started": 0}
        if args.download_heldout:
            get_heldout_bytes(download=True, stats=stats)
        report = run()
        report["network"] = stats["requests_started"] > 0
        report["download_requests_started"] = stats["requests_started"]
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, IndexError, OverflowError):
        print("留出实验未完成：请检查公开样本、校验值和网络状态。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
