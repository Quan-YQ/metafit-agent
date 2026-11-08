"""Fixed recording-level research comparison, offline; not a clinical model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "fusionposer"
OLD = DEFAULT_DATA
NEW = DEFAULT_DATA
REFERENCE_EXAMPLES = Path(__file__).resolve().parent
OLD_SHA256 = {
    "0625180826": ("59a1e0bce5ad3157137b43b7f41e3ca05d366272e93519976e5f8762233cebc9", "4dc960fcb05a83d3ec7b6997f134c0686ed638603ec1b15e6a4c4b23e4940ac6"),
    "0625181142": ("e16d71c0fa22fa68416a48ee3e33bc827f740a6d2133dbb508f05fb69776ba2c", "0b9e10b696618c650f538d3d43149151eb7750f716ac3591c1155264c376e392"),
    "0625181240": ("72f875584c346705e5d1ef4830644741c853b49f3246920797601b255ec46cc8", "244e8787ebfd18599433ecc463d3acbe634416fc20de6b775ee3675db17ae3b7"),
}
NEW_GIT = {
    "0625174933": ("7f8c54e58c5db24ff935576e652b636c25ff4a27", "aa74608f55130e6a91a2e4523d4116c738bef8e7"),
    "0625175157": ("85f65cc0fa5954ab8abd34b22f1faba0b925703d", "094c997be56125cc31b3b7af5857fda85f69465d"),
    "0625175446": ("7d8f89918b4d247e28d3539d41189e8096b20cd5", "086bc2c5bd3bcf8c0a81fdb5e04d949491af55ba"),
    "0625175813": ("ef894e79c5a97435499d558ffaeaa93d2a6b52cb", "484d514436f01a118d95b99a8090ce13f830a170"),
}
OLD_TRAIN = ("0625180826", "0625181142", "0625181240")
ADDITIONAL_TRAIN = ("0625175446", "0625175813")
TEST = ("0625174933", "0625175157")
ALPHA = 200.0

def read_pair(identifier):
    directory = OLD if identifier in OLD_SHA256 else NEW
    matrices = []
    for i, suffix in enumerate(("x", "y")):
        path = directory / f"motion_{identifier}_{suffix}.npz"
        raw = path.read_bytes()
        if identifier in OLD_SHA256:
            actual = hashlib.sha256(raw).hexdigest()
            expected = OLD_SHA256[identifier][i]
        else:
            actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            expected = NEW_GIT[identifier][i]
        if actual != expected:
            raise ValueError(f"文件摘要不匹配: {path.name}")
        # The upstream files have .npz names but are raw NPY arrays.
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
        if arr.dtype != np.dtype("<f8") or arr.ndim != 2 or arr.shape[1] != (430, 146)[i] or not np.isfinite(arr).all():
            raise ValueError(f"数据格式或数值无效: {path.name}")
        matrices.append(arr)
    x, y = matrices
    if x.shape[0] != y.shape[0] or x.shape[0] < 100:
        raise ValueError(f"配对行数无效: {identifier}")
    return np.array(x[:, 6*43:7*43]), np.array(y[:, :60])

def fit(records, names):
    x = np.concatenate([records[n][0] for n in names])
    y = np.concatenate([records[n][1] for n in names])
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales <= 1e-10] = 1.0
    z = np.column_stack((np.ones(len(x)), (x - means) / scales))
    penalty = np.diag([0.0] + [ALPHA] * 43)
    weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return means, scales, weights, y.mean(axis=0), len(x)

def predictions(model, x):
    means, scales, weights, _, _ = model
    return np.column_stack((np.ones(len(x)), (x-means)/scales)) @ weights

def measure(model, x, y):
    pred = predictions(model, x)
    distance = np.linalg.norm((pred-y).reshape(-1, 20, 3), axis=2)
    baseline = np.linalg.norm((model[3]-y).reshape(-1,20,3), axis=2)
    return {"frames": len(x), "mean_error": round(float(distance.mean()),3),
            "baseline_error": round(float(baseline.mean()),3),
            "frame_mean_error_p95": round(float(np.percentile(distance.mean(axis=1),95)),3),
            "per_point_mean_error": [round(float(v),3) for v in distance.mean(axis=0)]}

def independent_check(records):
    sys.path.insert(0, str(REFERENCE_EXAMPLES))
    from fusionposer_research import train_ridge, predict, mean_joint_error
    x, y = records[OLD_TRAIN[0]]
    x, y = x[:150], y[:150]
    python_model = train_ridge(x.tolist(), y.tolist(), alpha=ALPHA)
    numpy_model = fit({"subset": (x, y)}, ["subset"])
    p = predictions(numpy_model, x[:20])
    p_python = np.array([predict(python_model, row.tolist()) for row in x[:20]])
    delta = float(np.max(np.abs(p-p_python)))
    metric_delta = abs(float(np.linalg.norm((p-y[:20]).reshape(-1,20,3),axis=2).mean()) - mean_joint_error(p.tolist(), y[:20].tolist()))
    if delta > 1e-7 or metric_delta > 1e-10:
        raise ValueError("独立实现结果不一致")
    return {"max_prediction_difference": delta, "metric_difference": metric_delta}

def main():
    global OLD, NEW, REFERENCE_EXAMPLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--new-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-examples", type=Path, default=REFERENCE_EXAMPLES)
    args = parser.parse_args()
    OLD, NEW, REFERENCE_EXAMPLES = args.old_dir, args.new_dir, args.reference_examples
    all_ids = OLD_TRAIN + ADDITIONAL_TRAIN + TEST
    records = {n: read_pair(n) for n in all_ids}
    check = independent_check(records)
    original, expanded = fit(records, OLD_TRAIN), fit(records, OLD_TRAIN + ADDITIONAL_TRAIN)
    result = {"schema":"fusionposer-expanded-recording-1", "split_predefined": True,
              "training_original":list(OLD_TRAIN), "training_additional":list(ADDITIONAL_TRAIN),
              "test":list(TEST), "alpha": ALPHA, "numpy_version":np.__version__,
              "source_commit":"e8ba6f89b2f6cc031b37c55dd72fc674192df86f",
              "identity":"unknown", "check":check,
              "training_frames":{"original":original[4],"expanded":expanded[4]},
              "scores":{n:{"original":measure(original,*records[n]),"expanded":measure(expanded,*records[n])} for n in TEST},
              "notice":"Original and expanded models evaluated on the same held-out recordings. Units and field interpretation not independently confirmed; no cross-person or clinical claim."}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
