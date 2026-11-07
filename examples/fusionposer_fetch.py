"""核对或显式下载固定版本的公开研究样本；默认不联网。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

from fusionposer_cross_record import DATA, SAMPLES


BASE = "https://raw.githubusercontent.com/LuzyCat/FusionPoser/e8ba6f89b2f6cc031b37c55dd72fc674192df86f/Dataset/real/"
MAX_BYTES = 32 * 1024 * 1024


def check_or_fetch(data_dir=DATA, *, download=False, opener=urlopen):
    if download:
        data_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for name, (identifier, x_hash, y_hash) in SAMPLES.items():
        for suffix, expected in (("x", x_hash), ("y", y_hash)):
            filename = f"motion_{identifier}_{suffix}.npz"
            path = data_dir / filename
            state = "missing"
            if path.exists():
                if path.stat().st_size > MAX_BYTES:
                    raise ValueError("已有样本过大")
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError("已有样本摘要不符，不会覆盖")
                state = "verified_existing"
            elif download:
                with opener(BASE + filename, timeout=60) as response:
                    raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError("下载样本摘要不符")
                with path.open("xb") as handle:
                    handle.write(raw)
                state = "downloaded_and_verified"
            report.append({"recording": name, "file": filename, "status": state})
    return report


def main():
    parser = argparse.ArgumentParser(description="检查或获取 FusionPoser 公开研究样本")
    parser.add_argument("--download", action="store_true", help="从固定公开地址下载缺失样本，不自动重试")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        items = check_or_fetch(download=args.download)
        print(json.dumps({"network_enabled": args.download, "files": items}, ensure_ascii=False, indent=2))
        return 1 if any(item["status"] == "missing" for item in items) else 0
    except (OSError, ValueError):
        print("样本检查未完成；已有文件不会覆盖，下载失败不会自动重试。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
