"""数据获取显式联网、校验摘要且不覆盖已有文件。"""
import hashlib
import io
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import fusionposer_fetch as fetch


def configure(monkeypatch):
    digest = hashlib.sha256(b"known").hexdigest()
    monkeypatch.setattr(fetch, "SAMPLES", {"sample": ("abc", digest, digest)})


def test_offline_check_never_opens_network_or_creates_data(tmp_path, monkeypatch):
    configure(monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected network")
    data = tmp_path / "data"
    report = fetch.check_or_fetch(data, opener=forbidden)
    assert [r["status"] for r in report] == ["missing", "missing"]
    assert not data.exists()


def test_download_validates_and_preserves_existing_file(tmp_path, monkeypatch):
    configure(monkeypatch)
    calls = []
    def fake(url, timeout):
        calls.append(url)
        return io.BytesIO(b"known")
    report = fetch.check_or_fetch(tmp_path, download=True, opener=fake)
    assert len(calls) == 2 and all(r["status"] == "downloaded_and_verified" for r in report)
    report = fetch.check_or_fetch(tmp_path, download=True, opener=lambda *a, **k: pytest.fail("not needed"))
    assert all(r["status"] == "verified_existing" for r in report)
    (tmp_path / "motion_abc_x.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        fetch.check_or_fetch(tmp_path, download=True, opener=fake)
    assert (tmp_path / "motion_abc_x.npz").read_bytes() == b"changed"


def test_bad_download_does_not_write_file(tmp_path, monkeypatch):
    configure(monkeypatch)
    with pytest.raises(ValueError):
        fetch.check_or_fetch(tmp_path, download=True, opener=lambda *a, **k: io.BytesIO(b"wrong"))
    assert list(tmp_path.iterdir()) == []
