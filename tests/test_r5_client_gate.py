"""示例客户端必须在任何写操作前检查版本及模拟能力。"""
from io import BytesIO
import json
import sys

import pytest

from examples import r5_api_client as client


@pytest.mark.parametrize("change", ["version", "simulation", "real_model", "delivery", "missing_features"])
def test_client_rejects_untrusted_meta_before_writes(monkeypatch, change):
    meta = {"meta": {"contract_version": "0.3.0-draft", "simulation": True},
            "candidate": {"text": None, "may_display": False},
            "features": {"real_model": False, "delivery": False}, "status": "ready", "code": None, "required_action": "none"}
    if change == "version": meta["meta"]["contract_version"] = "0.2.2-draft"
    if change == "simulation": meta["meta"]["simulation"] = False
    if change == "real_model": meta["features"]["real_model"] = True
    if change == "delivery": meta["features"]["delivery"] = True
    if change == "missing_features": del meta["features"]
    calls = []
    def fake_open(req, timeout):
        calls.append(req.get_method())
        return BytesIO(json.dumps(meta).encode("utf-8"))
    monkeypatch.setattr(client, "urlopen", fake_open)
    monkeypatch.setattr(sys, "argv", ["r5_api_client.py"])
    with pytest.raises(SystemExit): client.main()
    assert calls == ["GET"]
