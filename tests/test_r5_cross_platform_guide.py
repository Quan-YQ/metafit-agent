"""本机用 POSIX sh/curl 和 .http 样例验证无需 PowerShell 的调用。"""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import subprocess
from threading import Thread
from uuid import uuid4

import pytest

from test_r5_simulation_api import service

ROOT = Path(__file__).resolve().parents[1]


def test_sh_curl_guide_commands(service, tmp_path):
    shell = shutil.which("sh")
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not shell or not curl:
        pytest.skip("未找到 POSIX sh 或 curl")
    guide = (ROOT / "docs" / "R5_App对接指南.md").read_text(encoding="utf-8")
    blocks = re.findall(r"<!-- bash-smoke:begin -->\s*```bash\s*\n(.*?)```\s*<!-- bash-smoke:end -->", guide, re.S)
    assert len(blocks) == 1
    blocks[0] = blocks[0].replace("127.0.0.1:8000", f"127.0.0.1:{service[1].server_port}")
    script = tmp_path / "guide.sh"
    script.write_text("set -e\n" + "\n".join(blocks), encoding="utf-8")
    proc = subprocess.run([shell, str(script)], cwd=ROOT, capture_output=True, timeout=30)
    output = proc.stdout.decode("utf-8", errors="replace")
    assert proc.returncode == 0, (output, proc.stderr.decode("utf-8", errors="replace"))
    assert "PASS: simulation contract and capabilities" in output
    assert " 200 " in output and " 201 " in output
    assert '"code": "pending_confirmation"' in output and '"event_saved": true' in output


@pytest.mark.parametrize("version,simulation,real_model,delivery,status,expected", [
    ("0.3.1-draft", True, False, False, 200, True),
    ("0.4.0-draft", True, False, False, 200, False),
    ("0.3.0-draft", False, False, False, 200, False),
    ("0.3.0-draft", True, True, False, 200, False),
    ("0.3.0-draft", True, False, True, 200, False),
    ("0.3.0-draft", 1, False, False, 200, False),
    ("0.3.0-draft", True, 0, False, 200, False),
    ("0.3.0-draft", True, False, False, 503, False),
])
def test_sh_gate_prevents_writes(tmp_path, version, simulation, real_model, delivery, status, expected):
    shell = shutil.which("sh")
    if not shell or not (shutil.which("curl") or shutil.which("curl.exe")):
        pytest.skip("未找到 POSIX sh 或 curl")
    writes = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps({"meta": {"contract_version": version, "simulation": simulation},
                               "features": {"real_model": real_model, "delivery": delivery}}).encode()
            self.send_response(status)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            writes.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        guide = (ROOT / "docs" / "R5_App对接指南.md").read_text(encoding="utf-8")
        block = re.search(r"<!-- bash-smoke:begin -->\s*```bash\s*\n(.*?)```", guide, re.S).group(1)
        block = block.replace("127.0.0.1:8000", f"127.0.0.1:{server.server_port}")
        script = tmp_path / "gate.sh"
        script.write_text(block, encoding="utf-8")
        result = subprocess.run([shell, str(script)], cwd=ROOT, capture_output=True, timeout=30)
        assert (result.returncode == 0) is expected, result.stderr.decode("utf-8", errors="replace")
        assert len(writes) == (2 if expected else 0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_collection_requests(service):
    template = (ROOT / "examples/r5_api_requests.http").read_text(encoding="utf-8")
    sid = "http_" + uuid4().hex
    chunks = [x.strip() for x in re.split(r"(?m)^###.*$", template)[1:] if x.strip()]
    assert len(chunks) == 6
    observations = []
    for chunk in chunks:
        request_part, _, body = chunk.partition("\n\n")
        lines = request_part.splitlines()
        method, url = lines[0].split(" ", 1)
        path = url.replace("{{base}}", "/sim/v0").replace("{{sid}}", sid).replace("{{user}}", "http_user")
        data = body.replace("{{sid}}", sid).replace("{{user}}", "http_user").strip()
        connection = http.client.HTTPConnection("127.0.0.1", service[1].server_port, timeout=5)
        connection.request(method, path, body=data.encode("utf-8") if data else None,
                           headers={"Content-Type": "application/json"} if data else {})
        response = connection.getresponse()
        value = json.loads(response.read())
        observations.append((response.status, value))
        connection.close()
    assert [status for status, _ in observations] == [200, 201, 200, 200, 400, 400]
    meta = observations[0][1]
    assert meta["meta"]["simulation"] is True and meta["meta"]["contract_version"] == "0.3.0-draft"
    assert meta["features"]["real_model"] is False and meta["features"]["delivery"] is False
    assert observations[2][1]["code"] == "pending_confirmation"
    assert observations[3][1]["code"] == "duplicate_event"
    assert observations[4][1]["code"] == "safety_report_unattributed"
    assert observations[5][1]["code"] == "safety_report_unattributed" and observations[5][1]["event_saved"] is False
