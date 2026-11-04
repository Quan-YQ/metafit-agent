"""实际执行指南中的 PowerShell 片段，避免交接命令只停留在纸面。"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_r5_simulation_api import service

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shell", ["pwsh", "powershell"])
def test_guide_commands_against_local_simulation(service, tmp_path, shell):
    executable = shutil.which(shell)
    if not executable:
        pytest.skip(f"未安装 {shell}")
    guide = (ROOT / "docs" / "R5_App对接指南.md").read_text(encoding="utf-8")
    blocks = re.findall(r"<!-- smoke:begin -->\s*```powershell\s*\n(.*?)```\s*<!-- smoke:end -->", guide, re.S)
    assert len(blocks) == 10
    blocks[0] = blocks[0].replace("127.0.0.1:8000", f"127.0.0.1:{service[1].server_port}")
    checks = [
        "",
        "if ($created.http -ne 201 -or $created.body.confirmation_state -ne 'not_asked') { throw 'creation' }",
        "if ($opened.body.confirmation_state -ne 'open' -or -not $shown.body.question_presented -or $answered.body.confirmation_state -ne 'confirmed') { throw 'confirmation' }",
        "if ($sent.http -ne 200 -or -not $sent.body.event_saved -or $queried.http -ne 200) { throw 'event' }",
        "if ($duplicate.body.code -ne 'duplicate_event') { throw 'duplicate' }",
        "if ($pending.body.code -ne 'pending_confirmation' -or -not $pending.body.event_saved) { throw 'pending' }",
        "if ($safety.body.status -ne 'safety_pending') { throw 'safety' }",
        "if (($fixtures | Where-Object name -eq 'server_busy').http_status -ne 503) { throw 'fixture' }",
        "if ($missing.http -ne 404 -or $unknownAnswer.http -ne 400 -or $unknownEvent.body.code -ne 'safety_report_unattributed' -or $badKnownAnswer.body.code -ne 'safety_report_unattributed' -or $unknownEvent.body.display.text -notmatch '报告未记录') { throw 'unknown' }",
        "if (-not $versionOK -or $versionBad) { throw 'version' }; "
        "$fakeMeta.body.meta.contract_version='0.3.0-draft'; $fakeMeta.body.meta.simulation=$false; "
        "if (Test-MFEnvironment $fakeMeta) { throw 'simulation flag' }; $fakeMeta.body.meta.simulation=$true; "
        "$fakeMeta.body.features.real_model=$true; if (Test-MFEnvironment $fakeMeta) { throw 'real model flag' }; "
        "$fakeMeta.body.features.real_model=$false; $fakeMeta.body.features.delivery=$true; "
        "if (Test-MFEnvironment $fakeMeta) { throw 'delivery flag' }",
    ]
    source = "$ErrorActionPreference = 'Stop'\n" + "\n".join(block + "\n" + check for block, check in zip(blocks, checks))
    script = tmp_path / "guide.ps1"
    script.write_text(source, encoding="utf-8-sig")
    proc = subprocess.run([executable, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                          cwd=ROOT, capture_output=True, timeout=60)
    assert proc.returncode == 0, (proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace"))
