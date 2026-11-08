"""通过真实命令验证入口及配置边界。"""

import os
import subprocess
import sys


def run_entrypoint(level: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if level is None:
        env.pop("METAFIT_LOG_LEVEL", None)
    else:
        env["METAFIT_LOG_LEVEL"] = level
    return subprocess.run(
        [sys.executable, "-m", "metafit_agent"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_missing_optional_configuration_uses_default():
    result = run_entrypoint(None)
    assert result.returncode == 0
    assert "INFO" in result.stdout
    assert "python -m metafit_agent.simulation_api" in result.stdout


def test_valid_configuration_is_accepted():
    result = run_entrypoint("debug")
    assert result.returncode == 0
    assert "DEBUG" in result.stdout


def test_invalid_configuration_fails_without_echoing_its_value():
    result = run_entrypoint("unexpected-sensitive-value")
    assert result.returncode != 0
    assert "METAFIT_LOG_LEVEL" in result.stderr
    assert "unexpected-sensitive-value" not in result.stderr
