"""密钥输入检查不能输出密钥，也不能使用网络。"""

import warnings

import pytest

from metafit_agent import check_key_input


@pytest.mark.parametrize("key,expected", [
    ("sk-this-is-a-dummy-key-123", 0),
    (" sk-this-is-a-dummy-key-123 ", 1),
    ("not-a-key", 1),
    ("sk-\x16this-is-a-dummy-key-123", 1),
])
def test_key_input_format_prints_no_value(monkeypatch, capsys, key, expected):
    monkeypatch.setattr(check_key_input.getpass, "getpass", lambda _: key)
    assert check_key_input.main() == expected
    output = capsys.readouterr()
    assert key not in output.out and key not in output.err
    assert "未发起网络请求" in output.out


def test_unsafe_terminal_does_not_read_key(monkeypatch, capsys):
    def unsafe(_):
        warnings.warn("input may be visible", check_key_input.getpass.GetPassWarning)

    monkeypatch.setattr(check_key_input.getpass, "getpass", unsafe)
    assert check_key_input.main() == 2
    assert "无法隐藏" in capsys.readouterr().err
