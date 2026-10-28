"""Issue 124: a malformed telegraph.endpoint (or another value the loader hands to the system) is a ConfigError that names
the key. It is never a bare ValueError, because bringup_config.load() translates only ConfigError."""
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import bringup_config, cli, telegraph as tg
from orchard import telegraph_cmd as cmd

TOML = """
[telegraph]
enabled = true
endpoint = "https://llm.example.org/v1"
models = ["small", "big"]
key_env = "TG_KEY"
"""
BRINGUP = """
runs_root = "runs"

[coder]
target = "mando2222/qwen3.8-27b-dflash2-p300x2-q4kv"
port = 8000
chips = 4
"""

BAD_ENDPOINTS = ["https://[::1", "https://host:99999/v1", "https://host:abc/v1", "https://host:-1/v1",
                 "https://host\x00/v1", "https://ho\x07st/v1", "https://host/v1\x7f", "https://host /v1",
                 "https://host/v1\n"]


def _toml_with(endpoint: str) -> str:
    # TOML basic strings take \u escapes, so control characters reach the loader as they would from a file
    escaped = "".join(c if (c.isprintable() and c != '"') else "\\u%04x" % ord(c) for c in endpoint)
    return TOML.replace("https://llm.example.org/v1", escaped)


def _write(tmp_path, text, name="tg.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.parametrize("endpoint", BAD_ENDPOINTS)
def test_a_malformed_endpoint_is_a_config_error_naming_the_key(endpoint, tmp_path):
    with pytest.raises(tg.ConfigError, match=r"telegraph\.endpoint") as info:
        tg.load({"enabled": True, "endpoint": endpoint, "models": ["m"]}, base=tmp_path)
    assert type(info.value) is tg.ConfigError


@pytest.mark.parametrize("endpoint", BAD_ENDPOINTS)
def test_a_malformed_endpoint_in_a_bringup_config_is_a_bringup_config_error(endpoint, tmp_path):
    path = _write(tmp_path, BRINGUP + _toml_with(endpoint), "bringup.toml")
    with pytest.raises(bringup_config.BringupConfigError, match=r"telegraph\.endpoint"):
        bringup_config.load(path)


def test_the_error_shows_the_offending_value(tmp_path):
    with pytest.raises(tg.ConfigError) as info:
        tg.load({"enabled": True, "endpoint": "https://[::1", "models": ["m"]}, base=tmp_path)
    assert "https://[::1" in str(info.value)


def test_the_shown_value_hides_a_password_in_the_url(tmp_path):
    with pytest.raises(tg.ConfigError) as info:
        tg.load({"enabled": True, "endpoint": "https://bob:hunter2@[::1", "models": ["m"]}, base=tmp_path)
    assert "hunter2" not in str(info.value) and "telegraph.endpoint" in str(info.value)


@pytest.mark.parametrize("endpoint", ["https://bob:hunter2@host：443/v1", "https://bob:hunter2@host:abc/v1",
                                      "https://bob:hunter2@host:99999/v1", "https://bob:hunter2@[" + "a" * 5000])
def test_no_parser_message_leaks_a_password_or_the_whole_value(endpoint, tmp_path):
    with pytest.raises(tg.ConfigError) as info:
        tg.load({"enabled": True, "endpoint": endpoint, "models": ["m"]}, base=tmp_path)
    assert "hunter2" not in str(info.value) and len(str(info.value)) < 400
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_the_shown_value_is_bounded_in_length(tmp_path):
    with pytest.raises(tg.ConfigError) as info:
        tg.load({"enabled": True, "endpoint": "https://[" + "a" * 5000, "models": ["m"]}, base=tmp_path)
    assert len(str(info.value)) < 400


@pytest.mark.parametrize("endpoint", ["https://host:8443/v1", "https://[::1]:8443/v1", "http://localhost:8000/v1",
                                      "https://llm.example.org/v1"])
def test_a_good_endpoint_with_or_without_a_port_still_loads(endpoint, tmp_path):
    assert tg.load({"enabled": True, "endpoint": endpoint, "models": ["m"]}, base=tmp_path).endpoint == endpoint


@pytest.mark.parametrize("key, value", [("key_file", "a\x00b"), ("dir", "~no-such-user-124/x")])
def test_a_path_value_the_system_cannot_read_is_a_config_error_naming_the_key(key, value, tmp_path):
    table = {"enabled": True, "endpoint": "https://h/v1", "models": ["m"], key: value}
    with pytest.raises(tg.ConfigError, match=f"telegraph.{key}") as info:
        tg.load(table, base=tmp_path)
    assert type(info.value) is tg.ConfigError


def test_the_dir_of_a_disabled_table_with_an_unreadable_path_does_not_raise(tmp_path):
    tg.load({"enabled": False, "dir": "~no-such-user-124/x"}, base=tmp_path)


def test_check_of_a_malformed_endpoint_names_the_key_and_exits_2(tmp_path):
    cfg = _write(tmp_path, TOML.replace("https://llm.example.org/v1", "https://[::1"))
    out = io.StringIO()
    code = cmd.main(cli._parser().parse_args(["telegraph", "check", "--telegraph-config", str(cfg)]), out=out,
                    environ={"TG_KEY": "k" * 20}, home=tmp_path)
    assert code == cmd.EXIT_PROBLEM == 2
    assert "telegraph.endpoint" in out.getvalue()
    assert "Traceback" not in out.getvalue()


def test_the_command_line_exits_non_zero_for_a_malformed_endpoint(tmp_path):
    """Through the real entry point in a subprocess, so the exit status is the process's own."""
    cfg = _write(tmp_path, TOML.replace("https://llm.example.org/v1", "https://[::1"))
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run([sys.executable, "-m", "orchard.cli", "telegraph", "check", "--telegraph-config", str(cfg)],
                          cwd=root, capture_output=True, text=True, env={**os.environ, "HOME": str(tmp_path)})
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "telegraph.endpoint" in proc.stdout
