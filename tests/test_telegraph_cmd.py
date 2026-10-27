"""`tt-orchard telegraph check | ask | report | usage`, and the [telegraph] table of bringup.toml."""
import io
import json
import os
import stat
from pathlib import Path

import pytest

from orchard import bringup_config, cli, telegraph as tg
from orchard import telegraph_cmd as cmd

KEY = "k3y-VALUE-0123456789"
TOML = """
[telegraph]
enabled = true
endpoint = "https://llm.example.org/v1"
models = ["small", "big"]
key_env = "TG_KEY"
"""


def write(tmp_path, text=TOML, name="tg.toml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def run(argv, tmp_path, *, transport=None, environ=None, getter=None):
    out = io.StringIO()
    env = {"TG_KEY": KEY} if environ is None else environ
    code = cmd.main(cli._parser().parse_args(["telegraph", *argv]), out=out, environ=env, transport=transport,
                    getter=getter, home=tmp_path)
    return code, out.getvalue()


def ok_transport(body_text="try fill"):
    seen = []

    def send(url, headers, body, timeout):
        seen.append((url, json.loads(body)))
        return 200, json.dumps({"choices": [{"message": {"content": body_text}}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
    send.seen = seen
    return send


# ------------------------------------------------------------------------------------------------ config table

BRINGUP = """
runs_root = "runs"

[coder]
target = "mando2222/qwen3.8-27b-dflash2-p300x2-q4kv"
port = 8000
chips = 4
"""


def test_a_bringup_config_may_hold_a_telegraph_table_and_it_is_loaded(tmp_path):
    cfg = bringup_config.load(write(tmp_path, BRINGUP + TOML, "bringup.toml"))
    assert isinstance(cfg.telegraph, tg.Config) and cfg.telegraph.enabled and cfg.telegraph.models == ("small", "big")


def test_a_bringup_config_without_the_table_has_telegraph_off(tmp_path):
    assert bringup_config.load(write(tmp_path, BRINGUP, "bringup.toml")).telegraph.enabled is False


def test_a_bad_telegraph_table_in_the_bringup_config_is_a_config_error_naming_it(tmp_path):
    with pytest.raises(bringup_config.BringupConfigError, match="telegraph.*modles"):
        bringup_config.load(write(tmp_path, BRINGUP + TOML.replace("models", "modles"), "bringup.toml"))


def test_a_relative_key_file_in_the_bringup_config_is_relative_to_the_config(tmp_path):
    toml = TOML.replace('key_env = "TG_KEY"', 'key_file = "secrets/key"')
    cfg = bringup_config.load(write(tmp_path, BRINGUP + toml, "bringup.toml"))
    assert cfg.telegraph.key_file == (tmp_path / "secrets" / "key").resolve()


def test_the_table_is_read_from_a_standalone_file(tmp_path):
    cfg = cmd.load_config(write(tmp_path), None)
    assert cfg.enabled and cfg.models == ("small", "big")


def test_a_missing_or_disabled_table_loads_as_off(tmp_path):
    assert cmd.load_config(write(tmp_path, "x = 1\n"), None).enabled is False


def test_a_bad_table_is_an_error_with_the_key_named(tmp_path):
    with pytest.raises(tg.ConfigError, match="modles"):
        cmd.load_config(write(tmp_path, TOML.replace("models", "modles")), None)


# ------------------------------------------------------------------------------------------------ check

def test_check_says_what_is_set_and_never_prints_the_key(tmp_path):
    code, text = run(["check", "--telegraph-config", str(write(tmp_path))], tmp_path)
    assert code == 0 and KEY not in text
    for needle in ("enabled: yes", "https://llm.example.org/v1", "small, big", "TG_KEY", "key: present"):
        assert needle in text, needle


def test_check_makes_no_request_without_ping(tmp_path):
    def boom(*a, **k):
        raise AssertionError("a request was made")
    code, _ = run(["check", "--telegraph-config", str(write(tmp_path))], tmp_path, transport=boom, getter=boom)
    assert code == 0


def test_check_of_a_missing_key_exits_2_and_says_which_variable(tmp_path):
    code, text = run(["check", "--telegraph-config", str(write(tmp_path))], tmp_path, environ={})
    assert code == 2 and "TG_KEY is not set" in text


def test_check_of_a_disabled_telegraph_says_so_and_exits_0(tmp_path):
    code, text = run(["check", "--telegraph-config", str(write(tmp_path, "[telegraph]\nenabled = false\n"))], tmp_path)
    assert code == 0 and "enabled: no" in text


def test_check_warns_about_a_key_file_other_users_can_read(tmp_path):
    keyfile = tmp_path / "key"
    keyfile.write_text(KEY)
    keyfile.chmod(0o644)
    toml = TOML.replace('key_env = "TG_KEY"', f'key_file = "{keyfile}"')
    code, text = run(["check", "--telegraph-config", str(write(tmp_path, toml))], tmp_path, environ={})
    assert code == 0 and "warning" in text and "readable by other users" in text and KEY not in text
    keyfile.chmod(0o600)
    assert "readable by other users" not in run(["check", "--telegraph-config", str(write(tmp_path, toml))], tmp_path)[1]


def test_check_ping_lists_the_models_and_marks_each_configured_one(tmp_path):
    def getter(url, headers, timeout):
        assert url == "https://llm.example.org/v1/models" and headers["Authorization"] == f"Bearer {KEY}"
        return 200, json.dumps({"data": [{"id": "small"}, {"id": "other"}]}).encode()
    code, text = run(["check", "--ping", "--telegraph-config", str(write(tmp_path))], tmp_path, getter=getter)
    assert code == 2 and "small: listed" in text and "big: NOT listed" in text and KEY not in text


def test_check_ping_with_every_model_listed_exits_0(tmp_path):
    def getter(url, headers, timeout):
        return 200, json.dumps({"data": [{"id": "small"}, {"id": "big"}]}).encode()
    assert run(["check", "--ping", "--telegraph-config", str(write(tmp_path))], tmp_path, getter=getter)[0] == 0


@pytest.mark.parametrize("answer", [(401, b"no"), (200, b"not json")])
def test_check_ping_that_fails_exits_2_with_a_reason(tmp_path, answer):
    code, text = run(["check", "--ping", "--telegraph-config", str(write(tmp_path))], tmp_path,
                     getter=lambda u, h, t: answer)
    assert code == 2 and "ping failed" in text


def test_check_ping_of_a_transport_error_exits_2_without_a_traceback(tmp_path):
    def getter(u, h, t):
        raise OSError("unreachable")
    code, text = run(["check", "--ping", "--telegraph-config", str(write(tmp_path))], tmp_path, getter=getter)
    assert code == 2 and "ping failed" in text and "OSError" in text


# ------------------------------------------------------------------------------------------------ ask, report, usage

def first_call_id(directory):
    """The id of the first call in a ledger (the lines before it are budget reservations)."""
    for line in (Path(directory) / "telegraph.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "call_id" in row:
            return row["call_id"]


def files(tmp_path):
    (tmp_path / "task.txt").write_text("write a softmax kernel")
    (tmp_path / "fail.txt").write_text("Deadlock detected")
    return ["--task-file", str(tmp_path / "task.txt"), "--failure-file", str(tmp_path / "fail.txt")]


def test_ask_prints_the_reply_as_untrusted_advice_and_one_usage_line(tmp_path):
    t = ok_transport()
    code, text = run(["ask", "--problem", "softmax", *files(tmp_path), "--dir", str(tmp_path / "d"),
                      "--telegraph-config", str(write(tmp_path))], tmp_path, transport=t)
    assert code == 0 and "untrusted text" in text and "try fill" in text
    assert "call " in text and "small" in text and "100 in" in text and "20 out" in text and KEY not in text
    assert "Deadlock detected" in t.seen[0][1]["messages"][-1]["content"]


def test_ask_of_a_disabled_telegraph_exits_3_and_sends_nothing(tmp_path):
    t = ok_transport()
    code, text = run(["ask", "--problem", "p", *files(tmp_path), "--dir", str(tmp_path / "d"),
                      "--telegraph-config", str(write(tmp_path, "x = 1\n"))], tmp_path, transport=t)
    assert code == 3 and "not enabled" in text and t.seen == []


def test_ask_that_fails_exits_3_with_the_reason(tmp_path):
    code, text = run(["ask", "--problem", "p", *files(tmp_path), "--dir", str(tmp_path / "d"),
                      "--telegraph-config", str(write(tmp_path))], tmp_path, transport=ok_transport(), environ={})
    assert code == 3 and "TG_KEY" in text


def test_ask_refuses_a_task_file_that_is_missing(tmp_path):
    code, text = run(["ask", "--problem", "p", "--task-file", str(tmp_path / "nope"), "--failure-file",
                      str(tmp_path / "nope"), "--telegraph-config", str(write(tmp_path))], tmp_path)
    assert code == 2 and "cannot read" in text


def test_report_and_usage_round_trip(tmp_path):
    d, cfg = str(tmp_path / "d"), str(write(tmp_path))
    run(["ask", "--problem", "p", *files(tmp_path), "--dir", d, "--telegraph-config", cfg], tmp_path,
        transport=ok_transport())
    call_id = first_call_id(tmp_path / "d")
    assert run(["report", call_id, "--helped", "--dir", d, "--telegraph-config", cfg], tmp_path)[0] == 0
    code, text = run(["usage", "--dir", d, "--telegraph-config", cfg], tmp_path)
    assert code == 0 and "calls: 1" in text and "helped: 1" in text and "tokens: 100 in, 20 out" in text
    code, text = run(["usage", "--json", "--dir", d, "--telegraph-config", cfg], tmp_path)
    assert json.loads(text)["helped"] == 1


def test_report_of_an_unknown_call_exits_2(tmp_path):
    code, text = run(["report", "nope", "--not-helped", "--dir", str(tmp_path / "d"),
                      "--telegraph-config", str(write(tmp_path))], tmp_path)
    assert code == 2 and "no call nope" in text


def test_the_default_directory_is_under_the_cache_and_usage_of_nothing_is_zero(tmp_path):
    code, text = run(["usage", "--telegraph-config", str(write(tmp_path))], tmp_path)
    assert code == 0 and "calls: 0" in text
    assert cmd.default_dir(tmp_path) == tmp_path / ".cache" / "tt-orchard" / "telegraph"


def test_ledger_files_hold_no_credential_after_a_whole_session(tmp_path):
    d, cfg = tmp_path / "d", str(write(tmp_path))
    run(["ask", "--problem", "p", *files(tmp_path), "--dir", str(d), "--telegraph-config", cfg], tmp_path,
        transport=ok_transport(f"the key {KEY} is shown"))
    for p in d.rglob("*"):
        if p.is_file():
            assert KEY not in p.read_text() and stat.S_IMODE(p.stat().st_mode) == 0o600


def test_report_not_helped_is_recorded_as_not_helped(tmp_path):
    d, cfg = str(tmp_path / "d"), str(write(tmp_path))
    run(["ask", "--problem", "p", *files(tmp_path), "--dir", d, "--telegraph-config", cfg], tmp_path,
        transport=ok_transport())
    call_id = first_call_id(tmp_path / "d")
    assert run(["report", call_id, "--not-helped", "--dir", d, "--telegraph-config", cfg], tmp_path)[0] == 0
    text = run(["usage", "--dir", d, "--telegraph-config", cfg], tmp_path)[1]
    assert "helped: 0, did not help: 1" in text and "problem p: 1 calls, 0 helped" in text
    assert "model small: 1 calls" in text


def test_a_ping_failure_never_prints_the_key_even_when_the_error_carries_it(tmp_path):
    def getter(u, h, t):
        raise OSError(f"rejected {KEY}")
    code, text = run(["check", "--ping", "--telegraph-config", str(write(tmp_path))], tmp_path, getter=getter)
    assert code == 2 and "ping failed" in text and KEY not in text


def test_a_missing_config_file_exits_2_with_the_reason(tmp_path):
    code, text = run(["check", "--telegraph-config", str(tmp_path / "absent.toml")], tmp_path)
    assert code == 2 and "cannot read" in text


def test_the_command_is_reachable_from_the_cli_entry_point(tmp_path):
    out = io.StringIO()
    code = cli.main(["telegraph", "usage", "--telegraph-config", str(write(tmp_path)), "--dir", str(tmp_path / "d")],
                    env={"HOME": str(tmp_path)}, stdout=out)
    assert code == 0 and "calls: 0" in out.getvalue()


def test_ping_of_an_anthropic_configuration_uses_that_apis_headers(tmp_path):
    toml = TOML.replace('endpoint = "https://llm.example.org/v1"', 'endpoint = "https://llm.example.org"\napi = "anthropic"')
    seen = {}

    def getter(url, headers, timeout):
        seen.update(url=url, headers=headers)
        return 200, json.dumps({"data": [{"id": "small"}, {"id": "big"}]}).encode()
    code, _ = run(["check", "--ping", "--telegraph-config", str(write(tmp_path, toml))], tmp_path, getter=getter)
    assert code == 0 and seen["url"] == "https://llm.example.org/v1/models"
    assert seen["headers"]["x-api-key"] == KEY and seen["headers"]["anthropic-version"] == tg.ANTHROPIC_VERSION
    assert "Authorization" not in seen["headers"]


def test_report_that_cannot_write_exits_with_an_error_and_does_not_say_recorded(tmp_path, monkeypatch):
    d, cfg = str(tmp_path / "d"), str(write(tmp_path))
    run(["ask", "--problem", "p", *files(tmp_path), "--dir", d, "--telegraph-config", cfg], tmp_path,
        transport=ok_transport())
    call_id = first_call_id(tmp_path / "d")
    monkeypatch.setattr(tg.Telegraph, "_append", lambda self, rec: False)
    code, text = run(["report", call_id, "--helped", "--dir", d, "--telegraph-config", cfg], tmp_path)
    assert code == 2 and "recorded" not in text and "could not" in text


def test_the_suggested_report_command_carries_the_directory_and_the_config(tmp_path):
    d, cfg = str(tmp_path / "my dir"), str(write(tmp_path))
    code, text = run(["ask", "--problem", "p", *files(tmp_path), "--dir", d, "--telegraph-config", cfg], tmp_path,
                     transport=ok_transport())
    line = next(l for l in text.splitlines() if "telegraph report" in l)
    import shlex
    assert f"--dir {shlex.quote(d)}" in line and f"--telegraph-config {shlex.quote(cfg)}" in line


def test_the_suggested_report_command_names_the_default_directory_when_none_was_given(tmp_path):
    cfg = str(write(tmp_path))
    code, text = run(["ask", "--problem", "p", *files(tmp_path), "--telegraph-config", cfg], tmp_path,
                     transport=ok_transport())
    line = next(l for l in text.splitlines() if "telegraph report" in l)
    assert f"--dir {cmd.default_dir(tmp_path)}" in line


def test_a_config_path_with_a_space_is_quoted_in_the_suggested_command(tmp_path):
    sub = tmp_path / "my cfg"
    sub.mkdir()
    cfg = str(write(sub))
    code, text = run(["ask", "--problem", "p", *files(tmp_path), "--telegraph-config", cfg], tmp_path,
                     transport=ok_transport())
    line = next(l for l in text.splitlines() if "telegraph report" in l)
    import shlex
    assert f"--telegraph-config {shlex.quote(cfg)}" in line and "'" in line


def test_a_paid_answer_that_could_not_be_recorded_shows_the_warning_and_no_report_command(tmp_path, monkeypatch):
    real = tg.Telegraph._append

    def fail_settle(self, rec):
        return False if rec.get("outcome") == "ok" else real(self, rec)
    monkeypatch.setattr(tg.Telegraph, "_append", fail_settle)
    d, cfg = str(tmp_path / "d"), str(write(tmp_path))
    code, text = run(["ask", "--problem", "p", *files(tmp_path), "--dir", d, "--telegraph-config", cfg], tmp_path,
                     transport=ok_transport())
    assert code == 0 and "try fill" in text and "could not record" in text
    assert "telegraph report" not in text
