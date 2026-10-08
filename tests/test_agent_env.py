# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The agent's environment and tools: no tokens, checked commands, writes only in the stage dir."""
import os
import shutil
import time

import pytest

from orchard.agent import Tools, agent_env

# Every test here may start real bash, so every one runs with stub tools first on PATH.
pytestmark = pytest.mark.usefixtures("stub_tools")

TOKENS = {"HF_TOKEN": "hf_supervisor_secret", "HUGGING_FACE_HUB_TOKEN": "hf_other_secret",
          "GH_TOKEN": "ghp_supervisor_secret", "GITHUB_TOKEN": "ghs_supervisor_secret",
          "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock", "AWS_SECRET_ACCESS_KEY": "aws_secret"}


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run" / "stages" / "0" / "evidence").mkdir(parents=True)
    return tmp_path / "run"


def tools(run, **kw):
    return Tools(run, run / "stages" / "0", agent_env(run), **kw)


def test_the_environment_is_an_allow_list_with_a_home_inside_the_run(run):
    env = agent_env(run, source={"PATH": "/usr/bin", "HOME": "/home/alice", "LANG": "C.UTF-8", **TOKENS})
    assert not set(TOKENS) & set(env)
    assert env["PATH"] == "/usr/bin" and env["LANG"] == "C.UTF-8"
    assert env["HOME"] == str(run.resolve() / "home") and os.path.isdir(env["HOME"])
    assert env["HF_TOKEN_PATH"].startswith(env["HOME"])


def test_agent_shells_get_a_device_mask_that_matches_no_chip(run):
    # UNVERIFIED on hardware: how UMD treats a mask that names no chip.
    env = agent_env(run, source={"PATH": "/usr/bin", "TT_VISIBLE_DEVICES": "0000:01:00.0",
                                 "TT_METAL_VISIBLE_DEVICES": "0"})
    assert env["TT_VISIBLE_DEVICES"] == "0000:ff:00.0"
    assert env["TT_METAL_VISIBLE_DEVICES"] == "0000:ff:00.0"
    assert tools(run).shell("printenv TT_VISIBLE_DEVICES").splitlines()[1] == "0000:ff:00.0"
    for name in ("TT_VISIBLE_DEVICES", "TT_METAL_VISIBLE_DEVICES"):
        with pytest.raises(ValueError, match="device mask"):
            agent_env(run, extra={name: "0000:01:00.0"})


def test_git_over_ssh_uses_no_key_and_no_config(run):
    # OpenSSH finds ~/.ssh from the passwd entry and ignores HOME, so HOME alone does not hide keys.
    assert agent_env(run)["GIT_SSH_COMMAND"] == (
        "ssh -F /dev/null -o IdentitiesOnly=yes -o IdentityFile=/dev/null -o BatchMode=yes")


def test_extra_variables_that_look_like_credentials_are_refused(run):
    assert agent_env(run, extra={"HF_HOME": "/mnt/models/hf"})["HF_HOME"] == "/mnt/models/hf"
    for name in ("HF_TOKEN", "OPENAI_API_KEY", "GH_AUTH", "DB_PASSWORD"):
        with pytest.raises(ValueError, match="credential"):
            agent_env(run, extra={name: "x"})


def test_a_shell_command_finds_no_token_in_its_environment_or_home(run, tmp_path, monkeypatch):
    for k, v in TOKENS.items():
        monkeypatch.setenv(k, v)
    real_home = tmp_path / "real-home"
    (real_home / ".cache" / "huggingface").mkdir(parents=True)
    (real_home / ".cache" / "huggingface" / "token").write_text("hf_planted_secret")
    monkeypatch.setenv("HOME", str(real_home))
    t = tools(run)
    out = t.shell("env")
    assert out.startswith("exit 0") and "secret" not in out and "ssh-agent" not in out
    out = t.shell("cat ~/.cache/huggingface/token")
    assert out.startswith("exit 1") and "hf_planted_secret" not in out


def test_agent_shells_in_these_tests_find_only_stub_tools(run, stub_tools):
    # The tests below run real bash. If a runner rule regressed, a refused command would run, so
    # every tool that could touch hardware, a remote or a lease must resolve to a stub here.
    path = agent_env(run)["PATH"]
    for name in stub_tools.NAMES:
        found = shutil.which(name, path=path)
        assert found == str(stub_tools.dir / name), (name, found)


def test_a_refused_command_never_starts(run, stub_tools):
    t = tools(run)
    out = t.shell("touch started.txt && git push origin main")
    assert out.startswith("refused:") and "git push" in out
    assert not (run / "started.txt").exists()
    assert t.shell("rm -f /nonexistent-orchard-dir/x").startswith("refused:")
    assert stub_tools.calls() == []          # no stub ran, so no real tool could have run either


def test_shell_runs_in_the_run_directory(run):
    assert os.path.realpath(run) in tools(run).shell("pwd")


def test_a_command_past_its_timeout_is_killed(run):
    t0 = time.monotonic()
    out = tools(run, timeout=0.5).shell("sleep 30")
    assert out.startswith("killed after 0.5 s") and time.monotonic() - t0 < 10


def test_long_output_keeps_its_head_and_tail(run):
    out = tools(run, limit=100).shell("seq 1 5000")
    assert out.splitlines()[1] == "1" and out.splitlines()[-1] == "5000" and "characters cut" in out


def test_write_file_stays_inside_the_stage_directory(run, tmp_path):
    t = tools(run)
    assert t.write_file("evidence/a.txt", "x").startswith("wrote 1 characters to stages/0/evidence/a.txt")
    (tmp_path / "outside").mkdir()
    os.symlink(tmp_path / "outside", run / "stages" / "0" / "link")
    for path in ("../escape.txt", str(tmp_path / "abs.txt"), "link/x.txt", "."):
        assert t.write_file(path, "x").startswith("refused:"), path
    assert not list((tmp_path / "outside").iterdir())
    assert not (run / "stages" / "escape.txt").exists()
    assert t.write_file("ledger.jsonl", "{}").startswith("refused: the ledger")


def test_a_write_without_a_path_says_a_path_is_needed(run):
    """Coder-Next sometimes sent write_file with no path; the answer said "None is outside your stage
    directory", which told it nothing it could act on."""
    t = tools(run)
    for path in (None, "", "  "):
        out = t.write_file(path, "x")
        assert out.startswith("error: write_file needs a path") and "None" not in out, path


def test_writing_the_same_content_again_changes_nothing_and_says_how_to_finish(run):
    """On the first lab run the agent wrote handoff.json, then wrote it again unchanged until the
    watchdog stopped the run. The answer to a repeat says nothing changed and how to end the step."""
    t = tools(run)
    assert t.write_file("handoff.json", "{}").startswith("wrote ")
    f = run / "stages" / "0" / "handoff.json"
    before = f.stat().st_mtime_ns
    out = t.write_file("handoff.json", "{}")
    assert out.startswith("unchanged: stages/0/handoff.json already holds exactly this content")
    assert "no tool call" in out and not out.startswith("wrote ")
    assert f.stat().st_mtime_ns == before
    assert t.write_file("handoff.json", '{"a": 1}').startswith("wrote ")


def test_bad_tool_calls_get_an_answer_and_raise_nothing(run):
    t = tools(run)
    assert t.call("shell", "not json").startswith("error:")
    assert t.call("shell", "[1]").startswith("error:")
    assert t.call("browse", "{}").startswith("error: there is no tool")
    assert t.call("shell", '{"command": ""}').startswith("error:")
