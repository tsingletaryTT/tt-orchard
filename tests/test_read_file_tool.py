# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The read_file tool (orchard/agent.py): a read-only way to look at a text file in the run directory.

Qwen3-Coder-Next calls a `read_file` tool by habit (4 of 50 replayed turns in the role-fit test, where the
27B used `shell` with `cat`). Without the tool each such call costs a turn and an error message. The tool is
stricter than `cat`: it never leaves the run directory, even through a symlink."""
import json
import os
from pathlib import Path

import pytest

from orchard import context
from orchard.agent import TOOL_SCHEMAS, Tools


@pytest.fixture
def tools(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    stage.mkdir(parents=True)
    return Tools(run, stage, {"PATH": os.environ["PATH"]}, limit=200), run, stage


def read(t, **args):
    return t.call("read_file", json.dumps(args))


def test_the_schema_lists_the_three_tools_with_a_required_path():
    assert [s["function"]["name"] for s in TOOL_SCHEMAS] == ["shell", "read_file", "write_file"]
    spec = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "read_file")["function"]
    assert spec["parameters"]["required"] == ["path"]


def test_a_path_relative_to_the_run_directory_is_read(tools):
    t, run, stage = tools
    (stage / "evidence").mkdir()
    (stage / "evidence" / "a.txt").write_text("measured 0.94")
    assert read(t, path="stages/2/evidence/a.txt") == "measured 0.94"


def test_a_path_relative_to_the_stage_directory_is_read_when_the_run_relative_one_is_missing(tools):
    t, run, stage = tools
    (stage / "handoff.json").write_text('{"goal": "x"}')
    assert read(t, path="handoff.json") == '{"goal": "x"}'


def test_the_run_relative_file_wins_when_both_exist(tools):
    t, run, stage = tools
    (run / "notes.txt").write_text("run")
    (stage / "notes.txt").write_text("stage")
    assert read(t, path="notes.txt") == "run"


@pytest.mark.parametrize("path", ["../outside.txt", "stages/../../outside.txt", "/etc/hostname"])
def test_a_path_outside_the_run_directory_is_refused(tools, tmp_path, path):
    (tmp_path / "outside.txt").write_text("secret")
    t, run, stage = tools
    out = read(t, path=path)
    assert out.startswith("refused:") and "run directory" in out and "secret" not in out


def test_a_symlink_that_leaves_the_run_directory_is_refused(tools, tmp_path):
    (tmp_path / "outside.txt").write_text("secret")
    t, run, stage = tools
    (run / "link.txt").symlink_to(tmp_path / "outside.txt")
    out = read(t, path="link.txt")
    assert out.startswith("refused:") and "secret" not in out


def test_a_symlink_that_stays_inside_the_run_directory_is_followed(tools):
    t, run, stage = tools
    (run / "real.txt").write_text("inside")
    (run / "link.txt").symlink_to(run / "real.txt")
    assert read(t, path="link.txt") == "inside"


def test_a_missing_file_says_where_it_looked(tools):
    t, run, stage = tools
    out = read(t, path="nope.txt")
    assert out.startswith("error:") and "nope.txt" in out and "stage directory" in out


def test_a_directory_is_an_error_that_lists_nothing(tools):
    t, run, stage = tools
    (stage / "evidence").mkdir()
    out = read(t, path="stages/2/evidence")
    assert out.startswith("error:") and "directory" in out


@pytest.mark.parametrize("args", [{}, {"path": ""}, {"path": 3}, {"path": None}])
def test_a_missing_or_wrong_path_argument_is_an_error(tools, args):
    t, run, stage = tools
    assert read(t, **args).startswith("error:")


def test_a_long_file_is_clipped_to_the_limit_and_says_so(tools):
    t, run, stage = tools
    (run / "big.txt").write_text("x" * 5000)
    out = read(t, path="big.txt")
    assert len(out) < 5000 and "characters cut" in out and out.startswith("x") and out.endswith("x")


def test_a_file_that_is_not_utf8_is_read_with_replacement_characters(tools):
    t, run, stage = tools
    (run / "bin.dat").write_bytes(b"ok \xff\xfe end")
    assert read(t, path="bin.dat").startswith("ok ")


def test_reading_never_changes_a_file(tools):
    t, run, stage = tools
    (run / "a.txt").write_text("keep")
    before = (run / "a.txt").stat().st_mtime_ns
    read(t, path="a.txt")
    assert (run / "a.txt").read_text() == "keep" and (run / "a.txt").stat().st_mtime_ns == before


def test_the_unknown_tool_message_lists_all_three_tools(tools):
    t, run, stage = tools
    out = t.call("browse", "{}")
    assert "shell, read_file and write_file" in out


def test_the_rules_the_agent_reads_name_read_file_and_say_it_is_read_only():
    rules = " ".join(context.RULES.split())
    assert "shell, read_file and write_file" in rules
    assert "read_file reads a text file inside the run directory" in rules


def test_a_tilde_is_not_expanded_so_a_home_path_cannot_be_reached(tools):
    t, run, stage = tools
    out = read(t, path="~/.ssh/id_rsa")
    assert out.startswith("error:") and "does not exist" in out
