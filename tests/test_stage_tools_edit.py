# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`edit` (orchard/skills/full-port-templates/stage_tools.py, issue 524): change one exact piece of a port file instead of writing the whole file again.

In run 18 of the tiny Qwen4 rehearsal the coder wrote the whole port file in every repair round (11,728 to 13,638 characters per write, often cut off at the
token cap). `edit FILE --old TEXT --new TEXT` sends only the changed text. Each guard below has a mutation in the PR body that turns its test red."""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from orchard import runaway
from orchard.watchdog import command_shape
from test_stage_tools_last_check import TOOL, load_tool

BODY = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(1, 41))     # 40 small functions, 120 lines


def stage_with(tmp_path, body=BODY):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text(body)
    return stage


def cli(stage, *args, cwd=None):
    shutil.copy2(TOOL, stage / "stage_tools.py")
    return subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), "edit", *args], cwd=cwd or stage.parent.parent,
                          env={"PATH": os.environ.get("PATH", "")}, capture_output=True, text=True, timeout=30)


def record(done):
    """The one JSON line the tool prints for a successful call."""
    lines = [l for l in done.stdout.splitlines() if l.startswith("EDIT_RECORD ")]
    assert len(lines) == 1, done.stdout
    return json.loads(lines[0][len("EDIT_RECORD "):])


def port(stage):
    return (stage / "port" / "port_model.py").read_text()


# ---- the change --------------------------------------------------------------------------------------------

def test_one_exact_match_is_replaced_and_the_rest_of_the_file_is_untouched(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "return x + 7\n", "--new", "return x * 7\n")
    assert done.returncode == 0, done.stdout
    assert port(stage) == BODY.replace("return x + 7\n", "return x * 7\n")


def test_the_call_prints_one_record_with_the_file_the_sizes_and_the_changed_lines(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "return x + 7\n", "--new", "return x * 7\n")
    rec = record(done)
    assert rec["tool"] == "edit" and rec["file"] == "stages/2/port/port_model.py"
    assert rec["bytes_before"] == len(BODY) and rec["bytes_after"] == len(BODY)
    assert rec["lines_changed"] == 1 and rec["chars_old"] == len("return x + 7\n") and rec["chars_new"] == len("return x * 7\n")


def test_a_longer_replacement_reports_the_growth_and_counts_the_larger_side_of_the_changed_lines(tmp_path):
    stage = stage_with(tmp_path)
    new = "y = x\nz = y\nreturn z + 7\n"
    rec = record(cli(stage, "stages/2/port/port_model.py", "--old", "return x + 7\n", "--new", new))
    assert rec["bytes_after"] == len(BODY) - len("return x + 7\n") + len(new) and rec["lines_changed"] == 3


def test_lines_that_stay_the_same_around_the_change_are_not_counted(tmp_path):
    stage = stage_with(tmp_path)
    old = "def f3(x):\n    return x + 3\n"
    new = "def f3(x):\n    return x - 3\n"
    assert record(cli(stage, "stages/2/port/port_model.py", "--old", old, "--new", new))["lines_changed"] == 1


def test_the_file_may_be_named_from_the_stage_directory_or_the_port_directory(tmp_path):
    stage = stage_with(tmp_path)
    assert cli(stage, "port/port_model.py", "--old", "return x + 1\n", "--new", "return 1\n").returncode == 0
    assert cli(stage, "port_model.py", "--old", "return x + 2\n", "--new", "return 2\n").returncode == 0
    assert "return 1\n" in port(stage) and "return 2\n" in port(stage)


def test_a_multi_line_change_comes_from_files(tmp_path):
    stage = stage_with(tmp_path)
    (stage / "old.txt").write_text("def f3(x):\n    return x + 3\n")
    (stage / "new.txt").write_text("def f3(x):\n    y = x\n    return y - 3\n")
    done = cli(stage, "stages/2/port/port_model.py", "--old-file", "stages/2/old.txt", "--new-file", "stages/2/new.txt")
    assert done.returncode == 0, done.stdout
    assert "    y = x\n    return y - 3\n" in port(stage) and port(stage).count("def f3(") == 1


def test_an_empty_new_text_deletes_the_match(tmp_path):
    stage = stage_with(tmp_path)
    assert cli(stage, "port_model.py", "--old", "def f9(x):\n    return x + 9\n\n", "--new", "").returncode == 0
    assert "def f9(" not in port(stage)


@pytest.mark.parametrize("args", [
    ["--old", "a"],                                      # no new text
    ["--new", "a"],                                      # no old text
    ["--old", "a", "--old-file", "x", "--new", "b"],     # both forms of the old text
    ["--old", "a", "--new", "b", "--new-file", "x"],     # both forms of the new text
    ["--old", "", "--new", "b"],                         # an empty old text matches everywhere
    ["--old", "a", "--new", "a"],                        # no change
])
def test_a_call_that_is_not_one_old_text_and_one_new_text_is_refused_and_writes_nothing(tmp_path, args):
    stage = stage_with(tmp_path)
    done = cli(stage, "port_model.py", *args)
    assert done.returncode == 2 and port(stage) == BODY and "EDIT_RECORD" not in done.stdout


# ---- exactly one match -------------------------------------------------------------------------------------

def test_no_match_is_refused_and_names_the_closest_lines(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "port_model.py", "--old", "return x + 70\n", "--new", "return 0\n")     # f7 and f70: close, not equal
    assert done.returncode == 1 and port(stage) == BODY
    assert "does not occur" in done.stdout and "closest" in done.stdout and "line " in done.stdout
    assert "return x + 7" in done.stdout


def test_no_match_says_what_to_change(tmp_path):
    done = cli(stage_with(tmp_path), "port_model.py", "--old", "nothing like it zzz", "--new", "b")
    assert done.returncode == 1 and "--old" in done.stdout and "`show" in done.stdout


def test_two_matches_are_refused_with_their_line_numbers_and_the_file_is_unchanged(tmp_path):
    stage = stage_with(tmp_path, body="a = 1\nb = 2\na = 1\n")
    done = cli(stage, "port_model.py", "--old", "a = 1", "--new", "a = 3")
    assert done.returncode == 1 and port(stage) == "a = 1\nb = 2\na = 1\n"
    assert "2 places" in done.stdout and "lines 1, 3" in done.stdout and "more of the surrounding" in done.stdout


def test_overlapping_matches_count_as_more_than_one(tmp_path):
    stage = stage_with(tmp_path, body="aaa\n")
    done = cli(stage, "port_model.py", "--old", "aa", "--new", "b")     # at offsets 0 and 1
    assert done.returncode == 1 and port(stage) == "aaa\n"


# ---- inside the port directory -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["../stage_tools.py", "stages/2/stage_tools.py", "stages/2/port/../port_check.py", "../../ledger.jsonl", "/etc/hostname",
                                  "stages/2/port/../handoff.json", "stages/1/port/port_model.py", "stages/2/port", "port", ""])
def test_a_file_outside_the_port_directory_is_refused_and_nothing_under_the_run_changes(tmp_path, name):
    stage = stage_with(tmp_path)
    (stage.parent.parent / "ledger.jsonl").write_text("secret = 1\n")
    (stage / "port_check.py").write_text("secret = 1\n")
    (stage / "handoff.json").write_text("secret = 1\n")
    shutil.copy2(TOOL, stage / "stage_tools.py")
    before = {p: p.read_bytes() for p in stage.parent.parent.rglob("*") if p.is_file()}
    done = cli(stage, name, "--old", "secret = 1", "--new", "secret = 2")
    assert done.returncode == 2 and "EDIT_RECORD" not in done.stdout
    assert {p: p.read_bytes() for p in stage.parent.parent.rglob("*") if p.is_file()} == before


def test_the_stage_tool_file_itself_is_not_editable(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "stages/2/stage_tools.py", "--old", "edit", "--new", "x")
    assert done.returncode == 2 and "port directory" in done.stdout


def test_a_link_to_a_file_outside_the_port_directory_is_refused_and_the_target_is_unchanged(tmp_path):
    stage = stage_with(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    os.symlink(outside, stage / "port" / "linked.py")
    done = cli(stage, "stages/2/port/linked.py", "--old", "secret = 1", "--new", "secret = 2")
    assert done.returncode == 2 and outside.read_text() == "secret = 1\n"


def test_a_link_to_a_file_inside_the_port_directory_is_refused_too(tmp_path):
    stage = stage_with(tmp_path)
    os.symlink("port_model.py", stage / "port" / "alias.py")
    done = cli(stage, "stages/2/port/alias.py", "--old", "return x + 1\n", "--new", "return 1\n")
    assert done.returncode == 2 and port(stage) == BODY


def test_a_link_for_a_directory_on_the_way_is_refused(tmp_path):
    stage = stage_with(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "m.py").write_text("secret = 1\n")
    os.symlink(other, stage / "port" / "sub")
    done = cli(stage, "stages/2/port/sub/m.py", "--old", "secret = 1", "--new", "secret = 2")
    assert done.returncode == 2 and (other / "m.py").read_text() == "secret = 1\n"


def test_a_port_directory_that_is_a_link_is_refused(tmp_path):
    stage = stage_with(tmp_path)
    real = tmp_path / "realport"
    shutil.move(stage / "port", real)
    os.symlink(real, stage / "port")
    done = cli(stage, "stages/2/port/port_model.py", "--old", "return x + 1\n", "--new", "return 1\n")
    assert done.returncode == 2 and (real / "port_model.py").read_text() == BODY


def test_a_fifo_is_refused_without_blocking(tmp_path):
    stage = stage_with(tmp_path)
    os.mkfifo(stage / "port" / "pipe.py")
    done = cli(stage, "stages/2/port/pipe.py", "--old", "a", "--new", "b")
    assert done.returncode == 2 and "regular file" in done.stdout


def test_the_old_and_new_files_must_be_inside_the_run_directory(tmp_path):
    stage = stage_with(tmp_path)
    outside = tmp_path / "old.txt"
    outside.write_text("return x + 7\n")
    (stage / "new.txt").write_text("return 7\n")
    done = cli(stage, "port_model.py", "--old-file", str(outside), "--new-file", "stages/2/new.txt")
    assert done.returncode == 2 and port(stage) == BODY
    os.symlink(outside, stage / "linked-old.txt")
    done = cli(stage, "port_model.py", "--old-file", "stages/2/linked-old.txt", "--new-file", "stages/2/new.txt")
    assert done.returncode == 2 and port(stage) == BODY


# ---- the write ---------------------------------------------------------------------------------------------

def test_the_write_goes_through_a_temporary_file_that_is_renamed_over_the_port_file_and_leaves_nothing_behind(tmp_path):
    stage = stage_with(tmp_path)
    before = os.stat(stage / "port" / "port_model.py")
    assert cli(stage, "port_model.py", "--old", "return x + 7\n", "--new", "return 7\n").returncode == 0
    after = os.stat(stage / "port" / "port_model.py")
    assert after.st_ino != before.st_ino                      # a new file took the name: the old one was never half-written
    assert sorted(os.listdir(stage / "port")) == ["port_model.py"]
    assert after.st_mode == before.st_mode


def test_a_failure_before_the_rename_leaves_the_old_file_and_no_temporary_file(tmp_path, monkeypatch, capsys):
    stage = stage_with(tmp_path)
    mod = load_tool()
    monkeypatch.setattr(mod, "HERE", stage)

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(mod.os, "fsync", boom)
    code, lines = mod.edit_cmd(stage, ["port_model.py", "--old", "return x + 7\n", "--new", "return 7\n"])
    assert code == 2 and "disk full" in " ".join(lines) and "unchanged" in " ".join(lines)
    assert port(stage) == BODY and sorted(os.listdir(stage / "port")) == ["port_model.py"]


def test_a_file_that_changed_after_it_was_read_is_not_overwritten(tmp_path, monkeypatch):
    stage = stage_with(tmp_path)
    mod = load_tool()
    real, swapped = mod.os.fsync, []

    def swap_then_fsync(fd):
        if not swapped:                       # the first fsync is the temporary file's, just before the identity check
            swapped.append(1)
            (stage / "port" / "other.py").write_text("x = 1\n")
            os.replace(stage / "port" / "other.py", stage / "port" / "port_model.py")      # another writer replaced the file after the read
        return real(fd)
    monkeypatch.setattr(mod.os, "fsync", swap_then_fsync)
    code, lines = mod.edit_cmd(stage, ["port_model.py", "--old", "return x + 7\n", "--new", "return 7\n"])
    assert code == 2 and "changed" in " ".join(lines)
    assert port(stage) == "x = 1\n" and sorted(os.listdir(stage / "port")) == ["port_model.py"]


# ---- sizes -------------------------------------------------------------------------------------------------

def test_the_caps_are_the_ones_the_agents_own_write_bound_uses():
    mod = load_tool()
    assert mod.EDIT_MAX_BYTES == runaway.MAX_FILE_BYTES and mod.EDIT_MAX_LINES == runaway.MAX_FILE_LINES


def test_a_result_larger_than_the_cap_is_refused(tmp_path):
    stage = stage_with(tmp_path, body="x = 1\n" + "y" * (runaway.MAX_FILE_BYTES - 100) + "\n")
    done = cli(stage, "port_model.py", "--old", "x = 1\n", "--new", "x = 1\n" + "z" * 500 + "\n")
    assert done.returncode == 2 and "limit" in done.stdout and port(stage).startswith("x = 1\ny")


def test_a_result_with_more_lines_than_the_cap_is_refused(tmp_path):
    stage = stage_with(tmp_path, body="x = 1\n" * (runaway.MAX_FILE_LINES - 2) + "# marker\n")
    done = cli(stage, "port_model.py", "--old", "# marker\n", "--new", "a\nb\nc\nd\n")
    assert done.returncode == 2 and "limit" in done.stdout and "# marker" in port(stage)


def test_a_result_that_is_empty_is_refused(tmp_path):
    stage = stage_with(tmp_path, body="only = 1\n")
    done = cli(stage, "port_model.py", "--old", "only = 1\n", "--new", "")
    assert done.returncode == 2 and "empty" in done.stdout and port(stage) == "only = 1\n"


def test_a_file_already_over_the_cap_is_refused_before_it_is_read_whole(tmp_path):
    stage = stage_with(tmp_path)
    with open(stage / "port" / "port_model.py", "wb") as f:
        f.truncate(runaway.MAX_FILE_BYTES + 1)
    done = cli(stage, "port_model.py", "--old", "a", "--new", "b")
    assert done.returncode == 2 and "limit" in done.stdout


def test_a_file_that_is_not_text_is_refused(tmp_path):
    stage = stage_with(tmp_path)
    (stage / "port" / "port_model.py").write_bytes(b"x = '\xff\xfe'\n")
    done = cli(stage, "port_model.py", "--old", "x", "--new", "y")
    assert done.returncode == 2 and "UTF-8" in done.stdout


# ---- the watchdog counts the call --------------------------------------------------------------------------

def tool(args: str) -> str:
    return f"python3 stages/2/stage_tools.py edit {args}"


def test_two_different_edits_of_one_file_have_different_shapes_and_the_same_edit_repeated_has_one():
    a = tool("stages/2/port/port_model.py --old 'return x + 1' --new 'return x + 2'")
    b = tool("stages/2/port/port_model.py --old 'return x + 3' --new 'return x + 4'")
    assert command_shape(a) != command_shape(b)
    assert command_shape(a) == command_shape("cd /runs/x && " + a + " 2>&1 | tail -5")
    assert command_shape(a).startswith("stage_tools edit stages/2/port/port_model.py")


def test_edits_with_long_arguments_that_differ_late_still_differ():
    head = "x" * 300
    assert command_shape(tool(f"f.py --old '{head}a' --new b")) != command_shape(tool(f"f.py --old '{head}b' --new b"))


def test_the_same_edit_repeated_is_stopped_as_a_repeated_call_and_two_edits_are_not():
    from orchard.watchdog import NoFileWritten, RepeatedToolCall, TurnRepeat, replay
    from test_watchdog_stage_tools import turn

    def stuck(commands):
        evs = []
        for k, c in enumerate(commands, start=1):
            evs += turn(k, c, "edit: changed " + c)     # a different edit prints a different record
        return replay(evs, [RepeatedToolCall(), TurnRepeat(), NoFileWritten(40)])
    assert stuck([tool("p.py --old 'a' --new 'b'")] * 6)
    assert not stuck([tool(f"p.py --old 'a{i}' --new 'b{i}'") for i in range(6)])


# ---- the agent's own file guard sees what an edit wrote ----------------------------------------------------

def test_an_edit_that_makes_the_file_a_loop_is_put_back_by_the_shell_guard_after_the_command(tmp_path):
    from orchard.agent import Tools
    from orchard.agent import agent_env
    stage = stage_with(tmp_path)
    run = stage.parent.parent
    shutil.copy2(TOOL, stage / "stage_tools.py")
    t = Tools(run, stage, agent_env(run))
    loop = "x = 1\\n" * 9000
    out = t.shell(f"python3 stages/2/stage_tools.py edit stages/2/port/port_model.py --old 'def f1(x):' --new '{loop}'")
    assert "refused and restored" in out and "degenerate repetition" in out
    assert port(stage) == BODY


# ---- the task text and the repair text ask for it -----------------------------------------------------------

def test_the_task_text_asks_for_edits_after_the_first_write():
    from orchard import port_facts
    note = port_facts.prepare_note(facts=True, skeleton=True)
    line = next(l for l in note.splitlines() if " edit " in l and l.startswith("- "))
    assert "after your first write" in line and "--old" in line and "--new" in line and "`--old-file` and `--new-file`" in note
    assert "Write your first version of `port_model.py`" in note          # the first attempt is still a whole-file write


@pytest.mark.parametrize("failure", [{"kind": "numerics", "n": 16, "first_mismatch": 0, "mismatches": 7, "agreement": 0.5625, "rows": [(0, 5, 6)],
                                      "port_sha256": "ab" * 32},
                                     {"kind": "request", "op": "prefill"}])
def test_the_repair_text_asks_for_edits_and_not_for_the_whole_file_again(failure):
    from orchard.supervisor import Supervisor
    text = Supervisor._repair_text(None, 2, 1, 3, failure, "out")
    cmds = [m.group(1) for m in re.finditer(r"`(python3 stages/2/stage_tools\.py edit[^`]*)`", text)]
    assert cmds and "do not write the whole file again" in text
    assert command_shape(cmds[0]).startswith("stage_tools edit ")
