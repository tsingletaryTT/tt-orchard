# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 562, the measuring half: in run 20 (stage 2, nine repair rounds) every repair step rewrote port_model.py with write_file and the `edit`
stage tool was never called, although the repair text names it. The issue's own first step is to measure how often a round rewrites the whole
file and how often it edits. `Tools` now counts a write_file that replaces an existing long file with different content (a whole-file
rewrite) and a shell command that runs the `edit` stage tool, and the step's ledger record carries both counts when they are not zero.
Nothing is refused: whether to refuse a rewrite needs runs to evaluate (the issue says so), and the coder changed its design in most steps
of run 20. Everything here runs against scripted fakes: no model, no chip."""
import json

from fake_model import FakeModel, call, final, turn
from orchard.agent import REWRITE_MIN_LINES, AgentStep
from orchard.ledger import Ledger
from test_tool_dedupe import put, run, shell, step, tools, write   # noqa: F401  (run is a fixture)

LONG = "\n".join(f"line {i}" for i in range(REWRITE_MIN_LINES + 20)) + "\n"
EDIT_CMD = "python3 stages/0/stage_tools.py edit port/port_model.py --old 'line 1' --new 'line one'"


def test_replacing_an_existing_long_file_with_different_content_is_a_counted_rewrite(run):
    put(run, "port/port_model.py", LONG)
    t = tools(run)
    out = write(t, 3, "port/port_model.py", LONG + "extra\n")
    assert out.startswith("wrote ")
    assert t.rewrite_count == 1
    (path, turn_no, old_lines, new_chars) = t.rewrite_calls[0]
    assert path.endswith("port/port_model.py") and turn_no == 3 and old_lines == REWRITE_MIN_LINES + 20 and new_chars == len(LONG + "extra\n")


def test_a_new_file_a_short_file_and_a_repeat_of_the_same_content_are_not_rewrites(run):
    t = tools(run)
    write(t, 1, "port/port_model.py", LONG)                       # a new file: the first write of an attempt
    put(run, "short.txt", "a\nb\n")
    write(t, 2, "short.txt", "c\nd\n")                              # a short file
    out = write(t, 3, "port/port_model.py", LONG)                  # exactly the same content: answered, not done
    assert out.startswith("unchanged:")
    assert t.rewrite_count == 0 and t.rewrite_calls == []


def test_a_shell_command_that_runs_the_edit_tool_is_counted_and_others_are_not(run):
    t = tools(run)
    shell(t, 1, EDIT_CMD)                                          # fails (no such file), and is still an edit call
    shell(t, 2, "ls stages/0")
    shell(t, 3, "echo stage_tools.py edit")                        # the words are not a call of the tool
    assert t.edit_calls == 1


def test_the_listed_rewrites_are_bounded(run):
    put(run, "port/port_model.py", LONG)
    t = tools(run)
    for n in range(1, 40):
        write(t, n, "port/port_model.py", LONG + f"v{n}\n")
    assert t.rewrite_count == 39 and len(t.rewrite_calls) <= 12


def transcript(led):
    return [e["data"] for e in led.read() if e["event"] == "evidence" and e["data"].get("what") == "transcript"][-1]


def test_the_ledger_record_of_a_step_carries_the_counts(run):
    put(run, "port/port_model.py", LONG)
    script = [call("write_file", path="port/port_model.py", content=LONG + "again\n"), call("shell", command=EDIT_CMD), final("done")]
    with Ledger(run / "ledger.jsonl") as led, FakeModel(lambda r: script[turn(r)]) as fm:
        out = step(run, led, fm.endpoint)[0].run("system", "user")
        rec = transcript(led)
    assert out.status == "done"
    assert rec["whole_file_rewrites"] == 1 and rec["edit_calls"] == 1
    assert rec["whole_file_rewrite_calls"][0][0].endswith("port/port_model.py")


def test_the_ledger_record_of_a_step_with_neither_has_no_such_keys(run):
    put(run, "evidence/a.txt", "text")
    script = [call("read_file", path="evidence/a.txt"), final("done")]
    with Ledger(run / "ledger.jsonl") as led, FakeModel(lambda r: script[turn(r)]) as fm:
        step(run, led, fm.endpoint)[0].run("system", "user")
        rec = transcript(led)
    assert not any(k in rec for k in ("whole_file_rewrites", "whole_file_rewrite_calls", "edit_calls"))
