# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A port file changed by a shell command counts as a write for the watchdog (issue 867).

In a stage 2 trial an agent made 34 `stage_tools.py edit` calls and 35 `py_compile` calls in 60 turns and was escalated by `no_file_written`:
the watchdog counted a write only for the `write_file` tool, and a shell command arrives as a tool result with no `wrote` value. The task text
tells the agent to change the port with `edit` after its first write, so the supervisor steered the agent to a write path its own watchdog did
not see, and the first-write hint and the nudge fired on an agent that was making progress.

The evidence is the port directory, read by the port guard before and after each shell command (orchard/port_guard.py): a file whose accepted
content differs from what it held before the command is a write. The text of the command's output is not evidence. Each guard below has a
mutation in the PR body that turns its test red."""
import shutil

import pytest

from fake_model import FakeModel, call, final, turn
from orchard import port_facts, runaway
from orchard.ledger import Ledger
from orchard.port_guard import PortGuard
from orchard.watchdog import FirstWriteDeadline, NoFileWritten, replay
from test_stage_tools_last_check import TOOL
from test_tool_dedupe import put, run, shell, step, tools   # noqa: F401  (run is a fixture)

PORT = "stages/0/port/port_model.py"
BODY = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(1, 21))


@pytest.fixture
def stage(run):
    put(run, "port/port_model.py", BODY)
    shutil.copy2(TOOL, run / "stages" / "0" / "stage_tools.py")
    return run / "stages" / "0"


def edit(old, new, name=PORT):
    return f"python3 stages/0/stage_tools.py edit {name} --old {old!r} --new {new!r}"


# ---- the port guard names the files a command changed ------------------------------------------------------------------------------------

def guard(stage):
    return PortGuard(stage, stage.parent.parent)


def test_a_port_file_whose_content_changed_is_listed_after_judge(stage):
    g = guard(stage)
    g.remember()
    (stage / "port" / "port_model.py").write_text(BODY.replace("x + 3", "x * 3"))
    g.judge()
    assert g.written == [("port_model.py",)]


def test_a_new_port_file_is_listed(stage):
    g = guard(stage)
    g.remember()
    (stage / "port" / "helpers.py").write_text("H = 1\n")
    g.judge()
    assert g.written == [("helpers.py",)]


def test_a_rewrite_with_the_same_bytes_is_not_a_write(stage):
    g = guard(stage)
    g.remember()
    (stage / "port" / "port_model.py").write_text(BODY)               # a new mtime, the same content
    g.judge()
    assert g.written == []


def test_a_cache_directory_is_not_a_write(stage):
    g = guard(stage)
    g.remember()
    (stage / "port" / "__pycache__").mkdir()
    (stage / "port" / "__pycache__" / "port_model.cpython-312.pyc").write_bytes(b"\0\1\2")
    g.judge()
    assert g.written == []


def test_a_file_the_guard_puts_back_is_not_a_write(stage):
    g = guard(stage)
    g.remember()
    (stage / "port" / "port_model.py").write_text("x = 1\n" * (runaway.MAX_FILE_BYTES // 6 + 100))
    notes = g.judge()
    assert notes and g.written == []
    assert (stage / "port" / "port_model.py").read_text() == BODY


def test_the_list_is_the_last_command_only(stage):
    g = guard(stage)
    g.remember()
    (stage / "port" / "port_model.py").write_text(BODY + "# one\n")
    g.judge()
    g.remember()
    g.judge()
    assert g.written == []


# ---- a shell call says whether it wrote -------------------------------------------------------------------------------------------------

def test_a_shell_call_has_no_answer_before_any_call(run):
    assert getattr(tools(run), "last_shell_wrote", None) is None


def test_sed_in_place_on_a_port_file_is_a_write(stage, run):
    t = tools(run)
    out = shell(t, 1, f"sed -i 's/x + 3/x * 3/' {PORT}")
    assert out.startswith("exit 0"), out
    assert t.last_shell_wrote is True


def test_a_redirect_that_makes_a_new_port_file_is_a_write(stage, run):
    t = tools(run)
    out = shell(t, 1, "echo 'H = 1' > stages/0/port/helpers.py")
    assert (stage / "port" / "helpers.py").is_file(), out
    assert t.last_shell_wrote is True


def test_the_edit_tool_changing_a_file_is_a_write(stage, run):
    t = tools(run)
    out = shell(t, 1, edit("x + 3", "x * 3"))
    assert "EDIT_RECORD" in out and out.startswith("exit 0"), out
    assert t.last_shell_wrote is True


def test_an_edit_the_tool_refused_is_not_a_write_and_says_so(stage, run):
    t = tools(run)
    out = shell(t, 1, edit("no such text", "x"))
    assert "EDIT_RECORD" not in out
    assert (stage / "port" / "port_model.py").read_text() == BODY
    assert t.last_shell_wrote is False


@pytest.mark.parametrize("command", ["ls stages/0/port", f"grep -n def {PORT}", f"python3 -m py_compile {PORT}", "echo hello"])
def test_a_command_that_changes_no_port_file_has_no_answer(stage, run, command):
    t = tools(run)
    shell(t, 1, command)
    assert t.last_shell_wrote is None


def test_a_printed_record_is_not_evidence_of_a_write(stage, run):
    """The text of the output can say anything. Only the file counts."""
    t = tools(run)
    shell(t, 1, 'echo \'EDIT_RECORD {"tool": "edit", "file": "stages/0/port/port_model.py", "bytes_after": 1}\'')
    assert t.last_shell_wrote is None
    shell(t, 2, edit("no such text", "x") + ' ; echo \'EDIT_RECORD {"tool": "edit"}\'')
    assert t.last_shell_wrote is False


def test_each_call_answers_for_itself(stage, run):
    t = tools(run)
    shell(t, 1, edit("x + 3", "x * 3"))
    assert t.last_shell_wrote is True
    shell(t, 2, "ls")
    assert t.last_shell_wrote is None


@pytest.mark.parametrize("command", ["", "   ", "grep -R x /"])
def test_a_call_that_ran_nothing_does_not_keep_the_last_answer(stage, run, command):
    t = tools(run)
    shell(t, 1, edit("x + 3", "x * 3"))
    assert t.last_shell_wrote is True
    shell(t, 2, command)                       # an empty command, and a search the runner refuses: nothing ran
    assert t.last_shell_wrote is None


def test_a_stage_without_a_port_directory_never_writes(run):
    t = tools(run)
    shell(t, 1, "echo hi > stages/0/evidence/a.txt")
    assert t.last_shell_wrote is None


# ---- the agent's events and the watchdog's detectors -----------------------------------------------------------------------------------

def events_of(run, commands):
    """The events AgentStep feeds the watchdog for a scripted model that runs each command in its own turn and then ends."""
    script = [call("shell", command=c) for c in commands] + [final("done")]
    with Ledger(run / "ledger.jsonl") as led, FakeModel(lambda r: script[turn(r)]) as fm:
        s, events, _ = step(run, led, fm.endpoint)
        s.run("system", "user")
    return events


def detectors():
    return [NoFileWritten(turns=5), FirstWriteDeadline(lambda n: f"hint {n}", 5)]


def test_the_result_event_of_a_shell_edit_says_it_wrote(stage, run):
    evs = events_of(run, [edit("x + 3", "x * 3")])
    results = [e for e in evs if e.kind == "tool_result" and e.tool == "shell"]
    assert [e.wrote for e in results] == [True]


def test_a_run_of_shell_edits_fires_neither_detector(stage, run):
    cmds = [edit(f"def f{i}(x):", f"def f{i}(x, k=0):") for i in range(1, 13)]
    evs = events_of(run, cmds)
    assert [e.wrote for e in evs if e.kind == "tool_result"] == [True] * 12
    assert replay(evs, detectors()) == []


def test_the_same_turns_without_a_write_fire_both_detectors(stage, run):
    """The instrument can fire: twelve reads and refused edits in a row are the case the detectors are for."""
    cmds = ["ls stages/0/port", edit("no such text", "x")] * 6
    evs = events_of(run, cmds)
    assert [e.wrote for e in evs if e.kind == "tool_result"] == [None, False] * 6
    assert {f.detector for f in replay(evs, detectors())} == {"no_file_written", "first_write_deadline"}


# ---- the nudge and the hint say what counts ---------------------------------------------------------------------------------------------

def test_the_nudge_names_the_calls_that_count_as_writes(stage, run):
    f = replay([_resp(t) for t in range(1, 8)], [NoFileWritten(turns=5)])[0]
    assert "write_file" in f.nudge and "stage_tools.py edit" in f.nudge and "shell command that changes a port file" in f.nudge


@pytest.mark.parametrize("skeleton", [True, False])
def test_the_hint_names_the_calls_that_count_as_writes(skeleton):
    text = port_facts.write_deadline_hint(12, port_is_skeleton=skeleton, excerpts=True)
    assert "shell command that changes a port file" in text and "stage_tools.py edit" in text
    assert port_facts.is_write_deadline_hint(text)


def _resp(turn_no):
    from orchard.watchdog import Event
    return Event(ts=float(turn_no), agent="a", kind="response", had_tool_call=True, turn=turn_no)
