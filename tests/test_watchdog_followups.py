# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Follow-ups from the PR 462, 501, 504 and 506 reviews (issues 474, 509, 511, 521).

474: a capped reply that still ran complete tool calls counts toward the two-in-a-row restart rule.
509: the first-write deadline hint is not carried across a context restart (the restart text asks the agent to re-read state; the hint says to stop reading).
511: command_shape keeps the directory of a leading `cd` for sed ranges and for commands with no path-like argument.
521: command_shape keeps quoted operator-like arguments ("|", ">", "&&") in the shape.
"""
import json

import pytest

from fake_model import FakeModel, call, final, truncated
from orchard import port_facts, runaway
from orchard.watchdog import command_shape
from test_agent_runaway_round3 import Control, by_request, cut_call, run, step      # noqa: F401  (run is a fixture)


# ---- 474: capped replies that ran complete calls --------------------------------------------------------------------------------------

def capped_write(path: str, call_id: str, tokens: int = 16384) -> dict:
    """finish_reason "length" with one complete write_file call: the server's parser dropped the runaway tail."""
    args = json.dumps({"path": path, "content": "x = 1\n"})
    return {"role": "assistant", "content": "", "finish_reason": "length", "completion_tokens": tokens,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "write_file", "arguments": args}}]}


def test_two_capped_replies_that_each_ran_a_call_restart_the_step_and_the_second_calls_ran(run):
    run_dir, ledger = run
    answers = [capped_write("port/one.py", "a"), capped_write("port/two.py", "b"), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1).run("system text", "task text")
    assert out.status == "done"
    assert (run_dir / "stages" / "0" / "port" / "one.py").exists() and (run_dir / "stages" / "0" / "port" / "two.py").exists(), \
        "the complete call of the reply that triggered the restart must still run"
    fresh = fm.requests[2]["messages"]
    assert [m["role"] for m in fresh] == ["system", "user"] and "restarted" in fresh[1]["content"]
    assert any(e["data"].get("decision") == "context restarted after capped replies" for e in ledger.read())


def test_two_capped_replies_that_each_ran_a_call_end_a_step_that_may_not_restart(run):
    run_dir, ledger = run
    answers = [capped_write("port/one.py", "a"), capped_write("port/two.py", "b"), final("never")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "error" and len(fm.requests) == 2
    assert "cut off at max_tokens twice in a row" in out.detail
    assert (run_dir / "stages" / "0" / "port" / "two.py").exists()


def test_a_normal_reply_between_two_capped_replies_resets_the_count(run):
    run_dir, ledger = run
    answers = [capped_write("port/one.py", "a"), call("shell", "n1", command="echo plain"),
               capped_write("port/two.py", "b"), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1).run("s", "u")
    assert out.status == "done"
    assert not any(e["data"].get("decision") == "context restarted after capped replies" for e in ledger.read())
    assert len(fm.requests[3]["messages"]) > 2, "the conversation was restarted although the capped replies were not in a row"


def test_a_capped_reply_with_a_call_then_a_capped_reply_with_no_call_restarts_the_step(run):
    run_dir, ledger = run
    answers = [capped_write("port/one.py", "a"), truncated(16384), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1).run("system text", "task text")
    assert out.status == "done"
    assert any(e["data"].get("decision") == "context restarted after capped replies" for e in ledger.read())
    assert [m["role"] for m in fm.requests[2]["messages"]] == ["system", "user"]


# ---- 509: the first-write deadline hint is not carried into a restart ---------------------------------------------------------------

@pytest.mark.parametrize("skeleton", [True, False])
def test_a_write_deadline_hint_is_not_carried_across_a_restart(run, skeleton):
    run_dir, ledger = run
    hint = port_facts.write_deadline_hint(12, port_is_skeleton=skeleton, excerpts=True)
    control = Control()
    control.nudges.append(hint)
    answers = [truncated(16384), truncated(16384), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, control=control, unstick=runaway.Unstick(), restart_on_caps=1).run("s", "task text")
    assert out.status == "done"
    first = fm.requests[0]["messages"]
    assert any(hint in str(m.get("content")) for m in first), "the hint was not delivered in the conversation it was written for"
    prompt = fm.requests[2]["messages"][1]["content"]
    assert hint not in prompt and "Stop reading" not in prompt and "still the skeleton" not in prompt, "a stale hint reached the restart prompt"
    assert "read a file again when you need its text" in prompt


def test_the_recognizer_matches_both_hints_the_supervisor_can_write_and_nothing_else():
    for skeleton in (True, False):
        for excerpts in (True, False):
            assert port_facts.is_write_deadline_hint(port_facts.write_deadline_hint(9, port_is_skeleton=skeleton, excerpts=excerpts))
    assert not port_facts.is_write_deadline_hint("20 model turns have passed and no file was written. Stop investigating: write the files now.")
    assert not port_facts.is_write_deadline_hint("GATE FEEDBACK: result.json is missing the field `chips`; write it.")
    assert not port_facts.is_write_deadline_hint(None)


def test_another_nudge_is_still_carried_across_a_restart(run):
    run_dir, ledger = run
    control = Control()
    control.nudges.append("SUPERVISOR NUDGE: write the files now and stop investigating.")
    answers = [truncated(16384), truncated(16384), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        step(run_dir, ledger, fm.endpoint, control=control, unstick=runaway.Unstick(), restart_on_caps=1).run("s", "task text")
    assert "SUPERVISOR NUDGE: write the files now" in fm.requests[2]["messages"][1]["content"]


# ---- 511: the directory of a leading cd stays in the shape ------------------------------------------------------------------------

def test_the_same_command_in_different_directories_has_different_shapes():
    assert command_shape("cd stage0 && ls -la") != command_shape("cd stage1 && ls -la")
    assert command_shape("cd stage0 && ls -la") == command_shape("cd stage0 && ls -la")
    assert command_shape("ls -la") == "ls -la"                          # no cd: unchanged


def test_a_sed_range_of_the_same_file_name_in_different_directories_has_different_shapes():
    shapes = {command_shape(f"cd {d} && sed -n '1,20p' config.json") for d in ("a", "b", "c", "d")}
    assert len(shapes) == 4
    assert command_shape("sed -n '1,20p' config.json") == "sed config.json 1,20p"       # no cd: the shape the other tests pin


def test_a_relative_program_with_no_arguments_keeps_its_directory():
    assert command_shape("cd a && ./run.sh") != command_shape("cd b && ./run.sh")


# ---- 521: quoted operator-like arguments ------------------------------------------------------------------------------------------

def test_quoted_operators_in_a_stage_tool_call_give_three_shapes():
    base = "python3 /run/stages/2/stage_tools.py facts --find"
    shapes = {command_shape(f'{base} "{op}"') for op in ("|", ">", "&&")}
    assert len(shapes) == 3 and command_shape(base) not in shapes


def test_quoted_operators_in_an_ordinary_command_give_different_shapes():
    assert command_shape('echo "&&"') != command_shape('echo ">"') != command_shape('echo "|"')


def test_a_pipe_inside_quotes_does_not_cut_the_command_but_a_real_pipe_does():
    assert command_shape('echo "a|b"') == "echo a|b"
    assert command_shape("echo hi | head") == "echo hi"
    assert command_shape("grep -rn x /src | head -50") == "grep /src"


def test_an_unbalanced_quote_still_gives_a_shape():
    assert command_shape('echo "unclosed | thing') != ""
