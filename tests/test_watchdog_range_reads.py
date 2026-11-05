# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Reading different line ranges of one file is progress. It is not a loop (issue 427, third rehearsal run).

In the third stage 2 rehearsal the coder paged through one reference module with `sed -n 'a,bp' <file>`: four calls, four different
ranges, four different pieces of text. `command_shape` dropped the range, the four calls shared one shape, and the near-duplicate rule
blocked the run as `agent-stuck`. The right handling is the no-file-written nudge, which asks for a first write. Calls repeat only when
the same command (range included) comes back or the same text comes back.

The fake agents below feed the real detectors the events AgentStep would send: a response, then a tool call with its args hash and its
shape (from `command_shape`), then the result with its output hash."""
import hashlib
import json

import pytest

from orchard.watchdog import (Event, NoFileWritten, RepeatedToolCall, TurnRepeat, command_shape, replay)

FILE = "/opt/venv/lib/python3.12/site-packages/transformers/models/toy/modeling_toy.py"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_turn(k: int, command: str, content: str) -> list:
    """One model turn that runs one shell command and gets `content` back."""
    args = json.dumps({"command": command})
    return [Event(ts=10 * k, agent="a", kind="response", input_tokens=10 * k, output_tokens=7, had_tool_call=True, turn=k),
            Event(ts=10 * k + 1, agent="a", kind="tool_call", tool="shell", args_hash=sha(args), shape=command_shape(command), turn=k),
            Event(ts=10 * k + 2, agent="a", kind="tool_result", tool="shell", output_hash=sha(content), turn=k)]


def write_turn(k: int) -> list:
    return [Event(ts=10 * k, agent="a", kind="response", input_tokens=10 * k, output_tokens=7, had_tool_call=True, turn=k),
            Event(ts=10 * k + 1, agent="a", kind="tool_result", tool="write_file", wrote=True, turn=k)]


def detectors(turns: int = 20):
    return [RepeatedToolCall(), TurnRepeat(), NoFileWritten(turns)]


def ranges(n: int, width: int = 250):
    return [f"sed -n '{1 + i * width},{(i + 1) * width}p' {FILE}" for i in range(n)]


def test_eight_reads_of_different_ranges_then_a_write_is_never_blocked():
    evs = []
    for k, cmd in enumerate(ranges(8), start=1):
        evs += read_turn(k, cmd, f"text of range {k}")
    evs += write_turn(9) + [Event(ts=100, agent="a", kind="response", input_tokens=1, output_tokens=1, had_tool_call=False, turn=10)]
    assert replay(evs, detectors()) == []


def test_the_same_call_four_times_is_a_loop():
    cmd = f"sed -n '1,250p' {FILE}"
    evs = []
    for k in range(1, 6):
        evs += read_turn(k, cmd, "same text")
    found = replay(evs, detectors())
    assert found, "four identical calls must be caught"
    assert {f.detector for f in found} & {"turn_repeat", "repeated_tool_call"}


def test_different_ranges_that_return_the_same_text_are_still_caught():
    """The results repeat: three different ranges came back with one text each time (a file that is shorter than the ranges)."""
    evs = []
    for k, cmd in enumerate(ranges(4), start=1):
        evs += read_turn(k, cmd, "")
    found = replay(evs, detectors())
    assert any(f.detector == "repeated_tool_call" and f.evidence.get("what") == "output" for f in found)


def test_reads_with_no_write_reach_the_no_file_written_nudge_and_it_names_the_excerpts_file():
    evs = []
    for k, cmd in enumerate(ranges(20), start=1):
        evs += read_turn(k, cmd, f"text of range {k}")
    evs.append(Event(ts=999, agent="a", kind="response", input_tokens=1, output_tokens=1, had_tool_call=True, turn=21))
    found = replay(evs, detectors())
    assert [f.detector for f in found] == ["no_file_written"]
    assert "stages/2/reference_excerpts.md" in found[0].nudge
    assert "write" in found[0].nudge and "port_model.py" in found[0].nudge


def test_the_nudge_text_still_asks_for_a_write_when_there_is_no_excerpts_file():
    nudge = NoFileWritten(1)
    f = nudge.feed(Event(ts=1, agent="a", kind="response", input_tokens=1, output_tokens=1, turn=1))
    assert f is not None and "no file was written" in f.nudge


@pytest.mark.parametrize("a,b", [
    (f"sed -n '1,250p' {FILE}", f"sed -n '251,500p' {FILE}"),
    (f"sed -n 100,200p {FILE}", f"sed -n 200,300p {FILE}"),
    (f"sed -n '40p' {FILE}", f"sed -n '41p' {FILE}"),
    (f"sed -n '1,250p' {FILE} | head -50", f"sed -n '300,400p' {FILE} | head -50"),
])
def test_sed_calls_with_different_ranges_have_different_shapes(a, b):
    assert command_shape(a) != command_shape(b)


def test_the_same_sed_range_has_the_same_shape_whatever_the_spacing_or_trailing_pipe():
    assert command_shape(f"sed -n '1,250p'   {FILE}") == command_shape(f"sed -n '1,250p' {FILE} 2>/dev/null")


def test_commands_other_than_sed_keep_their_shape():
    assert command_shape(f"grep -rn x {FILE} | head -5") == command_shape(f"grep -rn y {FILE} | head -9")
    assert command_shape(f"cat {FILE} | head -5") == command_shape(f"cat {FILE} | head -9")
