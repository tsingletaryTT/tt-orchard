# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The watchdog counts a call of a stage tool (orchard/skills/full-port-templates/stage_tools.py, issue 505) by tool name plus arguments.

Before this, `command_shape` cut a shell command to its first word and its first path-like argument, so `python3 stages/2/stage_tools.py facts a`
and `python3 stages/2/stage_tools.py facts b` shared the shape `python3 stages/2/stage_tools.py`, and the near-duplicate rule could stop a
coder that was reading different facts. With the arguments in the shape, two calls share one only when they are the same call."""
import hashlib
import json

import pytest

from orchard.watchdog import Event, NoFileWritten, RepeatedToolCall, TurnRepeat, command_shape, replay

RUN = "/runs/x"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tool(args: str, stage: int = 2, prefix: str = "") -> str:
    return f"{prefix}python3 stages/{stage}/stage_tools.py {args}".rstrip()


@pytest.mark.parametrize("a,b", [
    (tool("facts config.hc_count"), tool("facts config.ple_layer_ids")),
    (tool("show port/port_model.py --lines 1:100"), tool("show port/port_model.py --lines 101:200")),
    (tool("show port/port_model.py --lines 1:100"), tool("show port/other.py --lines 1:100")),
    (tool("last-check"), tool("layer-diff")),
    (tool("layer-diff --threshold 0.05"), tool("layer-diff --threshold 0.2")),
    (tool('facts --find "ple"'), tool('facts --find "hc"')),
])
def test_different_tools_or_arguments_have_different_shapes(a, b):
    assert command_shape(a) != command_shape(b)


@pytest.mark.parametrize("a,b", [
    (tool("last-check"), tool("last-check", prefix=f"cd {RUN} && ")),
    (tool("last-check"), tool("last-check", prefix=f"cd {RUN}/other; ")),
    (tool("last-check"), "python3 -P -s stages/2/stage_tools.py last-check"),
    (tool("last-check"), tool("last-check") + " 2>&1 | head -50"),
    (tool("last-check"), tool("last-check") + " > /dev/null"),
    (tool("facts config.hc_count"), f"/usr/bin/python3 {RUN}/stages/2/stage_tools.py   facts   config.hc_count"),
    (tool("last-check"), tool("last-check", stage=3)),
])
def test_the_same_call_has_the_same_shape_whatever_surrounds_it(a, b):
    assert command_shape(a) == command_shape(b)


def test_the_shape_names_the_tool_and_the_arguments():
    assert command_shape(f"cd {RUN} && " + tool("facts config.ple_layer_ids")) == "stage_tools facts config.ple_layer_ids"
    assert command_shape(tool("")) == "stage_tools"


def test_a_pipe_inside_a_quoted_argument_is_part_of_the_arguments_and_a_real_pipe_is_not():
    assert command_shape(tool('facts --find "ple|PLE" | head -5')) == "stage_tools facts --find ple|PLE"
    assert command_shape(tool('facts --find "ple|PLE"')) != command_shape(tool('facts --find "ple"'))


@pytest.mark.parametrize("cmd", ["cat stages/2/stage_tools.py", "grep -n facts stages/2/stage_tools.py", "sed -n '1,50p' stages/2/stage_tools.py",
                                 "echo python3 stages/2/stage_tools.py facts"])
def test_reading_the_tool_file_is_not_a_tool_call(cmd):
    assert not command_shape(cmd).startswith("stage_tools")


def test_a_command_that_is_not_python_but_names_the_file_is_not_a_tool_call():
    assert not command_shape("bash stages/2/stage_tools.py facts").startswith("stage_tools")


@pytest.mark.parametrize("cmd", ["", "   ", "python3 stages/2/stage_tools.py 'unbalanced", tool("facts " + "x" * 5000)])
def test_odd_commands_give_a_shape_and_no_exception(cmd):
    assert isinstance(command_shape(cmd), str) and len(command_shape(cmd)) <= 300


def turn(k: int, command: str, content: str) -> list:
    args = json.dumps({"command": command})
    return [Event(ts=10 * k, agent="a", kind="response", input_tokens=10 * k, output_tokens=7, had_tool_call=True, turn=k),
            Event(ts=10 * k + 1, agent="a", kind="tool_call", tool="shell", args_hash=sha(args), shape=command_shape(command), turn=k),
            Event(ts=10 * k + 2, agent="a", kind="tool_result", tool="shell", output_hash=sha(content), turn=k)]


def test_six_different_fact_reads_in_a_row_are_progress_and_the_same_read_six_times_is_a_loop():
    keys = ["config.hc_count", "config.ple_layer_ids", "layer_schedule", "config.head_dim", "tensors.0", "model_type"]
    evs = []
    for k, key in enumerate(keys, start=1):
        evs += turn(k, f"cd {RUN} && " + tool(f"facts {key}"), f"value of {key}")
    assert replay(evs, [RepeatedToolCall(), TurnRepeat(), NoFileWritten(40)]) == []
    evs = []
    for k in range(1, 7):
        evs += turn(k, f"cd {RUN} && " + tool("facts config.hc_count"), "same value")
    assert replay(evs, [RepeatedToolCall(), TurnRepeat(), NoFileWritten(40)])


def test_python_running_another_script_that_gets_the_tool_file_as_an_argument_is_not_a_tool_call():
    assert not command_shape("python3 other.py stages/2/stage_tools.py facts").startswith("stage_tools")


def test_another_file_whose_name_ends_the_same_way_is_not_the_tool():
    assert not command_shape("python3 stages/2/not_stage_tools.py facts a").startswith("stage_tools")


def test_two_long_calls_that_differ_only_at_the_end_have_different_shapes():
    base = "x" * 300
    assert command_shape(tool(f"facts --find {base}a")) != command_shape(tool(f"facts --find {base}b"))
    assert command_shape(tool(f"facts --find {base}a")) == command_shape(tool(f"facts --find {base}a"))
