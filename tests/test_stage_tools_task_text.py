# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 task text names the stage tools (issue 505): one line each, the exact command, and the optional `layer_outputs` hook that
`layer-diff` needs. The commands in the text are the ones the watchdog counts by tool name and arguments."""
import re

import pytest

from orchard import full_port, port_facts
from orchard.watchdog import command_shape

TOOLS = ("last-check", "layer-diff", "facts", "show", "cheat")


def command_lines(text: str):
    return [m.group(1) for m in re.finditer(r"`(python3 stages/2/stage_tools\.py[^`]*)`", text)]


def test_the_prepare_note_names_each_tool_with_its_command_and_one_line():
    note = port_facts.prepare_note(facts=True, skeleton=True, excerpts=True)
    cmds = command_lines(note)
    assert len(cmds) == len(TOOLS)
    for tool, cmd in zip(TOOLS, cmds):
        assert cmd.split()[2] == tool
    for line in note.splitlines():
        if "stage_tools.py" in line and line.startswith("- "):
            assert len(line) < 330


def test_every_command_in_the_text_is_one_the_watchdog_counts_by_name_and_arguments():
    for cmd in command_lines(port_facts.prepare_note(facts=True, skeleton=True)):
        assert command_shape(cmd).startswith("stage_tools "), cmd


def test_the_text_says_what_layer_outputs_must_return_and_that_it_is_optional():
    note = port_facts.prepare_note(facts=True, skeleton=True)
    assert "layer_outputs(model, ids)" in note and "optional" in note
    assert "after each decoder layer" in note and "below 0.85" in note


def test_the_text_tells_the_coder_to_use_the_tools_instead_of_cat_grep_and_python_c():
    note = port_facts.prepare_note(facts=True, skeleton=True)
    assert "`cat`" in note and "`grep`" in note and "`python3 -c`" in note


def test_the_tools_text_carries_the_same_agreement_bar_as_the_gate():
    assert str(full_port.TOP1_MIN) in port_facts.tools_text(2)


def test_without_a_facts_file_the_text_does_not_offer_the_facts_tool():
    note = port_facts.prepare_note(facts=False, skeleton=True)
    assert "port_facts.json" not in note and "stage_tools.py last-check" in note and "stage_tools.py show" in note and "stage_tools.py facts" not in note


def test_stage_3_gets_no_stage_2_tool_text():
    assert "stage_tools" not in port_facts.stage_3_note(chips=2, copied=True)


def test_the_repair_text_names_the_tools_for_a_numerics_failure_and_for_a_run_failure(tmp_path):
    from types import SimpleNamespace
    from orchard import supervisor
    failure = {"kind": "numerics", "agreement": 0.5625, "n": 32, "mismatches": 14, "first_mismatch": 0, "rows": [(0, 100, 7)], "port_sha256": "a" * 64}
    text = supervisor.Supervisor._repair_text(SimpleNamespace(), 2, 1, 3, failure, "")
    assert "stage_tools.py layer-diff" in text and "stage_tools.py last-check" in text and "layer_outputs(model, ids)" in text
    run = {"kind": "port", "op": "prefill"}
    text = supervisor.Supervisor._repair_text(SimpleNamespace(), 2, 1, 3, run, "FAILED: boom")
    assert "stage_tools.py last-check" in text and "layer-diff" not in text
