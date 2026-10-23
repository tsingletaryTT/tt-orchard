# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the stage 2 swap skill says about prepare_swap.py's "no stop token" warning.

The skill once told the agent to write the warning in `result.json` and stop. After a normal prepare reply the
supervisor reads `hw_test.json` and `handoff.json` (`Supervisor._read_test`) and reports them missing; it never reads
`result.json` there, so the operator would have seen "hw_test.json and handoff.json must exist" and not the warning.
The warning predicts a failure from one bundle's reading rule, so the agent goes on and the hardware test decides."""
import json
from pathlib import Path

from test_swap_skill_generation_config import bullet

START = '- `prepare_swap.py` warns "no stop token"'


def test_the_no_stop_token_warning_is_advisory_and_the_agent_goes_on_to_the_hardware_test():
    b = bullet(START)
    assert "advisory" in b and "`hw_test.json` and `handoff.json`" in b


def test_the_skill_does_not_send_the_agent_to_result_json_and_a_stop_for_it():
    b = bullet(START)
    assert "`result.json`" not in b and "and stop" not in b and "write the warning" not in b


def test_the_supervisor_really_has_no_prepare_failure_outcome_the_skill_could_use():
    """The reason for the rule: if the supervisor ever reads a prepare-failure file, change the skill with it."""
    src = (Path(__file__).parent.parent / "orchard" / "supervisor.py").read_text(encoding="utf-8")
    body = src.split("out, prep = self._step(spec, \"prepare\"", 1)[1].split("self._hardware_phase(spec, stage_dir, test)", 1)[0]
    assert "result.json" not in body and "_read_test(stage_dir)" in body


TOKENIZER_START = '- `prepare_swap.py` warns "the tokenizer\'s eos token'


def test_the_tokenizer_eos_warning_is_advisory_and_the_agent_goes_on_to_the_hardware_test():
    b = bullet(TOKENIZER_START)
    assert "advisory" in b and "`hw_test.json` and `handoff.json`" in b and "Do not try to fix it" in b


def test_the_tokenizer_eos_warning_does_not_send_the_agent_to_result_json_and_a_stop():
    b = bullet(TOKENIZER_START)
    assert "`result.json`" not in b and "and stop" not in b
