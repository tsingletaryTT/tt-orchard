# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The live narration of stage 2's sampling choice (issue 107): the trial's start, the decision with its reason, a skipped trial and the trial's test."""
import pytest

from orchard.ledger import Ledger
from test_supervisor_graft import narrated


def run_with(tmp_path, *entries):
    run = tmp_path / "run"
    with Ledger(run / "ledger.jsonl") as led:
        led.append("run_start", None, model="x/y")
        for event, stage, data in entries:
            led.append(event, stage, **data)
    return run


def test_the_trial_start_says_what_is_served_and_that_it_takes_a_while(tmp_path):
    run = run_with(tmp_path, ("decision", 2, {"decision": "sampling trial started", "needed": True, "need_source": "generation_config"}))
    [line] = [x for x in narrated(run) if "sampling trial" in x]
    assert "drafter off" in line and "minutes" in line and "generation_config" in line


def test_a_drafter_off_decision_is_narrated_with_its_reason(tmp_path):
    run = run_with(tmp_path, ("decision", 2, {"decision": "sampling decision", "choice": "drafter-off", "flag": None,
                                              "reason": "the model needs sampling and the first configuration ignores it", "needed": True}))
    [line] = [x for x in narrated(run) if "sampling decision" in x]
    assert "drafter-off configuration is chosen" in line and "ignores it" in line


def test_a_keep_decision_is_narrated_as_the_served_configuration_staying(tmp_path):
    run = run_with(tmp_path, ("decision", 2, {"decision": "sampling decision", "choice": "keep", "flag": None,
                                              "reason": "sampling is not needed for this model", "needed": False}))
    [line] = [x for x in narrated(run) if "sampling decision" in x]
    assert "configuration stage 2 served stays" in line and "not needed" in line


@pytest.mark.parametrize("flag", ["no-variant-serves-purpose", "sampling-applied-below-floor", "unverified", "variant-not-tried"])
def test_a_flagged_decision_is_marked_and_names_the_flag(tmp_path, flag):
    run = run_with(tmp_path, ("decision", 2, {"decision": "sampling decision", "choice": "keep", "flag": flag, "reason": "neither works",
                                              "needed": True}))
    [line] = [x for x in narrated(run) if "sampling decision" in x]
    assert flag in line and "needs the operator" in line


def test_a_skipped_trial_is_narrated_with_the_reason(tmp_path):
    run = run_with(tmp_path, ("decision", 2, {"decision": "sampling trial skipped", "reason": "the run uses a lab"}))
    [line] = [x for x in narrated(run) if "sampling trial skipped" in x]
    assert "the run uses a lab" in line


@pytest.mark.parametrize("data,words", [({"returncode": 0}, "passed (exit 0)"), ({"returncode": 3}, "failed (exit 3)"),
                                        ({"returncode": None, "timed_out": True}, "timed out")])
def test_the_trials_test_result_is_narrated(tmp_path, data, words):
    run = run_with(tmp_path, ("evidence", 2, {"what": "sampling trial test", "path": "p", "sha256": "0" * 64, **data}))
    [line] = [x for x in narrated(run) if "sampling trial test" in x]
    assert words in line
