# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the review of PR 114 (issue 107): the drafter-off variant is admitted only when its trial succeeded and its own checks pass, a trial that
failed after writing its swap check leaves nothing behind, and the other findings of that review."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sampling_choice_files import make_run, report, save, write  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402


def trial_record(s2, **kw):
    write(s2 / sc.TRIAL_DIR / "test-result.json", {"command": "x", "returncode": 0, "timed_out": False, **kw})


# ---- finding 6: a failed trial test leaves no variant -------------------------------------------------------------------------------

@pytest.mark.parametrize("rec", [{"returncode": 3}, {"returncode": None, "timed_out": True}, {"returncode": 0, "timed_out": True},
                                 {"returncode": -9}, {"returncode": None}])
def test_a_trial_test_that_failed_leaves_no_variant_even_when_it_wrote_a_swap_check(tmp_path, rec):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    trial_record(s2, **rec)
    assert sc.variant_side(s2) is None


def test_a_trial_test_that_succeeded_leaves_the_variant(tmp_path):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    trial_record(s2)
    assert sc.variant_side(s2)["verdict"] == "sampled-full-speed"


def test_no_test_record_leaves_the_variant_as_before(tmp_path):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    assert sc.variant_side(s2) is not None


def test_a_timeout_after_the_swap_check_is_not_a_stage_failure_in_the_gate(tmp_path):
    """The supervisor records keep / variant-not-tried for a trial that timed out. The gate must recompute the same, though the variant's swap check
    is on disk."""
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    trial_record(s2, returncode=None, timed_out=True)
    d = sc.decide(sc.need(run), sc.served_side(s2), None, variant_note="the sampling trial timed out and was stopped")
    assert d["choice"] == "keep" and d["flag"] == "variant-not-tried"
    sc.write_decision(s2, d, trial=False)
    assert sc.decision_problems(s2, run) == []


def test_an_unreadable_test_record_counts_as_a_failed_test(tmp_path):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    (s2 / sc.TRIAL_DIR / "test-result.json").write_text("{not json")
    assert sc.variant_side(s2) is None
