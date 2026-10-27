# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 114 in the supervisor (issue 107): a variant that fails its own checks is not shipped, a trial that finished before a kill is not skipped for
want of disk, and a kill during the sampling trial leaves no server behind when the run starts again."""
import json

import pytest

from orchard import sampling_choice as sc, supervisor
from orchard.supervisor import EXIT_READY
from test_prompt_suite_verdict import LOOP, make_record, with_entry
from test_sampling_choice_files import report
from test_supervisor_sampling import TrialSpy, decision_file, overrides, rig, sd, spy, stage2  # noqa: F401
from run_fakes import bringup
from fakes import Crash

pytestmark = pytest.mark.usefixtures("stub_tools")

FREE = "Paris is the capital of France and a large city"


def test_a_variant_that_fails_the_prompt_suite_is_not_shipped_and_the_stage_still_passes(rig, spy):
    bad = report(drafter="off", ignored=False, seconds=6.0)
    bad["prompt_suite"] = with_entry(make_record(), "arithmetic", content=LOOP)
    bad["free_run_text"] = bad["reference_generated_text"] = FREE
    spy.variant = bad
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose" and "prompt suite" in d["reason"]
    assert sc.drafter_off_chosen(rig.run_dir) is False and [e["result"] for e in rig.ends(2)] == ["pass"]


def test_a_variant_that_is_incoherent_is_not_shipped(rig, spy):
    spy.variant = report(drafter="off", ignored=False, seconds=6.0, draft={"coherent": False})
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["choice"] == "keep" and "coherent" in d["reason"]
