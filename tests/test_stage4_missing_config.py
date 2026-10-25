# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4 when no installed bundle or package serves a required chip count (issue 68).

On 2026-10-25 a fine-tune whose only installed package is a 4-chip container package passed stage 2 and then stage 4 left its list to the agent
(the drafter found nothing for the required 2-chip configuration). The agent wrote the same file three times, the watchdog stopped it, and the run
blocked as `agent-stuck` 20 minutes of chip time later. A required count that nothing serves is now a block before any agent step, naming the count and
the way out."""
import json

import pytest

from orchard import swap_draft as sd
from test_supervisor import Stop
from test_supervisor_container_base import wide_rig, stage2_tests  # noqa: F401
from test_swap_draft import NEAREST, containers, run  # noqa: F401  (fixtures)

pytestmark = pytest.mark.usefixtures("stub_tools")


def test_the_drafter_reports_the_counts_nothing_serves(run, tmp_path):
    home = containers(tmp_path / "op", {"a/four": (NEAREST, 4, "default")})
    missing: list[int] = []
    configs, tests, notes = sd.draft_stage4(**{k: v for k, v in run.items() if k != "chips"}, operator_home=home, counts=[2, 4],
                                           missing=missing)
    assert missing == [2] and set(configs) == {4} and tests is None and notes


def test_an_ambiguous_count_is_not_reported_as_missing(run, tmp_path):
    home = containers(tmp_path / "op", {"a/x": (NEAREST, 2, "default"), "b/y": (NEAREST, 2, "default")})
    missing: list[int] = []
    sd.draft_stage4(**{k: v for k, v in run.items() if k != "chips"}, operator_home=home, counts=[2], missing=missing)
    assert missing == []            # two packages fit: the operator chooses; nothing is missing


def blocked_notices(r):
    return [e["data"] for e in r.entries() if e["event"] == "notice" and e["data"].get("blocked")]


def test_a_required_count_nothing_serves_blocks_before_any_agent_step(tmp_path):
    r = wide_rig(tmp_path)
    r.args.required_chips = (2, 4)
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop())
    try:
        with pytest.raises(Stop):
            r.run()
        notes = blocked_notices(r)
        why = [b["reason"] for b in notes if "2-chip" in b["reason"]]
        assert why, notes
        assert "required_chips to 4" in why[0] and "--change-required-chips" in why[0] and "no installed bundle or container package" in why[0]
        stage4 = [e for e in r.entries() if e["stage"] == 4]
        assert not [e for e in stage4 if e["data"].get("decision") == "agent step"], "no agent step ran"
        assert not (r.run_dir / "stages" / "4" / "configs").exists()
    finally:
        r.close()


def test_an_optional_count_nothing_serves_does_not_block(tmp_path):
    """With only 4 chips required, the 2-chip gap is not this block's business (the scripted fake agent then stops the stage for its own
    reasons, so the run is ended at its first pause)."""
    r = wide_rig(tmp_path)
    r.args.required_chips = (4,)
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop())
    try:
        with pytest.raises(Stop):
            r.run()
        assert not [b for b in blocked_notices(r) if "no installed bundle or container package serves" in b["reason"]]
        assert [e for e in r.entries() if e["stage"] == 4 and e["data"].get("decision") == "agent step"]
    finally:
        r.close()


def test_when_no_required_count_can_be_served_the_block_says_so(tmp_path):
    r = wide_rig(tmp_path)
    r.args.required_chips = (2,)
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop())
    try:
        with pytest.raises(Stop):
            r.run()
        why = [b["reason"] for b in blocked_notices(r) if "2-chip" in b["reason"]]
        assert why and "No required count can be served" in why[0] and "--required-chips" not in why[0]
    finally:
        r.close()


def test_an_explicit_four_chip_package_with_nothing_for_four_chips_is_missing(run, tmp_path):
    """Review of PR 70: naming four_chip_package skipped the missing-count report when no package serves 4 chips at all."""
    home = containers(tmp_path / "op", {"a/two": (NEAREST, 2, "default")})
    missing: list[int] = []
    configs, tests, notes = sd.draft_stage4(**{k: v for k, v in run.items() if k != "chips"}, operator_home=home, counts=[4],
                                           four_chip_package="a/two", missing=missing)
    assert missing == [4] and not configs


def test_an_explicit_package_that_is_not_among_several_is_not_reported_as_missing(run, tmp_path):
    home = containers(tmp_path / "op", {"a/four": (NEAREST, 4, "default")})
    missing: list[int] = []
    sd.draft_stage4(**{k: v for k, v in run.items() if k != "chips"}, operator_home=home, counts=[4], four_chip_package="b/other",
                    missing=missing)
    assert missing == []                      # a package exists for 4 chips; the named one is a mistake for the operator to fix
