# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 194: a supervisor killed after the trial script wrote `trial-result.json` but before the supervisor wrote its `test-result.json` leaves trial outputs on
disk. A restart with too little cache space saves a decision that says no trial was run. The stage 2 gate recomputes the decision from the files, so the leftovers
must not be read as a served variant."""
import pytest

from orchard import sampling_choice as sc, supervisor
from orchard.supervisor import EXIT_READY
from fakes import Crash
from test_supervisor_sampling import TrialSpy, decision_file, overrides, rig, sd, spy  # noqa: F401

pytestmark = pytest.mark.usefixtures("stub_tools")


def _kill_between_the_two_writes(rig_):
    """Crash once the trial script has written its outputs, then remove the supervisor's test record, as if the kill had come before that write."""
    with pytest.raises(Crash):
        rig_.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    trial = sd(rig_) / sc.TRIAL_DIR
    (trial / "test-result.json").unlink()
    assert (trial / sc.TRIAL_RESULT).is_file() and (trial / "variant" / "evidence" / "swap-check.json").is_file()
    assert not (sd(rig_) / sc.DECISION_FILE).exists()


def test_a_restart_with_too_little_space_sets_the_stale_outputs_aside_and_the_gate_passes(rig, spy, monkeypatch):
    _kill_between_the_two_writes(rig)
    real = supervisor.check_disk
    monkeypatch.setattr(supervisor, "SAMPLING_TRIAL_DISK_GB", 12345.0)              # only the trial's own requirement is unmet
    monkeypatch.setattr(supervisor, "check_disk", lambda path, gb, usage=None: (False, 7.0) if gb == 12345.0 else real(path, gb, usage=usage))
    assert rig.run(pid=200) == EXIT_READY
    d = decision_file(rig)
    assert d["flag"] == "variant-not-tried" and d["trial"] is None
    trial = sd(rig) / sc.TRIAL_DIR
    assert not (trial / sc.TRIAL_RESULT).exists() and not (trial / "variant").exists()
    assert (trial / "stale" / sc.TRIAL_RESULT).is_file() and (trial / "stale" / "variant").is_dir()      # kept as evidence
    assert sc.decision_problems(sd(rig), rig.run_dir) == []


def test_a_restart_with_room_runs_the_trial_again_and_leaves_nothing_aside(rig, spy):
    _kill_between_the_two_writes(rig)
    assert rig.run(pid=200) == EXIT_READY
    assert decision_file(rig)["choice"] == "drafter-off"
    assert not (sd(rig) / sc.TRIAL_DIR / "stale").exists()
