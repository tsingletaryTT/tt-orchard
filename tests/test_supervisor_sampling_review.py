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


# ---- finding 2: a restart sweeps the sampling trial's servers too ---------------------------------------------------------------------

def test_a_restart_stops_the_sampling_trials_servers_before_it_recovers_leases(rig, spy, monkeypatch):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    calls = []
    monkeypatch.setattr(supervisor.Supervisor, "_stop_trial_servers",
                        lambda self, n, d, when: calls.append((n, d.name, when)))
    assert rig.run(pid=200) == EXIT_READY
    assert (2, "sampling-trial", "when the run started again") in calls


def test_a_restart_sweeps_the_graft_trial_directory_as_before(rig, spy, monkeypatch):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    (sd(rig) / "graft-trial").mkdir(exist_ok=True)
    calls = []
    monkeypatch.setattr(supervisor.Supervisor, "_stop_trial_servers",
                        lambda self, n, d, when: calls.append((n, d.name, when)))
    rig.run(pid=200)
    assert (2, "graft-trial", "when the run started again") in calls


def test_a_restart_with_no_trial_directory_sweeps_nothing(rig, spy, monkeypatch):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 1)
    assert not (sd(rig) / "sampling-trial").exists() and not (sd(rig) / "graft-trial").exists()
    calls = []
    monkeypatch.setattr(supervisor.Supervisor, "_stop_trial_servers", lambda self, n, d, when: calls.append((d.name, when)))
    rig.run(pid=200)
    assert [c for c in calls if c[1] == "when the run started again"] == []


def test_a_server_the_killed_supervisor_left_in_the_sampling_trial_is_stopped_on_restart(rig, spy):
    import os
    import signal
    import subprocess
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    proc = subprocess.Popen(["sleep", "300"], start_new_session=True,
                            env={**os.environ, supervisor.GRAFT_TRIAL_ENV: str((sd(rig) / "sampling-trial").resolve())})
    try:
        assert rig.run(pid=200) == EXIT_READY
        assert proc.wait(timeout=20) is not None                 # stopped by the sweep at the restart
        stopped = [e["data"] for e in rig.entries() if e["event"] == "decision" and e["data"].get("decision") == "graft trial servers stopped"]
        assert stopped and stopped[0]["when"] == "when the run started again" and stopped[0]["servers"][0]["pid"] == proc.pid
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)


def short_disk(monkeypatch):
    """After this, only the sampling trial's own disk check sees a short disk (its threshold is made distinct)."""
    real = supervisor.check_disk
    monkeypatch.setattr(supervisor, "SAMPLING_TRIAL_DISK_GB", 12345.0)
    monkeypatch.setattr(supervisor, "check_disk", lambda path, gb, usage=None: (False, 7.0) if gb == 12345.0 else real(path, gb, usage=usage))


# ---- finding 8: the disk prerequisite is for a trial that still has to run ----------------------------------------------------------------

def test_a_finished_trial_is_read_after_a_restart_even_when_its_own_cache_left_the_disk_short(rig, spy, monkeypatch):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    short_disk(monkeypatch)                         # the variant's new tensor cache now fills the disk
    assert rig.run(pid=200) == EXIT_READY
    d = decision_file(rig)
    assert d["choice"] == "drafter-off" and d["flag"] is None and d["trial"] == f"{sc.TRIAL_DIR}/{sc.TRIAL_RESULT}"
    assert len(spy.sampling_trials) == 1 and not stage2(rig, "sampling trial skipped")


def test_a_trial_that_has_not_run_is_still_skipped_for_want_of_disk(rig, spy, monkeypatch):
    short_disk(monkeypatch)
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["flag"] == "variant-not-tried" and "12345 GB" in d["reason"] and spy.sampling_trials == []


def test_a_failed_trial_marker_is_not_a_reason_to_run_again_and_still_says_it_failed(rig, spy, monkeypatch):
    spy.s_code = 3
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    short_disk(monkeypatch)
    assert rig.run(pid=200) == EXIT_READY
    d = decision_file(rig)
    assert "did not finish" in d["reason"] and len(spy.sampling_trials) == 1


def test_the_real_sweep_runs_on_both_directories_without_error(rig, spy):
    """The real sweep (no servers here) must not fail on the sampling trial directory."""
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    assert rig.run(pid=200) == EXIT_READY
