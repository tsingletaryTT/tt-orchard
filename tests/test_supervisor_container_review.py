# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The four findings of Copilot's review of PR 64 (stage 2 through a container package), each reproduced before its fix:
a restart sweeps the run's containers before it gives leases back; a container that survives the sweep keeps the lease it
runs under; a lab run and the sidecar stage do not get the container fallback."""
import hashlib
import json
from pathlib import Path

import pytest

from orchard import supervisor
from orchard.adapters import AdapterError
from orchard.ledger import Ledger
from run_fakes import FakeContainers, MachineAdapter
from test_supervisor import Rig, Stop
from test_supervisor_container_base import container_rig, drafted
from test_supervisor_lab import _installed_for_draft  # noqa: F401  (a v6 bundle and snapshots for the lab rig)

pytestmark = pytest.mark.usefixtures("stub_tools")


def label(run_dir) -> str:
    return "orchard.test=" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]


def seed_stale_lease(rig):
    """A ledger that shows a test lease an earlier supervisor took and never released."""
    stale = MachineAdapter(rig.m, owner_pid=999).acquire(2, "orchard:supervisor", "stage 2 test")
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", versions={}, inputs={})
        led.append("decision", 2, decision="test lease taken", config=2, test_lease=stale.record())
    return stale


@pytest.fixture
def plain(tmp_path):
    r = Rig(tmp_path)
    r.args.unattended = True
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop("the unattended run waited for the operator"))
    yield r
    r.close()


# ---- 1. a restart sweeps the run's containers before it gives a lease back ------------------------------------------------------

def test_a_restart_stops_the_runs_containers_before_it_gives_back_the_old_test_lease(plain):
    plain.containers = FakeContainers([["c1"], []])
    stale = seed_stale_lease(plain)
    plain.on_sleep = None
    plain.run()
    es = plain.entries()
    swept = next(i for i, e in enumerate(es) if e["data"].get("what") == "a hardware test left containers running; stopping them")
    gave = next(i for i, e in enumerate(es) if e["data"].get("decision") == "test lease released" and e["data"].get("lease_id") == stale.lease_id)
    assert swept < gave
    assert ("remove", "c1") in plain.containers.calls
    assert ("list", label(plain.run_dir)) in plain.containers.calls


def test_a_container_that_survives_the_restart_sweep_keeps_the_old_lease_and_blocks(plain):
    plain.containers = FakeContainers([["c1"]])
    stale = seed_stale_lease(plain)
    assert plain.run() == supervisor.EXIT_BLOCKED
    assert stale.lease_id in plain.m.leases or not [e for e in plain.entries() if e["data"].get("decision") == "test lease released"
                                                     and e["data"].get("lease_id") == stale.lease_id]
    blocked = [e for e in plain.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked and "still running" in blocked[0]["data"]["reason"]
    assert not [e for e in plain.entries() if e["event"] == "decision" and e["data"].get("decision") == "coder starting"]


def test_a_restart_with_clean_docker_writes_no_sweep_notice_and_goes_on(plain):
    plain.containers = FakeContainers([[]])
    stale = seed_stale_lease(plain)
    with Ledger(plain.run_dir / "ledger.jsonl") as led:
        led.append("decision", 2, decision="test lease released", lease_id=stale.lease_id)
    plain.m.leases.pop(stale.lease_id, None)
    plain.on_sleep = None
    assert plain.run() == supervisor.EXIT_READY
    assert not [e for e in plain.entries() if e["data"].get("what") == "a hardware test left containers running; stopping them"]
    assert plain.containers.calls and plain.containers.calls[0] == ("list", label(plain.run_dir))


# ---- 2. a surviving container keeps its lease through the shutdown --------------------------------------------------------------

def test_an_unattended_run_does_not_release_the_test_lease_while_a_container_survives(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"]])
    r.args.unattended = True
    try:
        assert r.run() == supervisor.EXIT_BLOCKED
        released = [e for e in r.entries() if e["data"].get("decision") == "test lease released"]
        assert released == []
        kept = [e for e in r.entries() if e["event"] == "notice" and "test lease" in str(e["data"].get("what", ""))]
        assert kept and "c1" in json.dumps(kept[0]["data"])
        assert r.m.leases, "the lease the container runs under must still be held"
    finally:
        r.close()


def test_the_test_lease_goes_back_once_the_container_is_gone(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"], []])
    try:
        assert r.run() == supervisor.EXIT_READY
        assert [e for e in r.entries() if e["data"].get("decision") == "test lease released" and e["stage"] == 2]
    finally:
        r.close()


# ---- 3. the lab run and the sidecar stage are not given the container fallback ----------------------------------------------------

def spy_on_draft(monkeypatch):
    from orchard import swap_draft
    seen, real = [], swap_draft.draft
    monkeypatch.setattr(swap_draft, "draft", lambda **kw: (seen.append(kw), real(**kw))[1])
    return seen


def test_a_lab_run_is_not_drafted_a_container_configuration(monkeypatch, tmp_path):
    from test_supervisor_lab import LabRig
    seen = spy_on_draft(monkeypatch)
    rig = LabRig(tmp_path)
    try:
        _installed_for_draft(rig)
        assert rig.run() == supervisor.EXIT_READY
    finally:
        rig.close()
    assert seen and all(kw["operator_home"] is None for kw in seen)


def test_the_sidecar_stage_is_not_drafted_a_container_configuration_and_the_swap_stage_is(monkeypatch, tmp_path):
    from orchard.stages import SIDECAR_STAGE_2, WEIGHTS_ONLY_STAGE_2
    seen = spy_on_draft(monkeypatch)
    r = container_rig(tmp_path)
    try:
        r.run_dir.mkdir(parents=True, exist_ok=True)
        with Ledger(r.run_dir / "ledger.jsonl") as led:
            sup = supervisor.build(r.args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m),
                                   versions={"tt_model": "test"}, clock=r.clock, sleep=r.sleep, disk_usage=lambda p: r.usage(p),
                                   home=r.home, containers=r.containers)
            sup._draft(SIDECAR_STAGE_2, tmp_path / "side")
            sup._draft(WEIGHTS_ONLY_STAGE_2, tmp_path / "swap")
    finally:
        r.close()
    assert [kw["operator_home"] is None for kw in seen] == [True, False]


# ---- the second look: a parked coder, a cleared survivor, and the lab phase -----------------------------------------------------------

def test_a_restart_sweeps_even_when_the_ledger_shows_no_test_lease(plain):
    """Stage 2 parks the coder and runs its container test under the coder's lease, which records no `test lease taken` (Copilot, PR 64)."""
    plain.containers = FakeContainers([["c1"], []])
    plain.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(plain.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", versions={}, inputs={})
        led.append("decision", 2, decision="hardware phase", action="park", free_boards=[], server_boards=[0, 1])
    plain.on_sleep = None
    plain.run()
    es = plain.entries()
    first_stage = next(i for i, e in enumerate(es) if e["event"] == "stage_start")
    assert [e for e in es[:first_stage] if e["data"].get("what") == "a hardware test left containers running; stopping them"]
    assert ("remove", "c1") in plain.containers.calls


def test_a_survivor_that_is_gone_when_the_lease_is_next_released_no_longer_keeps_it(tmp_path):
    from orchard.adapters import Lease  # noqa: F401
    r = container_rig(tmp_path, listings=[["c1"], ["c1"], []])      # the sweep's list, its re-check, then clean when asked again
    try:
        with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
            sup = supervisor.build(r.args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m),
                                   versions={"tt_model": "test"}, clock=r.clock, sleep=r.sleep, disk_usage=lambda p: r.usage(p),
                                   home=r.home, containers=r.containers)
            lease = sup.test_adapter.acquire(2, "orchard:supervisor", "stage 4 test")
            sup.test_lease = lease
            with pytest.raises(supervisor.Blocked):
                sup._sweep(4)
            assert sup.surviving_containers == ["c1"]
            assert sup._release_test_lease() is True          # docker now says the container is gone
            assert sup.surviving_containers == [] and sup.test_lease is None
    finally:
        r.close()


def test_a_survivor_that_docker_still_lists_keeps_the_lease_and_docker_that_cannot_answer_does_too(tmp_path):
    for i, listings in enumerate(([["c1"], ["c1"], ["c1"]], [["c1"], ["c1"], None])):
        d = tmp_path / str(i)
        d.mkdir()
        r = container_rig(d, listings=listings)
        try:
            with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
                sup = supervisor.build(r.args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m),
                                       versions={"tt_model": "test"}, clock=r.clock, sleep=r.sleep, disk_usage=lambda p: r.usage(p),
                                       home=r.home, containers=r.containers)
                sup.test_lease = sup.test_adapter.acquire(2, "orchard:supervisor", "stage 4 test")
                with pytest.raises(supervisor.Blocked):
                    sup._sweep(4)
                assert sup._release_test_lease() is False and sup.test_lease is not None and sup.surviving_containers
        finally:
            r.close()


def test_the_lab_phase_runs_the_cleanup_callback_before_it_gives_the_lease_back(monkeypatch, tmp_path):
    from test_supervisor_lab import LabRig
    rig = LabRig(tmp_path)
    try:
        _installed_for_draft(rig)
        monkeypatch.setattr(supervisor.Supervisor, "_container_config", staticmethod(lambda stage_dir: True))
        real = supervisor.Supervisor._sweep

        def sweep(self, n):
            rig.lab.adapter.calls.append(("sweep", n))          # one list holds the lab's lease calls and the sweeps
            return real(self, n)
        monkeypatch.setattr(supervisor.Supervisor, "_sweep", sweep)
        assert rig.run() == supervisor.EXIT_READY
        calls = rig.lab.adapter.calls
    finally:
        rig.close()
    first_acquire = next(i for i, c in enumerate(calls) if c[0] == "acquire")
    first_release = next(i for i, c in enumerate(calls) if c[0] == "release" and i > first_acquire)
    assert ("sweep", 2) in calls[first_acquire:first_release], calls[first_acquire:first_release + 1]


def test_the_kept_lease_notice_names_the_containers_docker_lists_now(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"], ["c1"], ["c2", "c3"]])
    try:
        with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
            sup = supervisor.build(r.args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m),
                                   versions={"tt_model": "test"}, clock=r.clock, sleep=r.sleep, disk_usage=lambda p: r.usage(p),
                                   home=r.home, containers=r.containers)
            sup.test_lease = sup.test_adapter.acquire(2, "orchard:supervisor", "stage 4 test")
            with pytest.raises(supervisor.Blocked):
                sup._sweep(4)
            assert sup._release_test_lease() is False
            assert sup.surviving_containers == ["c2", "c3"]
            kept = [e for e in led.read() if e["data"].get("what") == "the test lease was kept: a test container is still running"]
            assert kept and kept[-1]["data"]["containers"] == ["c2", "c3"]
    finally:
        r.close()


# ---- the third look: an unconfirmed cleanup and an attended start ----------------------------------------------------------------------

def bare_supervisor(r, led):
    return supervisor.build(r.args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m),
                            versions={"tt_model": "test"}, clock=r.clock, sleep=r.sleep, disk_usage=lambda p: r.usage(p),
                            home=r.home, containers=r.containers)


@pytest.mark.parametrize("after", [None, ["c1"]])
def test_a_cleanup_docker_cannot_confirm_is_a_survivor_and_keeps_the_lease(tmp_path, after):
    """Copilot (PR 64, third look): the first listing found c1, the re-listing returned None, and the sweep returned as if clean."""
    r = container_rig(tmp_path, listings=[["c1"], after])
    try:
        with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
            sup = bare_supervisor(r, led)
            sup.test_lease = sup.test_adapter.acquire(2, "orchard:supervisor", "stage 2 test")
            with pytest.raises(supervisor.Blocked) as e:
                sup._sweep(2)
            assert sup.surviving_containers == ["c1"]
            assert ("could not" in str(e.value) or "still running" in str(e.value))
            if after is None:
                assert "confirm" in str(e.value)
            assert sup._release_test_lease() is False and sup.test_lease is not None
    finally:
        r.close()


def test_a_sweep_that_finds_nothing_and_one_that_cleans_up_are_still_fine(tmp_path):
    for i, listings in enumerate(([[]], [["c1"], []])):
        d = tmp_path / str(i)
        d.mkdir()
        r = container_rig(d, listings=listings)
        try:
            with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
                sup = bare_supervisor(r, led)
                sup._sweep(2)
                assert sup.surviving_containers == []
        finally:
            r.close()


def test_docker_that_cannot_list_at_all_keeps_the_old_notice_and_does_not_block(tmp_path):
    r = container_rig(tmp_path, listings=[None])
    try:
        with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
            sup = bare_supervisor(r, led)
            sup._sweep(2)
            assert [e for e in led.read() if e["data"].get("what") == "docker could not be asked for test containers"]
    finally:
        r.close()


def test_an_attended_start_blocked_by_a_container_retries_the_recovery_after_the_operator_resumes(plain):
    """Copilot (PR 64, third look): after a pause at start, `resume` went into the stage loop without sweeping again or giving the old lease back."""
    from orchard.supervisor import Control
    plain.args.unattended = False
    plain.containers = FakeContainers([["c1"], ["c1"], []])       # the sweep finds c1, cannot remove it; after the operator's cleanup it is gone
    stale = seed_stale_lease(plain)
    plain.on_sleep = lambda: Control(plain.run_dir).write("resume")
    assert plain.run() == supervisor.EXIT_READY
    es = plain.entries()
    gave = [e for e in es if e["data"].get("decision") == "test lease released" and e["data"].get("lease_id") == stale.lease_id]
    assert gave, "the lease the earlier supervisor left was never given back"
    first_stage = next(i for i, e in enumerate(es) if e["event"] == "stage_start")
    assert es.index(gave[0]) < first_stage
    assert stale.lease_id not in plain.m.leases


def test_an_attended_start_blocked_by_a_container_ends_aborted_when_the_operator_aborts(plain):
    from orchard.supervisor import Control
    plain.args.unattended = False
    plain.containers = FakeContainers([["c1"]])
    stale = seed_stale_lease(plain)
    plain.on_sleep = lambda: Control(plain.run_dir).write("abort")
    assert plain.run() == supervisor.EXIT_ABORTED
    assert not [e for e in plain.entries() if e["event"] == "stage_start"]


# ---- the fourth look: a signal during a failed cleanup, and the graft trial on a container -------------------------------------------

def test_a_failed_cleanup_does_not_replace_an_interrupt_that_is_already_in_flight(tmp_path, monkeypatch):
    """Copilot (PR 64): the cleanup's Blocked, raised from a finally block, replaced Interrupted, so a SIGTERM during a container test
    paused the run instead of aborting it."""
    from orchard.supervisor import Interrupted, Supervisor
    r = container_rig(tmp_path, listings=[["c1"]])
    try:
        def interrupted(self, spec, stage_dir, test, lease, **kw):
            raise Interrupted("SIGTERM")
        monkeypatch.setattr(Supervisor, "_run_test", interrupted)
        code = r.run()
        assert code == supervisor.EXIT_ABORTED
        es = r.entries()
        assert [e for e in es if e["data"].get("decision") == "abort" and "SIGTERM" in str(e["data"].get("by"))]
        assert [e for e in es if e["event"] == "notice" and e["data"].get("blocked")], "the survivor must still be recorded"
        assert not [e for e in es if e["data"].get("decision") == "test lease released"], "its lease must still be kept"
    finally:
        r.close()


def test_a_failed_cleanup_after_a_normal_test_still_blocks(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"]])
    r.args.unattended = True
    try:
        assert r.run() == supervisor.EXIT_BLOCKED
    finally:
        r.close()


def graft_container_rig(tmp_path, *, bundle):
    from test_supervisor_graft import GRAFT_DELTA
    from run_fakes import bringup
    r = container_rig(tmp_path, bundle=bundle)
    r.script = lambda req: bringup(req, overrides=GRAFT_DELTA)
    return r


def test_a_container_only_stage_2_skips_the_graft_trial_and_records_the_safe_decision(tmp_path, monkeypatch):
    """Copilot (PR 64): the trial's scripts need stage_dir/run.sh, which container staging never writes, so it would take the hardware, possibly park
    the coder, and fail with no result."""
    from test_supervisor_graft import DECISION_FILE, TrialSpy
    r = graft_container_rig(tmp_path, bundle=False)
    spy = TrialSpy(supervisor.spawn_checked)
    monkeypatch.setattr(supervisor, "spawn_checked", spy)
    try:
        assert r.run() == supervisor.EXIT_READY
        assert spy.trials == [], "the trial must not start"
        d = json.loads((r.run_dir / "stages" / "2" / DECISION_FILE).read_text())
        assert d["choice"] == "safe" and "container package" in d["reason"]
        skipped = [e for e in r.entries() if e["data"].get("decision") == "graft trial skipped"]
        assert skipped and "container package" in skipped[0]["data"]["reason"]
    finally:
        r.close()


def test_a_bundle_stage_2_with_a_candidate_still_runs_the_trial(tmp_path, monkeypatch):
    from test_supervisor_graft import TrialSpy
    r = graft_container_rig(tmp_path, bundle=True)
    spy = TrialSpy(supervisor.spawn_checked)
    monkeypatch.setattr(supervisor, "spawn_checked", spy)
    try:
        assert r.run() == supervisor.EXIT_READY
        assert len(spy.trials) == 1
    finally:
        r.close()


def test_a_failed_cleanup_does_not_replace_an_interrupt_when_the_coder_was_parked_for_the_test(tmp_path, monkeypatch):
    """The same finding on the park path: a 4-chip coder holds both boards, so stage 2's test parks it."""
    from orchard.supervisor import Interrupted, Supervisor
    r = container_rig(tmp_path, listings=[["c1"]], chips=4)
    try:
        monkeypatch.setattr(Supervisor, "_run_test", lambda self, spec, stage_dir, test, lease, **kw: (_ for _ in ()).throw(Interrupted("SIGTERM")))
        parked = []
        real = Supervisor._handoff
        monkeypatch.setattr(Supervisor, "_handoff", lambda self, stage, lease: (parked.append(stage), real(self, stage, lease))[1])
        assert r.run() == supervisor.EXIT_ABORTED
        assert 2 in parked, "the test must have parked the coder"
    finally:
        r.close()


# ---- the fifth look: the coder on the other board, and an aborted run's retained lease -------------------------------------------------

def test_a_signal_with_a_surviving_container_still_stops_the_coder_on_the_other_board(tmp_path, monkeypatch):
    """Copilot (PR 64): `release_test_lease() and release_all()` skipped the coder's release when the test lease had to stay held, so the supervisor
    exited with the coder serving under its lease."""
    from orchard.supervisor import Interrupted, Supervisor
    r = container_rig(tmp_path, listings=[["c1"]])
    try:
        monkeypatch.setattr(Supervisor, "_run_test", lambda self, spec, stage_dir, test, lease, **kw: (_ for _ in ()).throw(Interrupted("SIGTERM")))
        assert r.run() == supervisor.EXIT_ABORTED
        assert r.m.coder_running is False, "the coder is on its own board and must be stopped"
        held = r.m.leases
        assert len(held) == 1, "only the test lease, which a live container still uses, stays held"
        released = [e for e in r.entries() if e["data"].get("decision") == "test lease released"]
        assert released == []
    finally:
        r.close()


def test_an_error_with_a_surviving_container_also_stops_the_coder(tmp_path, monkeypatch):
    from orchard.supervisor import Supervisor
    r = container_rig(tmp_path, listings=[["c1"]])
    try:
        def boom(self, spec, stage_dir, test, lease, **kw):
            raise RuntimeError("the test crashed")
        monkeypatch.setattr(Supervisor, "_run_test", boom)
        with pytest.raises(RuntimeError):
            r.run()
        assert r.m.coder_running is False
        assert len(r.m.leases) == 1
    finally:
        r.close()


def test_a_normal_signal_with_no_survivor_releases_both_leases(tmp_path, monkeypatch):
    from orchard.supervisor import Interrupted, Supervisor
    r = container_rig(tmp_path, listings=[[]])
    try:
        monkeypatch.setattr(Supervisor, "_run_test", lambda self, spec, stage_dir, test, lease, **kw: (_ for _ in ()).throw(Interrupted("SIGTERM")))
        assert r.run() == supervisor.EXIT_ABORTED
        assert r.m.coder_running is False and r.m.leases == {}
    finally:
        r.close()


def seed_aborted_run(rig):
    stale = seed_stale_lease(rig)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("decision", None, decision="abort", by="signal SIGTERM")
    return stale


def test_an_aborted_run_started_again_gives_back_the_lease_a_container_kept_once_docker_is_clean(plain):
    plain.containers = FakeContainers([[]])
    stale = seed_aborted_run(plain)
    plain.on_sleep = None
    assert plain.run() == supervisor.EXIT_ABORTED
    es = plain.entries()
    assert [e for e in es if e["data"].get("decision") == "test lease released" and e["data"].get("lease_id") == stale.lease_id]
    assert not [e for e in es if e["event"] == "stage_start"]
    assert stale.lease_id not in plain.m.leases


def test_an_aborted_run_with_a_container_still_there_keeps_the_lease_and_stays_aborted(plain):
    plain.containers = FakeContainers([["c1"]])
    stale = seed_aborted_run(plain)
    assert plain.run() == supervisor.EXIT_ABORTED
    es = plain.entries()
    assert not [e for e in es if e["data"].get("decision") == "test lease released" and e["data"].get("lease_id") == stale.lease_id]
    assert [e for e in es if e["event"] == "notice" and e["data"].get("blocked")]
    assert not [e for e in es if e["event"] == "stage_start"]


def test_a_finished_run_started_again_does_not_sweep(plain):
    plain.containers = FakeContainers([["c1"]])
    plain.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(plain.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", versions={}, inputs={})
        led.append("decision", None, decision="ready for operator review", bundle="stages/8/bundle")
    plain.on_sleep = None
    assert plain.run() == supervisor.EXIT_READY
    assert plain.containers.calls == []


# ---- the sixth look: the ledger-free release, and survivors remembered across a restart --------------------------------------------------

def test_the_ledger_free_release_keeps_a_test_lease_a_container_still_uses(tmp_path):
    """Copilot (PR 64): when the ledger cannot be written, _release_from_memory released the test lease without looking at the survivors."""
    r = container_rig(tmp_path, listings=[["c1"]])
    try:
        with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
            sup = bare_supervisor(r, led)
            sup.test_lease = sup.test_adapter.acquire(2, "orchard:supervisor", "stage 2 test")
            with pytest.raises(supervisor.Blocked):
                sup._sweep(2)
            assert sup._release_from_memory() is False
            assert sup.test_lease is not None and sup.test_lease.lease_id in r.m.leases
    finally:
        r.close()


def test_the_ledger_free_release_still_releases_a_test_lease_when_nothing_survives(tmp_path):
    r = container_rig(tmp_path, listings=[[]])
    try:
        with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
            sup = bare_supervisor(r, led)
            lease = sup.test_adapter.acquire(2, "orchard:supervisor", "stage 2 test")
            sup.test_lease = lease
            assert sup._release_from_memory() is True
            assert lease.lease_id not in r.m.leases
    finally:
        r.close()


def survivors_recorded(rig, stale):
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("decision", 2, decision="test containers survived", containers=["c1"])


def test_a_restart_with_docker_unavailable_keeps_a_lease_the_ledger_says_a_container_still_used(plain):
    """Copilot (PR 64): the survivor state started empty after a restart, and an unavailable docker listing let recovery release the lease."""
    plain.containers = FakeContainers([None])
    stale = seed_stale_lease(plain)
    survivors_recorded(plain, stale)
    assert plain.run() == supervisor.EXIT_BLOCKED
    es = plain.entries()
    assert not [e for e in es if e["data"].get("decision") == "test lease released" and e["data"].get("lease_id") == stale.lease_id]
    assert any("could not be asked whether the test containers" in str(e["data"].get("reason", "")) for e in es
               if e["event"] == "notice" and e["data"].get("blocked"))


def test_a_restart_with_docker_unavailable_and_no_known_survivor_goes_on(plain):
    plain.containers = FakeContainers([None])
    stale = seed_stale_lease(plain)
    plain.on_sleep = None
    assert plain.run() == supervisor.EXIT_READY
    assert [e for e in plain.entries() if e["data"].get("decision") == "test lease released" and e["data"].get("lease_id") == stale.lease_id]


def test_a_restart_that_finds_docker_clean_records_the_survivors_as_cleared_and_releases(plain):
    plain.containers = FakeContainers([[]])
    stale = seed_stale_lease(plain)
    survivors_recorded(plain, stale)
    plain.on_sleep = None
    assert plain.run() == supervisor.EXIT_READY
    es = plain.entries()
    cleared = next(i for i, e in enumerate(es) if e["data"].get("decision") == "test containers cleared")
    gave = next(i for i, e in enumerate(es) if e["data"].get("decision") == "test lease released" and e["data"].get("lease_id") == stale.lease_id)
    assert cleared < gave


def test_a_clean_sweep_with_nothing_ever_recorded_writes_no_cleared_decision(plain):
    plain.containers = FakeContainers([[]])
    seed_stale_lease(plain)
    plain.on_sleep = None
    plain.run()
    assert not [e for e in plain.entries() if e["data"].get("decision") == "test containers cleared"]


def test_a_sweep_that_blocks_records_the_survivors_and_one_that_clears_them_records_that(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"], ["c1"], ["c1"], []])
    try:
        with Ledger((r.run_dir.mkdir(parents=True, exist_ok=True) or r.run_dir) / "ledger.jsonl") as led:
            sup = bare_supervisor(r, led)
            with pytest.raises(supervisor.Blocked):
                sup._sweep(2)
            surv = [e for e in led.read() if e["data"].get("decision") == "test containers survived"]
            assert surv and surv[-1]["data"]["containers"] == ["c1"]
            sup.surviving_containers = []
            sup._sweep(2)                                        # docker now lists nothing (the last listing repeats)
            assert [e for e in led.read() if e["data"].get("decision") == "test containers cleared"]
    finally:
        r.close()


def test_survivors_the_ledger_shows_as_cleared_do_not_block_a_restart_when_docker_is_unavailable(plain):
    plain.containers = FakeContainers([None])
    stale = seed_stale_lease(plain)
    with Ledger(plain.run_dir / "ledger.jsonl") as led:
        led.append("decision", 2, decision="test containers survived", containers=["c1"])
        led.append("decision", 2, decision="test containers cleared", containers=["c1"])
    plain.on_sleep = None
    assert plain.run() == supervisor.EXIT_READY
    assert [e for e in plain.entries() if e["data"].get("decision") == "test lease released" and e["data"].get("lease_id") == stale.lease_id]
