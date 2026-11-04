# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 431, round 4 (issue 390). Fixed here: what can hold or free hardware wrongly. The other round 4 findings are public issues.

1. A parked hardware test runs under the coder's lease and records no `test_lease`. A helper it left behind must keep that lease in
   `_release_all` and `_release_from_memory`, and only that lease: an unrelated test lease is still released.
2. A restart's `_recover_test_leases` gave back a test lease before anything looked at the helpers the ledger remembers.
3. The helpers' first ledger entry ("hardware test helpers stopped", with `stopped: false` records) is enough to remember them: the second entry
   ("... survived") may never have been written.
4. `_gone` read every failure of /proc/<pid>/stat as "the process is gone". A file that cannot be read or parsed leaves the process unobservable,
   and an unobservable helper counts as running.
5. The identities of live helpers are in memory before the first ledger append of `_record_test_helpers`, as on the interrupted path.

Fake /proc trees and one real sleeping process. No chip is used."""
import os
from types import SimpleNamespace

import pytest

from orchard import agent, kernelprobe as kp
from orchard.adapters import Lease
from orchard.supervisor import Blocked, Supervisor

from test_spawn_checked_round3 import FakeAdapter, FakeLedger, sleeper, stub_supervisor  # noqa: F401  (sleeper is a fixture)


def survivors_of(rec):
    out = agent.Survivors()
    out.append(rec)
    return out


def coder_lease(lease_id="C1"):
    return Lease(lease_id=lease_id, chips=("0000:01:00.0",), dev_indices=(0,), env={}, units=("u",))


def coder_ledger(lease):
    return FakeLedger([{"event": "decision", "stage": None, "data": {"decision": "coder started", "lease": lease.record(), "server": {"owned": True}}}])


class Coder:
    """A coder whose stop must not be reached while a helper still holds its lease."""
    owned = True

    def adopt(self, rec):
        pass

    def confirm_stopped(self):
        raise AssertionError("the release went on to stop the coder although a helper holds its lease")


# ---- 1. a parked test's helper holds the coder's lease -----------------------------------------------------------------------------------------

def test_a_helper_of_a_parked_test_keeps_the_coders_lease_in_release_all(sleeper):
    child, rec = sleeper
    lease = coder_lease("C1")
    s = stub_supervisor(coder_ledger(lease))
    s.test_lease = None
    s.coder = Coder()
    s.adapter = FakeAdapter()
    with pytest.raises(Blocked):
        s._record_test_helpers(2, survivors_of(rec), lease_ids=["C1"])
    assert s._release_all() is False and s.adapter.released == []
    assert any("lease was kept" in str(e["data"].get("what", "")) for e in s.ledger.entries)


def test_release_from_memory_keeps_only_the_lease_the_helper_used(sleeper):
    child, rec = sleeper
    s = stub_supervisor()
    s.test_lease = SimpleNamespace(lease_id="T1", chips=["c1"])
    s.coder_lease = SimpleNamespace(lease_id="C1", chips=["c0"])
    s.coder = SimpleNamespace(owned=False)             # a coder another process owns: its lease goes back without a stop
    s.adapter = FakeAdapter()
    with pytest.raises(Blocked):
        s._record_test_helpers(2, survivors_of(rec), lease_ids=["C1"])
    assert s._release_from_memory() is False
    assert s.adapter.released == [] and [x.lease_id for x in s.test_adapter.released] == ["T1"]


def test_the_leases_a_helper_used_come_back_from_the_ledger_after_a_restart(sleeper):
    child, rec = sleeper
    first = stub_supervisor()
    with pytest.raises(Blocked):
        first._record_test_helpers(2, survivors_of(rec), lease_ids=["C1"])
    again = stub_supervisor(FakeLedger(first.ledger.entries))
    assert [r["pid"] for r in again._helpers_holding("C1")] == [rec["pid"]]
    assert again._helpers_holding("OTHER") == []


def test_a_helper_with_no_recorded_lease_holds_every_lease(sleeper):
    child, rec = sleeper
    s = stub_supervisor()
    s.surviving_helpers = [rec]
    assert s._helpers_holding("ANY") != []


# ---- 2. startup recovery looks at the remembered helpers first ---------------------------------------------------------------------------------

def test_startup_recovery_does_not_give_back_a_lease_a_remembered_helper_holds(sleeper, tmp_path):
    child, rec = sleeper
    taken = {"lease_id": "T9", "chips": ["c0"], "dev_indices": [0], "env": {}, "units": ["u"]}
    ledger = FakeLedger([
        {"event": "decision", "stage": 2, "data": {"decision": "test lease taken", "test_lease": taken}},
        {"event": "decision", "stage": 2, "data": {"decision": "hardware test helpers survived", "servers": [rec], "lease_ids": ["T9"]}}])
    s = stub_supervisor(ledger)
    s.test_lease = None
    s.run_dir = tmp_path
    s.lab = None
    s._sweep = lambda n: None
    s._recover_test_leases()
    assert s.test_adapter.released == []
    assert any("kept" in str(e["data"].get("note", "")) for e in ledger.entries if e["data"].get("decision") == "test lease released") or \
        any("still running" in str(e["data"].get("what", "")) for e in ledger.entries)


def test_startup_recovery_still_gives_back_a_lease_when_the_helper_is_gone(sleeper, tmp_path):
    child, rec = sleeper
    child.kill()
    child.wait()
    taken = {"lease_id": "T9", "chips": ["c0"], "dev_indices": [0], "env": {}, "units": ["u"]}
    ledger = FakeLedger([
        {"event": "decision", "stage": 2, "data": {"decision": "test lease taken", "test_lease": taken}},
        {"event": "decision", "stage": 2, "data": {"decision": "hardware test helpers survived", "servers": [rec], "lease_ids": ["T9"]}}])
    s = stub_supervisor(ledger)
    s.test_lease = None
    s.run_dir = tmp_path
    s.lab = None
    s._sweep = lambda n: None
    s._recover_test_leases()
    assert [x.lease_id for x in s.test_adapter.released] == ["T9"]


# ---- 3. the first ledger entry is enough ---------------------------------------------------------------------------------------------------------

def test_a_stopped_entry_with_a_live_helper_is_remembered_without_the_survived_entry(sleeper):
    child, rec = sleeper
    ledger = FakeLedger([{"event": "decision", "stage": 2, "data": {"decision": "hardware test helpers stopped", "servers": [rec, {"pid": 5, "stopped": True}],
                                                                   "problems": [], "lease_ids": ["L1"]}}])
    s = stub_supervisor(ledger)
    assert [r["pid"] for r in s._helpers_running()] == [rec["pid"]]
    assert s._release_test_lease() is False and s.test_adapter.released == []


def test_a_cleared_entry_ends_the_memory_also_of_a_stopped_entry(sleeper):
    child, rec = sleeper
    ledger = FakeLedger([{"event": "decision", "stage": 2, "data": {"decision": "hardware test helpers stopped", "servers": [rec], "problems": []}},
                         {"event": "decision", "stage": 2, "data": {"decision": "hardware test helpers cleared", "servers": [rec]}}])
    assert stub_supervisor(ledger)._helpers_running() == []


# ---- 4. an unobservable process is not a gone process ------------------------------------------------------------------------------------------

def fake_proc(tmp_path, pid, stat=None, *, as_dir=False):
    root = tmp_path / "proc"
    (root / str(pid)).mkdir(parents=True, exist_ok=True)
    if as_dir:
        (root / str(pid) / "stat").mkdir()
    elif stat is not None:
        (root / str(pid) / "stat").write_text(stat)
    return str(root)


GOOD = "7 (x y) S 1 7 7 0 -1 4194304 1 0 0 0 0 0 0 0 20 0 1 0 555 1 1 18446744073709551615"


def test_a_process_that_is_there_is_not_gone(tmp_path):
    root = fake_proc(tmp_path, 7, GOOD)
    assert kp._gone(root, 7, 555) is False and kp._gone(root, 7, 556) is True


def test_a_missing_entry_is_gone(tmp_path):
    root = fake_proc(tmp_path, 8, None)                 # the directory with no stat file
    (tmp_path / "proc" / "8").rmdir()
    assert kp._gone(root, 8, 1) is True


def test_a_stat_file_that_cannot_be_read_or_parsed_is_not_gone(tmp_path):
    assert kp._gone(fake_proc(tmp_path, 9, as_dir=True), 9, 1) is False        # an error other than "no such file"
    assert kp._gone(fake_proc(tmp_path, 10, "garbage"), 10, 1) is False         # malformed
    assert kp.still_running({"pid": 10, "start": 1}, proc_root=str(tmp_path / "proc")) is True


# ---- 5. the identities are in memory before the ledger is touched ------------------------------------------------------------------------------------

def test_the_identities_are_in_memory_when_the_ledger_cannot_be_written(sleeper):
    child, rec = sleeper
    s = stub_supervisor(FakeLedger(fail=True))
    with pytest.raises(OSError):
        s._record_test_helpers(2, survivors_of(rec), lease_ids=["T1"])
    assert s.surviving_helpers == [rec]


# ---- wiring: `_spawn` hands the leases the test ran under to the recorder -------------------------------------------------------------------------

def spawn_stub(tmp_path, ledger=None):
    s = stub_supervisor(ledger)
    s.run_dir = tmp_path
    s.extra_env = {}
    s.run_label = "orchard.test=x"
    s.lab = None
    s.clock = lambda: 0.0
    s.paths = SimpleNamespace(cache_root=str(tmp_path / "cache"))
    s._moved_aside_record = lambda p: {}
    s._block_if_off_bus = lambda *a, **k: None
    return s


@pytest.mark.parametrize("how", ["normal", "interrupted"])
def test_spawn_gives_the_recorder_the_lease_ids_of_the_test(how, sleeper, tmp_path, monkeypatch):
    child, rec = sleeper
    from orchard import supervisor

    def fake_spawn_checked(command, run_dir, env, deadline, out):
        agent._SURVIVORS.get().append(dict(rec))
        if how == "interrupted":
            raise KeyboardInterrupt()
        return 0, False
    monkeypatch.setattr(supervisor, "spawn_checked", fake_spawn_checked)
    s = spawn_stub(tmp_path)
    try:
        s._spawn(2, "true", ["c0", "c1"], [0, 1], 10.0, tmp_path / "out.txt", lease_ids=["T1", "T2"])
    except (Blocked, KeyboardInterrupt):
        pass
    assert s.surviving_helper_leases == ["T1", "T2"]
    kept = s.ledger.decisions("hardware test helpers stopped")
    assert kept and kept[0]["lease_ids"] == ["T1", "T2"]


# ---- 6. the final sweeps' records survive an interrupt -----------------------------------------------------------------------------------------

from orchard import kernelprobe as _kp  # noqa: E402
from orchard.agent import Outcome  # noqa: E402
from test_kernelprobe import FACTS, NO_KERNEL, check_writer, fake_kenv  # noqa: E402,F401  (fake_kenv is a fixture)


class _QuietStep:
    def __init__(self, **kw):
        self.tools = kw["tools"]

    def run(self, system, user):
        return Outcome("done", 1, "done")


def test_the_final_sweeps_records_are_in_the_saved_attempt_when_an_interrupt_ends_the_sweep(tmp_path, fake_kenv, monkeypatch):
    """The sweep after the agent step stopped a helper and was then cut short by an interrupt: the record is in `into`, not in the return value.
    The partial attempt on disk must still list it (PR 431, round 4)."""
    import json
    monkeypatch.setattr(_kp, "AgentStep", _QuietStep)
    rec = {"pid": 4242, "cmdline": "srv", "stopped": True, "boot_id": "b", "start": 7}

    def sweep(run_dir, *, into=None, status=None, **kw):
        into.append(dict(rec))
        status["problems"] = ["an interrupt cut a scan short"]
        raise KeyboardInterrupt()
    monkeypatch.setattr(_kp, "sweep_strays", sweep)
    with pytest.raises(KeyboardInterrupt):
        _kp.run_attempt(op="add", n=1, probe_dir=tmp_path / "probe", env=fake_kenv, endpoint="http://x/v1", model="m", coder_model="m",
                        probe_id="t", run_watched=check_writer(NO_KERNEL), tt_lang=FACTS)
    saved = json.loads((_kp.verdict_dir(tmp_path / "probe", "add", 1) / "attempt.json").read_text())
    assert saved["status"] == "interrupted" and rec in saved["swept"]
    assert "an interrupt cut a scan short" in saved["sweep_problems"]
