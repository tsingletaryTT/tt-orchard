# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 431, round 3 (issue 390): the cleanup after a killed command must not be cut short by a second interrupt, must say when it cannot
vouch for its result, and must keep what it found on every way out.

Four causes stood behind the review's eleven findings:
  A. the stop-signal handlers were swapped and restored in steps that a signal could cut short, and a freeze interrupted by a signal skipped
     the SIGKILL pass over processes it had already stopped;
  B. the rescan loop ended on its bound without freezing the process its last scan found, and said nothing about it;
  C. a process list that could not be read looked like a scan that found nothing, and a sweep that failed looked like an empty one, so the
     supervisor could release a lease or restore the coder while a helper still ran;
  D. what the sweep stopped was lost when an interrupt ended the call (the tools' record, the supervisor's ledger entry, the partial attempt).

Everything here uses a fake /proc, recorders in place of kill, and processes that only sleep. No chip is used."""
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from orchard import agent, kernelprobe as kp, supervisor
from orchard.supervisor import Blocked, Supervisor


# ---- A: the held stop signals -----------------------------------------------------------------------------------------------------------------

class Cut(BaseException):
    """Stands in for the BaseException that the supervisor's SIGINT and SIGTERM handlers raise."""


@pytest.fixture
def handlers():
    """Two recognisable original handlers for SIGINT and SIGTERM, restored when the test ends."""
    def mine_int(signum, frame):
        pass

    def mine_term(signum, frame):
        pass
    before = {s: signal.getsignal(s) for s in agent.STOP_SIGNALS}
    signal.signal(signal.SIGINT, mine_int)
    signal.signal(signal.SIGTERM, mine_term)
    try:
        yield {signal.SIGINT: mine_int, signal.SIGTERM: mine_term}
    finally:
        for s, h in before.items():
            signal.signal(s, h)


class FlakySignal:
    """A stand-in for the `signal` module in orchard.agent: the named call raises `Cut` before it takes effect (a stop signal arrived and the
    handler that was still in place raised), every other call passes through."""

    def __init__(self, fail_on):
        self.fail_on = list(fail_on)      # [(signum, "install" | "restore")], each used once
        self.calls = []

    def __getattr__(self, name):
        return getattr(signal, name)

    def signal(self, signum, handler):
        kind = "install" if getattr(handler, "__name__", "") == "_hold" else "restore"
        self.calls.append((signum, kind))
        if (signum, kind) in self.fail_on:
            self.fail_on.remove((signum, kind))
            raise Cut()
        return signal.signal(signum, handler)


def test_an_interrupt_during_the_install_is_kept_and_both_signals_end_up_held_and_then_restored(handlers, monkeypatch):
    """A signal that arrives before SIGTERM is held runs SIGTERM's original handler, which raises. The constructor must not raise, must keep the
    interrupt, must still hold both signals for the cleanup, and must put both originals back."""
    flaky = FlakySignal([(signal.SIGTERM, "install")])
    monkeypatch.setattr(agent, "signal", flaky)
    held = agent._HeldStopSignals()                       # does not raise
    assert isinstance(held.exc, Cut)
    assert getattr(signal.getsignal(signal.SIGINT), "__name__", "") == "_hold"
    assert getattr(signal.getsignal(signal.SIGTERM), "__name__", "") == "_hold"      # the second try took effect
    late = held.release()
    assert isinstance(late, Cut)                                                     # the interrupt is not lost
    assert signal.getsignal(signal.SIGINT) is handlers[signal.SIGINT]
    assert signal.getsignal(signal.SIGTERM) is handlers[signal.SIGTERM]


def test_an_interrupt_during_the_restore_still_puts_every_handler_back(handlers, monkeypatch):
    held = agent._HeldStopSignals()
    flaky = FlakySignal([(signal.SIGINT, "restore")])
    monkeypatch.setattr(agent, "signal", flaky)
    late = held.release()
    assert isinstance(late, Cut)
    assert signal.getsignal(signal.SIGINT) is handlers[signal.SIGINT]
    assert signal.getsignal(signal.SIGTERM) is handlers[signal.SIGTERM]


def test_the_cleanup_runs_both_steps_when_the_install_was_interrupted(handlers, monkeypatch):
    ran = []
    monkeypatch.setattr(agent, "_kill_session", lambda proc: ran.append("kill"))
    monkeypatch.setattr(agent, "_sweep_helpers", lambda tag, stdout: ran.append("sweep"))
    monkeypatch.setattr(agent, "signal", FlakySignal([(signal.SIGINT, "install")]))
    late = agent._cleanup(None, "tag", None)
    assert ran == ["kill", "sweep"]
    assert isinstance(late, Cut)
    assert signal.getsignal(signal.SIGINT) is handlers[signal.SIGINT]
    assert signal.getsignal(signal.SIGTERM) is handlers[signal.SIGTERM]


def test_a_signal_that_arrives_while_held_is_replayed_after_the_cleanup(handlers):
    seen = []
    signal.signal(signal.SIGTERM, lambda s, f: seen.append(s))
    held = agent._HeldStopSignals()
    signal.raise_signal(signal.SIGTERM)                  # recorded, not acted on
    assert seen == []
    assert held.release() is None
    assert seen == [signal.SIGTERM]


# ---- B and C: the sweep ----------------------------------------------------------------------------------------------------------------------

class FakeProc:
    """A fake /proc with processes that carry ORCHARD_RUN_DIR=<dir>, and the kill stand-in that removes an entry when a SIGKILL arrives."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "proc"
        (self.root / "sys" / "kernel" / "random").mkdir(parents=True)
        (self.root / "sys" / "kernel" / "random" / "boot_id").write_text("boot-1\n")
        self.dir = tmp_path / "attempt"
        self.dir.mkdir()
        self.sent: list[tuple[int, int]] = []
        self.on_signal = None
        monkeypatch.setattr(kp.os, "getpgid", lambda pid: pid + 100000)      # never a group leader: the pid itself is signalled
        monkeypatch.setattr(kp, "SWEEP_POLL_S", 0.0)

    def add(self, pid: int, start: int = 500, state: str = "S") -> None:
        d = self.root / str(pid)
        d.mkdir(exist_ok=True)
        (d / "environ").write_bytes(b"ORCHARD_RUN_DIR=" + str(self.dir.resolve()).encode() + b"\0")
        (d / "cmdline").write_bytes(b"helper\0--serve\0")
        (d / "stat").write_text(f"{pid} (helper) {state} " + " ".join(["1"] * 18) + f" {start} 0 0\n")

    def kill(self, pid, sig):
        self.sent.append((pid, int(sig)))
        if self.on_signal is not None:
            self.on_signal(pid, sig)
        if sig == signal.SIGKILL:
            import shutil
            shutil.rmtree(self.root / str(pid), ignore_errors=True)

    def sweep(self, **kw):
        return kp.sweep_strays(self.dir, proc_root=str(self.root), kill=self.kill, killpg=self.kill, grace_s=0.0, **kw)

    def sent_to(self, pid: int) -> list[int]:
        return [s for p, s in self.sent if p == pid]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return FakeProc(tmp_path, monkeypatch)


def test_a_scan_that_still_finds_a_new_process_on_its_last_round_stops_it_and_reports_an_unsettled_set(fake, monkeypatch):
    """B. A helper that keeps starting children: every scan finds a new pid. The bound ends the loop, and the process the last scan found
    must have been stopped too (it was not before), with the sweep saying that the set never settled."""
    monkeypatch.setattr(kp, "SWEEP_FREEZE_ROUNDS", 3)
    counter = iter(range(7001, 7100))

    def scan(run_dir, **kw):
        pid = next(counter)
        fake.add(pid)
        return [pid]
    monkeypatch.setattr(kp, "scan_strays", scan)
    status: dict = {}
    records = fake.sweep(identity=True, status=status)
    pids = [r["pid"] for r in records]
    assert len(pids) == 1 + 3                                      # the first scan, then one new pid per round
    last = pids[-1]
    assert signal.SIGSTOP in fake.sent_to(last)                    # the one the last scan found is stopped before the SIGKILLs go out
    assert all(signal.SIGKILL in fake.sent_to(p) for p in pids)
    assert status["complete"] is False and any("never settled" in p for p in status["problems"])
    assert all(r["stopped"] for r in records)


def test_a_set_that_settles_is_complete(fake, monkeypatch):
    fake.add(7001)
    status: dict = {}
    records = fake.sweep(identity=True, status=status)
    assert [r["pid"] for r in records] == [7001] and records[0]["stopped"] is True
    assert status == {"problems": [], "complete": True}


def test_an_interrupt_during_the_freeze_still_sends_every_sigkill_and_then_propagates(fake):
    """A. A signal that lands while the freeze stops processes (SIGSTOP) raised past an `except Exception`, so no SIGKILL went out and the
    processes stayed frozen. Every recorded process must get its SIGKILL, and the first interrupt is the one that propagates."""
    fake.add(7001)
    fake.add(7002)
    armed = []

    def on_signal(pid, sig):
        if sig == signal.SIGSTOP and not armed:
            armed.append(1)
            raise Cut()
    fake.on_signal = on_signal
    records: list = []
    status: dict = {}
    with pytest.raises(Cut):
        fake.sweep(into=records, status=status)
    assert {r["pid"] for r in records} == {7001, 7002}
    assert all(signal.SIGKILL in fake.sent_to(p) for p in (7001, 7002))
    assert all(r["stopped"] for r in records)                       # the records say so, although the sweep was cut short
    assert any("interrupt cut the freeze" in p for p in status["problems"]) and status["complete"] is False


def test_an_interrupt_during_the_sigkill_pass_runs_the_pass_again(fake):
    """A second signal can land in the SIGKILL loop itself. The pass is run again, so every recorded process still gets its SIGKILL."""
    fake.add(7001)
    fake.add(7002)
    armed = []

    def on_signal(pid, sig):
        if sig == signal.SIGKILL and not armed:
            armed.append(1)
            raise Cut()
    fake.on_signal = on_signal
    with pytest.raises(Cut):
        fake.sweep()
    assert all(signal.SIGKILL in fake.sent_to(p) for p in (7001, 7002))
    assert 7001 not in [int(p.name) for p in fake.root.iterdir() if p.name.isdigit()]


def test_an_interrupt_in_the_first_scan_does_not_skip_the_sweep(fake, monkeypatch):
    fake.add(7001)
    real_scan = kp.scan_strays
    calls = []

    def scan(run_dir, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise Cut()
        return real_scan(run_dir, **kw)
    monkeypatch.setattr(kp, "scan_strays", scan)
    records: list = []
    with pytest.raises(Cut):
        fake.sweep(into=records)
    assert [r["pid"] for r in records] == [7001]
    assert signal.SIGKILL in fake.sent_to(7001)


def test_an_error_in_the_sweep_is_a_problem_and_never_an_exception(fake, monkeypatch):
    def broken(run_dir, **kw):
        raise OSError("proc went away")
    monkeypatch.setattr(kp, "scan_strays", broken)
    status: dict = {}
    assert fake.sweep(status=status) == []
    assert status["complete"] is False and any("OSError" in p for p in status["problems"])


def test_a_process_list_that_cannot_be_read_is_not_an_empty_sweep(tmp_path):
    """C. scan_strays gave [] when /proc could not be listed, which the supervisor read as 'nothing is left'."""
    missing = tmp_path / "no-proc"
    errors: list = []
    assert kp.scan_strays(tmp_path, proc_root=str(missing), errors=errors) == [] and "could not be read" in errors[0]
    status: dict = {}
    assert kp.sweep_strays(tmp_path, proc_root=str(missing), grace_s=0.0, status=status) == []
    assert status["complete"] is False and any("could not be read" in p for p in status["problems"])


def test_a_readable_list_with_nothing_in_it_is_a_complete_sweep(fake):
    status: dict = {}
    assert fake.sweep(status=status) == []
    assert status == {"problems": [], "complete": True}


def test_still_running_judges_a_record_by_its_identity(fake):
    fake.add(7001, start=500)
    root = str(fake.root)
    rec = {"pid": 7001, "start": 500, "boot_id": "boot-1"}
    assert kp.still_running(rec, proc_root=root) is True
    assert kp.still_running({**rec, "start": 501}, proc_root=root) is False       # the pid was reused by another process
    assert kp.still_running({**rec, "boot_id": "boot-0"}, proc_root=root) is False   # another boot
    assert kp.still_running({"pid": 7002, "start": 1}, proc_root=root) is False   # no such process
    fake.add(7003, state="Z")
    assert kp.still_running({"pid": 7003}, proc_root=root) is False               # a zombie is gone
    assert kp.still_running({"cmdline": "x"}, proc_root=root) is True             # no pid: cannot be judged, so it counts as running
    assert kp.still_running({"pid": True}, proc_root=root) is True


# ---- D: what the sweep found reaches the collector on every way out -------------------------------------------------------------------------------

def test_the_records_reach_the_collector_and_stdout_when_an_interrupt_cuts_the_sweep_short(monkeypatch, tmp_path):
    def sweep(tag, **kw):
        kw["into"].append({"pid": 4242, "cmdline": "srv", "stopped": True, "boot_id": "b", "start": 9})
        raise Cut()
    monkeypatch.setattr(kp, "sweep_strays", sweep)
    out = (tmp_path / "out.txt").open("wb")
    with agent.collect_survivors() as survivors:
        with pytest.raises(Cut):
            agent._sweep_helpers("tag", out)
    out.close()
    assert [r["pid"] for r in survivors] == [4242]
    assert b"pid 4242" in (tmp_path / "out.txt").read_bytes()


def test_a_sweep_that_failed_is_a_problem_on_the_collector_and_not_an_empty_list(monkeypatch):
    def sweep(tag, **kw):
        raise OSError("boom")
    monkeypatch.setattr(kp, "sweep_strays", sweep)
    with agent.collect_survivors() as survivors:
        agent._sweep_helpers("tag", None)
    assert list(survivors) == [] and any("OSError" in p for p in survivors.problems)


def test_the_sweeps_own_problems_reach_the_collector(monkeypatch):
    def sweep(tag, **kw):
        kw["status"].update(problems=["the process list /proc could not be read (OSError)"], complete=False)
        return []
    monkeypatch.setattr(kp, "sweep_strays", sweep)
    with agent.collect_survivors() as survivors:
        agent._sweep_helpers("tag", None)
    assert survivors.problems == ["the process list /proc could not be read (OSError)"]


def test_a_sweep_that_returns_its_records_without_filling_into_still_reaches_the_collector(monkeypatch):
    rec = {"pid": 5, "cmdline": "x", "stopped": False, "boot_id": "b", "start": 1}
    monkeypatch.setattr(kp, "sweep_strays", lambda tag, **kw: [rec])
    with agent.collect_survivors() as survivors:
        agent._sweep_helpers("tag", None)
    assert list(survivors) == [rec]


def test_the_tools_keep_the_records_when_spawn_checked_is_interrupted(monkeypatch, tmp_path):
    """The records were lost: `self.swept.extend(swept)` ran after the call and an interrupt skipped it, so an interrupted kernel-probe attempt
    omitted exactly the helpers that its own cleanup had just stopped."""
    def interrupted(command, run_dir, env, timeout, out):
        sink = agent._SURVIVORS.get()
        sink.append({"pid": 77, "cmdline": "srv", "stopped": True, "boot_id": "b", "start": 3})
        sink.problems.append("a scan that never settled")
        raise Cut()
    monkeypatch.setattr(agent, "spawn_checked", interrupted)
    monkeypatch.setattr(agent, "check_string", lambda *a, **k: None)
    (tmp_path / "stage").mkdir()
    tools = agent.Tools(tmp_path, tmp_path / "stage", {})
    with pytest.raises(Cut):
        tools.shell("sleep 1")
    assert [r["pid"] for r in tools.swept] == [77]
    assert tools.sweep_problems == ["a scan that never settled"]


class FakeHwCheck:
    def __init__(self, records):
        self.records = records

    def summary(self):
        return {"calls": len(self.records)}


def _kw(tmp_path):
    return {"op": "add", "n": 1, "probe_dir": tmp_path, "coder_model": "m", "probe_id": "p", "skills": None}


def test_a_partial_attempt_is_saved_when_only_the_sweep_has_records(tmp_path):
    """A shell timeout, then an interrupt on a later turn, with no hardware call: the helpers' identities were dropped with the attempt."""
    tools = SimpleNamespace(swept=[{"pid": 9, "stopped": True}], sweep_problems=[])
    saved = kp._partial_attempt({"tools": tools}, _kw(tmp_path), Cut())
    assert saved is not None and saved["swept"] == [{"pid": 9, "stopped": True}]
    assert saved["agent_hw"] == {"enabled": False}
    assert list(tmp_path.rglob("attempt.json"))                       # and it was written to disk, in the probe's verdict directory


def test_a_partial_attempt_is_saved_for_sweep_problems_alone(tmp_path):
    tools = SimpleNamespace(swept=[], sweep_problems=["a scan that never settled"])
    saved = kp._partial_attempt({"tools": tools}, _kw(tmp_path), Cut())
    assert saved is not None and saved["sweep_problems"] == ["a scan that never settled"]


def test_a_partial_attempt_with_nothing_to_keep_is_still_none(tmp_path):
    assert kp._partial_attempt({"tools": SimpleNamespace(swept=[], sweep_problems=[])}, _kw(tmp_path), Cut()) is None
    assert kp._partial_attempt({}, _kw(tmp_path), Cut()) is None


def test_a_partial_attempt_with_hardware_calls_keeps_them_as_before(tmp_path):
    saved = kp._partial_attempt({"hwcheck": FakeHwCheck([1]), "tools": SimpleNamespace(swept=[], sweep_problems=[])}, _kw(tmp_path), Cut())
    assert saved is not None and saved["agent_hw"] == {"calls": 1}


# ---- C: the supervisor does not let a lease go, or the coder come back, over a helper that still runs ---------------------------------------------

class FakeLedger:
    def __init__(self, entries=None, fail=False):
        self.entries = list(entries or [])
        self.fail = fail

    def append(self, event, stage, **data):
        if self.fail:
            raise OSError("ledger is full")
        self.entries.append({"event": event, "stage": stage, "data": data})

    def read(self):
        return list(self.entries)

    def decisions(self, name):
        return [e["data"] for e in self.entries if e["event"] == "decision" and e["data"].get("decision") == name]


class FakeAdapter:
    def __init__(self):
        self.released = []

    def release(self, lease, **kw):
        self.released.append(lease)


@pytest.fixture
def sleeper():
    """A real process that sleeps, and the identity a sweep records for it (pid, boot id, start time)."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        for _ in range(100):
            if kp._proc_state("/proc", child.pid):
                break
            time.sleep(0.01)
        yield child, {"pid": child.pid, "cmdline": "sleeper", "stopped": False, "boot_id": kp._boot_id("/proc"),
                      "start": kp._proc_state("/proc", child.pid)[1]}
    finally:
        child.kill()
        child.wait()


def stub_supervisor(ledger=None):
    s = object.__new__(Supervisor)
    s.ledger = ledger or FakeLedger()
    s.surviving_containers = []
    s.surviving_helpers = []
    s.test_adapter = FakeAdapter()
    s.test_lease = SimpleNamespace(lease_id="L1", chips=["c0"])
    s.coder_lease = None
    return s


def test_the_test_lease_is_kept_while_a_remembered_helper_runs_and_released_when_it_is_gone(sleeper):
    child, rec = sleeper
    s = stub_supervisor()
    s.surviving_helpers = [rec]
    assert s._release_test_lease() is False and s.test_adapter.released == []
    assert any(e["data"].get("what", "").startswith("the test lease was kept: a helper") for e in s.ledger.entries)
    child.kill()
    child.wait()
    assert s._release_test_lease() is True and len(s.test_adapter.released) == 1
    assert s.ledger.decisions("hardware test helpers cleared") and s.surviving_helpers == []


def test_a_restart_remembers_a_running_helper_from_the_ledger(sleeper):
    """The memory is gone after a restart; the ledger still names the helper, so the lease stays and the entry is not forgotten."""
    child, rec = sleeper
    ledger = FakeLedger([{"event": "decision", "stage": 2, "data": {"decision": "hardware test helpers survived", "servers": [rec]}}])
    s = stub_supervisor(ledger)
    assert s.surviving_helpers == []
    assert s._release_test_lease() is False and s.test_adapter.released == []


def test_a_cleared_entry_in_the_ledger_ends_the_memory(sleeper):
    child, rec = sleeper
    ledger = FakeLedger([{"event": "decision", "stage": 2, "data": {"decision": "hardware test helpers survived", "servers": [rec]}},
                         {"event": "decision", "stage": 2, "data": {"decision": "hardware test helpers cleared", "servers": [rec]}}])
    s = stub_supervisor(ledger)
    assert s._release_test_lease() is True


def test_the_coder_is_not_restored_over_a_helper_that_still_runs(sleeper):
    child, rec = sleeper
    s = stub_supervisor()
    s.surviving_helpers = [rec]
    with pytest.raises(Blocked) as err:
        s._ensure_coder()
    assert "still running" in str(err.value) and str(rec["pid"]) in str(err.value)


def test_the_release_from_memory_keeps_the_test_lease_under_a_running_helper(sleeper):
    child, rec = sleeper
    s = stub_supervisor()
    s.surviving_helpers = [rec]
    assert s._release_from_memory() is False and s.test_adapter.released == []
    child.kill()
    child.wait()
    assert s._release_from_memory() is True and len(s.test_adapter.released) == 1


def test_the_hardware_test_that_left_a_helper_remembers_its_identity_and_blocks(sleeper):
    child, rec = sleeper
    s = stub_supervisor()
    survivors = agent.Survivors()
    survivors.append(rec)
    with pytest.raises(Blocked):
        s._record_test_helpers(2, survivors)
    assert s.surviving_helpers == [rec]
    kept = s.ledger.decisions("hardware test helpers survived")
    assert len(kept) == 1 and kept[0]["servers"] == [rec]
    assert len(s.ledger.decisions("hardware test helpers stopped")) == 1


def test_a_sweep_that_could_not_be_completed_blocks_although_it_found_nothing():
    """C. An empty list from a sweep that failed is not proof that nothing is left."""
    s = stub_supervisor()
    survivors = agent.Survivors()
    survivors.problems.append("the process list /proc could not be read (OSError)")
    with pytest.raises(Blocked) as err:
        s._record_test_helpers(2, survivors)
    assert "could not be completed" in str(err.value) and "could not be read" in str(err.value)
    assert s.surviving_helpers == []                  # nothing can be named, so no lease is held back; the run waits for the operator


def test_a_clean_sweep_with_nothing_found_writes_nothing_and_blocks_nothing():
    s = stub_supervisor()
    s._record_test_helpers(2, agent.Survivors())
    assert s.ledger.entries == []


def test_a_stopped_helper_is_recorded_and_does_not_block():
    s = stub_supervisor()
    survivors = agent.Survivors()
    survivors.append({"pid": 5, "cmdline": "x", "stopped": True, "boot_id": "b", "start": 1})
    s._record_test_helpers(2, survivors)
    assert len(s.ledger.decisions("hardware test helpers stopped")) == 1 and s.surviving_helpers == []


def test_on_an_interrupted_exit_the_helpers_are_recorded_remembered_and_nothing_raises(sleeper):
    child, rec = sleeper
    s = stub_supervisor()
    survivors = agent.Survivors()
    survivors.append(rec)
    survivors.append({"pid": 6, "cmdline": "y", "stopped": True, "boot_id": "b", "start": 2})
    s._note_test_helpers_on_exit(2, survivors)                 # no Blocked: the interrupt on its way is the one that propagates
    assert s.surviving_helpers == [rec]
    stopped = s.ledger.decisions("hardware test helpers stopped")
    assert len(stopped) == 1 and stopped[0]["interrupted"] is True and len(stopped[0]["servers"]) == 2
    assert s.ledger.decisions("hardware test helpers survived")[0]["servers"] == [rec]


def test_on_an_interrupted_exit_a_failing_ledger_does_not_hide_the_helpers_from_memory(sleeper):
    child, rec = sleeper
    s = stub_supervisor(FakeLedger(fail=True))
    survivors = agent.Survivors()
    survivors.append(rec)
    s._note_test_helpers_on_exit(2, survivors)                 # does not raise
    assert s.surviving_helpers == [rec]


def test_an_incomplete_sweep_of_the_graft_trial_servers_blocks(monkeypatch, tmp_path):
    s = stub_supervisor()

    def sweep(target, **kw):
        kw["status"].update(problems=["new helpers were still appearing after 8 scans, so the set of helpers never settled"], complete=False)
        return []
    monkeypatch.setattr(supervisor.kernelprobe, "sweep_strays", sweep)
    with pytest.raises(Blocked) as err:
        s._stop_trial_servers(2, tmp_path, "after the trial test")
    assert "could not be completed" in str(err.value)
