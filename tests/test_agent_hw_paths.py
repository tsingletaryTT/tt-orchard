# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Every path between taking a queue ticket and the final release, through the probe's own wiring (kernelprobe.make_hardware_check
over hwlease.GozerLeases) and a simulated gozer that keeps real state: a queue, a history and leases owned by a pid. The table of
paths, with the test of each, is in docs/superpowers/specs/2026-10-27-agent-hardware-check-design.md. No test touches the real
gozer (tests/conftest.py refuses it) or a chip."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from orchard import hwlease, hwtool, kernelprobe as kp
from orchard.commands import CommandResult
from test_agent_hw import CLOCK, GOOD_HW_SET, attempt_dir, canned, env_ready, fake_kenv, probe_args, skills_current, ttlang_skill  # noqa: F401

PID, WHO = 4321, "claude:test"
CHIPS = ["0000:01:00.0", "0000:02:00.0"]


def res(code=0, out="", err="", **kw):
    return CommandResult(("gozer",), code, out, err, **kw)


class SimGozer:
    """A gozer with state. `busy_calls` acquire answers say queued before the lease is granted (the first is the request, the
    rest are claims of the same ticket). Failure switches: `cancel_mode`, `release_mode`, `acquire_fail_at`, `no_env`."""

    def __init__(self, busy_calls=0):
        self.busy_calls = busy_calls
        self.queue, self.history, self.calls = [], [], []
        self.acquires = 0
        self.cancel_mode = "ok"            # ok | fail | gone
        self.release_mode = "ok"           # ok | reset_failed | refused | overrun | interrupt_once | gone
        self.acquire_fail_at = {}          # acquire call number -> an exception or a CommandResult
        self.no_env = False
        self.history_fails = False
        self.held = {}                     # lease id -> seconds the lease has been held (fake clock stamp)
        self.counter = 0
        self.interrupted_release = False

    # the command runner
    def run(self, argv, timeout, env=None, kill_on_timeout=True):
        argv = list(argv)
        self.calls.append({"argv": argv, "timeout": timeout, "kill_on_timeout": kill_on_timeout})
        sub = argv[1]
        return getattr(self, f"_{sub}")(argv)

    def subs(self):
        return [c["argv"][1] for c in self.calls]

    def _acquire(self, argv):
        self.acquires += 1
        failure = self.acquire_fail_at.get(self.acquires)
        if isinstance(failure, BaseException):
            raise failure
        if failure is not None:
            return failure
        reason = argv[argv.index("--reason") + 1]
        ticket = argv[argv.index("--ticket") + 1] if "--ticket" in argv else None
        if self.busy_calls > 0:
            self.busy_calls -= 1
            if ticket is None:
                ticket = f"T{len(self.history) + len(self.queue) + 1}"
                self.queue.append({"ticket": ticket, "pid": PID, "who": WHO, "reason": reason})
            return res(10, json.dumps({"granted": False, "queued": True, "ticket": ticket, "position": 1, "ahead": []}))
        if ticket is not None:
            self.queue = [q for q in self.queue if q["ticket"] != ticket]
        self.counter += 1
        lease = f"L{self.counter}"
        self.held[lease] = CLOCK["now"]
        self.history.append({"event": "granted", "lease_id": lease, "pid": PID, "who": WHO, "reason": reason, "chips": CHIPS,
                             "ts": "2026-10-27T10:00:00Z"})
        out = {"granted": True, "lease_id": lease, "chips": [] if self.no_env else CHIPS, "owner_pid": PID}
        if not self.no_env:
            out["env"] = {"TT_VISIBLE_DEVICES": ",".join(CHIPS)}
        return res(0, json.dumps(out))

    def _cancel(self, argv):
        if self.cancel_mode == "fail":
            return res(1, "", "boom")
        if self.cancel_mode == "gone":
            return res(13, "no such ticket")
        self.queue = [q for q in self.queue if q["ticket"] != argv[2]]
        return res(0, "cancelled")

    def _release(self, argv):
        lease = argv[2]
        mode = self.release_mode
        if mode == "interrupt_once" and not self.interrupted_release:
            self.interrupted_release = True
            raise KeyboardInterrupt
        if mode == "refused":
            return res(15, "", "device still open")
        if mode == "overrun":
            return res(None, timed_out=True, left_running=True, pid=999)
        if mode == "gone":
            return res(13, "no such lease")
        self.finish_release(lease)
        if mode == "reset_failed":
            return res(0, json.dumps({"message": "released; NOT marked clean"}))
        return res(0, json.dumps({"message": "released"}))

    def finish_release(self, lease, seconds=None):
        """Write the `released` event (what gozer does when a release finishes, even one the probe stopped waiting for)."""
        self.history.append({"event": "released", "lease_id": lease, "duration_s": seconds if seconds is not None else
                             round(CLOCK["now"] - self.held.get(lease, CLOCK["now"]) + 42.0, 1)})

    def _history(self, argv):
        return res(1, "", "boom") if self.history_fails else res(0, json.dumps({"history": self.history}))

    def _queue(self, argv):
        return res(0, json.dumps({"queue": self.queue}))

    # reading back
    def released(self):
        return [c["argv"][2] for c in self.calls if c["argv"][1] == "release"]

    def cancelled(self):
        return [c["argv"][2] for c in self.calls if c["argv"][1] == "cancel"]

    def requests(self):
        return [c["argv"] for c in self.calls if c["argv"][1] == "acquire" and "--ticket" not in c["argv"]]

    def claims(self):
        return [c["argv"] for c in self.calls if c["argv"][1] == "acquire" and "--ticket" in c["argv"]]


class Rig:
    def __init__(self, tmp_path, fake_kenv, *, sim=None, policy=None, sleep=None):
        self.adir = attempt_dir(tmp_path, fake_kenv)
        self.sim = sim or SimGozer()
        self.harness_calls = []
        self.harness = None                          # an optional callable(argv, kw) replacing the harness run
        CLOCK["now"] = 0.0
        self.leases = hwlease.GozerLeases(who=WHO, pid=PID, run=self.sim.run, clock=lambda: CLOCK["now"],
                                          sleep=sleep or (lambda s: CLOCK.__setitem__("now", CLOCK["now"] + s)))
        self.policy = policy or hwtool.HwPolicy()
        self.check = self.make(self.adir, 1)

    def make(self, adir, n):
        return kp.make_hardware_check(adir, op="add", n=n, shared_python="/shared/python", policy=self.policy,
                                      run_watched=self._watched, leases=self.leases, clock=lambda: CLOCK["now"],
                                      sleep=lambda s: CLOCK.__setitem__("now", CLOCK["now"] + s))

    def _watched(self, argv, **kw):
        self.harness_calls.append({"argv": list(argv), "env": kw.get("env")})
        if self.harness is not None:
            return self.harness(argv, kw)
        Path(argv[argv.index("--out") + 1]).write_text(
            json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 1.0}))
        return kp.Watched(0, None, False, 3.0)

    def call(self):
        return self.check.call({})


@pytest.fixture
def rig(tmp_path, fake_kenv):
    return Rig(tmp_path, fake_kenv)


def busy_rig(tmp_path, fake_kenv, busy_calls, **kw):
    return Rig(tmp_path, fake_kenv, sim=SimGozer(busy_calls=busy_calls), **kw)


# P1 ---------------------------------------------------------------------------------------------------------------


def test_p1_a_free_board_is_leased_by_pid_the_harness_runs_under_it_and_it_is_released(rig):
    text = rig.call()
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS")
    request = rig.sim.requests()[0]
    assert request[request.index("--owner-pid") + 1] == str(PID) and request[request.index("--who") + 1] == WHO
    assert request[request.index("--chips") + 1] == "1" and request[request.index("--exact") + 1] == "1"
    assert rig.harness_calls[0]["argv"][0] == "/shared/python" and "gozer" not in rig.harness_calls[0]["argv"][0]
    assert rig.harness_calls[0]["env"]["TT_VISIBLE_DEVICES"] == ",".join(CHIPS)
    assert rig.sim.released() == ["L1"] and rig.leases.unresolved() is None


def test_p1b_a_busy_board_is_claimed_on_the_one_ticket_until_it_is_granted(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=3)
    text = rig.call()
    assert "PASS" in text and len(rig.sim.requests()) == 1 and len(rig.sim.claims()) == 3
    assert all(c[-1] == "T1" and c[-2] == "--ticket" for c in rig.sim.claims())
    assert rig.sim.cancelled() == [] and rig.sim.released() == ["L1"] and rig.leases.unresolved() is None
    assert rig.check.summary()["calls"][0]["waited_s"] >= 4.0                  # two sleeps of 2 s


# P2 ---------------------------------------------------------------------------------------------------------------


def test_p2_a_wait_that_runs_out_cancels_its_one_ticket_and_runs_nothing(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=6.0))
    text = rig.call()
    assert "BUSY" in text and rig.sim.cancelled() == ["T1"] and rig.harness_calls == [] and rig.sim.released() == []
    assert rig.leases.unresolved() is None and len(rig.sim.requests()) == 1


def test_p2b_a_zero_wait_asks_once_and_cancels(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=0.0))
    assert "BUSY" in rig.call() and rig.sim.cancelled() == ["T1"] and len(rig.sim.claims()) == 1


# P3 ---------------------------------------------------------------------------------------------------------------


def test_p3_a_ticket_that_cannot_be_cancelled_is_named_and_blocks_the_next_call(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=0.0))
    rig.sim.cancel_mode = "fail"
    text = rig.call()
    assert "could not be cancelled" in text and "T1" in text and "boom" in text
    requests_before = len(rig.sim.requests())
    second = rig.call()
    assert second.startswith("hardware_check refused") and "ticket T1 is still queued" in second
    assert len(rig.sim.requests()) == requests_before
    assert rig.check.summary()["unresolved"].startswith("ticket T1")


def test_p3b_the_next_call_may_start_once_the_ticket_has_left_the_queue(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=0.0))
    rig.sim.cancel_mode = "fail"
    rig.call()
    rig.sim.queue.clear()                                                    # gozer expired it
    rig.sim.busy_calls = 0
    assert rig.call().splitlines()[0].startswith("hardware_check 2 of 6: PASS")


# P4 ---------------------------------------------------------------------------------------------------------------


def test_p4_a_claim_whose_answer_was_lost_is_found_by_the_pid_in_gozers_history(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=1)
    original = rig.sim._acquire

    def claim_lost(argv):
        if "--ticket" in argv:
            rig.sim.busy_calls = 0
            original(argv)                                                   # gozer grants the claim ...
            raise OSError("output lost")                                     # ... but the probe never sees the answer
        return original(argv)
    rig.sim._acquire = claim_lost
    text = rig.call()
    assert "PASS" in text and rig.sim.released() == ["L1"] and rig.leases.unresolved() is None


def test_p4b_a_request_that_gozer_could_not_answer_after_granting_is_not_lost(tmp_path, fake_kenv):
    """The first request itself is cut off after gozer granted it: nothing came back, but the pid's lease is in history."""
    rig = Rig(tmp_path, fake_kenv)
    real = rig.sim._acquire

    def lost(argv):
        real(argv)
        raise OSError("output lost")
    rig.sim._acquire = lost
    text = rig.call()
    assert "gozer could not give a lease" in text and rig.harness_calls == []
    assert rig.sim.released() == ["L1"] and rig.leases.unresolved() is None   # reclaimed: the lease was never seen


# P5 ---------------------------------------------------------------------------------------------------------------


def test_p5_an_interrupt_during_the_wait_cancels_the_ticket_and_reraises(tmp_path, fake_kenv):
    def interrupt(s):
        raise KeyboardInterrupt
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, sleep=interrupt)
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.sim.cancelled() == ["T1"] and rig.sim.queue == [] and rig.sim.released() == []
    assert rig.check.summary()["calls"][0]["outcome"] == "interrupted"


def test_p5b_an_interrupt_during_the_wait_releases_a_lease_granted_in_the_meantime(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=1)
    real = rig.sim._acquire

    def claim_then_interrupt(argv):
        if "--ticket" in argv:
            rig.sim.busy_calls = 0
            real(argv)                                                       # gozer grants the claim ...
            raise KeyboardInterrupt                                          # ... and the probe is interrupted
        return real(argv)
    rig.sim._acquire = claim_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.sim.released() == ["L1"] and rig.leases.unresolved() is None


# P6 ---------------------------------------------------------------------------------------------------------------


def test_p6_a_request_gozer_refuses_is_a_harness_error_with_nothing_held(rig):
    rig.sim.acquire_fail_at = {1: res(12, json.dumps({"granted": False, "error": "no chips"}))}
    text = rig.call()
    assert "FAIL: harness_error" in text and "no chips" in text and rig.harness_calls == []
    assert rig.leases.unresolved() is None


# P7 ---------------------------------------------------------------------------------------------------------------


def test_p7_a_grant_without_a_device_list_is_released_and_never_run(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    rig.sim.no_env = True
    text = rig.call()
    assert rig.harness_calls == [] and rig.sim.released() == ["L1"] and "FAIL: harness_error" in text


# P8 ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("boom", [RuntimeError("callback failed"), KeyboardInterrupt()])
def test_p8_an_error_or_interrupt_in_the_grant_callback_still_releases_the_lease(rig, boom):
    original = rig.check.run

    def run(k, timeout_s, wait_s, on_grant, ctx):
        def failing():
            raise boom
        return original(k, timeout_s, wait_s, failing, ctx)
    rig.check.run = run
    with pytest.raises(type(boom)):
        rig.call()
    assert rig.sim.released() == ["L1"] and rig.leases.unresolved() is None and rig.harness_calls == []


# P9 ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["sweep", "gate"])
def test_p9_an_exception_in_the_checks_after_the_grant_releases_the_lease(rig, monkeypatch, where):
    calls = []
    if where == "sweep":
        def sweep(run_dir, **kw):
            calls.append(1)
            if len(calls) == 3:                    # prepare, before the request, then under the lease
                raise RuntimeError("sweep failed")
            return []
        monkeypatch.setattr(kp, "sweep_strays", sweep)
    else:
        real = kp._hw_gate

        def gate(kernel, recorded):
            calls.append(1)
            if len(calls) == 3:                    # prepare, before the request, then under the lease
                raise RuntimeError("gate failed")
            return real(kernel, recorded)
        monkeypatch.setattr(kp, "_hw_gate", gate)
    with pytest.raises(RuntimeError):
        rig.call()
    assert rig.sim.released() == ["L1"] and rig.harness_calls == [] and rig.leases.unresolved() is None


# P10 --------------------------------------------------------------------------------------------------------------


def test_p10_an_interrupt_in_the_harness_run_releases_the_lease(rig):
    def harness(argv, kw):
        raise KeyboardInterrupt
    rig.harness = harness
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.sim.released() == ["L1"] and rig.check.summary()["calls"][0]["outcome"] == "interrupted"
    assert rig.check.summary()["calls"][0]["lease_incomplete"] is False       # settled from gozer's duration


def test_p10b_an_error_in_the_harness_run_releases_the_lease_and_propagates(rig):
    def harness(argv, kw):
        raise OSError("fork failed")
    rig.harness = harness
    with pytest.raises(OSError):
        rig.call()
    assert rig.sim.released() == ["L1"]


def test_p10c_a_hang_releases_the_lease_and_turns_the_tool_off(rig):
    rig.harness = lambda argv, kw: kp.Watched(-15, "stalled: no output and no CPU use for 60 s", True, 66.0)
    text = rig.call()
    assert "HANG" in text and rig.sim.released() == ["L1"] and rig.call().startswith("hardware_check refused")


# P11 --------------------------------------------------------------------------------------------------------------


def test_p11_a_release_that_exits_0_without_a_reset_is_told_to_the_agent_and_does_not_block(rig):
    rig.sim.release_mode = "reset_failed"
    text = rig.call()
    assert "PASS" in text and "could not release the lease" in text and "NOT marked clean" in text
    assert rig.check.summary()["calls"][0]["release_error"].startswith("the lease was released but its chip reset")
    assert rig.leases.unresolved() is None and rig.call().splitlines()[0].startswith("hardware_check 2 of 6")


# P12 --------------------------------------------------------------------------------------------------------------


def test_p12_a_refused_release_blocks_the_next_call_until_history_shows_it_released(rig):
    rig.sim.release_mode = "refused"
    text = rig.call()
    assert "device still open" in text and "gozer status" in text
    second = rig.call()
    assert second.startswith("hardware_check refused") and "lease L1 may still be held" in second
    assert len(rig.sim.requests()) == 1
    rig.sim.finish_release("L1")                                              # the lease was released after all
    rig.sim.release_mode = "ok"
    assert rig.call().splitlines()[0].startswith("hardware_check 2 of 6: PASS")


# P13 --------------------------------------------------------------------------------------------------------------


def test_p13_a_release_that_overruns_is_left_running_and_its_time_is_settled_from_gozer(rig):
    rig.sim.release_mode = "overrun"
    text = rig.call()
    assert "still running" in text and [c for c in rig.sim.calls if c["argv"][1] == "release"][0]["kill_on_timeout"] is False
    rec = rig.check.summary()["calls"][0]
    assert rec["lease_incomplete"] is True and rec["lease_id"] == "L1"
    assert rig.call().startswith("hardware_check refused") and "lease L1" in rig.call()
    rig.sim.finish_release("L1", seconds=130.0)                               # gozer's reset finished; it wrote the duration
    rig.sim.release_mode = "ok"
    CLOCK["now"] += 1.0
    assert rig.call().splitlines()[0].startswith("hardware_check 2 of 6: PASS")
    first = rig.check.summary()["calls"][0]
    assert first["lease_s"] == 130.0 and first["lease_incomplete"] is False and first["lease_reconciled"] is True


# P14 --------------------------------------------------------------------------------------------------------------


def test_p14_an_interrupt_during_the_release_reclaims_what_gozer_still_shows(rig):
    rig.sim.release_mode = "interrupt_once"
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.sim.released() == ["L1", "L1"]                                  # the interrupted release, then the reclaim
    assert rig.leases.unresolved() is None


def test_p14b_a_reclaim_that_cannot_release_leaves_the_lease_shown_by_gozer_and_blocking(rig):
    rig.sim.release_mode = "interrupt_once"
    original = rig.sim._release

    def refuse_after(argv):
        if rig.sim.interrupted_release:
            return res(15, "", "device still open")
        return original(argv)
    rig.sim._release = refuse_after
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    rig.sim.release_mode = "ok"
    rig.sim._release = original
    assert rig.call().startswith("hardware_check refused") and "lease L1" in rig.call()


# P15 --------------------------------------------------------------------------------------------------------------


def test_p15_history_that_cannot_be_read_blocks_the_next_call_and_says_so(rig):
    rig.call()
    rig.sim.history_fails = True
    assert "cannot confirm" in rig.call()


# P16 --------------------------------------------------------------------------------------------------------------


def test_p16_a_lease_left_by_one_attempt_blocks_the_next_attempts_first_call(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    rig.sim.release_mode = "refused"
    rig.call()
    second_dir = tmp_path / "probe" / "add" / "attempt-2"
    kp.prepare_attempt(second_dir, op="add", env=fake_kenv, tt_lang=kp_facts())
    (second_dir / "kernel.py").write_text(rig.adir.joinpath("kernel.py").read_text())
    second = rig.make(second_dir, 2)                       # a new attempt: a new check over the same gozer and the same pid
    requests_before = len(rig.sim.requests())
    text = second.call({})
    assert text.startswith("hardware_check refused") and "lease L1" in text
    assert len(rig.sim.requests()) == requests_before


def kp_facts():
    from test_agent_hw import FACTS
    return FACTS


def test_p16b_main_stops_after_an_attempt_that_left_something_unresolved(tmp_path, monkeypatch, fake_kenv, env_ready):
    seen = []

    class FakeLeases:
        def unresolved(self):
            seen.append("asked")
            return "lease L1 may still be held; look with `gozer status`"

    monkeypatch.setattr(kp.hwlease, "GozerLeases", lambda **kw: FakeLeases())

    def fake_run_attempt(*, op, n, **kw):
        seen.append(n)
        return dict(canned(op, n), agent_hw={"enabled": True, "calls": [{"k": 1}], "calls_counted": 1})
    monkeypatch.setattr(kp, "run_attempt", fake_run_attempt)
    code = kp.main(probe_args(tmp_path, attempts=3), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True)
    assert code == kp.EXIT_BLOCKED and seen == [1, "asked"]
    report = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert "lease L1 may still be held" in report["header"]["stopped"] and len(report["attempts"]) == 1


def test_p16c_main_asks_gozer_only_after_an_attempt_that_called_the_tool(tmp_path, monkeypatch, fake_kenv, env_ready):
    class FakeLeases:
        def unresolved(self):
            raise AssertionError("gozer must not be asked when no hardware call was made")

    monkeypatch.setattr(kp.hwlease, "GozerLeases", lambda **kw: FakeLeases())
    monkeypatch.setattr(kp, "run_attempt", lambda *, op, n, **kw: dict(canned(op, n), agent_hw={"enabled": True, "calls": []}))
    assert kp.main(probe_args(tmp_path, attempts=2), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True) == 0


def test_p16d_main_gives_every_attempt_the_same_leases_object(tmp_path, monkeypatch, fake_kenv, env_ready):
    made, passed = [], []

    class FakeLeases:
        def __init__(self, **kw):
            made.append(kw)

        def unresolved(self):
            return None
    monkeypatch.setattr(kp.hwlease, "GozerLeases", FakeLeases)
    monkeypatch.setattr(kp, "run_attempt", lambda *, op, n, **kw: passed.append(kw["leases"]) or canned(op, n))
    kp.main(probe_args(tmp_path, attempts=3), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True)
    assert len(made) == 1 and made[0]["who"] == kp.GOZER_WHO and len(passed) == 3 and len({id(x) for x in passed}) == 1


# the unresolved record -------------------------------------------------------------------------------------------


def test_p17_the_attempt_record_and_the_calls_file_say_what_is_unresolved(rig):
    rig.sim.release_mode = "refused"
    rig.call()
    assert "L1" in rig.check.summary()["unresolved"]
    saved = json.loads(kp.hwcheck_calls_path(rig.adir.parent.parent, "add", 1).read_text())
    assert "L1" in saved["unresolved"]


def test_p18_the_report_says_what_is_unresolved(tmp_path):
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    summary = dict(hw_summary({}), unresolved="lease L1 may still be held; look with `gozer status`")
    md = _report_md(tmp_path, [_row_attempt(agent_hw=summary)])
    assert "UNRESOLVED" in md and "lease L1 may still be held" in md


def test_p19_a_failing_lease_check_blocks_the_next_call_and_says_so():
    from test_agent_hw import World

    def health():
        raise RuntimeError("history unreadable")
    w = World()
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep, health=health)
    text = c.call({})
    assert text.startswith("hardware_check refused") and "the lease check failed" in text and w.runs == []


# the budget ------------------------------------------------------------------------------------------------------------


def test_the_cleanup_reserve_covers_the_harness_stop_and_the_separate_release():
    p = hwtool.HwPolicy()
    assert p.cleanup_s >= kp.GOZER_TERM_GRACE_S + hwlease.RELEASE_TIMEOUT_S
    assert (p.lease_s - p.cleanup_s) + kp.GOZER_TERM_GRACE_S + hwlease.RELEASE_TIMEOUT_S <= p.lease_s
