# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A busy board is waited for on ONE queue ticket (`gozer acquire`'s ticket and `gozer wait`), never by asking again in a
loop (PR 116, Copilot's review: the old loop gave up its place in the queue on every try). The probe runs the harness
under the lease that `gozer wait` grants and releases it itself. Interrupted waits and runs record the time spent.

Fakes only: no gozer, no chip."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from orchard import hwtool, kernelprobe as kp
from test_agent_hw import (FACTS, GOOD_HW_SET, KERNELS, QUEUED, World, attempt_dir, fake_kenv, good_hw,  # noqa: F401
                           hw_writer, make_check, queued, queued_runner, skills_current, ttlang_skill, CLOCK)


# ---- the policy object ----------------------------------------------------------------------------------------------


class QWorld(World):
    """A World whose busy answer keeps its ticket and whose wait_for is scripted: `waits` holds callables or answers."""

    def __init__(self, answers=None, waits=None):
        super().__init__(answers)
        self.waits = list(waits or [])
        self.wait_calls = []

    def wait_for(self, k, q, wait_s, timeout_s, on_grant):
        self.wait_calls.append({"k": k, "ticket": q.get("ticket"), "wait_s": wait_s, "timeout_s": timeout_s})
        item = self.waits.pop(0)
        return item(self, on_grant) if callable(item) else item

    def check(self, **policy):
        return hwtool.HardwareCheck(hwtool.HwPolicy(**policy), prepare=self.prepare, run=self.run, wait_for=self.wait_for,
                                    clock=self.clock, sleep=self.sleep, stamp=lambda: "2026-10-27T00:00:00Z")


def kept_ticket(ticket="T1"):
    return (dict(QUEUED, queued={"ticket": ticket, "cancel_error": None, "kept": True}), "queued", "")


def timed_out_wait(cancel_error=None, ticket="T1", took=120.0):
    def wait(world, on_grant):
        world.now += took
        return (dict(QUEUED, elapsed_s=0.0, queued={"ticket": ticket, "cancel_error": cancel_error}), "queued", "")
    return wait


def granted_pass(waited=21.0, elapsed=47.0):
    def wait(world, on_grant):
        world.now += waited
        on_grant()
        world.now += elapsed
        return (dict(good_hw(), elapsed_s=elapsed, waited_s=waited), None, "")
    return wait


def test_a_busy_board_waits_on_the_one_ticket_and_never_runs_gozer_again():
    w = QWorld([kept_ticket()], [timed_out_wait()])
    c = w.check()
    text = c.call({})
    assert len(w.runs) == 1 and len(w.wait_calls) == 1 and w.sleeps == []          # one try, one wait, no loop
    assert w.wait_calls[0]["ticket"] == "T1"
    assert "BUSY" in text and "simulator" in text and "were cancelled" in text
    assert "waited on 1 queue ticket" in text


def test_the_wait_gets_what_is_left_of_wait_s_after_the_first_try():
    w = QWorld([kept_ticket()], [timed_out_wait()])
    w.check(wait_s=120.0).call({})
    assert w.wait_calls[0]["wait_s"] == 119.5                                      # the busy try itself took 0.5 s


def test_the_wait_also_gets_the_runs_timeout_with_the_cleanup_reserved():
    w = QWorld([kept_ticket()], [timed_out_wait()])
    w.check(lease_s=300.0, call_timeout_s=240.0, cleanup_s=120.0).call({})
    assert w.wait_calls[0]["timeout_s"] == 180.0


def test_a_zero_wait_still_hands_the_ticket_to_wait_for_so_that_it_is_cancelled():
    w = QWorld([kept_ticket()], [timed_out_wait(took=0.0)])
    w.check(wait_s=0.0).call({})
    assert w.wait_calls[0]["wait_s"] == 0.0


def test_a_board_granted_during_the_wait_runs_the_kernel_and_records_the_wait_and_the_lease_time():
    w = QWorld([kept_ticket()], [granted_pass(waited=21.0, elapsed=47.0)])
    c = w.check()
    text = c.call({})
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS")
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 21.0 and rec["lease_s"] == 47.0 and rec["queued_polls"] == 1 and rec["cancelled"] is None
    assert rec["outcome"] == "pass" and c.summary()["lease_s_used"] == 47.0
    assert "waited 21.0 s for a lease" in text


def test_a_wait_that_ran_out_is_a_busy_call_that_uses_no_lease_time_and_counts():
    w = QWorld([kept_ticket()], [timed_out_wait()])
    c = w.check()
    c.call({})
    rec = c.summary()["calls"][0]
    assert rec["outcome"] == "busy" and rec["cancelled"] is True and rec["lease_s"] == 0.0 and rec["counted"] is True
    assert rec["queued_polls"] == 1 and rec["waited_s"] >= 119.0 and c.summary()["lease_s_used"] == 0.0
    assert c.summary()["calls_counted"] == 1


def test_a_ticket_that_could_not_be_cancelled_is_named_in_the_answer():
    w = QWorld([kept_ticket("T9")], [timed_out_wait(cancel_error="gozer cancel exited 1", ticket="T9")])
    c = w.check()
    text = c.call({})
    assert "could not be cancelled" in text and "T9" in text and "gozer cancel exited 1" in text
    assert c.summary()["calls"][0]["cancelled"] is False


def test_a_busy_answer_without_a_ticket_is_not_waited_on():
    nothing = dict(QUEUED, queued={"ticket": None, "cancel_error": "no ticket in gozer's output"})
    w = QWorld([(nothing, "queued", "")], [])
    text = w.check().call({})
    assert w.wait_calls == [] and "BUSY" in text and "no ticket" in text


def test_without_a_wait_function_a_busy_answer_is_final():
    w = World([queued()])
    c = w.check()
    assert "BUSY" in c.call({}) and len(w.runs) == 1


def test_a_hang_after_the_grant_is_charged_and_turns_the_tool_off():
    def hang(world, on_grant):
        world.now += 10.0
        on_grant()
        world.now += 66.0
        return ({"passed": False, "category": "hang", "first_line": "stalled: x", "pcc_min": None, "sets": [],
                 "violations": [], "elapsed_s": 66.0, "waited_s": 10.0}, "hang", "")
    w = QWorld([kept_ticket()], [hang])
    c = w.check()
    text = c.call({})
    assert "HANG" in text and c.summary()["lease_s_used"] == 66.0 and c.summary()["calls"][0]["waited_s"] == 10.0
    assert "hung" in c.call({})


# ---- interrupted waits and runs record the time ---------------------------------------------------------------------


def test_an_interrupt_while_waiting_records_the_wait_and_no_board_time():
    def interrupted(world, on_grant):
        world.now += 30.0
        raise KeyboardInterrupt
    w = QWorld([kept_ticket()], [interrupted])
    c = w.check()
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["outcome"] == "interrupted" and rec["waited_s"] == 30.5 and rec["total_s"] == 30.5
    assert rec["lease_s"] == 0.0 and not rec.get("lease_upper_bound") and rec["queued_polls"] == 1
    assert c.summary()["lease_s_used"] == 0.0


def test_an_interrupt_after_the_grant_records_the_wait_and_the_board_time_as_an_upper_bound():
    def interrupted(world, on_grant):
        world.now += 20.0
        on_grant()
        world.now += 5.0
        raise KeyboardInterrupt
    w = QWorld([kept_ticket()], [interrupted])
    c = w.check()
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 20.5 and rec["lease_s"] == 5.0 and rec["lease_upper_bound"] is True
    assert c.summary()["lease_s_used"] == 5.0


def test_an_interrupt_in_the_first_try_records_no_wait():
    w = QWorld()

    def boom(k, t):
        w.now += 7.0
        raise KeyboardInterrupt
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=boom, wait_for=w.wait_for, clock=w.clock,
                             sleep=w.sleep)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 0.0 and rec["lease_s"] == 7.0 and rec["lease_upper_bound"] is True


def test_the_report_row_of_a_waited_call_says_one_ticket_and_whether_it_was_cancelled():
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    busy = {"outcome": "busy", "passed": False, "lease_s": 0.0, "waited_s": 120.0, "queued_polls": 1, "cancelled": True}
    granted = {"outcome": "pass", "waited_s": 21.0, "queued_polls": 1, "cancelled": None}
    md = _report_md(__import__("pathlib").Path(__import__("tempfile").mkdtemp()),
                    [_row_attempt(agent_hw=hw_summary(busy, granted))])
    assert "1 ticket, cancelled" in md and "1 ticket, granted" in md


# ---- the probe's side: one gozer run, then a ticket, a leased harness run and a release -------------------------------


class FakeGozer:
    """Scripted gozer: `run` is the fake run_watched, `wait` and `release` and `cancel` record their calls."""

    def __init__(self, wait_answer=None, release_answer=(0, "")):
        self.events = []
        self.cancelled = []
        self.released = []
        self.waited = []
        self.wait_answer = wait_answer
        self.release_answer = release_answer
        self.argvs = []
        self.envs = []

    def run_watched(self, argv, **kw):
        self.argvs.append(list(argv))
        self.envs.append(kw.get("env"))
        if argv[0] == "gozer":
            self.events.append("gozer-run")
            with open(kw["log_path"], "a") as f:
                f.write("queued: ticket T1\n")
            return kp.Watched(10, None, False, 0.4)
        self.events.append("harness")
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(
                json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 1.5}))
        return kp.Watched(0, None, False, 3.0)

    def gozer_wait(self, ticket, wait_s):
        self.events.append("wait")
        self.waited.append((ticket, wait_s))
        if isinstance(self.wait_answer, BaseException):
            raise self.wait_answer
        return self.wait_answer if self.wait_answer is not None else (None, "")

    def gozer_release(self, lease):
        self.events.append("release")
        self.released.append(lease)
        return self.release_answer

    def cancel(self, ticket):
        self.events.append("cancel")
        self.cancelled.append(ticket)
        return None


GRANTED = (0, json.dumps({"granted": True, "lease_id": "L1", "chips": ["0000:01:00.0", "0000:02:00.0"],
                          "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}}))


def build(adir, g, **policy):
    CLOCK["now"] = 0.0
    return kp.make_hardware_check(adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(**policy),
                                  run_watched=g.run_watched, cancel=g.cancel, gozer_wait=g.gozer_wait,
                                  gozer_release=g.gozer_release, clock=lambda: CLOCK["now"],
                                  sleep=lambda s: CLOCK.__setitem__("now", CLOCK["now"] + s))


def test_a_busy_board_asks_gozer_once_waits_on_the_ticket_and_runs_the_harness_under_the_granted_lease(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=GRANTED)
    text = build(adir, g).call({})
    assert g.events == ["gozer-run", "wait", "harness", "release"], g.events
    assert g.waited == [("T1", pytest.approx(120.0, abs=1.0))] and g.cancelled == [] and g.released == ["L1"]
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS"), text
    harness = g.argvs[1]
    assert harness[0] == "/shared/python" and "gozer" not in harness[0]               # no second gozer run
    assert sum(1 for a in g.argvs if a[0] == "gozer") == 1
    assert g.envs[1]["TT_VISIBLE_DEVICES"] == "0000:01:00.0,0000:02:00.0"
    assert Path(harness[harness.index("--kernel") + 1]).is_absolute()


def test_a_wait_that_runs_out_cancels_the_ticket_once_and_runs_nothing(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=(None, ""))
    c = build(adir, g)
    text = c.call({})
    assert g.events == ["gozer-run", "wait", "cancel"] and g.cancelled == ["T1"] and g.released == []
    assert "BUSY" in text and c.summary()["calls"][0]["cancelled"] is True


def test_a_zero_wait_cancels_at_once_without_waiting(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=GRANTED)
    build(adir, g, wait_s=0.0).call({})
    assert g.events == ["gozer-run", "cancel"] and g.waited == []


def test_a_wait_that_answers_with_something_that_is_not_a_grant_cancels_the_ticket(tmp_path, fake_kenv):
    for answer in [(0, "not json"), (0, json.dumps({"granted": False})), (0, json.dumps({"granted": True})),
                   (1, GRANTED[1]), (0, json.dumps({"granted": True, "lease_id": "bad id"})),
                   (0, json.dumps({"granted": False, "lease_id": "L1"}))]:
        adir = attempt_dir(tmp_path / str(abs(hash(answer))), fake_kenv)
        g = FakeGozer(wait_answer=answer)
        text = build(adir, g).call({})
        assert g.cancelled == ["T1"] and g.released == [] and "BUSY" in text, (answer, g.events)
        assert "harness" not in g.events


def test_a_ticket_that_could_not_be_cancelled_is_reported(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=(None, ""))
    g.cancel = lambda t: "gozer cancel exited 1"
    text = build(adir, g).call({})
    assert "could not be cancelled" in text and "T1" in text and "gozer status" in text


def test_the_lease_is_released_when_the_harness_run_is_interrupted(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=GRANTED)
    real = g.run_watched

    def run(argv, **kw):
        if argv[0] != "gozer":
            raise KeyboardInterrupt
        return real(argv, **kw)
    g.run_watched = run
    c = build(adir, g)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    assert g.released == ["L1"] and g.cancelled == []                 # a granted ticket is spent: nothing to cancel
    assert c.summary()["calls"][0]["outcome"] == "interrupted"


def test_the_lease_is_released_after_a_hang(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=GRANTED)
    real = g.run_watched

    def run(argv, **kw):
        if argv[0] != "gozer":
            return kp.Watched(-15, "stalled: no output and no CPU use for 60 s", True, 66.0)
        return real(argv, **kw)
    g.run_watched = run
    c = build(adir, g)
    text = c.call({})
    assert g.released == ["L1"] and "HANG" in text and c.summary()["calls"][0]["outcome"] == "hang"


def test_a_failed_release_is_told_to_the_agent_and_recorded(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=GRANTED, release_answer=(1, "chips [0, 1] still open by [123]"))
    c = build(adir, g)
    text = c.call({})
    assert "PASS" in text and "could not release the lease" in text and "still open by [123]" in text
    assert "gozer status" in text
    assert c.summary()["calls"][0]["release_error"].startswith("chips [0, 1] still open")


def test_an_interrupt_while_waiting_cancels_the_ticket(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=KeyboardInterrupt())
    c = build(adir, g)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    assert g.cancelled == ["T1"] and g.released == []


def test_the_background_processes_are_swept_before_the_gozer_run_and_again_before_the_leased_run(tmp_path, fake_kenv,
                                                                                                monkeypatch):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=GRANTED)
    monkeypatch.setattr(kp, "sweep_strays", lambda run_dir, **kw: g.events.append("sweep") or [])
    build(adir, g).call({})
    assert g.events.index("wait") < len(g.events)
    before_harness = g.events[:g.events.index("harness")]
    assert before_harness[-1] == "sweep" and before_harness.count("sweep") >= 3, g.events      # prepare, run, leased run


def test_the_copy_is_checked_again_after_the_wait(tmp_path, fake_kenv):
    """The agent can write anywhere the operator can, and a wait can last two minutes."""
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=GRANTED)
    real = g.gozer_wait

    def wait(ticket, wait_s):
        for p in (tmp_path / "probe" / ".hwcheck").glob("add-1-1/authoritative/kernel.py"):
            p.write_text(p.read_text() + "\n# edited while waiting\n")
        return real(ticket, wait_s)
    g.gozer_wait = wait
    text = build(adir, g).call({})
    assert "harness" not in g.events and "harness_error" in text and g.released == ["L1"]


def test_a_relative_python_path_is_made_absolute_for_the_child_whose_directory_changes(tmp_path, fake_kenv, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adir = Path("kernel-probe") / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
    g = FakeGozer(wait_answer=GRANTED)
    CLOCK["now"] = 0.0
    kp.make_hardware_check(adir, op="add", n=1, shared_python="venv/bin/python", policy=hwtool.HwPolicy(),
                           run_watched=g.run_watched, cancel=g.cancel, gozer_wait=g.gozer_wait,
                           gozer_release=g.gozer_release, clock=lambda: CLOCK["now"], sleep=lambda s: None).call({})
    assert Path(g.argvs[0][g.argvs[0].index("--") + 1]).is_absolute()
    assert Path(g.argvs[1][0]).is_absolute() and g.argvs[1][0] == str(tmp_path / "venv" / "bin" / "python")


def test_a_bare_python_name_is_left_for_the_path_lookup():
    assert kp._abs_executable("python3") == "python3"
    assert kp._abs_executable("/usr/bin/python3") == "/usr/bin/python3"


# ---- the real gozer wait and release wrappers ---------------------------------------------------------------------


class FakeProc:
    def __init__(self, hang=False, out="{}", code=0):
        self.hang, self.out, self.returncode, self.killed = hang, out, code, False
        self.timeouts = []

    def communicate(self, timeout=None):
        self.timeouts.append(timeout)
        if self.hang and not self.killed:
            raise subprocess.TimeoutExpired("gozer wait", timeout)
        return self.out, ""

    def kill(self):
        self.killed = True


def test_gozer_wait_asks_for_the_ticket_json_and_a_timeout():
    seen = {}

    def popen(argv, **kw):
        seen["argv"], seen["kw"] = list(argv), kw
        return FakeProc(out='{"granted": true}')
    code, out = kp.gozer_wait("T1", 42.0, popen=popen)
    assert seen["argv"] == ["gozer", "wait", "--json", "--timeout", "42", "T1"] and code == 0 and "granted" in out


def test_gozer_wait_is_stopped_by_the_probe_at_the_deadline_because_gozers_own_timeout_was_seen_to_run_over():
    proc = FakeProc(hang=True)
    code, out = kp.gozer_wait("T1", 25.0, popen=lambda argv, **kw: proc)
    assert code is None and out == "" and proc.killed is True and proc.timeouts[0] == 25.0


def test_gozer_wait_refuses_a_ticket_that_could_pass_as_an_option():
    code, out = kp.gozer_wait("--force", 5.0, popen=lambda *a, **k: pytest.fail("nothing may run"))
    assert code is None


def test_gozer_wait_kills_the_waiter_on_an_interrupt():
    class Boom(FakeProc):
        def communicate(self, timeout=None):
            if not self.killed:
                raise KeyboardInterrupt
            return "", ""
    proc = Boom()
    with pytest.raises(KeyboardInterrupt):
        kp.gozer_wait("T1", 5.0, popen=lambda argv, **kw: proc)
    assert proc.killed is True


def test_gozer_release_runs_release_with_the_lease_id_and_reports_the_result():
    seen = {}

    def run(argv, **kw):
        seen["argv"], seen["kw"] = list(argv), kw
        return subprocess.CompletedProcess(argv, 1, stdout="out", stderr="still open")
    code, text = kp.gozer_release("2a03c8", run=run)
    assert seen["argv"] == ["gozer", "release", "2a03c8"] and code == 1 and "still open" in text
    assert seen["kw"]["timeout"] >= kp.GOZER_TERM_GRACE_S


def test_gozer_release_refuses_a_lease_id_that_could_pass_as_an_option():
    code, text = kp.gozer_release("--force", run=lambda *a, **k: pytest.fail("nothing may run"))
    assert code is None and "lease" in text


def test_gozer_release_reports_a_timeout_and_an_os_error():
    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    def gone(argv, **kw):
        raise OSError("no gozer")
    assert kp.gozer_release("L1", run=slow)[0] is None
    assert kp.gozer_release("L1", run=gone)[0] is None


def test_the_wait_and_the_lease_time_are_measured_from_the_clock(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    g = FakeGozer(wait_answer=GRANTED)
    real_wait, real_run = g.gozer_wait, g.run_watched

    def wait(ticket, wait_s):
        CLOCK["now"] += 21.0
        return real_wait(ticket, wait_s)

    def run(argv, **kw):
        if argv[0] != "gozer":
            CLOCK["now"] += 47.0
        return real_run(argv, **kw)
    g.gozer_wait, g.run_watched = wait, run
    c = build(adir, g)
    text = c.call({})
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 21.0 and rec["lease_s"] == 47.0 and rec["total_s"] == 68.0, rec
    assert "waited 21.0 s for a lease" in text and c.summary()["lease_s_used"] == 47.0


def test_the_report_row_shows_a_failed_release():
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    rec = {"outcome": "pass", "waited_s": 21.0, "queued_polls": 1, "cancelled": None,
           "release_error": "chips [0, 1] still open by [123]"}
    md = _report_md(Path(__import__("tempfile").mkdtemp()), [_row_attempt(agent_hw=hw_summary(rec))])
    assert "release failed: chips [0, 1] still open by [123]" in md
