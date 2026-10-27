# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for Copilot's first review of PR 116 (the agent's `hardware_check` tool). Fakes only: no gozer, no chip."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from fake_model import FakeModel, call, final, turn
from orchard import hwtool, kernelprobe as kp
from test_agent_hw import (CLOCK, FACTS, GOOD_HW_SET, KERNELS, QUEUED, World, attempt_dir, canned, env_ready,  # noqa: F401
                           fake_kenv, good_hw, hang_answer, hw_writer, make_check, probe_args, queued,
                           queued_runner, run_attempt, scripted_hw_agent, sim_pass_then_hw, skills_current,
                           ttlang_skill)


# ---- 1. background processes are swept before the kernel is copied and before each gozer try ------------------------


def traced(monkeypatch):
    events = []
    monkeypatch.setattr(kp, "sweep_strays", lambda run_dir, **kw: events.append("sweep") or [])
    real = kp.copy_kernel

    def copy(adir, clean):
        events.append("copy")
        return real(adir, clean)
    monkeypatch.setattr(kp, "copy_kernel", copy)
    return events


def test_the_attempts_background_processes_are_swept_before_the_copy_and_before_the_gozer_run(tmp_path, fake_kenv, monkeypatch):
    adir = attempt_dir(tmp_path, fake_kenv)
    events = traced(monkeypatch)

    def fake(argv, **kw):
        events.append("run")
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None}))
        return kp.Watched(0, None, False, 1.0)
    make_check(adir, tmp_path, fake).call({})
    assert events == ["sweep", "copy", "sweep", "run", "sweep"], events      # the last one is the sweep after the run


def test_every_try_of_a_call_that_waits_for_a_board_is_swept_first(tmp_path, fake_kenv, monkeypatch):
    adir = attempt_dir(tmp_path, fake_kenv)
    events = traced(monkeypatch)
    inner = queued_runner(["T1", "T2", "T3", "T4", "T5", "T6"])

    def fake(argv, **kw):
        events.append("run")
        return inner(argv, **kw)
    make_check(adir, tmp_path, fake, wait_s=12.0, poll_s=5.0).call({})
    runs = [i for i, e in enumerate(events) if e == "run"]
    assert len(runs) >= 3
    for i in runs:
        assert "sweep" in events[max(i - 2, 0):i], events           # a sweep just before this try (the gate sits between)


def test_a_process_that_rewrites_the_kernel_during_the_sweep_cannot_get_its_edit_run_unchecked(tmp_path, fake_kenv, monkeypatch):
    """The sweep runs before the copy, so what a stray writes before it is stopped is copied and gated like any kernel."""
    adir = attempt_dir(tmp_path, fake_kenv)
    sweeps = []

    def sweep(run_dir, **kw):
        sweeps.append(1)
        if len(sweeps) == 1:
            (Path(run_dir) / "kernel.py").write_text("import os\n")       # the stray's last write
        return []
    monkeypatch.setattr(kp, "sweep_strays", sweep)
    calls = []
    text = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls)).call({})
    assert calls == [] and "FAIL:" in text                                # the copy was gated, and refused


# ---- 2. the cleanup time is reserved ------------------------------------------------------------------------------


def test_the_default_policy_reserves_the_release_time_and_has_room_for_six_calls():
    p = hwtool.HwPolicy()
    assert p.cleanup_s == 120.0 and p.lease_s == 600.0
    assert p.max_calls * 47 + p.cleanup_s <= p.lease_s


def test_the_run_deadline_leaves_the_cleanup_time_inside_the_lease_budget():
    w = World()
    w.check(lease_s=300.0, call_timeout_s=240.0, cleanup_s=120.0).call({})
    assert w.runs == [(1, 180.0)]                       # 300 - 120: a hang that takes the whole deadline still resets in budget


def test_the_worst_case_of_a_call_fits_the_lease_cap():
    """deadline + the release grace never exceeds what is left, call after call."""
    w = World([(good_hw(elapsed_s=47.0), None, "")] * 10)
    c = w.check(lease_s=600.0, cleanup_s=120.0, min_call_s=60.0)
    for _ in range(10):
        c.call({})
    spent = 0.0
    for (k, timeout), _ in zip(w.runs, range(10)):
        assert timeout + 120.0 <= 600.0 - spent + 1e-9
        spent += 47.0


def test_a_call_needs_the_minimum_run_time_after_the_cleanup_reserve():
    w = World([(good_hw(elapsed_s=100.0), None, "")])
    c = w.check(lease_s=300.0, min_call_s=60.0, cleanup_s=120.0)
    c.call({})
    assert len(w.runs) == 1 and w.runs[0][1] == 180.0
    # 200 s are left: 200 - 120 = 80 s, enough for a run
    c.call({})
    assert len(w.runs) == 2 and w.runs[1][1] == 80.0
    w2 = World([(good_hw(elapsed_s=121.0), None, "")])
    c2 = w2.check(lease_s=300.0, min_call_s=60.0, cleanup_s=120.0)
    c2.call({})
    text = c2.call({})                                  # 179 left: 59 s for a run is under the minimum
    assert len(w2.runs) == 1 and "lease seconds" in text and "cleanup" in text


def test_the_policy_reports_the_reserve():
    assert hwtool.HwPolicy().as_dict()["cleanup_s"] == 120.0


def test_the_probe_reserves_at_least_the_gozer_term_grace():
    assert hwtool.HwPolicy().cleanup_s >= kp.GOZER_TERM_GRACE_S


# ---- 3. polling stops at the first ticket that could not be cancelled -------------------------------------------


def test_a_ticket_that_cannot_be_cancelled_stops_the_polling_and_the_answer_names_that_ticket():
    t1 = dict(QUEUED, queued={"ticket": "T1", "cancel_error": "gozer cancel exited 1"})
    t2 = dict(QUEUED, queued={"ticket": "T2", "cancel_error": None})
    w = World([(t1, "queued", ""), (t2, "queued", ""), (t2, "queued", "")])
    c = w.check(wait_s=120.0, poll_s=10.0)
    text = c.call({})
    assert len(w.runs) == 1, w.runs
    assert "T1" in text and "gozer cancel exited 1" in text and "T2" not in text
    assert c.summary()["calls"][0]["cancelled"] is False


def test_a_later_pass_cannot_hide_a_ticket_that_is_still_queued():
    t1 = dict(QUEUED, queued={"ticket": "T1", "cancel_error": "boom"})
    w = World([(t1, "queued", ""), (good_hw(), None, "")])
    text = w.check().call({})
    assert "PASS" not in text and "T1" in text and len(w.runs) == 1


def test_a_missing_ticket_also_stops_the_polling():
    nothing = dict(QUEUED, queued={"ticket": None, "cancel_error": "no ticket in gozer's output"})
    w = World([(nothing, "queued", ""), queued()])
    w.check().call({})
    assert len(w.runs) == 1


# ---- 4. a hang does not claim a cleanup nobody saw ----------------------------------------------------------------


def test_a_hang_answer_does_not_claim_the_lease_was_released_or_the_chip_reset():
    text = World([hang_answer()]).check().call({})
    assert "released the lease and reset the chip" not in text and "gozer released" not in text
    assert "gozer status" in text
    assert "stalled: no output and no CPU use for 60 s" in text
    assert "simulator" in text and "off for the rest of this attempt" in text


def test_a_hang_answer_says_what_to_do_about_a_lease_that_stayed():
    text = World([hang_answer()]).check().call({})
    assert "gozer reconcile" in text or "gozer-gatekeeper" in text
    assert "by hand" in text                              # never reset a chip by hand


# ---- 5. the calls survive an interruption -------------------------------------------------------------------------


def test_each_call_is_written_to_disk_as_it_finishes(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    check = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], []))
    check.call({"note": "one"})
    sidecar = kp.hwcheck_calls_path(tmp_path / "probe", "add", 1)
    data = json.loads(sidecar.read_text())
    assert data["calls_counted"] == 1 and data["calls"][0]["note"] == "one"
    check.call({})
    assert len(json.loads(sidecar.read_text())["calls"]) == 2


def test_a_refused_call_and_an_interrupted_call_are_written_too(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv, "forbidden_import.py")
    check = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], []))
    check.call({})
    assert json.loads(kp.hwcheck_calls_path(tmp_path / "probe", "add", 1).read_text())["calls"][0]["counted"] is False
    adir2 = attempt_dir(tmp_path / "two", fake_kenv)

    def boom(argv, **kw):
        raise KeyboardInterrupt
    check2 = make_check(adir2, tmp_path, boom)
    with pytest.raises(KeyboardInterrupt):
        check2.call({})
    calls = json.loads(kp.hwcheck_calls_path(tmp_path / "two" / "probe", "add", 1).read_text())["calls"]
    assert calls[0]["outcome"] == "interrupted"


def hw_agent_then_interrupt(calls):
    """run_watched: the hardware harness works; the simulator check (run after the agent step) is interrupted."""
    def fake(argv, **kw):
        calls.append(list(argv))
        if argv[0] == "gozer":
            Path(argv[argv.index("--out") + 1]).write_text(
                json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 1.0}))
            return kp.Watched(0, None, False, 45.0)
        raise KeyboardInterrupt
    return fake


def test_an_interrupt_in_the_authoritative_check_keeps_the_calls_in_a_saved_partial_attempt(tmp_path, fake_kenv):
    good = (KERNELS / "add_good.py").read_text()
    with FakeModel(scripted_hw_agent(good)) as server:
        with pytest.raises(KeyboardInterrupt) as info:
            run_attempt(tmp_path, fake_kenv, server, hw_agent_then_interrupt([]), hw_policy=hwtool.HwPolicy(),
                        shared_python="/p")
    partial = info.value.partial_attempt
    assert partial["op"] == "add" and partial["attempt"] == 1 and partial["status"] == "interrupted"
    assert partial["agent_hw"]["calls_counted"] == 1 and partial["agent_hw"]["calls"][0]["outcome"] == "pass"
    assert partial["passed_sim"] is False and partial["hw"] is None
    on_disk = json.loads((tmp_path / "probe" / ".verdict" / "add-1" / "attempt.json").read_text())
    assert on_disk == partial


def test_a_crash_of_the_agent_step_keeps_the_calls_too(tmp_path, fake_kenv):
    good = (KERNELS / "add_good.py").read_text()

    def crash(argv, **kw):
        if argv[0] == "gozer":
            return hw_agent_then_interrupt([])(argv, **kw)
        raise RuntimeError("the simulator check crashed")
    with FakeModel(scripted_hw_agent(good)) as server:
        with pytest.raises(RuntimeError) as info:
            run_attempt(tmp_path, fake_kenv, server, crash, hw_policy=hwtool.HwPolicy(), shared_python="/p")
    assert info.value.partial_attempt["agent_hw"]["calls_counted"] == 1


def test_an_interrupt_with_no_hardware_call_adds_no_partial_attempt(tmp_path, fake_kenv):
    def interrupt(argv, **kw):
        raise KeyboardInterrupt
    good = (KERNELS / "add_good.py").read_text()
    with FakeModel(lambda r: call("write_file", path="kernel.py", content=good) if turn(r) == 0 else final("done")) as server:
        with pytest.raises(KeyboardInterrupt) as info:
            run_attempt(tmp_path, fake_kenv, server, interrupt, hw_policy=hwtool.HwPolicy(), shared_python="/p")
    assert not hasattr(info.value, "partial_attempt")
    assert not (tmp_path / "probe" / ".verdict" / "add-1" / "attempt.json").exists()


def test_the_stopped_run_report_lists_the_partial_attempt_and_its_calls(tmp_path, monkeypatch, fake_kenv, env_ready):
    partial = dict(canned("add", 1), status="interrupted", agent_hw={
        "enabled": True, "policy": hwtool.HwPolicy().as_dict(), "calls_counted": 1, "bad_calls": 0, "lease_s_used": 47.0,
        "calls": [{"k": 1, "counted": True, "outcome": "pass", "passed": True, "total_s": 47.0, "lease_s": 47.0,
                   "waited_s": 0.0, "device_s": 0.5, "note": "", "first_line": "", "queued_polls": 0,
                   "cancelled": None, "timed_out": False}]})

    def interrupted(*, op, n, **kw):
        exc = KeyboardInterrupt()
        exc.partial_attempt = partial
        raise exc
    monkeypatch.setattr(kp, "run_attempt", interrupted)
    out = tmp_path / "out"
    code = kp.main(probe_args(tmp_path), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True)
    assert code == kp.EXIT_ABORTED
    data = json.loads((out / "kernel-probe-report.json").read_text())
    assert [a["attempt"] for a in data["attempts"]] == [1]
    md = (out / "kernel-probe-report.md").read_text()
    assert "Hardware checks called by the coder" in md and "1 call in 1 attempt" in md
    assert "interrupted during attempt 1" in md


def test_a_crash_partial_attempt_is_in_the_report_too(tmp_path, monkeypatch, fake_kenv, env_ready):
    partial = dict(canned("add", 1), status="interrupted")

    def crashed(*, op, n, **kw):
        exc = RuntimeError("boom")
        exc.partial_attempt = partial
        raise exc
    monkeypatch.setattr(kp, "run_attempt", crashed)
    code = kp.main(probe_args(tmp_path), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True)
    assert code == kp.EXIT_ERROR
    data = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert [a["attempt"] for a in data["attempts"]] == [1]
