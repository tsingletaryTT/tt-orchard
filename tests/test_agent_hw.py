# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `hardware_check` tool of a kernel-probe attempt (orchard/hwtool.py and its wiring in orchard/kernelprobe.py).

The agent cannot run gozer, so the probe offers a tool that runs the agent's current kernel.py on one leased chip
and gives back a short text. A short-lived wrapper holds the lease; the probe owns the copy of the kernel, the static gate, the stall watcher and
the judgement. The tool takes at most a short note: no command and no path.

Every test uses a fake `run_watched`, a fake gozer (tests/hwfakes.py FakeGozer) under the real wrapper (orchard/hwtool.py) and a fake clock. No test runs gozer or opens a chip."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import time
from pathlib import Path

import pytest

from fake_model import FakeModel, call, final, turn
from orchard import hwtool, kernelprobe as kp
from hwfakes import FakeGozer, wrapped
from test_kernelprobe import (FACTS, GOOD_HW_SET, KERNELS, canned, env_ready, fake_kenv, hw_writer,  # noqa: F401
                              probe_args, skills_current, ttlang_skill, _row_attempt, _report_md)

SKILL_LINES = None


# ---- a fake world for the policy object ---------------------------------------------------------------------------


class World:
    """Fake clock and sleep, and scripted answers for prepare and run. `answers` is a list of (hw, event, text)."""

    def __init__(self, answers=None, *, blocked=None, step_s=0.0):
        self.now = 0.0
        self.answers = list(answers or [])
        self.blocked = blocked
        self.step_s = step_s
        self.runs = []                       # (k, timeout_s) of every run
        self.run_args = []
        self.prepares = []
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def prepare(self, k):
        self.prepares.append(k)
        return self.blocked, "ab" * 32

    def run(self, k, timeout_s, wait_s=0.0):
        self.runs.append((k, timeout_s))
        self.run_args.append({"wait_s": wait_s})
        item = self.answers.pop(0) if self.answers else (good_hw(), None, "")
        if callable(item):                   # a script: item(world) -> (hw, event, text)
            return item(self)
        hw, event, text = item
        self.now += hw.get("elapsed_s", 0.0) if event != "queued" else 0.5
        return hw, event, text

    def check(self, **policy):
        return hwtool.HardwareCheck(hwtool.HwPolicy(**policy), prepare=self.prepare, run=self.run,
                                    clock=self.clock, sleep=self.sleep, stamp=lambda: "2026-10-27T00:00:00Z")


def good_hw(**kw):
    hw = {"passed": True, "category": None, "first_line": "", "pcc_min": 0.99995, "slope_range": [1.0, 1.0],
          "rel_l2_max": 0.003, "sets": [{"name": "seed0", "pcc": 0.99995}, {"name": "seed1", "pcc": 0.9999}],
          "violations": [], "elapsed_s": 47.0, "device_s": 3.2, "watch": {}}
    hw.update(kw)
    return hw


def bad_hw(category="hw_numeric_mismatch", line="set seed0: pcc 0.2 is below 0.99", **kw):
    return good_hw(**{"passed": False, "category": category, "first_line": line, "pcc_min": 0.2, **kw})


QUEUED = {"passed": False, "category": "queued", "first_line": "gozer queued the request", "pcc_min": None,
          "sets": [], "violations": [], "queued": {"ticket": "T1", "cancel_error": None}, "elapsed_s": 0.5}


def queued(**kw):
    return (dict(QUEUED, **kw), "queued", "")


# ---- the policy defaults ------------------------------------------------------------------------------------------


def test_the_defaults_are_six_calls_a_lease_cap_a_call_timeout_and_a_120_s_wait():
    p = hwtool.HwPolicy()
    assert (p.max_calls, p.wait_s) == (6, 120.0)
    assert p.lease_s == 600.0 and p.call_timeout_s == 240.0 and p.enabled is True
    assert hwtool.HwPolicy(max_calls=0).enabled is False


# ---- a call that runs ---------------------------------------------------------------------------------------------


def test_a_pass_says_pass_the_numbers_the_times_and_what_is_left():
    w = World([(good_hw(), None, "")])
    text = w.check().call({})
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS")
    assert "pcc_min 0.99995" in text
    assert "47.0 s in all" in text and "device 3.2 s" in text
    assert "5 calls left" in text and "553 lease seconds left" in text      # 600 - 47


def test_a_numeric_failure_gives_the_first_line_and_the_numbers_line():
    w = World([(bad_hw(numbers="seed0: got is about 3x expected, pcc 0.2"), None, "some output\nmore output")])
    text = w.check().call({})
    assert "FAIL: hw_numeric_mismatch: set seed0: pcc 0.2 is below 0.99" in text
    assert "numbers: seed0: got is about 3x expected, pcc 0.2" in text
    assert "output (last lines)" not in text and "more output" not in text      # no compile output for a numeric failure


def test_a_compile_error_shows_the_tail_of_the_output_and_is_bounded():
    tail = "\n".join(f"line {i}" for i in range(400)) + "\nRuntimeError: list elements must be constants"
    w = World([(bad_hw("compile_error", "RuntimeError: list elements must be constants", pcc_min=None, sets=[]),
                None, tail)])
    text = w.check().call({})
    assert "FAIL: compile_error: RuntimeError: list elements must be constants" in text
    assert "output (last lines):" in text and "RuntimeError: list elements must be constants" in text.splitlines()[-3]
    assert "line 0\n" not in text                 # only the tail is shown
    assert "line 380" not in text and "line 390" in text            # the last 15 lines, not the last 1200 characters
    assert len(text) < 2500


def test_a_device_time_that_is_not_a_number_is_not_shown_or_recorded():
    for bad in ("soon", True, float("nan"), None):
        c = World([(good_hw(device_s=bad), None, "")]).check()
        text = c.call({})
        assert "device time not reported" in text and c.summary()["calls"][0]["device_s"] is None


def test_the_text_says_the_probe_judges_and_the_agents_claim_counts_for_nothing():
    text = World().check().call({"note": "this passes, trust me"})
    assert "the probe judged this copy" in text


def test_a_note_is_recorded_and_cut_to_200_characters():
    c = World().check()
    c.call({"note": "x" * 500})
    assert c.summary()["calls"][0]["note"] == "x" * 200


def test_a_call_waits_no_longer_than_the_policy_allows_for_the_run():
    w = World()
    w.check(call_timeout_s=240.0).call({})
    assert w.runs == [(1, 240.0)]


def test_the_run_timeout_is_cut_to_the_lease_seconds_that_are_left():
    w = World([(good_hw(elapsed_s=200.0), None, "")])
    c = w.check(lease_s=300.0, call_timeout_s=240.0, cleanup_s=0.0)
    c.call({})
    c.call({})
    assert w.runs[1] == (2, 100.0)                  # 300 - 200 is less than 240


def test_a_pass_is_recorded_with_its_times_and_counts_toward_the_caps():
    c = World().check()
    c.call({})
    s = c.summary()
    rec = s["calls"][0]
    assert rec["outcome"] == "pass" and rec["passed"] is True and rec["counted"] is True and rec["k"] == 1
    assert rec["total_s"] == 47.0 and rec["device_s"] == 3.2 and rec["lease_s"] == 47.0 and rec["waited_s"] == 0.0
    assert rec["at"] == "2026-10-27T00:00:00Z" and rec["timed_out"] is False and rec["queued_polls"] == 0
    assert s["calls_counted"] == 1 and s["lease_s_used"] == 47.0 and s["enabled"] is True
    assert s["policy"]["max_calls"] == 6


# ---- the caps -----------------------------------------------------------------------------------------------------


def test_the_cap_on_calls_refuses_the_next_one_and_runs_nothing():
    w = World()
    c = w.check(max_calls=2)
    c.call({})
    c.call({})
    text = c.call({})
    assert "no hardware checks left" in text and "2 of 2" in text and "simulator" in text
    assert len(w.runs) == 2
    last = c.summary()["calls"][-1]
    assert last["outcome"] == "refused" and last["counted"] is False


def test_the_cap_on_lease_seconds_refuses_the_next_call_and_runs_nothing():
    w = World([(good_hw(elapsed_s=250.0), None, "")])
    c = w.check(lease_s=300.0, min_call_s=60.0, cleanup_s=0.0)
    c.call({})
    text = c.call({})
    assert "lease seconds" in text and "50" in text and "simulator" in text
    assert len(w.runs) == 1


def test_a_call_needs_the_minimum_call_time_to_be_left_of_the_lease_cap():
    w = World([(good_hw(elapsed_s=240.0), None, "")])
    c = w.check(lease_s=300.0, min_call_s=60.0, cleanup_s=0.0)
    c.call({})
    assert len(w.runs) == 1 and "lease seconds" in c.call({})              # 60 s left is enough: it is not refused
    w2 = World([(good_hw(elapsed_s=241.0), None, "")])
    c2 = w2.check(lease_s=300.0, min_call_s=60.0, cleanup_s=0.0)
    c2.call({})
    assert "lease seconds" in c2.call({}) and len(w2.runs) == 1            # 59 s left is not


def test_a_boundary_of_exactly_the_minimum_runs_the_call():
    w = World([(good_hw(elapsed_s=240.0), None, "")])
    c = w.check(lease_s=300.0, min_call_s=60.0, cleanup_s=0.0)
    c.call({})
    text = c.call({})
    assert len(w.runs) == 2 and text.startswith("hardware_check 2 of 6")


def test_the_text_of_every_answer_states_the_remaining_budget():
    w = World([(good_hw(elapsed_s=47.0), None, ""), queued(), queued(), queued()])
    c = w.check(wait_s=1.0)
    first = c.call({})
    assert "5 calls left" in first
    busy = c.call({})
    assert "4 calls left" in busy and "lease seconds left" in busy


def test_a_policy_with_zero_calls_is_not_enabled_and_never_runs():
    w = World()
    c = w.check(max_calls=0)
    assert "no hardware checks left" in c.call({}) and w.runs == [] and c.summary()["enabled"] is False


# ---- the board is busy --------------------------------------------------------------------------------------------


# ---- a hang -------------------------------------------------------------------------------------------------------


def hang_answer(reason="stalled: no output and no CPU use for 60 s"):
    return ({"passed": False, "category": "hang", "first_line": reason, "pcc_min": None, "sets": [], "violations": [],
             "elapsed_s": 66.0}, "hang", "PROGRESS run seed0\n")


def test_a_hang_is_reported_with_the_watchers_reason_and_charged_to_the_lease_budget():
    w = World([hang_answer()])
    c = w.check()
    text = c.call({})
    assert "stopped" in text and "stalled: no output and no CPU use for 60 s" in text
    assert "gozer status" in text and "released the lease" in text
    rec = c.summary()["calls"][0]
    assert rec["outcome"] == "hang" and rec["lease_s"] == 66.0 and rec["timed_out"] is True
    assert c.summary()["lease_s_used"] == 66.0


def test_hardware_checks_are_off_for_the_rest_of_the_attempt_after_a_hang():
    w = World([hang_answer()])
    c = w.check()
    c.call({})
    text = c.call({})
    assert "hung" in text and "simulator" in text and len(w.runs) == 1
    assert c.summary()["calls"][-1]["outcome"] == "refused"


def test_a_backstop_stop_is_also_a_timeout():
    w = World([hang_answer("backstop: still running after 240 s")])
    c = w.check()
    c.call({})
    assert c.summary()["calls"][0]["timed_out"] is True


# ---- what the agent may send --------------------------------------------------------------------------------------


@pytest.mark.parametrize("arguments", [{"command": "gozer release x"}, {"path": "/etc/passwd"}, {"chips": 4},
                                       {"note": "ok", "extra": 1}, {"note": 5}, {"note": ["a"]}, {"note": None},
                                       {"note": {"a": 1}}, {"note": True}])
def test_anything_but_a_short_note_is_an_error_and_runs_nothing(arguments):
    w = World()
    c = w.check()
    text = c.call(arguments)
    assert text.startswith("error:") and "note" in text
    assert w.runs == [] and w.prepares == [] and c.summary()["calls_counted"] == 0
    assert c.summary()["bad_calls"] == 1 and c.summary()["calls"] == []


def test_an_empty_call_is_fine_and_a_note_may_be_empty():
    w = World()
    c = w.check()
    assert c.call({}).startswith("hardware_check 1")
    assert c.call({"note": ""}).startswith("hardware_check 2")


# ---- no chip is touched when the kernel cannot be run -------------------------------------------------------------


def test_a_kernel_the_gate_refuses_is_answered_without_a_lease_and_does_not_count():
    blocked = {"passed": False, "category": "forbidden_code", "first_line": "1 violation: import at line 1",
               "pcc_min": None, "sets": [],
               "violations": [{"rule": "import", "line": 1, "detail": "import os", "remedy": "import only ttl"}]}
    w = World(blocked=blocked)
    c = w.check()
    text = c.call({})
    assert "FAIL: forbidden_code" in text and "import only ttl" in text and "no chip was used" in text
    assert w.runs == [] and c.summary()["calls_counted"] == 0
    rec = c.summary()["calls"][0]
    assert rec["counted"] is False and rec["outcome"] == "forbidden_code" and rec["lease_s"] == 0.0
    assert "5 calls left" not in text and "6 calls left" in text


def test_a_missing_kernel_is_answered_without_a_lease():
    w = World(blocked={"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist",
                       "pcc_min": None, "sets": [], "violations": []})
    c = w.check()
    assert "kernel.py does not exist" in c.call({}) and w.runs == []


# ---- one call at a time -------------------------------------------------------------------------------------------


def test_a_second_call_while_one_runs_is_answered_at_once_and_does_not_queue():
    started, release = threading.Event(), threading.Event()
    w = World()
    real = w.run

    def slow(k, t, *rest):
        started.set()
        assert release.wait(20)
        return real(k, t)
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=slow, clock=w.clock, sleep=w.sleep)
    first = []
    th = threading.Thread(target=lambda: first.append(c.call({})))
    th.start()
    assert started.wait(20)
    t0 = time.monotonic()
    second = c.call({})
    answered_in = time.monotonic() - t0
    release.set()
    th.join(20)
    assert "already running" in second and answered_in < 1.0          # answered at once, not after a wait
    assert first and first[0].startswith("hardware_check 1")
    assert c.summary()["calls_counted"] == 1


def test_an_exception_in_the_run_is_recorded_and_raised_and_the_lock_is_released():
    w = World()

    def boom(k, t, *rest):
        raise KeyboardInterrupt
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=boom, clock=w.clock, sleep=w.sleep)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    assert c.summary()["calls"][0]["outcome"] == "interrupted"
    w.answers = []
    c2 = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep)
    assert c2.call({}).startswith("hardware_check 1")
    # the same object can be called again after the exception (the lock is free)


# ---- the wiring in the probe --------------------------------------------------------------------------------------


def attempt_dir(tmp_path, env, kernel="add_good.py"):
    adir = tmp_path / "probe" / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=env, tt_lang=FACTS)
    if kernel:
        (adir / "kernel.py").write_text((KERNELS / kernel).read_text())
    return adir


CLOCK = {"now": 0.0}


WHO = "claude:test"


def fake_gozer(**kw):
    """A FakeGozer (tests/hwfakes.py): by default every acquire is granted at once. No test may reach the real gozer."""
    return FakeGozer(**kw)


def make_check(adir, tmp_path, run_watched, *, gozer=None, **policy):
    """A hardware check with a fake clock in CLOCK: sleeping moves it, and so does a fake run that advances it. `run_watched` plays the
    harness; the real wrapper (orchard/hwtool.py) runs in process around it against `gozer` (hwfakes.wrapped)."""
    CLOCK["now"] = 0.0

    def sleep(s):
        CLOCK["now"] += s
    gozer = gozer or fake_gozer()
    check = kp.make_hardware_check(adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(**policy),
                                   run_watched=wrapped(run_watched, gozer, lambda: CLOCK["now"]), who=WHO, status=gozer.sh,
                                   clock=lambda: CLOCK["now"], sleep=sleep)
    check.gozer = gozer
    return check


def test_the_tool_runs_the_shipped_hardware_phase_on_a_copy_of_the_current_kernel_under_one_chip_lease(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    calls = []
    c = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls))
    text = c.call({})
    argv = calls[0]
    assert argv[0] == "/shared/python"                       # the harness runs directly, under the lease's environment
    acquire = c.gozer.acquires()[0]
    assert acquire[acquire.index("--chips") + 1] == "1" and acquire[acquire.index("--exact") + 1] == "1"
    assert acquire[acquire.index("--who") + 1] == "claude:test" and "--owner-pid" in acquire
    assert Path(argv[argv.index("--spec") + 1]) == kp.TEMPLATES / "ops" / "add.json"
    assert Path(argv[argv.index("--phase") + 1:][0]).name == "hw"
    kernel = Path(argv[argv.index("--kernel") + 1])
    assert kernel != adir / "kernel.py" and kernel.read_bytes() == (adir / "kernel.py").read_bytes()
    assert kernel.parent.parent.parent == tmp_path / "probe" / ".hwcheck"
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS")


def test_the_copy_the_tool_runs_is_not_the_file_the_agent_can_change_afterwards(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    calls = []
    c = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls))
    c.call({})
    first = Path(calls[0][calls[0].index("--kernel") + 1])
    (adir / "kernel.py").write_text("# changed after the call\n")
    assert first.read_text() == (KERNELS / "add_good.py").read_text()


def test_each_call_gets_its_own_directory_and_a_fresh_copy(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    calls = []
    c = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls))
    c.call({})
    (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text() + "# second\n")
    c.call({})
    k1, k2 = (Path(a[a.index("--kernel") + 1]) for a in calls)
    assert k1 != k2 and k2.read_text().endswith("# second\n") and not k1.read_text().endswith("# second\n")


def test_the_tool_uses_the_shipped_harness_and_never_a_file_from_the_attempt(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    (adir / "kernel_harness.py").write_text("raise SystemExit('the agent edited the harness')\n")
    calls = []
    make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls)).call({})
    harness = next(Path(a) for a in calls[0] if a.endswith("kernel_harness.py"))
    assert harness == kp.TEMPLATES / "kernel_harness.py"


def test_the_tool_runs_in_the_hardware_environment_of_the_probe_with_the_stall_watcher_and_term_grace(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    seen = {}

    def fake(argv, **kw):
        seen.update(kw)
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None}))
        return kp.Watched(0, None, False, 1.0)
    make_check(adir, tmp_path, fake, call_timeout_s=200.0).call({})
    assert seen["backstop_s"] == 120.0 + 200.0 and seen["term_grace_s"] == kp.GOZER_TERM_GRACE_S
    assert seen["watcher"].quiet_s == kp.HW_QUIET_S and seen["watcher"].repeat_lines == kp.HW_REPEAT_LINES
    assert seen["env"]["ORCHARD_RUN_DIR"] == str(adir.resolve())
    assert "TT_METAL_VISIBLE_DEVICES" not in seen["env"] and "HF_TOKEN" not in seen["env"]


def test_a_kernel_that_breaks_the_static_gate_is_never_run(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv, "forbidden_import.py")
    calls = []
    text = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls)).call({})
    assert calls == [] and "FAIL: forbidden_code" in text and "no chip was used" in text


def test_a_missing_kernel_is_never_run(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv, kernel=None)
    calls = []
    text = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls)).call({})
    assert calls == [] and "kernel.py does not exist" in text


def test_a_file_planted_in_the_calls_directory_is_removed_before_the_copy(tmp_path, fake_kenv):
    """The harness runs with the copy's directory as its working directory, so a planted ttnn.py there would be imported."""
    adir = attempt_dir(tmp_path, fake_kenv)
    planted = kp.hwcheck_dir(tmp_path / "probe", "add", 1, 1) / kp.CLEAN_DIR / "ttnn.py"
    planted.parent.mkdir(parents=True)
    planted.write_text("raise SystemExit('planted')\n")
    calls = []
    make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls)).call({})
    assert calls and not planted.exists()
    assert sorted(p.name for p in planted.parent.iterdir()) == ["kernel.py"]


def test_a_kernel_that_is_a_link_is_never_run(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv, kernel=None)
    target = tmp_path / "elsewhere.py"
    target.write_text((KERNELS / "add_good.py").read_text())
    (adir / "kernel.py").symlink_to(target)
    calls = []
    text = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls)).call({})
    assert calls == [] and "forbidden_code" in text
    assert "file at line 0" in text and "plain file of at most 256 KiB" in text      # the rule and its remedy are shown


def test_a_clean_copy_that_changes_before_the_run_is_not_run(tmp_path, fake_kenv, monkeypatch):
    adir = attempt_dir(tmp_path, fake_kenv)
    calls = []
    check = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls))
    real = kp._hw_gate
    seen = []

    def tamper(kernel, recorded):
        seen.append(kernel)
        if len(seen) == 2:                      # the second look, just before the gozer run
            Path(kernel).write_text(Path(kernel).read_text() + "\n# edited between the copy and the run\n")
        return real(kernel, recorded)
    monkeypatch.setattr(kp, "_hw_gate", tamper)
    text = check.call({})
    assert len(seen) == 2 and calls == [] and "harness_error" in text and "changed" in text


def test_a_failed_numeric_run_is_judged_with_the_hardware_bars_not_the_agents_words(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    bad = dict(GOOD_HW_SET, pcc=0.2, slope=3.0)
    c = make_check(adir, tmp_path, hw_writer([bad], []))
    text = c.call({"note": "this is correct"})
    assert "FAIL: hw_numeric_mismatch" in text and "numbers:" in text
    assert c.summary()["calls"][0]["passed"] is False


def test_the_device_seconds_come_from_the_harnesss_result(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)

    def fake(argv, **kw):
        Path(argv[argv.index("--out") + 1]).write_text(
            json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 2.75}))
        CLOCK["now"] += 44.0
        return kp.Watched(0, None, False, 44.0)
    text = make_check(adir, tmp_path, fake).call({})
    assert "device 2.8 s" in text or "device 2.7 s" in text
    assert "44.0 s in all" in text


def test_a_result_with_no_device_seconds_says_so(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    text = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], [])).call({})
    assert "device time not reported" in text


@pytest.mark.parametrize("raw", ['"soon"', "true", "NaN", "-1", "1e9", "null"])
def test_a_device_seconds_value_that_is_not_a_plausible_number_is_ignored(tmp_path, fake_kenv, raw):
    adir = attempt_dir(tmp_path, fake_kenv)

    def fake(argv, **kw):
        Path(argv[argv.index("--out") + 1]).write_text(
            '{"phase": "hw", "sets": [' + json.dumps(GOOD_HW_SET) + '], "error": null, "device_s": ' + raw + "}")
        return kp.Watched(0, None, False, 1.0)
    assert "device time not reported" in make_check(adir, tmp_path, fake).call({})


def queued_runner(tickets):
    """A fake run_watched for gozer that queues: it logs `queued: ticket X` and exits 10."""
    it = iter(tickets)

    def fake(argv, **kw):
        with open(kw["log_path"], "a") as f:
            f.write(f"queued: ticket {next(it)}\n")
        return kp.Watched(10, None, False, 0.4)
    return fake


def test_a_hang_in_the_real_path_stops_the_agents_hardware_and_names_the_reason(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)

    def hung(argv, **kw):
        return kp.Watched(-15, "stalled: no output and no CPU use for 60 s", True, 66.0)
    c = make_check(adir, tmp_path, hung)
    text = c.call({})
    assert "stalled: no output and no CPU use for 60 s" in text and "stopped" in text
    assert c.summary()["calls"][0]["outcome"] == "hang"
    assert "hung" in c.call({})


def test_the_hardware_phase_after_the_attempt_still_raises_on_a_hang_and_a_queue(tmp_path, fake_kenv):
    """The shared core must keep the probe's own phase as it was: a hang or a queue ends the probe."""
    from test_kernelprobe import _passed_attempt

    adir, attempt = _passed_attempt(tmp_path, fake_kenv)

    def hung(argv, **kw):
        return kp.Watched(-15, "stalled: x", True, 66.0)
    with pytest.raises(kp.ProbeBlocked):
        kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hung)
    assert attempt["hw"]["category"] == "hang"
    adir2, attempt2 = _passed_attempt(tmp_path / "two", fake_kenv)
    cancelled = []
    with pytest.raises(kp.ProbeBlocked):
        kp.hardware_phase(adir2, attempt2, env=fake_kenv, shared_python="/p", run_watched=queued_runner(["T7"]),
                          cancel=lambda t: cancelled.append(t))
    assert cancelled == ["T7"] and attempt2["hw"]["category"] == "queued"


def test_the_hardware_phase_records_the_device_seconds_too(tmp_path, fake_kenv):
    from test_kernelprobe import _passed_attempt
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)

    def fake(argv, **kw):
        Path(argv[argv.index("--out") + 1]).write_text(
            json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 2.5}))
        return kp.Watched(0, None, False, 40.0)
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=fake)
    assert hw["device_s"] == 2.5


# ---- the tool in the agent's tool list ----------------------------------------------------------------------------


def tools_with(adir, check):
    return kp.attempt_tools(adir, hardware=check)


def test_the_tool_is_offered_when_a_check_exists_and_not_otherwise(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    names = lambda t: [s["function"]["name"] for s in t.schemas]
    assert "hardware_check" not in names(kp.attempt_tools(adir))
    check = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], []))
    assert "hardware_check" in names(tools_with(adir, check))
    assert names(tools_with(adir, check))[:3] == ["shell", "read_file", "write_file"]


def test_the_schema_takes_only_a_note_and_states_the_cost_and_the_use(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    check = make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], []))
    schema = next(s for s in tools_with(adir, check).schemas if s["function"]["name"] == "hardware_check")
    params = schema["function"]["parameters"]
    assert list(params["properties"]) == ["note"] and params.get("required", []) == []
    assert params.get("additionalProperties") is False
    desc = schema["function"]["description"]
    assert "47 s" in desc and "simulator" in desc and "6" in desc


def test_calling_the_tool_through_the_tools_object(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    calls = []
    t = tools_with(adir, make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls)))
    out = t.call("hardware_check", json.dumps({"note": "first try"}))
    assert out.startswith("hardware_check 1 of 6: PASS") and len(calls) == 1


@pytest.mark.parametrize("arguments", ["not json", "[1, 2]", '"text"', "3", "null"])
def test_garbage_arguments_to_the_tool_are_an_error_and_run_nothing(tmp_path, fake_kenv, arguments):
    adir = attempt_dir(tmp_path, fake_kenv)
    calls = []
    t = tools_with(adir, make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], calls)))
    assert t.call("hardware_check", arguments).startswith("error:") and calls == []


def test_without_a_check_the_tool_name_is_unknown(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    out = kp.attempt_tools(adir).call("hardware_check", "{}")
    assert out.startswith("error:") and "no tool named" in out


def test_a_shell_command_still_cannot_run_gozer(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    t = tools_with(adir, make_check(adir, tmp_path, hw_writer([GOOD_HW_SET], [])))
    out = t.call("shell", json.dumps({"command": "gozer run --chips 1 -- ls"}))
    assert out.startswith("refused")


# ---- run_attempt --------------------------------------------------------------------------------------------------


def scripted_hw_agent(kernel_text):
    def script(request):
        n = turn(request)
        if n == 0:
            return call("write_file", path="kernel.py", content=kernel_text)
        if n == 1:
            return call("hardware_check", note="try it on the chip")
        return final("done")
    return script


def run_attempt(tmp_path, env, server, run_watched, gozer=None, **kw):
    gozer = gozer or fake_gozer()
    args = dict(op="add", n=1, probe_dir=tmp_path / "probe", env=env, endpoint=server.endpoint, model="fake",
                coder_model="fake-coder", probe_id="t", max_turns=10, wall_s=120, tt_lang=FACTS,
                run_watched=wrapped(run_watched, gozer, lambda: CLOCK["now"]), hw_status=gozer.sh)
    args.update(kw)
    return kp.run_attempt(**args)


def sim_pass_then_hw(calls):
    """run_watched for both the simulator check (writes check json) and the hardware harness."""
    sim = {"passed": True, "category": None, "first_line": "", "sets": [GOOD_HW_SET]}

    def fake(argv, **kw):
        calls.append(list(argv))
        if any(str(a).endswith("kernel_harness.py") and "--phase" in argv and argv[argv.index("--phase") + 1] == "hw" for a in argv):
            Path(argv[argv.index("--out") + 1]).write_text(
                json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 3.0}))
            return kp.Watched(0, None, False, 45.0)
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(sim))
        return kp.Watched(0, None, False, 0.1)
    return fake


def test_run_attempt_gives_the_agent_the_tool_runs_it_and_records_the_call(tmp_path, fake_kenv):
    good = (KERNELS / "add_good.py").read_text()
    calls = []
    tools_seen = []

    def script(request):
        tools_seen.append([t["function"]["name"] for t in request.get("tools", [])])
        return scripted_hw_agent(good)(request)
    with FakeModel(script) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, sim_pass_then_hw(calls),
                              hw_policy=hwtool.HwPolicy(), shared_python="/shared/python")
    assert "hardware_check" in tools_seen[0]
    harness = [c for c in calls if c[0] == "/shared/python"]
    assert len(harness) == 1
    rec = attempt["agent_hw"]
    assert rec["enabled"] is True and rec["calls_counted"] == 1 and rec["calls"][0]["outcome"] == "pass"
    assert rec["calls"][0]["note"] == "try it on the chip" and rec["lease_s_used"] >= 0.0
    on_disk = json.loads((tmp_path / "probe" / ".verdict" / "add-1" / "attempt.json").read_text())
    assert on_disk["agent_hw"] == rec and on_disk == attempt
    assert attempt["hw"] is None                          # the tool's runs are not the attempt's hardware verdict


def test_run_attempt_without_a_policy_offers_no_tool_and_records_it_as_off(tmp_path, fake_kenv):
    tools_seen = []

    def script(request):
        tools_seen.append([t["function"]["name"] for t in request.get("tools", [])])
        return final("done")
    with FakeModel(script) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, sim_pass_then_hw([]))
    assert "hardware_check" not in tools_seen[0] and attempt["agent_hw"] == {"enabled": False}


def test_run_attempt_with_zero_calls_offers_no_tool(tmp_path, fake_kenv):
    tools_seen = []

    def script(request):
        tools_seen.append([t["function"]["name"] for t in request.get("tools", [])])
        return final("done")
    with FakeModel(script) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, sim_pass_then_hw([]),
                              hw_policy=hwtool.HwPolicy(max_calls=0), shared_python="/p")
    assert "hardware_check" not in tools_seen[0] and attempt["agent_hw"] == {"enabled": False}


def test_a_call_that_ends_the_attempt_by_interrupt_still_leaves_a_record(tmp_path, fake_kenv):
    good = (KERNELS / "add_good.py").read_text()

    def fake(argv, **kw):
        if any(str(a).endswith("kernel_harness.py") and "--phase" in argv and argv[argv.index("--phase") + 1] == "hw" for a in argv):
            raise KeyboardInterrupt
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"passed": True, "sets": []}))
        return kp.Watched(0, None, False, 0.1)
    with FakeModel(scripted_hw_agent(good)) as server:
        with pytest.raises(KeyboardInterrupt):
            run_attempt(tmp_path, fake_kenv, server, fake, hw_policy=hwtool.HwPolicy(), shared_python="/p")


# ---- the command line ---------------------------------------------------------------------------------------------


def test_the_policy_comes_from_the_flags_with_the_defaults():
    ns = argparse.Namespace()
    p = kp.hw_policy_from_args(ns)
    assert (p.max_calls, p.lease_s, p.call_timeout_s, p.wait_s) == (6, 600.0, 240.0, 120.0)
    p = kp.hw_policy_from_args(argparse.Namespace(hw_checks=2, hw_lease_s=100, hw_call_timeout_s=90, hw_wait_s=30))
    assert (p.max_calls, p.lease_s, p.call_timeout_s, p.wait_s) == (2, 100.0, 90.0, 30.0)


def test_no_agent_hw_is_the_same_as_zero_checks():
    assert kp.hw_policy_from_args(argparse.Namespace(no_agent_hw=True)).enabled is False
    assert kp.hw_policy_from_args(argparse.Namespace(hw_checks=0)).enabled is False
    assert kp.hw_policy_from_args(argparse.Namespace(hw_checks=3, no_agent_hw=True)).enabled is False


@pytest.mark.parametrize("bad", [dict(hw_checks=-1), dict(hw_checks=51), dict(hw_lease_s=0), dict(hw_lease_s=-5),
                                 dict(hw_call_timeout_s=0), dict(hw_wait_s=-1), dict(hw_wait_s=100000),
                                 dict(hw_checks="many")])
def test_a_flag_value_out_of_range_is_refused(bad):
    with pytest.raises(kp._Refused):
        kp.hw_policy_from_args(argparse.Namespace(**bad))


def test_the_cli_has_the_flags():
    from orchard import cli
    p = cli._parser()
    ns = p.parse_args(["kernel-probe", "--op", "add", "--hw-checks", "3", "--hw-lease-s", "200", "--hw-call-timeout-s",
                       "100", "--hw-wait-s", "60", "--no-agent-hw"])
    assert (ns.hw_checks, ns.hw_lease_s, ns.hw_call_timeout_s, ns.hw_wait_s, ns.no_agent_hw) == (3, 200.0, 100.0, 60.0, True)
    d = p.parse_args(["kernel-probe", "--op", "add"])
    assert d.hw_checks is None and d.no_agent_hw is False


def test_main_passes_the_policy_and_the_shared_python_to_each_attempt(tmp_path, monkeypatch, fake_kenv):
    from test_kernelprobe import fake_attempts, env_ready  # noqa: F401
    seen = []

    def fake_run_attempt(*, op, n, **kw):
        seen.append(kw)
        return canned(op, n)
    monkeypatch.setattr(kp, "run_attempt", fake_run_attempt)
    monkeypatch.setattr(kp.ttlang_env, "ensure", lambda env, **k: [kp.ttlang_env.Result("verify", kp.ttlang_env.OK, "x")])
    args = probe_args(tmp_path, hw_checks=4, hw_wait_s=30.0, shared_python="/shared/python")
    kp.main(args, env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True)
    assert seen and seen[0]["hw_policy"].max_calls == 4 and seen[0]["hw_policy"].wait_s == 30.0
    assert seen[0]["shared_python"] == "/shared/python"


def test_main_with_no_agent_hw_passes_a_disabled_policy(tmp_path, monkeypatch, fake_kenv):
    seen = []

    def fake_run_attempt(*, op, n, **kw):
        seen.append(kw)
        return canned(op, n)
    monkeypatch.setattr(kp, "run_attempt", fake_run_attempt)
    monkeypatch.setattr(kp.ttlang_env, "ensure", lambda env, **k: [kp.ttlang_env.Result("verify", kp.ttlang_env.OK, "x")])
    kp.main(probe_args(tmp_path, no_agent_hw=True), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True)
    assert seen[0]["hw_policy"].enabled is False


def test_main_refuses_a_bad_flag_before_any_attempt(tmp_path, monkeypatch, fake_kenv):
    monkeypatch.setattr(kp, "run_attempt", lambda **k: pytest.fail("no attempt may start"))
    monkeypatch.setattr(kp.ttlang_env, "ensure", lambda env, **k: [kp.ttlang_env.Result("verify", kp.ttlang_env.OK, "x")])
    assert kp.main(probe_args(tmp_path, hw_checks=-2), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True) == kp.EXIT_REFUSED


# ---- the report ---------------------------------------------------------------------------------------------------


def hw_summary(*calls, enabled=True, used=None):
    recs = []
    for i, c in enumerate(calls, start=1):
        base = {"k": i, "counted": True, "at": "2026-10-27T00:00:00Z", "outcome": "pass", "passed": True, "total_s": 47.0,
                "device_s": 3.0, "lease_s": 47.0, "waited_s": 0.0, "queued_polls": 0, "cancelled": None,
                "timed_out": False, "note": "", "first_line": ""}
        base.update(c)
        recs.append(base)
    lease = sum(r["lease_s"] for r in recs)
    return {"enabled": enabled, "policy": {"max_calls": 6, "lease_s": 300.0, "call_timeout_s": 240.0, "wait_s": 120.0},
            "calls": recs, "calls_counted": sum(1 for r in recs if r["counted"]), "bad_calls": 0,
            "lease_s_used": lease if used is None else used}


def test_the_attempt_table_has_an_agent_hardware_column_with_the_counts_and_the_chip_seconds(tmp_path):
    md = _report_md(tmp_path, [_row_attempt(agent_hw=hw_summary({}, {"outcome": "hw_numeric_mismatch", "passed": False}))])
    header = next(l for l in md.splitlines() if l.startswith("| attempt"))
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    assert header.rstrip().endswith("| agent hw |")
    assert row.count(" | ") == header.count(" | ")
    assert row.rstrip().endswith("2 calls (1 passed), 94 s |"), row


def test_a_probe_where_the_tool_was_off_says_so_and_the_column_is_a_dash(tmp_path):
    md = _report_md(tmp_path, [_row_attempt(agent_hw={"enabled": False})])
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    assert row.rstrip().endswith("| - |")
    assert "Hardware checks called by the coder: off" in md


def test_a_record_from_before_the_tool_existed_gets_a_dash_and_no_section(tmp_path):
    md = _report_md(tmp_path, [_row_attempt()])
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    assert row.rstrip().endswith("| - |") and "Hardware checks called by the coder" not in md


def test_the_report_lists_every_call_with_its_outcome_time_and_wait_and_totals_the_chip_time(tmp_path):
    calls = [{}, {"outcome": "busy", "passed": False, "lease_s": 0.0, "total_s": 121.0, "waited_s": 120.0,
                 "queued_polls": 12, "cancelled": True},
             {"outcome": "hang", "passed": False, "lease_s": 66.0, "total_s": 66.0, "timed_out": True}]
    md = _report_md(tmp_path, [_row_attempt(agent_hw=hw_summary(*calls))])
    assert "## Hardware checks called by the coder" in md
    section = md.split("## Hardware checks called by the coder")[1]
    lines = section.splitlines()
    assert any(l.startswith("| silu_mul | 1 | 1 | pass |") for l in lines)
    busy = next(l for l in lines if "| busy |" in l)
    assert "121" in busy and "120" in busy and "cancelled" in busy
    hang = next(l for l in lines if "| hang |" in l)
    assert "timed out" in hang
    assert "3 calls in 1 attempt" in section and "113 s of chip time" in section
    assert "1 busy" in section and "1 hang" in section and "1 passed" in section


def test_a_note_with_a_pipe_or_a_line_break_cannot_break_the_table(tmp_path):
    md = _report_md(tmp_path, [_row_attempt(agent_hw=hw_summary({"note": "a | b\nc"}))])
    section = md.split("## Hardware checks called by the coder")[1]
    row = next(l for l in section.splitlines() if l.startswith("| silu_mul |"))
    assert "\n" not in row and row.count(" | ") == next(l for l in section.splitlines() if l.startswith("| op ")).count(" | ")


def test_the_column_and_the_totals_count_only_the_calls_that_reached_the_lease(tmp_path):
    refused = {"counted": False, "outcome": "refused", "passed": False, "lease_s": 0.0, "k": None}
    md = _report_md(tmp_path, [_row_attempt(agent_hw=hw_summary({}, refused, refused))])
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    assert row.rstrip().endswith("1 call (1 passed), 47 s |"), row
    assert "1 call in 1 attempt" in md


def test_the_totals_say_how_long_the_agents_waited_for_boards(tmp_path):
    busy = {"outcome": "busy", "passed": False, "lease_s": 0.0, "total_s": 121.0, "waited_s": 120.0, "queued_polls": 12,
            "cancelled": True}
    md = _report_md(tmp_path, [_row_attempt(agent_hw=hw_summary(busy))])
    assert "120 s waited for a board" in md


def test_a_call_row_shows_the_agents_note_and_a_ticket_that_was_not_cancelled(tmp_path):
    busy = {"outcome": "busy", "passed": False, "lease_s": 0.0, "waited_s": 120.0, "queued_polls": 3, "cancelled": False,
            "note": "waiting for the board"}
    md = _report_md(tmp_path, [_row_attempt(agent_hw=hw_summary(busy))])
    row = next(l for l in md.splitlines() if "| busy |" in l)
    assert "note: waiting for the board" in row and "NOT cancelled" in row


def test_the_report_json_keeps_the_calls(tmp_path):
    data = kp.write_report(tmp_path, [_row_attempt(agent_hw=hw_summary({}))],
                           {"probe_id": "t", "bars": {}, "hw": None, "ops": ["silu_mul"]})
    assert data["attempts"][0]["agent_hw"]["calls"][0]["outcome"] == "pass"
    assert json.loads((tmp_path / "kernel-probe-report.json").read_text())["attempts"][0]["agent_hw"]["enabled"] is True


# ---- the skill ----------------------------------------------------------------------------------------------------


def skill_flat():
    return " ".join(kp.SKILL.read_text().split())


def test_the_skill_stays_at_100_lines_with_one_fenced_block():
    text = kp.SKILL.read_text()
    assert len(text.splitlines()) <= 100
    assert text.count("```") == 2


def test_the_skill_names_the_tool_its_cost_its_uses_and_the_simulator_as_the_cheap_path():
    flat = skill_flat()
    assert "`hardware_check`" in flat
    assert "about 47 seconds" in flat and "about 3 seconds" not in flat      # measured: a whole call, chip reset included
    assert "ttl.raw_element_read" in flat
    assert "hardware precision" in flat and "performance" in flat
    assert "the simulator is much cheaper" in flat
    assert "calls left" in flat


def test_the_skill_no_longer_tells_the_agent_that_no_device_may_be_touched():
    assert "or touch any device (the check runs on a simulator)" not in skill_flat()
    assert "run a command on a chip" not in skill_flat() or "hardware_check" in skill_flat()


# ---- the harness reports its own device time ----------------------------------------------------------------------


def test_the_harness_writes_device_seconds_into_its_result():
    src = (kp.TEMPLATES / "kernel_harness.py").read_text()
    assert 'result["device_s"]' in src and "time.monotonic()" in src


def test_the_tools_own_description_states_the_measured_cost_of_a_call():
    desc = hwtool.schema(hwtool.HwPolicy())["description"] if "description" in hwtool.schema(hwtool.HwPolicy()) else \
        hwtool.schema(hwtool.HwPolicy())["function"]["description"]
    assert "about 47 s" in desc and "about 3 s" not in desc
