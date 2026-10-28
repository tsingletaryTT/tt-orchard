# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 116, sixth review round: the budget shown to the agent is computed after the call's chip time is settled, and the attempt record
and report carry the reconciled hardware state (what was found, released, charged and left unresolved) even when no second call
happens. Fakes only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fake_model import FakeModel
from orchard import hwtool, kernelprobe as kp
from test_agent_hw import (CLOCK, FACTS, GOOD_HW_SET, KERNELS, World, _report_md, _row_attempt, good_hw, hw_summary, run_attempt,
                           scripted_hw_agent, skills_current, ttlang_skill, fake_kenv)  # noqa: F401
from test_agent_hw_paths import Rig, SimGozer
from test_agent_hw_queue import incomplete_pass, make


# ---- the budget in the answer is the reconciled one ---------------------------------------------------------------------------


def test_the_budget_text_of_a_call_is_computed_after_its_chip_time_is_settled():
    w = World([incomplete_pass(60.0)])
    c = make(w, {"L1": 500.0}, lease_s=600.0)                  # gozer already has the duration: 500 s, not the 60 s estimate
    text = c.call({})
    assert "100 lease seconds left" in text and "540 lease seconds left" not in text
    assert c.summary()["lease_s_used"] == 500.0


def test_the_budget_text_of_a_hang_is_settled_too():
    def hang(world, on_grant, ctx):
        ctx["lease_id"] = "L1"
        on_grant()
        world.now += 66.0
        return ({"passed": False, "category": "hang", "first_line": "stalled: x", "pcc_min": None, "sets": [], "violations": [],
                 "elapsed_s": 66.0, "waited_s": 0.0, "lease_id": "L1", "lease_incomplete": True}, "hang", "")
    w = World([hang])
    c = make(w, {"L1": 300.0}, lease_s=600.0)
    assert "300 lease seconds left" in c.call({})


def test_the_budget_text_of_a_busy_answer_and_of_a_blocked_one_use_the_settled_use():
    """An earlier call's long release finished: its duration counts before this answer's budget is written."""
    w = World([incomplete_pass(60.0)])
    durations = {}
    c = make(w, durations, lease_s=1000.0)
    c.call({})
    durations["L1"] = 400.0
    w.answers = [(dict(good_hw(), passed=False, category="queued", first_line="q", queued={"ticket": "T1", "cancel_error": None},
                       elapsed_s=0.0), "queued", "")]
    assert "600 lease seconds left" in c.call({})              # 1000 - 400


def test_a_settle_inside_the_budget_text_is_persisted_by_the_end_of_the_call():
    w = World([incomplete_pass(60.0)])
    saved = []
    c = hwtool.HardwareCheck(hwtool.HwPolicy(lease_s=600.0), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep,
                             settle=lambda lease: 500.0, on_record=saved.append)
    c.call({})
    assert saved[-1]["calls"][0]["lease_s"] == 500.0 and saved[-1]["lease_s_used"] == 500.0


# ---- reconcile: what the attempt record carries ---------------------------------------------------------------------------


def test_reconcile_settles_refreshes_and_persists_without_a_new_call():
    w = World([incomplete_pass(60.0)])
    durations, saved, health = {}, [], {"text": "lease L1 may still be held"}
    c = hwtool.HardwareCheck(hwtool.HwPolicy(lease_s=1000.0), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep,
                             settle=durations.get, on_record=saved.append,
                             health=lambda: health["text"] if w.runs else None)
    c.call({})
    assert c.summary()["unresolved"] and c.summary()["calls"][0]["lease_incomplete"] is True
    durations["L1"], health["text"] = 130.0, None                      # gozer finishes the release afterwards
    summary = c.reconcile()
    assert summary["calls"][0]["lease_s"] == 130.0 and summary["calls"][0]["lease_incomplete"] is False
    assert "unresolved" not in summary and summary["lease_s_used"] == 130.0
    assert saved[-1] == summary                                        # persisted too


def test_reconcile_with_nothing_to_settle_changes_nothing_and_writes_nothing_new():
    w = World()
    saved = []
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep, on_record=saved.append)
    c.call({})
    n = len(saved)
    assert c.reconcile()["calls"][0]["lease_s"] == 47.0 and len(saved) == n


def test_reconcile_never_raises_when_the_checks_fail():
    w = World([incomplete_pass(60.0)])

    def boom(*a):
        raise RuntimeError("gozer unreadable")
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep, settle=boom,
                             health=lambda: boom() if w.runs else None)
    c.call({})
    assert c.reconcile()["calls"][0]["lease_incomplete"] is True


def attempt_with_late_release(tmp_path, fake_kenv, *, finish):
    """An attempt whose only hardware call leaves a refused release; gozer finishes it (or not) while the simulator check runs."""
    rig = Rig(tmp_path, fake_kenv)
    rig.sim.release_mode = "refused"
    good = (KERNELS / "add_good.py").read_text()
    sim = {"passed": True, "category": None, "first_line": "", "sets": [GOOD_HW_SET]}

    def watched(argv, **kw):
        if any(str(a).endswith("kernel_harness.py") for a in argv) and "--phase" in argv and argv[argv.index("--phase") + 1] == "hw":
            Path(argv[argv.index("--out") + 1]).write_text(
                json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 3.0}))
            return kp.Watched(0, None, False, 45.0)
        if finish:
            rig.sim.release_mode = "ok"
            rig.sim.finish_release("L1", seconds=130.0)                # gozer finishes the long release during the check
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(sim))
        return kp.Watched(0, None, False, 0.1)
    with FakeModel(scripted_hw_agent(good)) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, watched, hw_policy=hwtool.HwPolicy(), shared_python="/shared/python",
                              leases=rig.leases)
    return rig, attempt


def test_the_final_attempt_record_carries_the_state_after_gozer_finished_the_release(tmp_path, fake_kenv):
    rig, attempt = attempt_with_late_release(tmp_path, fake_kenv, finish=True)
    rec = attempt["agent_hw"]
    assert rec["calls"][0]["lease_s"] == 130.0 and rec["calls"][0]["lease_incomplete"] is False
    assert "unresolved" not in rec and rec["lease_s_used"] == 130.0
    on_disk = json.loads((tmp_path / "probe" / ".verdict" / "add-1" / "attempt.json").read_text())
    assert on_disk["agent_hw"] == rec
    calls_file = json.loads(kp.hwcheck_calls_path(tmp_path / "probe", "add", 1).read_text())
    assert calls_file["calls"][0]["lease_s"] == 130.0 and "unresolved" not in calls_file


def test_the_final_attempt_record_still_says_unresolved_when_gozer_still_shows_the_lease(tmp_path, fake_kenv):
    rig, attempt = attempt_with_late_release(tmp_path, fake_kenv, finish=False)
    rec = attempt["agent_hw"]
    assert "lease L1" in rec["unresolved"] and rec["calls"][0]["lease_incomplete"] is True


def test_the_report_of_a_reconciled_attempt_has_the_settled_chip_time_and_no_unresolved_line(tmp_path, fake_kenv):
    rig, attempt = attempt_with_late_release(tmp_path, fake_kenv, finish=True)
    md = _report_md(tmp_path / "rep", [attempt])
    assert "UNRESOLVED" not in md and "130 s of chip time" in md


def test_a_partial_attempt_is_reconciled_before_it_is_saved(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    rig.sim.release_mode = "refused"
    good = (KERNELS / "add_good.py").read_text()

    def watched(argv, **kw):
        if any(str(a).endswith("kernel_harness.py") for a in argv) and "--phase" in argv and argv[argv.index("--phase") + 1] == "hw":
            Path(argv[argv.index("--out") + 1]).write_text(
                json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 3.0}))
            return kp.Watched(0, None, False, 45.0)
        rig.sim.release_mode = "ok"
        rig.sim.finish_release("L1", seconds=130.0)
        raise KeyboardInterrupt                                           # the simulator check is interrupted
    with FakeModel(scripted_hw_agent(good)) as server:
        with pytest.raises(KeyboardInterrupt) as info:
            run_attempt(tmp_path, fake_kenv, server, watched, hw_policy=hwtool.HwPolicy(), shared_python="/shared/python",
                        leases=rig.leases)
    partial = info.value.partial_attempt["agent_hw"]
    assert partial["calls"][0]["lease_s"] == 130.0 and "unresolved" not in partial


def test_a_settle_made_while_the_answer_was_formatted_is_written_once_and_not_again():
    w = World([incomplete_pass(60.0)])
    saved = []
    c = hwtool.HardwareCheck(hwtool.HwPolicy(lease_s=600.0), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep,
                             settle=lambda lease: 500.0, on_record=saved.append)
    c.call({})
    n = len(saved)
    c.reconcile()
    c.reconcile()
    assert len(saved) == n                                              # nothing changed since: nothing is written again


def test_reconcile_with_no_call_made_asks_gozer_nothing():
    """An attempt whose agent never called the tool has nothing to reconcile: no gozer command may run for it."""
    w = World()
    asked = []
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep,
                             health=lambda: asked.append("health"), settle=lambda lease: asked.append("settle"))
    assert c.reconcile()["calls"] == [] and asked == []
