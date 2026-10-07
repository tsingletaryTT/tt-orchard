# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Replaying the ledger: the next stage, pauses, escalations, caps, the coder's lease, stage dirs."""
import calendar
import time

import pytest

from orchard.ledger import Ledger
from orchard.stages import (STAGES, attempt_started_ts, budget_cap, coder_state, open_stage_dir,
                            run_progress)

T0 = calendar.timegm(time.strptime("2026-10-02T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"))


def entries(*rows):
    """Ledger-shaped dicts. Each row: (event, stage, seconds after T0, data)."""
    return [{"seq": i, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 + t)),
             "event": ev, "stage": st, "data": d} for i, (ev, st, t, d) in enumerate(rows, 1)]


START = ("run_start", None, 0, {"model": "Altworld/Hemmingway-1"})


def test_a_stage_with_no_end_is_open_and_runs_next():
    p = run_progress(entries(START, ("stage_start", 0, 1, {}), ("stage_end", 0, 2, {"result": "pass"}),
                             ("stage_start", 1, 3, {})))
    assert p.started and p.done == (0,) and p.open_stage == 1 and p.next_stage == 1
    assert p.clock_start == T0


def test_a_skipped_stage_counts_as_done():
    rows = [START] + [r for n in range(7) for r in (("stage_start", n, 1, {}),
                                                    ("stage_end", n, 2, {"result": "pass"}))]
    rows += [("stage_start", 7, 3, {"skip": True}), ("stage_end", 7, 3, {"result": "skipped"})]
    assert run_progress(entries(*rows)).next_stage == 8


def test_an_escalated_stage_runs_again_and_is_counted():
    passed = [r for n in (0, 1) for r in (("stage_start", n, 1, {}), ("stage_end", n, 1, {"result": "pass"}))]
    p = run_progress(entries(START, *passed, ("stage_start", 2, 1, {}),
                             ("escalate", 2, 2, {"by": "stage machine"}),
                             ("stage_end", 2, 2, {"result": "escalate"})))
    assert p.next_stage == 2 and p.open_stage is None
    assert p.escalated == {2} and p.escalations == 1


def test_a_pause_lasts_until_a_resume_and_the_resume_resets_the_caps():
    rows = [START, ("escalate", 2, 1, {}), ("restore", 2, 2, {"step": "ready", "seconds": 1800.0}),
            ("restore", 3, 2, {"step": "ready", "seconds": 150.0}),     # a warm restart
            ("decision", None, 2, {"decision": "coder died; restarting it once"}),
            ("decision", 2, 3, {"decision": "pause", "reason": "watchdog: loop"})]
    p = run_progress(entries(*rows))
    assert p.paused == "watchdog: loop" and p.escalations == 1 and p.cold_boots == 1
    assert p.coder_deaths == 1
    p = run_progress(entries(*rows, ("decision", None, 4, {"decision": "resume"})))
    assert p.paused is None and (p.escalations, p.cold_boots, p.coder_deaths) == (0, 0, 0)
    assert p.clock_start == T0 + 4


def test_abort_and_ready_are_read_from_decisions():
    assert run_progress(entries(START, ("decision", None, 1, {"decision": "abort"}))).aborted
    assert run_progress(entries(START, ("decision", None, 1, {"decision": "ready for operator review"}))).finished


def test_a_restart_keeps_the_attempts_clock():
    rows = entries(START, ("stage_start", 3, 100, {}), ("stage_start", 3, 500, {"resumed": True}))
    assert attempt_started_ts(rows, 3) == T0 + 100
    rows = entries(START, ("stage_start", 3, 100, {}), ("stage_end", 3, 200, {"result": "escalate"}),
                   ("stage_start", 3, 300, {}))
    assert attempt_started_ts(rows, 3) == T0 + 300
    assert attempt_started_ts(rows, 4) is None


@pytest.mark.parametrize("rows, words", [
    ([("escalate", 2, 1, {})] * 3, "3 escalations"),
    ([("decision", None, 1, {"decision": "coder started", "ready_s": 1800.0})] * 3, "3 cold coder boots"),
])
def test_budget_caps_pause_the_run(rows, words):
    assert words in budget_cap(run_progress(entries(START, *rows)), now=T0 + 10)


def test_the_wall_clock_cap_counts_from_run_start_or_the_last_resume():
    p = run_progress(entries(START))
    assert budget_cap(p, now=T0 + 3600) is None
    assert "the run has lasted" in budget_cap(p, now=T0 + 72 * 3600)
    # Without the restart at a resume, a resumed run would pause again at once, forever.
    p = run_progress(entries(START, ("decision", None, 72 * 3600, {"decision": "resume"})))
    assert budget_cap(p, now=T0 + 72 * 3600 + 60) is None


def test_coder_state_reads_the_latest_lease_and_the_baseline_canary():
    lease1, lease2 = {"lease_id": "L1"}, {"lease_id": "L2"}
    rows = entries(START,
                   ("decision", None, 1, {"decision": "coder starting", "lease": lease1, "server": {"port": 8000}}),
                   ("decision", None, 2, {"decision": "coder started", "lease": lease1, "canary": {"path": "c1"}}),
                   ("decision", 2, 3, {"decision": "test lease taken", "test_lease": {"lease_id": "T9"}}),
                   ("restore", 2, 4, {"step": "serve", "lease": lease2, "server": {"port": 8000, "pid": 7}}))
    assert coder_state(rows) == (lease2, {"port": 8000, "pid": 7}, {"path": "c1"})
    assert coder_state(entries(START)) == (None, None, None)


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield led


def test_a_partial_stage_directory_is_moved_aside_and_never_deleted(tmp_path, ledger):
    run = tmp_path / "run"
    d, resumed = open_stage_dir(run, STAGES[2], resuming=True, ledger=ledger)
    assert (d / "evidence").is_dir() and not resumed
    (d / "evidence" / "work.txt").write_text("half done")
    d, resumed = open_stage_dir(run, STAGES[2], resuming=True, ledger=ledger)  # no marker yet
    assert not resumed and not (d / "evidence" / "work.txt").exists()
    assert (run / "stages" / "2.partial-1" / "evidence" / "work.txt").read_text() == "half done"
    open_stage_dir(run, STAGES[2], resuming=False, ledger=ledger)
    assert (run / "stages" / "2.partial-2").is_dir()
    moved = [e["data"]["path"] for e in ledger.read() if e["event"] == "decision"]
    assert moved == ["stages/2.partial-1", "stages/2.partial-2"]


def test_a_resumed_stage_with_its_marker_keeps_its_directory(tmp_path, ledger):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, STAGES[2], resuming=False, ledger=ledger)
    (d / "test-result.json").write_text("{}")
    again, resumed = open_stage_dir(run, STAGES[2], resuming=True, ledger=ledger)
    assert again == d and resumed and (d / "test-result.json").exists()
    entry = ledger.read()[-1]["data"]
    assert entry["decision"] == "resume from marker"
    assert entry["marker"]["path"] == "stages/2/test-result.json" and len(entry["marker"]["sha256"]) == 64
    # A new attempt (after an escalation) never resumes, marker or not.
    _, resumed = open_stage_dir(run, STAGES[2], resuming=False, ledger=ledger)
    assert not resumed and (run / "stages" / "2.partial-1" / "test-result.json").exists()
