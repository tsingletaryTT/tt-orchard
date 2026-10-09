# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/run_view.py: what the UI says a run is doing now, its results, its timeline and comparisons.

The first lab run showed "chips resting" while a test ran on node4 for half an hour, and its results lived
only in raw JSON. These tests pin what the page now says instead.
"""
import json
import time
from pathlib import Path

from orchard import narrate, run_view, status, webui

T0 = 1_791_500_000


def e(seq, dt, event, stage=None, **data):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 + dt))
    return {"seq": seq, "ts": ts, "event": event, "stage": stage, "data": data}


LAB_TEST = [
    e(1, 0, "run_start", model="Altworld/Hemmingway-1", lab={"host": "node4"}, mode="lab"),
    e(2, 10, "decision", None, decision="coder starting"),
    e(3, 400, "decision", None, decision="coder started"),
    e(4, 401, "stage_start", 2),
    e(5, 402, "decision", 2, decision="agent step", phase="prepare", tier="small"),
    e(6, 500, "evidence", 2, what="transcript", status="done"),
    e(7, 501, "decision", 2, decision="copying files to the lab", where="lab node4"),
    e(8, 560, "decision", 2, decision="files copied to the lab", seconds=59.0,
      repos=["models--Altworld--Hemmingway-1", "models--incoai--Qwen3.8-27B-DFlash2"], where="lab node4"),
    e(9, 561, "decision", 2, decision="test lease taken", where="lab node4", test_lease={"chips": ["0000:01:00.0"]}),
    e(10, 562, "decision", 2, decision="hardware test started", chips=["0000:01:00.0", "0000:03:00.0"],
      deadline_s=3600.0, where="lab node4", command="python3 stages/2/serve_and_compare.py"),
]


def test_a_test_on_the_lab_is_what_the_run_is_doing_now_and_names_the_box():
    now = run_view.activity(LAB_TEST, "running", T0 + 900)
    assert now["kind"] == "test" and now["host"] == "node4"
    assert "Hardware test on node4" in now["text"] and "stage 2" in now["text"]
    assert now["elapsed_s"] == 900 - 562 and now["deadline_s"] == 3600.0


def test_the_copy_the_coder_boot_and_the_agent_each_read_plainly():
    assert run_view.activity(LAB_TEST[:7], "running", T0 + 520)["text"].startswith("Copying the run's files to node4")
    assert run_view.activity(LAB_TEST[:2], "running", T0 + 60)["kind"] == "coder"
    agent = run_view.activity(LAB_TEST[:5], "running", T0 + 450)
    assert agent["kind"] == "agent" and agent["actor"] == "grafter" and "prepare step" in agent["text"]


def test_a_finished_run_has_no_open_activity_and_a_plain_next_step():
    done = LAB_TEST + [e(11, 2000, "evidence", 2, what="hardware test", returncode=0),
                       e(12, 2001, "stage_end", 2, result="pass")]
    assert run_view.activity(done, "ready-for-operator-review", T0 + 3000)["text"] == "Finished: ready for your review"
    assert "RESULTS.md" in run_view.next_step("ready-for-operator-review")
    assert "sleep" not in " ".join(run_view.NEXT.values())               # not the runbook's words


def test_the_timeline_has_a_span_for_each_kind_and_marks_what_is_still_open():
    t = run_view.timeline(LAB_TEST, T0 + 900)
    kinds = {s["kind"]: s for s in t["spans"]}
    assert set(kinds) == {"stage", "agent", "coder", "copy", "test"}
    assert kinds["coder"]["end"] - kinds["coder"]["start"] == 390
    assert kinds["copy"]["seconds"] == 59.0 and kinds["copy"]["host"] == "node4"
    assert kinds["test"]["open"] is True and kinds["stage"]["open"] is True
    assert t["start"] == T0 and t["end"] == T0 + 900


def test_results_read_stage_0_2_and_4_and_where_each_test_ran(tmp_path):
    run = tmp_path / "run"
    (run / "stages" / "0").mkdir(parents=True)
    (run / "stages" / "0" / "delta.json").write_text(json.dumps({"class": "weights-only", "path": "weights-only",
                                                                 "nearest_model": "Qwen/Qwen3.8-27B", "model": "a/b"}))
    (run / "stages" / "2" / "evidence").mkdir(parents=True)
    (run / "stages" / "2" / "evidence" / "swap-check.json").write_text(json.dumps({"result_draft": {
        "top1_agreement": 0.9375, "n_tokens": 32, "server_ready_s": 1768.3, "coherent": True, "free_run_text": "x" * 500}}))
    (run / "stages" / "2" / "test-result.json").write_text(json.dumps({"seconds": 1780.0}))
    (run / "stages" / "4" / "tests" / "1").mkdir(parents=True)
    (run / "stages" / "4" / "tests" / "4").mkdir(parents=True)
    (run / "stages" / "4" / "result.json").write_text(json.dumps({"configs": [
        {"chips": 1, "pass": True, "top1_agreement": 0.97, "server_ready_s": 1728.3},
        {"chips": 4, "pass": False, "reason": "not run"}]}))
    (run / "stages" / "4" / "tests" / "4" / "test-result.json").write_text(json.dumps(
        {"not_run": "not run: the lab node4 has 2 chips"}))
    entries = [e(1, 0, "decision", 2, decision="hardware test started", where="lab node4"),
               e(2, 1, "decision", 4, decision="hardware test started", config=1, where="lab node4")]
    r = run_view.results(run, entries)
    assert r["triage"]["class"] == "weights-only"
    assert r["swap"]["top1_agreement"] == 0.9375 and r["swap"]["host"] == "node4"
    assert len(r["swap"]["free_run_text"]) == run_view.FREE_TEXT_CHARS
    rows = {c["chips"]: c for c in r["configs"]}
    assert rows[1]["pass"] is True and rows[1]["host"] == "node4"
    assert rows[4]["pass"] is False and "2 chips" in rows[4]["reason"]


def test_the_narration_names_the_lab_and_what_was_copied():
    lines = [l.text for ent in LAB_TEST for l in narrate._decision(ent, ent["data"]) if ent["event"] == "decision"]
    joined = "\n".join(lines)
    assert "copied to node4 in 59.0 s (with Altworld/Hemmingway-1, incoai/Qwen3.8-27B-DFlash2)" in joined
    assert "test lease taken on node4" in joined and "hardware test started on node4" in joined


def test_the_ledger_tab_shows_a_measurements_value_and_a_lab_copy():
    m = {"event": "measurement", "stage": None, "data": {"name": "queue_wait_seconds", "value": 0.03, "unit": "s"}}
    assert status.summarize(m) == "queue_wait_seconds = 0.03 s"
    assert "in 59.0 s" in status.summarize(LAB_TEST[7]) and "Hemmingway" in status.summarize(LAB_TEST[7])


def test_the_files_results_cite_can_be_opened():
    for rel in ("stages/2/evidence/server.log", "stages/2/swap_config.json", "stages/4/tests/plan.json",
                "stages/4/tests/2/output.txt", "stages/4/configs/1/swap_config.json",
                "stages/4/configs/1/evidence/swap-check.json", "evidence/coder-canary-00007.txt"):
        assert webui.ALLOWED.match(rel), rel
    for rel in ("ledger.jsonl", "home/.cache/x", "stages/2/model-dir/config.json", "../etc/passwd"):
        assert not webui.ALLOWED.match(rel), rel
