# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Live narration of a run (orchard/narrate.py): what is starting, who is doing what, what was tried.

The narrator reads the ledger and the agent logs without taking a lock and writes nothing. Each test
names the line a person would have needed on a run that blocked and gave no hint."""
import io
import json
from datetime import timezone
from pathlib import Path

import pytest

from orchard import narrate, ui
from orchard.ledger import Ledger

PLAIN = ui.PLAIN


def entry(event, stage=None, seq=1, **data):
    return {"event": event, "stage": stage, "seq": seq, "ts": "2026-10-07T05:20:31Z", "data": data}


def texts(entries):
    return [l.text for e in entries for l in narrate.lines_for(e)]


def one(e):
    got = narrate.lines_for(e)
    assert len(got) == 1, got
    return got[0]


# ---- ledger events ----------------------------------------------------------------------------

def test_a_stage_start_names_the_stage_and_the_attempt():
    l = one(entry("stage_start", 2, escalated=False, resumed=False))
    assert l.actor == "orchardist" and "stage 2" in l.text and "graft" in l.text


def test_an_escalated_stage_start_says_so():
    assert "escalated" in one(entry("stage_start", 2, escalated=True, resumed=False)).text


def test_a_failed_stage_end_shows_the_first_reasons():
    l = one(entry("stage_end", 2, result="fail", reasons=["serves must be true", "no evidence", "third"]))
    assert "fail" in l.text and "serves must be true" in l.text and "no evidence" in l.text


def test_a_passing_stage_end_says_pass():
    assert "pass" in one(entry("stage_end", 4, result="pass")).text


def test_the_coder_start_names_the_model_and_port():
    got = narrate.lines_for(entry("decision", decision="coder starting",
                                  server={"target": "raahemnabeel/qwen3-coder-next-blackhole", "port": 8001}))
    assert "raahemnabeel/qwen3-coder-next-blackhole" in got[0].text and "8001" in got[0].text


def test_the_coder_ready_line_says_the_canary_passed():
    assert "ready" in one(entry("decision", decision="coder started")).text


def test_a_lease_names_the_chips():
    assert "0000:01:00.0" in one(entry("decision", decision="lease granted", chips=["0000:01:00.0", "0000:02:00.0"],
                                       lease_id="abc", waited_s=1.0)).text


def test_a_hardware_test_start_shows_the_command_and_deadline():
    l = one(entry("decision", 2, decision="hardware test started", command="python3 stages/2/run_sidecar_checks.py",
                  deadline_s=7200.0, chips=["0000:03:00.0"], lease_id="x"))
    assert "run_sidecar_checks.py" in l.text and "2h" in l.text


@pytest.mark.parametrize("rc, word", [(0, "passed"), (4, "failed"), (None, "failed")])
def test_a_hardware_test_result_says_passed_or_failed_with_the_exit_code(rc, word):
    l = one(entry("evidence", 2, what="hardware test", returncode=rc, timed_out=rc is None, path="p", sha256="s"))
    assert word in l.text and (rc is None or f"exit {rc}" in l.text)


def test_a_timed_out_test_says_so():
    assert "timed out" in one(entry("evidence", 2, what="hardware test", returncode=None, timed_out=True,
                                    path="p", sha256="s")).text


def test_an_agent_step_names_the_tier_actor_model_and_phase():
    l = one(entry("decision", 2, decision="agent step", tier="small", model="Qwen/Qwen3-Coder-Next", phase="finish",
                  escalated=False, skill="/x/weights-swap-check.md"))
    assert l.actor == "grafter" and "Qwen/Qwen3-Coder-Next" in l.text and "finish" in l.text
    assert "weights-swap-check" in l.text


def test_the_large_tier_is_the_head_grower_and_cpu_the_seasonal_hand():
    base = dict(decision="agent step", model="m", phase="run", escalated=True, skill="s.md")
    assert one(entry("decision", 1, tier="large", **base)).actor == "head grower"
    assert one(entry("decision", 1, tier="cpu", **base)).actor == "seasonal hand"


def test_a_gate_feedback_lists_the_reasons_the_agent_was_given():
    l = one(entry("decision", 2, decision="gate feedback", file="stages/2/result.json", phase="finish",
                  reasons=["wiring_check must have passed", "head_sha256 mismatch"]))
    assert "wiring_check must have passed" in l.text and "head_sha256 mismatch" in l.text


def test_no_gate_feedback_after_a_failed_test_says_why():
    l = one(entry("decision", 2, decision="no gate feedback: the hardware test failed",
                  problem="the hardware test exited with code 4", returncode=4, timed_out=False))
    assert "exited with code 4" in l.text


def test_a_watchdog_escalation_names_the_detector_and_the_sheepdog():
    l = one(entry("escalate", 1, rung="escalate", watchdog=True, agent="stage-agent",
                  findings=[{"detector": "repeated_tool_call", "summary": "the same tool call (write_file) 3 times"}]))
    assert l.actor == "sheepdog" and "write_file" in l.text and "repeated_tool_call" in l.text


def test_a_nudge_shows_the_message():
    assert "write a file" in one(entry("retry", 2, message="write a file now")).text


def test_a_pause_and_a_block_show_the_reason_in_full():
    reason = "stage 1 failed after escalation: the agent step ended: turns no final answer after 60 turns"
    assert reason in one(entry("decision", decision="pause", reason=reason)).text
    l = one(entry("decision", decision="blocked", code="stage-failed", reason=reason))
    assert "stage-failed" in l.text and reason in l.text


def test_a_notice_names_what_happened():
    assert "cut off" in one(entry("notice", 2, what="reply cut off at max_tokens", in_a_row=1, turn=12)).text


def test_a_park_step_is_shown():
    assert "reset" in one(entry("park", 4, step="reset")).text


def test_a_measurement_shows_name_value_and_unit():
    assert "queue_wait_seconds" in one(entry("measurement", name="queue_wait_seconds", value=0.036, unit="s",
                                             label="measured")).text


def test_evidence_files_are_quiet():
    assert narrate.lines_for(entry("evidence", 2, what="evidence file", path="p", sha256="s")) == []


def test_an_unknown_decision_is_still_shown():
    assert "some new thing" in one(entry("decision", decision="some new thing")).text


def test_a_tier_substitution_is_shown_in_the_words_of_the_ledger():
    assert "tier small serves" in one(entry("decision", 1, decision="tier substituted",
                                            note="tier large is not serving; tier small serves the same model")).text


# ---- the agent's turns ------------------------------------------------------------------------

def call(name, **args):
    return {"id": f"id-{name}-{len(args)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def turn_record(n, calls=(), content="", sent_tools=()):
    return {"turn": n, "sent": [{"role": "system", "content": "x"}] + [
        {"role": "tool", "tool_call_id": i, "content": c} for i, c in sent_tools],
            "received": {"role": "assistant", "content": content, "tool_calls": list(calls) or None},
            "finish_reason": "tool_calls" if calls else "stop"}


def test_a_shell_call_shows_the_command():
    tn = narrate.TurnNarrator()
    got = tn.lines_for(turn_record(1, [call("shell", command="ls -la /tmp")]), actor="grafter")
    assert got[0].actor == "grafter" and "ls -la /tmp" in got[0].text


def test_read_and_write_calls_show_the_path_and_the_size_written():
    tn = narrate.TurnNarrator()
    got = tn.lines_for(turn_record(1, [call("read_file", path="stages/2/x.json"),
                                       call("write_file", path="stages/2/y.py", content="a" * 123)]), actor="grafter")
    assert "reads stages/2/x.json" in got[0].text
    assert "writes stages/2/y.py" in got[1].text and "123" in got[1].text


def test_the_result_of_a_call_appears_with_the_next_turn():
    tn = narrate.TurnNarrator()
    c = call("shell", command="tt-model list")
    tn.lines_for(turn_record(1, [c]), actor="grafter")
    got = tn.lines_for(turn_record(2, [], content="done", sent_tools=[(c["id"], "exit 0\nNo bundles installed.\n")]),
                       actor="grafter")
    joined = " | ".join(l.text for l in got)
    assert "exit 0" in joined and "No bundles installed." in joined


def test_a_failing_result_is_marked_and_long_output_is_cut():
    tn = narrate.TurnNarrator()
    c = call("shell", command="false")
    tn.lines_for(turn_record(1, [c]), actor="grafter")
    got = tn.lines_for(turn_record(2, sent_tools=[(c["id"], "exit 1\n" + "x" * 5000)]), actor="grafter")
    text = " ".join(l.text for l in got)
    assert "exit 1" in text and len(text) < 400


def test_assistant_text_is_shown_short():
    tn = narrate.TurnNarrator()
    got = tn.lines_for(turn_record(1, content="I will check the log now. " * 40), actor="grafter")
    assert got and len(got[0].text) < 300


def test_an_empty_or_cut_off_reply_is_called_out():
    tn = narrate.TurnNarrator()
    rec = turn_record(1)
    rec["finish_reason"] = "length"
    assert "cut off" in tn.lines_for(rec, actor="grafter")[0].text
    rec2 = turn_record(2)
    assert "empty" in tn.lines_for(rec2, actor="grafter")[0].text


# ---- the narrator over a run directory --------------------------------------------------------

@pytest.fixture
def run(tmp_path):
    return tmp_path


def write_ledger(run, *events):
    with Ledger(run / "ledger.jsonl") as led:
        for ev, stage, data in events:
            led.append(ev, stage, **data)


def narrator(run, out, *, replay=False, clock=None, heartbeat_s=60):
    return narrate.Narrator(run, PLAIN, out, replay=replay, clock=clock or (lambda: 0.0), heartbeat_s=heartbeat_s,
                            tz=timezone.utc)


def test_new_ledger_entries_are_printed_once_each(run):
    write_ledger(run, ("stage_start", 0, {"escalated": False, "resumed": False}))
    out = io.StringIO()
    n = narrator(run, out)            # starts at the end: that entry is history
    n.poll()
    assert out.getvalue() == ""
    with Ledger(run / "ledger.jsonl") as led:
        led.append("stage_start", 1, escalated=False, resumed=False)
    n.poll()
    n.poll()
    assert out.getvalue().count("stage 1") == 1 and "stage 0" not in out.getvalue()


def test_replay_prints_the_history_first(run):
    write_ledger(run, ("stage_start", 0, {"escalated": False, "resumed": False}),
                 ("stage_end", 0, {"result": "pass"}))
    out = io.StringIO()
    narrator(run, out, replay=True).poll()
    assert "stage 0" in out.getvalue() and "pass" in out.getvalue()


def test_a_line_is_prefixed_with_the_time_and_the_actor(run):
    write_ledger(run, ("stage_start", 0, {"escalated": False, "resumed": False}))
    out = io.StringIO()
    narrator(run, out, replay=True).poll()
    import re
    assert re.match(r"\d\d:\d\d:\d\d  orchardist ", out.getvalue().splitlines()[0])


def test_no_right_hand_border_is_printed(run):
    write_ledger(run, ("stage_end", 2, {"result": "fail", "reasons": ["a"]}))
    out = io.StringIO()
    narrator(run, out, replay=True).poll()
    assert not any(l.rstrip().endswith(("║", "│", "|")) for l in out.getvalue().splitlines())


def test_agent_turns_are_followed_live_and_only_new_ones(run):
    log = run / "stages/2/log"
    log.mkdir(parents=True)
    f = log / "prepare-00005.jsonl"
    f.write_text(json.dumps(turn_record(1, [call("shell", command="old")])) + "\n")
    write_ledger(run, ("decision", 2, {"decision": "agent step", "tier": "small", "model": "m", "phase": "prepare",
                                       "escalated": False, "skill": "s.md"}))
    out = io.StringIO()
    n = narrator(run, out)
    n.poll()
    with open(f, "a") as fh:
        fh.write(json.dumps(turn_record(2, [call("shell", command="new command")])) + "\n")
    n.poll()
    n.poll()
    assert out.getvalue().count("new command") == 1 and "old" not in out.getvalue()


def test_a_half_written_line_waits_for_its_newline(run):
    log = run / "stages/2/log"
    log.mkdir(parents=True)
    f = log / "prepare-00005.jsonl"
    f.write_text("")
    write_ledger(run, ("stage_start", 2, {"escalated": False, "resumed": False}))
    out = io.StringIO()
    n = narrator(run, out)
    n.poll()
    text = json.dumps(turn_record(1, [call("shell", command="half")]))
    with open(f, "a") as fh:
        fh.write(text[:20])
    n.poll()
    assert "half" not in out.getvalue()
    with open(f, "a") as fh:
        fh.write(text[20:] + "\n")
    n.poll()
    assert "half" in out.getvalue()


def test_a_log_file_created_after_the_start_is_read_from_its_beginning(run):
    write_ledger(run, ("stage_start", 2, {"escalated": False, "resumed": False}))
    out = io.StringIO()
    n = narrator(run, out)
    n.poll()
    log = run / "stages/2/log"
    log.mkdir(parents=True)
    (log / "run-00009.jsonl").write_text(json.dumps(turn_record(1, [call("shell", command="first")])) + "\n")
    n.poll()
    assert "first" in out.getvalue()


def test_a_broken_log_line_does_not_stop_the_narration(run):
    log = run / "stages/2/log"
    log.mkdir(parents=True)
    write_ledger(run, ("stage_start", 2, {"escalated": False, "resumed": False}))
    out = io.StringIO()
    n = narrator(run, out)
    n.poll()
    (log / "run-1.jsonl").write_text("not json\n" + json.dumps(turn_record(1, [call("shell", command="after")])) + "\n")
    n.poll()
    assert "after" in out.getvalue()


def test_a_missing_ledger_is_not_an_error(run):
    out = io.StringIO()
    narrator(run, out).poll()
    assert out.getvalue() == ""


# ---- the heartbeat ----------------------------------------------------------------------------

def test_a_quiet_minute_prints_what_it_is_waiting_for(run):
    now = [0.0]
    write_ledger(run, ("decision", None, {"decision": "coder starting",
                                          "server": {"target": "org/coder", "port": 8001}}))
    out = io.StringIO()
    n = narrator(run, out, replay=True, clock=lambda: now[0])
    n.poll()
    before = out.getvalue()
    now[0] = 30
    n.poll()
    assert out.getvalue() == before                 # not yet
    now[0] = 75
    n.poll()
    assert "still" in out.getvalue()[len(before):] and "org/coder" in out.getvalue()[len(before):]
    assert "1m" in out.getvalue()[len(before):]


def test_a_running_hardware_test_shows_the_tail_of_its_output(run):
    now = [0.0]
    write_ledger(run, ("decision", 2, {"decision": "hardware test started", "command": "python3 x.py",
                                       "deadline_s": 3600.0, "chips": ["c"], "lease_id": "l"}))
    ev = run / "stages/2/evidence"
    ev.mkdir(parents=True)
    (ev / "hw-test-output.txt").write_text("compiling\nloading weights 40%\n\n")
    out = io.StringIO()
    n = narrator(run, out, replay=True, clock=lambda: now[0])
    n.poll()
    now[0] = 100
    n.poll()
    assert "loading weights 40%" in out.getvalue()


def test_no_heartbeat_once_the_test_has_a_result(run):
    now = [0.0]
    write_ledger(run, ("decision", 2, {"decision": "hardware test started", "command": "x", "deadline_s": 60.0,
                                       "chips": [], "lease_id": "l"}),
                 ("evidence", 2, {"what": "hardware test", "returncode": 0, "timed_out": False, "path": "p",
                                  "sha256": "s"}))
    out = io.StringIO()
    n = narrator(run, out, replay=True, clock=lambda: now[0])
    n.poll()
    before = out.getvalue()
    now[0] = 500
    n.poll()
    assert "still" not in out.getvalue()[len(before):]


# ---- what was tried, and how to unblock -------------------------------------------------------

def test_the_digest_lists_the_last_things_the_agent_tried_in_the_stage(run):
    log = run / "stages/1/log"
    log.mkdir(parents=True)
    rows = [turn_record(i, [call("shell", command=f"cmd-{i}")]) for i in range(1, 13)]
    (log / "run-00009.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_ledger(run, ("stage_start", 1, {"escalated": False, "resumed": False}),
                 ("escalate", 1, {"rung": "escalate", "watchdog": True, "agent": "a",
                                  "findings": [{"detector": "repeated_tool_call", "summary": "same call 3 times"}]}),
                 ("decision", 1, {"decision": "pause", "reason": "stage 1 failed after escalation"}))
    lines = narrate.what_was_tried(run, stage=1, limit=5)
    text = "\n".join(lines)
    assert "cmd-12" in text and "cmd-8" in text and "cmd-7" not in text
    assert "same call 3 times" in text


def test_the_digest_reports_a_failed_hardware_test_with_its_last_output(run):
    ev = run / "stages/2/evidence"
    ev.mkdir(parents=True)
    (ev / "hw-test-output.txt").write_text("loading\nRuntimeError: Unset sample_on_device_mode\n")
    write_ledger(run, ("stage_start", 2, {"escalated": False, "resumed": False}),
                 ("evidence", 2, {"what": "hardware test", "returncode": 4, "timed_out": False, "path": "p",
                                  "sha256": "s"}))
    text = "\n".join(narrate.what_was_tried(run, stage=2, limit=5))
    assert "exit 4" in text and "Unset sample_on_device_mode" in text


def test_the_digest_says_so_when_there_is_nothing_to_show(run):
    write_ledger(run, ("stage_start", 3, {"escalated": False, "resumed": False}))
    assert "nothing" in "\n".join(narrate.what_was_tried(run, stage=3, limit=5)).lower()


@pytest.mark.parametrize("code", ["needs-new-model-code", "retry-budget-spent", "stage-failed", "agent-stuck",
                                  "disk-full", "hardware-unhealthy", "coder-unusable", "blocked", "unclassified"])
def test_every_block_code_has_advice_that_names_a_concrete_next_step(code):
    advice = narrate.how_to_unblock(code)
    assert len(advice) >= 2 and all(a.strip() for a in advice)
    assert any("tt-orchard bringup" in a or "tt-orchard status" in a or "gozer" in a or "df " in a or "read" in a.lower()
               for a in advice)


def test_the_block_codes_and_the_advice_table_cannot_drift():
    from orchard import blocked
    assert set(narrate.UNBLOCK) == set(blocked.REASONS)


def test_blocked_md_carries_what_was_tried_and_how_to_unblock(run):
    from orchard import blocked
    log = run / "stages/2/log"
    log.mkdir(parents=True)
    (log / "finish-1.jsonl").write_text(json.dumps(turn_record(1, [call("shell", command="grep -rn timeout /vllm")])) + "\n")
    write_ledger(run, ("run_start", None, {"model": "Cloudflare/clef"}),
                 ("stage_start", 2, {"escalated": False, "resumed": False}),
                 ("decision", 2, {"decision": "pause", "reason": "watchdog: 20 model turns in a row and no file written",
                                  "watchdog": True, "findings": [{"detector": "no_file_written", "summary": "s"}]}))
    from orchard.ledger import read_entries
    blocked.write_bundle(run, read_entries(run / "ledger.jsonl"), "agent-stuck", "watchdog: 20 turns", now=0)
    md = (run / "BLOCKED.md").read_text()
    assert "## What was tried" in md and "grep -rn timeout /vllm" in md
    assert "## How to unblock" in md and "agent-stuck" in md.split("## How to unblock")[0] or "stuck" in md.split("## How to unblock")[1]


def test_a_narrator_that_breaks_says_so_and_does_not_raise(run, monkeypatch):
    out = io.StringIO()
    n = narrator(run, out)

    def boom():
        raise RuntimeError("bad ledger")
    monkeypatch.setattr(n, "poll", boom)
    n.start()                  # must not raise: the run goes on
    n.stop()
    assert "live narration stopped: RuntimeError: bad ledger" in out.getvalue()


def test_the_background_thread_prints_new_entries_without_a_poll_call(run, monkeypatch):
    import time as _t
    monkeypatch.setattr(narrate, "POLL_S", 0.05)
    out = io.StringIO()
    n = narrator(run, out)
    n.start()
    write_ledger(run, ("stage_start", 5, {"escalated": False, "resumed": False}))
    deadline = _t.time() + 5
    while "stage 5" not in out.getvalue() and _t.time() < deadline:
        _t.sleep(0.05)
    n.stop()
    assert "stage 5" in out.getvalue()


def test_stop_prints_what_arrived_since_the_last_poll(run):
    out = io.StringIO()
    n = narrator(run, out)
    n.start()
    write_ledger(run, ("stage_start", 6, {"escalated": False, "resumed": False}))
    n.stop()
    assert "stage 6" in out.getvalue()


def test_finished_is_set_by_ready_or_blocked_and_cleared_by_a_resume(run):
    n = narrator(run, io.StringIO(), replay=True)
    write_ledger(run, ("decision", None, {"decision": "blocked", "code": "x", "reason": "r"}))
    n.poll()
    assert n.finished
    with Ledger(run / "ledger.jsonl") as led:
        led.append("decision", None, decision="resume", by="retry")
    n.poll()
    assert not n.finished


def test_a_partial_line_after_complete_lines_is_not_lost_or_repeated(run):
    log = run / "stages/2/log"
    log.mkdir(parents=True)
    f = log / "prepare-1.jsonl"
    f.write_text("")
    write_ledger(run, ("stage_start", 2, {"escalated": False, "resumed": False}))
    out = io.StringIO()
    n = narrator(run, out)
    n.poll()
    first = json.dumps(turn_record(1, [call("shell", command="one")]))
    second = json.dumps(turn_record(2, [call("shell", command="two")]))
    with open(f, "a") as fh:
        fh.write(first + "\n" + second[:15])
    n.poll()
    with open(f, "a") as fh:
        fh.write(second[15:] + "\n")
    n.poll()
    assert out.getvalue().count("runs: one") == 1 and out.getvalue().count("runs: two") == 1


def test_a_failure_in_a_later_poll_is_reported_by_the_thread_and_stops_it(run, monkeypatch):
    import time as _t
    monkeypatch.setattr(narrate, "POLL_S", 0.02)
    out = io.StringIO()
    n = narrator(run, out)
    real = n.poll
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("later failure")
        real()
    monkeypatch.setattr(n, "poll", flaky)
    n.start()
    deadline = _t.time() + 5
    while "later failure" not in out.getvalue() and _t.time() < deadline:
        _t.sleep(0.02)
    n.stop()
    assert "live narration stopped: RuntimeError: later failure" in out.getvalue()


def test_a_failed_hardware_test_is_followed_by_the_end_of_its_output(run):
    ev = run / "stages/2/evidence"
    ev.mkdir(parents=True)
    (ev / "hw-test-output.txt").write_text("loading\nRuntimeError: Unset sample_on_device_mode\nexit now\n")
    write_ledger(run, ("evidence", 2, {"what": "hardware test", "returncode": 4, "timed_out": False, "path": "p",
                                       "sha256": "s"}))
    out = io.StringIO()
    narrator(run, out, replay=True).poll()
    text = out.getvalue()
    assert "hardware test failed (exit 4)" in text and "last output" in text
    assert "Unset sample_on_device_mode" in text


def test_a_passing_hardware_test_shows_no_output_tail(run):
    ev = run / "stages/2/evidence"
    ev.mkdir(parents=True)
    (ev / "hw-test-output.txt").write_text("all good\n")
    write_ledger(run, ("evidence", 2, {"what": "hardware test", "returncode": 0, "timed_out": False, "path": "p",
                                       "sha256": "s"}))
    out = io.StringIO()
    narrator(run, out, replay=True).poll()
    assert "all good" not in out.getvalue()
