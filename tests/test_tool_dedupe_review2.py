# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 488, round 1: five findings on the tool-result dedupe.

1. DedupeStreak must start a step's count on the step's first response, not by comparing turn numbers of tool results: two steps that
   both make their tool calls on turn 1 shared one window and one `_fired` flag.
2. scripts/dedupe_replay.py forgot nothing at a context restart although the live Tools do (reset_dedupe).
3. It paired results with calls by position although a capped reply can log an incomplete call that never ran.
4. It crashed on tool arguments that are already a dict (AgentStep keeps the dict in the logged `received`).
5. It replayed each log file as its own step although a gate-feedback or wrap-up continuation shares the conversation and the Tools.
Each test is written against what the live loop does (orchard/agent.py), and the replay tests build their records the way AgentStep logs them."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchard.watchdog import DedupeStreak, Event   # noqa: E402

_spec = importlib.util.spec_from_file_location("dedupe_replay_under_test", ROOT / "scripts" / "dedupe_replay.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


# ---- 1. the watchdog detector ---------------------------------------------------------------------------------------------------------

def response(turn, ts=0.0):
    return Event(ts=ts, agent="a", kind="response", turn=turn)


def result(turn, deduped, ts=0.0, name="stages/0/evidence/a.txt"):
    return Event(ts=ts, agent="a", kind="tool_result", tool="read_file", output_hash="h", turn=turn, deduped=deduped,
                 name=name if deduped else None)


def test_a_step_whose_calls_all_come_on_turn_one_starts_with_an_empty_window_after_a_step_that_did_the_same():
    d = DedupeStreak()
    # Step A: one response (turn 1) with three repeated reads: the hint fires.
    d.feed(response(1))
    assert [d.feed(result(1, True)) for _ in range(3)][-1] is not None
    # Step B starts: its first response is turn 1 again, and its three repeated reads on turn 1 must fire again.
    d.feed(response(1))
    found = [d.feed(result(1, True)) for _ in range(3)]
    assert found[0] is None and found[1] is None and found[2] is not None


def test_two_deduped_results_left_over_from_a_step_do_not_count_toward_the_next_step():
    d = DedupeStreak()
    d.feed(response(1))
    d.feed(result(1, True))
    d.feed(result(1, True))
    d.feed(response(1))                                     # a new step: the two earlier notes are not this step's
    assert d.feed(result(1, True)) is None                  # one note, not three


def test_several_tool_results_within_one_response_are_one_step_and_still_count():
    d = DedupeStreak()
    d.feed(response(1))
    found = [d.feed(result(1, True)) for _ in range(3)]
    assert found[2] is not None                              # the first response does not reset the count mid-response


def test_a_later_response_in_the_same_step_does_not_reset_the_count():
    d = DedupeStreak()
    d.feed(response(1))
    d.feed(result(1, True))
    d.feed(response(2))
    d.feed(result(2, True))
    d.feed(response(3))
    assert d.feed(result(3, True)) is not None               # three notes over three turns of one step


def test_a_continuation_that_keeps_counting_turns_is_not_a_new_step():
    d = DedupeStreak()
    d.feed(response(1)); d.feed(result(1, True))
    d.feed(response(2)); d.feed(result(2, True))
    d.feed(response(7))                                       # `continue_with` numbers its turns on from the step's
    assert d.feed(result(7, True)) is not None


# ---- helpers that build records the way AgentStep.log them -------------------------------------------------------------------------

SYSTEM = {"role": "system", "content": "rules"}
USER = {"role": "user", "content": "task"}


def call(name, args, ident="c"):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, (str, dict)) else args}}


def rec(turn, sent, calls=(), finish="tool_calls"):
    msg = {"role": "assistant", "content": ""}
    if calls:
        msg["tool_calls"] = list(calls)
    return {"turn": turn, "sent": sent, "received": msg, "finish_reason": finish, "usage": {}, "request": {}}


def tool(text):
    return {"role": "tool", "tool_call_id": "c", "content": text}


def read_a():
    return call("read_file", json.dumps({"path": "stages/0/a.txt"}))


# ---- 2. a context restart makes the step forget what it read --------------------------------------------------------------------------------

def test_a_repeat_read_after_a_context_restart_is_not_a_note(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_a()]),
        rec(2, [tool("AAA" * 100)], [read_a()]),           # the first read's answer is cached here
        rec(3, [SYSTEM, {"role": "user", "content": "restart: everything is on disk"}], [read_a()]),     # the replacement conversation
        rec(4, [tool("AAA" * 100)], [], finish="stop"),     # a read of the same unchanged file in the NEW context: the live Tools forgot it
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["read_calls"] == 2 and out["read_notes"] == 0


def test_without_a_restart_the_same_repeat_is_a_note(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_a()]),
        rec(2, [tool("AAA" * 100)], [read_a()]),
        rec(3, [tool("AAA" * 100)], [], finish="stop"),
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["read_calls"] == 2 and out["read_notes"] == 1


def test_a_repeat_read_only_command_after_a_context_restart_is_not_a_note(tmp_path):
    ls = call("shell", json.dumps({"command": "ls"}))
    records = [
        rec(1, [SYSTEM, USER], [ls]),
        rec(2, [tool("exit 0\nstages\n")], [ls]),
        rec(3, [SYSTEM, {"role": "user", "content": "restart"}], [ls]),
        rec(4, [tool("exit 0\nstages\n")], [], finish="stop"),
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["shell_calls"] == 2 and out["shell_notes"] == 0


def test_the_same_command_without_a_restart_is_a_note(tmp_path):
    ls = call("shell", json.dumps({"command": "ls"}))
    records = [
        rec(1, [SYSTEM, USER], [ls]),
        rec(2, [tool("exit 0\nstages\n")], [ls]),
        rec(3, [tool("exit 0\nstages\n")], [], finish="stop"),
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["shell_calls"] == 2 and out["shell_notes"] == 1


# ---- 3. a capped reply can log a call that never ran -------------------------------------------------------------------------------------

def test_an_incomplete_call_in_a_capped_reply_is_not_paired_with_the_result_of_the_read(tmp_path):
    cut = call("write_file", '{"path": "stages/0/b.txt", "content": "abc')           # cut off inside the arguments: the live loop does not run it
    records = [
        rec(1, [SYSTEM, USER], [cut, read_a()], finish="length"),
        rec(2, [tool("AAA" * 100)], [], finish="stop"),                               # the one result is the read's
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["read_calls"] == 1 and out["chars"] == 300


def test_an_incomplete_write_does_not_clear_the_caches(tmp_path):
    cut = call("write_file", '{"path": "stages/0/a.txt", "content": "abc')
    records = [
        rec(1, [SYSTEM, USER], [read_a()]),
        rec(2, [tool("AAA" * 100)], [cut, read_a()], finish="length"),
        rec(3, [tool("AAA" * 100)], [], finish="stop"),     # the cut write never ran, so a.txt is unchanged and the repeat is a note
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["read_notes"] == 1


def test_complete_calls_are_all_kept_when_the_reply_was_not_capped(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_a(), call("read_file", json.dumps({"path": "stages/0/b.txt"}))]),
        rec(2, [tool("AAA"), tool("BBB")], [], finish="stop"),
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["read_calls"] == 2


# ---- 4. arguments that are already a dict, and shapes that are neither ----------------------------------------------------------------

def test_dict_arguments_do_not_crash_and_are_replayed(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [call("read_file", {"path": "stages/0/a.txt"})]),
        rec(2, [tool("AAA" * 100)], [call("read_file", {"path": "stages/0/a.txt"})]),
        rec(3, [tool("AAA" * 100)], [], finish="stop"),
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["read_calls"] == 2 and out["read_notes"] == 1


def test_dict_arguments_keep_their_path_so_two_files_with_the_same_text_are_not_one_read(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [call("read_file", {"path": "stages/0/a.txt"}), call("read_file", {"path": "stages/0/b.txt"})]),
        rec(2, [tool("SAME" * 100), tool("SAME" * 100)], [], finish="stop"),
    ]
    out = replay.replay_step(records, tmp_path)
    assert out["read_calls"] == 2 and out["read_notes"] == 0


def test_arguments_of_any_other_shape_do_not_crash(tmp_path):
    for odd in (["a"], 5, None, "[1, 2]", "not json", "", "{" * 5000):
        records = [rec(1, [SYSTEM, USER], [call("read_file", odd)]), rec(2, [tool("x")], [], finish="stop")]
        out = replay.replay_step(records, tmp_path)
        assert isinstance(out, dict) and out["read_notes"] == 0      # no exception is the point


# ---- 5. continuations share the conversation ------------------------------------------------------------------------------------------------

def write_log(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def make_run(tmp_path):
    run = tmp_path / "run"
    log = run / "stages" / "2" / "log"
    # The first log ends after one read; the gate-feedback continuation (a user message, no system prompt) reads the same unchanged file again.
    write_log(log / "prepare-00045.jsonl", [
        rec(1, [SYSTEM, USER], [read_a()]),
        rec(2, [tool("AAA" * 100)], [], finish="stop"),
    ])
    write_log(log / "prepare-continuation-00059.jsonl", [
        rec(3, [{"role": "user", "content": "the gate found a problem"}], [read_a()]),
        rec(4, [tool("AAA" * 100)], [], finish="stop"),
    ])
    return run


def test_a_gate_feedback_continuation_shares_the_caches_and_is_one_step(tmp_path):
    out = replay.replay_run(make_run(tmp_path))
    assert out["steps"] == 1 and out["read_calls"] == 2 and out["read_notes"] == 1


def test_a_wrapup_continuation_is_grouped_the_same_way(tmp_path):
    run = make_run(tmp_path)
    (run / "stages" / "2" / "log" / "prepare-continuation-00059.jsonl").rename(run / "stages" / "2" / "log" / "prepare-wrapup-00059.jsonl")
    out = replay.replay_run(run)
    assert out["steps"] == 1 and out["read_notes"] == 1


def test_a_log_of_another_phase_is_a_new_step_with_empty_caches(tmp_path):
    run = make_run(tmp_path)
    write_log(run / "stages" / "2" / "log" / "finish-00070.jsonl", [
        rec(1, [SYSTEM, USER], [read_a()]),
        rec(2, [tool("AAA" * 100)], [], finish="stop"),
    ])
    out = replay.replay_run(run)
    assert out["steps"] == 2 and out["read_notes"] == 1        # the finish step's read is its first, so it is not a note


def test_continuations_are_applied_in_execution_order_not_file_name_order(tmp_path):
    run = tmp_path / "run"
    log = run / "stages" / "2" / "log"
    write_log(log / "repair-00100.jsonl", [rec(1, [SYSTEM, USER], [read_a()]), rec(2, [tool("AAA" * 100)], [], finish="stop")])
    # The wrap-up (ledger sequence 105) ran before the gate-feedback continuation (110) although its file name sorts after it: it rewrote a.txt.
    write_log(log / "repair-wrapup-00105.jsonl", [
        rec(3, [{"role": "user", "content": "wrap up"}], [call("write_file", json.dumps({"path": "stages/0/a.txt", "content": "new"}))]),
        rec(4, [tool("wrote")], [read_a()]),
        rec(5, [tool("BBB" * 100)], [], finish="stop"),
    ])
    write_log(log / "repair-continuation-00110.jsonl", [
        rec(6, [{"role": "user", "content": "gate"}], [read_a()]),
        rec(7, [tool("BBB" * 100)], [], finish="stop"),                   # the same text as the wrap-up's read: a note, only when that read came first
    ])
    out = replay.replay_run(run)
    assert out["steps"] == 1 and out["read_calls"] == 3 and out["read_notes"] == 1


def test_a_continuation_of_another_phase_does_not_join_the_step_before_it(tmp_path):
    run = tmp_path / "run"
    log = run / "stages" / "2" / "log"
    write_log(log / "prepare-00045.jsonl", [rec(1, [SYSTEM, USER], [read_a()]), rec(2, [tool("AAA" * 100)], [], finish="stop")])
    write_log(log / "finish-continuation-00059.jsonl", [rec(3, [{"role": "user", "content": "gate"}], [read_a()]), rec(4, [tool("AAA" * 100)], [], finish="stop")])
    out = replay.replay_run(run)
    assert out["steps"] == 2 and out["read_notes"] == 0


def test_an_unreadable_log_is_skipped_and_the_rest_of_the_group_is_still_replayed(tmp_path):
    run = make_run(tmp_path)
    (run / "stages" / "2" / "log" / "prepare-wrapup-00070.jsonl").write_text("{not json\n", encoding="utf-8")
    out = replay.replay_run(run)
    assert out["steps"] == 1 and out["read_calls"] == 2 and out["read_notes"] == 1


def test_a_restart_inside_a_continuation_clears_the_group_caches(tmp_path):
    run = make_run(tmp_path)
    write_log(run / "stages" / "2" / "log" / "prepare-wrapup-00070.jsonl", [
        rec(5, [SYSTEM, {"role": "user", "content": "restart"}], [read_a()]),
        rec(6, [tool("AAA" * 100)], [], finish="stop"),
    ])
    out = replay.replay_run(run)
    assert out["read_calls"] == 3 and out["read_notes"] == 1     # the post-restart read is not a note


# ======================================================================================================================================
# Round 2 of the review (Copilot on head 3ac5b8dd, two findings in code that round 1 left unchanged)
#
# 6. A dedupe hint says "its text is already in your context, do not read it again". A context restart drops that conversation, but
#    `_restart_fresh` carried every watchdog nudge into the replacement prompt, so the agent was told a file was present when it was not.
# 7. scripts/dedupe_replay.py stripped every `stages/<n>/` prefix, so `stages/1/evidence/a.txt` and `stages/2/evidence/a.txt` were one file.
# ======================================================================================================================================

import pytest                                                                   # noqa: E402

from orchard import runaway                                                      # noqa: E402
from orchard.defaults import DEDUPE_HINT_PREFIX                                  # noqa: E402
from orchard.ledger import Ledger                                                # noqa: E402
from fake_model import FakeModel, call as fcall, final as ffinal, truncated      # noqa: E402
from test_tool_dedupe import run, put, step                                      # noqa: E402,F401  (the fixture and the AgentStep builder)
from test_agent_runaway_round3 import LOOP, by_request                           # noqa: E402


def real_hint() -> str:
    """The nudge text the detector really sends."""
    d = DedupeStreak()
    found = [d.feed(result(n, True)) for n in (1, 2, 3)]
    return found[-1].nudge


def restart_prompt(fm) -> str:
    """The first user message of the replacement conversation: the request that follows the restart carries it."""
    for req in fm.requests:
        users = [m["content"] for m in req["messages"] if m["role"] == "user"]
        if len(req["messages"]) == 2 and users and "was restarted" in users[0]:
            return users[0]
    raise AssertionError("no request carried the restart prompt")


def run_hint_then_capped_replies(run, hint: str):
    put(run, "evidence/a.txt", "file text")
    answers = [fcall("read_file", path="evidence/a.txt"), truncated(16384, content=LOOP), truncated(16384, content=LOOP),
               fcall("read_file", path="evidence/a.txt"), ffinal("done")]
    with Ledger(run / "ledger.jsonl") as led, FakeModel(by_request(*answers)) as fm:
        s, _, control = step(run, led, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1)
        control.nudges.append(hint)                       # the ladder delivered the hint before the first model turn
        out = s.run("system", "task")
    return out, fm


def test_the_hint_the_detector_sends_starts_with_the_marker():
    assert real_hint().startswith(DEDUPE_HINT_PREFIX) and "Do not read it again" in real_hint()


def test_a_dedupe_hint_is_not_carried_into_the_restart_prompt(run):
    out, fm = run_hint_then_capped_replies(run, real_hint())
    assert out.status == "done"
    prompt = restart_prompt(fm)
    assert "Do not read it again" not in prompt and DEDUPE_HINT_PREFIX not in prompt and "already in your context" not in prompt


def test_the_hint_reached_the_first_conversation_before_the_restart(run):
    out, fm = run_hint_then_capped_replies(run, real_hint())
    first = [m["content"] for m in fm.requests[0]["messages"] if m["role"] == "user"]
    assert any(DEDUPE_HINT_PREFIX in t for t in first)           # it was sent; only the carrying across a restart is dropped


def test_other_nudges_are_still_carried_into_the_restart_prompt(run):
    out, fm = run_hint_then_capped_replies(run, "You have not written a file for 20 turns. Write the port now.")
    assert "Write the port now." in restart_prompt(fm)


def test_the_restart_prompt_allows_reads_again(run):
    out, fm = run_hint_then_capped_replies(run, real_hint())
    prompt = restart_prompt(fm).lower()
    assert "read a file again" in prompt and "no longer in this conversation" in prompt


def test_a_note_given_before_a_restart_is_not_given_after_it(run):
    put(run, "evidence/a.txt", "file text")
    answers = [fcall("read_file", path="evidence/a.txt"), fcall("read_file", path="evidence/a.txt"),
               truncated(16384, content=LOOP), truncated(16384, content=LOOP),
               fcall("read_file", path="evidence/a.txt"), ffinal("done")]
    with Ledger(run / "ledger.jsonl") as led, FakeModel(by_request(*answers)) as fm:
        s, _, _ = step(run, led, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1)
        out = s.run("system", "task")
    assert out.status == "done"
    before = [m["content"] for m in fm.requests[2]["messages"] if m["role"] == "tool"]
    after = [m["content"] for m in fm.requests[-1]["messages"] if m["role"] == "tool"]
    assert len(before) == 2 and "unchanged since turn" in before[1]          # a note was given in the first conversation
    assert after == ["file text"]                                           # the new conversation holds the text itself, not a note about turn 1


# ---- 7. the stage component of a path -------------------------------------------------------------------------------------------------

def read_of(path):
    return call("read_file", json.dumps({"path": path}))


def write_of(path):
    return call("write_file", json.dumps({"path": path, "content": "x"}))


def one_step_run(tmp_path, stage, records, name="prepare-00045.jsonl", dirname=None):
    run_dir = tmp_path / "run"
    log = run_dir / "stages" / (dirname or str(stage)) / "log"
    write_log(log / name, records)
    return run_dir


def test_the_same_file_name_in_two_stages_is_two_files(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_of("stages/1/evidence/a.txt"), read_of("stages/2/evidence/a.txt")]),
        rec(2, [tool("SAME" * 100), tool("SAME" * 100)], [], finish="stop"),
    ]
    out = replay.replay_run(one_step_run(tmp_path, 2, records))
    assert out["read_calls"] == 2 and out["read_notes"] == 0


def test_a_write_in_one_stage_does_not_end_the_note_for_the_same_name_in_another(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_of("stages/1/evidence/a.txt")]),
        rec(2, [tool("AAA" * 100)], [write_of("stages/2/evidence/a.txt")]),
        rec(3, [tool("wrote")], [read_of("stages/1/evidence/a.txt")]),
        rec(4, [tool("AAA" * 100)], [], finish="stop"),
    ]
    out = replay.replay_run(one_step_run(tmp_path, 2, records))
    assert out["read_calls"] == 2 and out["read_notes"] == 1


def test_a_stage_relative_path_is_the_file_of_the_steps_own_stage(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_of("evidence/a.txt"), read_of("stages/2/evidence/a.txt")]),
        rec(2, [tool("SAME" * 100), tool("SAME" * 100)], [], finish="stop"),
    ]
    out = replay.replay_run(one_step_run(tmp_path, 2, records))
    assert out["read_notes"] == 1                                          # the alias and the explicit path are one file in stage 2


def test_the_same_alias_in_a_step_of_another_stage_is_another_file(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_of("evidence/a.txt"), read_of("stages/2/evidence/a.txt")]),
        rec(2, [tool("SAME" * 100), tool("SAME" * 100)], [], finish="stop"),
    ]
    out = replay.replay_run(one_step_run(tmp_path, 3, records))
    assert out["read_notes"] == 0                                          # in stage 3 `evidence/a.txt` is stages/3/evidence/a.txt


def test_a_write_by_stage_relative_path_ends_the_note_for_that_file_in_that_stage_only(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_of("stages/2/evidence/a.txt"), read_of("stages/1/evidence/a.txt")]),
        rec(2, [tool("AAA" * 100), tool("AAA" * 100)], [write_of("evidence/a.txt")], ),
        rec(3, [tool("wrote")], [read_of("stages/2/evidence/a.txt"), read_of("stages/1/evidence/a.txt")]),
        rec(4, [tool("AAA" * 100), tool("AAA" * 100)], [], finish="stop"),     # the rewrite left the same text: a write still ends the note
    ]
    out = replay.replay_run(one_step_run(tmp_path, 2, records))
    assert out["read_calls"] == 4 and out["read_notes"] == 1               # stage 1's copy is unchanged (a note), stage 2's was rewritten (not)


def test_a_path_that_exists_in_the_run_directory_is_the_run_relative_file(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "research").mkdir(parents=True)
    (run_dir / "research" / "x.json").write_text("{}")
    write_log(run_dir / "stages" / "2" / "log" / "prepare-00045.jsonl", [
        rec(1, [SYSTEM, USER], [read_of("research/x.json"), read_of("stages/2/research/x.json")]),
        rec(2, [tool("SAME" * 100), tool("SAME" * 100)], [], finish="stop"),
    ])
    out = replay.replay_run(run_dir)
    assert out["read_notes"] == 0                                          # the live Tools read the run directory first: research/x.json is run-relative


def test_the_stage_of_a_step_moved_aside_is_still_its_stage(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_of("evidence/a.txt"), read_of("stages/2/evidence/a.txt")]),
        rec(2, [tool("SAME" * 100), tool("SAME" * 100)], [], finish="stop"),
    ]
    out = replay.replay_run(one_step_run(tmp_path, 2, records, dirname="2.partial-1"))
    assert out["read_notes"] == 1                                          # the agent saw the directory as stages/2 while it was running


def test_offsets_still_separate_pages_of_one_file(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [call("read_file", json.dumps({"path": "evidence/a.txt", "offset": 0})),
                                call("read_file", json.dumps({"path": "evidence/a.txt", "offset": 4000}))]),
        rec(2, [tool("SAME" * 100), tool("SAME" * 100)], [], finish="stop"),
    ]
    out = replay.replay_run(one_step_run(tmp_path, 2, records))
    assert out["read_notes"] == 0


def test_an_absolute_path_inside_the_run_directory_is_the_run_relative_file(tmp_path):
    run_dir = tmp_path / "run"
    absolute = str(run_dir / "stages" / "2" / "evidence" / "a.txt")
    records = [
        rec(1, [SYSTEM, USER], [read_of(absolute), read_of("stages/2/evidence/a.txt")]),
        rec(2, [tool("SAME" * 100), tool("SAME" * 100)], [], finish="stop"),
    ]
    out = replay.replay_run(one_step_run(tmp_path, 2, records))
    assert out["read_notes"] == 1


def test_a_write_by_explicit_stages_path_ends_the_note_for_that_file(tmp_path):
    records = [
        rec(1, [SYSTEM, USER], [read_of("stages/2/evidence/a.txt")]),
        rec(2, [tool("AAA" * 100)], [write_of("stages/2/evidence/a.txt")]),
        rec(3, [tool("wrote")], [read_of("stages/2/evidence/a.txt")]),
        rec(4, [tool("AAA" * 100)], [], finish="stop"),                    # the same text again, but the file was written in between
    ]
    out = replay.replay_run(one_step_run(tmp_path, 2, records))
    assert out["read_calls"] == 2 and out["read_notes"] == 0
