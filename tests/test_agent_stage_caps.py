# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 458, item 4: capped replies counted for the STAGE. The step's own rule (two capped replies in a row restart the context once) never
fired in tiny Qwen4 run 22, where capped replies were separated by complete calls: twelve capped replies in stage 2, no two in a row, and the
stage ended 'no final answer after 60 turns', which names nothing. A count over the stage restarts the context once and then stops the run with
a block that names the coder's replies."""
import pytest

from fake_model import FakeModel, call, final, truncated
from orchard import blocked, narrate, runaway
from test_agent_runaway import LOOP, by_request, run, step  # noqa: F401  (run is a fixture)

CAP = truncated(16384, content=LOOP)


def note_caps(ledger, stage, n, *, loops=0):
    for i in range(n):
        ledger.append("notice", stage, watchdog=True, what="reply cut off at max_tokens", reply_capped=True, capped_in_step=i + 1,
                      repetition=i < loops)


def spaced(n, tail="done"):
    """n capped replies, each followed by one complete call, so no two capped replies are in a row; then a final answer."""
    out = []
    for i in range(n):
        out += [CAP, call("shell", command=f"echo {i}")]
    return out + [final(tail)]


def decisions(ledger, name):
    return [e["data"] for e in ledger.read() if e["event"] == "decision" and e["data"].get("decision") == name]


RESTARTED = "context restarted after capped replies"


# ---- the restart ---------------------------------------------------------------------------------------------------------------------

def test_capped_replies_that_are_never_in_a_row_restart_the_context_at_the_stage_count(run):
    run_dir, ledger = run
    with FakeModel(by_request(*spaced(4))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1, stage_restart_after=3, stage_block_after=9).run("s", "u")
    assert out.status == "done"
    stage = [d for d in decisions(ledger, RESTARTED) if d.get("trigger") == "stage"]
    assert len(stage) == 1 and stage[0]["capped_in_stage"] == 3
    assert not [d for d in decisions(ledger, RESTARTED) if d.get("trigger") != "stage"]          # no two in a row: the old rule did not fire
    after = fm.requests[stage_request_index(fm)]["messages"]
    assert [m["role"] for m in after] == ["system", "user"] and "restarted" in after[1]["content"]


def stage_request_index(fm):
    """The index of the first request whose conversation was reset to a system and a user message after the first one."""
    for i, r in enumerate(fm.requests):
        if i and [m["role"] for m in r["messages"]] == ["system", "user"]:
            return i
    raise AssertionError("no request after a reset")


def test_the_stage_restart_happens_once(run):
    run_dir, ledger = run
    with FakeModel(by_request(*spaced(6))) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1, stage_restart_after=2, stage_block_after=99).run("s", "u")
    assert len([d for d in decisions(ledger, RESTARTED) if d.get("trigger") == "stage"]) == 1


def test_the_stage_count_includes_earlier_steps_and_attempts_of_the_stage(run):
    run_dir, ledger = run
    note_caps(ledger, 0, 2)
    with FakeModel(by_request(*spaced(2))) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1, stage_restart_after=3, stage_block_after=99).run("s", "u")
    stage = [d for d in decisions(ledger, RESTARTED) if d.get("trigger") == "stage"]
    assert len(stage) == 1 and stage[0]["capped_in_stage"] == 3


# ---- what is counted -----------------------------------------------------------------------------------------------------------------

def test_capped_replies_before_a_resume_do_not_count(run):
    run_dir, ledger = run
    note_caps(ledger, 0, 6)
    ledger.append("decision", 0, decision="resume", by="operator")
    with FakeModel(by_request(*spaced(2))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1, stage_restart_after=5, stage_block_after=5).run("s", "u")
    assert out.status == "done" and not decisions(ledger, RESTARTED)


def test_capped_replies_of_another_stage_do_not_count(run):
    run_dir, ledger = run
    note_caps(ledger, 1, 9)
    with FakeModel(by_request(*spaced(2))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1, stage_restart_after=5, stage_block_after=5).run("s", "u")
    assert out.status == "done" and not decisions(ledger, RESTARTED)


def test_a_restart_before_a_resume_does_not_use_up_the_stage_restart_after_it(run):
    run_dir, ledger = run
    ledger.append("decision", 0, decision=RESTARTED, trigger="stage", capped_in_stage=4)
    ledger.append("decision", 0, decision="resume", by="operator")
    with FakeModel(by_request(*spaced(3))) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1, stage_restart_after=3, stage_block_after=99).run("s", "u")
    assert len([d for d in decisions(ledger, RESTARTED) if d.get("trigger") == "stage"]) == 2


# ---- the end of the step -------------------------------------------------------------------------------------------------------------

def test_the_stage_cap_ends_the_step_with_a_text_that_names_the_coders_replies(run):
    run_dir, ledger = run
    note_caps(ledger, 0, 3, loops=2)
    with FakeModel(by_request(*spaced(5))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1, stage_restart_after=4, stage_block_after=5).run("s", "u")
    assert out.status == "error"
    assert out.detail.startswith(blocked.REPLIES_CAPPED_MARK)
    assert "5 replies in this stage were cut off at max_tokens" in out.detail
    assert "repetition loops" in out.detail and "sampling" in out.detail and "max_tokens" in out.detail and "tier" in out.detail
    assert "The port may be fine" in out.detail


def test_the_text_counts_the_repetition_loops_of_the_stage(run):
    run_dir, ledger = run
    note_caps(ledger, 0, 4, loops=3)
    with FakeModel(by_request(*spaced(1))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1, stage_restart_after=99, stage_block_after=5).run("s", "u")
    assert out.status == "error" and "4 of them repetition loops" in out.detail          # three earlier loops and this reply, which is a loop too


def test_the_stage_rules_are_off_when_their_counts_are_zero(run):
    run_dir, ledger = run
    note_caps(ledger, 0, 30)
    with FakeModel(by_request(*spaced(3))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1).run("s", "u")
    assert out.status == "done" and not decisions(ledger, RESTARTED)


# ---- the block code ------------------------------------------------------------------------------------------------------------------

def test_the_block_code_exists_is_not_retried_by_itself_and_has_an_unblock_text():
    assert "replies-capped" in blocked.REASONS and "replies-capped" not in blocked.AUTO_RETRY
    assert narrate.UNBLOCK["replies-capped"] and "sampling" in " ".join(narrate.UNBLOCK["replies-capped"])


def test_a_pause_with_the_marked_text_has_the_new_code_even_when_it_names_a_chip_and_a_coder():
    text = blocked.REPLIES_CAPPED_MARK + ": 8 replies in this stage were cut off ... the coder's chip tier ... disk ..."
    assert blocked.block_code({"reason": "blocked: " + text}) == "replies-capped"
    assert not blocked.retryable("replies-capped", text)


# ---- the supervisor ------------------------------------------------------------------------------------------------------------------

def test_a_run_whose_coder_keeps_running_away_ends_blocked_with_the_new_code(tmp_path):
    from test_supervisor import Rig
    from orchard.supervisor import EXIT_BLOCKED
    r = Rig(tmp_path)
    try:
        r.args.unattended = True          # a block ends the run; an attended run would wait for the operator
        ordinary = r.script
        r.script = lambda request: CAP if request.get("tools") else ordinary(request)        # the canary and the probes still get their answers
        assert r.run() == EXIT_BLOCKED
        pauses = [e["data"] for e in r.entries() if e["event"] == "decision" and e["data"].get("decision") == "pause"]
        assert pauses and blocked.block_code(pauses[-1]) == "replies-capped", pauses[-1:]
        assert blocked.REPLIES_CAPPED_MARK in pauses[-1]["reason"]
    finally:
        r.close()
