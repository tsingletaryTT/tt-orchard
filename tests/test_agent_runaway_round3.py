# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 462, round 3 (issue 458): recovery from capped replies must keep the supervisor's instructions, record the replacement
conversation, and catch a capped reply whose tool call has truncated arguments.

Each test uses the fake coder of tests/fake_model.py; no server, no hardware."""
import json

import pytest

from fake_model import FakeModel, call, final, truncated
from orchard import runaway
from orchard.agent import AgentStep, Tools, agent_env
from orchard.ledger import Ledger

LOOP = ("Now I see the issue clearly. The model dictionary opens at position 5350 and never closes.\n"
        "- Position 17628: } (closes something, but not the model dictionary)\n") * 120


class Control:
    def __init__(self):
        self.nudges, self.stop = [], None

    def nudge(self, agent, message):
        self.nudges.append(message)

    def escalate(self, agent, stage):
        self.stop = "escalate"

    def pause(self, reason, evidence):
        self.stop = "pause"

    def take_nudges(self, agent):
        out, self.nudges = self.nudges, []
        return out

    def stop_reason(self):
        return self.stop


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run" / "stages" / "0" / "evidence").mkdir(parents=True)
    (tmp_path / "run" / "stages" / "0" / "port").mkdir(parents=True)
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield tmp_path / "run", led


def step(run, ledger, endpoint, control=None, **kw):
    sd = run / "stages" / "0"
    return AgentStep(agent="stage-agent", endpoint=endpoint, model="fake",
                     tools=Tools(run, sd, agent_env(run)), ledger=ledger, stage=0, phase="run",
                     feed=lambda e: None, control=control or Control(), run_dir=run,
                     evidence_dir=sd / "evidence", log_path=sd / "log" / "run-1.jsonl",
                     clock=lambda: 1000.0, **kw)


def by_request(*answers):
    seen = []

    def script(request):
        seen.append(1)
        return answers[min(len(seen), len(answers)) - 1]
    return script


def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def cut_call(name="write_file", call_id="cut1", tokens=16384, **extra):
    """A reply cut off at max_tokens in the middle of a tool call's JSON arguments."""
    args = '{"path": "port/port_model.py", "content": "def f():\\n    return 1\\n    retu'
    msg = {"role": "assistant", "content": "", "finish_reason": "length", "completion_tokens": tokens,
           "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]}
    msg.update(extra)
    return msg


def two_calls(second_complete: bool):
    first = {"id": "ok1", "type": "function", "function": {"name": "shell", "arguments": json.dumps({"command": "echo kept"})}}
    cut = '{"path": "port/x.py", "content": "abc'
    second = {"id": "cut2", "type": "function", "function": {"name": "write_file", "arguments": cut}}
    return {"role": "assistant", "content": "", "finish_reason": "length", "completion_tokens": 16384, "tool_calls": [first, second]}


# ---- finding: continuation instructions survive a restart ----

def test_a_restart_inside_a_gate_feedback_continuation_keeps_the_feedback(run, tmp_path):
    run_dir, ledger = run
    answers = [final("first pass done"),                                        # the step ends
               truncated(16384, content=LOOP), truncated(16384, content=LOOP),     # the continuation loops twice: restart
               call("shell", command="echo fixing"), final("fixed")]
    with FakeModel(by_request(*answers)) as fm:
        s = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1)
        s.run("system text", "task text")
        out = s.continue_with("GATE FEEDBACK: result.json is missing the field `chips`; write it.", max_turns=6,
                              log_path=tmp_path / "continue.jsonl")
    assert out.status == "done"
    fresh = fm.requests[3]["messages"]                                          # the first request after the restart (0 run, 1 and 2 capped)
    assert [m["role"] for m in fresh] == ["system", "user"]
    assert "task text" in fresh[1]["content"]
    assert "GATE FEEDBACK: result.json is missing the field `chips`" in fresh[1]["content"], "the continuation's instruction was dropped"


def test_a_watchdog_nudge_before_a_restart_is_carried_into_the_fresh_context(run):
    run_dir, ledger = run
    control = Control()
    control.nudges.append("SUPERVISOR NUDGE: write the files now and stop investigating.")
    answers = [truncated(16384, content=LOOP), truncated(16384, content=LOOP), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, control=control, unstick=runaway.Unstick(), restart_on_caps=1).run("s", "task text")
    assert out.status == "done"
    fresh = fm.requests[2]["messages"]
    assert "SUPERVISOR NUDGE: write the files now" in fresh[1]["content"]


def test_the_carried_instructions_are_bounded(run, tmp_path):
    run_dir, ledger = run
    answers = [final("a"), final("b"), final("c"), final("d"), final("e"), final("f"),
               truncated(16384, content=LOOP), truncated(16384, content=LOOP), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        s = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1)
        s.run("s", "task text")
        for i in range(5):
            s.continue_with(f"FEEDBACK {i} " + "z" * 100_000, max_turns=2, log_path=tmp_path / f"c{i}.jsonl")
        s.continue_with("LAST " + "y" * 100_000, max_turns=4, log_path=tmp_path / "last.jsonl")
    fresh = fm.requests[-1]["messages"][1]["content"]
    assert "LAST" in fresh, "the active instruction must be kept"
    assert len(fresh) < 60_000, "the carried instructions are not bounded"


# ---- finding: the transcript records the replacement conversation and the sampling ----

def test_the_row_after_a_restart_records_the_restart_prompt_and_the_sampling(run):
    run_dir, ledger = run
    answers = [truncated(16384, content=LOOP), truncated(16384, content=LOOP), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1).run("s", "task text")
    log = rows(run_dir / "stages" / "0" / "log" / "run-1.jsonl")
    last = log[-1]
    assert last["sent"], "the row after a restart replays as an empty prompt"
    assert any("restarted" in str(m.get("content", "")) for m in last["sent"]), last["sent"]
    assert last["request"]["temperature"] == 0.5 and last["request"]["max_tokens"] == 8192, "the sampling profile of the request is not recorded"
    assert log[0]["request"]["temperature"] == 0, "an ordinary request records its own sampling"


# ---- finding: a capped reply with truncated tool arguments goes through recovery ----

def test_a_capped_reply_with_truncated_arguments_is_counted_and_arms_the_retry_profile(run):
    run_dir, ledger = run
    answers = [cut_call(), final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "done"
    notices = [e["data"] for e in ledger.read() if e["event"] == "notice" and e["data"].get("reply_capped")]
    assert len(notices) == 1 and notices[0]["capped_in_step"] == 1 and notices[0]["incomplete_tool_calls"] == 1
    assert fm.requests[1]["temperature"] == 0.5 and fm.requests[1]["max_tokens"] == 8192, "the retry after the cap used the plain request"
    assert not (run_dir / "stages" / "0" / "port" / "port_model.py").exists(), "a call with truncated arguments must not run"


def test_a_truncated_call_is_not_put_into_the_conversation(run):
    run_dir, ledger = run
    with FakeModel(by_request(cut_call(), final("ok"))) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    sent = fm.requests[1]["messages"]
    assert not any(m.get("tool_calls") for m in sent), "an assistant message with an unanswered tool call was kept"
    assert not any(m.get("role") == "tool" for m in sent)


def test_two_capped_replies_with_truncated_arguments_restart_the_step(run):
    run_dir, ledger = run
    with FakeModel(by_request(cut_call(call_id="a"), cut_call(call_id="b"), final("ok"))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1).run("system text", "task text")
    assert out.status == "done"
    assert [m["role"] for m in fm.requests[2]["messages"]] == ["system", "user"] and "restarted" in fm.requests[2]["messages"][1]["content"]


def test_two_capped_replies_with_truncated_arguments_end_a_step_that_may_not_restart(run):
    run_dir, ledger = run
    with FakeModel(by_request(cut_call(call_id="a"), cut_call(call_id="b"), final("never"))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "error" and len(fm.requests) == 2
    assert "cut off at max_tokens twice in a row" in out.detail


def test_a_complete_call_in_a_capped_reply_still_runs_and_the_cap_is_still_counted(run):
    run_dir, ledger = run
    with FakeModel(by_request(two_calls(True), final("ok"))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "done"
    second = fm.requests[1]["messages"]
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert len(tool_msgs) == 1 and "kept" in tool_msgs[0]["content"] and tool_msgs[0]["tool_call_id"] == "ok1"
    calls = [c for m in second if m.get("tool_calls") for c in m["tool_calls"]]
    assert [c["id"] for c in calls] == ["ok1"], "only the complete call may stay in the conversation"
    assert fm.requests[1]["temperature"] == 0.5, "the cap was not counted when one call in the reply was complete"
    assert sum(1 for e in ledger.read() if e["event"] == "notice" and e["data"].get("reply_capped")) == 1


def test_a_complete_call_in_a_reply_that_ended_normally_is_untouched(run):
    run_dir, ledger = run
    with FakeModel(by_request(call("shell", command="echo plain"), final("ok"))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "done" and fm.requests[1]["temperature"] == 0
    assert not any(e["data"].get("reply_capped") for e in ledger.read() if e["event"] == "notice")


def test_a_looping_truncated_argument_is_named_as_a_repetition_loop(run):
    run_dir, ledger = run
    looped = cut_call()
    looped["tool_calls"][0]["function"]["arguments"] = '{"path": "port/p.py", "content": "' + ("x = 1\\n" * 3000)
    with FakeModel(by_request(looped, final("ok"))) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    notice = [e["data"] for e in ledger.read() if e["event"] == "notice" and e["data"].get("reply_capped")][0]
    assert notice["repetition"] is True


def test_the_narrator_says_a_tool_call_was_cut_off_and_not_run():
    from orchard import narrate
    rec = {"turn": 1, "sent": [], "finish_reason": "length", "received": cut_call()}
    lines = narrate.TurnNarrator().lines_for(rec, actor="stage-agent")
    text = " | ".join(getattr(l, "text", str(l)) for l in lines)
    assert "cut off" in text and "not run" in text and "writes ?" not in text


def test_the_narrator_still_narrates_a_complete_call_in_a_capped_reply():
    from orchard import narrate
    msg = two_calls(True)
    rec = {"turn": 1, "sent": [], "finish_reason": "length", "received": msg}
    text = " | ".join(getattr(l, "text", str(l)) for l in narrate.TurnNarrator().lines_for(rec, actor="a"))
    assert "echo kept" in text and "cut off" in text


def test_a_continuation_outlives_newer_nudges_when_the_carried_list_is_full(run):
    run_dir, ledger = run
    s = step(run_dir, ledger, "http://127.0.0.1:9", control=Control(), unstick=runaway.Unstick(), restart_on_caps=1)
    s._carry_instruction("continuation", "CONTINUATION: change the result file")
    for i in range(5):
        s._carry_instruction("nudge", f"NUDGE {i}")
    kinds = [k for k, _ in s._carry]
    assert len(kinds) == 4 and "continuation" in kinds, kinds
    assert [x for k, x in s._carry if k == "nudge"] == ["NUDGE 2", "NUDGE 3", "NUDGE 4"], "the oldest nudges must go first"


def test_a_second_restart_does_not_nest_the_first_restarts_text(run):
    run_dir, ledger = run
    answers = [truncated(16384, content=LOOP)] * 2 + [truncated(16384, content=LOOP)] * 2 + [final("ok")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=2).run("s", "task text")
    assert out.status == "done"
    second = fm.requests[4]["messages"][1]["content"]
    assert second.count("The conversation was restarted") == 1, "the first restart's note was copied into the second restart"
    assert second.startswith("task text")


def test_a_refused_retry_profile_is_recorded_as_the_plain_request_that_was_served(run):
    run_dir, ledger = run
    seen = []

    def script(request):
        seen.append(request)
        if "frequency_penalty" in request:
            return 400                                                       # the server does not take the profile
        return truncated(16384, content=LOOP) if len(seen) == 1 else final("ok")
    with FakeModel(script) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "done"
    last = rows(run_dir / "stages" / "0" / "log" / "run-1.jsonl")[-1]
    assert "frequency_penalty" not in last["request"] and last["request"]["temperature"] == 0, last["request"]


def test_a_reply_that_ended_normally_with_bad_arguments_still_reaches_the_tool_for_its_error(run):
    run_dir, ledger = run
    bad = {"role": "assistant", "content": "", "tool_calls": [{"id": "b1", "type": "function",
                                                                 "function": {"name": "shell", "arguments": "{not json"}}]}
    with FakeModel(by_request(bad, final("ok"))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "done"
    tool = [m for m in fm.requests[1]["messages"] if m.get("role") == "tool"]
    assert len(tool) == 1 and tool[0]["tool_call_id"] == "b1", "a call that ended normally must get its answer from the tool"


def test_the_transcript_shows_the_truncated_call_the_coder_sent(run):
    run_dir, ledger = run
    with FakeModel(by_request(two_calls(True), final("ok"))) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    first = rows(run_dir / "stages" / "0" / "log" / "run-1.jsonl")[0]
    ids = [c["id"] for c in first["received"].get("tool_calls", [])]
    assert ids == ["ok1", "cut2"], "the log must show what the coder sent, including the call that was cut off"
