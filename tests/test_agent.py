"""The agent step loop against a scripted fake model endpoint."""
import pytest

from fake_model import FakeModel, call, empty, final, truncated, turn
from orchard.agent import AgentStep, Tools, agent_env, probe_model
from orchard.ledger import Ledger
from orchard.watchdog import Event, IdenticalResponses, Ladder, StageOverBudget, Watchdog


class Control:
    """The supervisor side of a step: the watchdog's actuator plus the stop question."""

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
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield tmp_path / "run", led


def step(run, ledger, endpoint, *, control=None, feed=None, clock=None, **kw):
    sd = run / "stages" / "0"
    control = control or Control()
    events = []
    return AgentStep(agent="stage-agent", endpoint=endpoint, model="fake",
                     tools=Tools(run, sd, agent_env(run)), ledger=ledger, stage=0, phase="run",
                     feed=feed or events.append, control=control, run_dir=run,
                     evidence_dir=sd / "evidence", log_path=sd / "log" / "run-1.jsonl",
                     clock=clock or (lambda: 1000.0), **kw), events, control


SCRIPT = [call("write_file", path="evidence/a.txt", content="hello"),
          call("shell", command="cat stages/0/evidence/a.txt"), final("stage done")]


def test_a_step_runs_tool_calls_until_a_final_answer(run):
    run_dir, ledger = run
    with FakeModel(lambda r: SCRIPT[turn(r)]) as fm:
        s, events, _ = step(run_dir, ledger, fm.endpoint)
        out = s.run("system text", "user text")
    assert (out.status, out.turns, out.final_text) == ("done", 3, "stage done")
    assert fm.requests[2]["messages"][-1] == {"role": "tool", "tool_call_id": "c1",
                                              "content": "exit 0\nhello"}
    ev = [e["data"] for e in ledger.read() if e["event"] == "evidence"]
    assert ev[0]["path"] == "stages/0/evidence/a.txt" and len(ev[0]["sha256"]) == 64
    assert ev[-1]["what"] == "transcript" and ev[-1]["status"] == "done"
    kinds = [e.kind for e in events]
    assert kinds.count("response") == 3 and kinds.count("tool_call") == 2
    assert kinds.count("tool_result") == 2 and kinds.count("evidence") == 1


def test_the_request_is_greedy_non_streaming_and_offers_all_three_tools(run):
    run_dir, ledger = run
    with FakeModel(lambda r: final()) as fm:
        step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    req = fm.requests[0]
    assert req["temperature"] == 0 and req["stream"] is False and req["model"] == "fake"
    assert [t["function"]["name"] for t in req["tools"]] == ["shell", "read_file", "write_file"]
    assert req["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def repeating(stop_after):
    """The same visible text every turn; the tool arguments change, so only the text repeats."""
    def script(request):
        n = turn(request)
        if n >= stop_after:
            return final("gave up")
        msg = call("shell", command=f"echo {n}")
        msg["content"] = "let me look again"
        return msg
    return script


def test_a_nudge_from_the_ladder_is_added_to_the_next_request(run):
    run_dir, ledger = run
    control = Control()
    wd = Watchdog([IdenticalResponses()], Ladder(control, ledger, {"stage-agent"}))
    with FakeModel(repeating(4)) as fm:
        out = step(run_dir, ledger, fm.endpoint, control=control, feed=wd.feed)[0].run("s", "u")
    assert out.status == "done"
    users = [m["content"] for m in fm.requests[3]["messages"] if m["role"] == "user"]
    assert len(users) == 2 and "repeating itself" in users[1]
    assert not any("repeating itself" in m.get("content", "") for m in fm.requests[2]["messages"])


def test_a_second_loop_escalates_and_stops_the_step(run):
    run_dir, ledger = run
    control = Control()
    wd = Watchdog([IdenticalResponses()], Ladder(control, ledger, {"stage-agent"}))
    with FakeModel(repeating(50)) as fm:
        out = step(run_dir, ledger, fm.endpoint, control=control, feed=wd.feed)[0].run("s", "u")
    assert out.status == "escalate" and out.turns == 6
    assert [e["event"] for e in ledger.read() if e["data"].get("watchdog")] == ["retry", "escalate"]


def test_the_stage_budget_pause_stops_the_step(run):
    run_dir, ledger = run
    control = Control()
    wd = Watchdog([StageOverBudget({0: 60})], Ladder(control, ledger, {"stage-agent"}))
    wd.feed(Event(ts=0, agent="supervisor", kind="ledger", name="stage_start", stage=0))
    with FakeModel(repeating(50)) as fm:
        out = step(run_dir, ledger, fm.endpoint, control=control, feed=wd.feed,
                   clock=lambda: 100.0)[0].run("s", "u")
    assert out.status == "pause" and len(fm.requests) == 1


def test_a_failed_request_is_retried_once_and_no_more(run):
    run_dir, ledger = run
    with FakeModel(lambda r: 500) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert out.status == "error" and len(fm.requests) == 2
    assert [e["data"]["attempt"] for e in ledger.read() if e["event"] == "retry"] == [1, 2]


def test_the_turn_limit_ends_the_step(run):
    run_dir, ledger = run
    with FakeModel(repeating(50)) as fm:
        out = step(run_dir, ledger, fm.endpoint, max_turns=2)[0].run("s", "u")
    assert out.status == "turns" and len(fm.requests) == 2


def test_probe_model_reads_the_model_list():
    with FakeModel(lambda r: final(), models=["Qwen/Qwen3.8-27B"]) as fm:
        assert probe_model(fm.endpoint, "Qwen/Qwen3.8-27B") is True
        assert probe_model(fm.endpoint, "other") is False
    assert probe_model("http://127.0.0.1:9/v1", "x", timeout=1) is False


# ---- replies cut off at max_tokens (finish_reason "length") -------------------------------------

NUDGE = "Your last reply was cut off before you ran any command. Think briefly, then call a tool."


def by_request(*answers):
    """One answer per request, in order, whatever the conversation holds."""
    seen = []
    def script(request):
        seen.append(1)
        return answers[len(seen) - 1]
    return script


def test_a_truncated_reply_is_nudged_not_treated_as_done(run):
    run_dir, ledger = run
    script = by_request(truncated(), call("shell", command="echo hi"), final("all done"))
    with FakeModel(script) as fm:
        s, events, _ = step(run_dir, ledger, fm.endpoint)
        out = s.run("s", "u")
    assert (out.status, out.turns, out.final_text) == ("done", 3, "all done")
    # The empty reply is not added to the conversation; the nudge is.
    second = fm.requests[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "user"]
    assert second[-1]["content"] == NUDGE
    assert all(m["role"] != "assistant" for m in second)
    # Later requests keep the nudge in the conversation.
    assert NUDGE in [m["content"] for m in fm.requests[2]["messages"] if m["role"] == "user"]


def test_a_truncated_reply_is_logged_with_its_finish_reason(run):
    import json
    run_dir, ledger = run
    with FakeModel(by_request(truncated(), final("ok"))) as fm:
        step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    records = [json.loads(l) for l in (run_dir / "stages" / "0" / "log" / "run-1.jsonl").read_text().splitlines()]
    assert [r["finish_reason"] for r in records] == ["length", "stop"]
    notices = [e["data"] for e in ledger.read() if e["event"] == "notice" and e["data"].get("watchdog")]
    assert len(notices) == 1 and notices[0]["finish_reason"] == "length"


def test_two_truncated_replies_in_a_row_end_the_step_with_an_error(run):
    run_dir, ledger = run
    with FakeModel(by_request(truncated(8192), truncated(8192), final("never"))) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert out.status == "error" and out.turns == 2 and len(fm.requests) == 2
    assert out.detail == ("the reply was cut off at max_tokens twice in a row (8192 tokens each); "
                          "no action was taken")
    assert [e["data"]["status"] for e in ledger.read()
            if e["event"] == "evidence" and e["data"].get("what") == "transcript"] == ["error"]


def test_a_successful_turn_resets_the_truncation_count(run):
    run_dir, ledger = run
    script = by_request(truncated(), call("shell", command="echo a"), truncated(),
                        call("shell", command="echo b"), final("fin"))
    with FakeModel(script) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert (out.status, out.turns) == ("done", 5)


def test_a_truncated_reply_with_complete_tool_calls_is_executed(run):
    run_dir, ledger = run
    cut = call("shell", command="echo ran")
    cut["finish_reason"] = "length"
    with FakeModel(by_request(cut, final("fin"))) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert out.status == "done" and out.turns == 2
    assert fm.requests[1]["messages"][-1]["content"] == "exit 0\nran\n"
    assert not any(m.get("content") == NUDGE for m in fm.requests[1]["messages"])


def test_a_stop_reply_without_tool_calls_is_still_done(run):
    run_dir, ledger = run
    with FakeModel(lambda r: final("all finished")) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert (out.status, out.turns, len(fm.requests)) == ("done", 1, 1)


def test_a_reply_with_no_finish_reason_keeps_the_old_behaviour(run):
    run_dir, ledger = run
    def http(url, request, timeout):
        return {"choices": [{"message": {"role": "assistant", "content": "x"}}], "usage": {}}
    out = step(run_dir, ledger, "http://unused/v1", http=http)[0].run("s", "u")
    assert (out.status, out.final_text) == ("done", "x")


# ---- empty replies (no text and no tool call, any finish_reason) --------------------------------
# The live Qwen3.8 run, stage 1, attempt 3, turn 49: 159 completion tokens of reasoning, content "",
# no tool calls, finish_reason "stop". The loop took it as the final answer and the stage failed.

EMPTY_NUDGE = ("Your last reply was empty: it had no text and no command. Say what you will do next, "
               "then call a tool, or state that the stage is finished and name the output files you "
               "wrote.")


def test_an_empty_reply_is_nudged_not_treated_as_done(run):
    run_dir, ledger = run
    script = by_request(empty(), call("shell", command="echo hi"), final("all done"))
    with FakeModel(script) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert (out.status, out.turns, out.final_text) == ("done", 3, "all done")
    second = fm.requests[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "user"]
    assert second[-1]["content"] == EMPTY_NUDGE
    notices = [e["data"] for e in ledger.read() if e["event"] == "notice" and e["data"].get("watchdog")]
    assert len(notices) == 1 and notices[0]["what"] == "empty reply"
    assert notices[0]["finish_reason"] == "stop" and notices[0]["completion_tokens"] == 159


@pytest.mark.parametrize("content", ["", "   \n\t", None])
def test_whitespace_or_null_content_counts_as_empty(run, content):
    run_dir, ledger = run
    with FakeModel(by_request(empty(content=content), final("ok"))) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert (out.status, out.turns, out.final_text) == ("done", 2, "ok")
    assert fm.requests[1]["messages"][-1]["content"] == EMPTY_NUDGE


def test_two_empty_replies_in_a_row_end_the_step_with_an_error(run):
    run_dir, ledger = run
    with FakeModel(by_request(empty(), empty(), final("never"))) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert out.status == "error" and out.turns == 2 and len(fm.requests) == 2
    assert out.detail == ("the reply was empty (no text and no command) twice in a row; "
                          "no action was taken")


@pytest.mark.parametrize("first,second,detail", [
    (truncated(8192), empty(), "two replies in a row ran no command: cut off at max_tokens, then "
                               "empty (no text and no command); no action was taken"),
    (empty(), truncated(8192), "two replies in a row ran no command: empty (no text and no command), "
                               "then cut off at max_tokens; no action was taken"),
])
def test_empty_and_truncated_replies_count_toward_the_same_limit(run, first, second, detail):
    run_dir, ledger = run
    with FakeModel(by_request(first, second, final("never"))) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert (out.status, out.turns, out.detail) == ("error", 2, detail)


def test_a_tool_call_resets_the_empty_reply_count(run):
    run_dir, ledger = run
    script = by_request(empty(), call("shell", command="echo a"), empty(), final("fin"))
    with FakeModel(script) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert (out.status, out.turns, out.final_text) == ("done", 4, "fin")


def test_a_request_asks_for_the_default_max_tokens(run):
    run_dir, ledger = run
    with FakeModel(lambda r: final()) as fm:
        step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert fm.requests[0]["max_tokens"] == 16384


def test_a_request_switches_thinking_off_by_default(run):
    run_dir, ledger = run
    with FakeModel(lambda r: final()) as fm:
        step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert fm.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_step_built_with_thinking_on_does_not_send_the_switch(run):
    run_dir, ledger = run
    with FakeModel(lambda r: final()) as fm:
        s = step(run_dir, ledger, fm.endpoint)[0]
        s.thinking = True
        s.run("s", "u")
    assert "chat_template_kwargs" not in fm.requests[0]


# ---- continuing the same conversation (gate feedback) -------------------------------------------

def test_a_continuation_adds_one_user_message_to_the_same_conversation(run):
    import json
    run_dir, ledger = run
    script = by_request(call("shell", command="echo one"), final("first done"),
                        call("shell", command="echo two"), final("fixed"))
    with FakeModel(script) as fm:
        s = step(run_dir, ledger, fm.endpoint)[0]
        first = s.run("s", "u")
        log2 = run_dir / "stages" / "0" / "log" / "run-1-continuation.jsonl"
        second = s.continue_with("fix the gate", max_turns=5, log_path=log2)
    assert (first.status, first.turns) == ("done", 2)
    assert (second.status, second.turns, second.final_text) == ("done", 2, "fixed")
    sent = fm.requests[2]["messages"]
    # The whole first conversation, then the feedback as one user message.
    assert sent[:len(fm.requests[1]["messages"])] == fm.requests[1]["messages"]
    assert sent[-2] == {"role": "assistant", "content": "first done"}
    assert sent[-1] == {"role": "user", "content": "fix the gate"}
    # The continuation has its own log; the first log keeps its two records.
    log1 = run_dir / "stages" / "0" / "log" / "run-1.jsonl"
    assert len(log1.read_text().splitlines()) == 2
    records = [json.loads(l) for l in log2.read_text().splitlines()]
    assert len(records) == 2 and records[0]["sent"] == [{"role": "user", "content": "fix the gate"}]
    transcripts = [e["data"]["path"] for e in ledger.read()
                   if e["event"] == "evidence" and e["data"].get("what") == "transcript"]
    assert transcripts == ["stages/0/log/run-1.jsonl", "stages/0/log/run-1-continuation.jsonl"]


def test_a_continuation_has_its_own_turn_cap(run):
    run_dir, ledger = run
    with FakeModel(repeating(50)) as fm:
        s = step(run_dir, ledger, fm.endpoint, max_turns=3)[0]
        out = s.run("s", "u")
        assert out.status == "turns"
        more = s.continue_with("go on", max_turns=2,
                               log_path=run_dir / "stages" / "0" / "log" / "run-1-continuation.jsonl")
    assert (more.status, more.turns) == ("turns", 2) and len(fm.requests) == 5


def test_a_step_that_never_ran_cannot_continue(run):
    run_dir, ledger = run
    s = step(run_dir, ledger, "http://unused/v1")[0]
    with pytest.raises(RuntimeError):
        s.continue_with("x", max_turns=1, log_path=run_dir / "c.jsonl")


def test_a_write_file_result_says_whether_the_write_succeeded(run):
    run_dir, ledger = run
    script = [call("write_file", path="notes.txt", content="x"),
              call("write_file", path="../../outside.txt", content="x"),
              call("shell", command="echo hi"), final("done")]
    with FakeModel(lambda r: script[turn(r)]) as fm:
        s, events, _ = step(run_dir, ledger, fm.endpoint)
        s.run("s", "u")
    results = [(e.tool, e.wrote) for e in events if e.kind == "tool_result"]
    assert results == [("write_file", True), ("write_file", False), ("shell", None)]


def test_every_event_carries_its_turn_and_a_continuation_keeps_counting(run):
    run_dir, ledger = run
    script = by_request(call("write_file", path="evidence/a.txt", content="x"), final("first done"),
                        call("shell", command="echo two"), final("fixed"))
    with FakeModel(script) as fm:
        s, events, _ = step(run_dir, ledger, fm.endpoint)
        s.run("s", "u")
        s.continue_with("fix the gate", max_turns=5,
                        log_path=run_dir / "stages" / "0" / "log" / "run-1-continuation.jsonl")
    assert [(e.kind, e.turn) for e in events] == [
        ("response", 1), ("tool_call", 1), ("tool_result", 1), ("evidence", 1),
        ("response", 2),
        ("response", 3), ("tool_call", 3), ("tool_result", 3),
        ("response", 4)]


def test_a_shell_call_event_carries_the_command_shape(run):
    # TurnRepeat's shape track reads Event.shape; the step must set it for every shell call.
    from orchard.watchdog import command_shape
    run_dir, ledger = run
    cmd = "grep -rn hello stages/0/evidence/a.txt | head -5"
    script = [call("write_file", path="evidence/a.txt", content="hello"), call("shell", command=cmd),
              final("done")]
    with FakeModel(lambda r: script[turn(r)]) as fm:
        s, events, _ = step(run_dir, ledger, fm.endpoint)
        s.run("s", "u")
    calls = [e for e in events if e.kind == "tool_call"]
    assert [(e.tool, e.shape) for e in calls] == [("write_file", None), ("shell", command_shape(cmd))]
    assert calls[1].shape == "grep stages/0/evidence/a.txt"
