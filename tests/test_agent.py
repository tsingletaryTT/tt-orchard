"""The agent step loop against a scripted fake model endpoint."""
import pytest

from fake_model import FakeModel, call, final, turn
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


def test_the_request_is_greedy_non_streaming_and_offers_both_tools(run):
    run_dir, ledger = run
    with FakeModel(lambda r: final()) as fm:
        step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    req = fm.requests[0]
    assert req["temperature"] == 0 and req["stream"] is False and req["model"] == "fake"
    assert [t["function"]["name"] for t in req["tools"]] == ["shell", "write_file"]
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
