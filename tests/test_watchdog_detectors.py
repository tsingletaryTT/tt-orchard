"""The detectors on synthetic event streams."""
import pytest

from orchard.adapters import AdapterError, ChipState
from orchard.watchdog import (Event, IdenticalResponses, LeaseIdle, NoFileWritten, NoNewEvidence,
                              RepeatedToolCall, StageOverBudget, ThinkingWithoutAction, replay)


def resp(ts, i=1000, o=100, think=50, text=None, tool=False, agent="a"):
    return Event(ts=ts, agent=agent, kind="response", input_tokens=i, output_tokens=o,
                 thinking_tokens=think, text_hash=text, had_tool_call=tool)


def times(findings):
    return [f.ts for f in findings]


def test_event_kind_is_checked():
    with pytest.raises(ValueError):
        Event(ts=0, agent="a", kind="chat")


def test_three_identical_responses_fire_once_and_the_detector_re_arms():
    found = replay([resp(t) for t in range(1, 7)], [IdenticalResponses()])
    assert times(found) == [3, 6]
    assert found[0].detector == "identical_responses" and found[0].evidence["count"] == 3


def test_identical_counts_with_a_tool_call_are_not_a_repeat():
    assert replay([resp(t, tool=True) for t in range(5)], [IdenticalResponses()]) == []


def test_the_same_nonempty_text_is_a_repeat_whatever_the_counts():
    evs = [resp(t, i=1000 + t, text="ab" * 32, tool=True) for t in range(3)]
    assert times(replay(evs, [IdenticalResponses()])) == [2]


def test_agents_are_counted_separately():
    evs = [resp(1, agent="a"), resp(2, agent="b"), resp(3, agent="a"), resp(4, agent="b")]
    assert replay(evs, [IdenticalResponses()]) == []


def test_identical_needs_n_of_at_least_two():
    with pytest.raises(ValueError):
        IdenticalResponses(n=1)


@pytest.mark.parametrize("think,tool,fires", [(20001, False, True), (20000, False, False),
                                              (27939, True, False), (None, False, False)])
def test_thinking_without_action(think, tool, fires):
    found = replay([resp(1, think=think, tool=tool)], [ThinkingWithoutAction()])
    assert bool(found) is fires


def test_no_new_evidence_fires_after_the_window_and_resets_on_evidence():
    d = NoNewEvidence(window_s=100)
    evs = [resp(0), resp(99), resp(100), resp(150),
           Event(ts=160, agent="a", kind="evidence", name="evidence/pcc.json"),
           resp(250), resp(260)]
    assert times(replay(evs, [d])) == [100, 260]


@pytest.mark.parametrize("name", ["measurement", "stage_end"])
def test_a_named_progress_event_from_the_supervisor_counts_for_every_agent(name):
    d = NoNewEvidence(window_s=100)
    evs = [resp(0, agent="a"), resp(0, agent="b"),
           Event(ts=90, agent="supervisor", kind="ledger", name=name),
           resp(150, agent="a"), resp(150, agent="b")]
    assert replay(evs, [d]) == []


@pytest.mark.parametrize("name", ["retry", "escalate", "notice", "decision", "tick", "stage_start"])
def test_ladder_entries_and_polls_never_reset_the_clock(name):
    d = NoNewEvidence(window_s=100)
    evs = [resp(0), Event(ts=90, agent="supervisor", kind="ledger", name=name), resp(100)]
    assert times(replay(evs, [d])) == [100]


def test_each_agent_has_its_own_clock():
    d = NoNewEvidence(window_s=100)
    evs = [resp(0, agent="a"), resp(0, agent="b"),
           Event(ts=90, agent="b", kind="evidence", name="evidence/b.json"),
           resp(100, agent="a"), resp(100, agent="b")]
    found = replay(evs, [d])
    assert [(f.agent, f.ts) for f in found] == [("a", 100)]


def call(ts, tool="read_file", args="1" * 64):
    return Event(ts=ts, agent="a", kind="tool_call", tool=tool, args_hash=args)


def result(ts, tool="read_file", out="2" * 64):
    return Event(ts=ts, agent="a", kind="tool_result", tool=tool, output_hash=out)


def test_the_same_tool_call_three_times_fires():
    assert times(replay([call(1), call(2), call(3)], [RepeatedToolCall()])) == [3]


def test_the_same_tool_with_new_arguments_is_not_a_repeat():
    assert replay([call(1, args="1" * 64), call(2, args="3" * 64), call(3)], [RepeatedToolCall()]) == []


def test_calls_and_results_are_counted_separately():
    evs = [call(1), result(2), call(3), result(4), call(5)]
    found = replay(evs, [RepeatedToolCall()])
    assert times(found) == [5] and found[0].evidence["what"] == "call"


def test_the_same_output_three_times_fires():
    found = replay([result(1), result(2), result(3)], [RepeatedToolCall()])
    assert times(found) == [3] and found[0].evidence["what"] == "output"


class Status:
    def __init__(self, state="CLAIMED"):
        self.state, self.calls = state, 0

    def __call__(self):
        self.calls += 1
        if self.state == "error":
            raise AdapterError("gozer status failed")
        return [ChipState("0000:01:00.0", self.state, "orchard:run")]


def tick(ts):
    return Event(ts=ts, agent="sup", kind="ledger", name="tick")


def test_a_lease_idle_past_the_limit_fires():
    status = Status()
    d = LeaseIdle(status, lambda: {"0000:01:00.0"}, agent="coder", idle_s=1800, poll_s=60)
    found = replay([tick(t) for t in range(0, 1861, 60)], [d])
    assert times(found) == [1800] and found[0].agent == "coder"
    assert found[0].notice_only and not found[0].pause      # a notice, never the repeat ladder


def test_an_open_device_resets_the_idle_clock():
    status = Status()
    d = LeaseIdle(status, lambda: {"0000:01:00.0"}, agent="coder", idle_s=1800, poll_s=60)
    replay([tick(0), tick(900)], [d])
    status.state = "HELD"
    replay([tick(960)], [d])
    status.state = "CLAIMED"
    assert replay([tick(1020), tick(2700)], [d]) == []


def test_lease_idle_reads_status_at_most_once_per_poll():
    status = Status()
    d = LeaseIdle(status, lambda: {"0000:01:00.0"}, agent="coder", idle_s=1800, poll_s=60)
    replay([tick(t) for t in range(0, 600)], [d])
    assert status.calls == 10


def test_an_exempt_lease_and_an_unreadable_status_never_fire():
    d = LeaseIdle(Status(), lambda: {"0000:01:00.0"}, agent="c", idle_s=60, poll_s=1,
                  exempt=lambda: True)
    assert replay([tick(t) for t in range(200)], [d]) == []
    d = LeaseIdle(Status("error"), lambda: {"0000:01:00.0"}, agent="c", idle_s=60, poll_s=1)
    assert replay([tick(t) for t in range(200)], [d]) == []


def stage_event(ts, name, stage=2):
    return Event(ts=ts, agent="supervisor", kind="ledger", name=name, stage=stage)


def test_a_stage_past_its_budget_asks_for_a_pause_once():
    d = StageOverBudget({2: 100})
    found = replay([stage_event(0, "stage_start"), resp(50), resp(101), resp(200)], [d])
    assert times(found) == [101] and found[0].pause is True


def test_a_finished_stage_or_one_without_a_budget_never_fires():
    d = StageOverBudget({2: 100})
    evs = [stage_event(0, "stage_start"), stage_event(50, "stage_end"), resp(500),
           stage_event(600, "stage_start", stage=5), resp(5000)]
    assert replay(evs, [d]) == []


# ---- NoFileWritten -------------------------------------------------------------------------------

def wrote(ts, ok=True, tool="write_file", agent="a"):
    """The result of a file-writing tool call. ok=None is a transcript, which does not say."""
    return Event(ts=ts, agent=agent, kind="tool_result", tool=tool, output_hash="2" * 64, wrote=ok)


def test_writeless_turns_default_is_twenty():
    from orchard.defaults import WRITELESS_TURNS
    assert WRITELESS_TURNS == 20 and NoFileWritten().turns == 20


def test_twenty_turns_without_a_write_fire_once():
    found = replay([resp(t, i=t) for t in range(1, 51)], [NoFileWritten()])
    assert times(found) == [20]
    f = found[0]
    assert f.detector == "no_file_written" and f.evidence["turns"] == 20
    assert f.pause is False and f.notice_only is False          # nudge level
    assert "20 model turns" in f.nudge and "no file was written" in f.nudge
    assert "write the files the skill names now" in f.nudge and "blocks" in f.nudge


def test_a_write_re_arms_the_detector():
    evs = [resp(t, i=t) for t in range(1, 26)] + [wrote(26)] + [resp(t, i=t) for t in range(27, 47)]
    assert times(replay(evs, [NoFileWritten()])) == [20, 46]


def test_a_write_resets_the_count_before_it_fires():
    evs = [resp(t, i=t) for t in range(1, 16)] + [wrote(16)] + [resp(t, i=t) for t in range(17, 36)]
    assert replay(evs, [NoFileWritten()]) == []


def test_a_new_evidence_file_counts_as_a_write():
    evs = ([resp(t, i=t) for t in range(1, 16)]
           + [Event(ts=16, agent="a", kind="evidence", name="stages/2/evidence/x.json")]
           + [resp(t, i=t) for t in range(17, 36)])
    assert replay(evs, [NoFileWritten()]) == []


def test_a_refused_write_or_a_shell_result_is_not_a_write():
    evs = []
    for t in range(1, 21):
        evs += [resp(t, i=t), wrote(t + 0.5, ok=False),
                Event(ts=t + 0.6, agent="a", kind="tool_result", tool="shell", output_hash="3" * 64)]
    assert times(replay(evs, [NoFileWritten()])) == [20]


@pytest.mark.parametrize("tool", ["write_file", "edit", "replace"])
def test_a_transcript_write_tool_counts_when_the_source_does_not_say(tool):
    evs = [resp(t, i=t) for t in range(1, 16)] + [wrote(16, ok=None, tool=tool)] + \
          [resp(t, i=t) for t in range(17, 36)]
    assert replay(evs, [NoFileWritten()]) == []


def test_writeless_turns_are_counted_per_agent():
    evs = [resp(t, i=t, agent="a" if t % 2 else "b") for t in range(1, 39)]
    assert replay(evs, [NoFileWritten()]) == []



@pytest.mark.parametrize("name", ["qwen_quiet_signature.jsonl", "qwen_loop_signature.jsonl"])
def test_no_file_written_stays_quiet_on_the_committed_signatures(name):
    # The transcripts do not say whether a write succeeded, so a result from qwen-code's write
    # tools counts. The longest run without one is 18 turns (loop chat) and 16 (quiet chat).
    from pathlib import Path

    from orchard.transcripts import load_signature
    evs = load_signature(Path(__file__).resolve().parent / "fixtures" / name)
    assert replay(evs, [NoFileWritten()]) == []
