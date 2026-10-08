# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The response ladder, the retry guard and the watchdog that joins them."""
import pytest

from orchard.ledger import Ledger
from orchard.watchdog import (Event, Finding, IdenticalResponses, Ladder, RetryGuard,
                              StageOverBudget, Watchdog, nudge_message)


class Actuator:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def nudge(self, agent, message):
        self.calls.append(("nudge", agent))
        if self.fail:
            raise RuntimeError("proxy down")

    def escalate(self, agent, stage):
        self.calls.append(("escalate", agent, stage))

    def pause(self, reason, evidence):
        self.calls.append(("pause", reason))


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        yield led


def finding(agent="coder", pause=False, summary="3 identical responses in a row"):
    return Finding("identical_responses", agent, 1.0, summary, {"count": 3}, pause=pause)


def rungs(ledger):
    return [(e["event"], e["data"].get("rung")) for e in ledger.read() if e["data"].get("watchdog")]


def test_a_launched_agent_climbs_nudge_escalate_pause_once_each(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    got = [ladder.respond("coder", [finding()], 2) for _ in range(5)]
    assert got == ["nudge", "escalate", "pause", "none", "none"]
    assert [c[0] for c in act.calls] == ["nudge", "escalate", "pause"]
    assert rungs(ledger) == [("retry", "nudge"), ("escalate", "escalate"), ("decision", "pause")]


def test_the_ledger_entry_comes_before_the_action(ledger):
    with pytest.raises(RuntimeError):
        Ladder(Actuator(fail=True), ledger, {"coder"}).respond("coder", [finding()], 2)
    # A restarted supervisor reads the nudge from the ledger and does not nudge again.
    act = Actuator()
    assert Ladder(act, ledger, {"coder"}).respond("coder", [finding()], 2) == "escalate"


def test_a_foreign_agent_gets_a_notice_and_no_action(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    assert [ladder.respond("someone", [finding("someone")], 2) for _ in range(4)] == ["notice"] * 4
    assert act.calls == []
    notes = [e["data"] for e in ledger.read() if e["event"] == "notice"]
    assert len(notes) == 4 and all(n["report_only"] is True for n in notes)
    assert notes[0]["findings"][0]["evidence"] == {"count": 3}


def test_a_budget_finding_goes_straight_to_pause(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    assert ladder.respond("supervisor", [finding("supervisor", pause=True)], 2) == "pause"
    assert ladder.respond("supervisor", [finding("supervisor", pause=True)], 2) == "none"
    assert [c[0] for c in act.calls] == ["pause"]


def test_an_idle_lease_gets_a_notice_and_never_a_nudge(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    idle = Finding("lease_idle", "coder", 1.0, "lease chips held idle", {}, notice_only=True)
    assert [ladder.respond("coder", [idle], 2) for _ in range(3)] == ["notice"] * 3
    assert act.calls == []
    assert len([e for e in ledger.read() if e["event"] == "notice"]) == 3
    assert ladder.respond("coder", [idle, finding()], 2) == "nudge"   # the repeat still climbs
    assert "repeating itself" not in str([e["data"] for e in ledger.read()
                                          if e["event"] == "notice"])


@pytest.mark.parametrize("by", ["retry", "operator"])
def test_a_resume_gives_the_ladder_back_its_rungs(ledger, by):
    """On the lab run stage 2 used its nudge, escalate and pause before an unattended block. After
    `resume by retry` the ladder still read them from the ledger and answered every finding with
    "none": the agent ran one pair of commands 25 times and nothing stopped it, not even the budget."""
    first = Ladder(Actuator(), ledger, {"coder"})
    assert [first.respond("coder", [finding()], 2) for _ in range(3)] == ["nudge", "escalate", "pause"]
    ledger.append("decision", None, decision="resume", by=by)
    act = Actuator()
    again = Ladder(act, ledger, {"coder"})
    assert [again.respond("coder", [finding()], 2) for _ in range(4)] == ["nudge", "escalate", "pause", "none"]
    assert [c[0] for c in act.calls] == ["nudge", "escalate", "pause"]


def test_rungs_taken_after_a_resume_still_count_across_a_restart(ledger):
    ledger.append("decision", None, decision="resume", by="retry")
    Ladder(Actuator(), ledger, {"coder"}).respond("coder", [finding()], 2)       # the nudge
    assert Ladder(Actuator(), ledger, {"coder"}).respond("coder", [finding()], 2) == "escalate"


def test_a_budget_pause_works_again_after_a_resume(ledger):
    Ladder(Actuator(), ledger, {"coder"}).respond("coder", [finding(pause=True)], 2)
    ledger.append("decision", None, decision="resume", by="retry")
    assert Ladder(Actuator(), ledger, {"coder"}).respond("coder", [finding(pause=True)], 2) == "pause"


def test_rungs_are_counted_per_stage(ledger):
    ladder = Ladder(Actuator(), ledger, {"coder"})
    assert ladder.respond("coder", [finding()], 2) == "nudge"
    assert ladder.respond("coder", [finding()], 3) == "nudge"


def test_caps_can_allow_more_than_one_nudge(ledger):
    ladder = Ladder(Actuator(), ledger, {"coder"}, caps={"nudge": 2, "escalate": 1, "pause": 1})
    assert [ladder.respond("coder", [finding()], 2) for _ in range(3)] == ["nudge", "nudge", "escalate"]


def test_caps_must_name_every_rung(ledger):
    with pytest.raises(ValueError):
        Ladder(Actuator(), ledger, {"coder"}, caps={"nudge": 1})


def test_the_nudge_names_the_repeat_and_asks_for_something_different():
    msg = nudge_message([finding(summary="3 identical responses in a row (145299 input, 33348 output tokens)")])
    assert "3 identical responses in a row" in msg
    assert "greedily" in msg and "different" in msg


def test_retry_guard_allows_one_retry_of_the_same_call():
    g = RetryGuard()
    req = {"messages": [{"role": "user", "content": "x"}], "temperature": 0}
    assert [g.allow("coder", req) for _ in range(3)] == [True, True, False]
    nudged = {"messages": req["messages"] + [{"role": "user", "content": "try another way"}]}
    assert g.allow("coder", nudged) is True
    assert g.allow("other", req) is True


def resp(ts):
    return Event(ts=ts, agent="coder", kind="response", input_tokens=10, output_tokens=5)


def test_the_watchdog_tracks_the_stage_and_answers_once_per_agent(ledger):
    act = Actuator()
    wd = Watchdog([IdenticalResponses(), StageOverBudget({2: 1000})], Ladder(act, ledger, {"coder"}))
    wd.feed(Event(ts=0, agent="supervisor", kind="ledger", name="stage_start", stage=2))
    found = [f for t in (1, 2, 3) for f in wd.feed(resp(t))]
    assert wd.stage == 2 and [f.detector for f in found] == ["identical_responses"]
    assert act.calls == [("nudge", "coder")]
    entries = [e for e in ledger.read() if e["data"].get("watchdog")]
    assert entries[0]["stage"] == 2


def test_a_finding_with_its_own_nudge_text_sends_that_text(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    f = Finding("no_file_written", "coder", 1.0, "20 model turns and no file written", {"turns": 20},
                nudge="20 model turns have passed and no file was written. Write the files now.")
    assert ladder.respond("coder", [f], 2) == "nudge"
    msg = [e["data"]["message"] for e in ledger.read() if e["event"] == "retry"][0]
    assert "no file was written" in msg and "repeating itself" not in msg
    # Mixed with a repeat finding, both texts are sent.
    both = nudge_message([f, finding()])
    assert "no file was written" in both and "3 identical responses in a row" in both


def test_twenty_writeless_turns_reach_the_ladder_as_a_nudge(ledger):
    from orchard.watchdog import NoFileWritten
    act = Actuator()
    wd = Watchdog([NoFileWritten()], Ladder(act, ledger, {"coder"}))
    found = [f for t in range(1, 21) for f in wd.feed(Event(ts=t, agent="coder", kind="response",
                                                             input_tokens=t, output_tokens=5))]
    assert [f.detector for f in found] == ["no_file_written"]
    assert act.calls == [("nudge", "coder")]
