# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The outside helper inside the port repair rounds, end to end (orchard/repair_help.py, Supervisor._port_repair_rounds). The chips, the coder and
the helper's endpoint are fakes; the supervisor, the stage machine, the gate, the check template (which runs for real on the CPU with a fake
`ttnn`) and the real `Telegraph` are the real ones.

Run11 of the tiny Qwen4 rehearsal is the model for these: a port that fails at load, then at prefill, a coder that loops, and a helper that was
asked twice at attempt starts and never inside a round."""
import hashlib
import json

import pytest

from orchard import phone_a_friend as paf
from orchard import port_repair, repair_help as rh
from orchard import supervisor as sv
from orchard import telegraph as tg
from orchard.supervisor import EXIT_ABORTED, Control
from port_check_fakes import HONEST
from run_fakes import bringup, call, turn, where
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, make, with_snapshot
from test_supervisor_port_repair import BROKEN, broken, repairs
from test_telegraph import KEY, Wire, openai_reply, table

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")

PREFILL_BROKEN = HONEST.replace("return _peak(100)", "raise RuntimeError('Sizes of tensors must match except in dimension 1. Expected size 192 but got size 320')")
STUB = HONEST.replace("return _peak(100)", "raise NotImplementedError('prefill not implemented')")
UNDEFINED = HONEST.replace("return _peak(token_id + 1)", "return _peak(helper_nobody_defined(token_id))")
PORT = "port/port_model.py"


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    yield r
    r.close()


def reply(code):
    return lambda body: openai_reply(f"Here is the corrected port.\n```python\n{code}\n```\n", 1200, 400)


def wire_helper(monkeypatch, tmp_path, replies, *, models=("helper-model",), key=KEY, **kw):
    """A real `Telegraph` with a recording transport; `replies` is a list (the last one repeats) of code strings or functions of the request."""
    state = {"i": 0}

    def answer(body):
        r = replies[min(state["i"], len(replies) - 1)]
        state["i"] += 1
        return r(body) if callable(r) else reply(r)(body)
    wire = Wire(answer)
    cfg = tg.load(table(models=list(models), prices={m: [3.0, 15.0] for m in models}, max_calls_per_problem=8, max_calls=20, **kw), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=wire, environ={"TG_KEY": key} if key else {}, clock=lambda: 1000.0, hostname="node9",
                       home="/home/someone")
    monkeypatch.setattr(sv, "_telegraph_from", lambda *a, **k: tel)
    return tel, wire


def start(rig, port, per_round):
    """Stage 2's prepare writes `port`; round k of a repair writes `per_round[k - 1]` (the last repeats). An attended run ends at the pause before stage 3."""
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: port}})
    seen = repairs(rig, [{PORT: text} for text in per_round])
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    return seen


def decs(rig, *names):
    return [d for d in rig.decisions() if d.get("decision") in names]


def port_text(rig):
    return (rig.run_dir / "stages" / "2" / "port" / "port_model.py").read_text()


def stage_ends(rig):
    return [d["result"] for d in rig.ends(2)]


# ---- the helper at the start of the first round ------------------------------------------------------------------------------------------------

def test_the_first_failed_round_asks_the_helper_places_its_port_and_the_coder_starts_from_it(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [HONEST])
    seen = start(rig, BROKEN, [HONEST + "# the coder checked it\n"])
    assert rig.run() == EXIT_ABORTED
    assert stage_ends(rig) == ["pass"]
    asked, answered, placed = decs(rig, rh.ASKED), decs(rig, rh.ANSWERED), decs(rig, rh.PLACED)
    assert len(asked) == len(answered) == len(placed) == 1 and len(wire.requests) == 1
    assert asked[0]["round"] == 1 and asked[0]["failure_op"] == "load" and asked[0]["coder_failed"] is False and asked[0]["k"] == 1
    assert asked[0]["port_sha256"] == hashlib.sha256(BROKEN.encode()).hexdigest()
    assert placed[0]["sent_sha256"] == placed[0]["found_sha256"] == asked[0]["port_sha256"] and "still the one" in placed[0]["why"]
    assert (rig.run_dir / "stages" / "2" / "telegraph" / "repair-1.py").is_file()
    assert "has just replaced" in seen[0] and "stages/2/telegraph/repair-1.py" in seen[0] and "OLD file" in seen[0]
    assert "# the coder checked it" in port_text(rig)


def test_the_request_holds_the_port_that_failed_and_the_failure_and_no_home_path(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [HONEST])
    start(rig, BROKEN + f"# {rig.home}/secret\n", [HONEST + "# c\n"])
    rig.run()
    sent = wire.requests[0]["body"]["messages"][-1]["content"]
    assert "COMPLETE corrected" in sent and "def prefill(model, ids)" in sent and "model.layers.2.ple.conv1d.weight" in sent
    assert "What failed:" in sent and "round 1 of 3" in sent and "'load'" in sent
    assert str(rig.home) not in sent


def test_the_ask_is_in_the_ledger_before_the_answer_and_the_answer_before_the_placement(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [HONEST])
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    seqs = {e["data"]["decision"]: e["seq"] for e in rig.entries()
            if e["event"] == "decision" and e["data"].get("decision") in (rh.ASKED, rh.ANSWERED, rh.PLACED, rh.PROGRESS)}
    assert seqs[rh.ASKED] < seqs[rh.ANSWERED] < seqs[rh.PLACED] < seqs[rh.PROGRESS]


def test_the_helper_is_asked_with_this_runs_own_problem_id(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [HONEST])
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    assert decs(rig, rh.ASKED)[0]["problem"] == paf.problem_id(rig.run_dir)


def test_a_check_that_passes_after_the_help_is_a_helped_verdict_in_both_ledgers(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [HONEST])
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    verdict = decs(rig, rh.PROGRESS)
    assert len(verdict) == 1 and verdict[0]["helped"] is True and verdict[0]["outcome"] == "passed"
    assert verdict[0]["before_op"] == "load" and verdict[0]["round"] == 1
    assert tg.summarise(tmp_path / "tg")["helped"] == 1


# ---- progress, not a flat verdict ---------------------------------------------------------------------------------------------------------------

def test_a_failure_that_moved_from_load_to_prefill_is_progress_and_the_next_round_lets_the_coder_go_first(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n", HONEST + "# b\n"])
    assert rig.run() == EXIT_ABORTED
    assert stage_ends(rig) == ["pass"]
    assert len(decs(rig, rh.ASKED)) == 1 and len(wire.requests) == 1                      # round 2 asked nobody
    progress = decs(rig, rh.PROGRESS)
    assert [(p["before_op"], p["after_op"], p["outcome"], p["helped"]) for p in progress] == [("load", "prefill", "further", True)]


def test_a_failure_at_the_same_request_is_not_progress_and_the_next_round_asks_again(rig, monkeypatch, tmp_path):
    still_load = BROKEN.replace("model.layers.2.ple.conv1d.weight", "model.layers.3.other.weight")
    tel, wire = wire_helper(monkeypatch, tmp_path, [still_load, HONEST])
    start(rig, BROKEN, [still_load + "# a\n", HONEST + "# b\n"])
    assert rig.run() == EXIT_ABORTED
    assert stage_ends(rig) == ["pass"]
    assert [d["round"] for d in decs(rig, rh.ASKED)] == [1, 2] and [d["k"] for d in decs(rig, rh.ASKED)] == [1, 2]
    progress = decs(rig, rh.PROGRESS)
    assert [(p["outcome"], p["helped"]) for p in progress] == [("same request", False), ("passed", True)]
    assert (rig.run_dir / "stages" / "2" / "telegraph" / "repair-2.py").is_file()


# ---- a stub is never placed ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad,needle", [(STUB, "prefill raises NotImplementedError"), (UNDEFINED, "helper_nobody_defined"),
                                         (HONEST.replace("def close(model):\n    pass\n", ""), "no close function")])
def test_a_reply_that_is_not_a_whole_port_is_recorded_as_a_stub_and_not_placed(rig, monkeypatch, tmp_path, bad, needle):
    tel, wire = wire_helper(monkeypatch, tmp_path, [bad, HONEST])
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    stubs = decs(rig, rh.STUB)
    assert len(stubs) == 1 and needle in " ".join(stubs[0]["problems"])
    assert not decs(rig, rh.PLACED)
    verdict = decs(rig, rh.PROGRESS)
    assert verdict[0]["outcome"] == "stub" and verdict[0]["helped"] is False
    assert tg.summarise(tmp_path / "tg")["not_helped"] >= 1


def test_after_a_stub_the_next_ask_of_the_run_goes_to_the_next_model_of_the_list(rig, monkeypatch, tmp_path):
    still_load = BROKEN.replace("model.layers.2.ple.conv1d.weight", "model.layers.3.other.weight")
    tel, wire = wire_helper(monkeypatch, tmp_path, [STUB, HONEST], models=("small-model", "big-model"))
    start(rig, BROKEN, [still_load, HONEST + "# c\n"])
    rig.run()
    assert [r["body"]["model"] for r in wire.requests][:2] == ["small-model", "big-model"]


# ---- the file the coder changed is not overwritten ----------------------------------------------------------------------------------------------

def test_a_port_the_coder_changed_while_the_helper_answered_is_not_overwritten(rig, monkeypatch, tmp_path):
    def answer_while_the_coder_edits(body):
        (rig.run_dir / "stages" / "2" / "port" / "port_model.py").write_text(BROKEN + "# the coder edited this while the helper answered\n")
        return reply(HONEST)(body)
    wire_helper(monkeypatch, tmp_path, [answer_while_the_coder_edits])
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    not_placed = decs(rig, rh.NOT_PLACED)
    assert len(not_placed) == 1 and "coder changed" in not_placed[0]["reason"]
    assert not_placed[0]["sent_sha256"] != not_placed[0]["found_sha256"]
    assert not decs(rig, rh.PLACED)


# ---- the helper after a coder step that failed --------------------------------------------------------------------------------------------------

def looping_coder(rig):
    """Round k >= 2's repair step reads the same file again and again, so the watchdog pauses it; round 1 and everything else follow the script."""
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "repair") and "round 2 of" in request["messages"][1]["content"]:
            return call("read_file", path="stages/2/port_facts.json")
        return base(request)
    rig.script = script


def test_a_repair_step_the_watchdog_paused_asks_the_helper_and_the_round_goes_on_to_its_check(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    looping_coder(rig)
    assert rig.run() == EXIT_ABORTED
    assert stage_ends(rig) == ["pass"]
    asked = decs(rig, rh.ASKED)
    assert [(a["round"], a["coder_failed"]) for a in asked] == [(1, False), (2, True)]
    assert "ended" in asked[1]["why"]
    placed = decs(rig, rh.PLACED)
    assert len(placed) == 2 and placed[1]["sent_sha256"] == placed[1]["found_sha256"]         # the looping coder wrote nothing, so the file is the one sent
    assert port_text(rig).startswith("# Replaced by the supervisor")


def test_a_step_that_stopped_for_an_operator_abort_asks_nobody(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "repair") and "round 2 of" in request["messages"][1]["content"]:
            Control(rig.run_dir).write("abort")
        return base(request)
    rig.script = script
    rig.run()
    assert [a["round"] for a in decs(rig, rh.ASKED)] == [1]


# ---- limits and refusals ------------------------------------------------------------------------------------------------------------------------

def test_the_helper_off_in_the_repair_rounds_is_the_older_behaviour(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [HONEST])
    monkeypatch.setattr(sv.Supervisor, "REPAIR_HELP_DEFAULT", False)
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    assert not wire.requests and not decs(rig, rh.ASKED, rh.NOT_ASKED)
    assert stage_ends(rig) == ["pass"]


def test_without_a_helper_no_repair_entry_is_written(rig):
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    assert not decs(rig, rh.ASKED, rh.NOT_ASKED, rh.PLACED, rh.PROGRESS) and stage_ends(rig) == ["pass"]


def test_a_refusal_by_the_helper_is_a_ledger_decision_and_the_coder_repairs_as_before(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [HONEST], key=None)                      # no key: the helper refuses
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    answered = decs(rig, rh.ANSWERED)
    assert len(answered) == 1 and answered[0]["ok"] is False and not decs(rig, rh.PLACED)
    assert stage_ends(rig) == ["pass"]


def test_a_port_over_the_request_size_is_not_sent_and_the_ledger_says_so(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [HONEST])
    monkeypatch.setattr(rh, "PORT_CAP", 40)
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    refused = decs(rig, rh.NOT_ASKED)
    assert refused and "more than the 40" in refused[0]["reason"] and not wire.requests
    assert stage_ends(rig) == ["pass"]


def test_the_stage_stops_asking_at_the_limit_counting_the_attempt_start_asks(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [BROKEN.replace("conv1d", "conv1e")])           # a helper whose port always fails the same way
    monkeypatch.setattr(rh, "TOTAL_PER_STAGE", 3)          # the real limit is 6; a smaller one reaches it inside one run
    start(rig, BROKEN, [BROKEN + "# edit\n"])
    rig.args.unattended = True
    from test_supervisor_full_port import never_wait
    never_wait(rig)
    rig.run()
    assert len(decs(rig, paf.ASKED, rh.ASKED)) == 3 and len(wire.requests) == 3
    assert any("already asked 3 times" in d["reason"] for d in decs(rig, rh.NOT_ASKED))


def test_the_limit_is_six_in_the_stage():
    assert rh.TOTAL_PER_STAGE == 6 and rh.PER_ROUND == 1


def test_the_helper_text_is_never_run_by_the_supervisor(rig, monkeypatch, tmp_path):
    marker = tmp_path / "ran"
    evil = HONEST + f"\nopen({str(marker)!r}, 'w').write('ran')\n"          # runs only if something imports it at the supervisor's end; the check runs it on the (fake) device
    wire_helper(monkeypatch, tmp_path, [evil])
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    import ast
    ast.parse((rig.run_dir / "stages" / "2" / "telegraph" / "repair-1.py").read_text())          # the saved reply is text
    assert not decs(rig, rh.STUB)                                                               # and the supervisor judged it by reading it
    assert not marker.exists(), "the helper's text ran: the supervisor must only read it"          # (finding, PR 496 round 3: the marker was never checked)


# ---- review of PR 496 ------------------------------------------------------------------------------------------------------------------------------

def test_after_a_helper_rescued_a_paused_repair_step_the_next_round_still_asks_the_coder(rig, monkeypatch, tmp_path):
    """Finding: the handled watchdog stop stayed set, so the next repair step ended before it sent a request, and the pause entry stayed unresumed."""
    tel, wire = wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, PREFILL_BROKEN + "# helper\n", HONEST])
    seen = start(rig, BROKEN, [PREFILL_BROKEN + "# a\n", PREFILL_BROKEN + "# b\n", HONEST])
    looping_coder(rig)
    assert rig.run() == EXIT_ABORTED
    assert any("round 3 of" in text for text in seen), "the coder was never asked in round 3"
    entries = rig.entries()
    seqs = [(e["seq"], e["data"].get("decision")) for e in entries if e["stage"] in (2, None) and e["event"] == "decision"]
    pauses = [s for s, d in seqs if d == "pause"]
    end2 = next(e["seq"] for e in entries if e["stage"] == 2 and e["event"] == "stage_end")
    # every pause the watchdog wrote inside stage 2 is answered by a resume before the stage ended
    for p in [p for p in pauses if p < end2]:
        assert any(s > p and s < end2 and d == "resume" for s, d in seqs), f"the pause at {p} was never resumed"


def test_a_repair_request_over_the_helpers_prompt_cap_is_not_sent(rig, monkeypatch, tmp_path):
    """Finding: the helper cuts the middle out of a request over max_prompt_chars, and the answer then replaced the whole file."""
    tel, wire = wire_helper(monkeypatch, tmp_path, [HONEST], max_prompt_chars=tg.MIN_PROMPT_CHARS)
    start(rig, BROKEN, [HONEST + "# c\n"])
    rig.run()
    refused = decs(rig, rh.NOT_ASKED)
    assert refused and any("prompt" in d["reason"] for d in refused), refused
    assert not decs(rig, rh.ASKED) and not wire.requests


def test_a_pause_the_stage_budget_caused_is_not_lifted_by_a_helper_rescue(rig, monkeypatch, tmp_path):
    """Finding (round 3): a round that skipped its start ask could rescue a deadline-expired stage and run another hardware check. A pause the
    stage_over_budget detector caused stays; a pause from the ladder (the test above) is still rescued."""
    tel, wire = wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    looping_coder(rig)
    # The looping coder reaches the escalate rung; here the same stop arrives as the budget detector's pause.
    def as_deadline(self, agent, stage):
        self.pause("watchdog: stage over budget", {"agent": agent, "findings": [{"detector": "stage_over_budget"}]})
    monkeypatch.setattr(sv.RunActuator, "escalate", as_deadline)
    orig_pause = sv.RunActuator.pause
    monkeypatch.setattr(sv.RunActuator, "pause", lambda self, reason, evidence: orig_pause(
        self, reason, {**evidence, "findings": [{**f, "detector": "stage_over_budget"} for f in evidence["findings"]]}))
    rig.run()
    asked = decs(rig, rh.ASKED)
    assert asked and not [a for a in asked if a["coder_failed"]], ("the helper was asked to rescue a deadline pause", [a["why"] for a in asked])
    assert stage_ends(rig) != ["pass"]


@pytest.mark.parametrize("text,why", [
    ("I do not think I can help with this one.", "no Python code block"),
    ("Here:\n```python\ndef prefill(:\n```\n", "does not parse"),
])
def test_a_reply_with_no_usable_code_is_reported_as_not_helped_even_when_the_coder_then_passes(rig, monkeypatch, tmp_path, text, why):
    """Finding (round 3): an ok reply with no Python block stayed eligible for a progress verdict, so the coder's own passing check counted it
    as helped and kept its model on the helper's ladder."""
    tel, wire = wire_helper(monkeypatch, tmp_path, [lambda body: openai_reply(text, 300, 40)])
    start(rig, BROKEN, [HONEST + "# the coder fixed it\n"])
    rig.run()
    assert stage_ends(rig) == ["pass"]
    verdicts = decs(rig, rh.PROGRESS)
    assert verdicts and all(v["helped"] is False and v["outcome"].startswith("unusable reply") and why in v["outcome"] for v in verdicts), verdicts


# ---- issue 650: an unreachable helper -----------------------------------------------------------------------------------------------

class _Down:
    def __init__(self):
        self.calls = 0

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        raise tg.Unreachable("cannot connect to llm.example.org:443: timed out after 10 s")


def _down_helper(monkeypatch, tmp_path, **kw):
    down = _Down()
    cfg = tg.load(table(models=["helper-model"], prices={"helper-model": [3.0, 15.0]}, max_calls_per_problem=8, max_calls=20, unreachable_after=1, **kw),
                  base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=down, environ={"TG_KEY": KEY}, clock=lambda: 1000.0, hostname="node9", home="/home/someone")
    monkeypatch.setattr(sv, "_telegraph_from", lambda *a, **k: tel)
    return tel, down


def test_an_unattended_run_whose_helper_cannot_be_reached_ends_blocked_with_the_helper_unreachable_code(rig, monkeypatch, tmp_path):
    rig.args.unattended = True
    tel, down = _down_helper(monkeypatch, tmp_path, unreachable_action="block")
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: BROKEN}})
    repairs(rig, [{PORT: BROKEN}])
    assert rig.run() == sv.EXIT_BLOCKED
    block = [d for d in rig.decisions() if d.get("decision") == "blocked"]
    assert len(block) == 1 and block[0]["code"] == "helper-unreachable" and "unreachable" in block[0]["reason"]
    assert len(decs(rig, "helper unreachable, not asking")) == 1 and down.calls == 1
    # The block comes from the repair round that made the first ask, in the first attempt: the stage never ended or escalated first (a hook
    # missing from this call site would let the attempt end and block at the next attempt's start instead).
    assert rig.ends(2) == [] and len(decs(rig, rh.ASKED)) == 1


def test_by_default_an_unattended_run_goes_on_without_the_helper_and_asks_no_more(rig, monkeypatch, tmp_path):
    rig.args.unattended = True
    tel, down = _down_helper(monkeypatch, tmp_path)
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: BROKEN}})
    repairs(rig, [{PORT: BROKEN}])
    rig.run()
    assert not [d for d in rig.decisions() if d.get("decision") == "blocked" and d.get("code") == "helper-unreachable"]
    assert len(decs(rig, "helper unreachable, not asking")) == 1 and down.calls == 1      # the breaker kept every later ask off the wire
    assert "helper-unreachable" not in _block_codes(rig)                                  # a helper that is down never stops the run (operator, 2026-11-09)


def test_a_restart_in_an_outage_with_the_default_action_probes_first_then_goes_on_without_spending_the_call_caps(rig, monkeypatch, tmp_path):
    rig.args.unattended = True
    state = {"up": False}
    _fresh_helper(monkeypatch, tmp_path, state)                              # the default action: continue
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: BROKEN}})
    repairs(rig, [{PORT: BROKEN}])
    rig.run(pid=100)
    assert state["calls"] == 1 and len(decs(rig, "helper unreachable, not asking")) == 1
    rig.run(pid=101)
    assert "helper-unreachable" not in _block_codes(rig)
    assert state["calls"] == 1 and state["probes"] >= 1                      # the restart probed, and no request went out while it was down


def _fresh_helper(monkeypatch, tmp_path, state, **kw):
    """A helper whose transport and probe follow `state` (state['up'] switches the endpoint on), and a NEW Telegraph (a new breaker) for every
    supervisor process, as a restart makes one."""
    state.setdefault("calls", 0)
    state.setdefault("probes", 0)

    def transport(url, headers, body, timeout):
        state["calls"] += 1
        if not state["up"]:
            raise tg.Unreachable("cannot connect to llm.example.org:443: timed out after 10 s")
        return 200, json.dumps(openai_reply(f"Here is the corrected port.\n```python\n{HONEST}\n```\n", 1200, 400)).encode()

    def prober(url, headers, timeout, connect_timeout=None):
        state["probes"] += 1
        if not state["up"]:
            raise tg.Unreachable("cannot connect to llm.example.org:443: timed out after 10 s")
        return 200

    def clock():
        state["now"] = state.get("now", 1000.0) + state.get("tick", 0.0)
        return state["now"]

    def factory(*a, **k):
        cfg = tg.load(table(models=["helper-model"], prices={"helper-model": [3.0, 15.0]}, max_calls_per_problem=8, max_calls=20,
                            unreachable_after=1, **kw), base=tmp_path)
        return tg.Telegraph(cfg, tmp_path / "tg", transport=transport, prober=prober, environ={"TG_KEY": KEY}, clock=clock,
                            hostname="node9", home="/home/someone")

    monkeypatch.setattr(sv, "_telegraph_from", factory)


def test_refused_asks_during_an_outage_do_not_use_up_the_quota_and_the_first_ask_after_the_endpoint_returns_goes_through(rig, monkeypatch, tmp_path):
    """Review of PR 651: the default action continues, and every refused ask used to be recorded as asked, so ten rounds in an outage spent the
    six-ask stage limit and the helper could not come back."""
    rig.args.unattended = True
    rig.args.port_repair_rounds = 10
    state = {"up": False}
    _fresh_helper(monkeypatch, tmp_path, state)                              # the default action: continue; the clock stands still
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: BROKEN}})
    repairs(rig, [{PORT: BROKEN + f"\n# change {i}\n"} for i in range(10)])
    rig.run(pid=100)
    rounds = [d for d in rig.decisions() if d.get("decision") == port_repair.DECISION]
    assert len(rounds) >= 7                                                  # the rounds went on without the helper
    assert state["calls"] == 1                                               # one real attempt; the breaker kept every other ask off the wire
    assert len(decs(rig, rh.ASKED)) == 1                                     # and no ask was written for a refusal the breaker made by itself
    assert "helper-unreachable" not in _block_codes(rig)
    entries = rig.entries()
    assert rh.asks_in_stage(entries, 2) == 0                                 # nothing the helper never saw is counted
    assert not [d for d in rig.decisions() if "already asked" in str(d.get("reason", ""))]
    # The endpoint returns and the probe interval passes: the next ask goes through.
    state["up"] = True
    state["now"] = state.get("now", 1000.0) + 10_000
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run(pid=101)
    assert state["calls"] >= 2 and len(decs(rig, rh.ANSWERED)) >= 2


def _block_codes(rig):
    return [d.get("code") for d in rig.decisions() if d.get("decision") == "blocked"]


def test_a_restart_in_the_middle_of_an_outage_probes_first_blocks_again_without_asking_and_goes_on_when_the_endpoint_is_back(rig, monkeypatch, tmp_path):
    """Review of PR 651: the breaker is new in a new process, so a restart must not forget the outage and spend its caps without help."""
    rig.args.unattended = True
    state = {"up": False}
    _fresh_helper(monkeypatch, tmp_path, state, unreachable_action="block")
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: BROKEN}})
    repairs(rig, [{PORT: BROKEN}])
    assert rig.run(pid=100) == sv.EXIT_BLOCKED
    assert _block_codes(rig) == ["helper-unreachable"] and state["calls"] == 1
    # The second process: the ledger holds the outage, the probe fails, no ask goes out, the run blocks with the same code.
    assert rig.run(pid=101) == sv.EXIT_BLOCKED
    assert _block_codes(rig) == ["helper-unreachable", "helper-unreachable"] and state["calls"] == 1 and state["probes"] == 1
    assert len(decs(rig, "helper unreachable, not asking")) == 1               # the outage is one event, not one per restart
    # The third process: the endpoint is back, the probe is answered, and the run asks.
    state["up"] = True
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run(pid=102)
    assert len(decs(rig, "helper reachable again")) == 1 and state["calls"] >= 2
    assert _block_codes(rig)[:2] == ["helper-unreachable"] * 2 and "helper-unreachable" not in _block_codes(rig)[2:]     # later blocks, if any, are the stage's own


def test_a_round_interrupted_by_the_outage_is_taken_again_after_the_restart_and_not_skipped(rig, monkeypatch, tmp_path):
    """Review of PR 651: the round's decision is written before its ask; a block at the ask must not make the resume advance to the next round
    (or end the stage when it was the last allowed round)."""
    rig.args.unattended = True
    rig.args.port_repair_rounds = 1                                         # the only round: skipping it would end the repair rounds as spent
    state = {"up": False}
    _fresh_helper(monkeypatch, tmp_path, state, unreachable_action="block")
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: BROKEN}})
    repairs(rig, [{PORT: BROKEN}])
    assert rig.run(pid=100) == sv.EXIT_BLOCKED
    assert port_repair.rounds_used(rig.entries(), 2) == 0                   # the interrupted round is taken back
    state["up"] = True
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run(pid=101)
    rounds = [d for d in rig.decisions() if d.get("decision") == port_repair.DECISION]
    assert [d["round"] for d in rounds][:2] == [1, 1] and {d["round"] for d in rounds} == {1}   # round 1 again, never a round 2
    assert not [d for d in rig.decisions() if d.get("decision") == port_repair.STOPPED and d.get("reason") == "spent"]
    assert [d for d in rig.decisions() if d.get("decision") == port_repair.WITHDRAWN][0]["round"] == 1
    assert len(decs(rig, rh.ANSWERED)) >= 1 and state["calls"] >= 2


def test_the_retaken_round_asks_the_helper_again_and_a_restart_in_a_continuing_outage_blocks_at_that_ask(rig, monkeypatch, tmp_path):
    """Review of PR 651: the interrupted round's ask is withdrawn, so the resume is due to ask in that round (it must not see the round as
    already asked and go on without help), and the probe at that ask finds the outage still open."""
    rig.args.unattended = True
    rig.args.port_repair_rounds = 1
    state = {"up": False}
    _fresh_helper(monkeypatch, tmp_path, state, unreachable_action="block")
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: BROKEN}})
    repairs(rig, [{PORT: BROKEN}])
    assert rig.run(pid=100) == sv.EXIT_BLOCKED
    assert rig.run(pid=101) == sv.EXIT_BLOCKED                              # still down: the second process blocks too
    assert _block_codes(rig) == ["helper-unreachable"] * 2 and state["calls"] == 1 and state["probes"] == 1
    assert not decs(rig, rh.NOT_ASKED)                                       # the round was due to ask: it never went on without the helper
    assert [d["round"] for d in rig.decisions() if d.get("decision") == port_repair.WITHDRAWN] == [1, 1]
