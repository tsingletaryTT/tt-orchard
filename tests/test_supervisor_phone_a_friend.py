# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Phone a friend, end to end (orchard/phone_a_friend.py): a full-port stage 2 attempt fails, and the next attempt asks the outside helper for a
draft port. The chips, the coder and the helper's endpoint are fakes (the helper is a recording transport behind the real `Telegraph`); the
supervisor, the stage machine, the gate, the hardware-test runner and the port check template are the real ones."""
import pytest

from orchard import phone_a_friend as paf
from orchard import supervisor as sv
from orchard import telegraph as tg
from orchard.supervisor import EXIT_ABORTED, Control
from port_check_fakes import HONEST
from run_fakes import bringup, where
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, _snapshot_with_weights, make, with_snapshot
from test_supervisor_port_repair import BROKEN, broken, repairs
from test_telegraph import KEY, Wire, openai_reply, table

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    yield r
    r.close()


def helper_reply(code=HONEST):
    return lambda body: openai_reply(f"Here is the port.\n```python\n{code}\n```\n", 900, 300)


def wire_helper(rig, monkeypatch, tmp_path, wire=None, **kw):
    """Put a real `Telegraph` with a recording transport in the supervisor the rig builds (`--telegraph-config` is read by `build`)."""
    wire = wire or Wire(helper_reply())
    cfg = tg.load(table(models=["helper-model"], prices={"helper-model": [3.0, 15.0]}, **kw), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=wire, environ={"TG_KEY": KEY}, clock=lambda: 1000.0, hostname="node9", home="/home/someone")
    monkeypatch.setattr(sv, "_telegraph_from", lambda *a, **k: tel)
    monkeypatch.setattr(sv.Supervisor, "REPAIR_HELP_DEFAULT", False)      # these tests are about the asks at attempt starts; tests/test_supervisor_telegraph_in_repair.py turns it on
    return tel, wire


def coder(rig, uses_the_draft=True):
    """Attempt 1's prepare and every repair round write a port that fails at load. Attempt 2's prepare writes the working port when the
    supervisor's message names a draft and `uses_the_draft`, as a coder that copies it would. Returns the prepare messages seen."""
    make(rig, over={(2, "prepare"): {**PREPARE, "port/port_model.py": BROKEN}})
    repairs(rig, [{"port/port_model.py": broken(1)}, {"port/port_model.py": broken(2)}, {"port/port_model.py": broken(3)}])
    base = rig.script
    prepares = []

    def script(request):
        if "tools" in request and where(request) == (2, "prepare"):
            text = request["messages"][1]["content"]
            if len(prepares) == len(set(prepares)) and text not in prepares:
                prepares.append(text)
            if "stages/2/telegraph/draft-1.py" in text and uses_the_draft:
                return bringup(request, overrides={(2, "prepare"): {**PREPARE, "port/port_model.py": HONEST}})
        return base(request)
    rig.script = script
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")          # an attended run ends at the pause before stage 3
    return prepares


def telegraph_decisions(rig):
    names = {paf.ASKED, paf.ANSWERED, paf.NOT_ASKED, paf.REPORTED, paf.DRAFT_FAILED}
    return [d for d in rig.decisions() if d.get("decision") in names]


# ---- the helper is asked in the attempt after a failure, once -------------------------------------------------------------------------------

def test_the_second_attempt_asks_once_writes_the_draft_and_tells_the_coder(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path)
    prepares = coder(rig)
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == ["escalate", "pass"]
    assert len(wire.requests) == 1                                         # one call for the whole stage
    names = [d["decision"] for d in telegraph_decisions(rig)]
    assert names == [paf.ASKED, paf.ANSWERED, paf.REPORTED]
    draft = rig.run_dir / "stages" / "2" / "telegraph" / "draft-1.py"
    assert draft.is_file() and "def " in draft.read_text() and draft.read_text().startswith("# A draft from an outside helper model")
    answered = next(d for d in telegraph_decisions(rig) if d["decision"] == paf.ANSWERED)
    assert answered["ok"] is True and answered["parses"] is True and answered["tokens_in"] == 900 and answered["tokens_out"] == 300
    assert answered["draft"]["path"] == "stages/2/telegraph/draft-1.py" and len(answered["draft"]["sha256"]) == 64
    assert "stages/2/telegraph/draft-1.py" in prepares[-1] and "may be wrong" in prepares[-1]
    assert "draft-1.py" not in prepares[0]                                 # the first attempt had no helper


def test_the_request_holds_the_coders_brief_the_facts_and_the_failure_and_no_home_path(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path)
    _snapshot_with_weights(rig)                                              # a snapshot the facts file can be written from
    coder(rig)
    rig.run()
    sent = wire.requests[0]["body"]["messages"][-1]["content"]
    assert "Reply with ONE Python code block" in sent and "## What the local coder was told" in sent
    assert "## The facts file (port_facts.json)" in sent and "stages/2/port/" in sent
    assert "What failed:" in sent and "The latest attempt ended with" in sent
    assert str(rig.home) not in sent
    asked = next(d for d in telegraph_decisions(rig) if d["decision"] == paf.ASKED)
    assert asked["k"] == 1 and asked["problem"] == paf.problem_id(rig.run_dir) and asked["inputs"]["facts"]["chars"] > 0 and asked["failed_attempts"] == 1


def test_a_prompt_cap_too_small_for_the_request_is_recorded_on_the_ask(rig, monkeypatch, tmp_path):
    wire_helper(rig, monkeypatch, tmp_path, max_prompt_chars=2000)
    coder(rig)
    rig.run()
    asked = next(d for d in telegraph_decisions(rig) if d["decision"] == paf.ASKED)
    assert asked["over_cap"] is True and asked["max_prompt_chars"] == 2000


def test_a_roomy_prompt_cap_is_not_flagged(rig, monkeypatch, tmp_path):
    wire_helper(rig, monkeypatch, tmp_path, max_prompt_chars=400_000)
    coder(rig)
    rig.run()
    assert next(d for d in telegraph_decisions(rig) if d["decision"] == paf.ASKED)["over_cap"] is False


def test_the_ask_is_in_the_ledger_before_the_answer(rig, monkeypatch, tmp_path):
    wire_helper(rig, monkeypatch, tmp_path)
    coder(rig)
    rig.run()
    seqs = {d["decision"]: e["seq"] for e in rig.entries() if e["event"] == "decision" for d in [e["data"]] if d.get("decision") in
            (paf.ASKED, paf.ANSWERED, paf.REPORTED)}
    assert seqs[paf.ASKED] < seqs[paf.ANSWERED] < seqs[paf.REPORTED]


def test_a_working_draft_is_reported_as_helping_in_both_ledgers(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path)
    coder(rig)
    rig.run()
    reported = next(d for d in telegraph_decisions(rig) if d["decision"] == paf.REPORTED)
    assert reported["helped"] is True
    summary = tg.summarise(tmp_path / "tg")
    assert summary["helped"] == 1 and summary["calls"] == 1


def test_a_draft_the_coder_ignores_is_reported_as_not_helping(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path)
    coder(rig, uses_the_draft=False)
    rig.run()
    reported = [d for d in telegraph_decisions(rig) if d["decision"] == paf.REPORTED]
    assert [r["helped"] for r in reported] == [False]
    assert tg.summarise(tmp_path / "tg")["not_helped"] == 1


# ---- nothing is sent when it should not be --------------------------------------------------------------------------------------------------

def test_without_a_helper_the_run_is_what_it_was_and_the_ledger_has_no_telegraph_entry(rig):
    prepares = coder(rig, uses_the_draft=False)
    rig.run()
    assert telegraph_decisions(rig) == []
    assert all("draft-1.py" not in p for p in prepares)
    assert not (rig.run_dir / "stages" / "2" / "telegraph").exists()


def test_a_first_attempt_asks_nobody(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path)
    make(rig)                                                                # the working port from the start: no failed attempt
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert wire.requests == [] and telegraph_decisions(rig) == []


def test_a_refusal_by_the_helper_is_a_ledger_decision_and_the_stage_goes_on(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path, max_calls=0)         # a cap of zero calls: every ask is refused
    prepares = coder(rig)
    rig.run()
    assert wire.requests == []
    answered = next(d for d in telegraph_decisions(rig) if d["decision"] == paf.ANSWERED)
    assert answered["ok"] is False and answered["reason"]
    assert [d["decision"] for d in telegraph_decisions(rig)] == [paf.ASKED, paf.ANSWERED]          # a refusal has no draft step after it
    assert all("draft-1.py" not in p for p in prepares)
    assert not (rig.run_dir / "stages" / "2" / "telegraph").exists()
    assert [d["result"] for d in rig.ends(2)][0] == "escalate"               # the stage ran on as it would have without the helper


def test_no_key_is_a_refusal_not_a_failure(rig, monkeypatch, tmp_path):
    cfg = tg.load(table(models=["helper-model"]), base=tmp_path)
    wire = Wire(helper_reply())
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=wire, environ={}, clock=lambda: 1000.0, hostname="node9", home="/home/someone")
    monkeypatch.setattr(sv, "_telegraph_from", lambda *a, **k: tel)
    # The subject here is the refusal for a missing key at an attempt start. With the helper on, a later attempt now restores the furthest port
    # (issue 596) and skips the attempt-start draft, so the keeping of the best port is switched off for this test.
    monkeypatch.setattr(sv.Supervisor, "_best_port_on", lambda self: False)
    coder(rig)
    rig.run()
    assert wire.requests == []
    assert next(d for d in telegraph_decisions(rig) if d["decision"] == paf.ANSWERED)["ok"] is False


def test_a_reply_with_no_code_leaves_no_draft_and_says_so(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path, wire=Wire(lambda body: openai_reply("Check the index convention.", 500, 20)))
    prepares = coder(rig)
    rig.run()
    names = [d["decision"] for d in telegraph_decisions(rig)]
    assert paf.DRAFT_FAILED in names and "draft-1.py" not in prepares[-1]
    assert not (rig.run_dir / "stages" / "2" / "telegraph" / "draft-1.py").exists()


def test_a_draft_that_does_not_parse_is_still_given_with_the_warning(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path, wire=Wire(helper_reply("def load(:\n    pass")))
    prepares = coder(rig, uses_the_draft=False)
    rig.run()
    answered = next(d for d in telegraph_decisions(rig) if d["decision"] == paf.ANSWERED)
    assert answered["parses"] is False
    assert "does not parse as Python (SyntaxError)" in prepares[-1]


def test_the_helper_text_is_never_run_by_the_supervisor(rig, monkeypatch, tmp_path):
    marker = tmp_path / "ran"
    evil = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ran')\n"
    wire_helper(rig, monkeypatch, tmp_path, wire=Wire(helper_reply(evil)))
    coder(rig, uses_the_draft=False)
    rig.run()
    assert (rig.run_dir / "stages" / "2" / "telegraph" / "draft-1.py").is_file() and not marker.exists()


def test_a_draft_that_cannot_be_written_is_a_ledger_decision_and_the_stage_goes_on(rig, monkeypatch, tmp_path):
    wire_helper(rig, monkeypatch, tmp_path)
    prepares = coder(rig)

    def refuse(*a, **k):
        raise OSError("telegraph is not a plain directory (a link)")
    monkeypatch.setattr(paf, "write_draft", refuse)
    rig.run()
    failed = next(d for d in telegraph_decisions(rig) if d["decision"] == paf.DRAFT_FAILED)
    assert "could not be written" in failed["reason"] and "OSError" in failed["reason"]
    assert all("draft-1.py" not in p for p in prepares)
    assert [d["result"] for d in rig.ends(2)][0] == "escalate"


def test_an_operator_abort_in_the_attempt_gives_no_verdict_on_the_draft(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path)
    prepares = coder(rig, uses_the_draft=False)
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "prepare") and "draft-1.py" in request["messages"][1]["content"]:
            Control(rig.run_dir).write("abort")                       # the operator aborts while the coder works from the draft
        return base(request)
    rig.script = script
    rig.run()
    names = [d["decision"] for d in telegraph_decisions(rig)]
    assert paf.ASKED in names and paf.REPORTED not in names and "draft-1.py" in prepares[-1]
    assert tg.summarise(tmp_path / "tg")["helped"] == 0 and tg.summarise(tmp_path / "tg")["not_helped"] == 0


# ---- the counts --------------------------------------------------------------------------------------------------------------------------

def test_the_ledger_counts_one_call_in_this_run_and_the_limits_are_not_exceeded(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(rig, monkeypatch, tmp_path)
    coder(rig, uses_the_draft=False)
    rig.run()
    state = paf.attempt_state(rig.entries(), 2)
    assert len(wire.requests) == state["asked_in_stage"] == 1 and state["asked_in_stage"] <= paf.PER_STAGE


# ---- issue 650: an unreachable helper at an attempt start ---------------------------------------------------------------------------

def test_an_unattended_run_whose_helper_cannot_be_reached_at_an_attempt_start_ends_blocked_with_the_helper_unreachable_code(rig, monkeypatch, tmp_path):
    """The attempt-start ask is the other call site of the helper: attempt 1 fails, the next attempt asks, the endpoint cannot be reached."""
    rig.args.unattended = True

    class Down:
        calls = 0

        def __call__(self, url, headers, body, timeout):
            Down.calls += 1
            raise tg.Unreachable("cannot connect to llm.example.org:443: timed out after 10 s")

    wire_helper(rig, monkeypatch, tmp_path, wire=Down(), after_failures=1, unreachable_after=1, unreachable_action="block")
    coder(rig)
    rig.run()
    block = [d for d in rig.decisions() if d.get("decision") == "blocked"]
    assert len(block) == 1 and block[0]["code"] == "helper-unreachable"
    assert Down.calls == 1 and len([d for d in rig.decisions() if d.get("decision") == "helper unreachable, not asking"]) == 1


def test_with_the_breaker_open_an_attempt_start_writes_no_ask_and_sends_nothing_and_the_run_is_not_blocked(rig, monkeypatch, tmp_path):
    """Review of PR 651: the default action continues, and an ask the breaker would only refuse must not be recorded as an ask (it would use up
    the stage's quota during an outage). The probe interval has not passed (the clock stands still), so the due ask is left unwritten."""
    rig.args.unattended = True
    wire = Wire(helper_reply())
    tel, _ = wire_helper(rig, monkeypatch, tmp_path, wire=wire, after_failures=1)
    tel.hold_open()
    coder(rig, uses_the_draft=False)
    rig.run()
    assert not [d for d in rig.decisions() if d.get("decision") == paf.ASKED] and not wire.requests
    assert [d for d in rig.decisions() if d.get("decision") == paf.NOT_ASKED and "not due yet" in d.get("reason", "")]
    assert "helper-unreachable" not in [d.get("code") for d in rig.decisions() if d.get("decision") == "blocked"]


def test_an_attempt_start_ask_that_could_not_connect_is_not_counted_and_the_default_run_goes_on(rig, monkeypatch, tmp_path):
    rig.args.unattended = True

    class Down:
        calls = 0

        def __call__(self, url, headers, body, timeout):
            Down.calls += 1
            raise tg.Unreachable("cannot connect to llm.example.org:443: timed out after 10 s")

    wire_helper(rig, monkeypatch, tmp_path, wire=Down(), after_failures=1, unreachable_after=1)          # the default action: continue
    coder(rig, uses_the_draft=False)
    rig.run()
    asks = [d for d in rig.decisions() if d.get("decision") == paf.ASKED]
    assert len(asks) == 1 and Down.calls == 1
    assert paf.attempt_state(rig.entries(), 2)["asked_in_stage"] == 0                                    # the ask never reached the helper
    assert "helper-unreachable" not in [d.get("code") for d in rig.decisions() if d.get("decision") == "blocked"]
