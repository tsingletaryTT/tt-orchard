# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 182: the drafter decision from measured numbers, recorded in the ledger and stated on the card.

`orchard/spec_break_even.py` (PR 190) does the arithmetic. This slice reads a measurement record that a hardware run leaves, `evidence/spec-measure.json` in stage
2's directory (plain step time, verify pass time, accepted tokens per pass by prompt kind, the sampling it was measured at and where the prompts came from), decides
from it (`orchard/spec_decision.py`), writes `evidence/spec-decision.json`, records a ledger decision and gives the package card a sentence. The measuring run
itself needs a board and is not built: the tests here use recorded numbers, among them the published example (2.7 accepted per pass, a 4.6 ms plain step)."""
import json
import math

import pytest

from orchard import package_card, spec_decision as sd

GOOD = {"schema": 1, "plain_step_ms": 4.6, "verify_pass_ms": 8.0, "prompt_source": "real-text",
        "sampling": {"temperature": 0.0}, "acceptance_by_kind": {"code": [3.1, 3.3], "chat": [2.7, 2.7]}}


def rec(**over):
    return {**json.loads(json.dumps(GOOD)), **over}


# ---- the decision from recorded numbers -----------------------------------------------------------------------------------

def test_the_published_example_gives_a_break_even_pass_of_12_4_ms_and_a_gain_when_the_pass_is_faster():
    out = sd.decide(rec(acceptance_by_kind={"chat": [2.7]}, verify_pass_ms=8.0))
    assert out["verdict"]["result"]["break_even_pass_ms"] == pytest.approx(12.42) and out["enable_drafter"] is True
    assert out["verdict"]["result"]["verdict"] == "gain" and out["verdict"]["kind_used"] == "chat"


def test_a_verify_pass_at_the_break_even_time_leaves_the_drafter_off():
    out = sd.decide(rec(acceptance_by_kind={"chat": [2.7]}, verify_pass_ms=12.42))
    assert out["verdict"]["result"]["verdict"] == "break-even" and out["enable_drafter"] is False


def test_a_slow_verify_pass_is_a_loss_and_the_drafter_is_correctly_off():
    out = sd.decide(rec(acceptance_by_kind={"chat": [2.7]}, verify_pass_ms=20.0))
    assert out["verdict"]["result"]["verdict"] == "loss" and out["enable_drafter"] is False
    assert "off" in out["statement"].lower() and "20.0" in out["statement"]


def test_the_verdict_uses_the_lowest_kind_and_says_which():
    out = sd.decide(rec(acceptance_by_kind={"code": [5.0], "thai": [1.0]}, verify_pass_ms=8.0))
    assert out["verdict"]["kind_used"] == "thai" and out["enable_drafter"] is False and "thai" in out["statement"]
    assert set(out["verdict"]["per_kind"]) == {"code", "thai"}


def test_a_gain_on_every_kind_turns_the_drafter_on_and_the_statement_names_the_kinds():
    out = sd.decide(rec())
    assert out["enable_drafter"] is True and "code" in out["statement"] and "chat" in out["statement"]


# ---- sampling ---------------------------------------------------------------------------------------------------------------

def test_a_measurement_at_greedy_says_the_drafter_was_measured_at_greedy_only():
    out = sd.decide(rec(sampling={"temperature": 0.0}))
    assert out["greedy_only"] is True and "greedy" in out["statement"].lower()


def test_a_measurement_at_a_sampled_setting_names_the_setting_and_is_not_greedy_only():
    out = sd.decide(rec(sampling={"temperature": 0.7, "top_p": 0.95}))
    assert out["greedy_only"] is False and "0.7" in out["statement"] and "greedy only" not in out["statement"].lower()


@pytest.mark.parametrize("sampling", [None, {}, {"temperature": "hot"}, {"temperature": -1}, {"temperature": math.inf}, {"temperature": True}, "greedy",
                                      {"temperature": 0.7, "top_p": 1.5}, {"temperature": 0.7, "top_p": 0}, {"temperature": 0.7, "top_p": "x"}])
def test_a_sampling_that_is_not_a_setting_is_refused(sampling):
    with pytest.raises(sd.SpecRecordError):
        sd.decide(rec(sampling=sampling))


# ---- the record ---------------------------------------------------------------------------------------------------------------

def test_random_token_prompts_are_refused_because_their_acceptance_misleads():
    with pytest.raises(sd.SpecRecordError, match="real text"):
        sd.decide(rec(prompt_source="random-tokens"))


@pytest.mark.parametrize("bad", [{"schema": 2}, {"schema": "1"}, {"plain_step_ms": 0}, {"plain_step_ms": -1}, {"plain_step_ms": "4.6"}, {"plain_step_ms": True},
                                 {"verify_pass_ms": math.nan}, {"verify_pass_ms": None}, {"acceptance_by_kind": {}}, {"acceptance_by_kind": {"code": []}},
                                 {"acceptance_by_kind": {"code": [-1.0]}}, {"acceptance_by_kind": {"code": ["x"]}}, {"acceptance_by_kind": [1, 2]},
                                 {"acceptance_by_kind": {"": [1.0]}}, {"prompt_source": 7}, {"prompt_source": None}])
def test_a_record_with_a_bad_field_is_refused_with_a_reason(bad):
    with pytest.raises(sd.SpecRecordError):
        sd.decide(rec(**bad))


@pytest.mark.parametrize("missing", ["schema", "plain_step_ms", "verify_pass_ms", "prompt_source", "sampling", "acceptance_by_kind"])
def test_a_record_missing_a_field_is_refused(missing):
    r = rec()
    del r[missing]
    with pytest.raises(sd.SpecRecordError, match=missing):
        sd.decide(r)


@pytest.mark.parametrize("not_a_record", [None, [], "x", 5])
def test_a_record_that_is_not_an_object_is_refused(not_a_record):
    with pytest.raises(sd.SpecRecordError):
        sd.decide(not_a_record)


def test_the_kind_names_and_the_statement_are_bounded():
    out = sd.decide(rec(acceptance_by_kind={"k" * 500: [3.0], **{f"kind{i}": [3.0] for i in range(40)}}))
    assert len(out["statement"]) <= 1500


def test_a_huge_integer_is_refused_not_crashed():
    with pytest.raises(sd.SpecRecordError):
        sd.decide(rec(plain_step_ms=10 ** 400))


# ---- reading the file ------------------------------------------------------------------------------------------------------------

def write(tmp_path, data, name=sd.MEASURE_FILE):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data if isinstance(data, str) else json.dumps(data))
    return p


def test_read_returns_the_record_and_none_when_there_is_no_file(tmp_path):
    assert sd.read_measure(tmp_path) is None
    write(tmp_path, GOOD)
    assert sd.read_measure(tmp_path) == GOOD


@pytest.mark.parametrize("text", ["{nope", "[]", "NaN", '{"a": NaN}', '{"a": 1, "a": 2}'])
def test_read_refuses_a_file_that_is_not_one_json_object(tmp_path, text):
    write(tmp_path, text)
    with pytest.raises(sd.SpecRecordError):
        sd.read_measure(tmp_path)


def test_read_refuses_a_file_over_the_bound(tmp_path, monkeypatch):
    write(tmp_path, GOOD)
    monkeypatch.setattr(sd, "MEASURE_MAX", 10)
    with pytest.raises(sd.SpecRecordError):
        sd.read_measure(tmp_path)


def test_read_refuses_a_link(tmp_path):
    real = tmp_path / "real.json"
    real.write_text(json.dumps(GOOD))
    (tmp_path / "evidence").mkdir()
    (tmp_path / sd.MEASURE_FILE).symlink_to(real)
    with pytest.raises(sd.SpecRecordError):
        sd.read_measure(tmp_path)


# ---- the supervisor records the decision ------------------------------------------------------------------------------------------

def sup_entries(tmp_path, data, *, call=2):
    from test_supervisor_stock import Rig, _supervisor
    r = Rig(tmp_path)
    try:
        r.run()
        stage = r.run_dir / "stages" / str(call)
        if data is not None:
            write(stage, data)
        sup, led = _supervisor(r)
        try:
            sup._spec_decision_record(call, stage)
            entries = led.read()
        finally:
            led.__exit__(None, None, None)
        return entries, stage
    finally:
        r.close()


def test_stage_2_with_a_measurement_writes_the_decision_file_and_two_ledger_entries(tmp_path):
    entries, stage = sup_entries(tmp_path, GOOD)
    dec = json.loads((stage / sd.DECISION_FILE).read_text())
    assert dec["enable_drafter"] is True and dec["verdict"]["result"]["verdict"] == "gain"
    decisions = [e["data"] for e in entries if e["event"] == "decision" and e["data"].get("decision") == sd.LEDGER_DECISION]
    assert len(decisions) == 1 and decisions[0]["enable_drafter"] is True and decisions[0]["kind_used"] in ("code", "chat")
    ev = [e["data"] for e in entries if e["event"] == "evidence" and e["data"].get("what") == "speculative decision"]
    assert len(ev) == 1 and ev[0]["path"].endswith("evidence/spec-decision.json") and len(ev[0]["sha256"]) == 64


def test_a_loss_is_recorded_as_the_drafter_off(tmp_path):
    entries, stage = sup_entries(tmp_path, rec(verify_pass_ms=40.0))
    d = [e["data"] for e in entries if e["event"] == "decision" and e["data"].get("decision") == sd.LEDGER_DECISION]
    assert d and d[0]["enable_drafter"] is False and d[0]["verdict"] == "loss"


def test_stage_2_without_a_measurement_records_nothing(tmp_path):
    entries, stage = sup_entries(tmp_path, None)
    assert not [e for e in entries if e["event"] in ("decision", "evidence") and "speculative" in json.dumps(e["data"])]
    assert not (stage / sd.DECISION_FILE).exists()


@pytest.mark.parametrize("data", ["{nope", rec(prompt_source="random-tokens"), rec(plain_step_ms=-1)])
def test_a_measurement_that_is_refused_is_a_ledger_decision_with_the_reason_and_never_fails_the_stage(tmp_path, data):
    entries, stage = sup_entries(tmp_path, data)
    d = [e["data"] for e in entries if e["event"] == "decision" and e["data"].get("decision") == sd.REFUSED_DECISION]
    assert len(d) == 1 and d[0]["reason"] and not (stage / sd.DECISION_FILE).exists()


# ---- the card ------------------------------------------------------------------------------------------------------------------------

def card_facts(**over):
    from test_package_card import facts
    return facts(**over)


def test_the_card_limits_carry_the_decision_sentence():
    text = sd.decide(rec())["statement"]
    card = package_card.render_card(card_facts(spec_statement=text))
    assert text in " ".join(card.split())


def test_a_card_without_a_decision_is_as_before():
    assert package_card.render_card(card_facts()) == package_card.render_card(card_facts(spec_statement=None))


def test_the_staged_package_card_carries_the_decision_sentence(tmp_path, monkeypatch, stub_tools):
    """Stage 7's card builder reads the decision stage 2 recorded: the sentence is in the staged README, in the limits."""
    import test_package_card_use_wiring as wiring
    real = wiring.make_run
    statement = sd.decide(rec(verify_pass_ms=40.0))["statement"]

    def with_decision(tmp, source):
        r = real(tmp, source)
        d = r["run"] / "stages" / "2" / sd.DECISION_FILE
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(json.dumps({"statement": statement}))
        return r
    monkeypatch.setattr(wiring, "make_run", with_decision)
    built = wiring.build(tmp_path, monkeypatch)
    assert statement in " ".join((built["pkg"] / "README.md").read_text().split())


def test_a_staged_package_without_a_decision_has_no_sentence(tmp_path, monkeypatch, stub_tools):
    import test_package_card_use_wiring as wiring
    built = wiring.build(tmp_path, monkeypatch)
    assert "Speculative decoding is" not in (built["pkg"] / "README.md").read_text().replace("Speculative decoding is off, so decode speed", "")


# ---- bounds and the stage end -------------------------------------------------------------------------------------------------------

def test_more_kinds_than_the_card_names_are_summarised_and_the_statement_is_cut_to_its_bound(monkeypatch):
    many = {f"kind{i}": [3.0] for i in range(9)}
    out = sd.decide(rec(acceptance_by_kind=many))
    assert "3 more kinds" in out["statement"]
    monkeypatch.setattr(sd, "STATEMENT_MAX", 50)
    assert len(sd.decide(rec())["statement"]) == 50


def test_a_dangling_link_in_place_of_the_record_is_refused_not_read_as_missing(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / sd.MEASURE_FILE).symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(sd.SpecRecordError):
        sd.read_measure(tmp_path)


def test_a_record_over_the_real_bound_is_refused(tmp_path):
    write(tmp_path, "{}" + " " * (1024 * 1024))                # valid JSON, two bytes over a 1 MiB bound
    with pytest.raises(sd.SpecRecordError):
        sd.read_measure(tmp_path)


@pytest.mark.parametrize("statement", ["", "x" * 5000, 7, None, ["a"]])
def test_the_card_sentence_is_read_back_only_when_it_is_a_bounded_string(tmp_path, statement):
    write(tmp_path / "stages" / "2", {"statement": statement}, name=sd.DECISION_FILE)
    assert sd.statement_from_run(tmp_path) is None


def test_the_card_sentence_is_read_back_from_stage_2(tmp_path):
    write(tmp_path / "stages" / "2", {"statement": "Speculative decoding is off: x"}, name=sd.DECISION_FILE)
    assert sd.statement_from_run(tmp_path) == "Speculative decoding is off: x"
    assert sd.statement_from_run(tmp_path / "missing") is None


def test_a_run_whose_stage_2_passes_with_a_measurement_records_the_decision_at_the_stage_end(tmp_path):
    """Through the stage machine: the pass of stage 2 triggers the record."""
    from run_fakes import FILES, bringup
    from test_supervisor import Rig
    r = Rig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides={(2, "finish"): {**FILES[(2, "finish")], sd.MEASURE_FILE: GOOD}})
        r.run()
        d = [e["data"] for e in r.entries() if e["event"] == "decision" and e["data"].get("decision") == sd.LEDGER_DECISION]
        assert len(d) == 1 and d[0]["enable_drafter"] is True and r.ends(2)[0]["result"] == "pass"
        assert (r.run_dir / "stages" / "2" / sd.DECISION_FILE).is_file()
    finally:
        r.close()
