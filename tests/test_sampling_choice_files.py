# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The files behind the sampling choice (orchard/sampling_choice.py): reading a swap check into a side, the recorded decision, and the stage 2 gate
check that recomputes it from the files on disk. A forged decision file is refused."""
import json

import pytest

from orchard import sampling_choice as sc
from orchard.ledger import Ledger

GREEDY = "The lighthouse keeper climbed the stairs one more time that night and found the lamp already burning, though nobody had lit it."
STORY = "The lighthouse keeper climbed the long spiral stairs every evening, counting the steps and listening to the sea below him."
OTHER = "On the island there was a bell that rang by itself whenever the fog came in, and nobody remembered who had hung it there."


def req(text, seconds):
    return {"params": {}, "accepted": True, "error": None, "seconds": seconds, "completion_tokens": 96, "text": text,
            "finish_reason": "length", "drafter_log": ""}


def probe(ignored: bool, seconds=2.0):
    """A sampling probe record: ignored means every sampled request returned the greedy text."""
    a, b = (GREEDY, GREEDY) if ignored else (STORY, OTHER)
    return {"ran": True, "prompt": "p", "max_tokens": 96,
            "requests": {"greedy": req(GREEDY, seconds), "sampled_a": req(a, seconds), "sampled_b": req(b, seconds)}}


def report(drafter="on (dflash2)", ignored=True, seconds=2.0, serves=True, with_probe=True):
    r = {"serving": {"drafter": drafter, "sampling": "host"}, "result_draft": {"serves": serves}}
    if with_probe:
        r["sampling_probe"] = probe(ignored, seconds)
    return r


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def make_run(tmp_path, *, needed=True, stage2=None, variant=None, override="auto", graft=None):
    run = tmp_path / "run"
    s2 = run / "stages" / "2"
    write(run / "stages/0/evidence/genconfig-license.json",
          {"model_generation_config": {"temperature": 0.6} if needed else {}, "model_generation_config_present": needed,
           "card_sampling_hints": {}, "card_warns_greedy": []})
    write(s2 / "evidence/swap-check.json", stage2 or report())
    if variant is not None:
        write(s2 / sc.TRIAL_DIR / "variant/evidence/swap-check.json", variant)
    if graft is not None:
        write(s2 / "graft-decision.json", {"choice": graft["choice"], "reason": "r"})
        if "report" in graft:
            write(s2 / "graft-trial/variant/evidence/swap-check.json", graft["report"])
    Ledger(run / "ledger.jsonl").append("run_start", None, model="x/y", **({"sampling": override} if override else {}))
    return run, s2


def save(run, s2, override="auto"):
    cur = sc.served_side(s2)
    var = sc.variant_side(s2)
    d = sc.decide(sc.need(run, override), cur, var)
    sc.write_decision(s2, d, trial=var is not None)
    return d


# ---- sides ----------------------------------------------------------------------------------------------------------------------

def test_a_side_comes_from_the_raw_probe_requests_and_the_serving_record():
    s = sc.side_from_report(report(ignored=True, seconds=2.0))
    assert s == {"drafter": "on", "verdict": "sampled-ignored", "greedy_tok_s": pytest.approx(48.0), "sampled_tok_s": pytest.approx(48.0),
                 "served": True}
    off = sc.side_from_report(report(drafter="off", ignored=False, seconds=6.0))
    assert off["drafter"] == "off" and off["verdict"] == "sampled-full-speed" and off["sampled_tok_s"] == pytest.approx(16.0)


@pytest.mark.parametrize("text,expect", [("off", "off"), ("on (dflash2)", "on"), ("not set by the run script", None), (None, None), (5, None)])
def test_drafter_state_is_read_from_the_serving_text(text, expect):
    assert sc.drafter_of({"drafter": text}) == expect


def test_a_report_without_a_probe_has_the_not_run_verdict_and_no_speeds():
    s = sc.side_from_report(report(with_probe=False))
    assert s["verdict"] == "not-run" and s["greedy_tok_s"] is None and s["sampled_tok_s"] is None


def test_a_serves_false_report_says_not_served():
    assert sc.side_from_report(report(serves=False))["served"] is False


@pytest.mark.parametrize("bad", [None, [], "x", 3])
def test_a_report_that_is_not_an_object_gives_no_side(bad):
    assert sc.side_from_report(bad) is None


def test_a_summary_field_in_the_record_is_not_believed():
    r = report(ignored=True)
    r["sampling_probe"]["verdict"] = "sampled-full-speed"
    r["sampling_probe"]["requests"]["greedy"]["tokens_per_s"] = 9999
    s = sc.side_from_report(r)
    assert s["verdict"] == "sampled-ignored" and s["greedy_tok_s"] == pytest.approx(48.0)


def test_the_served_side_is_the_graft_variants_when_the_graft_was_chosen(tmp_path):
    run, s2 = make_run(tmp_path, graft={"choice": "graft", "report": report(ignored=False, seconds=1.6)})
    s = sc.served_side(s2)
    assert s["drafter"] == "on" and s["verdict"] == "sampled-full-speed" and s["greedy_tok_s"] == pytest.approx(60.0)


def test_the_served_side_is_stage_2s_own_when_the_graft_was_not_chosen(tmp_path):
    run, s2 = make_run(tmp_path, graft={"choice": "safe", "report": report(ignored=False)})
    assert sc.served_side(s2)["verdict"] == "sampled-ignored"


# ---- the override --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["auto", "needed", "not-needed"])
def test_the_override_is_read_from_the_ledgers_run_start(tmp_path, value):
    run, _ = make_run(tmp_path, override=value)
    assert sc.read_override(run) == value


def test_a_run_start_without_the_key_or_with_a_bad_value_reads_as_auto(tmp_path):
    run, _ = make_run(tmp_path, override=None)
    assert sc.read_override(run) == "auto"
    run2, _ = make_run(tmp_path / "b", override="sometimes")
    assert sc.read_override(run2) == "auto"


def test_no_ledger_reads_as_auto(tmp_path):
    assert sc.read_override(tmp_path) == "auto"


# ---- the decision file --------------------------------------------------------------------------------------------------------------

def test_the_decision_is_written_whole_and_read_back(tmp_path):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    d = save(run, s2)
    assert d["choice"] == "drafter-off"
    back = sc.read_decision(s2)
    assert back["choice"] == "drafter-off" and back["trial"] == "sampling-trial/trial-result.json"
    assert not (s2 / "sampling-decision.json.tmp").exists()
    assert sc.drafter_off_chosen(run) is True


def test_a_decision_with_no_trial_records_none(tmp_path):
    run, s2 = make_run(tmp_path, needed=False)
    save(run, s2)
    assert sc.read_decision(s2)["trial"] is None and sc.drafter_off_chosen(run) is False


@pytest.mark.parametrize("content", ["{not json", "[]", '{"choice": "maybe"}', '"keep"'])
def test_a_missing_or_malformed_decision_chooses_nothing(tmp_path, content):
    run, s2 = make_run(tmp_path)
    (s2 / sc.DECISION_FILE).write_text(content, encoding="utf-8")
    assert sc.read_decision(s2) is None and sc.drafter_off_chosen(run) is False


def test_a_missing_decision_file_chooses_nothing(tmp_path):
    run, s2 = make_run(tmp_path)
    assert sc.read_decision(s2) is None and sc.drafter_off_chosen(run) is False


# ---- the gate ----------------------------------------------------------------------------------------------------------------------

def test_no_decision_file_is_no_problem(tmp_path):
    run, s2 = make_run(tmp_path)
    assert sc.decision_problems(s2, run) == []


def test_an_honest_drafter_off_decision_passes(tmp_path):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    save(run, s2)
    assert sc.decision_problems(s2, run) == []


def test_an_honest_keep_decision_passes(tmp_path):
    run, s2 = make_run(tmp_path, needed=False)
    save(run, s2)
    assert sc.decision_problems(s2, run) == []


def test_a_decision_that_says_keep_while_the_files_say_drafter_off_is_refused(tmp_path):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    d = save(run, s2)
    d["choice"] = "keep"
    sc.write_decision(s2, d, trial=True)
    assert any("drafter-off" in p for p in sc.decision_problems(s2, run))


def test_a_decision_that_says_drafter_off_with_no_variant_on_disk_is_refused(tmp_path):
    run, s2 = make_run(tmp_path)
    d = sc.decide(sc.need(run), sc.served_side(s2), None)
    d.update(choice="drafter-off", flag=None)
    sc.write_decision(s2, d, trial=True)
    problems = sc.decision_problems(s2, run)
    assert any("does not show it serving" in p for p in problems)


def test_a_decision_that_hides_its_flag_is_refused(tmp_path):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=True, seconds=2.0))      # the variant ignores sampling too
    d = save(run, s2)
    assert d["flag"] == "no-variant-serves-purpose"
    d["flag"] = None
    sc.write_decision(s2, d, trial=True)
    assert any("flag" in p for p in sc.decision_problems(s2, run))


def test_a_decision_made_under_another_override_is_refused(tmp_path):
    run, s2 = make_run(tmp_path, needed=True, override="not-needed")
    d = sc.decide(sc.need(run, "needed"), sc.served_side(s2), None)
    sc.write_decision(s2, d, trial=False)
    assert any("override" in p for p in sc.decision_problems(s2, run))


def test_a_decision_that_ignores_the_files_evidence_of_need_is_refused(tmp_path):
    run, s2 = make_run(tmp_path, needed=True, variant=report(drafter="off", ignored=False, seconds=6.0))
    d = sc.decide(sc.need(tmp_path / "nowhere"), sc.served_side(s2), None)          # decided as if no evidence existed
    sc.write_decision(s2, d, trial=False)
    assert any("keep" in p and "drafter-off" in p for p in sc.decision_problems(s2, run))


def test_a_decision_made_from_a_swap_check_that_is_gone_cannot_be_checked(tmp_path):
    run, s2 = make_run(tmp_path, needed=False)
    save(run, s2)
    (s2 / "evidence/swap-check.json").unlink()
    assert any("cannot be checked" in p for p in sc.decision_problems(s2, run))


@pytest.mark.parametrize("edit,word", [({"flag": "loud"}, "flag"), ({"reason": ""}, "reason"), ({"choice": "x"}, "choice")])
def test_a_malformed_decision_is_refused(tmp_path, edit, word):
    run, s2 = make_run(tmp_path, needed=False)
    d = save(run, s2)
    d.update(edit)
    (s2 / sc.DECISION_FILE).write_text(json.dumps(d), encoding="utf-8")
    assert any(word in p for p in sc.decision_problems(s2, run))


def test_a_decision_file_that_is_not_an_object_is_refused(tmp_path):
    run, s2 = make_run(tmp_path)
    (s2 / sc.DECISION_FILE).write_text("[1]", encoding="utf-8")
    assert any("not a JSON object" in p for p in sc.decision_problems(s2, run))


def test_a_variant_note_is_kept_when_the_trial_was_skipped(tmp_path):
    run, s2 = make_run(tmp_path)
    d = sc.decide(sc.need(run), sc.served_side(s2), None, variant_note="the run uses a lab")
    d["variant_note"] = "the run uses a lab"
    sc.write_decision(s2, d, trial=False)
    assert d["flag"] == "variant-not-tried" and sc.decision_problems(s2, run) == []


def test_a_decision_after_a_chosen_graft_is_judged_on_the_grafted_configuration(tmp_path):
    run, s2 = make_run(tmp_path, graft={"choice": "graft", "report": report(ignored=False, seconds=1.6)})
    d = save(run, s2)
    assert d["choice"] == "keep" and d["flag"] is None and d["numbers"]["served_drafter"] == "on"
    assert sc.decision_problems(s2, run) == []


def test_an_unreadable_ledger_reads_as_auto(tmp_path):
    run, _ = make_run(tmp_path, override="needed")
    (run / "ledger.jsonl").write_text("{broken\n", encoding="utf-8")
    assert sc.read_override(run) == "auto"
