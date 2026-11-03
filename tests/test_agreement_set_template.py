# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The agreement set in serve_and_compare.py (issue 15): the template teacher-forces six prompts against a fake server, records the raw ids,
and the stage gate recomputes the verdict from them. A run with every fourth token wrong (the base weights by mistake, as the fake shows it)
must fail the gate; a good run must pass; a record scored against a reference other than stage 1's must fail it."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weights_swap_templates import NEAREST, TEMPLATES, swap, report  # noqa: E402,F401  (swap is a fixture)

from orchard import stages  # noqa: E402

IDS = ["chat-explain", "code-reverse", "tool-call-json", "non-english-german", "non-english-japanese", "long-context-needle"]
REF_REL = ("stages", "1", "evidence", "reference", "agreement-reference.json")


def make_sets(n=6, wrong_every=None):
    """n prompts with ids that share no prefix with each other or with the main prompt [1..5]; 32 distinct-looking reference ids each."""
    sets = []
    for i in range(n):
        sets.append({"id": IDS[i] if i < len(IDS) else f"extra-{i}", "kind": "chat", "prompt_ids": [30 + i, 30 + i, 30 + i, i + 1],
                     "generated_ids": [(j + i) % 33 + 1 for j in range(32)], "wrong_every": wrong_every})
    return sets


def record_reference(run, path):
    """What Supervisor._record_agreement_reference writes: the sha256 of stage 1's reference file, in the ledger, before the stage 2 test."""
    from orchard.ledger import Ledger
    led = Ledger(run / "ledger.jsonl")
    try:
        led.append("decision", 2, decision=stages.AGREEMENT_LEDGER_DECISION, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        led.close()


def install(swap, sets, *, set_sha=None, ledger=True):
    """Copy the helper and the prompt set next to the script and write the stage 1 reference file for `sets`."""
    stage, run = swap["stage"], swap["run"]
    shutil.copy(TEMPLATES / "agreement_core.py", stage)
    shutil.copy(TEMPLATES / "agreement_prompts.json", stage)
    sha = set_sha or hashlib.sha256((TEMPLATES / "agreement_prompts.json").read_bytes()).hexdigest()
    ref = {"prompt_set_sha256": sha,
           "entries": [{"id": s["id"], "kind": s["kind"], "prompt_ids": s["prompt_ids"], "generated_ids": s["generated_ids"]} for s in sets]}
    path = run.joinpath(*REF_REL)
    path.write_text(json.dumps(ref))
    if ledger:
        record_reference(run, path)
    (run / "stages" / "0").mkdir(parents=True, exist_ok=True)
    (run / "stages" / "0" / "delta.json").write_text(json.dumps({"nearest_model": NEAREST}))     # the gate asks stage 0 for the nearest model
    return path


def server_sets(sets):
    return [{"prompt_ids": s["prompt_ids"], "generated_ids": s["generated_ids"], "wrong_every": s["wrong_every"]} for s in sets]


def gate_reasons(swap):
    return stages._suite_reasons_in(swap["stage"] / "evidence" / "swap-check.json", swap["run"])


def write_control(swap, path, model_id=NEAREST):
    """Save the last run's swap-check.json as a control, with the model id it would have if the nearest model had been served."""
    rep = json.loads((swap["stage"] / "evidence" / "swap-check.json").read_text())
    rep["agreement_set"]["model_id"] = model_id
    path.write_text(json.dumps(rep))


def run(swap, sets, **overrides):
    return swap["start"]("perfect", server_extra={"agreement_sets": server_sets(sets)}, agreement_set=True, **overrides)


def test_a_good_run_records_every_prompt_and_passes_the_gate(swap):
    sets = make_sets()
    install(swap, sets)
    r = run(swap, sets)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = report(swap)["agreement_set"]
    assert rec["ran"] is True and len(rec["entries"]) == 6 and rec["n_tokens"] == 32
    assert all(e["chip_ids"] == e["reference_ids"] and e["error"] is None for e in rec["entries"])
    assert rec["baseline"] == {"check": "32-token teacher-forced, one prompt", "top1_agreement": 1.0, "n_tokens": 32}
    assert gate_reasons(swap) == []


def test_every_fourth_token_wrong_fails_the_gate(swap):
    sets = make_sets(wrong_every=4)
    install(swap, sets)
    assert run(swap, sets).returncode == 0
    rec = report(swap)["agreement_set"]
    assert all(sum(a != b for a, b in zip(e["chip_ids"], e["reference_ids"])) == 8 for e in rec["entries"])
    reasons = gate_reasons(swap)
    assert any("pooled top-1" in r for r in reasons), reasons


def test_one_prompt_that_drifts_fails_the_gate_when_the_others_are_perfect(swap):
    sets = make_sets(n=8)
    sets[5]["wrong_every"] = 1                    # every position of prompt p5 is wrong
    install(swap, sets)
    assert run(swap, sets).returncode == 0
    reasons = gate_reasons(swap)
    assert any(f"'{IDS[5]}'" in r and "per-prompt bar" in r for r in reasons), reasons


def test_the_run_without_the_agreement_flag_records_no_set(swap):
    r = swap["start"]("perfect")
    assert r.returncode == 0
    assert "agreement_set" not in report(swap)
    assert gate_reasons(swap) == []


def test_a_missing_reference_file_exits_2_before_any_server_starts(swap):
    sets = make_sets()
    path = install(swap, sets)
    path.unlink()
    r = run(swap, sets)
    assert r.returncode == 2 and "cannot read the agreement reference" in r.stderr
    assert not swap["pid_file"].exists()


def test_a_reference_made_for_another_prompt_set_exits_2(swap):
    sets = make_sets()
    install(swap, sets, set_sha="0" * 64)
    r = run(swap, sets)
    assert r.returncode == 2 and "made for prompt set" in r.stderr
    assert not swap["pid_file"].exists()


def test_a_missing_helper_or_prompt_file_exits_2(swap):
    sets = make_sets()
    install(swap, sets)
    (swap["stage"] / "agreement_prompts.json").unlink()
    r = run(swap, sets)
    assert r.returncode == 2 and "agreement_prompts.json" in r.stderr


def test_a_record_scored_against_other_reference_ids_fails_the_gate(swap):
    sets = make_sets()
    install(swap, sets)
    assert run(swap, sets).returncode == 0
    ref_path = swap["run"].joinpath(*REF_REL)
    changed = json.loads(ref_path.read_text())
    changed["entries"][0]["generated_ids"][0] += 1             # stage 1's file is not the one the record used
    ref_path.write_text(json.dumps(changed))
    reasons = gate_reasons(swap)
    assert any("other than the one stage 1 wrote" in r for r in reasons) and any(f"'{IDS[0]}'" in r and "not stage 1's" in r for r in reasons), reasons


def test_a_record_with_a_prompt_set_other_than_the_shipped_one_fails_the_gate(swap):
    sets = make_sets()
    install(swap, sets)
    assert run(swap, sets).returncode == 0
    path = swap["stage"] / "evidence" / "swap-check.json"
    rep = json.loads(path.read_text())
    rep["agreement_set"]["prompt_set_sha256"] = "f" * 64
    path.write_text(json.dumps(rep))
    assert any("not the one the repository ships" in r for r in gate_reasons(swap))


def test_a_record_without_a_stage_1_reference_file_fails_the_gate(swap):
    sets = make_sets()
    install(swap, sets)
    assert run(swap, sets).returncode == 0
    swap["run"].joinpath(*REF_REL).unlink()
    assert any("cannot be read" in r for r in gate_reasons(swap))


def test_a_control_that_agrees_equally_does_not_separate(swap, tmp_path):
    sets = make_sets()
    install(swap, sets)
    assert run(swap, sets).returncode == 0
    control = tmp_path / "control.json"
    write_control(swap, control)                                   # an identical run is no control
    assert run(swap, sets, agreement_control=str(control)).returncode == 0
    assert any("cannot tell" in r for r in gate_reasons(swap))


def test_a_nearest_model_control_that_agrees_less_lets_the_new_model_pass(swap, tmp_path):
    sets = make_sets()
    install(swap, sets)
    assert run(swap, make_sets(wrong_every=2)).returncode == 0           # the nearest model serves and agrees on half the tokens
    control = tmp_path / "weak.json"
    write_control(swap, control)
    assert run(swap, sets, agreement_control=str(control)).returncode == 0
    rec = report(swap)["agreement_set"]
    assert rec["control"]["entries"] and rec["control"]["sha256"] == hashlib.sha256(control.read_bytes()).hexdigest()
    assert gate_reasons(swap) == []



def test_a_control_served_from_another_model_exits_2_and_the_model_id_is_embedded(swap, tmp_path):
    sets = make_sets()
    install(swap, sets)
    assert run(swap, make_sets(wrong_every=2)).returncode == 0
    control = tmp_path / "other.json"
    write_control(swap, control, model_id="Someone/Else")
    r = run(swap, sets, agreement_control=str(control))
    assert r.returncode == 2 and "not a control of the nearest model" in r.stderr
    write_control(swap, control)
    assert run(swap, sets, agreement_control=str(control)).returncode == 0
    assert report(swap)["agreement_set"]["control"]["model_id"] == NEAREST


@pytest.mark.parametrize("change", [lambda e: e.update(generated_ids=["bad"]), lambda e: e.update(generated_ids=[]),
                                    lambda e: e.update(prompt_ids=[True]), lambda e: e.update(generated_ids=[-1] * 40),
                                    lambda e: e.update(id=["x"])])
def test_a_malformed_reference_entry_exits_2_before_any_server_starts(swap, change):
    sets = make_sets()
    path = install(swap, sets)
    ref = json.loads(path.read_text())
    change(ref["entries"][0])
    path.write_text(json.dumps(ref))
    r = run(swap, sets)
    assert r.returncode == 2 and "cannot read the agreement reference" in r.stderr
    assert not swap["pid_file"].exists()


def test_a_control_for_other_reference_ids_exits_2(swap, tmp_path):
    sets = make_sets()
    install(swap, sets)
    assert run(swap, sets).returncode == 0
    rep = json.loads((swap["stage"] / "evidence" / "swap-check.json").read_text())
    rep["agreement_set"]["entries"][0]["reference_ids"][0] += 1
    rep["agreement_set"]["model_id"] = NEAREST
    control = tmp_path / "control.json"
    control.write_text(json.dumps(rep))
    r = run(swap, sets, agreement_control=str(control))
    assert r.returncode == 2 and "other reference ids" in r.stderr


def test_a_run_that_compares_fewer_tokens_than_the_minimum_fails_the_gate(swap):
    sets = make_sets()
    install(swap, sets)
    r = run(swap, sets, agreement_tokens=16)
    assert r.returncode == 0
    assert all(len(e["reference_ids"]) == 16 for e in report(swap)["agreement_set"]["entries"])
    assert any("compared 16 tokens" in x for x in gate_reasons(swap))


def test_the_template_reader_refuses_a_fifo_a_link_and_an_oversize_file(tmp_path, monkeypatch):
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("sac_under_test", TEMPLATES / "serve_and_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fifo = tmp_path / "f"
    os.mkfifo(fifo)
    target = tmp_path / "t.json"
    target.write_text("{}")
    link = tmp_path / "l.json"
    link.symlink_to(target)
    for bad in (fifo, link):
        with pytest.raises(OSError):
            mod.read_small_file(bad)
    assert mod.read_small_file(target) == b"{}"
    monkeypatch.setattr(mod, "AGREEMENT_FILE_MAX_BYTES", 1)
    with pytest.raises(OSError):
        mod.read_small_file(target)


def test_a_refused_request_marks_the_rest_of_that_prompt_as_a_mismatch_and_the_run_goes_on(swap):
    sets = make_sets(n=7)
    install(swap, sets)
    served = server_sets(sets)
    served[2]["fail_at"] = 10
    r = swap["start"]("perfect", server_extra={"agreement_sets": served}, agreement_set=True)
    assert r.returncode == 0, r.stdout + r.stderr
    e = report(swap)["agreement_set"]["entries"][2]
    assert e["chip_ids"][:10] == e["reference_ids"][:10] and e["chip_ids"][10:] == [None] * 22 and "HTTP 500" in e["error"]
    assert any(f"'{IDS[2]}'" in x for x in gate_reasons(swap))


# ---- the gate's reading of the record (Copilot review of PR 394) ------------------------------------------------------------------------

def good_run(swap, **overrides):
    sets = make_sets()
    install(swap, sets)
    assert run(swap, sets, **overrides).returncode == 0
    return sets


def edit_report(swap, change):
    path = swap["stage"] / "evidence" / "swap-check.json"
    rep = json.loads(path.read_text())
    change(rep)
    path.write_text(json.dumps(rep))


def test_a_reference_the_supervisor_never_recorded_in_the_ledger_is_refused(swap):
    sets = make_sets()
    install(swap, sets, ledger=False)
    assert run(swap, sets).returncode == 0
    assert any("no record of stage 1's agreement reference" in r for r in gate_reasons(swap))


def test_a_reference_replaced_together_with_the_recorded_hash_is_refused(swap):
    good_run(swap)
    ref_path = swap["run"].joinpath(*REF_REL)
    changed = json.loads(ref_path.read_text())
    changed["entries"][0]["generated_ids"][0] += 1
    ref_path.write_text(json.dumps(changed))
    new_sha = hashlib.sha256(ref_path.read_bytes()).hexdigest()

    def change(rep):
        rep["agreement_set"]["reference_sha256"] = new_sha
        rep["agreement_set"]["entries"][0]["reference_ids"][0] = changed["entries"][0]["generated_ids"][0]
    edit_report(swap, change)
    reasons = gate_reasons(swap)
    assert any("not the one the supervisor recorded" in r for r in reasons), reasons


def test_five_copies_of_one_prompt_are_refused(swap):
    good_run(swap)

    def change(rep):
        for e in rep["agreement_set"]["entries"]:
            e["id"] = IDS[0]
    edit_report(swap, change)
    reasons = gate_reasons(swap)
    assert any("more than once" in r for r in reasons) and any("missing" in r for r in reasons), reasons


def test_a_reference_file_that_repeats_a_prompt_makes_the_template_exit_2(swap):
    sets = make_sets()
    for s_ in sets[1:]:
        s_["id"] = sets[0]["id"]
    install(swap, sets)
    r = run(swap, sets)
    assert r.returncode == 2 and "unique" in r.stderr


def test_a_reference_file_that_repeats_a_prompt_or_lacks_one_is_refused_at_the_gate(swap):
    good_run(swap)
    ref_path = swap["run"].joinpath(*REF_REL)
    changed = json.loads(ref_path.read_text())
    changed["entries"][1]["id"] = changed["entries"][0]["id"]
    ref_path.write_text(json.dumps(changed))
    reasons = gate_reasons(swap)
    assert any("stage 1's agreement reference" in r and "more than once" in r for r in reasons), reasons
    assert any("stage 1's agreement reference" in r and "missing" in r for r in reasons), reasons


def test_a_record_with_a_missing_shipped_prompt_is_refused(swap):
    good_run(swap)
    edit_report(swap, lambda rep: rep["agreement_set"]["entries"].pop())
    assert any("missing from the agreement set" in r for r in gate_reasons(swap))


def test_a_control_served_from_another_model_is_refused_at_the_gate_too(swap, tmp_path):
    good_run(swap)
    control = tmp_path / "c.json"
    write_control(swap, control)
    assert run(swap, make_sets(), agreement_control=str(control)).returncode == 0
    edit_report(swap, lambda rep: rep["agreement_set"]["control"].update(model_id="Someone/Else"))
    assert any("not a control of the nearest model" in r for r in gate_reasons(swap))


def test_an_edited_control_with_other_reference_ids_is_refused_at_the_gate(swap, tmp_path):
    good_run(swap)
    control = tmp_path / "c.json"
    write_control(swap, control)
    assert run(swap, make_sets(), agreement_control=str(control)).returncode == 0
    edit_report(swap, lambda rep: rep["agreement_set"]["control"]["entries"][0]["reference_ids"].__setitem__(0, 999))
    reasons = gate_reasons(swap)
    assert any("control" in r and "reference ids" in r for r in reasons), reasons


def test_an_incomplete_control_is_refused_at_the_gate(swap, tmp_path):
    good_run(swap)
    control = tmp_path / "c.json"
    write_control(swap, control)
    assert run(swap, make_sets(wrong_every=2), agreement_control=str(control)).returncode == 0
    edit_report(swap, lambda rep: rep["agreement_set"]["control"]["entries"].pop())
    assert any("control cannot be used" in r for r in gate_reasons(swap))


@pytest.mark.parametrize("change", [lambda rep: rep.pop("agreement_set"), lambda rep: rep["agreement_set"].update(ran=False),
                                    lambda rep: rep.update(agreement_set="x"), lambda rep: rep.update(agreement_set=None)])
def test_a_run_that_asked_for_the_set_needs_a_completed_record(swap, change):
    good_run(swap)
    edit_report(swap, change)
    assert any("asks for the agreement set" in r for r in gate_reasons(swap))


def test_a_missing_swap_check_when_the_set_was_asked_for_is_a_reason(swap):
    good_run(swap)
    (swap["stage"] / "evidence" / "swap-check.json").unlink()
    assert any("asks for the agreement set" in r for r in gate_reasons(swap))


@pytest.mark.parametrize("change", [lambda rep: rep["agreement_set"].update(entries=None), lambda rep: rep["agreement_set"].update(entries=[{"id": ["x"]}]),
                                    lambda rep: rep["agreement_set"].update(control={"entries": None}), lambda rep: rep["agreement_set"].update(control=7)])
def test_malformed_record_values_are_reasons_and_never_exceptions(swap, change):
    good_run(swap)
    edit_report(swap, change)
    assert gate_reasons(swap)


def test_a_malformed_reference_file_is_a_reason_at_the_gate(swap):
    good_run(swap)
    ref_path = swap["run"].joinpath(*REF_REL)
    ref_path.write_text(json.dumps({"entries": None}))
    assert gate_reasons(swap)
    ref_path.write_text(json.dumps({"entries": [{"id": ["x"]}]}))
    assert gate_reasons(swap)


def test_the_run_that_did_not_ask_for_the_set_is_not_judged_on_it(swap):
    assert swap["start"]("perfect").returncode == 0
    assert gate_reasons(swap) == []
