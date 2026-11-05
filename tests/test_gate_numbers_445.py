# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Four gates used to read a number an agent typed (issue 445). This file pins what each one does now.

* The sidecar parity object (stage 2, weights+sidecar): recomputed from `per_record` in `evidence/sidecar-parity.json`.
* The stage 7 boot check (it feeds the card): recomputed from `teacher_forced` in `verify/evidence/verify.json` against stage 1's reference ids.
* The stage 2 decoder PCC and the stage 3 `top1` have no raw evidence today. The gate says so (`GateResult.unverified`).

Bound to the ledger like the 32-token check of issue 433: the supervisor hashes the raw file when the hardware test ends (`raw_evidence`),
a bound run must carry the raw records, and a run with no such record is judged as before. No device, no server, no model."""
import copy
import hashlib
import json
import math

import pytest

from orchard import raw_evidence, sidecar_numbers, stages, token_check
from orchard.ledger import Ledger
from orchard.stages import gate_decoder, gate_full_model, gate_weights_swap_sidecar
from test_stages import stage, write
from test_stages_class import PARITY, RESULT, SIDE_DELTA

SP = "stages/2/evidence/sidecar-parity.json"


def rec(i, pcc=0.998, diff=0.004, n=3, agree=3):
    """One record as hidden_parity.py writes it (`per_record` entries: id, tokens, hidden_pcc, prob_max_abs_diff, top1_agree, n_questions)."""
    return {"id": f"r{i}", "tokens": 40 + i, "hidden_pcc": pcc, "prob_max_abs_diff": diff, "top1_agree": agree, "n_questions": n,
            "questions": [f"q{i}-{j}" for j in range(n)]}


RECORDS = [rec(0), rec(1, pcc=0.9985, diff=0.006), rec(2, n=4, agree=4), rec(3, pcc=0.9975, diff=0.003, n=3, agree=3),
           rec(4, n=2, agree=2), rec(5, n=2, agree=2)]          # 17 questions in six records, like PARITY


def parity_doc(records=RECORDS, **over):
    """What hidden_parity.aggregate gives (the schema of PARITY plus per_record), so the numbers are the template's own."""
    pccs = [r["hidden_pcc"] for r in records]
    nq = sum(r["n_questions"] for r in records)
    doc = {**PARITY, "n_records": len(records), "n_questions": nq, "hidden_pcc_min": min(pccs),
           "hidden_pcc_mean": sum(pccs) / len(pccs), "prob_max_abs_diff": max(r["prob_max_abs_diff"] for r in records),
           "top1_agree_fraction": sum(r["top1_agree"] for r in records) / nq, "per_record": copy.deepcopy(records)}
    doc.update(over)
    return doc


def sidecar_stage(tmp_path, evidence=None, result_sp=None, bind=False, evidence_text=None):
    """Stage 2 of a weights+sidecar run: result.json carries `result_sp` (default: the evidence's numbers) and the evidence file is
    `evidence`. `bind` writes the ledger entry the supervisor appends after the hardware test."""
    ev = parity_doc() if evidence is None else evidence
    sp = {k: v for k, v in ev.items() if k != "per_record"} if result_sp is None else result_sp
    run = tmp_path / "run"
    write(run, "stages/0/delta.json", SIDE_DELTA)
    path = write(run, SP, evidence_text if evidence_text is not None else ev)
    if bind:
        led = Ledger(run / "ledger.jsonl")
        try:
            led.append("evidence", 2, what=raw_evidence.RAW_EVIDENCE_WHAT, path=SP, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        finally:
            led.close()
    sd, run = stage(tmp_path, 2, "result.json", {**RESULT, "sidecar_parity": sp})
    return gate_weights_swap_sidecar(sd, run), run


def mentions(reasons, *words):
    return [r for r in reasons if all(w in r for w in words)]


# ---- the recompute matches the template's own aggregate -------------------------------------------------------------------------------

def test_the_recompute_gives_what_the_template_aggregates():
    got = sidecar_numbers.recompute(RECORDS)
    want = parity_doc()
    assert got == {k: want[k] for k in ("n_records", "n_questions", "hidden_pcc_min", "hidden_pcc_mean", "prob_max_abs_diff",
                                        "top1_agree_fraction")}


def test_the_recompute_equals_hidden_parity_aggregate():
    """The number the gate recomputes is the number the real template writes (torch is needed only to import the template)."""
    pytest.importorskip("torch")
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "orchard/skills/sidecar-parity-templates/hidden_parity.py"
    spec = importlib.util.spec_from_file_location("hp445", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    agg = mod.aggregate(copy.deepcopy(RECORDS), {"prob_max_abs_diff": 0.002}, {"cosine": 0.9999, "passed": True}, {})
    got = sidecar_numbers.recompute(agg["per_record"])
    assert all(math.isclose(got[k], agg[k], abs_tol=1e-12) for k in got)


# ---- sidecar parity: the unedited record, the edits, the binding ----------------------------------------------------------------------

def test_an_unedited_sidecar_record_passes_with_nothing_unverified(tmp_path):
    for bind in (False, True):
        g, _ = sidecar_stage(tmp_path / str(bind), bind=bind)
        assert g.ok, g.reasons
        assert g.unverified == ()


def test_raising_the_numbers_in_result_json_and_the_evidence_together_is_refused(tmp_path):
    """The edit: hidden_pcc_min 0.9999 and top1 1.0 in both files. The six records still give 0.9975 and 1.0."""
    ev = parity_doc(hidden_pcc_min=0.99999)
    g, _ = sidecar_stage(tmp_path, evidence=ev)
    assert not g.ok
    assert mentions(g.reasons, "hidden_pcc_min", "per_record", "0.9975")


def test_a_failing_record_hidden_behind_a_raised_summary_is_refused(tmp_path):
    """One record has a PCC of 0.9 and the summary says 0.998 in both files: the gate's bar is 0.99, the recompute gives 0.9."""
    records = [rec(0), rec(1, pcc=0.9), rec(2), rec(3), rec(4), rec(5)]
    ev = parity_doc(records, hidden_pcc_min=0.998)
    g, _ = sidecar_stage(tmp_path, evidence=ev)
    assert not g.ok
    assert mentions(g.reasons, "hidden_pcc_min", "0.9")


def test_a_result_json_that_differs_from_the_evidence_summary_is_refused(tmp_path):
    ev = parity_doc()
    g, _ = sidecar_stage(tmp_path, evidence=ev, result_sp={**{k: v for k, v in ev.items() if k != "per_record"}, "top1_agree_fraction": 1.0,
                                                          "prob_max_abs_diff": 0.001})
    assert not g.ok and mentions(g.reasons, "prob_max_abs_diff")


def test_each_summary_number_is_checked(tmp_path):
    for field, bad in (("n_questions", 40), ("hidden_pcc_min", 0.9999), ("prob_max_abs_diff", 0.0001), ("top1_agree_fraction", 0.5),
                       ("n_records", 9)):
        ev = parity_doc(**{field: bad})
        g, _ = sidecar_stage(tmp_path / field, evidence=ev)
        assert mentions(g.reasons, field, "per_record"), (field, g.reasons)


def test_a_wiring_check_that_failed_in_the_records_is_not_hidden(tmp_path):
    """result.json says wiring passed; the evidence file the test wrote says it did not."""
    ev = parity_doc(wiring_check={"cosine": 0.5, "passed": False})
    g, _ = sidecar_stage(tmp_path, evidence=ev, result_sp={k: v for k, v in parity_doc().items() if k != "per_record"})
    assert not g.ok and mentions(g.reasons, "wiring_check")


def test_a_bound_evidence_file_edited_after_the_test_is_refused(tmp_path):
    ev = parity_doc()
    run = tmp_path / "run"
    g, run = sidecar_stage(tmp_path, evidence=ev, bind=True)
    assert g.ok
    # The agent edits the records and the summary after the supervisor hashed the file.
    edited = parity_doc([rec(i, pcc=0.9999, diff=0.0001) for i in range(6)])
    (run / SP).write_text(json.dumps(edited))
    sd = run / "stages" / "2"
    g2 = gate_weights_swap_sidecar(sd, run)
    assert not g2.ok and mentions(g2.reasons, "sidecar-parity.json", "supervisor hashed")


def test_a_bound_run_without_per_record_is_refused(tmp_path):
    ev = {k: v for k, v in parity_doc().items() if k != "per_record"}
    g, _ = sidecar_stage(tmp_path, evidence=ev, bind=True)
    assert not g.ok and mentions(g.reasons, "per_record")


def test_a_bound_run_whose_evidence_file_was_deleted_is_refused(tmp_path):
    g, run = sidecar_stage(tmp_path, bind=True)
    (run / SP).unlink()
    g2 = gate_weights_swap_sidecar(run / "stages" / "2", run)
    assert not g2.ok and mentions(g2.reasons, "sidecar-parity.json", "supervisor hashed")


def test_an_unbound_legacy_evidence_file_passes_and_is_marked_unverified(tmp_path):
    """The existing tests write `{}` as the evidence file: no records, no ledger entry. Judged as before, and the gate says the numbers are
    the agent's."""
    g, _ = sidecar_stage(tmp_path, evidence={}, result_sp=PARITY)
    assert g.ok, g.reasons
    assert any("sidecar parity" in u for u in g.unverified)


@pytest.mark.parametrize("bad", [
    "nope", [], None, 7, [None], [{}], [{"hidden_pcc": "x"}], [{"id": "a", "hidden_pcc": 10 ** 3000, "prob_max_abs_diff": 0, "top1_agree": 0,
                                                              "n_questions": 1}],
    [{"id": "a", "hidden_pcc": True, "prob_max_abs_diff": 0, "top1_agree": 0, "n_questions": 1}],
    [{"id": "a", "hidden_pcc": 0.9, "prob_max_abs_diff": 0, "top1_agree": 5, "n_questions": 1}],
    [{"id": "a", "hidden_pcc": 0.9, "prob_max_abs_diff": 0, "top1_agree": 0, "n_questions": 0}],
    [{"id": "a", "hidden_pcc": 0.9, "prob_max_abs_diff": 0, "top1_agree": 0, "n_questions": 10 ** 3000}],
])
def test_hostile_per_record_values_are_refused_without_raising(tmp_path, bad):
    ev = {**{k: v for k, v in parity_doc().items() if k != "per_record"}, "per_record": bad}
    g, _ = sidecar_stage(tmp_path, evidence=ev, bind=True)
    assert not g.ok


def test_a_huge_record_list_is_counted_not_listed(tmp_path):
    ev = parity_doc([rec(i) for i in range(5000)])
    ev["hidden_pcc_min"] = 0.5
    g, _ = sidecar_stage(tmp_path, evidence=ev, bind=True)
    assert not g.ok
    assert all(len(r) <= token_check.MAX_REASON_CHARS + 80 for r in g.reasons)


@pytest.mark.parametrize("text", ["[1,2", "{\"per_record\": NaN}", "", "\x00", "[" * 5000])
def test_an_evidence_file_that_is_not_json_is_refused_on_a_bound_run(tmp_path, text):
    g, _ = sidecar_stage(tmp_path, evidence_text=text, bind=True)
    assert not g.ok


def test_the_recompute_never_raises_on_odd_values():
    for odd in (None, 1, "x", [], {}, [None] * 3, [{"id": 1}], [[]], [{"hidden_pcc": float("nan")}]):
        got, why = sidecar_numbers.safe_recompute(odd)
        assert got is None and why


# ---- the stage 7 boot check ----------------------------------------------------------------------------------------------------------

REF = [2000 + 11 * i for i in range(32)]
REF_PATH = "stages/1/evidence/reference/generated-ids.json"


def boot_report(ref=REF, bad=(5,)):
    """verify.json as verify_bundle.py writes it: the free-run text, the counts, the mismatches and the raw first token at each position."""
    forced, mismatches = [], []
    for k, want in enumerate(ref):
        got = want + 1 if k in bad else want
        forced.append({"position": k, "first_id": got})
        if k in bad:
            mismatches.append({"position": k, "expected_id": want, "got_id": got, "got_text": "g"})
    matches = len(ref) - len(bad)
    return {"label": "measured", "n_tokens": len(ref), "matches": matches, "top1_agreement": matches / len(ref), "coherent": True,
            "teacher_forced": forced, "mismatches": mismatches}


def boot_ledger(run, ids=True, raw=False, path="stages/7/verify/evidence/verify.json"):
    led = Ledger(run / "ledger.jsonl")
    try:
        if ids:
            led.append("decision", 2, decision=stages.REFERENCE_IDS_LEDGER_DECISION,
                       sha256=hashlib.sha256((run / REF_PATH).read_bytes()).hexdigest())
        if raw:
            led.append("evidence", 7, what=raw_evidence.RAW_EVIDENCE_WHAT, path=path,
                       sha256=hashlib.sha256((run / path).read_bytes()).hexdigest())
    finally:
        led.close()


def boot_reasons(tmp_path, report, ids=True, raw=False, ref=REF):
    run = tmp_path / "run"
    write(run, REF_PATH, {"generated_ids": ref})
    write(run, "stages/7/verify/evidence/verify.json", report)
    if ids or raw:
        boot_ledger(run, ids=ids, raw=raw)
    return stages.boot_check_reasons(run / "stages" / "7", run, report)


def test_the_unedited_boot_check_passes(tmp_path):
    for ids, raw in ((True, True), (True, False), (False, False)):
        reasons, unverified = boot_reasons(tmp_path / f"{ids}{raw}", boot_report(), ids=ids, raw=raw)
        assert reasons == [] and unverified == ()


def test_raising_the_boot_top1_while_the_raw_replies_stay_is_refused(tmp_path):
    r = boot_report()
    r["top1_agreement"], r["matches"], r["mismatches"] = 1.0, 32, []
    reasons, _ = boot_reasons(tmp_path, r, ids=True, raw=True)
    assert mentions(reasons, "verify.json", "top1_agreement", "31 of 32")


def test_editing_the_boot_records_and_the_numbers_together_is_refused_when_the_file_was_hashed(tmp_path):
    run = tmp_path / "run"
    write(run, REF_PATH, {"generated_ids": REF})
    write(run, "stages/7/verify/evidence/verify.json", boot_report())
    boot_ledger(run, ids=True, raw=True)
    write(run, "stages/7/verify/evidence/verify.json", boot_report(bad=()))
    reasons, _ = stages.boot_check_reasons(run / "stages" / "7", run, boot_report(bad=()))
    assert mentions(reasons, "verify.json", "supervisor hashed")


def test_a_report_that_differs_from_the_hashed_file_is_refused(tmp_path):
    """The gate's own read of verify.json and the hashed file must be the same document."""
    run = tmp_path / "run"
    write(run, REF_PATH, {"generated_ids": REF})
    write(run, "stages/7/verify/evidence/verify.json", boot_report())
    boot_ledger(run, ids=True, raw=True)
    reasons, _ = stages.boot_check_reasons(run / "stages" / "7", run, boot_report(bad=()))
    assert mentions(reasons, "changed while the gate was reading")


def test_a_bound_boot_check_without_records_is_refused(tmp_path):
    r = {k: v for k, v in boot_report().items() if k != "teacher_forced"}
    reasons, _ = boot_reasons(tmp_path, r, ids=True, raw=True)
    assert mentions(reasons, "teacher_forced")


def test_an_unbound_boot_check_without_records_passes_and_is_marked_unverified(tmp_path):
    r = {k: v for k, v in boot_report().items() if k != "teacher_forced"}
    reasons, unverified = boot_reasons(tmp_path, r, ids=False, raw=False)
    assert reasons == [] and any("boot check" in u for u in unverified)


def test_a_boot_check_with_the_wrong_number_of_positions_is_refused_when_bound(tmp_path):
    r = boot_report(ref=REF[:20], bad=())
    reasons, _ = boot_reasons(tmp_path, r, ids=True, raw=True)
    assert mentions(reasons, "20", "32")


@pytest.mark.parametrize("bad", [None, "x", 7, [], {}, [None], [{"position": 0}], [{"position": True, "first_id": 1}],
                                  [{"position": 0, "first_id": 10 ** 3000}], [{"position": 0, "first_id": "a"}]])
def test_hostile_boot_records_are_refused_without_raising(tmp_path, bad):
    r = {**boot_report(), "teacher_forced": bad}
    reasons, _ = boot_reasons(tmp_path, r, ids=True, raw=True)
    assert reasons


@pytest.mark.parametrize("value", [10 ** 3000, float("inf"), "0.9", None, True, [1], {"a": 1}])
def test_hostile_boot_summary_numbers_are_refused_without_raising(tmp_path, value):
    for field in ("top1_agreement", "n_tokens", "matches"):
        r = {**boot_report(), field: value}
        reasons, _ = boot_reasons(tmp_path / field, r, ids=True, raw=True)
        assert reasons


def test_the_boot_recompute_never_raises():
    assert token_check.boot_problems("not a dict", REF, require_records=True)
    assert token_check.boot_problems({"teacher_forced": object()}, REF, require_records=True)


# ---- the two numbers that have no raw evidence -----------------------------------------------------------------------------------------

def test_the_decoder_gate_says_its_pcc_is_agent_reported(tmp_path):
    sd, run = stage(tmp_path, 2, "result.json", {"pcc": 0.995, "argmax_match": True, "evidence": ["stages/2/evidence/diff.txt"]})
    g = gate_decoder(sd, run)
    assert g.ok, g.reasons
    assert any("pcc" in u and "agent" in u for u in g.unverified)


def test_a_failing_decoder_gate_still_fails(tmp_path):
    sd, run = stage(tmp_path, 2, "result.json", {"pcc": 0.5, "argmax_match": True, "evidence": ["stages/2/evidence/diff.txt"]})
    g = gate_decoder(sd, run)
    assert not g.ok


def test_the_full_model_gate_says_its_top1_is_agent_reported(tmp_path):
    sd, run = stage(tmp_path, 3, "result.json", {"parity": True, "top1": 0.97, "evidence": ["stages/3/evidence/diff.txt"]})
    g = gate_full_model(sd, run)
    assert g.ok, g.reasons
    assert any("top1" in u and "agent" in u for u in g.unverified)


def test_a_gate_with_a_recompute_has_nothing_unverified_for_it(tmp_path):
    g, _ = sidecar_stage(tmp_path)
    assert g.unverified == ()


# ---- the supervisor hashes the raw file when the test ends ----------------------------------------------------------------------------

def test_raw_evidence_is_hashed_for_the_files_that_exist(tmp_path):
    run = tmp_path / "run"
    write(run, SP, parity_doc())
    got = raw_evidence.record_after_test(run, run / "stages" / "2", 2)
    assert got == [{"path": SP, "sha256": hashlib.sha256((run / SP).read_bytes()).hexdigest()}]
    assert raw_evidence.record_after_test(run, run / "stages" / "3", 3) == []          # no raw file named for stage 3


def test_the_supervisor_records_the_raw_file_in_the_ledger_after_a_test(tmp_path, monkeypatch):
    """The wiring: Supervisor._run_test calls the recorder after the marker is written, and the entry is the one the gates look for."""
    from orchard import supervisor as sup
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    led = Ledger(run / "ledger.jsonl")
    s = sup.Supervisor.__new__(sup.Supervisor)
    s.run_dir, s.ledger = run, led

    def fake_spawn(self, n, command, chips, ids, deadline, out_path, **kw):
        write(run, SP, parity_doc())
        out_path.write_text("ok")
        return {"returncode": 0, "timed_out": False}
    monkeypatch.setattr(sup.Supervisor, "_spawn", fake_spawn)
    spec = stages.STAGES[2]
    lease = type("L", (), {"chips": list(range(4)), "dev_indices": list(range(4)), "lease_id": "x"})()
    s._run_test(spec, run / "stages" / "2", {"command": "true", "deadline_s": 5}, lease)
    entries = [e for e in led.read() if e["event"] == "evidence" and e["data"].get("what") == raw_evidence.RAW_EVIDENCE_WHAT]
    led.close()
    assert len(entries) == 1 and entries[0]["data"]["path"] == SP
    assert raw_evidence.recorded_sha(run, 2, SP) == hashlib.sha256((run / SP).read_bytes()).hexdigest()


def test_the_supervisor_notes_unverified_numbers_in_the_ledger(tmp_path):
    """A passing gate that has unverified numbers leaves a decision the operator bundle reads."""
    from orchard import supervisor as sup
    run = tmp_path / "run"
    run.mkdir()
    led = Ledger(run / "ledger.jsonl")
    s = sup.Supervisor.__new__(sup.Supervisor)
    s.run_dir, s.ledger = run, led
    s._note_unverified(3, stages.GateResult(True, unverified=("stage 3 top1: agent reported",)))
    s._note_unverified(3, stages.GateResult(True))
    got = [e for e in led.read() if e["data"].get("decision") == stages.UNVERIFIED_LEDGER_DECISION]
    led.close()
    assert len(got) == 1 and got[0]["data"]["numbers"] == ["stage 3 top1: agent reported"]


# ---- the operator bundle and the card say so ------------------------------------------------------------------------------------------

def test_the_bundle_lists_unverified_numbers_in_thin_evidence():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "orchard/skills/operator-bundle-templates/build_bundle.py"
    spec = importlib.util.spec_from_file_location("bb445", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeRun:
        results = {}
        reference = {}
        unverified = [{"stage": 3, "numbers": ["stage 3 top1: the agent reported it; no raw replies are recorded"]}]

        def configs(self):
            return []
    items = mod.thin_evidence(FakeRun(), lambda s: s)
    assert any("stage 3 top1" in i and "agent" in i for i in items)
