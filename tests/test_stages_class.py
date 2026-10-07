"""The outcome class in the stage machine: orchard/classes.py, the stage 0 gate, run_class, spec_for and the
sidecar parity gate. The class is stage 0's answer to "what kind of bring-up is this?" and each class has a
contract (docs/superpowers/specs/2026-10-06-bringup-command-design.md, section 3)."""
import dataclasses
import json
from pathlib import Path

import pytest

from orchard import classes
from orchard import defaults
from orchard.stages import (STAGES, delta_class, gate_delta, gate_sidecar_parity, gate_weights_swap,
                            gate_weights_swap_sidecar, run_class, run_path, spec_for)
from test_stages import GOOD_DELTA, SWAP, delta_file, stage, stage0_pass, write

HEAD_SHA, CODE_SHA = "a" * 64, "b" * 64
SIDECAR = {"file": "joint_head.safetensors", "size": 256_000_000, "sha256": HEAD_SHA, "num_tensors": 122,
           "dtypes": ["BF16"], "tensor_names": ["evidence_layers.0.attention.in_proj_weight"]}
CODE = {"file": "joint_schema_model.py", "sha256": CODE_SHA}
SIDE_DELTA = {**GOOD_DELTA, "class": "weights+sidecar", "sidecars": [SIDECAR], "code_files": [CODE]}


# ---- the class table --------------------------------------------------------------------------

def test_each_class_maps_to_the_path_the_existing_machinery_knows():
    assert classes.path_of("weights-only") == "weights-only"
    assert classes.path_of("weights+sidecar") == "weights-only"      # the backbone takes the weights-only path
    assert classes.path_of("full-port") == "full-port"
    assert classes.path_of("unknown") == "full-port"                 # nothing unproven runs as weights-only
    assert classes.path_of("nonsense") is None and classes.path_of(None) is None


def test_a_path_alone_implies_a_class_for_a_run_that_predates_classes():
    assert classes.class_of_path("weights-only") == "weights-only"
    assert classes.class_of_path("full-port") == "full-port"
    assert classes.class_of_path("maybe") is None


def test_the_class_set_is_the_documented_four():
    assert classes.CLASSES == ("weights-only", "weights+sidecar", "full-port", "unknown")


# ---- the stage 0 gate -------------------------------------------------------------------------

def gate(tmp_path, delta):
    return gate_delta(*stage(tmp_path, 0, "delta.json", delta))


@pytest.mark.parametrize("delta", [
    {**GOOD_DELTA, "class": "weights-only", "sidecars": []},
    SIDE_DELTA,
    {**GOOD_DELTA, "path": "full-port", "class": "full-port", "sidecars": []},
    {**GOOD_DELTA, "path": "full-port", "class": "unknown", "sidecars": []},
    GOOD_DELTA,                                                      # no class: a run from before classes
])
def test_the_gate_accepts_each_consistent_class(tmp_path, delta):
    g = gate(tmp_path, delta)
    assert g.ok, g.reasons


@pytest.mark.parametrize("delta,word", [
    ({**GOOD_DELTA, "class": "maybe"}, "must be one of"),
    ({**GOOD_DELTA, "class": "full-port"}, "path"),                  # path says weights-only
    ({**SIDE_DELTA, "path": "full-port"}, "path"),
    ({**GOOD_DELTA, "class": "weights+sidecar", "sidecars": []}, "sidecars"),
    ({**GOOD_DELTA, "class": "weights-only", "sidecars": [SIDECAR]}, "sidecars"),
    ({**SIDE_DELTA, "sidecars": [{**SIDECAR, "sha256": "short"}]}, "sha256"),
    ({**SIDE_DELTA, "sidecars": [{k: v for k, v in SIDECAR.items() if k != "file"}]}, "file"),
    ({**SIDE_DELTA, "sidecars": "joint_head.safetensors"}, "sidecars"),
    ({**SIDE_DELTA, "code_files": [{"file": "x.py"}]}, "code_files"),
    ({**SIDE_DELTA, "code_files": "x.py"}, "code_files"),
])
def test_the_gate_refuses_an_inconsistent_class(tmp_path, delta, word):
    g = gate(tmp_path, delta)
    assert not g.ok and any(word in r for r in g.reasons), g.reasons


# ---- reading the class back -------------------------------------------------------------------

@pytest.mark.parametrize("cls", ["weights-only", "weights+sidecar", "full-port", "unknown"])
def test_the_class_is_read_from_stage_0s_passing_end(tmp_path, cls):
    assert run_class(stage0_pass(path=classes.path_of(cls), **{"class": cls}), tmp_path) == cls


def test_a_ledger_without_a_class_derives_it_from_the_recorded_path(tmp_path):
    assert run_class(stage0_pass(path="weights-only"), tmp_path) == "weights-only"
    assert run_class(stage0_pass(path="full-port"), tmp_path) == "full-port"


def test_a_ledger_with_neither_reads_delta_json(tmp_path):
    delta_file(tmp_path, SIDE_DELTA)
    assert run_class(stage0_pass(), tmp_path) == "weights+sidecar"
    delta_file(tmp_path, GOOD_DELTA)                                  # no class: derived from its path
    assert run_class(stage0_pass(), tmp_path) == "weights-only"


def test_the_ledger_class_wins_over_a_delta_json_edited_later(tmp_path):
    delta_file(tmp_path, {**GOOD_DELTA, "class": "weights-only", "sidecars": []})
    assert run_class(stage0_pass(path="weights-only", **{"class": "weights+sidecar"}), tmp_path) == "weights+sidecar"


@pytest.mark.parametrize("bad", ["maybe", 3, None])
def test_an_unknown_recorded_class_is_none_never_weights_only(tmp_path, bad):
    delta_file(tmp_path, SIDE_DELTA)
    assert run_class(stage0_pass(path="weights-only", **{"class": bad}), tmp_path) is None


def test_before_stage_0_passes_there_is_no_class(tmp_path):
    delta_file(tmp_path, SIDE_DELTA)
    assert run_class([], tmp_path) is None


def test_the_class_and_the_path_never_disagree_in_what_run_path_reports(tmp_path):
    entries = stage0_pass(path="weights-only", **{"class": "weights+sidecar"})
    assert run_path(entries, tmp_path) == "weights-only" and run_class(entries, tmp_path) == "weights+sidecar"


def test_delta_class_reads_a_valid_class_or_derives_it(tmp_path):
    delta_file(tmp_path, SIDE_DELTA)
    assert delta_class(tmp_path) == "weights+sidecar"
    delta_file(tmp_path, {**GOOD_DELTA, "path": "full-port"})
    assert delta_class(tmp_path) == "full-port"
    delta_file(tmp_path, {**GOOD_DELTA, "class": "maybe"})            # a bad value is not trusted or derived
    assert delta_class(tmp_path) is None
    delta_file(tmp_path, "{not json")
    assert delta_class(tmp_path) is None


# ---- the stage table follows the class --------------------------------------------------------

def test_the_sidecar_class_gives_stage_2_its_own_skill_and_gate():
    s = spec_for(2, "weights-only", None, "weights+sidecar")
    assert s.skill == "weights-sidecar-check" and s.gate is gate_weights_swap_sidecar
    assert (s.boards, s.gate_file, s.marker) == (1, "result.json", STAGES[2].marker)


def test_every_other_stage_of_the_sidecar_class_is_the_weights_only_spec():
    for n in (3, 4, 5, 6, 7, 8):
        assert spec_for(n, "weights-only", "v6", "weights+sidecar") == spec_for(n, "weights-only", "v6")


@pytest.mark.parametrize("cls", [None, "weights-only"])
def test_without_a_sidecar_stage_2_is_the_plain_swap_check(cls):
    s = spec_for(2, "weights-only", None, cls)
    assert s.skill == "weights-swap-check" and s.gate is gate_weights_swap


def test_a_class_on_the_wrong_path_does_not_change_the_table():
    assert spec_for(2, "full-port", None, "weights+sidecar") == STAGES[2]
    assert spec_for(2, None, None, "weights+sidecar") == STAGES[2]


def test_the_sidecar_skill_file_exists():
    from orchard.supervisor import SKILLS_DIR
    assert (SKILLS_DIR / "weights-sidecar-check.md").is_file()


# ---- the sidecar parity gate ------------------------------------------------------------------

PARITY = {"measured": True, "n_records": 6, "n_questions": 17, "hidden_pcc_min": 0.997,
          "hidden_pcc_mean": 0.998, "prob_max_abs_diff": 0.01, "top1_agree_fraction": 1.0,
          "noise_floor": {"prob_max_abs_diff": 0.002}, "wiring_check": {"cosine": 0.9999, "passed": True},
          "tt_metal_sha": "d59904e", "code_sha256": CODE_SHA, "head_sha256": HEAD_SHA, "mesh_shape": [1, 2]}
RESULT = {**SWAP, "sidecar_parity": PARITY,
          "evidence": ["stages/2/evidence/diff.txt", "stages/2/evidence/sidecar-parity.json"]}


def sidecar_gate(tmp_path, result=None, delta=SIDE_DELTA, parity_file=True, **parity):
    run = tmp_path / "run"
    write(run, "stages/0/delta.json", delta)
    data = {**(RESULT if result is None else result)}
    if parity:
        data["sidecar_parity"] = {**data["sidecar_parity"], **{k: v for k, v in parity.items() if v is not ...}}
        for k, v in parity.items():
            if v is ...:
                data["sidecar_parity"].pop(k)
    if parity_file:
        write(run, "stages/2/evidence/sidecar-parity.json", "{}")
    return gate_weights_swap_sidecar(*stage(tmp_path, 2, "result.json", data))


def test_a_complete_sidecar_result_passes(tmp_path):
    g = sidecar_gate(tmp_path)
    assert g.ok, g.reasons


def test_the_swap_fields_are_still_checked(tmp_path):
    g = sidecar_gate(tmp_path, result={**RESULT, "top1_agreement": 0.1})
    assert not g.ok and any("top1_agreement" in r for r in g.reasons)


def test_a_result_with_no_sidecar_parity_fails(tmp_path):
    g = sidecar_gate(tmp_path, result={k: v for k, v in RESULT.items() if k != "sidecar_parity"})
    assert not g.ok and any("sidecar_parity" in r for r in g.reasons)


@pytest.mark.parametrize("change,word", [
    ({"measured": False}, "measured"),
    ({"measured": ...}, "measured"),
    ({"wiring_check": {"cosine": 0.5, "passed": False}}, "wiring"),
    ({"wiring_check": ...}, "wiring"),
    ({"n_questions": 3}, "n_questions"),
    ({"n_questions": True}, "n_questions"),
    ({"hidden_pcc_min": defaults.SIDECAR_PCC_MIN - 0.001}, "hidden_pcc_min"),
    ({"hidden_pcc_min": "high"}, "hidden_pcc_min"),
    ({"top1_agree_fraction": defaults.SIDECAR_TOP1_MIN - 0.01}, "top1_agree_fraction"),
    ({"top1_agree_fraction": 1.5}, "top1_agree_fraction"),
    ({"prob_max_abs_diff": 0.5}, "prob_max_abs_diff"),
    ({"prob_max_abs_diff": ...}, "prob_max_abs_diff"),
])
def test_each_failing_parity_field_gets_its_own_reason(tmp_path, change, word):
    g = sidecar_gate(tmp_path, **change)
    assert not g.ok and any(word in r for r in g.reasons), g.reasons


def test_the_probability_bar_widens_to_four_times_a_noisy_reference(tmp_path):
    loose = defaults.SIDECAR_PROB_DIFF_MAX * 3
    assert not sidecar_gate(tmp_path, prob_max_abs_diff=loose).ok                          # tight reference
    assert sidecar_gate(tmp_path / "b", prob_max_abs_diff=loose,
                        noise_floor={"prob_max_abs_diff": loose / defaults.SIDECAR_NOISE_FACTOR}).ok


def test_a_parity_run_on_a_different_head_file_is_refused(tmp_path):
    g = sidecar_gate(tmp_path, head_sha256="c" * 64)
    assert not g.ok and any("head_sha256" in r and "stage 0" in r for r in g.reasons), g.reasons


def test_a_parity_run_with_different_code_is_refused(tmp_path):
    g = sidecar_gate(tmp_path, code_sha256="d" * 64)
    assert not g.ok and any("code_sha256" in r for r in g.reasons), g.reasons


def test_a_model_without_code_files_needs_no_code_hash(tmp_path):
    delta = {**SIDE_DELTA, "code_files": []}
    g = sidecar_gate(tmp_path, delta=delta, code_sha256=...)
    assert g.ok, g.reasons


def test_a_missing_stage_0_delta_is_a_reason_not_a_crash(tmp_path):
    run = tmp_path / "run"
    write(run, "stages/2/evidence/sidecar-parity.json", "{}")
    g = gate_weights_swap_sidecar(*stage(tmp_path, 2, "result.json", RESULT))
    assert not g.ok and any("delta.json" in r for r in g.reasons)


def test_the_parity_evidence_file_must_exist_and_be_listed(tmp_path):
    g = sidecar_gate(tmp_path, parity_file=False)
    assert not g.ok
    g = sidecar_gate(tmp_path / "b", result={**RESULT, "evidence": ["stages/2/evidence/diff.txt"]})
    assert not g.ok and any("sidecar-parity.json" in r for r in g.reasons), g.reasons


def test_gate_sidecar_parity_alone_checks_only_the_parity_object(tmp_path):
    reasons = gate_sidecar_parity(PARITY, {"sidecars": [SIDECAR], "code_files": [CODE]})
    assert reasons == []
    assert gate_sidecar_parity("not a dict", {}) != []


def test_the_thresholds_are_labelled_choices_in_defaults():
    src = (Path(defaults.__file__)).read_text()
    for name in ("SIDECAR_PCC_MIN", "SIDECAR_TOP1_MIN", "SIDECAR_PROB_DIFF_MAX", "SIDECAR_MIN_QUESTIONS",
                 "SIDECAR_NOISE_FACTOR"):
        line = next(l for l in src.splitlines() if l.startswith(name))
        assert "choice" in line.lower() or "measured" in line.lower(), line
