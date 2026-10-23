# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 notices when the nearest model's mtp.* tensors could be grafted onto a model that has none.

AREX-2 ships no mtp.* tensors, so every swap check and package served it with the drafter off. A hand
experiment grafted the nearest model's tensors and the drafter ran about three times faster with the same
agreement. Stage 0 records the candidate and one hazard; it never changes the path or the class, and grafts
nothing itself: stage 2's graft trial serves the graft once and keeps it only when the measurement says so
(orchard/graft_trial.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_delta_triage_template import (BASE_REV, H, NO_MTP, delta, make_snapshot,  # noqa: E402,F401
                                        run_triage, stage_for, template_module, text_case, tokenizer_json)

from orchard.stages import gate_delta  # noqa: E402


def graft_hazards(d):
    return [h for h in d["hazards"] if h["area"] == "drafter_graft"]


def test_a_model_without_mtp_tensors_gets_a_graft_candidate_and_one_hazard(text_case):
    stage = text_case(tensor_change=NO_MTP)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    g = d["graft_candidate"]
    assert g["base_tensors"] == 2 and g["base_revision"] == BASE_REV
    assert g["hidden_size"] == H and g["shapes_match"] is True and g["graftable"] is True
    assert g["dtypes"] == ["BF16"] and len(g["base_shards"]) == 1
    [h] = graft_hazards(d)
    assert "graft trial" in h["finding"] and "mtp.* tensors" in h["finding"]
    assert h["evidence"]


def test_the_candidate_changes_neither_the_path_nor_the_class(text_case):
    with_graft = text_case(tensor_change=NO_MTP)
    assert run_triage(with_graft).returncode == 0
    d = delta(with_graft)
    assert d["path"] == "weights-only" and d["class"] == "weights-only" and not d["path_reasons"]


def test_the_draft_with_a_graft_candidate_still_passes_the_stage_0_gate(text_case):
    stage = text_case(tensor_change=NO_MTP)
    assert run_triage(stage).returncode == 0
    gate = gate_delta(stage, stage.parent.parent)
    assert gate.ok, gate.reasons


def test_a_model_that_keeps_its_mtp_tensors_has_no_candidate(text_case):
    stage = text_case()
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert "graft_candidate" not in d and graft_hazards(d) == []


def test_no_candidate_when_the_nearest_model_has_no_mtp_tensors_either(tmp_path, tokenizer_json):
    model = make_snapshot(tmp_path / "new", "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf", nested=False, vision=False,
                          tokenizer=tokenizer_json, tensor_change=NO_MTP)
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json,
                            tensor_change=NO_MTP)
    stage = stage_for(tmp_path, model, nearest)
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert "graft_candidate" not in d and graft_hazards(d) == []


def test_a_head_that_does_not_fit_is_recorded_as_not_graftable_and_the_hazard_says_so(text_case):
    stage = text_case(tensor_change=NO_MTP, base_kw={"tensor_change": {"mtp.norm.weight": ("BF16", [H + 1])}})
    assert run_triage(stage).returncode == 0
    d = delta(stage)
    assert d["graft_candidate"]["shapes_match"] is False and d["graft_candidate"]["graftable"] is False
    [h] = graft_hazards(d)
    assert "cannot be grafted" in h["finding"]


# ---- the function itself ----------------------------------------------------------------------------

def info(dtype, shape, shard):
    return {"dtype": dtype, "shape": shape, "shard": shard}


NEAREST = {"mtp.fc.weight": info("BF16", [8, 16], "s2"), "mtp.norm.weight": info("BF16", [8], "s2"),
           "lm_head.weight": info("BF16", [4, 8], "s2")}


def candidate(model=None, nearest=None, hidden=8, revision=BASE_REV):
    return template_module().graft_candidate(model or {"lm_head.weight": info("BF16", [4, 8], "m")},
                                             NEAREST if nearest is None else nearest, hidden, revision)


def test_the_function_counts_only_the_mtp_tensors_of_the_nearest_model():
    g = candidate()
    assert (g["base_tensors"], g["base_shards"], g["dtypes"], g["graftable"]) == (2, ["s2"], ["BF16"], True)


def test_tensors_under_a_prefix_count_and_a_lookalike_name_does_not():
    g = candidate(nearest={**NEAREST, "model.mtp.layers.0.weight": info("BF16", [2], "s2"),
                           "model.attn_mtpx.weight": info("BF16", [2], "s2")})
    assert g["base_tensors"] == 3


@pytest.mark.parametrize("model", [{"mtp.fc.weight": info("BF16", [8, 16], "m")}])
def test_a_model_with_mtp_tensors_has_no_candidate(model):
    assert candidate(model=model) is None


def test_no_mtp_tensors_in_the_nearest_model_means_no_candidate():
    assert candidate(nearest={"lm_head.weight": info("BF16", [4, 8], "s2")}) is None


def test_two_shards_make_it_not_graftable():
    g = candidate(nearest={**NEAREST, "mtp.norm.weight": info("BF16", [8], "s1")})
    assert g["base_shards"] == ["s1", "s2"] and g["graftable"] is False


def test_a_dtype_other_than_bf16_makes_it_not_graftable():
    g = candidate(nearest={**NEAREST, "mtp.fc.weight": info("F32", [8, 16], "s2")})
    assert g["dtypes"] == ["BF16", "F32"] and g["graftable"] is False


def test_the_hidden_size_is_matched_against_mtp_norm_weight():
    assert candidate(hidden=9)["shapes_match"] is False
    assert candidate(hidden=8)["shapes_match"] is True


def test_an_unknown_hidden_size_is_not_a_mismatch():
    g = candidate(hidden=None)
    assert g["shapes_match"] is None and g["graftable"] is True


def test_a_nearest_model_without_the_norm_tensor_is_not_a_mismatch():
    g = candidate(nearest={"mtp.fc.weight": info("BF16", [8, 16], "s2")})
    assert g["shapes_match"] is None and g["graftable"] is True


def test_a_snapshot_not_named_by_a_commit_has_no_revision():
    assert candidate(revision=None)["base_revision"] is None


def test_the_hidden_size_is_read_from_a_nested_text_config(tmp_path, tokenizer_json):
    model = make_snapshot(tmp_path / "new", "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf", nested=True, vision=True,
                          tokenizer=tokenizer_json, tensor_change=NO_MTP)
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    stage = stage_for(tmp_path, model, nearest)
    assert run_triage(stage).returncode == 0
    assert delta(stage)["graft_candidate"]["hidden_size"] == H


def test_a_snapshot_directory_that_is_not_a_commit_gives_no_base_revision(tmp_path, tokenizer_json):
    model = make_snapshot(tmp_path / "new", "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf", nested=False, vision=False,
                          tokenizer=tokenizer_json, tensor_change=NO_MTP)
    nearest = make_snapshot(tmp_path / "base", "main", nested=True, vision=True, tokenizer=tokenizer_json)
    stage = stage_for(tmp_path, model, nearest)
    assert run_triage(stage).returncode == 0
    assert delta(stage)["graft_candidate"]["base_revision"] is None


def test_the_hazard_names_each_reason_the_head_cannot_be_grafted():
    g = {"base_tensors": 2, "base_revision": None, "base_shards": ["a", "b"], "dtypes": ["BF16", "F32"], "hidden_size": 8,
         "shapes_match": False, "graftable": False}
    h = template_module().graft_hazard("Qwen/Qwen3.8-27B@1", g, lambda name: f"evidence/{name}")
    assert h["area"] == "drafter_graft" and h["evidence"] == ["evidence/tensor-compare.json"]
    for reason in ("not all BF16 (BF16, F32)", "they sit in 2 shards", "does not match the new model's hidden size 8"):
        assert reason in h["finding"]


def test_a_graftable_head_says_stage_2_tries_it_and_keeps_it_only_when_it_measures_better():
    g = {"base_tensors": 15, "base_revision": None, "base_shards": ["a"], "dtypes": ["BF16"], "hidden_size": 8,
         "shapes_match": True, "graftable": True}
    h = template_module().graft_hazard("Qwen/Qwen3.8-27B@1", g, lambda name: name)
    f = h["finding"]
    assert "15 mtp.* tensors of Qwen/Qwen3.8-27B@1" in f and "cannot be grafted" not in f
    assert "graft trial" in f and "graft-decision.json" in f and "only when" in f
    assert "off unless a person turns it on" not in f
