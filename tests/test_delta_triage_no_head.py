# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 and a model with no output head (issue 380). An embedding model such as Qwen3-Embedding or Qwen3-VL-Embedding has no lm_head, and
its architecture is a bare backbone. The nearest model's lm_head is then not a tensor the new model is missing: the new model does not read
a vocabulary head. Without this rule the nearest model's lm_head was listed under "only in the nearest model" and made the class full-port.
These tests run the real triage script on the restricted-head fixture's layout (no lm_head, no sidecar) with a config that names a bare
backbone, and a control with a causal language model architecture, which must still be full-port."""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("tokenizers")
from test_delta_triage_alignment import decider_case, run, template  # noqa: E402,F401  (decider_case is a fixture)
from test_delta_triage_template import tokenizer_json  # noqa: E402,F401  (the fixture decider_case uses)
from orchard.stages import gate_delta  # noqa: E402


def set_architectures(snap, archs):
    path = snap / "config.json"
    cfg = json.loads(path.read_text())
    cfg["architectures"] = archs
    path.write_text(json.dumps(cfg))


def test_a_bare_backbone_with_no_lm_head_is_weights_only_and_the_nearest_head_is_not_missing(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    set_architectures(new, ["Qwen3_5Model"])
    stage, d = run(tmp_path, new, base)
    assert d["class"] == "weights-only" and d["path"] == "weights-only" and d["path_reasons"] == []
    assert d["head"]["not_used"] is True and d["head"]["restricted"] is False
    note = [h for h in d["hazards"] if "no lm_head tensor" in h["finding"]]
    assert len(note) == 1 and note[0]["area"] == "other" and "scoring" in note[0]["finding"]
    assert gate_delta(stage, tmp_path / "run").ok


def test_the_same_layout_named_a_causal_language_model_is_still_full_port(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    set_architectures(new, ["Qwen3_5ForCausalLM"])
    stage, d = run(tmp_path, new, base)
    assert d["head"]["not_used"] is False and d["class"] == "full-port"
    assert any("lm_head.weight" in r for r in d["path_reasons"])
    assert not [h for h in d["hazards"] if "no lm_head tensor" in h["finding"]]


@pytest.mark.parametrize("archs", [None, [], "Qwen3_5Model", [3], [None]])
def test_a_config_with_no_usable_architecture_list_is_not_taken_for_a_headless_model(decider_case, archs):
    tmp_path, base, new, head, ids, with_rows = decider_case
    path = new / "config.json"
    cfg = json.loads(path.read_text())
    if archs is None:
        cfg.pop("architectures")
    else:
        cfg["architectures"] = archs
    path.write_text(json.dumps(cfg))
    stage, d = run(tmp_path, new, base)
    assert d["head"]["not_used"] is False and d["class"] == "full-port"


@pytest.mark.parametrize("name,expected", [
    ("Qwen3Model", True), ("Qwen3VLForEmbedding", True), ("BertModel", True), ("XLMRobertaModel", True),
    ("Qwen3ForCausalLM", False), ("Qwen3VLForConditionalGeneration", False), ("GPT2LMHeadModel", False), ("BertForMaskedLM", False),
    ("T5ForSeq2SeqLM", False), ("BertForPreTraining", False),
])
def test_the_architecture_names_that_read_a_vocabulary_head(tmp_path, name, expected):
    t = template()
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [name]}))
    assert t.has_no_output_head(tmp_path, {"model.embed_tokens.weight": {}}) is expected


def test_a_model_that_has_an_lm_head_tensor_has_an_output_head_whatever_its_architecture_is_called(tmp_path):
    t = template()
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3Model"]}))
    assert t.has_no_output_head(tmp_path, {"lm_head.weight": {}}) is False
    assert t.has_no_output_head(tmp_path, {"model.lm_head.weight": {}}) is False


def test_a_model_directory_with_no_config_or_an_unreadable_one_is_not_headless(tmp_path):
    t = template()
    assert t.has_no_output_head(tmp_path, {}) is False
    (tmp_path / "config.json").write_text("{not json")
    assert t.has_no_output_head(tmp_path, {}) is False
    (tmp_path / "config.json").write_text("[1]")
    assert t.has_no_output_head(tmp_path, {}) is False
