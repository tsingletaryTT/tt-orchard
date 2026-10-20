# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 1 template loads the model the way its checkpoint was saved, and uses stage 0's name map when it did not.

On the pplx-decider run (2026-10-19) the CPU model ran on random values: every weight was reported missing. The cause was not the template,
which resolves `Qwen3_5Model` from the checkpoint's architecture and would have loaded it. The stage 1 agent edited its copy of the script
and put `cls = transformers.AutoModelForCausalLM` in its place, and that class found none of the decider's `language_model.*` / `visual.*`
keys. A tiny Qwen3.5 model shows the effect in a second: AutoModelForCausalLM on a headless Qwen3_5Model checkpoint loads nothing, the
model's own class loads everything, and transformers' `key_mapping` repairs the rest. (The guard against an edited script is in
tests/test_gate_reference_evidence.py.) These tests pin what the template itself does: the class follows the architecture (and is found
in the model's own module, or by name, when transformers does not export it), a load that still misses keys is retried once with a remap
derived from stage 0's `alignment`, and a model with no lm_head fails with evidence instead of crashing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "reference-gate-templates" / "reference_gate.py"
STACK = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "tokenizers"))
pytestmark = pytest.mark.skipif(not STACK, reason="SKIPPED: torch, transformers or tokenizers is not importable, so the loader was not tested")


def template():
    spec = importlib.util.spec_from_file_location("reference_gate_loader_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_qwen35(root: Path, *, headless: bool = True, architectures=None) -> Path:
    """A random 4-layer Qwen3.5 (two linear-attention and two full-attention layers) saved like a snapshot. headless: the base class,
    whose keys are `language_model.*` and `visual.*` with no lm_head (the pplx-decider layout)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(["hello world, a boiler broke today"] * 40, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    snap = root / "snapshots" / ("1" * 40)
    snap.mkdir(parents=True)
    fast.save_pretrained(snap)
    torch.manual_seed(0)
    text = dict(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, layer_types=["linear_attention", "full_attention"] * 2,
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                max_position_embeddings=256)
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, in_channels=3, patch_size=4, spatial_merge_size=1,
                  temporal_patch_size=1, out_hidden_size=32, num_position_embeddings=16)
    cfg = Qwen3_5Config(text_config=text, vision_config=vision)
    model = (m.Qwen3_5Model(cfg) if headless else m.Qwen3_5ForConditionalGeneration(cfg)).to(torch.bfloat16)
    model.save_pretrained(snap, safe_serialization=True)
    if architectures:
        data = json.loads((snap / "config.json").read_text())
        data["architectures"] = architectures
        (snap / "config.json").write_text(json.dumps(data))
    return snap


@pytest.fixture(scope="module")
def headless(tmp_path_factory):
    return tiny_qwen35(tmp_path_factory.mktemp("headless"))


def load(snap, remap=None):
    import torch
    import transformers
    t = template()
    return t.load_model(torch, transformers, snap, remap=remap) if remap is not None else t.load_model(torch, transformers, snap)


# ---- the class -------------------------------------------------------------------------------------------------------------------

def test_a_headless_checkpoint_loads_through_its_own_class_with_no_missing_keys(headless):
    model, rep = load(headless)
    assert rep["class"] == "Qwen3_5Model" and rep["missing_keys"] == [] and rep["unexpected_keys"] == [] and rep["pass"] is True


def test_a_class_the_top_level_does_not_export_is_found_in_the_models_own_module(headless):
    """A transformers-like namespace with no `Qwen3_5Model` attribute: the module `transformers.models.qwen3_5.modeling_qwen3_5` has it."""
    from types import SimpleNamespace
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(str(headless))
    bare = SimpleNamespace(AutoModel=transformers.AutoModel, AutoModelForCausalLM=transformers.AutoModelForCausalLM)
    cls = template().resolve_class(bare, cfg, "Qwen3_5Model")
    assert cls.__name__ == "Qwen3_5Model" and cls is not transformers.AutoModel


def test_an_architecture_transformers_cannot_resolve_falls_back_by_name(tmp_path):
    """No class of that name anywhere: a name with `For` is a head model (the causal LM), a bare name is the base model."""
    t = template()
    snap = tiny_qwen35(tmp_path / "a", architectures=["Qwen3_5SomethingNew"])
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(str(snap))
    assert t.resolve_class(transformers, cfg, "Qwen3_5SomethingNew") is transformers.AutoModel
    assert t.resolve_class(transformers, cfg, "Qwen3_5ForSomethingNew") is transformers.AutoModelForCausalLM
    assert t.resolve_class(transformers, cfg, None) is transformers.AutoModelForCausalLM


# ---- the remap -------------------------------------------------------------------------------------------------------------------

DECIDER_RULES = [{"from": "language_model", "to": "model.language_model", "tensors": 850},
                 {"from": "visual", "to": "model.visual", "tensors": 333}]


def write_delta(run: Path, rules):
    d = run / "stages" / "0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "delta.json").write_text(json.dumps({"alignment": {"raw_rules": rules}}))
    return run


def test_the_remap_is_built_from_stage_0s_raw_rules(tmp_path):
    t = template()
    remap = t.remap_from_delta(write_delta(tmp_path, DECIDER_RULES))
    assert remap == {r"^language_model\.": "model.language_model.", r"^visual\.": "model.visual."}


def test_a_rule_that_drops_a_prefix_removes_it_from_the_key(tmp_path):
    remap = template().remap_from_delta(write_delta(tmp_path, [{"from": "text", "to": "", "tensors": 9}]))
    assert remap == {r"^text\.": ""}


def test_a_rule_with_few_tensors_or_no_change_is_not_a_remap(tmp_path):
    t = template()
    rules = [{"from": "x", "to": "y", "tensors": 2}, {"from": "same", "to": "same", "tensors": 99}]
    assert t.remap_from_delta(write_delta(tmp_path, rules)) == {}


def test_a_missing_unreadable_or_oversized_delta_gives_no_remap(tmp_path):
    t = template()
    assert t.remap_from_delta(tmp_path) == {}
    d = tmp_path / "stages" / "0"
    d.mkdir(parents=True)
    (d / "delta.json").write_text("not json")
    assert t.remap_from_delta(tmp_path) == {}
    (d / "delta.json").write_text(json.dumps({"alignment": {"raw_rules": DECIDER_RULES}, "pad": "x" * (3 * 1024 * 1024)}))
    assert t.remap_from_delta(tmp_path) == {}


def test_a_load_that_misses_keys_is_retried_once_with_the_remap_and_both_attempts_are_reported(tmp_path):
    """The decider's failure in miniature: a headless checkpoint read through a causal-LM class finds none of its keys."""
    snap = tiny_qwen35(tmp_path / "wrongclass", architectures=["Qwen3_5ForCausalLM"])
    model, rep = load(snap, remap={r"^language_model\.": "model.language_model.", r"^visual\.": "model.visual."})
    attempts = rep["attempts"]
    assert len(attempts) == 2 and attempts[0]["key_mapping"] is None and attempts[1]["key_mapping"]
    assert len(attempts[0]["missing_keys"]) > 10 and attempts[0]["unexpected_keys"]
    assert rep["missing_keys"] == ["lm_head.weight"] and rep["unexpected_keys"] == []          # only the genuinely absent head
    assert rep["pass"] is False                                                                # still not a pass: there is no lm_head


def test_a_retry_that_does_not_help_is_not_kept(tmp_path):
    """A remap that points the keys nowhere makes things no better: the first load stays the chosen one."""
    snap = tiny_qwen35(tmp_path / "wrongclass3", architectures=["Qwen3_5ForCausalLM"])
    model, rep = load(snap, remap={r"^language_model\.": "nowhere.", r"^visual\.": "nowhere."})
    first, second = rep["attempts"]
    assert first["chosen"] is True and second["chosen"] is False
    assert set(first["missing_keys"]) <= set(rep["missing_keys"]) and rep["unexpected_keys"]       # the report is the first load's


def test_a_load_that_passes_is_never_retried(headless):
    model, rep = load(headless, remap={r"^language_model\.": "model.language_model."})
    assert len(rep["attempts"]) == 1 and rep["pass"] is True


def test_without_a_remap_a_failed_load_is_one_attempt(tmp_path):
    snap = tiny_qwen35(tmp_path / "wrongclass2", architectures=["Qwen3_5ForCausalLM"])
    model, rep = load(snap)
    assert len(rep["attempts"]) == 1 and rep["pass"] is False


# ---- no logits ----------------------------------------------------------------------------------------------------------------------

def run_gate(tmp_path: Path, snap: Path, *, delta_rules=None):
    run = tmp_path / "run"
    stage = run / "stages" / "1"
    stage.mkdir(parents=True)
    if delta_rules is not None:
        write_delta(run, delta_rules)
    (stage / "reference_gate.py").write_text(TEMPLATE.read_text())
    (stage / "reference_config.json").write_text(json.dumps({"run_dir": str(run), "model_snapshot": str(snap), "python": sys.executable}))
    r = subprocess.run([sys.executable, str(stage / "reference_gate.py")], capture_output=True, text=True, timeout=600)
    return stage, r


def test_a_model_with_no_lm_head_fails_with_evidence_instead_of_crashing(headless, tmp_path):
    stage, r = run_gate(tmp_path, headless)
    assert r.returncode == 3, r.stderr[-600:]
    ref = json.loads((stage / "reference.json").read_text())
    summary = json.loads((stage / "evidence" / "summary.json").read_text())
    assert ref["verdict"] == "fail" and summary["verdict"] == "fail"
    assert summary["template_sha256"] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    note = next(c["note"] for c in ref["checks"] if c["name"] == "decodes forward")
    assert "no lm_head" in note and "logits" in note
    assert json.loads((stage / "evidence" / "load-report.json").read_text())["missing_keys"] == []        # it did load


def test_a_load_that_raises_also_writes_the_summary_the_gate_reads(tmp_path):
    snap = tiny_qwen35(tmp_path / "m")
    (snap / "model.safetensors").unlink()                         # from_pretrained now raises
    stage, r = run_gate(tmp_path, snap)
    assert r.returncode == 3
    summary = json.loads((stage / "evidence" / "summary.json").read_text())
    assert summary["verdict"] == "fail" and summary["checks"]["loads"] is False
    assert summary["template_sha256"] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
