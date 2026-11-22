# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 789: stage 1 can record a longer sampled reference for the stage 2 fidelity score.

The greedy reference is 32 tokens and stays so (the agreement check reads it). With `fidelity_tokens` in `reference_config.json` the template
also samples that many tokens at temperature 1 from the same prompt, keeps the log-probability of each, and writes
`evidence/reference/fidelity-ids.json`. The tests run the template for real on a tiny random Qwen2 (CPU) the way test_reference_gate_template does."""
from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from test_reference_gate_template import STACK, TEMPLATE, build_model, ev, needs_stack, template_module

N = 100


def run_with(tmp_path, snap, **config):
    stage = tmp_path / "run" / "stages" / "1"
    stage.mkdir(parents=True)
    (stage / "reference_gate.py").write_text(TEMPLATE.read_text())
    (stage / "reference_config.json").write_text(json.dumps(
        {"run_dir": str(tmp_path / "run"), "model_snapshot": str(snap), "python": sys.executable, **config}))
    r = subprocess.run([sys.executable, str(stage / "reference_gate.py")], capture_output=True, text=True, timeout=600)
    return stage, r


@pytest.fixture(scope="module")
def sampled(tmp_path_factory):
    if not STACK:
        pytest.skip("SKIPPED: torch, transformers or tokenizers is not importable")
    tmp = tmp_path_factory.mktemp("fidtok")
    snap = build_model(tmp / "model", thai=False)
    stage, r = run_with(tmp / "a", snap, fidelity_tokens=N)
    stage2, r2 = run_with(tmp / "b", snap, fidelity_tokens=N)
    plain, rp = run_with(tmp / "c", snap)
    return {"snap": snap, "stage": stage, "r": r, "stage2": stage2, "r2": r2, "plain": plain, "rp": rp}


def fid(stage):
    return json.loads(ev(stage, "reference", "fidelity-ids.json").read_text())


@needs_stack
def test_the_option_writes_a_complete_sampled_reference_of_the_asked_length(sampled):
    assert sampled["r"].returncode == 0, sampled["r"].stdout + sampled["r"].stderr
    f = fid(sampled["stage"])
    assert f["complete"] is True and f["temperature"] == 1.0 and isinstance(f["seed"], int)
    assert len(f["ids"]) == N and len(f["logprobs"]) == N
    assert all(isinstance(v, float) and math.isfinite(v) and v <= 0 for v in f["logprobs"])
    assert all(isinstance(i, int) and not isinstance(i, bool) for i in f["ids"])


@needs_stack
def test_the_greedy_reference_stays_32_tokens_and_the_stage_still_passes_its_gate(sampled):
    from orchard.stages import gate_reference
    stage = sampled["stage"]
    assert len(json.loads(ev(stage, "reference", "generated-ids.json").read_text())["generated_ids"]) == 32
    assert len(json.loads(ev(stage, "reference", "top5-logits.json").read_text())["steps"]) == 32
    gate = gate_reference(stage, stage.parent.parent)
    assert gate.ok, gate.reasons


@needs_stack
def test_the_sampled_text_is_sampled_and_not_the_greedy_text(sampled):
    greedy = json.loads(ev(sampled["stage"], "reference", "generated-ids.json").read_text())["generated_ids"]
    assert fid(sampled["stage"])["ids"][:32] != greedy      # at temperature 1 on a random model the two differ at once


@needs_stack
def test_the_same_seed_gives_the_same_sampled_reference_twice(sampled):
    a, b = fid(sampled["stage"]), fid(sampled["stage2"])
    assert a["seed"] == b["seed"] and a["ids"] == b["ids"] and a["logprobs"] == b["logprobs"]


@needs_stack
def test_each_logprob_is_the_models_log_probability_of_its_token_given_the_prefix_without_the_cache(sampled):
    # The sampling loop uses the key-value cache. Its numbers must match a full forward pass over prompt plus the ids before the token.
    import torch
    from transformers import AutoModelForCausalLM
    stage, snap = sampled["stage"], sampled["snap"]
    f = fid(stage)
    prompt = json.loads(ev(stage, "reference", "prompt-ids.json").read_text())["prompt_ids"]
    model = AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.bfloat16)
    model.eval()
    diffs = []
    for k in list(range(0, N, 7)) + [N - 1]:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([prompt + f["ids"][:k]]), use_cache=False).logits[0, -1].float()
        diffs.append(abs(float(torch.log_softmax(logits, -1)[f["ids"][k]]) - f["logprobs"][k]))
    # measured on this model: cached and uncached agree to 5e-7; a loop that dropped the cache and fed only the last token is off by 0.06 on average (0.28 at worst)
    assert max(diffs) < 1e-3, diffs


@needs_stack
def test_the_record_is_bound_to_its_prompt(sampled):
    import hashlib
    stage = sampled["stage"]
    prompt = json.loads(ev(stage, "reference", "prompt-ids.json").read_text())["prompt_ids"]
    want = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
    assert fid(stage)["prompt_ids_sha256"] == want


@needs_stack
def test_without_the_option_no_sampled_reference_is_written(sampled):
    assert sampled["rp"].returncode == 0
    assert not ev(sampled["plain"], "reference", "fidelity-ids.json").exists()
    assert "fidelity_reference" not in json.loads(ev(sampled["plain"], "summary.json").read_text())


@needs_stack
def test_the_summary_names_the_sampled_reference_and_the_artifact_list_carries_it(sampled):
    stage = sampled["stage"]
    s = json.loads(ev(stage, "summary.json").read_text())["fidelity_reference"]
    assert s["tokens"] == N and s["complete"] is True and s["seconds"] >= 0
    ref = json.loads((stage / "reference.json").read_text())
    assert ref["reference_artifacts"]["fidelity_ids"].endswith("reference/fidelity-ids.json")


@needs_stack
@pytest.mark.parametrize("bad", [3, 15, 8193, -1, "100", 100.0, True, None])
def test_a_bad_fidelity_tokens_value_exits_2_before_the_model_loads(tmp_path, bad):
    stage = tmp_path / "run" / "stages" / "1"
    stage.mkdir(parents=True)
    (stage / "reference_gate.py").write_text(TEMPLATE.read_text())
    (stage / "reference_config.json").write_text(json.dumps(
        {"run_dir": str(tmp_path / "run"), "model_snapshot": str(tmp_path), "python": sys.executable, "fidelity_tokens": bad}))
    r = subprocess.run([sys.executable, str(stage / "reference_gate.py")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "fidelity_tokens" in r.stderr, r.stderr


def test_zero_and_a_missing_key_mean_off_and_the_range_matches_stage_2s_cap_limit():
    mod = template_module()
    assert mod.fidelity_setting({}) == 0 and mod.fidelity_setting({"fidelity_tokens": 0}) == 0
    assert mod.fidelity_setting({"fidelity_tokens": 4000}) == 4000
    assert mod.FIDELITY_MAX == 8192 and mod.FIDELITY_MIN == 16


def test_a_failure_while_sampling_is_recorded_and_does_not_stop_the_stage():
    mod = template_module()

    class Broken:
        def __call__(self, **kw):
            raise RuntimeError("no cache for this model")

    class FakeTorch:
        long = None

        class Generator:
            def manual_seed(self, s):
                return self

        @staticmethod
        def inference_mode():
            import contextlib
            return contextlib.nullcontext()

        @staticmethod
        def tensor(*a, **k):
            return None

    out = mod.sample_reference(FakeTorch, Broken(), [1, 2, 3], 20, 7, writer=lambda doc: None)
    assert out["complete"] is False and "no cache for this model" in out["error"] and out["ids"] == []
