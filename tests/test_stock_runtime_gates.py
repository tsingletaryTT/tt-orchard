# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The two gates of the stock-runtime path: stage 0's delta.json (written by supervisor code, so no template stamp)
and stage 2's evidence/stock-check.json (written by the shipped template, which the gate re-reads and re-judges)."""
import json
from pathlib import Path

import pytest

from orchard import stages
from orchard.stages import gate_delta_stock, gate_stock_runtime
from test_stages import write

STOCK = {"name": "Llama-3.1-8B-Instruct", "matched": "unsloth/Llama-3.1-8B-Instruct", "via": "model_params",
         "boards": ["N150", "P100", "P150"], "blackhole": True, "params_b": 8.0, "layers": 32,
         "config_equal": True, "keys_differing": [], "layout_problems": [],
         "params_dir": "models/tt_transformers/model_params/Llama-3.1-8B-Instruct"}
DELTA = {"model": "unsloth/Llama-3.1-8B-Instruct", "nearest_model": "tt-transformers:Llama-3.1-8B-Instruct",
         "path": "stock-runtime", "class": "stock-runtime", "stock_runtime": STOCK, "sidecars": [],
         "differences": [{"area": a, "finding": f"{a} checked", "evidence": ["stages/0/evidence/diff.txt"]}
                         for a in ("config", "tensors", "tensor_names")],
         "hazards": [{"area": "other", "finding": "run on the host tree"}]}


def stage0(tmp_path, data):
    run = tmp_path / "run"
    write(run, "stages/0/evidence/diff.txt", "evidence")
    write(run, "stages/0/delta.json", data)
    return run / "stages" / "0", run


def test_a_good_stock_delta_passes_with_no_template_in_the_stage(tmp_path):
    g = gate_delta_stock(*stage0(tmp_path, DELTA))
    assert g.ok, g.reasons


@pytest.mark.parametrize("mutate,word", [
    (lambda d: d.update(path="weights-only"), "path"),
    (lambda d: d.update({"class": "weights-only"}), "class"),
    (lambda d: d.pop("stock_runtime"), "stock_runtime"),
    (lambda d: d["stock_runtime"].update(config_equal=False), "config_equal"),
    (lambda d: d["stock_runtime"].update(layout_problems=["no lm_head"]), "layout_problems"),
    (lambda d: d["stock_runtime"].update(keys_differing=["hidden_size"]), "keys_differing"),
    (lambda d: d["stock_runtime"].update(name=""), "name"),
    (lambda d: d["stock_runtime"].update(params_b="8"), "params_b"),
    (lambda d: d["stock_runtime"].update(params_b=True), "params_b"),
    (lambda d: d.update(sidecars=[{"file": "x"}]), "sidecars"),
    (lambda d: d.update(differences=[]), "differences"),
    (lambda d: d.pop("nearest_model"), "nearest_model"),
])
def test_a_stock_delta_that_does_not_support_its_class_fails(tmp_path, mutate, word):
    d = json.loads(json.dumps(DELTA))
    mutate(d)
    g = gate_delta_stock(*stage0(tmp_path, d))
    assert not g.ok and any(word in r for r in g.reasons), g.reasons


def test_a_full_port_answer_from_the_stock_triage_is_a_valid_stage_zero_result(tmp_path):
    # the measurement decides: a model whose config differs is a full port, and the run then stops as any full port does
    d = {**DELTA, "path": "full-port", "class": "full-port",
         "stock_runtime": {**STOCK, "config_equal": False, "keys_differing": ["num_hidden_layers"]},
         "path_reasons": ["config differs: num_hidden_layers"]}
    g = gate_delta_stock(*stage0(tmp_path, d))
    assert g.ok, g.reasons


# ---- stage 2 ----------------------------------------------------------------------------------------------

GOOD_TEXT = ("Paris is the capital of France. It sits on the Seine and has been a centre of art, science and politics "
             "for centuries, and its museums draw visitors from around the world every year.")
CHECK = {"model_id": "unsloth/Llama-3.1-8B-Instruct", "chips": 2, "device_name": "P300", "optimizations": "accuracy",
         "tt_metal": {"tag": "v0.80.0", "dirty": False},
         "teacher_forced": {"tokens": 511, "top1": 0.90, "top5": 0.98},
         "free_run": {"prompt": "What is the capital of France?", "text": GOOD_TEXT, "new_tokens": 32, "finish": "length"},
         "speed": {"decode_tokens_per_s": 40.0, "ttft_s": 0.2},
         "reference": {"source": "generated", "model_id": "unsloth/Llama-3.1-8B-Instruct", "snapshot": "4699cc75", "sha256": "a" * 64},
         "hub_fetch": [], "scratch_patch": "evidence/scratch.patch"}


def stage2(tmp_path, check=None, *, delta=None, template=True, copy=True):
    run = tmp_path / "run"
    write(run, "stages/0/evidence/diff.txt", "e")
    write(run, "stages/0/delta.json", delta or DELTA)
    sd = run / "stages" / "2"
    text = "print('stock check')\n"
    sha = stages._sha256(stages.SHIPPED_TEMPLATES["stock_check.py"])
    if copy:
        (sd).mkdir(parents=True, exist_ok=True)
        (sd / "stock_check.py").write_text(stages.SHIPPED_TEMPLATES["stock_check.py"].read_text())
    c = dict(CHECK if check is None else check)
    if template:
        c.setdefault("template_sha256", sha)
    write(run, "stages/2/evidence/stock-check.json", c)
    write(run, "stages/2/evidence/scratch.patch", "")
    return sd, run


def test_a_good_stage_two_check_passes(tmp_path):
    g = gate_stock_runtime(*stage2(tmp_path))
    assert g.ok, g.reasons
    assert "stages/2/evidence/stock-check.json" in g.evidence or "evidence/stock-check.json" in g.evidence


def change(**top):
    c = json.loads(json.dumps(CHECK))
    for k, v in top.items():
        c[k] = v
    return c


def tf(**kw):
    return change(teacher_forced={**CHECK["teacher_forced"], **kw})


@pytest.mark.parametrize("check,word", [
    (tf(top1=0.84), "top1"),
    (tf(top5=0.94), "top5"),
    (tf(top1="0.9"), "top1"),
    (tf(top1=True), "top1"),
    (tf(top1=float("nan")), "top1"),
    (tf(top5=1.5), "top5"),
    (tf(tokens=127), "tokens"),
    (tf(tokens=True), "tokens"),
    (change(device_name=""), "device_name"),
    (change(device_name=None), "device_name"),
    (change(chips=3), "chips"),
    (change(optimizations="turbo"), "optimizations"),
    (change(tt_metal={"tag": "", "dirty": False}), "tt_metal"),
    (change(tt_metal=None), "tt_metal"),
    (change(hub_fetch=["unsloth/x"]), "Hub"),
    (change(free_run={"prompt": "p", "text": "", "new_tokens": 0, "finish": "length"}), "free_run"),
    (change(free_run={"prompt": "p", "text": "a" * 5, "new_tokens": 3, "finish": "length"}), "free_run"),
    (change(free_run={"prompt": "p", "text": "ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok",
                      "new_tokens": 24, "finish": "length"}), "free_run"),
    (change(free_run={"prompt": "p", "text": GOOD_TEXT + " <|eot_id|>", "new_tokens": 32, "finish": "stop"}), "free_run"),
    (change(speed={"decode_tokens_per_s": 0, "ttft_s": 0.2}), "speed"),
    (change(speed={"decode_tokens_per_s": "x", "ttft_s": 0.2}), "speed"),
    (change(model_id="other/model"), "model_id"),
    # the reference must be made from this run's snapshot (review of PR 78)
    (change(reference=None), "reference"),
    (change(reference={**CHECK["reference"], "source": "tree"}), "reference"),
    (change(reference={**CHECK["reference"], "model_id": "other/model"}), "reference"),
    (change(reference={**CHECK["reference"], "snapshot": ""}), "reference"),
    (change(reference={**CHECK["reference"], "sha256": "xyz"}), "reference"),
])
def test_a_stage_two_check_that_does_not_meet_the_bars_fails(tmp_path, check, word):
    g = gate_stock_runtime(*stage2(tmp_path, check))
    assert not g.ok and any(word in r for r in g.reasons), g.reasons


def test_the_bar_is_the_small_model_bar_for_a_small_model(tmp_path):
    small = {**DELTA, "stock_runtime": {**STOCK, "name": "Llama-3.2-1B-Instruct", "params_b": 1.2}}
    ok = gate_stock_runtime(*stage2(tmp_path, tf(top1=0.78), delta=small))
    assert ok.ok, ok.reasons
    bad = gate_stock_runtime(*stage2(tmp_path, tf(top1=0.74), delta=small))
    assert not bad.ok and any("top1" in r for r in bad.reasons)
    # the same 0.78 does not pass for the 8B
    big = gate_stock_runtime(*stage2(tmp_path, tf(top1=0.78)))
    assert not big.ok


def test_the_bar_boundary_is_inclusive(tmp_path):
    assert gate_stock_runtime(*stage2(tmp_path, tf(top1=0.85, top5=0.95))).ok


def test_the_stage_needs_the_shipped_template_and_its_stamp(tmp_path):
    g = gate_stock_runtime(*stage2(tmp_path, copy=False))
    assert not g.ok and any("stock_check.py" in r for r in g.reasons)
    g = gate_stock_runtime(*stage2(tmp_path, template=False))
    assert not g.ok and any("template_sha256" in r for r in g.reasons)
    sd, run = stage2(tmp_path)
    (sd / "stock_check.py").write_text("print('edited')\n")
    g = gate_stock_runtime(sd, run)
    assert not g.ok and any("differs from the shipped template" in r for r in g.reasons)


def test_a_missing_or_unreadable_check_fails(tmp_path):
    sd, run = stage2(tmp_path)
    (sd / "evidence" / "stock-check.json").write_text("{")
    assert not gate_stock_runtime(sd, run).ok
    (sd / "evidence" / "stock-check.json").unlink()
    g = gate_stock_runtime(sd, run)
    assert not g.ok and any("stock-check.json" in r for r in g.reasons)


def test_a_stage_two_for_a_run_with_no_stock_record_fails(tmp_path):
    sd, run = stage2(tmp_path, delta={**DELTA, "class": "weights-only", "path": "weights-only"})
    g = gate_stock_runtime(sd, run)
    assert not g.ok and any("stock" in r for r in g.reasons)


def test_the_patch_file_is_required_evidence(tmp_path):
    sd, run = stage2(tmp_path)
    (sd / "evidence" / "scratch.patch").unlink()
    g = gate_stock_runtime(sd, run)
    assert not g.ok and any("scratch_patch" in r for r in g.reasons)


def test_a_pinned_revision_in_stage_zero_must_be_the_snapshot_the_reference_was_made_from(tmp_path):
    pinned = {**DELTA, "model": "unsloth/Llama-3.1-8B-Instruct@4699cc75"}
    assert gate_stock_runtime(*stage2(tmp_path, delta=pinned)).ok
    other = change(reference={**CHECK["reference"], "snapshot": "ffff0000"})
    g = gate_stock_runtime(*stage2(tmp_path, other, delta=pinned))
    assert not g.ok and any("reference" in r and "4699cc75" in r for r in g.reasons)
    # an unpinned model id has no revision to compare
    assert gate_stock_runtime(*stage2(tmp_path, other)).ok
