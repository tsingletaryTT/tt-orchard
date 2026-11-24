# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard tier-plan` tensor-parallel placement (Inkling-2-Mini study, 2026-11-24, learning 3).

Raahem's published Inkling-2-Mini bundle serves the model on 1, 2 or 4 dies by dividing the attention heads, the experts' intermediate width and
the LM head over the dies; the bundle asserts that the query and key-value head counts divide by the die count. The plan offered only
"replicated" and "experts split", so a model that fits only when attention is divided too was reported as needing host experts. The new row is a
layout the plan assumes, and it names the counts that must divide."""
import json
from pathlib import Path

import pytest

from orchard import tier_plan

GIB = float(1 << 30)
FIX = Path(__file__).parent / "fixtures" / "model-configs"
INKLING = json.loads((FIX / "inkling-2-mini.config.json").read_text())
HYBRID = json.loads((FIX / "qwen3.5-35b-a3b.json").read_text())

# hidden 64, 2 layers, 4 heads of 16, 4 KV heads, vocab 100, 8 experts of width 64 (top 2), untied head. By hand, in parameters:
# attention 2 x 16,384 = 32,768; router 2 x 512 = 1,024; embedding 6,400; head 6,400; one expert 3 x 64 x 64 = 12,288; all experts 8 x 2 x 12,288 = 196,608
CFG = {"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
       "head_dim": 16, "vocab_size": 100, "intermediate_size": 128, "moe_intermediate_size": 64, "num_experts": 8,
       "num_experts_per_tok": 2, "tie_word_embeddings": False}
# bf16 bytes: replicated 243,200 x 2 = 486,400; experts split over 2 = (46,592 + 98,304) x 2 = 289,792;
# tensor-parallel over 2 = ((32,768 + 6,400 + 196,608) / 2 + 1,024 + 6,400) x 2 = 250,624
REPLICATED, SPLIT2, TP2 = 486_400, 289_792, 250_624


def place(cfg, chips, budget, fmts=("bf16", "bf16")):
    return tier_plan.place(tier_plan.shape(cfg), fmts[0], fmts[1], chips, budget)


def test_a_model_that_fits_only_when_attention_is_divided_too_gets_the_tensor_parallel_row():
    r = place(CFG, 2, 260_000)                       # the split layout needs 289,792 per chip, tensor-parallel 250,624
    assert r["verdict"] == "fits on chips"
    assert r["placement"].startswith("tensor-parallel")
    assert r["per_chip_bytes"] == pytest.approx(TP2)
    assert r["resident_experts_per_layer"] == 8 and r["hit_share"] == 1.0


def test_the_row_says_it_is_a_layout_the_plan_assumes_and_names_what_must_divide():
    r = place(CFG, 2, 260_000)
    note = r["tensor_parallel_note"]
    assert "layout the plan assumes" in note
    assert "attention heads" in note and "key-value heads" in note and "divisible by 2" in note
    assert "routers and the embedding table stay on every die" in note


def test_the_simpler_layouts_are_still_tried_first():
    assert place(CFG, 2, REPLICATED)["placement"] == "replicated"
    assert place(CFG, 2, SPLIT2)["placement"] == "experts split over the chips"
    assert place(CFG, 2, SPLIT2 - 1)["placement"].startswith("tensor-parallel")       # one byte less and only the new layout fits


def test_one_chip_has_nothing_to_divide():
    r = place(CFG, 1, 250_000)
    assert r["verdict"] != "fits on chips" and r["tensor_parallel_note"] is None


def test_a_head_count_that_does_not_divide_blocks_the_layout_and_the_row_says_which():
    cfg = dict(CFG, num_key_value_heads=2)           # 2 key-value heads cannot be divided over 4 dies
    r = place(cfg, 4, 150_000)      # neither replicated (486,400) nor split (191,488 per chip) fits
    assert r["placement"] != "tensor-parallel: about 1/4 of the weights per die"
    assert "key-value heads (2)" in r["tensor_parallel_blocked"]
    assert r["verdict"] in ("fits with host experts", "does not fit")


def test_an_expert_width_that_does_not_fill_whole_tiles_on_every_die_blocks_the_layout():
    cfg = dict(CFG, moe_intermediate_size=32)        # 32 / 2 = 16 columns per die, half a tile
    r = place(cfg, 2, 150_000)
    assert r["tensor_parallel_blocked"] and "expert intermediate width (32)" in r["tensor_parallel_blocked"]


def test_a_row_that_fits_without_the_layout_carries_no_note():
    r = place(CFG, 2, REPLICATED)
    assert r["tensor_parallel_note"] is None and r["tensor_parallel_blocked"] is None


def test_the_counts_come_from_the_config():
    s = tier_plan.shape(INKLING)
    got = {label: value for label, value, _ in s.tp_counts}
    assert got["attention heads"] == 24 and got["key-value heads"] == 8
    assert got["expert intermediate width"] == 1152


def test_inkling_all_bf16_on_two_chips_fits_when_divided_over_the_dies():
    res = tier_plan.plan(INKLING, chips_list=(2,), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    r = next(x for x in res["rows"] if x["name"] == "all bf16")
    # (attention 622,854,144 + head 463,159,296 + routed experts 24,461,180,928) / 2 + (router 7,077,888 + embedding 463,159,296), all x 2 bytes
    want = ((622_854_144 + 463_159_296 + 24_461_180_928) / 2 + 7_077_888 + 463_159_296) * 2
    assert r["placement"].startswith("tensor-parallel") and r["verdict"] == "fits on chips"
    assert r["per_chip_bytes"] == pytest.approx(want)
    assert want / GIB == pytest.approx(24.67, abs=0.01)


def test_inkling_rows_that_fit_without_the_layout_do_not_change():
    res = tier_plan.plan(INKLING, chips_list=(1, 4), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    by = {(x["name"], x["chips"]): x for x in res["rows"]}
    assert by[("all bfp8", 4)]["placement"] == "replicated"
    assert by[("all bfp8", 1)]["verdict"] == "fits with host experts"


def test_a_hybrid_model_names_its_linear_attention_heads_among_the_counts():
    s = tier_plan.shape(HYBRID)
    labels = [label for label, _, _ in s.tp_counts]
    assert "attention heads" in labels
    assert any(label.startswith("linear-attention") for label in labels)
    for chips in (2, 4):
        blocked = tier_plan.tp_blockers(s, chips)
        bad = [(l, v, u) for l, v, u in s.tp_counts if v % (chips * u)]
        assert bool(blocked) == bool(bad)


def test_the_text_output_prints_the_layout_note_for_a_tensor_parallel_row():
    res = tier_plan.plan(INKLING, chips_list=(2,), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    text = tier_plan.render(res)
    assert any(l.startswith("all bf16, 2 chip(s): tensor-parallel") and "layout the plan assumes" in l for l in text.splitlines())


def test_the_layout_is_not_claimed_when_it_does_not_fit_either():
    r = place(CFG, 2, 120_000)                       # tensor-parallel needs 250,624 per die
    assert r["verdict"] == "fits with host experts"
    assert not r["placement"].startswith("tensor-parallel") and r["tensor_parallel_note"] is None


def test_the_text_output_says_why_the_layout_was_not_offered():
    cfg = dict(CFG, num_key_value_heads=2)
    res = tier_plan.plan(cfg, chips_list=(4,), chip_gib=0.00015, reserve_gib=0.0, kv_gib=0.0)
    text = tier_plan.render(res)
    assert any("tensor-parallel layout over 4 dies needs these counts divisible by 4" in l and "key-value heads (2)" in l for l in text.splitlines())


def test_a_row_that_fits_by_the_tensor_parallel_layout_has_no_overhead_limit():
    # usable 400,000 bytes, overhead 140,000, so a budget of 260,000: only the tensor-parallel layout (250,624) fits
    r = tier_plan.place(tier_plan.shape(CFG), "bf16", "bf16", 2, 260_000, usable_bytes=400_000, overhead_bytes=140_000)
    assert r["placement"].startswith("tensor-parallel") and r["overhead_limit_for_chips_bytes"] is None
