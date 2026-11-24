# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard tier-plan` fit headroom (Inkling-2-Mini study, 2026-11-24, learning 2).

The plan reserves 3 GiB per chip and a flat 4 GiB of KV cache before it counts weights, whatever the model's attention layout. For
Inkling-2-Mini that makes one chip with all-bfp8 weights "fit with host experts", while Raahem's published bundle serves it on one die.
A row that misses only because of the reserve and the KV share now says how large the reserve plus the KV share may be for it to fit on the
chips, so a reader sees which assumption decides the row. No constant and no verdict changes."""
import json
from pathlib import Path

import pytest

from orchard import tier_plan

GIB = float(1 << 30)
FIX = Path(__file__).parent / "fixtures" / "model-configs"
INKLING = json.loads((FIX / "inkling-2-mini.config.json").read_text())

# the tiny config of tests/test_tier_plan.py: bf16 weights are 46,592 always-resident and 98,304 expert parameters, two bytes each
CFG = {"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
       "head_dim": 16, "vocab_size": 100, "intermediate_size": 128, "moe_intermediate_size": 32, "num_experts": 8,
       "num_experts_per_tok": 2, "tie_word_embeddings": False}
ALL_BF16 = (46_592 + 98_304) * 2                         # 289,792 bytes: the replicated fit


def place(chips, usable, overhead, fmts=("bf16", "bf16")):
    """`usable` is what one chip's DRAM allows for weights before the reserve and the KV share; the budget is that less `overhead`."""
    return tier_plan.place(tier_plan.shape(CFG), fmts[0], fmts[1], chips, usable - overhead, usable_bytes=usable, overhead_bytes=overhead)


def test_a_row_that_misses_by_the_overhead_says_how_much_overhead_it_could_carry():
    # the chip offers 300,000 bytes, 100,000 are held back, so the budget is 200,000 and the 289,792 bytes of weights do not fit
    r = place(1, 300_000, 100_000)
    assert r["verdict"] == "fits with host experts"
    assert r["overhead_limit_for_chips_bytes"] == pytest.approx(300_000 - ALL_BF16)
    assert r["assumed_overhead_bytes"] == 100_000


def test_the_limit_uses_the_split_layout_when_it_needs_less_per_chip():
    # two chips: the split layout needs 93,184 + 196,608 / 2 = 191,488 per chip; the usable 300,000 leaves 108,512
    r = place(2, 300_000, 150_000)
    assert r["verdict"] == "fits with host experts"
    assert r["overhead_limit_for_chips_bytes"] == pytest.approx(300_000 - 191_488)


def test_without_an_overhead_the_row_has_no_limit():
    r = tier_plan.place(tier_plan.shape(CFG), "bf16", "bf16", 1, 200_000)
    assert r["overhead_limit_for_chips_bytes"] is None and r["assumed_overhead_bytes"] is None


def test_a_row_that_already_fits_on_the_chips_has_no_limit():
    r = place(1, 299_792, 10_000)
    assert r["verdict"] == "fits on chips"
    assert r["overhead_limit_for_chips_bytes"] is None


def test_a_row_that_would_not_fit_even_with_no_overhead_has_no_limit():
    # usable 100,000 bytes against 289,792 of weights on one chip: removing the whole overhead would not help
    r = place(1, 100_000, 10_000)
    assert r["overhead_limit_for_chips_bytes"] is None


def _res(**kw):
    kw.setdefault("chips_list", (1, 2, 4))
    return tier_plan.plan(INKLING, chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0, **kw)


def row(res, name, chips):
    return next(r for r in res["rows"] if r["name"] == name and r["chips"] == chips)


def test_inkling_all_bfp8_on_one_chip_misses_by_the_assumed_overhead_and_the_plan_says_so():
    r = row(_res(), "all bfp8", 1)
    assert r["verdict"] == "fits with host experts"
    # usable 32 x 0.934 = 29.888 GiB; the weights replicated take 25.75 GiB, so 4.14 GiB may go to reserve and KV; the plan assumes 3 + 4 = 7
    assert r["overhead_limit_for_chips_bytes"] / GIB == pytest.approx(29.888 - 25.75, abs=0.01)
    assert r["assumed_overhead_bytes"] / GIB == pytest.approx(7.0)


def test_the_rendered_plan_names_the_row_the_limit_and_the_flags_that_set_it():
    text = tier_plan.render(_res())
    line = next(l for l in text.splitlines() if l.startswith("all bfp8, 1 chip(s)") and "would fit on the chips" in l)
    assert "4.1" in line and "7.0" in line
    assert "--reserve-gib" in line and "--kv-gib" in line


def test_the_limit_is_in_the_json_and_changes_no_verdict():
    base = _res()
    for r in base["rows"]:
        assert r["verdict"] in ("fits on chips", "fits with host experts", "does not fit")
    assert row(base, "all bfp8", 4)["overhead_limit_for_chips_bytes"] is None      # fits on 4 chips already
    assert json.dumps(base)                                                       # serialisable
