# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/ab_sheet_counterfactual.py: the form counter and the mechanical rewrite (the simulator and compiler steps are covered by the
runs recorded in docs/run-logs/2026-11-22-plain-sheet-chip-claims.md; these tests need neither)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ab_sheet_counterfactual", REPO / "scripts" / "ab_sheet_counterfactual.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

OLD = """
    one = ttl.math.fill(1.0, shape=(1, 1))
    total = ttl.math.reduce_sum(x_blk * x_blk, dims=[-1])
    peak = ttl.math.reduce_max(x_blk, dims=[-1])
    row = ttl.math.broadcast(total, dims=[-1], shape=(1, col_tiles))
"""
GOOD = """
    one = ttl.block.fill(1.0, shape=(1, 1))
    total = ttl.math.reduce_sum(x_blk * x_blk, dims=[-1], shape=(1, 1))
    row = ttl.block.broadcast(total, dims=[-1], shape=(1, col_tiles))
"""


def test_the_counter_finds_each_wrong_claim_form_and_ignores_the_right_ones():
    old, good = mod.count_forms(OLD), mod.count_forms(GOOD)
    assert old["ttl.math.fill"] == 1 and old["ttl.math.broadcast"] == 1 and old["reduce without shape"] == 2
    assert set(mod.WRONG) == {"ttl.math.fill", "ttl.math.broadcast", "reduce without shape"}, "the script treats each wrong-claim form as wrong"
    assert all(good[name] == 0 for name in mod.WRONG)
    assert good["ttl.block.fill"] == 1 and good["ttl.block.broadcast"] == 1 and good["reduce with shape"] == 1


def test_the_rewrite_changes_only_the_wrong_forms_and_a_second_pass_changes_nothing():
    new = mod.rewrite(OLD)
    assert "ttl.math.fill" not in new and "ttl.math.broadcast" not in new
    assert "reduce_sum(x_blk * x_blk, dims=[-1], shape=(1, 1))" in new and "reduce_max(x_blk, dims=[-1], shape=(1, 1))" in new
    assert mod.rewrite(new) == new
    assert mod.rewrite(GOOD) == GOOD, "a kernel with the right forms is left alone"


def test_a_reduction_that_already_has_shape_is_not_given_a_second_one():
    text = "t = ttl.math.reduce_sum(x, dims=[-1], shape=(1, col_tiles))\n"
    assert mod.rewrite(text) == text
