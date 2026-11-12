# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/expert_skew.py: the skew figures from per-expert pick counts (issue 723)."""
import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("expert_skew", Path(__file__).resolve().parent.parent / "scripts" / "expert_skew.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)


def test_a_uniform_router_gives_the_share_of_experts_and_full_entropy():
    r = es.skew_from_counts([10] * 20)
    assert r["top_10pct_share"] == pytest.approx(0.10) and r["top_25pct_share"] == pytest.approx(0.25) and r["top_50pct_share"] == pytest.approx(0.50)
    assert r["entropy_vs_max"] == pytest.approx(1.0) and r["never_picked"] == 0


def test_one_expert_taking_every_pick_is_fully_skewed():
    r = es.skew_from_counts([100] + [0] * 9)
    assert r["top_10pct_share"] == 1.0 and r["never_picked"] == 0.9 and r["entropy_vs_max"] == pytest.approx(0.0)


def test_the_busiest_share_counts_the_largest_experts_not_the_first_ones():
    counts = [1, 1, 1, 1, 1, 1, 1, 1, 50, 42]
    assert es.skew_from_counts(counts)["top_25pct_share"] == pytest.approx((50 + 42 + 1) / 100)        # 25% of 10 experts rounds up to 3
    assert es.skew_from_counts(counts)["top_10pct_share"] == pytest.approx(0.50)


def test_entropy_is_between_the_two_extremes_for_a_moderate_skew():
    r = es.skew_from_counts([40, 30, 20, 10])
    assert 0.8 < r["entropy_vs_max"] < 1.0 and r["entropy_vs_max"] == pytest.approx(-sum(p * math.log(p) for p in (0.4, 0.3, 0.2, 0.1)) / math.log(4))


def test_no_picks_or_no_experts_are_refused():
    for bad in ([], [0, 0]):
        with pytest.raises(ValueError):
            es.skew_from_counts(bad)


def test_the_mean_over_layers_averages_each_figure_and_leaves_out_the_counts():
    layers = [es.skew_from_counts([10] * 10), es.skew_from_counts([100] + [0] * 9)]
    m = es.mean_over_layers(layers)
    assert m["top_10pct_share"] == pytest.approx((0.1 + 1.0) / 2) and "picks" not in m and "experts" not in m
