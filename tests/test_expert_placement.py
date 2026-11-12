# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Expert placement across chips (issue 723, item 3): who owns which expert, and that splitting the computation by owner gives the same answer."""
import pytest
import torch

from orchard import expert_placement as ep


def test_experts_split_into_equal_contiguous_blocks_when_nothing_is_known_about_load():
    p = ep.place_experts(n_experts=8, chips=4, bytes_per_expert=10, budget_bytes_per_chip=1000)
    assert p.owner == [0, 0, 1, 1, 2, 2, 3, 3]
    assert p.experts_per_chip == [2, 2, 2, 2]
    assert p.bytes_per_chip == [20, 20, 20, 20]


def test_a_count_that_does_not_divide_gives_the_extra_experts_to_the_first_chips():
    p = ep.place_experts(n_experts=10, chips=4, bytes_per_expert=1, budget_bytes_per_chip=100)
    assert p.experts_per_chip == [3, 3, 2, 2]
    assert sorted(set(p.owner)) == [0, 1, 2, 3]


def test_a_placement_that_cannot_fit_says_how_many_chips_would():
    with pytest.raises(ep.PlacementError, match=r"at least 3 chips"):
        ep.place_experts(n_experts=512, chips=2, bytes_per_expert=10, budget_bytes_per_chip=2000)   # 5120 bytes need 3 chips of 2000


def test_flash_next_routed_experts_at_bfp4_fit_four_chips_and_not_two():
    per_layer_expert = 3 * 2560 * 640 * 0.5625 * 48          # all 48 layers of one expert index live on its owner
    usable = 29.9 * 2**30 * 0.85
    p = ep.place_experts(512, 4, per_layer_expert, usable)
    assert p.experts_per_chip == [128] * 4 and max(p.bytes_per_chip) < usable
    with pytest.raises(ep.PlacementError):
        ep.place_experts(512, 2, per_layer_expert, usable)


def test_load_aware_placement_spreads_the_hottest_experts_over_different_chips():
    hot = [70, 1, 80, 1, 1, 90, 1, 100]          # the hot experts are not the first four
    p = ep.place_experts(8, 4, 1, 100, load=hot)
    assert len({p.owner[i] for i in (0, 2, 5, 7)}) == 4       # the four hot experts sit on four chips
    assert max(p.load_per_chip) - min(p.load_per_chip) <= 30
    assert p.experts_per_chip == [2, 2, 2, 2]


def test_load_aware_placement_respects_the_byte_budget():
    p = ep.place_experts(6, 2, 10, 30, load=[9, 9, 9, 1, 1, 1])
    assert max(p.bytes_per_chip) <= 30 and sum(p.experts_per_chip) == 6


def test_bad_inputs_are_refused():
    for kw in ({"n_experts": 0}, {"chips": 0}, {"bytes_per_expert": 0}, {"budget_bytes_per_chip": 0}):
        args = dict(n_experts=4, chips=2, bytes_per_expert=1, budget_bytes_per_chip=10) | kw
        with pytest.raises(ep.PlacementError):
            ep.place_experts(**args)
    with pytest.raises(ep.PlacementError, match="load"):
        ep.place_experts(4, 2, 1, 10, load=[1, 2, 3])


def _moe(x, router, experts, top_k):
    """Reference: softmax router, top-k, renormalised weights, each chosen expert is a gated MLP, weighted sum."""
    probs = (x @ router).softmax(-1)
    w, idx = probs.topk(top_k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(top_k):
            g, u, d = experts[int(idx[t, k])]
            out[t] += w[t, k] * ((torch.nn.functional.silu(x[t] @ g) * (x[t] @ u)) @ d)
    return out


@pytest.mark.parametrize("chips,load_aware", [(1, False), (2, False), (4, False), (4, True), (3, True)])
def test_splitting_the_experts_over_chips_gives_the_same_output_as_one_chip(chips, load_aware):
    torch.manual_seed(0)
    n, dim, ff, top_k, tokens = 16, 12, 8, 3, 9
    router = torch.randn(dim, n)
    experts = [(torch.randn(dim, ff), torch.randn(dim, ff), torch.randn(ff, dim)) for _ in range(n)]
    x = torch.randn(tokens, dim)
    load = [float(i % 5 + 1) for i in range(n)] if load_aware else None
    p = ep.place_experts(n, chips, 1, 100, load=load)
    got = ep.sharded_moe(x, router, experts, top_k, p)
    assert torch.allclose(got, _moe(x, router, experts, top_k), atol=1e-5)


def test_each_chip_only_touches_the_experts_it_owns():
    torch.manual_seed(1)
    n, dim, ff = 8, 6, 4
    router = torch.randn(dim, n)
    touched = []

    class Spy(tuple):
        pass
    experts = [Spy((torch.randn(dim, ff), torch.randn(dim, ff), torch.randn(ff, dim))) for _ in range(n)]
    p = ep.place_experts(n, 4, 1, 10)
    ep.sharded_moe(torch.randn(5, dim), router, experts, 2, p, on_expert=lambda chip, e: touched.append((chip, e)))
    assert touched and all(p.owner[e] == chip for chip, e in touched)


def test_the_traffic_estimate_counts_one_vector_per_remote_expert_hit():
    p = ep.place_experts(8, 4, 1, 10)
    chosen = torch.tensor([[0, 2], [1, 6]])                  # token 0 hits experts on chips 0 and 1, token 1 on chips 0 and 3
    assert ep.chips_touched(chosen, p) == [{0, 1}, {0, 3}]
