# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which chip owns which routed expert, and a CPU model of the split computation (issue 723, item 3).

A mixture-of-experts layer with hundreds of experts does not fit one chip. Expert parallelism gives each chip a subset of the experts of
every layer. A token's router picks `top_k` experts; each chip computes the picked experts it owns for that token, and the partial outputs
are summed across the chips (an all-reduce, or a gather to one chip).

This module holds the parts that need no chip:

  place_experts   the owner of each expert, from the expert count, the chip count, the bytes one expert takes on its chip (all layers of
                  that expert index together) and the per-chip budget. Without load figures the experts are cut into contiguous blocks.
                  With load figures (how often each expert was picked on sample text) the hottest experts go to different chips, so
                  one chip does not serve most of the picks. Every chip holds the same number of experts, give or take one, so the
                  memory is even.
  sharded_moe     the computation as the chips would do it, in torch on the CPU. A test compares it with the unsplit computation; the
                  chip emitter has to produce the same numbers.
  chips_touched   which chips each token needs, which sizes the cross-chip traffic.

Nothing here measures speed. The generator does not write the multi-chip block yet; `orchard/plan.py` prices the split.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class PlacementError(ValueError):
    """The experts cannot be placed as asked; the message says what would work."""


@dataclass
class Placement:
    owner: list[int]                                   # owner[e] is the chip that holds expert e
    chips: int
    bytes_per_expert: float
    experts_per_chip: list[int] = field(default_factory=list)
    bytes_per_chip: list[float] = field(default_factory=list)
    load_per_chip: list[float] = field(default_factory=list)

    def experts_of(self, chip: int) -> list[int]:
        return [e for e, c in enumerate(self.owner) if c == chip]


def _counts(n_experts: int, chips: int) -> list[int]:
    return [n_experts // chips + (1 if i < n_experts % chips else 0) for i in range(chips)]


def place_experts(n_experts: int, chips: int, bytes_per_expert: float, budget_bytes_per_chip: float, load: list[float] | None = None) -> Placement:
    for name, v in (("n_experts", n_experts), ("chips", chips), ("bytes_per_expert", bytes_per_expert), ("budget_bytes_per_chip", budget_bytes_per_chip)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
            raise PlacementError(f"{name} must be a positive number, got {v!r}")
    n_experts, chips = int(n_experts), int(chips)
    if load is not None and len(load) != n_experts:
        raise PlacementError(f"load has {len(load)} entries for {n_experts} experts")
    fit = int(budget_bytes_per_chip // bytes_per_expert)               # experts one chip can hold
    if fit < 1:
        raise PlacementError(f"one expert takes {bytes_per_expert:.0f} bytes and a chip's budget is {budget_bytes_per_chip:.0f}")
    counts = _counts(n_experts, chips)
    if max(counts) > fit:
        need = -(-n_experts // fit)
        raise PlacementError(f"{n_experts} experts of {bytes_per_expert:.0f} bytes do not fit {chips} chips of {budget_bytes_per_chip:.0f} bytes; "
                             f"at least {need} chips are needed, or a smaller format")
    owner = [0] * n_experts
    if load is None:
        e = 0
        for chip, c in enumerate(counts):
            for _ in range(c):
                owner[e] = chip
                e += 1
    else:
        room = list(counts)
        total = [0.0] * chips
        for e in sorted(range(n_experts), key=lambda i: -load[i]):       # hottest first, onto the least loaded chip that has room
            chip = min((c for c in range(chips) if room[c] > 0), key=lambda c: (total[c], c))
            owner[e] = chip
            room[chip] -= 1
            total[chip] += load[e]
    per = [sum(1 for c in owner if c == i) for i in range(chips)]
    loads = [sum((load[e] if load else 1.0) for e in range(n_experts) if owner[e] == i) for i in range(chips)]
    return Placement(owner=owner, chips=chips, bytes_per_expert=bytes_per_expert, experts_per_chip=per,
                     bytes_per_chip=[c * bytes_per_expert for c in per], load_per_chip=loads)


def chips_touched(chosen, placement: Placement) -> list[set[int]]:
    """For each token, the set of chips that own at least one of its chosen experts. `chosen` is a [tokens, top_k] integer tensor."""
    return [{placement.owner[int(e)] for e in row} for row in chosen]


def sharded_moe(x, router, experts, top_k: int, placement: Placement, on_expert=None):
    """The MoE layer computed chip by chip. x [tokens, dim]; router [dim, n_experts]; experts[e] = (gate, up, down) with gate and up [dim, ff] and
    down [ff, dim]. Softmax router, top_k picks with weights renormalised over the picks, silu-gated expert MLP, weighted sum.
    `on_expert(chip, expert)` is called for each expert a chip runs (a test uses it to check that no chip touches another chip's expert)."""
    import torch
    import torch.nn.functional as F

    probs = (x @ router).softmax(-1)
    w, idx = probs.topk(top_k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    total = torch.zeros_like(x)
    for chip in range(placement.chips):
        partial = torch.zeros_like(x)
        for e in placement.experts_of(chip):
            hit = (idx == e)
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            if on_expert is not None:
                on_expert(chip, e)
            g, u, d = experts[e]
            xe = x[rows]
            out = (F.silu(xe @ g) * (xe @ u)) @ d
            partial[rows] += (w[rows] * hit[rows]).sum(-1, keepdim=True) * out
        total += partial                                            # the cross-chip reduction
    return total
