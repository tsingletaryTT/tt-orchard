# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 15: the routed experts of a MoE layer split over the two chips of one board, with the partial outputs added by an all-reduce (issue 723, item 3).

Each chip owns half of the experts (`orchard.expert_placement.place_experts`). For every expert slot j, chip c holds expert `owner_experts[c][j]`;
the tokens' routing weight for that expert is a [T, 1] column on that chip. Each chip runs the dense-masked form of probe 11 on its own experts
(every local expert on every token, the output times the routing column) and sums the results into a partial [T, H]. `ttnn.all_reduce` over the mesh
adds the two partials. The sum is the same as the unsplit layer by linearity, so the probe compares the device result with `common.moe_experts_ref`.

It reports, per preset: the error against the reference, the wall time of the whole split layer (second call, after compilation), and the wall time of the
all-reduce alone on a [T, H] tensor. The `flash` preset uses Flash-Next's widths (hidden 2560, expert width 640) with 16 of the 512 experts and top-4.

Run on chips only inside a gozer lease of one board (two chips). `--selftest` checks the CPU side (the partial sums add up to the reference) and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common  # noqa: E402
from orchard import expert_placement as ep  # noqa: E402

PRESETS = {"tiny": dict(tokens=32, hidden=64, experts=8, top_k=2, inter=64, shared=64),
           "flash": dict(tokens=32, hidden=2560, experts=16, top_k=4, inter=640, shared=640),
           "flash_prefill": dict(tokens=256, hidden=2560, experts=16, top_k=4, inter=640, shared=640)}
CHIPS = 2


def routing_matrix(indices: torch.Tensor, weights: torch.Tensor, experts: int) -> torch.Tensor:
    """Dense [T, E] routing weights: the token's weight for each expert it picked, 0 elsewhere."""
    r = torch.zeros(indices.shape[0], experts)
    r.scatter_(1, indices, weights.float())
    return r


def chip_partials(t: dict, route: torch.Tensor, placement: ep.Placement) -> torch.Tensor:
    """The partial output of each chip, [chips, T, H], by the dense-masked form on the experts that chip owns."""
    out = torch.zeros(placement.chips, t["hidden"].shape[0], t["down"].shape[1])
    for c in range(placement.chips):
        for e in placement.experts_of(c):
            x = t["hidden"].float()
            y = (torch.nn.functional.silu(x @ t["gate"][e].float().t()) * (x @ t["up"][e].float().t())) @ t["down"][e].float().t()
            out[c] += y * route[:, e:e + 1]
    return out


def selftest() -> dict:
    t = common.moe_inputs(**{k: v for k, v in PRESETS["tiny"].items() if k != "tokens"}, tokens=PRESETS["tiny"]["tokens"])
    _, weights, indices = common.moe_router_ref(t["hidden"], t["router"], t["top_k"])
    ref = common.moe_experts_ref(t["hidden"], indices, weights, t["gate"], t["up"], t["down"])
    placement = ep.place_experts(t["gate"].shape[0], CHIPS, 1, 100)
    parts = chip_partials(t, routing_matrix(indices, weights, t["gate"].shape[0]), placement)
    stats = common.error_stats(parts.sum(0), ref)
    return {"probe": "p15_moe_expert_parallel", "mode": "selftest", "ok": bool(stats["max_abs"] < 1e-4), "stats": stats}


def run_device(ttnn, mesh, name: str, cfg: dict, repeat: int) -> dict:
    t = common.moe_inputs(**cfg)
    n_exp = t["gate"].shape[0]
    _, weights, indices = common.moe_router_ref(t["hidden"], t["router"], t["top_k"])
    ref = common.moe_experts_ref(t["hidden"], indices, weights, t["gate"], t["up"], t["down"])
    route = routing_matrix(indices, weights, n_exp)
    placement = ep.place_experts(n_exp, CHIPS, 1, 10 ** 9)
    slots = [placement.experts_of(c) for c in range(CHIPS)]
    per = len(slots[0])
    assert all(len(s) == per for s in slots), "this probe wants an even split"
    shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)

    def up_tensor(stack):                                    # [chips, 1, a, b] with chip c's slice on chip c
        return ttnn.from_torch(stack.bfloat16(), device=mesh, mesh_mapper=shard0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    x_dev = ttnn.from_torch(t["hidden"].bfloat16()[None, None], device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    gate = [up_tensor(torch.stack([t["gate"][slots[c][j]].t()[None] for c in range(CHIPS)])) for j in range(per)]    # [chips, 1, H, I]
    upw = [up_tensor(torch.stack([t["up"][slots[c][j]].t()[None] for c in range(CHIPS)])) for j in range(per)]
    down = [up_tensor(torch.stack([t["down"][slots[c][j]].t()[None] for c in range(CHIPS)])) for j in range(per)]     # [chips, 1, I, H]
    rcol = [up_tensor(torch.stack([route[:, slots[c][j]][None, :, None] for c in range(CHIPS)])) for j in range(per)]  # [chips, 1, T, 1]

    def layer():
        acc = None
        for j in range(per):
            h = ttnn.multiply(ttnn.silu(ttnn.matmul(x_dev, gate[j])), ttnn.matmul(x_dev, upw[j]))
            y = ttnn.multiply(ttnn.matmul(h, down[j]), rcol[j])
            acc = y if acc is None else ttnn.add(acc, y)
        return ttnn.all_reduce(acc, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)

    out, times = common.timed(layer, ttnn, mesh, repeat=repeat)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    res = {"preset": name, "experts": n_exp, "experts_per_chip": per, "tokens": cfg["tokens"], "layer_seconds": times,
           "stats_chip0": common.error_stats(got[0].reshape(cfg["tokens"], -1), ref),
           "chips_agree": bool(torch.equal(got[0], got[1])) if got.shape[0] > 1 else None}
    part = ttnn.from_torch(torch.randn(CHIPS, 1, cfg["tokens"], cfg["hidden"]).bfloat16(), device=mesh, mesh_mapper=shard0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    _, ar = common.timed(lambda: ttnn.all_reduce(part, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear), ttnn, mesh, repeat=repeat)
    res["all_reduce_seconds"] = ar
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--presets", nargs="+", default=["tiny", "flash", "flash_prefill"], choices=sorted(PRESETS))
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args(argv)
    if args.selftest:
        r = selftest()
        common.emit(r)
        return 0 if r["ok"] else 1
    import ttnn
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, CHIPS))
    results = []
    try:
        for name in args.presets:
            try:
                results.append(run_device(ttnn, mesh, name, PRESETS[name], args.repeat))
            except Exception as exc:  # noqa: BLE001
                results.append({"preset": name, "error": common.short_exc(exc)})
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    common.emit({"probe": "p15_moe_expert_parallel", "chips": CHIPS, "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
