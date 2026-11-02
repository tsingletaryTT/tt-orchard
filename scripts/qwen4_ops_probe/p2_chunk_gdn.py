# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 2: ttnn.transformer.chunk_gated_delta_rule against the Qwen4Exp torch_chunk_gated_delta_rule.

Defaults are the Flash-Next head counts (16 key heads, 48 value heads, head dim 128). Each input layout and dtype setting
the op takes is one entry of "configs":
  flat_cs32_bf16 / flat_cs32_fp32  rank-3 flat q/k/v (raw, the op normalises and scales), chunk 32, num_key_heads given,
                                   g and beta as bf16 or fp32
  head_cs32_fp32 / head_cs64_fp32  rank-4 head-split q/k/v (host L2-normalised), chunk 32 or 64
  lofi_refused                     a LoFi compute_kernel_config, to record whether the op refuses it (the doc says it does)
q/k/v are always cast to bf16 by the op; initial_state and the outputs are fp32.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ALL_CONFIGS = ["flat_cs32_bf16", "flat_cs32_fp32", "head_cs32_fp32", "head_cs64_fp32", "lofi_refused"]


def selftest() -> dict:
    """The chunked reference (chunk 16 and 64) must agree with the recurrent reference, final state included, with GQA."""
    q, k, v, g, beta = common.make_gdn_inputs(1, 40, 2, 4, 16, 16)
    qe, ke = common.expand_heads(q, 4), common.expand_heads(k, 4)
    rec_o, rec_s = common.torch_recurrent_gated_delta_rule(qe, ke, v, g, beta, output_final_state=True,
                                                          use_qk_l2norm_in_kernel=True)
    out = {}
    for cs in (16, 64):
        o, s = common.torch_chunk_gated_delta_rule(qe, ke, v, g, beta, chunk_size=cs, output_final_state=True,
                                                   use_qk_l2norm_in_kernel=True)
        out[f"chunk{cs}_o"] = common.error_stats(o, rec_o)["max_abs"]
        out[f"chunk{cs}_state"] = common.error_stats(s, rec_s)["max_abs"]
    out["ok"] = all(x < 1e-4 for x in out.values())
    out["probe"], out["mode"] = "p2_chunk_gdn", "selftest"
    return out


def run_chunk_op(ttnn, dev, inputs, layout, chunk_size, gb_dtype, heads_k, compute_config=None):
    """One call of the op. inputs = (q, k, v, g, beta) raw CPU tensors. Returns (o torch [B,T,HV,V], state torch
    [B,HV,K,V], state device tensor). layout is "flat" or "head"."""
    q, k, v, g, beta = inputs
    batch, seq = q.shape[:2]
    dtype = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}[gb_dtype]

    def up(t, dt):
        return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    kwargs = dict(output_final_state=True, chunk_size=chunk_size)
    if compute_config is not None:
        kwargs["compute_kernel_config"] = compute_config
    if layout == "flat":
        tq, tk, tv = (up(x.reshape(batch, seq, -1), ttnn.bfloat16) for x in (q, k, v))
        kwargs["num_key_heads"] = heads_k
    else:
        tq, tk, tv = up(common.l2norm(q), ttnn.bfloat16), up(common.l2norm(k), ttnn.bfloat16), up(v, ttnn.bfloat16)
    tg, tb = up(g, dtype), up(beta, dtype)
    o, state = ttnn.transformer.chunk_gated_delta_rule(tq, tk, tv, tg, tb, **kwargs)
    return ttnn.to_torch(o).float(), ttnn.to_torch(state).float(), state


def reference(inputs, heads_v):
    q, k, v, g, beta = inputs
    return common.torch_chunk_gated_delta_rule(common.expand_heads(q, heads_v), common.expand_heads(k, heads_v), v, g,
                                               beta, chunk_size=64, output_final_state=True,
                                               use_qk_l2norm_in_kernel=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference side; no ttnn")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--heads-k", type=int, default=16)
    ap.add_argument("--heads-v", type=int, default=48)
    ap.add_argument("--dim-k", type=int, default=128)
    ap.add_argument("--dim-v", type=int, default=128)
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS), help="comma list of: " + ", ".join(ALL_CONFIGS))
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    inputs = common.make_gdn_inputs(args.batch, args.seq, args.heads_k, args.heads_v, args.dim_k, args.dim_v)
    ref_o, ref_s = reference(inputs, args.heads_v)
    ttnn, dev = common.open_device()
    results = []
    try:
        for name in args.configs.split(","):
            entry = {"config": name}
            try:
                if name == "lofi_refused":
                    cfg = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                                 fp32_dest_acc_en=False)
                    layout, cs, gb = "flat", 32, "fp32"
                else:
                    layout, cs, gb = name.split("_")[0], int(name.split("_")[1][2:]), name.split("_")[2]
                    cfg = None
                (o, s, _), times = common.timed(lambda: run_chunk_op(ttnn, dev, inputs, layout, cs, gb, args.heads_k,
                                                                     cfg), ttnn, dev)
                entry.update(wall_s=times, o_shape=list(o.shape), o=common.error_stats(o, ref_o),
                             state=common.error_stats(s, ref_s))
            except Exception as exc:  # recorded, the next config still runs
                entry["failure"] = common.short_exc(exc)
            results.append(entry)
    finally:
        ttnn.close_device(dev)
    common.emit({"probe": "p2_chunk_gdn", "shape": {"B": args.batch, "T": args.seq, "Hk": args.heads_k,
                                                    "Hv": args.heads_v, "K": args.dim_k, "V": args.dim_v},
                 "reference": "torch_chunk_gated_delta_rule fp32 chunk 64", "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
