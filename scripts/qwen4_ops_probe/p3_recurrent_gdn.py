# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 3: ttnn.transformer.fused_recurrent_gated_delta_rule for one decode step against torch_recurrent_gated_delta_rule.

The state comes from a prefill of --seq tokens (probe 2's op, flat chunk-32 form), then one new token is decoded:
  state_ttnn   the device fp32 state the chunk op returned is fed straight in (the carry that matters), compared with
               the torch recurrent step started from the torch chunk reference's state
  state_torch  the torch reference state is uploaded as fp32 and fed in (isolates the recurrent op from the chunk op)
Each runs with g and beta as fp32 and as bf16. q/k are L2-normalised on the host as the op requires. The op is first
tried with H key heads and then, if that fails, with q/k expanded to the value head count on the host.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import p2_chunk_gdn  # noqa: E402


def selftest() -> dict:
    """A one-token recurrent step from a chunked state must equal the last position of a T+1 recurrent run."""
    q, k, v, g, beta = common.make_gdn_inputs(1, 33, 2, 4, 16, 16)
    qe, ke = common.expand_heads(q, 4), common.expand_heads(k, 4)
    full_o, full_s = common.torch_recurrent_gated_delta_rule(qe, ke, v, g, beta, output_final_state=True,
                                                            use_qk_l2norm_in_kernel=True)
    _, pre_s = common.torch_chunk_gated_delta_rule(qe[:, :32], ke[:, :32], v[:, :32], g[:, :32], beta[:, :32],
                                                   chunk_size=16, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o, s = common.torch_recurrent_gated_delta_rule(qe[:, 32:], ke[:, 32:], v[:, 32:], g[:, 32:], beta[:, 32:],
                                                  initial_state=pre_s, output_final_state=True,
                                                  use_qk_l2norm_in_kernel=True)
    out = {"step_o": common.error_stats(o, full_o[:, 32:])["max_abs"], "step_state": common.error_stats(s, full_s)["max_abs"]}
    out["ok"] = all(x < 1e-4 for x in out.values())
    out["probe"], out["mode"] = "p3_recurrent_gdn", "selftest"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference side; no ttnn")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128, help="prefill length that produces the carried state")
    ap.add_argument("--heads-k", type=int, default=16)
    ap.add_argument("--heads-v", type=int, default=48)
    ap.add_argument("--dim-k", type=int, default=128)
    ap.add_argument("--dim-v", type=int, default=128)
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    total = args.seq + 1
    allq, allk, allv, allg, allb = common.make_gdn_inputs(args.batch, total, args.heads_k, args.heads_v, args.dim_k,
                                                          args.dim_v, seed=1)
    pre = tuple(x[:, :args.seq] for x in (allq, allk, allv, allg, allb))
    step = tuple(x[:, args.seq:] for x in (allq, allk, allv, allg, allb))
    _, ref_state0 = p2_chunk_gdn.reference(pre, args.heads_v)
    sq, sk, sv, sg, sb = step
    ref_o, ref_s = common.torch_recurrent_gated_delta_rule(
        common.expand_heads(sq, args.heads_v), common.expand_heads(sk, args.heads_v), sv, sg, sb,
        initial_state=ref_state0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    ttnn, dev = common.open_device()
    results = []
    try:
        _, _, tt_state0 = p2_chunk_gdn.run_chunk_op(ttnn, dev, pre, "flat", 32, "fp32", args.heads_k)
        torch_state_dev = ttnn.from_torch(ref_state0.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                          device=dev)

        def up(t, dt):
            return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

        for state_name, state in (("state_ttnn", tt_state0), ("state_torch", torch_state_dev)):
            for gb in ("fp32", "bf16"):
                for expand in (False, True):
                    entry = {"config": f"{state_name}_gb{gb}_{'expanded' if expand else 'gqa'}"}
                    try:
                        dt = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}[gb]
                        qn, kn = common.l2norm(sq), common.l2norm(sk)
                        if expand:
                            qn, kn = common.expand_heads(qn, args.heads_v), common.expand_heads(kn, args.heads_v)

                        def call():
                            return ttnn.transformer.fused_recurrent_gated_delta_rule(
                                up(qn, ttnn.bfloat16), up(kn, ttnn.bfloat16), up(sv, ttnn.bfloat16), up(sg, dt),
                                up(sb, dt), initial_state=state, output_final_state=True)

                        (o, s), times = common.timed(call, ttnn, dev)
                        entry.update(wall_s=times, o_shape=list(ttnn.to_torch(o).shape),
                                     o=common.error_stats(ttnn.to_torch(o).float(), ref_o),
                                     state=common.error_stats(ttnn.to_torch(s).float(), ref_s))
                    except Exception as exc:
                        entry["failure"] = common.short_exc(exc)
                    results.append(entry)
    finally:
        ttnn.close_device(dev)
    common.emit({"probe": "p3_recurrent_gdn", "shape": {"B": args.batch, "prefill_T": args.seq, "decode_T": 1,
                                                        "Hk": args.heads_k, "Hv": args.heads_v, "K": args.dim_k,
                                                        "V": args.dim_v},
                 "reference": "torch_recurrent_gated_delta_rule fp32, state from torch_chunk_gated_delta_rule",
                 "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
