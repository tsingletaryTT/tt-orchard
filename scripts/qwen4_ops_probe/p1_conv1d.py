# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 1: a depthwise (groups = channels) dilated causal convolution, the PLE short conv of Qwen4Exp.

Defaults are Flash-Next: 4 * 2560 = 10240 channels, kernel 4, dilation 3, causal (left pad 9), then silu.
Two implementations are compared with the torch reference (`short_conv_ref`):
  conv1d    ttnn.conv1d(groups=channels, dilation=3, padding=(9, 0)), and if that fails, host-padded input with padding=0
  shifted   the fallback: four ttnn slices of the left-padded input, each multiplied by its weight row (a broadcast),
            summed, then ttnn.silu
Errors are for the convolution output before silu where a silu-free output exists (conv1d), and after silu otherwise;
the "compared" field of each entry says which.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def make_case(seq, channels, kernel, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, seq, channels, generator=gen).bfloat16().float()
    w = (torch.randn(channels, 1, kernel, generator=gen) * 0.5).bfloat16().float()
    return x, w


def selftest() -> dict:
    """The shifted-multiply-add form must equal the torch dilated depthwise convolution, and the dilation must matter."""
    x, w = make_case(24, 16, 4)
    ref = common.short_conv_ref(x, w, 4, 3)
    out = {"shifted_vs_ref": common.error_stats(common.short_conv_shifted(x, w, 4, 3), ref)["max_abs"],
           "dilation1_vs_ref": common.error_stats(common.short_conv_ref(x, w, 4, 1), ref)["max_abs"]}
    out["ok"] = out["shifted_vs_ref"] < 1e-5 and out["dilation1_vs_ref"] > 1e-2
    out["probe"], out["mode"] = "p1_conv1d", "selftest"
    return out


def run_conv1d(ttnn, dev, x, w, kernel, dilation, host_pad, groups=None):
    """ttnn.conv1d depthwise. Returns the output as torch [1, T, C] (before silu)."""
    seq, channels = x.shape[1], x.shape[2]
    groups = channels if groups is None else groups
    state_len = (kernel - 1) * dilation
    inp = F.pad(x, (0, 0, state_len, 0)) if host_pad else x
    tt_in = ttnn.from_torch(inp, dtype=ttnn.bfloat16)
    tt_w = ttnn.from_torch(w, dtype=ttnn.bfloat16)
    out, length = ttnn.conv1d(
        input_tensor=tt_in, weight_tensor=tt_w, device=dev, in_channels=channels, out_channels=channels,
        batch_size=1, input_length=inp.shape[1], kernel_size=kernel, stride=1,
        padding=0 if host_pad else (state_len, 0), dilation=dilation, groups=groups,
        conv_config=ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16), return_output_dim=True)
    res = ttnn.to_torch(ttnn.from_device(out)).float().reshape(1, -1, channels)[:, :length]
    return res


def run_shifted(ttnn, dev, x, w, kernel, dilation):
    """The fallback on ttnn ops. Returns the output as torch [1, T, C] after silu."""
    seq, channels = x.shape[1], x.shape[2]
    state_len = (kernel - 1) * dilation
    xpad = ttnn.from_torch(F.pad(x, (0, 0, state_len, 0)).reshape(1, 1, seq + state_len, channels),
                           dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    acc = None
    for j in range(kernel):
        piece = ttnn.slice(xpad, [0, 0, j * dilation, 0], [1, 1, j * dilation + seq, channels])
        piece = ttnn.to_layout(piece, ttnn.TILE_LAYOUT)
        wj = ttnn.from_torch(w[:, 0, j].reshape(1, 1, 1, channels), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev)
        term = ttnn.multiply(piece, wj)
        acc = term if acc is None else ttnn.add(acc, term)
    out = ttnn.silu(acc)
    return ttnn.to_torch(out).float().reshape(1, seq, channels)


def diagnose(ttnn, dev, seq, channels, kernel):
    """Which setting makes ttnn.conv1d wrong? Runs (dilation, groups) in {1, 3} x {1, channels} against torch."""
    out = []
    for dil in (1, 3):
        for groups in (1, channels):
            gen = torch.Generator().manual_seed(2)
            x = torch.randn(1, seq, channels, generator=gen).bfloat16().float()
            w = (torch.randn(channels, channels // groups, kernel, generator=gen) * 0.3).bfloat16().float()
            ref = F.conv1d(F.pad(x.transpose(1, 2), ((kernel - 1) * dil, 0)), w, groups=groups,
                           dilation=dil).transpose(1, 2)
            entry = {"dilation": dil, "groups": groups}
            try:
                got = run_conv1d(ttnn, dev, x, w, kernel, dil, False, groups)
                entry.update(common.error_stats(got, ref))
            except Exception as exc:
                entry["failure"] = common.short_exc(exc)
            out.append(entry)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference side; no ttnn")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--channels", type=int, default=10240)
    ap.add_argument("--kernel", type=int, default=4)
    ap.add_argument("--dilation", type=int, default=3)
    ap.add_argument("--diagnose", action="store_true", help="also run the (dilation, groups) matrix")
    ap.add_argument("--l1-small", type=int, default=32768, help="L1_SMALL bytes reserved at device open (conv1d needs it)")
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    x, w = make_case(args.seq, args.channels, args.kernel)
    ref_full = common.short_conv_ref(x, w, args.kernel, args.dilation)
    ref_conv = F.conv1d(F.pad(x.transpose(1, 2), ((args.kernel - 1) * args.dilation, 0)), w, groups=args.channels,
                        dilation=args.dilation).transpose(1, 2)
    ttnn, dev = common.open_device(args.l1_small)
    results = []
    try:
        for host_pad in (False, True):
            entry = {"config": "conv1d_" + ("host_pad" if host_pad else "op_pad"), "compared": "conv before silu"}
            try:
                got, times = common.timed(lambda: run_conv1d(ttnn, dev, x, w, args.kernel, args.dilation, host_pad),
                                          ttnn, dev)
                entry.update(wall_s=times, out_shape=list(got.shape), **common.error_stats(got, ref_conv))
                entry["silu_on_host_error"] = common.error_stats(F.silu(got), ref_full)
            except Exception as exc:
                entry["failure"] = common.short_exc(exc)
            results.append(entry)
        if args.diagnose:
            results.append({"config": "diagnose_matrix", "matrix": diagnose(ttnn, dev, args.seq, args.channels,
                                                                           args.kernel)})
        entry = {"config": "shifted_ttnn", "compared": "conv after silu"}
        try:
            got, times = common.timed(lambda: run_shifted(ttnn, dev, x, w, args.kernel, args.dilation), ttnn, dev)
            entry.update(wall_s=times, **common.error_stats(got, ref_full))
        except Exception as exc:
            entry["failure"] = common.short_exc(exc)
        results.append(entry)
    finally:
        ttnn.close_device(dev)
    common.emit({"probe": "p1_conv1d", "shape": {"T": args.seq, "C": args.channels, "kernel": args.kernel,
                                                 "dilation": args.dilation, "groups": args.channels,
                                                 "l1_small_size": args.l1_small},
                 "dtype": "bf16 input and weights", "reference": "torch F.conv1d depthwise dilated, fp32",
                 "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
