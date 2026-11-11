# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 14 (CPU only): the memory and the per-token reads of the routed experts of Flash-Next, so that the plan for the real model rests on arithmetic (issue 723).

Dimensions (the official config of Qwen/Qwen3.8-Flash-Next, model type qwen4_exp): 48 layers, hidden 2560, 512 routed experts with expert width 640 and k = 10 active
per token, a shared expert of width 640, a router of [512, 2560]. Every layer is a sparse MoE block.

Bytes per element come from the tile sizes of the formats: a 32 x 32 tile is 2048 B in bfloat16, 1088 B in bfloat8_b (1024 mantissa bytes and 64 shared-exponent bytes)
and 576 B in bfloat4_b (512 and 64). Chip memory: the nominal capacity of a Blackhole chip is 32 GB and the ACE-Step bundle card in this repository reports 29.9 GB usable;
the repository records 512 GB/s of bandwidth per die (docs/run-logs/2026-10-28-decode-floor-break-even-hazards.md). The numbers below are for the routed experts only. The
attention, the shared experts, the embeddings, the KV cache and the 104 GB host-resident PLE table of the real model are not counted: `expert_share` is the share of a
chip's usable memory that the plan lets the routed experts take (0.6, an assumption).

`--selftest` checks the arithmetic; without it the probe prints the table. There is no device mode: nothing here opens a device.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "bfp4": 576}
BYTES_PER_ELEM = {k: v / 1024 for k, v in TILE_BYTES.items()}
FLASH_NEXT = dict(layers=48, hidden=2560, inter=640, shared=640, experts=512, top_k=10)
CHIP_USABLE_GB = 29.9
CHIP_NOMINAL_GB = 32.0
DIE_BANDWIDTH_GBS = 512.0
EXPERT_SHARE = 0.6


def routed_layer_bytes(dims: dict, dtype: str) -> float:
    """Bytes of the routed experts of one layer: experts x (gate, up, down) x hidden x inter."""
    return dims["experts"] * 3 * dims["hidden"] * dims["inter"] * BYTES_PER_ELEM[dtype]


def routed_total_bytes(dims: dict, dtype: str) -> float:
    return dims["layers"] * routed_layer_bytes(dims, dtype)


def shared_layer_bytes(dims: dict, dtype: str) -> float:
    return 3 * dims["hidden"] * dims["shared"] * BYTES_PER_ELEM[dtype] + dims["hidden"] * BYTES_PER_ELEM[dtype]


def router_layer_bytes(dims: dict) -> float:
    return dims["experts"] * dims["hidden"] * BYTES_PER_ELEM["bf16"]


def active_read_bytes_per_token(dims: dict, dtype: str) -> float:
    """Bytes of routed-expert weights read to decode one token: layers x k x (gate, up, down)."""
    return dims["layers"] * dims["top_k"] * 3 * dims["hidden"] * dims["inter"] * BYTES_PER_ELEM[dtype]


def chips_needed(total_bytes: float, share: float = 1.0, usable_gb: float = CHIP_USABLE_GB) -> int:
    return max(1, math.ceil(total_bytes / (usable_gb * 1e9 * share)))


def table(dims: dict = FLASH_NEXT) -> dict:
    out = {}
    for dtype in TILE_BYTES:
        total = routed_total_bytes(dims, dtype)
        read = active_read_bytes_per_token(dims, dtype)
        out[dtype] = {
            "bytes_per_element": round(BYTES_PER_ELEM[dtype], 4),
            "per_layer_gb": round(routed_layer_bytes(dims, dtype) / 1e9, 3), "all_layers_gb": round(total / 1e9, 1),
            "min_chips_all_memory": chips_needed(total), f"min_chips_at_{int(EXPERT_SHARE * 100)}pct": chips_needed(total, EXPERT_SHARE),
            "per_chip_gb": {str(n): round(total / n / 1e9, 1) for n in (1, 2, 4)},
            "fits_at_share": {str(n): total / n <= CHIP_USABLE_GB * 1e9 * EXPERT_SHARE for n in (1, 2, 4)},
            "decode_read_gb_per_token": round(read / 1e9, 3),
            "decode_floor_ms_per_token": {str(n): round(read / (DIE_BANDWIDTH_GBS * 1e9 * n) * 1e3, 2) for n in (1, 2, 4)},
        }
    out["shared_expert_and_router_per_layer_mb_bf16"] = round((shared_layer_bytes(dims, "bf16") + router_layer_bytes(dims)) / 1e6, 1)
    return out


def selftest() -> dict:
    checks = {}
    d = FLASH_NEXT
    checks["one_expert_layer_in_bf16_is_5_033_gb"] = abs(routed_layer_bytes(d, "bf16") - 512 * 3 * 2560 * 640 * 2) < 1 and abs(routed_layer_bytes(d, "bf16") / 1e9 - 5.033) < 0.001
    checks["tile_sizes_give_the_bytes_per_element"] = (BYTES_PER_ELEM["bf16"] == 2.0 and abs(BYTES_PER_ELEM["bfp8"] - 1.0625) < 1e-9 and abs(BYTES_PER_ELEM["bfp4"] - 0.5625) < 1e-9)
    checks["lower_precision_is_smaller"] = routed_total_bytes(d, "bfp4") < routed_total_bytes(d, "bfp8") < routed_total_bytes(d, "bf16")
    checks["all_layers_is_48_times_a_layer"] = abs(routed_total_bytes(d, "bf16") - 48 * routed_layer_bytes(d, "bf16")) < 1
    t = table()
    checks["bfp4_routed_experts_need_a_memory_share_of_four_chips"] = t["bfp4"][f"min_chips_at_{int(EXPERT_SHARE * 100)}pct"] == 4 and t["bfp4"]["min_chips_all_memory"] == 3
    checks["bf16_routed_experts_do_not_fit_four_chips"] = t["bf16"]["fits_at_share"]["4"] is False and t["bf16"]["min_chips_all_memory"] > 4
    checks["per_chip_memory_falls_as_chips_rise"] = t["bfp4"]["per_chip_gb"]["1"] > t["bfp4"]["per_chip_gb"]["2"] > t["bfp4"]["per_chip_gb"]["4"]
    checks["decode_floor_falls_as_dies_rise_and_with_precision"] = (t["bfp4"]["decode_floor_ms_per_token"]["1"] > t["bfp4"]["decode_floor_ms_per_token"]["4"]
                                                                  and t["bf16"]["decode_floor_ms_per_token"]["1"] > t["bfp4"]["decode_floor_ms_per_token"]["1"])
    checks["active_reads_are_k_over_experts_of_the_weights"] = abs(active_read_bytes_per_token(d, "bf16") * d["experts"] / d["top_k"] - routed_total_bytes(d, "bf16")) < 1
    checks["chips_needed_rounds_up"] = chips_needed(29.9e9 + 1) == 2 and chips_needed(1) == 1
    return {"probe": "p14_moe_memory", "mode": "selftest", "checks": checks, "flash_next": t, "ok": all(checks.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="check the arithmetic")
    ap.add_argument("--experts", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    dims = dict(FLASH_NEXT)
    for key in ("experts", "top_k", "layers"):
        if getattr(args, key) is not None:
            dims[key] = getattr(args, key)
    common.emit({"probe": "p14_moe_memory", "mode": "estimate", "ok": True, "dims": dims, "table": table(dims), "chip_usable_gb": CHIP_USABLE_GB,
                 "die_bandwidth_gbs": DIE_BANDWIDTH_GBS, "expert_share": EXPERT_SHARE, "workarounds": []})
    return 0


if __name__ == "__main__":
    sys.exit(main())
