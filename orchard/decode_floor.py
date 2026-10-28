# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The DRAM-bandwidth floor on decode time per token (issue 181).

Decode at a batch of one reads every active weight once per token, so time per token cannot be below

    floor_ms = (active weight bytes per token) / (dies x per-die DRAM bandwidth)

`decode_floor` counts the active bytes from a model config and a dtype policy and returns the floor in milliseconds and tokens per
second. `share_of_floor(measured_tok_s, floor)` says how much of that a measured speed reaches. The result is an ESTIMATE and carries
its inputs (`as_dict()`), so a reader can redo it. What the estimate leaves out is listed in `ASSUMPTIONS`.

Bytes per token. Counted per layer from the config (shapes in `_attention_params`, `_linear_params`, `_mlp_params`):
  * attention (full, sliding-window, gated delta net "linear attention", multi-latent attention),
  * the dense MLP, or for a mixture-of-experts layer the router, the top-k routed experts and the shared expert,
  * the LM head (read in full for every token, even when it is tied to the embedding table; the embedding lookup reads one row and is
    left out).
Norm weights, biases, the shared-expert gate, rotary tables, the KV cache and recurrent state are not counted.

Bandwidth. `per_die_bandwidth_gb_s` is a required input with no default. `BOARDS` holds the figure for each board type with the
document it comes from, for callers who want it, and `floor_for_board` uses it. The figures are the datasheet peak. A kernel that
streams weights reaches less (the tt-metal tech report on saturating DRAM bandwidth reports over 92% on Wormhole and Grayskull; there
is no equivalent figure on record for Blackhole), so a share of 100% is not reachable and a share above 100% means an input is wrong or
the speed comes from something other than one token per weight read (speculative decoding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from orchard import model_config as mc

# ---- bytes per weight ---------------------------------------------------------------------------------
# Tenstorrent block formats store 16 values per block with one shared 8-bit exponent: bfp8 is 8 bits of sign and mantissa per value
# plus 8/16 of a bit, 8.5 bits or 1.0625 bytes; bfp4 is 4.5 bits or 0.5625 bytes; bfp2 is 2.5 bits or 0.3125 bytes.
DTYPE_BYTES: dict[str, float] = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "bfp8": 1.0625, "bfp4": 0.5625, "bfp2": 0.3125}

# The groups a dtype policy can name. `default` fills any group the policy does not name.
GROUPS = ("attention", "dense_mlp", "router", "routed_experts", "shared_experts", "lm_head")

ASSUMPTIONS = ("batch of one, one token per pass (no speculative decoding); weights spread evenly over the dies so their bandwidth adds; "
               "datasheet peak bandwidth; the KV cache, recurrent state, norms, biases, rotary tables and the embedding lookup are "
               "not counted; the dtype policy is whatever the caller gives and is not read from the bundle.")

# Where the peak figures come from. The public Tenstorrent docs site (repository tenstorrent/tenstorrent.github.io), page
# core/aibs/blackhole/index.md, table "Memory Bandwidth" row: p100a 448 GB/sec, p150a and p150b 512 GB/sec; and
# core/aibs/blackhole/p300.md: the p300c card has two ASICs and 1024 GB/sec in all. The same page states 512 GB/s per ASIC as
# 8 channels of 32 bits at 16 Gb/s (a 256-bit bus at 16 GT/s is 512 GB/s).
_DOCS = "tenstorrent.github.io core/aibs/blackhole/index.md and p300.md, Memory Bandwidth row (datasheet peak, GB/s)"


@dataclass(frozen=True)
class Board:
    dies: int                 # Blackhole ASICs on the board
    per_die_gb_s: float       # peak DRAM bandwidth of one ASIC
    source: str


BOARDS: dict[str, Board] = {
    "p100": Board(1, 448.0, f"{_DOCS}: p100a, 28 GB GDDR6, 448 GB/s"),
    "p150": Board(1, 512.0, f"{_DOCS}: p150a and p150b, 32 GB GDDR6, 512 GB/s"),
    "p300": Board(2, 512.0, f"{_DOCS}: p300c, two ASICs, 1024 GB/s in all, so 512 GB/s per die"),
}


class FloorInputError(ValueError):
    """An input to the floor is missing, is not a number, is out of range, or describes a model the counting does not cover."""


def _check_dies(dies) -> int:
    if isinstance(dies, bool) or not isinstance(dies, int) or dies < 1:
        raise FloorInputError("dies must be a whole number of at least 1")
    return dies


def _check_bandwidth(bw) -> float:
    if isinstance(bw, bool) or not isinstance(bw, (int, float)) or not math.isfinite(bw) or bw <= 0:
        raise FloorInputError("per-die bandwidth must be a positive finite number of GB/s")
    return float(bw)


def _check_policy(policy) -> dict:
    """A dtype policy: a dict of group name to dtype name, with `default` or every group named."""
    if not isinstance(policy, dict) or not policy:
        raise FloorInputError("dtype policy must be a non-empty dict of group to dtype name")
    for k, v in policy.items():
        if k != "default" and k not in GROUPS:
            raise FloorInputError(f"dtype policy names an unknown group {k!r} (groups: default, {', '.join(GROUPS)})")
        if v not in DTYPE_BYTES:
            raise FloorInputError(f"dtype policy gives {k!r} an unknown dtype {v!r} (known: {', '.join(DTYPE_BYTES)})")
    missing = [g for g in GROUPS if g not in policy]
    if missing and "default" not in policy:
        raise FloorInputError(f"dtype policy has no default and does not name {', '.join(missing)}")
    return dict(policy)


# ---- parameter counts per layer -----------------------------------------------------------------------

def _attention_params(d: mc.ModelDims) -> int:
    """Grouped-query attention: q (twice as wide when the layer has an output gate), k, v and the output projection."""
    q = d.hidden * d.heads * d.head_dim * (2 if d.attn_output_gate else 1)
    kv = 2 * d.hidden * d.kv_heads * d.head_dim
    return q + kv + d.heads * d.head_dim * d.hidden


def _mla_params(d: mc.ModelDims) -> int:
    """Multi-latent attention: a low-rank query (or a full one), the compressed key-value down projection, its up projection, output."""
    m = d.mla
    qk = m["qk_nope_head_dim"] + m["qk_rope_head_dim"]
    q = d.hidden * m["q_lora_rank"] + m["q_lora_rank"] * d.heads * qk if m["q_lora_rank"] else d.hidden * d.heads * qk
    kv_a = d.hidden * (m["kv_lora_rank"] + m["qk_rope_head_dim"])
    kv_b = m["kv_lora_rank"] * d.heads * (m["qk_nope_head_dim"] + m["v_head_dim"])
    return q + kv_a + kv_b + d.heads * m["v_head_dim"] * d.hidden


def _linear_params(d: mc.ModelDims) -> int:
    """Gated delta net (the Qwen3-Next and Qwen3.5 linear attention): q, k, v and the z gate in, the two per-head scalars, the short
    convolution over q, k and v, and the output projection."""
    lin = d.linear
    qkv_width = 2 * lin["key_dim"] + lin["value_dim"]
    return (d.hidden * qkv_width + d.hidden * lin["value_dim"] + d.hidden * 2 * lin["value_heads"]
            + qkv_width * lin["conv_kernel"] + lin["value_dim"] * d.hidden)


def _mlp_params(d: mc.ModelDims, width: int) -> int:
    """Gate, up and down projections of a gated MLP (three matrices), or up and down for the two-projection families."""
    return d.mlp_projections * d.hidden * width


PLAIN_MIXERS = ("full_attention", "sliding_attention", "attention")


def _mixer_params(d: mc.ModelDims, kind: str) -> int:
    if kind == "linear_attention":
        return _linear_params(d)
    if kind in PLAIN_MIXERS:
        return _mla_params(d) if d.mla else _attention_params(d)
    raise FloorInputError(f"the layer type {kind!r} is not one the floor can count (known: linear_attention, {', '.join(PLAIN_MIXERS)})")


@dataclass(frozen=True)
class DecodeFloor:
    """One floor estimate with everything needed to redo it. `by_group` holds bytes read per token for each group."""
    bytes_per_token: float
    by_group: dict
    active_params: float
    stored_params: float
    floor_ms: float
    floor_tok_s: float
    dies: int
    per_die_bandwidth_gb_s: float
    dtype_policy: dict
    board: str | None = None
    label: str = "estimate"
    assumptions: str = ASSUMPTIONS
    model_summary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """A plain JSON-ready record: the numbers, the inputs and the assumptions, labelled an estimate."""
        return {"label": self.label, "bytes_per_token": self.bytes_per_token, "floor_ms": self.floor_ms,
                "floor_tok_s": self.floor_tok_s, "bytes_by_group": dict(self.by_group), "active_params": self.active_params,
                "stored_params": self.stored_params,
                "inputs": {"dies": self.dies, "per_die_bandwidth_gb_s": self.per_die_bandwidth_gb_s,
                           "dtype_policy": dict(self.dtype_policy), "board": self.board, "model": dict(self.model_summary)},
                "assumptions": self.assumptions}


def decode_floor(model_config, dtype_policy, dies, per_die_bandwidth_gb_s, *, board: str | None = None) -> DecodeFloor:
    """The floor on decode time per token for `model_config` (a Hugging Face config dict) under `dtype_policy` on `dies` dies.

    `dtype_policy` maps a group (attention, dense_mlp, router, routed_experts, shared_experts, lm_head) or `default` to a dtype name in
    DTYPE_BYTES. `per_die_bandwidth_gb_s` is required; there is no default. `board` is a label for the record only.
    Raises FloorInputError for a bad input or a layer type it cannot count."""
    dies = _check_dies(dies)
    bw = _check_bandwidth(per_die_bandwidth_gb_s)
    policy = _check_policy(dtype_policy)
    try:
        d = mc.dims(model_config)
    except mc.ModelConfigError as exc:
        raise FloorInputError(str(exc)) from exc

    def width(group: str) -> float:
        return DTYPE_BYTES[policy.get(group, policy.get("default"))]

    active: dict[str, float] = {g: 0 for g in GROUPS}
    stored = 0
    for i in range(d.layers):
        attn = _mixer_params(d, d.layer_types[i])
        active["attention"] += attn
        stored += attn
        if d.moe_layers[i]:
            router = d.hidden * d.routed_experts
            expert = _mlp_params(d, d.moe_intermediate)
            shared = _mlp_params(d, d.shared_intermediate) if d.shared_intermediate else 0
            active["router"] += router
            active["routed_experts"] += d.top_k * expert
            active["shared_experts"] += shared
            stored += router + d.routed_experts * expert + shared
        else:
            mlp = _mlp_params(d, d.intermediate)
            active["dense_mlp"] += mlp
            stored += mlp
    head = d.hidden * d.vocab
    active["lm_head"] = head
    stored += head + (0 if d.tied_embeddings else head)      # the embedding table is stored but its lookup is not a full read

    by_group = {g: p * width(g) for g, p in active.items() if p}
    total = sum(by_group.values())
    floor_ms = total / (dies * bw * 1e9) * 1000.0
    summary = {"hidden_size": d.hidden, "layers": d.layers, "vocab_size": d.vocab, "routed_experts": d.routed_experts,
               "experts_per_token": d.top_k, "linear_attention_layers": d.layer_types.count("linear_attention"),
               "moe_layers": sum(d.moe_layers)}
    return DecodeFloor(bytes_per_token=total, by_group=by_group, active_params=sum(active.values()), stored_params=stored,
                       floor_ms=floor_ms, floor_tok_s=1000.0 / floor_ms, dies=dies, per_die_bandwidth_gb_s=bw, dtype_policy=policy,
                       board=board, model_summary=summary)


def floor_for_board(model_config, dtype_policy, board: str, *, dies: int | None = None) -> DecodeFloor:
    """`decode_floor` with the per-die bandwidth and die count of a board type from BOARDS. `dies` overrides the board's count
    (for a run that uses one die of a two-die board)."""
    if board not in BOARDS:
        raise FloorInputError(f"unknown board {board!r} (known: {', '.join(BOARDS)}); pass the bandwidth to decode_floor instead")
    b = BOARDS[board]
    return decode_floor(model_config, dtype_policy, b.dies if dies is None else dies, b.per_die_gb_s, board=board)


# ---- share of the floor -------------------------------------------------------------------------------

def share_of_floor(measured_tok_s, floor: DecodeFloor) -> float:
    """Measured tokens per second divided by the floor's tokens per second. Refuses a negative, non-finite or non-numeric speed."""
    if isinstance(measured_tok_s, bool) or not isinstance(measured_tok_s, (int, float)) or not math.isfinite(measured_tok_s) \
            or measured_tok_s < 0:
        raise FloorInputError("measured tokens per second must be a finite number of at least 0")
    return measured_tok_s / floor.floor_tok_s


def share_text(measured_tok_s, floor: DecodeFloor) -> str:
    """One sentence for a card or bundle: the speed, the share and the floor it is a share of, labelled an estimate."""
    share = share_of_floor(measured_tok_s, floor)
    text = (f"{measured_tok_s:.1f} tok/s is {share:.0%} of an estimated floor of {floor.floor_tok_s:.1f} tok/s "
            f"({floor.floor_ms:.1f} ms per token, {floor.bytes_per_token / 1e9:.1f} GB read per token over {floor.dies} "
            f"{'die' if floor.dies == 1 else 'dies'} at {floor.per_die_bandwidth_gb_s:.0f} GB/s each)")
    if share > 1.0:
        text += ("; the speed is above the floor, so an input is wrong (a smaller weight dtype than the policy says, more dies) or "
                 "the run emits several tokens per weight read (speculative decoding)")
    return text + "."


# Likely causes of a low share, most likely first (the order in published fused-decode work and in issue 181).
LOW_SHARE_CAUSES = ("op count and gaps between ops", "weight streaming below the bank bandwidth", "collectives at layer boundaries",
                    "one-row-in-a-tile ops", "host round trips")


def low_share_hint(share, *, threshold: float) -> str | None:
    """Text only, never a gate: when `share` is below `threshold`, the likely causes in order; otherwise None."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise FloorInputError("threshold must be a positive finite number")
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not math.isfinite(share) or share < 0:
        raise FloorInputError("share must be a finite number of at least 0")
    if share >= threshold:
        return None
    return (f"Decode reaches {share:.0%} of the estimated floor, below the {threshold:.0%} threshold. Likely causes, most likely first: "
            + "; ".join(f"{i}. {c}" for i, c in enumerate(LOW_SHARE_CAUSES, 1)) + ".")


def render_table(rows) -> str:
    """A text table for a log. `rows` is a list of (model name, DecodeFloor, [(label, measured tok/s), ...]).
    Every share is a share of an estimated floor and the header says so."""
    out = ["| model | floor tok/s (estimate) | GB per token | measurement | tok/s | share of the estimated floor |",
           "| --- | --- | --- | --- | --- | --- |"]
    for name, floor, measured in rows:
        if not measured:
            out.append(f"| {name} | {floor.floor_tok_s:.1f} | {floor.bytes_per_token / 1e9:.1f} | | | |")
        for label, tok_s in measured:
            out.append(f"| {name} | {floor.floor_tok_s:.1f} | {floor.bytes_per_token / 1e9:.1f} | {label} | {tok_s:.1f} | "
                       f"{share_of_floor(tok_s, floor) * 100:.1f}% |")
    return "\n".join(out)


def main(argv=None) -> int:
    """`python3 -m orchard.decode_floor CONFIG --board p300 --policy default=bf16,routed_experts=bfp4 [--measured 18.5]`: print one floor."""
    import argparse
    import os
    import json
    ap = argparse.ArgumentParser(description="Estimate the DRAM-bandwidth floor on decode time per token for a model config.")
    ap.add_argument("config", help="path to the model's config.json")
    ap.add_argument("--board", choices=sorted(BOARDS), help="board type (sets dies and per-die bandwidth)")
    ap.add_argument("--dies", type=int, help="die count (overrides the board's)")
    ap.add_argument("--per-die-gb-s", type=float, help="per-die bandwidth in GB/s (required without --board)")
    ap.add_argument("--policy", default="default=bf16", help="comma list of group=dtype, for example default=bfp8,routed_experts=bfp4")
    ap.add_argument("--measured", type=float, help="a measured tok/s to state as a share of the floor")
    a = ap.parse_args(argv)
    try:
        policy = dict(p.split("=", 1) for p in a.policy.split(",") if p)
        cfg = mc.read_model_config(os.path.realpath(a.config))      # a Hugging Face cache snapshot file is a link into blobs/
        if a.board:
            f = floor_for_board(cfg, policy, a.board, dies=a.dies)
        elif a.per_die_gb_s is not None and a.dies:
            f = decode_floor(cfg, policy, a.dies, a.per_die_gb_s)
        else:
            raise FloorInputError("give --board, or both --dies and --per-die-gb-s")
        print(json.dumps(f.as_dict(), indent=2))
        if a.measured is not None:
            print(share_text(a.measured, f))
    except ValueError as exc:        # FloorInputError, ModelConfigError, or a malformed --policy
        print(f"decode_floor: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
