# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hazard scan: model dimensions that break known kernels, predicted from the config alone (issue 183).

Published porting notes for mixture-of-experts models on Blackhole list failures that came from model dimensions: an expert mask 256
bits wide dropped experts past index 255, a boundary kernel rejected a hidden size of 3 tiles, and a sampler index wrapped at 65,536
vocabulary columns per die. Each fails with wrong output or a hang and no clean error, and each can be seen in the config before any
chip time is spent.

`HAZARD_TABLE` is the table: one row per hazard with its limit, why it matters, where the limit comes from and the targeted check
stage 2 should run first. Add a row when a bundle reports a new limit, and add the check that finds it in `scan_model_hazards`.
`scan_model_hazards(model_config, nearest_bundle_config)` returns the hits as named hazards. A hit never stops a run. It is a
prediction from the config: it says where to look first when output is wrong.

Limits that are not on record. The width the router kernel supports, the KV-head count and the sliding-window layer count
have no confirmed figure here. They are inputs (`limits=`), and the scan also compares the router top-k and the KV-head count with
the nearest bundle's, which needs no figure. A hazard whose dimension the nearest bundle shares (and the nearest bundle is known to
run) is marked `also_on_nearest`: a hit that the bundle already survives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from orchard import model_config as mc

TILE = 32
EXPERT_MASK_LIMITS = (256, 512)
VOCAB_PER_DIE_LIMIT = 65_536
# A hidden size of this many tiles was rejected by a boundary kernel (issue 183, published Blackhole porting notes).
REJECTED_HIDDEN_TILES = (3,)

_ISSUE = "issue 183 (published Blackhole porting notes for mixture-of-experts models)"
_SUPPLIED = "a limit supplied by the caller; no figure is on record in this repository"
_NEAREST = "the nearest installed bundle's config: the bundle is known to run, a model above it is not covered by that evidence"

# Dimensions checked for whole tiles. The key is the name used in hazard names and in the facts record.
TILED_DIMENSIONS = ("hidden_size", "intermediate_size", "head_dim", "moe_intermediate_size", "shared_intermediate_size")
# The limits a caller may supply, and what each limits.
LIMIT_NAMES = ("router_top_k_max", "kv_heads_max", "sliding_window_layers_max")
# Mixers: how each is detected, so the hazard can say what it saw. A model's mixer set is its layer types (full and sliding-window
# attention both count as plain attention) plus these key-based detections.
MIXER_RULES = {
    "multi_latent_attention": "the config has kv_lora_rank",
    "sparse_attention": "the config has an attention indexer key (index_topk, index_n_heads or index_head_dim)",
    "hyper_connections": "the config has a hyper-connection key (hc_mult, hc_eps or a key starting with hyper_connection)",
}


@dataclass(frozen=True)
class TableEntry:
    """One row of the hazard table."""
    name: str
    limit: str            # the limit in words
    why: str              # what goes wrong
    source: str           # where the limit comes from
    stage2_check: str     # the targeted check for stage 2


def _tile_rows() -> list[TableEntry]:
    rows = []
    for dim in TILED_DIMENSIONS:
        rows.append(TableEntry(
            f"{dim}_not_whole_tiles", "a whole number of 32-wide tiles",
            f"{dim} is not a multiple of 32, so a tiled matmul pads it or reads past the edge",
            "tile layout: every tensor on the chip is stored in 32 by 32 tiles",
            f"run one matmul or norm with {dim} at its real width and compare with the CPU result, looking at the last tile"))
    return rows


HAZARD_TABLE: tuple[TableEntry, ...] = tuple([
    TableEntry("routed_experts_above_256", f"routed experts at most {EXPERT_MASK_LIMITS[0]}",
               "an expert mask 256 bits wide drops the experts past index 255", _ISSUE,
               "run the router with a token routed to expert index {last} (the highest) and check that the expert's output arrives"),
    TableEntry("routed_experts_above_512", f"routed experts at most {EXPERT_MASK_LIMITS[1]}",
               "a 512-expert model needed a wider mask than 256 bits; above 512 the wider mask is also short", _ISSUE,
               "run the router with a token routed to expert index {last} (the highest) and check that the expert's output arrives"),
    TableEntry("vocab_per_die_above_65536", f"vocabulary columns per die at most {VOCAB_PER_DIE_LIMIT}",
               "a sampler index wraps at 65,536 columns per die and returns a wrong token", _ISSUE,
               "sample with the greedy and the sampled path on logits whose maximum sits in the last column of each die"),
    TableEntry("vocab_not_whole_tiles", "vocabulary a whole number of 32-wide tiles",
               "the vocabulary is not a multiple of 32, so the LM head and the sampler pad it", "tile layout: every tensor on the chip is stored in 32 by 32 tiles",
               "run the LM head and the sampler on logits whose maximum sits in the padded columns"),
    TableEntry("mla_flash_decode_needs_core_grid", "multi-latent attention decode with an explicit core grid of at most 64 cores",
               "flash_multi_latent_attention_decode with one latent key-value head at batch 1 spreads over all 110 Blackhole cores and fails with 'Tree reduction max 6 "
               "rounds (64 cores/head)'; an SDPAProgramConfig with an 8 by 8 grid works", "issue 213 (MiniCPM3-4B port on one Blackhole chip, overnight 2026-10-28)",
               "run the flash multi-latent decode op alone at the model's head count, latent rank and rope width at batch 1, first with no program config and then "
               "with an 8 by 8 SDPAProgramConfig, and compare with the CPU result"),
    TableEntry("hidden_size_tile_count_rejected", "hidden size not 3 tiles (96)",
               "a boundary kernel rejected a hidden size of 3 tiles", _ISSUE,
               "run the layer-boundary kernel alone at the model's hidden size"),
    TableEntry("router_top_k_above_nearest", "top-k at most the nearest bundle's",
               "the router kernel was exercised only up to the nearest bundle's top-k", _NEAREST,
               "run the router at the model's top-k and compare the selected experts with the CPU router"),
    TableEntry("router_top_k_above_limit", "top-k at most a supplied limit", "top-k above the width the router kernel supports",
               _SUPPLIED, "run the router at the model's top-k and compare the selected experts with the CPU router"),
    TableEntry("kv_heads_above_nearest", "KV heads at most the nearest bundle's",
               "the attention kernels were exercised only up to the nearest bundle's KV-head count", _NEAREST,
               "run one decode attention step at the model's KV-head count against the CPU result"),
    TableEntry("kv_heads_above_limit", "KV heads at most a supplied limit", "more KV heads than the supplied limit", _SUPPLIED,
               "run one decode attention step at the model's KV-head count against the CPU result"),
    TableEntry("sliding_window_layers_above_limit", "sliding-window layers at most a supplied limit",
               "more sliding-window layers than the supplied limit, counted over the whole model (the config does not say how layers are grouped)", _SUPPLIED,
               "run decode past the window length and compare with the CPU result"),
    TableEntry("mixer_linear_attention_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no linear attention kernels", _NEAREST,
               "run one linear-attention layer on the chip against the CPU result before the full model"),
    TableEntry("mixer_multi_latent_attention_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no multi-latent attention kernels", _NEAREST,
               "run one multi-latent attention layer (prefill and decode) against the CPU result before the full model"),
    TableEntry("mixer_sparse_attention_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no sparse attention (indexer) kernels", _NEAREST,
               "run one sparse-attention layer against the CPU result at a context longer than the indexer's top-k"),
    TableEntry("mixer_hyper_connections_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no hyper-connection kernels", _NEAREST,
               "run one decoder layer with its hyper-connections against the CPU result before the full model"),
] + _tile_rows())
_BY_NAME = {e.name: e for e in HAZARD_TABLE}


class HazardInputError(ValueError):
    """A config or an option of the scan is unusable."""


@dataclass(frozen=True)
class Hazard:
    name: str
    value: object                    # the model's number (or the mixer name)
    limit: object                    # the limit it was compared with
    why: str
    source: str
    stage2_check: str
    also_on_nearest: bool = False    # the nearest bundle, known to run, has the same hit

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "limit": self.limit, "why": self.why, "source": self.source,
                "stage2_check": self.stage2_check, "also_on_nearest": self.also_on_nearest}


@dataclass(frozen=True)
class HazardScan:
    hazards: tuple
    facts: dict                      # the dimensions the scan looked at, including tile counts
    notes: tuple = ()
    dies: int = 2

    def as_record(self) -> dict:
        """The record for stage 0's facts: plain JSON, with the dimensions, every hazard and its stage 2 check."""
        return {"label": "prediction from the config", "dies": self.dies, "facts": self.facts,
                "hazards": [h.as_dict() for h in self.hazards], "notes": list(self.notes)}


def _dims(config, what: str) -> mc.ModelDims:
    try:
        return mc.dims(config)
    except mc.ModelConfigError as exc:
        raise HazardInputError(f"{what}: {exc}") from exc


def _mixers(d: mc.ModelDims) -> list[str]:
    """The mixer kinds a model uses, sorted. Full and sliding-window attention are both `attention`."""
    kinds = {"attention" if t in ("full_attention", "sliding_attention", "attention") else t for t in d.layer_types}
    if d.mla:
        kinds.add("multi_latent_attention")
    if d.keys & {"index_topk", "index_n_heads", "index_head_dim"}:
        kinds.add("sparse_attention")
    if d.keys & {"hc_mult", "hc_eps"} or any(k.startswith("hyper_connection") for k in d.keys):
        kinds.add("hyper_connections")
    return sorted(kinds)


def _dimension_values(d: mc.ModelDims) -> dict[str, int]:
    out = {"hidden_size": d.hidden, "intermediate_size": d.intermediate, "head_dim": d.head_dim}
    if d.routed_experts:
        out["moe_intermediate_size"] = d.moe_intermediate
    if d.shared_intermediate:
        out["shared_intermediate_size"] = d.shared_intermediate
    return out


def _limits(limits) -> dict:
    if limits is None:
        return {}
    if not isinstance(limits, dict):
        raise HazardInputError("limits must be a dict")
    for k, v in limits.items():
        if k not in LIMIT_NAMES:
            raise HazardInputError(f"limits names an unknown limit {k!r} (known: {', '.join(LIMIT_NAMES)})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 1:
            raise HazardInputError(f"limits {k} must be a finite number of at least 1")
    return dict(limits)


def _tile_multiples(tile_multiples) -> dict:
    if tile_multiples is None:
        return {}
    if not isinstance(tile_multiples, dict):
        raise HazardInputError("tile_multiples must be a dict")
    for k, v in tile_multiples.items():
        if k not in TILED_DIMENSIONS:
            raise HazardInputError(f"tile_multiples names an unknown dimension {k!r} (known: {', '.join(TILED_DIMENSIONS)})")
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise HazardInputError(f"tile_multiples {k} must be a whole number of at least 1")
    return dict(tile_multiples)


def _hits(d: mc.ModelDims, dies: int, limits: dict, tile_multiples: dict) -> dict[str, tuple]:
    """The absolute checks (no nearest bundle needed) on one model: hazard name -> (value, limit, why override or None)."""
    hits: dict[str, tuple] = {}
    if d.routed_experts > EXPERT_MASK_LIMITS[0]:
        hits["routed_experts_above_256"] = (d.routed_experts, EXPERT_MASK_LIMITS[0], None)
    if d.routed_experts > EXPERT_MASK_LIMITS[1]:
        hits["routed_experts_above_512"] = (d.routed_experts, EXPERT_MASK_LIMITS[1], None)
    per_die = math.ceil(d.vocab / dies)
    if per_die > VOCAB_PER_DIE_LIMIT:
        hits["vocab_per_die_above_65536"] = (per_die, VOCAB_PER_DIE_LIMIT, None)
    if d.vocab % TILE:
        hits["vocab_not_whole_tiles"] = (d.vocab, f"a multiple of {TILE}", None)
    for dim, value in _dimension_values(d).items():
        if value % TILE:
            hits[f"{dim}_not_whole_tiles"] = (value, f"a multiple of {TILE}", None)
        elif dim in tile_multiples and (value // TILE) % tile_multiples[dim]:
            hits[f"{dim}_tiles_not_multiple_of_{tile_multiples[dim]}"] = (
                value // TILE, f"a multiple of {tile_multiples[dim]} tiles",
                f"{dim} is {value // TILE} tiles, and the caller says the fused kernels accept only multiples of {tile_multiples[dim]}")
    if d.mla:
        hits["mla_flash_decode_needs_core_grid"] = (d.mla["kv_lora_rank"] + d.mla["qk_rope_head_dim"], "an explicit program config on at most 64 cores", None)
    if d.hidden % TILE == 0 and d.hidden // TILE in REJECTED_HIDDEN_TILES:
        hits["hidden_size_tile_count_rejected"] = (d.hidden // TILE, f"not {d.hidden // TILE} tiles", None)
    if "router_top_k_max" in limits and d.top_k > limits["router_top_k_max"]:
        hits["router_top_k_above_limit"] = (d.top_k, limits["router_top_k_max"], None)
    if "kv_heads_max" in limits and d.kv_heads > limits["kv_heads_max"]:
        hits["kv_heads_above_limit"] = (d.kv_heads, limits["kv_heads_max"], None)
    sliding = sum(1 for t in d.layer_types if t == "sliding_attention")
    if "sliding_window_layers_max" in limits and sliding > limits["sliding_window_layers_max"]:
        hits["sliding_window_layers_above_limit"] = (sliding, limits["sliding_window_layers_max"], None)
    return hits


def scan_model_hazards(model_config, nearest_bundle_config, *, dies: int = 2, limits=None, tile_multiples=None) -> HazardScan:
    """Scan `model_config` against the hazard table and the nearest bundle's config (None when there is none).

    `dies` is the number of dies the vocabulary is split over (default 2, one p300 board). `limits` supplies the limits that are not
    on record (router_top_k_max, kv_heads_max, sliding_window_layers_max). `tile_multiples` maps a dimension to the multiple of tiles
    the fused kernels accept, when the caller knows it. Raises HazardInputError for an unusable config or option.

    A hazard the nearest bundle's own config also hits is marked `also_on_nearest`. Mixers and the top-k and KV-head comparisons need
    the nearest config; without it they are skipped and the notes say so."""
    if isinstance(dies, bool) or not isinstance(dies, int) or dies < 1:
        raise HazardInputError("dies must be a whole number of at least 1")
    lim, mult = _limits(limits), _tile_multiples(tile_multiples)
    d = _dims(model_config, "model config")
    n = _dims(nearest_bundle_config, "nearest bundle config") if nearest_bundle_config is not None else None
    mine = _hits(d, dies, lim, mult)
    theirs = _hits(n, dies, lim, mult) if n is not None else {}
    notes: list[str] = []

    found: list[Hazard] = []
    for name, (value, limit, why) in mine.items():
        e = _BY_NAME.get(name) or _generated_entry(name, why)
        check = e.stage2_check.replace("{last}", str(value - 1)) if isinstance(value, int) else e.stage2_check
        found.append(Hazard(name, value, limit, why or e.why, e.source, check, also_on_nearest=name in theirs))

    if n is None:
        notes.append("no nearest bundle config was given, so the mixer, top-k and KV-head comparisons with it were skipped")
    else:
        if n.top_k and d.top_k > n.top_k:
            e = _BY_NAME["router_top_k_above_nearest"]
            found.append(Hazard(e.name, d.top_k, n.top_k, e.why, e.source, e.stage2_check))
        if d.kv_heads > n.kv_heads:
            e = _BY_NAME["kv_heads_above_nearest"]
            found.append(Hazard(e.name, d.kv_heads, n.kv_heads, e.why, e.source, e.stage2_check))
        have = set(_mixers(n))
        for mixer in _mixers(d):
            name = f"mixer_{mixer}_not_in_nearest"
            if mixer not in have and name in _BY_NAME:
                e = _BY_NAME[name]
                rule = MIXER_RULES.get(mixer, "the config's layer_types names it")
                found.append(Hazard(name, mixer, sorted(have), f"{e.why} (detected because {rule})", e.source, e.stage2_check))
            elif mixer not in have:
                notes.append(f"the model uses the mixer {mixer!r}, which the nearest bundle does not and the table has no row for")

    facts = {"dimensions": {"hidden_size": d.hidden, "intermediate_size": d.intermediate, "head_dim": d.head_dim,
                            "vocab_size": d.vocab, "layers": d.layers, "heads": d.heads, "kv_heads": d.kv_heads},
             "tiles": {k: {"value": v, "tiles": v / TILE, "whole": v % TILE == 0} for k, v in _dimension_values(d).items()},
             "routed_experts": d.routed_experts, "router_top_k": d.top_k, "vocab_per_die": math.ceil(d.vocab / dies),
             "mixers": _mixers(d)}
    return HazardScan(tuple(found), facts, tuple(notes), dies)


def _generated_entry(name: str, why: str | None) -> TableEntry:
    """A hazard generated from a caller-supplied tile multiple: it has no table row because the multiple is the caller's input."""
    return TableEntry(name, "", why or "", "a tile multiple supplied by the caller; no figure is on record in this repository",
                      "run the fused kernel alone at the model's width and compare with the CPU result")


def main(argv=None) -> int:
    """`python3 -m orchard.hazards MODEL_CONFIG --nearest NEAREST_CONFIG [--dies N]`: print the scan record as JSON. Exit 2 on a bad input."""
    import argparse
    import os
    import json
    ap = argparse.ArgumentParser(description="Scan a model config for dimensions that break known kernels.")
    ap.add_argument("config", help="the model's config.json")
    ap.add_argument("--nearest", help="the nearest bundle's model config.json (without it the comparisons with the bundle are skipped)")
    ap.add_argument("--dies", type=int, default=2, help="dies the vocabulary is split over (default 2)")
    a = ap.parse_args(argv)
    try:
        model = mc.read_model_config(os.path.realpath(a.config))      # a Hugging Face cache snapshot file is a link into blobs/
        nearest = mc.read_model_config(os.path.realpath(a.nearest)) if a.nearest else None
        print(json.dumps(scan_model_hazards(model, nearest, dies=a.dies).as_record(), indent=2))
    except (HazardInputError, mc.ModelConfigError) as exc:
        print(f"hazards: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
