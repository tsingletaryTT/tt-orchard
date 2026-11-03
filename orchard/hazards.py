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
# Key spellings per family: the DeepSeek and GLM indexer is index_*, the Qwen4 preview (model_type qwen4_exp, Qwen3.8-Flash-Next) writes indexer_*;
# DeepSeek hyper-connections are hc_mult and hc_eps, the Qwen4 gated residual is hc_count and hc_lowrank. Only the key names are read, so a
# later model that keeps these spellings under another model_type is still seen.
SPARSE_ATTENTION_KEYS = frozenset({"index_topk", "index_n_heads", "index_head_dim", "indexer_budget", "indexer_n_heads", "indexer_head_dim"})
HYPER_CONNECTION_KEYS = frozenset({"hc_mult", "hc_eps", "hc_count", "hc_lowrank"})
QWEN4_TYPES = ("qwen4_exp", "qwen4_exp_text")       # transformers' model types for the Qwen4 preview (and a dense variant, if one appears)
NGRAM_EMBEDDING_KEYS = frozenset({"ngram_size", "ngram_vocab_size_base"})
MIXER_RULES = {
    "multi_latent_attention": "the config has kv_lora_rank",
    "sparse_attention": "the config has an attention indexer key (index_topk, index_n_heads, index_head_dim, indexer_budget, indexer_n_heads or indexer_head_dim)",
    "hyper_connections": "the config has a hyper-connection key (hc_mult, hc_eps, hc_count, hc_lowrank or a key starting with hyper_connection)",
    "ngram_embedding": "the config has an n-gram embedding key (ngram_size or ngram_vocab_size_base)",
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
    TableEntry("mixer_mamba_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no Mamba-2 (state-space) kernels: the short convolution, the state update and the gated norm", _NEAREST,
               "run one Mamba-2 layer (a prefill, then single-token steps that carry the conv and state caches) against the CPU result before the full model"),
    TableEntry("mixer_multi_latent_attention_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no multi-latent attention kernels", _NEAREST,
               "run one multi-latent attention layer (prefill and decode) against the CPU result before the full model"),
    TableEntry("mixer_sparse_attention_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no sparse attention (indexer) kernels", _NEAREST,
               "run one sparse-attention layer against the CPU result at a context longer than the indexer's top-k"),
    TableEntry("mixer_hyper_connections_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no hyper-connection kernels", _NEAREST,
               "run one decoder layer with its hyper-connections against the CPU result before the full model"),
    TableEntry("mixer_ngram_embedding_not_in_nearest", "mixer implemented by the nearest bundle",
               "the nearest bundle has no n-gram embedding: a table indexed by hashes of the last few token ids (table size not computed "
               "from the config; check the checkpoint headers for its size and whether it fits device memory)", _NEAREST,
               "run the n-gram lookup for one token (hash the last ngram_size token ids for every head, read the rows) and the layer's "
               "gate and convolution against the CPU result, then three tokens in a row so the history carries over"),
    TableEntry("dense_model_nearest_is_moe", "a model of the same shape as the nearest bundle (routed experts or none)",
               "the model has no routed experts and the nearest bundle does, so the bundle's router, expert and shared-expert paths are not the model's MLP", _NEAREST,
               "run one decoder layer's dense MLP (gate, up, down and the activation) at the model's width against the CPU result before the full model"),
    TableEntry("dense_qwen4_not_flash_next_shape", "a Qwen4-family model with routed experts (Qwen3.8-Flash-Next has them in every layer)",
               "a qwen4_exp config with no routed experts is a dense variant. transformers 5.19.0 builds a mixture-of-experts block in every layer of qwen4_exp, so "
               "there is no upstream dense reference, and the Flash-Next ports have no dense MLP path. PROVISIONAL: no dense Qwen4 config or weights exist yet",
               "transformers qwen4_exp (5.19.0) requires num_experts > 0; the dense reading comes from public reports (TensorFold issue 1244, llama.cpp issue 31072)",
               "build the CPU reference with orchard/skills/reference-gate-templates/qwen4_dense.py, then run one dense layer with its hyper-connections against it"),
    TableEntry("hyper_connection_width_not_divisible_by_dies", "hc_count x hidden_size divisible by the die count (checked at 2, 4 and the scan's dies)",
               "the residual stream is hc_count copies of the hidden state; a width that does not split evenly over the dies leaves an uneven shard", _SUPPLIED,
               "run the gated residual (norm, down projection, up projection, gate) on a tensor of the full width sharded over the dies and compare with the CPU result"),
    TableEntry("ffn_per_die_not_whole_tiles", "a dense FFN width that splits over the dies into a multiple of 32 columns (checked at 2, 4 and the scan's dies)",
               "a dense MLP column-sharded over the dies needs whole 32-wide tiles per die, or the shard is padded", "tile layout: every tensor on the chip is stored in 32 by 32 tiles",
               "run the dense MLP sharded over the dies at the model's width and compare with the CPU result, looking at the last tile of each shard"),
    TableEntry("attention_heads_not_divisible_by_dies", "query heads divisible by the die count (checked at 2, 4 and the scan's dies)",
               "attention heads are split over the dies; a count that does not divide leaves a die with a different number of heads", _SUPPLIED,
               "run one attention layer sharded over the dies at the model's head count and compare with the CPU result"),
    TableEntry("kv_heads_not_divisible_by_dies", "KV heads divisible by the die count, or the die count a multiple of the KV heads (replication)",
               "KV heads are split over the dies, or copied when there are fewer than dies; a count that does neither evenly leaves dies with different caches", _SUPPLIED,
               "run one decode attention step sharded over the dies at the model's KV-head count and compare with the CPU result"),
    TableEntry("qsa_heads_not_divisible_by_dies", "sparse-attention indexer heads divisible by the die count (checked at 2, 4 and the scan's dies)",
               "the indexer scores are computed per head and summed over heads; heads that do not divide over the dies need a cross-die sum of uneven parts", _SUPPLIED,
               "run the indexer at the model's head count sharded over the dies against the CPU result at a context longer than the top-k"),
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
    if d.keys & SPARSE_ATTENTION_KEYS:
        kinds.add("sparse_attention")
    if d.keys & HYPER_CONNECTION_KEYS or any(k.startswith("hyper_connection") for k in d.keys):
        kinds.add("hyper_connections")
    if d.keys & NGRAM_EMBEDDING_KEYS:
        kinds.add("ngram_embedding")
    return sorted(kinds)


def _dimension_values(d: mc.ModelDims) -> dict[str, int]:
    out = {"hidden_size": d.hidden, "intermediate_size": d.intermediate, "head_dim": d.head_dim}
    if d.routed_experts:
        out["moe_intermediate_size"] = d.moe_intermediate
    if d.shared_intermediate:
        out["shared_intermediate_size"] = d.shared_intermediate
    return out


CHECKED_DIES = (2, 4)           # the die counts the division checks always cover (one board, two boards), beside the scan's own


def _optional_whole(cfg: dict, key: str) -> int | None:
    """A whole number of at least 1 at `key`, None when the key is absent or null (there is no default), HazardInputError otherwise."""
    if cfg.get(key) is None:
        return None
    try:
        return mc._num(cfg, key)
    except mc.ModelConfigError as exc:
        raise HazardInputError(str(exc)) from exc


def _extras(config) -> dict:
    """Config values the scan reads straight from the file with no built-in default: hc_count, hc_lowrank, the indexer's head count and
    the model type. A missing value is None."""
    cfg = mc.text_config(config)
    qsa = next((k for k in ("indexer_n_heads", "index_n_heads") if cfg.get(k) is not None), None)
    return {"hc_count": _optional_whole(cfg, "hc_count"), "hc_lowrank": _optional_whole(cfg, "hc_lowrank"),
            "qsa_heads": _optional_whole(cfg, qsa) if qsa else None, "model_type": str(cfg.get("model_type") or "")}


def _dies_checked(dies: int) -> list[int]:
    return sorted({*CHECKED_DIES, dies} - {1})


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


def _hits(d: mc.ModelDims, dies: int, limits: dict, tile_multiples: dict, x: dict | None = None) -> dict[str, tuple]:
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
    hits.update(_division_hits(d, dies, x or {}))
    sliding = sum(1 for t in d.layer_types if t == "sliding_attention")
    if "sliding_window_layers_max" in limits and sliding > limits["sliding_window_layers_max"]:
        hits["sliding_window_layers_above_limit"] = (sliding, limits["sliding_window_layers_max"], None)
    return hits


def _division_hits(d: mc.ModelDims, dies: int, x: dict) -> dict[str, tuple]:
    """Checks that a width or a head count splits over the dies, at 2 and 4 dies and the scan's own count. Each hit's value is the list of
    die counts that fail."""
    hits: dict[str, tuple] = {}
    checked = _dies_checked(dies)
    stated = f"divisible by every die count checked ({', '.join(map(str, checked))})"
    if x.get("hc_count"):
        width = x["hc_count"] * d.hidden
        bad = [n for n in checked if width % n]
        if bad:
            hits["hyper_connection_width_not_divisible_by_dies"] = (
                bad, stated, f"the residual width hc_count x hidden_size = {x['hc_count']} x {d.hidden} = {width} does not divide evenly over {bad} dies")
    if any(not m for m in d.moe_layers) or not d.routed_experts:       # at least one layer has a plain MLP of intermediate_size
        bad = [n for n in checked if d.intermediate % n or (d.intermediate // n) % TILE]
        if bad:
            hits["ffn_per_die_not_whole_tiles"] = (
                bad, f"intermediate_size per die a multiple of {TILE}",
                f"the dense FFN width {d.intermediate} does not split into a multiple of {TILE} columns per die over {bad} dies")
    bad = [n for n in checked if d.heads % n]
    if bad:
        hits["attention_heads_not_divisible_by_dies"] = (bad, stated, f"{d.heads} query heads do not divide evenly over {bad} dies")
    bad = [n for n in checked if d.kv_heads % n and n % d.kv_heads]
    if bad:
        hits["kv_heads_not_divisible_by_dies"] = (bad, "divisible by the die count, or the die count a multiple of the KV heads",
                                                  f"{d.kv_heads} KV heads neither divide over nor replicate across {bad} dies")
    if x.get("qsa_heads"):
        bad = [n for n in checked if x["qsa_heads"] % n]
        if bad:
            hits["qsa_heads_not_divisible_by_dies"] = (bad, stated, f"{x['qsa_heads']} indexer heads do not divide evenly over {bad} dies")
    return hits


# The hazards of a full vocabulary projection: the LM head and the sampler that reads it (issue 380). A reranker reads two rows of the head
# (yes and no), and an embedding model has no head, so neither runs the projection or the sampler these two rows describe.
FULL_HEAD_HAZARDS = ("vocab_per_die_above_65536", "vocab_not_whole_tiles")


def scan_model_hazards(model_config, nearest_bundle_config, *, dies: int = 2, limits=None, tile_multiples=None,
                       kind: str = "generative", has_output_head: bool | None = None) -> HazardScan:
    """Scan `model_config` against the hazard table and the nearest bundle's config (None when there is none).

    `dies` is the number of dies the vocabulary is split over (default 2, one p300 board). `limits` supplies the limits that are not
    on record (router_top_k_max, kv_heads_max, sliding_window_layers_max). `tile_multiples` maps a dimension to the multiple of tiles
    the fused kernels accept, when the caller knows it. Raises HazardInputError for an unusable config or option.

    A hazard the nearest bundle's own config also hits is marked `also_on_nearest`. Mixers and the top-k and KV-head comparisons need
    the nearest config; without it they are skipped and the notes say so.

    `kind` is the model kind (orchard/model_kind.py): for a `reranker` or an `embedding` model, and for any model whose config describes
    a bare backbone (`has_output_head` False), the two hazards of a full vocabulary projection (FULL_HEAD_HAZARDS) are left out, and a
    note says so. Every other hazard is scanned as before, because the backbone is the same."""
    if isinstance(dies, bool) or not isinstance(dies, int) or dies < 1:
        raise HazardInputError("dies must be a whole number of at least 1")
    if kind not in ("generative", "reranker", "embedding"):
        raise HazardInputError(f"kind must be generative, reranker or embedding, got {kind!r}")
    lim, mult = _limits(limits), _tile_multiples(tile_multiples)
    d = _dims(model_config, "model config")
    n = _dims(nearest_bundle_config, "nearest bundle config") if nearest_bundle_config is not None else None
    xm = _extras(model_config)
    xn = _extras(nearest_bundle_config) if nearest_bundle_config is not None else None
    mine = _hits(d, dies, lim, mult, xm)
    theirs = _hits(n, dies, lim, mult, xn) if n is not None else {}
    notes: list[str] = []
    if kind != "generative" or has_output_head is False:
        skipped = [h for h in FULL_HEAD_HAZARDS if h in mine]
        for h in FULL_HEAD_HAZARDS:
            mine.pop(h, None)
            theirs.pop(h, None)
        why = ("it reads two rows of the output head (yes and no), so the full-vocabulary projection and the sampler do not run" if kind == "reranker"
               else "it has no output head, so no vocabulary projection or sampler runs")
        notes.append(f"the model kind is {kind}" + (" with a bare backbone" if has_output_head is False and kind == "generative" else "")
                     + f": {why}, and the vocabulary hazards ({', '.join(FULL_HEAD_HAZARDS)}) were not applied"
                     + (f" (they would have hit: {', '.join(skipped)})" if skipped else ""))
    if xm["hc_count"] and xm["hc_lowrank"] is None:
        notes.append("the config has hc_count but no hc_lowrank; the scan does not assume one (a full-size reading is 640, a TP2 shard 320)")
    if xm["hc_lowrank"] and xm["hc_count"] is None:
        notes.append("the config has hc_lowrank but no hc_count; the hyper-connection width was not computed")

    found: list[Hazard] = []
    for name, (value, limit, why) in mine.items():
        e = _BY_NAME.get(name) or _generated_entry(name, why)
        check = e.stage2_check.replace("{last}", str(value - 1)) if isinstance(value, int) and not isinstance(value, bool) else e.stage2_check
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
        if not d.routed_experts and n.routed_experts:
            e = _BY_NAME["dense_model_nearest_is_moe"]
            found.append(Hazard(e.name, 0, n.routed_experts, e.why, e.source, e.stage2_check))
        have = set(_mixers(n))
        for mixer in _mixers(d):
            name = f"mixer_{mixer}_not_in_nearest"
            if mixer not in have and name in _BY_NAME:
                e = _BY_NAME[name]
                rule = MIXER_RULES.get(mixer, "the config's layer_types names it")
                found.append(Hazard(name, mixer, sorted(have), f"{e.why} (detected because {rule})", e.source, e.stage2_check))
            elif mixer not in have:
                notes.append(f"the model uses the mixer {mixer!r}, which the nearest bundle does not and the table has no row for")

    if not d.routed_experts and xm["model_type"] in QWEN4_TYPES:
        e = _BY_NAME["dense_qwen4_not_flash_next_shape"]
        found.append(Hazard(e.name, 0, "routed experts in every layer", e.why, e.source, e.stage2_check))

    hc = {"hc_count": xm["hc_count"], "hc_lowrank": xm["hc_lowrank"], "width": xm["hc_count"] * d.hidden if xm["hc_count"] else None}
    facts = {"dimensions": {"hidden_size": d.hidden, "intermediate_size": d.intermediate, "head_dim": d.head_dim,
                            "vocab_size": d.vocab, "layers": d.layers, "heads": d.heads, "kv_heads": d.kv_heads},
             "tiles": {k: {"value": v, "tiles": v / TILE, "whole": v % TILE == 0} for k, v in _dimension_values(d).items()},
             "routed_experts": d.routed_experts, "router_top_k": d.top_k, "vocab_per_die": math.ceil(d.vocab / dies),
             "mixers": _mixers(d), "hyper_connections": hc if (hc["hc_count"] or hc["hc_lowrank"]) else None,
             "qsa_heads": xm["qsa_heads"], "dies_checked": _dies_checked(dies), "dense": not d.routed_experts, "model_kind": kind}
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
