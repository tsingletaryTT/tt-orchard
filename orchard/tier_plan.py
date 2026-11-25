# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What a mixture-of-experts model needs on N chips, and when some experts have to live in host memory (issue 739, step 1).

    tt-orchard tier-plan SOURCE [--chips 1,2,4] [--json] ...

SOURCE is a Hugging Face repository id, a `config.json`, or a directory that holds one. Only `config.json` and, when the repository has it,
`model.safetensors.index.json` are read. No weight is downloaded and no chip is touched. `tt-orchard plan` answers a related question from the
safetensors headers of a snapshot that is already on disk; this command answers it before the download, from the config.

Where each number comes from (the output labels them the same way):

  config-derived     parameter counts per group, from `orchard/decode_floor.py`'s counting of the model's config (attention or other mixer,
                     dense MLP, router, shared experts, embedding table, output head, routed experts);
  assumed constant   bytes per element of bf16, bfp8 and bfp4 (`orchard/plan.py`), DRAM per chip, the share of it that is usable, the
                     per-chip reserve and the KV cache allowance (`orchard/defaults.py`), and the host memory margin below;
  estimate           the share of expert picks that land on an expert kept on the chips. It comes from the busiest-expert shares measured on
                     one model (OLMoE-1B-7B, issue 739 and scripts/expert_skew.py) unless the caller passes figures for the model itself.
                     Router skew is a property of the model and of the text, so this number is never presented as a measurement of the model
                     being planned.

The placements, tried in this order for each precision and chip count:

  replicated                      every chip holds every weight (what the generated ports do today);
  experts split over the chips    attention, routers, shared experts and the tables on every chip, the routed experts divided between the chips;
  tensor-parallel                 attention heads, key-value heads, the expert intermediate width and the output head divided over the chips,
                                  routers and the embedding table on every chip: about 1/N of the weights per chip. A layout the plan assumes,
                                  after the published Inkling-2-Mini bundle; it needs the head counts and the expert width (in whole 32-wide
                                  tiles per chip) divisible by N, and a row that cannot use it says which count does not divide;
  hot experts on chips, rest host the same always-resident weights on every chip, as many whole experts (one expert index in every layer) as
                                  fit beside them, and every expert pinned in host memory with a margin (the Strata rule the issue quotes).
                                  The host path is not written (step 3 of the issue); the row says what it would need.

A second layout is reported beside the third placement, because the hybrid bundles Tenstorrent publishes use it: whole layers on the chips (every
expert of a layer) and the other layers' experts in host memory, which needs only the host copies of those layers plus the margin. Every layer takes
the same number of picks per token, so the share of picks that stay on the chips is the share of layers on the chips; that figure follows from the
layout and is not an estimate. Which layout a real port would use is a design choice for step 3.

Nothing here measures speed or quality. A row that fits says only that the weights fit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import http.client
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from orchard import decode_floor as df
from orchard import defaults
from orchard import drop_check
from orchard import model_config as mc
from orchard import plan as weights_plan
from orchard import safeio

GIB = float(1 << 30)              # finite: ok a constant
BYTES_PER_ELEMENT = weights_plan.BYTES_PER_ELEMENT
USABLE_FRACTION = weights_plan.USABLE_FRACTION
TILE = 32                         # a tile is 32 values wide; a die's share of an expert's width is a whole number of them
HOST_MARGIN_GIB = 10.0            # choice, from the issue text (Strata's rule: RAM at least the experts plus about 10 GB)

# The busiest-expert shares of OLMoE-1B-7B (64 experts, top 8) over 16,384 tokens of project text: the share of picks that go to the busiest 10, 25
# and 50 percent of the experts. They are the mean over layers in docs/run-logs/2026-11-11-expert-skew-olmoe-1b-7b.json, and a test keeps them equal.
OLMOE_SKEW = ((0.10, 0.414), (0.25, 0.606), (0.50, 0.824))
OLMOE_LABEL = ("estimate: the busiest-expert shares were measured on OLMoE-1B-7B (64 experts, top 8) over 16,384 tokens of project text "
               "(docs/run-logs/2026-11-11-expert-skew-olmoe-1b-7b.json); not measured for this model")


# No real model has this many parameters in one group (the largest published ones are a few trillion in all). A config that claims more is refused
# before any arithmetic, so a number such as 10**400 cannot overflow the float conversions below (hardening checklist item 9).
MAX_GROUP_PARAMS = 10 ** 15


class TierPlanError(ValueError):
    """The model cannot be planned. The message says why in one sentence and holds no file content."""


@dataclass(frozen=True)
class Shape:
    """Parameter counts of a mixture-of-experts model, per group. All are counts of values; bytes are computed from them later."""
    attention_params: int          # attention or other token mixer, every layer
    dense_mlp_params: int          # layers whose MLP has no routed experts
    router_params: int
    shared_params: int             # shared experts, every MoE layer
    embedding_params: int
    head_params: int               # 0 when the output head is tied to the embedding table
    experts: int                   # routed experts per MoE layer
    top_k: int
    moe_layers: int
    expert_params: int             # one routed expert in one layer
    layers: int
    # counts a tensor-parallel split must divide: (label, value, unit); the die count times the unit must divide the value
    tp_counts: tuple = ()

    @property
    def always_params(self) -> int:
        """What stays on every chip in every placement."""
        return self.attention_params + self.dense_mlp_params + self.router_params + self.shared_params + self.embedding_params + self.head_params

    @property
    def expert_index_params(self) -> int:
        """One expert index in every MoE layer: the unit the chips keep or hand to the host."""
        return self.moe_layers * self.expert_params

    @property
    def expert_params_total(self) -> int:
        return self.experts * self.expert_index_params


def shape(config) -> Shape:
    """The parameter groups of the model a Hugging Face config describes. Raises TierPlanError for a dense model, a family whose layers the
    counting does not know ("unknown family") and a config with a missing or unusable number."""
    try:
        d = mc.dims(config)
    except mc.ModelConfigError as exc:
        text = str(exc)
        raise TierPlanError(text if "layer type" not in text and "layer_types" not in text else f"unknown family: {text}") from exc
    if not d.routed_experts:
        raise TierPlanError("this model has no routed experts, so there is nothing to place in host memory; use `tt-orchard plan` for a dense model")
    attention = dense = router = shared = 0
    moe = 0
    for i in range(d.layers):
        try:
            attention += df._mixer_params(d, d.layer_types[i])
        except df.FloorInputError as exc:
            raise TierPlanError(f"unknown family: {exc}") from exc
        if d.moe_layers[i]:
            moe += 1
            router += d.hidden * d.routed_experts + df._latent_params(d)       # the router, and the latent projections around the experts if any
            shared += df._mlp_params(d, d.shared_intermediate) if d.shared_intermediate else 0
        else:
            dense += df._mlp_params(d, d.intermediate)
    table = d.hidden * d.vocab
    expert = df._expert_params(d)
    for name, n in (("attention", attention), ("dense MLP", dense), ("router", router), ("shared experts", shared), ("embedding table", table),
                    ("routed experts", d.routed_experts * moe * expert)):
        if n > MAX_GROUP_PARAMS:
            raise TierPlanError(f"the config describes more than {MAX_GROUP_PARAMS:.0e} parameters in the {name}; no real model is that large")
    counts = [("attention heads", d.heads, 1), ("key-value heads", d.kv_heads, 1)]
    if d.swa_heads:
        counts.append(("sliding-window attention heads", d.swa_heads, 1))
    if d.linear:
        counts += [("linear-attention key heads", d.linear["key_heads"], 1), ("linear-attention value heads", d.linear["value_heads"], 1)]
    if d.mamba:
        counts.append(("mamba heads", d.mamba["heads"], 1))
    if d.moe_intermediate:
        counts.append(("expert intermediate width", d.moe_intermediate, TILE))      # every die holds whole tiles of the expert width
    return Shape(attention_params=attention, dense_mlp_params=dense, router_params=router, shared_params=shared, embedding_params=table,
                 head_params=0 if d.tied_embeddings else table, experts=d.routed_experts, top_k=d.top_k, moe_layers=moe,
                 expert_params=expert, layers=d.layers, tp_counts=tuple(counts))


def tp_blockers(s: Shape, chips: int) -> list[str]:
    """The counts that a tensor-parallel split over `chips` dies cannot divide, each as "label (value)". Empty when the split is possible."""
    return [f"{label} ({value})" for label, value, unit in s.tp_counts if value % (chips * unit)]


def caveats(config) -> list[str]:
    """One sentence for each feature the config names that the parameter count leaves out. The always-resident figure is then a lower bound.
    Found by key name in the language model's settings (and `vision_config` at the top level); a key set to null or 0 does not count."""
    try:
        cfg = mc.text_config(config)
    except mc.ModelConfigError:
        return []
    top = config if isinstance(config, dict) else {}

    def has(*prefixes) -> bool:
        return any(k.startswith(prefixes) and cfg[k] not in (None, 0, False, "", [], {}) for k in cfg)
    out = []
    if has("hc_"):
        out.append("hyper-connections (hc_ keys): their weights are not counted")
    if has("indexer_"):
        out.append("the sparse-attention indexer (indexer_ keys): its weights are not counted")
    if has("ngram_", "ple_", "heads_per_ngram", "make_ngram"):
        out.append("a hashed n-gram embedding (ngram_ and ple_ keys): its table is not counted and can be large")
    if has("engram_"):
        table = mc.engram_table(cfg)
        size = (f": {table['params'] / 1e9:.1f}B parameters in {table['layers']} layers, {table['bytes'] / (1 << 30):.1f} GiB as "
                f"{'FP8 rows with a bf16 scale each' if table['fp8'] else 'bf16'}, "
                f"kept in host memory by the published bundle (the card's figure; the plan does not place it)") if table else " (the table sizes in the config could not be read)"
        out.append("an Engram n-gram memory (engram_ keys)" + size)
    if has("mtp_num_hidden_layers", "num_nextn_predict_layers"):
        out.append("a multi-token-prediction layer (mtp keys): its weights are not counted")
    if isinstance(cfg.get("loop_config"), dict):
        out.append("looped layers (loop_config): the gate weights a looped block reads on each pass (loop controllers) are not counted. The fit counts "
                   "each looped layer's weights once; decode reads them once per pass (orchard/decode_floor.py)")
    if top.get("vision_config") or cfg.get("vision_config"):
        out.append("a vision tower (vision_config): only the language model is counted")
    return out


def budget_bytes(chip_gib: float, *, chips: int, reserve_gib: float, kv_gib: float, kv_per_chip_gib: float | None = None) -> float:
    """Bytes of one chip that weights may use: the usable share of its DRAM, less the per-chip reserve and this chip's share of the KV cache
    allowance. Never negative. With `kv_per_chip_gib` (an estimate for a model whose cache is the same on every die) each chip loses that much
    and the allowance is not divided."""
    kv_here = kv_gib / chips if kv_per_chip_gib is None else kv_per_chip_gib
    return max(0.0, (chip_gib * USABLE_FRACTION - reserve_gib - kv_here) * GIB)


KV_BYTES_PER_VALUE = 2          # the bundles cache the latent rows in bf16 (the Hy4-flash card: "KV: bf16")
KV_ESTIMATE_LABEL = ("estimate from the config: the latent rows only (kv_lora_rank + qk_rope_head_dim values, bf16), the same on every die; the sparse-attention "
                     "indexer's keys and any other state are not counted")
MAX_CONTEXT_TOKENS = 1 << 31
MAX_USERS = 1 << 20


def kv_estimate(config, context_tokens: int, users: int = 1) -> dict | None:
    """A KV cache estimate for a model with multi-latent attention, or None for any other model (the flat allowance applies) or a layout it cannot read.

    A full-attention layer (`full_attention`, `attention` or `deepseek_sparse_attention`) caches one latent row for each of the `context_tokens`
    held across all users (a shared pool, as the Hy4-flash card describes). A sliding-window layer keeps a ring of `sliding_window` rows for each user
    (or `context_tokens` rows when that is fewer). A row is `kv_lora_rank + qk_rope_head_dim` values. Latent attention shares the row between heads,
    so a tensor-parallel split over heads repeats it on every die: the estimate is per die and is not divided by the die count."""
    try:
        d = mc.dims(config)
        cfg = mc.text_config(config)
    except mc.ModelConfigError:
        return None
    if not d.mla:
        return None
    full = sum(1 for t in d.layer_types if t in ("full_attention", "attention", "deepseek_sparse_attention"))
    sliding = sum(1 for t in d.layer_types if t == "sliding_attention")
    if full + sliding != d.layers:
        return None
    window = cfg.get("sliding_window")
    if sliding and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
        return None
    row = (d.mla["kv_lora_rank"] + d.mla["qk_rope_head_dim"]) * KV_BYTES_PER_VALUE
    ring = min(context_tokens, window) if sliding else 0
    return {"bytes": full * row * context_tokens + sliding * row * ring * users, "label": KV_ESTIMATE_LABEL, "row_bytes": row,
            "full_layers": full, "sliding_layers": sliding, "window": window if sliding else None, "context_tokens": context_tokens, "users": users}


def parse_skew(text: str) -> tuple:
    """Points "fraction:share,..." (the busiest `fraction` of the experts take `share` of the picks) as a tuple of (fraction, share).
    A router cannot take less than its uniform share, a larger set of experts cannot take less, and the fractions must increase."""
    pts = []
    for part in (text or "").split(","):
        try:
            frac, share = (float(x) for x in part.split(":"))   # finite: ok the argument is a string, and the range check below refuses inf and nan
        except ValueError as exc:
            raise TierPlanError(f"a skew point is FRACTION:SHARE, for example 0.25:0.61; got {part!r}") from exc
        if not (0 < frac < 1 and 0 < share <= 1):
            raise TierPlanError(f"skew point {part!r}: the fraction must be between 0 and 1 and the share above 0 and at most 1")
        if share < frac:
            raise TierPlanError(f"skew point {part!r}: the busiest {frac:.0%} of the experts cannot take less than {frac:.0%} of the picks")
        if pts and (frac <= pts[-1][0] or share < pts[-1][1]):
            raise TierPlanError("skew points must have rising fractions and shares that do not fall")
        pts.append((frac, share))
    if not pts:
        raise TierPlanError("a skew needs at least one FRACTION:SHARE point")
    return tuple(pts)


def hit_share(fraction: float, points) -> float:
    """The share of expert picks that land on the busiest `fraction` of the experts, by straight lines through the points, (0, 0) and (1, 1).
    With no points the router is uniform and the share equals the fraction."""
    if not points:
        return fraction
    xs = [(0.0, 0.0), *points, (1.0, 1.0)]
    for (x0, y0), (x1, y1) in zip(xs, xs[1:]):
        if fraction <= x1:
            return y0 + (y1 - y0) * (fraction - x0) / (x1 - x0)
    return 1.0


def place(s: Shape, other_fmt: str, expert_fmt: str, chips: int, budget: float, *, host_margin_gib: float = HOST_MARGIN_GIB,
          host_free_gib: float | None = None, skew=OLMOE_SKEW, skew_source: str | None = None,
          usable_bytes: float | None = None, overhead_bytes: float | None = None) -> dict:
    """One row: the first placement that fits for this precision and chip count, or the reason none does. `budget` is the weight bytes one chip may use.

    `usable_bytes` is the share of one chip's DRAM that weights could use before the reserve and the KV share come off, and `overhead_bytes` is that
    reserve plus the KV share (what the budget lost). With both, a row that does not fit on the chips only because of that overhead says how large the
    overhead could be for it to fit (`overhead_limit_for_chips_bytes`). The limit changes no verdict; it shows which assumption decides the row."""
    other_b, expert_b = BYTES_PER_ELEMENT[other_fmt], BYTES_PER_ELEMENT[expert_fmt]
    always = s.always_params * other_b
    experts = s.expert_params_total * expert_b
    index = s.expert_index_params * expert_b
    row = {"other_format": other_fmt, "expert_format": expert_fmt, "chips": chips, "budget_bytes_per_chip": budget,
           "always_resident_bytes": always, "expert_bytes_total": experts, "bytes_per_expert_index": index,
           "resident_experts_per_layer": 0, "host_ram_gib": 0.0, "hit_share": None, "hit_share_label": None, "host_expert_picks_per_token": 0.0,
           "per_chip_bytes": None, "placement": None, "verdict": "does not fit", "why": None, "whole_layers": None,
           "tensor_parallel_note": None, "tensor_parallel_blocked": None,
           "assumed_overhead_bytes": overhead_bytes, "overhead_limit_for_chips_bytes": None}
    if usable_bytes is not None and overhead_bytes is not None:
        # the least one chip must hold when everything stays on the chips: replicated, or the experts split (never more than replicated)
        need = always + experts / chips if chips > 1 else always + experts
        if budget < need <= usable_bytes:
            row["overhead_limit_for_chips_bytes"] = usable_bytes - need

    def on_chips(placement: str, per_chip: float) -> dict:
        row.update(verdict="fits on chips", placement=placement, per_chip_bytes=per_chip, resident_experts_per_layer=s.experts, hit_share=1.0,
                   overhead_limit_for_chips_bytes=None)            # a row that fits by any layout has nothing to loosen
        return row

    if always + experts <= budget:
        return on_chips("replicated", always + experts)
    if always + experts / chips <= budget:
        return on_chips("experts split over the chips", always + experts / chips)
    if chips > 1:
        blocked = tp_blockers(s, chips)
        if blocked:
            row["tensor_parallel_blocked"] = (f"a tensor-parallel layout over {chips} dies needs these counts divisible by {chips} (the expert width in "
                                              f"whole {TILE}-wide tiles per die): " + ", ".join(blocked))
        else:
            # attention, dense and shared MLPs, the output head and the experts are divided over the dies; routers and the embedding table are not
            divided = (s.attention_params + s.dense_mlp_params + s.shared_params + s.head_params) * other_b
            per_die = (divided + experts) / chips + (s.router_params + s.embedding_params) * other_b
            if per_die <= budget:
                row["tensor_parallel_note"] = (f"tensor-parallel is a layout the plan assumes, after the published Inkling-2-Mini bundle (raahemnabeel/inkling-2-mini): attention "
                                               f"heads, key-value heads, the expert intermediate width and the output head are divided over the {chips} dies, so "
                                               f"each holds about 1/{chips} of the weights; routers and the embedding table stay on every die. It needs the "
                                               f"head counts and the expert width divisible by {chips} (they are); nothing here has been run")
                return on_chips(f"tensor-parallel: about 1/{chips} of the weights per die", per_die)
    if always > budget:
        row["why"] = (f"the always-resident weights (attention, routers, shared experts, embedding and head) need {always / GIB:.2f} GiB on every chip "
                      f"and a chip offers {budget / GIB:.2f} GiB for weights")
        return row
    kept = min(s.experts, int(chips * (budget - always) // index))
    if kept < 1:
        row["why"] = (f"the chips have {chips * (budget - always) / GIB:.2f} GiB left beside the always-resident weights and one expert "
                      f"(its index in every layer) takes {index / GIB:.2f} GiB")
        return row
    host_gib = (experts + host_margin_gib * GIB) / GIB
    per_layer = s.experts * s.expert_params * expert_b
    layers_on = int(chips * (budget - always) // per_layer)      # the chips hold the experts of this many whole layers (fewer than all: else the split row fit)
    wl_host = ((s.moe_layers - layers_on) * per_layer + host_margin_gib * GIB) / GIB
    wl = {"layers_on_chips": layers_on, "moe_layers": s.moe_layers, "host_ram_gib": wl_host, "hit_share": layers_on / s.moe_layers,
          "hit_share_label": "by construction: every layer takes the same number of picks per token, so the share of picks on the chips is the share of layers on the chips",
          "fits_host_ram": host_free_gib is None or wl_host <= host_free_gib}
    uniform_fits = host_free_gib is None or host_gib <= host_free_gib
    if not (uniform_fits or wl["fits_host_ram"]):
        row["why"] = (f"the experts and the margin need {wl_host:.1f} GiB of host memory at the least and {host_free_gib:.1f} GiB is free")
        return row
    share = hit_share(kept / s.experts, skew)
    row.update(whole_layers=wl, uniform_fits_host_ram=uniform_fits, verdict="fits with host experts", placement="hot experts on chips, the rest in host memory", resident_experts_per_layer=kept,
               per_chip_bytes=always + kept * index / chips, host_ram_gib=host_gib, hit_share=share,
               hit_share_label=(OLMOE_LABEL if skew_source is None else f"estimate: {skew_source}"),
               host_expert_picks_per_token=s.top_k * s.moe_layers * (1 - share))
    return row


# ---- reading the source ------------------------------------------------------------------------------------------------

MAX_INDEX_BYTES = 32 << 20        # a weight index lists every tensor; the largest real ones are a few megabytes
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
REVISION_RE = re.compile(r"main|[0-9a-f]{40}")


class HubReader(drop_check.HubClient):
    """Reads `config.json` and the weight index of any public Hugging Face repository, with no token. `drop_check.HubClient` is limited to the
    organisations that command checks; this one takes any organisation and keeps the rest of its protections: the answer must end on the Hub or
    its content hosts, the size is capped, and only these two file names can be asked for, so no weight file can be requested through it."""

    def get(self, repo, revision, name, *, max_bytes):
        if not isinstance(repo, str) or not drop_check.REPO_RE.fullmatch(repo) or ".." in repo:
            raise ValueError("the repository id must be org/name with letters, digits, '.', '_' and '-' only")
        if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
            raise ValueError("the revision must be main or a 40-character commit hash")
        if name not in (CONFIG_NAME, INDEX_NAME):
            raise ValueError(f"only {CONFIG_NAME} and {INDEX_NAME} are read; asked for {name!r}")
        return self._open_same_repo(f"{self.base}/{repo}/resolve/{revision}/{name}", repo, max_bytes)

    def _open_same_repo(self, url: str, repo: str, max_bytes: int) -> bytes | None:
        """`drop_check.HubClient._open` with its redirect rule replaced: after a redirect the answer must come from the Hub's own content hosts, or
        from huggingface.co under this same repository (the Hub answers `resolve/` with a redirect to `api/resolve-cache/models/<repo>/<commit>/`)."""
        req = urllib.request.Request(url, headers={"User-Agent": drop_check.USER_AGENT})
        try:
            with self.opener(req, timeout=drop_check.TIMEOUT_S) as resp:
                problem = redirect_problem_for(repo, url, resp.geturl())     # checked before the body is read
                if problem:
                    raise drop_check.RedirectRefused(f"{url.rsplit('/', 1)[-1]}: {problem}; nothing from it was read")
                data = resp.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                return None
            raise drop_check.HubUnreachable(f"the Hub answered HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise drop_check.HubUnreachable(f"the Hub did not answer ({type(exc).__name__})") from exc
        if len(data) > max_bytes:
            raise drop_check.TooLarge(f"{url.rsplit('/', 1)[-1]} is larger than {max_bytes} bytes")
        return data


def redirect_problem_for(repo: str, asked: str, final) -> str | None:
    """None when a request for `asked` (a file of `repo`) that ended on `final` may be read; else why not. No redirect is accepted. After one the
    URL must be https on the default port with no user name, on a content host of the Hub, or on huggingface.co under this same repository."""
    if final == asked:
        return None
    if not isinstance(final, str):
        return "the response did not say which URL it came from"
    try:
        u = urllib.parse.urlsplit(final)
        host, port = (u.hostname or "").lower(), u.port
    except ValueError:
        return "the response came from a URL that cannot be parsed"
    if u.scheme != "https" or port not in (None, 443) or u.username is not None or u.password is not None:
        return f"the request was redirected to {u.scheme}://{host or '?'}{u.path}, which is not a plain https URL"
    if host == drop_check.CONTENT_DOMAIN or host.endswith("." + drop_check.CONTENT_DOMAIN):
        return None
    if host != drop_check.HUB_HOST:
        return f"the request was redirected to {host or '?'}, outside the Hub and its content hosts"
    if u.path.startswith((f"/{repo}/", f"/api/resolve-cache/models/{repo}/")):
        return None
    return f"the request was redirected to {host}{u.path}, which is not under {repo}"


def _index_total(raw) -> int | None:
    """`metadata.total_size` of a weight index, or None when there is no usable index."""
    try:
        idx = safeio.parse_json_bytes(raw, max_bytes=MAX_INDEX_BYTES, expect=dict)
    except safeio.SafeReadError:
        return None
    total = (idx.get("metadata") or {}).get("total_size") if isinstance(idx.get("metadata"), dict) else None
    return total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else None


def load(source: str, *, client=None, revision: str = "main") -> tuple[dict, int | None, str]:
    """(config, index total_size or None, where it came from). A path that exists is read as a path; otherwise `source` must be an org/name
    repository id, whose config.json and weight index are fetched. Raises TierPlanError with one plain sentence."""
    path = Path(source)
    if path.is_file() or path.is_dir():
        cfg_path = path / CONFIG_NAME if path.is_dir() else path
        try:
            cfg = mc.read_model_config(cfg_path)
        except mc.ModelConfigError as exc:
            raise TierPlanError(str(exc)) from exc
        total = None
        if path.is_dir():
            try:
                total = _index_total(safeio.read_bytes(path / INDEX_NAME, max_bytes=MAX_INDEX_BYTES))
            except safeio.SafeReadError:
                total = None
        return cfg, total, f"local {cfg_path.name}"
    if not drop_check.REPO_RE.fullmatch(source) or ".." in source:
        raise TierPlanError(f"{source!r} is not a file, a directory or a Hugging Face repository id (org/name)")
    client = client or HubReader()
    try:
        raw_cfg = client.get(source, revision, CONFIG_NAME, max_bytes=mc.MAX_CONFIG_BYTES)
        if raw_cfg is None:
            raise TierPlanError(f"{source} has no config.json that can be read (the repository is missing, private or gated)")
        raw_idx = client.get(source, revision, INDEX_NAME, max_bytes=MAX_INDEX_BYTES)
    except (drop_check.Unreadable, drop_check.HubUnreachable, ValueError) as exc:
        raise TierPlanError(str(exc)) from exc
    try:
        cfg = safeio.parse_json_bytes(raw_cfg, max_bytes=mc.MAX_CONFIG_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        raise TierPlanError(f"the config.json of {source} is not usable: {exc.reason}") from exc
    return cfg, (_index_total(raw_idx) if raw_idx is not None else None), f"{source} at {revision} (config.json and weight index only)"


def checkpoint_note(s: Shape, index_total: int | None) -> str | None:
    """A sentence when the weight index's total size is far from the config count priced in bf16; None when it agrees or there is no index."""
    if index_total is None:
        return None
    bf16 = (s.always_params + s.expert_params_total) * BYTES_PER_ELEMENT["bf16"]
    diff = index_total / bf16 - 1
    if abs(diff) <= 0.10:
        return None
    return (f"the weight index says the checkpoint is {index_total / GIB:.1f} GiB and the config count in bf16 is {bf16 / GIB:.1f} GiB ({diff:+.0%}): "
            f"the checkpoint may be stored in a different format, or the count may miss tensors")


# ---- the plan ----------------------------------------------------------------------------------------------------------

def plan(config, *, chips_list, chip_gib, reserve_gib, kv_gib, host_margin_gib=HOST_MARGIN_GIB, host_free_gib=None, skew=OLMOE_SKEW,
         skew_source=None, scores=None, rule=None, flags=frozenset(), index_total=None, where=None, kv_estimate=None) -> dict:
    """The rows for every precision combination and chip count, with each input and where it came from. `flags` names the inputs the caller set.
    `kv_estimate` is a result of `kv_estimate()`: when given, each chip loses that whole estimate (the latent cache is the same on every die) in place of its
    share of the flat `kv_gib` allowance."""
    s = shape(config)
    rule = rule or weights_plan.DEFAULT_RULE

    def entry(name, value, constant_note):
        return {"value": value, "source": "caller's flag" if name in flags else "assumed constant", "note": constant_note}

    inputs = {
        "parameters_per_group": {"value": {"attention": s.attention_params, "dense_mlp": s.dense_mlp_params, "router": s.router_params,
                                           "shared_experts": s.shared_params, "embedding": s.embedding_params, "head": s.head_params,
                                           "routed_experts_total": s.expert_params_total}, "source": "config-derived",
                                 "note": "counted by orchard/decode_floor.py from the config; norms, biases and rotary tables are not counted"},
        "bytes_per_element": {"value": dict(BYTES_PER_ELEMENT), "source": "assumed constant", "note": "orchard/plan.py: one shared exponent byte per 16 values"},
        "chip_gib": entry("chip_gib", chip_gib, "DRAM per chip as the cards and run logs state it (orchard/defaults.py)"),
        "usable_fraction": {"value": USABLE_FRACTION, "source": "assumed constant", "note": "probe 14: 29.9 GiB usable of 32 (orchard/plan.py)"},
        "reserve_gib": entry("reserve_gib", reserve_gib, "per chip: trace regions, program binaries and circular buffers (orchard/defaults.py)"),
        "kv_gib": ({"value": kv_estimate["bytes"] / GIB, "source": "estimate from the config", "note": kv_estimate["label"]} if kv_estimate else
                   entry("kv_gib", kv_gib, "KV cache and state for the whole model, spread over the chips (orchard/defaults.py)")),
        "host_margin_gib": entry("host_margin_gib", host_margin_gib, "host memory kept free beside the pinned experts (the issue text: about 10 GB)"),
        "expert_pick_skew": {"value": [list(p) for p in skew], "source": "caller's flag" if skew_source else "estimate",
                             "note": skew_source or OLMOE_LABEL},
    }
    if host_free_gib is not None:
        inputs["host_free_gib"] = {"value": host_free_gib, "source": "caller's flag", "note": "the free host memory the caller says there is"}
    rows = []
    for chips in chips_list:
        kv_here = kv_estimate["bytes"] / GIB if kv_estimate else None          # per chip, not divided: the latent cache is repeated on every die
        budget = budget_bytes(chip_gib, chips=chips, reserve_gib=reserve_gib, kv_gib=kv_gib, kv_per_chip_gib=kv_here)
        usable = chip_gib * USABLE_FRACTION * GIB
        overhead = (reserve_gib + (kv_gib / chips if kv_here is None else kv_here)) * GIB
        for name, other_fmt, expert_fmt in weights_plan.COMBINATIONS:
            r = place(s, other_fmt, expert_fmt, chips, budget, host_margin_gib=host_margin_gib, host_free_gib=host_free_gib, skew=skew,
                      skew_source=skew_source, usable_bytes=usable, overhead_bytes=overhead)
            r["name"] = name
            r["format_gate"], r["format_gate_reason"] = weights_plan.row_gate(other_fmt, expert_fmt, scores, rule)
            rows.append(r)
    notes = [n for n in (checkpoint_note(s, index_total),) if n]
    cav = caveats(config)
    out = {"source": where, "caveats": cav, "always_resident_is_lower_bound": bool(cav), "model": {"layers": s.layers, "moe_layers": s.moe_layers, "routed_experts_per_layer": s.experts,
                                       "experts_per_token": s.top_k, "always_resident_params": s.always_params,
                                       "routed_expert_params_total": s.expert_params_total, "params_per_expert_index": s.expert_index_params},
            "inputs": inputs, "fidelity_rule": rule.as_dict(), "rows": rows, "notes": notes}
    if kv_estimate:
        out["kv_estimate"] = dict(kv_estimate)
    return out


def render(res: dict) -> str:
    m = res["model"]
    lines = [f"{res['source']}: {m['layers']} layers, {m['moe_layers']} with {m['routed_experts_per_layer']} routed experts (top {m['experts_per_token']}); "
             f"{m['always_resident_params'] / 1e9:.2f} B always-resident and {m['routed_expert_params_total'] / 1e9:.2f} B routed-expert parameters [config-derived]"]
    if res.get("caveats"):
        lines.append("the always-resident figure is a lower bound: it leaves out " + "; ".join(res["caveats"]) + ". Every row below is optimistic by that much.")
    i = res["inputs"]
    if res.get("kv_estimate"):
        k = res["kv_estimate"]
        kv_text = (f"{i['kv_gib']['value']:.2f} GiB KV cache on each chip [estimate from the config for {k['context_tokens']:,} tokens and {k['users']} user(s): "
                   f"latent rows only, the sparse-attention indexer's keys are not counted]")
    else:
        kv_text = f"{i['kv_gib']['value']} GiB KV cache over the chips"
    lines.append(f"chip DRAM {i['chip_gib']['value']} GiB, {i['usable_fraction']['value']:.1%} usable, {i['reserve_gib']['value']} GiB reserve per chip, "
                 f"{kv_text}, host margin {i['host_margin_gib']['value']} GiB [assumed constant unless a flag was given; "
                 f"the sources are in --json]")
    rule = res.get("fidelity_rule") or {}
    if rule.get("name") not in (None, weights_plan.DEFAULT_RULE.name):
        lines.append(f"fidelity rule: {rule['name']} (ratio bar {weights_plan.FIDELITY_BAR}" + "".join(
            f"; {label} {rule[key]}" for key, label in (("kl_max", "KL at most"), ("top1_min", "top-1 at least"), ("abs_dppl_percent_max", "|perplexity change| percent at most"))
            if rule.get(key) is not None) + ")")
    lines += ["", f"{'combination':21} {'chips':>5}  {'verdict':23} {'GiB/chip':>8} {'experts on chips':>17} {'host GiB':>9} {'hit share':>10} {'host picks/token':>17}  fidelity rule"]
    for r in res["rows"]:
        fits_host = r["verdict"] == "fits with host experts"
        per_chip = f"{r['per_chip_bytes'] / GIB:.2f}" if r["per_chip_bytes"] else "-"
        resident = f"{r['resident_experts_per_layer']} of {m['routed_experts_per_layer']}" if r["verdict"] != "does not fit" else "-"
        host_gib = (f"{r['host_ram_gib']:.1f}" + ("" if r.get("uniform_fits_host_ram", True) else "!")) if r["host_ram_gib"] else "-"
        hit = f"{r['hit_share']:.0%}" if fits_host else "-"
        picks = f"{r['host_expert_picks_per_token']:.1f}" if fits_host else "-"
        lines.append(f"{r['name']:21} {r['chips']:>5}  {r['verdict']:23} {per_chip:>8} {resident:>17} {host_gib:>9} {hit:>10} {picks:>17}  {r['format_gate']}")
    if any(not r.get("uniform_fits_host_ram", True) for r in res["rows"]):
        lines.append("! more than the free host memory in this layout; the whole-layer layout below asks for less")
    wl_rows = [r for r in res["rows"] if r["whole_layers"]]
    if wl_rows:
        lines += ["", "whole layers on chips instead (the layout of the published hybrid bundles; host copies of the other layers only):",
                  f"{'combination':21} {'chips':>5}  {'whole layers on chips':>22} {'host GiB':>9} {'hit share':>10}"]
        for r in wl_rows:
            w = r["whole_layers"]
            lines.append(f"{r['name']:21} {r['chips']:>5}  {str(w['layers_on_chips']) + ' of ' + str(w['moe_layers']):>22} {w['host_ram_gib']:>9.1f} {w['hit_share']:>10.0%}"
                         + ("" if w["fits_host_ram"] else "  (more than the free host memory)"))
        lines.append("whole-layer hit share: " + wl_rows[0]["whole_layers"]["hit_share_label"])
    label = next((r["hit_share_label"] for r in res["rows"] if r["hit_share_label"]), None)
    lines.append("")
    if label:
        lines.append(f"hit share: {label}")
    for r in res["rows"]:
        if r["why"]:
            lines.append(f"{r['name']}, {r['chips']} chip(s): {r['why']}")
    for r in res["rows"]:
        if r.get("tensor_parallel_note"):
            lines.append(f"{r['name']}, {r['chips']} chip(s): {r['tensor_parallel_note']}")
        elif r.get("tensor_parallel_blocked") and r["verdict"] != "fits on chips":
            lines.append(f"{r['name']}, {r['chips']} chip(s): {r['tensor_parallel_blocked']}")
    for r in res["rows"]:
        if r.get("overhead_limit_for_chips_bytes") is not None:
            lines.append(f"{r['name']}, {r['chips']} chip(s): would fit on the chips if the reserve plus this chip's share of the KV cache were at most "
                         f"{r['overhead_limit_for_chips_bytes'] / GIB:.1f} GiB per chip; the plan assumes {r['assumed_overhead_bytes'] / GIB:.1f} GiB "
                         f"(--reserve-gib and --kv-gib set them). The KV allowance is a flat figure for every model, so a model with mostly sliding-window "
                         f"attention may need much less")
    lines += [f"note: {n}" for n in res["notes"]]
    lines.append("the weights fit or they do not; speed and quality are not measured here, and the host path for experts is not written (issue 739, step 3)")
    return "\n".join(lines)


def _chips(text: str) -> tuple:
    try:
        vals = tuple(int(x) for x in text.split(","))
    except ValueError as exc:
        raise TierPlanError("--chips is a list of whole numbers such as 1,2,4") from exc
    if not vals or any(not 1 <= v <= 64 for v in vals):
        raise TierPlanError("--chips values must be 1 to 64")
    return vals


def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard tier-plan", add_help=add_help,
                                description="say what precision and expert placement a mixture-of-experts model needs on 1, 2 or 4 chips, from its config alone "
                                            "(reads config.json and the weight index; downloads no weight)")
    p.add_argument("source", help="a Hugging Face repository id (org/name), a config.json, or a directory holding one; a path that exists is read as a path")
    p.add_argument("--revision", default="main", help="the repository revision: main or a commit hash (default main)")
    p.add_argument("--chips", default="1,2,4", help="chip counts to plan for, comma separated (default 1,2,4)")
    p.add_argument("--chip-gib", type=float, help=f"DRAM per chip in GiB (default {defaults.CHIP_DRAM_GIB})")
    p.add_argument("--reserve-gib", type=float, help=f"per-chip reserve in GiB (default {defaults.CHIP_RESERVE_GIB})")
    p.add_argument("--kv-gib", type=float, help=f"KV cache allowance for the whole model in GiB, spread over the chips (default {defaults.KV_BUDGET_GIB})")
    p.add_argument("--context", type=int, metavar="TOKENS",
                   help="plan for this many tokens of context held across all users: for a model with multi-latent attention the KV allowance then comes from the "
                        "config (an estimate, the same on every die) in place of the flat --kv-gib figure; other models keep the flat figure")
    p.add_argument("--users", type=int, metavar="N", help="with --context: how many users share it (each sliding-window layer keeps a ring per user; default 1)")
    p.add_argument("--host-margin-gib", type=float, help=f"host memory kept free beside the pinned experts (default {HOST_MARGIN_GIB})")
    p.add_argument("--host-free-gib", type=float, help="free host memory in GiB; a row that needs more is a no (default: not checked)")
    p.add_argument("--skew", metavar="FRACTION:SHARE,...",
                   help="expert-use skew measured on this model: the busiest FRACTION of the experts take SHARE of the picks, for example 0.10:0.41,0.25:0.61 "
                        "(default: the OLMoE-1B-7B figures, an estimate that is not measured for the model being planned)")
    p.add_argument("--skew-source", help="where the --skew figures came from, printed beside the hit share")
    weights_plan.add_fidelity_arguments(p)
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def _positive(name: str, v, *, zero_ok: bool = False):
    if v is not None and not (safeio.is_finite_number(v) and (v >= 0 if zero_ok else v > 0)):
        raise TierPlanError(f"{name} must be a finite number {'of zero or more' if zero_ok else 'above zero'}")


def _context_and_users(args) -> tuple:
    """The `--context` and `--users` values as whole numbers (None, 1 when not given), or TierPlanError: a context with an explicit `--kv-gib`, users with
    no context, and a value that is not a whole number from 1 up are all refused."""
    ctx, users = getattr(args, "context", None), getattr(args, "users", None)
    if users is not None and ctx is None:
        raise TierPlanError("--users needs --context (it says how many users share that context)")
    if ctx is not None and args.kv_gib is not None:
        raise TierPlanError("--context and --kv-gib both set the KV allowance: give one of them")
    for flag, v, hi in (("--context", ctx, MAX_CONTEXT_TOKENS), ("--users", users, MAX_USERS)):
        if v is not None:
            try:
                safeio.whole_number(v, lo=1, hi=hi, what=flag)
            except safeio.BadValue as exc:
                raise TierPlanError(f"{flag} must be a whole number from 1 to {hi}") from exc
    return ctx, 1 if users is None else users


def main(argv=None, out=None, client=None) -> int:
    return run(parser().parse_args(argv), out, client)


def run(args, out=None, client=None) -> int:
    """Run the command for parsed arguments (`tt-orchard tier-plan` parses them with the shared options). Returns 0, or 2 with one line on stderr."""
    out = out or sys.stdout
    try:
        chips = _chips(args.chips)
        _positive("--chip-gib", args.chip_gib)
        _positive("--reserve-gib", args.reserve_gib, zero_ok=True)
        _positive("--kv-gib", args.kv_gib, zero_ok=True)
        _positive("--host-margin-gib", args.host_margin_gib, zero_ok=True)
        _positive("--host-free-gib", args.host_free_gib)
        context, users = _context_and_users(args)
        skew = parse_skew(args.skew) if args.skew is not None else OLMOE_SKEW
        try:
            scores, rule = weights_plan.fidelity_inputs(args)
        except weights_plan.FidelityInputError as exc:
            raise TierPlanError(f"fidelity input: {exc}") from exc
        flags = {n for n, v in (("chip_gib", args.chip_gib), ("reserve_gib", args.reserve_gib), ("kv_gib", args.kv_gib),
                                ("host_margin_gib", args.host_margin_gib)) if v is not None}
        cfg, total, where = load(args.source, client=client, revision=args.revision)
        est = kv_estimate(cfg, context, users) if context is not None else None
        res = plan(cfg, chips_list=chips, chip_gib=args.chip_gib or defaults.CHIP_DRAM_GIB,
                   reserve_gib=defaults.CHIP_RESERVE_GIB if args.reserve_gib is None else args.reserve_gib,
                   kv_gib=defaults.KV_BUDGET_GIB if args.kv_gib is None else args.kv_gib,
                   host_margin_gib=HOST_MARGIN_GIB if args.host_margin_gib is None else args.host_margin_gib, host_free_gib=args.host_free_gib,
                   skew=skew, skew_source=(args.skew_source or "the figures the caller gave with --skew") if args.skew is not None else None,
                   scores=scores, rule=rule, flags=flags, index_total=total, where=where, kv_estimate=est)
        if context is not None and est is None:
            res["notes"].append("no model-aware KV estimate for this config (it has no multi-latent attention, or its layer layout cannot be read), so the flat "
                                f"{defaults.KV_BUDGET_GIB} GiB allowance stays and --context changed nothing")
    except TierPlanError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2) if args.json else render(res), file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
