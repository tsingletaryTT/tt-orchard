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
            router += d.hidden * d.routed_experts
            shared += df._mlp_params(d, d.shared_intermediate) if d.shared_intermediate else 0
        else:
            dense += df._mlp_params(d, d.intermediate)
    table = d.hidden * d.vocab
    expert = df._mlp_params(d, d.moe_intermediate)
    for name, n in (("attention", attention), ("dense MLP", dense), ("router", router), ("shared experts", shared), ("embedding table", table),
                    ("routed experts", d.routed_experts * moe * expert)):
        if n > MAX_GROUP_PARAMS:
            raise TierPlanError(f"the config describes more than {MAX_GROUP_PARAMS:.0e} parameters in the {name}; no real model is that large")
    return Shape(attention_params=attention, dense_mlp_params=dense, router_params=router, shared_params=shared, embedding_params=table,
                 head_params=0 if d.tied_embeddings else table, experts=d.routed_experts, top_k=d.top_k, moe_layers=moe,
                 expert_params=expert, layers=d.layers)


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
    if has("mtp_num_hidden_layers", "num_nextn_predict_layers"):
        out.append("a multi-token-prediction layer (mtp keys): its weights are not counted")
    if top.get("vision_config") or cfg.get("vision_config"):
        out.append("a vision tower (vision_config): only the language model is counted")
    return out


def budget_bytes(chip_gib: float, *, chips: int, reserve_gib: float, kv_gib: float) -> float:
    """Bytes of one chip that weights may use: the usable share of its DRAM, less the per-chip reserve and this chip's share of the KV cache
    allowance. Never negative."""
    return max(0.0, (chip_gib * USABLE_FRACTION - reserve_gib - kv_gib / chips) * GIB)


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
          host_free_gib: float | None = None, skew=OLMOE_SKEW, skew_source: str | None = None) -> dict:
    """One row: the first placement that fits for this precision and chip count, or the reason none does. `budget` is the weight bytes one chip may use."""
    other_b, expert_b = BYTES_PER_ELEMENT[other_fmt], BYTES_PER_ELEMENT[expert_fmt]
    always = s.always_params * other_b
    experts = s.expert_params_total * expert_b
    index = s.expert_index_params * expert_b
    row = {"other_format": other_fmt, "expert_format": expert_fmt, "chips": chips, "budget_bytes_per_chip": budget,
           "always_resident_bytes": always, "expert_bytes_total": experts, "bytes_per_expert_index": index,
           "resident_experts_per_layer": 0, "host_ram_gib": 0.0, "hit_share": None, "hit_share_label": None, "host_expert_picks_per_token": 0.0,
           "per_chip_bytes": None, "placement": None, "verdict": "does not fit", "why": None, "whole_layers": None}

    def on_chips(placement: str, per_chip: float) -> dict:
        row.update(verdict="fits on chips", placement=placement, per_chip_bytes=per_chip, resident_experts_per_layer=s.experts, hit_share=1.0)
        return row

    if always + experts <= budget:
        return on_chips("replicated", always + experts)
    if always + experts / chips <= budget:
        return on_chips("experts split over the chips", always + experts / chips)
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
         skew_source=None, scores=None, flags=frozenset(), index_total=None, where=None) -> dict:
    """The rows for every precision combination and chip count, with each input and where it came from. `flags` names the inputs the caller set."""
    s = shape(config)

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
        "kv_gib": entry("kv_gib", kv_gib, "KV cache and state for the whole model, spread over the chips (orchard/defaults.py)"),
        "host_margin_gib": entry("host_margin_gib", host_margin_gib, "host memory kept free beside the pinned experts (the issue text: about 10 GB)"),
        "expert_pick_skew": {"value": [list(p) for p in skew], "source": "caller's flag" if skew_source else "estimate",
                             "note": skew_source or OLMOE_LABEL},
    }
    if host_free_gib is not None:
        inputs["host_free_gib"] = {"value": host_free_gib, "source": "caller's flag", "note": "the free host memory the caller says there is"}
    rows = []
    for chips in chips_list:
        budget = budget_bytes(chip_gib, chips=chips, reserve_gib=reserve_gib, kv_gib=kv_gib)
        for name, other_fmt, expert_fmt in weights_plan.COMBINATIONS:
            r = place(s, other_fmt, expert_fmt, chips, budget, host_margin_gib=host_margin_gib, host_free_gib=host_free_gib, skew=skew,
                      skew_source=skew_source)
            r["name"] = name
            r["format_gate"] = weights_plan._worst(weights_plan.format_gate(other_fmt, scores), weights_plan.format_gate(expert_fmt, scores))
            rows.append(r)
    notes = [n for n in (checkpoint_note(s, index_total),) if n]
    cav = caveats(config)
    return {"source": where, "caveats": cav, "always_resident_is_lower_bound": bool(cav), "model": {"layers": s.layers, "moe_layers": s.moe_layers, "routed_experts_per_layer": s.experts,
                                       "experts_per_token": s.top_k, "always_resident_params": s.always_params,
                                       "routed_expert_params_total": s.expert_params_total, "params_per_expert_index": s.expert_index_params},
            "inputs": inputs, "rows": rows, "notes": notes}


def render(res: dict) -> str:
    m = res["model"]
    lines = [f"{res['source']}: {m['layers']} layers, {m['moe_layers']} with {m['routed_experts_per_layer']} routed experts (top {m['experts_per_token']}); "
             f"{m['always_resident_params'] / 1e9:.2f} B always-resident and {m['routed_expert_params_total'] / 1e9:.2f} B routed-expert parameters [config-derived]"]
    if res.get("caveats"):
        lines.append("the always-resident figure is a lower bound: it leaves out " + "; ".join(res["caveats"]) + ". Every row below is optimistic by that much.")
    i = res["inputs"]
    lines.append(f"chip DRAM {i['chip_gib']['value']} GiB, {i['usable_fraction']['value']:.1%} usable, {i['reserve_gib']['value']} GiB reserve per chip, "
                 f"{i['kv_gib']['value']} GiB KV cache over the chips, host margin {i['host_margin_gib']['value']} GiB [assumed constant unless a flag was given; "
                 f"the sources are in --json]")
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
    p.add_argument("--host-margin-gib", type=float, help=f"host memory kept free beside the pinned experts (default {HOST_MARGIN_GIB})")
    p.add_argument("--host-free-gib", type=float, help="free host memory in GiB; a row that needs more is a no (default: not checked)")
    p.add_argument("--skew", metavar="FRACTION:SHARE,...",
                   help="expert-use skew measured on this model: the busiest FRACTION of the experts take SHARE of the picks, for example 0.10:0.41,0.25:0.61 "
                        "(default: the OLMoE-1B-7B figures, an estimate that is not measured for the model being planned)")
    p.add_argument("--skew-source", help="where the --skew figures came from, printed beside the hit share")
    p.add_argument("--fidelity-bfp4", type=float, help=f"a fidelity score measured for this model's port with bfp4 weights (the bar is {weights_plan.FIDELITY_BAR})")
    p.add_argument("--fidelity-tokens", type=int, default=0, help=f"scored tokens behind --fidelity-bfp4 (at least {weights_plan.MIN_SCORED_TOKENS} to count)")
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def _positive(name: str, v, *, zero_ok: bool = False):
    if v is not None and not (safeio.is_finite_number(v) and (v >= 0 if zero_ok else v > 0)):
        raise TierPlanError(f"{name} must be a finite number {'of zero or more' if zero_ok else 'above zero'}")


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
        skew = parse_skew(args.skew) if args.skew is not None else OLMOE_SKEW
        scores = None
        if args.fidelity_bfp4 is not None:
            if not safeio.is_finite_number(args.fidelity_bfp4) or not 0 <= args.fidelity_bfp4 <= 10.5 or args.fidelity_tokens < 0:
                raise TierPlanError("--fidelity-bfp4 must be a score from 0 to 10 and --fidelity-tokens a count")
            scores = {"bfp4": (args.fidelity_bfp4, args.fidelity_tokens)}
        flags = {n for n, v in (("chip_gib", args.chip_gib), ("reserve_gib", args.reserve_gib), ("kv_gib", args.kv_gib),
                                ("host_margin_gib", args.host_margin_gib)) if v is not None}
        cfg, total, where = load(args.source, client=client, revision=args.revision)
        res = plan(cfg, chips_list=chips, chip_gib=args.chip_gib or defaults.CHIP_DRAM_GIB,
                   reserve_gib=defaults.CHIP_RESERVE_GIB if args.reserve_gib is None else args.reserve_gib,
                   kv_gib=defaults.KV_BUDGET_GIB if args.kv_gib is None else args.kv_gib,
                   host_margin_gib=HOST_MARGIN_GIB if args.host_margin_gib is None else args.host_margin_gib, host_free_gib=args.host_free_gib,
                   skew=skew, skew_source=(args.skew_source or "the figures the caller gave with --skew") if args.skew is not None else None,
                   scores=scores, flags=flags, index_total=total, where=where)
    except TierPlanError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2) if args.json else render(res), file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
