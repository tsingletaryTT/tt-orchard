# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0's class decision on a corpus of models, from their Hub headers, set beside independent labels.

    python3 -m orchard.classify_bench collect --corpus corpus.json --base-snapshot <nearest model's snapshot> --out DIR
    python3 -m orchard.classify_bench compare --out DIR

`collect` reads each model's file list, config and safetensors headers from the Hub by range request (no weights; the research package's
fetcher: allow-listed hosts, no token, a request budget per model). The nearest model's snapshot is local, so every comparison is against the
same copy. Only a restricted-head candidate costs a download: its small sidecar file, so the template's own check can compare its rows with the
local base `lm_head`. The class is the one the stage 0 template's own functions give (`triage_rules.load()`), not a copy of its rules.

Each model gets two outputs. `tool.json` holds the class and the reasons. `packets/<slug>.md` is what a labeler reads: the evidence a person would
look at, with no verdict in it. A labeler (a person, or a Claude agent that has not seen the tool's output) writes `labels/<name>.json`, and
`compare` reports where the tool and the labelers disagree and which of them the real data supports. The labels are a second opinion. The corpus
may name a `truth` for models whose bring-up was run for real; those anchor the comparison.

What the bench cannot see: weight values (a fine-tune and a copy of the base look alike), the model's own code, and any repo the Hub will not
serve without a token. Those models are reported as skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from orchard import classes
from orchard.research import headers, triage_rules
from orchard.research.http import FetchError
from orchard.research.types import valid_id

HUB_MODEL = "https://huggingface.co/api/models/{id}?blobs=true"
RAW_URL = headers.RAW_URL
SIDECAR_DOWNLOAD_MAX = 8 * 1024 * 1024        # the fetcher's own body cap; a bigger sidecar is not downloaded
# The preflight estimator is cautious (64 shards, an 8 MiB body, 600 requests, 240 s). The bench reads further, so that "unknown" means the evidence
# decided nothing and not that a cap was hit; what it still cannot read is flagged `evidence_complete: false`.
BENCH_MAX_SHARDS = 256
BENCH_CONFIG_MAX = 64 * 1024 * 1024
BENCH_MAX_REQUESTS = 4000
BENCH_DEADLINE_S = 1200.0
MAX_EXTRA_HEADERS = 6
MAX_LISTED = 20
UNREADABLE = re.compile(r"HTTP (401|403)")
CLASSES = classes.CLASSES
QUANT_HINTS = (".qweight", ".scales", ".qzeros", ".weight_scale", ".weight_scale_2", ".weight_packed", ".input_scale", ".g_idx", ".weight_scale_inv")


class CorpusError(ValueError):
    """The corpus or a labels file is not what the bench reads."""


def slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def load_corpus(path) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorpusError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise CorpusError("the corpus must be a non-empty list of {id, stratum?, truth?, note?}")
    seen = set()
    for m in data:
        if not isinstance(m, dict) or not valid_id(m.get("id")):
            raise CorpusError(f"{m!r}: id must be a Hugging Face org/name")
        if m["id"] in seen:
            raise CorpusError(f"{m['id']} appears twice")
        seen.add(m["id"])
        if m.get("truth") is not None and m["truth"] not in CLASSES:
            raise CorpusError(f"{m['id']}: truth {m['truth']!r} is not one of {CLASSES}")
    return data


# ---- the nearest model, read once ---------------------------------------------------------------------------------------------------

def read_base(base_dir) -> dict:
    t = triage_rules.load()
    d = Path(base_dir)
    shards, tensors, errors = t.read_tensors(d)
    cfg, _ = t.read_json(d / "config.json")
    return {"dir": d, "shards": shards, "tensors": tensors, "errors": errors, "config": cfg or {},
            "files": sorted(p.name for p in d.iterdir() if p.is_file())}


# ---- one model ------------------------------------------------------------------------------------------------------------------------

def _skipped(repo: str, why: str) -> dict:
    return {"packet": {"id": repo, "status": "skipped", "why": why},
            "tool": {"class": None, "status": "skipped", "why": why, "reasons": []}}


def _hub(fetch, repo: str):
    try:
        meta = fetch.json(HUB_MODEL.format(id=repo))
    except FetchError as exc:
        return None, f"hub metadata unavailable: {getattr(exc, 'reason', exc)}"
    if not isinstance(meta, dict) or not isinstance(meta.get("siblings"), list):
        return None, "hub metadata has no file list"
    return meta, None


def _extra_weight_files(t, files: list[str], shards: list[str], base_files: list[str]) -> list[str]:
    """Weight files outside the backbone that the nearest model lacks: what stage 0's compare_sidecars would list."""
    skip = set(shards) | set(base_files)
    # Stage 0 scans the snapshot directory's own files (compare_sidecars uses iterdir), so a weights file in a subdirectory is not a sidecar.
    return [f for f in files if "/" not in f and f.endswith(t.WEIGHT_SUFFIXES) and f not in skip and not t.BACKBONE_BIN.fullmatch(f)]


def _sidecars(fetch, repo: str, t, extras: list[str], sizes: dict, backbone_names: set[str]) -> tuple[dict, dict]:
    """(the sidecar record stage 0 builds, {file: {tensor: shape}} for the packet). Header-only."""
    entries, problems, shapes = [], [], {}
    backbone = backbone_names | {t.normalize(n) for n in backbone_names}
    for name in extras:
        entry = {"file": name, "size": sizes.get(name)}
        entries.append(entry)
        if not name.endswith(".safetensors"):
            problems.append(f"sidecar {name} is not a safetensors file, so its tensors cannot be read without loading it")
            continue
        if len([e for e in entries if e["file"].endswith(".safetensors")]) > MAX_EXTRA_HEADERS:
            problems.append(f"more than {MAX_EXTRA_HEADERS} extra safetensors files; {name} was not read")
            continue
        try:
            header = headers.shard_header(fetch, repo, name)
        except (FetchError, ValueError, UnicodeDecodeError) as exc:
            problems.append(f"sidecar {name} cannot be read: {getattr(exc, 'reason', exc)}")
            continue
        shapes[name] = {k: v["shape"] for k, v in header.items()}
        clash = sorted(n for n in header if n in backbone or t.normalize(n) in backbone)
        if clash:
            problems.append(f"sidecar {name} has {len(clash)} tensor names that overlap the backbone")
    return {"sidecars": entries, "problems": problems}, shapes


def _head_candidate(base: dict, model_tensors: dict, shapes: dict) -> str | None:
    """A sidecar file with a 2-D tensor as wide as the base lm_head and shorter than it, when the model has no lm_head of its own. The rows
    must be fetched to know whether they are copies; this only says whether that is worth a download."""
    head = base["tensors"].get("lm_head.weight")
    if not head or len(head["shape"]) != 2 or any(re.search(r"(^|\.)lm_head(\.|$)", n) for n in model_tensors):
        return None
    vocab, dim = head["shape"]
    for fname, tensors in shapes.items():
        if any(len(s) == 2 and s[1] == dim and s[0] < vocab for s in tensors.values()):
            return fname
    return None


def collect_one(fetch, repo: str, base: dict) -> dict:
    """{"packet": the blind evidence, "tool": the class and why}. A model the Hub will not serve is skipped, not classified."""
    t = triage_rules.load()
    meta, why = _hub(fetch, repo)
    if meta is None:
        return _skipped(repo, why)
    sizes = {s["rfilename"]: s.get("size") for s in meta["siblings"] if isinstance(s, dict) and isinstance(s.get("rfilename"), str)}
    files = sorted(sizes)
    cfg_errors, cfg = [], None                  # None, not {}: stage 0 reads a missing config as "could not compare", an empty one as "all values differ"
    if "config.json" in files:
        try:
            cfg = fetch.json(RAW_URL.format(repo=repo, name="config.json"), max_bytes=BENCH_CONFIG_MAX)
        except FetchError as exc:
            cfg_errors.append(f"config.json: {getattr(exc, 'reason', exc)}")
        if cfg is not None and not isinstance(cfg, dict):
            cfg_errors.append("config.json is not a JSON object")
            cfg = None
    else:
        cfg_errors.append("config.json is missing")
    res = headers.read_model(fetch, repo, files, max_shards=BENCH_MAX_SHARDS)
    data = res.data
    if not data["tensors"] and any(UNREADABLE.search(e) for e in data["errors"]):
        return _skipped(repo, "the Hub will not serve this repo's files without a token: " + data["errors"][0][:120])
    sc, shapes = _sidecars(fetch, repo, t, _extra_weight_files(t, files, data["shards"], base["files"]), sizes, set(data["tensors"]))
    c = t.compare_config_dicts(cfg, base["config"], cfg_errors, "model/config.json", "base/config.json")
    tens = t.compare_tensor_maps(data["tensors"], base["tensors"], mshards=data["shards"], bshards=base["shards"], merr=data["errors"],
                                 berr=base["errors"], model_path=repo, base_path="base")
    head, downloaded = {"restricted": False, "note": "no sidecar looks like a cut-down lm_head"}, 0
    candidate = _head_candidate(base, data["tensors"], shapes)
    if candidate:
        size = sizes.get(candidate)
        if not isinstance(size, int) or size > SIDECAR_DOWNLOAD_MAX:
            head = {"restricted": None, "note": f"{candidate} is over the {SIDECAR_DOWNLOAD_MAX}-byte download cap, so its rows were not compared"}
        else:
            try:
                blob = fetch.get(RAW_URL.format(repo=repo, name=candidate), max_bytes=size + 1024)
            except FetchError as exc:
                head = {"restricted": None, "note": f"{candidate} could not be fetched: {getattr(exc, 'reason', exc)}"}
            else:
                downloaded = len(blob)
                with tempfile.TemporaryDirectory() as tmp:
                    (Path(tmp) / candidate).write_bytes(blob)
                    head = t.detect_restricted_head(Path(tmp), base["dir"], {"sidecars": [{"file": candidate}]})
    tens["head"] = head
    reasons = t.decide_path(c, tens) + sc["problems"]
    cls = t.decide_class(c, tens, reasons, sc)
    gaps = (list(data["errors"]) + [u.reason for u in res.unavailable if "no safetensors" not in u.reason]
            + [e for e in cfg_errors if e.startswith("config.json:")] + [p for p in sc["problems"] if " cannot be read:" in p])    # a failed fetch, not a gguf
    tool = {"class": cls, "status": "classified", "reasons": [str(r)[:240] for r in reasons[:6]], "sidecars": [e["file"] for e in sc["sidecars"]],
            "evidence_complete": not gaps, "incomplete_because": [g[:160] for g in gaps[:5]],
            "head": {k: head.get(k) for k in ("restricted", "rows", "base_rows", "matched_rows", "token_ids", "min_cosine", "note")},
            "downloaded_bytes": downloaded}
    packet = _packet(repo, meta, sizes, files, cfg, c, tens, data, sc, shapes, base)
    packet["evidence_gaps"] = tool["incomplete_because"]
    return {"packet": packet, "tool": tool}


# ---- the packet: evidence without a verdict -----------------------------------------------------------------------------------------

def _packet(repo, meta, sizes, files, cfg, c, tens, data, sc, shapes, base) -> dict:
    ns = tens["normalized_name_sets"]
    shard_bytes = sum(sizes.get(s) or 0 for s in data["shards"])
    names = list(data["tensors"])
    hints = Counter(h for n in names for h in QUANT_HINTS if n.endswith(h))
    card = meta.get("cardData") if isinstance(meta.get("cardData"), dict) else {}
    diffs = [{"key": k, "model": v.get("model"), "base": v.get("base")}
             for k, v in ((k, c.get("text_key_values", {}).get(k, {})) for k in c.get("decisive_keys_differing", []))]
    head = base["tensors"].get("lm_head.weight")
    return {
        "id": repo, "status": "ok", "revision": str(meta.get("sha", ""))[:12], "license": card.get("license"), "gated": meta.get("gated") or False,
        "downloads": meta.get("downloads"), "pipeline_tag": meta.get("pipeline_tag"), "tags": [str(x) for x in (meta.get("tags") or [])[:12]],
        "card_base_model": card.get("base_model"),
        "files": {"backbone_shards": len(data["shards"]), "backbone_gib": round(shard_bytes / (1 << 30), 2),
                  "other_weight_files": [{"file": e["file"], "mb": round((e["size"] or 0) / 1e6, 1)} for e in sc["sidecars"]],
                  "code_files": [f for f in files if f.endswith(".py")][:MAX_LISTED], "has_tokenizer": "tokenizer.json" in files,
                  "has_chat_template": any("chat_template" in f for f in files), "total_files": len(files)},
        "config": {"architectures": (cfg or {}).get("architectures"), "model_type": (cfg or {}).get("model_type"),
                   "could_not_compare": c.get("errors") if "text_key_values" not in c else [], "differing_text_values": diffs},
        "tensors": {"model_count": len(names), "base_count": len(base["tensors"]), "shared_by_name": tens["raw_name_sets"]["common"],
                    "shared_after_alignment": ns["common"], "aligned_rules": ns["aligned"]["rules"],
                    "only_in_model": ns["only_in_model"][:MAX_LISTED], "only_in_base": ns["only_in_base"][:MAX_LISTED],
                    "shape_differences": [f"{x['name']} {x['model']} vs {x['base']}" for x in tens["shape_diffs_normalized"][:5]],
                    "dtype_differences": [f"{x['name']} {x['model']} here, {x['base']} in the nearest model" for x in tens["dtype_diffs_normalized"][:5]],
                    "dtype_difference_count": len(tens["dtype_diffs_normalized"]),
                    "dtype_histogram": tens["model"]["dtype_histogram"], "quantization_hints": dict(hints),
                    "model_has_lm_head": any(re.search(r"(^|\.)lm_head(\.|$)", n) for n in names),
                    "base_lm_head_shape": head["shape"] if head else None, "model_mtp_tensors": tens["mtp_tensors"]["model"],
                    "base_mtp_tensors": tens["mtp_tensors"]["base"], "read_errors": data["errors"][:5]},
        "other_files": [{"file": f, "tensors": [{"name": k, "shape": v} for k, v in list(shapes.get(f, {}).items())[:6]],
                         "tensor_count": len(shapes.get(f, {}))} for f in [e["file"] for e in sc["sidecars"]]],
    }


def render_packet(p: dict) -> str:
    """The packet as the short Markdown a labeler reads."""
    if p.get("status") != "ok":
        return f"# {p['id']}\n\nNot classifiable by the bench: {p.get('why')}\n"
    f, c, t = p["files"], p["config"], p["tensors"]
    out = [f"# {p['id']}", "",
           f"revision {p['revision']} · license {p['license']} · gated {p['gated']} · {p['downloads']} downloads · pipeline {p['pipeline_tag']}",
           f"card says it is based on: {p['card_base_model']}", f"tags: {', '.join(p['tags'])}", "",
           "## Files",
           f"- backbone: {f['backbone_shards']} safetensors shards, {f['backbone_gib']} GiB; {f['total_files']} files in all; tokenizer {f['has_tokenizer']}; chat template {f['has_chat_template']}"]
    for o in f["other_weight_files"]:
        out.append(f"- other weight file: {o['file']} ({o['mb']} MB)")
    if f["code_files"]:
        out.append(f"- code files (not read): {', '.join(f['code_files'])}")
    out += ["", "## Config against the nearest model", f"- architectures {c['architectures']}, model_type {c['model_type']}"]
    if c["could_not_compare"]:
        out.append(f"- the configs could not be compared: {'; '.join(map(str, c['could_not_compare']))}")
    out.append("- text-config values that differ: " + ("; ".join(f"{d['key']} {d['model']} vs {d['base']}" for d in c["differing_text_values"]) or "none of the values that decide whether the nearest runtime can load it"))
    out += ["", "## Tensors against the nearest model",
            f"- this model has {t['model_count']} tensors, the nearest has {t['base_count']}; {t['shared_by_name']} names match as written, {t['shared_after_alignment']} after pairing names that differ only by a prefix"]
    for r in t["aligned_rules"]:
        out.append(f"- prefix pairing: names starting `{r['model_prefix'] or '(none)'}` here are `{r['base_prefix'] or '(none)'}` there ({r['tensors']} tensors, {r['same_shape']} with the same shape and dtype)")
    out.append(f"- only in this model: {', '.join(t['only_in_model']) or 'none'}")
    out.append(f"- only in the nearest model: {', '.join(t['only_in_base']) or 'none'}")
    if t["shape_differences"]:
        out.append("- shape differences: " + "; ".join(t["shape_differences"]))
    if t["dtype_difference_count"]:
        out.append(f"- {t['dtype_difference_count']} tensors differ in dtype from the nearest model, for example: " + "; ".join(t["dtype_differences"]))
    out.append(f"- dtypes: {t['dtype_histogram']}; quantization-style tensor names: {t['quantization_hints'] or 'none'}")
    out.append(f"- has its own lm_head: {t['model_has_lm_head']}; the nearest model's lm_head shape: {t['base_lm_head_shape']}; mtp tensors here {t['model_mtp_tensors']}, there {t['base_mtp_tensors']}")
    if t["read_errors"]:
        out.append(f"- header read problems: {'; '.join(t['read_errors'])}")
    if p.get("evidence_gaps"):
        out.append(f"- NOTE the evidence is incomplete: {'; '.join(p['evidence_gaps'])}")
    for o in p["other_files"]:
        shown = ", ".join(f"{x['name']} {x['shape']}" for x in o["tensors"]) or "(not a safetensors file, or unreadable)"
        out.append(f"- {o['file']}: {o['tensor_count']} tensors: {shown}")
    return "\n".join(out) + "\n"


# ---- comparing the tool with the labelers -------------------------------------------------------------------------------------------

def _check_labels(name: str, labels: dict, known: set[str]) -> None:
    for model, lab in labels.items():
        if model not in known:
            raise CorpusError(f"labeler {name}: {model!r} is not a model in tool.json")
        if not isinstance(lab, dict) or lab.get("class") not in CLASSES:
            raise CorpusError(f"labeler {name}: {model!r} needs a class from {CLASSES}")


def compare(tool: dict, labelers: dict, truth: dict | None = None) -> dict:
    truth = truth or {}
    for name, labels in labelers.items():
        _check_labels(name, labels, set(tool))
    rows, confusion = [], defaultdict(lambda: defaultdict(int))
    skipped = [m for m, d in tool.items() if d.get("class") is None]
    agree = 0
    for model, d in sorted(tool.items()):
        if d.get("class") is None:
            continue
        given = {n: l[model] for n, l in labelers.items() if model in l}
        votes = Counter(l["class"] for l in given.values())
        top = votes.most_common(2)
        majority = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None
        unanimous = len(votes) <= 1
        if majority is None:
            flag = "labelers disagree"
        elif d["class"] != majority:
            flag = "tool differs from the majority"
        elif not unanimous:
            flag = "labelers disagree"
        else:
            flag = ""
        if majority is not None and d["class"] == majority:
            agree += 1
        if majority is not None:
            confusion[majority][d["class"]] += 1
        rows.append({"id": model, "tool": d["class"], "tool_reasons": d.get("reasons", []), "majority": majority,
                     "labels": {n: {"class": l["class"], "confidence": l.get("confidence"), "why": l.get("why")} for n, l in given.items()},
                     "truth": truth.get(model), "flag": flag})
    anchored = [r for r in rows if r["truth"]]
    rep = {"n_compared": len(rows), "n_skipped": len(skipped), "skipped": skipped, "tool_vs_majority": {"agree": agree, "of": sum(1 for r in rows if r["majority"])},
           "rows": rows, "confusion": {k: dict(v) for k, v in confusion.items()},
           "truth": {"tool": {"n": len(anchored), "correct": sum(1 for r in anchored if r["tool"] == r["truth"])},
                     "labelers": {n: {"n": sum(1 for r in anchored if n in r["labels"]),
                                      "correct": sum(1 for r in anchored if n in r["labels"] and r["labels"][n]["class"] == r["truth"])}
                                  for n in labelers}}}
    return rep


def render_report(rep: dict) -> str:
    out = ["# Classification bench", "",
           f"{rep['n_compared']} models compared, {rep['n_skipped']} skipped (the Hub would not serve them without a token, or the repo has nothing to read).",
           f"The tool matched the labelers' majority on {rep['tool_vs_majority']['agree']} of {rep['tool_vs_majority']['of']} models that had one.", ""]
    t = rep["truth"]
    if t["tool"]["n"]:
        out.append(f"Against {t['tool']['n']} models with a known outcome: the tool is right on {t['tool']['correct']}; "
                   + "; ".join(f"labeler {n} on {v['correct']} of {v['n']}" for n, v in t["labelers"].items()) + ".")
        out.append("")
    if rep["confusion"]:
        out += ["## Majority label (rows) against the tool (columns)", ""]
        cols = sorted({c for r in rep["confusion"].values() for c in r})
        out.append("| label \\ tool | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for k, r in sorted(rep["confusion"].items()):
            out.append(f"| {k} | " + " | ".join(str(r.get(c, 0)) for c in cols) + " |")
        out.append("")
    flagged = [r for r in rep["rows"] if r["flag"]]
    out.append(f"## Where they differ ({len(flagged)})")
    for r in flagged:
        out += ["", f"### {r['id']}: {r['flag']}", f"- tool: {r['tool']} ({'; '.join(r['tool_reasons']) or 'no reasons'})"]
        if r["truth"]:
            out.append(f"- known outcome: {r['truth']}")
        for n, l in r["labels"].items():
            out.append(f"- {n}: {l['class']} ({l['confidence']}): {l['why']}")
    if rep["skipped"]:
        out += ["", "## Skipped", ", ".join(rep["skipped"])]
    return "\n".join(out) + "\n"


LABELING = """# Labeling guide

You are labeling models for a bench of stage 0's class decision. For each packet in `packets/` (a short evidence summary of one Hugging Face model
against its nearest supported model, Qwen3.8-27B), decide which class stage 0 SHOULD give. Use only the packet. Do not run code, do not fetch
anything, do not look at any file outside `packets/` and this guide, and do not guess what some tool would say.

Classes:
- weights-only: the nearest model's architecture, only the weights differ. Every tensor has an analogue with the same shape and dtype (a model may
  name its tensors with a different prefix; vision-tower, projector and mtp.* tensors may differ), and the config values that decide loading match.
- weights+sidecar: weights-only for the backbone, plus an extra weights file the nearest runtime does not load (a task head, for example) whose
  tensor names do not overlap the backbone's.
- restricted-head: the nearest model's backbone with its lm_head cut down to a few rows kept in a sidecar (the sidecar is as wide as the head, much
  shorter, and the model has no lm_head of its own).
- full-port: the model needs new model code (a different shape, a missing or extra text tensor, a different config value that decides loading).
- unknown: the evidence cannot decide: nothing to read (no safetensors, a gguf only, an adapter only), headers that did not read, or a layout
  that does not line up at all.

Write `labels/<your name>.json`: `{"<model id with / as -->": {"class": "<one of the five>", "confidence": "high|medium|low", "why": "<one sentence>"}}`
for every packet whose status is ok. Skip packets that say "Not classifiable by the bench".
"""


# ---- the command line ---------------------------------------------------------------------------------------------------------------------

def _collect(args) -> int:
    from orchard.research.http import Fetcher
    corpus = load_corpus(args.corpus)[: args.limit or None]
    out = Path(args.out)
    (out / "packets").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    base = read_base(args.base_snapshot)
    cache = Path(args.cache_dir or out / "http-cache")
    tool, truth = {}, {}
    for i, m in enumerate(corpus, 1):
        fetch = Fetcher(cache, offline=args.offline, refresh=args.refresh, max_requests=BENCH_MAX_REQUESTS, deadline_s=BENCH_DEADLINE_S)
        try:
            got = collect_one(fetch, m["id"], base)
        except Exception as exc:                                   # noqa: BLE001 - one bad repo must not stop the batch
            got = _skipped(m["id"], f"{type(exc).__name__}: {exc}"[:200])
        s = slug(m["id"])
        (out / "packets" / f"{s}.json").write_text(json.dumps(got["packet"], indent=2) + "\n", encoding="utf-8")
        (out / "packets" / f"{s}.md").write_text(render_packet(got["packet"]), encoding="utf-8")
        tool[s] = {**got["tool"], "id": m["id"], "stratum": m.get("stratum")}
        if m.get("truth"):
            truth[s] = m["truth"]
        print(f"[{i}/{len(corpus)}] {m['id']}: {got['tool']['class'] or 'skipped (' + got['tool'].get('why', '') + ')'}", flush=True)
    (out / "tool.json").write_text(json.dumps(tool, indent=2) + "\n", encoding="utf-8")
    (out / "truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    (out / "LABELING.md").write_text(LABELING, encoding="utf-8")
    print(f"wrote {len(tool)} packets and tool.json under {out}")
    return 0


def _compare(args) -> int:
    out = Path(args.out)
    tool = json.loads((out / "tool.json").read_text(encoding="utf-8"))
    truth = json.loads((out / "truth.json").read_text(encoding="utf-8")) if (out / "truth.json").is_file() else {}
    labelers = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "labels").glob("*.json"))}
    if not labelers:
        print("no labels in labels/: nothing to compare", file=sys.stderr)
        return 2
    rep = compare(tool, labelers, truth)
    (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(rep), encoding="utf-8")
    print(render_report(rep))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--corpus", required=True)
    c.add_argument("--base-snapshot", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--cache-dir")
    c.add_argument("--offline", action="store_true")
    c.add_argument("--refresh", action="store_true")
    k = sub.add_parser("compare")
    k.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        return _collect(args) if args.cmd == "collect" else _compare(args)
    except CorpusError as exc:
        print(f"classify_bench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
