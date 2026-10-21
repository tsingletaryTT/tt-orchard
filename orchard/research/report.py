# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Merge the collectors into a ranked list of base candidates and write it to the run directory.

A candidate is a model stage 0 could compare against and a bundle could serve. Ranking (design 3.2): an
installed v6 bundle serves it (group 0), a published v6 bundle (1), a container bundle (2), no bundle (3);
then confidence (stated, inferred, weak), fewer differing config keys, and the cheaper class. Only a stated
or inferred candidate in group 0 is `selected`. Nothing here asks a model; every row says why it is there."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from orchard import nearest
from orchard.research import arch_match, headers, lineage, ports
from orchard.research.http import FetchError
from orchard.research.types import (CONFIDENCE_RANK, INFERRED, STATED, WEAK, Result, Unavailable, clean,
                                    valid_id)

MANIFEST_URL = "https://huggingface.co/{name}/resolve/main/tt_kernel_manifest.json"
MAX_PUBLISHED = 12          # published bundles whose manifests are read
MAX_CANDIDATES = 30         # candidate repos whose config.json is read
MAX_ESTIMATES = 5           # candidates whose headers are read for a class estimate
CLASS_RANK = {"weights-only": 0, "weights+sidecar": 1, "full-port": 2, "unknown": 3}
V6 = "6"
RESEARCH_DIR = "research"


def _unique(items) -> list[str]:
    out: list[str] = []
    for item in items:
        if valid_id(item) and item not in out:
            out.append(item)
    return out


def _empty(model_id: str, now: str, offline: bool, reason: str) -> dict:
    return {"model": model_id, "generated_at": now, "offline": offline, "new_model_gated": None,
            "stated_bases": [], "candidates": [],
            "bundle_choices": [], "selected": None, "ports": [], "evidence": [],
            "unavailable": [Unavailable("research", reason).as_dict()]}


def _guard(res: Result, source: str, call, fallback):
    """Run one collector. An exception the collector did not expect (a field of the wrong type in an answer from
    the internet, say) becomes an Unavailable row with the exception's type, and `fallback` stands in for its
    answer, so `build` never raises."""
    try:
        return call()
    except Exception as exc:
        res.unavailable.append(Unavailable(source, type(exc).__name__))
        return fallback() if callable(fallback) else fallback


def _bundles(installed, search, family: str, fetch, res: Result) -> dict[str, dict]:
    """name -> {"name", "installed", "schema", "repo", "code"} for installed and published bundles."""
    rows = {b.name: {"name": b.name, "installed": True, "schema": str(b.schema), "repo": b.weights_repo, "code": []}
            for b in installed}
    try:
        found = search(family) if search is not None else []
    except Exception as exc:                                  # a failing search leaves a gap in the report
        res.unavailable.append(Unavailable("bundle search", f"{type(exc).__name__}"))
        found = []
    for item in (found or [])[:MAX_PUBLISHED]:
        name = item.get("name") if isinstance(item, dict) else None
        if not valid_id(name) or name in rows:
            continue
        try:
            manifest = fetch.json(MANIFEST_URL.format(name=name))
            repo = manifest["weights"]["repo_id"]
            schema = str(manifest.get("schema_version", ""))
        except (FetchError, KeyError, TypeError, AttributeError) as exc:
            res.unavailable.append(Unavailable("bundle manifest", clean(f"{name}: {getattr(exc, 'reason', 'unreadable')}")))
            continue
        except Exception as exc:
            res.unavailable.append(Unavailable("bundle manifest", f"{name}: {type(exc).__name__}"))
            continue
        if not valid_id(repo):
            continue
        code = []
        try:
            payload = fetch.json(lineage.HUB_MODEL.format(id=name))
            code = ports.bundle_code(payload.get("siblings"), name, fetch.now()) if isinstance(payload, dict) else []
        except Exception:                       # the code listing is optional; the bundle is still a candidate
            pass
        rows[name] = {"name": name, "installed": False, "schema": schema, "repo": repo, "code": code}
    return rows


def _note_header_errors(res: Result, repo: str, read: Result) -> None:
    """A header that could not be read leaves the estimate empty; say why, so the gap is visible."""
    errors = read.data.get("errors") or []
    if errors:
        more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
        res.unavailable.append(Unavailable("headers", clean(f"{repo}: {errors[0]}{more}")))


def _config(fetch, repo: str, res: Result) -> dict | None:
    """A repo's config.json. When it cannot be read (a gated or private repo answers 401 or 403) the reason,
    with its HTTP status, goes into `res.unavailable`, so the candidate is not silently left without a comparison."""
    if not valid_id(repo):
        return None
    try:
        cfg = fetch.json(arch_match.CONFIG_URL.format(repo=repo))
    except FetchError as exc:
        res.unavailable.append(Unavailable("config", clean(f"{repo}: {exc.reason}")))
        return None
    except Exception as exc:
        res.unavailable.append(Unavailable("config", f"{repo}: {type(exc).__name__}"))
        return None
    if not isinstance(cfg, dict):
        res.unavailable.append(Unavailable("config", f"{repo}: config.json is not an object"))
        return None
    return cfg


def _mismatch(cmp: dict) -> str:
    if not cmp.get("same_family"):
        return "another architecture family"
    return "other decisive config values: " + ", ".join(cmp["differing"])


def _group(bundles: list[dict]) -> int:
    if any(b["installed"] and b["schema"] == V6 for b in bundles):
        return 0
    if any(b["schema"] == V6 for b in bundles):
        return 1
    return 2 if bundles else 3


def build(*, model_id, hub, base_override, installed, search, hub_info, fetch, run=subprocess.run,
          which=shutil.which, no_github: bool = False) -> dict:
    now = fetch.now()
    if getattr(fetch, "offline", False):
        return _empty(model_id, now, True, "--offline")
    res = Result()
    lin = _guard(res, "lineage", lambda: lineage.chain(fetch, model_id), lambda: Result(data={"ancestors": [], "edges": []}))
    res.extend(lin)
    stated = _unique([base_override] + list(hub.base_models if hub else []) + lin.data["ancestors"])
    before = len(res.unavailable)
    new_cfg = _config(fetch, model_id, res) or {}
    # Research sends no token, so a gated or private model's config and headers answer 401 or 403. Say so, so an
    # empty or weak candidate list is not read as "there is no base".
    gated = bool(hub and (hub.gated or hub.private)) or any(
        u.reason.startswith((f"{model_id}: HTTP 401", f"{model_id}: HTTP 403")) for u in res.unavailable[before:])
    letters = re.match(r"[a-z]+", str(new_cfg.get("model_type") or "").lower())
    family = nearest.family_query(stated[0]) if stated else (letters.group(0) if letters else "")
    sib = _guard(res, "siblings", lambda: lineage.siblings(fetch, stated[0], exclude={model_id}),
                 lambda: Result(data={"siblings": []})) if stated else Result(data={"siblings": []})
    res.extend(sib)
    sibling_ids = set(sib.data["siblings"])
    # The direct bases: --base, the card, and the tags of the model itself. A base further up the lineage (a base
    # of the base) is a relation the card does not state for this model, so its own config decides its row.
    direct = {base_override, *(hub.base_models if hub else []),
              *(e["parent"] for e in lin.data["edges"] if e["child"] == model_id)}
    parent_of = {e["parent"]: e["child"] for e in reversed(lin.data["edges"])}
    rows = _guard(res, "bundles", lambda: _bundles(installed, search, family, fetch, res), dict) if family else {}
    by_repo: dict[str, list[dict]] = {}
    for row in rows.values():
        by_repo.setdefault(row["repo"], []).append(row)
        res.evidence += row["code"]

    candidates: dict[str, dict] = {}
    cfgs: dict[str, dict] = {}
    # The model is never its own base, even when an installed bundle already serves it.
    for repo in [r for r in _unique(stated + list(by_repo)) if r != model_id][:MAX_CANDIDATES]:
        cfg = _config(fetch, repo, res)
        no_match = {"confidence": None, "differing": {}}
        cmp = _guard(res, "config", lambda: arch_match.compare(new_cfg, cfg), no_match) \
            if cfg is not None and new_cfg else no_match
        known = "same_family" in cmp                 # the comparison ran (both configs read, compare did not fail)
        if repo == base_override:
            confidence, why = STATED, "named with --base"
        elif repo in direct:
            # The card's word stands unless both configs are known and disagree.
            confidence = STATED if not known or cmp["confidence"] == INFERRED else WEAK
            why = "named as the base model" + ("" if confidence == STATED else ", but " + _mismatch(cmp))
        elif repo in stated:
            confidence = INFERRED if not known or cmp["confidence"] == INFERRED else WEAK
            why = f"a base of {parent_of.get(repo, stated[0])} in the lineage" + (
                "" if confidence == INFERRED else ", " + _mismatch(cmp))
        elif repo in sibling_ids and not cmp["confidence"] and cfg is not None and new_cfg:
            continue                          # a fine-tune of the base in another architecture family: not a candidate
        elif repo in sibling_ids:
            # A sibling shares the base, which says nothing about this model's own architecture: when both configs
            # are known the comparison decides; when one is missing the relationship alone stays "inferred".
            confidence = cmp["confidence"] or INFERRED
            why = f"another fine-tune of {stated[0]}" + (
                "" if confidence == INFERRED else ", other decisive config values: " + ", ".join(cmp["differing"]))
        elif cmp["confidence"]:
            confidence = cmp["confidence"]
            why = ("same architecture family and the same decisive config values" if confidence == INFERRED
                   else "same architecture family, other decisive config values: " + ", ".join(cmp["differing"]))
        else:
            continue
        if cfg is not None:
            cfgs[repo] = cfg
        bundles = [{"name": b["name"], "installed": b["installed"], "schema": b["schema"]}
                   for b in sorted(by_repo.get(repo, []), key=lambda b: (not b["installed"], b["schema"] != V6, b["name"]))]
        candidates[repo] = {"repo": repo, "confidence": confidence, "why": clean(why), "differing": cmp["differing"],
                            "group": _group(bundles), "bundles": bundles, "class_estimate": None, "download_gb": None}

    ordered = sorted(candidates.values(), key=lambda c: (c["group"], CONFIDENCE_RANK[c["confidence"]],
                                                         len(c["differing"]), c["repo"]))
    new_model = None
    for cand in [c for c in ordered if c["confidence"] != WEAK][:MAX_ESTIMATES]:
        try:
            info, err = hub_info(cand["repo"])
        except Exception as exc:                              # a failing lookup becomes a row in the report
            info, err = None, type(exc).__name__
        if info is None:
            res.unavailable.append(Unavailable("class estimate", clean(f"{cand['repo']}: {err}")))
            continue
        cand["download_gb"] = round(info.total_bytes / 1e9, 1)
        if new_model is None:
            new_model = (_guard(res, "headers", lambda: headers.read_model(fetch, model_id, list(hub.files)),
                                lambda: Result(data={"tensors": {}, "errors": []})) if hub is not None
                         else Result(unavailable=[Unavailable("headers", f"{model_id}: the hub did not answer")]))
            res.extend(Result(unavailable=new_model.unavailable))
            _note_header_errors(res, model_id, new_model)
        other = _guard(res, "headers", lambda: headers.read_model(fetch, cand["repo"], list(info.files)),
                       lambda: Result(data={"tensors": {}, "errors": []}))
        res.extend(Result(unavailable=other.unavailable))
        _note_header_errors(res, cand["repo"], other)
        if new_model.data.get("tensors") and other.data.get("tensors") and cand["repo"] in cfgs:
            cand["class_estimate"] = _guard(
                res, "class estimate", lambda: headers.estimate_class(new_cfg, new_model.data, cfgs[cand["repo"]], other.data), None)
    ordered.sort(key=lambda c: (c["group"], CONFIDENCE_RANK[c["confidence"]], len(c["differing"]),
                                CLASS_RANK.get((c["class_estimate"] or {}).get("class"), 3), c["repo"]))

    choices = [{"name": b["name"], "installed": b["installed"], "repo": c["repo"], "schema": b["schema"],
                "class": (c["class_estimate"] or {}).get("class"), "confidence": c["confidence"], "group": c["group"]}
               for c in ordered for b in c["bundles"]]
    selected = None
    for c in ordered:
        if c["group"] == 0 and c["confidence"] in (STATED, INFERRED):
            b = next(b for b in c["bundles"] if b["installed"] and b["schema"] == V6)
            selected = {"repo": c["repo"], "bundle": b["name"], "confidence": c["confidence"],
                        "class": (c["class_estimate"] or {}).get("class")}
            break
    port_res = _guard(res, "ports", lambda: ports.collect(fetch, new_cfg, family=family or "", run=run, which=which,
                                                          no_github=no_github, ids=_unique(stated + [model_id])), Result) if family else Result()
    res.extend(port_res)
    return {"model": model_id, "generated_at": now, "offline": False, "new_model_gated": gated, "stated_bases": stated,
            "candidates": ordered, "bundle_choices": choices, "selected": selected,
            "ports": [e.as_dict() for e in port_res.evidence],
            "evidence": [e.as_dict() for e in res.evidence if e not in port_res.evidence],
            "unavailable": [u.as_dict() for u in res.unavailable]}


def markdown(report: dict) -> str:
    lines = [f"# Base candidates for {report['model']}", "",
             f"Generated {report['generated_at']}. Every row says where it came from; a source that could not be "
             "read is listed under Unavailable."]
    if report["offline"]:
        lines += ["", "Research was skipped (`--offline`)."]
    if report.get("new_model_gated"):
        lines += ["", "The new model is gated or private on the hub. Research sends no token, so it could not read the "
                  "model's config.json or safetensors headers: no architecture comparison and no class estimate were "
                  "made for it. A short or empty candidate list does not mean there is no base."]
    lines += ["", "## Candidates", ""]
    for c in report["candidates"]:
        est = c["class_estimate"]
        cls = f"expected class {est['class']}" + (" (a sidecar is possible)" if est["sidecar_possible"] else "") if est \
            else "no class estimate"
        names = ", ".join(f"{b['name']} ({'installed' if b['installed'] else 'published'}, schema {b['schema']})"
                          for b in c["bundles"]) or "no bundle"
        size = f", {c['download_gb']} GB" if c["download_gb"] is not None else ""
        lines.append(f"- **{c['repo']}** [{c['confidence']}] {c['why']}. {cls}{size}. Bundles: {names}.")
    if not report["candidates"]:
        lines.append("- none found")
    lines += ["", "## Existing ports", ""]
    lines += [f"- [{p['confidence']}] {p['claim']} ({p['url']})" for p in report["ports"]] or ["- none found"]
    lines += ["", "## Unavailable", ""]
    lines += [f"- {u['source']}: {u['reason']}" for u in report["unavailable"]] or ["- nothing"]
    return "\n".join(lines) + "\n"


STAGE0_FILE = "base-candidates.stage0.json"
CLASSES = tuple(CLASS_RANK)
CONFIDENCES = tuple(CONFIDENCE_RANK)


def _enum(value, allowed):
    return value if isinstance(value, str) and value in allowed else None


def _id(value):
    return value if valid_id(value) else None


def stage0_view(report: dict) -> dict:
    """What the stage 0 agent may read: candidates only. The full report holds text from the internet (issue and
    pull request titles, claims, reasons, config values), which an agent must not read as instructions. Every
    string in this view is a model id that passed `valid_id`, one of the fixed confidence values, or a class
    name; anything else becomes None, and a candidate whose repo is not a model id is left out."""
    rows = report.get("candidates") if isinstance(report, dict) else None
    out = []
    for c in rows if isinstance(rows, list) else []:
        if not isinstance(c, dict) or _id(c.get("repo")) is None:
            continue
        est = c.get("class_estimate")
        bundles = c.get("bundles")
        out.append({"repo": c["repo"], "confidence": _enum(c.get("confidence"), CONFIDENCES),
                    "group": c["group"] if isinstance(c.get("group"), int) else None,
                    "class_estimate": _enum(est.get("class") if isinstance(est, dict) else None, CLASSES),
                    "bundles": [b["name"] for b in bundles if isinstance(b, dict) and _id(b.get("name"))]
                    if isinstance(bundles, list) else []})
    sel = report.get("selected") if isinstance(report, dict) else None
    selected = None
    if isinstance(sel, dict) and _id(sel.get("repo")) and _id(sel.get("bundle")):
        selected = {"repo": sel["repo"], "bundle": sel["bundle"],
                    "confidence": _enum(sel.get("confidence"), CONFIDENCES), "class": _enum(sel.get("class"), CLASSES)}
    return {"model": _id(report.get("model")) if isinstance(report, dict) else None,
            "selected": selected, "candidates": out}


def _write_atomic(path: Path, text: str) -> None:
    """Write beside the target and rename over it, so a reader sees the old file or the whole new one."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write(run_dir, report: dict) -> Path:
    """`<run_dir>/research/base-candidates.json`, `.md` and `base-candidates.stage0.json` (the view stage 0
    reads). Nothing else is written. Each file is replaced atomically, and all three texts are built before the
    first file is touched. A resume builds the report again and overwrites these files; `Evidence.fetched_at`
    in them is the time of that build, because the cache stores the fetch time but the evidence is stamped
    from `fetch.now()` when it is made."""
    out = Path(run_dir) / RESEARCH_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / "base-candidates.json"
    texts = {path: json.dumps(report, indent=2, sort_keys=True) + "\n",
             out / "base-candidates.md": markdown(report),
             out / STAGE0_FILE: json.dumps(stage0_view(report), indent=2, sort_keys=True) + "\n"}
    for target, text in texts.items():
        _write_atomic(target, text)
    return path
