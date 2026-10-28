# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The gate's check of the audio class decision that stage 0's template writes under `delta.json` `audio.decision`.

The template decides (orchard/skills/delta-triage-templates/delta_triage.py, `decide_audio`); this module re-reads the record the way `gate_package` re-reads
a package, so an agent that edited delta.json cannot pass a class its own evidence does not support. A delta.json with no decision (a text model, or a run
from before the decision existed) has nothing to check.

What it requires:
- the fields: class, path, alternative_path, confidence, strict, differences, overrides, and the class and path equal to delta.json's;
- each difference: a key, a category (serving, structural or unknown), `decided_by` (`allow-list` or `override`), and a non-empty `source` when an override decided it;
- each override: a kind (category or path), a non-empty source, and the fields its kind needs;
- consistency: a weights-only decision has only serving differences (none at all when strict), unless a path override with a source explains it, and has
  no tensor-level reason. A class of unknown has path full-port and confidence low.
"""
from __future__ import annotations

CLASSES = ("weights-only", "full-port", "unknown")
PATHS = ("weights-only", "full-port")
CONFIDENCES = ("high", "medium", "low")
CATEGORIES = ("serving", "structural", "unknown")
DECIDERS = ("allow-list", "override")
FIELDS = ("class", "path", "alternative_path", "confidence", "strict", "differences", "overrides")


def _text(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def decision_problems(delta) -> list[str]:
    """The reasons the decision in `delta` (a parsed delta.json) is malformed or does not support its class. Empty when there is no decision."""
    if not isinstance(delta, dict):
        return []
    audio = delta.get("audio")
    if not isinstance(audio, dict) or "decision" not in audio:
        return []
    d = audio["decision"]
    if not isinstance(d, dict):
        return ["delta.json audio.decision must be an object"]
    out = []
    for f in FIELDS:
        if f not in d:
            out.append(f"audio.decision has no {f}")
    if out:
        return out
    if d["class"] not in CLASSES:
        out.append(f"audio.decision class must be one of {CLASSES}")
    if d["path"] not in PATHS:
        out.append(f"audio.decision path must be one of {PATHS}")
    if d["alternative_path"] is not None and (d["alternative_path"] not in PATHS or d["alternative_path"] == d["path"]):
        out.append("audio.decision alternative_path must be null or the path that is not the chosen one")
    if d["confidence"] not in CONFIDENCES:
        out.append(f"audio.decision confidence must be one of {CONFIDENCES}")
    if not isinstance(d["strict"], bool):
        out.append("audio.decision strict must be true or false")
    if not isinstance(d["differences"], list):
        out.append("audio.decision differences must be a list")
    if not isinstance(d["overrides"], list):
        out.append("audio.decision overrides must be a list")
    if out:
        return out
    if delta.get("class") != d["class"]:
        out.append(f"audio.decision class {d['class']!r} is not delta.json's class {delta.get('class')!r}")
    if delta.get("path") != d["path"]:
        out.append(f"audio.decision path {d['path']!r} is not delta.json's path {delta.get('path')!r}")
    categories = []
    for i, x in enumerate(d["differences"]):
        if not isinstance(x, dict) or not _text(x.get("key")):
            out.append(f"audio.decision differences[{i}] needs a key")
            continue
        if x.get("category") not in CATEGORIES:
            out.append(f"audio.decision differences[{i}] category must be one of {CATEGORIES}")
            continue
        if x.get("decided_by") not in DECIDERS:
            out.append(f"audio.decision differences[{i}] decided_by must be one of {DECIDERS}")
            continue
        if x["decided_by"] == "override" and not _text(x.get("source")):
            out.append(f"audio.decision differences[{i}] was decided by an override and needs a source")
        categories.append(x["category"])
    path_override = None
    for i, o in enumerate(d["overrides"]):
        if not isinstance(o, dict) or o.get("kind") not in ("category", "path"):
            out.append(f"audio.decision overrides[{i}] needs a kind of category or path")
            continue
        if not _text(o.get("source")):
            out.append(f"audio.decision overrides[{i}] needs a source")
        if o["kind"] == "path":
            path_override = o
    if d["class"] == "unknown" and (d["path"] != "full-port" or d["confidence"] != "low"):
        out.append("audio.decision class unknown needs path full-port and confidence low")
    if d["class"] == "weights-only" and path_override is None:
        wrong = [c for c in categories if c != "serving"]
        if wrong:
            out.append(f"audio.decision class weights-only has {len(wrong)} difference(s) that are not serving settings and no path override explains them")
        if d["strict"] and categories:
            out.append("audio.decision class weights-only is strict but lists differences and no path override explains them")
    return out
