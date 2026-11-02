# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 for a scoring model (issue 358, docs/superpowers/specs/2026-11-02-scoring-model-check-design.md): what the gate checks.

The supervisor copies skills/scoring-check-templates/scoring_check.py to stages/2/scoring_check.py, drafts
stages/2/scoring_check_config.json (`draft_config`), and runs COMMAND under a lease. The template writes evidence/scoring-check.json
(RECORD_SCHEMA) and prints `SCORING_CHECK_SHA256 <hex>` of it as its last line. This module reads the files that run leaves and returns
the reasons the result cannot be believed. It holds no state and starts nothing. Slice 1 has no stage-machine wiring and no result.json:
`check(stage_dir, run_dir)` is the whole gate.

What it checks, each as a named reason:

- the supervisor's test ran the fixed COMMAND, exited 0 and did not time out; the output file is the one it recorded and holds exactly one
  mark line that matches the record;
- the stage's copy of the template is the shipped file and the record carries its sha256; the config is the file the record names;
- the record names the stage 1 pair file and reference file with their current sha256, and the reference scores and group names in the
  record are the ones on disk;
- the gate recomputes every parity number from the record's scores (`parity`, a second copy of the template's) and the record must report
  the same numbers;
- the recomputed numbers meet the bounds. The shipped defaults are DEFAULT_BOUNDS. The config may state stricter bounds and never looser
  ones: a config bound that is looser than the default is a reason, and the stricter value is the one applied;
- the port's source files are the ones the record lists, unchanged, none is a link, one mentions ttnn, none reads the reference.

What this does not show: the same limits as the full-port check. The numbers come from a test the supervisor ran, and a port that returns
the reference's scores without computing them looks the same as one that computed them. The bounds are choices made from two public
bundles' accuracy tables and have not been checked against a chip run through the supervisor.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

from orchard import safeio
from orchard.full_port import _inside, _json, _sha

RECORD_SCHEMA = "orchard-scoring-check/1"
KIND = "reranker"
REFERENCE_FILE = "stages/1/evidence/reference/scoring-reference.json"
PAIRS_FILE = "stages/1/evidence/reference/scoring-pairs.json"
PORT_DIR = "stages/2/port/"
PORT_MODULE = "stages/2/port/port_model.py"
RECORD_FILE = "evidence/scoring-check.json"
OUTPUT_FILE = "evidence/hw-test-output.txt"
OUTPUT_MARK = "SCORING_CHECK_SHA256"
MODEL_ROLE = "model"
UNHASHED_SUFFIXES = (".pyc", ".pyo", ".so", ".pyd", ".pth")      # importable and not in the manifest; the template's list is the same
TEMPLATE = Path(__file__).resolve().parent / "skills" / "scoring-check-templates" / "scoring_check.py"
TEMPLATE_NAME = "scoring_check.py"
CONFIG_NAME = "scoring_check_config.json"
COMMAND = "python3 -P -s stages/2/scoring_check.py"
DEADLINE_S = 3600
ORDER_GAP = 0.5                       # the same value as the template's; a test checks the two files agree
JSON_MAX = 16 * 1024 * 1024           # choice: the pair file is a few MB at most (32 prompts of up to 8192 ids)
SOURCE_MAX = 2 * 1024 * 1024
MAX_PAIRS = 4096                      # the same limit as the template's: more pairs than this are not the set the recorder wrote
MAX_VALUE = 1e6                       # the same limit as the template's, for one score
TOLERANCE = 1e-9                      # the gate's recomputed number against the number the record reports

# Bounds, with the evidence each is drawn from. The 4B reranker bundle (raahemnabeel/qwen3-reranker-4b, doc/accuracy.md) reports Pearson r
# 0.9997, mean/max absolute difference 0.140/0.64 logits, top-1 33/33 groups and 100% of separated pairs in order; the 8B VL bundle reports
# 0.9991, 0.107/0.39 and 41/41. Each bound below leaves room around those numbers and sits at or inside the policies the 4B bundle
# measured and did not select (bfp8 MLP: 0.217/0.63 with top-1 31/32; MLP LoFi: 0.359/0.93 with r 0.99885). They are choices, not
# measurements of this repository's own port runs. "min" bounds are floors, "max" bounds are ceilings.
DEFAULT_BOUNDS = {
    "pearson_min": 0.998,             # the 4B bundle's selected policy: 0.9997; its LoFi policy: 0.99885
    "mean_abs_diff_max": 0.25,        # logits; the selected policy 0.14, the bfp8 policy 0.217
    "max_abs_diff_max": 1.0,          # logits; the selected policy 0.64, the LoFi policy 0.93
    "top1_match_min": 0.95,           # share of groups with the same top document; bundles 1.0, bfp8 policy 31/32 = 0.969
    "order_kept_min": 0.98,           # share of reference-separated pairs kept in order; bundles 1.0
    "min_pairs": 24,                  # a set smaller than the shipped one (32) cannot carry the shares above
    "min_groups": 6,
    "min_order_pairs": 10,            # separated pairs needed for the order share to mean something
}
_FLOORS = ("pearson_min", "top1_match_min", "order_kept_min", "min_pairs", "min_groups", "min_order_pairs")


def template_sha256() -> str | None:
    try:
        return hashlib.sha256(safeio.read_bytes(TEMPLATE, max_bytes=JSON_MAX)).hexdigest()
    except safeio.SafeReadError:
        return None


def draft_config(model_snapshot: str) -> dict:
    """The config the supervisor writes next to the template: the snapshot, the kind and the bounds in force."""
    return {"model_snapshot": model_snapshot, "kind": KIND, "bounds": dict(DEFAULT_BOUNDS)}


def _num(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def _score(x) -> bool:
    """A finite number no larger in size than MAX_VALUE: a logit difference. A larger one would overflow the sums in `parity`."""
    return _num(x) and abs(x) <= MAX_VALUE


# ---- the numbers (a second copy of the template's, so the gate does not take them from the record) ----------------------------

def pearson(a, b) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    if saa <= 0.0 or sbb <= 0.0:
        return 0.0
    den = math.sqrt(saa) * math.sqrt(sbb)         # not sqrt(saa * sbb): the product of two small variances can underflow to 0
    return sab / den if den > 0.0 else 0.0


def _first_max(values, idx):
    best = idx[0]
    for i in idx:
        if values[i] > values[best]:
            best = i
    return best


def parity(reference, chip, groups) -> dict:
    """Pearson r, mean and max absolute difference, groups with the same top document, and the order of document pairs within a group that
    the reference separates by at least ORDER_GAP. Lists of equal, non-zero length."""
    n = len(reference)
    diffs = [abs(a - b) for a, b in zip(reference, chip)]
    members: dict = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    top1 = sum(1 for idx in members.values() if _first_max(reference, idx) == _first_max(chip, idx))
    kept = total = 0
    for idx in members.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                gap = reference[i] - reference[j]
                if abs(gap) >= ORDER_GAP:
                    total += 1
                    if (chip[i] - chip[j]) * gap > 0:
                        kept += 1
    return {"pearson": pearson(reference, chip), "mean_abs_diff": sum(diffs) / n, "max_abs_diff": max(diffs),
            "top1_groups_matching": top1, "top1_groups": len(members),
            "order_pairs": total, "order_pairs_kept": kept, "order_kept": (kept / total) if total else None}


# ---- bounds ------------------------------------------------------------------------------------------------------------------

def effective_bounds(config_bounds, reasons: list) -> dict:
    """The bounds to apply: the default for each key, replaced by the config's value when that is at least as strict. A looser value, a
    value that is not a finite number and a key nobody defined are each a reason."""
    out = dict(DEFAULT_BOUNDS)
    if config_bounds is None:
        return out
    if not isinstance(config_bounds, dict):
        reasons.append(f"{CONFIG_NAME} bounds must be an object, got {type(config_bounds).__name__}")
        return out
    for key, val in config_bounds.items():
        if key not in DEFAULT_BOUNDS:
            reasons.append(f"{CONFIG_NAME} bounds has an unknown key {str(key)[:60]!r}")
        elif not _num(val):
            reasons.append(f"{CONFIG_NAME} bound {key} must be a finite number, got {val!r}")
        elif (val < DEFAULT_BOUNDS[key]) if key in _FLOORS else (val > DEFAULT_BOUNDS[key]):
            reasons.append(f"{CONFIG_NAME} bound {key} {val} is looser than the shipped {DEFAULT_BOUNDS[key]}: a config may tighten a bound and never loosen it")
        else:
            out[key] = val
    return out


def bound_reasons(m: dict, bounds: dict, n_pairs: int) -> list[str]:
    """The reasons the numbers `m` (from `parity`) miss the bounds."""
    r = []
    if n_pairs < bounds["min_pairs"]:
        r.append(f"the record scores {n_pairs} pairs; at least {bounds['min_pairs']} are needed")
    if m["top1_groups"] < bounds["min_groups"]:
        r.append(f"the record has {m['top1_groups']} groups; at least {bounds['min_groups']} are needed")
    if m["pearson"] < bounds["pearson_min"]:
        r.append(f"pearson {m['pearson']:.5f} is below the minimum of {bounds['pearson_min']}")
    if m["mean_abs_diff"] > bounds["mean_abs_diff_max"]:
        r.append(f"mean absolute score difference {m['mean_abs_diff']:.4f} is above the maximum of {bounds['mean_abs_diff_max']}")
    if m["max_abs_diff"] > bounds["max_abs_diff_max"]:
        r.append(f"max absolute score difference {m['max_abs_diff']:.4f} is above the maximum of {bounds['max_abs_diff_max']}")
    share = m["top1_groups_matching"] / m["top1_groups"]
    if share < bounds["top1_match_min"]:
        r.append(f"the top document matches in {m['top1_groups_matching']} of {m['top1_groups']} groups ({share:.3f}), below the minimum of {bounds['top1_match_min']}")
    if m["order_pairs"] < bounds["min_order_pairs"] or m["order_kept"] is None:
        r.append(f"the reference separates only {m['order_pairs']} pairs by {ORDER_GAP}; at least {bounds['min_order_pairs']} are needed to judge the order")
    elif m["order_kept"] < bounds["order_kept_min"]:
        r.append(f"{m['order_pairs_kept']} of {m['order_pairs']} separated pairs keep the reference's order ({m['order_kept']:.3f}), below the minimum of {bounds['order_kept_min']}")
    return r


# ---- the gate ----------------------------------------------------------------------------------------------------------------

def check(stage_dir, run_dir) -> tuple[list[str], list[str]]:
    """(reasons, evidence paths) for a stage 2 directory of a scoring model. No reasons means the check passes."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    reasons: list[str] = []
    seen: list[str] = []
    record = _json(stage_dir / RECORD_FILE, RECORD_FILE, reasons)
    marker = _json(stage_dir / "test-result.json", "test-result.json", reasons)
    config = _json(stage_dir / CONFIG_NAME, CONFIG_NAME, reasons)
    if record is None:
        return reasons, seen
    seen.append(os.path.relpath(stage_dir / RECORD_FILE, run_dir))
    if isinstance(marker, dict):
        _marker_reasons(marker, reasons)
    if record.get("schema") != RECORD_SCHEMA:
        reasons.append(f"{RECORD_FILE} schema must be {RECORD_SCHEMA!r}, got {record.get('schema')!r}")
    if record.get("kind") != KIND:
        reasons.append(f"{RECORD_FILE} kind must be {KIND!r}, got {record.get('kind')!r}")
    if isinstance(config, dict) and config.get("kind") != KIND:
        reasons.append(f"{CONFIG_NAME} kind must be {KIND!r}, got {config.get('kind')!r}")
    bounds = effective_bounds(config.get("bounds") if isinstance(config, dict) else None, reasons)

    ref_scores, groups = _disk_reasons(record, run_dir, reasons)
    chip = record.get("chip_scores")
    ok_scores = (isinstance(chip, list) and 0 < len(chip) <= MAX_PAIRS and all(_score(s) for s in chip)
                 and ref_scores is not None and groups is not None and len(chip) == len(ref_scores) == len(groups))
    if not ok_scores:
        reasons.append(f"chip_scores must be a list of 1 to {MAX_PAIRS} finite numbers of size at most {MAX_VALUE:g}, one for each pair of the stage 1 pair file")
    else:
        m = parity(ref_scores, chip, groups)
        reported = record.get("metrics")
        if not isinstance(reported, dict):
            reasons.append(f"{RECORD_FILE} reports no metrics object")
        else:
            for key in ("pearson", "mean_abs_diff", "max_abs_diff", "order_kept"):
                got = reported.get(key)
                want = m[key]
                if want is None and got is None:
                    continue
                if not _num(got) or want is None or abs(got - want) > TOLERANCE:
                    reasons.append(f"{RECORD_FILE} metrics.{key} {got!r} differs from the {want!r} the gate recomputed from the record's scores")
            for key in ("top1_groups_matching", "top1_groups", "order_pairs", "order_pairs_kept"):
                if reported.get(key) != m[key] or type(reported.get(key)) is not int:
                    reasons.append(f"{RECORD_FILE} metrics.{key} {reported.get(key)!r} differs from the {m[key]!r} the gate recomputed")
        reasons += bound_reasons(m, bounds, len(chip))
    if isinstance(marker, dict):
        rc, mc = record.get("chips"), marker.get("chips")
        if not all(isinstance(c, list) and c and all(isinstance(x, str) for x in c) for c in (rc, mc)) or sorted(rc) != sorted(mc):
            reasons.append(f"{RECORD_FILE} chips {rc!r} are not the chips the supervisor leased for the test {mc!r} (both must be non-empty lists of chip names)")
        _output_reasons(stage_dir, run_dir, marker, reasons)
    for key in ("load_s", "score_s"):
        v = record.get(key)
        if not _num(v) or v <= 0:
            reasons.append(f"{RECORD_FILE} {key} must be a positive number of seconds, got {v!r}")
    _template_reasons(record, stage_dir, reasons)
    _source_reasons(record, stage_dir, run_dir, reasons)
    return reasons, seen


def _marker_reasons(marker: dict, reasons: list) -> None:
    code, timed_out = marker.get("returncode"), marker.get("timed_out")
    if not (type(code) is int and code == 0 and timed_out is False):
        if timed_out:
            why = "the hardware test ran past its deadline and was stopped"
        elif code is None:
            why = "the hardware test has no exit code (it was refused or did not start)"
        else:
            why = f"the hardware test exited with code {code!r}"
        reasons.append(f"test-result.json: {why}")
    if marker.get("command") != COMMAND:
        reasons.append(f"test-result.json: the hardware test ran {marker.get('command')!r}; the supervisor's command for this stage is {COMMAND!r}")


def _disk_reasons(record: dict, run_dir: Path, reasons: list):
    """The record against the stage 1 files on disk. Returns (reference scores, group names) from disk, or None for each that cannot be read."""
    ref_doc = pairs_doc = None
    for key, want in (("reference", REFERENCE_FILE), ("pairs", PAIRS_FILE)):
        entry = record.get(key)
        if not isinstance(entry, dict) or entry.get("path") != want:
            reasons.append(f"{key}.path must be {want!r}, the stage 1 file, got {entry.get('path') if isinstance(entry, dict) else entry!r}")
            continue
        f = _inside(run_dir, want)
        if f is None:
            reasons.append(f"{key}: {want} is not a file in the run directory")
            continue
        if entry.get("sha256") != _sha(f):
            reasons.append(f"{key}.sha256 does not match {want} as it is on disk")
        try:
            doc = safeio.read_json(f, max_bytes=JSON_MAX, expect=dict)
        except safeio.SafeReadError as exc:
            reasons.append(f"{want} is not readable JSON: {exc.reason}")
            continue
        if key == "reference":
            ref_doc = doc
        else:
            pairs_doc = doc
    scores = groups = None
    if ref_doc is not None:
        s = ref_doc.get("scores")
        if isinstance(s, list) and 0 < len(s) <= MAX_PAIRS and all(_score(x) for x in s):
            scores = s
        else:
            reasons.append(f"{REFERENCE_FILE} holds no list of 1 to {MAX_PAIRS} finite scores of size at most {MAX_VALUE:g}")
        if ref_doc.get("model_kind") != KIND:
            reasons.append(f"{REFERENCE_FILE} is for kind {ref_doc.get('model_kind')!r}, not {KIND!r}")
        named, used = ref_doc.get("pairs"), record.get("pairs")
        if not isinstance(named, dict):
            reasons.append(f"{REFERENCE_FILE} pairs must be an object that names the pair file")
        elif pairs_doc is not None and isinstance(used, dict) and named.get("sha256") != used.get("sha256"):
            reasons.append(f"{REFERENCE_FILE} names a pair file other than the one the record used")
    if pairs_doc is not None:
        rows = pairs_doc.get("pairs")
        if isinstance(rows, list) and 0 < len(rows) <= MAX_PAIRS and all(isinstance(p, dict) and isinstance(p.get("group"), str) for p in rows):
            groups = [p["group"] for p in rows]
            if record.get("groups") != groups:
                reasons.append(f"{RECORD_FILE} groups are not the groups of {PAIRS_FILE}: the comparison was made on another grouping")
            if scores is not None and len(scores) != len(rows):
                reasons.append(f"{REFERENCE_FILE} has {len(scores)} scores for the {len(rows)} pairs of {PAIRS_FILE}")
        else:
            reasons.append(f"{PAIRS_FILE} holds no list of 1 to {MAX_PAIRS} pairs with group names")
    if scores is not None and record.get("reference_scores") != scores:
        reasons.append(f"{RECORD_FILE} reference_scores are not the scores in {REFERENCE_FILE}: the comparison was made against numbers the reference did not produce")
    return scores, groups


def _output_reasons(stage_dir: Path, run_dir: Path, marker: dict, reasons: list) -> None:
    out = stage_dir / OUTPUT_FILE
    rec = marker.get("output")
    want = os.path.relpath(out, run_dir)
    if not isinstance(rec, dict) or rec.get("path") != want or rec.get("sha256") != _sha(out):
        reasons.append(f"{OUTPUT_FILE} is not the output file the supervisor recorded in test-result.json")
        return
    try:
        text = safeio.read_bytes(out, max_bytes=JSON_MAX).decode("utf-8", "replace")
    except safeio.SafeReadError as exc:
        reasons.append(f"{OUTPUT_FILE} is not readable: {exc.reason}")
        return
    marks = re.findall(rf"^{OUTPUT_MARK} ([0-9a-f]{{64}})\s*$", text, re.M)
    starts = [l for l in text.splitlines() if l.startswith(OUTPUT_MARK)]
    if len(starts) != 1:
        reasons.append(f"{OUTPUT_FILE} has {len(starts)} lines that begin with {OUTPUT_MARK}; the template prints exactly one "
                       "(and prefixes every line the port prints), so something else wrote a line of its own")
    if not marks or marks[-1] != _sha(stage_dir / RECORD_FILE):
        reasons.append(f"{OUTPUT_FILE} does not carry a {OUTPUT_MARK} line that matches {RECORD_FILE}, so the record was not written by the test the supervisor ran")


def _template_reasons(record: dict, stage_dir: Path, reasons: list) -> None:
    shipped = template_sha256()
    if shipped is None:
        reasons.append(f"the shipped template {TEMPLATE.name} cannot be read, so the record cannot be tied to it")
        return
    copy = stage_dir / TEMPLATE_NAME
    mine = _sha(copy) if copy.is_file() and not copy.is_symlink() else None
    if mine is None:
        reasons.append(f"{TEMPLATE_NAME} is not in this stage directory as a regular file: the supervisor copies it there and the gate compares it with the shipped one")
    elif mine != shipped:
        reasons.append(f"{TEMPLATE_NAME} in this stage differs from the shipped template (sha256 {mine[:12]} against {shipped[:12]}), so the record is not the template's")
    got = record.get("template_sha256")
    if got != shipped:
        reasons.append(f"{RECORD_FILE} template_sha256 {str(got)[:12]} is not the shipped template's ({shipped[:12]}): the record was not written by it")
    cfg = record.get("config")
    cfg_path = stage_dir / CONFIG_NAME
    if not isinstance(cfg, dict) or cfg.get("sha256") != (_sha(cfg_path) if cfg_path.is_file() else None):
        reasons.append(f"{CONFIG_NAME} is not the file the record names (missing, or changed after the test)")


def _source_reasons(record: dict, stage_dir: Path, run_dir: Path, reasons: list) -> None:
    sources = record.get("port_sources")
    if not isinstance(sources, list) or not sources:
        reasons.append("port_sources must list the port's source files (stages/2/port/)")
        return
    texts, listed = [], set()
    for i, s in enumerate(sources):
        path = s.get("path") if isinstance(s, dict) else None
        f = _inside(run_dir, path) if isinstance(path, str) and path.startswith(PORT_DIR) else None
        if f is None or s.get("role") != MODEL_ROLE:
            reasons.append(f"port_sources[{i}] needs a file under {PORT_DIR} and a role of {MODEL_ROLE!r}")
            continue
        listed.add(path)
        if s.get("sha256") != _sha(f):
            reasons.append(f"port_sources[{i}] {path} changed after the record was written (sha256 differs)")
        try:
            texts.append((path, safeio.read_bytes(f, max_bytes=SOURCE_MAX).decode("utf-8", "replace")))
        except safeio.SafeReadError:
            reasons.append(f"port_sources[{i}] {path} is not readable")
    if not texts:
        reasons.append(f"port_sources must include at least one readable file with the role {MODEL_ROLE!r}")
        return
    if PORT_MODULE not in listed:
        reasons.append(f"port_sources does not list {PORT_MODULE}, the port module the template calls")
    links = [os.path.join(r, n) for r, ds, fs in os.walk(stage_dir / "port") for n in ds + fs if os.path.islink(os.path.join(r, n))]
    if links:
        reasons.append(f"{os.path.relpath(links[0], run_dir)} is a link under {PORT_DIR}: its target is not hashed or read by the checks, so files there must be regular files")
    unhashed = [os.path.join(r, n) for r, ds, fs in os.walk(stage_dir / "port") for n in ds + fs
                if n == "__pycache__" or n.endswith(UNHASHED_SUFFIXES)]
    if unhashed:
        reasons.append(f"{os.path.relpath(unhashed[0], run_dir)} is importable code that port_sources does not hash (bytecode, a compiled module "
                       "or a path file): cached bytecode could differ from the .py file the checks read")
    on_disk = {os.path.relpath(f, run_dir) for f in (stage_dir / "port").rglob("*.py") if f.is_file()}
    if on_disk - listed:
        reasons.append(f"{sorted(on_disk - listed)[0]} is in {PORT_DIR} but not in port_sources: it was added after the record was written")
    if not any("ttnn" in t for _, t in texts):
        reasons.append("no model source in port_sources mentions ttnn, so nothing shows the port runs on the chip")
    for path, t in texts:
        if "scoring-reference" in t or "scoring-pairs" in t or "evidence/reference" in t:
            reasons.append(f"model source {path} refers to the stage 1 reference files; model code does not read the reference, and one that does could copy its scores")
