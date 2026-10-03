"""The stage table, the exit gates and the replay that drives the stage machine (spec sections 5, 10).

This module owns what each stage is: its number, owner skill, the boards its hardware test needs,
its budget and free-disk need (orchard/defaults.py), the file its exit gate reads, the gate itself
and its resume marker. It also owns the questions the supervisor asks the ledger: which stage
runs next, whether the run is paused, how many escalations and coder starts it has used, and
where the coder's lease is recorded. Nothing here starts a process or calls a model.

A gate checks the shape of a stage's result file and that every evidence path it lists is a file
inside the run directory. A gate cannot tell whether a claim is true. The operator reviews the
bundle; the spec's `stage-review` skill is not wired in by plan 4.

Skills are referenced by name. `resolve_skill` finds `<dir>/<name>.md` or `<dir>/<name>/SKILL.md`
in the directories the run is given, so the existing tt-model-bringup skills are used where they
are installed and are not copied here.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from orchard.defaults import (COLD_START_S, LONG_STAGE_S, RUN_COLD_BOOT_CAP, RUN_ESCALATION_CAP,
                              RUN_WALL_CLOCK_S, STAGE2_PCC_MIN, STAGE_BUDGET_S, STAGE_DISK_GB)
from orchard.tiers import TierConfig


class TierUnavailable(Exception):
    """No tier that can serve this stage answers."""


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reasons: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()      # run-directory-relative paths the gate relied on


@dataclass(frozen=True)
class StageSpec:
    number: int
    name: str
    skill: str                          # owner skill, by name
    refs: tuple[str, ...]               # related existing skills, by name
    boards: int                         # boards the hardware test needs; 0 means no hardware phase
    gate_file: str | None               # the file in the stage directory the gate reads
    gate: Callable[[Path, Path], GateResult] | None
    marker: str | None                  # resume marker: a file in the stage directory
    skip: str | None = None             # why plan 4 skips this stage

    @property
    def budget_s(self) -> float:
        return STAGE_BUDGET_S[self.number]

    @property
    def disk_gb(self) -> float:
        return STAGE_DISK_GB[self.number]


# ---- evidence and gate helpers ------------------------------------------------------------------

def evidence_record(run_dir, path) -> dict:
    """The path (relative to the run directory) and sha256 that an evidence entry records.

    Relative paths keep the machine's home directory out of the ledger's evidence lists.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(run_dir))
    return {"path": rel, "sha256": h.hexdigest()}


def inside(run_dir, rel) -> Path | None:
    """The file `rel` names, if it is a regular file inside the run directory (links resolved)."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        return None
    root = os.path.realpath(run_dir)
    p = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, p]) != root or not os.path.isfile(p):
        return None
    return Path(p)


def _load(stage_dir, name: str) -> tuple[dict | None, str | None]:
    path = Path(stage_dir) / name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"{name} is missing"
    except (OSError, ValueError) as exc:
        return None, f"{name} is not readable JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{name} is not a JSON object"
    return data, None


def _evidence(run_dir, items, where: str, reasons: list, seen: list) -> None:
    if not isinstance(items, list) or not items:
        reasons.append(f"{where} lists no evidence")
        return
    for rel in items:
        if inside(run_dir, rel) is None:
            reasons.append(f"{where}: evidence {rel!r} is not a file inside the run directory")
        else:
            seen.append(rel)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _done(reasons: list, seen: list) -> GateResult:
    return GateResult(not reasons, tuple(reasons), tuple(seen))


# ---- the gates (one per stage) ------------------------------------------------------------------

DELTA_AREAS = ("config", "architecture", "tensors", "tensor_names", "tokenizer", "chat_template",
               "files", "generation_config", "license", "other")
HAZARD_AREAS = ("tensor_cache", "drafter", "disk", "license", "other")
PATHS = ("weights-only", "full-port")


def gate_delta(stage_dir, run_dir) -> GateResult:
    """Stage 0: a written list of what differs from the nearest supported model."""
    d, err = _load(stage_dir, "delta.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    for key in ("model", "nearest_model"):
        if not _text(d.get(key)):
            reasons.append(f"delta.json needs {key}")
    if d.get("path") not in PATHS:
        reasons.append(f"delta.json path must be one of {PATHS}")
    diffs = d.get("differences")
    if not isinstance(diffs, list) or not diffs:
        reasons.append("delta.json differences must be a non-empty list")
        diffs = []
    for i, item in enumerate(diffs):
        if not isinstance(item, dict) or item.get("area") not in DELTA_AREAS or not _text(item.get("finding")):
            reasons.append(f"differences[{i}] needs an area from {DELTA_AREAS} and a finding")
            continue
        _evidence(run_dir, item.get("evidence"), f"differences[{i}]", reasons, seen)
    hazards = d.get("hazards", [])
    if not isinstance(hazards, list):
        reasons.append("delta.json hazards must be a list")
        hazards = []
    for i, item in enumerate(hazards):
        if not isinstance(item, dict) or item.get("area") not in HAZARD_AREAS or not _text(item.get("finding")):
            reasons.append(f"hazards[{i}] needs an area from {HAZARD_AREAS} and a finding")
    return _done(reasons, seen)


def gate_reference(stage_dir, run_dir) -> GateResult:
    """Stage 1: the CPU reference reproduces the model card's published behavior."""
    d, err = _load(stage_dir, "reference.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    if d.get("verdict") != "pass":
        reasons.append("reference.json verdict is not 'pass'")
    checks = d.get("checks")
    if not isinstance(checks, list) or not checks:
        reasons.append("reference.json checks must be a non-empty list")
        checks = []
    for i, c in enumerate(checks):
        if not isinstance(c, dict) or not _text(c.get("name")):
            reasons.append(f"checks[{i}] needs a name")
            continue
        if c.get("pass") is not True:
            reasons.append(f"check {c['name']!r} did not pass")
        _evidence(run_dir, c.get("evidence"), f"check {c['name']!r}", reasons, seen)
    return _done(reasons, seen)


def gate_decoder(stage_dir, run_dir) -> GateResult:
    """Stage 2: PCC at or above the functional-decoder bar, and argmax agreement."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    if not _number(d.get("pcc")) or d["pcc"] < STAGE2_PCC_MIN:
        reasons.append(f"pcc must be a number at or above {STAGE2_PCC_MIN}, got {d.get('pcc')!r}")
    if d.get("argmax_match") is not True:
        reasons.append("argmax_match is not true")
    _evidence(run_dir, d.get("evidence"), "result.json", reasons, seen)
    return _done(reasons, seen)


def gate_full_model(stage_dir, run_dir) -> GateResult:
    """Stage 3: end-to-end parity with the reference, with the top-1 agreement recorded."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    if d.get("parity") is not True:
        reasons.append("parity is not true")
    if not _number(d.get("top1")) or not 0 <= d["top1"] <= 1:
        reasons.append(f"top1 must be a fraction from 0 to 1, got {d.get('top1')!r}")
    _evidence(run_dir, d.get("evidence"), "result.json", reasons, seen)
    return _done(reasons, seen)


def gate_mesh(stage_dir, run_dir) -> GateResult:
    """Stage 4: evidence for each mesh configuration, as mesh-shrink requires."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    configs = d.get("configs")
    if not isinstance(configs, list) or not configs:
        reasons.append("result.json configs must be a non-empty list")
        configs = []
    for i, c in enumerate(configs):
        if not isinstance(c, dict) or isinstance(c.get("chips"), bool) or not isinstance(c.get("chips"), int):
            reasons.append(f"configs[{i}] needs an integer chips")
            continue
        if c.get("pass") is not True:
            reasons.append(f"the {c['chips']}-chip configuration did not pass")
        _evidence(run_dir, c.get("evidence"), f"configs[{i}]", reasons, seen)
    return _done(reasons, seen)


SERVING_CHECKS = ("boots", "passkey", "canary")


def gate_serving(stage_dir, run_dir) -> GateResult:
    """Stage 5: the black-box server checks pass (boot, passkey or needle, canary)."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    checks = d.get("checks") if isinstance(d.get("checks"), dict) else {}
    for name in SERVING_CHECKS:
        c = checks.get(name)
        if not isinstance(c, dict):
            reasons.append(f"checks.{name} is missing")
            continue
        if c.get("pass") is not True:
            reasons.append(f"checks.{name} did not pass")
        _evidence(run_dir, c.get("evidence"), f"checks.{name}", reasons, seen)
    return _done(reasons, seen)


def gate_numbers(stage_dir, run_dir) -> GateResult:
    """Stage 6: every number labelled measured or TODO; measured ones carry evidence."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    numbers = d.get("numbers")
    if not isinstance(numbers, list) or not numbers:
        reasons.append("result.json numbers must be a non-empty list")
        numbers = []
    measured = 0
    for i, n in enumerate(numbers):
        if not isinstance(n, dict) or not _text(n.get("name")) or not _text(n.get("unit")):
            reasons.append(f"numbers[{i}] needs a name and a unit")
            continue
        if n.get("label") == "measured":
            measured += 1
            if not _number(n.get("value")):
                reasons.append(f"number {n['name']!r} is labelled measured and has no numeric value")
            _evidence(run_dir, n.get("evidence"), f"number {n['name']!r}", reasons, seen)
        elif n.get("label") != "TODO":
            reasons.append(f"number {n['name']!r} must be labelled 'measured' or 'TODO'")
    if numbers and not measured:
        reasons.append("no number is measured")
    qual = d.get("qualitative")
    if not isinstance(qual, dict):
        reasons.append("result.json qualitative is missing")
    else:
        _evidence(run_dir, qual.get("evidence"), "qualitative", reasons, seen)
    return _done(reasons, seen)


BUNDLE_FILES = ("RESULTS.md", "RISKS.md", "PUBLISH_COMMANDS.txt", "ledger.jsonl")


def gate_bundle(stage_dir, run_dir) -> GateResult:
    """Stage 8: results, ledger, open risks, publish commands as text, and a clean scrub."""
    from orchard.scrub import scrub_bundle
    bundle = Path(stage_dir) / "bundle"
    reasons, seen = [], []
    for name in BUNDLE_FILES:
        f = bundle / name
        if not f.is_file() or not f.read_text(encoding="utf-8", errors="replace").strip():
            reasons.append(f"bundle/{name} is missing or empty")
        else:
            seen.append(os.path.relpath(f, run_dir))
    for hit in scrub_bundle(bundle):
        reasons.append(f"scrub: {hit}")
    return _done(reasons, seen)


SKIP_7 = ("plan 4 builds no package or container image; the operator bundle reports the "
          "package as not built")

STAGES: tuple[StageSpec, ...] = (
    StageSpec(0, "intake and delta triage", "delta-triage", ("model-bringup",), 0,
              "delta.json", gate_delta, "RESUME.md"),
    StageSpec(1, "environment and CPU reference", "reference-gate", ("model-bringup",), 0,
              "reference.json", gate_reference, "RESUME.md"),
    StageSpec(2, "functional decoder on one chip", "functional-decoder", ("tt-device-usage",), 1,
              "result.json", gate_decoder, "test-result.json"),
    StageSpec(3, "full model", "full-model", ("tt-device-usage",), 1,
              "result.json", gate_full_model, "test-result.json"),
    StageSpec(4, "multichip, then shrink to fewer chips", "mesh-shrink",
              ("multichip", "tt-device-usage"), 1, "result.json", gate_mesh, "test-result.json"),
    StageSpec(5, "serving integration", "serving-check", ("vllm-integration", "tt-device-usage"), 1,
              "result.json", gate_serving, "test-result.json"),
    StageSpec(6, "qualitative check and benchmark", "serving-check",
              ("qualitative-check", "benchmark-model"), 1, "result.json", gate_numbers,
              "test-result.json"),
    StageSpec(7, "package and container build", "", (), 0, None, None, None, skip=SKIP_7),
    StageSpec(8, "operator bundle", "operator-bundle", (), 0, "bundle/RESULTS.md", gate_bundle, None),
)


def validate_table(stages=STAGES) -> None:
    """Refuse a table the spec forbids: gaps, and long stages with no resume marker."""
    if [s.number for s in stages] != list(range(9)):
        raise ValueError("the stage table must list stages 0 to 8 in order")
    for s in stages:
        if s.skip:
            continue
        if s.gate is None or not s.gate_file or not s.skill:
            raise ValueError(f"stage {s.number} needs a skill, a gate file and a gate")
        if s.budget_s > LONG_STAGE_S and not s.marker:
            raise ValueError(f"stage {s.number} has a budget of {s.budget_s} s and declares no "
                             "resume marker (spec section 10)")


validate_table()


# ---- tiers, skills and disk ---------------------------------------------------------------------

def tier_for(cfg: TierConfig, stage: int, *, phase: str, escalated: bool) -> str:
    """The tier for one agent step. Escalated: the stage's diagnose tier, else [escalation] default.

    A stage with a `plan` tier (stage 4) prepares with it: large plans, small runs.
    """
    entry = cfg.stages[stage]
    if entry["run"] == "none":
        raise ValueError(f"stage {stage} loads no model")
    if escalated:
        return entry.get("diagnose", cfg.escalation)
    if phase == "prepare" and "plan" in entry:
        return entry["plan"]
    return entry["run"]


def resolve_endpoint(cfg: TierConfig, tier: str, probe) -> tuple[str, str, str | None]:
    """(tier used, endpoint, note). `probe(endpoint, model)` says whether a server answers.

    The large and small tiers can name the same model on different chip counts. Only one of them
    can be loaded at a time on this box, so when the named tier is down and another chip tier with
    the same model answers, that one serves the step and the note says so.
    """
    t = cfg.tiers[tier]
    if probe(t["endpoint"], t["model"]):
        return tier, t["endpoint"], None
    for name, other in cfg.tiers.items():
        if (name != tier and other["model"] == t["model"] and other["placement"] == t["placement"]
                and probe(other["endpoint"], other["model"])):
            return name, other["endpoint"], (f"tier {tier} is not serving; tier {name} serves the "
                                             f"same model {t['model']}")
    raise TierUnavailable(f"tier {tier} ({t['model']} at {t['endpoint']}) is not serving")


def resolve_skill(name: str, dirs) -> Path | None:
    for d in dirs:
        for candidate in (Path(d) / f"{name}.md", Path(d) / name / "SKILL.md"):
            if candidate.is_file():
                return candidate
    return None


def check_disk(path, need_gb: float, usage=shutil.disk_usage) -> tuple[bool, float]:
    free_gb = usage(path).free / 1e9
    return free_gb >= need_gb, round(free_gb, 1)


# ---- stage 0 compared with a reference answer ---------------------------------------------------

# Labels used in the hand-written Hemmingway-1 reference (`N. Label: ...`) and the delta area each
# one is about. A label not listed here makes the comparison refuse, so a new reference cannot be
# half-read.
REFERENCE_AREAS = {"text config": "config", "tensors": "tensors", "tensor name prefix": "tensor_names",
                   "tokenizer": "tokenizer", "files": "files", "generation config": "generation_config"}
# Hazard lines are matched by word, in this order ("disk" first: that line also says "caches").
HAZARD_WORDS = (("disk", "disk"), ("drafter", "drafter"), ("cache", "tensor_cache"))


def reference_items(md: str) -> tuple[list[str], list[str], str | None]:
    """The difference areas, hazard areas and expected path a reference answer lists."""
    diffs, hazards, path, section = [], [], None, None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("Differences from"):
            section = "diff"
        elif s.startswith("Hazards"):
            section = "hazard"
        elif s.startswith("Expected path:"):
            path = "weights-only" if "weights-only" in s else "full-port"
            section = None
        elif section == "diff" and (m := re.match(r"\d+\.\s+([^:]+):", s)):
            label = m.group(1).strip().lower()
            if label not in REFERENCE_AREAS:
                raise ValueError(f"reference item {label!r} has no delta area")
            diffs.append(REFERENCE_AREAS[label])
        elif section == "hazard" and s.startswith("- "):
            area = next((a for word, a in HAZARD_WORDS if word in s.lower()), None)
            if area is None:
                raise ValueError(f"reference hazard {s[:60]!r} has no hazard area")
            hazards.append(area)
    return diffs, hazards, path


def compare_delta(delta: dict, reference_md: str) -> dict:
    """Does a stage 0 delta cover every area the reference lists, and agree on the path?

    This compares areas and the path only. Whether each finding is right is for a person.
    """
    want_d, want_h, want_path = reference_items(reference_md)
    have_d = {i.get("area") for i in delta.get("differences") or [] if isinstance(i, dict)}
    have_h = {i.get("area") for i in delta.get("hazards") or [] if isinstance(i, dict)}
    missing = ([f"difference: {a}" for a in want_d if a not in have_d]
               + [f"hazard: {a}" for a in want_h if a not in have_h])
    matches = delta.get("path") == want_path
    return {"ok": not missing and matches, "missing": missing, "path": delta.get("path"),
            "reference_path": want_path, "path_matches": matches}


# ---- command line: compare a stage 0 delta with a reference answer ------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m orchard.stages")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare-delta", help="compare a stage 0 delta.json with a reference answer")
    c.add_argument("delta")
    c.add_argument("reference")
    a = p.parse_args(argv)
    delta = json.loads(Path(a.delta).read_text(encoding="utf-8"))
    out = compare_delta(delta, Path(a.reference).read_text(encoding="utf-8"))
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
