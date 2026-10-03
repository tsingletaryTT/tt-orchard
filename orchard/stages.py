"""The stage table, the exit gates and the replay that drives the stage machine (spec sections 5, 10).

This module owns what each stage is: its number, owner skill, the boards its hardware test needs,
its budget and free-disk need (orchard/defaults.py), the file its exit gate reads, the gate itself
and its resume marker. It also owns the questions the supervisor asks the ledger: which stage
runs next, whether the run is paused, how many escalations and coder starts it has used, and
where the coder's lease is recorded. Nothing here starts a process or calls a model.

The table holds one spec per stage. The path stage 0 chose can replace a spec: on the weights-only
path stage 2 uses the weights-swap-check skill and `gate_weights_swap`, stage 3 is skipped, and
stage 4 runs one hardware test per chip configuration (`WEIGHTS_ONLY_STAGE_4`, `gate_mesh_swap`)
(`spec_for`, `run_path`).

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
import dataclasses
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
                              RUN_WALL_CLOCK_S, STAGE2_PCC_MIN, STAGE4_SWAP_DISK_GB, STAGE_BUDGET_S,
                              STAGE_DISK_GB, SWAP_MIN_TOKENS, SWAP_TOP1_MIN)
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
    tests: bool = False                 # the hardware phase runs a list of tests (hw_tests.json,
                                        # orchard/hwtests.py) in place of one hw_test.json
    disk: float | None = None           # free disk the stage needs, when it differs from STAGE_DISK_GB

    @property
    def budget_s(self) -> float:
        return STAGE_BUDGET_S[self.number]

    @property
    def disk_gb(self) -> float:
        return self.disk if self.disk is not None else STAGE_DISK_GB[self.number]


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


def gate_weights_swap(stage_dir, run_dir) -> GateResult:
    """Stage 2 on the weights-only path: the existing TT implementation serves the new weights,
    and its greedy tokens agree with the stage 1 CPU reference (the weights-swap-check skill).

    It reads result.json as that skill describes. Each failing field gets its own reason, which
    names the field. After a failed hardware test the skill writes `serves` false with a `failure`
    field. The gate fails that result, and its serves reason quotes the failure text.
    """
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = _swap_reasons(d), []
    _evidence(run_dir, d.get("evidence"), "result.json", reasons, seen)
    return _done(reasons, seen)


def _swap_reasons(d: dict, where: str = "") -> list[str]:
    """The swap fields of one result (stage 2's result.json, or one stage 4 configuration), checked
    against the stage 2 bar. Each failing field gets its own reason, which names the field."""
    reasons = []
    if d.get("serves") is not True:
        reason = f"{where}serves must be true (the server started and answered), got {d.get('serves')!r}"
        failure = d.get("failure")
        if isinstance(failure, str) and failure.strip():
            # The finish step records a failed hardware test here. The reason carries it so the
            # escalation and the operator see why the test failed.
            reason += f"; the recorded failure: {failure.strip()[:500]}"
        reasons.append(reason)
    if d.get("coherent") is not True:
        reasons.append(f"{where}coherent must be true (the free-run text is readable), got {d.get('coherent')!r}")
    n = d.get("n_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or n < SWAP_MIN_TOKENS:
        reasons.append(f"{where}n_tokens must be a whole number of at least {SWAP_MIN_TOKENS}, got {n!r}")
    top1 = d.get("top1_agreement")
    if not _number(top1) or not 0 <= top1 <= 1:
        reasons.append(f"{where}top1_agreement must be a fraction from 0 to 1, got {top1!r}")
    elif top1 < SWAP_TOP1_MIN:
        reasons.append(f"{where}top1_agreement {top1} is below the minimum of {SWAP_TOP1_MIN}")
    ready = d.get("server_ready_s")
    if not _number(ready) or ready <= 0:
        reasons.append(f"{where}server_ready_s must be a positive number of seconds, got {ready!r}")
    return reasons


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


def gate_mesh(stage_dir, run_dir, required=None) -> GateResult:
    """Stage 4: evidence for each mesh configuration, as mesh-shrink requires.

    `required` is the tuple of chip counts the operator named for this run. Without it, every
    listed configuration must pass. With it:
    - each required count needs at least one entry, and every entry for a required count must
      pass, so a passing duplicate cannot hide a failing one;
    - an entry for any other count must still be well formed, and one that claims a pass must
      carry valid evidence, but an optional entry that did not pass adds no failure reason.
    """
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    configs = d.get("configs")
    if not isinstance(configs, list) or not configs:
        reasons.append("result.json configs must be a non-empty list")
        configs = []
    required = tuple(required or ())
    listed = set()
    for i, c in enumerate(configs):
        if not isinstance(c, dict) or isinstance(c.get("chips"), bool) or not isinstance(c.get("chips"), int):
            reasons.append(f"configs[{i}] needs an integer chips")
            continue
        listed.add(c["chips"])
        optional = bool(required) and c["chips"] not in required
        passed = c.get("pass") is True
        if not passed and not optional:
            reasons.append(f"the {c['chips']}-chip configuration did not pass")
        if passed or not optional:
            _evidence(run_dir, c.get("evidence"), f"configs[{i}]", reasons, seen)
    for chips in required:
        if chips not in listed:
            reasons.append(f"the {chips}-chip configuration is required and has no entry")
    return _done(reasons, seen)


def hw_record_path(stage_dir, chips: int) -> Path:
    """Where the supervisor records one configuration's hardware test (orchard/hwtests.py). Only
    the supervisor writes this file."""
    return Path(stage_dir) / "tests" / str(chips) / "test-result.json"


def hw_record_problem(stage_dir, chips: int) -> str | None:
    """Why the supervisor's record does not show a finished test on `chips` chips, or None."""
    path = hw_record_path(stage_dir, chips)
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"the supervisor has no record of its test (tests/{chips}/test-result.json)"
    except (OSError, ValueError) as exc:
        return f"tests/{chips}/test-result.json is not readable JSON: {exc}"
    if not isinstance(rec, dict):
        return f"tests/{chips}/test-result.json is not a JSON object"
    if rec.get("timed_out") is not False:
        return f"its test did not finish before its deadline (timed_out {rec.get('timed_out')!r})"
    if rec.get("returncode") != 0:
        return f"its test exited {rec.get('returncode')!r}"
    if not isinstance(rec.get("chips"), list) or len(rec["chips"]) != chips:
        return f"its test ran on {rec.get('chips')!r}, which is not {chips} chips"
    return None


def gate_mesh_swap(stage_dir, run_dir, required=None) -> GateResult:
    """Stage 4 on the weights-only path: `gate_mesh`, and for every configuration that claims a
    pass, proof that the supervisor ran its test and that the test measured a passing swap.

    For a passing entry with N chips:
    - tests/N/test-result.json must show an exit code of 0, no timeout and N chips. Only the
      supervisor writes it, so a configuration whose test never ran cannot pass.
    - the entry must hold the stage 2 fields and meet the stage 2 bar (`_swap_reasons`).
    - its evidence must include configs/N/evidence/swap-check.json, and that file's result_draft
      must hold the same top1_agreement.
    An entry that does not claim a pass is judged by `gate_mesh` alone.
    """
    base = gate_mesh(stage_dir, run_dir, required)
    d, err = _load(stage_dir, "result.json")
    if err:
        return base
    reasons, seen = list(base.reasons), list(base.evidence)
    rel = os.path.relpath(os.path.realpath(stage_dir), os.path.realpath(run_dir))
    for c in d.get("configs") if isinstance(d.get("configs"), list) else []:
        if (not isinstance(c, dict) or c.get("pass") is not True or isinstance(c.get("chips"), bool)
                or not isinstance(c.get("chips"), int)):
            continue
        n = c["chips"]
        where = f"the {n}-chip configuration"
        problem = hw_record_problem(stage_dir, n)
        if problem:
            reasons.append(f"{where} claims a pass, but {problem}")
        else:
            seen.append(f"{rel}/tests/{n}/test-result.json")
        reasons += _swap_reasons(c, f"{where}: ")
        swap = f"{rel}/configs/{n}/evidence/swap-check.json"
        if swap not in (c.get("evidence") if isinstance(c.get("evidence"), list) else []):
            reasons.append(f"{where}: evidence must include {swap}")
            continue
        try:
            draft = json.loads(inside(run_dir, swap).read_text(encoding="utf-8"))["result_draft"]
        except (AttributeError, OSError, ValueError, KeyError, TypeError):
            draft = None
        if not isinstance(draft, dict) or draft.get("top1_agreement") != c.get("top1_agreement"):
            reasons.append(f"{where}: top1_agreement {c.get('top1_agreement')!r} is not the "
                           f"result_draft's in {swap}")
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


# ---- the path stage 0 chose ---------------------------------------------------------------------
# Stage 0 writes "path" in delta.json: "weights-only" when the new model has the nearest model's
# architecture and only the weights differ, "full-port" when it needs new model code. On the
# weights-only path stage 2 serves the new weights with the existing TT implementation and compares
# the chip's tokens with the stage 1 reference (the weights-swap-check skill). Any other path,
# including one the supervisor cannot read, keeps the table above.

WEIGHTS_ONLY_STAGE_2 = dataclasses.replace(
    STAGES[2], name="weights swap check on one board", skill="weights-swap-check",
    gate=gate_weights_swap)


# Stage 3 builds and checks the full TT model. On the weights-only path that model already exists,
# and stage 2 has served the new weights through it and compared every token with the reference.
SKIP_3_WEIGHTS_ONLY = "weights-only path: the stage 2 serve-and-compare covers the full model"
WEIGHTS_ONLY_STAGE_3 = dataclasses.replace(STAGES[3], skip=SKIP_3_WEIGHTS_ONLY)

# Stage 4 shows the new weights working on each chip configuration the packages will ship for.
# Nothing is parallelised or shrunk: an existing package or bundle serves each configuration, so the
# stage runs one serve-and-compare test per configuration (the weights-swap-configs skill writes
# hw_tests.json; orchard/hwtests.py reads it) and `gate_mesh_swap` checks the result. `boards` is
# the most any one test needs (the 4-chip configuration needs both boards). The resume marker is
# the supervisor's copy of the validated test list, so a resumed stage skips the prepare step and
# goes on at the first configuration without a test record.
WEIGHTS_ONLY_STAGE_4 = dataclasses.replace(
    STAGES[4], name="weights swap on each chip configuration", skill="weights-swap-configs",
    refs=("tt-device-usage",), boards=2, gate=gate_mesh_swap, marker="tests/plan.json", tests=True,
    disk=STAGE4_SWAP_DISK_GB)


def delta_path(run_dir) -> str | None:
    """The path in stages/0/delta.json, or None when the file is missing, unreadable or names
    another value. The stage 0 gate validated the file; this only reads it."""
    try:
        data = json.loads((Path(run_dir) / "stages" / "0" / "delta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    path = data.get("path") if isinstance(data, dict) else None
    return path if path in PATHS else None


def run_path(entries: list[dict], run_dir) -> str | None:
    """The path this run follows: "weights-only", "full-port", or None when it is not known.

    The supervisor records the path in stage 0's passing stage_end. That entry decides, so a
    resumed run makes the same choice as the run that crashed, even if an agent has since edited
    delta.json. A ledger written before the path was recorded there falls back to delta.json.
    Before stage 0 passes there is no path.
    """
    end = None
    for e in entries:
        if e["event"] == "stage_end" and e["stage"] == 0 and e["data"].get("result") == "pass":
            end = e["data"]
    if end is None:
        return None
    if "path" in end:
        return end["path"] if end["path"] in PATHS else None
    return delta_path(run_dir)


def spec_for(number: int, path: str | None) -> StageSpec:
    """The stage spec for `number` on `path`. Only the weights-only path changes the table: stage 2
    gets the swap skill and gate, stage 3 is skipped (the supervisor records it as skipped, as it
    does stage 7), and stage 4 runs one test per chip configuration."""
    if path == "weights-only" and number == 2:
        return WEIGHTS_ONLY_STAGE_2
    if path == "weights-only" and number == 3:
        return WEIGHTS_ONLY_STAGE_3
    if path == "weights-only" and number == 4:
        return WEIGHTS_ONLY_STAGE_4
    return STAGES[number]


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


# ---- replaying the ledger -----------------------------------------------------------------------

@dataclass(frozen=True)
class RunProgress:
    started: bool
    run_start: dict | None
    clock_start: float | None           # run start, and again at each operator resume
    finished: bool                      # decision "ready for operator review"
    aborted: bool
    paused: str | None                  # the reason, while the latest pause has no resume after it
    done: tuple[int, ...]               # stages that passed or were skipped
    open_stage: int | None              # a stage_start with no stage_end after it
    escalated: frozenset[int]           # stages with an escalate entry
    escalations: int                    # since the last operator resume
    cold_boots: int                     # coder starts slower than COLD_START_S, since the last resume
    coder_deaths: int                   # since the last operator resume
    next_stage: int | None


def ledger_ts(entry: dict) -> float:
    return float(calendar.timegm(time.strptime(entry["ts"], "%Y-%m-%dT%H:%M:%SZ")))


def run_progress(entries: list[dict]) -> RunProgress:
    run_start = clock_start = None
    finished = aborted = False
    paused = None
    done: list[int] = []
    open_stage = None
    escalated: set[int] = set()
    escalations = cold_boots = coder_deaths = 0
    for e in entries:
        ev, stage, d = e["event"], e["stage"], e["data"]
        if ev == "run_start":
            run_start, clock_start = d, ledger_ts(e)
        elif ev == "stage_start":
            open_stage = stage
        elif ev == "stage_end":
            if stage == open_stage:
                open_stage = None
            if d.get("result") in ("pass", "skipped") and stage not in done:
                done.append(stage)
        elif ev == "escalate":
            escalated.add(stage)
            escalations += 1
        elif ev == "restore" and d.get("step") == "ready" and (d.get("seconds") or 0) >= COLD_START_S:
            cold_boots += 1
        elif ev == "decision":
            what = d.get("decision")
            if what == "pause":
                paused = d.get("reason") or "paused"
            elif what == "resume":
                paused, escalations, cold_boots, coder_deaths = None, 0, 0, 0
                clock_start = ledger_ts(e)
            elif what == "abort":
                aborted = True
            elif what == "ready for operator review":
                finished = True
            elif what == "coder started" and (d.get("ready_s") or 0) >= COLD_START_S:
                cold_boots += 1
            elif what == "coder died; restarting it once":
                coder_deaths += 1
    next_stage = next((s.number for s in STAGES if s.number not in done), None)
    return RunProgress(run_start is not None, run_start, clock_start, finished, aborted, paused,
                       tuple(done), open_stage, frozenset(escalated), escalations, cold_boots,
                       coder_deaths, next_stage)


def attempt_started_ts(entries: list[dict], stage: int) -> float | None:
    """When the current attempt at `stage` first started: the first stage_start after its last
    stage_end. A restart keeps the attempt's clock, so a crash does not reset the budget."""
    t = None
    for e in entries:
        if e["stage"] != stage:
            continue
        if e["event"] == "stage_end":
            t = None
        elif e["event"] == "stage_start" and t is None:
            t = ledger_ts(e)
    return t


def budget_cap(p: RunProgress, now: float) -> str | None:
    """Spec section 10: a run-wide cap that pauses the run, or None."""
    if p.escalations >= RUN_ESCALATION_CAP:
        return f"{p.escalations} escalations since the last resume (cap {RUN_ESCALATION_CAP})"
    if p.cold_boots >= RUN_COLD_BOOT_CAP:
        return f"{p.cold_boots} cold coder boots since the last resume (cap {RUN_COLD_BOOT_CAP})"
    if p.clock_start is not None and now - p.clock_start >= RUN_WALL_CLOCK_S:
        return (f"the run has lasted {int(now - p.clock_start)} s since it started or was last "
                f"resumed (cap {int(RUN_WALL_CLOCK_S)} s)")
    return None


def coder_state(entries: list[dict]) -> tuple[dict | None, dict | None, dict | None]:
    """The coder's latest lease record, server record and baseline canary, from the ledger.

    Park and restore entries carry the lease (orchard/handoff.py). The supervisor's own
    "coder starting" and "coder started" decisions carry it too.
    """
    lease = server = canary = None
    for e in entries:
        d = e["data"]
        own = e["event"] == "decision" and d.get("decision") in ("coder starting", "coder started")
        if e["event"] in ("park", "restore") or own:
            if isinstance(d.get("lease"), dict):
                lease = d["lease"]
            if isinstance(d.get("server"), dict):
                server = d["server"]
            if own and isinstance(d.get("canary"), dict):
                canary = d["canary"]
    return lease, server, canary


def open_stage_dir(run_dir, spec: StageSpec, *, resuming: bool, ledger) -> tuple[Path, bool]:
    """The stage directory for this attempt, and whether it resumes from the stage's marker.

    A resumed stage with its marker present keeps its directory. Otherwise an existing directory
    is moved aside to `<n>.partial-<k>` (never deleted) and a fresh one is made (spec section 10).
    """
    d = Path(run_dir) / "stages" / str(spec.number)
    if d.exists():
        marker = d / spec.marker if spec.marker else None
        if resuming and marker is not None and marker.is_file():
            ledger.append("decision", spec.number, decision="resume from marker",
                          marker=evidence_record(run_dir, marker))
            return d, True
        k = 1
        while (aside := d.with_name(f"{spec.number}.partial-{k}")).exists():
            k += 1
        os.rename(d, aside)
        ledger.append("decision", spec.number, decision="moved the partial stage directory aside",
                      path=os.path.relpath(aside, run_dir))
    (d / "evidence").mkdir(parents=True)
    return d, False


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
