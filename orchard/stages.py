# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage table, the exit gates and the replay that drives the stage machine (spec sections 5, 10).

This module owns what each stage is: its number, owner skill, the boards its hardware test needs,
its budget and free-disk need (orchard/defaults.py), the file its exit gate reads, the gate itself
and its resume marker. It also owns the questions the supervisor asks the ledger: which stage
runs next, whether the run is paused, how many escalations and coder starts it has used, and
where the coder's lease is recorded. Nothing here starts a process or calls a model.

The table holds one spec per stage. The path stage 0 chose can replace a spec: on the weights-only
path stage 2 uses the weights-swap-check skill and `gate_weights_swap`, stage 4 runs one hardware
test per chip configuration (`WEIGHTS_ONLY_STAGE_4`, `gate_mesh_swap`), and stages 3, 5 and 6 are
skipped (`spec_for`, `run_path`). On the weights-only path of a run started with --package-format
v6, stage 7 is `PACKAGE_STAGE_7`: supervisor code (orchard/package.py) does its work, with no agent,
and `gate_package` checks the package again from the files on disk (`package_format`).

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
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from orchard import audio_decision, audio_facts, audio_runtime, full_port, model_kind, safeio, scoring_check
from orchard import text_runtime as _text_rt
from orchard.classes import CLASSES, SIDECAR_CLASSES, class_of_path, path_of
from orchard import agreement_core, prompt_suite, sampling_choice, sampling_probe, token_check
from orchard.graft_trial import DECISION_FILE as GRAFT_DECISION_FILE
from orchard.graft_trial import chosen as graft_chosen
from orchard.graft_trial import read_decision as read_graft_decision
from orchard.graft_trial import decision_problems
from orchard.defaults import (STOCK_MIN_TOKENS, STOCK_SMALL_PARAMS_MAX_B, STOCK_SMALL_TOP1_MIN,
                              STOCK_SMALL_TOP5_MIN, STOCK_TOP1_MIN, STOCK_TOP5_MIN, SUITE_BLOCKING_KINDS, COLD_START_S, LONG_STAGE_S, PACKAGE_FORMATS, RUN_COLD_BOOT_CAP, RUN_PARK_BOOT_CAP,
                              RUN_ESCALATION_CAP, RUN_WALL_CLOCK_S, STAGE2_PCC_MIN,
                              STAGE4_SWAP_DISK_GB, STAGE_BUDGET_S, STAGE_DISK_GB, SWAP_MIN_TOKENS,
                              SWAP_TOP1_MIN, SIDECAR_MIN_QUESTIONS, SIDECAR_NOISE_FACTOR,
                              SIDECAR_PCC_MIN, SIDECAR_PROB_DIFF_MAX, SIDECAR_TOP1_MIN)
from orchard.tiers import TierConfig


class TierUnavailable(Exception):
    """No tier that can serve this stage answers."""


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reasons: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()      # run-directory-relative paths the gate relied on
    waived: tuple[str, ...] = ()        # fields an operator option waived, from the gate's own read (full-port rehearsal, issue 357)


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
    harness: bool = False               # supervisor code does the work; no agent and no model
    tests: bool = False                 # the hardware phase runs a list of tests (hw_tests.json,
                                        # orchard/hwtests.py) in place of one hw_test.json
    disk: float | None = None           # free disk the stage needs, when it differs from STAGE_DISK_GB
    draft: str | None = None            # a config the supervisor drafts before a fresh prepare step
                                        # ("swap_config": orchard/swap_draft.py)
    needs_coder: bool = True            # False: the stage never asks the local coder anything, so the supervisor does not boot it for the stage
                                        # (issue 143). Only the audio runtime's stages say so today.

    @property
    def budget_s(self) -> float:
        return STAGE_BUDGET_S[self.number]

    @property
    def disk_gb(self) -> float:
        return self.disk if self.disk is not None else STAGE_DISK_GB[self.number]


# ---- evidence and gate helpers ------------------------------------------------------------------

EVIDENCE_HASH_MAX = 512 * 1024 * 1024    # the largest file a ledger evidence entry will hash (an agent transcript is the largest in practice)


def _within(root: str, p: str) -> bool:
    return p == root or p.startswith(root.rstrip(os.sep) + os.sep)


def locate(run_dir, rel) -> tuple[Path | None, str | None]:
    """(path, None) for the regular file `rel` names inside the run directory (links resolved), or (None, why) with a named reason.

    This decides whether an evidence path an agent wrote is usable. It never raises: a path with a null byte,
    one that cannot be resolved, a link that leaves the run directory, a FIFO or device, a missing file and a file over
    `EVIDENCE_HASH_MAX` are each a reason a gate can show the agent (issue 335: a null byte made `os.path.realpath` raise ValueError
    out of the gate)."""
    if not isinstance(rel, str) or not rel:
        return None, "the path is not a non-empty string"
    if "\0" in rel:
        return None, "the path holds a null byte"
    if os.path.isabs(rel):
        return None, "the path is absolute"
    try:
        root = os.path.realpath(run_dir)
        p = os.path.realpath(os.path.join(root, rel))
        if not _within(root, p):
            return None, "the path leaves the run directory"
        st = os.stat(p)
    except FileNotFoundError:
        return None, "the file does not exist"
    except (OSError, ValueError):
        return None, "the path cannot be examined"
    if not stat.S_ISREG(st.st_mode):
        return None, "the path is not a regular file (a directory, FIFO or device)"
    if st.st_size > EVIDENCE_HASH_MAX:
        return None, f"the file is larger than {EVIDENCE_HASH_MAX} bytes"
    return Path(p), None


def inside(run_dir, rel) -> Path | None:
    """The file `rel` names, if it is a regular file inside the run directory (links resolved). `locate` says why when it is not."""
    return locate(run_dir, rel)[0]


def evidence_record(run_dir, path) -> dict:
    """The path (relative to the run directory) and sha256 that an evidence entry records.

    Relative paths keep the machine's home directory out of the ledger's evidence lists. The file is read through
    `safeio.sha256_file`: a FIFO, a device, an oversize file or a change during the read is a `safeio.SafeReadError` with a
    named reason (issue 335: a plain `open()` blocked on a FIFO). A link that stays inside the run directory is followed once, as
    the gates follow it; a link that leaves the run directory is refused. A caller records evidence only for a file its gate
    accepted, so the refusal fails the stage (Supervisor._end_stage).
    """
    target = safeio.resolved(path)
    root = safeio.resolved(run_dir)
    if not _within(root, target):
        raise safeio.PathRefused("the evidence file is not inside the run directory")
    rel = os.path.relpath(target, root)
    # Open beneath the run directory, one component at a time without following a link: a parent directory replaced by a link to somewhere
    # else after the path was resolved is refused, and the hash is never taken from a file outside the run directory.
    with safeio.pinned_root(root) as pin:
        digest = safeio.sha256_file(rel, max_bytes=EVIDENCE_HASH_MAX, anchor=pin.fd)
    return {"path": rel, "sha256": digest}


TEXT_MAX_BYTES = 16 * 1024 * 1024       # the largest card, script or bundle text file a gate will read
JSON_MAX_BYTES = 16 * 1024 * 1024       # the largest stage record or evidence file a gate will read (docs/safe-reading.md)


def _safe_json(path, *, max_bytes: int = JSON_MAX_BYTES, expect=None):
    """The JSON in `path`, read through orchard.safeio: a size cap, no FIFO or device, no NaN or Infinity, no repeated key, bounded
    nesting. Raises only safeio.SafeReadError."""
    return safeio.read_json(path, max_bytes=max_bytes, expect=expect)


def _is_missing(exc: "safeio.SafeReadError") -> bool:
    """True when `exc` says the file does not exist (as opposed to a refusal of what is there)."""
    return isinstance(exc, safeio.ReadFailed) and "ENOENT" in exc.reason


def _load(stage_dir, name: str) -> tuple[dict | None, str | None]:
    path = Path(stage_dir) / name
    try:
        data = _safe_json(path)
    except safeio.SafeReadError as exc:
        if _is_missing(exc):
            return None, f"{name} is missing"
        return None, f"{name} is not readable JSON: {exc.reason}"
    if not isinstance(data, dict):
        return None, f"{name} is not a JSON object"
    return data, None


def _evidence(run_dir, items, where: str, reasons: list, seen: list) -> None:
    if not isinstance(items, list) or not items:
        reasons.append(f"{where} lists no evidence")
        return
    for rel in items:
        found, why = locate(run_dir, rel)
        if found is None:
            reasons.append(f"{where}: evidence {rel!r} is not a file inside the run directory ({why})")
        else:
            seen.append(rel)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _done(reasons: list, seen: list) -> GateResult:
    return GateResult(not reasons, tuple(reasons), tuple(seen))


# ---- the gates (one per stage) ------------------------------------------------------------------

STOCK_OPTIMIZATIONS = ("accuracy", "performance")
STOCK_MIN_FREE_CHARS = 20          # choice: the shortest free-run text the gate will judge
DELTA_AREAS = ("config", "architecture", "tensors", "tensor_names", "tokenizer", "chat_template",
               "files", "generation_config", "license", "other")
HAZARD_AREAS = ("tensor_cache", "drafter", "drafter_graft", "disk", "license", "generation_config", "stop_tokens", "sampling_defaults", "chat_template", "other")
PATHS = ("weights-only", "full-port", "stock-runtime")
AUDIO_CLASSES = ("weights-only", "full-port", "unknown")     # the classes an audio model can take (orchard/audio_facts.py)


def gate_delta(stage_dir, run_dir) -> GateResult:
    """Stage 0: a written list of what differs from the nearest supported model."""
    d, err = _load(stage_dir, "delta.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    _delta_reasons(d, run_dir, reasons, seen, Path(stage_dir))
    reasons += _template_reasons(Path(stage_dir), "delta_triage.py")
    reasons += _stamp_reasons(d, "delta_triage.py", "delta.json")
    return _done(reasons, seen)


def _delta_reasons(d: dict, run_dir, reasons: list, seen: list, stage_dir: Path | None = None) -> None:
    """The checks of delta.json that do not depend on who wrote it (shared by the agent stage and the stock-runtime stage). `stage_dir` is the
    stage of a template-written delta.json: the audio decision is then held to the current template's stamp and the stage's triage configuration
    (orchard/audio_decision.py). The stock-runtime stage has no template and passes none."""
    for key in ("model", "nearest_model"):
        if not _text(d.get(key)):
            reasons.append(f"delta.json needs {key}")
    if d.get("path") not in PATHS:
        reasons.append(f"delta.json path must be one of {PATHS}")
    reasons += _class_reasons(d)
    if stage_dir is None:
        reasons += audio_decision.decision_problems(d)
    else:
        try:
            shipped = _sha256(SHIPPED_TEMPLATES["delta_triage.py"])
        except safeio.SafeReadError:
            shipped = None
        reasons += audio_decision.stage_problems(d, stage_dir, shipped)
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


def gate_delta_stock(stage_dir, run_dir) -> GateResult:
    """Stage 0 of a run with stock parameters (orchard/stock_runtime.py). Supervisor code wrote delta.json, so there is no
    template copy or stamp to check. The record has to support the class it names: `stock-runtime` needs equal configs and no layout
    problem, and any other class (the measurement said full-port) is accepted as an answer."""
    d, err = _load(stage_dir, "delta.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    _delta_reasons(d, run_dir, reasons, seen)
    rec = d.get("stock_runtime")
    if not isinstance(rec, dict):
        reasons.append("delta.json needs a stock_runtime record (stage 0 of a run with stock parameters writes one)")
        return _done(reasons, seen)
    if not _text(rec.get("name")):
        reasons.append("stock_runtime.name must be the tt-transformers model name")
    if d.get("class") == "stock-runtime":
        if rec.get("config_equal") is not True:
            reasons.append("class stock-runtime needs stock_runtime.config_equal to be true")
        if rec.get("keys_differing"):
            reasons.append(f"class stock-runtime needs stock_runtime.keys_differing to be empty, got {rec.get('keys_differing')!r}")
        if rec.get("layout_problems"):
            reasons.append(f"class stock-runtime needs stock_runtime.layout_problems to be empty, got {rec.get('layout_problems')!r}")
        if not _number(rec.get("params_b")) or rec["params_b"] <= 0:
            reasons.append(f"stock_runtime.params_b must be a positive number of billions, got {rec.get('params_b')!r}")
    return _done(reasons, seen)


def _class_reasons(d: dict) -> list[str]:
    """The class, sidecars and code files of delta.json. A delta.json with none of them is a run from before
    classes and passes. With a class, the path, the sidecars and the class must agree."""
    reasons = []
    if "class" in d:
        if d["class"] not in CLASSES:
            reasons.append(f"delta.json class must be one of {CLASSES}")
        elif path_of(d["class"]) != d.get("path"):
            reasons.append(f"delta.json class {d['class']!r} takes the path {path_of(d['class'])!r}, "
                           f"but its path is {d.get('path')!r}")
    if "facts" in d:
        reasons += ["delta.json " + r for r in audio_facts.facts_problems(d["facts"], required=("form", "modality"))]
        if (isinstance(d["facts"], dict) and d["facts"].get("modality") == "audio" and d.get("class") in CLASSES
                and d["class"] not in AUDIO_CLASSES):
            reasons.append(f"delta.json class {d['class']!r} is a text class; an audio model is {' or '.join(AUDIO_CLASSES)}")
    sidecars = d.get("sidecars", [])
    if not isinstance(sidecars, list):
        reasons.append("delta.json sidecars must be a list")
        sidecars = []
    for i, item in enumerate(sidecars):
        if not isinstance(item, dict) or not _text(item.get("file")):
            reasons.append(f"sidecars[{i}] needs a file")
        elif not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            reasons.append(f"sidecars[{i}] ({item['file']}) needs a sha256 of 64 hex digits")
    if d.get("class") in SIDECAR_CLASSES and not sidecars:
        reasons.append(f"class {d['class']} needs at least one entry in sidecars")
    if "class" in d and d["class"] not in SIDECAR_CLASSES and sidecars:
        reasons.append(f"class {d['class']!r} cannot have sidecars; only {' and '.join(SIDECAR_CLASSES)} do")
    if d.get("class") == "restricted-head":
        head = d.get("head")
        if not isinstance(head, dict) or head.get("restricted") is not True:
            reasons.append("class restricted-head needs delta.json head.restricted to be true "
                           "(the template's restricted-head check found the sidecar rows in the nearest model's lm_head)")
    vt = d.get("vocab_trim")
    if d.get("class") == "vocab-trim":
        if not isinstance(vt, dict):
            reasons.append("class vocab-trim needs a vocab_trim record in delta.json (stage 0's vocabulary check)")
        else:
            if vt.get("trimmed") is not True:
                reasons.append("class vocab-trim needs vocab_trim.trimmed to be true")
            if vt.get("kept_subset") is not True:
                reasons.append("class vocab-trim needs vocab_trim.kept_subset to be true "
                               "(every kept row must be a row of the nearest model's matrices)")
            sizes = (vt.get("model_vocab"), vt.get("base_vocab"))
            if any(isinstance(v, bool) or not isinstance(v, int) for v in sizes) or not sizes[0] < sizes[1]:
                reasons.append("class vocab-trim needs vocab_trim.model_vocab and base_vocab to be whole numbers, the model's smaller")
    elif vt is not None:
        reasons.append("delta.json has a vocab_trim record but its class is not vocab-trim")
    if "mtp" in d:
        m = d["mtp"]
        counts = (m.get("nearest_tensors"), m.get("new_tensors")) if isinstance(m, dict) else ()
        if len(counts) != 2 or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts):
            reasons.append('delta.json mtp must be {"nearest_tensors": N, "new_tensors": M} with whole numbers '
                           "of zero or more")
    code = d.get("code_files", [])
    if not isinstance(code, list):
        reasons.append("delta.json code_files must be a list")
        code = []
    for i, item in enumerate(code):
        if (not isinstance(item, dict) or not _text(item.get("file"))
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            reasons.append(f"code_files[{i}] needs a file and a sha256 of 64 hex digits")
    return reasons


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
    reasons += _reference_evidence_reasons(Path(stage_dir), d)
    reasons += _template_reasons(Path(stage_dir), "reference_gate.py")
    if (Path(stage_dir) / "qwen4_dense.py").is_file():          # the dense Qwen4 shim shapes the model the evidence describes
        reasons += _template_reasons(Path(stage_dir), "qwen4_dense.py")
    return _done(reasons, seen)


EVIDENCE_MAX_BYTES = 2 * 1024 * 1024        # the reference template's evidence files are a few KB to a few hundred KB

_SKILLS_ROOT = Path(__file__).with_name("skills")
# The template scripts whose output a gate trusts, and the shipped file each stage's copy must equal.
SHIPPED_TEMPLATES = {"delta_triage.py": _SKILLS_ROOT / "delta-triage-templates" / "delta_triage.py",
                     "stock_check.py": _SKILLS_ROOT / "stock-runtime-templates" / "stock_check.py",
                     "reference_gate.py": _SKILLS_ROOT / "reference-gate-templates" / "reference_gate.py",
                     # loaded by reference_gate.py from its own directory for a dense qwen4_exp config; optional in a stage
                     "qwen4_dense.py": _SKILLS_ROOT / "reference-gate-templates" / "qwen4_dense.py"}


def _sha256(path: Path) -> str:
    """The sha256 of a file through `safeio.sha256_file` (no link, no FIFO or device, a size cap). Raises only safeio.SafeReadError."""
    return safeio.sha256_file(path, max_bytes=EVIDENCE_HASH_MAX)


def _template_reasons(stage_dir: Path, name: str) -> list[str]:
    """A stage whose evidence comes from a shipped template is only as trustworthy as the script. If the stage's copy differs from the
    shipped file, the evidence is not the template's. On the pplx-decider run the stage 1 agent edited reference_gate.py (it put
    AutoModelForCausalLM in place of the class lookup), the edited script loaded none of the weights, and the gate accepted the result.
    A stage with no copy fails too: an agent could run an edited script under another name, or delete the copy after running it, and
    the gate would have nothing to compare. The evidence the template writes also carries the sha256 of the script that wrote it
    (`template_sha256`); `_stamp_reasons` compares that with the shipped file."""
    copy, shipped = stage_dir / name, SHIPPED_TEMPLATES[name]
    try:
        theirs = _sha256(shipped)
    except safeio.SafeReadError:
        return []                      # the shipped file is part of this repository; a gate cannot compare against what is not there
    if not os.path.lexists(copy):
        return [f"{name} is not in this stage directory. Copy it from the skill's templates directory and run it from there, "
                "so the gate can check that the evidence came from the shipped script."]
    try:
        mine = _sha256(copy)
    except safeio.SafeReadError as exc:
        # The copy is in the agent's directory. A link, a FIFO or a file over the cap is not the template, and says so.
        return [f"{name} in this stage cannot be compared with the shipped template: {exc.reason}. Copy {name} from the skill's "
                "templates directory again."]
    if mine == theirs:
        return []
    return [f"{name} in this stage differs from the shipped template (sha256 {mine[:12]} against {theirs[:12]}), so the evidence it wrote "
            f"is not the template's. Copy {name} from the skill's templates directory again and change only the config file."]


def _stamp_reasons(data: dict, name: str, where: str) -> list[str]:
    """`data` is a file the template named `name` wrote (delta.json or evidence/summary.json). It must carry the sha256 of the
    script that wrote it, equal to the shipped file's. This is best effort, because the agent can write the file too; it catches a
    run of an edited script under another name, and a copy deleted after the run, which the copy check alone cannot see."""
    try:
        theirs = _sha256(SHIPPED_TEMPLATES[name])
    except safeio.SafeReadError:
        return []
    got = data.get("template_sha256")
    if got == theirs:
        return []
    if not isinstance(got, str):
        return [f"{where} has no template_sha256, so it was not written by the shipped {name}. Run the copy from the skill's templates directory."]
    return [f"{where} template_sha256 {got[:12]} differs from the shipped {name} ({theirs[:12]}), so the evidence is not the template's."]


def _evidence_json(stage_dir: Path, name: str, *, required: bool, reasons: list):
    """evidence/<name> as a dict, or None. A missing file is a reason only when `required`; a file that is unreadable, too large
    or a link that leaves the stage directory is always one, because an agent that cannot change the verdict can still hide
    what the evidence says by swelling or redirecting the file."""
    rel = f"evidence/{name}"
    path = stage_dir / rel
    if not path.exists() and not path.is_symlink():
        if required:
            reasons.append(f"{rel} is missing: reference_gate.py was copied into this stage, so it must have written it")
        return None
    # A link that stays inside the stage directory is followed once here (safeio.resolved never raises, so a link loop
    # becomes a path that the read refuses); the read itself refuses links, FIFOs and devices and stops at the cap.
    target = safeio.resolved(path)
    base = safeio.resolved(stage_dir)
    if not (target == base or target.startswith(base.rstrip(os.sep) + os.sep)):
        reasons.append(f"{rel} is a link to a file outside the stage directory, so it was not read")
        return None
    try:
        data = _safe_json(target, max_bytes=EVIDENCE_MAX_BYTES)
    except safeio.TooLarge:
        reasons.append(f"{rel} is too large to read (the cap is {EVIDENCE_MAX_BYTES} bytes)")
        return None
    except safeio.SafeReadError as exc:
        reasons.append(f"{rel} is not readable JSON: {exc.reason}")
        return None
    if not isinstance(data, dict):
        reasons.append(f"{rel} is not a JSON object")
        return None
    return data


def _reference_evidence_reasons(stage_dir: Path, agent: dict) -> list[str]:
    """The reference template (orchard/skills/reference-gate-templates) writes evidence/summary.json (its own verdict and one
    boolean per check) and evidence/load-report.json (the keys the CPU model did not find in the checkpoint). The stage agent
    writes reference.json. The evidence decides: on the pplx-decider run the template said fail with all 851 model keys missing and
    the agent wrote pass, and this gate used to accept it."""
    reasons: list[str] = []
    used_template = (stage_dir / "reference_gate.py").is_file()
    summary = _evidence_json(stage_dir, "summary.json", required=used_template, reasons=reasons)
    report = _evidence_json(stage_dir, "load-report.json", required=False, reasons=reasons)
    if isinstance(summary, dict):
        reasons += _stamp_reasons(summary, "reference_gate.py", "evidence/summary.json")
        if summary.get("verdict") != "pass":
            reasons.append(f"evidence/summary.json (written by the reference template) says verdict {summary.get('verdict')!r}, "
                           f"but reference.json says {agent.get('verdict')!r}; the evidence decides")
        measured = summary.get("checks") if isinstance(summary.get("checks"), dict) else {}
        for c in agent.get("checks") or []:
            if isinstance(c, dict) and c.get("pass") is True and measured.get(c.get("name")) is False:
                reasons.append(f"check {c['name']!r}: evidence/summary.json says it failed, reference.json says it passed")
    if isinstance(report, dict) and report.get("dense_qwen4") is not None:
        # The model was built with the dense-layer shim. Its hash is in the load report whether or not the stage still has a copy.
        dense = report["dense_qwen4"]
        got = dense.get("shim_sha256") if isinstance(dense, dict) else None
        try:
            shipped = _sha256(SHIPPED_TEMPLATES["qwen4_dense.py"])
        except safeio.SafeReadError as exc:
            # The shipped shim is missing or refused: the report's hash cannot be checked, which is a reason and not a pass.
            shipped = None
            reasons.append(f"evidence/load-report.json says the model was built with the dense Qwen4 shim, but the shipped shim cannot be read to "
                           f"check its sha256 ({exc.reason})")
        if shipped is not None and got != shipped:
            reasons.append("evidence/load-report.json says the model was built with the dense Qwen4 shim, but it does not carry the shipped "
                           "shim's sha256 (qwen4_dense.py was edited, or the report was not written by the shipped reference_gate.py)")
    if isinstance(report, dict) and isinstance(report.get("missing_keys"), list):
        ignore = []
        for pattern in report.get("ignore_patterns") or []:
            try:
                ignore.append(re.compile(pattern))
            except (re.error, TypeError):
                continue
        missing = [k for k in report["missing_keys"] if isinstance(k, str) and not any(p.search(k) for p in ignore)]
        if missing:
            total = report.get("model_key_count")
            of = f" of {total}" if isinstance(total, int) and not isinstance(total, bool) else ""
            reasons.append(f"evidence/load-report.json: {len(missing)}{of} model keys were not found in the checkpoint, so those "
                           "weights did not load and the CPU model ran on uninitialised values")
    return reasons


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


def gate_full_port(stage_dir, run_dir, *, rehearsal: bool = False) -> GateResult:
    """Stage 2 of a full port (run started with --full-port): the port ran on the leased chips and its tokens agree with the stage 1
    reference. orchard/full_port.py holds the checks; each failing field has its own reason. `rehearsal` (the run's --full-port-rehearsal,
    read from the ledger by the supervisor) waives the `coherent` field only."""
    waived: list[str] = []
    reasons, seen = full_port.check(stage_dir, run_dir, rehearsal=rehearsal, waived=waived)
    return dataclasses.replace(_done(reasons, seen), waived=tuple(waived))


def gate_scoring_reference(stage_dir, run_dir) -> GateResult:
    """Stage 1 of a scoring model (a reranker or an embedding model, issue 378): the pair set and the Hugging Face scores or vectors that
    `python3 -m orchard.scoring_reference` writes under stages/1/evidence/reference/. The kind is the one the pair file names; the supervisor
    checks that it is the run's kind (`Supervisor._check_gate`)."""
    kind = None
    try:
        doc = _safe_json(Path(run_dir) / scoring_check.PAIRS_FILE, max_bytes=scoring_check.JSON_MAX, expect=dict)
        kind = doc.get("model_kind")
    except safeio.SafeReadError as exc:
        return GateResult(False, (f"{scoring_check.PAIRS_FILE} is not readable JSON: {exc.reason}",))
    reasons, seen = scoring_check.reference_reasons(run_dir, kind if isinstance(kind, str) else "")
    return _done(reasons, seen)


def gate_scoring(stage_dir, run_dir) -> GateResult:
    """Stage 2 of a scoring model (issue 378): the supervisor's own scoring check ran the port on the leased chip and its scores or vectors
    match the stage 1 reference within the bounds. orchard/scoring_check.py holds the checks; the kind is the config's (the supervisor
    passes the run's kind when it checks the gate itself)."""
    reasons, seen = scoring_check.check(stage_dir, run_dir)
    return _done(reasons, seen)


def _delta_refusal(run_dir) -> str | None:
    """A reason when stages/0/delta.json is there and the safe reader refuses it (NaN, a repeated key, over the cap, a link, a FIFO), else None.
    A missing file is not a refusal. The readers that take the class or the nearest model from delta.json answer None for a refused file, and
    a gate that asks "is this a vocabulary trim?" must not take that None as "no"."""
    try:
        _safe_json(Path(run_dir) / "stages" / "0" / "delta.json")
    except safeio.SafeReadError as exc:
        if not _is_missing(exc):
            return f"stage 0's delta.json is not readable JSON: {exc.reason}, so the run's class cannot be told and the vocabulary-trim check cannot be made"
    return None


def _trim_vocab(run_dir) -> int | None:
    """The model's vocabulary size when stage 0 classed the run vocab-trim, else None (no check is made)."""
    delta, err = _load(Path(run_dir) / "stages" / "0", "delta.json")
    if err or not isinstance(delta, dict) or delta.get("class") != "vocab-trim":
        return None
    vt = delta.get("vocab_trim")
    want = vt.get("model_vocab") if isinstance(vt, dict) else None
    return want if isinstance(want, int) and not isinstance(want, bool) else 0


def _trim_reasons(want: int, sources, serving, where: str = "") -> list[str]:
    """One served configuration of a vocabulary trim (issue 56). The served config.json must carry the model's vocab_size, as
    model-dir-sources.json (`sources`, None when unreadable) records, or the engine fails at load. The speculative drafter works on the
    nearest model's token ids, so swap-check.json's `serving` record must say it was off."""
    reasons = []
    vocab = sources.get("vocab") if isinstance(sources, dict) else None
    if sources is None:
        reasons.append(f"{where}this model is a vocabulary trim and model-dir-sources.json is missing or unreadable, so the served "
                       "config.json's vocab_size cannot be checked")
    elif not isinstance(vocab, dict) or vocab.get("trimmed") is not True or vocab.get("model") != want:
        reasons.append(f"{where}this model is a vocabulary trim: model-dir-sources.json must record that the served config.json carries "
                       f"vocab_size {want} (vocab.trimmed true, vocab.model {want}), got {vocab!r}")
    if not isinstance(serving, dict) or serving.get("drafter") != "off":
        reasons.append(f"{where}this model is a vocabulary trim: the speculative drafter works on the nearest model's token ids, so "
                       f"swap-check.json serving.drafter must be 'off', got {serving.get('drafter') if isinstance(serving, dict) else serving!r}")
    return reasons


def vocab_trim_problems(stage_dir, run_dir) -> list[str]:
    """Stage 2's check of a vocabulary trim (see _trim_reasons). Nothing is asked of any other class."""
    refused = _delta_refusal(run_dir)
    if refused:
        return [refused]
    want = _trim_vocab(run_dir)
    if want is None:
        return []
    sources, _ = _load(Path(stage_dir), "model-dir-sources.json")
    check, _ = _load(Path(stage_dir) / "evidence", "swap-check.json")
    return _trim_reasons(want, sources, check.get("serving") if isinstance(check, dict) else None)


def _stock_record(run_dir) -> dict | None:
    delta, err = _load(Path(run_dir) / "stages" / "0", "delta.json")
    rec = delta.get("stock_runtime") if not err and isinstance(delta, dict) and delta.get("class") == "stock-runtime" else None
    return rec if isinstance(rec, dict) else None


def _unit_number(value) -> bool:
    return _number(value) and 0 <= value <= 1        # a NaN or an infinity fails both comparisons


def gate_stock_runtime(stage_dir, run_dir) -> GateResult:
    """Stage 2 of the stock-runtime path: tt-transformers served the model on one board. The gate re-reads the template's
    evidence/stock-check.json and judges it again: teacher-forced top-1 and top-5 against the stage 1 reference at the bars in
    orchard/defaults.py (the small-model bars for a model of STOCK_SMALL_PARAMS_MAX_B billion parameters or fewer), enough compared
    tokens, a recorded device name and tt-metal tag, the optimization mode, a free run with no blocking finding, a positive decode
    speed, no model fetched from the Hub, and the scratch patch as evidence. The numbers come from the template; nothing here
    trusts a result file the agent wrote."""
    stage_dir = Path(stage_dir)
    rec = _stock_record(run_dir)
    if rec is None:
        return GateResult(False, ("stage 0 did not record the class stock-runtime with a stock record, so there is nothing to check this against",))
    reasons, seen = [], []
    reasons += _template_reasons(stage_dir, "stock_check.py")
    d = _evidence_json(stage_dir, "stock-check.json", required=True, reasons=reasons)
    if d is None:
        return _done(reasons, seen)
    seen.append("stages/2/evidence/stock-check.json")
    reasons += _stamp_reasons(d, "stock_check.py", "evidence/stock-check.json")
    delta, _ = _load(Path(run_dir) / "stages" / "0", "delta.json")
    want = str((delta or {}).get("model", "")).split("@", 1)[0]
    if str(d.get("model_id", "")).split("@", 1)[0] != want:
        reasons.append(f"model_id {d.get('model_id')!r} is not the model of this run ({want!r})")
    if not _text(d.get("device_name")):
        reasons.append("device_name must name the board type the runtime saw (P150, P300 ...)")
    if d.get("chips") not in (1, 2):
        reasons.append(f"chips must be 1 or 2 (one board), got {d.get('chips')!r}")
    if d.get("optimizations") not in STOCK_OPTIMIZATIONS:
        reasons.append(f"optimizations must be one of {STOCK_OPTIMIZATIONS}, got {d.get('optimizations')!r}")
    tm = d.get("tt_metal")
    if not isinstance(tm, dict) or not _text(tm.get("tag")) or not isinstance(tm.get("dirty"), bool):
        reasons.append('tt_metal must be {"tag": <text>, "dirty": <true or false>}: the tree the runtime came from')
    small = _number(rec.get("params_b")) and rec["params_b"] <= STOCK_SMALL_PARAMS_MAX_B
    top1_min, top5_min = ((STOCK_SMALL_TOP1_MIN, STOCK_SMALL_TOP5_MIN) if small else (STOCK_TOP1_MIN, STOCK_TOP5_MIN))
    t = d.get("teacher_forced") if isinstance(d.get("teacher_forced"), dict) else {}
    if isinstance(t.get("tokens"), bool) or not isinstance(t.get("tokens"), int) or t["tokens"] < STOCK_MIN_TOKENS:
        reasons.append(f"teacher_forced.tokens must be a whole number of at least {STOCK_MIN_TOKENS}, got {t.get('tokens')!r}")
    for key, bar in (("top1", top1_min), ("top5", top5_min)):
        v = t.get(key)
        if not _unit_number(v):
            reasons.append(f"teacher_forced.{key} must be a fraction from 0 to 1, got {v!r}")
        elif v < bar:
            reasons.append(f"teacher_forced.{key} {v} is below the bar {bar}")
    f = d.get("free_run") if isinstance(d.get("free_run"), dict) else {}
    text = f.get("text") if isinstance(f.get("text"), str) else ""
    if len("".join(text.split())) < STOCK_MIN_FREE_CHARS:
        reasons.append(f"free_run.text has fewer than {STOCK_MIN_FREE_CHARS} non-space characters, so there is nothing to judge")
    else:
        for finding in prompt_suite.findings({"content": text}):
            if finding["kind"] in SUITE_BLOCKING_KINDS:
                reasons.append(f"free_run: {finding['kind']}: {finding['detail']}")
    sp = d.get("speed") if isinstance(d.get("speed"), dict) else {}
    if not _number(sp.get("decode_tokens_per_s")) or not sp["decode_tokens_per_s"] > 0:
        reasons.append(f"speed.decode_tokens_per_s must be a positive number, got {sp.get('decode_tokens_per_s')!r}")
    ref = d.get("reference")
    want_rev = str((delta or {}).get("model", "")).split("@", 1)[1] if "@" in str((delta or {}).get("model", "")) else None
    if not isinstance(ref, dict) or ref.get("source") != "generated":
        reasons.append("reference must record a reference made from this run's snapshot (source 'generated'): the file the host tree ships "
                       f"for a model name belongs to the model the tree lists, got {ref!r}")
    else:
        if str(ref.get("model_id", "")).split("@", 1)[0] != want:
            reasons.append(f"reference model_id {ref.get('model_id')!r} is not the model of this run ({want!r})")
        if not _text(ref.get("snapshot")):
            reasons.append("reference must name the snapshot it was made from")
        elif want_rev and ref["snapshot"] != want_rev:
            reasons.append(f"reference was made from snapshot {ref['snapshot']!r}, but stage 0 names revision {want_rev!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256", ""))):
            reasons.append("reference needs the sha256 of the reference file (64 hex digits)")
    reasons += _hub_fetch_reasons(d)
    patch = d.get("scratch_patch")
    if not _text(patch) or inside(run_dir, f"stages/2/{patch}") is None:
        reasons.append(f"scratch_patch must name a file under the stage directory (the patch of tt-metal-scratch against the host tree), got {patch!r}")
    else:
        seen.append(f"stages/2/{patch}")
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
    reasons += _suite_reasons_in(Path(stage_dir) / "evidence" / "swap-check.json", run_dir)
    reasons += _token_check_reasons_in(Path(stage_dir) / "evidence" / "swap-check.json", [("result.json", d)], run_dir)
    reasons += decision_problems(stage_dir, run_dir)         # the graft trial's decision, when stage 2 made one
    reasons += sampling_choice.decision_problems(stage_dir, run_dir)       # the sampling choice (issue 107), when stage 2 made one
    reasons += vocab_trim_problems(stage_dir, run_dir)
    _evidence(run_dir, d.get("evidence"), "result.json", reasons, seen)
    return _done(reasons, seen)


def gate_sidecar_parity(sp, delta: dict) -> list[str]:
    """The reasons a result's `sidecar_parity` object is not good enough (empty when it is). `delta` is
    stage 0's delta.json: the parity run must have used the exact head file and code file stage 0 recorded."""
    if not isinstance(sp, dict):
        return ["sidecar_parity must be an object (the parity check's measurements)"]
    reasons = []
    if sp.get("measured") is not True:
        reasons.append(f"sidecar_parity measured must be true, got {sp.get('measured')!r}")
    wiring = sp.get("wiring_check")
    if not isinstance(wiring, dict) or wiring.get("passed") is not True:
        reasons.append("sidecar_parity wiring_check must have passed: the script's own layer loop must "
                       "agree with the model's prefill before its hidden states mean anything")
    nq = sp.get("n_questions")
    if isinstance(nq, bool) or not isinstance(nq, int) or nq < SIDECAR_MIN_QUESTIONS:
        reasons.append(f"sidecar_parity n_questions must be a whole number of at least {SIDECAR_MIN_QUESTIONS}, "
                       f"got {nq!r}")
    pcc = sp.get("hidden_pcc_min")
    if not _number(pcc):
        reasons.append(f"sidecar_parity hidden_pcc_min must be a number, got {pcc!r}")
    elif pcc < SIDECAR_PCC_MIN:
        reasons.append(f"sidecar_parity hidden_pcc_min {pcc} is below the minimum of {SIDECAR_PCC_MIN}")
    top1 = sp.get("top1_agree_fraction")
    if not _number(top1) or not 0 <= top1 <= 1:
        reasons.append(f"sidecar_parity top1_agree_fraction must be a fraction from 0 to 1, got {top1!r}")
    elif top1 < SIDECAR_TOP1_MIN:
        reasons.append(f"sidecar_parity top1_agree_fraction {top1} is below the minimum of {SIDECAR_TOP1_MIN}")
    diff = sp.get("prob_max_abs_diff")
    floor = (sp.get("noise_floor") or {}).get("prob_max_abs_diff") if isinstance(sp.get("noise_floor"), dict) else None
    bar = max(SIDECAR_PROB_DIFF_MAX, SIDECAR_NOISE_FACTOR * floor if _number(floor) else 0.0)
    if not _number(diff):
        reasons.append(f"sidecar_parity prob_max_abs_diff must be a number, got {diff!r}")
    elif diff > bar:
        reasons.append(f"sidecar_parity prob_max_abs_diff {diff} is above the bar of {bar:g} "
                       f"(the larger of {SIDECAR_PROB_DIFF_MAX} and {SIDECAR_NOISE_FACTOR:g} times the "
                       "CPU-versus-CPU difference)")
    heads = {x.get("sha256") for x in delta.get("sidecars", []) if isinstance(x, dict)}
    if sp.get("head_sha256") not in heads:
        reasons.append("sidecar_parity head_sha256 does not match any sidecar stage 0 recorded; the parity "
                       "run used a different head file")
    codes = {x.get("sha256") for x in delta.get("code_files", []) if isinstance(x, dict)}
    if codes and sp.get("code_sha256") not in codes:
        reasons.append("sidecar_parity code_sha256 does not match any code file stage 0 recorded; the parity "
                       "run used different code")
    return reasons


def gate_weights_swap_sidecar(stage_dir, run_dir) -> GateResult:
    """Stage 2 on the weights+sidecar class: everything gate_weights_swap checks, and the sidecar parity
    measured in the same lease (orchard/skills/weights-sidecar-check.md)."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = _swap_reasons(d), []
    reasons += _suite_reasons_in(Path(stage_dir) / "evidence" / "swap-check.json", run_dir)
    reasons += _token_check_reasons_in(Path(stage_dir) / "evidence" / "swap-check.json", [("result.json", d)], run_dir)
    reasons += decision_problems(stage_dir, run_dir)
    reasons += sampling_choice.decision_problems(stage_dir, run_dir)
    delta, derr = _load(Path(run_dir) / "stages" / "0", "delta.json")
    if derr:
        reasons.append(f"stage 0's {derr}, so the sidecar parity cannot be tied to the files stage 0 recorded")
    else:
        reasons += gate_sidecar_parity(d.get("sidecar_parity"), delta)
    listed = d.get("evidence")
    if not isinstance(listed, list) or not any(Path(str(e)).name == "sidecar-parity.json" for e in listed):
        reasons.append("result.json evidence must list evidence/sidecar-parity.json")
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
        rec = _safe_json(path)
    except safeio.SafeReadError as exc:
        if _is_missing(exc):
            return f"the supervisor has no record of its test (tests/{chips}/test-result.json)"
        return f"tests/{chips}/test-result.json is not readable JSON: {exc.reason}"
    if not isinstance(rec, dict):
        return f"tests/{chips}/test-result.json is not a JSON object"
    if rec.get("timed_out") is not False:
        return f"its test did not finish before its deadline (timed_out {rec.get('timed_out')!r})"
    if rec.get("returncode") != 0:
        return f"its test exited {rec.get('returncode')!r}"
    if not isinstance(rec.get("chips"), list) or len(rec["chips"]) != chips:
        return f"its test ran on {rec.get('chips')!r}, which is not {chips} chips"
    return None


def _nearest_model_of(run_dir) -> str | None:
    """The nearest model's id as stage 0 recorded it (delta.json's `nearest_model`), or None when stage 0 gives none. The gates ask
    stage 0 and not the stage's own swap_config.json, because an agent writes that file and a control is only worth as much as
    the model it was served from."""
    delta, err = _load(Path(run_dir) / "stages" / "0", "delta.json")
    value = delta.get("nearest_model") if not err and isinstance(delta, dict) else None
    return value if isinstance(value, str) else None


def _hub_fetch_reasons(report, where: str = "") -> list[str]:
    """A container test whose report records a Hub fetch measured the fetched model, not the model directory (the package ignored MODEL_WEIGHTS_DIR).
    A report with no such key, or an empty list, has none; a record that is present and is not a list (null included) is a reason too."""
    if not isinstance(report, dict) or "hub_fetch" not in report:
        return []                                  # a report from before the record existed has none
    fetched = report["hub_fetch"]
    if fetched == []:
        return []
    if not isinstance(fetched, list):
        return [f"{where}swap-check.json hub_fetch must be a list, got {fetched!r}"]
    return [f"{where}the container fetched {', '.join(str(r) for r in fetched)} from the Hub, so the test did not serve the model directory: "
            f"the package ignores MODEL_WEIGHTS_DIR, and its numbers describe the fetched model"]


AGREEMENT_PROMPTS_FILE = Path(__file__).resolve().parent / "skills" / "weights-swap-templates" / "agreement_prompts.json"
AGREEMENT_REFERENCE_PATH = ("stages", "1", "evidence", "reference", "agreement-reference.json")
AGREEMENT_REFERENCE_MAX_BYTES = 8 * 1024 * 1024
AGREEMENT_LEDGER_DECISION = "agreement reference recorded"     # the supervisor writes it when it drafts stage 2's config (Supervisor._record_agreement_reference)


def _shipped_agreement_ids() -> list[str] | None:
    """The prompt ids of the shipped agreement set, or None when the file cannot be read."""
    try:
        data = safeio.read_json(AGREEMENT_PROMPTS_FILE, max_bytes=AGREEMENT_REFERENCE_MAX_BYTES)
    except safeio.SafeReadError:
        return None
    prompts = data.get("prompts") if isinstance(data, dict) else None
    ids = [p.get("id") for p in prompts if isinstance(p, dict)] if isinstance(prompts, list) else []
    return ids if ids and all(isinstance(i, str) for i in ids) else None


def _ledger_decision_sha(run_dir, decision: str) -> str | None:
    """The sha256 the supervisor recorded in the ledger under the decision `decision` (an evidence file it hashed before any stage 2 test
    ran), or None when no such entry exists or the ledger cannot be read. The last entry wins."""
    from orchard.ledger import LedgerCorrupt, read_entries
    try:
        entries = read_entries(Path(run_dir) / "ledger.jsonl")
    except (LedgerCorrupt, OSError):
        return None
    found = None
    for e in entries:
        data = e.get("data")
        if e.get("event") == "decision" and isinstance(data, dict) and data.get("decision") == decision \
                and isinstance(data.get("sha256"), str):
            found = data["sha256"]
    return found


def _ledger_reference_sha(run_dir) -> str | None:
    """The sha256 of stage 1's agreement reference file as the supervisor recorded it in the ledger before any stage 2 test ran, or None
    when no such entry exists (or the ledger cannot be read). The last entry wins."""
    return _ledger_decision_sha(run_dir, AGREEMENT_LEDGER_DECISION)


def agreement_reference_record(run_dir) -> dict | None:
    """{path, sha256} of stage 1's agreement reference file when it exists and can be hashed, else None. The supervisor calls this when it
    drafts stage 2's config: a record means the run can use the agreement set, and the sha256 goes into the ledger as the trusted value
    the gate compares the file with later."""
    path = Path(run_dir).joinpath(*AGREEMENT_REFERENCE_PATH)
    if not os.path.lexists(path):
        return None
    try:
        return evidence_record(run_dir, path)
    except safeio.SafeReadError:
        return None


def _agreement_binding_reasons(record, run_dir) -> list[str]:
    """The agreement-set record must have been scored against the reference stage 1 wrote and the prompt set the repository ships. The
    reference file's sha256 is checked against the one the supervisor put in the ledger, so a file replaced after stage 1 (together with the
    hash the template recorded) is refused. The prompt ids must be the shipped ids, each once, in the reference, the record and the control;
    every reference id list is compared with stage 1's, and the control must have been served from the nearest model."""
    if not isinstance(record, dict) or record.get("ran") is not True:
        return []
    out = []
    shipped_ids = _shipped_agreement_ids()
    try:
        shipped = safeio.sha256_file(AGREEMENT_PROMPTS_FILE, max_bytes=AGREEMENT_REFERENCE_MAX_BYTES)
    except safeio.SafeReadError as exc:
        return [f"the shipped agreement prompt set cannot be read: {exc.reason}"]
    if shipped_ids is None:
        return ["the shipped agreement prompt set has no usable prompt ids"]
    if record.get("prompt_set_sha256") != shipped:
        out.append("the agreement set was run with a prompt set that is not the one the repository ships (its sha256 differs)")
    path = Path(run_dir).joinpath(*AGREEMENT_REFERENCE_PATH)
    try:
        raw = safeio.read_bytes(path, max_bytes=AGREEMENT_REFERENCE_MAX_BYTES)
        ref = safeio.parse_json_bytes(raw, max_bytes=AGREEMENT_REFERENCE_MAX_BYTES)
    except safeio.SafeReadError as exc:
        return out + [f"the agreement set's CPU reference (stage 1 {AGREEMENT_REFERENCE_PATH[-1]}) cannot be read: {exc.reason}"]
    actual = hashlib.sha256(raw).hexdigest()
    recorded = _ledger_reference_sha(run_dir)
    if recorded is None:
        out.append("the ledger has no record of stage 1's agreement reference (the supervisor writes it before stage 2's test), so the file cannot be trusted")
    elif recorded != actual:
        out.append("stage 1's agreement reference file is not the one the supervisor recorded in the ledger (its sha256 differs)")
    if record.get("reference_sha256") != actual:
        out.append("the agreement set was scored against a reference file other than the one stage 1 wrote (its sha256 differs)")
    entries = ref.get("entries") if isinstance(ref, dict) else None
    out += [f"stage 1's agreement reference: {r}" for r in agreement_core.id_problems(entries, shipped_ids)]
    wanted = ({e["id"]: e.get("generated_ids") for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
              if isinstance(entries, list) else {})

    def check(rows, label):
        for e in rows if isinstance(rows, list) else []:
            pid = e.get("id") if isinstance(e, dict) else None
            ids = e.get("reference_ids") if isinstance(e, dict) else None
            full = wanted.get(pid) if isinstance(pid, str) else None
            if not isinstance(ids, list) or not isinstance(full, list) or full[:len(ids)] != ids:
                out.append(f"the {label}'s reference ids for prompt {pid if isinstance(pid, str) else None!r} are not stage 1's")

    check(record.get("entries"), "agreement set")
    control = record.get("control")
    if isinstance(control, dict):
        check(control.get("entries"), "control")
        problem = prompt_suite.control_identity_problem(_nearest_model_of(run_dir), control.get("model_id"))
        if problem:
            out.append(f"the agreement control is not a control of the nearest model: {problem}")
    return out


def _agreement_reasons(record, run_dir, requested: bool) -> list[str]:
    """All gate reasons for one swap-check.json's `agreement_set`. A run whose swap_config.json asked for the set (`requested`) needs a
    completed record; a run that did not ask has none unless a record is present, in which case it is judged like any other."""
    ran = isinstance(record, dict) and record.get("ran") is True
    if requested and not ran:
        return ["swap_config.json asks for the agreement set and swap-check.json holds no completed agreement_set record (missing, malformed or ran false)"]
    if not ran:
        return []
    return agreement_core.judge(record, expected_ids=_shipped_agreement_ids() or [])["reasons"] + _agreement_binding_reasons(record, run_dir)


def _suite_reasons(report, run_dir, where: str = "", agreement_requested: bool = False) -> list[str]:
    """The prompt-suite reasons (orchard/prompt_suite.py) in one swap-check.json report, each prefixed with `where`.
    A report without a suite record has none. The gates read the template's swap-check.json and not the agent's
    result.json: the record holds the raw replies, and the verdict is computed here from them. A control in the record has
    to have been served from the run's nearest model (stage 0's), which is checked here again whatever the template did."""
    probe = [where + r for r in sampling_probe.problems(report.get("sampling_probe") if isinstance(report, dict) else None)]
    # The agreement set (issue 15): every number is recomputed from the raw ids in the record, with the same file the template used.
    record = report.get("agreement_set") if isinstance(report, dict) else None
    agreement = [where + r for r in _agreement_reasons(record, run_dir, agreement_requested)]
    return (_hub_fetch_reasons(report, where) + probe + agreement
            + [where + r for r in prompt_suite.evaluate(report, nearest_model=_nearest_model_of(run_dir)).reasons])


# Stage 1's CPU reference ids, the tokens the 32-token check feeds the server one at a time (issue 433). The supervisor puts the file's sha256
# in the ledger when it drafts stage 2's config, before any stage 2 test; the gate compares the file with it and recomputes the check.
REFERENCE_IDS_PATH = ("stages", "1", "evidence", "reference", "generated-ids.json")
REFERENCE_IDS_MAX_BYTES = 8 * 1024 * 1024
REFERENCE_IDS_LEDGER_DECISION = "stage 1 reference ids recorded"      # Supervisor._draft_stage_files, beside AGREEMENT_LEDGER_DECISION


def reference_ids_record(run_dir) -> dict | None:
    """{path, sha256} of stage 1's `generated-ids.json` when it exists and can be hashed, else None. The supervisor calls this when it drafts
    stage 2's config, and the sha256 goes into the ledger as the value the gate compares the file with later."""
    path = Path(run_dir).joinpath(*REFERENCE_IDS_PATH)
    if not os.path.lexists(path):
        return None
    try:
        return evidence_record(run_dir, path)
    except safeio.SafeReadError:
        return None


def _reference_ids(run_dir) -> tuple[list | None, list[str], bool]:
    """(ids, reasons, bound) for the 32-token recompute: stage 1's generated ids, read through safeio and checked against the ledger's hash.
    `bound` says the ledger holds the file's hash, so the supervisor governed the run from its stage 2 draft and the file cannot have been
    swapped since. (None, [], False) when there is nothing to compare with: no file and no ledger record (a run from before this check, or a
    stage 2 without a stage 1 reference), which the gate judges as before. A file the ledger recorded and that is now missing, unreadable or
    different is a reason; so is a file that is not a list of ids."""
    path = Path(run_dir).joinpath(*REFERENCE_IDS_PATH)
    recorded = _ledger_decision_sha(run_dir, REFERENCE_IDS_LEDGER_DECISION)
    name = REFERENCE_IDS_PATH[-1]
    bound = recorded is not None
    if not os.path.lexists(path):
        if not bound:
            return None, [], False
        return None, [f"stage 1's {name} is missing, though the supervisor recorded it in the ledger, so the 32-token check cannot be recomputed"], True
    try:
        raw = safeio.read_bytes(path, max_bytes=REFERENCE_IDS_MAX_BYTES)
        data = safeio.parse_json_bytes(raw, max_bytes=REFERENCE_IDS_MAX_BYTES)
    except safeio.SafeReadError as exc:
        return None, [f"stage 1's {name} cannot be read, so the 32-token check cannot be recomputed: {exc.reason}"], bound
    reasons = []
    if recorded is not None and hashlib.sha256(raw).hexdigest() != recorded:
        reasons.append(f"stage 1's {name} is not the file the supervisor recorded in the ledger (its sha256 differs), so the 32-token check "
                       "cannot be recomputed from it")
    ids = data.get("generated_ids") if isinstance(data, dict) else None
    if not isinstance(ids, list):
        return None, reasons + [f"stage 1's {name} has no generated_ids list"], bound
    return (None if reasons else ids), reasons, bound


def _token_check_reasons(report, results, run_dir, where: str = "") -> list[str]:
    """The reasons one swap-check.json's 32-token numbers, and the results that repeat them (`results`: (label, object) pairs), are not what
    its raw `teacher_forced` records give against stage 1's reference ids (issue 433, orchard/token_check.py). Empty when the report is not an
    object (the other checks say so) or when the run has no stage 1 reference to compare with."""
    if not isinstance(report, dict):
        # A run whose ledger holds the reference ids' hash was governed from the start: a swap-check.json that is not an object is what an
        # edit would leave, and the recompute is what the binding is for (review of PR 444). An unbound run is judged as before.
        if _ledger_decision_sha(run_dir, REFERENCE_IDS_LEDGER_DECISION) is not None:
            return [f"{where}swap-check.json is not an object, though the supervisor recorded stage 1's reference ids in the ledger, so the "
                    "32-token check cannot be recomputed"]
        return []
    ids, reasons, bound = _reference_ids(run_dir)
    if ids is None:
        return [where + r for r in reasons]
    return [where + r for r in reasons] + token_check.problems(report, results, ids, where, require_records=bound)


def _token_check_reasons_in(path, results, run_dir, where: str = "") -> list[str]:
    """`_token_check_reasons` for the swap-check.json at `path`. A file that the safe reader refuses has none here: `_suite_reasons_in`
    already says why. A file that is missing is a reason when the ledger holds the reference ids' hash (the run was governed from the start,
    so the recompute cannot be skipped by deleting the report); otherwise a result with no swap check is judged as before."""
    try:
        report = _safe_json(path)
    except safeio.SafeReadError as exc:
        if _is_missing(exc) and _ledger_decision_sha(run_dir, REFERENCE_IDS_LEDGER_DECISION) is not None:
            return [f"{where}swap-check.json is missing, though the supervisor recorded stage 1's reference ids in the ledger, so the "
                    "32-token check cannot be recomputed"]
        return []
    return _token_check_reasons(report, results, run_dir, where)


def _suite_reasons_in(path, run_dir, where: str = "") -> list[str]:
    """`_suite_reasons` for the swap-check.json at `path`. A file that is missing has none, and neither has a JSON value that
    is not an object: the other checks of the gate already say what they need of those. A file that is there and that the safe
    reader refuses (not JSON, NaN, over the size cap, a link, a FIFO) is a reason: the report could have held a failing record
    (a non-empty hub_fetch, a suite finding), and a refused read must not drop it."""
    # Whether the run asked for the agreement set is read from the swap_config.json beside the evidence directory, which the supervisor
    # drafts; a config that cannot be read counts as not asking.
    requested = False
    try:
        cfg = _safe_json(Path(path).parent.parent / "swap_config.json")
        requested = isinstance(cfg, dict) and cfg.get("agreement_set") is True
    except safeio.SafeReadError:
        pass
    try:
        report = _safe_json(path)
    except safeio.SafeReadError as exc:
        if _is_missing(exc):
            return [f"{where}swap_config.json asks for the agreement set and swap-check.json is missing"] if requested else []
        return [f"{where}swap-check.json is not readable JSON: {exc.reason}"]
    if not isinstance(report, dict):
        return [f"{where}swap_config.json asks for the agreement set and swap-check.json is not an object"] if requested else []
    return _suite_reasons(report, run_dir, where, requested)


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
    trim = _trim_vocab(run_dir)
    refused = _delta_refusal(run_dir)
    if refused:
        reasons.append(refused)
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
        swap_path, swap_why = locate(run_dir, swap)
        try:
            report = _safe_json(swap_path) if swap_path is not None else None
            draft = report["result_draft"]
        except (AttributeError, safeio.SafeReadError, KeyError, TypeError) as exc:
            report, draft = None, None
            # Say why the swap check could not be used: an agent that links it to a device or swells it needs the cause, and a bare
            # "top1_agreement is not the result_draft's" names none.
            cause = swap_why if swap_path is None else (exc.reason if isinstance(exc, safeio.SafeReadError) else "it has no result_draft object")
            reasons.append(f"{where}: {swap} cannot be used ({cause})")
        if not isinstance(draft, dict) or draft.get("top1_agreement") != c.get("top1_agreement"):
            reasons.append(f"{where}: top1_agreement {c.get('top1_agreement')!r} is not the "
                           f"result_draft's in {swap}")
        if isinstance(report, dict):
            reasons += _suite_reasons(report, run_dir, f"{where}: ")
            reasons += _token_check_reasons(report, [("the configuration's result", c)], run_dir, f"{where}: ")
        reasons += _graft_reasons(stage_dir, run_dir, n, where)
        reasons += _sampling_reasons(report, run_dir, where)
        if trim is not None:
            cdir = Path(stage_dir) / "configs" / str(n)
            reasons += _trim_reasons(trim, _read_checked(cdir / "model-dir-sources.json", f"configs/{n}/model-dir-sources.json", reasons),
                                     report.get("serving") if isinstance(report, dict) else None, f"{where}: ")
    return _done(reasons, seen)


def decision_file_problems(run_dir) -> list[str]:
    """One reason for each of stage 2's decision files (graft-decision.json, sampling-decision.json) that is there and refused. Stage 4's graft
    check and the package gate both use this, with the one reader and limit of `sampling_choice.read_decision_file`; a missing file is no reason."""
    out = []
    for fname in (GRAFT_DECISION_FILE, sampling_choice.DECISION_FILE):
        problem = sampling_choice.read_decision_file(Path(run_dir) / "stages" / "2" / fname).problem(f"stages/2/{fname}")
        if problem:
            out.append(problem)
    return out


def _graft_reasons(stage_dir, run_dir, n: int, where: str) -> list[str]:
    """A bundle configuration was built the way stage 2's graft decision says. A configuration served by the nearest model's own bundle
    has `bundle_dir` in its swap_config.json; the supervisor drafted `graft_mtp` there from the decision, and the agent's prepare step can
    change it afterwards, so the check reads what prepare_swap.py recorded it built (`mtp_graft` in model-dir-sources.json). A container
    configuration is another server and is not held to the decision."""
    # Stage 2's two decision files decide what this configuration is held to. One that is there and refused (a link, a FIFO, an oversize file)
    # reads as no decision in `graft_chosen` and `drafter_off_chosen`, which would let an ungrafted bundle through.
    bad = decision_file_problems(run_dir)
    if bad:
        return [f"{where}: {bad[0]}, so stage 2's decision cannot be checked against this configuration"]
    cdir = Path(stage_dir) / "configs" / str(n)
    got = safeio.read_json_checked(cdir / "swap_config.json", max_bytes=JSON_MAX_BYTES)
    if got.refused:
        # A swap_config.json that is there and refused is not an absent one, with or without a recorded decision: it could name the bundle
        # whose graft the decision did not choose (issue 287).
        return [f"{where}: {got.problem(f'configs/{n}/swap_config.json')}, so how this configuration serves cannot be told and stage 2's "
                "graft decision cannot be checked against it"]
    cfg = got.value
    kind = _serving_type(cfg)
    if kind is None and read_graft_decision(Path(run_dir) / "stages" / "2") is not None:
        # Once stage 2 has recorded a decision, a config that is gone or cannot be read must not let a configuration through: it may be an
        # ungrafted bundle whose config was lost. Only a configuration that names its container package is exempt.
        return [f"{where}: swap_config.json is missing, cannot be read, or names neither a bundle (bundle_dir) nor a container package "
                "(package and profile), so how this configuration serves cannot be told and stage 2's graft decision cannot be checked "
                "against it"]
    if kind != "bundle":
        return []
    try:
        sources = _safe_json(cdir / "model-dir-sources.json")
    except safeio.SafeReadError as exc:
        if not _is_missing(exc):
            # A record that is there and refused is not an absent record: it could hold the graft the decision did not choose.
            return [f"{where}: configs/{n}/model-dir-sources.json is not readable JSON: {exc.reason}, so what this configuration was built "
                    "with cannot be checked against stage 2's graft decision"]
        sources = None
    built = isinstance(sources, dict) and isinstance(sources.get("mtp_graft"), dict)
    chose = graft_chosen(run_dir)
    if chose and not built:
        return [f"{where}: stage 2 chose the graft ({GRAFT_DECISION_FILE}), but configs/{n}/model-dir-sources.json records no mtp_graft, "
                "so this configuration was built without the grafted head"]
    if built and not chose:
        return [f"{where}: configs/{n}/model-dir-sources.json records a grafted head, but stage 2 did not choose the graft"]
    return []


def _sampling_reasons(report, run_dir, where: str) -> list[str]:
    """A configuration was served the way stage 2's sampling decision says (issue 107). When stage 2 chose the drafter-off configuration, the swap
    check's `serving` record (written by the template from the run script it served with) must say the drafter was off. A decision of keep, or none,
    asks nothing."""
    if not sampling_choice.drafter_off_chosen(run_dir):
        return []
    serving = report.get("serving") if isinstance(report, dict) else None
    got = serving.get("drafter") if isinstance(serving, dict) else None
    if got == "off":
        return []
    return [f"{where}: stage 2 chose the drafter-off configuration because the model needs sampling ({sampling_choice.DECISION_FILE}), so "
            f"swap-check.json serving.drafter must be 'off', got {got!r}"]


def _serving_type(cfg) -> str | None:
    """"bundle" when a bundle configuration's swap_config.json names its bundle_dir, "container" when it names a container package and its
    profile, None when it is not an object or names neither."""
    if not isinstance(cfg, dict):
        return None
    if isinstance(cfg.get("bundle_dir"), str) and cfg["bundle_dir"]:
        return "bundle"
    if all(isinstance(cfg.get(k), str) and cfg[k] for k in ("package", "profile")):
        return "container"
    return None


def _read_checked(path: Path, name: str, reasons: list):
    """The JSON in `path` (any type), or None. A missing file is None with no reason, because a caller decides whether it is required. A file
    that is there and refused (a FIFO, a link, a device, a file over the cap, NaN, a repeated key) is None with a named reason in `reasons`
    (issue 287: a refusal used to come back as None, which every caller read as "no file")."""
    got = safeio.read_json_checked(path, max_bytes=JSON_MAX_BYTES)
    if got.refused:
        reasons.append(got.problem(name) or "")
        return None
    return got.value if got.ok else None


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


BUNDLE_FILES = ("RESULTS.md", "RISKS.md", "card.md", "PUBLISH_COMMANDS.txt", "ledger.jsonl")
BUNDLE_RECORD = "evidence/bundle-build.json"


def gate_bundle(stage_dir, run_dir) -> GateResult:
    """Stage 8: results, ledger, open risks, the card, publish commands as text, and a clean scrub.
    Every file is the one build_bundle.py wrote: its record holds each file's sha256, and only the
    ledger copy (which the supervisor writes again just before this gate) may differ. On run 2 an
    agent rewrote RESULTS.md with stage names and an evidence path the run never had, and the gate
    of the time passed it."""
    from orchard.scrub import scrub_bundle
    stage_dir = Path(stage_dir)
    bundle = stage_dir / "bundle"
    reasons, seen = [], []
    for name in BUNDLE_FILES:
        f = bundle / name
        got = safeio.read_text_checked(f, max_bytes=TEXT_MAX_BYTES, errors="replace")
        if got.refused:
            reasons.append(got.problem(f"bundle/{name}") or "")
        elif not got.ok or not str(got.value).strip():
            reasons.append(f"bundle/{name} is missing or empty")
        else:
            seen.append(os.path.relpath(f, run_dir))
    try:
        built = _safe_json(stage_dir / BUNDLE_RECORD)
        recorded = {f["path"]: f["sha256"] for f in built["files"]}
    except (safeio.SafeReadError, KeyError, TypeError):
        reasons.append(f"stages/8/{BUNDLE_RECORD} is missing or unreadable: the bundle was not built "
                       "by build_bundle.py")
    else:
        seen.append(os.path.relpath(stage_dir / BUNDLE_RECORD, run_dir))
        found = set()
        for f in sorted(bundle.rglob("*")) if bundle.is_dir() else ():
            if not f.is_file() or f == bundle / "ledger.jsonl":
                continue
            rel = f"stages/8/bundle/{f.relative_to(bundle).as_posix()}"
            found.add(rel)
            if rel not in recorded:
                reasons.append(f"{rel} was not written by build_bundle.py")
            else:
                try:
                    unchanged = _sha256(f) == recorded[rel]
                except safeio.SafeReadError as exc:
                    reasons.append(f"{rel} cannot be read to check it against build_bundle.py's record: {exc.reason}")
                    continue
                if not unchanged:
                    reasons.append(f"{rel} changed after build_bundle.py wrote it")
        for rel in sorted(set(recorded) - found - {"stages/8/bundle/ledger.jsonl"}):
            reasons.append(f"{rel} was written by build_bundle.py and is gone")
    reasons += _bundle_benchmark_reasons(bundle, run_dir)
    for hit in scrub_bundle(bundle):
        reasons.append(f"scrub: {hit}")
    return _done(reasons, seen)


def _bundle_benchmark_reasons(bundle: Path, run_dir) -> list[str]:
    """A bundle whose RESULTS.md heads its audio timings as a benchmark must rest on at least `resident.MIN_REPEATS` measured repeats of each model that
    stage 2 repeated, counted here from `stages/2/audio_result.json` and not taken from any label. Fewer repeats are a single sample, which the bundle
    heads differently. A result file that cannot be read gives no support for the heading."""
    from orchard import resident
    got = safeio.read_text_checked(bundle / "RESULTS.md", max_bytes=TEXT_MAX_BYTES, errors="replace")
    if got.refused:
        return [got.problem("bundle/RESULTS.md") or ""]        # the gate's own required-file check reports it too; here it also blocks the benchmark heading
    if not got.ok:
        return []
    text = got.value
    if resident.BENCH_HEADING not in text.splitlines():
        return []
    rel = f"stages/2/{audio_runtime.RESULT_FILE}"
    try:
        res = safeio.read_json(Path(run_dir) / rel, max_bytes=64 << 20, expect=dict)
    except (safeio.SafeReadError, OSError):
        return [f"RESULTS.md heads the audio timings as a benchmark, but {rel} cannot be read to support it"]
    warm = (res.get("timing") or {}).get("warmup_runs") if isinstance(res.get("timing"), dict) else None
    if not isinstance(warm, int) or isinstance(warm, bool) or warm < resident.WARMUP_RUNS:
        return [f"RESULTS.md heads the audio timings as a benchmark, but {rel} records {warm!r} warm-up runs; the rule is {resident.WARMUP_RUNS}"]
    reps = res.get("repeats")
    runs = res.get("runs")
    counts = {}
    # Every model the result recorded a run for must have its repeats; a role with a missing, empty or malformed list counts zero.
    for role in sorted({*(reps if isinstance(reps, dict) else ()), *(runs if isinstance(runs, dict) else ())}, key=str):
        lst = reps.get(role) if isinstance(reps, dict) else None
        counts[role] = sum(1 for e in lst if isinstance(e, dict) and isinstance(e.get("timings"), dict)
                           and isinstance(e["timings"].get(resident.MAIN_PART), (int, float))
                           and not isinstance(e["timings"].get(resident.MAIN_PART), bool)) if isinstance(lst, list) else 0   # safeio.read_json already refuses NaN and Infinity
    if not counts:
        return [f"RESULTS.md heads the audio timings as a benchmark, but {rel} lists no repeats"]
    return [f"RESULTS.md heads the audio timings as a benchmark, but {role} has {n} measured repeat(s) in {rel}; "
            f"{resident.MIN_REPEATS} are needed" for role, n in sorted(counts.items()) if n < resident.MIN_REPEATS]



def _inside_dir(run_dir, rel) -> Path | None:
    """The directory `rel` names, if it is a directory inside the run directory (links resolved)."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        return None
    root = os.path.realpath(run_dir)
    p = os.path.realpath(os.path.join(root, rel))
    return Path(p) if os.path.commonpath([root, p]) == root and os.path.isdir(p) else None


def _shipped_names(directory: Path) -> tuple[str, ...]:
    """The file names in a package's base_config/ (the files it takes from the nearest model), sorted."""
    try:
        return tuple(sorted(f.name for f in directory.iterdir() if f.is_file()))
    except OSError:
        return ()


def delta_model_has_mtp(delta) -> bool | None:
    """Does the new model ship mtp.* tensors, as stage 0's delta.json says (`mtp.new_tensors`)? None when it does not say."""
    mtp = delta.get("mtp") if isinstance(delta, dict) else None
    n = mtp.get("new_tensors") if isinstance(mtp, dict) else None
    return n > 0 if isinstance(n, int) and not isinstance(n, bool) else None


def gate_package(stage_dir, run_dir) -> GateResult:
    """Stage 7: a staged v6 package whose boot check passed.

    Everything that matters is checked again from the files on disk: the license (read from the
    new model's snapshot), each staged directory's scrub, run.sh's weights settings, the manifest's
    weights, the card, the boot check's verify.json and the publish commands. package.json only
    says where to look."""
    # Imported here: orchard.package imports this module.
    from orchard.package import (GRAFT_FILE, PUBLISH_FILE, PackageError, acceptance_sources, graft_evidence_problems, graft_problems,
                                 graft_trial_numbers, publish_problems, read_reported, sampling_numbers,
                                 sampling_summary, sampling_summary_problems, serving_problems,
                                 served_drafter_state, split_model_id, stage_2_serving, stage_2_vocab, weights_wiring_problems)
    from orchard.can import hardening_problems
    from orchard.card_use import read_use_facts
    from orchard.package_card import card_evidence_paths, card_problems, read_license
    from orchard.scrub import scrub_package
    d, err = _load(stage_dir, "package.json")
    if err:
        return GateResult(False, (err,))
    run = Path(run_dir)
    reasons, seen = [], []
    if d.get("format") != "v6":
        reasons.append(f"package.json format must be 'v6', got {d.get('format')!r}")
    delta, _ = _load(run / "stages" / "0", "delta.json")
    swap, _ = _load(run / "stages" / "2", "swap_config.json")
    delta, swap = delta or {}, swap or {}
    # Stage 0 writes <repo>@<revision>; the manifest keeps the repo and the revision apart.
    new_repo, new_rev = split_model_id(delta.get("model"))
    nearest_repo = split_model_id(delta.get("nearest_model"))[0]
    license_id = read_license(swap.get("new_snapshot") or "/nonexistent")
    if not license_id:
        reasons.append("the new model's license cannot be read from its snapshot's README.md")
    elif d.get("license") != license_id:
        reasons.append(f"package.json says the license is {d.get('license')!r}; the model's is "
                       f"{license_id!r}")
    reasons += sampling_summary_problems(run)
    reasons += [f"{p}, so the package cannot be checked against stage 2's decision" for p in decision_file_problems(run)]
    drafter_off, host_sampling = served_drafter_state(run)
    sampling_decision = sampling_choice.read_decision(run / "stages" / "2")
    model_has_mtp = delta_model_has_mtp(delta)
    # A grafted package serves with the drafter and on-device sampling on, whatever stage 2's run.sh says.
    graft_expected = isinstance(d.get("mtp_graft"), dict)
    # A drafter-off sampling decision conflicts with any graft: a graft keeps the drafter on. The gate does not rely on read_run's check, so it refuses
    # package-level and per-profile graft metadata itself, and the graft exemption below no longer switches the serving checks off for that decision.
    off_by_sampling = sampling_choice.drafter_off_chosen(run)
    if off_by_sampling and graft_expected:
        reasons.append("package.json records an mtp_graft, but stage 2's sampling decision chose the drafter-off configuration, and a graft keeps the "
                       "drafter on")
    decision = read_graft_decision(run / "stages" / "2")
    graft_chosen_by_stage_2 = decision is not None and decision["choice"] == "graft"
    trial_decision = decision["choice"] if decision is not None and decision.get("trial") is not None else None
    trial_rows = tuple(n.name for n in graft_trial_numbers(run))
    sampling = sampling_summary(run)
    sampling_expected = tuple(sampling_numbers(run))
    acceptance_measured = bool(acceptance_sources(tuple(graft_trial_numbers(run)) + sampling_expected))
    try:
        reported, reported_source = read_reported(run)
    except PackageError as exc:
        reasons.append(str(exc))
        reported, reported_source = (), ""
    sidecars = [s.get("file") for s in delta.get("sidecars") or () if isinstance(s, dict)]
    profiles = d.get("profiles") if isinstance(d.get("profiles"), list) else []
    boot_check, _ = _load(Path(stage_dir) / "verify" / "evidence", "verify.json")
    if not any(isinstance(p, dict) and p.get("required") and p.get("verified") is True
               for p in profiles):
        failed = [p.get("verify", {}).get("failed") for p in profiles if isinstance(p, dict)]
        reasons.append("no required profile passed its boot check"
                       + (f": {failed[0]}" if failed and failed[0] else ""))
    for p in profiles:
        name = p.get("name") if isinstance(p, dict) else None
        out = _inside_dir(run, p.get("dir")) if isinstance(p, dict) else None
        if out is None:
            reasons.append(f"profile {name!r}: its dir is not a directory inside the run directory")
            continue
        try:
            m = _safe_json(out / "tt_kernel_manifest.json", expect=dict)
        except safeio.SafeReadError as exc:
            reasons.append(f"profile {name}: tt_kernel_manifest.json is not readable JSON: {exc.reason}")
            continue
        texts = {fname: safeio.read_text_checked(out / fname, max_bytes=TEXT_MAX_BYTES) for fname in ("run.sh", "README.md")}
        bad = [got.problem(f"profile {name}: {fname}") or f"profile {name}: {fname} is missing" for fname, got in texts.items() if not got.ok]
        if bad:
            reasons += bad
            continue
        run_sh, card = texts["run.sh"].value, texts["README.md"].value
        w = m.get("weights") or {}
        if (w.get("repo_id"), w.get("revision")) != (new_repo, new_rev) or d.get("revision") != new_rev:
            reasons.append(f"profile {name}: the manifest's weights are {w.get('repo_id')}@"
                           f"{w.get('revision')} and package.json's revision is "
                           f"{d.get('revision')}; stage 0 names {delta.get('model')}")
        reasons += [f"profile {name}: {x}" for x in
                    weights_wiring_problems(run_sh, nearest_model=nearest_repo)]
        has_graft = (out / GRAFT_FILE).exists()
        grafted = (graft_expected or has_graft) and not off_by_sampling         # the exemption from the serving checks
        if off_by_sampling and has_graft:
            reasons.append(f"profile {name}: {GRAFT_FILE} is present, but stage 2's sampling decision chose the drafter-off configuration, and a graft "
                           "keeps the drafter on")
        if graft_chosen_by_stage_2 and not has_graft:
            reasons.append(f"profile {name}: stage 2's graft trial chose the graft, so the package needs {GRAFT_FILE}")
        if graft_expected or has_graft:
            reasons += [f"profile {name}: {x}" for x in
                        graft_problems(out, nearest_model=nearest_repo,
                                       base_revision=Path(swap.get("base_snapshot") or "").name)]
        reasons += [f"profile {name}: {x}" for x in graft_evidence_problems(out, boot_check)]
        reasons += [f"profile {name}: {x}" for x in
                    serving_problems(run_sh, m.get("env") or {},
                                     drafter_off=drafter_off and not grafted,
                                     host_sampling=host_sampling and not grafted,
                                     model_has_mtp=model_has_mtp, graft=has_graft)]
        reasons += [f"profile {name}: scrub: {x}" for x in
                    scrub_package(out, namespace=d.get("namespace"))]
        reasons += [f"profile {name}: {x}" for x in hardening_problems(out)]     # the canning checks (orchard/can.py)
        if license_id:
            reasons += [f"profile {name}: card: {x}" for x in
                        card_problems(card, license_id=license_id, run_dir=run, sidecars=sidecars,
                                      base_files=_shipped_names(out / "base_config"), reported=reported, reported_source=reported_source,
                                      model_id=new_repo, nearest_model=nearest_repo, graft=has_graft,
                                      graft_decision=trial_decision, vocab=stage_2_vocab(run),
                                      trial_rows=trial_rows if p.get("required") else (),
                                      sampling=sampling if p.get("required") else None,
                                      sampling_decision=sampling_decision if p.get("required") else None,
                                      drafter_off_by_sampling=bool(sampling_decision and sampling_decision.get("choice") == "drafter-off"),
                                      sampling_first_verdict=((sampling_decision or {}).get("numbers") or {}).get("served_verdict"),
                                      sampling_expected=sampling_expected if p.get("required") else (),
                                      acceptance_measured=acceptance_measured if p.get("required") else False,
                                      use=read_use_facts(swap.get("new_snapshot") or "/nonexistent", run_sh, run,
                                                           identity=(new_repo, new_rev), measured=bool(p.get("required"))))]
        reasons += [f"profile {name}: the card cites {e}, which is missing at evidence/{e}"
                    for e in card_evidence_paths(card) if not (out / "evidence" / e).is_file()]
        if p.get("verified") is True:
            v, verr = _load(Path(stage_dir) / "verify" / "evidence", "verify.json")
            if verr:
                reasons.append(f"profile {name}: {verr}")
                continue
            top1, n = v.get("top1_agreement"), v.get("n_tokens")
            if not _number(top1) or top1 < SWAP_TOP1_MIN:
                reasons.append(f"profile {name}: the boot check's top1_agreement {top1!r} is below "
                               f"{SWAP_TOP1_MIN}")
            if v.get("coherent") is not True:
                reasons.append(f"profile {name}: the boot check's free-run text is not coherent")
            if isinstance(n, bool) or not isinstance(n, int) or n < SWAP_MIN_TOKENS:
                reasons.append(f"profile {name}: the boot check compared {n!r} tokens; at least "
                               f"{SWAP_MIN_TOKENS} are needed")
            served = v.get("served_model") or ""
            if (not served.endswith("/model-dir")
                    or set((v.get("server_weights_env") or {}).values()) != {served}):
                reasons.append(f"profile {name}: the boot check's server did not serve its own "
                               "model-dir with MODEL_WEIGHTS_DIR and HF_MODEL set to it")
            _evidence(run_dir, v.get("evidence"), f"profile {name} boot check", reasons, seen)
    got = safeio.read_text_checked(Path(stage_dir) / PUBLISH_FILE, max_bytes=TEXT_MAX_BYTES)
    if got.ok:
        reasons += publish_problems(got.value)
    else:
        reasons.append(got.problem(PUBLISH_FILE) or f"{PUBLISH_FILE} is missing")
    return _done(reasons, seen)


SKIP_7 = ("stage 7 builds a package only on the weights-only path of a run started with "
          "--package-format v6; this run builds none, and the operator bundle reports the package "
          "as not built")

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
    # Stage 8 is supervisor code: build_bundle.py writes the whole bundle, summary included, in one
    # call. An agent added nothing the gate checks and, on lab run 2, made the bundle wrong.
    StageSpec(8, "operator bundle", "", (), 0, "bundle/RESULTS.md", gate_bundle, None, harness=True),
)


def validate_table(stages=STAGES) -> None:
    """Refuse a table the spec forbids: gaps, and long stages with no resume marker."""
    if [s.number for s in stages] != list(range(9)):
        raise ValueError("the stage table must list stages 0 to 8 in order")
    for s in stages:
        if s.skip:
            continue
        if s.gate is None or not s.gate_file or not (s.skill or s.harness):
            raise ValueError(f"stage {s.number} needs a skill (or harness code), a gate file "
                             "and a gate")
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
    gate=gate_weights_swap, draft="swap_config")


# A full port the operator allowed (`--full-port`, class `full-port` only): stage 2 has the coder write the port under stages/2/port/ and
# one hardware test compares the chip's tokens with the stage 1 reference (the full-port-check skill, orchard/full_port.py). The other
# stages keep the generic table; the supervisor pauses before stage 3 until the later slices of the design exist.
FULL_PORT_STAGE_2 = dataclasses.replace(
    STAGES[2], name="full port on one board and agreement with the CPU reference", skill="full-port-check",
    gate=gate_full_port, draft="full_port")


# A scoring model (a reranker or an embedding model, issue 378, docs/superpowers/specs/2026-11-02-scoring-model-check-design.md): stage 1 records
# a pair set and the Hugging Face scores or vectors instead of generated ids, and stage 2 runs the supervisor's scoring check on the coder's port
# module (skill scoring-check, orchard/scoring_check.py). No agent writes a finish step for stage 2: the check's own record is the result, so
# the gate file is that record.
SCORING_STAGE_1 = dataclasses.replace(
    STAGES[1], name="CPU reference scores or vectors for a scoring model", skill="scoring-reference", refs=(),
    gate_file=scoring_check.PAIRS_FILE.removeprefix("stages/1/"), gate=gate_scoring_reference)
SCORING_STAGE_2 = dataclasses.replace(
    STAGES[2], name="scoring model on one board and parity with the CPU reference", skill="scoring-check",
    gate_file=scoring_check.RECORD_FILE, gate=gate_scoring, draft="scoring_check")


def scoring_routed(kind, path, cls, full_port: bool) -> bool:
    """Whether a run of a scoring kind takes the scoring route: class `weights-only` (the backbone is the nearest model's and a port module
    reads its output), or class `full-port` with the operator's `--full-port` (the port also carries new model code). Any other class has
    no scoring route yet, and the supervisor pauses (SCORING_NOT_ROUTED)."""
    if kind not in model_kind.SCORING_KINDS:
        return False
    return (path == "weights-only" and cls == "weights-only") or (full_port and path == "full-port" and cls == "full-port")


# The weights+sidecar class (orchard/classes.py): stage 2 also measures the sidecar head's parity in the same
# hardware lease (the weights-sidecar-check skill), and its gate checks both. Every other stage is the
# weights-only spec: the backbone is what those stages test.
SIDECAR_STAGE_2 = dataclasses.replace(
    WEIGHTS_ONLY_STAGE_2, name="weights swap check and sidecar parity on one board",
    skill="weights-sidecar-check", gate=gate_weights_swap_sidecar)


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
    disk=STAGE4_SWAP_DISK_GB, draft="stage4")


# Stage 5 is the black-box serving check: boot, a passkey or needle, a canary. On the weights-only
# path stage 4's serve-and-compare tests already boot the server on each configuration, compare its
# tokens with the CPU reference and check that its free-run text is coherent.
SKIP_5_WEIGHTS_ONLY = ("weights-only path: stage 4's serve-and-compare tests boot each configuration, "
                       "check the output against the CPU reference and check it is coherent (the "
                       "black-box serving check)")
WEIGHTS_ONLY_STAGE_5 = dataclasses.replace(STAGES[5], skip=SKIP_5_WEIGHTS_ONLY)

# Stage 6 is the qualitative check and the benchmark. The operator deferred both on the
# weights-only path (2026-10-03). Stage 4's coherence check is the only quality evidence there.
SKIP_6_WEIGHTS_ONLY = ("weights-only path: the operator deferred the qualitative check and benchmark; "
                       "stage 4 already shows the output is coherent")
WEIGHTS_ONLY_STAGE_6 = dataclasses.replace(STAGES[6], skip=SKIP_6_WEIGHTS_ONLY)

# Stage 7 on the weights-only path of a run started with --package-format v6 (plan 5): supervisor
# code stages a v6 thin package, installs a copy and boots it on one board (orchard/package.py).
# The hardware phase is the one-test phase stages 2 and 3 use, so the coder is parked when no
# board is free.
PACKAGE_STAGE_7 = dataclasses.replace(
    STAGES[7], name="package (v6 thin bundle) and boot check", boards=1, gate_file="package.json",
    gate=gate_package, marker="test-result.json", skip=None, harness=True)
validate_table(tuple(PACKAGE_STAGE_7 if s.number == 7 else s for s in STAGES))


# The stock-runtime path (issue 61). Stage 0 is supervisor code (orchard/stock_runtime.py), because the comparison is with a tt-metal
# parameters directory and not with a weights snapshot. Stage 2 serves the model through tt-transformers on one board (the
# stock-runtime-check skill). Nothing is swapped or ported, so the other stages are skipped with a reason.
STOCK_STAGE_0 = dataclasses.replace(
    STAGES[0], name="intake and stock-runtime triage", skill="", refs=(), gate=gate_delta_stock, marker=None, harness=True)
STOCK_STAGE_2 = dataclasses.replace(
    STAGES[2], name="serve through tt-transformers on one board", skill="stock-runtime-check", refs=("tt-device-usage",),
    gate=gate_stock_runtime, gate_file="evidence/stock-check.json", draft="stock_config")
SKIP_STOCK_3 = "stock-runtime path: stage 2 served the full model through tt-transformers and compared it with the CPU reference"
SKIP_STOCK_4 = ("stock-runtime path: the 2-chip and 4-chip checks are not wired into the supervisor yet (issue 93). stock_check.py ran by hand with "
                "a 2-chip config (P300) gives a record of the same shape (Llama-3.1-8B-Instruct, 2026-10-29); a 4-chip run needs both boards. "
                "The n_kv_heads of the model divide 1, 2 and 4 chips only as a read of the config")
SKIP_STOCK_5 = "stock-runtime path: stage 2 booted the runtime, compared its tokens with the CPU reference and judged a free run"
SKIP_STOCK_6 = ("stock-runtime path: the qualitative check and benchmark are deferred, as on the weights-only path; stage 2's teacher-forced "
                "agreement and free run are the only quality evidence")
SKIP_STOCK_7 = "stock-runtime path: no package is built (the pilot writes no package, no push and no publish)"


# The audio runtime path (orchard/audio_runtime.py, orchard/audio_stages.py): a run whose model the harness's own Stable Audio 3 runtime serves.
# Stage 0 is the normal triage. Stage 1 makes the CPU reference runs, stage 2 makes the clip on one chip and compares it with them, stage 8 builds the
# bundle with the clips. Stages 3 to 7 are skipped with a reason. None of them asks the local coder anything.
def _audio_verb(verb: str, run_dir, args: dict) -> list[str]:
    """Run a numeric audio gate in the configured reference interpreter (the supervisor needs only the standard library). A failure to run it is a
    reason, never a pass."""
    from orchard import audio_remote, audio_runtime
    from orchard.audio_plan import AudioStageError
    paths = audio_runtime.selection_from_run_dir(run_dir)
    if paths is None:
        return ["the ledger holds no audio runtime selection, so there is no interpreter to check the stage with"]
    try:
        result = audio_remote.call(paths.reference_python, verb, args, timeout_s=1800.0, env={"PATH": os.environ.get("PATH", "")})
    except AudioStageError as exc:
        return [str(exc)]
    return [str(x) for x in result] if isinstance(result, list) else ["the audio gate gave an answer that is not a list"]


def gate_audio_reference(stage_dir, run_dir) -> GateResult:
    reasons = _audio_verb("gate_reference", run_dir, {"stage_dir": str(stage_dir), "run_dir": str(run_dir)})
    seen = [os.path.relpath(Path(stage_dir) / audio_runtime.REFERENCE_FILE, run_dir)] if not reasons else []
    return _done(reasons, seen)


def gate_audio_chip(stage_dir, run_dir) -> GateResult:
    ref_stage = Path(run_dir) / "stages" / "1"
    reasons = _audio_verb("gate_chip", run_dir, {"stage_dir": str(stage_dir), "run_dir": str(run_dir), "ref_stage_dir": str(ref_stage)})
    seen = [os.path.relpath(Path(stage_dir) / audio_runtime.RESULT_FILE, run_dir)] if not reasons else []
    return _done(reasons, seen)


AUDIO_STAGE_1 = StageSpec(1, "CPU reference for the audio model", "", (), 0, audio_runtime.REFERENCE_FILE, gate_audio_reference,
                          audio_runtime.REFERENCE_FILE, harness=True, disk=audio_runtime.STAGE_DISK_GB, needs_coder=False)
AUDIO_STAGE_2 = StageSpec(2, "clip on one chip and agreement with the CPU reference", "", ("tt-device-usage",), 1, audio_runtime.RESULT_FILE,
                          gate_audio_chip, "test-result.json", harness=True, disk=audio_runtime.STAGE_DISK_GB, needs_coder=False)
SKIP_AUDIO_3 = ("audio runtime path: stage 2 ran the diffusion transformer on the chip with the library's own code for the rest; there is no "
                "separate full-model stage")
SKIP_AUDIO_4 = ("audio runtime path: one chip runs the diffusion transformer; a multichip port and a shrink to fewer chips are not part of "
                "this path (design step 1 uses one chip)")
SKIP_AUDIO_5 = ("audio runtime path: no server is built (design step 4); the chip run of stage 2 makes the clip directly")
SKIP_AUDIO_6 = ("audio runtime path: no quality check or benchmark exists for music; stage 2's sound checks and its agreement with the "
                "CPU reference are the only evidence, and nobody has listened to the clip")
SKIP_AUDIO_7 = "audio runtime path: no package is built (design step 4 is not built)"
AUDIO_TABLE = {1: AUDIO_STAGE_1, 2: AUDIO_STAGE_2,
               3: dataclasses.replace(STAGES[3], skip=SKIP_AUDIO_3, needs_coder=False),
               4: dataclasses.replace(STAGES[4], skip=SKIP_AUDIO_4, needs_coder=False),
               5: dataclasses.replace(STAGES[5], skip=SKIP_AUDIO_5, needs_coder=False),
               6: dataclasses.replace(STAGES[6], skip=SKIP_AUDIO_6, needs_coder=False),
               7: dataclasses.replace(STAGES[7], skip=SKIP_AUDIO_7, needs_coder=False),
               8: dataclasses.replace(STAGES[8], needs_coder=False)}


# The text-model runtime path (orchard/text_runtime.py, orchard/text_stages.py): a run started with the input `text_runtime=<name>`, for a text model whose
# scripts are in this repository. Stage 0 to 2, 6 and 8 are supervisor code and none of them asks the local coder anything. Stages 3 to 5 are skipped with a
# reason; stage 7 is skipped until the package builder is wired in (issue 61).
def _text_verb(verb: str, run_dir, args: dict) -> list[str]:
    """Run a numeric text gate in the configured interpreter (the supervisor needs only the standard library). A failure to run it is a reason, never a pass."""
    from orchard import text_remote
    from orchard.text_runtime import TextStageError
    sel = _text_rt.selection_from_run_dir(run_dir)
    if sel is None:
        return ["the ledger holds no text runtime selection, so there is no interpreter to check the stage with"]
    try:
        result = text_remote.call(sel[1].python, verb, {**args, "runtime": sel[0]}, timeout_s=1800.0, env={"PATH": os.environ.get("PATH", "")})
    except TextStageError as exc:
        return [str(exc)]
    return [str(x) for x in result] if isinstance(result, list) else ["the text gate gave an answer that is not a list"]


def gate_text_triage(stage_dir, run_dir) -> GateResult:
    """Stage 0 of a text-runtime run. Supervisor code wrote delta.json from the model's own files, so there is no template copy to check. The record has to
    name a registered runtime, the runtime has to accept the model's configuration, and delta.json has to pass the checks every stage 0 passes."""
    d, err = _load(stage_dir, "delta.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    _delta_reasons(d, run_dir, reasons, seen)
    rec = d.get("text_runtime")
    if not isinstance(rec, dict) or _text_rt.get(rec.get("name")) is None:
        reasons.append("delta.json needs a text_runtime record that names a registered text runtime")
        return _done(reasons, seen)
    if rec.get("config_problems"):
        reasons.append(f"the {rec['name']} runtime does not accept this model's configuration: {'; '.join(map(str, rec['config_problems']))[:300]}")
    if d.get("facts", {}).get("modality") != "text":
        reasons.append("a text-runtime run needs facts.modality to be text")
    return _done(reasons, seen)


def gate_text_reference(stage_dir, run_dir) -> GateResult:
    reasons = _text_verb("gate_reference", run_dir, {"stage_dir": str(stage_dir), "run_dir": str(run_dir)})
    return _done(reasons, [os.path.relpath(Path(stage_dir) / _text_rt.REFERENCE_FILE, run_dir)] if not reasons else [])


def gate_text_chip(stage_dir, run_dir) -> GateResult:
    reasons = _text_verb("gate_chip", run_dir, {"stage_dir": str(stage_dir), "run_dir": str(run_dir), "ref_stage_dir": str(Path(run_dir) / "stages" / "1")})
    return _done(reasons, [os.path.relpath(Path(stage_dir) / _text_rt.RESULT_FILE, run_dir)] if not reasons else [])


def gate_text_bench(stage_dir, run_dir) -> GateResult:
    reasons = _text_verb("gate_bench", run_dir, {"stage_dir": str(stage_dir), "run_dir": str(run_dir), "ref_stage_dir": str(Path(run_dir) / "stages" / "1")})
    return _done(reasons, [os.path.relpath(Path(stage_dir) / _text_rt.BENCH_FILE, run_dir)] if not reasons else [])


TEXT_STAGE_0 = StageSpec(0, "intake and triage for a text-runtime model", "", (), 0, "delta.json", gate_text_triage, None, harness=True, needs_coder=False)
TEXT_STAGE_1 = StageSpec(1, "CPU reference for the text model", "", (), 0, _text_rt.REFERENCE_FILE, gate_text_reference, _text_rt.REFERENCE_FILE,
                         harness=True, disk=_text_rt.STAGE_DISK_GB, needs_coder=False)
TEXT_STAGE_2 = StageSpec(2, "one chip: teacher-forced agreement and perplexity against the CPU reference", "", ("tt-device-usage",), 1, _text_rt.RESULT_FILE,
                         gate_text_chip, "test-result.json", harness=True, disk=_text_rt.STAGE_DISK_GB, needs_coder=False)
TEXT_STAGE_6 = StageSpec(6, "benchmark with repeats and the share of the decode floor", "", ("tt-device-usage",), 1, _text_rt.BENCH_FILE, gate_text_bench,
                         "test-result.json", harness=True, disk=_text_rt.STAGE_DISK_GB, needs_coder=False)
SKIP_TEXT_3 = "text runtime path: stage 2 ran the whole model on the chip with the runtime's scripts; there is no separate full-model stage"
SKIP_TEXT_4 = ("text runtime path: one chip runs the model; a multichip port and a shrink to fewer chips are not part of this path (the runtime may ship "
               "a tensor-parallel version that these stages do not run)")
SKIP_TEXT_5 = "text runtime path: no serving integration stage; the package's server is checked when the package is built"
SKIP_TEXT_7 = "text runtime path: the package is built from this run's results by the package builder, not by this stage (issue 61)"
TEXT_TABLE = {0: TEXT_STAGE_0, 1: TEXT_STAGE_1, 2: TEXT_STAGE_2, 6: TEXT_STAGE_6,
              3: dataclasses.replace(STAGES[3], skip=SKIP_TEXT_3, needs_coder=False),
              4: dataclasses.replace(STAGES[4], skip=SKIP_TEXT_4, needs_coder=False),
              5: dataclasses.replace(STAGES[5], skip=SKIP_TEXT_5, needs_coder=False),
              7: dataclasses.replace(STAGES[7], skip=SKIP_TEXT_7, needs_coder=False),
              8: dataclasses.replace(STAGES[8], needs_coder=False)}
TEXT_SPECS = (TEXT_STAGE_0, TEXT_STAGE_1, TEXT_STAGE_2, TEXT_STAGE_6)


def scratch_stage(number: int, path) -> bool:
    """Whether an agent in this stage may write to <run>/tt-metal-scratch: stages 2 to 4 of the stock-runtime path only."""
    return path == "stock-runtime" and number in (2, 3, 4)


def stock_params(entries: list[dict]) -> str | None:
    """The tt-transformers parameters directory the run was started with (run_start's input `stock_params`), or None. Its presence is
    what makes stage 0 supervisor code; a run with none keeps the agent's stage 0."""
    for e in entries:
        if e.get("event") == "run_start":
            value = (e.get("data", {}).get("inputs") or {}).get("stock_params")
            return value if isinstance(value, str) and value else None
    return None


def delta_path(run_dir) -> str | None:
    """The path in stages/0/delta.json, or None when the file is missing, unreadable or names
    another value. The stage 0 gate validated the file; this only reads it."""
    try:
        data = _safe_json(Path(run_dir) / "stages" / "0" / "delta.json")
    except safeio.SafeReadError:
        return None
    path = data.get("path") if isinstance(data, dict) else None
    return path if path in PATHS else None


def delta_facts(run_dir) -> dict | None:
    """The `facts` record of stages/0/delta.json (form and modality; orchard/audio_facts.py), or None when the file has none or
    is unreadable. The stage 0 gate validated it; this only reads it."""
    try:
        data = _safe_json(Path(run_dir) / "stages" / "0" / "delta.json")
    except safeio.SafeReadError:
        return None
    facts = data.get("facts") if isinstance(data, dict) else None
    return dict(facts) if isinstance(facts, dict) else None


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


def delta_class(run_dir) -> str | None:
    """The class in stages/0/delta.json (a delta.json with a path and no class implies one), or None when
    the file is missing, unreadable, or names a value that is not a class."""
    try:
        data = _safe_json(Path(run_dir) / "stages" / "0" / "delta.json")
    except safeio.SafeReadError:
        return None
    if not isinstance(data, dict):
        return None
    if "class" in data:
        return data["class"] if data["class"] in CLASSES else None
    return class_of_path(data.get("path"))


def run_class(entries: list[dict], run_dir) -> str | None:
    """The class this run follows, read the way run_path reads the path: stage 0's passing stage_end holds
    it, so a resumed run uses the class the first attempt chose even if delta.json is edited later. An
    older ledger with only a path gets the class that path implies; one with neither reads delta.json."""
    end = None
    for e in entries:
        if e["event"] == "stage_end" and e["stage"] == 0 and e["data"].get("result") == "pass":
            end = e["data"]
    if end is None:
        return None
    if "class" in end:
        return end["class"] if end["class"] in CLASSES else None
    if "path" in end:
        return class_of_path(end["path"])
    return delta_class(run_dir)


def run_facts(entries: list[dict], run_dir) -> dict | None:
    """The run's form, serving and modality (orchard/audio_facts.py), read the way run_class reads the class: from stage 0's passing
    `stage_end`, so a resume uses what the first attempt decided. A ledger whose stage 0 passed without them is a run from before the
    facts and gets the default (full weights, a thin bundle, text). None before stage 0 passes, or when the record fails the check."""
    end = None
    for e in entries:
        if e["event"] == "stage_end" and e["stage"] == 0 and e["data"].get("result") == "pass":
            end = e["data"]
    if end is None:
        return None
    if "facts" not in end:
        return dict(audio_facts.DEFAULT_FACTS)
    facts = end["facts"]
    return dict(facts) if not audio_facts.facts_problems(facts) else None


def run_model_kind(entries: list[dict]) -> str:
    """The model kind of the run (orchard/model_kind.py), read from stage 0's passing `stage_end` like the class: a resumed run keeps the kind
    the first attempt decided. A ledger written before kinds, or one whose record is malformed, gives "generative", which is every run
    before the scoring check."""
    end = None
    for e in entries:
        if e["event"] == "stage_end" and e["stage"] == 0 and e["data"].get("result") == "pass":
            end = e["data"]
    rec = end.get("model_kind") if end is not None else None
    kind = rec.get("kind") if isinstance(rec, dict) else None
    return kind if kind in model_kind.KINDS else "generative"


def spec_for(number: int, path: str | None, package_format: str | None = None,
             cls: str | None = None, *, stock: bool = False, audio_runtime: bool = False, text_runtime: bool = False,
             full_port: bool = False, kind: str = "generative") -> StageSpec:
    """The stage spec for `number` on `path`. Only the weights-only path changes the table: stage 2
    gets the swap skill and gate, stage 4 runs one test per chip configuration, stages 3, 5 and 6
    are skipped (the supervisor records each as skipped), and when the run was started with
    --package-format v6, stage 7 packages the model. Every other run skips stage 7."""
    if text_runtime and number in TEXT_TABLE:
        return TEXT_TABLE[number]
    if number == 0 and stock:
        return STOCK_STAGE_0
    if audio_runtime and number in AUDIO_TABLE:
        return AUDIO_TABLE[number]
    if number in (1, 2) and scoring_routed(kind, path, cls, full_port):
        return SCORING_STAGE_1 if number == 1 else SCORING_STAGE_2       # a scoring model has no generated tokens to compare
    if full_port and number == 2 and path == "full-port" and cls == "full-port":
        return FULL_PORT_STAGE_2          # only stage 0's own `full-port` answer: `unknown` and `restricted-head` keep their pauses
    if path == "stock-runtime":
        stock_table = {2: STOCK_STAGE_2, 3: dataclasses.replace(STAGES[3], skip=SKIP_STOCK_3),
                       4: dataclasses.replace(STAGES[4], skip=SKIP_STOCK_4), 5: dataclasses.replace(STAGES[5], skip=SKIP_STOCK_5),
                       6: dataclasses.replace(STAGES[6], skip=SKIP_STOCK_6), 7: dataclasses.replace(STAGES[7], skip=SKIP_STOCK_7)}
        if number in stock_table:
            return stock_table[number]
    if path == "weights-only":
        weights_only = {2: SIDECAR_STAGE_2 if cls == "weights+sidecar" else WEIGHTS_ONLY_STAGE_2, 3: WEIGHTS_ONLY_STAGE_3, 4: WEIGHTS_ONLY_STAGE_4,
                        5: WEIGHTS_ONLY_STAGE_5, 6: WEIGHTS_ONLY_STAGE_6}
        if number in weights_only:
            return weights_only[number]
        if number == 7 and package_format == "v6":
            return PACKAGE_STAGE_7
    return STAGES[number]


FULL_PORT_ALLOWED = "full-port route allowed"


def full_port_enabled(entries: list[dict]) -> bool:
    """Whether the operator allowed the full-port route (`--full-port`): run_start says so, or a later decision does (the flag came on a
    resume of a run that has not started stage 2). The ledger is the answer, so a resume without the flag keeps it."""
    for e in entries:
        if e["event"] == "run_start" and e["data"].get("full_port") is True:
            return True
        if e["event"] == "decision" and e["data"].get("decision") == FULL_PORT_ALLOWED:
            return True
    return False


def full_port_rehearsal(entries: list[dict]) -> bool:
    """Whether the operator started the run with `--full-port-rehearsal` (issue 357). Only run_start counts: the option waives a gate field,
    so it cannot be added later by a decision, and nothing the coder writes is read."""
    return any(e["event"] == "run_start" and e["data"].get("full_port_rehearsal") is True for e in entries)


PACKAGE_OPTIONS_SET = "package options set"


def package_options(entries: list[dict]) -> dict | None:
    """The run's stage 7 options: run_start's `package`, or a later "package options set" decision
    (a run started without them may get them on a resume, before stage 7 has started)."""
    opts = None
    for e in entries:
        if e["event"] == "run_start":
            opts = e["data"].get("package")
        elif e["event"] == "decision" and e["data"].get("decision") == PACKAGE_OPTIONS_SET:
            opts = e["data"].get("package")
    return opts if isinstance(opts, dict) else None


def package_format(entries: list[dict]) -> str | None:
    """The run's package format ("v6"), or None when the run builds no package."""
    fmt = (package_options(entries) or {}).get("format")
    return fmt if fmt in PACKAGE_FORMATS else None


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
    cold_boots: int                     # unplanned coder starts slower than COLD_START_S, since the last resume
    park_boots: int                     # park restores slower than COLD_START_S, since the last resume (planned)
    park_boot_s: float                  # the seconds those park restores waited for the coder
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
    escalations = cold_boots = coder_deaths = park_boots = 0
    park_boot_s = 0.0
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
            # A restore ends a planned park (or the recovery of an interrupted one). It is counted apart
            # from the coder starts that nobody planned, so a coder that parks at every hardware stage
            # does not spend the cold-boot cap (issue 317).
            park_boots += 1
            park_boot_s += float(d["seconds"])
        elif ev == "decision":
            what = d.get("decision")
            if what == "pause":
                paused = d.get("reason") or "paused"
            elif what == "resume":
                paused, escalations, cold_boots, coder_deaths = None, 0, 0, 0
                park_boots, park_boot_s = 0, 0.0
                clock_start = ledger_ts(e)
                if d.get("after") == "abort":
                    # The explicit transition (`--resume-aborted`, abort_can_resume). A plain operator resume
                    # never reaches an aborted run, because no supervisor is waiting to read it.
                    aborted = False
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
                       park_boots, park_boot_s, coder_deaths, next_stage)


def abort_can_resume(entries: list[dict]) -> bool:
    """Whether an explicit resume may follow the run's latest abort: the run started and did not finish, it is
    aborted now, and the latest abort was an operator's (`by: operator`) or a signal's (`by: signal SIGINT`).
    An abort of any other kind stays terminal."""
    p = run_progress(entries)
    if not p.started or not p.aborted or p.finished:
        return False
    by = None
    for e in entries:
        if e["event"] == "decision" and e["data"].get("decision") == "abort":
            by = e["data"].get("by")
    return isinstance(by, str) and (by == "operator" or (by.startswith("signal ") and len(by) > len("signal ")))


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
    if p.park_boots >= RUN_PARK_BOOT_CAP:
        return (f"{p.park_boots} cold park restores since the last resume (cap {RUN_PARK_BOOT_CAP}); "
                f"{int(p.park_boot_s / 60)} minutes were spent waiting for the coder")
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
        own = e["event"] == "decision" and d.get("decision") in ("coder starting", "coder adopted", "coder started",
                                                             "coder container started", "coder not started", "coder start pending", "coder start blocked")
        if e["event"] in ("park", "restore") or own:
            if isinstance(d.get("lease"), dict):
                lease = d["lease"]
            if isinstance(d.get("server"), dict):
                server = d["server"]
            if own and isinstance(d.get("canary"), dict):
                canary = d["canary"]
    return lease, server, canary


CODER_START_DECISIONS = ("coder starting", "coder adopted", "coder started", "coder container started", "coder not started",
                         "coder start pending", "coder start blocked")


def coder_start_blocked(entries: list[dict]) -> bool:
    """True when the latest coder start decision is "coder start pending" or "coder start blocked": the granted-board check was under way or
    blocked before `ServerControl.start` ran
    (issue 351). No coder was launched, so a resume retries the check and the launch under that lease. It does not finish a boot, compare
    a canary or count a coder death: nothing of the coder's exists to compare or to have died."""
    last = None
    for e in entries:
        if e["event"] == "decision" and e["data"].get("decision") in CODER_START_DECISIONS:
            last = e["data"]["decision"]
    return last in ("coder start pending", "coder start blocked")


SUPERVISOR_MARKER = "test-result.json"      # written by Supervisor._run_test, which then appends an evidence entry with its path and sha256


# Resume markers other than test-result.json that the supervisor writes itself (or whose writer it records), each with the `what` of the
# evidence entry that vouches for it (issue 344). Supervisor._run_test_list appends "hardware test list" after it writes tests/plan.json;
# the text and audio stage 1 bodies append "text stage record" or "audio stage record" after the reference file is written.
VOUCHED_MARKERS = {
    "tests/plan.json": ("hardware test list",),
    "text_reference.json": ("text stage record",),
    "audio_reference.json": ("audio stage record",),
}


# Both entries of a move-aside are boundaries between attempts. "moving ..." is written before the rename, so a kill after the rename and a second
# kill after the fresh directory is made still leave a boundary in the ledger (review of PR 403). "moved ..." follows the rename.
MOVE_ASIDE_DECISIONS = ("moving the partial stage directory aside", "moved the partial stage directory aside")


def recorded_by_supervisor(ledger, n: int, rec: dict, *, whats: tuple = ()) -> bool:
    """Does the ledger hold an evidence entry for stage `n` with this file's path and sha256? Supervisor._run_test appends one right
    after it writes the file. A file an agent wrote, or one changed after the record, has none.

    With `whats`, the entry must have one of those `what` values instead (the markers of VOUCHED_MARKERS)."""
    # Only _run_test's entry has `returncode` and `timed_out`. AgentStep's "evidence file" entries (an agent can link the marker from
    # evidence/) never do, so they do not count.
    entries = ledger.read()
    # Entries from before the stage's latest "moved the partial stage directory aside" belong to an earlier attempt, whose directory is
    # gone. A later attempt can write a marker with the same bytes at the same path, so those entries cannot vouch for it. A resume
    # appends no such entry, so the current attempt's own record stays valid across restarts. A stage_start with resumed=False is a
    # boundary too: a kill between the rename and the move-aside entry leaves no entry, and the next attempt then starts a fresh
    # directory. A stage_start with resumed=True is not one: it comes after the record the attempt resumed from.
    start = max((i for i, e in enumerate(entries) if e.get("stage") == n and (
        (e.get("event") == "decision" and e["data"].get("decision") in MOVE_ASIDE_DECISIONS)
        or (e.get("event") == "stage_start" and e["data"].get("resumed") is False))), default=-1) + 1
    return any(e.get("event") == "evidence" and e.get("stage") == n and e["data"].get("path") == rec["path"]
               and e["data"].get("sha256") == rec["sha256"]
               and ((e["data"].get("what") in whats) if whats else ("returncode" in e["data"] and "timed_out" in e["data"]))
               for e in entries[start:])


def open_stage_dir(run_dir, spec: StageSpec, *, resuming: bool, ledger) -> tuple[Path, bool]:
    """The stage directory for this attempt, and whether it resumes from the stage's marker.

    A resumed stage with its marker present keeps its directory. Otherwise an existing directory
    is moved aside to `<n>.partial-<k>` (never deleted) and a fresh one is made (spec section 10).
    """
    d = Path(run_dir) / "stages" / str(spec.number)
    if d.exists():
        marker = d / spec.marker if spec.marker else None
        if resuming and marker is not None and marker.is_file():
            try:
                rec = evidence_record(run_dir, marker)
            except safeio.SafeReadError as exc:
                # A marker that the bounded reader refuses (over the cap, a link that leaves the run directory, changed during the read) is not
                # a resume point. The reason is in the ledger, and the directory is moved aside below as for any marker that cannot be used.
                rec = None
                ledger.append("decision", spec.number, decision="marker cannot be read and is not a resume point",
                              marker=spec.marker, reason=exc.reason)
            if rec is None:
                pass
            elif marker.name == SUPERVISOR_MARKER and not recorded_by_supervisor(ledger, spec.number, rec):
                # An agent can write this file. Only the supervisor's own record of its hardware test vouches for it.
                ledger.append("decision", spec.number, decision="marker not written by the supervisor", marker=rec)
            elif spec.marker in VOUCHED_MARKERS and not recorded_by_supervisor(ledger, spec.number, rec, whats=VOUCHED_MARKERS[spec.marker]):
                # The plan and the stage 1 reference files are written by the supervisor too (issue 344). A file with no matching entry
                # after the stage's latest move-aside is not a resume point; the stage directory is moved aside and the step runs again.
                ledger.append("decision", spec.number, decision="marker not written by the supervisor", marker=rec)
            else:
                ledger.append("decision", spec.number, decision="resume from marker", marker=rec)
                return d, True
        k = 1
        while (aside := d.with_name(f"{spec.number}.partial-{k}")).exists():
            k += 1
        ledger.append("decision", spec.number, decision="moving the partial stage directory aside",
                      path=os.path.relpath(aside, run_dir))
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
