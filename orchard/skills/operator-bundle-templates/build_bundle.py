#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 8 template: build the operator bundle from the run's own files (the operator-bundle skill).

The stage 8 agent copies this file into stages/8/, writes stages/8/bundle_config.json next to it:

    {"run_dir": "<the run directory, absolute>"}

and runs `python3 stages/8/build_bundle.py` from the run directory. The script then builds
`<run_dir>/stages/8/bundle/` from scratch, deterministically, from these inputs:

    ledger.jsonl                     stage results, skip reasons and the run's counts
    stages/0/delta.json              differences and hazards
    stages/1/reference.json          the CPU reference checks
    stages/<2..6>/result.json        each stage that passed
    stages/2/swap_config.json, stages/4/configs/<N>/swap_config.json
                                     the tensor cache each test used (weights-only path)
    stages/7/package.json, stages/7/PUBLISH_COMMANDS.txt, stages/7/verify/evidence/verify.json,
    stages/7/package/<name>/         when stage 7 staged a package
    stages/2/audio_result.json       only for an audio run: adds bundle/audio/ (the FLAC clips, none over
                                     20 MB), the "Audio clip on a chip" section of RESULTS.md and the
                                     "Audio run" section of RISKS.md. Other runs get the same bundle as before.

It writes:

    bundle/RESULTS.md            each stage 0 to 8: its result, its numbers with labels (measured or
                                 TODO) and evidence paths relative to the run directory; skipped
                                 stages with the ledger's reason; the run's wall time, escalations,
                                 retries, pauses and operator commands, counted from the ledger
    bundle/RISKS.md              every TODO number, each stage 0 hazard with a `Dealt with:` line,
                                 thin evidence, and the license
    bundle/card.md               the model-level card, with stage 7's package cards copied in
    bundle/PUBLISH_COMMANDS.txt  stage 7's file, byte for byte (comment lines only when stage 7
                                 was skipped)
    bundle/package/              stage 7's package folders without venvs, caches, weights or
                                 unlisted wheels, plus the record, publish commands and cards the
                                 supervisor also copies (supervisor.py, _copy_package)
    bundle/ledger.jsonl          a copy of the run ledger
    evidence/bundle-build.json   every file in bundle/ with its sha256, and what was skipped

Text this script copies from a result file (findings, notes, reasons) has the run's machine paths
replaced by labels such as <CACHE_ROOT>, and any remaining home path, the hostname or a token
removed, so the generated files pass the scrub. Copied package files are not edited.

At the end it runs the scrub the stage 8 gate runs (orchard/scrub.py, scrub_bundle) and prints
every finding. It does not fail on them. The gate does.

The `Dealt with:` line of the tensor-cache hazard is computed: `yes` only when every tensor cache
stage 2 and stage 4 recorded is a distinct directory, each one is empty or holds the
`.orchard-model` marker naming this model, and none lies under the nearest model's cache path (a
path named in the hazard's finding, tt-model's cache, or the bundle a test served). The caches
are checked when this script runs, so a cache deleted since its test gets `Not shown`. Other
hazards say `Not shown` unless a result file holds the measurement that would settle them.

It never touches the network, never starts a process, and writes only under stages/8/bundle and
stages/8/evidence. Exit codes: 0 the bundle was built; 2 a required input is missing or
unreadable (the message names the file).

It needs the tt-orchard checkout for the scrub and the stage table. It finds the checkout from
the ledger's run_start `paths.orchard_dir`, or from an optional `orchard_dir` key in the config.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import sys
import time
from pathlib import Path

CONFIG_KEYS = ("run_dir",)
CONFIG_NAME = "bundle_config.json"
MAX_COPY_BYTES = 1 << 30            # 1 GB; a larger file in a package folder is skipped and listed
MARKER = ".orchard-model"           # written by serve_and_compare.py's cache guard
NC_LINE = "Non-commercial use only."
# The directory names and suffixes orchard/scrub.py's package scrub forbids in a staged package
# (FORBIDDEN_DIRS, FORBIDDEN_SUFFIXES), plus Python and git leftovers. The package copy leaves
# each of them out. tests/test_operator_bundle_template.py checks that these cover scrub.py's sets.
SKIP_DIRS = frozenset({".tt_cache", "tensors", "venv", ".venv", ".python", ".uv", ".hf", ".cache",
                       "model-dir", "__pycache__", ".git"})
SKIP_SUFFIXES = frozenset({".tensorbin", ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".pyc"})
ABS_PATH = re.compile(r"(?<![\w.~-])(/[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)+)")


class Missing(Exception):
    """A required input is missing or unreadable. The message names the file."""


# ---- small helpers ----------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, rel: str, *, required: bool) -> dict | None:
    """The JSON object at `path`, None when it is absent and not required."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if required:
            raise Missing(f"{rel} is missing") from None
        return None
    except (OSError, ValueError) as exc:
        if required:
            raise Missing(f"{rel} is not readable JSON: {exc}") from None
        return None
    if not isinstance(data, dict):
        if required:
            raise Missing(f"{rel} is not a JSON object")
        return None
    return data


def read_ledger(path: Path) -> list[dict]:
    """The ledger's complete lines as entries. A last line without its newline is a write a crash
    cut off (orchard/ledger.py) and is left out."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise Missing("ledger.jsonl is missing") from None
    lines = text.split("\n")[:-1]            # the part after the last newline does not count
    entries = []
    for i, line in enumerate(lines, 1):
        try:
            e = json.loads(line)
        except ValueError:
            raise Missing(f"ledger.jsonl line {i} is not JSON") from None
        if not isinstance(e, dict) or not isinstance(e.get("data"), dict):
            raise Missing(f"ledger.jsonl line {i} is not a ledger entry")
        entries.append(e)
    if not entries:
        raise Missing("ledger.jsonl has no entries")
    return entries


def recorded_paths(entries: list[dict]) -> dict:
    """run_start's paths, or the later "run paths recorded" decision (orchard/paths.py)."""
    found = {}
    for e in entries:
        d = e["data"]
        if e.get("event") == "run_start" and isinstance(d.get("paths"), dict):
            found = d["paths"]
        elif e.get("event") == "decision" and d.get("decision") == "run paths recorded":
            found = d.get("paths") or found
    return found


def fmt(value) -> str:
    if value is None:
        return "TODO"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return str(value)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def labelled(value) -> str:
    """A number with its label, for a table cell."""
    return f"{fmt(value)} (measured)" if is_number(value) else "TODO"


def ev(paths) -> str:
    items = [p for p in (paths or []) if isinstance(p, str) and p]
    return ", ".join(f"`{p}`" for p in items) if items else "-"


def one_line(text) -> str:
    return " ".join(str(text).split())


def end(text) -> str:
    """`text` on one line, ending with a full stop."""
    text = one_line(text)
    return text if not text or text[-1] in ".!?" else text + "."


def chips(n) -> str:
    return f"{n} chip" if n == 1 else f"{n} chips"


def split_model_id(model_id) -> tuple[str, str | None]:
    """'org/name@<40 hex>' -> ('org/name', '<40 hex>'), as orchard/package.py splits it."""
    text = model_id if isinstance(model_id, str) else ""
    repo, sep, rev = text.rpartition("@")
    if sep and re.fullmatch(r"[0-9a-f]{40}", rev):
        return repo, rev
    return text, None


def under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


class Redactor:
    """Replaces the run's machine paths with labels, then removes any remaining home path, the
    hostname and tokens, in text copied from a result file. Counts what it changed."""

    def __init__(self, paths: dict, run_dir: Path, scrub):
        pairs = [(str(run_dir), "<RUN_DIR>")]
        for key, label in (("cache_root", "<CACHE_ROOT>"), ("hf_home", "<HF_HOME>"),
                           ("tt_model_root", "<TT_MODEL_ROOT>"), ("operator_home", "<OPERATOR_HOME>"),
                           ("orchard_dir", "<ORCHARD_DIR>")):
            if isinstance(paths.get(key), str) and paths[key] not in ("", "/"):
                pairs.append((paths[key], label))
        for value, label in list(pairs):
            real = os.path.realpath(value)
            if real != value:
                pairs.append((real, label))
        self.pairs = sorted(pairs, key=lambda p: -len(p[0]))   # the longest path first
        self.scrub = scrub
        self.host = socket.gethostname()
        self.count = 0

    def __call__(self, text) -> str:
        out = str(text)
        for value, label in self.pairs:
            out = out.replace(value, label)
        out = self.scrub.HOME_PATH.sub("<home>", out)
        if self.host and self.host != "localhost":
            out = re.sub(rf"\b{re.escape(self.host)}\b", "<hostname>", out)
        for _, pattern in self.scrub.TOKEN_PATTERNS:
            out = pattern.sub("<token removed>", out)
        if out != str(text):
            self.count += 1
        return out


# ---- what the run produced --------------------------------------------------------------------------

class Run:
    """Everything the bundle is built from, read once. Raises Missing for a required input."""

    def __init__(self, run_dir: Path, entries: list[dict], stages_mod):
        self.dir = run_dir
        self.entries = entries
        self.paths = recorded_paths(entries)
        self.st = stages_mod
        self.path = stages_mod.run_path(entries, run_dir)
        self.cls = stages_mod.run_class(entries, run_dir)
        self.package_format = stages_mod.package_format(entries)
        # An audio run (the harness's own audio runtime) wrote stage 2's audio_result.json and has the audio stage table.
        self.audio_run = (run_dir / "stages" / "2" / "audio_result.json").is_file()
        # A text-runtime run (orchard/text_runtime.py) wrote stage 2's text_result.json and has the text stage table.
        self.text_run = (run_dir / "stages" / "2" / "text_result.json").is_file()
        self.specs = {n: stages_mod.spec_for(n, self.path, self.package_format, self.cls, audio_runtime=self.audio_run, text_runtime=self.text_run)
                      for n in range(9)}
        self.outcome = {}          # stage -> the data of its last stage_end
        self.attempts = {}         # stage -> stage_start entries that were not a skip
        for e in entries:
            n = e.get("stage")
            if e.get("event") == "stage_end" and isinstance(n, int):
                self.outcome[n] = e["data"]
            elif e.get("event") == "stage_start" and isinstance(n, int) and not e["data"].get("skip"):
                self.attempts[n] = self.attempts.get(n, 0) + 1
        s = run_dir / "stages"
        self.delta = read_json(s / "0" / "delta.json", "stages/0/delta.json", required=True)
        # An audio run has no reference.json: stage 1 wrote audio_reference.json, which the audio section reads through stage 2's result.
        self.reference = {} if (self.audio_run or self.text_run) else read_json(s / "1" / "reference.json", "stages/1/reference.json", required=True)
        self.results = {}
        for n in range(2, 7):
            spec = self.specs[n]
            if spec.skip or not spec.gate_file or (self.audio_run and n == 2) or self.text_run:
                continue            # an audio run's stage 2 file is the audio result, which the audio section reads
            rel = f"stages/{n}/{spec.gate_file}"
            self.results[n] = read_json(run_dir / rel, rel, required=self.result(n) == "pass")
        # How each swap check served (the drafter and where sampling happened), as the check recorded it.
        self.serving = {}
        sc = read_json(s / "2" / "evidence" / "swap-check.json", "", required=False) or {}
        if isinstance(sc.get("serving"), dict):
            self.serving[2] = sc["serving"]
        for d in sorted((s / "4" / "configs").glob("*/evidence/swap-check.json")):
            n = d.parent.parent.name
            sc = read_json(d, "", required=False) or {}
            if n.isdigit() and isinstance(sc.get("serving"), dict):
                self.serving[(4, int(n))] = sc["serving"]
        self.swap = {2: read_json(s / "2" / "swap_config.json", "", required=False) or {}}
        for d in sorted((s / "4" / "configs").glob("*/swap_config.json")):
            if d.parent.name.isdigit():
                self.swap[(4, int(d.parent.name))] = read_json(d, "", required=False) or {}
        # Stage 2's graft decision (orchard/graft_trial.py), and which stage 4 bundle configurations were built with the graft
        # (model-dir-sources.json's mtp_graft record; a container configuration has no bundle_dir and is not held to the decision).
        decision_path = s / "2" / "graft-decision.json"
        self.graft_trial_result = read_json(s / "2" / "graft-trial" / "trial-result.json", "", required=False)
        self.graft_decision_present = decision_path.exists()
        self.graft_decision = read_json(decision_path, "", required=False)
        if self.graft_decision is not None and self.graft_decision.get("choice") not in ("graft", "safe"):
            self.graft_decision = None
        # Stage 2's sampling decision (orchard/sampling_choice.py, issue 107): None when there is none or it names no known choice.
        self.sampling_decision = read_json(s / "2" / "sampling-decision.json", "", required=False)
        if not isinstance(self.sampling_decision, dict) or self.sampling_decision.get("choice") not in ("keep", "drafter-off"):
            self.sampling_decision = None
        rec2 = read_json(s / "2" / "model-dir-sources.json", "", required=False) or {}
        self.stage_2_grafted = isinstance(rec2.get("mtp_graft"), dict)       # stage 2's own swap grafted (a person set graft_mtp)
        self.graft_built = {}
        for key, cfg in self.swap.items():
            if key == 2 or not isinstance(cfg, dict) or not cfg.get("bundle_dir"):
                continue
            rec = read_json(s / "4" / "configs" / str(key[1]) / "model-dir-sources.json", "", required=False) or {}
            self.graft_built[key] = isinstance(rec.get("mtp_graft"), dict)
        # stage 7
        self.package = None
        self.publish = None
        self.verify = read_json(s / "7" / "verify" / "evidence" / "verify.json", "", required=False)
        r7 = self.result(7)
        if r7 != "skipped" or (s / "7" / "package.json").is_file():
            self.package = read_json(s / "7" / "package.json", "stages/7/package.json",
                                     required=r7 == "pass")
            try:
                self.publish = (s / "7" / "PUBLISH_COMMANDS.txt").read_bytes()
            except FileNotFoundError:
                raise Missing("stages/7/PUBLISH_COMMANDS.txt is missing: stage 7 produced no "
                              "publish commands, and the bundle must carry them. Stage 7 has to "
                              "write it before stage 8 can build the bundle") from None
            if not self.publish.strip():
                raise Missing("stages/7/PUBLISH_COMMANDS.txt is empty")
        # A stock-runtime run (orchard/stock_runtime.py) has the template's stock-check.json in stage 2's evidence (issue 93).
        self.stock = read_json(run_dir / STOCK_FILE, "", required=False)
        self.text = {name: read_json(run_dir / "stages" / str(n) / name, "", required=False) for n, name in TEXT_FILES}   # a text run's three result files
        self.audio = load_audio(run_dir)    # stage 2's audio_result.json: None for a run without one (see "audio runs")
        self.audio_clips: list[dict] = []   # set by build() once the clips are copied
        self.cards = {}             # profile name -> (run-relative card path, text)
        for p in self.profiles():
            rel = f"{p.get('dir')}/README.md"
            try:
                self.cards[p["name"]] = (rel, (run_dir / rel).read_text(encoding="utf-8"))
            except (OSError, KeyError, TypeError):
                continue

    def result(self, n: int) -> str | None:
        return (self.outcome.get(n) or {}).get("result")

    def profiles(self) -> list[dict]:
        p = (self.package or {}).get("profiles")
        return [x for x in p if isinstance(x, dict)] if isinstance(p, list) else []

    def model(self) -> tuple[str, str | None]:
        repo, rev = split_model_id(self.delta.get("model"))
        if self.package:
            repo = self.package.get("model") or repo
            rev = self.package.get("revision") or rev
        return repo, rev

    def license(self) -> tuple[str | None, bool | None]:
        if self.package and self.package.get("license"):
            return self.package["license"], self.package.get("non_commercial")
        g = read_json(self.dir / "stages" / "0" / "evidence" / "genconfig-license.json", "",
                      required=False) or {}
        lic = (g.get("model_license") or {}).get("license") if isinstance(g.get("model_license"), dict) else None
        return lic, None

    def configs(self) -> list[dict]:
        r = self.results.get(4) or {}
        c = r.get("configs")
        return [x for x in c if isinstance(x, dict)] if isinstance(c, list) else []


# ---- the ledger's counts ------------------------------------------------------------------------------

def ledger_ts(e: dict) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(e.get("ts", ""), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def run_counts(entries: list[dict]) -> list[tuple[str, str]]:
    stamps = [t for t in (ledger_ts(e) for e in entries) if t is not None]
    wall = stamps[-1] - stamps[0] if stamps else None
    count = lambda pred: sum(1 for e in entries if pred(e))         # noqa: E731
    return [
        ("Wall time (first to last ledger entry)",
         f"{wall:.0f} s ({wall / 3600:.2f} h)" if wall is not None else "TODO"),
        ("Escalations", str(count(lambda e: e.get("event") == "escalate"))),
        ("Retries", str(count(lambda e: e.get("event") == "retry"))),
        ("Pauses", str(count(lambda e: e.get("event") == "decision"
                             and e["data"].get("decision") == "pause"))),
        ("Operator commands", str(count(lambda e: e.get("event") == "decision"
                                        and e["data"].get("by") == "operator"))),
        ("Ledger entries", str(len(entries))),
    ]


# ---- the tensor-cache check ---------------------------------------------------------------------------

def caches_used(run: Run) -> list[tuple[str, str]]:
    """(which test, cache directory) for every tensor cache stage 2 and stage 4 recorded."""
    out = []
    r2 = run.results.get(2) or {}
    c2 = r2.get("cache_dir") or run.swap.get(2, {}).get("tt_cache")
    if isinstance(c2, str) and c2:
        out.append(("stage 2", c2))
    for c in run.configs():
        n = c.get("chips")
        cache = c.get("cache_dir") or run.swap.get((4, n), {}).get("tt_cache")
        if isinstance(cache, str) and cache and (c.get("pass") is True or c.get("cache_dir")):
            out.append((f"stage 4, {n} chips", cache))
    return out


def nearest_cache_roots(run: Run) -> list[str]:
    """Paths under which the nearest model's caches live: any absolute path the stage 0
    tensor-cache hazard names, tt-model's cache directory, and each bundle a test served."""
    roots = []
    for h in run.delta.get("hazards") or []:
        if isinstance(h, dict) and h.get("area") == "tensor_cache":
            roots += [m.rstrip(".") for m in ABS_PATH.findall(str(h.get("finding", "")))]
    ttm = run.paths.get("tt_model_root")
    if isinstance(ttm, str) and ttm:
        roots.append(str(Path(ttm).parent))
    for cfg in run.swap.values():
        if isinstance(cfg.get("bundle_dir"), str) and cfg["bundle_dir"]:
            roots.append(cfg["bundle_dir"])
    return sorted({os.path.realpath(r) for r in roots if r.count("/") >= 2})


def tensor_cache_dealt_with(run: Run, redact: Redactor) -> tuple[bool, list[str]]:
    caches = caches_used(run)
    if not caches:
        return False, ["No stage recorded a tensor cache directory, so the run's files cannot "
                       "show which caches were used."]
    model_ids = {run.delta.get("model")} | {c.get("new_model_id") for c in run.swap.values()}
    model_ids = {m for m in model_ids if isinstance(m, str) and m}
    real = [(who, os.path.realpath(path)) for who, path in caches]
    problems = []
    for i, (a_who, a) in enumerate(real):
        for b_who, b in real[i + 1:]:
            if under(a, b) or under(b, a):
                problems.append(f"{a_who} and {b_who} share a cache directory ({redact(a)}).")
    for who, path in real:
        p = Path(path)
        if not p.is_dir():
            problems.append(f"The {who} cache ({redact(path)}) no longer exists, so its marker "
                            "cannot be checked.")
            continue
        if not any(p.iterdir()):
            continue
        marker = p / MARKER
        if not marker.is_file():
            problems.append(f"The {who} cache ({redact(path)}) holds files and no {MARKER} marker.")
        elif marker.read_text(encoding="utf-8", errors="replace").strip() not in model_ids:
            problems.append(f"The {who} cache ({redact(path)}) has a {MARKER} marker that names "
                            "another model.")
    for who, path in real:
        for root in nearest_cache_roots(run):
            if under(path, root):
                problems.append(f"The {who} cache ({redact(path)}) is under the nearest model's "
                                f"cache path {redact(root)}.")
    if problems:
        return False, problems
    listed = "; ".join(f"{who}: {redact(path)}" for who, path in real)
    return True, [f"{len(real)} cache directories, all distinct, each empty or marked with {MARKER} "
                  f"for this model, and none under the nearest model's cache path ({listed}). "
                  "Checked when the bundle was built."]


# ---- RESULTS.md ---------------------------------------------------------------------------------------

def field_table(result: dict, rel: str, redact: Redactor) -> list[str]:
    """Each scalar field of a result file, with its label and evidence."""
    evidence = ev([rel, *(result.get("evidence") or [])])
    rows = ["| Field | Value | Label | Evidence |", "|---|---|---|---|"]
    for key in sorted(result):
        value = result[key]
        if key in ("evidence", "free_run_text", "failure", "cache_dir") or isinstance(value, (dict, list)):
            continue
        label = "measured" if is_number(value) else ("TODO" if value is None else "-")
        rows.append(f"| {key} | {redact(one_line(fmt(value)))} | {label} | {evidence} |")
    lines = rows + [""]
    for key, what in (("free_run_text", "Free-run text"), ("failure", "Recorded failure"),
                      ("cache_dir", "Tensor cache")):
        if isinstance(result.get(key), str) and result[key].strip():
            lines.append(f"{what}: {redact(one_line(result[key]))}")
    for i, n in enumerate(result.get("numbers") or []):        # stage 6
        if i == 0:
            lines += ["", "| Number | Value | Label | Evidence |", "|---|---|---|---|"]
        if isinstance(n, dict):
            shown = f"{fmt(n.get('value'))} {n.get('unit', '')}".strip() if n.get("label") == "measured" else "TODO"
            lines.append(f"| {redact(n.get('name'))} | {shown} | {n.get('label')} | "
                         f"{ev([rel, *(n.get('evidence') or [])]) if n.get('label') == 'measured' else '-'} |")
    checks = result.get("checks")
    if isinstance(checks, dict):                               # stage 5
        lines += ["", "| Check | Pass | Evidence |", "|---|---|---|"]
        for name, c in sorted(checks.items()):
            if isinstance(c, dict):
                lines.append(f"| {name} | {fmt(c.get('pass'))} | {ev(c.get('evidence'))} |")
    return lines


def serving_text(sv: dict) -> str:
    return f"drafter {sv.get('drafter', 'unknown')}; sampling {sv.get('sampling', 'unknown')}"


def parity_section(sp: dict, result: dict) -> list[str]:
    """The sidecar parity measurements of stage 2, each labelled, with the bars the gate used."""
    from orchard import defaults
    rel = next((e for e in result.get("evidence") or [] if isinstance(e, str) and e.endswith("sidecar-parity.json")),
               "stages/2/evidence/sidecar-parity.json")
    evidence = ev(["stages/2/result.json", rel])
    floor = (sp.get("noise_floor") or {}).get("prob_max_abs_diff") if isinstance(sp.get("noise_floor"), dict) else None
    wiring = (sp.get("wiring_check") or {}).get("cosine") if isinstance(sp.get("wiring_check"), dict) else None
    rows = [("n_records", sp.get("n_records")), ("n_questions", sp.get("n_questions")),
            ("hidden_pcc_min", sp.get("hidden_pcc_min")), ("hidden_pcc_mean", sp.get("hidden_pcc_mean")),
            ("prob_max_abs_diff", sp.get("prob_max_abs_diff")), ("top1_agree_fraction", sp.get("top1_agree_fraction")),
            ("cpu_vs_cpu_prob_diff", floor), ("wiring_cosine", wiring)]
    lines = ["", "### Sidecar parity", "",
             "The sidecar head run on the host from the device's final hidden states, against the same head run on "
             "the CPU reference's states, for a few fixed text records.", "",
             "| Field | Value | Label | Evidence |", "|---|---|---|---|"]
    lines += [f"| {k} | {fmt(v)} | {'measured' if is_number(v) else 'TODO'} | {evidence} |" for k, v in rows]
    lines += ["", f"Mesh: {json.dumps(sp.get('mesh_shape'))}. tt-metal: {sp.get('tt_metal_sha')}. Head file sha256 "
              f"{str(sp.get('head_sha256'))[:12]}. Code file sha256 {str(sp.get('code_sha256'))[:12]}.",
              f"Bars the gate used (orchard/defaults.py): hidden_pcc_min at least {defaults.SIDECAR_PCC_MIN}; "
              f"top1_agree_fraction at least {defaults.SIDECAR_TOP1_MIN}; prob_max_abs_diff at most the larger of "
              f"{defaults.SIDECAR_PROB_DIFF_MAX} and {defaults.SIDECAR_NOISE_FACTOR:g} times the CPU-versus-CPU "
              f"difference; at least {defaults.SIDECAR_MIN_QUESTIONS} questions. One model and one small set of "
              "records measured these numbers."]
    return lines


SAMPLING_SOURCES = {
    "base": "the sampling defaults and stop tokens the server uses are the nearest model's ({base})",
    "model": "the sampling defaults and stop tokens the server uses are the model's own",
    "none": "there is no generation_config.json, so the server has no sampling defaults or stop tokens from one",
}


def served_dir_section(run: Run, redact: Redactor) -> list[str]:
    """The "The served model directory" section, from stage 2's `model-dir-sources.json` (written by
    prepare_swap.py). Returns no lines when the file is missing or is not a JSON object, so a run staged before
    the record existed gets the same RESULTS.md as before. Every copied string goes through the redactor."""
    rec = read_json(run.dir / "stages" / "2" / "model-dir-sources.json", "stages/2/model-dir-sources.json",
                    required=False)
    if rec is None:
        return []
    text = lambda v: redact(one_line(v)) if isinstance(v, str) and v else None      # noqa: E731
    base = text(rec.get("base_model")) or "the nearest model"
    lines = ["", "## The served model directory", "",
             "Stage 2 serves a directory built from two models: configuration files from the nearest model "
             f"({base}) and the tokenizer and weights of the model under test. From "
             "`stages/2/model-dir-sources.json`.", ""]
    revs = [f"{label}: `{r}`" for label, r in (("nearest model revision", text(rec.get("base_revision"))),
                                                ("model revision", text(rec.get("model_revision")))) if r]
    if revs:
        lines += [", ".join(revs) + ".", ""]
    rows = []
    for f in rec.get("files") if isinstance(rec.get("files"), list) else []:
        name = text(f.get("name")) if isinstance(f, dict) else None
        if not name:
            continue
        src = text(f.get("source")) or "not recorded"
        how = {"copy": "copied", "link": "linked"}.get(f.get("how"), "not recorded")
        rows.append(f"| `{name}` | {src} | {how} | {text(f.get('reason')) or ''} |")
    w = rec.get("weights")
    if isinstance(w, dict) and is_number(w.get("count")):
        rows.append(f"| weights | {text(w.get('source')) or 'not recorded'} | "
                    f"{ {'copy': 'copied', 'link': 'linked'}.get(w.get('how'), 'not recorded') } | "
                    f"{fmt(w['count'])} safetensors files |")
    if rows:
        lines += ["| File | From | How | Note |", "|---|---|---|---|", *rows, ""]
    absent = [a for a in (text(x) for x in (rec.get("absent") if isinstance(rec.get("absent"), list) else []))
              if a]
    if absent:
        lines += ["Absent from the model's snapshot and not copied from the nearest model: "
                  + ", ".join(f"`{a}`" for a in absent) + ".", ""]
    origin = rec.get("sampling_defaults_from")
    if isinstance(origin, str) and origin in SAMPLING_SOURCES:
        lines.append("Generation config: " + SAMPLING_SOURCES[origin].format(base=base) + ".")
    chat = chat_facts_lines(rec, text)
    if chat:
        lines += ["", *chat]
    return lines


def sampling_section(run: Run, redact: Redactor) -> list[str]:
    """The "Sampling" section (issue 107): for stage 2 and for each stage 4 configuration whose `swap-check.json` holds a `sampling_probe` record, what
    the server did with a request that has temperature above 0. The verdict, the speeds and the drafter's accepted share are computed here from the
    raw requests by orchard/sampling_probe.py, so a summary written into the file is never read. No lines when no record exists, so a run from before
    the probe gets the same RESULTS.md as before. Every copied string goes through the redactor."""
    from orchard import sampling_probe
    where = [("stage 2", "stages/2/evidence/swap-check.json")]
    configs = run.dir / "stages" / "4" / "configs"
    if configs.is_dir():
        for d in sorted((p for p in configs.iterdir() if p.name.isdigit()), key=lambda p: int(p.name), reverse=True):
            where.append((f"stage 4, {int(d.name)} chips", f"stages/4/configs/{d.name}/evidence/swap-check.json"))
    rows, sentences = [], []
    for label, rel in where:
        report = read_json(run.dir / rel, rel, required=False)
        if not isinstance(report, dict) or report.get("sampling_probe") is None:
            continue
        s = sampling_probe.summarize(report["sampling_probe"])
        if s["verdict"] == "not-run":
            continue
        speed = lambda side: fmt(round(side["tokens_per_s"], 1)) if side and side.get("tokens_per_s") else "TODO"      # noqa: E731
        rate = lambda side: fmt(round(side["drafter_accept_rate"], 3)) if side and side.get("drafter_accept_rate") is not None else "TODO"   # noqa: E731
        rows.append(f"| {label} | {s['verdict']} | {speed(s['greedy'])} | {speed(s['sampled'])} | {rate(s['greedy'])} / {rate(s['sampled'])} | "
                    f"{'measured' if s['verdict'] != 'malformed' else 'TODO'} | `{rel}` |")
        text = redact(one_line(s["sentence"]))
        sentences.append(f"- {label}: {text}" + ("" if text.endswith(".") else "."))
        sentences += [f"  - {redact(one_line(r))}" for r in s["reasons"][:3]]
    if not rows:
        return []
    return ["", "## Sampling", "",
            "What the served configuration does with a request that has temperature above 0. Each stage's swap check sends one greedy request and two sampled "
            "requests (temperature above 0, top-p below 1, different seeds) to the server and records them raw; the verdict, speeds and drafter acceptance are "
            "computed from those records. A refused or slow sampled request is a fact about the configuration and does not fail a stage. "
            "The drafter column is the accepted share of drafted tokens for the greedy and sampled requests (TODO when the server logs none).", "",
            "| Configuration | Verdict | Greedy tokens/s | Sampled tokens/s | Drafter acceptance (greedy / sampled) | Label | Evidence |",
            "|---|---|---|---|---|---|---|", *rows, "", *sentences]


SAMPLING_FLAG_RISKS = {
    "no-variant-serves-purpose": "The model needs sampling, and no configuration that stage 2 measured both applies sampling and decodes above the "
                                 "usability floor. The run ships the selected configuration, which does not do what the model needs.",
    "sampling-applied-below-floor": "The model needs sampling. The selected configuration applies sampling but is slow: it decodes below the usability floor. "
                                    "It is shipped because a configuration that does not apply sampling cannot do what the model is for.",
    "unverified": "The model needs sampling, and a verdict or a speed that the choice rests on was not measured.",
    "variant-not-tried": "The model needs sampling, and the served configuration does not show sampling applied. The drafter-off variant was not "
                         "tried, so the served configuration is kept.",
}


def sampling_choice_section(run: Run, redact: Redactor) -> list[str]:
    """The "The sampling choice" section, from stage 2's `sampling-decision.json` (orchard/sampling_choice.py): which configuration the run ships,
    why, and the numbers it used. No lines when the run has no decision file, so a run from before the choice gets the same RESULTS.md as before.
    Every copied string goes through the redactor."""
    d = run.sampling_decision
    if d is None:
        return []
    reason = redact(one_line(d.get("reason"))) if isinstance(d.get("reason"), str) and d["reason"].strip() else "no reason recorded"
    why = [redact(one_line(x)) for x in d.get("need_evidence") or [] if isinstance(x, str)][:3]
    nums = d.get("numbers") if isinstance(d.get("numbers"), dict) else {}
    floor = (d.get("thresholds") or {}).get("min_tok_s") if isinstance(d.get("thresholds"), dict) else None
    lines = ["", "## The sampling choice", "",
             "A speculative drafter can ignore sampling: the Qwen3.8 drafter accepts a request with temperature above 0 and samples only its first "
             "token. Stage 2 therefore chooses the served configuration by what the model is for. It reads whether the model needs sampling from files, "
             "the sampling probe's verdict for the served configuration, and the speed of each configuration. From `stages/2/sampling-decision.json`.", "",
             ("Result: stage 2 chose the drafter-off configuration. " if d["choice"] == "drafter-off" else "Result: stage 2 kept the configuration it served. ")
             + f"{reason}.", "",
             f"Sampling needed: {'yes' if d.get('needed') else 'no'} ({redact(str(d.get('need_source')))}; override {redact(str(d.get('override')))}).",
             *[f"- {w}" for w in why]]
    if nums.get("served_verdict") == "sampled-matches-greedy":
        lines += ["", "The probe's verdict for the configuration stage 2 first served is `sampled-matches-greedy`: suspected, not confirmed. Its sampled replies "
                      "equal the greedy text or each other, which fits sampling not being applied and also happens when one token is eligible at every "
                      "step. The choice treats it like ignored sampling and tries the drafter-off variant."]
    if d.get("flag"):
        lines += ["", f"Flag: `{redact(str(d['flag']))}`. " + SAMPLING_FLAG_RISKS.get(d["flag"], "")]
    where = ev(["stages/2/sampling-decision.json"])
    label = lambda *keys: "measured" if any(is_number(nums.get(k)) for k in keys) else "TODO"        # noqa: E731

    def used(speed_key, basis_key):
        """The speed the rule used, with which request it came from. A greedy figure is a fallback (no usable sampled speed) and is named as one."""
        text = fmt(nums.get(speed_key))
        if not is_number(nums.get(speed_key)):
            return text
        basis = nums.get(basis_key)
        return text + {"sampled": " (sampled requests)", "greedy": " (greedy request; no sampled speed)"}.get(basis, " (basis not recorded)")
    lines += ["", "| Measure | Served configuration | Drafter-off variant | Label | Evidence |", "|---|---|---|---|---|",
              f"| Probe verdict | {redact(str(nums.get('served_verdict')))} | {redact(str(nums.get('variant_verdict')))} | recorded | {where} |",
              f"| Decode speed used for the decision (tokens/s) | {used('served_tok_s', 'served_speed_basis')} | "
              f"{used('variant_tok_s', 'variant_speed_basis')} | "
              f"{label('served_tok_s', 'variant_tok_s')} | {where} |",
              f"| Usability floor (tokens/s) | {fmt(floor)} | {fmt(floor)} | choice | `orchard/defaults.py` |"]
    return lines


def graft_section(run: Run, redact: Redactor) -> list[str]:
    """The "The graft trial" section, from stage 2's `graft-decision.json`: which variant the run ships, why, and both measurements. No lines when
    the run has no decision file, so a run from before the trial gets the same RESULTS.md as before. Every copied string goes through the redactor."""
    d = run.graft_decision
    if d is None:
        return []
    choice = d["choice"]
    reason = redact(one_line(d.get("reason"))) if isinstance(d.get("reason"), str) and d["reason"].strip() else "no reason recorded"
    lines = ["", "## The graft trial", "",
             "The nearest model's `mtp.*` head can be grafted onto a model that ships none, which lets the drafter and on-device sampling stay on. "
             "Stage 2 tries that once after the safe configuration passes its gate, and ships the graft only when the measurement favours it. "
             "From `stages/2/graft-decision.json`.", ""]
    if d.get("trial") is None:
        lines.append(f"Result: the trial did not run, and the safe configuration (drafter off, sampling on the host) was kept. {reason}.")
        return lines
    have_result = run.graft_trial_result is not None
    if not have_result:
        # A trial that timed out or died leaves a decision that still names its result file. Nothing was compared, and the file is not cited.
        lines.append("Result: the trial left no result file, so nothing was compared. "
                     + ("Stage 2 kept the safe configuration (drafter off, sampling on the host). " if choice == "safe" else
                        "The decision file says the graft, but the result it rests on is missing. ") + f"{reason}.")
    else:
        lines.append(("Result: stage 2 chose the graft. " if choice == "graft" else
                      "Result: stage 2 kept the safe configuration (drafter off, sampling on the host). ") + f"{reason}.")
    nums = d.get("numbers") if isinstance(d.get("numbers"), dict) else {}
    where = ev((["stages/2/graft-trial/trial-result.json"] if have_result else []) + ["stages/2/graft-decision.json"])
    rows = [("Decode speed (tokens/s, median of four prompts)", "safe_decode_tok_s", "variant_decode_tok_s"),
            ("Time to first token (s, median of four prompts)", "safe_ttft_s", "variant_ttft_s"),
            ("Top-1 agreement with the CPU reference", "safe_top1_agreement", "variant_top1_agreement")]
    lines += ["", "| Measure | Safe configuration | Grafted variant | Label | Evidence |", "|---|---|---|---|---|"]
    for name, safe_key, var_key in rows:
        safe, var = nums.get(safe_key), nums.get(var_key)
        label = "measured" if is_number(safe) or is_number(var) else "TODO"
        lines.append(f"| {name} | {fmt(safe)} | {fmt(var)} | {label} | {where} |")
    if is_number(nums.get("speedup")):
        lines.append(f"| Decode speed ratio (grafted over safe) | - | {fmt(nums['speedup'])} | measured | {where} |")
    rate_safe, rate_var = nums.get("safe_drafter_accept_rate"), nums.get("variant_drafter_accept_rate")
    lines.append(f"| Drafter acceptance (accepted over drafted tokens) | {fmt(rate_safe) if is_number(rate_safe) else '-'} | {fmt(rate_var)} | "
                 f"{'measured' if is_number(rate_var) else 'TODO'} | {where} |")
    th = d.get("thresholds") if isinstance(d.get("thresholds"), dict) else {}
    if th:
        lines += ["", "The graft is kept when the variant served, was coherent, agreed with the CPU reference at least as well as stage 2's bar "
                  f"({fmt(th.get('top1_min'))}), lost no more than {fmt(th.get('top1_margin'))} against the safe run, and decoded at least "
                  f"{fmt(th.get('min_speedup'))} times as fast. These values are provisional (`orchard/defaults.py`). "
                  + ("Drafter acceptance is shown for information only and does not enter the choice."
                     if is_number(rate_var) else "Drafter acceptance was not measured.")]
    return lines


def trial_gap(run: Run, d: dict) -> str | None:
    """Why the trial did not compare both configurations, or None when it did. The decision file names the trial's result even when the trial
    timed out or died before writing it, and its numbers are empty then, so the file and both speeds have to be on disk."""
    reason = one_line(d.get("reason") or "no reason recorded")
    if run.graft_trial_result is None:
        return (f"The graft trial left no `stages/2/graft-trial/trial-result.json`, so nothing was compared and the safe configuration was kept "
                f"({reason}). Evidence: `stages/2/graft-decision.json`.")
    nums = d.get("numbers") if isinstance(d.get("numbers"), dict) else {}
    missing = [name for name, key in (("the safe configuration's decode speed", "safe_decode_tok_s"),
                                      ("the grafted variant's decode speed", "variant_decode_tok_s")) if not is_number(nums.get(key))]
    if missing:
        return (f"The graft trial's decision holds no measurement of {' and '.join(missing)}, so the two configurations were not compared "
                f"({reason}). Evidence: `stages/2/graft-decision.json`, `stages/2/graft-trial/trial-result.json`.")
    return None


def graft_hazard_dealt_with(run: Run) -> tuple[bool, list[str]]:
    """The `drafter_graft` hazard: a graft was possible. Dealt with when stage 2's trial ran, decided, and everything built afterwards agrees with
    the decision: each stage 4 bundle configuration records the graft when the decision is graft and none does when it is safe, and the same for the
    package when there is one."""
    d = run.graft_decision
    if d is None:
        why = ("stages/2/graft-decision.json is not in this run, so no trial decided whether the graft helps."
               if not run.graft_decision_present else
               "stages/2/graft-decision.json is not readable or names no choice, so the trial's decision cannot be used.")
        return False, [why]
    if d.get("trial") is None:
        reason = one_line(d.get("reason") or "no reason recorded")
        return False, [f"The graft trial was skipped ({reason}). The safe configuration was served and nothing was measured with the graft."]
    why = trial_gap(run, d)
    if why:
        return False, [why]
    choice = d["choice"]
    want = choice == "graft"
    mismatched = sorted(k[1] for k, built in run.graft_built.items() if built != want)
    if mismatched:
        return False, [f"Stage 2's trial chose {'the graft' if want else 'the safe configuration'}, but stage 4's "
                       f"{', '.join(chips(n) for n in mismatched)} bundle configuration "
                       f"{'was built without' if want else 'carries'} the grafted head (`stages/4/configs/N/model-dir-sources.json`)."]
    if run.package is not None:
        packaged = isinstance(run.package.get("mtp_graft"), dict)
        if packaged != want:
            return False, [f"Stage 2's trial chose {'the graft' if want else 'the safe configuration'}, but the package "
                           f"{'carries no graft' if want else 'is grafted'} (`stages/7/package.json`)."]
    follows = ("a grafted head in stage 4's bundle configurations and in the package" if want else
               "no grafted head in stage 4's bundle configurations or in the package")
    nums = d.get("numbers") if isinstance(d.get("numbers"), dict) else {}
    speed = (f" Decode speed {fmt(nums.get('safe_decode_tok_s'))} tokens/s safe, {fmt(nums.get('variant_decode_tok_s'))} grafted." if
             is_number(nums.get("safe_decode_tok_s")) or is_number(nums.get("variant_decode_tok_s")) else "")
    rate = nums.get("variant_drafter_accept_rate")
    accept = (f"The grafted drafter's acceptance rate in the trial was {fmt(rate)} (accepted over drafted tokens)." if is_number(rate)
              else "The drafter's acceptance rate was not measured.")
    return True, [f"Stage 2's graft trial measured both configurations and {'chose the graft' if want else 'kept the safe configuration'}; "
                  f"what was built after it has {follows}.{speed} Evidence: `stages/2/graft-decision.json`, "
                  f"`stages/2/graft-trial/trial-result.json`. {accept}"]


def chat_template_sentence(rec: dict, text) -> str | None:
    """One sentence on the chat template the served directory holds, from the record's `chat_template`. None when the
    record has none or it has the wrong shape (a record from before the field existed)."""
    ct = rec.get("chat_template")
    if not isinstance(ct, dict) or "file" not in ct:
        return None
    file, sha, nearest = ct.get("file"), ct.get("sha256"), ct.get("nearest_sha256")
    if nearest is not None and not isinstance(nearest, str):
        return None
    if file is None:
        return ("the model directory has none, and the nearest model has one." if nearest else
                "the model directory has none, and neither has the nearest model.")
    if not isinstance(file, str) or not isinstance(sha, str):
        return None
    head = f"`{text(file)}` (sha256 {sha[:12]})"
    same = ct.get("same_as_nearest")
    if same is True:
        return head + ", the same as the nearest model's."
    if same is False:
        return head + ", different from the nearest model's."
    return head + ("." if nearest else ". The nearest model has none.")


def chat_facts_lines(rec: dict, text) -> list[str]:
    """The chat template line and, when any differ from the nearest model's, the special tokens line."""
    out = []
    sentence = chat_template_sentence(rec, text)
    if sentence:
        out.append("Chat template: " + sentence)
    diff = rec.get("special_tokens_differ_from_nearest")
    names = [n for n in (text(x) for x in diff) if n] if isinstance(diff, list) else []
    if names:
        out.append("Special tokens that differ from the nearest model's: " + ", ".join(f"`{n}`" for n in names) + ".")
    return out


def stage_results(run: Run, redact: Redactor) -> dict[int, list[str]]:
    """The body of each stage's section."""
    body: dict[int, list[str]] = {}
    d = run.delta
    lines = [f"Path: `{d.get('path')}`. Model: {d.get('model')}. Nearest model: {d.get('nearest_model')}.", ""]
    for reason in d.get("path_reasons") or []:
        lines.append(f"- Path reason: {redact(one_line(reason))}")
    lines += ["### Differences", ""]
    for x in d.get("differences") or []:
        if isinstance(x, dict):
            lines.append(f"- {x.get('area')}: {end(redact(x.get('finding', '')))} "
                         f"Evidence: {ev(x.get('evidence'))}.")
    lines += ["", "### Hazards", "", "RISKS.md says whether a later stage dealt with each one.", ""]
    for x in d.get("hazards") or []:
        if isinstance(x, dict):
            lines.append(f"- {x.get('area')}: {end(redact(x.get('finding', '')))} "
                         f"Evidence: {ev(x.get('evidence'))}.")
    body[0] = lines + ["", "Evidence: `stages/0/delta.json`."]

    r = run.reference
    env = r.get("environment") if isinstance(r.get("environment"), dict) else {}
    lines = [f"Verdict: {r.get('verdict')}."]
    if env:
        lines.append("Environment: " + ", ".join(f"{k} {redact(v)}" for k, v in sorted(env.items())) + ".")
    lines.append("")
    for c in r.get("checks") or []:
        if isinstance(c, dict):
            lines.append(f"- {c.get('name')}: {'pass' if c.get('pass') is True else 'did not pass'}. "
                         f"{end(redact(c.get('note', '')))} Evidence: {ev(c.get('evidence'))}.")
    arts = r.get("reference_artifacts")
    if isinstance(arts, dict) and arts:
        lines += ["", "Reference files: " + ev(list(arts.values())) + "."]
    body[1] = lines + ["", "Evidence: `stages/1/reference.json`."]
    if run.audio_run:
        body[2] = ["The clip, its comparisons with the CPU runs and the control are in the section Audio clip on a chip.", "",
                   "Evidence: `stages/2/audio_result.json`."]
        body[1] = ["The CPU reference runs (the model, its base as the control, and one run in the chip's interpreter) are listed in the section "
                   "Audio clip on a chip.", "", "Evidence: `stages/1/audio_reference.json`."]
    if run.text_run:
        body[1] = [f"The CPU reference runs are described in the section {TEXT_HEADING}.", "", "Evidence: `stages/1/text_reference.json`."]
        body[2] = [f"The chip comparison is in the section {TEXT_HEADING}.", "", "Evidence: `stages/2/text_result.json`."]
        body[6] = [f"The benchmark is in the section {TEXT_HEADING}.", "", "Evidence: `stages/6/text_bench.json`."]

    for n in (2, 3, 5, 6):
        res = run.results.get(n)
        if res is not None:
            body[n] = field_table(res, f"stages/{n}/{run.specs[n].gate_file}", redact)
    if 2 in run.serving and 2 in body:
        body[2] += ["", f"Served with: {serving_text(run.serving[2])}. Evidence: `stages/2/evidence/swap-check.json`."]
    if isinstance(run.results.get(2), dict) and isinstance(run.results[2].get("sidecar_parity"), dict):
        body[2] += parity_section(run.results[2]["sidecar_parity"], run.results[2])

    configs = run.configs()
    if configs:
        req = next((e["data"].get("required_chips") for e in run.entries
                    if e.get("event") == "run_start"), None)
        lines = [f"Required chip counts: {', '.join(map(str, req)) if req else 'not recorded'}.", "",
                 "| Chips | Pass | Serves | server_ready_s | Coherent | top1_agreement | n_tokens | "
                 "Package | Tensor cache | Evidence |", "|---|---|---|---|---|---|---|---|---|---|"]
        failed = []
        for c in sorted(configs, key=lambda c: c["chips"] if is_number(c.get("chips")) else 1 << 30):
            passed = c.get("pass") is True
            lines.append(
                f"| {c.get('chips')} | {'pass' if passed else 'fail'} | "
                f"{fmt(c['serves']) if 'serves' in c and c['serves'] is not None else '-'} | "
                f"{labelled(c.get('server_ready_s'))} | "
                f"{fmt(c['coherent']) if 'coherent' in c and c['coherent'] is not None else '-'} | "
                f"{labelled(c.get('top1_agreement'))} | {labelled(c.get('n_tokens'))} | "
                f"{redact(c.get('package') or '-')} | {redact(c.get('cache_dir') or '-')} | "
                f"{ev(['stages/4/result.json', *(c.get('evidence') or [])])} |")
            if not passed:
                failed.append(c)
        if failed:
            lines += ["", "Configurations that did not pass:", ""]
            for c in failed:
                lines.append(f"- {chips(c.get('chips'))}: {end(redact(c.get('reason') or 'no reason recorded'))} "
                             f"Evidence: {ev(c.get('evidence'))}.")
        served = [(c, run.serving[(4, c["chips"])]) for c in sorted(configs, key=lambda c: c.get("chips") or 0)
                  if (4, c.get("chips")) in run.serving]
        if served:
            lines += ["", "How each configuration served:", ""]
            lines += [f"- {chips(c.get('chips'))}: {serving_text(sv)}. Evidence: "
                      f"`stages/4/configs/{c.get('chips')}/evidence/swap-check.json`." for c, sv in served]
        body[4] = lines
    elif run.results.get(4) is not None:
        body[4] = field_table(run.results[4], "stages/4/result.json", redact)

    if run.package:
        p = run.package
        lic, nc = run.license()
        lines = [f"Stage 7 staged {len(run.profiles())} package(s) of {p.get('model')} at revision "
                 f"`{p.get('revision')}` in format {p.get('format')}. License: {lic}"
                 + (" (non-commercial)." if nc else "."), "",
                 "| Package | Chips | Mesh | Required | Boot check | top1_agreement | server_ready_s | "
                 "Evidence |", "|---|---|---|---|---|---|---|---|"]
        for prof in sorted(run.profiles(), key=lambda x: (x.get("chips") or 0)):
            verified = prof.get("verified") is True
            v = (run.verify or prof.get("verify") or {}) if verified else {}
            v_ev = (v.get("evidence") or ["stages/7/verify/evidence/verify.json"]) if verified else []
            lines.append(
                f"| {prof.get('name')} | {prof.get('chips')} | {prof.get('mesh')} | "
                f"{fmt(prof.get('required'))} | {'passed' if verified else 'not boot-checked'} | "
                f"{labelled(v.get('top1_agreement')) if verified else 'TODO'} | "
                f"{labelled(v.get('server_ready_s')) if verified else 'TODO'} | "
                f"{ev(['stages/7/package.json', *v_ev, run.cards.get(prof.get('name'), ('',))[0]])} |")
        for s in p.get("skipped_profiles") or []:
            if isinstance(s, dict):
                lines.append(f"\nNo package for {chips(s.get('chips'))}: {end(redact(s.get('reason', '')))}")
        lines += ["", "The publish commands are in `stages/7/PUBLISH_COMMANDS.txt`, copied to this "
                  "bundle's `PUBLISH_COMMANDS.txt`. The run never ran them.",
                  "", "Evidence: `stages/7/package.json`."]
        body[7] = lines
    else:
        body[7] = ["No package exists. Stage 7 staged none."]
    return body


def summary(run: Run, n_todo: int, n_hazards: int) -> str:
    passed = [n for n in range(8) if run.result(n) == "pass"]
    skipped = [n for n in range(8) if run.result(n) == "skipped"]
    other = [n for n in range(8) if n not in passed and n not in skipped]
    repo, _ = run.model()
    cls = f" (class {run.cls})" if run.cls and run.cls != run.path else ""
    text = (f"This run brought up {repo} on the {run.path or 'unknown'} path{cls}. "
            f"Stages that passed: {', '.join(map(str, passed)) or 'none'}. "
            f"Stages skipped: {', '.join(map(str, skipped)) or 'none'}. ")
    if other:
        text += f"Stages with another result: {', '.join(map(str, other))}. "
    profiles = run.profiles()
    if run.package:
        ok = sum(1 for p in profiles if p.get("verified") is True)
        text += f"Stage 7 staged {len(profiles)} package(s); {ok} passed its boot check. "
    else:
        text += "No package was staged. "
    text += f"RISKS.md lists {n_todo} TODO item(s) and {n_hazards} stage 0 hazard(s)."
    return text


def run_fact_rows(run: Run, redact: Redactor) -> list[str]:
    """Table rows for the sidecars and the MTP count stage 0 recorded; none for a model without either."""
    rows = []
    for x in run.delta.get("sidecars") or []:
        if isinstance(x, dict):
            rows.append(f"| Sidecar | {redact(str(x.get('file')))} ({(x.get('size') or 0) / 1e6:.1f} MB, "
                        f"{x.get('num_tensors')} tensors, sha256 {str(x.get('sha256'))[:12]}) |")
    m = run.delta.get("mtp")
    if isinstance(m, dict):
        rows.append(f"| MTP tensors | nearest model {m.get('nearest_tensors')}, new model {m.get('new_tensors')} |")
    return rows


REPORTED_FILE = "reported-results.json"
REPORTED_DRAFT = "stages/0/evidence/reported-results.draft.json"
REPORTED_HEADING = "Reported by the authors (not measured here)"
REPORTED_MAX_BYTES = 256 * 1024


def reported_cell(redact: Redactor, text) -> str:
    """A table cell from copied text: on one line, with a pipe replaced so the cell cannot end early."""
    return redact(one_line(text)).replace("|", "¦")


def draft_row_count(run: Run) -> int | None:
    """The number of rows in stage 0's draft of the reported scores, None when there is no readable draft."""
    path = run.dir / REPORTED_DRAFT
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > REPORTED_MAX_BYTES:
            return None
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows")
    except (OSError, ValueError, AttributeError):
        return None
    return len(rows) if isinstance(rows, list) else None


def reported_section(run: Run, redact: Redactor) -> list[str]:
    """The scores the model's authors report, as a person accepted them into `reported-results.json`. They are the authors'
    claims: every row is labelled "reported, not measured here", and nothing else in the bundle, no gate and no verdict reads
    them. A draft that nobody accepted gets one line and no rows. A run with neither file gets no section, so its RESULTS.md
    is the same as before. Every copied string goes through the redactor."""
    from orchard.package import PackageError, read_reported_file         # the checkout is on the path by now
    path = run.dir / REPORTED_FILE
    drafted = draft_row_count(run)
    head = ["", f"## {REPORTED_HEADING}", ""]
    if not (path.exists() or path.is_symlink()):
        if not drafted:
            return []
        return head + [f"Stage 0 drafted {drafted} {'row' if drafted == 1 else 'rows'} from the model card "
                       f"(`{REPORTED_DRAFT}`). Nobody accepted any, so none is in this bundle or on a package card. "
                       "`tt-orchard reported show RUN` lists them and `tt-orchard reported accept RUN` accepts chosen rows."]
    try:
        if path.is_symlink() or not path.is_file():
            raise PackageError("it is not a regular file")
        if path.stat().st_size > REPORTED_MAX_BYTES:
            raise PackageError(f"it is over {REPORTED_MAX_BYTES // 1024} KiB")
        rows, source = read_reported_file(path)
    except (PackageError, OSError) as exc:
        return head + [f"`{REPORTED_FILE}` is not valid ({end(redact(one_line(exc)))}) No row is listed."]
    lines = head + [f"Source: {reported_cell(redact, source)}.", "",
                    "These are the scores the model's authors report, copied from `reported-results.json` as a person accepted "
                    "them. They were not measured in this run: the protocol, tools and judge are the authors', and nothing here "
                    "reproduces them. No pass bar and no verdict uses them.", "",
                    "| Name | Value | Label | Note |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {reported_cell(redact, r.name)} | {reported_cell(redact, r.value)} | reported, not measured here | "
                     f"{reported_cell(redact, r.note) or '-'} |")
    if drafted and drafted != len(rows):
        lines += ["", f"Stage 0 drafted {drafted} rows from the model card; {len(rows)} were accepted."]
    return lines


def inventory_section(run: Run, redact: Redactor) -> list[str]:
    """The "Ops and kernels" section, from stage 0's `ops-inventory.json`. Returns no lines when the
    file is missing or is not a JSON object, so a run without an inventory gets the same RESULTS.md as
    before. Every copied string goes through the redactor."""
    inv = read_json(run.dir / "stages" / "0" / "ops-inventory.json", "stages/0/ops-inventory.json",
                    required=False)
    if inv is None:
        return []
    lines = ["", "## Ops and kernels", "", "From `stages/0/ops-inventory.json`.", ""]
    if inv.get("skipped"):
        lines.append(f"The inventory was skipped: {end(redact(one_line(inv['skipped'])))}")
        return lines
    traced = inv.get("traced_class")
    lines.append(f"Traced class: {redact(one_line(traced)) if isinstance(traced, str) and traced else 'not recorded'}.")
    summary = inv.get("summary") if isinstance(inv.get("summary"), dict) else {}
    distinct = summary.get("distinct") if isinstance(summary.get("distinct"), dict) else {}
    if distinct:
        lines.append("Distinct ops by status: "
                     + ", ".join(f"{redact(str(k))} {redact(fmt(v))}" for k, v in distinct.items()) + ".")
    dump = inv.get("ttnn_dump")
    version = dump.get("ttnn_version") if isinstance(dump, dict) else None
    if version:
        lines.append(f"ttnn dump: version {redact(one_line(version))}.")
    else:
        note = inv.get("ttnn_dump_note")
        note = note if isinstance(note, str) and note else "no reason recorded"
        lines.append("ttnn dump: none. " + end(redact(one_line(note))))
    ops_in = inv.get("ops")
    ops_in = ops_in if isinstance(ops_in, list) else []
    ops = [o for o in ops_in
           if isinstance(o, dict) and o.get("status") == "unmapped" and isinstance(o.get("op"), str)]
    ops.sort(key=lambda o: -(o["scaled_count"] if is_number(o.get("scaled_count")) else 0))
    if ops:
        lines += ["", "Unmapped ops, most frequent first (at most 10):", ""]
        for o in ops[:10]:
            count = o.get("scaled_count")
            shown = redact(fmt(count)) if is_number(count) else "count not recorded"
            lines.append(f"- `{redact(one_line(o['op']))}` ({shown})")
    cands = [o for o in ops_in if isinstance(o, dict) and o.get("status") == "candidate" and isinstance(o.get("op"), str)]
    cands.sort(key=lambda o: -(o["scaled_count"] if is_number(o.get("scaled_count")) else 0))
    if cands:
        lines += ["", "Candidate ops, most frequent first (at most 10). These have no mapping-table row. The ttnn names "
                      "are guesses from the op's name, each with the reason it was proposed; they are not `mapped`:", ""]
        for o in cands[:10]:
            cl = o.get("candidates")
            named = [c for c in (cl if isinstance(cl, list) else []) if isinstance(c, dict) and isinstance(c.get("ttnn"), str)]
            guess = ", ".join(f"`{redact(one_line(c['ttnn']))}` ({redact(one_line(str(c.get('why'))))})" for c in named[:3])
            count = o.get("scaled_count")
            shown = redact(fmt(count)) if is_number(count) else "count not recorded"
            lines.append(f"- `{redact(one_line(o['op']))}` ({shown}): {guess or 'none recorded'}")
    verified = [o for o in ops_in if isinstance(o, dict) and o.get("status") == "verified" and isinstance(o.get("op"), str)]
    if verified:
        lines += ["", "Verified ops (at most 10). Each agreed with a torch reference on the ttsim simulator at one small "
                      "shape. That is not a run on hardware and the ops are not `mapped`:", ""]
        for o in verified[:10]:
            lines.append(f"- `{redact(one_line(o['op']))}` -> `{redact(one_line(str(o.get('ttnn'))))}`")
    pats = inv.get("patterns")
    pats = [f for f in pats if isinstance(f, dict) and isinstance(f.get("pattern"), str)] if isinstance(pats, list) else []
    if pats:
        lines += ["", "Groups of ops that one ttnn op does at once, recognised from the order of the traced ops "
                      "(`pattern`; `pattern-verified` after the simulator check). The ops inside a group keep their own "
                      "statuses:", ""]
        for f in pats[:10]:
            count = f.get("scaled_count")
            shown = redact(fmt(count)) if is_number(count) else "count not recorded"
            lines.append(f"- {redact(one_line(f['pattern']))} -> `{redact(one_line(str(f.get('ttnn'))))}` "
                         f"({redact(one_line(str(f.get('status'))))}, {shown})")
    lines += ["", "`mapped` is a name match against a recorded ttnn dump. It does not say the op supports "
                  "the shape, dtype or layout the model needs."]
    return lines


# ---- audio runs (stage 2's audio_result.json) -----------------------------------------------------------
# A run whose model the harness's own audio runtime serves has `stages/2/audio_result.json`. Only then does the bundle get
# `audio/` (the clips), the "Audio clip on a chip" section of RESULTS.md and the "Audio run" section of RISKS.md. A run without
# that file gets the same bundle as before. The file comes from another stage and is untrusted: every field is checked for its
# type, a missing or wrongly typed field prints "not recorded", and nothing here raises on its content.

AUDIO_RESULT = "stages/2/audio_result.json"
AUDIO_HEADING = "Audio clip on a chip"
AUDIO_MAX_JSON_BYTES = 4 << 20       # a larger audio_result.json is not read
MAX_CLIP_BYTES = 20_000_000          # 20 MB; a larger clip is not copied and the results say so
MAX_NUMBER = 1e15                    # a recorded number beyond this is treated as not recorded
NOT_RECORDED = "not recorded"
# (role, where the clip's path is recorded, key there, file name in bundle/audio/, label)
AUDIO_CLIPS = (("chip", "runs", "chip", "chip.flac", "the fine-tune's clip from the chip run"),
               ("reference", "reference_runs", "reference", "reference.flac", "the fine-tune's clip from the CPU reference"),
               ("control_chip", "runs", "control_chip", "control-chip.flac", "the base model's clip from the chip run"),
               ("control_reference", "reference_runs", "control_reference", "control-reference.flac",
                "the base model's clip from the CPU reference"))
# Each timing key of the chip run, with what it measures. A key the file does not hold prints "not recorded".
AUDIO_TIMINGS = (
    ("generate_and_decode_s", "seconds", "the sampler's loop and the autoencoder's decode, end to end"),
    ("conditioning_s", "seconds", "the T5Gemma encoder turning the prompt into conditioning, on the host"),
    ("model_load_s", "seconds", "loading the model's weights on the host"),
    ("chip_weights_upload_s", "seconds", "copying the diffusion transformer's weights to the chip"),
    ("chip_transformer_call_s", "seconds", "wall time of the diffusion transformer's calls, summed over all passes: host preparation, uploads, the layers on the chip and the read back"),
    ("chip_passes", "passes", "how many times the diffusion transformer ran on the chip"))
AUDIO_AGREEMENTS = (("Chip against the CPU reference", ("agreement", "chip_vs_reference")),
                    ("Chip against the same-environment CPU run", ("agreement", "chip_vs_host_same_env")),
                    ("Environment floor: the same-environment CPU run against the reference", ("environment_floor",)))
AUDIO_TABLE_HEAD = ["| Comparison | Verdict | Final-latent PCC | Final-latent relative L2 | Waveform PCC | Waveform relative L2 | "
                    "Log-spectrum distance | Worst layer | Failed arrays |", "|---|---|---|---|---|---|---|---|---|"]
SAFE_KEY = re.compile(r"[A-Za-z0-9_.-]{1,60}")


def dig(obj, *keys):
    """`obj[k1][k2]...` when every step is a dict that holds the key, else None."""
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


def num(value, digits: int = 4) -> str:
    """A recorded number as text; anything that is not a finite number of sane size is "not recorded"."""
    if not is_number(value) or value != value or abs(value) > MAX_NUMBER:
        return NOT_RECORDED
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def txt(redact: Redactor, value, limit: int = 300) -> str:
    """A recorded string as one line for a table cell, shortened and redacted; a non-string is "not recorded"."""
    if not isinstance(value, str) or not value.strip():
        return NOT_RECORDED
    line = one_line(value)
    if len(line) > limit:
        line = line[:limit] + "..."
    return reported_cell(redact, line)


# ---- text-runtime runs (stage 1, 2 and 6 result files) ----------------------------------------------------------------
# A run whose model the harness's own text runtime serves (orchard/text_runtime.py) has stages/1/text_reference.json, stages/2/text_result.json and
# stages/6/text_bench.json. Only then does RESULTS.md get the "Text model on a chip" section and RISKS.md the "Text run" section. The files come from other
# stages and are untrusted: a missing or wrongly typed field prints "not recorded".
TEXT_FILES = ((1, "text_reference.json"), (2, "text_result.json"), (6, "text_bench.json"))
TEXT_HEADING = "Text model on a chip"


def as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def as_list(x) -> list:
    return x if isinstance(x, list) else []


def variants_of(d) -> list[tuple[str, dict]]:
    """The (name, record) pairs of a result's `variants`, with a record that is not an object read as empty and a `variants` that is not an object as none."""
    v = as_dict(as_dict(d).get("variants"))
    return [(str(k), as_dict(x)) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))]


def text_variant_rows(run: Run, redact: Redactor) -> list[str]:
    res, bench = as_dict(run.text.get("text_result.json")), as_dict(run.text.get("text_bench.json"))
    head = ["| Variant | Decode tok/s (median, min to max) | Timing label | Share of floor | Teacher-forced agreement | Worst block error over floor | "
            "Perplexity (chip, reference) | Stage 2 bar |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    bench_by_name = dict(variants_of(bench))
    for name, v in variants_of(res):
        b = bench_by_name.get(name, {})
        spread = as_dict(b.get("decode_tok_s"))
        agree = as_dict(dig(v, "numeric", "agreement"))
        ratio = as_dict(dig(v, "numeric", "worst_ratio_to_floor"))
        ppl = as_dict(v.get("perplexity"))
        share = b.get("share_of_floor_median")
        rows.append(f"| {txt(redact, name)} | {num(spread.get('median'), 3)} ({num(spread.get('min'), 3)} to {num(spread.get('max'), 3)}) | {txt(redact, b.get('label'))} | "
                    f"{num(share * 100, 3) + ' percent' if is_number(share) else NOT_RECORDED} | {num(agree.get('matches'))} of {num(agree.get('of'))} | "
                    f"{num(ratio.get('ratio'), 3)} at {txt(redact, ratio.get('name'))} | {num(ppl.get('chip_ppl'), 5)}, {num(ppl.get('reference_ppl'), 5)} | "
                    f"{'met' if v.get('ok') is True else 'not met'} |")
    return head + rows if rows else ["No variant is recorded in `stages/2/text_result.json`."]


def text_section(run: Run, redact: Redactor) -> list[str]:
    """The "Text model on a chip" section of RESULTS.md. No lines for a run without text_result.json."""
    if not run.text_run:
        return []
    res, ref, bench = (as_dict(run.text.get(n)) for n in ("text_result.json", "text_reference.json", "text_bench.json"))
    lines = ["", f"## {TEXT_HEADING}", "",
             f"Runtime: {txt(redact, res.get('runtime'))}. Evidence: `stages/1/text_reference.json`, `stages/2/text_result.json`, `stages/6/text_bench.json`. "
             "Each number below is recomputed by the stage gate from the recorded arrays and files.", "",
             f"The reference is a float32 CPU run of {num(ref.get('decode_tokens'))} greedy tokens after a {num(ref.get('prompt_tokens'))}-token prompt. "
             "The chip runs follow the reference's token path (teacher forcing).", "",
             f"Rule: {txt(redact, res.get('rule'), 600)}", ""]
    lines += text_variant_rows(run, redact)
    for name, e in variants_of(bench):
        lines += ["", f"Floor, {txt(redact, name)}: {txt(redact, e.get('share_text'), 400)}"]
    lines += ["", "### What these numbers do not show", ""]
    lines += [f"- {txt(redact, x, 500)}" for x in as_list(res.get("limits")) + as_list(bench.get("limits"))] or ["- No limits are recorded."]
    return lines


def text_risks(run: Run, redact: Redactor) -> list[str]:
    """The "Text run" section of RISKS.md: a stage 2 bar that was not met first, then the fixed limits. No lines without text_result.json."""
    if not run.text_run:
        return []
    res = as_dict(run.text.get("text_result.json"))
    lines = ["## Text run", ""]
    for name, v in variants_of(res):
        if v.get("ok") is not True:
            lines.append(f"- Variant {txt(redact, name)} did not meet the stage 2 bar: "
                         f"{txt(redact, '; '.join(map(str, as_list(dig(v, 'numeric', 'reasons')) + as_list(dig(v, 'perplexity', 'reasons')))), 400)}. "
                         "Evidence: `stages/2/text_result.json`.")
    lines += ["- The chip bar is a multiple of the CPU bfloat16 error and is provisional. Evidence: `stages/2/text_result.json`.",
              "- Agreement and perplexity cover one prompt and one text window. Evidence: `stages/2/text_result.json`.", ""]
    return lines


# ---- stock-runtime runs (stage 2's evidence/stock-check.json) ---------------------------------------------------------
# A run that tt-transformers already supports has the stock_check.py template's stock-check.json. Only then does RESULTS.md get the "Stock runtime on a chip"
# section and RISKS.md the "Stock runtime run" section. The file is written by a template and read again here as untrusted: a missing or wrongly typed field
# prints "not recorded".
STOCK_FILE = "stages/2/evidence/stock-check.json"
STOCK_HEADING = "Stock runtime on a chip"


STOCK_PATCH_MAX_BYTES = 2_000_000       # the template cuts its patch at this size (PATCH_LIMIT in stock_check.py); a larger file is not read
STOCK_PATCH_NAME = re.compile(r"evidence/[A-Za-z0-9._-]+")


def read_stock_patch(run: Run, name) -> list[str] | None:
    """The lines of the patch file `name` in stage 2, or None. The name comes from the evidence file, which is untrusted: it has to be `evidence/<one file name>`,
    the file has to be a regular file that is not a link (it is opened with O_NOFOLLOW, so a link in the last place is refused, and the directory is checked
    to be real), and it is read through a size limit. A path such as /dev/zero, a path with `..` or a link out of the stage gives None (Copilot, PR 275)."""
    if not isinstance(name, str) or not STOCK_PATCH_NAME.fullmatch(name) or name in ("evidence/.", "evidence/.."):
        return None
    stage = run.dir / "stages" / "2"
    folder = stage / "evidence"
    if folder.is_symlink() or not folder.is_dir():
        return None
    try:
        fd = os.open(folder / name.split("/", 1)[1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(STOCK_PATCH_MAX_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    return None if len(data) > STOCK_PATCH_MAX_BYTES else data.decode("utf-8", errors="replace").splitlines()


def patch_summary(run: Run) -> str:
    """Lines added and removed by the scratch patch of the host tree, counted from the patch file; "not recorded" when it cannot be read safely."""
    lines = read_stock_patch(run, dig(run.stock, "scratch_patch"))
    if lines is None:
        return NOT_RECORDED
    if not any(ln.strip() for ln in lines):
        return "no change to the copied host tree paths"
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return f"{added} lines added, {removed} removed"


def stock_section(run: Run, redact: Redactor) -> list[str]:
    """The "Stock runtime on a chip" section of RESULTS.md. No lines for a run without stock-check.json."""
    if run.stock is None:
        return []
    d = as_dict(run.stock)
    t, sp, ref, tree = (as_dict(d.get(k)) for k in ("teacher_forced", "speed", "reference", "tt_metal"))
    free = as_dict(d.get("free_run"))
    dirty = tree.get("dirty")
    rows = [("Board name the runtime saw", txt(redact, d.get("device_name"))), ("Chips", num(d.get("chips"))),
            ("Optimization mode", txt(redact, d.get("optimizations"))),
            ("Teacher-forced tokens compared", num(t.get("tokens"))), ("Top-1 agreement with the CPU reference", num(t.get("top1"), 5)),
            ("Top-5 agreement with the CPU reference", num(t.get("top5"), 5)),
            ("Decode speed (tokens per second per user, batch 1)", num(sp.get("decode_tokens_per_s"), 4)),
            ("Time to first token (seconds)", num(sp.get("ttft_s"), 3)),
            ("Free run", f"{num(free.get('new_tokens'))} new tokens, finish {txt(redact, free.get('finish'))}"),
            ("Reference source", txt(redact, ref.get("source"))), ("Reference model", txt(redact, ref.get("model_id"))),
            ("Reference snapshot", txt(redact, ref.get("snapshot"))), ("Reference file sha256", txt(redact, ref.get("sha256"))),
            ("tt-metal tag", txt(redact, tree.get("tag"))),
            ("tt-metal tree", "uncommitted changes" if dirty is True else "no uncommitted changes" if dirty is False else NOT_RECORDED),
            ("Scratch patch", f"{txt(redact, d.get('scratch_patch'))}: {patch_summary(run)}"),
            ("Fetched from the Hub during the run", f"{len(as_list(d.get('hub_fetch')))} fetches" if isinstance(d.get("hub_fetch"), list) else NOT_RECORDED),
            ("Wall time of the hardware test (seconds)", num(d.get("seconds"), 5))]
    lines = ["", f"## {STOCK_HEADING}", "",
             "tt-transformers served the model as it is, from a scratch copy of the host tt-metal tree's `models` directory. Evidence: "
             f"`{STOCK_FILE}`. The stage 2 gate judges the accuracy and speed fields again from this file.", "",
             "| Field | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += ["", "### What these numbers do not show", "",
              "- Top-1 and top-5 are teacher-forced against a float CPU reference over one text; they are not a quality benchmark.",
              "- The speed is one batch-1 demo run on the board named above. It is not a serving measurement.",
              "- Only the board named above was measured. A run on a different chip count has its own record."]
    return lines


def stock_risks(run: Run, redact: Redactor) -> list[str]:
    """The "Stock runtime run" section of RISKS.md. No lines for a run without stock-check.json."""
    if run.stock is None:
        return []
    d = as_dict(run.stock)
    lines = ["## Stock runtime run", ""]
    if as_dict(d.get("tt_metal")).get("dirty") is True:
        lines.append(f"- The host tt-metal tree has uncommitted changes, so the runtime is not a published build. Evidence: `{STOCK_FILE}`.")
    patch = patch_summary(run)
    if patch not in (NOT_RECORDED, "no change to the copied host tree paths"):
        lines.append(f"- The scratch patch changed the stock code ({patch}). The measured model is not the stock code until the patch is upstream. "
                     f"Evidence: `{STOCK_FILE}`.")
    lines += ["- No package was built for this model. A package needs the base-bundle step of issue 62.", ""]
    return lines


def load_audio(run_dir: Path):
    """None when the run has no audio_result.json. Otherwise a dict: the file's content when it is a readable JSON object
    of a sane size, and {"_unreadable": <reason>} when it is not."""
    path = run_dir / AUDIO_RESULT
    if not (path.exists() or path.is_symlink()):
        return None
    try:
        if path.is_symlink() or not path.is_file():
            return {"_unreadable": "it is not a regular file"}
        if path.stat().st_size > AUDIO_MAX_JSON_BYTES:
            return {"_unreadable": f"it is larger than {AUDIO_MAX_JSON_BYTES >> 20} MB"}
    except OSError:
        return {"_unreadable": "it cannot be read"}
    data = read_json(path, AUDIO_RESULT, required=False)
    return data if data is not None else {"_unreadable": "it is not a readable JSON object"}


def audio_summary_state(summary) -> str:
    """'ok', 'failed' or 'unknown' for one agreement summary."""
    if not isinstance(summary, dict):
        return "unknown"
    if summary.get("ok") is False or summary.get("comparable") is False:
        return "failed"
    return "ok" if summary.get("ok") is True else "unknown"


def copy_audio(run: Run, bundle: Path) -> list[dict]:
    """bundle/audio/: the clips the result names, under fixed file names. A clip that is not a regular .flac file inside the
    run directory, or is over MAX_CLIP_BYTES, is not copied; the record says why. Returns one record per clip the file names:
    {"role", "name", "label", "copied", "reason", "bytes"}."""
    out = []
    data = run.audio
    if not data or "_unreadable" in data:
        return out
    root = os.path.realpath(run.dir)
    for role, where, key, name, label in AUDIO_CLIPS:
        entry = dig(data, where, key)
        if not isinstance(entry, dict) or "clip" not in entry:
            continue
        rec = {"role": role, "name": name, "label": label, "copied": False, "reason": None, "bytes": None}
        out.append(rec)
        rel = entry.get("clip")
        if not isinstance(rel, str) or not rel or os.path.isabs(rel) or "\0" in rel:
            rec["reason"] = "its recorded path is not a path inside the run directory"
            continue
        src = Path(os.path.join(root, rel))
        real = os.path.realpath(src)
        if os.path.commonpath([root, real]) != root or src.is_symlink() or not src.is_file():
            rec["reason"] = "its recorded path is not a regular file inside the run directory"
            continue
        if src.suffix.lower() != ".flac":
            rec["reason"] = "it is not a .flac file"
            continue
        size = src.stat().st_size
        rec["bytes"] = size
        if size > MAX_CLIP_BYTES:
            rec["reason"] = f"it is {size} bytes, over the {MAX_CLIP_BYTES // 1_000_000} MB limit"
            continue
        dest = bundle / "audio" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        rec["copied"] = True
    return out


def audio_clip_line(rec: dict) -> str:
    if rec["copied"]:
        return f"- [{rec['label']}](audio/{rec['name']}) (`audio/{rec['name']}`, {rec['bytes']} bytes)"
    return f"- {rec['label']}: not copied into this bundle, because {rec['reason']}."


def audio_agreement_row(redact: Redactor, label: str, s) -> str:
    if not isinstance(s, dict):
        return f"| {label} | {NOT_RECORDED} |" + f" {NOT_RECORDED} |" * 7
    state = audio_summary_state(s)
    if state == "ok":
        verdict = "within the bars"
    elif state == "unknown":
        verdict = NOT_RECORDED
    else:
        verdict = "not comparable" if s.get("comparable") is False and s.get("ok") is not False else "outside the bars"
    lat, wav, worst = dig(s, "latent_final"), dig(s, "waveform"), dig(s, "worst_layer")
    if isinstance(worst, dict) and isinstance(worst.get("name"), str):
        worst_text = f"{txt(redact, worst['name'], 80)} (relative L2 {num(worst.get('rel_l2'))}, PCC {num(worst.get('pcc'))})"
    else:
        worst_text = NOT_RECORDED
    failed = s.get("failed")
    if isinstance(failed, list):
        names = [txt(redact, f, 80) for f in failed[:20] if isinstance(f, str)]
        failed_text = (", ".join(names) + (f" and {len(failed) - 20} more" if len(failed) > 20 else "")) if names else "none"
    else:
        failed_text = NOT_RECORDED
    return (f"| {label} | {verdict} | {num(dig(lat, 'pcc'))} | {num(dig(lat, 'rel_l2'))} | {num(dig(wav, 'pcc'))} | "
            f"{num(dig(wav, 'rel_l2'))} | {num(dig(wav, 'log_spectrum_distance'))} | {worst_text} | {failed_text} |")


AUDIO_REPEAT_PARTS = (("generate_and_decode_s", "the whole `generate` call and decode"), ("chip_transformer_call_s", "the diffusion transformer's calls"),
                      ("conditioning_s", "the T5Gemma encoder on the host"))
AUDIO_REPEAT_ROLES = (("chip", "reference", "Fine-tune, 8 steps"), ("control_chip", "control_reference", "Base control, 50 steps, guidance 7"))
AUDIO_MIN_REPEATS = 3            # fewer measured repeats is a single sample (orchard/resident.py MIN_REPEATS; orchard/stages.py gate_bundle checks the same heading)
AUDIO_BENCH_HEADING = "### Timing repeats: benchmark"
AUDIO_SINGLE_HEADING = "### Timing repeats: single sample"
AUDIO_UNWARMED_HEADING = "### Timing repeats: unwarmed"
AUDIO_MIN_WARMUP = 1             # discarded runs before the repeats (orchard/resident.py WARMUP_RUNS)


def audio_good_values(lst, key: str) -> list:
    """The finite numbers of timing `key` over a role's repeats (at most 20 entries are read)."""
    vals = [dig(e, "timings", key) if isinstance(e, dict) else None for e in (lst[:20] if isinstance(lst, list) else [])]
    return [v for v in vals if is_number(v) and abs(v) <= MAX_NUMBER]


def audio_repeat_counts(data: dict) -> dict:
    """Per role, how many repeats have a measured whole-call time. The builder counts these itself; a label written in the result decides nothing.
    A role the result recorded a run for (or lists repeats for) counts zero when its repeat list is missing, empty or not a list."""
    reps = dig(data, "repeats")
    runs = dig(data, "runs")
    out = {}
    for role, _ref, _label in AUDIO_REPEAT_ROLES:
        listed = isinstance(reps, dict) and role in reps
        recorded = isinstance(runs, dict) and role in runs
        if listed or recorded:
            lst = reps.get(role) if isinstance(reps, dict) else None
            out[role] = len(audio_good_values(lst, "generate_and_decode_s")) if isinstance(lst, list) else 0
    return out


def audio_repeat_lines(data: dict) -> list[str]:
    """The timing repeats of stage 2 as plain tables. With at least AUDIO_MIN_REPEATS measured repeats for every model that was repeated, the section is
    a benchmark: per model and part the seconds of each repeat and their min, median and max, with the CPU reference's one sample for the whole call, the
    load that the single loaded process reported once, and the host load average. With fewer it is headed as a single sample. Missing or malformed values
    print as not recorded; nothing here raises."""
    import statistics
    reps = dig(data, "repeats")
    counts = audio_repeat_counts(data)
    enough = bool(counts) and min(counts.values()) >= AUDIO_MIN_REPEATS
    warm = dig(data, "timing", "warmup_runs")
    warmed = isinstance(warm, int) and not isinstance(warm, bool) and warm >= AUDIO_MIN_WARMUP
    bench = enough and warmed
    lines = ["", AUDIO_BENCH_HEADING if bench else (AUDIO_UNWARMED_HEADING if enough else AUDIO_SINGLE_HEADING), ""]
    if not any(counts.values()) and not (isinstance(reps, dict) and any(isinstance(v, list) and v for v in reps.values())):
        return lines + [f"The chip run was not repeated, or the repeats are {NOT_RECORDED}. The timings above are a single sample: one run, with no spread."]
    warm_text = (f"{warm} discarded warm-up run(s) in the same process before the repeats" if isinstance(warm, int) and not isinstance(warm, bool) and warm >= 0
                 else f"Discarded warm-up runs before the repeats: {NOT_RECORDED}")
    lines += [("Each model was loaded once on the chip and then generated the same clip several times with the same settings and seed. " if bench else
               ("There are at least 3 measured repeats, but the discarded warm-up runs are missing or fewer than the rule asks for, so the first repeat "
                "carries start-up cost. It is not a benchmark. " if enough else
                "This is a single sample: fewer than 3 measured repeats for at least one model, so there is no spread to quote. It is not a benchmark. "))
              + f"{warm_text[0].upper() + warm_text[1:]}. The CPU column is the reference's single `generate` call, made once in the reference "
              "environment (one sample).", "",
              "| Run | Part | Seconds per repeat | Min | Median | Max | CPU reference, one sample |", "|---|---|---|---|---|---|---|"]
    for role, ref_role, label in AUDIO_REPEAT_ROLES:
        lst = reps.get(role) if isinstance(reps, dict) else None
        if not isinstance(lst, list) or not lst:
            continue
        for key, what in AUDIO_REPEAT_PARTS:
            vals = [dig(e, "timings", key) if isinstance(e, dict) else None for e in lst[:20]]
            good = audio_good_values(lst, key)
            cells = ", ".join(num(v) if is_number(v) and abs(v) <= MAX_NUMBER else NOT_RECORDED for v in vals)
            # a part that any listed repeat lacks has no spread: it is shown as not recorded, as the stage 2 record marks it incomplete
            lo, med, hi = (num(min(good)), num(statistics.median(good)), num(max(good))) if good and len(good) == len(vals) else (NOT_RECORDED,) * 3
            cpu = dig(data, "reference_runs", ref_role, "timings", key) if key == "generate_and_decode_s" else None
            lines.append(f"| {label} | {what} (`{key}`) | {cells} | {lo} | {med} | {hi} | "
                         f"{num(cpu) if is_number(cpu) and abs(cpu) <= MAX_NUMBER else ('n/a' if key != 'generate_and_decode_s' else NOT_RECORDED)} |")
    lines += ["", "Load, once for the process (the first repeat reports it, because the repeats share the loaded model):", ""]
    for role, _ref, label in AUDIO_REPEAT_ROLES:
        lst = reps.get(role) if isinstance(reps, dict) else None
        if isinstance(lst, list) and lst:
            lines.append(f"- {label}: model load {num(dig(lst[0], 'timings', 'model_load_s'))} seconds, weights upload to the chip "
                         f"{num(dig(lst[0], 'timings', 'chip_weights_upload_s'))} seconds.")
    lines += ["", "Host load average (one minute) when each repeat ended, because the host is shared:", ""]
    for role, _ref, label in AUDIO_REPEAT_ROLES:
        lst = reps.get(role) if isinstance(reps, dict) else None
        if isinstance(lst, list) and lst:
            vals = [dig(e, "timings", "host_loadavg_1m") if isinstance(e, dict) else None for e in lst[:20]]
            lines.append(f"- {label}: " + ", ".join(num(v) if is_number(v) and abs(v) <= MAX_NUMBER else NOT_RECORDED for v in vals))
    return lines + [f"", f"Repeats per model: {', '.join(str(len(v)) for v in reps.values() if isinstance(v, list))}."]


def audio_section(run: Run, redact: Redactor) -> list[str]:
    """The "Audio clip on a chip" section of RESULTS.md. No lines for a run without audio_result.json."""
    data = run.audio
    if data is None:
        return []
    head = ["", f"## {AUDIO_HEADING}", ""]
    if "_unreadable" in data:
        return head + [f"`{AUDIO_RESULT}` is present but {data['_unreadable']}. No audio number or clip is in this bundle."]
    chip = dig(data, "runs", "chip")
    st = dig(data, "settings")
    cell = lambda v: txt(redact, v) if isinstance(v, str) else num(v)    # noqa: E731
    lines = head + [
        f"From `{AUDIO_RESULT}`. Nobody has listened to the clips. The numbers below compare the chip run's arrays with the "
        "CPU reference's arrays and say nothing about how the audio sounds.", "",
        "What ran where: the diffusion transformer ran on one chip. The T5Gemma text encoder, the sampler and the "
        "autoencoder decoder ran on the host CPU.", "",
        "| Fact | Value |", "|---|---|",
        f"| Model | {txt(redact, dig(chip, 'model'))} |",
        f"| Revision | {txt(redact, dig(chip, 'revision'))} |",
        f"| Runtime library commit | {txt(redact, dig(chip, 'library_commit'))} |",
        f"| Device | {txt(redact, dig(chip, 'device'))} |",
        f"| Prompt | {cell(dig(st, 'prompt'))} |",
        f"| Duration (seconds) | {cell(dig(st, 'duration'))} |",
        f"| Steps | {cell(dig(st, 'steps'))} |",
        f"| Guidance scale | {cell(dig(st, 'cfg_scale'))} |",
        f"| Sampler | {cell(dig(st, 'sampler'))} |",
        f"| Seed | {cell(dig(st, 'seed'))} |",
        f"| Sample size | {cell(dig(st, 'sample_size'))} |", "",
        "### Wall time of the chip run", "",
        "| Part | Value | What it measures |", "|---|---|---|"]
    timings = dig(chip, "timings")
    for key, unit, what in AUDIO_TIMINGS:
        value = num(dig(timings, key))
        lines.append(f"| `{key}` | {value if value == NOT_RECORDED else f'{value} {unit}'} | {what} |")
    total = num(dig(chip, "total_seconds"))
    lines.append(f"| `total_seconds` | {total if total == NOT_RECORDED else f'{total} seconds'} | the whole chip run, as the "
                 "chip test measured it |")
    known = {t[0] for t in AUDIO_TIMINGS}
    extra = sorted(k for k, v in (timings.items() if isinstance(timings, dict) else ())
                   if isinstance(k, str) and SAFE_KEY.fullmatch(k) and is_number(v) and k not in known)
    for k in extra[:20]:
        lines.append(f"| `{k}` | {num(timings[k])} | another number the chip test recorded |")
    ref = dig(data, "reference_runs", "reference")
    if isinstance(ref, dict):
        own = dig(ref, "timings", "generate_and_decode_s")
        lines += ["", f"The CPU reference's own `generate` call and decode took {num(own)} seconds when it was made"
                  + (" (this run reused a saved reference)." if ref.get("from_cache") is True else ".")]
    lines += audio_repeat_lines(data)
    lines += ["", "### Agreement with the CPU reference", "",
              "PCC is the Pearson correlation between two arrays (1 is identical). Relative L2 is the size of the difference "
              "divided by the size of the reference (0 is identical). The log-spectrum distance compares the two waveforms' "
              "spectra on a log scale (0 is identical). The worst layer is the diffusion transformer layer whose output "
              "differs most.", "", *AUDIO_TABLE_HEAD]
    summaries = [(label, dig(data, *path)) for label, path in AUDIO_AGREEMENTS]
    for label, s in summaries:
        lines.append(audio_agreement_row(redact, label, s))
    lines += ["", "The pass bars are provisional. Nobody has set them from a measured noise floor. The environment "
              "floor is the amount two CPU environments already differ: the same-environment CPU run uses the same code and "
              "weights as the reference and differs only in the Python environment. A chip number close to that floor "
              "means the chip run differs from the reference about as much as a second CPU environment does."]
    for label, s in summaries:
        probs = s.get("problems") if isinstance(s, dict) else None
        shown = [p for p in probs[:10] if isinstance(p, str)] if isinstance(probs, list) else []
        if shown:
            lines += ["", f"Problems recorded for \"{label}\":", ""] + [f"- {txt(redact, p)}" for p in shown]
    lines += ["", "### Control: the base model", ""]
    control = dig(data, "runs", "control_chip")
    cdiff = dig(data, "control_difference")
    if not isinstance(control, dict):
        lines.append("The control did not run, so the run has no check that the chip served the fine-tune's weights.")
    else:
        lines += ["The control runs the base model on the chip with its own settings (fifty guided steps). It catches a chip run that "
                  "served the wrong weights. Two checks decide: the base model's clip has to differ from the fine-tune's clip, and "
                  "the base model's chip latent has to be nearer to the base model's own CPU reference than to the fine-tune's. "
                  "The provisional bars are shown for information and decide nothing for the control, because fifty guided steps "
                  "compound the per-pass error that the fine-tune's eight steps show.", "", *AUDIO_TABLE_HEAD,
                  audio_agreement_row(redact, "Base model on the chip against the base model's CPU reference",
                                      dig(data, "agreement", "control_chip_vs_control_reference")), ""]
        disc = dig(data, "control_discrimination")
        if isinstance(disc, dict) and disc.get("ok") in (True, False):
            lines += [f"Final-latent relative L2 error of the base model's chip run: {num(disc.get('own_rel_l2'))} against the base "
                      f"model's own reference and {num(disc.get('other_rel_l2'))} against the fine-tune's reference. "
                      + ("It is nearer its own." if disc["ok"] else "It is not nearer its own, so the chip may have served the wrong weights."), ""]
        else:
            lines += [f"Whether the base model's chip run is nearer its own reference than the fine-tune's: {NOT_RECORDED}.", ""]
        if isinstance(cdiff, dict) and cdiff.get("ok") in (True, False):
            lines.append(f"The base model's clip {'differs from' if cdiff['ok'] else 'does not differ from'} the fine-tune's "
                         f"clip (relative L2 {num(cdiff.get('rel_l2'))}, PCC {num(cdiff.get('pcc'))}). "
                         + ("That is the expected result." if cdiff["ok"] else
                            "The chip may have served the same weights for both runs."))
        else:
            lines.append(f"Whether the base model's clip differs from the fine-tune's: {NOT_RECORDED}.")
    lines += ["", "### Clips", "", "The clips are in `audio/` in this bundle. Nobody has listened to them.", ""]
    lines += [audio_clip_line(rec) for rec in run.audio_clips] or [f"- No clip is recorded in `{AUDIO_RESULT}`."]
    limits = data.get("limits")
    items = [x for x in limits if isinstance(x, str) and x.strip()] if isinstance(limits, list) else []
    lines += ["", "### Limits recorded by stage 2", ""]
    lines += [f"- {redact(one_line(x))}" for x in items[:20]] or [f"- {NOT_RECORDED}"]
    lines += ["", f"Evidence: `{AUDIO_RESULT}`."]
    return lines


def audio_risks(run: Run, redact: Redactor) -> list[str]:
    """The "Audio run" section of RISKS.md: a failed chip agreement first, then the fixed limits. No lines without audio."""
    data = run.audio
    if data is None:
        return []
    lines = ["## Audio run", ""]
    if "_unreadable" in data:
        return lines + [f"- `{AUDIO_RESULT}` is present but {data['_unreadable']}, so this bundle shows no audio result. "
                        f"Evidence: `{AUDIO_RESULT}`.", ""]
    chip_agree = dig(data, "agreement", "chip_vs_reference")
    state = audio_summary_state(chip_agree)
    if state == "failed":
        failed = chip_agree.get("failed") if isinstance(chip_agree.get("failed"), list) else []
        names = ", ".join(txt(redact, f, 80) for f in failed[:10] if isinstance(f, str))
        lines.append("- The chip run's agreement with the CPU reference failed" + (f" (failed arrays: {names})" if names else "")
                     + f". Treat the clip as unverified. Evidence: `{AUDIO_RESULT}`.")
    elif state == "unknown":
        lines.append(f"- The chip run's agreement with the CPU reference is {NOT_RECORDED}. Evidence: `{AUDIO_RESULT}`.")
    lines += ["- Audio quality is not judged: nobody has listened to the clips.",
              "- Bars are provisional: nobody has set the agreement bars from a measured noise floor.",
              "- Only the diffusion transformer ran on the chip. The T5Gemma encoder, the sampler and the autoencoder decoder "
              "ran on the host.",
              "- One prompt, one duration, one seed: the agreement is measured on a single case."]
    if not isinstance(dig(data, "runs", "control_chip"), dict):
        lines.append("- The control did not run, so the run has no check that the chip served the fine-tune's weights.")
    elif dig(data, "control_difference", "ok") is not True or dig(data, "control_discrimination", "ok") is not True:
        lines.append("- The base model's clip does not differ from the fine-tune's, or its chip run is not nearer its own reference than "
                     "the fine-tune's, or the file does not say. The chip may have served the wrong weights.")
    for rec in run.audio_clips:
        if not rec["copied"]:
            lines.append(f"- {rec['label'][0].upper() + rec['label'][1:]} is not in this bundle, because {rec['reason']}.")
    return lines + [""]


def render_results(run: Run, redact: Redactor, n_todo: int, n_hazards: int, skipped_files: list) -> str:
    repo, rev = run.model()
    lic, nc = run.license()
    lines = [f"# Results: {repo}", "", "## Summary", "", summary(run, n_todo, n_hazards), "",
             "## Run facts", "",
             "Every path in this file is relative to the run directory (the directory that holds "
             "`ledger.jsonl`). A number is labelled `measured` or `TODO`; `TODO` means not measured.", "",
             "| Fact | Value |", "|---|---|",
             f"| Model | {repo} |", f"| Revision | {rev or 'not recorded'} |",
             f"| Nearest model | {run.delta.get('nearest_model')} |",
             f"| Path | {run.path or 'not recorded'} |",
             f"| Class | {run.cls or run.delta.get('class') or 'not recorded'} |",
             *run_fact_rows(run, redact),
             f"| License | {lic or 'not recorded'}{' (non-commercial)' if nc else ''} |", "",
             "## The run", "", "Counted from `ledger.jsonl`.", "",
             "| Count | Value | Label | Evidence |", "|---|---|---|---|"]
    for name, value in run_counts(run.entries):
        lines.append(f"| {name} | {value} | {'TODO' if value == 'TODO' else 'measured'} | `ledger.jsonl` |")
    lines += inventory_section(run, redact)
    lines += served_dir_section(run, redact)
    lines += graft_section(run, redact)
    lines += sampling_section(run, redact)
    lines += sampling_choice_section(run, redact)
    lines += reported_section(run, redact)
    lines += audio_section(run, redact)
    lines += text_section(run, redact)
    lines += stock_section(run, redact)
    body = stage_results(run, redact)
    for n in range(9):
        spec = run.specs[n]
        lines += ["", f"## Stage {n}: {spec.name}", ""]
        out = run.outcome.get(n) or {}
        result = out.get("result")
        if n == 8:
            lines += ["Result: this bundle. The stage 8 gate checks it after the agent finishes.", "",
                      "Files: `RESULTS.md`, `RISKS.md`, `card.md`, `PUBLISH_COMMANDS.txt`, `package/` "
                      "and `ledger.jsonl`. `package/` holds stage 7's package folders without venvs, "
                      "caches, weights or wheels the package does not ship, plus stage 7's record, "
                      "publish commands and cards.",
                      "",
                      "`ledger.jsonl` is the operator's copy of the run record. It holds absolute "
                      "paths from the machine and is not for publishing. The scrub skips it.",
                      "", "The build record is `stages/8/evidence/bundle-build.json`."]
            for s in skipped_files:
                lines.append(f"- Left out of `package/`: `{s['path']}` ({s['reason']}).")
            continue
        if result == "skipped":
            reason = out.get("reason")
            lines.append(f"Result: skipped. Reason (from the ledger): "
                         f"{redact(one_line(reason)) if reason else 'the ledger gives no reason'}")
            continue
        if result is None:
            lines.append("Result: the ledger records no end for this stage.")
        else:
            lines.append(f"Result: {result}. Attempts: {run.attempts.get(n, 0)}.")
            if result != "pass" and out.get("reasons"):
                lines.append("Reasons: " + redact(one_line("; ".join(map(str, out["reasons"])))))
        lines.append("")
        lines += body.get(n) or ["No result file."]
    return "\n".join(lines).rstrip() + "\n"


# ---- RISKS.md -----------------------------------------------------------------------------------------

def card_section(text: str, heading: str) -> list[str]:
    """The lines under a card's `## heading`."""
    if f"\n{heading}\n" not in text:
        return []
    after = text.split(f"\n{heading}\n", 1)[1]
    return after.split("\n## ", 1)[0].splitlines()


def todo_items(run: Run, redact: Redactor) -> list[str]:
    items = []
    for p in run.profiles():
        if p.get("verified") is not True:
            card = run.cards.get(p.get("name"), ("",))[0]
            items.append(f"{p.get('name')} ({chips(p.get('chips'))}): its boot check did not run "
                         "(package.json records verified false). Its top1 agreement and server "
                         f"ready time are TODO. Evidence: {ev(['stages/7/package.json', card])}.")
    not_measured: dict[str, list[str]] = {}
    for name, (rel, text) in sorted(run.cards.items()):
        for row in card_section(text, "## Expected performance"):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if row.startswith("| ") and len(cells) == 4 and cells[2] == "TODO":
                items.append(f"{name}: {redact(cells[0])} is TODO. Evidence: `{rel}`.")
        for line in card_section(text, "## Not measured"):
            if line.startswith("- "):
                not_measured.setdefault(line[2:].strip(), []).append(rel)
    for what, rels in not_measured.items():
        items.append(f"Not measured: {redact(what)}. Listed in: {ev(rels)}.")
    for s in (run.package or {}).get("skipped_profiles") or []:
        if isinstance(s, dict):
            items.append(f"No package for {chips(s.get('chips'))}: {end(redact(s.get('reason', '')))} "
                         "Evidence: `stages/7/package.json`.")
    for c in run.configs():
        if c.get("pass") is not True:
            items.append(f"the {c.get('chips')}-chip configuration did not pass in stage 4: "
                         f"{end(redact(c.get('reason') or 'no reason recorded'))} Its numbers are "
                         f"TODO. Evidence: {ev(['stages/4/result.json', *(c.get('evidence') or [])])}.")
    for n in (2, 3, 5, 6):
        res = run.results.get(n) or {}
        rel = f"stages/{n}/{run.specs[n].gate_file}"
        for key in sorted(res):
            if res[key] is None and key not in ("failure",):
                items.append(f"stage {n} {key}: TODO. Evidence: `{rel}`.")
        for x in res.get("numbers") or []:
            if isinstance(x, dict) and x.get("label") == "TODO":
                items.append(f"stage {n} {redact(x.get('name'))}: TODO. Evidence: `{rel}`.")
    if not run.package:
        reason = (run.outcome.get(7) or {}).get("reason")
        items.append("No package was built, so no package was boot-checked"
                     + (f". The ledger's reason: {redact(one_line(reason))}" if reason else "") + ".")
    return items


def mtp_drafter_dealt_with(run: Run) -> tuple[bool, list[str]]:
    """The model has no mtp.* tensors, so a drafter that needs the MTP head cannot serve it. Dealt with when
    every swap check recorded that it served with the drafter off."""
    states = {k: v.get("drafter") for k, v in run.serving.items()}
    if not states:
        return False, ["No swap check recorded how it served, so the bundle cannot say whether the drafter was off."]
    label = lambda k: "stage 2" if k == 2 else f"stage 4, {chips(k[1])}"
    if run.graft_decision is not None and run.graft_decision["choice"] == "graft" and run.graft_decision.get("trial") is not None:
        # A graft was chosen: the drafter may be on where the head was grafted. Stage 2's own check is the safe configuration unless a person
        # grafted there too.
        ungrafted = sorted((k for k, v in states.items() if v != "off"
                            and not (run.stage_2_grafted if k == 2 else run.graft_built.get(k))), key=str)
        if not ungrafted:
            return True, ["Stage 2's graft trial chose the graft. Every swap check with a drafter on was built with the nearest model's mtp.* head "
                          f"grafted ({', '.join(label(k) for k in sorted(states, key=str) if states[k] != 'off') or 'none'}); stage 2's own check "
                          "served with the drafter off. The drafter's acceptance rate was not measured. Evidence: each check's `swap-check.json` "
                          "and `model-dir-sources.json`."]
        return False, ["A drafter was on in a swap check built without the grafted head: " + ", ".join(label(k) for k in ungrafted) + "."]
    if all(v == "off" for v in states.values()):
        return True, [f"Every swap check served with the drafter off ({', '.join(label(k) for k in sorted(states, key=str))}). "
                      "No figure with a drafter was measured, so speed numbers from the nearest model's packages "
                      "do not carry over. Evidence: each check's `swap-check.json`."]
    return False, ["Drafter state per check: " + "; ".join(f"{label(k)}: {v}" for k, v in sorted(states.items(), key=str)) + "."]


def generation_config_dealt_with(run: Run) -> tuple[bool, list[str]]:
    """The generation config hazard is dealt with when stage 2 served the nearest model's file (so the engine found
    stop tokens) and every package card says the model ships none. The settings are still the nearest model's."""
    rec = read_json(run.dir / "stages" / "2" / "model-dir-sources.json", "stages/2/model-dir-sources.json",
                    required=False)
    if rec is None:
        return False, ["Stage 2's record of the served directory, stages/2/model-dir-sources.json is not in this "
                       "run, so the bundle cannot say whose generation config was served."]
    origin = rec.get("sampling_defaults_from")
    if origin == "model":
        return False, ["Stage 2 served the new model's own generation_config.json (`stages/2/model-dir-sources.json`). "
                       "Read the finding: that file may lack the stop tokens."]
    if origin == "none":
        return False, ["Stage 2 served a directory with no generation_config.json "
                       "(`stages/2/model-dir-sources.json`)."]
    if origin != "base":
        return False, ["`stages/2/model-dir-sources.json` does not say whose generation config was served."]
    saying = [rel for rel, text in run.cards.values() if "ships no generation_config.json" in text]
    why = ["Stage 2 served the nearest model's generation_config.json (`stages/2/model-dir-sources.json`), so the "
           "stop tokens and the sampling defaults are the nearest model's.",
           f"Package cards that say so: {len(saying)} of {len(run.cards)}."]
    return bool(run.cards) and len(saying) == len(run.cards), why


def stop_tokens_dealt_with(run: Run) -> tuple[bool, list[str]]:
    """Never yes: the swap checks compare the first 32 generated tokens and stop there, so no stage showed which token
    ends a reply. The bundle names the stop tokens stage 2's directory holds when its record lists them."""
    why = ["Stage 2's and stage 4's swap checks compare the first 32 generated tokens with the CPU reference. They "
           "never run a request until it stops, so no stage showed which token ends a reply."]
    rec = read_json(run.dir / "stages" / "2" / "model-dir-sources.json", "stages/2/model-dir-sources.json", required=False)
    ids = rec.get("eos_token_ids") if isinstance(rec, dict) else None
    if isinstance(ids, list) and ids and all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        why.append(f"Stage 2's model directory stops on {fmt(ids)} (`stages/2/model-dir-sources.json`).")
    return False, why


SAMPLING_ORIGINS = {"base": "the nearest model's generation_config.json",
                    "model": "the new model's own generation_config.json",
                    "none": "no generation_config.json"}


def sampling_defaults_dealt_with(run: Run) -> tuple[bool, list[str]]:
    """Never yes: no stage compares the card's sampling settings with the defaults the server applies."""
    why = ["No stage compared the card's sampling settings with the defaults the server applies to a request that "
           "sets none."]
    rec = read_json(run.dir / "stages" / "2" / "model-dir-sources.json", "stages/2/model-dir-sources.json", required=False)
    origin = SAMPLING_ORIGINS.get(rec.get("sampling_defaults_from")) if isinstance(rec, dict) \
        and isinstance(rec.get("sampling_defaults_from"), str) else None
    if origin:
        why.append(f"Stage 2 served {origin} (`stages/2/model-dir-sources.json`).")
    return False, why


def chat_template_dealt_with(run: Run, redact: Redactor) -> tuple[bool, list[str]]:
    """Never yes: no stage sends a chat request (the swap checks use /v1/completions, which applies no template)."""
    why = ["No stage sent a chat request. The swap checks in stages 2 and 4 use /v1/completions, which applies no chat "
           "template."]
    rec = read_json(run.dir / "stages" / "2" / "model-dir-sources.json", "stages/2/model-dir-sources.json", required=False)
    text = lambda v: redact(one_line(v)) if isinstance(v, str) and v else None      # noqa: E731
    sentence = chat_template_sentence(rec, text) if isinstance(rec, dict) else None
    if sentence:
        held = "has no chat template" if rec["chat_template"].get("file") is None else "holds " + sentence.rstrip(".")
        why.append(f"Stage 2's model directory {held} (`stages/2/model-dir-sources.json`).")
    return False, why


def hazard_entries(run: Run, redact: Redactor) -> list[tuple[str, str, list[str], bool, list[str]]]:
    """(heading, finding, evidence, dealt with, why) for each stage 0 hazard."""
    out, seen = [], {}
    nc_cards = [rel for rel, text in run.cards.values() if NC_LINE in text]
    s6 = run.results.get(6) or {}
    for h in run.delta.get("hazards") or []:
        if not isinstance(h, dict):
            continue
        area = str(h.get("area"))
        seen[area] = seen.get(area, 0) + 1
        heading = area if seen[area] == 1 else f"{area} ({seen[area]})"
        if area == "tensor_cache":
            ok, why = tensor_cache_dealt_with(run, redact)
        elif area == "drafter" and "no mtp" in str(h.get("finding", "")).lower():
            ok, why = mtp_drafter_dealt_with(run)
        elif area == "other" and "ships code" in str(h.get("finding", "")).lower():
            sp = (run.results.get(2) or {}).get("sidecar_parity") if isinstance(run.results.get(2), dict) else None
            ok = False
            names = ", ".join(str(c.get("file")) for c in run.delta.get("code_files") or [] if isinstance(c, dict))
            why = ([f"Stage 2's parity run imported {names} only after its sha256 ({str(sp.get('code_sha256'))[:12]}) "
                    "matched the one stage 0 recorded. Nobody in the run read the code. Read it before running it "
                    "yourself."] if isinstance(sp, dict) and sp.get("measured") is True else
                   ["No stage used the file. Nobody in the run read it."])
        elif area == "other" and "sidecar" in str(h.get("finding", "")).lower():
            res2 = run.results.get(2) if isinstance(run.results.get(2), dict) else {}
            sp = res2.get("sidecar_parity")
            ok = isinstance(sp, dict) and sp.get("measured") is True
            why = ([f"Stage 2 measured the sidecar parity on the device and the host: hidden_pcc_min "
                    f"{fmt(sp.get('hidden_pcc_min'))}, top1_agree_fraction {fmt(sp.get('top1_agree_fraction'))}, "
                    f"prob_max_abs_diff {fmt(sp.get('prob_max_abs_diff'))}, over {fmt(sp.get('n_questions'))} "
                    "questions. Evidence: `stages/2/evidence/sidecar-parity.json`."] if ok else
                   ["No stage measured the sidecar's output. Read the finding and its evidence."])
        elif area == "drafter":
            rate = [x for x in s6.get("numbers") or [] if isinstance(x, dict)
                    and x.get("label") == "measured" and "accept" in str(x.get("name", "")).lower()]
            ok = bool(rate)
            why = ([f"Stage 6 measured {redact(rate[0].get('name'))}: {fmt(rate[0].get('value'))}. "
                    "Evidence: `stages/6/result.json`."] if ok else
                   ["No stage measured the drafter's acceptance rate on this model."])
        elif area == "disk":
            disk_pauses = sum(1 for e in run.entries if e.get("event") == "decision"
                              and e["data"].get("decision") == "pause"
                              and "disk" in str(e["data"].get("reason", "")).lower())
            ok, why = False, [f"The run's files do not record where each stage wrote its large files. "
                              f"Pauses whose reason names disk space: {disk_pauses} (ledger.jsonl)."]
        elif area == "drafter_graft":
            ok, why = graft_hazard_dealt_with(run)
        elif area == "generation_config":
            ok, why = generation_config_dealt_with(run)
        elif area == "stop_tokens":
            ok, why = stop_tokens_dealt_with(run)
        elif area == "sampling_defaults":
            ok, why = sampling_defaults_dealt_with(run)
        elif area == "chat_template":
            ok, why = chat_template_dealt_with(run, redact)
        elif area == "license":
            ok = False
            why = ["A license is the operator's decision. The run cannot settle it."]
            if run.cards:
                why.append(f"Package cards that say \"{NC_LINE}\": {len(nc_cards)} of {len(run.cards)}.")
        else:
            ok, why = False, ["The bundle has no check for this hazard. Read the finding and its evidence."]
        out.append((heading, redact(one_line(h.get("finding", ""))),
                    [e for e in h.get("evidence") or [] if isinstance(e, str)], ok, why))
    return out


def thin_evidence(run: Run, redact: Redactor) -> list[str]:
    items = []
    r2 = run.results.get(2) or {}
    if is_number(r2.get("top1_agreement")):
        items.append(f"Stage 2's top1 agreement ({fmt(r2['top1_agreement'])}) comes from one prompt of "
                     f"{fmt(r2.get('n_tokens'))} tokens. Evidence: `stages/2/result.json`.")
    for c in run.configs():
        if c.get("pass") is True and is_number(c.get("top1_agreement")):
            items.append(f"Stage 4's {c.get('chips')}-chip top1 agreement ({fmt(c['top1_agreement'])}) "
                         f"comes from one prompt of {fmt(c.get('n_tokens'))} tokens. Evidence: "
                         f"{ev(c.get('evidence'))}.")
    for c in run.reference.get("checks") or []:
        if isinstance(c, dict) and "form check" in str(c.get("note", "")).lower():
            items.append(f"Stage 1's check \"{c.get('name')}\" is a form check only. Evidence: "
                         f"{ev(c.get('evidence'))}.")
    items.append("Each gate checks the shape of a result file and that its evidence files exist. "
                 "A gate cannot tell whether a claim is true.")
    return items


def render_risks(run: Run, redact: Redactor, todo: list[str], hazards) -> str:
    repo, _ = run.model()
    lines = [f"# Risks: {repo}", "",
             "Every item names its evidence, relative to the run directory.", ""]
    lines += audio_risks(run, redact)
    lines += text_risks(run, redact)
    lines += stock_risks(run, redact)
    lines += ["## TODO numbers", ""]
    lines += [f"- {t}" for t in todo] or ["- None found in the run's files."]
    lines += ["", "## Stage 0 hazards", "",
              "Each hazard stage 0 found, and whether the run's files show that a later stage "
              "dealt with it. `Not shown` means the evidence cannot show it.", ""]
    for heading, finding, evidence, ok, why in hazards:
        lines += [f"### {heading}", "", finding, "", f"- Evidence: {ev(evidence)}.",
                  f"- Dealt with: {'yes' if ok else 'Not shown'}. {' '.join(why)}", ""]
    if not hazards:
        lines += ["Stage 0 listed no hazards.", ""]
    d = run.sampling_decision
    if d is not None and d.get("flag"):
        lines += ["## Sampling choice", "",
                  SAMPLING_FLAG_RISKS.get(d["flag"], "The sampling decision is flagged.") + " Evidence: `stages/2/sampling-decision.json`.", ""]
    lines += ["## Thin evidence", ""] + [f"- {t}" for t in thin_evidence(run, redact)]
    lic, nc = run.license()
    lines += ["", "## License", ""]
    if lic:
        source = "`stages/7/package.json`" if run.package else "`stages/0/evidence/genconfig-license.json`"
        lines.append(f"- License: {lic}. Evidence: {source}.")
        if nc:
            lines.append(f"- package.json records non_commercial true. The model is for non-commercial "
                         f"use only, and each package card must say \"{NC_LINE}\".")
            missing = [rel for rel, text in run.cards.values() if NC_LINE not in text]
            if missing:
                lines.append(f"- These cards do not say it: {ev(missing)}.")
        elif nc is False:
            lines.append("- package.json records non_commercial false.")
    else:
        lines.append("- The run's files do not record the license. Not shown.")
    return "\n".join(lines).rstrip() + "\n"


# ---- card.md ------------------------------------------------------------------------------------------

def strip_front_matter(text: str) -> str:
    if text.startswith("---\n") and "\n---\n" in text[4:]:
        return text[4:].split("\n---\n", 1)[1]
    return text


def render_card(run: Run, redact: Redactor) -> str:
    repo, rev = run.model()
    lic, nc = run.license()
    lines = [f"# {repo}: model card (draft)", "",
             f"- Model: {repo} at revision `{rev or 'not recorded'}`.",
             f"- Nearest supported model: {split_model_id(run.delta.get('nearest_model'))[0]}.",
             f"- License: {lic or 'not recorded'}."]
    if nc:
        lines.append(f"- {NC_LINE} Commercial use is not permitted by that license.")
    lines += ["", "## Profiles", ""]
    if run.package:
        lines += ["| Package | Chips | Mesh | Boot check |", "|---|---|---|---|"]
        for p in sorted(run.profiles(), key=lambda x: (x.get("chips") or 0)):
            status = "passed in stage 7" if p.get("verified") is True else "not boot-checked"
            lines.append(f"| {p.get('name')} | {p.get('chips')} | {p.get('mesh')} | {status} |")
        for s in run.package.get("skipped_profiles") or []:
            if isinstance(s, dict):
                lines.append(f"\nNo package for {chips(s.get('chips'))}: {end(redact(s.get('reason', '')))}")
        lines += ["", "## Package cards", "",
                  "Each card below is stage 7's card, copied as it is. Its front matter is left out "
                  "and its headings are moved down three levels.", ""]
        for name, (rel, text) in sorted(run.cards.items()):
            lines += [f"### Card: {name}", "", f"From `{rel}`.", ""]
            for line in strip_front_matter(text).splitlines():
                lines.append("###" + line if line.startswith("#") else line)
            lines.append("")
    else:
        lines.append("No package was staged, so no profile exists.")
    return "\n".join(lines).rstrip() + "\n"


# ---- the package copy ---------------------------------------------------------------------------------

def copy_package_tree(src: Path, dst: Path, *, shipped_wheels: set[str], skipped: list,
                      max_bytes: int = MAX_COPY_BYTES) -> list[Path]:
    """Copy one staged package folder, leaving out venvs, caches, weights, links, wheels the
    manifest does not list and files larger than `max_bytes`. Returns the files written; each
    file left out is appended to `skipped` as {"path", "reason"}."""
    src, dst = Path(src), Path(dst)
    written = []
    for dirpath, dirnames, filenames in os.walk(src):
        here = Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            if (here / d).is_symlink():
                skipped.append({"path": str(here / d), "reason": "a symbolic link"})
            elif d in SKIP_DIRS:
                skipped.append({"path": str(here / d), "reason": "a venv, cache or built model directory"})
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            f = here / name
            rel = f.relative_to(src).as_posix()
            if f.is_symlink():
                skipped.append({"path": str(f), "reason": "a symbolic link"})
                continue
            if f.suffix.lower() in SKIP_SUFFIXES:
                skipped.append({"path": str(f), "reason": "a weights, cache or bytecode file"})
                continue
            if f.suffix.lower() == ".whl" and rel not in shipped_wheels:
                skipped.append({"path": str(f), "reason": "a wheel the package manifest does not list"})
                continue
            if f.stat().st_size > max_bytes:
                skipped.append({"path": str(f), "reason": f"larger than {max_bytes} bytes (1 GB)"
                                if max_bytes == MAX_COPY_BYTES else f"larger than {max_bytes} bytes"})
                continue
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out)
            written.append(out)
    return written


def shipped_wheels(pkg: Path) -> set[str]:
    try:
        deps = json.loads((pkg / "tt_kernel_manifest.json").read_text(encoding="utf-8")).get("deps") or {}
    except (OSError, ValueError, AttributeError):
        return set()
    return {w for w in list(deps.get("wheels") or []) + list(deps.get("models_wheels") or [])
            if isinstance(w, str)}


def copy_packages(run: Run, bundle: Path) -> list[dict]:
    """bundle/package/: each profile's folder, then the files supervisor.py's _copy_package also
    copies (package.json, PUBLISH_COMMANDS.txt, <name>-README.md), so both copies agree."""
    skipped: list[dict] = []
    if not run.package:
        return skipped
    dest = bundle / "package"
    dest.mkdir(parents=True, exist_ok=True)
    root = os.path.realpath(run.dir)
    for p in run.profiles():
        rel = p.get("dir")
        src = Path(os.path.realpath(run.dir / str(rel)))
        if not isinstance(rel, str) or not under(str(src), root) or not src.is_dir():
            skipped.append({"path": str(rel), "reason": "not a directory inside the run directory"})
            continue
        copy_package_tree(src, dest / src.name, shipped_wheels=shipped_wheels(src), skipped=skipped)
    s7 = run.dir / "stages" / "7"
    shutil.copyfile(s7 / "package.json", dest / "package.json")
    shutil.copyfile(s7 / "PUBLISH_COMMANDS.txt", dest / "PUBLISH_COMMANDS.txt")
    for card in sorted((s7 / "package").glob("*/README.md")):
        shutil.copyfile(card, dest / f"{card.parent.name}-README.md")
    for s in skipped:
        s["path"] = os.path.relpath(s["path"], run.dir) if os.path.isabs(s["path"]) else s["path"]
    return skipped


# ---- main ---------------------------------------------------------------------------------------------

def find_orchard(cfg: dict, entries: list[dict]) -> Path:
    for candidate in (cfg.get("orchard_dir"), recorded_paths(entries).get("orchard_dir")):
        if isinstance(candidate, str) and (Path(candidate) / "orchard" / "scrub.py").is_file():
            return Path(candidate)
    raise Missing("orchard/scrub.py cannot be found: the ledger's run_start paths.orchard_dir does "
                  "not name a tt-orchard checkout. Add \"orchard_dir\" to bundle_config.json")


def load_config(here: Path) -> dict:
    cfg = read_json(here / CONFIG_NAME, f"stages/8/{CONFIG_NAME}", required=True)
    for key in CONFIG_KEYS:
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise Missing(f"stages/8/{CONFIG_NAME} needs {key!r}")
    if not Path(cfg["run_dir"]).is_dir():
        raise Missing(f"stages/8/{CONFIG_NAME}: run_dir {cfg['run_dir']!r} is not a directory")
    return cfg


def build(cfg: dict) -> int:
    run_dir = Path(os.path.abspath(cfg["run_dir"]))
    entries = read_ledger(run_dir / "ledger.jsonl")
    orchard_dir = find_orchard(cfg, entries)
    sys.path.insert(0, str(orchard_dir))
    from orchard import scrub, stages               # noqa: E402  (the checkout is found at run time)
    run = Run(run_dir, entries, stages)
    redact = Redactor(run.paths, run_dir, scrub)

    stage8 = run_dir / "stages" / "8"
    bundle = stage8 / "bundle"
    if bundle.is_symlink():
        raise Missing("stages/8/bundle is a symbolic link; remove it")
    if bundle.exists():
        shutil.rmtree(bundle)                       # the bundle is built from scratch every time
    bundle.mkdir(parents=True)

    todo = todo_items(run, redact)
    hazards = hazard_entries(run, redact)
    skipped = copy_packages(run, bundle)
    run.audio_clips = copy_audio(run, bundle)
    (bundle / "RESULTS.md").write_text(render_results(run, redact, len(todo), len(hazards), skipped),
                                       encoding="utf-8")
    (bundle / "RISKS.md").write_text(render_risks(run, redact, todo, hazards), encoding="utf-8")
    (bundle / "card.md").write_text(render_card(run, redact), encoding="utf-8")
    if run.publish is not None:
        (bundle / "PUBLISH_COMMANDS.txt").write_bytes(run.publish)
    else:
        reason = one_line(redact((run.outcome.get(7) or {}).get("reason") or "none recorded"))
        (bundle / "PUBLISH_COMMANDS.txt").write_text(
            "# No package was staged in this run, so there is nothing to publish.\n"
            f"# Stage 7 was skipped. The ledger's reason: {reason}\n", encoding="utf-8")
    shutil.copyfile(run_dir / "ledger.jsonl", bundle / "ledger.jsonl")

    files = [{"path": f"stages/8/bundle/{p.relative_to(bundle).as_posix()}", "sha256": sha256(p),
              "bytes": p.stat().st_size}
             for p in sorted(bundle.rglob("*")) if p.is_file()]
    for f in files:
        print(f"wrote {f['path']}")
    for s in skipped:
        print(f"left out {s['path']}: {s['reason']}")
    for rec in run.audio_clips:
        if not rec["copied"]:
            print(f"left out audio/{rec['name']}: {rec['reason']}")

    home = run.paths.get("operator_home") if isinstance(run.paths.get("operator_home"), str) else None
    hits = scrub.scrub_bundle(bundle, home=home)
    record = {"files": files, "skipped": skipped, "scrub": hits, "redacted_texts": redact.count,
              "path": run.path, "package_format": run.package_format}
    if run.audio is not None:                       # only an audio run adds a key, so other records stay as they were
        record["audio_clips"] = [{k: rec[k] for k in ("name", "copied", "reason")} for rec in run.audio_clips]
    (stage8 / "evidence").mkdir(parents=True, exist_ok=True)
    (stage8 / "evidence" / "bundle-build.json").write_text(json.dumps(record, indent=2) + "\n",
                                                           encoding="utf-8")
    print("wrote stages/8/evidence/bundle-build.json")
    print(f"texts with machine paths, hostnames or tokens removed: {redact.count}")
    if hits:
        print(f"scrub: {len(hits)} finding(s). The stage 8 gate fails the bundle until each is fixed "
              "at its source:")
        for hit in hits:
            print(f"scrub: {hit}")
    else:
        print("scrub: no findings")
    return 0


def main() -> int:
    here = Path(__file__).resolve().parent
    try:
        return build(load_config(here))
    except Missing as exc:
        print(f"build_bundle: required input missing or unreadable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
