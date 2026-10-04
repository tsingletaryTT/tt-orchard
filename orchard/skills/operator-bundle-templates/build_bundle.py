#!/usr/bin/env python3
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
        self.package_format = stages_mod.package_format(entries)
        self.specs = {n: stages_mod.spec_for(n, self.path, self.package_format) for n in range(9)}
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
        self.reference = read_json(s / "1" / "reference.json", "stages/1/reference.json", required=True)
        self.results = {}
        for n in range(2, 7):
            spec = self.specs[n]
            if spec.skip or not spec.gate_file:
                continue
            rel = f"stages/{n}/{spec.gate_file}"
            self.results[n] = read_json(run_dir / rel, rel, required=self.result(n) == "pass")
        self.swap = {2: read_json(s / "2" / "swap_config.json", "", required=False) or {}}
        for d in sorted((s / "4" / "configs").glob("*/swap_config.json")):
            if d.parent.name.isdigit():
                self.swap[(4, int(d.parent.name))] = read_json(d, "", required=False) or {}
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

    for n in (2, 3, 5, 6):
        res = run.results.get(n)
        if res is not None:
            body[n] = field_table(res, f"stages/{n}/{run.specs[n].gate_file}", redact)

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
    text = (f"This run brought up {repo} on the {run.path or 'unknown'} path. "
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
             f"| License | {lic or 'not recorded'}{' (non-commercial)' if nc else ''} |", "",
             "## The run", "", "Counted from `ledger.jsonl`.", "",
             "| Count | Value | Label | Evidence |", "|---|---|---|---|"]
    for name, value in run_counts(run.entries):
        lines.append(f"| {name} | {value} | {'TODO' if value == 'TODO' else 'measured'} | `ledger.jsonl` |")
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
        for row in card_section(text, "## Numbers"):
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
             "Every item names its evidence, relative to the run directory.", "",
             "## TODO numbers", ""]
    lines += [f"- {t}" for t in todo] or ["- None found in the run's files."]
    lines += ["", "## Stage 0 hazards", "",
              "Each hazard stage 0 found, and whether the run's files show that a later stage "
              "dealt with it. `Not shown` means the evidence cannot show it.", ""]
    for heading, finding, evidence, ok, why in hazards:
        lines += [f"### {heading}", "", finding, "", f"- Evidence: {ev(evidence)}.",
                  f"- Dealt with: {'yes' if ok else 'Not shown'}. {' '.join(why)}", ""]
    if not hazards:
        lines += ["Stage 0 listed no hazards.", ""]
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

    home = run.paths.get("operator_home") if isinstance(run.paths.get("operator_home"), str) else None
    hits = scrub.scrub_bundle(bundle, home=home)
    record = {"files": files, "skipped": skipped, "scrub": hits, "redacted_texts": redact.count,
              "path": run.path, "package_format": run.package_format}
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
