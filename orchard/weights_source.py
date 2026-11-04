# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which installed packages can serve swapped weights (issue 73).

A weights-only bring-up puts the new model's weights in a model directory and serves an installed package against it. That only
measures the new model when the package reads the directory. A catalog container package whose generator calls snapshot_download on a
constant repository ignores the directory and serves the base model: the first such run reported 32 of 32 tokens matching and had
measured the base. PR 72 made a test that fetches from the Hub fail. This module gives the `base` preflight row the same answer
before a coder boot.

Two kinds of installed package are judged:

- A container package (schema 5.1) is judged by a record, `orchard/data/weights_source.json`. Nothing in its manifest says what its
  generator does, so each row holds a status and the evidence for it: `hard-coded` (seen to fetch its own repository), `honours` (seen
  to read the directory) or `unprobed` (no probe recorded). Only `hard-coded` refuses the package.
- A v6 bundle is judged by a dry run of the edit that stage 2's prepare_swap.py makes to its run.sh. A run.sh that edit cannot rewrite
  (no single --model argument, a second --model option, a current-form script without its prepare line) cannot take swapped weights.
  The dry run calls the template's own functions, so the two cannot disagree, and it writes nothing.

A package that cannot be read (no directory, no run.sh, a file over the size bound) is not refused for that: the runtime check of PR 72
remains the net for what this module cannot see.
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from orchard import safeio

DATA_FILE = Path(__file__).resolve().parent / "data" / "weights_source.json"
TEMPLATE = Path(__file__).resolve().parent / "skills" / "weights-swap-templates" / "prepare_swap.py"

HONOURS, HARD_CODED, UNPROBED = "honours", "hard-coded", "unprobed"
STATUSES = (HONOURS, HARD_CODED, UNPROBED)

MAX_RECORD_BYTES = 1 * 1024 * 1024       # choice: the shipped file is a few kilobytes
MAX_RUN_SH_BYTES = 1 * 1024 * 1024       # choice: a bundle's run.sh is about ten kilobytes
MAX_EVIDENCE_CHARS = 2000                # choice: a row's evidence is a few sentences
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
ROW_KEYS = {"status", "evidence", "issue"}

PATCHED_GENERATOR = ("The only way to a swapped-weights bring-up through a package like that is a patched generator: a new package, "
                     "or a bundle, whose generator reads MODEL_WEIGHTS_DIR")


@dataclass(frozen=True)
class Record:
    name: str
    status: str
    evidence: str
    issue: int | None = None


def _bad(why: str) -> ValueError:
    return ValueError(f"weights_source record: {why}")


def load_records(path=DATA_FILE) -> dict[str, Record]:
    """The rows of the record file, or a ValueError that says what is wrong. The file is read once through safeio (no link, a size bound),
    and every row is checked: a known status, evidence text for every status, only known keys, an integer issue number when there is one."""
    try:
        data = safeio.read_json(path, max_bytes=MAX_RECORD_BYTES)
    except safeio.SafeReadError as exc:
        raise _bad(f"{Path(path).name} cannot be read as JSON: {exc.reason}") from None
    if not isinstance(data, dict):
        raise _bad("the file is not an object")
    if data.get("schema") != 1:
        raise _bad(f"schema must be 1, got {data.get('schema')!r}")
    rows = data.get("packages")
    if not isinstance(rows, dict):
        raise _bad("packages must be an object")
    out = {}
    for name, row in rows.items():
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise _bad(f"the package name {name!r} is not org/name")
        if not isinstance(row, dict):
            raise _bad(f"{name}: the row is not an object")
        if extra := sorted(set(row) - ROW_KEYS):
            raise _bad(f"{name}: unknown key {extra[0]!r}")
        status = row.get("status")
        if status not in STATUSES:
            raise _bad(f"{name}: status {status!r} is not one of {', '.join(STATUSES)}")
        evidence = row.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > MAX_EVIDENCE_CHARS:
            raise _bad(f"{name}: evidence must be text of 1 to {MAX_EVIDENCE_CHARS} characters")
        issue = row.get("issue")
        if issue is not None and (isinstance(issue, bool) or not isinstance(issue, int) or issue < 1):
            raise _bad(f"{name}: issue must be a whole number or absent")
        out[name] = Record(name, status, evidence.strip(), issue)
    return out


@lru_cache(maxsize=1)
def _shipped() -> dict[str, Record]:
    return load_records()


def records() -> dict[str, Record]:
    """The shipped record. A broken file raises, and the base row turns that into a result."""
    return dict(_shipped())


def record_for(name: str, recs: dict[str, Record]) -> Record:
    """The row for `name`, or an unprobed row when there is none."""
    return recs.get(name) or Record(name, UNPROBED, "No probe of this package is recorded.")


@lru_cache(maxsize=1)
def _template():
    spec = importlib.util.spec_from_file_location("prepare_swap_for_weights_source", str(TEMPLATE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_sh_problem(text: str, nearest: str) -> str | None:
    """Why stage 2's prepare_swap.py could not rewrite this bundle run.sh for a model directory, or None when it can. It runs the
    template's own edit on the text in memory: no file is read or written, and the bundle and model paths are placeholders. The
    template is asked to fail quietly for this caller (a context variable, so a preflight on another thread is not affected) and
    the failure comes back as an exception that carries its message. The process's stderr is not touched: the web UI runs
    preflights on request threads, and a redirect of the shared stderr lets overlapping dry runs swallow each other's messages."""
    tmpl = _template()
    token = tmpl.QUIET.set(True)
    try:
        tmpl.edit_run_script(text, Path("/nonexistent-bundle"), nearest, Path("/nonexistent-model-dir"))
    except tmpl.SwapFail as exc:
        return exc.message
    finally:
        tmpl.QUIET.reset(token)
    return None


def _read_run_sh(directory: str) -> str | None:
    """The bundle's run.sh text, or None when there is no directory, no file or a refused read."""
    if not directory:
        return None
    try:
        return safeio.read_bytes(Path(directory) / "run.sh", max_bytes=MAX_RUN_SH_BYTES).decode("utf-8", "replace")
    except (safeio.SafeReadError, OSError):
        return None


def bundle_problem(directory: str, base: str) -> str | None:
    """Why a v6 bundle's run.sh cannot take swapped weights, or None when it can or cannot be read (no directory, no file, a refused
    read: the runtime check of PR 72 is the net for those)."""
    text = _read_run_sh(directory)
    return run_sh_problem(text, base) if text is not None else None


def hard_coded_names(recs: dict[str, Record]) -> set[str]:
    """The names of the container packages the record says hard-code their repository."""
    return {name for name, rec in recs.items() if rec.status == HARD_CODED}


def unoffered_names(installed, recs: dict[str, Record], refused_names=()) -> set[str]:
    """The package names no picker, search result or ranked choice should offer for a bring-up: the packages judged and refused now, and
    the recorded hard-coded containers that are not installed (a download of one would be refused by the next preflight). A name that an
    installed v6 bundle carries is left out of the recorded part: the record is about container packages, and that bundle is judged by
    its own run.sh when it serves the base.
    Every installed v6 bundle is also judged against its own `weights_repo`, so a bundle for a different base (an alternative the picker
    could list) that cannot take swapped weights is left out too. `refused_names` only covers the packages that serve the base asked
    for (review of PR 442, round 3). A hard-coded container of another base is already in the recorded part."""
    bundles = {pkg.name for pkg in installed if pkg.schema == "6"}
    own_refused = {pkg.name for pkg in installed if pkg.schema == "6" and judge([pkg], getattr(pkg, "weights_repo", ""), recs)[1]}
    return set(refused_names) | own_refused | (hard_coded_names(recs) - bundles)


def judge(installed, base: str, recs: dict[str, Record]):
    """Split the installed packages that serve `base` into (usable, refused). `refused` holds (package, reason) pairs. The order of the
    usable ones is the order given. The record judges container packages only; a v6 bundle, even one with a recorded name (a patched
    replacement that took the container's name), is judged by its own run.sh."""
    usable, refused = [], []
    for pkg in installed:
        if pkg.schema == "6":
            problem = bundle_problem(getattr(pkg, "directory", ""), base)
            if problem:
                refused.append((pkg, f"{pkg.name}'s run.sh cannot take swapped weights: {problem}"))
                continue
        else:
            rec = record_for(pkg.name, recs)
            if rec.status == HARD_CODED:
                refused.append((pkg, f"{pkg.name} hard-codes its model repository, so it serves the base model whatever weights are "
                                     f"swapped in ({rec.evidence})"))
                continue
        usable.append(pkg)
    return usable, refused


def refusal_text(base: str, refused) -> str:
    """The base row's block text when every package that serves `base` was refused: one sentence per package, then the way out."""
    why = "; ".join(reason for _, reason in refused)
    return (f"no installed package can serve swapped weights for {base}, the model this one is based on, so stage 2 would measure the "
            f"wrong model. {why}. {PATCHED_GENERATOR}.")


def status_note(pkg, recs: dict[str, Record]) -> str:
    """One sentence for the base row's warning about a container package: what is known of its use of MODEL_WEIGHTS_DIR."""
    rec = record_for(pkg.name, recs)
    if rec.status == HONOURS:
        return f"{pkg.name} honours MODEL_WEIGHTS_DIR ({rec.evidence})"
    return (f"{pkg.name} has not been probed for MODEL_WEIGHTS_DIR (docs/weights-source.md says how). The test container runs with the "
            "Hub off, so a package that hard-codes its repository fails the swap test, but only after a coder boot")
