# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Records of version combinations that were run on this hardware (read by `tt-orchard updates`).

A record says: this exact set of versions was tried, on this date, with this result, and the evidence is at this path. The file
is `config/verifications.json`, or the path in $ORCHARD_VERIFICATIONS. Schema:

    {"schema": 1, "records": [
        {"versions": {"tt-lang": "<full sha>", "tt-lang-wheel": "1.1.10.dev20261026", "tt-metal": "v0.80.0"},
         "date": "2026-10-29", "result": "pass", "evidence": "docs/run-logs/2026-10-29-ttlang-pair-chip-check.md"}]}

`versions` maps a component name to a version string. A record covers a combination when every component the record names has
the same value in the combination (a record names only the components the run pinned). Only a record with result "pass" makes a
combination verified. Nothing writes this file in code: a person or a verification run adds a record after a real run, and
this module only reads it. A damaged file, or a record of the wrong shape, is skipped and reported in `problems`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from orchard import safeio

MAX_BYTES = 1024 * 1024
RESULTS = ("pass", "fail")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class Record:
    versions: dict
    date: str
    result: str
    evidence: str


@dataclass
class Verifications:
    records: list[Record] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def match(self, combo: dict) -> Record | None:
        """The newest record (pass or fail) whose every component equals the combination's, or None. The caller checks
        `result`, so a newer failing record hides an older passing one."""
        hits = [r for r in self.records if r.versions and all(combo.get(k) == v for k, v in r.versions.items())]
        hits.sort(key=lambda r: r.date)
        return hits[-1] if hits else None


def default_path(checkout: Path | None = None) -> Path:
    env = os.environ.get("ORCHARD_VERIFICATIONS")
    if env:
        return Path(env)
    return Path(checkout or Path(__file__).resolve().parent.parent) / "config" / "verifications.json"


def _record(raw) -> Record | str:
    if not isinstance(raw, dict):
        return "a record is not an object"
    v = raw.get("versions")
    if not isinstance(v, dict) or not v or not all(isinstance(k, str) and isinstance(x, str) and x for k, x in v.items()):
        return "versions must be a non-empty object of strings"
    if not isinstance(raw.get("date"), str) or not _DATE.match(raw["date"]):
        return "date must be YYYY-MM-DD"
    if raw.get("result") not in RESULTS:
        return "result must be pass or fail"
    ev = raw.get("evidence")
    if not isinstance(ev, str) or not ev or len(ev) > 300:
        return "evidence must be a path string"
    return Record(dict(v), raw["date"], raw["result"], ev)


def load(path) -> Verifications:
    """Read the file. A missing file is an empty result without a problem; anything else wrong is a problem."""
    out = Verifications()
    p = Path(path)
    if not p.exists():
        return out
    try:
        data = safeio.read_json(p, max_bytes=MAX_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        out.problems.append(f"{p.name} could not be read: {type(exc).__name__}")
        return out
    if data.get("schema") != 1 or not isinstance(data.get("records"), list):
        out.problems.append(f"{p.name}: schema must be 1 with a records list")
        return out
    for i, raw in enumerate(data["records"]):
        rec = _record(raw)
        if isinstance(rec, str):
            out.problems.append(f"{p.name}: record {i}: {rec}")
        else:
            out.records.append(rec)
    return out
