#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Count raw file access in orchard/ per file, for the ratchet in tests/test_hardening_ratchet.py.

Raw access is `.read_text(`, `.read_bytes(`, `.write_text(`, `.write_bytes(` and the builtin `open(`. Reviews of this repository kept
finding that such a call on a file something else controls can block (a FIFO), follow a link, read without a bound or write through a
refused path (docs/pr-hardening-checklist.md, items 1 and 2). New code should use orchard/safeio.py. A line that is right to keep
carries a `# hardening: ok <reason>` comment and is not counted. Comment-only lines are not counted.

    python3 scripts/hardening_baseline.py            # print the counts
    python3 scripts/hardening_baseline.py --update   # rewrite tests/hardening_baseline.json (do this when you removed raw access)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "hardening_baseline.json"
EXEMPT_FILES = {"orchard/safeio.py"}          # the module that implements the guarded access
EXEMPT_MARK = "hardening: ok"
RAW = re.compile(r"\.read_text\(|\.read_bytes\(|\.write_text\(|\.write_bytes\(|(?<![\w.])open\(")


def count_file(path: Path) -> int:
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        code = line.strip()
        if not code or code.startswith("#") or EXEMPT_MARK in line:
            continue
        n += len(RAW.findall(line))
    return n


def count_all(root: Path = ROOT) -> dict[str, int]:
    """Raw-access count per file under orchard/ (files with none are left out), keyed by the path relative to the repository."""
    out: dict[str, int] = {}
    for p in sorted((root / "orchard").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel in EXEMPT_FILES:
            continue
        n = count_file(p)
        if n:
            out[rel] = n
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    counts = count_all()
    if "--update" in argv:
        BASELINE.write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE.relative_to(ROOT)}: {sum(counts.values())} raw accesses in {len(counts)} files")
        return 0
    for rel, n in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"{n:5d}  {rel}")
    print(f"{sum(counts.values()):5d}  total in {len(counts)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
