#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Count the float checks in orchard/ that an integer too large for a float can crash, per file, for tests/test_finite_ratchet.py (issue 461).

`math.isfinite(10**400)` and `float(10**400)` raise OverflowError: an integer of more than about 308 digits does not fit a float. A check
meant to refuse such a value from JSON crashes on it instead. New code uses `safeio.is_finite_number`, `safeio.fits_float` or
`safeio.as_float`, which never raise.

Two kinds of call are counted:

* `math.isfinite(...)`;
* the builtin `float(...)` with one argument that is not a string literal (`float("inf")` and `float("nan")` cannot overflow and are
  left out).

Calls are found in the syntax tree, so spacing and line breaks do not hide a call, and text inside a string or a comment is not a call.
A call is left out of the count when a comment on its first or last line says `finite: ok <reason>` (the words, a space and a reason of
at least three characters). Use the marker where the value cannot be an integer too large for a float (it was parsed by `float()` from
text, or it is a float computed in this module) or where an OverflowError is caught. orchard/safeio.py holds the helpers and is not
counted. A file that does not parse raises SyntaxError.

    python3 scripts/finite_baseline.py            # print the counts
    python3 scripts/finite_baseline.py --update   # rewrite tests/finite_baseline.json when counts only fell (refuses growth)
"""
from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "finite_baseline.json"
EXEMPT_FILES = {"orchard/safeio.py"}
EXEMPT_RE = re.compile(r"finite: ok(?=\s)\s+(?=(?:\s*\S){3})")


def _marked_lines(source: str) -> set[int]:
    """Line numbers that carry a comment with the marker. A string holding the same words does not count."""
    marked: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT and EXEMPT_RE.search(tok.string):
            marked.add(tok.start[0])
    return marked


def _is_counted(node: ast.Call) -> bool:
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr == "isfinite" and isinstance(f.value, ast.Name) and f.value.id == "math":
        return True
    if isinstance(f, ast.Name) and f.id == "float" and len(node.args) == 1 and not node.keywords:
        arg = node.args[0]
        return not (isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    return False


def count_source(source: str) -> int:
    tree = ast.parse(source)
    marked = _marked_lines(source)
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_counted(node):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if node.lineno not in marked and end not in marked:
                n += 1
    return n


def count_file(path: Path) -> int:
    return count_source(path.read_text(encoding="utf-8"))       # hardening: ok our own source files, read by a test


def count_all(root: Path = ROOT) -> dict[str, int]:
    out = {}
    for f in sorted((root / "orchard").rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel in EXEMPT_FILES:
            continue
        n = count_file(f)
        if n:
            out[rel] = n
    return out


def main(argv: list[str]) -> int:
    now = count_all()
    if "--update" not in argv:
        for rel, n in now.items():
            print(f"{n:4d}  {rel}")
        print(f"{sum(now.values())} calls in {len(now)} files")
        return 0
    old = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}      # hardening: ok the repository's own baseline
    grew = {rel: (old.get(rel, 0), n) for rel, n in now.items() if n > old.get(rel, 0)}
    if grew and old:
        print("refused: these counts rose (was, now): " + ", ".join(f"{r} {a} -> {b}" for r, (a, b) in sorted(grew.items())), file=sys.stderr)
        return 1
    BASELINE.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n", encoding="utf-8")     # hardening: ok the repository's own baseline
    print(f"wrote {BASELINE.relative_to(ROOT)}: {sum(now.values())} calls in {len(now)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
