#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Count raw file access in orchard/ per file, for the ratchet in tests/test_hardening_ratchet.py.

Raw access is a call to the builtin `open` (also `builtins.open`, `io.open` and `codecs.open`), or to `.read_text`, `.read_bytes`, `.write_text` or `.write_bytes` on anything other than
`safeio` (`safeio.read_bytes(...)` and `orchard.safeio.read_bytes(...)` are the guarded readers and are not counted). Reviews of this
repository kept finding that such a call on a file something else controls can block (a FIFO), follow a link, read without a bound
or write through a refused path (docs/pr-hardening-checklist.md, items 1 and 2). New code should use orchard/safeio.py.

Calls are found in the syntax tree, so spacing and line breaks do not hide a call, and text inside a string or a comment is not a
call. A call is left out of the count when a comment on its first or last line says `hardening: ok <reason>` (the words `hardening: ok`, a space and a reason of at least three characters; a bare `ok` or `okay` does not count); the same words inside a
string do not count as that comment. A file that does not parse raises `SyntaxError`.

    python3 scripts/hardening_baseline.py            # print the counts
    python3 scripts/hardening_baseline.py --update   # rewrite tests/hardening_baseline.json when counts only fell (refuses growth)
    python3 scripts/hardening_baseline.py --accept-growth   # rewrite it even when counts rose (say why in the project log)
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
BASELINE = ROOT / "tests" / "hardening_baseline.json"
HAZARD_BASELINE = ROOT / "tests" / "hardening_hazards_baseline.json"   # the second count: path hazards (see count_hazards_source)
EXEMPT_FILES = {"orchard/safeio.py"}          # the module that implements the guarded access
# `hardening: ok` as whole words, then whitespace, then a reason of at least three non-space characters in all (not `okay`, not a bare `ok`)
EXEMPT_RE = re.compile(r"hardening: ok(?=\s)\s+(?=(?:\s*\S){3})")
RAW_METHODS = {"read_text", "read_bytes", "write_text", "write_bytes"}
OPEN_MODULES = {"builtins", "io", "codecs"}      # modules whose `open` is the plain file open (a bare `.open(` on any object is too broad: webbrowser, tarfile, Image)


def _marked_lines(source: str) -> set[int]:
    """Line numbers that carry a comment with the exemption words. Strings never count."""
    marked: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT and EXEMPT_RE.search(tok.string):
            marked.add(tok.start[0])
    return marked


def _is_safeio(node: ast.expr) -> bool:
    """True for `safeio` and for any dotted path that ends in `.safeio`."""
    return (isinstance(node, ast.Name) and node.id == "safeio") or (isinstance(node, ast.Attribute) and node.attr == "safeio")


def _is_file_open(f: ast.expr) -> bool:
    """True for the callee of the builtin `open` and of `builtins.open`, `io.open` and `codecs.open` (OPEN_MODULES): the plain file open."""
    if isinstance(f, ast.Name):
        return f.id == "open"
    return isinstance(f, ast.Attribute) and f.attr == "open" and isinstance(f.value, ast.Name) and f.value.id in OPEN_MODULES


def _is_raw_call(node: ast.Call) -> bool:
    f = node.func
    if _is_file_open(f):
        return True
    return isinstance(f, ast.Attribute) and f.attr in RAW_METHODS and not _is_safeio(f.value)


def count_source(source: str) -> int:
    tree = ast.parse(source)                    # a file that does not parse raises SyntaxError before anything else reads it
    marked = _marked_lines(source)
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_raw_call(node):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if node.lineno not in marked and end not in marked:     # a marker between the two lines belongs to an inner call
                n += 1
    return n


def _mentions_nofollow(node: ast.AST) -> bool:
    """True when `O_NOFOLLOW` (or a module constant named `NOFOLLOW`) appears anywhere in the expression: a name, an attribute, or a string given to getattr.
    Flags kept in a variable are not seen through, so such a call is counted until it is marked."""
    for n in ast.walk(node):
        if (isinstance(n, ast.Name) and n.id.endswith("NOFOLLOW")) or (isinstance(n, ast.Attribute) and n.attr == "O_NOFOLLOW"):   # `NOFOLLOW` module constants too
            return True
        if isinstance(n, ast.Constant) and n.value == "O_NOFOLLOW":
            return True
    return False


def _is_hazard_call(node: ast.Call) -> bool:
    """A call that makes one of the review mistakes about links (docs/pr-hardening-checklist.md, the symlink recipe):

    * `os.open(...)` whose flags never mention `O_NOFOLLOW`, with or without a `dir_fd` (a descriptor anchors the lookup but a link in the last part is still followed);
    * the builtin `open(..., "w")` (or "a", "x", "r+" ...) with a literal write mode: it follows a link and truncates the target, the
      way the ledger lock file once did (issue 493). A mode that is not a literal is not counted.
    """
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr == "open" and isinstance(f.value, ast.Name) and f.value.id == "os":
        # A descriptor-relative open (`dir_fd`) still follows a link in the last part, so it needs O_NOFOLLOW too.
        flags = node.args[1:2] + [kw.value for kw in node.keywords if kw.arg == "flags"]
        return not any(_mentions_nofollow(a) for a in flags)
    if _is_file_open(f):          # the builtin and its qualified forms (builtins.open, io.open, codecs.open), as the raw count sees them
        mode = node.args[1] if len(node.args) > 1 else next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
        return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(c in mode.value for c in "wax+")
    return False


def count_hazards_source(source: str) -> int:
    """Count the link hazards in `source`. The marker `# hardening: ok <reason>` leaves a call out, as for the raw count."""
    tree = ast.parse(source)
    marked = _marked_lines(source)
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_hazard_call(node):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if node.lineno not in marked and end not in marked:
                n += 1
    return n


def count_hazards_all(root: Path = ROOT) -> dict[str, int]:
    """Link-hazard count per file under orchard/ (files with none left out); safeio.py is exempt, as for the raw count."""
    out: dict[str, int] = {}
    for p in sorted((root / "orchard").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel in EXEMPT_FILES:
            continue
        n = count_hazards_source(p.read_text(encoding="utf-8", errors="replace"))
        if n:
            out[rel] = n
    return out


def count_file(path: Path) -> int:
    return count_source(path.read_text(encoding="utf-8", errors="replace"))


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


def growth(counts: dict[str, int], path: Path = BASELINE) -> list[str]:
    """What is above the recorded baseline in `path` (a count that rose, or a file that is new), as `file was -> now`. Writes nothing."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return [f"{rel} {old.get(rel, 0)} -> {n}" for rel, n in sorted(counts.items()) if n > old.get(rel, 0)]


def write_baseline(counts: dict[str, int], path: Path) -> None:
    path.write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def update_baseline(counts: dict[str, int], path: Path = BASELINE, accept_growth: bool = False) -> list[str]:
    """Write `counts` as the baseline. When any count is above the recorded one (or a file is new), write nothing and return what grew,
    unless `accept_growth` is true. An empty list means the baseline was written."""
    grew = growth(counts, path)
    if grew and not accept_growth:
        return grew
    write_baseline(counts, path)
    return []


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    counts = count_all()
    hazards = count_hazards_all()
    if "--update" in argv or "--accept-growth" in argv:
        # Both counts are checked before either file is written, so a refusal leaves both baselines as they were.
        grew = growth(counts, BASELINE) + growth(hazards, HAZARD_BASELINE)
        if grew and "--accept-growth" not in argv:
            print("refused: raw file access or link hazards grew (was -> now): " + ", ".join(grew) + ". Use orchard/safeio.py, mark a line that is right "
                  "to keep with `# hardening: ok <reason>`, or, when another merged change raised the counts, run with --accept-growth "
                  "and say why in the project log.", file=sys.stderr)
            return 1
        write_baseline(counts, BASELINE)
        write_baseline(hazards, HAZARD_BASELINE)
        print(f"wrote {BASELINE.relative_to(ROOT)}: {sum(counts.values())} raw accesses in {len(counts)} files; "
              f"{HAZARD_BASELINE.relative_to(ROOT)}: {sum(hazards.values())} link hazards in {len(hazards)} files")
        return 0
    for rel, n in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"{n:5d}  {rel}")
    print(f"{sum(counts.values()):5d}  total in {len(counts)} files")
    print(f"{sum(hazards.values()):5d}  link hazards (os.open without O_NOFOLLOW, open with a write mode) in {len(hazards)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
