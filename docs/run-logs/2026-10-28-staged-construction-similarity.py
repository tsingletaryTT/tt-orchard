#!/usr/bin/env python3
"""Normalized similarity of kernels to the orchard library kernels and the tt-lang checkout's examples.

Usage: similarity.py LIBRARY_DIR EXAMPLES_DIR KERNEL.py [KERNEL.py ...]
What is compared: the operation, which is the function decorated with `ttl.operation` (the whole `@ttl.operation ... def` block
and its nested thread functions), so host-side test code in an example does not dilute a copy of the kernel. A file with no such
function is compared whole. Normalization: comments and blank lines are removed, and identifiers are kept (a rename lowers the
score). The function's own name is replaced by `OPERATION` on both sides, since the probe accepts any name. Score:
difflib.SequenceMatcher ratio over the remaining lines, 0 to 1. Flag: 0.80 or more against any file.
The script refuses (it exits non-zero with a message) a comparison directory that holds no .py file and a kernel that is missing or empty, so a clean
result always comes from a real comparison. Prints one line per kernel.
"""
import ast
import difflib
import re
import sys
from pathlib import Path

FLAG = 0.80


def operation_text(text):
    """The source of the ttl.operation function, or the whole text when there is none (or it does not parse)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any("operation" in ast.unparse(d) for d in node.decorator_list):
            body = ast.get_source_segment(text, node) or text
            decorators = "\n".join(f"@{ast.unparse(d)}" for d in node.decorator_list)
            return decorators + "\n" + re.sub(r"^def\s+" + re.escape(node.name) + r"\b", "def OPERATION", body, count=1)
    return text


def norm(text):
    out = []
    for line in operation_text(text).splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def files(folder):
    found = sorted(Path(folder).glob("*.py"))
    if not found:
        sys.exit(f"refused: no .py file in {folder}; no similarity check ran")
    return found


def best(lines, folder):
    top = (0.0, "-")
    for p in files(folder):
        r = difflib.SequenceMatcher(None, lines, norm(p.read_text(encoding="utf-8", errors="replace")), autojunk=False).ratio()
        if r > top[0]:
            top = (r, p.name)
    return top


def main(argv):
    if len(argv) < 4:
        sys.exit(__doc__)
    lib, ex, kernels = argv[1], argv[2], argv[3:]
    for k in kernels:
        text = Path(k).read_text(encoding="utf-8", errors="replace") if Path(k).is_file() else ""
        lines = norm(text)
        if not lines:
            sys.exit(f"refused: {k} is missing or empty; no similarity check ran")
        a, b = best(lines, lib), best(lines, ex)
        flag = "FLAG near-copy" if max(a[0], b[0]) >= FLAG else "ok"
        print(f"{k}\tlibrary {a[0]:.2f} ({a[1]})\texamples {b[0]:.2f} ({b[1]})\t{flag}")


if __name__ == "__main__":
    main(sys.argv)
