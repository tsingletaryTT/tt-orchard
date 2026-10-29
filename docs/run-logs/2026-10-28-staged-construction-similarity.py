#!/usr/bin/env python3
"""Normalized similarity of kernels to the orchard library kernels and the tt-lang checkout's examples.

Usage: similarity.py LIBRARY_DIR EXAMPLES_DIR KERNEL.py [KERNEL.py ...]
Normalization: comments, docstrings-as-lines starting with '#', blank lines and leading/trailing whitespace are removed, and
identifiers are kept (a rename lowers the score). Score: difflib.SequenceMatcher ratio over the remaining lines, 0 to 1.
Flag: a score of 0.80 or more against any file. Prints one line per kernel: its best match for each source.
"""
import difflib
import sys
from pathlib import Path


def norm(text):
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def best(lines, folder):
    top = (0.0, "-")
    for p in sorted(Path(folder).glob("*.py")):
        r = difflib.SequenceMatcher(None, lines, norm(p.read_text(encoding="utf-8", errors="replace")), autojunk=False).ratio()
        if r > top[0]:
            top = (r, p.name)
    return top


lib, ex = sys.argv[1], sys.argv[2]
for k in sys.argv[3:]:
    lines = norm(Path(k).read_text(encoding="utf-8", errors="replace"))
    a, b = best(lines, lib), best(lines, ex)
    flag = "FLAG near-copy" if max(a[0], b[0]) >= 0.80 else "ok"
    print(f"{k}\tlibrary {a[0]:.2f} ({a[1]})\texamples {b[0]:.2f} ({b[1]})\t{flag}")
