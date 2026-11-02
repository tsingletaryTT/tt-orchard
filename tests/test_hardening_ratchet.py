# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ratchet on raw file access in orchard/ (docs/pr-hardening-checklist.md, items 1 and 2).

Reviews kept finding that `.read_text(`, `.read_bytes(`, `.write_text(`, `.write_bytes(` and `open(` on a file something else controls
can block on a FIFO, follow a link, read without a bound or write through a path a reader refused. Existing uses are recorded in
tests/hardening_baseline.json. A change may lower a count. It may not raise one or add a file with raw access: use orchard/safeio.py,
or mark a line that is right to keep with `# hardening: ok <reason>`.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("hardening_baseline", REPO / "scripts" / "hardening_baseline.py")
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)


def test_no_file_gained_raw_file_access():
    baseline = json.loads(hb.BASELINE.read_text(encoding="utf-8"))
    now = hb.count_all(REPO)
    grew = {rel: (baseline.get(rel, 0), n) for rel, n in now.items() if n > baseline.get(rel, 0)}
    assert not grew, (
        "raw file access grew (was, now): " + ", ".join(f"{rel} {a} -> {b}" for rel, (a, b) in sorted(grew.items()))
        + ". Read with orchard/safeio.py, write with safeio.atomic_write, or mark a line that is right to keep with "
          "`# hardening: ok <reason>`. See docs/pr-hardening-checklist.md.")


def test_the_counter_counts_what_it_says(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("\n".join([
        "a = p.read_text()",                               # 1
        "b = p.read_bytes() + q.read_bytes()",             # 2
        "p.write_text('x')",                               # 1
        "p.write_bytes(b'x')",                             # 1
        "with open(p) as fh: pass",                        # 1
        "os.open(p, 0)",                                   # not the builtin
        "webbrowser.open(url)",                            # not the builtin
        "# p.read_text()",                                 # a comment
        "c = p.read_text()  # hardening: ok our own file", # marked
        "def reopen(x): pass",                             # not open(
    ]) + "\n", encoding="utf-8")
    assert hb.count_file(f) == 6


def test_the_baseline_has_no_entry_for_a_missing_file():
    baseline = json.loads(hb.BASELINE.read_text(encoding="utf-8"))
    missing = [rel for rel in baseline if not (REPO / rel).is_file()]
    assert not missing, f"remove from tests/hardening_baseline.json (python3 scripts/hardening_baseline.py --update): {missing}"
