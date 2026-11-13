# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ratchet on float checks that an integer too large for a float can crash (issue 461).

`math.isfinite(10**400)` and `float(10**400)` raise OverflowError. Existing calls in orchard/ are recorded per file in
tests/finite_baseline.json. A change may lower a count. It may not raise one or add a file with such a call: use `safeio.is_finite_number`,
`safeio.fits_float` or `safeio.as_float`, or mark a line that is right to keep with `# finite: ok <reason>`. Lower the baseline with
`python3 scripts/finite_baseline.py --update` when a call is replaced.

This replaces the first ratchet of issue 461, which counted `math.isfinite` only and had no marker.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("finite_baseline", REPO / "scripts" / "finite_baseline.py")
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)


def test_no_file_gained_a_float_check_that_a_huge_integer_can_crash():
    baseline = json.loads(fb.BASELINE.read_text(encoding="utf-8"))
    now = fb.count_all(REPO)
    grew = {rel: (baseline.get(rel, 0), n) for rel, n in now.items() if n > baseline.get(rel, 0)}
    assert not grew, (
        "math.isfinite( or float( calls grew (was, now): " + ", ".join(f"{rel} {a} -> {b}" for rel, (a, b) in sorted(grew.items()))
        + ". Use safeio.is_finite_number, safeio.fits_float or safeio.as_float, or mark a line that is right to keep with "
          "`# finite: ok <reason>`.")


def test_the_baseline_names_only_files_that_exist():
    baseline = json.loads(fb.BASELINE.read_text(encoding="utf-8"))
    assert [rel for rel in baseline if not (REPO / rel).is_file()] == []


def test_the_helpers_module_is_not_counted():
    assert "orchard/safeio.py" not in fb.count_all(REPO)


def test_the_counter_counts_what_it_says():
    src = "\n".join([
        "import math",
        "a = math.isfinite(x)",                                   # 1
        "b = float(x)",                                           # 1
        "c = math.isfinite(y) and math.isfinite(float(z))",       # 3
        "d = float('inf') + float(\"nan\")",                      # string literals cannot overflow
        "e = float()",                                            # no argument
        "f = other.isfinite(w)",                                  # not math.isfinite
        "g = np.float64(x)",                                      # not the builtin
        "# math.isfinite(q) float(q)",                            # a comment
        "h = 'math.isfinite(q) float(q)'",                        # a string
        "i = float(x)  # finite: ok x is a float here",           # marked
    ]) + "\n"
    assert fb.count_source(src) == 5


def test_calls_split_over_lines_are_counted_and_a_marker_on_the_last_line_counts():
    assert fb.count_source("a = math.isfinite(\n    x)\n") == 1
    assert fb.count_source("a = float(\n    x)  # finite: ok x is a float here\n") == 0


def test_only_a_comment_with_a_reason_marks_a_line():
    assert fb.count_source("a = float(x); s = 'finite: ok x is a float'\n") == 1         # inside a string
    assert fb.count_source("a = float(x)  # finite: ok\n") == 1                         # no reason
    assert fb.count_source("a = float(x)  # finite: okay it is fine\n") == 1             # not the marker words
    assert fb.count_source("a = float(x)  # finite: ok it is a float\n") == 0


def test_a_marker_on_an_inner_call_does_not_exempt_the_call_around_it():
    nested = "a = math.isfinite(\n    float(x),  # finite: ok x is a float\n)\n"
    assert fb.count_source(nested) == 1
