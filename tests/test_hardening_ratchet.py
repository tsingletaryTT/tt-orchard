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


def _count(tmp_path, source: str) -> int:
    f = tmp_path / "m.py"
    f.write_text(source, encoding="utf-8")
    return hb.count_file(f)


def test_the_counter_counts_what_it_says(tmp_path):
    src = "\n".join([
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
    ]) + "\n"
    assert _count(tmp_path, src) == 6


def test_the_guarded_reader_is_not_raw_access(tmp_path):
    src = "import safeio\nx = safeio.read_bytes(p, max_bytes=10)\ny = orchard.safeio.read_bytes(p, max_bytes=10)\n"
    assert _count(tmp_path, src) == 0


def test_calls_with_odd_spacing_or_line_breaks_are_counted(tmp_path):
    src = "a = open (p)\nb = p.read_bytes ()\nc = p.read_text(\n    encoding='utf-8')\nd = (\n    open\n)(p)\n"
    assert _count(tmp_path, src) == 4


def test_text_inside_strings_and_trailing_comments_is_not_a_call(tmp_path):
    src = 'doc = "use p.read_text() or open(p)"\nx = 1  # p.read_bytes() would be raw\n"""p.write_text(x)"""\n'
    assert _count(tmp_path, src) == 0


def test_only_a_comment_can_mark_a_line_as_right_to_keep(tmp_path):
    marked = "a = p.read_text()  # hardening: ok our own file\n"
    in_string = "a = p.read_text(); s = 'hardening: ok'\n"
    multi = "a = p.read_text(\n    encoding='utf-8')  # hardening: ok our own file\n"
    assert _count(tmp_path, marked) == 0
    assert _count(tmp_path, in_string) == 1
    assert _count(tmp_path, multi) == 0


def test_a_marker_on_an_inner_call_does_not_exempt_the_call_around_it(tmp_path):
    nested = "dst.write_text(\n    src.read_text(),  # hardening: ok source is our own file\n)\n"
    assert _count(tmp_path, nested) == 1                  # the outer write is counted, the marked inner read is not
    first = "a = p.read_text(  # hardening: ok our own file\n    encoding='utf-8')\n"
    last = "a = p.read_text(\n    encoding='utf-8')  # hardening: ok our own file\n"
    assert _count(tmp_path, first) == 0 and _count(tmp_path, last) == 0
    middle = "a = p.read_text(\n    # hardening: ok our own file\n    encoding='utf-8')\n"
    assert _count(tmp_path, middle) == 1                  # a marker between the first and last line marks no call


def test_update_refuses_growth_unless_it_is_accepted(tmp_path):
    import json
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"a.py": 1, "d.py": 3}))
    grew = hb.update_baseline({"a.py": 2, "b.py": 1, "d.py": 1}, base)
    assert sorted(grew) == ["a.py 1 -> 2", "b.py 0 -> 1"]
    assert json.loads(base.read_text()) == {"a.py": 1, "d.py": 3}        # nothing written
    assert hb.update_baseline({"a.py": 2, "b.py": 1, "d.py": 1}, base, accept_growth=True) == []
    assert json.loads(base.read_text()) == {"a.py": 2, "b.py": 1, "d.py": 1}
    assert hb.update_baseline({"a.py": 1}, base) == []                  # only a decrease: written
    assert json.loads(base.read_text()) == {"a.py": 1}


def test_the_marker_needs_the_word_ok_and_a_reason(tmp_path):
    import pytest
    for comment in ("# hardening: ok", "# hardening: okay", "# hardening: ok ", "# hardening: ok-ish", "# hardening: okay our own file",
                    "# hardening ok our own file", "# Hardening: ok our own file", "# hardening: ok x", "# hardening: ok ab"):
        assert _count(tmp_path, f"a = p.read_text()  {comment}\n") == 1, comment
    for comment in ("# hardening: ok our own file", "#hardening: ok state file we wrote", "# note. hardening: ok written by this program",
                    "# hardening: ok a file this program wrote", "# hardening: ok I created this file", "# hardening: ok our file"):
        assert _count(tmp_path, f"a = p.read_text()  {comment}\n") == 0, comment


def test_a_file_that_does_not_parse_is_an_error_not_a_zero(tmp_path):
    import pytest
    with pytest.raises(SyntaxError):
        _count(tmp_path, "def broken(:\n")


def test_the_baseline_has_no_entry_for_a_missing_file():
    baseline = json.loads(hb.BASELINE.read_text(encoding="utf-8"))
    missing = [rel for rel in baseline if not (REPO / rel).is_file()]
    assert not missing, f"remove from tests/hardening_baseline.json (python3 scripts/hardening_baseline.py --update): {missing}"
