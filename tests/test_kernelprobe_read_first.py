# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernel-probe skill's read-first paragraph (version 0.9.36, branch probe-read-first).

A read-only analysis of 19 `rmsnorm` attempts found that the coder read on average 19% of the specification and the
section "Block states" in none of them, while `pop(): not a valid next dataflow step` ended 4 attempts and is explained
there. The skill now tells the coder to read three sections in full before it writes a kernel. It names section titles
only: no worked solution and no op-specific fix. These tests pin the wording, check each title against the pinned
specification (the same pattern as the tests that read the simulator source) and check that the instruction works with
the probe's `read_file`."""
import json
import os
import re
from pathlib import Path

import pytest

from orchard import kernelprobe as kp

SRC = Path(os.environ.get("ORCHARD_TTLANG_SRC", Path.home() / "orchard-src" / "tt-lang"))
SPEC = SRC / "docs" / "sphinx" / "specs" / "TTLangSpecification.md"
needs_spec = pytest.mark.skipif(not SPEC.is_file(), reason="SKIPPED: no tt-lang checkout with docs/sphinx/specs")
TITLES = ("Dataflow buffer", "Block", "Block states")


def _paragraph() -> str:
    """The read-first paragraph: from its first words to the next blank line, whitespace flattened."""
    text = kp.SKILL.read_text()
    start = text.index("Read before you write any kernel")
    return " ".join(text[start:].split("\n\n")[0].split())


def _re_read_sentence() -> str:
    (sentence,) = re.findall(r"[^.]*\bread \"Block states\" again[^.]*\.", _paragraph())
    return sentence


def test_the_paragraph_comes_before_the_steps_and_says_before_you_write():
    text = kp.SKILL.read_text()
    assert text.index("Read before you write any kernel") < text.index("\nSteps:\n")
    assert "Read before you write any kernel: three sections of `reference/TTLangSpecification.md`, each in full" \
        in _paragraph()


@pytest.mark.parametrize("title", TITLES)
def test_the_paragraph_names_each_section_in_quotes(title):
    assert f'"{title}"' in _paragraph()


def test_the_three_titles_are_named_in_one_list():
    assert '"Dataflow buffer", "Block" and "Block states"' in _paragraph()


def test_the_paragraph_names_the_commands_that_find_and_read_a_section():
    p = _paragraph()
    assert '`grep -n "^## " reference/TTLangSpecification.md`' in p
    assert "`read_file`" in p and "offset" in p and "limit" in p
    assert "follow the offset" in p                 # a section longer than one page comes back in pages


def test_a_dataflow_error_sends_the_coder_back_to_block_states():
    s = _re_read_sentence()
    assert "dataflow" in s and "pop()" in s and "before you change the kernel" in s


def test_the_paragraph_holds_no_worked_solution_and_no_op_name():
    p = _paragraph().lower()
    for word in ("rsqrt", "reduce_sum", "mean(", "eps", "rmsnorm", "broadcast", "fill", "store(", "```", "def "):
        assert word not in p, word


def test_the_skill_stays_within_100_lines_with_one_fenced_skeleton():
    text = kp.SKILL.read_text()
    assert len(text.splitlines()) <= 100
    assert len(re.findall(r"```", text)) == 2


@needs_spec
@pytest.mark.parametrize("title", TITLES)
def test_each_named_section_is_a_heading_of_the_pinned_specification(title):
    headings = [l[3:].strip() for l in SPEC.read_text().splitlines() if l.startswith("## ")]
    assert headings.count(title) == 1, headings


@needs_spec
def test_the_grep_the_skill_names_prints_the_three_headings_and_each_section_follows_the_next_heading():
    lines = SPEC.read_text().splitlines()
    printed = [(i + 1, l[3:].strip()) for i, l in enumerate(lines) if re.match(r"## ", l)]
    by_title = {t: n for n, t in printed}
    assert all(t in by_title for t in TITLES)
    nxt = {t: min(n for n, _ in printed if n > by_title[t]) for t in TITLES}
    assert nxt["Dataflow buffer"] == by_title["Block"]          # the sections the skill names are neighbours
    assert nxt["Block"] == by_title["Block states"]


@needs_spec
@pytest.mark.parametrize("title", TITLES)
def test_read_file_with_offset_and_limit_returns_the_whole_section_by_following_the_offsets(tmp_path, title):
    """The instruction works mechanically: the probe's read_file, given a heading's line as offset and the section's
    length as limit, returns pages that together hold every line of the section."""
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "TTLangSpecification.md").write_text(SPEC.read_text())
    tools = kp.ProbeTools(tmp_path, tmp_path, {})
    lines = SPEC.read_text().splitlines()
    starts = [i + 1 for i, l in enumerate(lines) if l.startswith("## ")] + [len(lines) + 1]
    first = next(n for n in starts if lines[n - 1][3:].strip() == title)
    length = min(n for n in starts if n > first) - first
    got, offset, end = [], first, first + length - 1
    while offset <= end:
        out = tools.call("read_file", json.dumps({"path": "reference/TTLangSpecification.md", "offset": offset,
                                                  "limit": end - offset + 1}))
        m = re.search(r"\n\[lines (\d+) to (\d+) of \d+", out)
        assert m and int(m.group(1)) == offset, out[:200]
        got += out[:m.start()].splitlines()
        nxt = re.search(r"read on with offset=(\d+)", out)
        offset = int(nxt.group(1)) if nxt else int(m.group(2)) + 1      # follow the offset the page names
        assert offset == int(m.group(2)) + 1, out[-120:]
    # a page drops the blank lines at its end, so compare the lines that hold text
    assert got[0] == f"## {title}"
    assert [l for l in got if l.strip()] == [l for l in lines[first - 1:end] if l.strip()]
