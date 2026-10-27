# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Every registered reader of a file or record that something else wrote, against the hostile corpus.

The registry (`READERS`) and the corpus are in tests/hostile_inputs.py. For each (reader, item) pair this file asserts the contract
described on `hostile_inputs.Reader`: the reader returns a value free of NaN, Infinity and lone surrogates, or raises an exception of
its documented type, finishes quickly, and does not read most of a 50 MB file or any part of a link to /dev/zero.

A reader that fails today is not weakened here. Its pair is listed in `KNOWN_DEFECTS` with the GitHub issue that tracks the defect and is
marked xfail(strict=True): when the reader is fixed the test turns red until the entry is removed, so the list cannot go stale.

The readers run in child processes with a 2 GiB address-space limit and a timeout (a FIFO must not hang the suite and /dev/zero must not
exhaust its memory). One shared fixture runs every reader once, in parallel, and the per-pair tests read that result.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import hostile_inputs as hi

ORCHARD = Path(hi.REPO) / "orchard"
PAIRS = [(r, i.name) for r in hi.READERS for i in hi.ITEMS if not (r.feed == "bytes" and i.kind not in ("bytes", "big"))]

_cache: dict[str, dict] = {}


def results_for(reader: hi.Reader) -> dict:
    """The child-process verdicts for `reader`, computed once for the whole registry (in parallel) on first use."""
    if not _cache:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for name, got in pool.map(lambda r: (r.name, hi.run_reader(r)), hi.READERS):
                _cache[name] = got
    return _cache[reader.name]


def _param(reader: hi.Reader, item: str):
    issue = hi.XFAIL_BY_PAIR.get((reader.name, item))
    marks = [pytest.mark.xfail(strict=True, reason=f"issue {issue}: {hi.KNOWN_DEFECT_TEXT[issue]}")] if issue else []
    return pytest.param(reader, item, id=f"{reader.name}::{item}", marks=marks)


@pytest.mark.parametrize("reader,item", [_param(r, i) for r, i in PAIRS])
def test_reader_keeps_its_contract_on_hostile_input(reader, item):
    got = results_for(reader)[item]
    assert got["ok"], got["why"]


# ----- the registry itself

def test_a_reader_is_registered_once_and_the_corpus_has_the_required_items():
    names = [r.name for r in hi.READERS]
    assert len(names) == len(set(names)) and len(names) >= 25
    needed = {"empty_array", "empty_object", "null", "empty_string", "string_x", "zero", "minus_one", "huge_exponent", "nan", "infinity",
              "true", "nest_1000", "big_string_50mb", "duplicate_keys", "lone_surrogate", "invalid_utf8", "bom", "truncated", "empty",
              "directory", "fifo", "link_to_regular_file", "link_to_dev_zero", "permission_denied"}
    assert needed <= {i.name for i in hi.ITEMS}


def test_every_known_defect_names_a_real_pair():
    pairs = {(r.name, i) for r, i in PAIRS}
    stale = [pair for pair in hi.XFAIL_BY_PAIR if pair not in pairs]
    assert stale == []
    assert set(hi.KNOWN_DEFECTS) == set(hi.KNOWN_DEFECT_TEXT)


def _modules_that_parse_json() -> set[str]:
    out = set()
    for path in ORCHARD.rglob("*.py"):
        rel = path.relative_to(ORCHARD)
        if rel.parts[0] in ("skills", "package_templates", "lm_eval_tasks", "web"):
            continue
        if re.search(r"\bjson\.loads?\(", path.read_text(encoding="utf-8")):
            out.add(str(rel))
    return out


def _module_of(reader_name: str) -> str | None:
    parts = reader_name.split(" ")[0].split(".")
    for n in range(len(parts), 0, -1):
        candidate = ORCHARD.joinpath(*parts[:n]).with_suffix(".py")
        if candidate.exists():
            return str(candidate.relative_to(ORCHARD))
    return None


def test_the_registry_covers_these_modules_and_the_rest_are_listed():
    """A module that parses JSON is either covered by a registered reader or named in hostile_inputs.UNREGISTERED_MODULES. A new module
    that parses JSON fails here until a reader is registered for it (or it is added to the list, which is the backlog); a module that
    gets a reader fails here until it leaves the list."""
    parsing = _modules_that_parse_json()
    covered = {_module_of(r.name) for r in hi.READERS}
    assert None not in covered, [r.name for r in hi.READERS if _module_of(r.name) is None]
    assert parsing - covered == set(hi.UNREGISTERED_MODULES)
    assert not (covered & set(hi.UNREGISTERED_MODULES))


# ----- the instrument can fail: readers that are wrong on purpose are caught, and a reader built on safeio is not

def _selftest(name: str) -> dict:
    return hi.run_reader(hi.SELFTEST[name])


def test_a_reader_built_on_safeio_passes_every_item():
    got = _selftest("selftest.safeio")
    assert len(got) >= 38
    assert {k: v["why"] for k, v in got.items() if not v["ok"]} == {}


@pytest.mark.parametrize("name,items", [
    ("selftest.passes_nan_on", {"huge_exponent", "nan", "infinity", "negative_infinity", "nan_in_object"}),
    ("selftest.raises_keyerror", {"empty_object", "null", "nan"}),
    ("selftest.hangs_on_a_fifo", {"fifo"}),
    ("selftest.reads_without_a_cap", {"big_string_50mb", "link_to_dev_zero"}),
    ("selftest.recurses", {"nest_100000"}),
])
def test_a_reader_that_is_wrong_on_purpose_is_reported(name, items):
    got = _selftest(name)
    assert items <= {k for k, v in got.items() if not v["ok"]}, {k: v for k, v in got.items() if k in items}
