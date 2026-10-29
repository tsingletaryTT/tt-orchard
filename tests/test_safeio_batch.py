# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the readers moved onto orchard/safeio.py in the safeio-batch change (issues 135 to 140, 239), and for `safeio.read_jsonl`.

The corpus-wide proofs are in tests/test_hostile_readers.py. This file holds the cases that the corpus does not express."""
from __future__ import annotations

import os

import pytest

from orchard import safeio


# ------------------------------------------------------------------------------------------------------------ read_jsonl

def test_read_jsonl_returns_each_line_and_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"a": 1}\n\n[2]\n "s" \n')
    assert safeio.read_jsonl(p, max_bytes=1000) == [{"a": 1}, [2], "s"]


def test_read_jsonl_does_not_split_on_unicode_line_separators(tmp_path):
    """lm-eval leaves U+2028 raw inside a JSON string; splitlines would cut the line there."""
    p = tmp_path / "x.jsonl"
    p.write_bytes('{"a": "x y"}\n'.encode("utf-8"))
    assert safeio.read_jsonl(p, max_bytes=1000) == [{"a": "x y"}]


def test_read_jsonl_names_the_bad_line_without_its_content(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"a": 1}\n{"secret-token": NaN}\n')
    with pytest.raises(safeio.NotJson) as info:
        safeio.read_jsonl(p, max_bytes=1000)
    assert "line 2" in info.value.reason and "secret-token" not in info.value.reason


def test_read_jsonl_refuses_a_file_over_the_cap_and_a_fifo(tmp_path):
    big = tmp_path / "big.jsonl"
    big.write_bytes(b'{"a": 1}\n' * 10)
    with pytest.raises(safeio.TooLarge):
        safeio.read_jsonl(big, max_bytes=20)
    fifo = tmp_path / "fifo.jsonl"
    os.mkfifo(fifo)
    with pytest.raises(safeio.NotRegular):
        safeio.read_jsonl(fifo, max_bytes=20)


def test_read_jsonl_applies_expect_to_every_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"a": 1}\n[1]\n')
    with pytest.raises(safeio.WrongType):
        safeio.read_jsonl(p, max_bytes=1000, expect=dict)
