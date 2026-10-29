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


# ------------------------------------------------------------------------------------------ Hugging Face cache snapshots (links into blobs)

def _hf_cache_snapshot(tmp_path, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")):
    """A snapshot laid out as the Hugging Face cache does: each file is a link into ../../blobs."""
    import json
    import struct
    blobs, snap = tmp_path / "blobs", tmp_path / "snapshots" / "rev"
    blobs.mkdir()
    snap.mkdir(parents=True)

    def put(name, data: bytes):
        (blobs / name.replace("/", "_")).write_bytes(data)
        os.symlink(os.path.relpath(blobs / name.replace("/", "_"), snap), snap / name)
    put("config.json", json.dumps({"architectures": ["LlamaForCausalLM"], "hidden_size": 64}).encode())
    header = json.dumps({"w%d" % i: {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]} for i in range(len(shards))}).encode()
    for s in shards:
        put(s, struct.pack("<Q", len(header)) + header + b"\0" * 8)
    put("model.safetensors.index.json", json.dumps({"weight_map": {"w%d" % i: s for i, s in enumerate(shards)}}).encode())
    return snap


def test_stock_runtime_reads_a_cache_shaped_snapshot_with_linked_config_and_index(tmp_path):
    from orchard import stock_runtime
    snap = _hf_cache_snapshot(tmp_path)
    cfg, why = stock_runtime._load_config(snap / "config.json")
    assert why is None and cfg["hidden_size"] == 64
    tensors, problems = stock_runtime.read_tensors(snap)
    assert problems == [] and set(tensors) == {"w0", "w1"}


def test_a_linked_config_that_points_at_a_device_or_loops_is_still_refused(tmp_path):
    from orchard import stock_runtime
    (tmp_path / "dev.json").symlink_to("/dev/zero")
    (tmp_path / "loop.json").symlink_to(tmp_path / "loop.json")
    for name in ("dev.json", "loop.json", "missing.json"):
        cfg, why = stock_runtime._load_config(tmp_path / name)
        assert cfg is None and why


def test_nearest_and_swap_draft_read_a_manifest_that_is_a_link_into_the_cache(tmp_path):
    import json
    from orchard import nearest, swap_draft
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "m").write_text(json.dumps({"weights": {"repo_id": "org/m"}, "device_count": 2, "schema_version": "5.1"}))
    manifest = tmp_path / "manifest.json"
    manifest.symlink_to(blobs / "m")
    home = tmp_path / "home"
    (home / ".cache" / "tt-model").mkdir(parents=True)
    (home / ".cache" / "tt-model" / "installed.json").write_text(
        json.dumps({"p": {"container": True, "image": "i", "manifest": str(manifest)}}))
    assert [c.weights_repo for c in nearest.read_containers(home)] == ["org/m"]
    assert [c["repo"] for c in swap_draft.installed_containers(home)] == ["org/m"]


def test_a_linked_run_file_and_a_linked_operator_file_are_still_read(tmp_path):
    """Files that were read through a link before the port still are: a run file, a suite, an items file."""
    import json
    from orchard import bench_session, package, prompt_suite, verdict
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"3": [1, 2]}))
    items = tmp_path / "items.json"
    items.symlink_to(real)
    assert bench_session.load_items(items) == {"3": [1, 2]}
    run = tmp_path / "run"
    (run / "stages/1").mkdir(parents=True)
    (run / "blob.json").write_text(json.dumps({"3": [1, 2]}))
    (run / "stages/1/reference.json").symlink_to(run / "blob.json")      # a link that stays inside the run
    assert verdict._read_json(run, "stages/1/reference.json") == {"3": [1, 2]}
    assert package._json(items, "x") == {"3": [1, 2]}


def test_a_ledger_appended_to_during_the_read_is_not_reported_corrupt(tmp_path, monkeypatch):
    """A live supervisor appends while status reads. The read must return the entries that were complete when it began."""
    from orchard import ledger
    path = tmp_path / "ledger.jsonl"
    with ledger.Ledger(path) as led:
        led.append("run_start", None, model="org/m")
        state = {"done": False}

        def hook(real):
            def wrapped(fd, n, *rest):
                if not state["done"]:
                    state["done"] = True                          # the supervisor appends after the reader's size check, before its read
                    led.append("notice", None, what="appended during the read")
                return real(fd, n, *rest)
            return wrapped
        monkeypatch.setattr(os, "pread", hook(os.pread))
        monkeypatch.setattr(os, "read", hook(os.read))
        entries = ledger.read_entries(path)
    assert state["done"] and [e["event"] for e in entries] == ["run_start"]


def test_read_prefix_refuses_a_file_cut_short_and_one_over_the_cap(tmp_path, monkeypatch):
    p = tmp_path / "f"
    p.write_bytes(b"abcdef")
    with pytest.raises(safeio.TooLarge):
        safeio.read_prefix(p, max_bytes=3)
    real = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, n, off: b"")
    with pytest.raises(safeio.ChangedWhileReading):
        safeio.read_prefix(p, max_bytes=10)
    monkeypatch.setattr(os, "pread", real)
    assert safeio.read_prefix(p, max_bytes=10) == b"abcdef"
