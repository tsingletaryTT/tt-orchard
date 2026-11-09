# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 647: a full-port class may list sidecars.

Spark-X2.6-4B-Loop-mHC keeps its loop, hyper-connection and n-gram tensors in loop_modules.safetensors, outside the shard index. Stage 0
measures that file (size, sha256, tensor names) and drafts class `full-port` with it listed, and the gate said "class 'full-port' cannot have
sidecars". The agent then changed the class to `unknown`, which pauses the run for a person, so --full-port did not help. For a full port those
are weights the port needs, so listing them is information. The gate now accepts them when each entry carries the evidence, and still refuses an
entry that lacks it or was refused by the scan. Stage 2's port facts name such extra weight files."""
import json
import os
import struct
from pathlib import Path

import pytest

from orchard import port_facts, stages

GOOD = {"file": "loop_modules.safetensors", "size": 1234, "sha256": "ab" * 32, "num_tensors": 2, "dtypes": ["BF16"],
        "tensor_names": ["loop.mixer.weight", "loop.hc.weight"]}


def reasons(cls="full-port", path="full-port", sidecars=None):
    return stages._class_reasons({"class": cls, "path": path, "sidecars": [GOOD] if sidecars is None else sidecars})


def test_a_full_port_class_accepts_a_sidecar_with_size_sha256_and_tensor_names():
    assert not [r for r in reasons() if "sidecar" in r]


@pytest.mark.parametrize("missing", ["size", "sha256", "tensor_names"])
def test_a_full_port_sidecar_without_its_evidence_is_refused(missing):
    entry = {k: v for k, v in GOOD.items() if k != missing}
    text = " ".join(reasons(sidecars=[entry]))
    assert "loop_modules.safetensors" in text and "sidecar" in text


@pytest.mark.parametrize("bad", [{"size": -1}, {"size": True}, {"size": "9"}, {"tensor_names": "loop"}, {"tensor_names": [1, 2]},
                                 {"sha256": "short"}])
def test_a_full_port_sidecar_with_a_malformed_field_is_refused(bad):
    assert [r for r in reasons(sidecars=[{**GOOD, **bad}]) if "loop_modules.safetensors" in r]


def test_a_full_port_sidecar_the_scan_refused_is_still_refused_and_says_why():
    entry = {"file": "loop_modules.safetensors", "size": None, "sha256": None, "num_tensors": None, "dtypes": [], "tensor_names": [],
             "refused": "a link to a place outside the snapshot"}
    text = " ".join(reasons(sidecars=[entry]))
    assert "refused" in text and "outside the snapshot" in text


def test_a_class_that_is_not_a_sidecar_class_or_full_port_still_cannot_have_sidecars():
    for cls, path in (("weights-only", "weights-only"), ("vocab-trim", "full-port"), ("stock-runtime", "weights-only")):
        assert any("cannot have sidecars" in r for r in reasons(cls=cls, path=path)), cls


def test_a_full_port_with_no_sidecars_is_unchanged():
    assert reasons(sidecars=[]) == [] or not any("sidecar" in r for r in reasons(sidecars=[]))


# ---- stage 2's port facts name the extra weight files --------------------------------------------------------------------------------------

def _write_st(path, tensors):
    header, off = {}, 0
    for name, shape in tensors.items():
        n = 2
        for d in shape:
            n *= d
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * off)


def snapshot_with_a_second_weights_file(tmp_path, *, index=True):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4}))
    _write_st(snap / "model.safetensors", {"model.layers.0.w": (2, 2), "lm_head.weight": (2, 2)})
    _write_st(snap / "loop_modules.safetensors", {"loop.mixer.weight": (2, 2), "loop.hc.dynamic": (2, 2), "engram.table": (4, 2)})
    if index:
        (snap / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"model.layers.0.w": "model.safetensors", "lm_head.weight": "model.safetensors"}}))
    return snap


def test_the_port_facts_list_a_weights_file_the_index_does_not_name(tmp_path):
    facts = port_facts.build(snapshot_with_a_second_weights_file(tmp_path))
    extra = facts["weights"]["extra_files"]
    assert [e["file"] for e in extra] == ["loop_modules.safetensors"]
    assert extra[0]["tensors"] == 3 and set(extra[0]["prefixes"]) == {"loop", "engram"}
    assert "loop_modules.safetensors" in facts["weights"]["note"]


def test_a_snapshot_with_no_extra_file_has_an_empty_list(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "llama"}))
    _write_st(snap / "model.safetensors", {"model.layers.0.w": (2, 2)})
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.layers.0.w": "model.safetensors"}}))
    assert port_facts.build(snap)["weights"]["extra_files"] == []


def test_the_extra_files_section_stays_inside_the_facts_size_cap_with_many_files_and_prefixes(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "llama"}))
    _write_st(snap / "model.safetensors", {"model.layers.0.w": (2, 2)})
    for k in range(20):
        _write_st(snap / f"extra{k:02d}.safetensors", {f"prefix{k}x{j}.w": (2, 2) for j in range(30)})
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.layers.0.w": "model.safetensors"}}))
    facts = port_facts.build(snap)
    assert len(json.dumps(facts, indent=2)) + 1 <= port_facts.MAX_CHARS
    assert len(facts["weights"]["extra_files"]) <= port_facts.MAX_EXTRA_FILES
    assert all(len(e["prefixes"]) <= port_facts.MAX_EXTRA_PREFIXES for e in facts["weights"]["extra_files"])
