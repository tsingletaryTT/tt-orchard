# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 643: the hub-wide blob store (regression from PR 622, found on the Spark bring-up).

With current huggingface_hub a snapshot file is a link `<hub>/models--org--name/snapshots/<rev>/file -> ../../blobs/<hash>`, and that
`blobs/<hash>` entry is itself a link into a store shared by the whole hub: `<hub>/blobs/<xx>/<hash...>`. PR 622 allowed a link target only
inside the snapshot or the model's own blobs/, so every legitimate file in this layout was refused. The hub-wide store is an allowed root
only when the snapshot is `<hub>/models--*/snapshots/<rev>`, `<hub>/blobs` is a real directory and not a link, and the final target is a regular
file inside it. Everything else is still refused."""
import hashlib
import importlib.util
import json
import os
import struct
from pathlib import Path

import pytest

from orchard import residency, safeio, stages

DT = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
_spec = importlib.util.spec_from_file_location("delta_triage_hubstore_under_test", DT)
dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dt)

REV = "a" * 40


def safetensors_bytes(name="model.layers.0.mlp.gate_proj.weight"):
    header = json.dumps({name: {"dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 128]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * 128


def hub_layout(tmp_path, *, files=None, hub_blobs_is_link=False):
    """The faithful two-level layout. Returns (hub, snapshot). files: {name: bytes}, each stored once in <hub>/blobs/<xx>/<hash> and reached
    through <hub>/models--org--name/blobs/<hash> (a link) and the snapshot link."""
    hub = tmp_path / "hub"
    model = hub / "models--org--name"
    snap = model / "snapshots" / REV
    (model / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    store = hub / "blobs"
    if hub_blobs_is_link:
        real = tmp_path / "elsewhere-store"
        real.mkdir()
        os.symlink(real, store)
        store = real
    else:
        store.mkdir()
    for name, data in (files or {}).items():
        h = hashlib.sha256(data).hexdigest()
        (store / h[:2]).mkdir(exist_ok=True)
        (store / h[:2] / h).write_bytes(data)
        os.symlink(store / h[:2] / h, model / "blobs" / h)
        os.symlink(f"../../blobs/{h}", snap / name)
    return hub, snap


# ---- safeio.resolved_inside / cache_roots -----------------------------------------------------------------------------------------------

def test_a_snapshot_file_reached_through_the_hub_wide_store_is_accepted(tmp_path):
    hub, snap = hub_layout(tmp_path, files={"model.safetensors": safetensors_bytes()})
    got = safeio.resolved_inside(snap / "model.safetensors", snap)
    assert got is not None and got.startswith(os.path.realpath(hub / "blobs") + os.sep)


def test_a_symlinked_hub_blobs_directory_gives_no_hub_root(tmp_path):
    hub, snap = hub_layout(tmp_path, files={"model.safetensors": safetensors_bytes()}, hub_blobs_is_link=True)
    assert safeio.resolved_inside(snap / "model.safetensors", snap) is None


def test_a_models_blobs_entry_that_points_outside_every_root_is_refused(tmp_path):
    hub, snap = hub_layout(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x" * 10)
    os.symlink(outside, hub / "models--org--name" / "blobs" / "evil")
    os.symlink("../../blobs/evil", snap / "model.safetensors")
    assert safeio.resolved_inside(snap / "model.safetensors", snap) is None


def test_a_target_in_the_hub_store_that_is_a_pipe_is_refused(tmp_path):
    hub, snap = hub_layout(tmp_path)
    (hub / "blobs" / "ab").mkdir()
    os.mkfifo(hub / "blobs" / "ab" / "pipe")
    os.symlink(hub / "blobs" / "ab" / "pipe", hub / "models--org--name" / "blobs" / "pipe")
    os.symlink("../../blobs/pipe", snap / "model.safetensors")
    assert safeio.resolved_inside(snap / "model.safetensors", snap) is None


def test_a_snapshot_not_under_a_models_directory_gets_no_hub_root(tmp_path):
    base = tmp_path / "plain" / "snapshots" / REV                       # not models--*: the hub rule does not apply
    base.mkdir(parents=True)
    (tmp_path / "plain" / "blobs").mkdir()
    store = tmp_path / "blobs"
    store.mkdir()
    (store / "f").write_bytes(b"x")
    os.symlink(store / "f", base / "f.safetensors")
    assert safeio.resolved_inside(base / "f.safetensors", base) is None


def test_a_link_inside_the_snapshot_and_a_plain_file_are_still_accepted(tmp_path):
    hub, snap = hub_layout(tmp_path)
    (snap / "plain.safetensors").write_bytes(b"x")
    os.symlink(snap / "plain.safetensors", snap / "alias.safetensors")
    assert safeio.resolved_inside(snap / "plain.safetensors", snap) is not None
    assert safeio.resolved_inside(snap / "alias.safetensors", snap) is not None


# ---- residency.read_headers -------------------------------------------------------------------------------------------------------------

def test_residency_reads_a_snapshot_in_the_two_level_layout(tmp_path):
    hub, snap = hub_layout(tmp_path, files={"model.safetensors": safetensors_bytes()})
    assert residency.read_headers(snap).checkpoint_bytes == 128


def test_residency_refuses_the_same_layout_when_the_hub_store_is_a_link(tmp_path):
    hub, snap = hub_layout(tmp_path, files={"model.safetensors": safetensors_bytes()}, hub_blobs_is_link=True)
    with pytest.raises(residency.ResidencyError, match="leaves"):
        residency.read_headers(snap)


# ---- stage 0: the sidecar scan and the link roots -----------------------------------------------------------------------------------------

def _scan(snap, tmp_path):
    empty = tmp_path / "empty_base"
    empty.mkdir(exist_ok=True)
    return dt.compare_sidecars(snap, empty, set())


def test_the_sidecar_scan_lists_size_and_sha256_for_a_sidecar_in_the_two_level_layout(tmp_path):
    """The Spark case: model.safetensors is the backbone, loop_modules.safetensors a sidecar, both reached through the hub-wide store."""
    sidecar = safetensors_bytes("loop.mixer.weight")
    hub, snap = hub_layout(tmp_path, files={"model.safetensors": safetensors_bytes(), "loop_modules.safetensors": sidecar})
    entry = next(e for e in _scan(snap, tmp_path)["sidecars"] if e["file"] == "loop_modules.safetensors")
    assert entry["size"] == len(sidecar) and entry["sha256"] == hashlib.sha256(sidecar).hexdigest() and not entry.get("refused")


def test_a_gguf_in_the_two_level_layout_keeps_its_size(tmp_path):
    data = b"g" * 321
    hub, snap = hub_layout(tmp_path, files={"x.gguf": data})
    assert next(e for e in _scan(snap, tmp_path)["ignored_files"] if e["file"] == "x.gguf")["size"] == 321


def test_the_sidecar_scan_still_refuses_a_hostile_hub_store(tmp_path):
    sidecar = safetensors_bytes("loop.mixer.weight")
    hub, snap = hub_layout(tmp_path, files={"model.safetensors": safetensors_bytes(), "loop_modules.safetensors": sidecar}, hub_blobs_is_link=True)
    entry = next(e for e in _scan(snap, tmp_path)["sidecars"] if e["file"] == "loop_modules.safetensors")
    assert entry["sha256"] is None and entry["size"] is None and entry.get("refused")


def test_the_link_roots_add_the_hub_store_only_for_a_real_hub_blobs_directory(tmp_path):
    hub, snap = hub_layout(tmp_path)
    assert os.path.realpath(hub / "blobs") in {str(r) for r in dt._link_roots(snap)}
    (tmp_path / "second").mkdir()
    hub2, snap2 = hub_layout(tmp_path / "second", hub_blobs_is_link=True)
    assert os.path.realpath(hub2 / "blobs") not in {str(r) for r in dt._link_roots(snap2)}


# ---- the gate says what happened ---------------------------------------------------------------------------------------------------------

def test_the_gate_names_a_refused_sidecar_instead_of_asking_for_a_sha256():
    d = {"class": "weights+sidecar", "path": "weights-only",
         "sidecars": [{"file": "loop_modules.safetensors", "size": None, "sha256": None, "refused": "a link to a place outside the snapshot"}]}
    reasons = stages._class_reasons(d)
    text = " ".join(reasons)
    assert "loop_modules.safetensors" in text and "refused" in text and "needs a sha256" not in text


def test_the_gate_names_a_refused_ignored_file_instead_of_asking_for_a_size():
    d = {"ignored_files": [{"file": "x.gguf", "size": None, "reason": "converted copy", "refused": "a link to a place outside"}]}
    text = " ".join(stages._class_reasons(d))
    assert "x.gguf" in text and "refused" in text and "needs a size" not in text


def test_an_ordinary_missing_sha256_still_gets_the_old_message():
    text = " ".join(stages._class_reasons({"class": "weights+sidecar", "path": "weights-only", "sidecars": [{"file": "a.safetensors"}]}))
    assert "needs a sha256" in text


def test_the_sidecar_scan_refuses_a_pipe_in_the_hub_store_and_never_reads_it(tmp_path):
    hub, snap = hub_layout(tmp_path)
    (hub / "blobs" / "ab").mkdir()
    os.mkfifo(hub / "blobs" / "ab" / "pipe")
    os.symlink(hub / "blobs" / "ab" / "pipe", hub / "models--org--name" / "blobs" / "pipe")
    os.symlink("../../blobs/pipe", snap / "x.gguf")
    entry = next(e for e in _scan(snap, tmp_path)["ignored_files"] if e["file"] == "x.gguf")
    assert entry["size"] is None and entry["refused"]
