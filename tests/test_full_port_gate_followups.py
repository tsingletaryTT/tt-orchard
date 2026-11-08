# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 546, 548 and 549: follow-ups from the review of PR 543 on the full-port record gate (orchard/full_port.py).

546: the record's `config.path` must be the stage's own `port_check_config.json`, and the layer floor is read from that file, not from a path the
     record names. A record that kept the right hash but named another in-run config with a smaller layer count lowered the matmul floor.
548: a Hugging Face cache snapshot stores `config.json` as a link; the layer floor must still be read (the link is resolved before the safe read).
549: supplied weight evidence that is not an object with whole-number element counts is refused (a null `weights` stays allowed)."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from orchard.stages import gate_full_port
from test_ttnn_gate import good_ops

MANY = {"num_hidden_layers": 40}
FEW = {"num_hidden_layers": 1}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def reseal(stage_dir: Path, mutate) -> None:
    """Change the record with `mutate(record)`, then rewrite the output line and the marker so only that change is what the gate can see."""
    path = stage_dir / "evidence" / "port-check.json"
    rec = json.loads(path.read_text())
    mutate(rec)
    path.write_text(json.dumps(rec))
    out = stage_dir / "evidence" / "hw-test-output.txt"
    out.write_text(f"PORT_CHECK_SHA256 {sha(path.read_bytes())}\n")
    marker = json.loads((stage_dir / "test-result.json").read_text())
    marker["output"]["sha256"] = sha(out.read_bytes())
    (stage_dir / "test-result.json").write_text(json.dumps(marker))


def snapshot(tmp_path, name, config, *, as_link=False) -> Path:
    snap = tmp_path / name
    snap.mkdir()
    if as_link:
        blobs = tmp_path / (name + "_blobs")
        blobs.mkdir()
        (blobs / "abc").write_text(json.dumps(config))
        os.symlink(blobs / "abc", snap / "config.json")
    else:
        (snap / "config.json").write_text(json.dumps(config))
    return snap


def build_with(tmp_path, snap: Path, ops):
    from test_full_port import build
    stage_dir, run = build(tmp_path, record={"device_ops": ops})
    cfg = stage_dir / "port_check_config.json"
    cfg.write_text(json.dumps({"model_snapshot": str(snap)}))
    reseal(stage_dir, lambda rec: rec["config"].update(sha256=sha(cfg.read_bytes())))
    return stage_dir, run


# ---- 546 -------------------------------------------------------------------------------------------------------------------------------

def test_a_record_that_names_another_config_file_is_refused_and_the_stage_config_sets_the_floor(tmp_path):
    many, few = snapshot(tmp_path, "many", MANY), snapshot(tmp_path, "few", FEW)
    low = good_ops(floor=1, prefill_matmul_calls=1, decode_matmul_calls_max=1)
    stage_dir, run = build_with(tmp_path, many, low)
    other = stage_dir / "other_config.json"            # an in-run config naming a snapshot with one layer
    other.write_text(json.dumps({"model_snapshot": str(few)}))
    # The record keeps the stage config's hash but names the other file as its path.
    reseal(stage_dir, lambda rec: rec["config"].update(path="stages/2/other_config.json"))
    g = gate_full_port(stage_dir, run)
    assert not g.ok
    assert any("config path" in r and "port_check_config.json" in r for r in g.reasons), g.reasons
    assert any("below the 40" in r for r in g.reasons), g.reasons       # the floor comes from the stage's own config, not the named file


def test_the_stage_config_path_in_the_record_is_accepted_when_it_is_the_stage_config(tmp_path):
    many = snapshot(tmp_path, "many", MANY)
    stage_dir, run = build_with(tmp_path, many, good_ops(floor=40, prefill_matmul_calls=40, decode_matmul_calls_max=40))
    g = gate_full_port(stage_dir, run)
    assert g.ok, g.reasons


# ---- 548 -------------------------------------------------------------------------------------------------------------------------------

def test_a_symlinked_snapshot_config_still_gives_the_layer_floor(tmp_path):
    many = snapshot(tmp_path, "many", MANY, as_link=True)
    stage_dir, run = build_with(tmp_path, many, good_ops(floor=1, prefill_matmul_calls=1, decode_matmul_calls_max=1))
    g = gate_full_port(stage_dir, run)
    assert not g.ok and any("below the 40" in r for r in g.reasons), g.reasons


# ---- 549 -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("weights", [
    {"host_elements": "900", "device_elements": 100}, {"host_elements": -1, "device_elements": 100},
    {"host_elements": True, "device_elements": 100}, {"host_elements": 0, "device_elements": "5000"},
    {"host_elements": 0}, {"device_elements": 5000}, [], "x", 5, {"host_elements": 1.5, "device_elements": 2}])
def test_malformed_supplied_weight_evidence_is_refused(tmp_path, weights):
    from test_full_port import build
    stage_dir, run = build(tmp_path, record={"device_ops": good_ops(weights=weights)})
    g = gate_full_port(stage_dir, run)
    assert not g.ok and any("weights" in r and "whole-number" in r for r in g.reasons), g.reasons


def test_absent_weight_evidence_stays_allowed(tmp_path):
    from test_full_port import build
    stage_dir, run = build(tmp_path, record={"device_ops": good_ops(weights=None)})
    assert gate_full_port(stage_dir, run).ok
