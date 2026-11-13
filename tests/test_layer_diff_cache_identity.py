# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 630 (review of PR 627) and the issues folded into it (612, 613): the layer-diff reference cache in
orchard/skills/full-port-templates/stage_tools.py.

630: the cache key held only the prompt, the snapshot and the dtype. A mixed dense/MoE checkpoint whose reference was cached under the
     MoE-only construction kept getting those outputs after the loader learned the dense shim. The key now also holds a loader revision and
     the sha256 of the shim the loader uses, and the shim module is built from the bytes that were hashed.
612: the reference side's forward hook converted each layer output to float32 before any size check. It now checks the element count
     against a budget first.
613: the port archive's bytes stayed referenced through the whole reference computation after its arrays were copied out.

The tests run `inner` in this process with `reference_layers` replaced, so no model is loaded."""
import hashlib
import json
import sys

import numpy as np
import pytest

from test_stage_tools_last_check import load_tool
from test_stage_tools_layer_diff import N, PROMPT, build, reference_layers

DENSE = {"model_type": "qwen4_exp", "num_hidden_layers": 2, "num_experts": 4, "mlp_only_layers": [0]}       # layer 0 dense, layer 1 MoE
SHIM_SRC = "MARK = 'a'\n\ndef needs_dense_shim(raw):\n    return raw.get('model_type') == 'qwen4_exp' and bool(raw.get('mlp_only_layers'))\n"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def setup(tmp_path, monkeypatch, *, config=None, shim=SHIM_SRC):
    """A stage whose cached reference was written by the tool itself (a first call that computes), and a tool whose `reference_layers`
    records each call. Returns (tool, stage, calls)."""
    stage = build(tmp_path)
    (stage / "evidence" / "layer-reference.json").unlink()                    # start with no cache: the first call computes and stores one
    (stage / "evidence" / "layer-reference.npz").unlink()
    snap = tmp_path / "snap"
    if config is not None:
        (snap / "config.json").write_text(json.dumps(config))
    if shim is not None:
        (stage.parent / "1" / "qwen4_dense.py").write_text(shim)
    tool = load_tool()
    tool.HERE = stage
    calls = []

    def fake_reference_layers(snapshot, ids, dtype, construction=None):
        calls.append(construction)
        return reference_layers()

    monkeypatch.setattr(tool, "reference_layers", fake_reference_layers)
    return tool, stage, calls


def run_inner(tool, stage):
    port = stage / "evidence" / "layer-port.npz"
    return tool.inner(["0.05", "float32", sha(port.read_bytes()), str(N)])


def sidecar(stage):
    return json.loads((stage / "evidence" / "layer-reference.json").read_text())


# ---- 630 ----------------------------------------------------------------------------------------------------------------------------------

def test_the_sidecar_holds_the_loader_revision_and_the_shim_hash_of_a_dense_checkpoint(tmp_path, monkeypatch):
    tool, stage, calls = setup(tmp_path, monkeypatch, config=DENSE)
    assert run_inner(tool, stage) == 0 and len(calls) == 1
    meta = sidecar(stage)
    assert meta["loader_revision"] == tool.LOADER_REVISION
    assert meta["dense_shim_sha256"] == sha(SHIM_SRC.encode())
    assert run_inner(tool, stage) == 0 and len(calls) == 1                    # same inputs: the cache is used


def test_a_cache_written_before_the_loader_revision_existed_is_not_used(tmp_path, monkeypatch):
    """The case of the issue: a sidecar with only the prompt, the snapshot and the dtype, from the MoE-only construction."""
    tool, stage, calls = setup(tmp_path, monkeypatch, config=DENSE)
    run_inner(tool, stage)
    meta = sidecar(stage)
    for k in ("loader_revision", "dense_shim_sha256"):
        meta.pop(k)
    (stage / "evidence" / "layer-reference.json").write_text(json.dumps(meta))
    assert run_inner(tool, stage) == 0 and len(calls) == 2


def test_a_new_loader_revision_misses_the_cache(tmp_path, monkeypatch):
    tool, stage, calls = setup(tmp_path, monkeypatch)
    run_inner(tool, stage)
    monkeypatch.setattr(tool, "LOADER_REVISION", tool.LOADER_REVISION + 1)
    assert run_inner(tool, stage) == 0 and len(calls) == 2


def test_a_changed_shim_misses_the_cache_of_a_dense_checkpoint(tmp_path, monkeypatch):
    tool, stage, calls = setup(tmp_path, monkeypatch, config=DENSE)
    run_inner(tool, stage)
    (stage.parent / "1" / "qwen4_dense.py").write_text(SHIM_SRC.replace("'a'", "'b'"))
    assert run_inner(tool, stage) == 0 and len(calls) == 2
    assert sidecar(stage)["dense_shim_sha256"] == sha(SHIM_SRC.replace("'a'", "'b'").encode())


def test_the_shim_does_not_enter_the_key_of_a_checkpoint_that_does_not_use_it(tmp_path, monkeypatch):
    """A Llama snapshot never loads through the shim, so an edit of the shim keeps its cache."""
    tool, stage, calls = setup(tmp_path, monkeypatch, config={"model_type": "llama", "num_hidden_layers": 2})
    run_inner(tool, stage)
    assert sidecar(stage)["dense_shim_sha256"] is None
    (stage.parent / "1" / "qwen4_dense.py").write_text(SHIM_SRC.replace("'a'", "'b'"))
    assert run_inner(tool, stage) == 0 and len(calls) == 1


def test_the_reference_is_computed_with_the_construction_the_key_names(tmp_path, monkeypatch):
    """The key and the load use one selection: the shim module handed to reference_layers has the hash that went into the sidecar."""
    tool, stage, calls = setup(tmp_path, monkeypatch, config=DENSE)
    run_inner(tool, stage)
    shim, raw, identity = calls[0]
    assert shim is not None and shim.MARK == "a" and raw == DENSE
    assert identity == {k: sidecar(stage)[k] for k in identity}


def test_the_shim_module_is_built_from_the_bytes_that_were_hashed(tmp_path, monkeypatch):
    """The file on disk says MARK = 'b'; the read the hash came from returned MARK = 'a'. The module and the digest must both be 'a'."""
    tool, stage, _ = setup(tmp_path, monkeypatch, config=DENSE, shim=SHIM_SRC.replace("'a'", "'b'"))
    hashed = SHIM_SRC.encode()
    real = tool.read_snapshot
    monkeypatch.setattr(tool, "read_snapshot",
                        lambda path, cap, dir_fd=None: (hashed, sha(hashed), None) if path.name == "qwen4_dense.py" else real(path, cap, dir_fd))
    shim, raw, identity = tool.reference_construction(str(tmp_path / "snap"))
    assert shim.MARK == "a" and identity["dense_shim_sha256"] == sha(hashed)


def test_a_link_in_place_of_the_stage_1_shim_is_a_reason_and_not_a_fallback(tmp_path, monkeypatch):
    """A shim that is present but cannot be read safely stops the reference. Falling through to the next candidate would load a shim that
    stage 1 did not use."""
    tool, stage, calls = setup(tmp_path, monkeypatch, config=DENSE, shim=None)
    real = tmp_path / "elsewhere.py"
    real.write_text(SHIM_SRC)
    (stage.parent / "1" / "qwen4_dense.py").symlink_to(real)
    assert run_inner(tool, stage) == 2 and calls == []


def test_a_run_with_no_shim_anywhere_has_no_shim_hash(tmp_path, monkeypatch):
    tool, stage, calls = setup(tmp_path, monkeypatch, config=DENSE, shim=None)
    monkeypatch.setattr(tool, "dense_shim_paths", lambda: [stage.parent / "1" / "qwen4_dense.py"])
    assert run_inner(tool, stage) == 0 and sidecar(stage)["dense_shim_sha256"] is None


# ---- 612 ----------------------------------------------------------------------------------------------------------------------------------

class HugeOutput:
    """Looks like a torch layer output to the hook: `numel` and `detach`. Converting it would allocate gigabytes, so `.float()` fails the test."""

    def __init__(self, numel):
        self._numel = numel

    def numel(self):
        return self._numel

    def detach(self):
        return self

    def float(self):
        raise AssertionError("the layer output was converted to float32 before its size was checked")


class SmallOutput(HugeOutput):
    def float(self):
        import torch
        return torch.zeros(self._numel)


class FakeLayer:
    def __init__(self, out):
        self.out, self.hooks = out, []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        layer = self

        class Handle:
            def remove(self):
                layer.hooks.remove(fn)
        return Handle()


class FakeModel:
    def __init__(self, outs):
        self.layers = [FakeLayer(o) for o in outs]

    def eval(self):
        return self

    def __call__(self, **_kw):
        for layer in self.layers:
            for fn in list(layer.hooks):
                fn(layer, (), (layer.out,))


def fake_load(tool, monkeypatch, outs):
    pytest.importorskip("torch")
    monkeypatch.setattr(tool, "load_reference_model", lambda snapshot, dtype, construction=None: FakeModel(outs))


def test_a_reference_layer_over_the_element_budget_is_refused_before_it_is_converted(monkeypatch):
    tool = load_tool()
    fake_load(tool, monkeypatch, [HugeOutput(tool.REF_MAX_NUMEL + 1)])
    with pytest.raises(ValueError, match="more than"):
        tool.reference_layers("snap", [1, 2], "float32")


def test_the_reference_budget_counts_what_earlier_layers_used(monkeypatch):
    tool = load_tool()
    fake_load(tool, monkeypatch, [SmallOutput(10), HugeOutput(tool.REF_MAX_NUMEL)])
    with pytest.raises(ValueError, match="more than"):
        tool.reference_layers("snap", [1, 2], "float32")


def test_reference_layers_under_the_budget_are_returned(monkeypatch):
    tool = load_tool()
    fake_load(tool, monkeypatch, [SmallOutput(10), SmallOutput(6)])
    out = tool.reference_layers("snap", [1, 2], "float32")
    assert [a.size for a in out] == [10, 6] and all(a.dtype == np.float32 for a in out)


# ---- 613 ----------------------------------------------------------------------------------------------------------------------------------

def test_the_port_archive_bytes_are_released_before_the_reference_is_computed(tmp_path, monkeypatch):
    """`inner` copies the port's arrays out of the archive bytes. The frame that calls reference_layers must no longer hold those bytes."""
    tool, stage, _ = setup(tmp_path, monkeypatch)
    seen = []

    def inspecting_reference_layers(snapshot, ids, dtype, construction=None):
        caller = sys._getframe(1).f_locals
        seen.append([k for k, v in caller.items() if isinstance(v, (bytes, bytearray)) and len(v) > 1000])
        return reference_layers()

    monkeypatch.setattr(tool, "reference_layers", inspecting_reference_layers)
    assert run_inner(tool, stage) == 0
    assert seen == [[]]
