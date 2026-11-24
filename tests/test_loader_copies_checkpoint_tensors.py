# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 813: a generated port must copy each checkpoint tensor out of the memory-mapped file before it can reach `ttnn.from_torch`.

On the PyPI ttnn 0.80.0 wheel, `ttnn.from_torch` of a tensor that `safe_open(...).get_tensor(name)` returned (it is backed by the mapped file) never returns:
the main thread sits at 100 percent CPU and the server looks hung before it is ready (Aion-1.0-Instruct-2.3B, 2026-11-22; confirmed again on a chip on
2026-11-24, `.clone()` fixed it). These tests do not need a chip. They write a real safetensors file, read a tensor back through the loader text a port
carries, and ask the operating system (`/proc/self/maps`) whether the returned tensor's storage still lies inside the mapping of that file. A stand-in for
`from_torch` raises on such a tensor, the way the wheel hangs on it.

Linux only: `/proc/self/maps` is the check.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from test_port_template import std_generate

REPO = Path(__file__).resolve().parent.parent
SKELETON = REPO / "orchard" / "skills" / "full-port-templates" / "port_model_skeleton.py"

pytestmark = pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="the mapping check reads /proc/self/maps")


def file_backed(tensor: torch.Tensor, path: Path) -> bool:
    """True when the tensor's storage lies inside a mapping of `path` in this process."""
    ptr = tensor.untyped_storage().data_ptr()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].strip() == str(path):
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            if lo <= ptr < hi:
                return True
    return False


def from_torch_stand_in(path: Path):
    """What the wheel does with a file-backed tensor, as a test double: it does not return. Here it raises, so a test fails instead of hanging."""
    def from_torch(tensor, *args, **kwargs):
        if file_backed(tensor, path):
            raise AssertionError("ttnn.from_torch was given a tensor backed by the memory-mapped checkpoint (issue 813: the PyPI wheel hangs on it)")
        return ("uploaded", tuple(tensor.shape))
    return from_torch


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"model.embed_tokens.weight": torch.arange(0, 6000, dtype=torch.float32).reshape(100, 60).to(torch.bfloat16),
               "model.norm.weight": torch.ones(60, dtype=torch.bfloat16)}, str(path))
    return path


def generated_tensors_class():
    """The `_Tensors` class exactly as the generator writes it into a port, compiled on its own (the rest of the port needs ttnn)."""
    text, _ = std_generate()
    match = re.search(r"^class _Tensors:\n(?:(?:    .*|)\n)+", text, flags=re.M)
    assert match, "the generated port has no _Tensors class"
    namespace: dict = {}
    exec("import glob, os\nfrom safetensors import safe_open\n" + match.group(0), namespace)
    return namespace["_Tensors"]


def skeleton_module(monkeypatch):
    """The stage 2 skeleton (`port_model_skeleton.py`) imported with an empty stand-in for ttnn."""
    monkeypatch.setitem(sys.modules, "ttnn", types.ModuleType("ttnn"))
    spec = importlib.util.spec_from_file_location("skeleton_under_test", SKELETON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_mapping_check_sees_a_tensor_read_straight_from_the_file(checkpoint):
    """The check itself: a tensor from `get_tensor` is file-backed and a copy of it is not. Without this the tests below could pass on any reader."""
    from safetensors import safe_open
    with safe_open(str(checkpoint), framework="pt") as f:
        raw = f.get_tensor("model.norm.weight")
        copied = f.get_tensor("model.norm.weight").clone()
    assert file_backed(raw, checkpoint) and not file_backed(copied, checkpoint)


def test_the_generated_tensors_reader_returns_a_copy_that_is_not_backed_by_the_file(checkpoint):
    tensors = generated_tensors_class()(str(checkpoint.parent))
    got = tensors["model.embed_tokens.weight"]
    assert not file_backed(got, checkpoint)
    assert got.shape == (100, 60) and got.dtype == torch.bfloat16


def test_the_generated_reader_still_returns_the_checkpoint_values(checkpoint):
    tensors = generated_tensors_class()(str(checkpoint.parent))
    expected = torch.arange(0, 6000, dtype=torch.float32).reshape(100, 60).to(torch.bfloat16)
    assert torch.equal(tensors["model.embed_tokens.weight"], expected)


def test_an_upload_of_a_tensor_the_generated_reader_returned_does_not_reach_from_torch_backed_by_the_file(checkpoint):
    tensors = generated_tensors_class()(str(checkpoint.parent))
    from_torch = from_torch_stand_in(checkpoint)
    assert from_torch(tensors["model.embed_tokens.weight"])[0] == "uploaded"


def test_the_skeleton_read_tensors_returns_copies_that_are_not_backed_by_the_file(checkpoint, monkeypatch):
    out = skeleton_module(monkeypatch).read_tensors(str(checkpoint.parent))
    assert set(out) == {"model.embed_tokens.weight", "model.norm.weight"}
    for name, tensor in out.items():
        assert not file_backed(tensor, checkpoint), name


def test_the_skeleton_reads_one_tensor_per_request_and_none_when_it_is_built(checkpoint, monkeypatch):
    """Copying must not make the skeleton hold the whole checkpoint in host memory: a tensor is read, and copied, only when it is asked for."""
    module = skeleton_module(monkeypatch)
    reads = []
    real = module.safe_open

    class Counting:
        def __init__(self, path, framework):
            self._cm = real(path, framework=framework)

        def __enter__(self):
            handle = self._cm.__enter__()
            outer = handle

            class Handle:
                def keys(self_inner):
                    return outer.keys()

                def get_tensor(self_inner, name):
                    reads.append(name)
                    return outer.get_tensor(name)
            return Handle()

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    monkeypatch.setattr(module, "safe_open", Counting)
    out = module.read_tensors(str(checkpoint.parent))
    assert len(out) == 2 and "model.norm.weight" in out and reads == []
    out["model.norm.weight"]
    assert reads == ["model.norm.weight"]


def test_an_upload_of_a_tensor_the_skeleton_returned_does_not_reach_from_torch_backed_by_the_file(checkpoint, monkeypatch):
    out = skeleton_module(monkeypatch).read_tensors(str(checkpoint.parent))
    from_torch = from_torch_stand_in(checkpoint)
    assert all(from_torch(t)[0] == "uploaded" for t in out.values())


def test_the_upload_stand_in_does_fail_on_a_tensor_read_straight_from_the_file(checkpoint):
    """The stand-in has to be able to fail, or the two upload tests above prove nothing."""
    from safetensors import safe_open
    with safe_open(str(checkpoint), framework="pt") as f:
        raw = f.get_tensor("model.norm.weight")
    with pytest.raises(AssertionError, match="memory-mapped"):
        from_torch_stand_in(checkpoint)(raw)


# ---- the repository scan: no new loader may skip the copy ------------------------------------------------------------------------------------

# Every place under orchard/ that calls `.get_tensor(` and does not copy. Each is host-side only: its result never reaches `ttnn.from_torch`.
HOST_ONLY_READERS = {
    "orchard/skills/sidecar-parity-templates/hidden_parity.py": "load_tensor feeds the embedding table to host-side comparison arithmetic "
                                                               "(wiring_check), never to ttnn.from_torch",
}


def get_tensor_calls():
    """(relative path, line number, line) of every `.get_tensor(` call in a Python file under orchard/, including the lines of the loader text the
    generator writes out as string literals."""
    found = []
    for path in sorted((REPO / "orchard").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if ".get_tensor(" in line:
                found.append((str(path.relative_to(REPO)), number, line.strip()))
    return found


def test_every_get_tensor_call_under_orchard_copies_or_is_listed_as_host_only():
    uncopied = [(p, n, line) for p, n, line in get_tensor_calls() if ".clone()" not in line and p not in HOST_ONLY_READERS]
    assert not uncopied, f"a loader returns a memory-mapped tensor without .clone() (issue 813); copy it or list it in HOST_ONLY_READERS with the reason: {uncopied}"


def test_the_host_only_list_names_files_that_still_exist_and_still_call_get_tensor():
    """A stale entry would let a later loader in that file skip the copy."""
    calls = {p for p, _, _ in get_tensor_calls()}
    for path in HOST_ONLY_READERS:
        assert path in calls, f"{path} no longer calls .get_tensor(: remove its entry"


def test_the_scan_does_see_the_three_known_readers():
    """The scan must be able to fail: it has to find the generator's loader text, the skeleton and the parity template."""
    files = {p for p, _, _ in get_tensor_calls()}
    assert {"orchard/port_template.py", "orchard/skills/full-port-templates/port_model_skeleton.py",
            "orchard/skills/sidecar-parity-templates/hidden_parity.py"} <= files
