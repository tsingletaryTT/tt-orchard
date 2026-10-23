# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""prepare_swap.py with `graft_mtp`: a model without mtp.* tensors is served with the nearest model's.

BAAI/AREX-2 ships no mtp.* tensors, so stage 2 served it with the speculative drafter off and the host
sampling. A hand experiment added the nearest model's 15 mtp.* tensors as one more shard and the drafter
started. These tests pin what the option does and when it refuses. The snapshots here hold real safetensors
files written by the test, so the byte copy is checked against the source bytes."""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weights_swap_templates import (DRAFTER_LINE, NEAREST, prep, run_prepare,  # noqa: E402,F401
                                         with_drafter)

SHARD = "model-mtp-from-base.safetensors"
MTP = {"mtp.fc.weight": ("BF16", [4, 8]), "mtp.norm.weight": ("BF16", [4]),
       "mtp.layers.0.mlp.up_proj.weight": ("BF16", [6, 4])}
BACKBONE = {"model.embed_tokens.weight": ("BF16", [10, 4]), "lm_head.weight": ("BF16", [10, 4])}
SIZES = {"BF16": 2, "F32": 4}


def payload(name: str, nbytes: int) -> bytes:
    seed = hashlib.sha256(name.encode()).digest()
    return (seed * (nbytes // len(seed) + 1))[:nbytes]


def nbytes(dtype: str, shape: list[int]) -> int:
    n = SIZES[dtype]
    for d in shape:
        n *= d
    return n


def write_safetensors(path: Path, tensors: dict) -> None:
    """A real safetensors file: header, then each tensor's bytes in the order given."""
    header, blobs, off = {"__metadata__": {"format": "pt"}}, [], 0
    for name, (dtype, shape) in tensors.items():
        data = payload(name, nbytes(dtype, shape))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def tensor_bytes(path: Path, name: str) -> bytes:
    h = read_header(path)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        a, b = h[name]["data_offsets"]
        f.seek(8 + n + a)
        return f.read(b - a)


def put_link(snap: Path, name: str, content: bytes | str) -> Path:
    """A blob in the snapshot's blobs/ directory and a relative link to it, as the HF cache lays it out."""
    blobs = snap.parent.parent / "blobs"
    blob = blobs / f"blob-{name}"
    if blob.exists():
        blob.unlink()
    (blob.write_bytes if isinstance(content, bytes) else blob.write_text)(content)
    link = snap / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(blob, snap))
    return blob


def base_with_mtp(prep, shards: dict, hidden: int | None = None):
    """The nearest model's snapshot holds `shards` ({file: {tensor: (dtype, shape)}}) and an index."""
    base, weight_map = prep["base"], {}
    blobs = base.parent.parent / "blobs"
    for name, tensors in shards.items():
        path = blobs / f"blob-{name}"
        write_safetensors(path, tensors)
        link = base / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(os.path.relpath(path, base))
        weight_map.update({t: name for t in tensors})
    put_link(base, "model.safetensors.index.json", json.dumps({"metadata": {}, "weight_map": weight_map}))
    return base


def new_model(prep, *, keys=("model.embed_tokens.weight", "lm_head.weight"), hidden=None):
    """The new model's weight index (a real one) lists `keys`, and its config may name a hidden size."""
    put_link(prep["new"], "model.safetensors.index.json", json.dumps(
        {"metadata": {"total_size": 100}, "weight_map": {k: "model-00001-of-00002.safetensors" for k in keys}}))
    if hidden is not None:
        put_link(prep["new"], "config.json", json.dumps({"text_config": {"hidden_size": hidden}}))


def graft_on(prep, value=True):
    path = prep["stage"] / "swap_config.json"
    cfg = json.loads(path.read_text())
    cfg["graft_mtp"] = value
    path.write_text(json.dumps(cfg))


@pytest.fixture
def grafted(prep):
    """A base with its mtp.* tensors in the last shard beside lm_head, a new model with none, graft on."""
    base_with_mtp(prep, {"model-00001-of-00002.safetensors": {"model.embed_tokens.weight": BACKBONE["model.embed_tokens.weight"]},
                         "model-00002-of-00002.safetensors": {"lm_head.weight": BACKBONE["lm_head.weight"], **MTP}})
    new_model(prep)
    with_drafter(prep)
    graft_on(prep)
    return prep


def model_dir(prep) -> Path:
    return prep["stage"] / "model-dir"


def test_the_nearest_models_mtp_tensors_become_one_shard_in_the_model_dir(grafted):
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    shard = model_dir(grafted) / SHARD
    assert shard.is_file() and not shard.is_symlink()
    header = {k: v for k, v in read_header(shard).items() if k != "__metadata__"}
    assert sorted(header) == sorted(MTP)
    src = grafted["base"] / "model-00002-of-00002.safetensors"
    for name, (dtype, shape) in MTP.items():
        assert header[name]["dtype"] == dtype and header[name]["shape"] == shape
        assert tensor_bytes(shard, name) == tensor_bytes(src, name)


def test_lm_head_and_the_other_base_tensors_are_not_copied(grafted):
    run_prepare(grafted["stage"])
    names = {k for k in read_header(model_dir(grafted) / SHARD) if k != "__metadata__"}
    assert "lm_head.weight" not in names and "model.embed_tokens.weight" not in names


def test_the_index_in_the_model_dir_lists_the_grafted_tensors_and_keeps_the_models(grafted):
    run_prepare(grafted["stage"])
    index = model_dir(grafted) / "model.safetensors.index.json"
    assert index.is_file() and not index.is_symlink()
    data = json.loads(index.read_text())
    assert all(data["weight_map"][k] == SHARD for k in MTP)
    assert data["weight_map"]["lm_head.weight"] == "model-00001-of-00002.safetensors"
    assert data["metadata"]["total_size"] > 100        # the grafted bytes are counted


def test_the_new_models_cached_index_is_not_written_through(grafted):
    blob = (grafted["new"] / "model.safetensors.index.json").resolve()
    before = blob.read_bytes()
    run_prepare(grafted["stage"])
    assert blob.read_bytes() == before
    assert not any(k.startswith("mtp.") for k in json.loads(blob.read_text())["weight_map"])


def test_the_base_snapshot_is_left_as_it_was(grafted):
    blobs = grafted["base"].parent.parent / "blobs"
    before = {p.name: p.read_bytes() for p in blobs.iterdir()}
    run_prepare(grafted["stage"])
    assert {p.name: p.read_bytes() for p in blobs.iterdir()} == before


def test_the_drafter_and_on_device_sampling_stay_on(grafted):
    run = grafted["bundle"] / "run.sh"
    run.write_text(run.read_text().replace(
        "--max_num_seqs 4", "--max_num_seqs 4 --additional-config '{\"tt\": {\"sample_on_device_mode\": \"decode_only\"}}'"))
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    text = (grafted["stage"] / "run.sh").read_text()
    assert DRAFTER_LINE.strip() in text and "sample_on_device_mode" in text
    assert "grafted" in r.stdout


def test_without_the_option_nothing_is_grafted_and_the_drafter_is_cleared(grafted):
    graft_on(grafted, False)
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (model_dir(grafted) / SHARD).exists()
    assert 'export QWEN36_DRAFTER=""' in (grafted["stage"] / "run.sh").read_text()
    assert "mtp_graft" not in json.loads((grafted["stage"] / "model-dir-sources.json").read_text())


def test_with_no_graft_mtp_key_at_all_nothing_is_grafted(grafted):
    path = grafted["stage"] / "swap_config.json"
    cfg = json.loads(path.read_text())
    del cfg["graft_mtp"]
    path.write_text(json.dumps(cfg))
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (model_dir(grafted) / SHARD).exists()
    assert 'export QWEN36_DRAFTER=""' in (grafted["stage"] / "run.sh").read_text()


def test_the_sources_record_says_what_was_grafted_and_from_where(grafted):
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((grafted["stage"] / "model-dir-sources.json").read_text())["mtp_graft"]
    assert rec["from"] == NEAREST and rec["base_revision"] == "0" * 40
    assert rec["file"] == SHARD and rec["tensors"] == len(MTP) and rec["source_file"] == "model-00002-of-00002.safetensors"
    assert rec["sha256"] == hashlib.sha256((model_dir(grafted) / SHARD).read_bytes()).hexdigest()
    assert "/" not in json.dumps(rec).replace(NEAREST, "")           # no machine paths


def test_the_shard_is_the_same_bytes_every_time(grafted):
    run_prepare(grafted["stage"])
    first = (model_dir(grafted) / SHARD).read_bytes()
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0 and (model_dir(grafted) / SHARD).read_bytes() == first


def test_only_the_shard_that_holds_the_mtp_tensors_is_opened(grafted):
    # The first base shard is not a safetensors file at all; the graft must not need to read it.
    put_link(grafted["base"], "model-00001-of-00002.safetensors", "not a safetensors file")
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_container_package_gets_the_shard_too(grafted):
    path = grafted["stage"] / "swap_config.json"
    cfg = json.loads(path.read_text())
    del cfg["bundle_dir"]
    cfg["package"] = "episod/some-container"
    path.write_text(json.dumps(cfg))
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (model_dir(grafted) / SHARD).is_file()


def refused(prep, reason):
    r = run_prepare(prep["stage"])
    assert r.returncode == 2, r.stdout + r.stderr
    assert reason in r.stderr
    assert not (prep["stage"] / "model-dir").exists() and not (prep["stage"] / "run.sh").exists()


def test_a_model_that_already_has_mtp_tensors_is_refused(grafted):
    new_model(grafted, keys=("lm_head.weight", "mtp.fc.weight"))
    refused(grafted, "already has mtp.")


def test_a_nearest_model_without_mtp_tensors_is_refused(grafted):
    base_with_mtp(grafted, {"model-00002-of-00002.safetensors": dict(BACKBONE)})
    refused(grafted, "nearest model has no mtp.")


def test_a_tensor_that_is_not_bf16_is_refused_and_named(grafted):
    base_with_mtp(grafted, {"model-00002-of-00002.safetensors": {**BACKBONE, **MTP, "mtp.norm.weight": ("F32", [4])}})
    refused(grafted, "mtp.norm.weight is F32; the graft copies BF16 tensors only")


def test_tensors_spread_over_two_shards_are_refused(grafted):
    base_with_mtp(grafted, {"model-00001-of-00002.safetensors": {"mtp.fc.weight": MTP["mtp.fc.weight"]},
                            "model-00002-of-00002.safetensors": {"lm_head.weight": BACKBONE["lm_head.weight"],
                                                                 "mtp.norm.weight": MTP["mtp.norm.weight"]}})
    refused(grafted, "more than one shard")


def test_a_tensor_the_index_names_but_the_shard_lacks_is_refused(grafted):
    path = grafted["base"].parent.parent / "blobs" / "blob-model.safetensors.index.json"
    data = json.loads(path.read_text())
    data["weight_map"]["mtp.ghost.weight"] = "model-00002-of-00002.safetensors"
    path.write_text(json.dumps(data))
    refused(grafted, "mtp.ghost.weight")


def test_a_hidden_size_that_disagrees_with_the_head_is_refused(grafted):
    new_model(grafted, hidden=8)          # mtp.norm.weight has 4 elements
    refused(grafted, "hidden size")


def test_a_hidden_size_that_agrees_is_accepted(grafted):
    new_model(grafted, hidden=4)
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr


def test_when_the_models_weights_cannot_be_read_the_graft_is_refused(grafted):
    (grafted["new"] / "model.safetensors.index.json").unlink()
    # the fixture's shards are text, so their headers cannot be read either
    refused(grafted, "cannot tell")


@pytest.mark.parametrize("value", ["yes", 1, None, "true"])
def test_graft_mtp_must_be_true_or_false(grafted, value):
    graft_on(grafted, value)
    refused(grafted, "graft_mtp")


def test_a_shard_cut_short_is_refused(grafted):
    blob = grafted["base"].parent.parent / "blobs" / "blob-model-00002-of-00002.safetensors"
    blob.write_bytes(blob.read_bytes()[:-5])
    refused(grafted, "does not match the bytes")


def test_a_header_that_claims_the_wrong_size_for_a_tensor_is_refused(grafted):
    blob = grafted["base"].parent.parent / "blobs" / "blob-model-00002-of-00002.safetensors"
    raw = blob.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    header["mtp.norm.weight"]["shape"] = [5]           # 5 elements in 4 elements of bytes
    new = json.dumps(header).encode()
    new += b" " * (-len(new) % 8)
    blob.write_bytes(struct.pack("<Q", len(new)) + new + raw[8 + n:])
    refused(grafted, "mtp.norm.weight")


def test_a_new_model_with_no_weight_index_still_gets_the_shard(grafted):
    (grafted["new"] / "model.safetensors.index.json").unlink()
    write_safetensors(put_link(grafted["new"], "model-00001-of-00002.safetensors", b""), dict(BACKBONE))
    write_safetensors(put_link(grafted["new"], "model-00002-of-00002.safetensors", b""), {"model.x.weight": ("BF16", [2])})
    write_safetensors(put_link(grafted["new"], "model-mtp.safetensors", b""), {"model.y.weight": ("BF16", [2])})   # a name, not a tensor
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (model_dir(grafted) / SHARD).is_file()
    assert not (model_dir(grafted) / "model.safetensors.index.json").exists()


def test_a_nearest_model_with_no_index_is_read_from_its_shard_headers(grafted):
    (grafted["base"] / "model.safetensors.index.json").unlink()
    r = run_prepare(grafted["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    names = {k for k in read_header(model_dir(grafted) / SHARD) if k != "__metadata__"}
    assert names == set(MTP)


def test_the_skill_tells_the_agent_not_to_set_graft_mtp_and_that_the_supervisor_tries_it():
    skill = " ".join((Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-check.md").read_text().split())
    assert '"graft_mtp": true' in skill and "Do not set it in this stage" in skill
    assert "graft-decision.json" in skill and "do not touch `stages/2/graft-trial/`" in skill
