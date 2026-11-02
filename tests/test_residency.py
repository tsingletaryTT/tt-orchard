# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The residency estimator (orchard/residency.py, issue 327): header reading, categories, the quantization policy, the chip table, the first-boot
needs, the stage 0 record and the command. Headers are built from model configs (tests/residency_headers.py); nothing touches a network."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from orchard import defaults, residency as rs
from residency_headers import dense_tensors, flash_next_tensors, header_bytes, write_snapshot

FIX = Path(__file__).parent / "fixtures" / "model-configs"
GIB = 1 << 30


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The hub call is replaced by one that fails the test: only a test that injects a fetch may use a repo id."""
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(huggingface_hub, "get_safetensors_metadata", boom)


@pytest.fixture(scope="module")
def flash_dir(tmp_path_factory):
    cfg = json.loads((FIX / "qwen3.8-flash-next.json").read_text())
    return write_snapshot(tmp_path_factory.mktemp("flash"), flash_next_tensors(cfg), shards=7, config=cfg)


@pytest.fixture(scope="module")
def dense_dir(tmp_path_factory):
    cfg = json.loads((FIX / "qwen3.8-27b.json").read_text())
    return write_snapshot(tmp_path_factory.mktemp("dense"), dense_tensors(cfg), shards=4, config=cfg)


@pytest.fixture(scope="module")
def flash(flash_dir):
    return rs.estimate(rs.read_headers(flash_dir))


@pytest.fixture(scope="module")
def dense(dense_dir):
    return rs.estimate(rs.read_headers(dense_dir))


def row(rec, chips):
    return next(r for r in rec["estimated"]["rows"] if r["chips"] == chips)


# ---- the Flash-Next numbers from the study -----------------------------------------------------------------------------------

def test_flash_next_bytes_by_category_match_the_study_table(flash):
    by = flash["measured"]["by_category"]
    assert by["routed_experts"]["bytes"] == 48 * 512 * (1280 * 2560 + 2560 * 640) * 2          # 241.59 GB, 96 fused tensors
    assert by["routed_experts"]["tensors"] == 96
    assert round(by["routed_experts"]["bytes"] / 1e9, 2) == 241.59
    assert round(by["ngram_ple"]["bytes"] / 1e9, 1) == 102.4 and by["ngram_ple"]["tensors"] >= 128
    assert by["embedding"]["bytes"] == 2 * 248320 * 2560 * 2 and by["embedding"]["tensors"] == 2           # 2.54 GB
    assert by["mtp"]["bytes"] > 5e9                      # the MTP layer's own experts are counted under MTP, not under routed experts
    assert 0.5e9 < by["vision"]["bytes"] < 1.3e9
    assert by["gdn"]["bytes"] > 0 and by["attention"]["bytes"] > 0 and by["shared_experts"]["bytes"] > 0


def test_every_tensor_lands_in_exactly_one_category(flash, flash_dir):
    headers = rs.read_headers(flash_dir)
    assert sum(c["bytes"] for c in flash["measured"]["by_category"].values()) == flash["measured"]["checkpoint_bytes"]
    assert sum(c["tensors"] for c in flash["measured"]["by_category"].values()) == flash["measured"]["tensors"] == len(headers.tensors)


def test_flash_next_bf4_experts_are_about_68_gb_and_need_four_chips(flash):
    est = flash["estimated"]
    assert est["by_category"]["routed_experts"]["stored_bytes"] == 241_591_910_400 // 2 * 9 // 16      # bfp4: 0.5625 bytes a value
    assert round(est["by_category"]["routed_experts"]["stored_bytes"] / 1e9, 1) == 67.9               # "about 68 GB" in the study
    assert [r["fits"] for r in (row(flash, 1), row(flash, 2), row(flash, 4), row(flash, 8))] == [False, False, True, True]
    assert est["chips_needed"] == 3 and est["smallest_mesh"] == 4


def test_the_ngram_table_is_host_resident_and_is_not_on_any_chip(flash):
    est = flash["estimated"]
    assert est["policy"]["ngram_ple"] == "host" and est["by_category"]["ngram_ple"]["placement"] == "host"
    assert est["host_resident_categories"] == ["ngram_ple"]
    assert est["host_bytes"] == flash["measured"]["by_category"]["ngram_ple"]["bytes"]
    on_chip = sum(c["stored_bytes"] for k, c in est["by_category"].items() if c["placement"] == "device")
    assert est["device_bytes"] == on_chip and est["device_bytes"] < 80e9


def test_dense_27b_fits_two_chips_and_not_one(dense):
    assert dense["measured"]["by_category"]["dense_mlp"]["bytes"] > dense["measured"]["by_category"]["attention"]["bytes"]
    assert "routed_experts" not in dense["measured"]["by_category"] and "ngram_ple" not in dense["measured"]["by_category"]
    assert round(dense["measured"]["checkpoint_bytes"] / 1e9) in range(50, 58)           # about 27B parameters in bf16
    assert [row(dense, n)["fits"] for n in (1, 2, 4, 8)] == [False, True, True, True]
    assert dense["estimated"]["chips_needed"] == 2 and dense["estimated"]["host_bytes"] == 0


# ---- categories ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,shape,want", [
    ("model.language_model.embed_tokens.weight", [10, 4], "embedding"), ("lm_head.weight", [10, 4], "embedding"),
    ("model.layers.3.self_attn.q_proj.weight", [4, 4], "attention"), ("model.layers.3.self_attn.indexer.index_qk_proj.weight", [4, 4], "attention"),
    ("model.layers.3.linear_attn.in_proj_qkv.weight", [4, 4], "gdn"), ("model.layers.3.mamba.in_proj.weight", [4, 4], "gdn"),
    ("model.layers.3.mlp.experts.gate_up_proj", [2, 4, 4], "routed_experts"),
    ("model.layers.3.block_sparse_moe.experts.7.w1.weight", [4, 4], "routed_experts"),
    ("model.layers.3.mlp.shared_expert.up_proj.weight", [4, 4], "shared_experts"),
    ("model.layers.3.mlp.shared_experts.up_proj.weight", [4, 4], "shared_experts"),
    ("model.layers.3.mlp.gate_proj.weight", [4, 4], "dense_mlp"), ("model.layers.3.feed_forward.w1.weight", [4, 4], "dense_mlp"),
    ("model.layers.3.mlp.gate.weight", [8, 4], "norms_other"),                         # the router
    ("model.layers.1.ple.ngram_embeddings.5.weight", [4, 4], "ngram_ple"), ("model.embed_tokens_per_layer.weight", [4, 4], "ngram_ple"),
    ("mtp.layers.0.mlp.experts.down_proj", [2, 4, 4], "mtp"), ("model.mtp.fc.weight", [4, 4], "mtp"),
    ("model.visual.blocks.0.attn.qkv.weight", [4, 4], "vision"), ("vision_tower.encoder.layers.0.self_attn.q_proj.weight", [4, 4], "vision"),
    ("model.layers.3.input_layernorm.weight", [4], "norms_other"), ("model.layers.3.self_attn.q_norm.weight", [4], "norms_other"),
    ("model.layers.3.hc_attn.weight", [4, 4], "norms_other"),
])
def test_category_by_name(name, shape, want):
    assert rs.categorize(name, shape) == want


# ---- the quantization policy --------------------------------------------------------------------------------------------------

def tiny(tmp_path, tensors, **kw):
    return rs.estimate(rs.read_headers(write_snapshot(tmp_path / "t", tensors, shards=1)), **kw)


def test_the_policy_scales_two_dimensional_float_tensors_by_the_block_format_width(tmp_path):
    t = [("model.layers.0.mlp.gate_proj.weight", "BF16", [1024, 1024]), ("model.layers.0.input_layernorm.weight", "BF16", [1024]),
         ("model.layers.0.mlp.ids", "I64", [1024, 8])]
    rec = tiny(tmp_path, t, policy={"dense_mlp": "bfp8"})
    dm = rec["estimated"]["by_category"]["dense_mlp"]
    assert dm["stored_bytes"] == int(1024 * 1024 * 1.0625) + 1024 * 8 * 8        # the I64 tensor is not a weight: it keeps its source bytes
    assert rec["estimated"]["by_category"]["norms_other"]["stored_bytes"] == 2048    # a one-dimensional tensor stays as stored
    assert rec["estimated"]["policy"]["dense_mlp"] == "bfp8"


def test_a_one_dimensional_tensor_keeps_its_bytes_even_when_the_caller_quantizes_its_category(tmp_path):
    t = [("model.layers.0.input_layernorm.weight", "BF16", [1024]), ("model.layers.0.mlp.gate_proj.weight", "BF16", [64, 64])]
    rec = tiny(tmp_path, t, policy={"norms_other": "bfp4"})
    assert rec["estimated"]["by_category"]["norms_other"]["stored_bytes"] == 2048


def test_a_host_policy_moves_a_category_off_the_chips_and_into_host_ram(tmp_path):
    t = [("model.layers.0.mlp.gate_proj.weight", "BF16", [1024, 1024]), ("model.layers.0.self_attn.q_proj.weight", "BF16", [1024, 1024])]
    rec = tiny(tmp_path, t, policy={"attention": "host"})
    assert rec["estimated"]["device_bytes"] == int(1024 * 1024 * 1.0625)
    assert rec["estimated"]["host_bytes"] == 1024 * 1024 * 2 and rec["estimated"]["host_resident_categories"] == ["attention"]


def test_a_policy_names_only_known_categories_and_dtypes(tmp_path):
    t = [("model.layers.0.mlp.gate_proj.weight", "BF16", [64, 64])]
    for bad in ({"nonsense": "bfp8"}, {"dense_mlp": "fp7"}):
        with pytest.raises(rs.ResidencyError):
            tiny(tmp_path, t, policy=bad)


def test_the_default_policy_is_labeled_a_choice_and_a_caller_policy_is_labeled_the_caller(flash, tmp_path):
    p = {a["name"]: a for a in flash["assumptions"]}
    assert p["policy"]["source"] == "default (choice)"
    t = [("model.layers.0.mlp.gate_proj.weight", "BF16", [64, 64])]
    assert {a["name"]: a for a in tiny(tmp_path, t, policy={"dense_mlp": "bf16"})["assumptions"]}["policy"]["source"] == "caller"


# ---- chips, DRAM, KV and the reserve ------------------------------------------------------------------------------------------

def test_dram_reserve_and_kv_are_inputs_and_the_output_labels_each_one(flash):
    a = {x["name"]: x for x in flash["assumptions"]}
    assert a["dram_gib_per_chip"]["value"] == defaults.CHIP_DRAM_GIB and a["dram_gib_per_chip"]["source"] == "default (choice)"
    assert a["reserve_gib_per_chip"]["value"] == defaults.CHIP_RESERVE_GIB and a["kv_budget_gib"]["value"] == defaults.KV_BUDGET_GIB
    assert all("choice" in a[k]["source"] for k in ("dram_gib_per_chip", "reserve_gib_per_chip", "kv_budget_gib"))
    assert any("allocator" in s for s in flash["not_measured"])


def test_a_stated_dram_and_kv_budget_change_the_answer_and_are_recorded_as_the_caller_s(flash_dir):
    h = rs.read_headers(flash_dir)
    big = rs.estimate(h, dram_gib=96, kv_gib=0, reserve_gib=0)
    assert big["estimated"]["chips_needed"] == 1 and big["estimated"]["smallest_mesh"] == 1
    a = {x["name"]: x for x in big["assumptions"]}
    assert a["dram_gib_per_chip"] == {**a["dram_gib_per_chip"], "value": 96.0, "source": "caller"}
    assert a["kv_budget_gib"]["source"] == "caller" and a["reserve_gib_per_chip"]["source"] == "caller"
    small = rs.estimate(h, dram_gib=16)
    assert small["estimated"]["chips_needed"] > rs.estimate(h)["estimated"]["chips_needed"]


def test_the_kv_budget_and_reserve_count_against_each_chip(flash_dir):
    h = rs.read_headers(flash_dir)
    rec = rs.estimate(h, chips=[4])
    r4 = row(rec, 4)
    dev = rec["estimated"]["device_bytes"]
    expect_used = dev / 4 + defaults.CHIP_RESERVE_GIB * GIB + defaults.KV_BUDGET_GIB * GIB / 4
    assert abs(r4["used_bytes_per_chip"] - expect_used) < 2
    assert r4["free_bytes_per_chip"] == pytest.approx(defaults.CHIP_DRAM_GIB * GIB - expect_used, abs=2)
    only_kv = rs.estimate(h, kv_gib=0, reserve_gib=0)
    assert row(only_kv, 4)["used_bytes_per_chip"] == pytest.approx(dev / 4, abs=2)


def test_the_table_has_the_rows_asked_for_and_the_box(flash_dir):
    rec = rs.estimate(rs.read_headers(flash_dir), chips=[3], box_chips=6)
    assert [r["chips"] for r in rec["estimated"]["rows"]] == [1, 2, 3, 4, 6, 8]


def test_reserve_at_or_above_dram_is_refused(flash_dir):
    with pytest.raises(rs.ResidencyError, match="reserve"):
        rs.estimate(rs.read_headers(flash_dir), dram_gib=4, reserve_gib=4)


@pytest.mark.parametrize("kw", [{"dram_gib": 0}, {"dram_gib": float("nan")}, {"kv_gib": -1}, {"reserve_gib": -1}, {"chips": [0]}, {"chips": [True]},
                                {"box_chips": 0}, {"dram_gib": "32"}])
def test_bad_numbers_are_refused(flash_dir, kw):
    with pytest.raises(rs.ResidencyError):
        rs.estimate(rs.read_headers(flash_dir), **kw)


# ---- labels ------------------------------------------------------------------------------------------------------------------

def test_the_record_separates_measured_figures_from_estimated_ones_and_names_what_is_missing(flash):
    assert flash["label"] == "estimate" and set(flash["labels"]) == {"measured", "estimated"}
    assert "read from the safetensors headers" in flash["labels"]["measured"]
    assert "arithmetic" in flash["labels"]["estimated"]
    assert set(flash["measured"]) == {"files", "tensors", "checkpoint_bytes", "by_category"}
    # every byte count outside `measured` is under `estimated`, and the not-measured list says what would replace them
    assert any("allocator" in s for s in flash["not_measured"]) and any("first boot" in s or "first-boot" in s for s in flash["not_measured"])


# ---- first boot ---------------------------------------------------------------------------------------------------------------

def test_first_boot_needs_disk_for_the_checkpoint_and_the_converted_cache_and_ram_for_the_host_table(flash):
    fb = flash["estimated"]["first_boot"]
    assert fb["cache_bytes"] == flash["estimated"]["device_bytes"]
    assert fb["checkpoint_bytes_to_download"] == 0 and fb["snapshot_local"] is True               # the snapshot is on disk already
    assert fb["disk_bytes"] == fb["cache_bytes"]
    assert fb["host_ram_bytes"] == flash["estimated"]["host_bytes"] + fb["conversion_in_flight_bytes"]
    # two copies of the largest layer's routed experts: about 10 GB, the figure the cards give
    assert 9e9 < fb["conversion_in_flight_bytes"] < 11e9


def test_a_repo_id_needs_the_download_as_well(flash_dir):
    cfg = json.loads((FIX / "qwen3.8-flash-next.json").read_text())
    h = rs.headers_from_metadata(_hub_metadata(flash_next_tensors(cfg)), source="hub:Qwen/Example")
    fb = rs.estimate(h)["estimated"]["first_boot"]
    assert fb["snapshot_local"] is False and fb["checkpoint_bytes_to_download"] == h.checkpoint_bytes
    assert fb["disk_bytes"] == fb["cache_bytes"] + h.checkpoint_bytes


# ---- reading headers ----------------------------------------------------------------------------------------------------------

def test_only_the_header_is_read_never_the_tensor_data(tmp_path, monkeypatch):
    d = write_snapshot(tmp_path / "s", [("model.layers.0.mlp.gate_proj.weight", "BF16", [64, 64])], shards=1)
    reads = []
    real = os.read
    monkeypatch.setattr(os, "read", lambda fd, n: (reads.append(n), real(fd, n))[1])
    h = rs.read_headers(d)
    assert h.checkpoint_bytes == 64 * 64 * 2 and all(n < 10_000 for n in reads)         # the file holds no data at all: a data read would come back short


def test_a_snapshot_link_into_blobs_is_followed_like_a_hugging_face_cache(tmp_path):
    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir()
    snap.mkdir()
    (blobs / "abc").write_bytes(header_bytes([("model.layers.0.mlp.gate_proj.weight", "BF16", [8, 8])]))
    (snap / "model.safetensors").symlink_to(blobs / "abc")
    assert rs.read_headers(snap).checkpoint_bytes == 128


def _write(path, header: dict | bytes, declared: int | None = None):
    body = header if isinstance(header, bytes) else json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(body) if declared is None else declared) + body)


def test_refusals_name_the_file_and_the_problem(tmp_path):
    ok = {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}
    # (header, declared length, the words the message must hold). Each case is meant to be stopped by one guard, so the words name it.
    cases = {
        "a header longer than the cap": ({"w": ok}, rs.MAX_HEADER_BYTES + 1, "cap"),
        "a header longer than the file": ({"w": ok}, 10_000, "longer than the file"),
        "a dtype the estimator has no width for": ({"w": {**ok, "dtype": "F7"}}, None, "dtype"),
        "a negative dimension": ({"w": {**ok, "shape": [-4, -4]}}, None, "short list"),         # the product is 16, so only the dimension check stops it
        "a boolean dimension": ({"w": {**ok, "shape": [True, 16], "data_offsets": [0, 32]}}, None, "short list"),
        "offsets that disagree with the shape": ({"w": {**ok, "data_offsets": [0, 31]}}, None, "disagree"),
        "offsets out of order": ({"w": {**ok, "data_offsets": [32, 0]}}, None, "ordered pair"),
        "a tensor entry that is not an object": ({"w": 5}, None, "not an object"),
        "a header that is not an object": (b"[1, 2]", None, "usable JSON"),
        "a header that is not JSON": (b"{nope", None, "usable JSON"),
        "NaN in the header": (b'{"w": {"dtype": "BF16", "shape": [NaN], "data_offsets": [0, 0]}}', None, "usable JSON"),
    }
    for why, (header, declared, words) in cases.items():
        d = tmp_path / why.replace(" ", "_")
        d.mkdir()
        _write(d / "model.safetensors", header, declared)
        with pytest.raises(rs.ResidencyError) as exc:
            rs.read_headers(d)
        assert "model.safetensors" in str(exc.value) and words in str(exc.value), (why, str(exc.value))


def test_a_header_length_larger_than_the_cap_is_refused_before_it_is_read(tmp_path, monkeypatch):
    d = tmp_path / "big"
    d.mkdir()
    (d / "m.safetensors").write_bytes(struct.pack("<Q", 2 ** 60) + b"{}")
    reads = []
    real = os.read
    monkeypatch.setattr(os, "read", lambda fd, n: (reads.append(n), real(fd, n))[1])
    with pytest.raises(rs.ResidencyError, match="header"):
        rs.read_headers(d)
    assert max(reads) < 1000                                    # no read asked for the declared length


def test_a_tensor_name_in_two_shards_and_an_empty_directory_are_refused(tmp_path):
    t = [("model.layers.0.mlp.gate_proj.weight", "BF16", [4, 4])]
    d = tmp_path / "dup"
    d.mkdir()
    for n in (1, 2):
        (d / f"m{n}.safetensors").write_bytes(header_bytes(t))
    with pytest.raises(rs.ResidencyError, match="twice"):
        rs.read_headers(d)
    e = tmp_path / "empty"
    e.mkdir()
    with pytest.raises(rs.ResidencyError, match="no .safetensors"):
        rs.read_headers(e)
    with pytest.raises(rs.ResidencyError):
        rs.read_headers(tmp_path / "missing")


def test_a_fifo_shard_is_refused_without_blocking(tmp_path):
    d = tmp_path / "fifo"
    d.mkdir()
    os.mkfifo(d / "m.safetensors")
    with pytest.raises(rs.ResidencyError, match="not a regular file"):
        rs.read_headers(d)


def test_a_read_error_becomes_a_residency_error(tmp_path, monkeypatch):
    d = write_snapshot(tmp_path / "s", [("model.layers.0.mlp.gate_proj.weight", "BF16", [4, 4])], shards=1)

    def fail(fd, n):
        raise OSError(5, "input/output error")
    monkeypatch.setattr(os, "read", fail)
    with pytest.raises(rs.ResidencyError, match="cannot be read"):
        rs.read_headers(d)


def test_a_directory_named_like_a_shard_is_refused(tmp_path):
    d = tmp_path / "dirshard"
    (d / "m.safetensors").mkdir(parents=True)
    with pytest.raises(rs.ResidencyError, match="not a regular file"):
        rs.read_headers(d)


def test_too_many_shards_or_tensors_are_refused(tmp_path, monkeypatch):
    t = [(f"model.layers.0.mlp.w{i}.weight", "BF16", [2, 2]) for i in range(5)]
    d = write_snapshot(tmp_path / "many", t, shards=1)
    monkeypatch.setattr(rs, "MAX_TENSORS", 4)
    with pytest.raises(rs.ResidencyError, match="tensors"):
        rs.read_headers(d)
    monkeypatch.undo()
    monkeypatch.setattr(rs, "MAX_FILES", 1)
    d2 = write_snapshot(tmp_path / "files", t, shards=2)
    with pytest.raises(rs.ResidencyError, match="files"):
        rs.read_headers(d2)


# ---- the hub ------------------------------------------------------------------------------------------------------------------

def _hub_metadata(tensors):
    from huggingface_hub.utils import SafetensorsFileMetadata, SafetensorsRepoMetadata, TensorInfo
    infos, offset = {}, 0
    for name, dtype, shape in tensors:
        n = 1
        for d in shape:
            n *= d
        size = n * {"BF16": 2, "F32": 4, "I64": 8}[dtype]
        infos[name] = TensorInfo(dtype=dtype, shape=list(shape), data_offsets=(offset, offset + size))
        offset += size
    return SafetensorsRepoMetadata(metadata=None, sharded=False, weight_map={}, files_metadata={
        "model.safetensors": SafetensorsFileMetadata(metadata={}, tensors=infos)})


def test_a_repo_id_goes_through_the_hub_metadata_call_and_gives_the_same_numbers(flash, flash_dir):
    cfg = json.loads((FIX / "qwen3.8-flash-next.json").read_text())
    calls = []

    def fetch(repo_id, revision=None):
        calls.append((repo_id, revision))
        return _hub_metadata(flash_next_tensors(cfg))
    h = rs.read_target("Qwen/Example-Model", revision="abc", fetch=fetch)
    assert calls == [("Qwen/Example-Model", "abc")] and h.source == "hub:Qwen/Example-Model@abc"
    assert rs.estimate(h)["measured"]["by_category"] == flash["measured"]["by_category"]


def test_a_directory_target_never_calls_the_hub(flash_dir):
    assert rs.read_target(str(flash_dir), fetch=lambda *a, **k: pytest.fail("the hub was called")).source.startswith("local:")


def test_a_hub_failure_becomes_a_residency_error(tmp_path):
    def fetch(repo_id, revision=None):
        raise RuntimeError("401 token abc123")
    with pytest.raises(rs.ResidencyError) as exc:
        rs.read_target("Qwen/Example-Model", fetch=fetch)
    assert "abc123" not in str(exc.value) and "Example-Model" in str(exc.value)


def test_a_target_that_is_neither_a_directory_nor_a_repo_id_is_refused(tmp_path):
    for bad in (str(tmp_path / "nope"), "", "has space/x", "../x/y"):
        with pytest.raises(rs.ResidencyError):
            rs.read_target(bad, fetch=lambda *a, **k: pytest.fail("the hub was called"))


# ---- stage 0 ------------------------------------------------------------------------------------------------------------------

def test_stage_0_writes_the_record_and_a_hazard_when_the_model_does_not_fit_the_box(flash_dir, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    rec = rs.write_residency(stage, model_dir=flash_dir, box_chips=2)
    assert json.loads((stage / rs.FILE).read_text()) == rec
    assert [h["name"] for h in rec["hazards"]] == ["model_does_not_fit_box"]
    reason = rec["hazards"][0]["reason"]
    assert "2 chips" in reason and "at least 3" in reason and "estimate" in reason and "choice" in reason
    assert rec["verdict"] == {"box_chips": 2, "fits_box": False, "chips_needed": 3}


def test_stage_0_has_no_hazard_when_the_box_holds_the_model(flash_dir, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    rec = rs.write_residency(stage, model_dir=flash_dir, box_chips=4)
    assert rec["hazards"] == [] and rec["verdict"]["fits_box"] is True
    assert rs.write_residency(stage, model_dir=flash_dir, box_chips=None)["verdict"]["fits_box"] is None      # the box is unknown: no claim


def test_stage_0_skips_with_a_reason_when_there_are_no_headers(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    for model_dir in (None, tmp_path / "missing", tmp_path):
        rec = rs.write_residency(stage, model_dir=model_dir, box_chips=4)
        assert rec["scan"] == "skipped" and rec["reason"] and rec["hazards"] == []


def test_the_triage_config_names_the_model_when_the_run_gives_none(flash_dir, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "triage_config.json").write_text(json.dumps({"model_snapshot": str(flash_dir)}))
    assert rs.write_residency(stage, model_dir=None, box_chips=4)["scan"] == "ran"


def test_read_record_returns_a_usable_record_and_none_for_a_bad_one(flash_dir, tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "0"
    stage.mkdir(parents=True)
    assert rs.read_record(run) is None
    rs.write_residency(stage, model_dir=flash_dir, box_chips=4)
    assert rs.read_record(run)["estimated"]["chips_needed"] == 3
    (stage / rs.FILE).write_text('{"scan": "ran", "estimated": {"chips_needed": "three"}}')
    assert rs.read_record(run) is None
    (stage / rs.FILE).write_text("[1")
    assert rs.read_record(run) is None
    (stage / rs.FILE).write_text('{"scan": "skipped", "estimated": {"chips_needed": 3}}')
    assert rs.read_record(run) is None


def test_chips_needed_for_the_box_is_read_back_from_the_record(flash_dir, tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "0"
    stage.mkdir(parents=True)
    rs.write_residency(stage, model_dir=flash_dir, box_chips=4)
    rec = rs.read_record(run)
    assert rs.shortfall(rec, 2) and "at least 3 chips" in rs.shortfall(rec, 2) and rs.shortfall(rec, 4) is None
    assert rs.shortfall(rec, None) is None


# ---- the command --------------------------------------------------------------------------------------------------------------

def run_cli(argv, capsys):
    code = rs.main(argv)
    return code, capsys.readouterr().out


def test_the_command_prints_a_table_and_exits_0_when_a_stated_chip_count_holds_the_model(flash_dir, capsys):
    code, out = run_cli([str(flash_dir), "--chips", "4", "8"], capsys)
    assert code == 0 and "routed_experts" in out and "ngram_ple" in out and "host" in out
    assert "estimate" in out and "default (choice)" in out and "allocator" in out


def test_the_command_exits_1_when_no_stated_chip_count_holds_the_model(flash_dir, capsys):
    code, out = run_cli([str(flash_dir), "--chips", "1", "2"], capsys)
    assert code == 1 and "does not fit" in out


def test_the_command_json_is_the_record(flash_dir, capsys):
    code, out = run_cli([str(flash_dir), "--json", "--dram-gib", "48", "--kv-gib", "2", "--reserve-gib", "1", "--policy", "attention=bfp4"], capsys)
    rec = json.loads(out)
    a = {x["name"]: x for x in rec["assumptions"]}
    assert code == 0 and a["dram_gib_per_chip"]["value"] == 48 and rec["estimated"]["policy"]["attention"] == "bfp4"


def test_the_command_exits_2_on_a_bad_input(tmp_path, capsys):
    assert run_cli([str(tmp_path / "missing")], capsys)[0] == 2
    assert run_cli([str(tmp_path), "--policy", "nonsense"], capsys)[0] == 2


def test_the_command_is_wired_into_tt_orchard(flash_dir, capsys):
    from orchard import cli
    code = cli.main(["residency", str(flash_dir), "--chips", "4"])
    assert code == 0 and "routed_experts" in capsys.readouterr().out
