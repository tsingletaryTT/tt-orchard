# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Engram table of Hy4-flash in the residency estimator (issue 844).

`tencent/Hy4-flash` keeps four Engram layers (`engram_layer_ids = [2, 12, 22, 32]`, which are layers 1, 11, 21 and 31 in the tensor names). Each holds an embedding table of about 15 GiB and a hash
lookup table; the published bundle keeps them in host RAM (60 GiB in all) and sends the looked-up rows to the chips. The estimator
used to file the tables under `norms_other`, which the default policy keeps on the chips in bf16, so a plan built from tensor names counted
60 GiB on the device. These tests use the real tensor names (tests/fixtures/hy4-flash-tensor-names.json: names only, no weights)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import residency as rs
from residency_headers import write_snapshot

FIX = Path(__file__).parent / "fixtures"
GIB = 1 << 30
NAMES = json.loads((FIX / "hy4-flash-tensor-names.json").read_text())["names"]
ENGRAM = [n for n in NAMES if "engram" in n]
OTHERS = [n for n in NAMES if "engram" not in n]
# the rest of the Engram layer: projections and norms that run on the mesh in the bundle
ENGRAM_TABLES = [n for n in ENGRAM if n.endswith("multi_head_embedding.embedding.weight") or n.endswith(".lookup_table")]
ENGRAM_ON_MESH = [n for n in ENGRAM if n not in ENGRAM_TABLES]

# the Engram keys of the real config (text_config), values cut to what the key checks need
HY4_ENGRAM_CONFIG = {"text_config": {
    "engram_embed_random_seed": 1234, "engram_fp8_embedding": True, "engram_isolate_fffd_namespace": False,
    "engram_layer_head_vocab_sizes": {"12": [[10312277, 10312283]]}, "engram_layer_ids": [2, 12, 22, 32],
    "engram_max_ngram_size": 3, "engram_n_embed_per_ngram": 768, "engram_n_head_per_ngram": 8, "engram_pad_id": 120002,
    "engram_seed": 0, "engram_vocab_size": [10312000, 10312000]}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(huggingface_hub, "get_safetensors_metadata", boom)


def test_the_fixture_holds_the_real_engram_names():
    """Four layers, each with one embedding table and one lookup table: the rest of the 164 Engram names are projections and norms."""
    assert len(ENGRAM_TABLES) == 8 and len(ENGRAM) == 164
    assert {n.split(".")[2] for n in ENGRAM_TABLES} == {"1", "11", "21", "31"}      # the config counts layers from 1: ids 2, 12, 22, 32


@pytest.mark.parametrize("name", ENGRAM_TABLES)
def test_an_engram_table_is_a_host_resident_table(name):
    assert rs.categorize(name, (10312000, 768)) == "ngram_ple"


@pytest.mark.parametrize("name", ENGRAM_ON_MESH)
def test_the_projections_and_norms_of_an_engram_layer_stay_on_the_mesh(name):
    """In the bundle the projections, the norms and the gate scales run on the device: they are not table tensors."""
    assert rs.categorize(name, (1536, 768)) != "ngram_ple"


def test_the_projection_weight_named_in_the_issue_stays_where_it_was():
    assert rs.categorize("model.layers.12.engram_layer.value_proj.weight", (1536, 6144)) == "norms_other"


@pytest.mark.parametrize("name,shape,want", [
    ("model.embed_tokens.weight", (120000, 4096), "embedding"),
    ("model.layers.0.self_attn.q_proj.weight", (4096, 4096), "attention"),
    ("model.layers.0.mlp.up_proj.weight", (4096, 4096), "dense_mlp"),
    ("model.layers.1.ple.ngram_embeddings.5.weight", (4, 4), "ngram_ple"),
    ("model.layers.3.notengram_embedding.weight", (4, 4), "norms_other"),         # the word must start with engram
    ("model.layers.3.engineering_table.weight", (4, 4), "norms_other"),
])
def test_other_tensors_are_not_moved(name, shape, want):
    assert rs.categorize(name, shape) == want


def test_every_non_engram_name_of_the_model_keeps_its_category_unless_it_is_a_table():
    """No name outside the Engram layers is filed as a table (the model has no per-layer embedding or n-gram module)."""
    assert [n for n in OTHERS if rs.categorize(n, (8, 8)) == "ngram_ple"] == []


@pytest.mark.parametrize("key", ["engram_layer_head_vocab_sizes", "engram_layer_ids", "engram_fp8_embedding", "engram_vocab_size",
                                 "engram_n_embed_per_ngram", "ENGRAM_Max_Size"])
def test_an_engram_config_key_names_a_host_table(key):
    assert rs._table_key(key) is True


@pytest.mark.parametrize("key", ["engineering_depth", "notengram_x", "multiple_of", "sample_rate", "hidden_size"])
def test_other_config_keys_do_not(key):
    assert rs._table_key(key) is False


def test_the_real_engram_config_names_a_host_table_even_without_the_ngram_key():
    cfg = {"text_config": {k: v for k, v in HY4_ENGRAM_CONFIG["text_config"].items() if k != "engram_n_embed_per_ngram"
                           and k != "engram_max_ngram_size" and k != "engram_n_head_per_ngram"}}
    assert rs.config_has_host_table(cfg) is True


def test_a_config_without_engram_or_ngram_keys_names_none():
    assert rs.config_has_host_table({"text_config": {"hidden_size": 4096, "num_hidden_layers": 43}}) is False


def test_the_weight_index_of_the_real_model_has_a_host_table_and_without_the_tables_it_has_none():
    assert rs.index_has_host_table(NAMES) is True
    assert rs.index_has_host_table([n for n in NAMES if n not in ENGRAM_TABLES]) is False


def test_the_hub_count_path_flags_the_table_from_the_config_and_from_the_names():
    assert rs.host_table_from(HY4_ENGRAM_CONFIG, NAMES) is True
    assert rs.host_table_from(HY4_ENGRAM_CONFIG, None) is True
    assert rs.host_table_from({"text_config": {"hidden_size": 8}}, NAMES) is True


def _estimate(tmp_path, **kw):
    """A snapshot of the model's real names with small shapes: the tables are the biggest tensors, as in the real checkpoint."""
    tensors = []
    for n in NAMES:
        if n in ENGRAM_TABLES and n.endswith(".lookup_table"):
            tensors.append((n, "I64", [64]))
        elif n in ENGRAM_TABLES:
            tensors.append((n, "F8_E4M3", [4096, 64]))
        elif "norm" in n or n.endswith(".bias"):
            tensors.append((n, "BF16", [64]))
        else:
            tensors.append((n, "BF16", [32, 32]))
    return rs.estimate(rs.read_headers(write_snapshot(tmp_path / "hy4", tensors, shards=2)), **kw)


def _est(tmp_path, **kw):
    return _estimate(tmp_path, **kw)["estimated"]


def test_a_plan_from_the_real_names_puts_the_engram_tables_on_the_host_and_counts_them_in_host_gib(tmp_path):
    est = _est(tmp_path)
    table_bytes = 4 * (4096 * 64) + 4 * 64 * 8
    assert est["by_category"]["ngram_ple"] == {"placement": "host", "stored_bytes": table_bytes}
    assert est["host_resident_categories"] == ["ngram_ple"]
    assert est["host_bytes"] >= table_bytes
    on_chips = sum(v["stored_bytes"] for k, v in est["by_category"].items() if v["placement"] == "device")
    assert est["device_bytes"] == on_chips and est["device_bytes"] < est["host_bytes"] + on_chips
    # the table is not in any chip figure of the plan
    assert est["by_category"]["norms_other"]["stored_bytes"] < est["by_category"]["ngram_ple"]["stored_bytes"] * 100


def test_the_printed_plan_names_the_table_as_host_memory(tmp_path):
    text = "\n".join(rs._text(_estimate(tmp_path), []))
    assert "ngram_ple" in text and "on the host" in text


def test_a_policy_that_puts_the_table_on_the_chips_is_still_honoured(tmp_path):
    est = _est(tmp_path, policy={"ngram_ple": "bf16"})
    assert est["by_category"]["ngram_ple"]["placement"] == "device"
    assert est["host_resident_categories"] == []
