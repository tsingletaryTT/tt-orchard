# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hy4-flash (Tencent, 220B mixture of experts with sparse MLA attention and an Engram n-gram memory) in the plan and the hazard scan
(Hy4-flash study, 2026-11-25, learnings 1, 2, 4 and 5).

Before this change `tt-orchard tier-plan tencent/Hy4-flash` stopped with "unknown family: the layer type 'deepseek_sparse_attention' is not one the
floor can count", the config's `head_dim` key (64, the rope part) was printed as the head width (the q/k head is 256 wide), and the Engram keys
matched neither the plan's caveats nor the hazard scan. The fixture is the public config.json at the pinned revision b2498eba."""
import copy
import json
from pathlib import Path

import pytest

from orchard import decode_floor as df, hazards, model_config as mc, tier_plan

FIX = Path(__file__).parent / "fixtures" / "model-configs"
HY4 = json.loads((FIX / "hy4-flash.pinned.config.json").read_text())
T = HY4["text_config"]


def mla_params(hidden, heads, q_lora, kv_lora, nope, rope, v):
    """One MLA mixer, written out here from the shapes (not from the module under test): low-rank query, compressed key-value down projection,
    its up projection, output projection."""
    qk = nope + rope
    return hidden * q_lora + q_lora * heads * qk + hidden * (kv_lora + rope) + kv_lora * heads * (nope + v) + heads * v * hidden


def test_the_config_reads_and_the_layer_mix_is_twelve_sparse_and_thirty_sliding():
    d = mc.dims(HY4)
    assert d.layers == 42 and d.routed_experts == 512 and d.top_k == 16
    assert d.layer_types.count("deepseek_sparse_attention") == 12 and d.layer_types.count("sliding_attention") == 30
    assert d.heads == 32 and d.swa_heads == 64


def test_the_head_width_is_the_query_key_width_not_the_rope_part_the_head_dim_key_names():
    d = mc.dims(HY4)
    assert T["head_dim"] == 64                       # the key names the rope part
    assert d.head_dim == 192 + 64 == T["qk_head_dim"]


def test_a_head_dim_key_on_another_mla_model_does_not_change_its_width():
    cfg = {"model_type": "deepseek_v3", "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
           "vocab_size": 256, "intermediate_size": 256, "kv_lora_rank": 32, "q_lora_rank": 0, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8,
           "v_head_dim": 16}
    assert mc.dims(cfg).head_dim == 24
    assert mc.dims(dict(cfg, head_dim=8)).head_dim == 24         # an MLA model's width is nope + rope


def test_sparse_and_sliding_layers_are_counted_as_mla_with_their_own_head_counts():
    d = mc.dims(HY4)
    sparse = mla_params(3072, 32, 1024, 512, 192, 64, 256)
    sliding = mla_params(3072, 64, 1024, 512, 192, 64, 256)
    assert df._mixer_params(d, "deepseek_sparse_attention") == sparse
    assert df._mixer_params(d, "sliding_attention") == sliding
    assert sliding > sparse                          # twice the heads


def test_the_plan_counts_the_attention_of_all_forty_two_layers():
    s = tier_plan.shape(HY4)
    assert s.attention_params == 12 * mla_params(3072, 32, 1024, 512, 192, 64, 256) + 30 * mla_params(3072, 64, 1024, 512, 192, 64, 256)


def test_the_plan_runs_and_names_what_it_leaves_out():
    res = tier_plan.plan(HY4, chips_list=(4,), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    text = " ".join(res["caveats"])
    assert res["always_resident_is_lower_bound"]
    assert "sparse-attention indexer" in text and "hyper-connections" in text and "vision tower" in text


def test_sliding_layers_with_other_mla_shapes_are_refused_not_guessed():
    for key, value in (("swa_qk_nope_head_dim", 128), ("swa_v_head_dim", 128), ("swa_qk_rope_head_dim", 32)):
        cfg = copy.deepcopy(HY4)
        cfg["text_config"][key] = value
        with pytest.raises(mc.ModelConfigError, match="sliding-window"):
            mc.dims(cfg)


def test_a_sliding_head_count_must_be_a_whole_number():
    cfg = copy.deepcopy(HY4)
    cfg["text_config"]["swa_num_attention_heads"] = "64"
    with pytest.raises(mc.ModelConfigError):
        mc.dims(cfg)


def test_the_sliding_heads_are_a_count_a_tensor_parallel_split_must_divide():
    s = tier_plan.shape(HY4)
    labels = {label: value for label, value, unit in s.tp_counts}
    assert labels["sliding-window attention heads"] == 64 and labels["attention heads"] == 32


# ---- the Engram memory (learning 2) ----

def test_the_engram_table_is_computed_from_the_config():
    e = mc.engram_table(mc.text_config(HY4))
    assert e["layers"] == 4 and e["orders"] == 2 and e["heads_per_order"] == 8 and e["values_per_row"] == 96
    assert e["rows"] == 660_006_416
    assert e["params"] == 660_006_416 * 96 == 63_360_615_936
    # the config says FP8 rows with a bf16 scale each: one byte per value and two per row
    assert e["bytes"] == e["params"] + 2 * e["rows"]
    assert e["bytes"] / (1 << 30) == pytest.approx(60.2385, abs=1e-3)


def test_an_engram_table_in_bf16_is_two_bytes_a_value():
    cfg = dict(mc.text_config(HY4), engram_fp8_embedding=False)
    e = mc.engram_table(cfg)
    assert e["bytes"] == e["params"] * 2


def test_a_config_without_engram_keys_has_no_table():
    assert mc.engram_table(mc.text_config(json.loads((FIX / "qwen3-32b.json").read_text()))) is None


def test_an_engram_config_with_an_unusable_table_list_gives_no_figure_and_does_not_crash():
    cfg = dict(mc.text_config(HY4), engram_layer_head_vocab_sizes={"2": "many"})
    assert mc.engram_table(cfg) is None
    cfg = dict(mc.text_config(HY4), engram_layer_head_vocab_sizes={"2": [[10 ** 400]]})
    assert mc.engram_table(cfg) is None


def test_the_plan_names_the_engram_table_and_says_it_is_host_memory():
    res = tier_plan.plan(HY4, chips_list=(4,), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    line = [c for c in res["caveats"] if "Engram" in c]
    assert len(line) == 1
    assert "60.2 GiB" in line[0] and "63.4B" in line[0] and "host" in line[0]


def test_the_hazard_scan_sees_the_engram_memory_as_an_ngram_embedding():
    d = mc.dims(HY4)
    assert "ngram_embedding" in hazards._mixers(d)


def test_a_config_without_engram_keys_is_not_flagged_as_an_ngram_embedding():
    d = mc.dims(json.loads((FIX / "qwen3-32b.json").read_text()))
    assert "ngram_embedding" not in hazards._mixers(d)


# ---- the two hazard rows whose cause is the stock op (learning 5) ----

def test_the_stock_op_rows_say_a_published_bundle_exists_and_that_the_scan_cannot_tell_if_they_bite():
    scan = hazards.scan_model_hazards(HY4, None, dies=4)
    names = {h.name: h for h in scan.hazards}
    assert "routed_experts_above_256" in names and "mla_flash_decode_needs_core_grid" in names
    text = " ".join(scan.notes)
    assert "gather" in text and "raahemnabeel/hy4-flash" in text and "claim" in text


def test_the_note_is_absent_for_a_model_that_trips_neither_row():
    scan = hazards.scan_model_hazards(json.loads((FIX / "qwen3-32b.json").read_text()), None, dies=4)
    assert not any("raahemnabeel/hy4-flash" in n for n in scan.notes)


# ---- cases the first mutation pass left green ----

def flat(**over):
    """A small flat config with no latent attention."""
    cfg = {"model_type": "x", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 16,
           "vocab_size": 100, "intermediate_size": 128}
    cfg.update(over)
    return cfg


def test_a_sparse_attention_layer_in_a_config_without_latent_shapes_is_refused():
    d = mc.dims(flat(layer_types=["deepseek_sparse_attention", "full_attention"]))
    with pytest.raises(df.FloorInputError, match="multi-latent attention shapes"):
        df._mixer_params(d, "deepseek_sparse_attention")


def test_sliding_heads_in_a_config_without_latent_attention_are_ignored():
    assert mc.dims(flat(swa_num_attention_heads=8)).swa_heads == 0


def test_an_engram_row_that_does_not_split_into_whole_heads_gives_no_figure():
    assert mc.engram_table(dict(mc.text_config(HY4), engram_n_embed_per_ngram=770)) is None


def test_an_engram_layer_with_no_tables_gives_no_figure():
    assert mc.engram_table(dict(mc.text_config(HY4), engram_layer_head_vocab_sizes={"2": []})) is None
    assert mc.engram_table(dict(mc.text_config(HY4), engram_layer_head_vocab_sizes={})) is None


def test_the_scan_with_a_nearest_config_does_not_report_sparse_attention_as_an_unknown_mixer():
    nearest = json.loads((FIX / "qwen3-32b.json").read_text())
    scan = hazards.scan_model_hazards(HY4, nearest, dies=4)
    assert "deepseek_sparse_attention" not in scan.facts["mixers"]
    assert not any("deepseek_sparse_attention" in n for n in scan.notes)


def test_the_bundle_note_needs_a_flagged_row():
    cfg = copy.deepcopy(HY4)
    cfg["text_config"]["n_routed_experts"] = 128            # at most 256 experts: the first row is not flagged
    del cfg["text_config"]["kv_lora_rank"]                  # no latent attention: the second row is not flagged
    scan = hazards.scan_model_hazards(cfg, None, dies=4)
    assert not any("raahemnabeel/hy4-flash" in n for n in scan.notes)


# ---- the latent MoE (found while recomputing the card's figures) ----
# Hy4-flash's routed experts work in a 1,536-wide latent (`routed_expert_hidden_size`), not in the 3,072-wide hidden state: a down projection
# (3072 to 1536) feeds them and an up projection (1536 to 3072) follows. Counting an expert as 3 x hidden x moe_intermediate doubled the expert
# parameters (297B instead of 148.6B) and everything the plan derives from them.

def test_the_expert_width_is_the_latent_width():
    d = mc.dims(HY4)
    assert d.expert_hidden == 1536 and d.hidden == 3072


def test_a_model_without_a_latent_width_keeps_experts_at_the_hidden_width():
    qwen = mc.dims(json.loads((FIX / "qwen3-coder-30b-a3b.json").read_text()))
    assert qwen.expert_hidden == 0
    s = tier_plan.shape(json.loads((FIX / "qwen3-coder-30b-a3b.json").read_text()))
    assert s.expert_params == 3 * qwen.hidden * qwen.moe_intermediate


def test_one_expert_is_three_latent_by_intermediate_matrices():
    s = tier_plan.shape(HY4)
    assert s.expert_params == 3 * 1536 * 1536
    assert s.expert_params_total == 41 * 512 * 3 * 1536 * 1536
    assert round(s.expert_params_total / 1e9, 1) == 148.6


def test_the_latent_projections_and_the_router_are_always_resident():
    s = tier_plan.shape(HY4)
    per_layer = 3072 * 512 + 2 * 3072 * 1536              # the router, then the down and up projections around the experts
    assert s.router_params == 41 * per_layer


def test_the_floor_reads_sixteen_latent_experts_a_layer():
    policy = {"default": "bf16"}
    f = df.decode_floor(HY4, policy, 4, 512.0)
    assert f.by_group["routed_experts"] == 41 * 16 * 3 * 1536 * 1536 * 2
    assert f.by_group["router"] == 41 * (3072 * 512 + 2 * 3072 * 1536) * 2


def test_a_latent_width_must_be_a_whole_number():
    cfg = copy.deepcopy(HY4)
    cfg["text_config"]["routed_expert_hidden_size"] = 15.5
    with pytest.raises(mc.ModelConfigError):
        mc.dims(cfg)


def test_an_expert_width_equal_to_the_hidden_size_is_not_a_latent_moe():
    cfg = copy.deepcopy(HY4)
    cfg["text_config"]["routed_expert_hidden_size"] = 3072
    d = mc.dims(cfg)
    assert d.expert_hidden == 0 and df._latent_params(d) == 0
    assert df._expert_params(d) == 3 * 3072 * 1536
