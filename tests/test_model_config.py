# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/model_config.py: read a Hugging Face config.json and normalise the numbers the floor and the hazard scan use."""
from __future__ import annotations

from pathlib import Path

import pytest

from orchard import model_config as mc

FIX = Path(__file__).parent / "fixtures" / "model-configs"


def dense(**over):
    cfg = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
           "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 320}
    cfg.update(over)
    return cfg


def test_text_config_is_merged_over_the_top_level_and_top_level_keys_fill_gaps():
    cfg = {"tie_word_embeddings": True, "vocab_size": 1, "text_config": dense(vocab_size=320)}
    flat = mc.text_config(cfg)
    assert flat["vocab_size"] == 320            # the text config wins
    assert flat["tie_word_embeddings"] is True  # a key only the top level holds is kept
    assert flat["hidden_size"] == 64


def test_dense_config_normalises_to_dims():
    d = mc.dims(dense())
    assert (d.hidden, d.intermediate, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab) == (64, 128, 2, 4, 2, 16, 320)
    assert d.layer_types == ("full_attention", "full_attention")
    assert d.routed_experts == 0 and d.top_k == 0
    assert d.tied_embeddings is False


def test_head_dim_defaults_to_hidden_over_heads():
    cfg = dense()
    del cfg["head_dim"]
    assert mc.dims(cfg).head_dim == 16


def test_layer_types_come_from_the_list_or_the_full_attention_interval():
    lin = dict(linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
    assert mc.dims(dense(layer_types=["linear_attention", "full_attention"], **lin)).layer_types == ("linear_attention", "full_attention")
    d = mc.dims(dense(num_hidden_layers=8, full_attention_interval=4, **lin))
    assert d.layer_types == ("linear_attention",) * 3 + ("full_attention",) + ("linear_attention",) * 3 + ("full_attention",)


def test_moe_names_from_several_families_are_understood():
    d = mc.dims(dense(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32))
    assert (d.routed_experts, d.top_k, d.moe_intermediate, d.shared_intermediate) == (8, 2, 32, 32)
    d = mc.dims(dense(n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, n_shared_experts=2))
    assert d.shared_intermediate == 64          # a count of shared experts times the expert width
    d = mc.dims(dense(num_local_experts=4, num_experts_per_tok=1))
    assert d.routed_experts == 4 and d.moe_intermediate == 128   # a family with no separate expert width uses the MLP width


def test_which_layers_are_moe():
    base = dict(num_hidden_layers=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32)
    assert mc.dims(dense(**base, first_k_dense_replace=1)).moe_layers == (False, True, True, True)
    assert mc.dims(dense(**base, mlp_only_layers=[0, 3])).moe_layers == (False, True, True, False)
    assert mc.dims(dense(**base, decoder_sparse_step=2)).moe_layers == (False, True, False, True)
    assert mc.dims(dense()).moe_layers == (False, False)


@pytest.mark.parametrize("key,bad", [("hidden_size", 0), ("hidden_size", -4), ("hidden_size", 1.5), ("hidden_size", True),
                                     ("hidden_size", float("nan")), ("vocab_size", "320"), ("num_hidden_layers", None)])
def test_a_bad_number_is_refused_and_names_the_key(key, bad):
    with pytest.raises(mc.ModelConfigError, match=key):
        mc.dims(dense(**{key: bad}))


def test_a_missing_required_key_is_refused():
    cfg = dense()
    del cfg["vocab_size"]
    with pytest.raises(mc.ModelConfigError, match="vocab_size"):
        mc.dims(cfg)


def test_layer_types_of_the_wrong_length_or_type_are_refused():
    with pytest.raises(mc.ModelConfigError, match="layer_types"):
        mc.dims(dense(layer_types=["full_attention"]))
    with pytest.raises(mc.ModelConfigError, match="layer_types"):
        mc.dims(dense(layer_types=["full_attention", 3]))


def test_linear_attention_layers_need_their_shape_keys():
    with pytest.raises(mc.ModelConfigError, match="linear_num_key_heads"):
        mc.dims(dense(layer_types=["linear_attention", "full_attention"]))


def test_the_qwen3_next_family_has_an_output_gate_without_a_config_key():
    assert mc.dims(dense(model_type="qwen3_next")).attn_output_gate is True
    assert mc.dims(dense(attn_output_gate=True)).attn_output_gate is True
    assert mc.dims(dense(model_type="llama")).attn_output_gate is False


def test_top_k_above_the_expert_count_is_refused():
    with pytest.raises(mc.ModelConfigError, match="num_experts_per_tok"):
        mc.dims(dense(num_experts=4, num_experts_per_tok=5, moe_intermediate_size=32))
    assert mc.dims(dense(num_experts=4, num_experts_per_tok=4, moe_intermediate_size=32)).top_k == 4


# ---- Copilot review of PR 190: layouts the counting must not assume -----------------------------------

def test_two_projection_mlp_families_are_counted_with_two_projections():
    assert mc.dims(dense(model_type="gpt_neox")).mlp_projections == 2
    assert mc.dims(dense(model_type="llama")).mlp_projections == 3
    assert mc.dims(dense()).mlp_projections == 3           # a family the table does not list is a gated MLP


@pytest.mark.parametrize("n", [10**30, 10**7, mc.MAX_LAYERS + 1])
def test_a_huge_layer_count_is_refused_before_any_per_layer_tuple_is_built(n):
    with pytest.raises(mc.ModelConfigError, match="num_hidden_layers"):
        mc.dims(dense(num_hidden_layers=n))


def test_the_layer_bound_leaves_real_models_alone():
    assert mc.dims(dense(num_hidden_layers=mc.MAX_LAYERS)).layers == mc.MAX_LAYERS


def test_mla_head_dim_falls_back_to_the_query_key_width():
    cfg = dense(hidden_size=7168, num_attention_heads=128, kv_lora_rank=512, q_lora_rank=1536, qk_nope_head_dim=128,
                qk_rope_head_dim=64, v_head_dim=128)
    del cfg["head_dim"]
    assert mc.dims(cfg).head_dim == 192                     # 128 + 64, and 7168 / 128 would give 56
    assert mc.dims(dict(cfg, head_dim=64)).head_dim == 64   # an explicit head_dim still wins


def test_a_mistral_sliding_window_makes_every_layer_a_sliding_layer():
    d = mc.dims(dense(model_type="mistral", sliding_window=4096))
    assert d.layer_types == ("sliding_attention", "sliding_attention")


def test_qwen2_style_windows_apply_from_max_window_layers_when_enabled():
    cfg = dense(model_type="qwen2", num_hidden_layers=4, sliding_window=4096, max_window_layers=2)
    assert mc.dims(dict(cfg, use_sliding_window=False)).layer_types == ("full_attention",) * 4
    assert mc.dims(dict(cfg, use_sliding_window=True)).layer_types == ("full_attention",) * 2 + ("sliding_attention",) * 2


def test_a_window_with_no_known_layout_is_refused_and_not_ignored():
    with pytest.raises(mc.ModelConfigError, match="sliding_window"):
        mc.dims(dense(model_type="somenewmodel", sliding_window=4096))
    assert mc.dims(dense(model_type="somenewmodel", sliding_window=None)).layer_types == ("full_attention",) * 2
    assert mc.dims(dense(model_type="somenewmodel", sliding_window=0)).layer_types == ("full_attention",) * 2
    # an explicit layer list settles it
    listed = mc.dims(dense(model_type="somenewmodel", sliding_window=4096, layer_types=["sliding_attention", "full_attention"]))
    assert listed.layer_types == ("sliding_attention", "full_attention")


def test_a_huge_mlp_only_layers_list_is_normalised_in_linear_time():
    import time
    cfg = dense(num_hidden_layers=mc.MAX_LAYERS, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
                mlp_only_layers=[mc.MAX_LAYERS - 1] * 200_000)
    start = time.monotonic()
    d = mc.dims(cfg)
    assert time.monotonic() - start < 2.0                      # a per-layer scan of the list takes billions of comparisons
    assert d.moe_layers[-1] is False and d.moe_layers[0] is True


def test_null_keys_are_not_listed_as_present():
    d = mc.dims(dense(kv_lora_rank=None, index_topk=None, hc_mult=3))
    assert "hc_mult" in d.keys and "kv_lora_rank" not in d.keys and "index_topk" not in d.keys


def test_kv_head_count_follows_each_family_spelling_and_multi_query():
    assert mc.dims(dense(num_key_value_heads=None)).kv_heads == 4                     # no key: one KV head per query head
    assert mc.dims(dense(num_key_value_heads=None, model_type="falcon", multi_query=True)).kv_heads == 1       # Falcon's legacy multi-query layout
    assert mc.dims(dense(num_key_value_heads=None, model_type="falcon", multi_query=True, num_kv_heads=2)).kv_heads == 2
    assert mc.dims(dense(num_key_value_heads=None, model_type="falcon", multi_query=False)).kv_heads == 4
    assert mc.dims(dense(num_key_value_heads=None, n_head_kv=2)).kv_heads == 2
    assert mc.dims(dense(num_key_value_heads=None, multi_query_group_num=2)).kv_heads == 2
    assert mc.dims(dense(num_key_value_heads=3, num_kv_heads=2)).kv_heads == 3       # the standard key wins
    with pytest.raises(mc.ModelConfigError, match="num_kv_heads"):
        mc.dims(dense(num_key_value_heads=None, num_kv_heads=0))


def test_not_a_dict_is_refused():
    with pytest.raises(mc.ModelConfigError):
        mc.dims([1, 2])


def test_read_model_config_reads_the_real_fixtures():
    cfg = mc.read_model_config(FIX / "qwen3-coder-next.json")
    assert mc.dims(cfg).routed_experts == 512
    d = mc.dims(mc.read_model_config(FIX / "qwen3.8-27b.json"))
    assert (d.hidden, d.layers, d.vocab, d.routed_experts) == (5120, 64, 248320, 0)
    assert d.layer_types.count("full_attention") == 16 and d.layer_types.count("linear_attention") == 48


def test_read_model_config_turns_every_read_failure_into_one_error(tmp_path):
    (tmp_path / "bad.json").write_text('{"a": NaN}')
    for name in ("bad.json", "missing.json"):
        with pytest.raises(mc.ModelConfigError):
            mc.read_model_config(tmp_path / name)
    (tmp_path / "list.json").write_text("[1]")
    with pytest.raises(mc.ModelConfigError):
        mc.read_model_config(tmp_path / "list.json")


def test_read_model_config_refuses_a_huge_file(tmp_path):
    (tmp_path / "big.json").write_bytes(b'{"a": "' + b"x" * (mc.MAX_CONFIG_BYTES + 10) + b'"}')
    with pytest.raises(mc.ModelConfigError):
        mc.read_model_config(tmp_path / "big.json")


def test_minicpm3_value_head_width_comes_from_the_hidden_size():
    cfg = dense(model_type="minicpm3", hidden_size=2560, num_attention_heads=40, kv_lora_rank=256, q_lora_rank=768, qk_nope_head_dim=64,
                qk_rope_head_dim=32)
    assert mc.dims(cfg).mla["v_head_dim"] == 64
    with pytest.raises(mc.ModelConfigError, match="v_head_dim"):      # another model type without the key is still refused
        mc.dims(dict(cfg, model_type="other_mla"))
