# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/decode_floor.py: bytes read per decoded token and the DRAM-bandwidth floor on time per token (issue 181).

The expected numbers are worked out by hand in each test, with plain arithmetic, so the module's counting is checked against a
second statement of the same formula and not against itself."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import decode_floor as df
from orchard import model_config as mc

FIX = Path(__file__).parent / "fixtures" / "model-configs"

DENSE = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 320}
# per layer: q 64*4*16 + k,v 2*64*2*16 + o 4*16*64 = 4096 + 4096 + 4096 = 12288 ; mlp 3*64*128 = 24576
DENSE_ATTN, DENSE_MLP, DENSE_HEAD = 12288 * 2, 24576 * 2, 64 * 320

MOE = dict(DENSE, num_hidden_layers=3, first_k_dense_replace=1, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
           shared_expert_intermediate_size=32)
# layer 0 dense mlp 24576 ; layers 1 and 2 moe: router 64*8, routed 2*3*64*32, shared 3*64*32 (each, times two layers)
MOE_ROUTER, MOE_ROUTED, MOE_SHARED = 2 * 64 * 8, 2 * 2 * 3 * 64 * 32, 2 * 3 * 64 * 32


def policy(**kw):
    return {"default": "bf16", **kw}


def test_dense_bytes_per_token_in_bf16():
    f = df.decode_floor(DENSE, policy(), dies=1, per_die_bandwidth_gb_s=100.0)
    params = DENSE_ATTN + DENSE_MLP + DENSE_HEAD
    assert f.bytes_per_token == pytest.approx(params * 2.0)
    assert f.by_group == {"attention": pytest.approx(DENSE_ATTN * 2.0), "dense_mlp": pytest.approx(DENSE_MLP * 2.0),
                          "lm_head": pytest.approx(DENSE_HEAD * 2.0)}


def test_floor_time_and_rate_follow_from_bytes_dies_and_bandwidth():
    f = df.decode_floor(DENSE, policy(), dies=2, per_die_bandwidth_gb_s=0.5)   # 1 GB/s in all
    assert f.floor_ms == pytest.approx(f.bytes_per_token / 1e9 * 1000)
    assert f.floor_tok_s == pytest.approx(1000.0 / f.floor_ms)
    one = df.decode_floor(DENSE, policy(), dies=1, per_die_bandwidth_gb_s=0.5)
    assert one.floor_ms == pytest.approx(2 * f.floor_ms)                        # half the dies, twice the time


def test_dtype_policy_sets_bytes_per_weight_by_group():
    f = df.decode_floor(DENSE, policy(attention="bfp8", dense_mlp="bfp4"), dies=1, per_die_bandwidth_gb_s=1.0)
    assert f.by_group["attention"] == pytest.approx(DENSE_ATTN * 1.0625)
    assert f.by_group["dense_mlp"] == pytest.approx(DENSE_MLP * 0.5625)
    assert f.by_group["lm_head"] == pytest.approx(DENSE_HEAD * 2.0)             # falls to the default


def test_moe_reads_only_the_top_k_experts_plus_the_shared_one():
    f = df.decode_floor(MOE, policy(routed_experts="bfp4"), dies=2, per_die_bandwidth_gb_s=512.0)
    assert f.by_group["routed_experts"] == pytest.approx(MOE_ROUTED * 0.5625)
    assert f.by_group["shared_experts"] == pytest.approx(MOE_SHARED * 2.0)
    assert f.by_group["router"] == pytest.approx(MOE_ROUTER * 2.0)
    assert f.by_group["dense_mlp"] == pytest.approx(24576 * 2.0)                  # the single dense layer
    # eight experts exist but two are read: what is stored is larger than what is read
    assert f.stored_params > f.active_params


def test_mixed_dense_and_moe_layers_sum_to_the_whole():
    f = df.decode_floor(MOE, policy(), dies=1, per_die_bandwidth_gb_s=1.0)
    attn = 12288 * 3
    expected = attn + 24576 + MOE_ROUTER + MOE_ROUTED + MOE_SHARED + DENSE_HEAD
    assert f.active_params == pytest.approx(expected)
    assert f.bytes_per_token == pytest.approx(expected * 2.0)
    # stored: all 8 experts in each of the two moe layers
    stored = attn + 24576 + MOE_ROUTER + 2 * 8 * 3 * 64 * 32 + MOE_SHARED + DENSE_HEAD + DENSE_HEAD   # head and embeddings
    assert f.stored_params == pytest.approx(stored)


def test_tied_embeddings_are_stored_once_but_the_head_is_still_read():
    cfg = dict(DENSE, tie_word_embeddings=True)
    f = df.decode_floor(cfg, policy(), 1, 1.0)
    assert f.by_group["lm_head"] == pytest.approx(DENSE_HEAD * 2.0)
    untied = df.decode_floor(DENSE, policy(), 1, 1.0)
    assert untied.stored_params - f.stored_params == DENSE_HEAD


def test_linear_attention_layers_are_priced_with_their_own_shapes():
    cfg = dict(DENSE, num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
               linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4)
    # key_dim 16, value_dim 32 ; qkv 64*(2*16+32) + z 64*32 + ba 64*2*4 + conv (2*16+32)*4 + out 32*64
    linear = 64 * 64 + 64 * 32 + 64 * 8 + 64 * 4 + 32 * 64
    f = df.decode_floor(cfg, policy(), 1, 1.0)
    assert f.by_group["attention"] == pytest.approx((linear + 12288) * 2.0)


def test_attention_output_gate_doubles_the_query_projection():
    plain = df.decode_floor(DENSE, policy(), 1, 1.0).by_group["attention"]
    gated = df.decode_floor(dict(DENSE, attn_output_gate=True), policy(), 1, 1.0).by_group["attention"]
    assert gated - plain == pytest.approx(2 * 64 * 4 * 16 * 2.0)                  # two layers, one extra q-sized matrix each


def test_multi_latent_attention_is_priced_from_its_ranks():
    cfg = dict(DENSE, num_hidden_layers=1, kv_lora_rank=8, q_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=10)
    # q: 64*16 + 16*4*(12+4) ; kv_a: 64*(8+4) ; kv_b: 8*4*(12+10) ; o: 4*10*64
    mla = 64 * 16 + 16 * 4 * 16 + 64 * 12 + 8 * 4 * 22 + 4 * 10 * 64
    assert df.decode_floor(cfg, policy(), 1, 1.0).by_group["attention"] == pytest.approx(mla * 2.0)


def test_an_unknown_layer_type_is_refused_not_guessed():
    with pytest.raises(df.FloorInputError, match="mamba_block"):
        df.decode_floor(dict(DENSE, layer_types=["mamba_block", "full_attention"]), policy(), 1, 1.0)


def test_the_result_is_labelled_an_estimate_and_lists_its_inputs():
    f = df.decode_floor(DENSE, policy(attention="bfp8"), dies=2, per_die_bandwidth_gb_s=512.0, board="p300")
    d = f.as_dict()
    assert d["label"] == "estimate"
    assert d["inputs"]["dies"] == 2 and d["inputs"]["per_die_bandwidth_gb_s"] == 512.0
    assert d["inputs"]["dtype_policy"]["attention"] == "bfp8"
    assert d["inputs"]["board"] == "p300"
    assert "batch of one" in d["assumptions"] and "KV cache" in d["assumptions"]
    json.dumps(d, allow_nan=False)                                                # plain JSON, no NaN


@pytest.mark.parametrize("dies", [0, -1, 1.5, True, "2", None, float("nan")])
def test_bad_dies_are_refused(dies):
    with pytest.raises(df.FloorInputError, match="dies"):
        df.decode_floor(DENSE, policy(), dies, 1.0)


@pytest.mark.parametrize("bw", [0, -1.0, float("nan"), float("inf"), True, "512", None])
def test_bad_bandwidth_is_refused(bw):
    with pytest.raises(df.FloorInputError, match="bandwidth"):
        df.decode_floor(DENSE, policy(), 1, bw)


def test_bandwidth_has_no_silent_default():
    with pytest.raises(TypeError):
        df.decode_floor(DENSE, policy(), 1)          # per-die bandwidth is a required input


@pytest.mark.parametrize("bad", [{"default": "int3"}, {"attention": "bf16"}, {}, {"default": "bf16", "nonsense": "bf16"}, "bf16", None])
def test_bad_dtype_policy_is_refused(bad):
    with pytest.raises(df.FloorInputError):
        df.decode_floor(DENSE, bad, 1, 1.0)


def test_bad_model_config_is_refused_with_the_floor_error():
    with pytest.raises(df.FloorInputError):
        df.decode_floor({"hidden_size": 5}, policy(), 1, 1.0)


def test_a_two_projection_mlp_is_counted_with_two_projections():
    three = df.decode_floor(DENSE, policy(), 1, 1.0).by_group["dense_mlp"]
    two = df.decode_floor(dict(DENSE, model_type="gpt_neox"), policy(), 1, 1.0).by_group["dense_mlp"]
    assert two == pytest.approx(three * 2 / 3)
    assert two == pytest.approx(2 * 2 * 64 * 128 * 2.0)       # two layers, two matrices each, bf16


def test_the_mla_floor_and_the_scan_use_the_same_head_width():
    from orchard import hazards as hz
    cfg = dict(DENSE, hidden_size=7168, num_attention_heads=128, intermediate_size=18432, num_key_value_heads=128,
               kv_lora_rank=512, q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, vocab_size=129280)
    del cfg["head_dim"]
    assert df.decode_floor(cfg, policy(), 1, 1.0).model_summary["hidden_size"] == 7168      # the floor still builds
    s = hz.scan_model_hazards(cfg, cfg)
    assert s.facts["tiles"]["head_dim"] == {"value": 192, "tiles": 6.0, "whole": True}
    assert not any(h.name == "head_dim_not_whole_tiles" for h in s.hazards)


@pytest.mark.parametrize("bad", [{"default": ["bf16"]}, {"default": {"a": 1}}, {"default": None}, {"default": 2}, {"default": "bf16", "attention": ["bfp8"]}])
def test_a_dtype_that_is_not_a_string_is_refused_with_the_floor_error(bad):
    with pytest.raises(df.FloorInputError):
        df.decode_floor(DENSE, bad, 1, 1.0)


def test_the_record_holds_every_normalised_dimension_the_count_used():
    a = df.decode_floor(DENSE, policy(), 1, 1.0).as_dict()["inputs"]["model"]
    b = df.decode_floor(dict(DENSE, intermediate_size=256), policy(), 1, 1.0).as_dict()["inputs"]["model"]
    assert a != b                                                   # a width that changes the bytes changes the record
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                "vocab_size", "tied_embeddings", "mlp_projections", "layer_types", "moe_layers", "routed_experts", "experts_per_token",
                "moe_intermediate_size", "shared_intermediate_size", "attn_output_gate", "linear_attention", "multi_latent_attention"):
        assert key in a, key
    moe = df.decode_floor(MOE, policy(), 1, 1.0).as_dict()["inputs"]["model"]
    assert moe["moe_layers"] == [False, True, True] and moe["layer_types"] == {"full_attention": 3}
    json.dumps(moe, allow_nan=False)


# ---- the board table -----------------------------------------------------------------------------

def test_board_table_states_the_figure_and_its_source():
    p300, p150 = df.BOARDS["p300"], df.BOARDS["p150"]
    assert (p300.dies, p300.per_die_gb_s) == (2, 512.0)
    assert (p150.dies, p150.per_die_gb_s) == (1, 512.0)
    for b in df.BOARDS.values():
        assert "tenstorrent.github.io" in b.source and "GB/s" in b.source


def test_floor_for_board_uses_the_table():
    f = df.floor_for_board(DENSE, policy(), "p300")
    assert f.dies == 2 and f.per_die_bandwidth_gb_s == 512.0
    assert f.board == "p300"
    assert df.floor_for_board(DENSE, policy(), "p300", dies=1).dies == 1     # one die of the board
    with pytest.raises(df.FloorInputError, match="board"):
        df.floor_for_board(DENSE, policy(), "p999")


# ---- share of the floor --------------------------------------------------------------------------

def test_share_of_floor_is_measured_over_floor():
    f = df.decode_floor(DENSE, policy(), 1, 1.0)
    assert df.share_of_floor(f.floor_tok_s / 4, f) == pytest.approx(0.25)
    assert df.share_of_floor(0, f) == 0.0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, "20", None])
def test_share_of_floor_refuses_bad_speeds(bad):
    f = df.decode_floor(DENSE, policy(), 1, 1.0)
    with pytest.raises(df.FloorInputError, match="measured"):
        df.share_of_floor(bad, f)


def test_share_text_names_the_estimate_and_flags_a_share_above_one():
    f = df.decode_floor(DENSE, policy(), 1, 1.0)
    low = df.share_text(f.floor_tok_s * 0.25, f)
    assert "25%" in low and "estimated floor" in low
    high = df.share_text(f.floor_tok_s * 2, f)
    assert "above the floor" in high and "speculative" in high


def test_low_share_hint_lists_causes_in_the_stated_order_and_only_below_the_threshold():
    assert df.low_share_hint(0.60, threshold=0.5) is None
    hint = df.low_share_hint(0.10, threshold=0.5)
    order = ["op count", "weight streaming", "collectives", "one-row", "host round trips"]
    positions = [hint.lower().find(w) for w in order]
    assert all(p >= 0 for p in positions) and positions == sorted(positions)
    with pytest.raises(df.FloorInputError):
        df.low_share_hint(0.1, threshold=0)


def test_command_line_honours_an_explicit_bandwidth_given_with_a_board(capsys):
    cfg = str(FIX / "qwen3.8-27b.json")
    assert df.main([cfg, "--board", "p300", "--per-die-gb-s", "256", "--policy", "default=bfp8"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inputs"]["per_die_bandwidth_gb_s"] == 256.0 and out["inputs"]["dies"] == 2 and out["inputs"]["board"] == "p300"
    assert df.main([cfg, "--board", "p300", "--per-die-gb-s", "256", "--dies", "1", "--policy", "default=bfp8"]) == 0
    assert json.loads(capsys.readouterr().out)["inputs"]["dies"] == 1


def test_a_share_exactly_at_the_threshold_gets_no_hint():
    assert df.low_share_hint(0.5, threshold=0.5) is None
    assert df.low_share_hint(0.4999, threshold=0.5) is not None


# ---- the repository's real models ----------------------------------------------------------------
# Checked against the shard headers of each model's safetensors files, summed by tensor group (counted in parameters):
#   Qwen3.8-27B: 27,781,427,952 in all, of which the vision tower holds 460,730,096 and the multi-token-prediction head 424,699,392.
#                The floor counts neither (decode does not read them), and it leaves out the norm vectors (about a million).
#   Qwen3-Coder-Next: 79,674,391,296 in all, with no tower and no extra head; norms and the shared-expert gate (0.3 million) are left out.
HEADER_PARAMS = {"qwen3.8-27b.json": 27_781_427_952 - 460_730_096 - 424_699_392, "qwen3-coder-next.json": 79_674_391_296}


@pytest.mark.parametrize("name", sorted(HEADER_PARAMS))
def test_stored_parameters_match_the_shard_headers_within_a_thousandth(name):
    f = df.decode_floor(mc.read_model_config(FIX / name), policy(), 2, 512.0)
    assert f.stored_params == pytest.approx(HEADER_PARAMS[name], rel=1e-3)


def test_qwen38_27b_reads_every_dense_weight_per_token():
    f = df.decode_floor(mc.read_model_config(FIX / "qwen3.8-27b.json"), policy(), 2, 512.0)
    assert f.active_params == f.stored_params - 248320 * 5120   # all but the embedding table, whose lookup reads one row
    assert f.active_params == pytest.approx(25.62e9, rel=1e-3)
    assert f.by_group["dense_mlp"] == pytest.approx(3 * 5.704e9 * 2, rel=1e-3)   # gate, up and down projections, 5.70e9 each, in bf16


def test_coder_next_reads_about_three_and_a_half_billion_of_eighty():
    f = df.decode_floor(mc.read_model_config(FIX / "qwen3-coder-next.json"), policy(), 2, 512.0)
    assert f.active_params == pytest.approx(3.56e9, rel=1e-2)    # ten of 512 experts per layer, the shared expert, attention, the head
    assert f.stored_params == pytest.approx(79.67e9, rel=1e-2)


# ---- the table printed in the run log ------------------------------------------------------------

def test_render_table_marks_every_row_as_a_share_of_an_estimate():
    f = df.decode_floor(DENSE, policy(), 1, 1.0)
    text = df.render_table([("fake", f, [("plain", f.floor_tok_s / 2)])])
    assert "estimate" in text.lower() and "50.0%" in text
    assert "fake" in text and "plain" in text


def test_command_line_prints_a_floor_and_a_share_and_refuses_missing_bandwidth(capsys):
    cfg = str(FIX / "qwen3.8-27b.json")
    assert df.main([cfg, "--board", "p300", "--policy", "default=bfp8", "--measured", "18.0"]) == 0
    out = capsys.readouterr().out
    assert '"label": "estimate"' in out and "of an estimated floor" in out
    assert df.main([cfg, "--policy", "default=bf16"]) == 2          # no board and no bandwidth: refused
    assert df.main([cfg, "--board", "p300", "--policy", "default=int3"]) == 2
