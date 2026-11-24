# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Looped layers: a config with `loop_config` runs a block of its layers several times for each token, so decode reads that block's
weights several times (Lottolabs study, 2026-11-24, learning 11). The fixture is the public config of Inkling-2-Mini (layers 6 to 17 run
3 times, 48 layer applications per token, 24 layers stored)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import decode_floor as df
from orchard import hazards as hz
from orchard import model_config as mc
from orchard import tier_plan as tp

FIX = Path(__file__).parent / "fixtures" / "model-configs"
INKLING = json.loads((FIX / "inkling-2-mini.config.json").read_text())
POLICY = {"default": "bfp8", "routed_experts": "bfp4"}

# a small dense model, used where a hand count is easier than the real config
TINY = {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "vocab_size": 1024}


def looped(cfg, **loop):
    return dict(cfg, loop_config=dict({"core_start": 1, "core_end": 3, "num_passes": 2}, **loop))


# ---- model_config -----------------------------------------------------------------------------------

def test_the_real_config_runs_layers_6_to_17_three_times_for_48_applications():
    d = mc.dims(INKLING)
    assert d.layers == 24
    assert d.loop_passes == (1,) * 6 + (3,) * 12 + (1,) * 6
    assert sum(d.loop_passes) == 48


def test_a_config_without_loop_config_applies_every_layer_once():
    assert mc.dims(TINY).loop_passes == (1, 1, 1, 1)


def test_loop_config_under_text_config_is_read_too():
    d = mc.dims({"text_config": looped(TINY)})
    assert d.loop_passes == (1, 2, 2, 1)


@pytest.mark.parametrize("loop", [
    {"core_start": 3, "core_end": 3},            # empty block
    {"core_start": 3, "core_end": 2},            # backwards
    {"core_start": -1},                          # before the first layer
    {"core_end": 5},                             # past the last layer (4 layers)
    {"num_passes": 0},
    {"num_passes": -2},
    {"num_passes": 2.5},
    {"num_passes": True},
    {"num_passes": "3"},
    {"num_passes": 10_000},                      # no real model loops that often
    {"core_start": None},
])
def test_a_bad_loop_config_is_refused_with_the_key_named(loop):
    with pytest.raises(mc.ModelConfigError, match="loop_config"):
        mc.dims(looped(TINY, **loop))


@pytest.mark.parametrize("bad", ["yes", 1, 0, [True]])
def test_is_varloop_must_be_true_or_false(bad):
    with pytest.raises(mc.ModelConfigError, match="is_varloop"):
        mc.dims(looped(TINY, is_varloop=bad))


@pytest.mark.parametrize("bad", ["3", 3, [1, 2], True])
def test_a_loop_config_that_is_not_an_object_is_refused(bad):
    with pytest.raises(mc.ModelConfigError, match="loop_config"):
        mc.dims(dict(TINY, loop_config=bad))


def test_a_loop_config_set_to_null_means_no_loop():
    assert mc.dims(dict(TINY, loop_config=None)).loop_passes == (1, 1, 1, 1)


def test_a_loop_whose_pass_count_varies_is_flagged_and_counted_at_its_maximum():
    d = mc.dims(looped(TINY, is_varloop=True))
    assert d.loop_variable is True and d.loop_passes == (1, 2, 2, 1)
    assert mc.dims(looped(TINY, is_varloop=False)).loop_variable is False
    assert mc.dims(TINY).loop_variable is False


# ---- decode_floor -----------------------------------------------------------------------------------

def test_a_loop_multiplies_the_bytes_read_for_the_looped_layers_and_nothing_else():
    flat = df.decode_floor(TINY, {"default": "bf16"}, 1, 100.0)
    loop = df.decode_floor(looped(TINY), {"default": "bf16"}, 1, 100.0)
    d = mc.dims(TINY)
    one_layer = df._mixer_params(d, "full_attention") * 2.0
    mlp = df._mlp_params(d, d.intermediate) * 2.0
    assert loop.by_group["attention"] == flat.by_group["attention"] + 2 * one_layer     # 2 looped layers, 1 extra pass each
    assert loop.by_group["dense_mlp"] == flat.by_group["dense_mlp"] + 2 * mlp
    assert loop.by_group["lm_head"] == flat.by_group["lm_head"]                         # the head runs once, after the loop
    assert loop.bytes_per_token > flat.bytes_per_token


def test_a_loop_multiplies_the_router_and_the_shared_and_routed_experts_per_pass():
    moe = dict(TINY, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64)
    flat = df.decode_floor(moe, {"default": "bf16"}, 1, 100.0).by_group
    loop = df.decode_floor(looped(moe), {"default": "bf16"}, 1, 100.0).by_group
    d = mc.dims(moe)
    for group, per_layer in (("router", d.hidden * d.routed_experts * 2.0),
                             ("routed_experts", d.top_k * df._mlp_params(d, d.moe_intermediate) * 2.0),
                             ("shared_experts", df._mlp_params(d, d.shared_intermediate) * 2.0)):
        assert loop[group] == flat[group] + 2 * per_layer, group      # 2 looped layers, one extra pass each


def test_a_loop_does_not_change_what_is_stored():
    assert df.decode_floor(looped(TINY), {"default": "bf16"}, 1, 100.0).stored_params == \
        df.decode_floor(TINY, {"default": "bf16"}, 1, 100.0).stored_params


def test_without_a_loop_the_count_is_the_one_every_earlier_caller_got():
    a = df.decode_floor(TINY, {"default": "bf16"}, 1, 100.0)
    b = df.decode_floor(dict(TINY, loop_config=None), {"default": "bf16"}, 1, 100.0)
    assert a.as_dict()["bytes_by_group"] == b.as_dict()["bytes_by_group"] and a.bytes_per_token == b.bytes_per_token


def test_the_real_config_streams_about_the_3_72_gb_per_token_the_card_reports():
    f = df.floor_for_board(INKLING, POLICY, "p150")
    # The expert bytes are exact: 48 applications x 8 of 128 experts x 3 x 2304 x 1152 weights x 0.5625 bytes.
    assert f.by_group["routed_experts"] == 48 * 8 * 3 * 2304 * 1152 * 0.5625
    # The card (map g4, BF16 router, BFP8 attention) reports 3.72 GB. The count leaves out the loop controllers (about 0.17 GB) and
    # the relative-position projection (about 0.05 GB), which the config does not describe, so it lands a few percent under.
    assert 3.5e9 < f.bytes_per_token < 3.72e9
    assert abs(f.bytes_per_token - 3.72e9) / 3.72e9 < 0.06


def test_the_real_config_attention_counts_the_output_gate_the_checkpoint_header_shows():
    # attn.gate_attn.weight is [3072, 2304] in every layer, the same size as the query projection: q counted twice as wide.
    d = mc.dims(INKLING)
    assert d.attn_output_gate is True
    per_pass = 2 * 2304 * 3072 + 2 * 2304 * 1024 + 3072 * 2304
    assert df._mixer_params(d, "full_attention") == per_pass


def test_the_summary_records_the_loop_so_a_saved_estimate_can_be_redone():
    m = df.decode_floor(INKLING, POLICY, 1, 512.0).as_dict()["inputs"]["model"]
    assert m["loop_applications"] == 48 and m["loop_passes_by_layer"] == [1] * 6 + [3] * 12 + [1] * 6


# ---- tier_plan --------------------------------------------------------------------------------------

def test_the_fit_plan_says_a_loop_controller_is_not_counted():
    out = tp.caveats(INKLING)
    assert any("loop" in c and "not counted" in c for c in out), out


def test_a_config_without_a_loop_gets_no_loop_caveat():
    assert not any("loop" in c for c in tp.caveats(TINY))


# ---- hazards ----------------------------------------------------------------------------------------

def test_a_looped_model_hits_the_looped_layers_mixer_hazard_against_a_plain_nearest_bundle():
    s = hz.scan_model_hazards(looped(TINY), TINY, dies=2)
    assert "looped_layers" in s.facts["mixers"]
    assert "mixer_looped_layers_not_in_nearest" in {h.name for h in s.hazards}
    why = next(h for h in s.hazards if h.name == "mixer_looped_layers_not_in_nearest")
    assert "loop_config" in why.why and "more than once for each token" in why.why     # the rule that detected it is named


def test_a_plain_model_has_no_looped_layers_mixer():
    s = hz.scan_model_hazards(TINY, TINY, dies=2)
    assert "looped_layers" not in s.facts["mixers"] and s.hazards == ()


def test_a_looped_model_against_a_looped_nearest_bundle_is_not_flagged():
    s = hz.scan_model_hazards(looped(TINY), looped(TINY), dies=2)
    assert "mixer_looped_layers_not_in_nearest" not in {h.name for h in s.hazards}
