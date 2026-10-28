# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/hazards.py: model dimensions that break known kernels, predicted from the config alone (issue 183)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import hazards as hz
from orchard import model_config as mc

FIX = Path(__file__).parent / "fixtures" / "model-configs"

# a tiny dense model whose every dimension is a whole number of tiles and trips nothing
CLEAN = {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 32000}


def cfg(**over):
    return dict(CLEAN, **over)


def names(scan):
    return {h.name for h in scan.hazards}


def test_a_clean_config_has_no_hazards_and_reports_its_tile_counts():
    s = hz.scan_model_hazards(CLEAN, CLEAN, dies=2)
    assert s.hazards == ()
    t = s.facts["tiles"]
    assert t["hidden_size"] == {"value": 128, "tiles": 4.0, "whole": True}
    assert t["intermediate_size"]["tiles"] == 8.0 and t["head_dim"]["tiles"] == 1.0


# ---- routed experts ---------------------------------------------------------------------------------

def test_257_experts_hit_the_256_mask_hazard_and_513_hit_both():
    moe = dict(num_experts_per_tok=2, moe_intermediate_size=32)
    assert names(hz.scan_model_hazards(cfg(num_experts=256, **moe), CLEAN)) == set()
    s = hz.scan_model_hazards(cfg(num_experts=257, **moe), CLEAN)
    assert names(s) == {"routed_experts_above_256"}
    s = hz.scan_model_hazards(cfg(num_experts=513, **moe), CLEAN)
    assert names(s) == {"routed_experts_above_256", "routed_experts_above_512"}


def test_512_experts_hit_only_the_256_hazard():
    s = hz.scan_model_hazards(cfg(num_experts=512, num_experts_per_tok=2, moe_intermediate_size=32), CLEAN)
    assert names(s) == {"routed_experts_above_256"}


def test_the_expert_hazard_suggests_a_router_check_at_a_high_index():
    s = hz.scan_model_hazards(cfg(num_experts=512, num_experts_per_tok=2, moe_intermediate_size=32), CLEAN)
    (h,) = s.hazards
    assert "expert index 511" in h.stage2_check
    assert h.limit == 256 and h.value == 512
    assert h.source and "issue 183" in h.source


def test_a_hazard_the_nearest_bundle_also_has_is_marked_covered():
    moe = cfg(num_experts=512, num_experts_per_tok=2, moe_intermediate_size=32)
    s = hz.scan_model_hazards(moe, moe)
    (h,) = s.hazards
    assert h.also_on_nearest is True
    assert hz.scan_model_hazards(moe, CLEAN).hazards[0].also_on_nearest is False


# ---- vocabulary per die -----------------------------------------------------------------------------

def test_vocabulary_per_die_above_65536_depends_on_the_die_count():
    big = cfg(vocab_size=131072)
    assert names(hz.scan_model_hazards(big, CLEAN, dies=1)) == {"vocab_per_die_above_65536"}
    assert names(hz.scan_model_hazards(big, CLEAN, dies=2)) == set()            # 65,536 per die equals the limit and is allowed
    assert names(hz.scan_model_hazards(cfg(vocab_size=131104), CLEAN, dies=2)) == {"vocab_per_die_above_65536"}
    assert names(hz.scan_model_hazards(cfg(vocab_size=131104), CLEAN, dies=4)) == set()


def test_vocabulary_per_die_rounds_up_when_it_does_not_divide():
    s = hz.scan_model_hazards(cfg(vocab_size=131073), CLEAN, dies=2)             # 65,536.5 per die: one die holds 65,537
    assert "vocab_per_die_above_65536" in names(s)
    assert s.facts["vocab_per_die"] == 65537


def test_vocabulary_not_a_whole_number_of_tiles_is_a_hazard():
    assert names(hz.scan_model_hazards(cfg(vocab_size=32001), CLEAN)) == {"vocab_not_whole_tiles"}
    assert names(hz.scan_model_hazards(cfg(vocab_size=32000), CLEAN)) == set()


def test_dies_must_be_a_positive_whole_number():
    for bad in (0, -1, 1.5, True, "2", None):
        with pytest.raises(hz.HazardInputError, match="dies"):
            hz.scan_model_hazards(CLEAN, CLEAN, dies=bad)


# ---- tile counts ------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,value", [("hidden_size", 130), ("intermediate_size", 250), ("head_dim", 40)])
def test_a_dimension_that_is_not_a_whole_number_of_tiles_is_a_hazard(key, value):
    s = hz.scan_model_hazards(cfg(**{key: value}), CLEAN)
    assert f"{key}_not_whole_tiles" in names(s)
    assert s.facts["tiles"][key]["whole"] is False
    h = next(h for h in s.hazards if h.name == f"{key}_not_whole_tiles")
    assert h.stage2_check and h.value == value


def test_expert_and_shared_widths_are_also_checked_for_tiles():
    moe = cfg(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=100, shared_expert_intermediate_size=96)
    s = hz.scan_model_hazards(moe, CLEAN)
    assert "moe_intermediate_size_not_whole_tiles" in names(s)
    assert "shared_intermediate_size_not_whole_tiles" not in names(s)       # 96 = 3 tiles


def test_a_hidden_size_of_three_tiles_is_a_known_rejected_count():
    s = hz.scan_model_hazards(cfg(hidden_size=96), CLEAN)
    assert names(s) == {"hidden_size_tile_count_rejected"}
    h = s.hazards[0]
    assert h.value == 3 and "boundary kernel" in h.why


def test_a_required_tile_multiple_can_be_supplied_and_is_checked():
    s = hz.scan_model_hazards(CLEAN, CLEAN, tile_multiples={"hidden_size": 8})   # hidden is 4 tiles
    assert names(s) == {"hidden_size_tiles_not_multiple_of_8"}
    assert names(hz.scan_model_hazards(CLEAN, CLEAN, tile_multiples={"hidden_size": 2})) == set()
    with pytest.raises(hz.HazardInputError, match="tile_multiples"):
        hz.scan_model_hazards(CLEAN, CLEAN, tile_multiples={"hidden_size": 0})
    with pytest.raises(hz.HazardInputError, match="tile_multiples"):
        hz.scan_model_hazards(CLEAN, CLEAN, tile_multiples={"banana": 2})


# ---- router top-k and kv heads -----------------------------------------------------------------------

def test_top_k_above_the_nearest_bundles_is_a_hazard():
    nearest = cfg(num_experts=64, num_experts_per_tok=4, moe_intermediate_size=64)
    ok = cfg(num_experts=64, num_experts_per_tok=4, moe_intermediate_size=64)
    bad = cfg(num_experts=64, num_experts_per_tok=8, moe_intermediate_size=64)
    assert names(hz.scan_model_hazards(ok, nearest)) == set()
    s = hz.scan_model_hazards(bad, nearest)
    assert names(s) == {"router_top_k_above_nearest"}
    assert s.hazards[0].limit == 4 and s.hazards[0].value == 8


def test_top_k_above_a_supplied_limit_is_a_hazard_and_there_is_no_built_in_limit():
    moe = cfg(num_experts=64, num_experts_per_tok=10, moe_intermediate_size=64)
    assert names(hz.scan_model_hazards(moe, moe)) == set()                          # no limit known: nothing invented
    s = hz.scan_model_hazards(moe, moe, limits={"router_top_k_max": 8})
    assert names(s) == {"router_top_k_above_limit"}
    assert s.hazards[0].limit == 8
    assert "supplied by the caller" in s.hazards[0].source


def test_top_k_exactly_at_a_supplied_limit_is_not_a_hazard():
    moe = cfg(num_experts=64, num_experts_per_tok=8, moe_intermediate_size=64)
    assert names(hz.scan_model_hazards(moe, moe, limits={"router_top_k_max": 8})) == set()


def test_kv_heads_above_the_nearest_or_a_supplied_limit():
    near = cfg(num_key_value_heads=2)
    assert names(hz.scan_model_hazards(cfg(num_key_value_heads=8), near)) == {"kv_heads_above_nearest"}
    assert names(hz.scan_model_hazards(cfg(num_key_value_heads=8), cfg(num_key_value_heads=8),
                                       limits={"kv_heads_max": 4})) == {"kv_heads_above_limit"}


def test_sliding_window_layers_above_a_supplied_limit():
    sw = cfg(num_hidden_layers=4, layer_types=["sliding_attention"] * 3 + ["full_attention"])
    assert names(hz.scan_model_hazards(sw, sw)) == set()
    assert names(hz.scan_model_hazards(sw, sw, limits={"sliding_window_layers_max": 2})) == {"sliding_window_layers_above_limit"}


def test_the_sliding_window_limit_counts_layers_over_the_whole_model_and_says_so():
    # two runs of three sliding layers: no run exceeds 3, the model has 6; the limit is defined over the whole model
    sw = cfg(num_hidden_layers=8, layer_types=["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"] * 3 + ["full_attention"])
    assert names(hz.scan_model_hazards(sw, sw, limits={"sliding_window_layers_max": 6})) == set()
    s = hz.scan_model_hazards(sw, sw, limits={"sliding_window_layers_max": 5})
    assert names(s) == {"sliding_window_layers_above_limit"} and s.hazards[0].value == 6
    entry = hz.HAZARD_TABLE[[e.name for e in hz.HAZARD_TABLE].index("sliding_window_layers_above_limit")]
    assert "whole model" in entry.why and "per group" not in entry.why


def test_a_mistral_window_is_seen_by_the_scan_through_the_normalised_layers():
    mistral = cfg(model_type="mistral", sliding_window=4096)
    assert names(hz.scan_model_hazards(mistral, mistral, limits={"sliding_window_layers_max": 1})) == {"sliding_window_layers_above_limit"}


def test_an_unknown_limit_name_or_a_bad_limit_is_refused():
    for bad in ({"banana": 1}, {"kv_heads_max": 0}, {"kv_heads_max": True}, {"kv_heads_max": float("nan")}, "x"):
        with pytest.raises(hz.HazardInputError, match="limits"):
            hz.scan_model_hazards(CLEAN, CLEAN, limits=bad)


# ---- mixer types ------------------------------------------------------------------------------------

def test_linear_attention_the_nearest_bundle_lacks_is_a_hazard():
    lin = cfg(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
              linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32)
    assert names(hz.scan_model_hazards(lin, CLEAN)) == {"mixer_linear_attention_not_in_nearest"}
    assert names(hz.scan_model_hazards(lin, lin)) == set()


def test_mla_sparse_attention_and_hyper_connections_are_detected_from_config_keys():
    mla = cfg(kv_lora_rank=64, q_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32)
    assert names(hz.scan_model_hazards(mla, CLEAN)) == {"mixer_multi_latent_attention_not_in_nearest"}
    sparse = cfg(index_topk=2048, index_n_heads=8, index_head_dim=128)
    assert names(hz.scan_model_hazards(sparse, CLEAN)) == {"mixer_sparse_attention_not_in_nearest"}
    hyper = cfg(hc_mult=4)
    assert names(hz.scan_model_hazards(hyper, CLEAN)) == {"mixer_hyper_connections_not_in_nearest"}


def test_each_mixer_hazard_names_its_detection_rule_and_a_check():
    mla = cfg(kv_lora_rank=64, q_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32)
    (h,) = hz.scan_model_hazards(mla, CLEAN).hazards
    assert "kv_lora_rank" in h.why and h.stage2_check


def test_without_a_nearest_bundle_config_the_scan_says_so_and_skips_the_comparisons():
    lin = cfg(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
              linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
              num_experts=512, num_experts_per_tok=2, moe_intermediate_size=64)
    s = hz.scan_model_hazards(lin, None)
    assert names(s) == {"routed_experts_above_256"}                                    # absolute limits still apply
    assert any("nearest" in n for n in s.notes)
    assert s.facts["mixers"] == ["attention", "linear_attention"]


# ---- table, record, errors ----------------------------------------------------------------------------

def test_the_table_has_a_limit_a_reason_and_a_source_for_every_entry():
    assert hz.HAZARD_TABLE
    for e in hz.HAZARD_TABLE:
        assert e.name and e.why and e.source and e.stage2_check
        assert e.name == e.name.lower().replace(" ", "_")
    assert len({e.name for e in hz.HAZARD_TABLE}) == len(hz.HAZARD_TABLE)


def test_every_table_entry_can_be_hit_by_some_config():
    """The table and the scan must agree: an entry no scan can produce is a dead row."""
    hit: set[str] = set()
    moe = dict(num_experts=600, num_experts_per_tok=2, moe_intermediate_size=100, shared_expert_intermediate_size=100)
    probes = [
        (cfg(hidden_size=96), CLEAN, {}),
        (cfg(hidden_size=130, intermediate_size=250, head_dim=40, vocab_size=131105, **moe), CLEAN, {"dies": 1}),
        (cfg(num_experts=64, num_experts_per_tok=8, moe_intermediate_size=64, num_key_value_heads=8),
         cfg(num_experts=64, num_experts_per_tok=4, moe_intermediate_size=64, num_key_value_heads=2), {}),
        (cfg(num_experts=64, num_experts_per_tok=8, moe_intermediate_size=64, num_hidden_layers=4,
             layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "linear_attention"],
             linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
             kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, index_topk=8, hc_mult=4), CLEAN,
         {"limits": {"router_top_k_max": 4, "kv_heads_max": 1, "sliding_window_layers_max": 1}, "tile_multiples": {"hidden_size": 8}}),
    ]
    for model, nearest, kw in probes:
        hit |= {h.name for h in hz.scan_model_hazards(model, nearest, **kw).hazards}
    table = {e.name for e in hz.HAZARD_TABLE}
    assert table <= hit, sorted(table - hit)


def test_record_is_plain_json_and_lists_names_and_checks():
    s = hz.scan_model_hazards(cfg(num_experts=512, num_experts_per_tok=2, moe_intermediate_size=64), CLEAN)
    rec = s.as_record()
    json.dumps(rec, allow_nan=False)
    assert rec["hazards"][0]["name"] == "routed_experts_above_256"
    assert rec["hazards"][0]["stage2_check"]
    assert rec["dies"] == 2 and rec["label"] == "prediction from the config"


def test_bad_configs_are_refused_with_the_hazard_error():
    with pytest.raises(hz.HazardInputError):
        hz.scan_model_hazards({"hidden_size": 5}, CLEAN)
    with pytest.raises(hz.HazardInputError, match="nearest"):
        hz.scan_model_hazards(CLEAN, {"hidden_size": 5})
    with pytest.raises(hz.HazardInputError):
        hz.scan_model_hazards("nope", CLEAN)


def test_the_real_catalog_configs_scan_without_error_and_the_coder_hits_the_expert_hazard():
    coder = mc.read_model_config(FIX / "qwen3-coder-next.json")
    qwen = mc.read_model_config(FIX / "qwen3.8-27b.json")
    s = hz.scan_model_hazards(coder, qwen, dies=2)
    assert "routed_experts_above_256" in names(s)
    assert "mixer_linear_attention_not_in_nearest" not in names(s)      # the 27B bundle implements it
    assert hz.scan_model_hazards(qwen, qwen, dies=2).facts["vocab_per_die"] == 124160


def test_command_line_prints_the_record_and_refuses_a_bad_file(capsys, tmp_path):
    rc = hz.main([str(FIX / "qwen3-coder-next.json"), "--nearest", str(FIX / "qwen3.8-27b.json"), "--dies", "2"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["hazards"][0]["name"] == "routed_experts_above_256"
    assert hz.main([str(tmp_path / "missing.json")]) == 2
    assert "hazards:" in capsys.readouterr().out
