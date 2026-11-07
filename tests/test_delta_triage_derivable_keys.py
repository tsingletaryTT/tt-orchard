# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 does not count a config key as a structural delta when a newer transformers release wrote it and its value follows from
other fields (issue 537).

Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged (saved by transformers 5.5) against Qwen/Qwen3-32B (saved by 4.51): the only
differences are `layer_types` (64 x full_attention) and `dtype` against `torch_dtype`. Stage 0 called it a full port. The two real
configs are in tests/fixtures/model-configs. The allow-list is DERIVABLE_DEFAULTS in the template; each entry has a condition, and
these tests check that the condition is what keeps a real difference a delta."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
FIX = ROOT / "fixtures" / "model-configs"


def template():
    spec = importlib.util.spec_from_file_location("delta_triage_derivable_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real():
    return (json.loads((FIX / "berkelium-qwen3-32b-cadquery-v1-merged.json").read_text()),
            json.loads((FIX / "qwen3-32b.json").read_text()))


def compare(m, b):
    return template().compare_config_dicts(m, b, [], "m/config.json", "b/config.json")


def test_real_pair_has_no_decisive_difference_and_reports_what_was_set_aside():
    m, b = real()
    c = compare(m, b)
    assert c["decisive_keys_differing"] == []
    assert c["decisive_keys_equal"] is True
    set_aside = {x["key"]: x for x in c["derivable_keys_ignored"]}
    assert set(set_aside) == {"layer_types", "dtype"}
    assert "full_attention" in set_aside["layer_types"]["why"]
    assert set_aside["dtype"]["value"] == "bfloat16"
    # the raw difference is still on record
    assert "layer_types" in c["text_config_diffs"]


def test_real_pair_reaches_weights_only_in_decide_path():
    t = template()
    m, b = real()
    c = compare(m, b)
    assert t.decide_path(c, {"errors": [], "normalized_name_sets": {"common": 1, "unexplained_only_in_model": [], "unexplained_only_in_base": []},
                             "shape_diffs_normalized": [], "dtype_diffs_normalized": []}) == []


def test_finding_text_names_the_ignored_keys():
    m, b = real()
    text = template().config_finding(compare(m, b))
    assert "layer_types" in text and "Set aside" in text


@pytest.mark.parametrize("entry", ["sliding_attention", "linear_attention"])
def test_a_non_full_attention_entry_is_still_a_delta(entry):
    m, b = real()
    m["layer_types"][10] = entry
    c = compare(m, b)
    assert c["decisive_keys_differing"] == ["layer_types"]
    assert all(x["key"] != "layer_types" for x in c["derivable_keys_ignored"])


def test_all_full_attention_with_a_sliding_window_in_use_is_still_a_delta():
    m, b = real()
    m["sliding_window"], m["use_sliding_window"] = 4096, True
    assert compare(m, b)["decisive_keys_differing"] == ["layer_types"]


def test_wrong_length_is_still_a_delta():
    m, b = real()
    m["layer_types"] = m["layer_types"][:-1]
    assert compare(m, b)["decisive_keys_differing"] == ["layer_types"]


def test_linear_attention_hint_on_either_side_is_still_a_delta():
    m, b = real()
    b["full_attention_interval"] = 4
    assert compare(m, b)["decisive_keys_differing"] == ["layer_types"]


def test_different_model_types_are_still_a_delta():
    m, b = real()
    m["model_type"] = "qwen3_next"
    assert compare(m, b)["decisive_keys_differing"] == ["layer_types"]


def test_layer_types_present_on_both_sides_and_different_is_a_delta():
    m, b = real()
    b["layer_types"] = ["full_attention"] * 63 + ["sliding_attention"]
    assert compare(m, b)["decisive_keys_differing"] == ["layer_types"]


def test_other_differences_stay_decisive():
    m, b = real()
    m["num_hidden_layers"] = 62
    c = compare(m, b)
    assert "num_hidden_layers" in c["decisive_keys_differing"]


def test_dtype_with_a_different_value_is_not_set_aside():
    m, b = real()
    m["dtype"] = "float16"
    c = compare(m, b)
    assert all(x["key"] != "dtype" for x in c["derivable_keys_ignored"])


def test_unrelated_extra_key_is_not_set_aside():
    m, b = real()
    m["some_new_key"] = 1
    c = compare(m, b)
    assert "some_new_key" in c["text_config_diffs"]
    assert all(x["key"] != "some_new_key" for x in c["derivable_keys_ignored"])


def test_identical_configs_set_nothing_aside():
    _, b = real()
    c = compare(copy.deepcopy(b), b)
    assert c["derivable_keys_ignored"] == []
