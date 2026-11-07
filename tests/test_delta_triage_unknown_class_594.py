# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 594: a model that shares no tensor name with its nearest model AND differs in the text config is a full port, not class `unknown`.
The Olmo-Hybrid-7B pilot against Qwen3.5-9B (different hidden size, heads and vocabulary, no common tensor name) got class `unknown`, so the
run paused with 'a person decides' although it had been started with --full-port. `unknown` stays for what stage 0 could not compare."""
import importlib.util
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
spec = importlib.util.spec_from_file_location("delta_triage_594_under_test", TEMPLATE)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

NO_COMMON = {"errors": [], "normalized_name_sets": {"common": 0}, "head": None}
SC = {"problems": [], "sidecars": []}
CFG_DIFF = ["text config hidden_size differs: 3840 vs 4096", "text config num_attention_heads differs: 30 vs 16",
            "the two models share no tensor name, even after the prefix is normalized"]


def test_no_shared_tensor_name_with_a_definite_config_difference_is_full_port():
    assert t.decide_class({"errors": None}, NO_COMMON, CFG_DIFF, SC) == "full-port"


def test_no_shared_tensor_name_and_nothing_else_known_stays_unknown():
    only = ["the two models share no tensor name, even after the prefix is normalized"]
    assert t.decide_class({"errors": None}, NO_COMMON, only, SC) == "unknown"


def test_unreadable_headers_stay_unknown_even_with_a_config_difference():
    bad = dict(NO_COMMON, errors=["m.safetensors: cannot read header"])
    assert t.decide_class({"errors": None}, bad, CFG_DIFF, SC) == "unknown"


def test_an_unreadable_config_stays_unknown():
    assert t.decide_class({"errors": ["config.json: bad"]}, NO_COMMON, CFG_DIFF, SC) == "unknown"
