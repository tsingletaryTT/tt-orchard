# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Huge numbers in the flat-rope path of the port generator (issue 461's rule; the finite ratchet's baseline for orchard/port_template.py).

`float(10**400)` raises OverflowError. The safe JSON reader refuses an integer of more than 100 digits, so a file cannot deliver such a value;
these tests call the planner directly, as the other hostile-value tests of the generator do. PR 791 closed this in the layer-keyed rope path. The older flat path still called `float(theta)` and `float(scaling)` on values that `_number` lets through and printed
`partial_rotary_factor` into a stub reason. A config or a constructed attribute with such a number must give a named stub (or the default
where the value is optional), never an exception, and a reason never carries the digits of a config value.
"""
from __future__ import annotations

import copy

import pytest

from orchard import port_template as pt
from test_port_template import STD_CONFIG, STD_EXCERPTS, headers_of, std_construction, std_weights

BIG = 10 ** 400


def generate(cfg=None, construction=None):
    weights = std_weights()
    return pt.generate(config=cfg if cfg is not None else STD_CONFIG, construction=construction or std_construction(),
                       tensors=headers_of(weights), excerpts=STD_EXCERPTS)


def attention_reasons(p):
    return [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]


def config_with(**changes):
    cfg = copy.deepcopy(STD_CONFIG)
    cfg.update(changes)
    return cfg


@pytest.mark.parametrize("where", ["top", "rope_parameters", "rope_scaling"])
@pytest.mark.parametrize("value", [BIG, -BIG], ids=["10**400", "-10**400"])
def test_a_huge_rope_theta_is_a_named_stub_and_not_an_overflow(where, value):
    cfg = config_with(rope_theta=10000.0) if where == "top" else config_with()
    if where == "top":
        cfg["rope_theta"] = value
    else:
        cfg.pop("rope_theta", None)
        cfg[where] = {"rope_theta": value}
    text, p = generate(cfg)
    reasons = attention_reasons(p)
    assert p.attention is None and reasons and all("rope_theta" in r for r in reasons), (where, reasons)
    assert "0" * 30 not in " ".join(reasons), "a reason must not carry the digits of a config value"


def test_the_rotary_plan_alone_refuses_a_huge_theta_without_raising():
    plan, why = pt._rotary_plan({"head_dim": 16, "num_attention_heads": 2, "rope_theta": BIG}, 32)
    assert plan is None and "rope_theta" in why


@pytest.mark.parametrize("value", [BIG, -BIG], ids=["10**400", "-10**400"])
def test_a_huge_partial_rotary_factor_is_a_named_stub_without_its_digits(value):
    text, p = generate(config_with(partial_rotary_factor=value))
    reasons = attention_reasons(p)
    assert p.attention is None and reasons and all("partial_rotary_factor" in r for r in reasons), reasons
    assert "0" * 30 not in " ".join(reasons)


@pytest.mark.parametrize("value", [BIG], ids=["10**400"])
def test_a_huge_attention_scaling_is_a_named_stub_and_not_an_overflow(value):
    construction = std_construction()
    seen = 0
    for entry in construction["classes"]:
        if entry["class"].endswith("Qwen3Attention"):
            for variant in entry["variants"]:
                variant["attrs"]["scaling"] = value
                seen += 1
    assert seen, "the fixture has an attention class"
    text, p = generate(construction=construction)
    reasons = attention_reasons(p)
    assert p.attention is None and reasons and all("scaling" in r for r in reasons), reasons
    assert "0" * 30 not in " ".join(reasons)


@pytest.mark.parametrize("value", [BIG, -BIG, 0, True, "16"], ids=["10**400", "-10**400", "0", "True", "str"])
def test_a_head_dim_that_is_not_a_small_whole_number_falls_back_to_hidden_over_heads(value):
    text, p = generate(config_with(head_dim=value))
    assert p.attention is not None and p.attention["head_dim"] == 16        # hidden 32 over 2 heads


@pytest.mark.parametrize("value", [BIG, -BIG, 0, 1, 2, True, float("nan"), float("inf")], ids=["10**400", "-10**400", "0", "1", "2", "True", "nan", "inf"])
def test_a_huge_or_out_of_range_norm_epsilon_keeps_the_default(value):
    assert pt._eps({"eps": value}, 1e-6) == 1e-6
    text, p = generate(config_with(rms_norm_eps=value))
    assert p.attention is not None                                           # the config's own eps is not what decides the plan


def test_an_ordinary_flat_rope_config_is_planned_with_a_float_theta_and_scale():
    text, p = generate()
    assert p.attention["theta"] == 10000.0 and isinstance(p.attention["theta"], float)
    assert p.attention["scale"] == 16 ** -0.5 and isinstance(p.attention["scale"], float)
