# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 441, Copilot round 3: values from config.json that the agreement table copies, or computes with, must never stop stage 0 or make the
evidence file invalid JSON.

Round 2 found three ways a hostile config.json could do that: a float that overflows to infinity (`1e400`) was copied into a row and written as
`Infinity`; the rotary-dimension product overflowed (`round(inf)`, or an integer too large for a float); and, in the same family, an integer
with thousands of digits cannot be written as JSON text at all (Python refuses to turn it into a string). A row whose value cannot be trusted is
"not comparable". Every test here calls the evidence function and then writes the result with `allow_nan=False`, the way the bundle reads it."""
from __future__ import annotations

import json

import pytest

from test_gguf_header import hybrid_config, module, rows_of


def strict(evidence) -> str:
    return json.dumps(evidence, allow_nan=False)


# ---- finding 1: non-finite values ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_config_float_is_not_comparable_and_the_evidence_stays_strict_json(tmp_path, bad):
    rows, e = rows_of(tmp_path, hybrid_config(rms_norm_eps=bad))
    strict(e)
    row = rows["rms norm epsilon"]
    assert row["verdict"] == "not comparable" and row["config"] is None


def test_a_non_finite_float_inside_a_config_list_is_removed_and_the_row_is_not_comparable(tmp_path):
    cfg = hybrid_config()
    cfg["text_config"]["rope_parameters"]["mrope_section"] = [1, float("inf"), 0]
    rows, e = rows_of(tmp_path, cfg)
    strict(e)
    row = rows["rope sections"]
    assert row["verdict"] == "not comparable" and None in row["config"]


def test_a_non_finite_rope_base_in_the_config_is_not_comparable(tmp_path):
    cfg = hybrid_config()
    cfg["text_config"]["rope_parameters"]["rope_theta"] = float("inf")
    rows, e = rows_of(tmp_path, cfg)
    strict(e)
    assert rows["rope base"]["verdict"] == "not comparable" and rows["rope base"]["config"] is None


def test_brief_removes_non_finite_floats_at_any_depth():
    m = module()
    assert m._brief([1.0, float("nan"), [float("inf")]]) == [1.0, None, [None]]
    assert m._brief({"a": [float("-inf")], "b": 2}) == {"a": [None], "b": 2}
    assert m._brief(float("inf")) is None and m._brief(1.5) == 1.5


def test_brief_survives_nesting_far_deeper_than_json_can_write():
    m = module()
    deep = x = []
    for _ in range(5000):
        x.append([])
        x = x[0]
    strict(m._brief(deep))                                    # a RecursionError before the nesting bound


def test_the_evidence_file_is_made_strict_at_the_write_site_too(tmp_path):
    """A float that reaches the evidence by a path that does not go through `_brief` is still removed before the file is written."""
    m = module()
    assert m._finite({"a": [float("inf"), {"b": float("nan")}], "c": 1.0}) == {"a": [None, {"b": None}], "c": 1.0}


# ---- finding 2: rotary arithmetic -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim, factor", [(128, 1e308), (10 ** 400, 0.25), (2 ** 70, 0.5)])
def test_rotary_arithmetic_that_overflows_leaves_the_rope_dimension_row_not_comparable(tmp_path, head_dim, factor):
    cfg = hybrid_config(head_dim=head_dim)
    cfg["text_config"]["rope_parameters"]["partial_rotary_factor"] = factor
    rows, e = rows_of(tmp_path, cfg)                          # OverflowError before the guard for the first two
    strict(e)
    assert rows["rope dimension"]["verdict"] == "not comparable"


def test_an_ordinary_rotary_dimension_still_agrees(tmp_path):
    rows, _ = rows_of(tmp_path, hybrid_config())
    assert rows["rope dimension"]["verdict"] == "agree"


# ---- the same family: integers too large to write ---------------------------------------------------------------------------------


def test_huge_integers_in_the_config_cannot_break_the_json_writer(tmp_path):
    cfg = hybrid_config(linear_num_value_heads=10 ** 4000, linear_value_head_dim=10 ** 4000)
    rows, e = rows_of(tmp_path, cfg)
    strict(e)                                                 # ValueError (integer string conversion limit) before the bound
    assert rows["linear inner size"]["verdict"] == "not comparable"
    assert rows["linear value heads"]["verdict"] == "not comparable"


def test_brief_replaces_an_integer_beyond_64_bits_with_a_short_text():
    m = module()
    shown = m._brief(10 ** 5000)
    assert isinstance(shown, str) and len(shown) < 80 and "bits" in shown
    assert m._brief(2 ** 64 - 1) == 2 ** 64 - 1               # the largest value a GGUF integer key holds is kept


def test_a_list_row_with_a_non_finite_or_oversized_element_is_not_comparable(tmp_path):
    cfg = hybrid_config()
    cfg["text_config"]["rope_parameters"]["mrope_section"] = [1, 10 ** 30, 0]
    rows, e = rows_of(tmp_path, cfg)
    strict(e)
    assert rows["rope sections"]["verdict"] == "not comparable"
