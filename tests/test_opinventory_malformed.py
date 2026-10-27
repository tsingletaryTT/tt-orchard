# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Optional inventory fields of the wrong type must be skipped, not raise: the files are read after an agent wrote them
(bundle, delta.json record) or by a command given any file (verify-candidates)."""
import json

import pytest

import test_operator_bundle_template as tb
from orchard import opinventory as oi
from orchard import opverify as ov

BAD = [7, True, {"a": 1}, 1.5]


def cand_row(value, op="aten.cand"):
    return {"op": op, "scaled_count": 9000, "status": "candidate", "modules": [], "candidates": value}


@pytest.mark.parametrize("bad", BAD + ["x"])
def test_the_bundle_skips_a_candidates_field_that_is_not_a_list(tmp_path, bad):
    inv = dict(tb.INVENTORY, ops=tb.INVENTORY["ops"] + [cand_row(bad)])
    text = tb.results_text(tmp_path, inv)
    assert "Traceback" not in text and "## Stage 8" in text
    sec = tb.section(text, "## Ops and kernels")
    assert "`aten.cand` (9000): none recorded" in sec          # the row is listed, with no names claimed


@pytest.mark.parametrize("bad", BAD)
def test_the_bundle_skips_verified_and_pattern_rows_of_the_wrong_shape(tmp_path, bad):
    inv = dict(tb.INVENTORY, ops=tb.INVENTORY["ops"] + [{"op": bad, "status": "verified", "scaled_count": bad},
                                                        {"op": "aten.v", "status": "verified", "ttnn": bad}],
               patterns=[{"pattern": bad, "ttnn": bad, "status": bad, "scaled_count": bad}, bad])
    text = tb.results_text(tmp_path, inv)
    assert "Traceback" not in text and "## Stage 8" in text


@pytest.mark.parametrize("bad", BAD)
def test_record_in_delta_skips_candidates_and_patterns_of_the_wrong_type(tmp_path, bad):
    sd = tmp_path / "s"
    sd.mkdir()
    inv = {"traced_class": "X", "summary": {"distinct": {}}, "ttnn_dump": None,
           "ops": [cand_row(bad), cand_row([{"ttnn": "t"}, bad, None], op="aten.ok")], "patterns": bad}
    (sd / "ops-inventory.json").write_text(json.dumps(inv))
    (sd / "delta.json").write_text("{}")
    assert oi.record_in_delta(sd) is True
    rec = json.loads((sd / "delta.json").read_text())["inventory"]
    assert rec["patterns"] == []
    got = {c["op"]: c["candidates"] for c in rec["candidate_ops"]}
    assert got == {"aten.cand": [], "aten.ok": ["t"]}


@pytest.mark.parametrize("bad", BAD)
def test_verify_candidates_skips_rows_of_the_wrong_shape(bad):
    inv = {"ops": [cand_row(bad),
                   {"op": "aten.s", "status": "candidate", "candidates": [{"ttnn": "t"}], "sample": {"output_shapes": bad}},
                   {"status": "candidate", "candidates": [{"ttnn": "t"}]},
                   {"op": "aten.n", "status": "candidate", "candidates": [{"ttnn": bad}, {"why": "x"}, bad]}],
           "patterns": [{"pattern": bad, "ttnn": "x", "status": "pattern"}, {"pattern": "rms_norm", "status": "pattern"},
                        bad, {"pattern": "ok", "ttnn": "t", "status": "pattern", "shape": bad, "eps": bad}]}
    cases, _ = ov.select_cases(inv)
    assert [c["id"] for c in cases] == ["aten.s->t", "pattern:ok"]
    assert cases[0]["rows"] == 32 and cases[1]["const"] == []


def test_the_bundle_skips_non_dict_items_inside_a_candidates_list(tmp_path):
    inv = dict(tb.INVENTORY, ops=tb.INVENTORY["ops"] + [cand_row(["junk", None, 3, {"ttnn": "good", "why": "w"}])])
    sec = tb.section(tb.results_text(tmp_path, inv), "## Ops and kernels")
    assert "`aten.cand` (9000): `good` (w)" in sec


@pytest.mark.parametrize("bad", BAD + ["x"])
def test_verify_candidates_skips_a_patterns_field_that_is_not_a_list(bad):
    assert ov.select_cases({"ops": [], "patterns": bad}) == ([], 0)


@pytest.mark.parametrize("bad", BAD)
def test_the_inventory_ignores_verified_entries_of_the_wrong_type(tmp_path, bad):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"entries": {"aten.a": bad, "aten.b": {"ttnn": bad}, "aten.c": {"ttnn": "x"}}}))
    assert oi.load_verified(p) == {"aten.c": {"ttnn": "x"}}
