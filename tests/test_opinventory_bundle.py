# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Candidate, verified and pattern rows in the stage 8 bundle's "Ops and kernels" section. The helpers are the ones of
test_operator_bundle_template.py (same run fixture, same script)."""
import json

import pytest

import test_operator_bundle_template as tb


def _cand(op, count, names):
    return {"op": op, "scaled_count": count, "status": "candidate", "modules": [],
            "candidates": [{"ttnn": n, "why": "synonym"} for n in names]}


INV = dict(tb.INVENTORY, summary={"distinct": {"mapped": 3, "composable": 1, "verified": 1, "candidate": 12,
                                               "unmapped": 1, "unchecked": 0, "ignored": 2}},
           ops=([_cand(f"aten.cand_{i:02d}", 500 - i, ["a_name", "b_name"]) for i in range(12)]
                + [{"op": "aten.ver", "scaled_count": 7, "status": "verified", "ttnn": "ver_ttnn", "modules": []},
                   {"op": "aten.zzz_unmapped", "scaled_count": 3, "status": "unmapped", "modules": []},
                   {"op": "aten.mm", "scaled_count": 9999, "status": "mapped", "modules": []}]),
           patterns=[{"pattern": "rms_norm", "ttnn": "rms_norm", "status": "pattern", "scaled_count": 17},
                     {"pattern": "rotary", "ttnn": "r", "status": "pattern-verified", "scaled_count": 8}])


def sec_of(tmp_path, inv):
    return tb.section(tb.results_text(tmp_path, inv), "## Ops and kernels")


def test_candidates_are_listed_as_guesses_with_their_reasons_and_never_as_unmapped_or_mapped(tmp_path):
    sec = sec_of(tmp_path, INV)
    assert "candidate 12" in sec and "verified 1" in sec
    assert "they are not `mapped`" in sec
    listed = [i for i in range(12) if f"aten.cand_{i:02d}" in sec]
    assert listed == list(range(10))                        # the ten most frequent, in order
    assert "`aten.cand_00` (500): `a_name` (synonym), `b_name` (synonym)" in sec
    unmapped_part = sec.split("Unmapped ops")[1].split("Candidate ops")[0]
    assert "aten.zzz_unmapped" in unmapped_part and "aten.cand_" not in unmapped_part
    assert "aten.mm" not in sec


def test_verified_ops_and_op_groups_say_they_are_simulator_checks_and_not_mapped(tmp_path):
    sec = sec_of(tmp_path, INV)
    assert "`aten.ver` -> `ver_ttnn`" in sec
    assert "That is not a run on hardware and the ops are not `mapped`" in sec
    assert "- rms_norm -> `rms_norm` (pattern, 17)" in sec and "- rotary -> `r` (pattern-verified, 8)" in sec
    assert "The ops inside a group keep their own statuses" in sec


def test_a_run_with_no_candidates_or_patterns_has_none_of_those_paragraphs(tmp_path):
    sec = sec_of(tmp_path, tb.INVENTORY)
    assert "Candidate ops" not in sec and "Verified ops" not in sec and "Groups of ops" not in sec


def test_candidate_text_copied_from_the_inventory_is_redacted_and_malformed_rows_are_skipped(tmp_path):
    inv = json.loads(json.dumps(INV))
    inv["ops"][0]["candidates"] = [{"ttnn": "/home/someone/x", "why": f"w {tb.TOKEN}"}, "junk", {"ttnn": 5}, None]
    inv["ops"].append({"op": "aten.nocands", "status": "candidate", "scaled_count": 9000})
    inv["ops"].append({"op": 7, "status": "candidate"})
    inv["patterns"] = [{"pattern": "/home/someone/p", "ttnn": tb.TOKEN, "status": "pattern"}, 5, {"ttnn": "x"}]
    text = tb.results_text(tmp_path, inv)
    assert "/home/someone" not in text and tb.TOKEN not in text
    assert "`aten.nocands` (9000): none recorded" in text and "`7`" not in text
    assert "Traceback" not in text


@pytest.mark.parametrize("patterns", ["x", 5, {"a": 1}, [None, 3]])
def test_a_malformed_patterns_field_cannot_break_the_build(tmp_path, patterns):
    text = tb.results_text(tmp_path, dict(INV, patterns=patterns))
    assert "Traceback" not in text and "## Stage 8" in text and "Groups of ops" not in text
