# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Second review of PR 114, finding 2 (issue 107): the speed the rule uses falls back to the greedy request's when no sampled speed exists (a refused or too
short sampled request). The decision records which one it used, and the stage 8 bundle labels the figure as the speed used for the decision and says so."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_sampling_choice import files, put  # noqa: E402
from test_operator_bundle_template import make_run, section  # noqa: E402
from test_sampling_choice import NEEDED, side, variant  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402


def test_the_basis_is_sampled_when_a_sampled_speed_exists_else_greedy_else_none():
    assert sc.speed_basis({"sampled_tok_s": 5.0, "greedy_tok_s": 9.0}) == "sampled"
    assert sc.speed_basis({"sampled_tok_s": None, "greedy_tok_s": 9.0}) == "greedy"
    assert sc.speed_basis({"sampled_tok_s": 0, "greedy_tok_s": 9.0}) == "greedy"
    assert sc.speed_basis({"sampled_tok_s": None, "greedy_tok_s": None}) is None
    assert sc.speed_basis(None) is None


def test_the_decision_records_the_basis_of_each_speed():
    d = sc.decide(NEEDED, side(sampled=None, greedy=59.0), variant(greedy=18.0, sampled=17.0))
    n = d["numbers"]
    assert n["served_tok_s"] == 59.0 and n["served_speed_basis"] == "greedy"
    assert n["variant_tok_s"] == 17.0 and n["variant_speed_basis"] == "sampled"


def test_no_speed_has_no_basis():
    d = sc.decide(NEEDED, side(sampled=None, greedy=None), variant(greedy=None, sampled=None))
    assert d["numbers"]["served_speed_basis"] is None and d["numbers"]["variant_speed_basis"] is None


def row(results):
    block = section(results, "## The sampling choice")
    return next(ln for ln in block.splitlines() if ln.startswith("| Decode speed"))


def test_the_bundle_labels_the_figure_as_the_speed_used_and_names_a_greedy_fallback(tmp_path):
    run = make_run(tmp_path)
    put(run, sc.decide(NEEDED, side(verdict="sampled-refused", sampled=None, greedy=59.0), variant(greedy=18.0, sampled=17.0)))
    results, _ = files(run)
    r = row(results)
    assert "used for the decision" in r and "of a sampled request" not in r
    assert "59.0 (greedy request" in r and "17.0 (sampled requests" in r


def test_the_bundle_says_when_both_figures_are_sampled(tmp_path):
    run = make_run(tmp_path)
    put(run, sc.decide(NEEDED, side(), variant()))
    r = row(files(run)[0])
    assert r.count("sampled requests") == 2 and "greedy request" not in r


def test_a_decision_written_before_the_basis_existed_still_renders(tmp_path):
    run = make_run(tmp_path)
    d = sc.decide(NEEDED, side(), variant())
    del d["numbers"]["served_speed_basis"], d["numbers"]["variant_speed_basis"]
    put(run, d)
    assert "basis not recorded" in row(files(run)[0])


def test_an_unmeasured_speed_has_no_basis_text(tmp_path):
    run = make_run(tmp_path)
    put(run, sc.decide(NEEDED, side(sampled=None, greedy=None, verdict="not-run"), variant(greedy=None, sampled=None)))
    r = row(files(run)[0])
    assert "greedy request" not in r and "TODO" in r and "basis not recorded" not in r
