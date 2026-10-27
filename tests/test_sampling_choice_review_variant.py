# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 114, finding 1 (issue 107): the drafter-off variant is admitted only when it passes the same checks as any swap check, the stage 2 swap
fields and the prompt suite. A healthy sampling probe on a variant that fails coherence, agreement or the suite must not select it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sampling_choice import NEEDED, side  # noqa: E402
from test_sampling_choice_files import make_run, report, save, write  # noqa: E402
from test_prompt_suite_verdict import LOOP, make_record, with_entry  # noqa: E402

from orchard import defaults, sampling_choice as sc  # noqa: E402

FREE = "Paris is the capital of France and a large city"


def variant_run(tmp_path, **kw):
    return make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0, **kw))


def suite_run(tmp_path, record):
    run, s2 = make_run(tmp_path)
    rep = report(drafter="off", ignored=False, seconds=6.0)
    rep["prompt_suite"] = record
    rep["free_run_text"] = rep["reference_generated_text"] = FREE
    write(s2 / sc.TRIAL_DIR / "variant/evidence/swap-check.json", rep)
    return run, s2


@pytest.mark.parametrize("draft,word", [({"coherent": False}, "coherent"), ({"coherent": None}, "coherent"),
                                         ({"top1_agreement": 0.5}, "top1_agreement"), ({"top1_agreement": None}, "top1_agreement"),
                                         ({"n_tokens": 4}, "n_tokens"), ({"server_ready_s": 0}, "server_ready_s")])
def test_a_variant_that_fails_a_swap_check_is_not_admitted(tmp_path, draft, word):
    run, s2 = variant_run(tmp_path, draft=draft)
    v = sc.variant_side(s2, run)
    assert v["served"] is False and word in v["failure"]
    assert v["greedy_tok_s"] is None and v["sampled_tok_s"] is None and v["verdict"] == "not-run"


def test_the_bar_is_stage_2s_top1_minimum(tmp_path):
    run, s2 = variant_run(tmp_path, draft={"top1_agreement": defaults.SWAP_TOP1_MIN})
    assert sc.variant_side(s2, run)["served"] is True
    run, s2 = variant_run(tmp_path / "b", draft={"top1_agreement": defaults.SWAP_TOP1_MIN - 0.01})
    assert sc.variant_side(s2, run)["served"] is False


def test_a_variant_with_a_blocking_prompt_suite_finding_is_not_admitted(tmp_path):
    run, s2 = suite_run(tmp_path, with_entry(make_record(), "arithmetic", content=LOOP))
    v = sc.variant_side(s2, run)
    assert v["served"] is False and "prompt suite" in v["failure"] and "arithmetic" in v["failure"]


def test_a_healthy_variant_with_a_healthy_prompt_suite_is_admitted(tmp_path):
    run, s2 = suite_run(tmp_path, make_record())
    assert sc.variant_side(s2, run)["served"] is True


def test_without_a_run_directory_only_the_swap_fields_are_checked(tmp_path):
    run, s2 = variant_run(tmp_path, draft={"coherent": False})
    assert sc.variant_side(s2)["served"] is False
    run, s2 = suite_run(tmp_path / "b", with_entry(make_record(), "arithmetic", content=LOOP))
    assert sc.variant_side(s2)["served"] is True


def test_the_rule_keeps_the_original_configuration_when_the_variant_fails_its_checks(tmp_path):
    run, s2 = variant_run(tmp_path, draft={"coherent": False})
    d = sc.decide(NEEDED, side(), sc.variant_side(s2, run))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"
    assert "coherent" in d["reason"] and "failed" in d["reason"]


def test_the_gate_recomputes_the_same_keep_and_refuses_a_forged_drafter_off(tmp_path):
    run, s2 = variant_run(tmp_path, draft={"coherent": False})
    d = save(run, s2)
    assert d["choice"] == "keep"
    sc.write_decision(s2, d, trial=True)
    assert sc.decision_problems(s2, run) == []
    sc.write_decision(s2, {**d, "choice": "drafter-off", "flag": None}, trial=True)
    assert any("drafter-off" in p for p in sc.decision_problems(s2, run))


def test_a_swap_check_with_no_result_draft_is_not_admitted(tmp_path):
    run, s2 = make_run(tmp_path)
    rep = report(drafter="off", ignored=False, seconds=6.0)
    del rep["result_draft"]
    write(s2 / sc.TRIAL_DIR / "variant/evidence/swap-check.json", rep)
    v = sc.variant_side(s2, run)
    assert v["served"] is False and "serves" in v["failure"]


def test_a_suite_failure_gives_keep_in_the_save_path_with_the_run_directory(tmp_path):
    run, s2 = suite_run(tmp_path, with_entry(make_record(), "arithmetic", content=LOOP))
    d = save(run, s2)
    assert d["choice"] == "keep" and "prompt suite" in d["reason"]
    assert sc.decision_problems(s2, run) == []
