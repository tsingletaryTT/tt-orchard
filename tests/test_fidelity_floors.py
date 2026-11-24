# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""KL and top-1 floors beside the perplexity score in the fidelity rule (Lottolabs study, 2026-11-24, learnings 3 and 5).

`FIDELITY_BAR = 9.9` reads one number, a perplexity ratio, and the LFM2.5 card passes it (perplexity +0.29 percent) with a KL of 0.0458 and a
top-1 agreement of 0.915. The LFM3 card states its own rule: KL at most 0.03 and |perplexity change| at most 1 percent. The default rule here is
the old one; the other rules are named and opt-in."""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import pytest

from orchard import cli, plan, tier_plan
from test_plan import ONLY_BFP4, snap

ROOT = Path(__file__).resolve().parent.parent
TOK = plan.MIN_SCORED_TOKENS


def rec(score=9.95, tokens=TOK, **more):
    return {"score": score, "tokens": tokens, **more}


# ---- the default rule is the one every caller had -----------------------------------------------------

def test_a_bare_score_and_token_count_still_gate_exactly_as_before():
    assert plan.format_gate("bfp4", {"bfp4": (plan.FIDELITY_BAR, TOK)}) == "ok"
    assert plan.format_gate("bfp4", {"bfp4": (plan.FIDELITY_BAR - 0.01, TOK)}) == "below-bar"
    assert plan.format_gate("bfp4", {"bfp4": (10.0, TOK - 1)}) == "too-few-tokens"
    assert plan.format_gate("bfp4", {}) == "unscored"
    assert plan.format_gate("bfp8", {}) == "ok" and plan.format_gate("bf16", None) == "ok"


def test_the_default_rule_ignores_a_kl_and_a_top1_that_a_record_carries():
    assert plan.format_gate("bfp4", {"bfp4": rec(kl=0.5, top1=0.1)}) == "ok"


def test_a_dict_record_with_only_a_score_and_tokens_equals_the_pair():
    assert plan.format_gate("bfp4", {"bfp4": rec(9.89)}) == plan.format_gate("bfp4", {"bfp4": (9.89, TOK)}) == "below-bar"


def test_the_default_rule_is_the_perplexity_ratio_with_no_other_limit():
    r = plan.DEFAULT_RULE
    assert (r.name, r.kl_max, r.top1_min, r.abs_dppl_percent_max) == ("perplexity", None, None, None)


# ---- the LFM3 card's rule -----------------------------------------------------------------------------

def test_the_lfm3_rule_is_named_with_its_two_limits_and_its_source():
    r = plan.RULES["lfm3-card"]
    assert r.kl_max == 0.03 and r.abs_dppl_percent_max == 1.0 and r.top1_min is None
    assert "LFM3" in r.source


def test_the_lfm3_rule_passes_the_lfm3_numbers():
    # LFM3-1.7B card: KL 0.0243, dPPL +0.55 percent
    assert plan.format_gate("bfp4", {"bfp4": rec(9.945, kl=0.0243, dppl_percent=0.55)}, plan.RULES["lfm3-card"]) == "ok"


def test_the_lfm3_rule_fails_the_lfm25_numbers_that_the_perplexity_bar_passes():
    lfm25 = rec(9.971, kl=0.0458, top1=0.915, dppl_percent=0.29)       # LFM2.5-24B-A2B card
    assert plan.format_gate("bfp4", {"bfp4": lfm25}) == "ok"                                  # perplexity ratio alone: passes
    assert plan.format_gate("bfp4", {"bfp4": lfm25}, plan.RULES["lfm3-card"]) == "below-bar"  # KL above 0.03


def test_the_lfm3_rule_boundary_is_inclusive_for_kl():
    assert plan.format_gate("bfp4", {"bfp4": rec(kl=0.03, dppl_percent=0.1)}, plan.RULES["lfm3-card"]) == "ok"
    assert plan.format_gate("bfp4", {"bfp4": rec(kl=0.0300001, dppl_percent=0.1)}, plan.RULES["lfm3-card"]) == "below-bar"


def test_a_missing_kl_is_unscored_and_the_reason_names_the_measure():
    gate, why = plan.gate_detail("bfp4", {"bfp4": rec(dppl_percent=0.1)}, plan.RULES["lfm3-card"])
    assert gate == "unscored" and "kl" in why.lower() and "lfm3-card" in why


def test_dppl_is_derived_from_the_score_when_the_record_has_none():
    # score 10 * ppl_ref / ppl_cand, so the perplexity change is 100 * (10 / score - 1)
    r = plan.RULES["lfm3-card"]
    assert plan.format_gate("bfp4", {"bfp4": rec(9.91, kl=0.01)}, r) == "ok"            # +0.91 percent
    assert plan.format_gate("bfp4", {"bfp4": rec(9.90, kl=0.01)}, r) == "below-bar"     # +1.01 percent: over the card's 1 percent
    assert "+1.01 percent" in plan.gate_detail("bfp4", {"bfp4": rec(9.90, kl=0.01)}, r)[1]    # the derived change is positive when the candidate is worse
    # a stated dppl_percent wins over the derivation (the card's own natural-text number)
    assert plan.format_gate("bfp4", {"bfp4": rec(9.90, kl=0.01, dppl_percent=0.5)}, r) == "ok"


def test_a_negative_perplexity_change_counts_by_its_size():
    r = plan.RULES["lfm3-card"]
    assert plan.format_gate("bfp4", {"bfp4": rec(kl=0.01, dppl_percent=-0.9)}, r) == "ok"
    assert plan.format_gate("bfp4", {"bfp4": rec(kl=0.01, dppl_percent=-1.5)}, r) == "below-bar"


def test_the_perplexity_bar_still_applies_under_a_named_rule():
    assert plan.format_gate("bfp4", {"bfp4": rec(9.5, kl=0.001, dppl_percent=0.1)}, plan.RULES["lfm3-card"]) == "below-bar"


def test_too_few_tokens_still_come_first():
    assert plan.format_gate("bfp4", {"bfp4": rec(10.0, TOK - 1, kl=0.9)}, plan.RULES["lfm3-card"]) == "too-few-tokens"


def test_a_named_rule_does_not_gate_bfp8_or_bf16():
    assert plan.format_gate("bfp8", {}, plan.RULES["lfm3-card"]) == "ok"


# ---- a top-1 floor ------------------------------------------------------------------------------------

def test_a_top1_floor_gates_on_the_agreement():
    r = plan.rule_from(None, top1_min=0.9)
    assert plan.format_gate("bfp4", {"bfp4": rec(top1=0.915)}, r) == "ok"
    assert plan.format_gate("bfp4", {"bfp4": rec(top1=0.899)}, r) == "below-bar"
    assert plan.format_gate("bfp4", {"bfp4": rec(top1=0.9)}, r) == "ok"
    gate, why = plan.gate_detail("bfp4", {"bfp4": rec()}, r)
    assert gate == "unscored" and "top-1" in why


def test_a_named_rule_takes_extra_limits_from_the_caller():
    r = plan.rule_from("lfm3-card", kl_max=0.02, top1_min=0.95)
    assert (r.kl_max, r.top1_min, r.abs_dppl_percent_max) == (0.02, 0.95, 1.0)
    assert plan.format_gate("bfp4", {"bfp4": rec(kl=0.025, top1=0.96, dppl_percent=0.1)}, r) == "below-bar"


@pytest.mark.parametrize("kw", [{"kl_max": 0}, {"kl_max": -0.1}, {"kl_max": float("nan")}, {"kl_max": float("inf")}, {"kl_max": "0.03"}, {"kl_max": True},
                                {"top1_min": 0}, {"top1_min": 1.1}, {"top1_min": -0.5}, {"top1_min": float("nan")}, {"top1_min": "0.9"}])
def test_a_limit_that_is_not_a_usable_number_is_refused(kw):
    with pytest.raises(plan.FidelityInputError):
        plan.rule_from(None, **kw)


def test_an_unknown_rule_name_is_refused_with_the_known_ones_named():
    with pytest.raises(plan.FidelityInputError, match="lfm3-card"):
        plan.rule_from("nope")


@pytest.mark.parametrize("bad", [
    {"kl": -0.01}, {"kl": float("nan")}, {"kl": float("inf")}, {"kl": "0.02"}, {"kl": True}, {"kl": None},
    {"top1": 1.5}, {"top1": -0.1}, {"top1": float("nan")}, {"top1": "0.9"},
    {"dppl_percent": float("nan")}, {"dppl_percent": float("inf")}, {"dppl_percent": "1"}, {"dppl_percent": 10**9},
    {"tokens": 4000.5}, {"tokens": -1}, {"tokens": True}, {"score": float("nan")}, {"score": "9.9"}, {"score": 11}, {"score": -1},
    {"surprise": 1},
])
def test_a_bad_measure_in_a_record_is_refused(bad):
    with pytest.raises(plan.FidelityInputError):
        plan.format_gate("bfp4", {"bfp4": {**rec(), **bad}}, plan.RULES["lfm3-card"])


@pytest.mark.parametrize("bad", [[], "9.9", 9.9, (9.9,), (9.9, 4000, 1), {}, {"tokens": 4000}, [9.9, "4000"]])
def test_a_record_that_is_not_a_pair_or_a_dict_with_a_score_is_refused(bad):
    with pytest.raises(plan.FidelityInputError):
        plan.format_gate("bfp4", {"bfp4": bad})


# ---- rows and the plan --------------------------------------------------------------------------------

def test_a_row_carries_the_reason_a_rule_failed_it(tmp_path):
    res = plan.plan(snap(tmp_path), 4, scores={"bfp4": rec(9.971, kl=0.0458, dppl_percent=0.29)}, rule=plan.RULES["lfm3-card"], **ONLY_BFP4)
    row = next(r for r in res["rows"] if r["name"] == "all bfp4")
    assert row["format_gate"] == "below-bar" and "0.0458" in row["format_gate_reason"] and "0.03" in row["format_gate_reason"]
    assert res["recommendation"] is None
    assert any("all bfp4" in n and "0.0458" in n for n in res["notes"])


def test_the_same_record_under_the_default_rule_recommends_the_row(tmp_path):
    res = plan.plan(snap(tmp_path), 4, scores={"bfp4": rec(9.971, kl=0.0458, dppl_percent=0.29)}, **ONLY_BFP4)
    assert res["recommendation"]["row"] == "all bfp4" and res["recommendation"]["gate"] == "ok"
    row = next(r for r in res["rows"] if r["name"] == "all bfp4")
    assert row["format_gate_reason"] is None


def test_the_plan_names_the_rule_it_used(tmp_path):
    d = snap(tmp_path)
    res = plan.plan(d, 4)
    assert res["fidelity_rule"]["name"] == "perplexity" and res["fidelity_rule"]["kl_max"] is None
    res = plan.plan(d, 4, rule=plan.RULES["lfm3-card"])
    assert res["fidelity_rule"]["name"] == "lfm3-card" and res["fidelity_rule"]["kl_max"] == 0.03


def test_the_rendered_plan_shows_the_rule_when_it_is_not_the_default(tmp_path):
    d = snap(tmp_path)
    text = plan.render(plan.plan(d, 4, rule=plan.RULES["lfm3-card"]))
    assert "fidelity rule: lfm3-card" in text and "KL at most 0.03" in text
    assert "fidelity rule:" not in plan.render(plan.plan(d, 4))


# ---- the command lines --------------------------------------------------------------------------------

def run_plan(d, *extra):
    out = io.StringIO()
    code = plan.main([str(d), "--chips", "4", "--chip-gib", "4.8e-5", "--weight-share", "1", "--json", *extra], out=out)
    return code, out.getvalue()


def test_the_plan_command_takes_kl_top1_and_a_rule(tmp_path):
    d = snap(tmp_path)
    flags = ["--fidelity-bfp4", "9.971", "--fidelity-tokens", "4096", "--fidelity-kl", "0.0458", "--fidelity-top1", "0.915"]
    code, text = run_plan(d, *flags)
    assert code == 0 and json.loads(text)["recommendation"]["gate"] == "ok"
    code, text = run_plan(d, *flags, "--fidelity-rule", "lfm3-card")
    assert code == 0 and json.loads(text)["recommendation"] is None
    code, text = run_plan(d, *flags, "--fidelity-top1-min", "0.95")
    assert code == 0 and json.loads(text)["recommendation"] is None


@pytest.mark.parametrize("extra", [
    ["--fidelity-kl", "0.01"],                                         # a KL with no score
    ["--fidelity-bfp4", "9.95", "--fidelity-tokens", "4096", "--fidelity-kl", "-1"],
    ["--fidelity-bfp4", "9.95", "--fidelity-tokens", "4096", "--fidelity-top1", "2"],
    ["--fidelity-bfp4", "9.95", "--fidelity-tokens", "4096", "--fidelity-kl", "nan"],
    ["--fidelity-bfp4", "9.95", "--fidelity-tokens", "4096", "--fidelity-rule", "lfm3-card", "--fidelity-kl-max", "0"],
    ["--fidelity-rule", "lfm3-card", "--fidelity-kl-max", "inf"],
])
def test_the_plan_command_refuses_a_bad_fidelity_input_with_one_line(tmp_path, capsys, extra):
    code, _ = run_plan(snap(tmp_path), *extra)
    err = capsys.readouterr().err
    assert code == 2 and err.count("\n") == 1 and "Traceback" not in err


def test_the_orchard_plan_wrapper_passes_every_fidelity_flag_through(tmp_path):
    """The wrapper in orchard/cli.py used to build the plan's argument list by hand and dropped --fidelity-bfp4 and --fidelity-tokens."""
    base = ["plan", str(snap(tmp_path)), "--chips", "4", "--chip-gib", "4.8e-5", "--weight-share", "1", "--json"]
    out = io.StringIO()
    code = cli.main(base + ["--fidelity-bfp4", "9.971", "--fidelity-tokens", "4096", "--fidelity-kl", "0.0458", "--fidelity-top1", "0.915",
                            "--fidelity-dppl-percent", "0.29", "--fidelity-rule", "lfm3-card", "--fidelity-kl-max", "0.05"], stdout=out)
    res = json.loads(out.getvalue())
    assert code == 0 and res["fidelity_rule"]["name"] == "lfm3-card+limits" and res["fidelity_rule"]["kl_max"] == 0.05
    assert next(r for r in res["rows"] if r["name"] == "all bfp4")["format_gate"] == "ok"        # KL 0.0458 is within the caller's 0.05


def test_the_wrapper_with_no_fidelity_flag_still_gives_the_default_rule(tmp_path):
    out = io.StringIO()
    code = cli.main(["plan", str(snap(tmp_path)), "--chips", "4", "--json"], stdout=out)
    assert code == 0 and json.loads(out.getvalue())["fidelity_rule"]["name"] == "perplexity"


# ---- tier-plan ----------------------------------------------------------------------------------------

CFG = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
       "vocab_size": 1000, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32}


def test_tier_plan_rows_carry_the_rule_outcome_and_the_reason():
    res = tier_plan.plan(CFG, chips_list=(1,), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0,
                         scores={"bfp4": rec(9.971, kl=0.0458, dppl_percent=0.29)}, rule=plan.RULES["lfm3-card"])
    row = next(r for r in res["rows"] if r["name"] == "all bfp4")
    assert row["format_gate"] == "below-bar" and "0.0458" in row["format_gate_reason"]
    assert res["fidelity_rule"]["name"] == "lfm3-card"
    plain = tier_plan.plan(CFG, chips_list=(1,), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0, scores={"bfp4": rec(9.971, kl=0.0458)})
    assert next(r for r in plain["rows"] if r["name"] == "all bfp4")["format_gate"] == "ok"


def test_the_tier_plan_command_takes_the_same_flags(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG))
    out = io.StringIO()
    code = tier_plan.main([str(cfg), "--chips", "1", "--chip-gib", "32", "--json", "--fidelity-bfp4", "9.971", "--fidelity-tokens", "4096",
                           "--fidelity-kl", "0.0458", "--fidelity-rule", "lfm3-card"], out=out)
    res = json.loads(out.getvalue())
    assert code == 0 and next(r for r in res["rows"] if r["name"] == "all bfp4")["format_gate"] == "below-bar"


def test_the_tier_plan_command_refuses_a_bad_kl(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG))
    code = tier_plan.main([str(cfg), "--chips", "1", "--fidelity-bfp4", "9.95", "--fidelity-tokens", "4096", "--fidelity-kl", "-2"], out=io.StringIO())
    assert code == 2 and "Traceback" not in capsys.readouterr().err


# ---- scripts/fidelity_score.py ------------------------------------------------------------------------

def test_the_score_script_reports_the_perplexity_change_and_the_flags_to_paste():
    import importlib.util
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("fidelity_score", ROOT / "scripts" / "fidelity_score.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    res = fs.fidelity_from_logprobs([-1.0] * 100, [-1.0101] * 100)
    res.update(kl_full=0.0123, top1_agree=0.97)
    extra = fs.plan_inputs(res)
    assert extra["dppl_percent"] == pytest.approx(100 * (10 / res["fidelity"] - 1))
    flags = extra["plan_flags"]
    assert f"--fidelity-bfp4 {res['fidelity']:.4f}" in flags and "--fidelity-tokens 100" in flags
    assert "--fidelity-kl 0.0123" in flags and "--fidelity-top1 0.97" in flags


def test_a_scored_run_carries_the_perplexity_change_and_the_flags_to_paste():
    """score_candidate is what `run` calls for each candidate; its result must carry the plan inputs, not only plan_inputs on its own."""
    import importlib.util
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("fidelity_score", ROOT / "scripts" / "fidelity_score.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)

    class Out:
        def __init__(self, logits):
            self.logits = logits

    class Fake:
        def __init__(self, scale):
            self.scale = scale

        def __call__(self, full):
            n = full.shape[1]
            base = torch.zeros(1, n, 5)
            base[0, :, 2] = 3.0 * self.scale          # a peaked distribution on token 2
            return Out(base)

    sample = torch.tensor([[1, 2, 2, 2, 2, 2]])
    res = fs.score_candidate(Fake(1.0), Fake(0.9), [sample], [2])
    assert res["tokens"] == 4 and res["plan_flags"].startswith("--fidelity-bfp4 ") and "--fidelity-kl" in res["plan_flags"]
    assert res["dppl_percent"] == pytest.approx(100 * (10 / res["fidelity"] - 1))
