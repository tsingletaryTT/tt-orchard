# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The probe's `sampled-matches-greedy` verdict (PR 109): sampled replies equal the greedy text or each other, which fits ignored sampling and also happens when one
token is eligible at every step. For a model that needs sampling the choice treats it like `sampled-ignored` (try the drafter-off variant: wrongly trusting it
ships a greedy-only server), but every record, the card and the bundle say the verdict is suspected and not confirmed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_sampling_choice import files, put  # noqa: E402
from test_operator_bundle_template import make_run as make_bundle_run, section  # noqa: E402
from test_package_card import facts  # noqa: E402
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_sampling_choice import drafter_on_run, section as card_section, stage_card  # noqa: E402
from test_sampling_choice import NEEDED, NOT_NEEDED, side, variant  # noqa: E402
from test_sampling_choice_files import GREEDY, make_run, report, req, save  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402
from orchard.card_use import PERF_IN_PROSE  # noqa: E402
from orchard.package_card import render_card  # noqa: E402
from orchard.stages import gate_package  # noqa: E402

MG = "sampled-matches-greedy"


def matches_report(drafter="on (dflash2)", seconds=2.0):
    """A swap check whose probe the merged judge reads as `sampled-matches-greedy`: the sampled replies equal the greedy text and there is no hot request."""
    return report(drafter=drafter, ignored=True, seconds=seconds, hot=False)


# ---- the verdict is read as it is ------------------------------------------------------------------------------------------------------

def test_the_probe_judges_a_record_without_a_hot_request_as_matches_greedy():
    assert sc.side_from_report(matches_report())["verdict"] == MG


def test_the_suspect_set_names_exactly_the_matches_greedy_verdict():
    assert sc.SUSPECT == (MG,)
    assert MG not in sc.APPLIED and MG not in sc.BAD


# ---- the choice -------------------------------------------------------------------------------------------------------------------------

def test_a_model_that_needs_sampling_gets_the_drafter_off_trial_after_a_matches_greedy_verdict():
    assert sc.needs_variant(NEEDED, side(verdict=MG)) is True


def test_a_model_that_does_not_need_sampling_is_not_tried():
    assert sc.needs_variant(NOT_NEEDED, side(verdict=MG)) is False


def test_a_good_drafter_off_variant_is_chosen_after_matches_greedy_with_no_flag():
    d = sc.decide(NEEDED, side(verdict=MG), variant())
    assert d["choice"] == "drafter-off" and d["flag"] is None
    assert d["numbers"]["served_verdict"] == MG


def test_the_reason_says_suspected_and_not_confirmed_and_never_says_ignored():
    d = sc.decide(NEEDED, side(verdict=MG), variant())
    r = d["reason"]
    assert "suspected" in r and "not confirmed" in r and "ignores sampling" not in r and "ignored" not in r.replace("ignores sampled", "")
    assert MG in r


def test_with_no_variant_the_choice_is_unverified_and_not_a_confirmed_failure():
    d = sc.decide(NEEDED, side(drafter="off", verdict=MG, greedy=18.0, sampled=18.0))
    assert d["choice"] == "keep" and d["flag"] == "unverified" and "suspected" in d["reason"]


def test_a_failed_variant_after_matches_greedy_is_unverified_not_no_variant_serves_purpose():
    d = sc.decide(NEEDED, side(verdict=MG), variant(served=False, greedy=None, sampled=None))
    assert d["choice"] == "keep" and d["flag"] == "unverified"


def test_a_variant_that_ignores_sampling_too_keeps_the_original_with_the_flag_for_the_variant_not_the_original():
    d = sc.decide(NEEDED, side(verdict=MG, greedy=59.0, sampled=59.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose" and "suspected" in d["reason"]


def test_a_matches_greedy_variant_does_not_count_as_a_good_variant():
    d = sc.decide(NEEDED, side(), variant(verdict=MG, greedy=18.0, sampled=18.0))
    assert d["flag"] == "unverified"


# ---- the gate ---------------------------------------------------------------------------------------------------------------------------

def test_the_gate_recomputes_a_matches_greedy_decision(tmp_path):
    run, s2 = make_run(tmp_path, stage2=matches_report(), variant=report(drafter="off", ignored=False, seconds=6.0))
    d = save(run, s2)
    assert d["choice"] == "drafter-off" and d["numbers"]["served_verdict"] == MG
    assert sc.decision_problems(s2, run) == []


def test_the_gate_refuses_a_decision_that_calls_the_verdict_confirmed(tmp_path):
    run, s2 = make_run(tmp_path, stage2=matches_report(), variant=report(drafter="off", ignored=False, seconds=6.0))
    d = save(run, s2)
    sc.write_decision(s2, {**d, "numbers": {**d["numbers"], "served_verdict": "sampled-ignored"}}, trial=True)
    assert any("`numbers.served_verdict`" in p for p in sc.decision_problems(s2, run))


# ---- the card ---------------------------------------------------------------------------------------------------------------------------

def test_the_served_clause_for_the_suspect_verdict_is_suspected_wording():
    c = sc.served_clause(MG)
    assert "suspected" in c and "not confirmed" in c and "ignored" not in c


def test_the_card_sentence_says_suspected_and_not_that_the_drafter_ignored_anything():
    s = sc.card_sentence(sc.decide(NEEDED, side(verdict=MG), variant()))
    assert "suspected" in s and "not confirmed" in s and "ignored sampled requests" not in s
    assert not PERF_IN_PROSE.search(s)


def test_the_cards_bullet_for_the_suspect_verdict_makes_no_ignored_claim():
    card = render_card(facts(drafter_off=True, drafter_off_by_sampling=True, sampling_first_verdict=MG))
    b = next(" ".join(ln.split()) for ln in card.splitlines() if ln.startswith("- Speculative decoding is off."))
    assert "suspected" in b and "not confirmed" in b and "ignored" not in b


def test_a_package_after_matches_greedy_gets_the_suspected_wording_and_the_gate_passes_it(world):
    import json
    drafter_on_run(world)
    s2 = world["run"] / "stages/2"
    swap = s2 / "evidence/swap-check.json"
    swap.write_text(json.dumps({**json.loads(swap.read_text()), **matches_report(seconds=1.6)}))      # the probe has no hot request: suspected only
    sc.write_decision(s2, sc.decide(sc.need(world["run"]), sc.served_side(s2), sc.variant_side(s2, world["run"])), trial=True)
    assert sc.served_side(s2)["verdict"] == MG
    stage_dir, pkg, card = stage_card(world)
    body = " ".join(card_section(card).split())
    assert "suspected" in body and "ignored sampled requests" not in body
    assert gate_package(stage_dir, world["run"]).ok


# ---- the bundle -------------------------------------------------------------------------------------------------------------------------

def test_the_bundle_says_the_verdict_is_suspected_and_not_confirmed(tmp_path):
    run = make_bundle_run(tmp_path)
    put(run, sc.decide(NEEDED, side(verdict=MG), variant()))
    results, _ = files(run)
    block = section(results, "## The sampling choice")
    assert "first served is `sampled-matches-greedy`: suspected, not confirmed." in block and "ignores sampled requests" not in block


def test_the_bundle_does_not_say_suspected_for_a_confirmed_verdict(tmp_path):
    run = make_bundle_run(tmp_path)
    put(run, sc.decide(NEEDED, side(verdict="sampled-ignored"), variant()))
    results, _ = files(run)
    assert "suspected" not in section(results, "## The sampling choice")
