# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Third review of PR 114, finding 2 (issue 107): the card's bullet that explains why speculative decoding is off says only what the probe established about
the configuration stage 2 first served. A drafter-off decision can follow an inconclusive, not-run or refused probe, and then the bullet must not claim that
the drafter ignored sampled requests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_card import facts, run_with_evidence  # noqa: E402
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_sampling_choice import drafter_on_run, stage_card  # noqa: E402
from test_sampling_choice_files import report  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402
from orchard.package_card import card_problems, render_card  # noqa: E402
from orchard.stages import gate_package  # noqa: E402


def bullet(card):
    return next(" ".join(ln.split()) for ln in card.splitlines() if ln.startswith("- Speculative decoding is off."))


def make(verdict, **kw):
    return render_card(facts(drafter_off=True, drafter_off_by_sampling=True, sampling_first_verdict=verdict, **kw))


def test_an_ignored_verdict_says_the_drafter_ignored_sampled_requests():
    b = bullet(make("sampled-ignored"))
    assert "ignored sampled requests" in b and "needs sampling" in b


@pytest.mark.parametrize("verdict", ["sampled-refused", "sampled-incoherent", "greedy-refused"])
def test_a_refused_or_bad_verdict_says_sampling_was_not_applied_and_not_that_it_was_ignored(verdict):
    b = bullet(make(verdict))
    assert "did not apply sampling" in b and "ignored" not in b


@pytest.mark.parametrize("verdict", ["inconclusive", "not-run", "malformed", None])
def test_an_unknown_verdict_gets_neutral_wording_that_claims_nothing_about_the_drafter(verdict):
    b = bullet(make(verdict))
    assert "no usable sampling verdict" in b and "ignored" not in b and "did not apply" not in b


def test_the_bullet_and_the_sampling_sentence_use_the_same_clause():
    for verdict in ("sampled-ignored", "sampled-refused", "inconclusive"):
        clause = sc.served_clause(verdict)
        assert clause in bullet(make(verdict))


def test_the_card_check_refuses_a_bullet_that_claims_more_than_the_verdict(tmp_path):
    run = run_with_evidence(tmp_path)
    card = make("inconclusive")
    wrong = make("sampled-ignored")
    ok = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, drafter_off_by_sampling=True, sampling_first_verdict="inconclusive",
                       model_id="Altworld/Hemmingway-1")
    assert not any("speculative decoding" in p.lower() for p in ok), ok
    bad = card_problems(wrong, license_id="cc-by-nc-4.0", run_dir=run, drafter_off_by_sampling=True, sampling_first_verdict="inconclusive",
                        model_id="Altworld/Hemmingway-1")
    assert any("speculative decoding" in p.lower() and "verdict" in p for p in bad), bad


def test_a_package_whose_first_probe_was_not_run_gets_the_neutral_bullet_and_the_gate_passes_it(world):
    swap = world["run"] / "stages/2/evidence/swap-check.json"
    dec = drafter_on_run(world, variant=report(drafter="off", ignored=False, seconds=6.0))
    import json
    d = json.loads(swap.read_text())
    d.pop("sampling_probe")
    swap.write_text(json.dumps(d))
    s2 = world["run"] / "stages/2"
    dec = sc.decide(sc.need(world["run"]), sc.served_side(s2), sc.variant_side(s2, world["run"]))
    assert dec["numbers"]["served_verdict"] == "not-run" and dec["choice"] == "drafter-off" and dec["flag"] == "unverified"
    sc.write_decision(s2, dec, trial=True)
    stage_dir, pkg, card = stage_card(world)
    assert "ignored" not in bullet(card) and "no usable sampling verdict" in bullet(card)
    assert gate_package(stage_dir, world["run"]).ok


def test_a_package_whose_first_probe_ignored_sampling_keeps_the_ignored_bullet(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    assert "ignored sampled requests" in bullet(card) and gate_package(stage_dir, world["run"]).ok


def test_a_vocabulary_trim_keeps_its_own_reason_and_is_not_asked_for_the_sampling_bullet(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts(drafter_off=True, drafter_off_by_sampling=True, sampling_first_verdict="sampled-ignored", vocab=(181760, 248320)))
    assert "token ids" in bullet(card) and "ignored sampled requests" not in bullet(card)
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, drafter_off_by_sampling=True, sampling_first_verdict="sampled-ignored",
                             model_id="Altworld/Hemmingway-1", vocab=(181760, 248320))
    assert not any("why speculative decoding is off" in p for p in problems), problems
