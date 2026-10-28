# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Second review of PR 114, finding 4 (issue 107): the card's limitation about a drafter-off choice says which checks covered this package. The sampling trial
ran on stage 2's chip count, stage 4 checked this profile's chip count, and only a package that stage 7 booted can claim the boot check. An optional profile
that was not verified claims neither the trial on its own chip count nor stage 7."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_card import facts  # noqa: E402
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_sampling_choice import drafter_on_run, stage_card  # noqa: E402

from orchard.package_card import render_card  # noqa: E402


def limitations(card):
    return " ".join(card.split("## Limitations", 1)[1].split("\n## ", 1)[0].split())


def line(card):
    text = limitations(card)
    return next(s for s in re.split(r"(?<=\.) (?=[A-Z])", text) if "drafter-off configuration this package serves" in s or "Stage 2's own check" in s) \
        if "Stage 2's own check" in text else ""


def test_a_verified_required_profile_claims_the_trial_stage_4_and_the_boot_check():
    card = render_card(facts(drafter_off=True, drafter_off_by_sampling=True, verified=True, chips=2, sampling_trial_chips=2))
    t = limitations(card)
    assert "sampling trial" in t and "stage 4 on 2 chips" in t and "boot check of this package" in t
    assert "did not boot" not in t


def test_an_unverified_profile_on_the_trials_chip_count_does_not_claim_stage_7():
    card = render_card(facts(drafter_off=True, drafter_off_by_sampling=True, verified=False, chips=2, sampling_trial_chips=2))
    t = limitations(card)
    assert "sampling trial and stage 4 on 2 chips. Stage 7 did not boot this package" in t
    assert "boot check of this package" not in t and "ran on" not in t


def test_an_optional_profile_names_the_trials_chip_count_and_its_own():
    card = render_card(facts(drafter_off=True, drafter_off_by_sampling=True, verified=False, chips=4, sampling_trial_chips=2))
    t = limitations(card)
    assert "which ran on 2 chips" in t and "stage 4 on this package's 4 chips. Stage 7 did not boot this package" in t
    assert "boot check of this package" not in t


def test_the_old_blanket_claim_is_gone():
    for v in (True, False):
        card = render_card(facts(drafter_off=True, drafter_off_by_sampling=True, verified=v, chips=2, sampling_trial_chips=2))
        assert "checked by the sampling trial, stage 4 and stage 7" not in limitations(card)


def test_a_card_with_no_sampling_choice_has_no_such_limitation():
    card = render_card(facts(drafter_off=True, drafter_off_by_sampling=False, verified=True, sampling_trial_chips=2))
    assert "sampling trial" not in limitations(card)


def test_an_unknown_trial_chip_count_is_not_invented():
    card = render_card(facts(drafter_off=True, drafter_off_by_sampling=True, verified=False, chips=4, sampling_trial_chips=None))
    t = limitations(card)
    assert "which ran on" not in t and "checked by stage 4 on 4 chips. Stage 7 did not boot this package" in t


def test_the_package_card_of_the_required_boot_checked_profile_claims_the_boot_check(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    t = limitations(card)
    assert "the sampling trial and stage 4 on 2 chips, and by the stage 7 boot check of this package." in t
