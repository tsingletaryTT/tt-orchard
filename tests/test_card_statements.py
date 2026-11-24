# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Statements the Lottolabs cards make that a package card now makes too (study of 2026-11-24, learnings 25 and 32): where each measured row came from and
whether it was measured on this package, what the checks do and do not prove, the firmware and tt-kmd the package was tested with, and a one-sentence
summary of the licence's restriction."""
from __future__ import annotations

import pytest

from orchard import package_card as pc
from orchard.package_card import Number, card_problems, render_card
from test_package_card import facts, run_with_evidence


def n(name, stage, label="measured"):
    ev = (f"stages/{stage}/result.json",) if label == "measured" else ()
    return Number(name, 0.9 if label == "measured" else None, "fraction", label, ev)


# ---- where the numbers came from ----------------------------------------------------------------------------------

def test_a_row_measured_by_stage_7_is_said_to_be_measured_on_this_package():
    s = pc.numbers_source_sentence((n("a", 7),))
    assert "`stages/7/`" in s and "measured on this package" in s
    assert "earlier in the run" not in s


def test_rows_from_stages_2_and_4_are_said_to_be_measured_before_the_package_existed():
    s = pc.numbers_source_sentence((n("a", 2), n("b", 4)))
    assert "`stages/2/`, `stages/4/`" in s and "before this package was built" in s
    assert "measured on this package" not in s


def test_both_kinds_of_row_get_both_clauses_in_stage_order():
    s = pc.numbers_source_sentence((n("a", 4), n("b", 7), n("c", 2)))
    assert s.index("`stages/7/`") < s.index("`stages/2/`, `stages/4/`")
    assert "measured on this package" in s and "before this package was built" in s


def test_the_sentence_says_no_row_is_copied_from_elsewhere():
    assert "No row is copied from another package or from the model's authors" in pc.numbers_source_sentence((n("a", 7),))


def test_unmeasured_rows_and_rows_with_other_evidence_get_no_sentence():
    assert pc.numbers_source_sentence((n("a", 7, "TODO"),)) is None
    assert pc.numbers_source_sentence(()) is None
    assert pc.numbers_source_sentence((Number("a", 1, "x", "measured", ("evidence/x.json",)),)) is None
    assert pc.numbers_source_sentence((Number("a", 1, "x", "measured", ()),)) is None


def test_only_the_first_evidence_file_decides_the_stage():
    # the first file is the one the value was read from; a later file under another stage does not change where the number came from
    s = pc.numbers_source_sentence((Number("a", 1, "x", "measured", ("stages/7/result.json", "stages/2/evidence/a.json")),))
    assert "`stages/7/`" in s and "`stages/2/`" not in s


def test_a_stage_path_with_a_prefix_or_a_digit_suffix_does_not_count():
    assert pc.numbers_source_sentence((Number("a", 1, "x", "measured", ("runs/stages/7/result.json",)),)) is None
    assert pc.numbers_source_sentence((Number("a", 1, "x", "measured", ("stages/70/result.json",)),)) is None


def test_the_card_carries_the_sentence_under_expected_performance():
    f = facts(numbers=(n("top1 agreement with the CPU reference (stage 7)", 7), n("top1 agreement (stage 2)", 2)))
    card = render_card(f)
    head = card.index("## Expected performance")
    assert "measured on this package" in card[head:head + 1500] and "No row is copied" in card[head:head + 1500]


# ---- what the checks prove ----------------------------------------------------------------------------------------

def test_every_card_says_what_its_numbers_do_not_show():
    card = render_card(facts())
    assert pc.PROOF_SCOPE in card
    head = card.index("## Expected performance")
    assert card.index(pc.PROOF_SCOPE) > head
    for phrase in ("other prompts", "quality on any benchmark", "output distribution"):
        assert phrase in pc.PROOF_SCOPE


def test_a_card_with_the_new_sentences_still_passes_its_own_check(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts())
    assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=run) == []


def test_the_scope_sentence_holds_no_number_with_a_unit_and_no_commercial_wording():
    assert not pc.PERF_IN_PROSE.search(pc.PROOF_SCOPE)
    for pattern in pc.COMMERCIAL_CLAIMS:
        import re
        assert not re.search(pattern, pc.PROOF_SCOPE, re.I), pattern


# ---- firmware and tt-kmd ------------------------------------------------------------------------------------------

def test_a_package_tested_with_known_firmware_and_kmd_says_so_in_its_limitations():
    card = render_card(facts(firmware="19.17.1", kmd="2.12.0"))
    lim = card[card.index("## Limitations"):card.index("## Risks and safety considerations")]
    assert "firmware 19.17.1" in lim and "tt-kmd 2.12.0" in lim and "other versions are not claimed" in lim


def test_one_known_version_is_stated_alone():
    card = render_card(facts(firmware="19.17.1"))
    assert "firmware 19.17.1" in card and "tt-kmd" not in card.split("## Limitations")[1].split("## Risks")[0]
    card = render_card(facts(kmd="2.12.0"))
    assert "tt-kmd 2.12.0" in card and "firmware 2" not in card


def test_no_version_no_sentence():
    card = render_card(facts())
    assert "firmware" not in card.lower().split("## limitations")[1].split("## risks")[0]


@pytest.mark.parametrize("bad", ["", " ", "19 17", "19.17.1\n- x", "a" * 41, "v|1", "<b>", "`x`", "../x"])
def test_a_firmware_or_kmd_that_is_not_a_version_is_refused(bad):
    if bad == "":
        render_card(facts(firmware=bad))          # an empty value means unknown
        return
    with pytest.raises(ValueError, match="firmware"):
        render_card(facts(firmware=bad))
    with pytest.raises(ValueError, match="tt-kmd"):
        render_card(facts(kmd=bad))


def test_the_versions_are_stated_as_tested_not_required():
    card = render_card(facts(firmware="19.17.1", kmd="2.12.0"))
    assert "required" not in card.split("firmware 19.17.1")[1][:200]


# ---- the licence in one sentence ----------------------------------------------------------------------------------

@pytest.mark.parametrize("lid,words", [("mit", "copyright and licence notice"), ("cc-by-4.0", "Credit is required and changes must be indicated"),
                                       ("cc-by-nc-sa-4.0", "same licence"), ("cc-by-nc-nd-4.0", "adapted versions may not be shared"),
                                       ("cc-by-nc-4.0", "Credit is required")])
def test_each_known_licence_gets_one_sentence_naming_its_restriction(lid, words):
    card = render_card(facts(license_id=lid))
    lic = card[card.index("## License"):card.index("## What it runs")]
    assert words in lic and "This is a summary" in lic and "the licence text at the link governs" in lic


def test_apache_gets_no_extra_restriction_sentence():
    lic = render_card(facts(license_id="apache-2.0")).split("## License")[1].split("## What it runs")[0]
    assert "This is a summary" not in lic


def test_an_unknown_licence_keeps_the_non_commercial_treatment_and_gets_no_summary():
    card = render_card(facts(license_id="other"))
    lic = card[card.index("## License"):card.index("## What it runs")]
    assert "Non-commercial use only." in lic and "This is a summary" not in lic


@pytest.mark.parametrize("lid", ["mit", "cc-by-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0", "cc-by-nc-4.0", "apache-2.0"])
def test_the_restriction_sentence_passes_the_cards_own_check(tmp_path, lid):
    run = run_with_evidence(tmp_path)
    assert card_problems(render_card(facts(license_id=lid)), license_id=lid, run_dir=run) == []


def test_the_restriction_table_covers_every_licence_the_module_names_except_apache():
    assert set(pc.LICENSE_RESTRICTIONS) == set(pc.LICENSES) - {"apache-2.0"}
