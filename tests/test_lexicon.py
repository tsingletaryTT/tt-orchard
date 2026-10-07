# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The orchard vocabulary (orchard/lexicon.py): one table, checked against the code it describes."""
import re

import pytest

from orchard import lexicon, status, ui
from orchard.stages import STAGES


@pytest.mark.parametrize("state", status.STATES)
def test_every_status_state_has_an_emoji_a_colour_role_and_a_phrase(state):
    e = lexicon.STATES[state]
    assert e.emoji and e.phrase
    assert e.role in ui.ROLES


def test_the_lexicon_has_no_state_the_status_command_cannot_print():
    assert set(lexicon.STATES) == set(status.STATES)


def test_every_stage_has_an_orchard_name_and_an_emoji():
    assert set(lexicon.STAGES) == {s.number for s in STAGES}
    for e in lexicon.STAGES.values():
        assert e.emoji and e.name


def test_orchard_names_are_flavour_and_the_real_stage_name_stays_available():
    assert lexicon.stage_label(0, "intake and delta triage", ui.Style("none", False)) == \
        "0 intake and delta triage"
    assert "survey" in lexicon.stage_label(0, "intake and delta triage", ui.Style("none", True))
    assert "intake and delta triage" in lexicon.stage_label(0, "intake and delta triage", ui.Style("none", True))


@pytest.mark.parametrize("state,word", [("ready-for-operator-review", "ripe"), ("aborted", "fallen")])
def test_terminal_states_have_a_closing_line_that_names_the_real_state(state, word):
    line = lexicon.closing_line(state, ui.Style("none", True), where="/r")
    assert word in line and state in line


def test_a_blocked_end_state_names_its_reason():
    line = lexicon.frost_line("needs-new-model-code", ui.Style("none", True))
    assert "frost" in line and "needs-new-model-code" in line


def test_running_states_have_no_closing_line():
    assert lexicon.closing_line("running", ui.Style("none", True), where="/r") is None


def test_the_actor_table_lists_every_role_the_readme_will_explain():
    for who in ("orchardist", "grafter", "head grower", "seasonal hand", "sheepdog", "almanac", "gate",
                "shed", "harvest basket"):
        assert who in lexicon.ACTORS


def test_no_lexicon_text_uses_the_phrasings_the_house_style_bans():
    text = " ".join(str(v) for table in (lexicon.STATES, lexicon.STAGES) for v in table.values())
    text += " ".join(lexicon.ACTORS.values())
    for banned in (", not ", "rather than", "instead of", "reads as", "honest", "load-bearing"):
        assert banned not in text.lower(), banned
    assert not re.search(r"\bthe one\b", text.lower())


def test_the_refusal_line_lists_every_reason_and_says_nothing_started():
    line = lexicon.refused_line(["credentials-needed", "coder-unusable"], ui.Style("none", True))
    assert "credentials-needed" in line and "coder-unusable" in line and "Nothing was started" in line
    assert lexicon.refused_line(["x"], ui.Style("none", False)).startswith("frost")
