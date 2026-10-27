# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package card's Not measured list and its limitation lines follow the measured drafter-acceptance rows, whichever step measured them (issue 107,
review of PR 109): stage 2's graft trial (the `grafted drafter` row) or the sampling probe (the greedy and sampled request rows). Before, only the graft
trial counted, so a card with probe rows still listed the rate as not measured."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_graft_auto import finished_auto, rows  # noqa: E402
from test_drafter_acceptance_card import DRAFTER, decided_with, not_measured  # noqa: E402
from test_package_sampling import RATE_ROWS, put_probe, section  # noqa: E402
from test_sampling_probe import LINE, OTHER, STORY, entry, record  # noqa: E402

from orchard.package import ACCEPTANCE_ROW  # noqa: E402
from orchard.stages import gate_package  # noqa: E402

PROBE = record(greedy=entry(log=LINE), a=entry(text=STORY, log=LINE), b=entry(text=OTHER, log=LINE))
LISTED = "the drafter's acceptance rate on this model"


def card_of(w):
    stage_dir, _, pkg = finished_auto(w)
    return stage_dir, (pkg / "README.md").read_text()


def flat(text):
    return " ".join(text.split())


def limits_of(card):
    return flat(card.split("## Limitations")[1].split("\n## ")[0])


def test_probe_only_measurement(world):
    """The graft trial chose the graft and has no acceptance number; the probe's rows are the only measurement."""
    decided_with(world, None)
    put_probe(world, PROBE)
    stage_dir, card = card_of(world)
    assert RATE_ROWS[0] in rows(card) and ACCEPTANCE_ROW not in rows(card)
    assert LISTED not in not_measured(card)
    assert "Its acceptance rate on this model is in the table below (sampling probe)." in flat(card)
    assert "is in the table above (sampling probe)" in limits_of(card) and "was not measured" not in limits_of(card)
    assert gate_package(stage_dir, world["run"]).ok


def test_graft_trial_only_measurement(world):
    decided_with(world, DRAFTER)
    stage_dir, card = card_of(world)
    assert ACCEPTANCE_ROW in rows(card) and RATE_ROWS[0] not in rows(card)
    assert LISTED not in not_measured(card)
    assert "Its acceptance rate on this model is in the table below (graft trial)." in flat(card)
    assert "is in the table above (graft trial)" in limits_of(card)
    assert gate_package(stage_dir, world["run"]).ok


def test_both_sources_are_named(world):
    decided_with(world, DRAFTER)
    put_probe(world, PROBE)
    stage_dir, card = card_of(world)
    assert ACCEPTANCE_ROW in rows(card) and RATE_ROWS[0] in rows(card)
    assert LISTED not in not_measured(card)
    assert "Its acceptance rate on this model is in the table below (graft trial and sampling probe)." in flat(card)
    assert "is in the table above (graft trial and sampling probe)" in limits_of(card)
    assert gate_package(stage_dir, world["run"]).ok


def test_neither_source_leaves_it_listed_as_not_measured(world):
    decided_with(world, None)
    stage_dir, card = card_of(world)
    assert LISTED in not_measured(card)
    assert "Its acceptance rate on this model is listed under Not measured." in flat(card)
    assert "was not measured" in limits_of(card)
    assert gate_package(stage_dir, world["run"]).ok


def test_a_grafted_package_with_no_rows_still_says_the_rate_was_not_measured(world):
    decided_with(world, None)
    _, card = card_of(world)
    assert "How often the drafter's guesses are accepted on this model was not measured." in limits_of(card)
    assert LISTED in not_measured(card)


def test_the_gate_refuses_a_card_that_lists_a_measured_rate_as_not_measured(world):
    decided_with(world, None)
    put_probe(world, PROBE)
    stage_dir, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    (pkg / "README.md").write_text(card.replace("## Not measured\n", f"## Not measured\n\n- {LISTED}\n", 1))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("Not measured" in r and "acceptance" in r for r in g.reasons), g.reasons


def test_only_a_drafter_off_probe_with_no_session_lines_adds_no_rows(world):
    put_probe(world, record())               # no session lines: the drafter was off
    _, card = card_of(world)
    assert RATE_ROWS[0] not in rows(card) and RATE_ROWS[1] not in rows(card)


def test_a_todo_row_is_not_a_measurement():
    from orchard.package import acceptance_sources
    from orchard.package_card import Number
    assert acceptance_sources([Number(ACCEPTANCE_ROW, None, "fraction", "TODO", ()), Number(RATE_ROWS[0], None, "fraction", "TODO", ())]) == ()
    assert acceptance_sources([Number(RATE_ROWS[1], 0.5, "fraction", "measured", ("x",))]) == ("sampling probe",)
