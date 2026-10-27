# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 114, finding 5 (issue 107): the card's warnings point readers to the sampling decision, so the package ships it. The card cites the packaged
path, the evidence extraction copies the file (scrubbed of machine paths like every evidence file), and the gate refuses a package whose copy is missing."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_sampling_choice import drafter_on_run, section, stage_card  # noqa: E402
from test_sampling_choice import NEEDED, NOT_NEEDED, side, variant  # noqa: E402
from test_sampling_choice_files import report  # noqa: E402

from orchard import package_card, sampling_choice as sc  # noqa: E402
from orchard.stages import gate_package  # noqa: E402

PACKAGED = "evidence/stages/2/sampling-decision.json"


def decisions():
    return [sc.decide(NEEDED, side(), variant()),                                                       # chosen, verified
            sc.decide(NOT_NEEDED, side()),                                                              # kept, not needed
            sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=9.0, sampled=8.0)),   # below-floor
            sc.decide(NEEDED, side(), variant(verdict="inconclusive", greedy=None, sampled=None)),       # unverified
            sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), variant(greedy=8.0, sampled=8.0)),       # no-variant-serves-purpose, kept
            sc.decide(NEEDED, side(), None, variant_note="x")]                                          # variant-not-tried


@pytest.mark.parametrize("d", decisions(), ids=lambda d: f"{d['choice']}-{d['flag']}")
def test_every_card_sentence_cites_the_packaged_decision_file(d):
    assert f"`{PACKAGED}`" in sc.card_sentence(d)


@pytest.mark.parametrize("d", decisions(), ids=lambda d: f"{d['choice']}-{d['flag']}")
def test_a_warning_points_to_the_packaged_path_and_not_to_a_bare_file_name(d):
    s = sc.card_sentence(d).replace(f"`{PACKAGED}`", "")
    assert "`sampling-decision.json`" not in s


def test_the_decision_path_constant_is_the_run_relative_path():
    assert sc.DECISION_PATH == "stages/2/sampling-decision.json"
    assert PACKAGED == "evidence/" + sc.DECISION_PATH


def test_the_evidence_paths_of_a_card_include_the_decision_file():
    d = sc.decide(NEEDED, side(), variant())
    card = "# T\n\n## Sampling\n\n" + sc.card_sentence(d) + "\n\n## Numbers\n\n"
    assert package_card.card_evidence_paths(card) == [sc.DECISION_PATH] or sc.DECISION_PATH in package_card.card_evidence_paths(card)


def test_a_card_with_no_decision_cites_no_decision_file():
    assert sc.DECISION_PATH not in package_card.card_evidence_paths("# T\n\n## Sampling\n\nSampling was tested.\n")


def test_the_package_ships_the_decision_file_scrubbed_of_machine_paths(world):
    drafter_on_run(world)
    run = world["run"]
    d = json.loads((run / "stages/2" / sc.DECISION_FILE).read_text())
    d["need_evidence"] = [f"the card said so (read from {run}/stages/2/evidence)"]
    (run / "stages/2" / sc.DECISION_FILE).write_text(json.dumps(d))
    stage_dir, pkg, card = stage_card(world)
    shipped = pkg / PACKAGED
    assert shipped.is_file()
    text = shipped.read_text()
    assert str(run) not in text and "<RUN_DIR>" in text
    assert json.loads(text)["choice"] == "drafter-off"


def test_the_card_cites_the_packaged_file_and_the_gate_passes(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    assert f"`{PACKAGED}`" in " ".join(section(card).split())
    assert gate_package(stage_dir, world["run"]).ok


def test_the_gate_refuses_a_package_whose_decision_copy_is_missing(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    (pkg / PACKAGED).unlink()
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("sampling-decision.json" in r for r in g.reasons), g.reasons


def test_a_keep_decision_ships_the_file_too(world):
    drafter_on_run(world, needed=False)
    stage_dir, pkg, card = stage_card(world)
    assert (pkg / PACKAGED).is_file() and gate_package(stage_dir, world["run"]).ok


def test_the_optional_profiles_card_does_not_cite_it(world):
    """Only the required profile's card has the Sampling section, so only its package carries the file."""
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    other = [p for p in (world["run"] / "stages/7/package").iterdir() if p.is_dir() and p != pkg]
    for o in other:
        assert not (o / PACKAGED).exists()


def test_a_mention_outside_the_sampling_section_is_not_collected():
    card = f"# T\n\n## Intended use\n\nThe decision and the numbers behind it are in `{PACKAGED}`.\n\n## Sampling\n\nSampling was tested.\n"
    assert package_card.card_evidence_paths(card) == []


def test_only_the_cited_sentence_form_is_collected():
    card = f"# T\n\n## Sampling\n\nLook in `{PACKAGED}` for something else.\n"
    assert package_card.card_evidence_paths(card) == []


def test_a_path_the_numbers_table_already_cites_is_listed_once():
    d = sc.decide(NEEDED, side(), variant())
    card = ("# T\n\n## Sampling\n\n" + sc.card_sentence(d) + "\n\n## Expected performance\n\nEvery number is labelled.\n\n| Number | Value | Label | Evidence |\n|---|---|---|---|\n"
            f"| x | 1 | measured | `{sc.DECISION_PATH}` |\n\n## Limitations\n")
    assert package_card.card_evidence_paths(card).count(sc.DECISION_PATH) == 1
