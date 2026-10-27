# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 114 in the package (issue 107): a choice that rests on a missing verdict or speed gets a card that makes no claim about it, and the gate
passes that card; a faster-of-two choice is packaged drafter off."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_sampling_choice import drafter_on_run, section, stage_card  # noqa: E402
from test_sampling_choice_files import report  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402
from orchard.package import read_run  # noqa: E402
from orchard.stages import gate_package  # noqa: E402


def test_an_unverified_choice_gets_a_card_without_the_unsupported_claims_and_the_gate_passes_it(world):
    dec = drafter_on_run(world, variant=report(drafter="off", with_probe=False))
    assert dec["choice"] == "drafter-off" and dec["flag"] == "unverified"
    stage_dir, pkg, card = stage_card(world)
    body = " ".join(section(card).split())
    assert "documented host-sampling path" in body and "does not claim" in body
    assert "applies sampling and decodes above the usability floor" not in body
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_a_faster_of_two_choice_is_packaged_drafter_off_and_the_card_says_it_does_not_do_what_the_model_needs(world):
    dec = drafter_on_run(world, variant=report(drafter="off", ignored=False, seconds=12.0))      # 8 tokens/s, below the floor; the first run is slower
    # Make the first configuration slower than the variant, so the variant is the faster of the two.
    import json
    swap = world["run"] / "stages/2/evidence/swap-check.json"
    d = json.loads(swap.read_text())
    d.update(report(drafter="on (dflash2)", ignored=True, seconds=19.2))                         # 5 tokens/s
    swap.write_text(json.dumps(d))
    s2 = world["run"] / "stages/2"
    dec = sc.decide(sc.need(world["run"]), sc.served_side(s2), sc.variant_side(s2, world["run"]))
    assert dec["choice"] == "drafter-off" and dec["flag"] == "no-variant-serves-purpose"
    sc.write_decision(s2, dec, trial=True)
    assert read_run(world["run"]).drafter_off is True
    stage_dir, pkg, card = stage_card(world)
    assert "does not do what the model needs" in " ".join(section(card).split())
    assert gate_package(stage_dir, world["run"]).ok
