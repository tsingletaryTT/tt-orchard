# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7 follows stage 2's sampling decision (orchard/sampling_choice.py, issue 107). A model whose stage 2 configuration had the drafter on and ignored
sampling, with a decision of drafter-off, is packaged with the drafter off and host sampling, as a model with no mtp.* tensors is. The card states the choice
and the reason, takes its Sampling rows from the configuration the package serves, and the gate refuses a card that leaves the choice out."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_graft_auto import finished_auto, rows  # noqa: E402
from test_sampling_choice_files import report, write as write_json  # noqa: E402
from package_fakes import write  # noqa: E402

from orchard import package, sampling_choice as sc  # noqa: E402
from orchard.package import PackageError, read_run  # noqa: E402
from orchard.stages import gate_package  # noqa: E402

GEN_NEEDED = {"model_generation_config": {"temperature": 0.6}, "model_generation_config_present": True, "card_sampling_hints": {}, "card_warns_greedy": []}


def drafter_on_run(w, *, needed=True, variant=None, choose=True):
    """Make the run a model with its own mtp.* tensors whose stage 2 served with the drafter on and whose probe showed sampling ignored. `variant` is the
    drafter-off variant's swap check (None: the default healthy one); `choose` writes the decision."""
    run, src = w["run"], w["src"]
    write(run / "stages/2/run.sh", (src / "run.sh").read_text())
    d = json.loads((run / "stages/0/delta.json").read_text())
    write(run / "stages/0/delta.json", {**d, "mtp": {"nearest_tensors": 3, "new_tensors": 3}})
    write_json(run / "stages/0/evidence/genconfig-license.json", GEN_NEEDED if needed else {**GEN_NEEDED, "model_generation_config": {},
                                                                                         "model_generation_config_present": False})
    old = json.loads((run / "stages/2/evidence/swap-check.json").read_text())
    write(run / "stages/2/evidence/swap-check.json", {**old, **report(drafter="on (dflash2)", ignored=True, seconds=1.6)})
    s2 = run / "stages/2"
    write(s2 / sc.TRIAL_DIR / "variant/evidence/swap-check.json", variant or report(drafter="off", ignored=False, seconds=6.0))
    write(s2 / sc.TRIAL_DIR / sc.TRIAL_RESULT, {"variant": {"served": True}})
    if choose:
        dec = sc.decide(sc.need(run), sc.served_side(s2), sc.variant_side(s2, run))
        sc.write_decision(s2, dec, trial=True)
        return dec


def stage_card(w):
    stage_dir, package_, pkg = finished_auto(w)
    return stage_dir, pkg, (pkg / "README.md").read_text()


# ---- read_run and the package settings ---------------------------------------------------------------------------------------------

def test_without_a_decision_a_drafter_on_stage_2_is_packaged_drafter_on(world):
    drafter_on_run(world, choose=False)
    facts = read_run(world["run"])
    assert facts.drafter_off is False and facts.host_sampling is False


def test_a_keep_decision_leaves_the_package_as_stage_2_served(world):
    drafter_on_run(world, needed=False)
    assert sc.read_decision(world["run"] / "stages/2")["choice"] == "keep"
    facts = read_run(world["run"])
    assert facts.drafter_off is False


def test_a_drafter_off_decision_makes_the_package_drafter_off_with_host_sampling(world):
    dec = drafter_on_run(world)
    assert dec["choice"] == "drafter-off"
    facts = read_run(world["run"])
    assert facts.drafter_off is True and facts.host_sampling is True


def test_the_staged_package_has_the_drafter_off_and_no_device_sampling(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    manifest = json.loads((pkg / "tt_kernel_manifest.json").read_text())
    assert manifest["env"]["QWEN36_DRAFTER"] == "" and not any(k.startswith("QWEN36_DFLASH") or k == "DFLASH_WEIGHTS" for k in manifest["env"])
    assert "sample_on_device_mode" not in (pkg / "run.sh").read_text()


def test_graft_mtp_with_a_drafter_off_decision_is_refused_because_the_settings_are_opposite(world):
    drafter_on_run(world)
    with pytest.raises(PackageError, match="drafter-off"):
        read_run(world["run"], graft_mtp=True)


def test_a_drafter_off_decision_beside_a_graft_decision_is_refused(world):
    drafter_on_run(world)
    write(world["run"] / "stages/2/graft-decision.json", {"choice": "graft", "reason": "r"})
    with pytest.raises(PackageError):
        read_run(world["run"])


def test_stage_7_refuses_a_sampling_decision_the_files_do_not_support(world):
    drafter_on_run(world)
    d = json.loads((world["run"] / "stages/2" / sc.DECISION_FILE).read_text())
    d["choice"] = "keep"
    write(world["run"] / "stages/2" / sc.DECISION_FILE, d)
    with pytest.raises(PackageError, match=sc.DECISION_FILE):
        read_run(world["run"])


# ---- the card ---------------------------------------------------------------------------------------------------------------------

def section(card, head="## Sampling"):
    return card.split(head, 1)[1].split("\n## ", 1)[0]


def test_the_card_states_the_choice_and_the_reason_in_the_sampling_section(world):
    dec = drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    body = " ".join(section(card).split())
    assert " ".join(sc.card_sentence(dec).split()) in body
    assert "chose the drafter-off configuration" in body and "generation_config.json" in body


def test_the_sampling_rows_come_from_the_drafter_off_variant_the_package_serves(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    table = rows(card)
    greedy = table["decode speed, greedy request (stage 2 sampling probe)"]
    assert float(greedy[0].split()[0]) == pytest.approx(16.0)          # 96 tokens in 6 s, the variant's, not the drafter-on run's 60
    assert "stages/2/sampling-trial/variant/evidence/swap-check.json" in greedy[1]
    assert "applied" in section(card) or "different text" in section(card)           # the variant's verdict sentence, not the drafter-on one
    assert "was not applied" not in section(card)


def test_the_cards_drafter_off_reason_is_the_sampling_choice_and_not_a_missing_mtp_head(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    flat = " ".join(card.split())
    assert "needs the MTP tensors of the model" not in flat
    assert "Speculative decoding is off" in flat and "stage 2 chose" in flat.lower()
    assert "Speculative decoding is off, so decode speed is that of plain decoding." in flat


def test_the_card_says_stage_2s_first_check_served_the_drafter_on(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    flat = " ".join(card.split())
    assert "stage 2" in flat.lower() and "with the drafter on" in flat


def test_the_gate_passes_the_finished_package(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_the_gate_refuses_a_card_that_leaves_the_choice_out(world):
    dec = drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    (pkg / "README.md").write_text(card.replace(sc.card_sentence(dec), "Sampling was tested."))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("sampling decision" in r.lower() for r in g.reasons), g.reasons


def test_the_gate_refuses_a_card_that_gives_the_wrong_choice(world):
    dec = drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    (pkg / "README.md").write_text(card.replace("Stage 2 chose the drafter-off configuration.", "Stage 2 kept the configuration it served."))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("sampling decision" in r.lower() for r in g.reasons), g.reasons


def test_a_keep_decision_is_stated_on_the_card_and_the_gate_requires_it(world):
    dec = drafter_on_run(world, needed=False)
    stage_dir, pkg, card = stage_card(world)
    assert "kept the configuration it served" in " ".join(section(card).split())
    assert gate_package(stage_dir, world["run"]).ok
    (pkg / "README.md").write_text(card.replace(sc.card_sentence(dec), "Sampling was tested."))
    assert any("sampling decision" in r.lower() for r in gate_package(stage_dir, world["run"]).reasons)


def test_a_flagged_decision_is_loud_on_the_card(world):
    drafter_on_run(world, variant=report(drafter="off", ignored=True, seconds=6.0))
    dec = sc.read_decision(world["run"] / "stages/2")
    assert dec["flag"] == "no-variant-serves-purpose" and dec["choice"] == "keep"
    stage_dir, pkg, card = stage_card(world)
    assert "does not do what the model needs" in " ".join(section(card).split())


def test_a_run_with_no_decision_gets_the_card_it_got_before(world):
    stage_dir, pkg, card = stage_card(world)
    assert "Stage 2 chose" not in card and "Stage 2 kept the configuration it served" not in card
    assert gate_package(stage_dir, world["run"]).ok
