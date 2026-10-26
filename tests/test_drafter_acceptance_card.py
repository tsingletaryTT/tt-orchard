# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package card and the stage 8 bundle show the drafter's acceptance when stage 2's graft trial measured it, and say it was not measured
when the trial has no number (issue 14). Fixtures come from the graft package and bundle tests."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_graft_auto import CANDIDATE, finished_auto, rows, trial  # noqa: E402
from package_fakes import write  # noqa: E402

from orchard.graft_trial import DECISION_FILE, TRIAL_DIR, TRIAL_RESULT, decide, write_decision  # noqa: E402
from orchard.package import ACCEPTANCE_ROW  # noqa: E402
from orchard.stages import gate_package  # noqa: E402

DRAFTER = {"accept_rate": 0.3125, "accepted_per_iter": 2.1875, "draft_length": 7, "iters": 800, "sessions": 12}


def decided_with(w, drafter):
    s2 = w["run"] / "stages" / "2"
    t = trial()
    t["variant"]["drafter"] = drafter
    write(s2 / TRIAL_DIR / TRIAL_RESULT, t)
    write_decision(s2, decide(t, safe_top1=0.94), candidate=CANDIDATE)


def not_measured(card: str) -> str:
    return card.split("## Not measured")[1]


def test_the_card_shows_the_measured_acceptance_and_does_not_list_it_as_not_measured(world):
    decided_with(world, DRAFTER)
    _, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    r = rows(card)
    assert r[ACCEPTANCE_ROW][0] == "0.3125 fraction"
    assert r[ACCEPTANCE_ROW][1] == f"`stages/2/{TRIAL_DIR}/{TRIAL_RESULT}`, `stages/2/{DECISION_FILE}`"
    assert "acceptance rate" not in not_measured(card)
    flat = " ".join(card.split())
    assert "acceptance rate on this model is listed under Not measured" not in flat
    assert "Its acceptance rate on this model is in the table below" in flat


def test_a_trial_with_no_acceptance_leaves_the_card_saying_it_was_not_measured(world):
    decided_with(world, None)
    _, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    assert ACCEPTANCE_ROW not in rows(card)
    assert "acceptance rate" in not_measured(card)
    assert "listed under Not measured" in " ".join(card.split())


def test_the_gate_passes_a_card_with_the_acceptance_row(world):
    decided_with(world, DRAFTER)
    stage_dir, _, _ = finished_auto(world)
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_the_gate_refuses_a_card_that_lost_the_acceptance_row(world):
    decided_with(world, DRAFTER)
    stage_dir, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    (pkg / "README.md").write_text("\n".join(ln for ln in card.splitlines() if ACCEPTANCE_ROW not in ln) + "\n")
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any(ACCEPTANCE_ROW in r for r in g.reasons), g.reasons


# ---- the stage 8 bundle -------------------------------------------------------------------------------------------------------------

from test_operator_bundle_graft import results, risks_block, with_graft  # noqa: E402
from test_operator_bundle_template import write_json  # noqa: E402


def with_rate(tmp_path, choice, rate, *, safe_rate=None):
    run = with_graft(tmp_path, choice, package_graft=(choice == "graft"))
    p = run / "stages" / "2" / "graft-decision.json"
    d = json.loads(p.read_text())
    d["numbers"]["variant_drafter_accept_rate"] = rate
    d["numbers"]["safe_drafter_accept_rate"] = safe_rate
    write_json(p, d)
    return run


def test_the_bundle_table_has_an_acceptance_row_with_the_number(tmp_path):
    text = results(with_rate(tmp_path, "graft", 0.3125))
    line = [ln for ln in text.splitlines() if ln.startswith("| Drafter acceptance")][0]
    assert "0.3125" in line and "measured" in line and "accepted over drafted" in line


def test_the_bundle_shows_both_sides_when_both_were_measured(tmp_path):
    text = results(with_rate(tmp_path, "graft", 0.3125, safe_rate=0.5))
    line = [ln for ln in text.splitlines() if ln.startswith("| Drafter acceptance")][0]
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[1:4] == ["0.5", "0.3125", "measured"]


def test_the_bundle_says_acceptance_was_not_measured_when_the_trial_has_no_number(tmp_path):
    text = results(with_rate(tmp_path, "graft", None))
    flat = " ".join(text.split())
    assert "Drafter acceptance was not measured" in flat
    line = [ln for ln in text.splitlines() if ln.startswith("| Drafter acceptance")][0]
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[1:4] == ["-", "TODO", "TODO"]                     # safe side, grafted side, label


def test_the_bundle_does_not_say_not_measured_when_there_is_a_number(tmp_path):
    flat = " ".join(results(with_rate(tmp_path, "graft", 0.3125)).split())
    assert "Drafter acceptance was not measured" not in flat


def test_the_bundle_hazard_text_quotes_the_rate_when_there_is_one(tmp_path):
    block = risks_block(with_rate(tmp_path, "graft", 0.3125))
    assert "0.3125" in block and "acceptance rate was not measured" not in block


def test_the_bundle_hazard_text_says_not_measured_without_a_rate(tmp_path):
    block = risks_block(with_rate(tmp_path, "graft", None))
    assert "acceptance rate was not measured" in block


def limits_of(card: str) -> str:
    flat = " ".join(card.split())
    return flat.split("## Limitations")[1].split("## ")[0]


NOT_MEASURED_LIMIT = "How often the drafter's guesses are accepted on this model was not measured"


def test_the_limitations_follow_a_measured_acceptance(world):
    decided_with(world, DRAFTER)
    _, _, pkg = finished_auto(world)
    limits = limits_of((pkg / "README.md").read_text())
    assert NOT_MEASURED_LIMIT not in limits
    assert "is in the table above" in limits


def test_the_limitations_keep_the_not_measured_sentence_without_a_rate(world):
    decided_with(world, None)
    _, _, pkg = finished_auto(world)
    assert NOT_MEASURED_LIMIT in limits_of((pkg / "README.md").read_text())
