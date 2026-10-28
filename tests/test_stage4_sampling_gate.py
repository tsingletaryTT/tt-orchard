# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4's gate and stage 2's sampling decision (issue 107): when stage 2 chose the drafter-off configuration, every passing configuration's swap check
must say it was served with the drafter off. A decision of keep, or no decision, asks nothing."""
import pytest

from orchard import sampling_choice as sc
from orchard.stages import gate_mesh_swap
from test_stages import mesh_stage, swap_config, write


def decided(run, choice):
    write(run, f"stages/2/{sc.DECISION_FILE}", {"choice": choice, "reason": "because", "flag": None})


def stage(tmp_path, *, serving, chips=(2,)):
    sd, run = mesh_stage(tmp_path, [swap_config(n) for n in chips])
    for n in chips:
        report = {"result_draft": {"top1_agreement": swap_config(n)["top1_agreement"]}}
        if serving is not None:
            report["serving"] = serving
        write(run, f"stages/4/configs/{n}/evidence/swap-check.json", report)
    return sd, run


def test_a_drafter_off_decision_with_a_configuration_served_drafter_off_passes(tmp_path):
    sd, run = stage(tmp_path, serving={"drafter": "off", "sampling": "host"})
    decided(run, "drafter-off")
    assert gate_mesh_swap(sd, run, required=(2,)).ok


@pytest.mark.parametrize("serving", [{"drafter": "on (dflash2)"}, {"drafter": "not set by the run script"}, {"drafter": None}, {}, None, "off"])
def test_a_drafter_off_decision_with_a_configuration_not_served_drafter_off_is_refused(tmp_path, serving):
    sd, run = stage(tmp_path, serving=serving)
    decided(run, "drafter-off")
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("2-chip" in r and "drafter" in r and "sampling" in r for r in g.reasons), g.reasons


@pytest.mark.parametrize("choice", ["keep", None])
def test_a_keep_decision_or_none_asks_nothing_about_the_drafter(tmp_path, choice):
    sd, run = stage(tmp_path, serving={"drafter": "on (dflash2)"})
    if choice:
        decided(run, choice)
    assert gate_mesh_swap(sd, run, required=(2,)).ok


def test_each_configuration_is_judged_on_its_own_record(tmp_path):
    sd, run = stage(tmp_path, serving={"drafter": "off"}, chips=(2, 4))
    write(run, "stages/4/configs/4/evidence/swap-check.json",
          {"result_draft": {"top1_agreement": swap_config(4)["top1_agreement"]}, "serving": {"drafter": "on (dflash2)"}})
    decided(run, "drafter-off")
    g = gate_mesh_swap(sd, run, required=(2, 4))
    assert not g.ok and any("4-chip" in r and "drafter" in r for r in g.reasons) and not any("2-chip" in r and "drafter" in r for r in g.reasons)
