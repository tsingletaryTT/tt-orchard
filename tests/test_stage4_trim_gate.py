# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4's gate for a vocabulary trim (issue 56): each configuration that claims a pass must have been served with the model's own
vocab_size (model-dir-sources.json) and with the speculative drafter off (swap-check.json's serving record). Stage 2's gate asks the
same of its one configuration (tests/test_stages_class.py)."""
import pytest

from orchard.stages import gate_mesh_swap
from test_stages import mesh_stage, swap_config, write
from test_stages_class import GOOD_DELTA, TRIM, TRIM_DELTA

GOOD_SOURCES = {"vocab": {"model": TRIM["model_vocab"], "base": TRIM["base_vocab"], "trimmed": True}}
OFF = {"drafter": "off", "sampling": "host"}


def build(tmp_path, *, chips=(2,), sources=GOOD_SOURCES, serving=OFF, klass="vocab-trim", extra=()):
    sd, run = mesh_stage(tmp_path, [swap_config(n) for n in chips] + list(extra))
    write(run, "stages/0/delta.json", TRIM_DELTA if klass == "vocab-trim" else {**GOOD_DELTA, "class": klass, "sidecars": []})
    for n in chips:
        if sources is not None:
            write(run, f"stages/4/configs/{n}/model-dir-sources.json", sources)
        report = {"result_draft": {"top1_agreement": swap_config(n)["top1_agreement"]}}
        if serving is not None:
            report["serving"] = serving
        write(run, f"stages/4/configs/{n}/evidence/swap-check.json", report)
    return sd, run


def test_a_trim_served_with_its_vocab_and_the_drafter_off_passes_every_configuration(tmp_path):
    sd, run = build(tmp_path, chips=(2, 4))
    g = gate_mesh_swap(sd, run, required=(2, 4))
    assert g.ok, g.reasons


def test_a_configuration_whose_served_config_kept_the_nearest_models_vocab_is_refused(tmp_path):
    sd, run = build(tmp_path, sources={"vocab": {"model": 181760, "base": 248320, "trimmed": False}})
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("2-chip" in r and "vocab" in r for r in g.reasons), g.reasons


def test_a_configuration_that_records_another_vocabulary_is_refused(tmp_path):
    sd, run = build(tmp_path, sources={"vocab": {"model": 7, "base": 248320, "trimmed": True}})
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("181760" in r for r in g.reasons), g.reasons


def test_a_configuration_with_no_sources_record_is_refused(tmp_path):
    sd, run = build(tmp_path, sources=None)
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("2-chip" in r and "model-dir-sources.json" in r for r in g.reasons), g.reasons


@pytest.mark.parametrize("serving", [{"drafter": "on", "sampling": "device"}, {"drafter": None}, {}, None, "off"])
def test_a_configuration_served_with_the_drafter_on_or_unrecorded_is_refused(tmp_path, serving):
    sd, run = build(tmp_path, serving=serving)
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("2-chip" in r and "drafter" in r for r in g.reasons), g.reasons


def test_each_configuration_is_judged_on_its_own_record(tmp_path):
    sd, run = build(tmp_path, chips=(2, 4))
    write(run, "stages/4/configs/4/model-dir-sources.json", {"vocab": {"model": 181760, "trimmed": False}})
    g = gate_mesh_swap(sd, run, required=(2, 4))
    assert not g.ok
    assert any("4-chip" in r and "vocab" in r for r in g.reasons) and not any("2-chip" in r and "vocab" in r for r in g.reasons), g.reasons


def test_a_configuration_that_does_not_claim_a_pass_is_not_checked(tmp_path):
    sd, run = build(tmp_path, extra=[swap_config(1, ok=False)])
    write(run, "stages/4/configs/1/model-dir-sources.json", {"vocab": {"trimmed": False}})
    assert gate_mesh_swap(sd, run, required=(2,)).ok


def test_other_classes_are_asked_nothing_extra(tmp_path):
    sd, run = build(tmp_path, klass="weights-only", sources={"vocab": {"trimmed": False}}, serving={"drafter": "on"})
    g = gate_mesh_swap(sd, run, required=(2,))
    assert g.ok, g.reasons


@pytest.mark.parametrize("size", [None, True, "181760"])
def test_a_delta_whose_model_vocab_is_not_a_whole_number_cannot_be_matched_by_a_record_that_repeats_it(tmp_path, size):
    sd, run = build(tmp_path, sources={"vocab": {"model": size, "base": 248320, "trimmed": True}})
    write(run, "stages/0/delta.json", {**TRIM_DELTA, "vocab_trim": {**TRIM, "model_vocab": size}})
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("2-chip" in r and "vocab" in r for r in g.reasons), g.reasons


def test_a_stage_0_record_the_safe_reader_refuses_does_not_switch_the_trim_check_off(tmp_path):
    """delta.json with a NaN in it is refused by the reader; the gate must not take that as 'this run is not a vocabulary trim'."""
    sd, run = build(tmp_path, sources=GOOD_SOURCES, serving={"drafter": "on"})
    (run / "stages/0/delta.json").write_text('{"class": "vocab-trim", "x": NaN}')
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("delta.json" in r and "not readable" in r for r in g.reasons), g.reasons
