# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 363: the one serving-time gap the stage 2 check can see. The TT vLLM plugin reuses a decode slot for a new request and expects
prefill to initialise that request's model state (docs/superpowers/specs/2026-11-02-port-interface-vllm-plugin-mapping.md). The check
calls `prefill` on the same prompt twice, once before the teacher-forced run and once before the free run. A port whose prefill does not
reset its state gives a different first id the second time. The record holds both ids, so the gate compares them."""
import pytest

from orchard.stages import gate_full_port
from test_full_port import build, one, REF_IDS


def fixed(record):
    """A record whose free run starts with the id the teacher-forced run started with."""
    chip = list(REF_IDS[:30]) + [0, 0]
    return {"chip_ids": chip, "free_run_ids": [chip[0]] + list(range(1, 48)), **record}


def test_a_free_run_that_starts_with_the_teacher_forced_first_id_passes(tmp_path):
    stage_dir, run = build(tmp_path, record=fixed({}))
    g = gate_full_port(stage_dir, run)
    assert g.ok, g.reasons


def test_a_second_prefill_that_gives_another_first_id_fails(tmp_path):
    chip = list(REF_IDS[:30]) + [0, 0]
    stage_dir, run = build(tmp_path, record={"chip_ids": chip, "free_run_ids": [chip[0] + 1] + list(range(1, 48))})
    g = gate_full_port(stage_dir, run)
    assert not g.ok and one(g, "prefill")


def test_the_reason_names_both_ids_and_the_slot_reuse_it_stands_for(tmp_path):
    chip = list(REF_IDS[:30]) + [0, 0]
    stage_dir, run = build(tmp_path, record={"chip_ids": chip, "free_run_ids": [7] + list(range(1, 48))})
    text = " ".join(gate_full_port(stage_dir, run).reasons)
    assert "7" in text and str(chip[0]) in text and "reset" in text


@pytest.mark.parametrize("bad", ["nope", [], None, [True], [1.5]])
def test_a_record_with_unusable_free_run_ids_fails_with_a_reason_and_does_not_raise(tmp_path, bad):
    stage_dir, run = build(tmp_path, record={"free_run_ids": bad})
    g = gate_full_port(stage_dir, run)
    assert not g.ok and one(g, "free_run_ids")
