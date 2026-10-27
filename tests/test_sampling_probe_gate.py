# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 and stage 4 gates and the sampling probe (issue 107). The probe records a capability, so a sampled request that is refused,
ignored or slow never fails a stage. A record that is not well formed does, because every later reader (the bundle, the card) would have to
guess what it meant. A swap-check.json with no probe record changes nothing."""
import copy
import json

from orchard.stages import gate_mesh_swap, gate_weights_swap
from test_prompt_suite_gate import HEALTHY_FREE, stage2, swap_check
from test_sampling_probe import STORY, entry, record, refused

BAD_SECONDS = "fast"


def with_probe(rec):
    return swap_check() | {"sampling_probe": rec}


def test_a_swap_check_without_the_probe_changes_nothing(tmp_path):
    assert stage2(tmp_path, swap_check()).ok


def test_a_healthy_probe_passes(tmp_path):
    g = stage2(tmp_path, with_probe(record()))
    assert g.ok, g.reasons


def test_a_refused_sampled_request_does_not_fail_the_stage(tmp_path):
    assert stage2(tmp_path, with_probe(record(a=refused(), b=refused()))).ok


def test_ignored_sampling_does_not_fail_the_stage(tmp_path):
    assert stage2(tmp_path, with_probe(record(a=entry(text=STORY), b=entry(text=STORY)))).ok


def test_a_slow_sampled_request_does_not_fail_the_stage(tmp_path):
    assert stage2(tmp_path, with_probe(record(a=entry(seconds=30.0), b=entry(text="x " * 40, seconds=30.0)))).ok


def test_a_malformed_probe_record_fails_the_stage_and_says_where(tmp_path):
    rec = copy.deepcopy(record())
    rec["requests"]["greedy"]["seconds"] = BAD_SECONDS
    g = stage2(tmp_path, with_probe(rec))
    assert not g.ok
    assert any("sampling_probe" in r and "greedy" in r and "seconds" in r for r in g.reasons)


def test_the_stage_4_gate_reads_the_probe_too_and_names_the_configuration(tmp_path):
    from test_prompt_suite_gate import mesh_stage, swap_config, write, GOOD_DELTA
    rec = copy.deepcopy(record())
    rec["requests"]["sampled_a"]["accepted"] = "yes"
    sd, run = mesh_stage(tmp_path, [swap_config(2), swap_config(4)])
    write(run, "stages/0/delta.json", GOOD_DELTA)
    write(run, "stages/4/configs/2/evidence/swap-check.json", with_probe(record()))
    write(run, "stages/4/configs/4/evidence/swap-check.json", with_probe(rec))
    g = gate_mesh_swap(sd, run, required=(2, 4))
    assert not g.ok
    assert any(r.startswith("the 4-chip configuration: ") and "sampling_probe" in r for r in g.reasons)
    assert not any(r.startswith("the 2-chip configuration: ") and "sampling_probe" in r for r in g.reasons)
