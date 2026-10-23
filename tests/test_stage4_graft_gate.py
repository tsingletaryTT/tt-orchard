# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4's gate and stage 2's graft decision: a passing bundle configuration must have been built the way the decision says. The drafted
config says graft_mtp, but the prepare step is an agent's and can rewrite it after the draft; what it built is in model-dir-sources.json."""
import json

import pytest

from orchard.graft_trial import DECISION_FILE
from orchard.stages import gate_mesh_swap
from test_stages import mesh_stage, swap_config, write


def decided(run, choice):
    write(run, f"stages/2/{DECISION_FILE}", {"choice": choice, "reason": "because"})


def bundle_config(run, n, *, grafted):
    write(run, f"stages/4/configs/{n}/swap_config.json", {"bundle_dir": "/b", "tt_cache": "/c", **({"graft_mtp": True} if grafted else {})})
    sources = {"files": []}
    if grafted:
        sources["mtp_graft"] = {"from": "Qwen/Qwen3.8-27B", "file": "model-mtp-from-base.safetensors", "tensors": 15}
    write(run, f"stages/4/configs/{n}/model-dir-sources.json", sources)


def gate(tmp_path, n=2):
    sd, run = mesh_stage(tmp_path, [swap_config(n)])
    return sd, run


def test_a_graft_decision_with_a_grafted_bundle_configuration_passes(tmp_path):
    sd, run = gate(tmp_path)
    decided(run, "graft")
    bundle_config(run, 2, grafted=True)
    assert gate_mesh_swap(sd, run, required=(2,)).ok


def test_a_graft_decision_with_a_bundle_configuration_built_without_the_graft_is_refused(tmp_path):
    sd, run = gate(tmp_path)
    decided(run, "graft")
    bundle_config(run, 2, grafted=False)
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and len(g.reasons) == 1
    assert "2-chip configuration" in g.reasons[0] and "graft" in g.reasons[0] and "model-dir-sources.json" in g.reasons[0]


def test_a_graft_decision_with_no_sources_record_is_refused(tmp_path):
    sd, run = gate(tmp_path)
    decided(run, "graft")
    write(run, "stages/4/configs/2/swap_config.json", {"bundle_dir": "/b", "tt_cache": "/c"})
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("model-dir-sources.json" in r for r in g.reasons)


@pytest.mark.parametrize("choice", ["safe", None])
def test_a_grafted_configuration_without_a_graft_decision_is_refused(tmp_path, choice):
    sd, run = gate(tmp_path)
    if choice:
        decided(run, choice)
    bundle_config(run, 2, grafted=True)
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and len(g.reasons) == 1 and "did not choose the graft" in g.reasons[0]


def test_a_container_configuration_is_not_held_to_the_decision(tmp_path):
    """No bundle_dir in its swap_config.json: a container package keeps its own serving, graft decision or not."""
    sd, run = gate(tmp_path, n=4)
    decided(run, "graft")
    write(run, "stages/4/configs/4/swap_config.json", {"package": "x/y", "profile": "p", "tt_cache": "/c"})
    write(run, "stages/4/configs/4/model-dir-sources.json", {"files": []})
    assert gate_mesh_swap(sd, run, required=(4,)).ok


def test_a_run_with_no_decision_and_an_ungrafted_bundle_configuration_passes_as_before(tmp_path):
    sd, run = gate(tmp_path)
    bundle_config(run, 2, grafted=False)
    assert gate_mesh_swap(sd, run, required=(2,)).ok


def test_a_configuration_that_does_not_claim_a_pass_is_not_checked(tmp_path):
    sd, run = mesh_stage(tmp_path, [swap_config(2), swap_config(1, ok=False)])
    decided(run, "graft")
    bundle_config(run, 2, grafted=True)
    bundle_config(run, 1, grafted=False)                    # the optional 1-chip test did not pass, so it is judged by gate_mesh alone
    assert gate_mesh_swap(sd, run, required=(2,)).ok


def test_a_swap_config_the_gate_cannot_read_is_treated_as_no_bundle_configuration(tmp_path):
    sd, run = gate(tmp_path)
    decided(run, "graft")
    write(run, "stages/4/configs/2/swap_config.json", "not json")
    assert gate_mesh_swap(sd, run, required=(2,)).ok


def test_the_check_reads_what_was_built_and_not_what_the_config_asks_for(tmp_path):
    sd, run = gate(tmp_path)
    decided(run, "graft")
    write(run, "stages/4/configs/2/swap_config.json", {"bundle_dir": "/b", "tt_cache": "/c", "graft_mtp": True})
    write(run, "stages/4/configs/2/model-dir-sources.json", {"files": []})
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("records no mtp_graft" in r for r in g.reasons)


def test_a_graft_record_that_is_not_an_object_is_not_a_record(tmp_path):
    sd, run = gate(tmp_path)
    decided(run, "graft")
    write(run, "stages/4/configs/2/swap_config.json", {"bundle_dir": "/b", "tt_cache": "/c"})
    write(run, "stages/4/configs/2/model-dir-sources.json", {"mtp_graft": True})
    assert not gate_mesh_swap(sd, run, required=(2,)).ok
