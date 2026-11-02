# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Slice 4 of the full-port route: the residency estimate is written at stage 0 and read before a full-port stage 2 (issue 327). The chip, the
coder and the model are fakes; the supervisor, the stage machine and the estimator are the real ones. The model is a directory of
safetensors headers (tests/residency_headers.py): a model too large for the box's 4 chips, and one that fits."""
import json

import pytest

from orchard import blocked, narrate, residency, status
from orchard.supervisor import EXIT_BLOCKED, Control
from residency_headers import write_snapshot
from test_supervisor_full_port import make, never_wait, started
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")

HUGE = [("model.layers.0.mlp.gate_proj.weight", "BF16", [1_000_000, 1_000_000])]          # 2 TB of weights: 35 chips at the defaults
SMALL = [("model.layers.0.mlp.gate_proj.weight", "BF16", [4096, 4096])]


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)          # the rig's box is two boards: four chips
    r.args.full_port = True
    yield r
    r.close()


def with_model(rig, tmp_path, tensors, name="model"):
    d = tmp_path / name
    if tensors is not None:
        write_snapshot(d, tensors, shards=1)
    d.mkdir(exist_ok=True)
    rig.args.input.append(f"model={d}")
    return d


def evidence(rig, what):
    return [e["data"] for e in rig.entries() if e["event"] == "evidence" and e["data"].get("what") == what]


def test_stage_0_writes_the_record_and_a_ledger_entry_that_names_the_box(rig, tmp_path):
    with_model(rig, tmp_path, SMALL)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    rec = json.loads((rig.run_dir / "stages" / "0" / residency.FILE).read_text())
    assert rec["scan"] == "ran" and rec["verdict"] == {"box_chips": 4, "fits_box": True, "chips_needed": 1}
    (entry,) = evidence(rig, "residency estimate")
    assert entry["chips_needed"] == 1 and entry["fits_box"] is True and entry["hazards"] == [] and entry["path"].endswith("residency.json")


def test_a_model_the_box_cannot_hold_blocks_stage_2_with_the_numbers_and_starts_no_prepare_step(rig, tmp_path):
    with_model(rig, tmp_path, HUGE)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    pauses = [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"]
    assert pauses and pauses[0].startswith("blocked: the model does not fit this box")
    assert "at least 35 chips" in pauses[0] and "the box has 4 chips" in pauses[0] and "choice" in pauses[0]
    assert not started(rig, 2)                                                         # no stage 2 step started: no prepare step
    stage2 = rig.run_dir / "stages" / "2"
    assert not stage2.exists() or not any(stage2.iterdir())
    notice = next(e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked"))
    assert notice["evidence"]["need_chips"] == 35 and notice["evidence"]["box_chips"] == 4
    (entry,) = evidence(rig, "residency estimate")
    assert entry["fits_box"] is False and entry["hazards"] == ["model_does_not_fit_box"]


def test_unattended_the_block_has_its_own_code_and_is_not_retried(rig, tmp_path):
    rig.args.unattended = True
    with_model(rig, tmp_path, HUGE)
    make(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert [b["code"] for b in block] == ["model-does-not-fit"]
    assert "at least 35 chips" in (rig.run_dir / "BLOCKED.md").read_text()
    assert not started(rig, 2) and "model-does-not-fit" not in blocked.AUTO_RETRY and not blocked.retryable("model-does-not-fit", block[0]["reason"])


def test_the_code_is_classified_before_the_hardware_words_and_has_an_unblock_text():
    why = residency.shortfall({"estimated": {"chips_needed": 6, "device_bytes": 5 << 30}, "assumptions": []}, 4)
    assert status.classify_pause({"reason": f"blocked: {why}"})[0] == "blocked"
    assert blocked.block_code({"reason": f"blocked: {why}"}) == "model-does-not-fit"
    assert "model-does-not-fit" in blocked.REASONS and narrate.UNBLOCK["model-does-not-fit"]


def test_a_model_that_fits_goes_on_into_stage_2(rig, tmp_path):
    with_model(rig, tmp_path, SMALL)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert started(rig, 2) and not any(str(d.get("reason", "")).startswith("blocked:") for d in rig.decisions())


def test_a_model_directory_without_headers_makes_no_claim(rig, tmp_path):
    with_model(rig, tmp_path, None)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    rec = json.loads((rig.run_dir / "stages" / "0" / residency.FILE).read_text())
    assert rec["scan"] == "skipped" and started(rig, 2)


def test_only_a_full_port_stage_2_is_gated(tmp_path):
    r = Rig(tmp_path)                      # no --full-port: the generic stage 2 runs and the estimate is a record only
    try:
        with_model(r, tmp_path, HUGE)
        r.on_sleep = lambda: Control(r.run_dir).write("abort")
        r.run()
        assert not any(str(d.get("reason", "")).startswith("blocked: the model does not fit") for d in r.decisions())
        assert started(r, 2)
    finally:
        r.close()
