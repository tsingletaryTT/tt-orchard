# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `restricted-head` class: the nearest model's backbone, with its lm_head cut down to a few rows kept in a sidecar.

The first model of this kind was perplexity-ai/pplx-decider-v1.2 (2026-10-19). Its weights are Qwen3.8-27B's, trained a little;
its readout rows are copies of the base head's rows for the answer-code tokens; and the way it reads them (at marker tokens, with
attention across a request's questions) is in its code, not in its config or tensors. The operator's ruling was that the harness
stops after stage 0 and reports, instead of attempting to serve it. These tests pin that: the class is known, its gate rules
hold, the run pauses before stage 1 (so no coder time goes on a reference for a model that will not be served), an unattended run
ends blocked with its own code, and the block says what to do.
"""
import json

import pytest

from orchard import blocked, classes, narrate, status, supervisor
from orchard.stages import gate_delta
from run_fakes import DELTA, FILES, TEMPLATE_SHA, TEMPLATE_TEXT, bringup
from test_supervisor import never_wait, rig  # noqa: F401  (rig is a fixture)

SIDECAR = {"file": "readout.safetensors", "size": 2611312, "sha256": "ab" * 32, "num_tensors": 1, "dtypes": ["BF16"],
           "tensor_names": ["weight"]}
HEAD = {"restricted": True, "rows": 255, "base_rows": 248320, "matched_rows": 255, "min_cosine": 0.9999,
        "sidecar": "readout.safetensors", "tensor": "weight", "token_ids": [32, 33], "note": "copies of lm_head rows"}
RESTRICTED = {**DELTA, "path": "full-port", "class": "restricted-head", "sidecars": [SIDECAR], "head": HEAD}
RESTRICTED_FILES = {(0, "run"): {**FILES[(0, "run")], "delta.json": RESTRICTED}}


# ---- the class --------------------------------------------------------------------------------------------------------------------

def test_the_class_exists_and_takes_the_full_port_path():
    assert "restricted-head" in classes.CLASSES
    assert classes.path_of("restricted-head") == "full-port"


def write_stage(tmp_path, delta):
    (tmp_path / "evidence").mkdir(exist_ok=True)
    (tmp_path / "evidence" / "e.json").write_text("{}")
    (tmp_path / "delta.json").write_text(json.dumps({"template_sha256": TEMPLATE_SHA["delta_triage.py"], **delta}))
    (tmp_path / "delta_triage.py").write_text(TEMPLATE_TEXT["delta_triage.py"])
    return tmp_path


def evidence_free(delta):
    """A delta.json with one finding that cites a file, so only the class rules are under test."""
    return {**delta, "differences": [{"area": "config", "finding": "x", "evidence": ["evidence/e.json"]}], "hazards": []}


def test_the_gate_accepts_a_restricted_head_with_its_sidecar_and_head_record(tmp_path):
    g = gate_delta(write_stage(tmp_path, evidence_free(RESTRICTED)), tmp_path)
    assert g.ok, g.reasons


def test_the_gate_wants_the_head_record_for_this_class(tmp_path):
    without = {k: v for k, v in evidence_free(RESTRICTED).items() if k != "head"}
    assert any("head" in r for r in gate_delta(write_stage(tmp_path, without), tmp_path).reasons)
    not_restricted = {**evidence_free(RESTRICTED), "head": {**HEAD, "restricted": False}}
    assert any("restricted" in r for r in gate_delta(write_stage(tmp_path, not_restricted), tmp_path).reasons)


def test_the_gate_wants_a_sidecar_for_this_class(tmp_path):
    bare = {**evidence_free(RESTRICTED), "sidecars": []}
    assert any("sidecar" in r for r in gate_delta(write_stage(tmp_path, bare), tmp_path).reasons)


def test_the_class_and_the_path_must_agree(tmp_path):
    wrong = {**evidence_free(RESTRICTED), "path": "weights-only"}
    assert any("takes the path" in r for r in gate_delta(write_stage(tmp_path, wrong), tmp_path).reasons)


def test_only_the_two_sidecar_classes_may_carry_sidecars(tmp_path):
    bad = {**evidence_free(RESTRICTED), "class": "full-port"}
    assert any("cannot have sidecars" in r for r in gate_delta(write_stage(tmp_path, bad), tmp_path).reasons)


# ---- the stop --------------------------------------------------------------------------------------------------------------------

def run_attended_until_abort(rig):
    rig.script = lambda r: bringup(r, overrides=RESTRICTED_FILES)
    from orchard.supervisor import Control
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    return rig.run()


def test_the_run_pauses_after_stage_0_and_never_starts_stage_1(rig):
    assert run_attended_until_abort(rig) == supervisor.EXIT_ABORTED
    assert [d["result"] for d in rig.ends(0)] == ["pass"]
    assert rig.ends(1) == [] and not any(e["stage"] == 1 and e["event"] == "stage_start" for e in rig.entries())
    assert rig.ends(0)[0]["class"] == "restricted-head" and rig.ends(0)[0]["path"] == "full-port"
    pauses = [d for d in rig.decisions() if d["decision"] == "pause"]
    assert len(pauses) == 1 and "restricted head" in pauses[0]["reason"]


def test_an_unattended_run_ends_blocked_with_its_own_code_and_never_waits(rig):
    rig.args.unattended = True
    rig.script = lambda r: bringup(r, overrides=RESTRICTED_FILES)
    never_wait(rig)
    assert rig.run() == supervisor.EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert len(block) == 1 and block[0]["code"] == "restricted-head" and "restricted head" in block[0]["reason"]
    assert rig.ends(1) == []
    text = (rig.run_dir / "BLOCKED.md").read_text()
    assert "restricted-head" in text and "lm_head" in text.split("## How to unblock")[1]


def test_an_unattended_retry_is_stopped_again_and_stage_1_still_never_runs(rig):
    """An unattended retry must not let the run slide past the stop: the first block is not an acknowledgement."""
    rig.args.unattended = True
    rig.script = lambda r: bringup(r, overrides=RESTRICTED_FILES)
    never_wait(rig)
    assert rig.run() == supervisor.EXIT_BLOCKED
    assert rig.run() == supervisor.EXIT_BLOCKED
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["restricted-head"] * 2
    assert rig.ends(1) == []


def test_the_block_releases_the_hardware(rig):
    rig.args.unattended = True
    rig.script = lambda r: bringup(r, overrides=RESTRICTED_FILES)
    never_wait(rig)
    rig.run()
    assert not rig.m.coder_running and rig.m.leases == {}


def test_an_operator_who_resumes_from_the_pause_is_not_stopped_again_at_stage_2(rig):
    from orchard.supervisor import Control
    rig.script = lambda r: bringup(r, overrides=RESTRICTED_FILES)
    calls = []

    def on_sleep():
        calls.append(1)
        Control(rig.run_dir).write("resume" if len(calls) == 1 else "abort")
    rig.on_sleep = on_sleep
    rig.run()
    reasons = [d["reason"] for d in rig.decisions() if d["decision"] == "pause"]
    assert len(reasons) == 1 and "restricted head" in reasons[0]          # no second pause for the full-port path at stage 2


def test_a_full_port_still_pauses_at_stage_2_and_lets_stage_1_run(rig):
    from run_fakes import DELTA as D
    full = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**D, "path": "full-port"}}}
    from orchard.supervisor import Control
    rig.script = lambda r: bringup(r, overrides=full)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert [d["result"] for d in rig.ends(1)] == ["pass"] and rig.ends(2) == []


# ---- what the operator is told ---------------------------------------------------------------------------------------------------

def test_the_pause_is_classified_and_has_a_block_code_with_an_unblock_text():
    pause = {"decision": "pause", "reason": supervisor.RESTRICTED_HEAD}
    assert status.classify_pause(pause)[0] == "restricted_head"
    assert blocked.block_code(pause) == "restricted-head"
    assert "restricted-head" in blocked.REASONS and narrate.UNBLOCK["restricted-head"]
    text = " ".join(narrate.UNBLOCK["restricted-head"])
    assert "lm_head" in text and "delta.json" in text


def test_a_restricted_head_pause_is_a_state_that_needs_a_human():
    f = {"state": "paused", "paused": {"kind": "restricted_head", "reason": supervisor.RESTRICTED_HEAD}, "pause": {"kind": "restricted_head"}}
    rule = next(r for r in status.HINT_RULES if r[0] == "needs a human")
    assert rule[1](f)
