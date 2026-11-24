# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 662 (follow-up to PR 660): the run records the digest of the named bundle's orchard-capabilities.json and the sha256 of its manifest at the
preflight, and stages 2 and 4 refuse a bundle whose files changed since.

Stage 0 judges the model against the bundle's capabilities file. A bundle updated in place before stage 2 (`tt-model pull --force`, a hand edit)
would run code stage 0 never judged, and the ledger would not show it. The two values go into the `run_start` inputs as
`bundle_capabilities_sha256` (the hex digest, or `none` when the bundle ships no file) and `bundle_manifest_sha256`. A resume keeps the recorded
values. A run that started before this change has neither key and is not compared.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from orchard import nearest
from orchard import swap_draft as sd
from test_bundle_identity import (CAPS, QWEN38, QWEN4_4, REPO, bringup, bundle, conf, run, world)  # noqa: F401

CAPS_KEY, MANIFEST_KEY = "bundle_capabilities_sha256", "bundle_manifest_sha256"
OTHER_CAPS = json.dumps({"schema": 1, "config_keys": {"num_hidden_layers": {"type": "int"}, "rope": {"type": "float"}}})


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---- the record ----------------------------------------------------------------------------------------------------------

def test_the_record_holds_the_capabilities_digest_and_the_manifest_sha(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4, CAPS)
    rec = nearest.bundle_record(str(d))
    assert rec == {CAPS_KEY: sha(CAPS), MANIFEST_KEY: sha((d / "tt_kernel_manifest.json").read_text())}


def test_a_bundle_with_no_capabilities_file_records_none(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4)
    assert nearest.bundle_record(str(d))[CAPS_KEY] == "none"


def test_an_unreadable_file_is_recorded_without_a_path(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4)
    real = tmp_path / "caps.json"
    real.write_text(CAPS)
    (d / nearest.CAPABILITIES_FILE).symlink_to(real)
    rec = nearest.bundle_record(str(d))
    assert rec[CAPS_KEY] == "unreadable" and str(tmp_path) not in json.dumps(rec)


# ---- the comparison --------------------------------------------------------------------------------------------------------

def test_an_unchanged_bundle_has_no_changes(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4, CAPS)
    assert nearest.bundle_changes(str(d), nearest.bundle_record(str(d))) == []


def test_an_edited_capabilities_file_is_named(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4, CAPS)
    rec = nearest.bundle_record(str(d))
    (d / nearest.CAPABILITIES_FILE).write_text(OTHER_CAPS)
    got = nearest.bundle_changes(str(d), rec)
    assert len(got) == 1 and nearest.CAPABILITIES_FILE in got[0] and "changed" in got[0]


def test_a_capabilities_file_added_or_removed_since_the_record_is_a_change(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4)
    rec = nearest.bundle_record(str(d))
    (d / nearest.CAPABILITIES_FILE).write_text(CAPS)
    assert nearest.CAPABILITIES_FILE in "".join(nearest.bundle_changes(str(d), rec))
    rec = nearest.bundle_record(str(d))
    (d / nearest.CAPABILITIES_FILE).unlink()
    assert nearest.CAPABILITIES_FILE in "".join(nearest.bundle_changes(str(d), rec))


def test_an_edited_manifest_is_named(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4, CAPS)
    rec = nearest.bundle_record(str(d))
    m = json.loads((d / "tt_kernel_manifest.json").read_text())
    m["device_count"] = 2
    (d / "tt_kernel_manifest.json").write_text(json.dumps(m))
    got = nearest.bundle_changes(str(d), rec)
    assert len(got) == 1 and "tt_kernel_manifest.json" in got[0]


def test_both_files_changed_gives_two_messages(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4, CAPS)
    rec = nearest.bundle_record(str(d))
    (d / nearest.CAPABILITIES_FILE).write_text(OTHER_CAPS)
    (d / "tt_kernel_manifest.json").write_text("{}")
    assert len(nearest.bundle_changes(str(d), rec)) == 2


@pytest.mark.parametrize("recorded", [{}, {"bundle": "/x"}, None])
def test_a_run_that_recorded_nothing_is_not_compared(tmp_path, recorded):
    d = bundle(tmp_path, "a/a", REPO, 4, CAPS)
    (d / nearest.CAPABILITIES_FILE).write_text(OTHER_CAPS)
    assert nearest.bundle_changes(str(d), recorded) == []


def test_a_recorded_value_of_the_wrong_type_is_a_change_not_a_crash(tmp_path):
    d = bundle(tmp_path, "a/a", REPO, 4, CAPS)
    got = nearest.bundle_changes(str(d), {CAPS_KEY: 7, MANIFEST_KEY: ["x"]})
    assert len(got) == 2


# ---- the preflight records it; a resume keeps it ------------------------------------------------------------------------------

def test_the_run_records_the_named_bundles_digest_and_manifest_sha_as_inputs(conf, world):  # noqa: F811
    code, out, rec = bringup(conf, ["--base", QWEN4_4], world["w"])
    assert code == 0, out
    assert f"{CAPS_KEY}={sha(CAPS)}" in rec.argv
    assert f"{MANIFEST_KEY}={sha((world['q4'] / 'tt_kernel_manifest.json').read_text())}" in rec.argv


def test_a_bundle_without_capabilities_records_none(conf, world):  # noqa: F811
    code, out, rec = bringup(conf, ["--base", QWEN38], world["w"])
    assert code == 0, out
    assert f"{CAPS_KEY}=none" in rec.argv


def test_a_resumed_run_keeps_the_recorded_values_not_the_files_now(conf, world, tmp_path):  # noqa: F811
    from orchard import bringup_config
    from orchard.ledger import Ledger
    cfg = bringup_config.load(conf)
    run_dir = bringup_config.run_dir(cfg, "jialinyyzz/humanizer")
    run_dir.mkdir(parents=True)
    recorded = nearest.bundle_record(str(world["q4"]))
    with Ledger(run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="jialinyyzz/humanizer", inputs={"model": "/m", "bundle": str(world["q4"]), **recorded})
    (world["q4"] / nearest.CAPABILITIES_FILE).write_text(OTHER_CAPS)         # the bundle was updated in place before the resume
    code, out, rec = bringup(conf, ["--base", QWEN4_4], world["w"])
    assert code == 0, out
    assert f"{CAPS_KEY}={recorded[CAPS_KEY]}" in rec.argv and f"{CAPS_KEY}={sha(OTHER_CAPS)}" not in rec.argv


def test_a_resumed_run_that_recorded_nothing_records_nothing_now(conf, world, tmp_path):  # noqa: F811
    """A run that started before this change: recording the files as they are now would hide a change made since stage 0."""
    from orchard import bringup_config
    from orchard.ledger import Ledger
    cfg = bringup_config.load(conf)
    run_dir = bringup_config.run_dir(cfg, "jialinyyzz/humanizer")
    run_dir.mkdir(parents=True)
    with Ledger(run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="jialinyyzz/humanizer", inputs={"model": "/m", "bundle": str(world["q4"])})
    code, out, rec = bringup(conf, ["--base", QWEN4_4], world["w"])
    assert code == 0, out
    assert not any(a.startswith((CAPS_KEY, MANIFEST_KEY)) for a in rec.argv)


# ---- stages 2 and 4 refuse a changed bundle ------------------------------------------------------------------------------------

def recorded_inputs(run, directory):
    return {**run["inputs"], "bundle": str(directory), **nearest.bundle_record(str(directory))}


def test_stage_2_drafts_from_a_bundle_that_is_unchanged(run):
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 2, CAPS)
    cfg, problems = sd.draft(**{**run, "inputs": recorded_inputs(run, q4)})
    assert problems == [] and cfg["bundle_dir"] == str(q4)


def test_stage_2_refuses_a_bundle_whose_capabilities_file_changed_since_the_record(run):
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 2, CAPS)
    inputs = recorded_inputs(run, q4)
    (q4 / nearest.CAPABILITIES_FILE).write_text(OTHER_CAPS)
    cfg, problems = sd.draft(**{**run, "inputs": inputs})
    assert cfg is None and len(problems) == 1 and nearest.CAPABILITIES_FILE in problems[0] and QWEN4_4 in problems[0]


def test_stage_4_refuses_a_bundle_whose_manifest_changed_since_the_record(run, tmp_path):
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    inputs = recorded_inputs(run, q4)
    m = json.loads((q4 / "tt_kernel_manifest.json").read_text())
    m["device_count"] = 8
    (q4 / "tt_kernel_manifest.json").write_text(json.dumps(m))
    configs, hw, notes = sd.draft_stage4(run_dir=run["run_dir"], tt_model_root=run["tt_model_root"], cache_root=run["cache_root"],
                                         hf_home=run["hf_home"], operator_home=tmp_path / "op", inputs=inputs, counts=[4])
    assert configs == {} and any("tt_kernel_manifest.json" in n for n in notes)


def test_an_agent_written_config_is_refused_when_the_run_bundle_changed(run):
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    rec = nearest.bundle_record(str(q4))
    stage = run["run_dir"] / "stages" / "2"
    stage.mkdir(parents=True)
    (stage / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": str(q4)}))
    args = dict(tt_model_root=run["tt_model_root"], bundle=str(q4))
    assert sd.nearest_problem(run["run_dir"], stage / "swap_config.json", recorded=rec, **args) is None
    (q4 / nearest.CAPABILITIES_FILE).write_text(OTHER_CAPS)
    msg = sd.nearest_problem(run["run_dir"], stage / "swap_config.json", recorded=rec, **args)
    assert msg and nearest.CAPABILITIES_FILE in msg
    assert sd.nearest_problem(run["run_dir"], stage / "swap_config.json", **args) is None      # no record, no comparison (an older run)


def test_the_supervisor_passes_the_run_record_to_its_stage_2_and_4_checks(run, tmp_path, monkeypatch):
    from orchard.supervisor import NOTE_KEYS, Supervisor
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    stage = run["run_dir"] / "stages" / "2"
    stage.mkdir(parents=True)
    (stage / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": str(q4)}))
    (stage / "hw_test.json").write_text(json.dumps({"command": "true", "deadline_s": 60}))
    (stage / "handoff.json").write_text(json.dumps({k: "x" for k in NOTE_KEYS}))
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir = run["run_dir"]
    sup.paths = type("P", (), {"tt_model_root": run["tt_model_root"], "operator_home": tmp_path / "op"})()
    sup.inputs = recorded_inputs(run, q4)
    (q4 / nearest.CAPABILITIES_FILE).write_text(OTHER_CAPS)
    _, problems = sup._read_test(stage)
    assert any(nearest.CAPABILITIES_FILE in m for m in problems), problems
    stage4 = run["run_dir"] / "stages" / "4"
    (stage4 / "configs" / "4").mkdir(parents=True)
    (stage4 / "configs" / "4" / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": str(q4)}))
    import orchard.supervisor as sup_mod
    monkeypatch.setattr(sup_mod, "read_plan", lambda *a, **k: ([type("T", (), {"chips": 4})()], []))
    sup.required_chips, sup.home = None, tmp_path
    got = sup._read_plan(type("S", (), {"boards": 2, "budget_s": 60})(), stage4)
    assert any(nearest.CAPABILITIES_FILE in m for m in got[1]), got
