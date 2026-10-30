# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 writes the hazard scan (orchard/stage0_scan.py, issue 183): the record, the refusals, the supervisor wiring and the bundle."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import stage0_scan as s0

FIX = Path(__file__).parent / "fixtures" / "model-configs"
CLEAN = {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 32000}
MOE = dict(CLEAN, num_experts=512, num_experts_per_tok=8, moe_intermediate_size=64)


def snap(tmp, name, cfg):
    d = tmp / name
    d.mkdir()
    if cfg is not None:
        (d / "config.json").write_text(json.dumps(cfg))
    return d


def record(stage):
    return json.loads((stage / s0.FILE).read_text())


def test_a_model_with_512_experts_hits_the_256_mask_and_the_check_names_the_last_expert(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    rec = s0.write_scan(stage, model_dir=snap(tmp_path, "m", MOE), nearest_dir=snap(tmp_path, "n", CLEAN))
    on_disk = record(stage)
    assert on_disk == rec
    hits = {h["name"]: h for h in rec["hazards"]}
    assert "routed_experts_above_256" in hits and "routed_experts_above_512" not in hits      # 512 is at the wider mask's limit
    assert "511" in hits["routed_experts_above_256"]["stage2_check"]
    assert rec["scan"] == "ran" and rec["label"] == "prediction from the config"


def test_the_nearest_directory_is_read_from_the_triage_config_when_the_stage_has_one(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    m, n = snap(tmp_path, "m", CLEAN), snap(tmp_path, "n", dict(CLEAN, num_key_value_heads=1))
    (stage / "triage_config.json").write_text(json.dumps({"model_snapshot": str(m), "nearest_snapshot": str(n)}))
    rec = s0.write_scan(stage, model_dir=None, nearest_dir=None)
    assert [h["name"] for h in rec["hazards"]] == ["kv_heads_above_nearest"]


def test_without_a_nearest_config_the_scan_still_runs_and_says_what_it_skipped(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    rec = s0.write_scan(stage, model_dir=snap(tmp_path, "m", MOE), nearest_dir=None)
    assert rec["scan"] == "ran" and any("nearest" in n for n in rec["notes"])


@pytest.mark.parametrize("cfg,why", [(None, "config.json"), ("not json", "not usable"), ({"hidden_size": 0}, "hidden_size")])
def test_a_model_directory_the_scan_cannot_read_is_recorded_as_skipped_and_never_raises(tmp_path, cfg, why):
    stage = tmp_path / "stage"
    stage.mkdir()
    d = snap(tmp_path, "m", None)
    if cfg == "not json":
        (d / "config.json").write_text("{nope")
    elif cfg is not None:
        (d / "config.json").write_text(json.dumps(cfg))
    rec = s0.write_scan(stage, model_dir=d, nearest_dir=None)
    assert rec["scan"] == "skipped" and rec["hazards"] == [] and why in rec["reason"]
    assert record(stage)["scan"] == "skipped"


def test_no_model_directory_at_all_is_skipped(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    assert s0.write_scan(stage, model_dir=None, nearest_dir=None)["scan"] == "skipped"


def test_a_triage_config_with_a_hostile_value_is_ignored_not_followed(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "triage_config.json").write_text('{"model_snapshot": 5, "nearest_snapshot": ["x"]}')
    assert s0.write_scan(stage, model_dir=snap(tmp_path, "m", CLEAN), nearest_dir=None)["scan"] == "ran"


# ---- the catalog: the configs this repository has run, and the failures the run logs and issues record ----------------------

def test_the_catalog_configs_hit_what_the_notes_say_they_hit(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    base = FIX.parent / "model-configs"
    got, also = {}, {}
    for f in sorted(base.glob("*.json")):
        d = snap(tmp_path, f.stem, json.loads(f.read_text()))
        hits = s0.write_scan(stage, model_dir=d, nearest_dir=snap(tmp_path, "n" + f.stem, json.loads((base / "qwen3.8-27b.json").read_text())))["hazards"]
        got[f.stem] = {h["name"] for h in hits}
        also[f.stem] = {h["name"] for h in hits if h["also_on_nearest"]}
    assert "routed_experts_above_256" in got["qwen3-coder-next"]        # 512 experts: the 256-bit expert mask
    assert "routed_experts_above_256" in got["qwen3-next-80b-a3b-instruct"]
    assert got["qwen3.8-27b"] == also["qwen3.8-27b"]                     # every hit of the bundle's own model is one it already survives
    assert "routed_experts_above_256" not in also["qwen3-coder-next"]    # the 27B bundle has no experts, so this one is new
    assert "mixer_mamba_not_in_nearest" in got["granite-4.0-h-1b"]


def test_minicpm3_hits_the_flash_decode_grid_hazard_of_issue_213(tmp_path):
    cfg = dict(CLEAN, hidden_size=2560, intermediate_size=6400, num_attention_heads=40, num_key_value_heads=40, vocab_size=73448,
               kv_lora_rank=256, qk_rope_head_dim=32, qk_nope_head_dim=64, v_head_dim=64, q_lora_rank=768, head_dim=96)
    stage = tmp_path / "stage"
    stage.mkdir()
    rec = s0.write_scan(stage, model_dir=snap(tmp_path, "m", cfg), nearest_dir=None)
    assert "mla_flash_decode_needs_core_grid" in {h["name"] for h in rec["hazards"]}


# ---- the supervisor writes it at stage 0's pass --------------------------------------------------------------------------------

def test_stage_zero_of_a_supervised_run_writes_the_scan_and_records_it_in_the_ledger(tmp_path):
    from test_supervisor_stock import Rig, ends
    r = Rig(tmp_path, cfg_edit={"vocab_size": 200_000})
    try:
        r.run()
        es = r.entries()
        rec = json.loads((r.run_dir / "stages/0" / s0.FILE).read_text())
        assert rec["scan"] == "ran" and "vocab_per_die_above_65536" in {h["name"] for h in rec["hazards"]}
        ev = [x for x in es if x["event"] == "evidence" and x["stage"] == 0 and x["data"].get("what") == "hazard scan"]
        assert len(ev) == 1 and "vocab_per_die_above_65536" in ev[0]["data"]["hazards"]
        assert ev[0]["data"]["path"] == f"stages/0/{s0.FILE}"
        assert ends(es)[0]["result"] == "pass"           # a hit never stops a run
    finally:
        r.close()


# ---- the operator bundle lists it -------------------------------------------------------------------------------------------

def _bundle(tmp_path, scan):
    from test_operator_bundle_template import build, bundle_of, make_run, section
    run = make_run(tmp_path)
    if scan is not None:
        (run / "stages/0" / s0.FILE).write_text(json.dumps(scan))
    r = build(run)
    assert r.returncode == 0, r.stdout + r.stderr
    b = bundle_of(run)
    risks = (b / "RISKS.md").read_text()
    return (section(risks, "## Stage 0 hazard scan") if "## Stage 0 hazard scan\n" in risks else None), (b / "RESULTS.md").read_text()


HIT = {"name": "routed_experts_above_256", "value": 512, "limit": 256, "why": "an expert mask 256 bits wide drops the experts past index 255",
       "source": "issue 183", "stage2_check": "run the router with a token routed to expert index 511", "also_on_nearest": False}


def test_the_bundle_lists_each_hit_with_its_numbers_and_its_stage_2_check_and_says_the_check_is_not_shown(tmp_path):
    rec = {"label": "prediction from the config", "scan": "ran", "dies": 2, "facts": {}, "notes": [], "hazards": [HIT]}
    risks, results = _bundle(tmp_path, rec)
    assert "routed_experts_above_256" in risks and "512" in risks and "256" in risks
    assert "expert index 511" in risks and "prediction from the model's config" in risks
    assert "Check run: Not shown" in risks
    assert "1 kernel hazard" in results and "stages/0/hazard-scan.json" in results


def test_the_bundle_says_when_the_scan_found_nothing_was_skipped_or_never_ran(tmp_path):
    ran = {"label": "prediction from the config", "scan": "ran", "dies": 2, "facts": {}, "notes": [], "hazards": []}
    assert "found no hazards" in _bundle(tmp_path / "a", ran)[0]
    skipped = {"label": "prediction from the config", "scan": "skipped", "reason": "no config.json", "hazards": []}
    assert "was skipped" in _bundle(tmp_path / "b", skipped)[0] and "no config.json" in _bundle(tmp_path / "c", skipped)[0]
    risks, results = _bundle(tmp_path / "d", None)       # a run from before the scan gets the same files as before
    assert risks is None and "Hazard scan" not in results


def test_a_hit_the_nearest_bundle_also_has_is_marked(tmp_path):
    rec = {"label": "prediction from the config", "scan": "ran", "dies": 2, "facts": {}, "notes": [],
           "hazards": [dict(HIT, also_on_nearest=True)]}
    assert "nearest bundle also hits it" in _bundle(tmp_path, rec)[0]


def test_a_malformed_scan_file_never_stops_the_bundle(tmp_path):
    risks, _ = _bundle(tmp_path, {"scan": "ran", "hazards": "nope"})
    assert "could not be read" in risks


def test_the_stage_2_skill_points_the_agent_at_the_scan_and_says_not_to_claim_its_checks_ran():
    text = " ".join((Path(__file__).parent.parent / "orchard/skills/weights-swap-check.md").read_text().split())
    assert "stages/0/hazard-scan.json" in text and "`stage2_check`" in text and "its check was not run" in text


# ---- review of PR 292 -------------------------------------------------------------------------------------------------------

def test_a_null_byte_in_a_snapshot_path_is_a_skipped_input_and_never_raises(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "triage_config.json").write_text(json.dumps({"model_snapshot": str(snap(tmp_path, "m", MOE)), "nearest_snapshot": "a\u0000b"}))
    rec = s0.write_scan(stage, model_dir=None, nearest_dir=None)
    assert rec["scan"] == "ran" and any("nearest" in n for n in rec["notes"]) and "routed_experts_above_256" in {h["name"] for h in rec["hazards"]}
    assert s0.write_scan(stage, model_dir="x\u0000y", nearest_dir=None)["scan"] == "skipped"


def test_a_nearest_config_with_unusable_dimensions_keeps_the_models_own_hazards(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    rec = s0.write_scan(stage, model_dir=snap(tmp_path, "m", MOE), nearest_dir=snap(tmp_path, "n", {}))
    assert rec["scan"] == "ran" and "routed_experts_above_256" in {h["name"] for h in rec["hazards"]}
    assert any("nearest" in n and "skipped" in n for n in rec["notes"])


def test_the_bundle_reads_the_scan_with_a_size_cap_and_refuses_a_fifo_or_a_device_link(tmp_path):
    import os
    import threading
    from test_operator_bundle_template import build, bundle_of, make_run
    run = make_run(tmp_path)
    fifo = run / "stages/0" / s0.FILE
    os.mkfifo(fifo)
    done = []
    t = threading.Thread(target=lambda: done.append(build(run)), daemon=True)
    t.start()
    t.join(30)
    assert done and done[0].returncode == 0, "the bundle build blocked on a FIFO"
    assert "could not be read" in (bundle_of(run) / "RISKS.md").read_text()
    fifo.unlink()
    os.symlink("/dev/zero", fifo)
    assert build(run).returncode == 0
    assert "could not be read" in (bundle_of(run) / "RISKS.md").read_text()


def test_a_scan_whose_notes_are_not_a_list_is_reported_unreadable_and_never_stops_the_bundle(tmp_path):
    risks, _ = _bundle(tmp_path, {"scan": "ran", "hazards": [], "notes": 1})
    assert "could not be read" in risks


def test_a_failure_to_write_the_scan_is_a_ledger_decision_and_the_stage_still_passes(tmp_path, monkeypatch):
    from orchard.stages import GateResult, spec_for
    from test_supervisor_stock import Rig, _supervisor

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    r = Rig(tmp_path)
    try:
        r.run()
        monkeypatch.setattr(s0, "write_scan", boom)
        sup, led = _supervisor(r)
        try:
            spec = spec_for(0, "weights-only")
            assert sup._end_stage(spec, r.run_dir / "stages/0", "pass", [], GateResult(True, (), ()), False) == "pass"
            entries = led.read()
        finally:
            led.__exit__(None, None, None)
        assert any(e["data"].get("decision") == "hazard scan not written" and "No space" in e["data"]["reason"] for e in entries)
    finally:
        r.close()


def test_a_scan_entry_with_no_name_makes_the_whole_scan_unreadable_and_never_a_clean_one(tmp_path):
    risks, results = _bundle(tmp_path, {"label": "x", "scan": "ran", "dies": 2, "facts": {}, "notes": [], "hazards": [{"value": 512, "limit": 256}]})
    assert "could not be read" in risks and "found no hazards" not in risks and "0 kernel hazard" not in results
