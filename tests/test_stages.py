"""The stage table, tier and skill lookup, the disk check, the exit gates and the stage 0 comparison."""
import dataclasses
import json
import os
import socket
from collections import namedtuple
from pathlib import Path

import pytest

from orchard.stages import (STAGES, TierUnavailable, check_disk, compare_delta, gate_bundle,
                            gate_decoder, gate_delta, gate_full_model, gate_mesh, gate_numbers,
                            gate_reference, gate_serving, main, reference_items, resolve_endpoint,
                            resolve_skill, tier_for, validate_table)
from orchard.tiers import TierConfig

REFERENCE = Path(__file__).parent / "fixtures" / "hemmingway_stage0_reference.md"


def cfg(escalation="large"):
    def tier(port, model, placement="chips"):
        return {"role": "r", "endpoint": f"http://127.0.0.1:{port}/v1", "model": model,
                "placement": placement}
    tiers = {"large": tier(8000, "Qwen/Qwen3.8-27B"), "small": tier(8001, "Qwen/Qwen3.8-27B"),
             "cpu": tier(11434, "qwen3-coder:30b", "cpu")}
    stages = {0: {"run": "large"}, 1: {"run": "small"}, 2: {"run": "small", "diagnose": "large"},
              3: {"run": "small", "diagnose": "large"},
              4: {"run": "small", "plan": "large", "diagnose": "large"},
              5: {"run": "small"}, 6: {"run": "small"}, 7: {"run": "none"}, 8: {"run": "small"}}
    return TierConfig(tiers, stages, escalation)


def write(root, rel, text="x"):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text if isinstance(text, str) else json.dumps(text))
    return p


# ---- the table ----------------------------------------------------------------------------------

def test_the_table_names_the_owner_skills_and_hardware_stages():
    assert [s.skill for s in STAGES] == ["delta-triage", "reference-gate", "functional-decoder",
                                         "full-model", "mesh-shrink", "serving-check",
                                         "serving-check", "", "operator-bundle"]
    assert [s.boards for s in STAGES] == [0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert STAGES[7].skip and all(s.skip is None for s in STAGES if s.number != 7)


def test_a_long_stage_without_a_resume_marker_is_refused():
    table = list(STAGES)
    table[3] = dataclasses.replace(STAGES[3], marker=None)
    with pytest.raises(ValueError, match="resume marker"):
        validate_table(tuple(table))


def test_tier_for_runs_plans_and_escalates():
    c = cfg(escalation="cpu")
    assert tier_for(c, 1, phase="run", escalated=False) == "small"
    assert tier_for(c, 4, phase="prepare", escalated=False) == "large"     # large plans
    assert tier_for(c, 4, phase="finish", escalated=False) == "small"      # small runs
    assert tier_for(c, 2, phase="finish", escalated=True) == "large"       # the diagnose tier
    assert tier_for(c, 5, phase="finish", escalated=True) == "cpu"         # [escalation] default
    with pytest.raises(ValueError):
        tier_for(c, 7, phase="run", escalated=False)


def test_resolve_endpoint_falls_back_only_to_the_same_model():
    c = cfg()
    only_large = lambda endpoint, model: endpoint.endswith(":8000/v1")
    assert resolve_endpoint(c, "large", only_large) == ("large", "http://127.0.0.1:8000/v1", None)
    used, endpoint, note = resolve_endpoint(c, "small", only_large)
    assert (used, endpoint) == ("large", "http://127.0.0.1:8000/v1")
    assert "tier small is not serving" in note
    with pytest.raises(TierUnavailable):
        resolve_endpoint(c, "cpu", only_large)        # no other tier serves qwen3-coder:30b


def test_resolve_skill_finds_flat_files_and_plugin_folders(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "x.md")
    write(b, "y/SKILL.md")
    write(b, "x.md")
    assert resolve_skill("x", [a, b]) == a / "x.md"       # the first directory wins
    assert resolve_skill("y", [a, b]) == b / "y" / "SKILL.md"
    assert resolve_skill("z", [a, b]) is None


def test_check_disk_compares_free_space_with_the_need():
    usage = lambda path: namedtuple("U", "total used free")(0, 0, 5e9)
    assert check_disk(".", 4, usage=usage) == (True, 5.0)
    assert check_disk(".", 6, usage=usage) == (False, 5.0)


# ---- the gates ----------------------------------------------------------------------------------

GOOD_DELTA = {
    "model": "Altworld/Hemmingway-1", "nearest_model": "Qwen/Qwen3.8-27B", "path": "weights-only",
    "differences": [
        {"area": a, "finding": f"{a} checked", "evidence": ["stages/0/evidence/diff.txt"]}
        for a in ("config", "tensors", "tensor_names", "tokenizer", "files", "generation_config")],
    "hazards": [{"area": a, "finding": f"{a} hazard"} for a in ("tensor_cache", "drafter", "disk")],
}


def stage(tmp_path, n, name, data):
    run = tmp_path / "run"
    write(run, f"stages/{n}/evidence/diff.txt", "evidence")
    write(run, f"stages/{n}/{name}", data)
    return run / "stages" / str(n), run


def test_delta_gate_passes_a_complete_delta(tmp_path):
    g = gate_delta(*stage(tmp_path, 0, "delta.json", GOOD_DELTA))
    assert g.ok, g.reasons
    assert "stages/0/evidence/diff.txt" in g.evidence


def test_delta_gate_names_what_is_missing(tmp_path):
    bad = {**GOOD_DELTA, "path": "maybe",
           "differences": [{"area": "vibes", "finding": "x", "evidence": ["a"]}]}
    g = gate_delta(*stage(tmp_path, 0, "delta.json", bad))
    assert not g.ok
    assert any("path must be" in r for r in g.reasons)
    assert any("differences[0] needs an area" in r for r in g.reasons)
    assert not gate_delta(tmp_path / "nothing", tmp_path).ok        # no delta.json at all


def test_evidence_outside_the_run_directory_does_not_count(tmp_path):
    outside = write(tmp_path, "outside.txt")
    sd, run = stage(tmp_path, 0, "delta.json", GOOD_DELTA)
    os.symlink(outside, run / "stages/0/evidence/link.txt")
    for rel in (str(outside), "../outside.txt", "stages/0/evidence/link.txt", "stages/0/evidence"):
        delta = {**GOOD_DELTA, "differences": [{"area": "config", "finding": "x", "evidence": [rel]}]}
        write(run, "stages/0/delta.json", delta)
        g = gate_delta(sd, run)
        assert not g.ok and "not a file inside the run directory" in g.reasons[0], rel


def test_reference_gate_needs_a_pass_verdict_and_passing_checks(tmp_path):
    good = {"verdict": "pass", "checks": [{"name": "greedy matches the card", "pass": True,
                                           "evidence": ["stages/1/evidence/diff.txt"]}]}
    assert gate_reference(*stage(tmp_path, 1, "reference.json", good)).ok
    bad = {"verdict": "pass", "checks": [{"name": "decodes forward", "pass": False,
                                          "evidence": ["stages/1/evidence/diff.txt"]}]}
    g = gate_reference(*stage(tmp_path, 1, "reference.json", bad))
    assert not g.ok and "'decodes forward' did not pass" in g.reasons[0]


def test_decoder_gate_uses_the_functional_decoder_bar_and_not_the_agents(tmp_path):
    ev = ["stages/2/evidence/diff.txt"]
    ok = {"pcc": 0.995, "argmax_match": True, "evidence": ev}
    assert gate_decoder(*stage(tmp_path, 2, "result.json", ok)).ok
    low = {"pcc": 0.9949, "pcc_threshold": 0.9, "argmax_match": True, "evidence": ev}
    assert not gate_decoder(*stage(tmp_path, 2, "result.json", low)).ok


def test_full_model_gate_needs_parity_and_a_top1_fraction(tmp_path):
    ev = ["stages/3/evidence/diff.txt"]
    assert gate_full_model(*stage(tmp_path, 3, "result.json", {"parity": True, "top1": 0.97, "evidence": ev})).ok
    assert not gate_full_model(*stage(tmp_path, 3, "result.json", {"parity": True, "top1": 97, "evidence": ev})).ok


def test_mesh_gate_needs_every_configuration_to_pass(tmp_path):
    ev = ["stages/4/evidence/diff.txt"]
    data = {"configs": [{"chips": 2, "pass": True, "evidence": ev}, {"chips": 1, "pass": False, "evidence": ev}]}
    g = gate_mesh(*stage(tmp_path, 4, "result.json", data))
    assert not g.ok and g.reasons == ("the 1-chip configuration did not pass",)


def test_serving_gate_needs_boot_passkey_and_canary(tmp_path):
    ev = ["stages/5/evidence/diff.txt"]
    data = {"checks": {"boots": {"pass": True, "evidence": ev}, "canary": {"pass": True, "evidence": ev}}}
    g = gate_serving(*stage(tmp_path, 5, "result.json", data))
    assert g.reasons == ("checks.passkey is missing",)


def test_numbers_gate_needs_a_label_on_every_number(tmp_path):
    ev = ["stages/6/evidence/diff.txt"]
    good = {"numbers": [{"name": "decode", "value": 80.1, "unit": "tok/s/user", "label": "measured", "evidence": ev},
                        {"name": "ttft", "value": None, "unit": "ms", "label": "TODO"}],
            "qualitative": {"evidence": ev}}
    assert gate_numbers(*stage(tmp_path, 6, "result.json", good)).ok
    guessed = {**good, "numbers": [{**good["numbers"][0], "label": "estimated"}]}
    assert not gate_numbers(*stage(tmp_path, 6, "result.json", guessed)).ok
    todo_only = {**good, "numbers": [good["numbers"][1]]}
    assert "no number is measured" in gate_numbers(*stage(tmp_path, 6, "result.json", todo_only)).reasons


def bundle(tmp_path, **files):
    b = tmp_path / "run" / "stages" / "8" / "bundle"
    b.mkdir(parents=True)
    base = {"RESULTS.md": "Results.", "RISKS.md": "Risks.",
            "PUBLISH_COMMANDS.txt": "tt-model push example/hemmingway-1-p300\n",
            "ledger.jsonl": "{}\n"}
    for name, text in {**base, **files}.items():
        if text is not None:
            (b / name).write_text(text)
    return b.parent, tmp_path / "run"


def test_a_complete_clean_bundle_passes_the_gate(tmp_path):
    g = gate_bundle(*bundle(tmp_path))
    assert g.ok, g.reasons
    assert "stages/8/bundle/PUBLISH_COMMANDS.txt" in g.evidence


def test_a_scrub_hit_or_a_missing_file_blocks_the_bundle(tmp_path):
    g = gate_bundle(*bundle(tmp_path, **{"RESULTS.md": f"Run on {socket.gethostname()}.",
                                         "PUBLISH_COMMANDS.txt": None}))
    assert not g.ok
    assert "bundle/PUBLISH_COMMANDS.txt is missing or empty" in g.reasons
    assert any(r.startswith("scrub: RESULTS.md: the hostname") for r in g.reasons)


# ---- stage 0 against the hand-written reference ---------------------------------------------------

def test_the_hemmingway_reference_is_read_item_by_item():
    diffs, hazards, path = reference_items(REFERENCE.read_text())
    assert diffs == ["config", "tensors", "tensor_names", "tokenizer", "files", "generation_config"]
    assert hazards == ["tensor_cache", "drafter", "disk"]
    assert path == "weights-only"


def test_a_delta_that_covers_the_reference_matches():
    assert compare_delta(GOOD_DELTA, REFERENCE.read_text())["ok"] is True


def test_a_delta_missing_an_item_or_with_the_wrong_path_does_not_match():
    delta = {**GOOD_DELTA, "path": "full-port",
             "differences": [d for d in GOOD_DELTA["differences"] if d["area"] != "tokenizer"],
             "hazards": GOOD_DELTA["hazards"][:2]}
    out = compare_delta(delta, REFERENCE.read_text())
    assert out["ok"] is False and out["path_matches"] is False
    assert out["missing"] == ["difference: tokenizer", "hazard: disk"]


def test_compare_delta_command_line(tmp_path, capsys):
    delta = write(tmp_path, "delta.json", GOOD_DELTA)
    assert main(["compare-delta", str(delta), str(REFERENCE)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    write(tmp_path, "delta.json", {**GOOD_DELTA, "path": "full-port"})
    assert main(["compare-delta", str(delta), str(REFERENCE)]) == 1
