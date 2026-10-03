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
                            gate_reference, gate_serving, gate_weights_swap, main, reference_items,
                            resolve_endpoint, resolve_skill, run_path, spec_for, tier_for,
                            validate_table)
from orchard.defaults import SWAP_TOP1_MIN
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


# ---- the path stage 0 chose ---------------------------------------------------------------------

def stage0_pass(**data):
    """A ledger holding stage 0's passing end, with `data` in it (the supervisor adds "path")."""
    return [{"seq": 1, "event": "stage_start", "stage": 0, "data": {}},
            {"seq": 2, "event": "stage_end", "stage": 0, "data": {"result": "pass", **data}}]


def delta_file(run, text):
    p = Path(run) / "stages" / "0" / "delta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text if isinstance(text, str) else json.dumps(text))


@pytest.mark.parametrize("path", ["weights-only", "full-port"])
def test_the_path_is_read_from_stage_0s_passing_end(tmp_path, path):
    assert run_path(stage0_pass(path=path), tmp_path) == path


def test_the_ledger_path_wins_over_a_delta_json_edited_later(tmp_path):
    # An agent in a later stage can rewrite stages/0/delta.json. The choice stays the one stage 0 made.
    delta_file(tmp_path, {**GOOD_DELTA, "path": "full-port"})
    assert run_path(stage0_pass(path="weights-only"), tmp_path) == "weights-only"


@pytest.mark.parametrize("path", ["weights-only", "full-port"])
def test_a_ledger_from_before_the_path_was_recorded_reads_delta_json(tmp_path, path):
    delta_file(tmp_path, {**GOOD_DELTA, "path": path})
    assert run_path(stage0_pass(), tmp_path) == path


@pytest.mark.parametrize("text", [None, "{not json", '["weights-only"]', {"path": "maybe"},
                                  {"path": None}, {"model": "x"}])
def test_an_unknown_path_is_none_and_never_weights_only(tmp_path, text):
    if text is not None:
        delta_file(tmp_path, text)
    assert run_path(stage0_pass(), tmp_path) is None


@pytest.mark.parametrize("recorded", [None, "maybe", 3])
def test_an_unknown_recorded_path_is_none(tmp_path, recorded):
    delta_file(tmp_path, {**GOOD_DELTA, "path": "weights-only"})
    assert run_path(stage0_pass(path=recorded), tmp_path) is None


def test_before_stage_0_passes_there_is_no_path(tmp_path):
    delta_file(tmp_path, GOOD_DELTA)              # written, but the gate has not passed it yet
    failed = [{"seq": 1, "event": "stage_end", "stage": 0, "data": {"result": "escalate"}}]
    assert run_path([], tmp_path) is None and run_path(failed, tmp_path) is None


def test_the_weights_only_path_gives_stage_2_the_swap_skill_and_gate():
    s = spec_for(2, "weights-only")
    assert (s.number, s.skill, s.gate, s.gate_file, s.boards) == (
        2, "weights-swap-check", gate_weights_swap, "result.json", 1)
    assert s.marker == STAGES[2].marker and s.skip is None


def test_the_weights_only_path_skips_stage_3_with_a_reason():
    s = spec_for(3, "weights-only")
    assert s.skip == ("weights-only path: the stage 2 serve-and-compare covers the full model")
    assert s.number == 3 and STAGES[3].skip is None


@pytest.mark.parametrize("path", ["full-port", None])
def test_a_full_port_or_unknown_path_keeps_todays_stage_2_and_3(path):
    assert spec_for(2, path) == STAGES[2] and spec_for(2, path).skill == "functional-decoder"
    assert spec_for(3, path) == STAGES[3]


@pytest.mark.parametrize("n", [0, 1, 4, 5, 6, 7, 8])
def test_the_path_changes_no_other_stage(n):
    for path in ("weights-only", "full-port", None):
        assert spec_for(n, path) == STAGES[n]


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


# The weights-swap-check skill's result.json, with the agreement of the corrected hand prototype
# (Hemmingway-1 weights served through MODEL_WEIGHTS_DIR, 30 of 32).
SWAP = {"serves": True, "server_ready_s": 280.5, "coherent": True, "free_run_text": "The sea was calm.",
        "top1_agreement": 0.94, "n_tokens": 32, "cache_dir": "cache/hemmingway-1/tt_cache",
        "evidence": ["stages/2/evidence/diff.txt"]}


def swap_reasons(tmp_path, **change):
    data = {k: v for k, v in {**SWAP, **change}.items() if v is not ...}
    return gate_weights_swap(*stage(tmp_path, 2, "result.json", data)).reasons


def test_weights_swap_gate_passes_the_prototype_numbers_and_records_the_evidence(tmp_path):
    g = gate_weights_swap(*stage(tmp_path, 2, "result.json", SWAP))
    assert g.ok, g.reasons
    assert g.evidence == ("stages/2/evidence/diff.txt",)


def test_weights_swap_gate_minimum_is_the_default(tmp_path):
    assert SWAP_TOP1_MIN == 0.85
    assert swap_reasons(tmp_path, top1_agreement=SWAP_TOP1_MIN) == ()


def test_weights_swap_gate_refuses_the_base_weights_standing_in(tmp_path):
    # The first hand prototype served the base Qwen3.8-27B weights by mistake (HF_MODEL in the
    # bundle's run.sh) and agreed with the Hemmingway-1 CPU reference on 25 of 32 tokens.
    reasons = swap_reasons(tmp_path, top1_agreement=25 / 32)
    assert len(reasons) == 1 and "below the minimum of 0.85" in reasons[0], reasons


@pytest.mark.parametrize("change,word", [
    ({"top1_agreement": 0.84}, "top1_agreement"),
    ({"top1_agreement": 0.0}, "top1_agreement"),
    ({"top1_agreement": 1.5}, "top1_agreement"),
    ({"top1_agreement": "0.78"}, "top1_agreement"),
    ({"top1_agreement": True}, "top1_agreement"),
    ({"top1_agreement": ...}, "top1_agreement"),
    ({"serves": False}, "serves"),
    ({"serves": "yes"}, "serves"),
    ({"coherent": False}, "coherent"),
    ({"coherent": ...}, "coherent"),
    ({"n_tokens": 15}, "n_tokens"),
    ({"n_tokens": 32.0}, "n_tokens"),
    ({"n_tokens": True}, "n_tokens"),
    ({"server_ready_s": 0}, "server_ready_s"),
    ({"server_ready_s": -3}, "server_ready_s"),
    ({"server_ready_s": "280"}, "server_ready_s"),
    ({"evidence": ["stages/2/evidence/missing.txt"]}, "evidence"),
    ({"evidence": []}, "evidence"),
])
def test_weights_swap_gate_names_the_field_that_fails(tmp_path, change, word):
    reasons = swap_reasons(tmp_path, **change)
    assert len(reasons) == 1 and word in reasons[0], reasons


def test_weights_swap_gate_refuses_an_honest_failure_and_names_the_test_failure(tmp_path):
    # The finish step's result after a failed hardware test: serves false, the failure text, and no
    # measured numbers. The gate must fail it, and its serves reason must carry the failure text,
    # so the escalation and the operator see why the test failed.
    reasons = swap_reasons(tmp_path, serves=False, failure="device init failed (exit code 4)",
                           coherent=None, n_tokens=None, top1_agreement=None, server_ready_s=None)
    assert "serves" in reasons[0] and "device init failed (exit code 4)" in reasons[0], reasons
    assert len(reasons) == 5                      # serves, coherent, n_tokens, top1, server_ready_s


def test_weights_swap_gate_without_a_failure_field_keeps_its_plain_serves_reason(tmp_path):
    [reason] = swap_reasons(tmp_path, serves=False)
    assert reason == "serves must be true (the server started and answered), got False"


def test_mesh_gate_needs_every_configuration_to_pass(tmp_path):
    ev = ["stages/4/evidence/diff.txt"]
    data = {"configs": [{"chips": 2, "pass": True, "evidence": ev}, {"chips": 1, "pass": False, "evidence": ev}]}
    g = gate_mesh(*stage(tmp_path, 4, "result.json", data))
    assert not g.ok and g.reasons == ("the 1-chip configuration did not pass",)


MESH_EV = ["stages/4/evidence/diff.txt"]


def mesh(tmp_path, configs, required):
    return gate_mesh(*stage(tmp_path, 4, "result.json", {"configs": configs}), required=required)


def cfg_entry(chips, ok, evidence=MESH_EV, **more):
    return {"chips": chips, "pass": ok, "evidence": evidence, **more}


def test_mesh_gate_passes_when_the_required_configurations_pass_and_an_optional_one_fails(tmp_path):
    g = mesh(tmp_path, [cfg_entry(2, True), cfg_entry(4, True),
                        cfg_entry(1, False, reason="does not fit one chip")], (2, 4))
    assert g.ok, g.reasons
    assert g.reasons == ()


def test_mesh_gate_names_a_required_configuration_that_is_missing(tmp_path):
    g = mesh(tmp_path, [cfg_entry(2, True), cfg_entry(1, True)], (2, 4))
    assert not g.ok
    assert g.reasons == ("the 4-chip configuration is required and has no entry",)


def test_mesh_gate_fails_a_required_configuration_that_did_not_pass(tmp_path):
    g = mesh(tmp_path, [cfg_entry(2, True), cfg_entry(4, False)], (2, 4))
    assert not g.ok and g.reasons == ("the 4-chip configuration did not pass",)


def test_mesh_gate_does_not_let_a_passing_duplicate_hide_a_failing_required_entry(tmp_path):
    g = mesh(tmp_path, [cfg_entry(2, True), cfg_entry(4, True), cfg_entry(4, False)], (2, 4))
    assert not g.ok and g.reasons == ("the 4-chip configuration did not pass",)
    g = mesh(tmp_path, [cfg_entry(2, True), cfg_entry(4, False), cfg_entry(4, True)], (2, 4))
    assert not g.ok


def test_mesh_gate_still_checks_the_shape_and_evidence_of_optional_entries(tmp_path):
    g = mesh(tmp_path, [cfg_entry(2, True), cfg_entry(1, True, evidence=["stages/4/evidence/none.txt"])], (2,))
    assert not g.ok                                  # an optional entry that claims a pass needs evidence
    g = mesh(tmp_path, [cfg_entry(2, True), {"chips": "one", "pass": False}], (2,))
    assert not g.ok and "configs[1] needs an integer chips" in g.reasons


def test_mesh_gate_with_no_required_list_keeps_every_listed_configuration_required(tmp_path):
    data = [cfg_entry(2, True), cfg_entry(1, False)]
    g = mesh(tmp_path, data, None)
    assert not g.ok and g.reasons == ("the 1-chip configuration did not pass",)
    assert not mesh(tmp_path, data, ()).ok
    assert not mesh(tmp_path, [], (2,)).ok           # a required list does not excuse an empty result


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


# ---- stage 4 on the weights-only path -------------------------------------------------------------

def swap_config(n, ok=True, **change):
    """One stage 4 configuration entry as the weights-swap-configs skill writes it."""
    if not ok:
        return {"chips": n, "pass": False, "reason": "no package fits", **change}
    swap = f"stages/4/configs/{n}/evidence/swap-check.json"
    entry = {**{k: v for k, v in SWAP.items() if k != "evidence"}, "chips": n, "pass": True,
             "evidence": [swap, f"stages/4/tests/{n}/output.txt"]}
    entry.update(change)
    return entry


def mesh_stage(tmp_path, configs, records=None, drafts=None):
    """A run with stage 4's result.json, the supervisor's test records and each test's
    swap-check.json. `records` maps chips to a test-result override (None leaves it out)."""
    run = tmp_path / "run"
    sd = run / "stages" / "4"
    write(run, "stages/4/result.json", {"configs": configs})
    for c in configs:
        n = c["chips"]
        write(run, f"stages/4/tests/{n}/output.txt", "test output")
        rec = {"returncode": 0, "timed_out": False, "chips": [f"chip{i}" for i in range(n)]}
        override = (records or {}).get(n, {})
        if override is not None:
            write(run, f"stages/4/tests/{n}/test-result.json", {**rec, **override})
        draft = (drafts or {}).get(n, {"top1_agreement": c.get("top1_agreement")})
        write(run, f"stages/4/configs/{n}/evidence/swap-check.json", {"result_draft": draft})
    return sd, run


def test_weights_only_stage_4_is_a_list_of_tests_with_its_own_gate_marker_and_disk():
    from orchard.defaults import STAGE4_SWAP_DISK_GB
    from orchard.stages import WEIGHTS_ONLY_STAGE_4 as s4, gate_mesh_swap
    assert s4.number == 4 and s4.tests and s4.boards == 2
    assert s4.skill == "weights-swap-configs" and s4.gate is gate_mesh_swap
    assert s4.marker == "tests/plan.json" and s4.disk_gb == STAGE4_SWAP_DISK_GB
    assert s4.budget_s == STAGES[4].budget_s
    assert STAGES[4].disk_gb == 80.0 and not STAGES[4].tests       # the plan 4 table is unchanged


def test_the_swap_mesh_gate_passes_required_configurations_with_records_and_an_optional_failure(tmp_path):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(2), swap_config(4), swap_config(1, ok=False)]),
                       required=(2, 4))
    assert g.ok, g.reasons
    assert "stages/4/tests/4/test-result.json" in g.evidence


def test_the_swap_mesh_gate_refuses_a_pass_for_a_configuration_whose_test_never_ran(tmp_path):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(2), swap_config(4)], records={4: None}),
                       required=(2, 4))
    assert g.reasons == ("the 4-chip configuration claims a pass, but the supervisor has no record "
                         "of its test (tests/4/test-result.json)",)


@pytest.mark.parametrize("record,words", [
    ({"returncode": 4}, "its test exited 4"),
    ({"returncode": None, "timed_out": True}, "did not finish before its deadline"),
    ({"chips": ["a", "b"]}, "which is not 4 chips"),
])
def test_the_swap_mesh_gate_refuses_a_pass_whose_test_did_not_finish_on_its_chips(tmp_path, record, words):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(4)], records={4: record}), required=(4,))
    assert len(g.reasons) == 1 and words in g.reasons[0], g.reasons


def test_the_swap_mesh_gate_holds_each_configuration_to_the_stage_2_bar(tmp_path):
    # The base weights standing in agreed 25 of 32 with the Hemmingway-1 reference.
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(4, top1_agreement=25 / 32)]), required=(4,))
    assert g.reasons == ("the 4-chip configuration: top1_agreement 0.78125 is below the minimum of 0.85",)


def test_the_swap_mesh_gate_checks_an_optional_configuration_that_claims_a_pass(tmp_path):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(2), swap_config(1)], records={1: None}),
                       required=(2,))
    assert g.reasons == ("the 1-chip configuration claims a pass, but the supervisor has no record "
                         "of its test (tests/1/test-result.json)",)


def test_the_swap_mesh_gate_needs_the_tests_own_report_with_the_same_agreement(tmp_path):
    from orchard.stages import gate_mesh_swap
    missing = swap_config(4, evidence=["stages/4/tests/4/output.txt"])
    g = gate_mesh_swap(*mesh_stage(tmp_path, [missing]), required=(4,))
    assert g.reasons == ("the 4-chip configuration: evidence must include "
                         "stages/4/configs/4/evidence/swap-check.json",)
    g = gate_mesh_swap(*mesh_stage(tmp_path / "b", [swap_config(4)], drafts={4: {"top1_agreement": 0.5}}),
                       required=(4,))
    assert g.reasons == ("the 4-chip configuration: top1_agreement 0.94 is not the result_draft's in "
                         "stages/4/configs/4/evidence/swap-check.json",)


def test_the_swap_mesh_gate_keeps_the_mesh_gates_rules(tmp_path):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(2)]), required=(2, 4))
    assert g.reasons == ("the 4-chip configuration is required and has no entry",)
