# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The scoring route in the supervisor (issue 378, docs/superpowers/specs/2026-11-02-scoring-model-check-design.md): stage 0 records the model
kind, a reranker or an embedding model of class weights-only gets the scoring stage 1 and 2 (no finish step), the real check runs on the CPU with
a fake `ttnn`, and the run pauses before stage 3. The chip, the coder and the model are fakes; the supervisor, the stage machine, the gates and the
hardware-test runner are the real ones."""
import hashlib
import json

import pytest

from orchard import blocked, scoring_check, supervisor
from orchard.supervisor import EXIT_ABORTED, EXIT_BLOCKED, Control, parse
from port_check_fakes import install_fake_ttnn, python_path
from run_fakes import DELTA, FILES, argv, bringup, note, where
from scoring_check_fakes import HONEST, NOISY, REFERENCE_SCORES, pair_rows
from scoring_embed_fakes import HONEST_EMBED, N_EMBED, embed_rows, fake_vector
from test_supervisor import Rig, Stop, step_skills

pytestmark = pytest.mark.usefixtures("stub_tools")

REF = "evidence/reference/"


def rerank_files():
    rows = json.dumps({"schema": "orchard-scoring-pairs/1", "model_kind": "reranker", "pairs": pair_rows()}) + "\n"
    ref = {"schema": "orchard-scoring-reference/1", "model_kind": "reranker", "precision": "fp32", "scores": REFERENCE_SCORES,
           "pairs": {"file": "scoring-pairs.json", "sha256": hashlib.sha256(rows.encode()).hexdigest(), "count": len(pair_rows())}}
    return {REF + "scoring-pairs.json": rows, REF + "scoring-reference.json": ref}


def embed_files():
    rows = embed_rows()
    pairs = json.dumps({"schema": "orchard-scoring-pairs/1", "model_kind": "embedding", "pairs": rows}) + "\n"
    ref = {"schema": "orchard-scoring-reference/1", "model_kind": "embedding", "precision": "fp32", "vectors": [fake_vector(n) for n in range(len(rows))],
           "pairs": {"file": "scoring-pairs.json", "sha256": hashlib.sha256(pairs.encode()).hexdigest(), "count": len(rows)}}
    return {REF + "scoring-pairs.json": pairs, REF + "scoring-reference.json": ref}


def prepare(port):
    return {"port/port_model.py": port, "hw_test.json": {"command": scoring_check.COMMAND, "deadline_s": 60}, "handoff.json": note(2)}


def make(rig, *, stage1=None, port=HONEST, delta=None, prepare_extra=None):
    files = {(0, "run"): {**FILES[(0, "run")], "delta.json": delta or DELTA}, (1, "run"): stage1 if stage1 is not None else rerank_files(),
             (2, "prepare"): {**prepare(port), **(prepare_extra or {})}}
    rig.script = lambda request: bringup(request, overrides=files)


def never_wait(rig):
    def fail():
        raise Stop("the run waited for the operator")
    rig.on_sleep = fail


def started(rig, n):
    return [e for e in rig.entries() if e["event"] == "stage_start" and e["stage"] == n]


def snapshot(tmp_path, *, tag="text-ranking", archs=("Qwen3ForCausalLM",)):
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"architectures": list(archs), "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 4,
                                                  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 151936}))
    (snap / "README.md").write_text(f"---\npipeline_tag: {tag}\n---\n" if tag else "# no front matter\n")
    install_fake_ttnn(tmp_path)
    return snap


def with_model(r, tmp_path, *extra, **kw):
    snap = snapshot(tmp_path, **kw)
    r.args = parse(argv(tmp_path, r.tiers, r.chip_server.endpoint, chips=2) + ["--input", f"model={snap}",
                                                                              "--env", f"PYTHONPATH={python_path(tmp_path)}", *extra])
    return snap


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_model(r, tmp_path)
    yield r
    r.close()


def abort_at_pause(rig):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


# ---- stage 0 ---------------------------------------------------------------------------------------------------------------

def test_stage_0_records_the_kind_from_the_card_and_the_hazard_scan_leaves_the_head_hazards_out(rig):
    make(rig)
    abort_at_pause(rig)
    rig.run()
    kind = rig.ends(0)[0]["model_kind"]
    assert kind["kind"] == "reranker" and kind["source"] == "model card" and kind["pipeline_tag"] == "text-ranking"
    scan = json.loads((rig.run_dir / "stages" / "0" / "hazard-scan.json").read_text())
    assert scan["facts"]["model_kind"] == "reranker" and not [h for h in scan["hazards"] if h["name"].startswith("vocab_")]
    assert any("vocabulary hazards" in n for n in scan["notes"])


def test_the_operator_flag_decides_the_kind_and_the_ledger_says_so(tmp_path):
    r = Rig(tmp_path)
    try:
        with_model(r, tmp_path, "--model-kind", "embedding")
        make(r, stage1=embed_files(), port=HONEST_EMBED)
        abort_at_pause(r)
        r.run()
        kind = r.ends(0)[0]["model_kind"]
        assert kind["kind"] == "embedding" and kind["source"] == "operator" and kind["pipeline_tag"] == "text-ranking"
        assert [d["result"] for d in r.ends(2)] == ["pass"]
    finally:
        r.close()


def test_a_model_with_no_card_tag_is_generative_and_takes_the_stages_it_always_took(tmp_path):
    r = Rig(tmp_path)
    try:
        with_model(r, tmp_path, tag=None)
        r.script = lambda request: bringup(request)
        assert r.run() == supervisor.EXIT_READY
        assert r.ends(0)[0]["model_kind"]["kind"] == "generative" and r.ends(0)[0]["model_kind"]["source"] == "default"
        assert not any("scoring" in s for n in range(8) for s in step_skills(r, n))
    finally:
        r.close()


# ---- stages 1 and 2 --------------------------------------------------------------------------------------------------------

def test_a_reranker_run_uses_the_scoring_skills_runs_the_real_check_and_has_no_finish_step(rig):
    make(rig)
    abort_at_pause(rig)
    assert rig.run() == EXIT_ABORTED
    assert step_skills(rig, 1) and all(s.endswith("orchard/skills/scoring-reference.md") for s in step_skills(rig, 1))
    assert step_skills(rig, 2) and all(s.endswith("orchard/skills/scoring-check.md") for s in step_skills(rig, 2))
    phases = [e["data"]["phase"] for e in rig.entries() if e["event"] == "decision" and e["stage"] == 2 and e["data"]["decision"] == "agent step"]
    assert phases == ["prepare"]                                             # the check's record is the result
    assert [d["result"] for d in rig.ends(1)] == ["pass"] and [d["result"] for d in rig.ends(2)] == ["pass"]
    stage = rig.run_dir / "stages" / "2"
    rec = json.loads((stage / "evidence" / "scoring-check.json").read_text())
    marker = json.loads((stage / "test-result.json").read_text())
    assert rec["kind"] == "reranker" and marker["returncode"] == 0 and rec["chips"] == marker["chips"] != []
    assert marker["command"] == scoring_check.COMMAND
    assert not (stage / "result.json").exists()


def test_the_ledger_holds_the_hashes_of_the_template_the_config_and_the_stage_files(rig):
    make(rig)
    abort_at_pause(rig)
    rig.run()
    copied = [d for d in rig.decisions() if d.get("decision") == "scoring check template copied"]
    assert copied and copied[0]["kind"] == "reranker"
    stage = rig.run_dir / "stages" / "2"
    assert copied[0]["template"] == {"path": "stages/2/scoring_check.py", "sha256": hashlib.sha256((stage / "scoring_check.py").read_bytes()).hexdigest()}
    assert copied[0]["config"]["path"] == "stages/2/scoring_check_config.json"
    cfg = json.loads((stage / "scoring_check_config.json").read_text())
    assert cfg["kind"] == "reranker" and cfg["bounds"] == scoring_check.DEFAULT_BOUNDS
    paths = {e["path"]: e["sha256"] for e in rig.ends(2)[0]["evidence"]}
    assert paths["stages/2/evidence/scoring-check.json"] == hashlib.sha256((stage / "evidence" / "scoring-check.json").read_bytes()).hexdigest()
    s1 = {e["path"]: e["sha256"] for e in rig.ends(1)[0]["evidence"]}
    ev = rig.run_dir / "stages" / "1" / "evidence" / "reference"
    assert s1["stages/1/evidence/reference/scoring-reference.json"] == hashlib.sha256((ev / "scoring-reference.json").read_bytes()).hexdigest()
    assert s1["stages/1/evidence/reference/scoring-pairs.json"] == hashlib.sha256((ev / "scoring-pairs.json").read_bytes()).hexdigest()


def test_an_embedding_run_passes_through_the_same_route(rig):
    rig.args.model_kind = "embedding"
    make(rig, stage1=embed_files(), port=HONEST_EMBED)
    abort_at_pause(rig)
    rig.run()
    rec = json.loads((rig.run_dir / "stages" / "2" / "evidence" / "scoring-check.json").read_text())
    assert rec["kind"] == "embedding" and len(rec["chip_vectors"]) == N_EMBED
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_a_port_that_misses_the_bounds_fails_stage_2_with_the_checks_reason_and_no_finish_step(rig):
    rig.args.unattended = True
    make(rig, port=NOISY)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert block["code"] == "stage-failed" and "mean absolute score difference" in block["reason"]
    assert not started(rig, 3)
    assert all(e["data"]["phase"] == "prepare" for e in rig.entries() if e["event"] == "decision" and e["stage"] == 2 and e["data"]["decision"] == "agent step")


def test_stage_1_files_of_the_other_kind_are_refused_by_stage_1(rig):
    rig.args.unattended = True
    rig.args.model_kind = "embedding"          # the run is an embedding run; the recorder files are a reranker's
    make(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert block["code"] == "stage-failed" and "pair file is for kind 'reranker'" in block["reason"]
    assert not started(rig, 2)


def test_a_reference_changed_after_stage_1_passed_is_refused_by_stage_2(rig):
    rig.args.unattended = True
    # Something rewrites stage 1's reference file while stage 2's prepare step runs: the same scores with one more field, so the check still
    # passes against it, and the record names the new file's hash. Only the ledger's stage 1 entry holds the hash of the file stage 1 accepted.
    make(rig)
    inner = rig.script
    ref_file = rig.run_dir / "stages" / "1" / "evidence" / "reference" / "scoring-reference.json"

    def script(request):
        if where(request) == (2, "prepare"):
            ref_file.write_text(json.dumps({**json.loads(ref_file.read_text()), "note": "edited after stage 1 passed"}))
        return inner(request)
    rig.script = script
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert block["code"] == "stage-failed" and "is not the file stage 1's gate accepted" in block["reason"]
    assert [d["result"] for d in rig.ends(2)] != ["pass"] and not started(rig, 3)


def test_a_coder_written_check_script_is_replaced_and_its_command_is_not_run(rig):
    rig.args.unattended = True
    forged = 'import pathlib\npathlib.Path("FORGED-CHECK-RAN").write_text("yes")\nprint("SCORING_CHECK_SHA256", "0" * 64)\n'
    make(rig, port=NOISY, prepare_extra={"scoring_check.py": forged, "hw_test.json": {"command": "python3 stages/2/scoring_check.py", "deadline_s": 60}})
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert not (rig.run_dir / "FORGED-CHECK-RAN").exists()
    assert (rig.run_dir / "stages" / "2" / "scoring_check.py").read_bytes() == scoring_check.TEMPLATE.read_bytes()
    assert json.loads((rig.run_dir / "stages" / "2" / "test-result.json").read_text())["command"] == scoring_check.COMMAND


def test_a_record_the_coder_wrote_during_prepare_is_set_aside(rig):
    make(rig, prepare_extra={"evidence/scoring-check.json": {"schema": "orchard-scoring-check/1", "forged": True}})
    abort_at_pause(rig)
    rig.run()
    assert any(d.get("decision") == "results written before the test were set aside" and "evidence/scoring-check.json" in d["files"] for d in rig.decisions())
    rec = json.loads((rig.run_dir / "stages" / "2" / "evidence" / "scoring-check.json").read_text())
    assert "forged" not in rec and rec["chip_scores"]


# ---- the pauses ------------------------------------------------------------------------------------------------------------

def test_the_run_pauses_before_stage_3_and_runs_none_of_stages_3_to_6(rig):
    make(rig)
    abort_at_pause(rig)
    rig.run()
    assert [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"] == [supervisor.SCORING_STAGES_UNBUILT]
    for n in (3, 4, 5, 6):
        assert not started(rig, n), n


def test_an_unattended_scoring_run_ends_blocked_with_the_scoring_route_code_and_a_retry_blocks_again(rig):
    rig.args.unattended = True
    make(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert rig.run() == EXIT_BLOCKED
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["scoring-route"] * 2
    assert "stage 2" in (rig.run_dir / "BLOCKED.md").read_text() and "scoring-route" in blocked.REASONS
    assert not started(rig, 3) and not rig.m.coder_running and rig.m.leases == {}


def test_an_attended_resume_after_the_pause_goes_on_to_stage_3_and_the_pause_is_not_repeated(rig):
    make(rig)
    waits = []

    def on_sleep():
        waits.append(1)
        if len(waits) > 1:
            raise Stop("the run paused a second time")      # the one resume is enough
        Control(rig.run_dir).write("resume")
    rig.on_sleep = on_sleep
    try:
        rig.run()
    except Stop:
        pass
    assert [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"][:1] == [supervisor.SCORING_STAGES_UNBUILT]
    assert started(rig, 3) and [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"].count(supervisor.SCORING_STAGES_UNBUILT) == 1


def test_a_scoring_model_of_a_class_with_no_route_pauses_before_stage_1(rig):
    rig.args.unattended = True
    make(rig, delta={**DELTA, "path": "full-port", "class": "unknown", "path_reasons": ["the safetensors headers could not be read"]})
    rig.args.full_port = True               # the flag does not open the route for class unknown
    never_wait(rig)
    rig.run()
    blocks = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert blocks and blocks[0]["code"] == "scoring-route" and "no scoring route" in blocks[0]["reason"]
    assert not started(rig, 1)


FULL = {**DELTA, "path": "full-port", "class": "full-port", "path_reasons": ["text config differs: x"]}


def test_a_full_port_scoring_model_without_the_flag_pauses_before_stage_1_and_says_to_pass_it(rig):
    rig.args.unattended = True
    make(rig, delta=FULL)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    blocks = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert [b["code"] for b in blocks] == ["scoring-route"] and "--full-port" in blocks[0]["reason"]
    assert not started(rig, 1)


def test_a_full_port_scoring_model_with_the_flag_takes_the_scoring_route(rig):
    rig.args.full_port = True
    make(rig, delta=FULL)
    abort_at_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and step_skills(rig, 2)[0].endswith("scoring-check.md")
