# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Slice 1 of the full-port route in the supervisor (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md): the opt-in flag, the
stage 2 route for class `full-port`, and the pause before stage 3. The chip, the coder and the model are fakes; the supervisor, the stage
machine, the gate and the hardware-test runner are the real ones."""
import json

import pytest

from fakes import Crash
from orchard import blocked, narrate, status, supervisor
from orchard.stages import SKIP_3_FULL_PORT, full_port_enabled
from orchard.supervisor import EXIT_ABORTED, EXIT_BLOCKED, EXIT_READY, Control, parse
from orchard import full_port
from port_check_fakes import HONEST, WRONG, install_fake_ttnn, python_path, tokenizer_json
from run_fakes import DELTA, FILES, argv, bringup, note
from test_supervisor import Rig, Stop, step_skills

pytestmark = pytest.mark.usefixtures("stub_tools")

REF_IDS = list(range(100, 132))

# What the coder's stage 1 would leave: the reference ids the stage 2 comparison reads.
STAGE_1 = {**FILES[(1, "run")], "evidence/reference/generated-ids.json": {"generated_ids": REF_IDS, "generated_text": "x"},
           "evidence/reference/prompt-ids.json": {"prompt_ids": [1, 2, 3], "prompt_text": "p"}}

# The port the coder writes (the check is the supervisor's template, which these tests run for real on the CPU with a fake `ttnn`).
PREPARE = {"port/port_model.py": HONEST,
           "hw_test.json": {"command": full_port.COMMAND, "deadline_s": 60}, "handoff.json": note(2)}
FINISH = {"result.json": {"runs": True, "load_s": 120.5, "coherent": True, "n_tokens": 32, "top1_agreement": 1.0,
                          "free_run_text": "w100 w101", "evidence": ["stages/2/evidence/port-check.json", "stages/2/evidence/hw-test-output.txt"]}}
FULL_PORT = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "path": "full-port"}}, (1, "run"): STAGE_1,
             (2, "prepare"): PREPARE, (2, "finish"): FINISH}


def from_record(rig):
    """The finish step's result.json, copied from the record the template wrote, as the skill tells the coder to do."""
    rec = json.loads((rig.run_dir / "stages" / "2" / "evidence" / "port-check.json").read_text())
    return {"result.json": {"runs": True, "coherent": True, "load_s": rec["load_s"], "n_tokens": len(rec["chip_ids"]),
                            "top1_agreement": rec["top1_agreement"], "free_run_text": rec["free_run_text"],
                            "evidence": ["stages/2/evidence/port-check.json", "stages/2/evidence/hw-test-output.txt"]}}


def make(rig, delta=None, over=None):
    files = {**FULL_PORT, **(over or {})}
    if delta is not None:
        files[(0, "run")] = {**FILES[(0, "run")], "delta.json": delta}

    def script(request):
        f = dict(files)
        record = rig.run_dir / "stages" / "2" / "evidence" / "port-check.json"
        if record.is_file() and "load_s" in record.read_text():          # not a record the coder wrote during prepare
            given = (over or {}).get((2, "finish"))
            if given is None:
                f[(2, "finish")] = from_record(rig)         # load_s is measured by the template, so the finish step copies it
            elif callable(given):
                f[(2, "finish")] = given(from_record(rig)["result.json"])
        return bringup(request, overrides=f)
    rig.script = script


def never_wait(rig):
    def fail():
        raise Stop("the run waited for the operator")
    rig.on_sleep = fail


def started(rig, n):
    return [e for e in rig.entries() if e["event"] == "stage_start" and e["stage"] == n]


def test_parse_has_a_full_port_flag_that_defaults_off(tmp_path):
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert parse(a).full_port is False and parse(a + ["--full-port"]).full_port is True


def test_run_start_records_the_flag_only_when_given(tmp_path):
    rig = Rig(tmp_path)
    try:
        make(rig)
        rig.args.full_port = True
        never_wait(rig)
        with pytest.raises(Stop):
            rig.run()
        start = next(e for e in rig.entries() if e["event"] == "run_start")
        assert start["data"]["full_port"] is True and full_port_enabled(rig.entries()) is True
    finally:
        rig.close()


def with_snapshot(r, tmp_path):
    """The run's `model=` input: a snapshot with a tokenizer.json, which the template decodes the free-run ids with. The fake runtime
    and the `tokenizers` package reach the test process through PYTHONPATH, as the real runtime reaches it through the venv."""
    snap = tmp_path / "snapshot"
    snap.mkdir()
    tokenizer_json(snap / "tokenizer.json")
    install_fake_ttnn(tmp_path)
    r.args = parse(argv(tmp_path, r.tiers, r.chip_server.endpoint, chips=2) + ["--input", f"model={snap}", "--env", f"PYTHONPATH={python_path(tmp_path)}"])


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    yield r
    r.close()


@pytest.fixture
def agent_finish(monkeypatch):
    """The tests that use this are about what the finish AGENT writes (a forged marker, an incoherent result, a swapped file). Since issue 456
    the supervisor writes result.json itself when the hardware test passed, so these tests switch that derivation off and the agent step runs
    as it did. tests/test_supervisor_full_port_derived.py covers the derived path."""
    from orchard import full_port_result
    monkeypatch.setattr(full_port_result, "derive", lambda *a, **k: (None, "switched off by the test"))


def test_a_full_port_with_the_flag_does_not_pause_before_stage_2_and_uses_the_full_port_skill(rig):
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")      # the pause before stage 3 is where the run ends
    assert rig.run() == EXIT_ABORTED
    assert not any(d.get("reason", "").startswith("stage 0 found a full port") for d in rig.decisions())
    assert step_skills(rig, 2) and all(s.endswith("orchard/skills/full-port-check.md") for s in step_skills(rig, 2))
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_stage_2_passes_through_the_real_test_runner_and_the_gate(rig):
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    stage = rig.run_dir / "stages" / "2"
    rec = json.loads((stage / "evidence" / "port-check.json").read_text())
    marker = json.loads((stage / "test-result.json").read_text())
    assert marker["returncode"] == 0 and rec["chips"] == marker["chips"] != []      # the test ran on the leased chips
    ends = rig.ends(2)
    assert any(e["path"] == "stages/2/result.json" for e in ends[0]["evidence"])


def test_the_run_skips_stage_3_pauses_before_stage_4_and_runs_none_of_stages_4_to_6(rig):
    """Issue 326: stage 3 runs the port on the configured chips. This run configures no count above one board (no --required-chips), so stage 2's
    board is that count: stage 3 is recorded as skipped with its reason, and the pause that was before stage 3 is now before stage 4."""
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    reasons = [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"]
    assert reasons == [supervisor.FULL_PORT_STAGES_UNBUILT]
    assert [(e["data"]["result"], e["data"].get("reason")) for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 3] == \
        [("skipped", SKIP_3_FULL_PORT)]
    for n in (4, 5, 6):
        assert not started(rig, n), n


def test_an_unattended_full_port_run_ends_blocked_with_its_own_code(rig):
    rig.args.unattended = True
    make(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert len(block) == 1 and block[0]["code"] == "full-port-stages-unbuilt"
    assert "stage 2" in (rig.run_dir / "BLOCKED.md").read_text()
    assert not rig.m.coder_running and rig.m.leases == {}          # the hardware is given back


def test_an_unattended_retry_does_not_pass_the_pause(rig):
    rig.args.unattended = True
    make(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert rig.run() == EXIT_BLOCKED
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["full-port-stages-unbuilt"] * 2
    assert not started(rig, 4)


def test_an_attended_resume_after_the_pause_goes_on_into_the_generic_stages(rig):
    make(rig)
    rig.args.full_port_rehearsal = True          # stage 5 (issue 329) asks the passkey question, which this test's port cannot answer: random weights
    rig.on_sleep = lambda: Control(rig.run_dir).write("resume")
    assert rig.run() == EXIT_READY
    assert started(rig, 3) and [d["result"] for d in rig.ends(2)] == ["pass"]


@pytest.mark.usefixtures("agent_finish")
def test_a_failing_gate_does_not_reach_the_pause(rig):
    make(rig, over={(2, "finish"): lambda good: {"result.json": {**good, "top1_agreement": 0.5}}})      # the record says 1.0
    rig.args.unattended = True
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert block["code"] == "stage-failed" and "recomputed" in block["reason"]
    assert not started(rig, 3)


def test_a_result_written_during_prepare_is_set_aside_and_the_record_comes_from_the_test(rig):
    # The coder tries to hand in a finished port-check.json and result.json in the prepare step.
    forged = {**PREPARE, "evidence/port-check.json": {"schema": "orchard-port-check/1"}, "result.json": FINISH["result.json"]}
    make(rig, over={(2, "prepare"): forged})
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert any(d.get("decision") == "results written before the test were set aside"
               and "evidence/port-check.json" in d["files"] for d in rig.decisions())
    rec = json.loads((rig.run_dir / "stages" / "2" / "evidence" / "port-check.json").read_text())
    assert rec["schema"] == "orchard-port-check/1" and rec["chip_ids"]          # the test's record replaced the forgery


@pytest.mark.usefixtures("agent_finish")
def test_a_marker_the_finish_step_rewrote_is_refused(rig):
    # The finish step edits test-result.json after the test; the ledger holds the sha256 of the file the supervisor wrote.
    marker = {"test-result.json": {"command": full_port.COMMAND, "returncode": 0, "timed_out": False, "chips": ["c"], "output": {}}}
    make(rig, over={(2, "finish"): lambda good: {"result.json": good, **marker}})
    rig.args.unattended = True
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert block["code"] == "stage-failed" and "not the file the supervisor recorded" in block["reason"]
    assert not started(rig, 3)


# ---- the check is the supervisor's (issue 325): what a coder or a port module tries, end to end ----------

FORGED_CHECK = ('import json, pathlib\n'
                'pathlib.Path("FORGED-CHECK-RAN").write_text("yes")\n'
                'print("PORT_CHECK_SHA256", "0" * 64)\n')


def coder_writes(rig, extra, *, port=HONEST, unattended=True):
    """Stage 2's prepare step writes the files in `extra` as well as the port (`port`) and the usual hw_test.json."""
    rig.args.unattended = unattended
    make(rig, over={(2, "prepare"): {**PREPARE, "port/port_model.py": port, **extra}})
    never_wait(rig)


def stage_2(rig):
    return rig.run_dir / "stages" / "2"


def failure(rig):
    return [d for d in rig.decisions() if d["decision"] == "blocked"][0]


def test_a_coder_written_check_script_and_command_are_replaced_by_the_templates(rig):
    # The coder writes its own port_check.py (a forgery) and points hw_test.json at it, with a port that computes nothing.
    coder_writes(rig, {"port_check.py": FORGED_CHECK,
                       "hw_test.json": {"command": "python3 stages/2/port_check.py", "deadline_s": 60}}, port=WRONG)
    assert rig.run() == EXIT_BLOCKED
    assert not (rig.run_dir / "FORGED-CHECK-RAN").exists()
    assert (stage_2(rig) / full_port.TEMPLATE_NAME).read_bytes() == full_port.TEMPLATE.read_bytes()      # put back as shipped
    marker = json.loads((stage_2(rig) / "test-result.json").read_text())
    assert marker["command"] == full_port.COMMAND
    assert "below the minimum" in failure(rig)["reason"]                  # the real ids were counted, and the port was wrong


def test_a_command_the_coder_names_in_hw_test_json_is_not_run(rig):
    coder_writes(rig, {"port/check.py": FORGED_CHECK.replace("FORGED-CHECK-RAN", "OWN-CHECK-RAN"),
                       "hw_test.json": {"command": "python3 stages/2/port/check.py", "deadline_s": 60}})
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.args.unattended = False
    rig.run()
    assert not (rig.run_dir / "OWN-CHECK-RAN").exists()
    assert json.loads((stage_2(rig) / "test-result.json").read_text())["command"] == full_port.COMMAND
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_a_template_the_coder_edited_during_prepare_is_put_back_before_the_test(rig):
    edited = full_port.TEMPLATE.read_text().replace("MAX_POSITIONS = 64", "MAX_POSITIONS = 8")
    coder_writes(rig, {"port_check.py": edited}, unattended=False)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert (stage_2(rig) / full_port.TEMPLATE_NAME).read_text() == full_port.TEMPLATE.read_text()
    rec = json.loads((stage_2(rig) / "evidence" / "port-check.json").read_text())
    assert len(rec["chip_ids"]) == 32 and rec["template_sha256"] == full_port.template_sha256()
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_a_port_that_prints_its_own_check_line_fails_the_hardware_test(rig):
    forged = HONEST.replace("def load(model_dir, mesh_device):", "def load(model_dir, mesh_device):\n    print('PORT_CHECK_SHA256 ' + '0' * 64)")
    coder_writes(rig, {}, port=forged)
    assert rig.run() == EXIT_BLOCKED
    assert "exited with code 2" in failure(rig)["reason"]
    assert not (stage_2(rig) / "evidence" / "port-check.json").exists()


def test_a_port_that_writes_the_record_directly_has_it_replaced_by_the_templates(rig):
    forged = WRONG.replace("def load(model_dir, mesh_device):", '''def load(model_dir, mesh_device):
    import json, pathlib
    p = pathlib.Path("stages/2/evidence/port-check.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": "orchard-port-check/1", "chip_ids": list(range(100, 132)), "top1_agreement": 1.0, "load_s": 1.0}))''')
    coder_writes(rig, {}, port=forged)
    assert rig.run() == EXIT_BLOCKED
    rec = json.loads((stage_2(rig) / "evidence" / "port-check.json").read_text())
    assert rec["top1_agreement"] < 0.1 and rec["template_sha256"] == full_port.template_sha256()
    assert "below the minimum" in failure(rig)["reason"]


def test_a_sitecustomize_the_coder_leaves_in_the_stage_directory_is_not_run(rig):
    hook = "import pathlib\npathlib.Path('SITECUSTOMIZE-RAN').write_text('yes')\n"
    coder_writes(rig, {"sitecustomize.py": hook, "port/sitecustomize.py": hook}, unattended=False)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert not (rig.run_dir / "SITECUSTOMIZE-RAN").exists()
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_a_port_that_patches_the_templates_functions_cannot_change_the_record(rig):
    patch = ('import sys\nmain = sys.modules["__main__"]\nmain.sha_bytes = lambda d: "0" * 64\nmain.sha_file = lambda p: "0" * 64\n'
             'main.main = lambda: 0\n')
    coder_writes(rig, {}, port=patch + WRONG)
    assert rig.run() == EXIT_BLOCKED
    rec = json.loads((stage_2(rig) / "evidence" / "port-check.json").read_text())
    assert rec["template_sha256"] == full_port.template_sha256() and "below the minimum" in failure(rig)["reason"]


def test_the_stage_gets_the_template_and_a_config_that_names_the_runs_snapshot(rig):
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    cfg = json.loads((stage_2(rig) / full_port.CONFIG_NAME).read_text())
    assert cfg["model_snapshot"].endswith("snapshot") and (stage_2(rig) / full_port.TEMPLATE_NAME).is_file()
    assert any(d.get("decision") == "port check template copied" for d in rig.decisions())


# ---- what the flag does not change ----------------------------------------------------------------

def test_the_flag_does_not_let_an_unknown_class_go_on(rig):
    rig.args.unattended = True
    unknown = {**DELTA, "path": "full-port", "class": "unknown"}
    make(rig, delta=unknown)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["needs-new-model-code"]
    assert not started(rig, 2)


def test_the_pause_for_an_unknown_class_says_the_class_is_unknown_and_why(rig):
    """Issue 437: the pause said "stage 0 found a full port" for a class that stage 0 could not decide, and the reason was in delta.json only."""
    rig.args.unattended = True
    reasons = ["sidecar head.pt cannot be read: it is not a safetensors file", "text config hidden_size differs: 2560 vs 5120"]
    make(rig, delta={**DELTA, "path": "full-port", "class": "unknown", "path_reasons": reasons})
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    [pause] = [d for d in rig.decisions() if d.get("decision") == "pause" and d.get("reason", "").startswith(supervisor.CLASS_UNKNOWN)]
    assert "class unknown" in pause["reason"] and "head.pt cannot be read" in pause["reason"] and "hidden_size differs" in pause["reason"]
    assert "found a full port" not in pause["reason"]
    assert status.classify_pause(pause)[0] == "full_port"
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["needs-new-model-code"]


def test_an_attended_resume_goes_on_after_the_unknown_class_pause_and_does_not_pause_again(rig):
    """The wiring behind is_full_port_pause: one resume is the acknowledgement, as it is for the full-port pause."""
    make(rig, delta={**DELTA, "path": "full-port", "class": "unknown", "path_reasons": ["sidecar head.pt cannot be read"]})
    waits = []

    def on_sleep():
        waits.append(1)
        if len(waits) > 1:
            raise Stop("paused a second time for the same unknown class")
        Control(rig.run_dir).write("resume")
    rig.on_sleep = on_sleep
    try:
        rig.run()
    except Stop:
        pass
    pauses = [d for d in rig.decisions() if d.get("decision") == "pause" and d.get("reason", "").startswith(supervisor.CLASS_UNKNOWN)]
    assert len(pauses) == 1 and started(rig, 2)


def test_a_full_port_class_without_the_flag_still_pauses_with_the_full_port_message(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    try:
        r.args.unattended = True
        make(r, delta={**DELTA, "path": "full-port", "class": "full-port"})
        never_wait(r)
        assert r.run() == EXIT_BLOCKED
        assert [d["reason"] for d in r.decisions() if d.get("decision") == "pause"] == [supervisor.FULL_PORT]
    finally:
        r.close()


def test_an_unknown_class_pause_is_an_acknowledged_full_port_pause_on_a_resume(tmp_path):
    entries = [{"event": "decision", "data": {"decision": "pause", "reason": supervisor.class_unknown_reason(["a reason"])}}]
    assert supervisor.is_full_port_pause(entries[0]["data"]["reason"]) and supervisor.is_full_port_pause(supervisor.FULL_PORT)
    assert not supervisor.is_full_port_pause("operator") and not supervisor.is_full_port_pause(supervisor.FULL_PORT_STAGES_UNBUILT)


@pytest.mark.parametrize("quoted", [
    "tensor model.layers.0.full-port route.pt has shape [2560]",           # the file name holds the full-port route words
    "a file named restricted head parity.bin is not readable",              # restricted head
    "the file scoring route.safetensors differs",                           # scoring route
    "sidecar audio path.gguf cannot be read",                               # audio path
    "weights failed after escalation.bin has the wrong size",               # stage failed
    "the repository says it is not escalated: text config differs",         # stage failed, the other phrase
    "the model needs a full port, said the card",                           # full port
    "cap since lasted",                                                     # budget words
])
def test_quoted_repository_text_cannot_change_what_kind_of_pause_the_unknown_class_pause_is(quoted):
    """Review of PR 438 (round 1): the pause reason quotes reasons from delta.json, which hold file names and text from the model's
    repository. `classify_pause` reads substrings, so a quoted phrase used to win over the fixed prefix and give the pause another
    code and another recovery text. The prefix is read first."""
    pause = {"decision": "pause", "reason": supervisor.class_unknown_reason([quoted])}
    assert status.classify_pause(pause) == ("full_port", None)
    assert blocked.block_code(pause) == "needs-new-model-code"
    assert supervisor.is_full_port_pause(pause["reason"])


def test_the_unknown_class_prefix_is_one_constant_that_status_and_the_supervisor_share():
    from orchard import classes
    assert supervisor.CLASS_UNKNOWN is classes.CLASS_UNKNOWN or supervisor.CLASS_UNKNOWN == classes.CLASS_UNKNOWN


def test_the_reasons_in_the_unknown_pause_are_few_and_short():
    text = supervisor.class_unknown_reason([f"reason {i} " + "x" * 500 for i in range(10)])
    assert "reason 0" in text and "reason 2" in text and "reason 3" not in text and len(text) < 1000
    assert supervisor.class_unknown_reason([]).endswith("a person decides whether to go on")


def test_a_full_port_without_the_flag_still_blocks_unattended(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    try:
        r.args.unattended = True
        make(r)
        never_wait(r)
        assert r.run() == EXIT_BLOCKED
        assert [d["code"] for d in r.decisions() if d["decision"] == "blocked"] == ["needs-new-model-code"]
    finally:
        r.close()


def test_the_flag_on_a_weights_only_run_changes_nothing(rig):
    assert rig.run() == EXIT_READY                      # the default scripted run is weights-only
    assert all(s.endswith("weights-swap-check.md") for s in step_skills(rig, 2))
    assert not any(d.get("reason") == supervisor.FULL_PORT_STAGES_UNBUILT for d in rig.decisions())


def test_a_resume_keeps_the_flag_the_run_started_with(rig):
    make(rig)
    never_wait(rig)
    with pytest.raises(Stop):
        rig.run()                                       # started with the flag, waiting at the pause before stage 3
    rig.args.full_port = False                          # the operator leaves the option out on the resume
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    assert rig.run(pid=200) == EXIT_ABORTED
    assert full_port_enabled(rig.entries()) is True


def test_the_flag_added_on_a_resume_before_stage_2_goes_on_into_the_full_port_stage_2(tmp_path):
    # The run of a model that was blocked with needs-new-model-code: stages 0 and 1 passed, stage 2 never started.
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    try:
        r.args.unattended = True
        make(r)
        never_wait(r)
        assert r.run() == EXIT_BLOCKED
        assert not started(r, 2) and not full_port_enabled(r.entries())
        r.args.full_port = True
        assert r.run(pid=200) == EXIT_BLOCKED               # now blocked at the pause before stage 3
        assert [d["code"] for d in r.decisions() if d["decision"] == "blocked"] == ["needs-new-model-code", "full-port-stages-unbuilt"]
        assert [d["by"] for d in r.decisions() if d["decision"] == "full-port route allowed"] == ["--full-port"]
        assert [d["result"] for d in r.ends(2)] == ["pass"] and full_port_enabled(r.entries())
        assert len([e for e in r.entries() if e["event"] == "run_start"]) == 1
    finally:
        r.close()


def test_a_flag_after_stage_2_started_without_it_is_refused(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    try:
        make(r)
        r.on_sleep = lambda: Control(r.run_dir).write("resume")        # an attended operator goes on past the full-port pause
        with pytest.raises(Crash):
            r.run(crash_if=lambda e: e["event"] == "stage_start" and e["stage"] == 2)
        r.args.full_port = True
        with pytest.raises(ValueError, match="stage 2 has begun"):
            r.run(pid=200)
    finally:
        r.close()


# ---- the block code ---------------------------------------------------------------------------------

def test_the_pause_is_classified_and_the_code_has_an_unblock_text():
    pause = {"reason": supervisor.FULL_PORT_STAGES_UNBUILT}
    assert status.classify_pause(pause)[0] == "full_port_unbuilt"
    assert blocked.block_code(pause) == "full-port-stages-unbuilt"
    assert "full-port-stages-unbuilt" in blocked.REASONS and narrate.UNBLOCK["full-port-stages-unbuilt"]
    assert "full-port-stages-unbuilt" not in blocked.AUTO_RETRY


def test_the_status_hint_says_a_person_decides():
    facts = {"state": "paused", "pause": {"kind": "full_port_unbuilt"}, "control_pending": False, "disk": {"run_dir_free_gb": 500},
             "pauses_at_stage": 1, "quiet_s": 0}
    hint = next(h for name, test, h in status.HINT_RULES if test(facts))
    assert "needs a decision" in hint


def test_the_older_full_port_pause_is_still_a_full_port_pause():
    assert status.classify_pause({"reason": supervisor.FULL_PORT})[0] == "full_port"


# ---- the rehearsal option (issue 357) ---------------------------------------------------------------

def incoherent(good):
    return {"result.json": {**good, "coherent": False}}


def test_parse_has_a_rehearsal_flag_that_defaults_off(tmp_path):
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert parse(a).full_port_rehearsal is False and parse(a + ["--full-port", "--full-port-rehearsal"]).full_port_rehearsal is True


@pytest.mark.usefixtures("agent_finish")
def test_without_the_rehearsal_flag_an_incoherent_port_fails_stage_2(rig):
    make(rig, over={(2, "finish"): incoherent})
    rig.args.unattended = True
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert failure(rig)["code"] == "stage-failed" and "coherent" in failure(rig)["reason"]
    assert not any(d.get("decision") == supervisor.REHEARSAL_WAIVED for d in rig.decisions())


@pytest.mark.usefixtures("agent_finish")
def test_with_the_rehearsal_flag_stage_2_passes_and_the_ledger_says_the_field_was_waived(rig):
    make(rig, over={(2, "finish"): incoherent})
    rig.args.full_port_rehearsal = True
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    waived = [d for d in rig.decisions() if d.get("decision") == supervisor.REHEARSAL_WAIVED]
    assert len(waived) == 1 and waived[0]["field"] == "coherent" and waived[0]["by"] == "--full-port-rehearsal"
    start = next(e for e in rig.entries() if e["event"] == "run_start")
    assert start["data"]["full_port_rehearsal"] is True


@pytest.mark.usefixtures("agent_finish")
def test_a_result_swapped_after_the_gate_read_it_still_gets_the_waiver_decision(rig, monkeypatch):
    # A writer replaces result.json with `coherent: true` right after the gate's read. The waiver the gate applied is on its result, so the
    # decision is written from that and the file is not opened again.
    real = full_port.check

    def swapping(stage_dir, run_dir, **kw):
        out = real(stage_dir, run_dir, **kw)
        path = stage_dir / "result.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "coherent": True}))
        return out
    monkeypatch.setattr(full_port, "check", swapping)
    make(rig, over={(2, "finish"): incoherent})
    rig.args.full_port_rehearsal = True
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    assert len([d for d in rig.decisions() if d.get("decision") == supervisor.REHEARSAL_WAIVED]) == 1


@pytest.mark.usefixtures("agent_finish")
def test_the_rehearsal_flag_keeps_every_other_check(rig):
    make(rig, over={(2, "finish"): lambda good: {"result.json": {**good, "coherent": False, "top1_agreement": 0.5}}})
    rig.args.full_port_rehearsal = True
    rig.args.unattended = True
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert failure(rig)["code"] == "stage-failed" and "recomputed" in failure(rig)["reason"]


def test_the_rehearsal_flag_needs_the_full_port_flag(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    try:
        r.args.full_port_rehearsal = True
        with pytest.raises(ValueError, match="needs --full-port"):
            r.run()
    finally:
        r.close()


def test_a_resume_keeps_the_rehearsal_the_run_started_with(rig):
    make(rig, over={(2, "finish"): incoherent})
    rig.args.full_port_rehearsal = True
    never_wait(rig)
    with pytest.raises(Stop):
        rig.run()                                       # stage 2 passed; waiting at the pause before stage 3
    rig.args.full_port_rehearsal = False
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    assert rig.run(pid=200) == EXIT_ABORTED
    assert supervisor.full_port_rehearsal(rig.entries()) is True


def test_the_rehearsal_flag_cannot_be_added_on_a_resume(rig):
    make(rig)
    never_wait(rig)
    with pytest.raises(Stop):
        rig.run()
    rig.args.full_port_rehearsal = True
    with pytest.raises(ValueError, match="only when the run starts"):
        rig.run(pid=200)


# ---- the facts file, the skeleton and the prepare note (orchard/port_facts.py) -----------------------------------------------

def _snapshot_with_weights(rig):
    from test_port_facts import CONFIG, TENSORS, write_safetensors
    snap = rig.tmp / "snapshot"
    (snap / "config.json").write_text(json.dumps(CONFIG))
    write_safetensors(snap / "model.safetensors", TENSORS)
    return snap


def _watch_first_prepare(rig):
    """Record what the stage directory and the prepare task hold when the coder's first prepare turn arrives."""
    from run_fakes import where
    seen = {}
    inner = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "prepare") and not seen:
            sd = rig.run_dir / "stages" / "2"
            facts, port = sd / "port_facts.json", sd / "port" / "port_model.py"
            seen.update(facts=json.loads(facts.read_text()) if facts.is_file() else None,
                        skeleton=port.read_text() if port.is_file() else None,
                        user=[m["content"] for m in request["messages"] if m["role"] == "user"][0])
        return inner(request)
    rig.script = script
    return seen


def test_the_coder_starts_from_a_facts_file_a_skeleton_and_a_task_that_says_to_write_early(rig):
    _snapshot_with_weights(rig)
    make(rig)
    seen = _watch_first_prepare(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert seen["facts"]["schema"] == "orchard-port-facts/1" and seen["facts"]["weights"]["layout"] == "single file"
    assert "NotImplementedError" in seen["skeleton"] and "def read_tensors" in seen["skeleton"]
    task = seen["user"].split("## From the supervisor", 1)[1]
    assert "port_facts.json" in task and "within 12 turns" in task and "skeleton" in task and full_port.COMMAND in task
    decisions = {d.get("decision") for d in rig.decisions()}
    assert {"port facts written", "port skeleton copied"} <= decisions


def test_a_snapshot_the_facts_cannot_read_leaves_no_facts_file_and_the_task_still_says_to_write_early(rig):
    make(rig)                                   # the rig's snapshot has a tokenizer and no config or weights
    seen = _watch_first_prepare(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert seen["facts"] is None and "NotImplementedError" in seen["skeleton"]
    task = seen["user"].split("## From the supervisor", 1)[1]
    assert "port_facts.json" not in task and "within 12 turns" in task
    problem = next(d for d in rig.decisions() if d.get("decision") == "port facts not written")
    assert "config.json" in problem["problem"]


def _stub(tmp_path, snapshot):
    from types import SimpleNamespace
    entries = []
    stub = SimpleNamespace(inputs={"model": str(snapshot)}, run_dir=tmp_path,
                           ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
    return stub, entries, supervisor.Supervisor


def _checkpoint(tmp_path, with_config=True):
    from test_port_facts import CONFIG, TENSORS, write_safetensors
    snap = tmp_path / "snap"
    snap.mkdir()
    if with_config:
        (snap / "config.json").write_text(json.dumps(CONFIG))
    write_safetensors(snap / "model.safetensors", TENSORS)
    return snap


def test_a_port_the_coder_already_wrote_is_kept_and_the_task_does_not_call_it_a_skeleton(tmp_path):
    from types import SimpleNamespace
    snap = _checkpoint(tmp_path)
    sd = tmp_path / "stages" / "2"
    (sd / "port").mkdir(parents=True)
    (sd / "port" / "port_model.py").write_text("# the coder's own port, mentions ttnn\n")
    stub, entries, S = _stub(tmp_path, snap)
    S._seed_full_port(stub, SimpleNamespace(number=2), sd)
    assert (sd / "port" / "port_model.py").read_text().startswith("# the coder's own port")
    assert stub._port_seed == {"facts": True, "skeleton": False}
    assert not any(k.get("decision") == "port skeleton copied" for _, k in entries)
    note = S._prepare_in_code(stub, SimpleNamespace(draft="full_port", skill="full-port-check", number=2), sd).note
    assert "port_facts.json" in note and "is a skeleton" not in note


@pytest.mark.timeout(30)
def test_a_config_that_is_a_fifo_leaves_no_facts_and_does_not_hang_the_stage(tmp_path):
    import os
    from types import SimpleNamespace
    snap = _checkpoint(tmp_path, with_config=False)
    os.mkfifo(snap / "config.json")
    sd = tmp_path / "stages" / "2"
    sd.mkdir(parents=True)
    stub, entries, S = _stub(tmp_path, snap)
    S._seed_full_port(stub, SimpleNamespace(number=2), sd)
    assert not (sd / "port_facts.json").exists() and stub._port_seed["facts"] is False
    assert any(k.get("decision") == "port facts not written" for _, k in entries)


def test_a_skeleton_that_cannot_be_copied_leaves_the_facts_and_a_ledger_decision_and_does_not_stop_the_stage(tmp_path):
    from types import SimpleNamespace
    snap = _checkpoint(tmp_path)
    sd = tmp_path / "stages" / "2"
    sd.mkdir(parents=True)
    (sd / "port").write_text("a file where the port directory should be")
    stub, entries, S = _stub(tmp_path, snap)
    S._seed_full_port(stub, SimpleNamespace(number=2), sd)
    assert (sd / "port_facts.json").is_file() and stub._port_seed == {"facts": True, "skeleton": False}
    assert any(k.get("decision") == "port skeleton not copied" for _, k in entries)


def test_an_evidence_record_the_ledger_refuses_leaves_a_ledger_decision_and_does_not_stop_the_stage(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orchard import safeio
    snap = _checkpoint(tmp_path)
    sd = tmp_path / "stages" / "2"
    sd.mkdir(parents=True)

    def refuse(run_dir, path):
        raise safeio.PathRefused("the evidence file is not inside the run directory")
    monkeypatch.setattr(supervisor, "evidence_record", refuse)
    stub, entries, S = _stub(tmp_path, snap)
    S._seed_full_port(stub, SimpleNamespace(number=2), sd)
    assert stub._port_seed["facts"] is False
    assert any(k.get("decision") == "port facts not written" for _, k in entries)


def test_an_unexpected_error_in_the_facts_step_leaves_a_ledger_decision_and_does_not_stop_the_stage(tmp_path, monkeypatch):
    # The facts are advisory: before they existed the coder worked from its skill. An error nobody listed (Copilot, PR 432: a 5,000-digit tensor
    # name raised ValueError) must not stop the run, and its message, which may quote a tensor name, stays out of the ledger.
    from types import SimpleNamespace
    from orchard import port_facts
    snap = _checkpoint(tmp_path)
    sd = tmp_path / "stages" / "2"
    sd.mkdir(parents=True)

    def boom(*a, **k):
        raise ValueError("Exceeds the limit (4300 digits) in model.layers.99999.weight")
    monkeypatch.setattr(port_facts, "build", boom)
    stub, entries, S = _stub(tmp_path, snap)
    S._seed_full_port(stub, SimpleNamespace(number=2), sd)
    assert stub._port_seed["facts"] is False and not (sd / "port_facts.json").exists() and stub._port_seed["skeleton"] is True
    problem = next(k for _, k in entries if k.get("decision") == "port facts not written")["problem"]
    assert "ValueError" in problem and "99999" not in problem
