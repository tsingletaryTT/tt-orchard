# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 420, end to end: a full-port stage 2 whose port check fails in seconds gives the coder a bounded number of repair rounds in the same
attempt. The chips, the coder and the model are fakes; the supervisor, the stage machine, the gate, the test runner and the check template
(which runs for real on the CPU with a fake `ttnn`) are the real ones."""
import json
import re

import pytest

from fakes import Crash
from orchard import full_port, port_repair
from orchard.supervisor import EXIT_ABORTED, EXIT_BLOCKED, Control, parse
from port_check_fakes import HONEST
from run_fakes import argv, bringup, call, where
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, STAGE_1, make, never_wait, stage_2, started, with_snapshot

pytestmark = pytest.mark.usefixtures("stub_tools")

# The port the tiny Qwen4 rehearsal's coder wrote: it reads a weight the checkpoint does not have, so `load` fails.
BROKEN = HONEST.replace('return {"dir": model_dir, "device": mesh_device}', "raise KeyError('model.layers.2.ple.conv1d.weight')")


def broken(k):
    """A port that still fails at load but is a different file, of a different size, from every other round's."""
    return BROKEN + "# edit " + "x" * k + "\n"          # a different length each time: write_file's reply names the size, and three equal replies in a row trip the watchdog


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    yield r
    r.close()


def repairs(rig, per_round):
    """Wrap the rig's script: a repair step writes `per_round[k - 1]` (the last entry repeats), k read from the supervisor's message."""
    base = rig.script
    seen = []

    def script(request):
        if "tools" in request and where(request) == (2, "repair"):
            found = re.search(r"round (\d+) of (\d+)", request["messages"][1]["content"])
            k = int(found.group(1)) if found else 1
            seen.append(request["messages"][1]["content"])
            return bringup(request, overrides={(2, "repair"): per_round[min(k - 1, len(per_round) - 1)]})
        return base(request)
    rig.script = script
    return seen


def start(rig, port=BROKEN, per_round=None, *, unattended=True):
    """Stage 2's prepare writes `port`; each repair round writes the files in `per_round`. Returns the repair prompts seen."""
    make(rig, over={(2, "prepare"): {**PREPARE, "port/port_model.py": port}})
    seen = repairs(rig, per_round if per_round is not None else [{"port/port_model.py": HONEST}])
    rig.args.unattended = unattended
    if unattended:
        never_wait(rig)
    else:
        rig.on_sleep = lambda: Control(rig.run_dir).write("abort")      # the pause before stage 3 is where an attended run ends
    return seen


def rounds(rig):
    return [d for d in rig.decisions() if d.get("decision") == port_repair.DECISION]


def stopped(rig):
    return [d for d in rig.decisions() if d.get("decision") == port_repair.STOPPED]


def attempts(rig):
    """The round numbers each attempt of stage 2 used, in order (a resume does not start a new attempt)."""
    out = []
    for e in rig.entries():
        if e["stage"] != 2:
            continue
        if e["event"] == "stage_start" and e["data"].get("resumed") is False:
            out.append([])
        elif e["event"] == "decision" and e["data"].get("decision") == port_repair.DECISION:
            out[-1].append(e["data"]["round"])
    return out


def exit_codes(rig):
    return [e["data"]["returncode"] for e in rig.entries() if e["event"] == "evidence" and e["stage"] == 2 and e["data"].get("what") == "hardware test"]


# ---- a port that fails at load on round 1 and passes on round 2 ---------------------------------------------------------------

def test_a_port_that_fails_at_load_is_repaired_in_the_same_attempt(rig):
    start(rig, unattended=False)
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and len(started(rig, 2)) == 1       # one attempt, no escalation
    assert exit_codes(rig) == [2, 0]                                                           # the failed check, then the same check again
    assert [(d["round"], d["of"], d["op"]) for d in rounds(rig)] == [(1, 3, "load")]


def test_the_ledger_holds_both_test_records_and_the_gate_accepts_the_last(rig):
    start(rig, unattended=False)
    rig.run()
    entries = rig.entries()
    marker = json.loads((stage_2(rig) / "test-result.json").read_text())
    assert marker["returncode"] == 0
    assert [e["data"]["returncode"] for e in entries if e["event"] == "evidence" and e["stage"] == 2
            and e["data"].get("what") == "hardware test"] == [2, 0]
    rec = json.loads((stage_2(rig) / "evidence" / "port-check.json").read_text())
    assert rec["template_sha256"] == full_port.template_sha256() and rec["top1_agreement"] == 1.0


def test_the_round_is_in_the_ledger_before_the_repair_step_starts_and_the_second_test_comes_after_the_step(rig):
    start(rig, unattended=False)
    rig.run()
    entries = rig.entries()
    decided = next(i for i, e in enumerate(entries) if e["data"].get("decision") == port_repair.DECISION)
    step = next(i for i, e in enumerate(entries) if e["data"].get("decision") == "agent step" and e["data"].get("phase") == "repair")
    second = [i for i, e in enumerate(entries) if e["event"] == "evidence" and e["data"].get("what") == "hardware test"][1]
    assert decided < step < second
    assert entries[-1]["event"] != "stage_start" and entries[decided]["stage"] == 2


def test_a_home_path_in_the_failure_text_is_redacted_in_the_ledger(rig):
    secret = f"{rig.home}/secret-model/weights.bin"
    start(rig, port=BROKEN.replace("model.layers.2.ple.conv1d.weight", secret), unattended=False)
    rig.run()
    failure = rounds(rig)[0]["failure"]
    assert str(rig.home) not in failure and "<home>/secret-model/weights.bin" in failure


def test_the_repair_prompt_names_the_round_the_failure_and_the_one_directory_the_coder_may_edit(rig):
    seen = start(rig, unattended=False)
    rig.run()
    text = seen[0]
    assert "round 1 of 3" in text and "KeyError" in text and "model.layers.2.ple.conv1d.weight" in text
    assert "stages/2/port/" in text and "Do not run the check" in text


def test_the_stage_clock_runs_through_every_round(rig):
    start(rig, unattended=False)
    rig.run()
    entries = rig.entries()
    first = next(i for i, e in enumerate(entries) if e["event"] == "stage_start" and e["stage"] == 2)
    last = max(i for i, e in enumerate(entries) if e["event"] == "stage_end" and e["stage"] == 2)
    tests = [i for i, e in enumerate(entries) if e["event"] == "evidence" and e["data"].get("what") == "hardware test"]
    assert first < tests[0] < tests[1] < last           # both tests are inside the one attempt the stage's clock and budget measure


# ---- a port that never passes: the rounds stop at the limit, and the stage fails as it did before --------------------------------

def test_a_port_that_never_passes_stops_after_the_limit_in_each_attempt(rig):
    rig.args.port_repair_rounds = 2
    start(rig, per_round=[{"port/port_model.py": broken(k)} for k in range(1, 6)])
    assert rig.run() == EXIT_BLOCKED
    assert attempts(rig) == [[1, 2], [1, 2]]                    # the first attempt, then the escalated one: never more than 2 each
    assert [d["reason"] for d in stopped(rig)] == ["spent", "spent"]
    block = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert block["code"] == "stage-failed" and "exited with code 2" in block["reason"]


def test_the_default_is_three_rounds(rig):
    start(rig, per_round=[{"port/port_model.py": broken(k)} for k in range(1, 6)])
    assert rig.run() == EXIT_BLOCKED
    assert attempts(rig) == [[1, 2, 3], [1, 2, 3]]
    start_entry = next(e for e in rig.entries() if e["event"] == "run_start")
    assert start_entry["data"]["port_repair_rounds"] == 3


def test_zero_rounds_is_the_old_behaviour(rig):
    rig.args.port_repair_rounds = 0
    start(rig)
    assert rig.run() == EXIT_BLOCKED
    assert rounds(rig) == [] and exit_codes(rig)[:1] == [2] and len(started(rig, 2)) == 2        # fail, escalate, fail: no round in between
    assert stopped(rig) == []                                       # and nothing in the ledger says rounds were considered


def test_a_round_that_changes_nothing_ends_the_rounds_and_runs_no_test(rig):
    start(rig, per_round=[{}])                    # the coder replies without writing
    assert rig.run() == EXIT_BLOCKED
    assert attempts(rig) == [[1], [1]] and [d["reason"] for d in stopped(rig)] == ["unchanged", "unchanged"]
    assert exit_codes(rig) == [2, 2]               # one test per attempt: an unchanged port is not run again


def test_the_rounds_ended_decision_names_the_count(rig):
    rig.args.port_repair_rounds = 1
    start(rig, per_round=[{"port/port_model.py": broken(1)}])
    rig.run()
    d = stopped(rig)[0]
    assert d["reason"] == "spent" and d["rounds"] == 1 and d["of"] == 1


def test_a_repair_step_that_writes_the_same_fix_again_ends_when_the_port_has_changed(rig):
    start(rig, unattended=False)
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "repair"):
            return call("write_file", path="port/port_model.py", content=HONEST)         # the same call every turn, and never a final reply
        return base(request)
    rig.script = script
    assert rig.run() == EXIT_ABORTED
    assert any(d.get("decision") == "step finished: repeated write" for d in rig.decisions())
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


# ---- what earns no round ---------------------------------------------------------------------------------------------------------

def test_a_failure_of_the_run_around_the_port_earns_no_round(rig):
    bad = {**STAGE_1, "evidence/reference/generated-ids.json": {"generated_ids": "x", "generated_text": "x"}}      # the template says setup
    start(rig, port=HONEST)
    rig.script = (lambda base: (lambda request: bringup(request, overrides={(1, "run"): bad}) if "tools" in request and where(request) == (1, "run")
                                else base(request)))(rig.script)
    assert rig.run() == EXIT_BLOCKED
    assert rounds(rig) == [] and exit_codes(rig)[0] == 2           # exit 2, kind setup: the stage fails as before


# ---- the repair step may not touch the supervisor's records -----------------------------------------------------------------------

FORGED_MARKER = {"command": full_port.COMMAND, "returncode": 0, "timed_out": False, "chips": ["c"], "output": {}}


def test_a_repair_step_that_writes_the_test_record_fails_the_attempt_and_runs_no_second_test(rig):
    start(rig, per_round=[{"port/port_model.py": HONEST, "test-result.json": FORGED_MARKER}])
    assert rig.run() == EXIT_BLOCKED
    assert exit_codes(rig)[:1] == [2] and exit_codes(rig).count(0) == 0
    block = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert "which only the supervisor writes" in block["reason"] and "test-result.json" in block["reason"]


def test_a_repair_step_that_rewrites_the_output_file_fails_the_attempt(rig):
    start(rig, per_round=[{"port/port_model.py": HONEST, "evidence/hw-test-output.txt": "all good\n"}])
    assert rig.run() == EXIT_BLOCKED
    assert exit_codes(rig).count(0) == 0
    assert "evidence/hw-test-output.txt" in [d for d in rig.decisions() if d["decision"] == "blocked"][0]["reason"]


def test_a_template_the_repair_step_edited_is_put_back_before_the_second_test(rig):
    edited = full_port.TEMPLATE.read_text().replace("MAX_POSITIONS = 64", "MAX_POSITIONS = 8")
    start(rig, per_round=[{"port/port_model.py": HONEST, "port_check.py": edited,
                           "hw_test.json": {"command": "python3 stages/2/port/check.py", "deadline_s": 60}}], unattended=False)
    rig.run()
    assert (stage_2(rig) / full_port.TEMPLATE_NAME).read_text() == full_port.TEMPLATE.read_text()
    assert json.loads((stage_2(rig) / "test-result.json").read_text())["command"] == full_port.COMMAND
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_a_result_the_repair_step_wrote_is_set_aside_and_the_record_comes_from_the_test(rig):
    forged = {"port/port_model.py": HONEST, "evidence/port-check.json": {"schema": "orchard-port-check/1"},
              "result.json": {"runs": True}}
    start(rig, per_round=[forged], unattended=False)
    rig.run()
    assert any(d.get("decision") == "results written before the test were set aside" and "evidence/port-check.json" in d["files"] for d in rig.decisions())
    rec = json.loads((stage_2(rig) / "evidence" / "port-check.json").read_text())
    assert rec["chip_ids"] and rec["template_sha256"] == full_port.template_sha256()


# ---- a kill in the middle of the rounds -----------------------------------------------------------------------------------------

def round_decision(k):
    return lambda e: e["event"] == "decision" and e["data"].get("decision") == port_repair.DECISION and e["data"].get("round") == k


def test_a_resume_keeps_the_count_of_the_attempt(rig):
    start(rig, per_round=[{"port/port_model.py": broken(k)} for k in range(1, 6)])
    with pytest.raises(Crash):
        rig.run(crash_if=round_decision(2))              # killed right after round 2 was written to the ledger
    assert rig.run(pid=200) == EXIT_BLOCKED
    assert attempts(rig) == [[1, 2, 3], [1, 2, 3]]       # the resume went on with round 3 and did not start again at 1


def test_a_resume_of_an_escalated_attempt_goes_on_in_its_stage_directory(rig):
    rig.args.port_repair_rounds = 2
    start(rig, per_round=[{"port/port_model.py": broken(k)} for k in range(1, 6)])
    firsts = []

    def second_attempts_first_round(e):               # the first attempt's round 1 passes; the escalated attempt's round 1 is the one that is killed
        if round_decision(1)(e):
            firsts.append(e)
        return round_decision(1)(e) and len(firsts) == 2
    with pytest.raises(Crash):
        rig.run(crash_if=second_attempts_first_round)
    assert rig.run(pid=200) == EXIT_BLOCKED
    assert attempts(rig) == [[1, 2], [1, 2]]
    assert not any(d.get("decision") == "not resuming from a failed hardware test" for d in rig.decisions())


def test_a_resume_after_the_rounds_ended_as_unchanged_does_not_start_another(rig):
    start(rig, per_round=[{}])
    ended = lambda e: e["event"] == "decision" and e["data"].get("decision") == port_repair.STOPPED and e["data"].get("reason") == "unchanged"
    firsts = []
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: ended(e) and not firsts.append(e) and len(firsts) == 1)
    assert rig.run(pid=200) == EXIT_BLOCKED
    assert attempts(rig) == [[1], [1]]


# ---- the option --------------------------------------------------------------------------------------------------------------------

def test_parse_has_the_option_default_none_and_checks_the_range(tmp_path):
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert parse(a).port_repair_rounds is None and parse(a + ["--port-repair-rounds", "2"]).port_repair_rounds == 2
    with pytest.raises(SystemExit):
        parse(a + ["--port-repair-rounds", "11"])
    with pytest.raises(SystemExit):
        parse(a + ["--port-repair-rounds", "-1"])


def test_a_resume_with_another_count_is_refused_and_one_without_the_option_keeps_the_recorded_value(rig):
    rig.args.port_repair_rounds = 2
    start(rig, unattended=False)
    rig.run()
    assert next(e for e in rig.entries() if e["event"] == "run_start")["data"]["port_repair_rounds"] == 2
    rig.args.port_repair_rounds = 3
    with pytest.raises(ValueError, match="port repair rounds"):
        rig.run(pid=200)
    rig.args.port_repair_rounds = None
    rig.run(pid=200)                                       # no option: the recorded 2 stands, and nothing is refused


# ---- review of PR 425 ------------------------------------------------------------------------------------------------------------

def test_a_secret_whose_at_sign_is_past_the_cut_does_not_reach_the_ledger_or_the_prompt(rig):
    secret = "S3CRETpassword" + "p" * 400
    seen = start(rig, port=BROKEN.replace("model.layers.2.ple.conv1d.weight", f"https://bob:{secret}@host/model"), unattended=False)
    rig.run()
    assert "S3CRET" not in rounds(rig)[0]["failure"] and "S3CRET" not in seen[0]


def test_a_crash_between_the_repair_edit_and_the_rerun_tests_the_fix_on_resume(rig):
    # Round 1 writes the fixed port; the supervisor is killed when the rerun starts. The resumed attempt's repair step changes nothing more.
    start(rig, per_round=[{"port/port_model.py": HONEST}, {}], unattended=False)
    reruns = []

    def at_the_rerun(e):
        if e["event"] == "decision" and e["data"].get("decision") == "hardware phase":
            reruns.append(e)
        return len(reruns) == 2
    with pytest.raises(Crash):
        rig.run(crash_if=at_the_rerun)
    assert rig.run(pid=200) == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    assert exit_codes(rig) == [2, 0] and stopped(rig) == []          # the fix was tested; the port was not called unchanged
