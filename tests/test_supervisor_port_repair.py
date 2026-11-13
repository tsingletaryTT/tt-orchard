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
from run_fakes import argv, bringup, call, turn, where
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, STAGE_1, make, never_wait, stage_2, started, with_snapshot

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")

# The port the tiny Qwen4 rehearsal's coder wrote: it reads a weight the checkpoint does not have, so `load` fails.
BROKEN = HONEST.replace('return {"dir": model_dir, "device": mesh_device}', "raise KeyError('model.layers.2.ple.conv1d.weight')")


def broken(k):
    """A port that still fails at load but is a different file, of a different size, from every other round's."""
    return BROKEN + "# edit " + "x" * k + "\n"          # a different length each time: write_file's reply names the size, and three equal replies in a row trip the watchdog


@pytest.fixture
def board_only(monkeypatch):
    """The CPU run before the board (issue 697) is switched off. BROKEN fails at load with a KeyError, which the CPU run finds before any board, so
    the tests that place a crash at the Nth hardware phase would count one phase fewer. They are about the board check and its resume, so they
    run without it; tests/test_cpu_run_before_board.py covers the CPU run."""
    from orchard.supervisor import Supervisor
    monkeypatch.setattr(Supervisor, "_cpu_run_precheck", lambda self, spec, stage_dir, test: False)


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


def same_length(k):
    """A port that still fails at load, a different file each round but of one size: write_file's reply (`wrote N characters`) is identical."""
    return BROKEN + f"# edit {k:03d}\n"


def test_independent_repair_rounds_do_not_share_loop_detection_history(rig):
    # Each repair step is its own conversation. Six distinct edits of one size give six identical `wrote N characters` replies; one watchdog
    # for the whole stage nudged at the third and escalated at the sixth, so --port-repair-rounds 8 never reached round 7 (review of PR 425).
    rig.args.port_repair_rounds = 8
    start(rig, per_round=[{"port/port_model.py": same_length(k)} for k in range(1, 10)])
    rig.run()
    assert attempts(rig)[0] == list(range(1, 9)), attempts(rig)
    assert all(e["data"].get("by") == "stage machine" for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 2)


def test_a_long_check_between_repair_steps_does_not_read_as_no_progress(rig):
    # The check that follows a repair step can take longer than the watchdog's no-progress window. Its evidence goes to the ledger, not to the
    # watchdog, so one watchdog for the stage read the next step's first reply as an hour of responses with no evidence and nudged or escalated
    # a stage that had just run a check (review of PR 425, round 3). Each repair step has its own detector clock.
    rig.args.port_repair_rounds = 3
    start(rig, per_round=[{"port/port_model.py": broken(k)} for k in range(1, 6)])
    base = rig.script

    def slow(request):
        if "tools" in request and where(request) == (2, "repair") and turn(request) == 0:
            rig.clock.now += 3700.0                    # more than NO_EVIDENCE_S (3600 s) since the last step; three of them stay inside stage 2's 14,400 s
        return base(request)
    rig.script = slow
    rig.run()
    assert attempts(rig)[0] == [1, 2, 3], attempts(rig)
    entries = rig.entries()
    first_end = next(i for i, e in enumerate(entries) if e["event"] == "stage_end" and e["stage"] == 2)
    assert not [e for e in entries[:first_end] if e["data"].get("watchdog")], "the watchdog acted on an attempt that was running its checks"


def test_a_long_check_before_the_finish_step_does_not_read_as_no_progress(rig):
    # The same reasoning for the step after the last rerun: the repair step's passing rerun can take longer than the no-progress window, and the
    # finish step must not inherit the repair step's detector clock (review of PR 425, round 3).
    rig.args.port_repair_rounds = 3
    start(rig)                                                  # round 1 writes the honest port, so the rerun passes and the finish step follows
    base = rig.script

    def slow(request):
        if "tools" in request and where(request) == (2, "finish") and turn(request) == 0:
            rig.clock.now += 3700.0
        return base(request)
    rig.script = slow
    rig.run()
    entries = rig.entries()
    first_end = next(i for i, e in enumerate(entries) if e["event"] == "stage_end" and e["stage"] == 2)
    assert entries[first_end]["data"]["result"] == "pass"
    assert not [e for e in entries[:first_end] if e["data"].get("watchdog")], "the watchdog acted on a finish step that followed a long check"


def watchdogs_made(monkeypatch):
    """Count the watchdogs the supervisor makes for stage 2: the list grows by one entry for each."""
    from orchard.supervisor import Supervisor
    made, real = [], Supervisor._watchdog

    def counting(self, spec):
        if spec.number == 2:
            made.append(spec.number)
        return real(self, spec)
    monkeypatch.setattr(Supervisor, "_watchdog", counting)
    return made


def test_without_the_option_an_attempt_has_one_watchdog_as_before(rig, monkeypatch):
    made = watchdogs_made(monkeypatch)
    start(rig, port=HONEST)
    rig.args.port_repair_rounds = 0                             # the option off: the stage is what it was before issue 420
    rig.run()
    assert len(made) == len(started(rig, 2)) and made, (made, len(started(rig, 2)))


def test_with_the_option_each_repair_step_and_the_finish_step_after_them_have_their_own_watchdog(rig, monkeypatch):
    made = watchdogs_made(monkeypatch)
    rig.args.port_repair_rounds = 3
    start(rig, per_round=[{"port/port_model.py": broken(1)}, {"port/port_model.py": HONEST}])
    rig.run()
    first_attempt = len(rounds(rig)[:2])                         # two repair steps in the first attempt, then its finish step
    assert len(made) >= len(started(rig, 2)) + first_attempt + 1, (made, len(started(rig, 2)))


def never_finishes(rig):
    """A repair step that reads a new file every turn and never gives a final answer, so the step ends without status `done`."""
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "repair"):
            return call("read_file", path=f"stages/2/notes-{turn(request)}.txt")
        return base(request)
    rig.script = script


def test_a_kill_between_the_escalate_entry_and_the_stage_end_gives_the_escalated_attempt_a_fresh_start(rig):
    # A repair step that ends early with rounds left ends the attempt. The stage machine writes `escalate` and then `stage_end`; a kill between
    # the two resumes the stage escalated while the latest `stage_start` is still the original attempt's. The failed port is still repairable
    # (rounds left), but the escalated attempt must start fresh: its own prepare, its own rounds. Review of PR 425, round 3.
    rig.args.port_repair_rounds = 3
    start(rig, per_round=[{"port/port_model.py": broken(1)}])
    never_finishes(rig)
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "escalate" and e["stage"] == 2)
    assert started(rig, 2)[-1]["data"].get("escalated") is not True              # the latest start is the original attempt's
    rig.run(pid=200)
    entries = rig.entries()
    cut = next(i for i, e in enumerate(entries) if e["event"] == "escalate" and e["stage"] == 2)
    first = next(e for e in entries[cut:] if e["event"] == "stage_start" and e["stage"] == 2)["data"]
    assert first.get("escalated") is True and first.get("resumed") is False, first             # the resume starts the escalated attempt afresh
    assert any(e["event"] == "decision" and e["data"].get("decision") == "not resuming from a failed hardware test" for e in entries[cut:])
    assert attempts(rig)[0] == [1]                  # the original attempt's count stopped at its one round; the escalated attempt did not continue it


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

def test_a_secret_the_worker_cut_at_2000_characters_does_not_reach_the_ledger_or_the_prompt(rig):
    from test_port_repair import straddling
    secret = "S3CRETpassword" + "p" * 400
    seen = start(rig, port=BROKEN.replace("model.layers.2.ple.conv1d.weight", straddling(secret)), unattended=False)
    rig.run()
    assert rounds(rig) and "S3CRET" not in rounds(rig)[0]["failure"] and "S3CRET" not in seen[0]
    assert "S3CRET" not in json.dumps(rig.entries())


def test_a_secret_whose_at_sign_is_past_the_cut_does_not_reach_the_ledger_or_the_prompt(rig):
    secret = "S3CRETpassword" + "p" * 400
    seen = start(rig, port=BROKEN.replace("model.layers.2.ple.conv1d.weight", f"https://bob:{secret}@host/model"), unattended=False)
    rig.run()
    assert "S3CRET" not in rounds(rig)[0]["failure"] and "S3CRET" not in seen[0]


def hardware_test_evidence(e):
    return e["event"] == "evidence" and e["stage"] == 2 and e["data"].get("what") == "hardware test"


def test_a_kill_after_the_rerun_replaced_the_record_and_before_its_evidence_entry_keeps_the_repaired_port(rig, monkeypatch):
    # The rerun's test writes test-result.json and then the supervisor writes the evidence entry that vouches for it. A kill between the two
    # leaves a record no entry vouches for. The resume must not read that as a forged marker and move the attempt aside (the repaired port and
    # the attempt's count would be lost): it keeps the port, does not trust the unvouched record, and runs the owed check again (review of PR 432).
    import test_supervisor as ts
    from fakes import Crash

    class CrashesBefore(ts.Ledger):
        """A ledger whose process dies just BEFORE the first append for which `crash_if(entry)` is true."""
        def __init__(self, path, crash_if=None):
            super().__init__(path)
            self.crash_if = crash_if

        def append(self, event, stage, **data):
            if self.crash_if and self.crash_if({"event": event, "stage": stage, "data": data}):
                raise Crash(f"killed before the ledger event {event}")
            return super().append(event, stage, **data)
    monkeypatch.setattr(ts, "CrashingLedger", CrashesBefore)
    seen = []

    def before_the_reruns_evidence(e):
        if hardware_test_evidence(e):
            seen.append(e)
        return hardware_test_evidence(e) and len(seen) == 2
    start(rig, per_round=[{"port/port_model.py": HONEST}], unattended=False)
    with pytest.raises(Crash):
        rig.run(crash_if=before_the_reruns_evidence)
    marker_after_the_kill = json.loads((stage_2(rig) / "test-result.json").read_text())
    assert marker_after_the_kill["returncode"] == 0                       # the rerun's record is on disk, and no entry vouches for it
    assert rig.run(pid=200) == EXIT_ABORTED
    assert not any(d.get("decision") == "marker not written by the supervisor" for d in rig.decisions()), "the unvouched record was refused as a forgery"
    assert attempts(rig) == [[1]]                                         # the attempt was not moved aside and did not start again
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    assert exit_codes(rig) == [2, 0]                                      # the interrupted rerun was never recorded; the check ran again and was


def test_a_kill_after_a_rerun_with_identical_record_bytes_was_recorded_does_not_repeat_the_test(rig):
    # An edit that leaves the same failure, with durations that round alike, gives a rerun record byte for byte equal to the sealed one. A kill
    # after the rerun's evidence entry and before `port repair tested` must still read the rerun as done: the evidence entry says so, and the
    # bytes cannot (review of PR 432). Repeating it would park the coder and reboot it again.
    rig.args.port_repair_rounds = 1
    start(rig, per_round=[{"port/port_model.py": broken(1)}], unattended=False)
    seen = []

    def at_the_reruns_evidence(e):
        if hardware_test_evidence(e):
            seen.append(e)
        return hardware_test_evidence(e) and len(seen) == 2
    with pytest.raises(Crash):
        rig.run(crash_if=at_the_reruns_evidence)
    assert seen[0]["data"]["sha256"] == seen[1]["data"]["sha256"], "the rerun's record is not byte for byte the sealed one, so this test does not show the case"
    rig.run(pid=200)
    first_end = next(i for i, e in enumerate(rig.entries()) if e["event"] == "stage_end" and e["stage"] == 2)
    tests_in_first_attempt = [e for e in rig.entries()[:first_end] if hardware_test_evidence(e)]
    assert len(tests_in_first_attempt) == 2, len(tests_in_first_attempt)      # the initial test and its one rerun; the rerun was not repeated


def test_a_record_changed_after_the_reruns_evidence_entry_is_not_trusted(rig):
    # The evidence entry vouches for one set of bytes (path and sha256). A record that differs by the time of the resume is not that record, so the
    # owed check runs again instead of a changed file standing in for its result.
    start(rig, per_round=[{"port/port_model.py": HONEST}], unattended=False)
    seen = []

    def at_the_reruns_evidence(e):
        if hardware_test_evidence(e):
            seen.append(e)
        return hardware_test_evidence(e) and len(seen) == 2
    with pytest.raises(Crash):
        rig.run(crash_if=at_the_reruns_evidence)
    marker = stage_2(rig) / "test-result.json"
    record = json.loads(marker.read_text())
    marker.write_text(json.dumps({**record, "seconds": record["seconds"] + 123.0}))
    rig.run(pid=200)
    assert exit_codes(rig) == [2, 0, 0], exit_codes(rig)                      # the initial test, the rerun that was recorded, and the check run again


def test_a_record_changed_with_no_rerun_in_flight_is_still_refused_on_resume(rig):
    # The exception for a rerun in flight is for that window only. With none in the ledger, a test record no entry vouches for is what it always
    # was: possibly written by an agent, and no resume point.
    start(rig, per_round=[{"port/port_model.py": HONEST}], unattended=False)
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "decision" and e["data"].get("decision") == port_repair.TESTED)      # after the first test, before any rerun
    marker = stage_2(rig) / "test-result.json"
    record = json.loads(marker.read_text())
    marker.write_text(json.dumps({**record, "returncode": 0}))
    rig.run(pid=200)
    assert any(d.get("decision") == "marker not written by the supervisor" for d in rig.decisions())
    assert not any(e["data"].get("unvouched") for e in rig.entries() if e["event"] == "decision")


@pytest.mark.usefixtures("board_only")
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


# ---- Copilot's second review of PR 425: a kill while a rerun is owed ------------------------------------------------------------------------

def repairs_by_attempt(rig, table):
    """Like `repairs`, but a repair step writes `table[(attempt, round)]` (nothing when absent). The attempt is the count of fresh stage 2 starts in the
    ledger, so an escalated attempt can write another file from the first attempt's round with the same number."""
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "repair"):
            found = re.search(r"round (\d+) of (\d+)", request["messages"][1]["content"])
            k = int(found.group(1)) if found else 1
            entries = []
            for line in (rig.run_dir / "ledger.jsonl").read_text().splitlines():      # the supervisor holds the ledger's lock: read the file, do not open it
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    pass                                                              # a line the supervisor is writing right now
            attempt = sum(1 for e in entries if e.get("stage") == 2 and e.get("event") == "stage_start" and e.get("data", {}).get("resumed") is False)
            return bringup(request, overrides={(2, "repair"): table.get((attempt, k), {})})
        return base(request)
    rig.script = script


def crash_at_rerun(rig, *, escalated, before_rerun=False):
    """Run until the supervisor dies when the repaired port's rerun starts. The first attempt's rerun is the second hardware phase; with
    `escalated` the first attempt spends its only round on a port that still fails, and the kill is on the escalated attempt's only round,
    its fourth hardware phase. With `before_rerun` the kill is earlier: when the round's failed output has been copied to the evidence
    directory, after the repair step's edit and before the rerun is recorded. Returns the count of matching ledger entries at the kill."""
    seen = []

    def at(e):
        if before_rerun:
            if e["event"] == "evidence" and e["data"].get("what") == "port repair round output":
                seen.append(e)
            return len(seen) == (2 if escalated else 1)
        if e["event"] == "decision" and e["data"].get("decision") == "hardware phase":
            seen.append(e)
        return len(seen) == (4 if escalated else 2)
    if escalated:
        rig.args.port_repair_rounds = 1
        make(rig, over={(2, "prepare"): {**PREPARE, "port/port_model.py": BROKEN}})
        repairs_by_attempt(rig, {(1, 1): {"port/port_model.py": broken(1)}, (2, 1): {"port/port_model.py": HONEST}})
        rig.args.unattended = False
        rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    else:
        start(rig, per_round=[{"port/port_model.py": HONEST}, {}], unattended=False)
    with pytest.raises(Crash):
        rig.run(crash_if=at)
    return len(seen)


def tests_started(rig):
    return [d for d in rig.decisions() if d.get("decision") == "hardware test started"]


@pytest.mark.parametrize("escalated", [False, True], ids=["first attempt", "escalated attempt"])
@pytest.mark.usefixtures("board_only")
def test_a_kill_when_the_last_repair_round_reruns_tests_the_fix_on_resume(rig, escalated):
    # The escalated variant: the attempt's only round is already counted when the kill comes, so `used < rounds` is false (Copilot, PR 425).
    crash_at_rerun(rig, escalated=escalated)
    assert rig.run(pid=200) == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == (["escalate", "pass"] if escalated else ["pass"])       # the fix was tested in the attempt that wrote it
    assert exit_codes(rig)[-1] == 0
    assert not any(d.get("decision") == "not resuming from a failed hardware test" for d in rig.decisions())


@pytest.mark.parametrize("escalated", [False, True], ids=["first attempt", "escalated attempt"])
@pytest.mark.usefixtures("board_only")
def test_a_kill_after_the_rerun_replaced_the_output_file_tests_the_fix_on_resume(rig, escalated):
    # The test's output file is opened fresh when the rerun starts, so from then until the record is written the old record's output hash no
    # longer matches. The resume used to read no failure there and went on to the finish step with the old failure (Copilot, PR 425).
    crash_at_rerun(rig, escalated=escalated)
    (stage_2(rig) / port_repair.OUTPUT).write_text("the first lines of the rerun, cut short by the kill\n")
    assert rig.run(pid=200) == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == (["escalate", "pass"] if escalated else ["pass"])       # no wasted attempt: the resume tested the fix
    assert exit_codes(rig)[-1] == 0
    assert [d["reason"] for d in stopped(rig)] == (["spent"] if escalated else [])      # only the first attempt's only round ended as spent
    kept = [e for e in rig.entries() if e["event"] == "evidence" and e["data"].get("what") == "port repair round output"]
    assert [e["data"]["round"] for e in kept] == [1] * len(kept) and len({e["data"]["sha256"] for e in kept}) == 1
    saved = (stage_2(rig) / "evidence" / "port-repair-round-1-output.txt").read_text()
    assert "first lines of the rerun" not in saved and "PORT_CHECK_FAILURE" in saved      # what the repair step was shown, not the cut-short rerun


@pytest.mark.usefixtures("board_only")
def test_a_kill_after_the_rerun_wrote_its_record_does_not_run_the_test_again(rig):
    start(rig, per_round=[{"port/port_model.py": HONEST}, {}], unattended=False)
    released = []

    def after_the_second_record(e):
        if e["event"] == "evidence" and e["stage"] == 2 and e["data"].get("what") == "hardware test":
            released.append(e)
        return len(released) == 2
    with pytest.raises(Crash):
        rig.run(crash_if=after_the_second_record)
    assert rig.run(pid=200) == EXIT_ABORTED
    assert len(tests_started(rig)) == 2 and exit_codes(rig)[-1] == 0
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


@pytest.mark.parametrize("escalated", [False, True], ids=["first attempt", "escalated attempt"])
def test_a_kill_after_the_last_round_edit_and_before_its_rerun_is_recorded_tests_the_fix_on_resume(rig, escalated):
    # No rerun entry exists yet, the old record is whole, and (escalated) the attempt's rounds are all counted: only the port's difference from the
    # port the latest test ran against says that a test is owed.
    crash_at_rerun(rig, escalated=escalated, before_rerun=True)
    assert len([d for d in rig.decisions() if d.get("decision") == port_repair.RERUN]) == (1 if escalated else 0)      # the first attempt's only
    assert rig.run(pid=200) == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == (["escalate", "pass"] if escalated else ["pass"])
    assert exit_codes(rig)[-1] == 0
    assert not any(d.get("decision") == "not resuming from a failed hardware test" for d in rig.decisions())


def test_repair_rounds_that_cannot_settle_end_in_a_failure_and_do_not_loop(rig, monkeypatch):
    # A ledger that follows the rules cannot do this; the bound is for a fault in the in-flight logic itself (a mutant of it spun forever,
    # appending a `tested` entry on every pass).
    start(rig, per_round=[{"port/port_model.py": HONEST}, {}], unattended=False)
    monkeypatch.setattr(port_repair, "rerun_inflight", lambda entries, n: {"round": 1, "record": {"path": "nothing", "sha256": "0" * 64}})
    rig.run()
    assert [d["reason"] for d in stopped(rig)].count("did not settle") >= 1
    assert "pass" not in [d["result"] for d in rig.ends(2)]               # a loop that could not settle is not a pass
    assert "did not settle within" in (rig.run_dir / "ledger.jsonl").read_text()        # the attempt's failure says why, in the ledger
    tested = [d for d in rig.decisions() if d.get("decision") == port_repair.TESTED]
    assert len(tested) <= len(started(rig, 2)) * (2 * port_repair.DEFAULT_ROUNDS + 3)         # each attempt has its passes and its first entry, and no more
