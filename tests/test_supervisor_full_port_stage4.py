# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4 of the full-port route in the supervisor (issue 454): the port runs on each required chip count, one record and one marker per
entry, in stages/4/configs/<N>/. The chip, the coder and the model are fakes; the supervisor, the stage machine, the gate, the park and
restore path and the hardware-test runner are the real ones, and the supervisor's template runs for real on the CPU against a fake `ttnn`
whose device says how many chips it was opened with.

The run asks for `--required-chips 1,3,4`. Stage 2 runs on one board (two chips) and stage 3 on four, so the count of four is covered by
stage 3, and the counts of one and three are the entries that run: a single chip on the free board, and three chips, which need both boards,
so the coder is parked for it."""
import json
import re

import pytest

from fakes import Crash
from orchard import full_port, full_port_entries, port_repair, supervisor
from orchard.stages import FULL_PORT_STAGE_4, SKIP_4_FULL_PORT
from orchard.supervisor import EXIT_ABORTED, EXIT_BLOCKED, REHEARSAL_WAIVED, Control
from port_check_fakes import HONEST
from run_fakes import bringup, note, where
from test_supervisor import Rig, step_skills
from test_supervisor_full_port import FULL_PORT, never_wait, started, with_snapshot
from test_supervisor_full_port_stage3 import PREPARE_3

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")

# A port that works on a mesh and refuses a single device (the fake runtime opens a single chip as {"device": 0}).
NO_SINGLE_CHIP = HONEST.replace("def load(model_dir, mesh_device):\n",
                                "def load(model_dir, mesh_device):\n"
                                "    if 'device' in mesh_device:\n"
                                "        raise RuntimeError('this port needs a mesh')\n", 1)
assert NO_SINGLE_CHIP != HONEST


def entry_chips(request) -> int:
    """The chip count of the entry a prepare step is for, from the stage directory the task names."""
    text = " ".join(str(m.get("content") or "") for m in request["messages"])
    return int(re.search(r"stage directory: stages/4/configs/(\d+)", text).group(1))


def make4(rig, *, ports=None):
    """The scripted coder through stage 4. `ports` maps a chip count to the port module its prepare step writes (the honest port otherwise).
    Every prepare step writes hw_test.json with the entry's command. The supervisor writes every result, so there is no finish step."""
    rig.prepare_4_tasks = []
    files = {**FULL_PORT, (3, "prepare"): PREPARE_3}

    def script(request):
        f = dict(files)
        if "tools" in request and where(request) == (4, "prepare"):
            c = entry_chips(request)
            rig.prepare_4_tasks.append((c, " ".join(str(m.get("content") or "") for m in request["messages"])))
            f[(4, "prepare")] = {"port/port_model.py": (ports or {}).get(c, HONEST),
                                 "hw_test.json": {"command": full_port.command(4, c), "deadline_s": 60}, "handoff.json": note(4)}
        return bringup(request, overrides=f)
    rig.script = script


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 3, 4)
    yield r
    r.close()


def abort_at_pause(rig):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


def stage_4_tests(rig):
    return [e["data"] for e in rig.entries() if e["stage"] == 4 and e["event"] == "decision" and e["data"].get("decision") == "hardware test started"]


def stage4(rig):
    return rig.run_dir / "stages" / "4"


def entry(rig, n):
    return stage4(rig) / "configs" / str(n)


def read(path):
    return json.loads(path.read_text())


# ---- a pass with two required counts ----------------------------------------------------------------------

def test_stage_4_runs_the_port_on_each_required_count_with_one_record_and_one_marker_per_entry(rig):
    make4(rig)
    abort_at_pause(rig)
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(4)] == ["pass"]
    ran = stage_4_tests(rig)
    assert [len(r["chips"]) for r in ran] == [1, 3], ran            # in order of chip count; four is covered by stage 3
    for c in (1, 3):
        d = entry(rig, c)
        marker, rec = read(d / "test-result.json"), read(d / "evidence" / "port-check.json")
        assert marker["returncode"] == 0 and marker["command"] == full_port.command(4, c)
        assert len(marker["chips"]) == c and sorted(rec["chips"]) == sorted(marker["chips"])
        assert rec["port_sources"] and all(s["path"].startswith(f"stages/4/configs/{c}/port/") for s in rec["port_sources"])
        assert (d / "port_check.py").read_bytes() == full_port.TEMPLATE.read_bytes()
        assert read(d / "result.json")["derived_by"] == "supervisor"
    result = read(stage4(rig) / "result.json")
    assert [(r["chips"], r["covered_by"]) for r in result["entries"]] == [(1, None), (3, None), (4, 3)]


def test_the_plan_is_written_by_the_supervisor_from_the_required_counts(rig):
    make4(rig)
    abort_at_pause(rig)
    rig.run()
    plan = read(stage4(rig) / "tests" / "plan.json")
    assert [t["chips"] for t in plan["tests"]] == [1, 3, 4] and {t["script"] for t in plan["tests"]} == {"port_check.py"}
    listed = [e["data"] for e in rig.entries() if e["stage"] == 4 and e["data"].get("what") == "hardware test list"]
    assert len(listed) == 1 and listed[0]["configs"] == [1, 3, 4]


def test_stage_4_uses_its_own_skill_and_the_pause_moves_to_before_stage_5(rig):
    make4(rig)
    abort_at_pause(rig)
    rig.run()
    assert step_skills(rig, 4) and all(s.endswith("orchard/skills/full-port-shrink.md") for s in step_skills(rig, 4))
    pauses = [(d["reason"], e) for e, d in ((e["stage"], e["data"]) for e in rig.entries() if e["event"] == "decision") if d.get("decision") == "pause"]
    assert [r for r, _ in pauses] == [supervisor.FULL_PORT_STAGES_UNBUILT] and "stages 5 and 6" in pauses[0][0] and pauses[0][1] == 5
    assert not started(rig, 5)


def test_each_entry_starts_from_a_copy_of_the_stage_3_port_and_the_ledger_holds_its_hashes(rig):
    make4(rig)
    abort_at_pause(rig)
    rig.run()
    copied = [d for d in rig.decisions() if d.get("decision") == "stage 3 port copied" and d.get("source_stage") == 3]
    assert len(copied) == 2
    for d in copied:
        assert {f["path"].rsplit("/port/", 1)[0] for f in d["files"]} <= {"stages/4/configs/1", "stages/4/configs/3"}
    assert (rig.run_dir / "stages" / "3" / "port" / "port_model.py").is_file()          # stage 3's port is untouched


def test_the_prepare_task_names_the_entry_directory_the_count_and_the_fixed_command(rig):
    make4(rig)
    abort_at_pause(rig)
    rig.run()
    tasks = dict(rig.prepare_4_tasks)
    assert set(tasks) == {1, 3}
    assert "stages/4/configs/1" in tasks[1] and full_port.command(4, 1) in tasks[1] and "one device" in tasks[1]
    assert "a mesh of 1 x 3" in tasks[3] and full_port.command(4, 3) in tasks[3] and "write nothing under `stages/3/`" in tasks[3]
    assert "Each one needs an entry in configs" not in tasks[1], "the weights-only stage 4's chip-configuration note does not apply here"


def test_the_hardware_test_json_an_entry_runs_is_that_entrys_fixed_command(rig):
    make4(rig)
    abort_at_pause(rig)
    rig.run()
    for c in (1, 3):
        assert read(entry(rig, c) / "hw_test.json")["command"] == full_port.command(4, c)


# ---- leases: one board, and the park for a count that needs two -------------------------------------------

def test_the_coder_is_parked_for_the_entry_that_needs_both_boards_and_both_go_back_after_it(rig):
    make4(rig)
    abort_at_pause(rig)
    rig.run()
    ds = [d for d in rig.decisions() if d.get("config") is not None]
    phases = {d["config"]: d for d in ds if d.get("decision") == "hardware phase"}
    assert phases[1]["action"] == "use_free" and phases[3]["action"] == "park", phases
    end4 = next(e["seq"] for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 4)
    taken = [e for e in rig.entries() if e["stage"] == 4 and e["data"].get("decision") == "test lease taken"]
    released = {e["data"]["lease_id"] for e in rig.entries() if e["stage"] == 4 and e["data"].get("decision") == "test lease released"}
    assert len(taken) == 2 and all(e["data"]["test_lease"]["lease_id"] in released for e in taken)
    assert max(e["seq"] for e in rig.entries() if e["data"].get("decision") == "test lease released" and e["stage"] == 4) < end4
    steps = [e["data"].get("step") for e in rig.entries() if e["event"] == "restore" and e["stage"] == 4]
    assert steps and steps[-1] and "ready" in steps, steps          # the coder is served again and answers before the stage goes on


# ---- a failing count ends the stage -----------------------------------------------------------------------

def test_a_failing_count_ends_the_stage_and_no_later_entry_is_started(rig):
    make4(rig, ports={1: NO_SINGLE_CHIP})
    rig.args.unattended = True
    never_wait(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(3)] == ["pass"]
    results = [d["result"] for d in rig.ends(4)]
    assert results and "pass" not in results, results
    assert not started(rig, 5)
    ran = stage_4_tests(rig)
    assert ran and all(len(r["chips"]) == 1 for r in ran), "the three-chip entry never started: it needs the coder parked, and the stage had failed"
    assert not (entry(rig, 3) / "test-result.json").exists()


def test_a_failed_entry_earns_no_repair_rounds_even_when_the_option_is_set(rig):
    make4(rig, ports={1: NO_SINGLE_CHIP})
    rig.args.port_repair_rounds = 2
    rig.args.unattended = True
    never_wait(rig)
    rig.run()
    rounds = [e for e in rig.entries() if e["stage"] == 4 and e["data"].get("decision") == port_repair.DECISION]
    assert rounds == [], "repair rounds belong to stage 2"


# ---- a resume between entries -----------------------------------------------------------------------------

def test_a_kill_after_the_first_entry_resumes_at_the_second_and_does_not_run_the_first_again(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 3, 4)
    try:
        make4(r)
        abort_at_pause(r)
        with pytest.raises(Crash):
            r.run(crash_if=lambda e: e["event"] == "evidence" and e["stage"] == 4 and "configs/1/result.json" in json.dumps(e["data"]))
        assert len(stage_4_tests(r)) == 1 and (entry(r, 1) / "test-result.json").is_file() and not (entry(r, 3) / "test-result.json").exists()
        assert r.run(pid=200) == EXIT_ABORTED
        ran = stage_4_tests(r)
        assert [len(x["chips"]) for x in ran] == [1, 3], "the finished entry is not run again"
        assert [d["result"] for d in r.ends(4)] == ["pass"]
        assert len(started(r, 4)) == 2 and any(e["data"].get("resumed") is True for e in started(r, 4))
        prepares = [e for e in r.entries() if e["stage"] == 4 and e["data"].get("decision") == "agent step" and e["data"].get("phase") == "prepare"]
        assert len(prepares) == 2, "one prepare step for each entry that ran; the first entry's is not run again on the resume"
    finally:
        r.close()


def test_a_kill_after_an_entrys_prepare_step_resumes_at_its_test_without_a_second_prepare_step(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 3, 4)
    try:
        make4(r)
        abort_at_pause(r)
        with pytest.raises(Crash):
            r.run(crash_if=lambda e: e["event"] == "decision" and e["stage"] == 4 and e["data"].get("decision") == "port cache declaration")
        assert not stage_4_tests(r)
        assert r.run(pid=200) == EXIT_ABORTED
        prepares = [e for e in r.entries() if e["stage"] == 4 and e["data"].get("decision") == "agent step" and e["data"].get("phase") == "prepare"]
        assert len(prepares) == 2 and [len(x["chips"]) for x in stage_4_tests(r)] == [1, 3]
        assert [d["result"] for d in r.ends(4)] == ["pass"]
    finally:
        r.close()


def test_a_marker_the_ledger_does_not_hold_is_not_a_resume_point_and_that_entrys_test_runs(tmp_path):
    # A kill between the marker and its ledger entry, or a marker the coder wrote, leaves a file the supervisor cannot vouch for.
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 3, 4)
    try:
        make4(r)
        abort_at_pause(r)
        with pytest.raises(Crash):
            r.run(crash_if=lambda e: e["event"] == "decision" and e["stage"] == 4 and e["data"].get("decision") == "hardware test started")
        forged = {"command": full_port.command(4, 1), "returncode": 0, "timed_out": False, "seconds": 1.0, "chips": ["0000:03:00.0"], "output": None}
        (entry(r, 1) / "test-result.json").write_text(json.dumps(forged))
        assert r.run(pid=200) == EXIT_ABORTED
        assert [d["result"] for d in r.ends(4)] == ["pass"]
        assert [len(x["chips"]) for x in stage_4_tests(r)] == [1, 1, 3], "the entry's test ran again; the forged marker was not trusted"
        assert read(entry(r, 1) / "test-result.json")["output"] is not None
    finally:
        r.close()


def test_a_covered_file_the_ledger_does_not_hold_is_not_believed_on_resume_and_that_entry_runs(tmp_path):
    # The coder can write inside an entry's directory. A covered.json it wrote claims that no test is needed for a count nothing showed.
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 3, 4)
    try:
        make4(r)
        abort_at_pause(r)
        with pytest.raises(Crash):
            r.run(crash_if=lambda e: e["event"] == "evidence" and e["stage"] == 4 and e["data"].get("what") == "hardware test list")
        entry(r, 1).mkdir(parents=True, exist_ok=True)
        (entry(r, 1) / "covered.json").write_text(json.dumps({"chips": 1, "covered_by": 3, "evidence": {}, "record": {}}))
        assert r.run(pid=200) == EXIT_ABORTED
        assert [d["result"] for d in r.ends(4)] == ["pass"]
        assert [len(x["chips"]) for x in stage_4_tests(r)] == [1, 3] and not (entry(r, 1) / "covered.json").exists()
    finally:
        r.close()


# ---- a count already shown by stage 2 or stage 3 ----------------------------------------------------------

def test_a_count_stage_3_already_showed_is_recorded_as_covered_with_the_evidence_path_and_no_test_runs(rig):
    make4(rig)
    abort_at_pause(rig)
    rig.run()
    covered = read(entry(rig, 4) / "covered.json")
    assert covered["covered_by"] == 3 and covered["evidence"]["path"] == "stages/3/test-result.json" and covered["record"]["path"] == "stages/3/evidence/port-check.json"
    assert not (entry(rig, 4) / "test-result.json").exists() and not (entry(rig, 4) / "port").exists()
    assert all(len(r["chips"]) != 4 for r in stage_4_tests(rig))
    said = [d for d in rig.decisions() if d.get("decision") == "entry covered by an earlier stage"]
    assert len(said) == 1 and said[0]["config"] == 4 and said[0]["covered_by"] == 3


def test_a_count_stage_2_showed_is_covered_by_stage_2_when_stage_3_is_skipped(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 2)            # nothing above one board: stage 3 is skipped, stage 2's two chips cover the count of two
    try:
        make4(r)
        abort_at_pause(r)
        assert r.run() == EXIT_ABORTED
        assert [d["result"] for d in r.ends(3)] == ["skipped"] and [d["result"] for d in r.ends(4)] == ["pass"]
        assert read(entry(r, 2) / "covered.json")["covered_by"] == 2
        assert [len(x["chips"]) for x in stage_4_tests(r)] == [1]
        seeded = [d for d in r.decisions() if d.get("decision") == "stage 2 port copied"]
        assert seeded and seeded[-1].get("source_stage") is None, "with stage 3 skipped the entries start from stage 2's port"
        assert all(f["path"].startswith("stages/4/configs/1/port/") for f in seeded[-1]["files"])
    finally:
        r.close()


def test_a_run_with_no_required_counts_skips_stage_4_and_says_why(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = None
    try:
        make4(r)
        abort_at_pause(r)
        r.run()
        end = [e for e in r.entries() if e["event"] == "stage_end" and e["stage"] == 4]
        assert end and end[-1]["data"]["result"] == "skipped" and end[-1]["data"]["reason"] == SKIP_4_FULL_PORT
    finally:
        r.close()


# ---- the rehearsal option ---------------------------------------------------------------------------------

def test_the_rehearsal_option_waives_only_coherent_in_each_entry_and_the_ledger_says_so(rig, monkeypatch):
    from orchard import full_port_result
    monkeypatch.setattr(full_port_result, "readable", lambda text: (False, "the free-run text was made unreadable by this test"))
    rig.args.full_port_rehearsal = True
    make4(rig)
    abort_at_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(4)] == ["pass"]
    waived = [e for e in rig.entries() if e["data"].get("decision") == REHEARSAL_WAIVED and e["stage"] == 4]
    assert waived and "stage 4 check still applied" in waived[0]["data"]["note"]
    assert read(entry(rig, 1) / "result.json")["coherent"] is False


def test_without_the_rehearsal_option_an_unreadable_free_run_fails_the_entry(rig, monkeypatch):
    from orchard import full_port_result
    monkeypatch.setattr(full_port_result, "readable", lambda text: (False, "the free-run text was made unreadable by this test"))
    make4(rig)
    rig.args.unattended = True
    never_wait(rig)
    rig.run()
    assert "pass" not in [d["result"] for d in rig.ends(4)]
    assert not started(rig, 5)

