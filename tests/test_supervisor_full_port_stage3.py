# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 3 of the full-port route in the supervisor (issue 326): the port that passed stage 2 on one board runs again on the chip count the run
is configured for (`--required-chips`, here 2,4: four chips, both boards). The chip, the coder and the model are fakes; the supervisor, the stage
machine, the gate, the park and restore path and the hardware-test runner are the real ones, and the supervisor's template runs for real on the
CPU against a fake `ttnn` whose mesh device says how many chips it was opened with."""
import functools
import json
from types import SimpleNamespace

import pytest

from fakes import Crash
from orchard import full_port, port_repair, supervisor
from orchard.stages import FULL_PORT_STAGE_3, SKIP_3_FULL_PORT
from orchard.supervisor import EXIT_ABORTED, EXIT_BLOCKED, REHEARSAL_WAIVED, Control, Supervisor
from port_check_fakes import HONEST
from run_fakes import note
from test_supervisor import Rig, step_skills
from test_supervisor_full_port import FULL_PORT, PREPARE, started, with_snapshot, never_wait
from run_fakes import bringup, where

pytestmark = pytest.mark.usefixtures("stub_tools")

# A port that works on one board and refuses a mesh of four: stage 2 passes it, stage 3 shows the gap.
NO_FOUR_CHIP_MESH = HONEST.replace("def load(model_dir, mesh_device):\n",
                                   "def load(model_dir, mesh_device):\n"
                                   "    if mesh_device.get('mesh') == (1, 4):\n"
                                   "        raise RuntimeError('this port does not shard across four chips')\n", 1)
assert NO_FOUR_CHIP_MESH != HONEST

PREPARE_3 = {"port/port_model.py": HONEST, "hw_test.json": {"command": full_port.command(3), "deadline_s": 60}, "handoff.json": note(3)}


def from_record(rig, n):
    """The finish step's result.json for stage `n`, copied from the record the template wrote, as the skill tells the coder to do."""
    rec = json.loads((rig.run_dir / "stages" / str(n) / "evidence" / "port-check.json").read_text())
    return {"result.json": {"runs": True, "coherent": True, "load_s": rec["load_s"], "n_tokens": len(rec["chip_ids"]),
                            "top1_agreement": rec["top1_agreement"], "free_run_text": rec["free_run_text"],
                            "evidence": [f"stages/{n}/evidence/port-check.json", f"stages/{n}/evidence/hw-test-output.txt"]}}


def make3(rig, *, stage3_port=None, stage2_port=None, coherent=True):
    """The scripted coder for a full-port run through stage 3. A finish step copies the numbers from the record the check just wrote.
    `coherent=False` has the finish steps report the free-run text as unreadable, as a random-weight rehearsal model's is."""
    files = {**FULL_PORT, (3, "prepare"): {**PREPARE_3, **({"port/port_model.py": stage3_port} if stage3_port else {})}}
    if stage2_port:
        files[(2, "prepare")] = {**PREPARE, "port/port_model.py": stage2_port}

    rig.prepare_3_task = []

    def script(request):
        f = dict(files)
        if "tools" in request and where(request) == (3, "prepare"):
            rig.prepare_3_task.append(" ".join(str(m.get("content") or "") for m in request["messages"]))
        for n in (2, 3):
            record = rig.run_dir / "stages" / str(n) / "evidence" / "port-check.json"
            if record.is_file() and "load_s" in record.read_text():          # not a record the coder wrote during prepare
                f[(n, "finish")] = from_record(rig, n)
                f[(n, "finish")]["result.json"]["coherent"] = coherent
        return bringup(request, overrides=f)
    rig.script = script


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (2, 4)
    yield r
    r.close()


def abort_at_pause(rig):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


def stage_3_tests(rig):
    return [e["data"] for e in rig.entries() if e["stage"] == 3 and e["event"] == "decision" and e["data"].get("decision") == "hardware test started"]


# ---- a pass -----------------------------------------------------------------------------------------

def test_stage_3_runs_the_port_on_all_four_configured_chips_and_passes(rig):
    make3(rig)
    abort_at_pause(rig)
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and [d["result"] for d in rig.ends(3)] == ["pass"]
    ran = stage_3_tests(rig)
    assert len(ran) == 1 and len(ran[0]["chips"]) == 4, ran
    stage = rig.run_dir / "stages" / "3"
    rec = json.loads((stage / "evidence" / "port-check.json").read_text())
    marker = json.loads((stage / "test-result.json").read_text())
    assert marker["returncode"] == 0 and marker["command"] == full_port.command(3)
    assert sorted(rec["chips"]) == sorted(marker["chips"]) and len(rec["chips"]) == 4
    assert rec["port_sources"] and all(s["path"].startswith("stages/3/port/") for s in rec["port_sources"])


def test_stage_3_uses_its_own_skill_and_the_pause_moves_to_stage_4(rig):
    make3(rig)
    abort_at_pause(rig)
    rig.run()
    assert step_skills(rig, 3) and all(s.endswith("orchard/skills/full-port-mesh.md") for s in step_skills(rig, 3))
    pauses = [(d["reason"]) for d in rig.decisions() if d.get("decision") == "pause"]
    assert pauses == [supervisor.FULL_PORT_STAGES_UNBUILT] and "stages 4 to 6" in pauses[0]
    for n in (4, 5, 6):
        assert not started(rig, n), n


def test_the_coder_is_parked_for_the_four_chip_test_and_both_boards_go_back_after_it(rig):
    make3(rig)
    abort_at_pause(rig)
    rig.run()
    ds = rig.decisions()
    park = [d for d in ds if d.get("decision") == "hardware phase" and d.get("action") == "park" and d.get("chips") == 4]
    assert park, "the coder's board was parked: a four-chip test needs both boards"
    taken = [d for d in ds if d.get("decision") == "test lease taken"]
    assert taken, "the further board was leased for the wide test"
    released = {d["lease_id"] for d in ds if d.get("decision") == "test lease released"}
    assert all(d["test_lease"]["lease_id"] in released for d in taken[-1:])
    end3 = next(e["seq"] for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 3)
    back = max(e["seq"] for e in rig.entries() if e["data"].get("decision") == "test lease released")
    assert back < end3, "the further board went back before stage 3 ended"


def test_stage_3_starts_from_a_copy_of_the_stage_2_port_and_the_ledger_holds_its_hashes(rig):
    make3(rig)
    abort_at_pause(rig)
    rig.run()
    copied = [d for d in rig.decisions() if d.get("decision") == "stage 2 port copied"]
    assert len(copied) == 1
    paths = {f["path"] for f in copied[0]["files"]}
    assert "stages/3/port/port_model.py" in paths and all(p.startswith("stages/3/port/") for p in paths)
    assert (rig.run_dir / "stages" / "2" / "port" / "port_model.py").is_file()           # stage 2's port is still there, untouched


def test_the_prepare_task_tells_the_coder_the_chip_count_the_copied_port_and_the_stage_3_paths(rig):
    make3(rig)
    abort_at_pause(rig)
    rig.run()
    task = rig.prepare_3_task[0]
    assert "on 4 chips (a mesh of 1 x 4" in task and "already holds the files of the stage 2 port" in task
    assert full_port.command(3) in task and "write nothing under `stages/2/`" in task


def test_the_hardware_test_json_stage_3_runs_is_the_fixed_stage_3_command(rig):
    make3(rig)
    abort_at_pause(rig)
    rig.run()
    hw = json.loads((rig.run_dir / "stages" / "3" / "hw_test.json").read_text())
    assert hw["command"] == full_port.command(3)


# ---- a failing chip count ---------------------------------------------------------------------------

def test_a_port_that_fails_on_the_configured_chips_does_not_pass_stage_3_and_the_run_never_reaches_stage_4(rig):
    make3(rig, stage3_port=NO_FOUR_CHIP_MESH)          # stage 2's port is the honest one and passes on two chips
    rig.args.unattended = True
    never_wait(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    results3 = [d["result"] for d in rig.ends(3)]
    assert results3 and "pass" not in results3, results3
    assert not started(rig, 4)
    marker = json.loads((rig.run_dir / "stages" / "3" / "test-result.json").read_text()) if (rig.run_dir / "stages" / "3" / "test-result.json").is_file() else None
    ran = stage_3_tests(rig)
    assert ran and all(len(r["chips"]) == 4 for r in ran), ran
    assert marker is None or marker["returncode"] != 0


def test_a_failed_stage_3_test_earns_no_repair_rounds_even_when_the_option_is_set(rig):
    make3(rig, stage3_port=NO_FOUR_CHIP_MESH)
    rig.args.port_repair_rounds = 2
    rig.args.unattended = True
    never_wait(rig)
    rig.run()
    rounds = [e for e in rig.entries() if e["stage"] == 3 and e["event"] == "decision" and e["data"].get("decision") == port_repair.DECISION]
    assert rounds == [], "repair rounds belong to stage 2"


def test_the_rehearsal_option_waives_coherent_in_stage_3_as_in_stage_2_and_the_ledger_says_so(rig):
    rig.args.full_port_rehearsal = True
    make3(rig, coherent=False)
    abort_at_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and [d["result"] for d in rig.ends(3)] == ["pass"]
    waived = [e for e in rig.entries() if e["data"].get("decision") == REHEARSAL_WAIVED]
    assert sorted(e["stage"] for e in waived) == [2, 3]
    assert "stage 3 check still applied" in next(e for e in waived if e["stage"] == 3)["data"]["note"]


def test_without_the_rehearsal_option_an_unreadable_free_run_fails_stage_3(rig):
    make3(rig, coherent=False)
    rig.args.unattended = True
    never_wait(rig)
    rig.run()
    assert not any(d["result"] == "pass" for d in rig.ends(2) + rig.ends(3))


def test_a_configured_count_larger_than_the_box_blocks_stage_3_with_the_numbers(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (2, 8)
    try:
        make3(r)
        r.args.unattended = True
        never_wait(r)
        r.run()
        text = json.dumps([e["data"] for e in r.entries() if e["event"] in ("block", "pause", "notice", "decision")])
        assert "the port's configured chip count needs 8 chips and this box has 4" in text
        assert not [d for d in r.ends(3) if d["result"] == "pass"]
    finally:
        r.close()


def test_an_unattended_run_that_passed_stage_3_ends_blocked_at_the_stage_4_pause_and_gives_the_hardware_back(rig):
    rig.args.unattended = True
    make3(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert [d["result"] for d in rig.ends(3)] == ["pass"]
    block = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert len(block) == 1 and block[0]["code"] == "full-port-stages-unbuilt"
    assert "stage 3" in (rig.run_dir / "BLOCKED.md").read_text()
    assert not rig.m.coder_running and rig.m.leases == {}
    assert not started(rig, 4)


# ---- the skip ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("counts", [None, (2,), (1, 2), (1,)])
def test_stage_3_is_skipped_with_its_reason_when_no_count_above_one_board_is_configured(tmp_path, counts):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = counts
    try:
        make3(r)
        abort_at_pause(r)
        r.run()
        ends = [(e["data"]["result"], e["data"].get("reason")) for e in r.entries() if e["event"] == "stage_end" and e["stage"] == 3]
        assert ends == [("skipped", SKIP_3_FULL_PORT)]
        assert not stage_3_tests(r)
    finally:
        r.close()


@pytest.mark.parametrize("counts,chips,boards", [((2, 4), 4, 2), ((2, 3), 4, 2), ((3,), 4, 2), ((1, 2, 4), 4, 2), ((2, 6), 6, 3), ((1, 2), None, 1)])
def test_the_configured_count_is_whole_boards_and_the_phase_holds_that_many(tmp_path, counts, chips, boards):
    """The lease is whole boards, so a count of 3 is tested on, and expected as, 4: the gate compares the chips the test ran on with this number."""
    stub = SimpleNamespace(required_chips=counts)
    stub._stage_3_chips = lambda: Supervisor._stage_3_chips(stub)
    assert stub._stage_3_chips() == chips
    assert Supervisor._phase_boards(stub, FULL_PORT_STAGE_3, tmp_path) == boards


def test_a_configured_count_of_three_is_tested_on_the_four_chips_the_lease_grants_and_the_gate_expects_four(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (2, 3)
    try:
        make3(r)
        abort_at_pause(r)
        r.run()
        assert [d["result"] for d in r.ends(3)] == ["pass"]
        assert len(stage_3_tests(r)[0]["chips"]) == 4
    finally:
        r.close()


# ---- a resume between stage 2 and stage 3 -----------------------------------------------------------

def test_a_kill_during_stage_3_resumes_without_running_stage_2_again_and_still_passes(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (2, 4)
    try:
        make3(r)
        abort_at_pause(r)
        with pytest.raises(Crash):
            r.run(crash_if=lambda e: e["event"] == "stage_start" and e["stage"] == 3)
        assert [d["result"] for d in r.ends(2)] == ["pass"] and not r.ends(3)
        assert r.run(pid=200) == EXIT_ABORTED
        assert len(started(r, 2)) == 1, "stage 2 passed and is not run again"
        assert [d["result"] for d in r.ends(3)] == ["pass"]
        assert len(stage_3_tests(r)) == 1
    finally:
        r.close()


def test_a_kill_after_the_stage_3_test_resumes_at_the_finish_step_without_a_second_test(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (2, 4)
    try:
        make3(r)
        abort_at_pause(r)
        with pytest.raises(Crash):
            r.run(crash_if=lambda e: e["event"] == "evidence" and e["stage"] == 3 and "test-result.json" in json.dumps(e["data"]))
        assert len(stage_3_tests(r)) == 1
        assert r.run(pid=200) == EXIT_ABORTED
        assert len(stage_3_tests(r)) == 1, "the hardware test is not run twice"
        assert [d["result"] for d in r.ends(3)] == ["pass"]
    finally:
        r.close()


# ---- the supervisor's own wiring --------------------------------------------------------------------

def gate_stub(*, required=(2, 4), entries=()):
    stub = SimpleNamespace(required_chips=required, ledger=SimpleNamespace(read=lambda: list(entries)))
    stub._stage_3_chips = lambda: Supervisor._stage_3_chips(stub)
    return stub


def test_the_supervisor_hands_the_gate_the_configured_chip_count_from_its_own_option():
    gate = Supervisor._gate(gate_stub(), FULL_PORT_STAGE_3)
    assert isinstance(gate, functools.partial) and gate.keywords == {"expected_chips": 4}
    assert Supervisor._gate(gate_stub(required=(2, 8)), FULL_PORT_STAGE_3).keywords == {"expected_chips": 8}


def test_the_rehearsal_flag_reaches_the_stage_3_gate_from_the_ledger_only():
    start = {"event": "run_start", "stage": None, "data": {"full_port": True, "full_port_rehearsal": True}}
    assert Supervisor._gate(gate_stub(entries=[start]), FULL_PORT_STAGE_3).keywords == {"expected_chips": 4, "rehearsal": True}


def test_repair_is_never_pending_in_stage_3_even_with_a_rerun_in_flight_in_its_ledger_entries(tmp_path):
    entries = [{"event": "stage_start", "stage": 3, "data": {"escalated": False}},
               {"event": "decision", "stage": 3, "data": {"decision": port_repair.RERUN, "round": 1, "record": {"path": "x", "sha256": "0" * 64}}}]
    assert port_repair.rerun_inflight(entries, 3) is not None             # the question would be yes if the stage were 2
    stub = SimpleNamespace(port_repair_rounds=2, ledger=SimpleNamespace(read=lambda: entries))
    assert Supervisor._repair_pending(stub, FULL_PORT_STAGE_3, tmp_path) is False


# ---- the seed ---------------------------------------------------------------------------------------

class Ledger:
    def __init__(self):
        self.rows = []

    def append(self, event, stage, **data):
        self.rows.append((event, stage, data))


def seeder(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "3"
    stage.mkdir(parents=True)
    (run / "stages" / "2" / "port").mkdir(parents=True)
    return SimpleNamespace(run_dir=run, ledger=Ledger()), stage, run / "stages" / "2" / "port"


def seed(sup, stage):
    Supervisor._seed_stage_3_port(sup, SimpleNamespace(number=3), stage)
    return [d.get("decision") for _, _, d in sup.ledger.rows]


def test_the_seed_copies_the_regular_files_and_skips_caches(tmp_path):
    sup, stage, src = seeder(tmp_path)
    (src / "port_model.py").write_text("import ttnn\n")
    (src / "layers").mkdir()
    (src / "layers" / "attn.py").write_text("x = 1\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "port_model.cpython-312.pyc").write_bytes(b"\0")
    assert seed(sup, stage) == ["stage 2 port copied"]
    assert (stage / "port" / "port_model.py").read_text() == "import ttnn\n" and (stage / "port" / "layers" / "attn.py").is_file()
    assert not (stage / "port" / "__pycache__").exists()
    files = sup.ledger.rows[0][2]["files"]
    assert {f["path"] for f in files} == {"stages/3/port/port_model.py", "stages/3/port/layers/attn.py"}
    assert all(len(f["sha256"]) == 64 for f in files)


def test_the_seed_refuses_a_link_and_leaves_no_half_a_port(tmp_path):
    sup, stage, src = seeder(tmp_path)
    (src / "a_first.py").write_text("import ttnn\n")
    (src / "z_link.py").symlink_to(src / "a_first.py")
    assert seed(sup, stage) == ["stage 2 port not copied"]
    assert "is not a regular file" in sup.ledger.rows[0][2]["problem"], "the supervisor's own refusal names the link, not only the reader's"
    assert not (stage / "port").exists(), "a copy that stopped on a problem is removed"


def test_the_seed_refuses_too_many_files_and_a_file_over_the_cap(tmp_path, monkeypatch):
    sup, stage, src = seeder(tmp_path)
    monkeypatch.setattr(supervisor, "STAGE_3_SEED_FILES", 2)
    for i in range(3):
        (src / f"f{i}.py").write_text("import ttnn\n")
    assert seed(sup, stage) == ["stage 2 port not copied"] and "more than 2 files" in sup.ledger.rows[0][2]["problem"]
    sup2, stage2, src2 = seeder(tmp_path / "b")
    monkeypatch.setattr(supervisor, "STAGE_3_SEED_FILES", 200)
    monkeypatch.setattr(supervisor, "STAGE_3_SEED_FILE_BYTES", 10)
    (src2 / "big.py").write_text("x" * 100)
    assert seed(sup2, stage2) == ["stage 2 port not copied"] and not (stage2 / "port").exists()


@pytest.mark.timeout(30)
def test_the_seed_does_not_open_a_fifo(tmp_path):
    import os
    sup, stage, src = seeder(tmp_path)
    (src / "a.py").write_text("import ttnn\n")
    os.mkfifo(src / "b_pipe.py")
    assert seed(sup, stage) == ["stage 2 port not copied"] and "is not a regular file" in sup.ledger.rows[0][2]["problem"]


def test_the_seed_problem_text_carries_no_path_when_a_file_cannot_be_read(tmp_path):
    import os
    sup, stage, src = seeder(tmp_path)
    secret = src / "locked_model.py"
    secret.write_text("import ttnn\n")
    os.chmod(secret, 0)
    try:
        if os.access(secret, os.R_OK):
            pytest.skip("this account can read a mode-0 file (running as root)")
        assert seed(sup, stage) == ["stage 2 port not copied"]
        problem = sup.ledger.rows[0][2]["problem"]
        assert str(tmp_path) not in problem and "locked_model" not in problem and problem.startswith(("SafeReadError", "ReadFailed", "PermissionError"))
    finally:
        os.chmod(secret, 0o644)


def test_the_seed_keeps_a_port_already_in_the_stage_directory(tmp_path):
    sup, stage, src = seeder(tmp_path)
    (src / "port_model.py").write_text("from_stage_2\n")
    (stage / "port").mkdir()
    (stage / "port" / "port_model.py").write_text("the coder's work on a resumed stage\n")
    assert seed(sup, stage) == []
    assert (stage / "port" / "port_model.py").read_text() == "the coder's work on a resumed stage\n"


def test_an_empty_stage_2_port_is_not_a_seed(tmp_path):
    sup, stage, _ = seeder(tmp_path)
    assert seed(sup, stage) == ["stage 2 port not copied"]
