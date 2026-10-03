"""Stage 4 on the weights-only path: one hardware test per chip configuration, each under its own
lease, with the coder parked only when its boards are needed."""
import hashlib
import json

import pytest

from orchard import supervisor
from orchard.adapters import ChipState, Queued
from orchard.supervisor import EXIT_READY
from run_fakes import (BOARDS, FILES, FakeContainers, MachineAdapter, bringup, swap_entry,
                       swap_prepare, where)
from test_supervisor import Rig, Stop, escalation_aware

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)                     # the coder on two chips, board B0
    yield r
    r.close()


def stage4(rig, decision):
    return [e["data"] for e in rig.entries() if e["stage"] == 4 and e["event"] == "decision"
            and e["data"].get("decision") == decision]


def stop():
    raise Stop()


def test_a_weights_only_run_gives_stage_4_the_configs_skill_and_runs_each_configuration_once(rig):
    assert rig.run() == EXIT_READY
    skills = [d["skill"].rsplit("/", 1)[1] for d in stage4(rig, "agent step")]
    assert skills == ["weights-swap-configs.md"] * 2                  # prepare, finish
    assert [d["config"] for d in stage4(rig, "hardware test started")] == [1, 2, 4]
    sd = rig.run_dir / "stages" / "4"
    assert json.loads((sd / "tests" / "plan.json").read_text())["tests"][2]["chips"] == 4
    assert [json.loads((sd / "tests" / str(n) / "test-result.json").read_text())["returncode"]
            for n in (1, 2, 4)] == [0, 0, 0]
    summary = json.loads((sd / "test-result.json").read_text())
    assert [t["config"] for t in summary["tests"]] == [1, 2, 4]
    assert [d["result"] for d in rig.ends(4)] == ["pass"]
    listed = [e["data"] for e in rig.entries() if e["event"] == "evidence" and e["stage"] == 4
              and e["data"]["what"] == "hardware test list"]
    assert listed[0]["configs"] == [1, 2, 4] and listed[0]["path"] == "stages/4/tests/plan.json"


def test_the_tests_get_their_chips_device_ids_and_the_runs_container_label(rig):
    assert rig.run() == EXIT_READY
    label = "orchard.test=" + hashlib.sha256(str(rig.run_dir.resolve()).encode()).hexdigest()[:12]
    for n, chips, ids in ((1, BOARDS["B1"][:1], "2"), (2, BOARDS["B1"], "2,3"),
                          (4, BOARDS["B0"] + BOARDS["B1"], "0,1,2,3")):
        env = (rig.run_dir / "stages" / "4" / "configs" / str(n) / "evidence" / "devices.txt").read_text()
        assert env.splitlines() == [f"TT_VISIBLE_DEVICES={','.join(chips)}", f"ORCHARD_DEVICE_IDS={ids}",
                                    f"ORCHARD_TEST_LABEL={label}"], n
    assert ("list", label) in rig.containers.calls


class SnapshotAtSpawn:
    """Wraps spawn_checked: records the machine's state at the moment each test starts."""

    def __init__(self, rig, real):
        self.rig, self.real, self.seen = rig, real, []

    def __call__(self, command, run_dir, env, timeout, out):
        m = self.rig.m
        self.seen.append({"command": command, "chips": env["TT_VISIBLE_DEVICES"].split(","),
                          "coder_running": m.coder_running,
                          "leases": {lid: (set(lease.chips), owner) for lid, (lease, owner) in m.leases.items()}})
        return self.real(command, run_dir, env, timeout, out)


class BoardOneBusy(MachineAdapter):
    """Another tenant holds board B1 until `busy` lease attempts for it have been turned away. While
    it is busy, gozer status shows B1's chips HELD by someone else."""
    busy = 0
    coder_up_while_queued: list = []

    def acquire(self, chips, who, reason, *, queue=False, exact=None):
        if BoardOneBusy.busy and (exact is None or exact in BOARDS["B1"]):
            BoardOneBusy.busy -= 1
            BoardOneBusy.coder_up_while_queued.append(self.m.coder_running)
            raise Queued("t-busy")
        return super().acquire(chips, who, reason, queue=queue, exact=exact)

    def status(self):
        out = super().status()
        if not BoardOneBusy.busy:
            return out
        return [ChipState(c.bdf, "HELD", "someone-else", board=c.board, dev_index=c.dev_index, lease_pid=999)
                if c.board == "B1" else c for c in out]


def test_the_4_chip_test_waits_for_a_board_another_tenant_holds_and_runs_only_on_this_runs_leases(rig, monkeypatch):
    BoardOneBusy.busy, BoardOneBusy.coder_up_while_queued = 0, []
    rig.adapter_cls = BoardOneBusy

    def script(request):
        if "tools" in request and where(request) == (4, "prepare"):
            BoardOneBusy.busy = 3                  # the other tenant takes B1 as stage 4 starts
        return bringup(request)
    rig.script = script
    spy = SnapshotAtSpawn(rig, supervisor.spawn_checked)
    monkeypatch.setattr(supervisor, "spawn_checked", spy)
    assert rig.run() == EXIT_READY
    # While B1 was busy the coder stayed up: the further board is leased before the park.
    assert BoardOneBusy.coder_up_while_queued == [True, True, True]
    four = next(s for s in spy.seen if s["command"].endswith("configs/4/serve_and_compare_container.py"))
    assert four["coder_running"] is False
    assert sorted(four["chips"]) == sorted(BOARDS["B0"] + BOARDS["B1"])
    assert all(owner == rig.m.owner_pid for _, owner in four["leases"].values())
    assert set().union(*(chips for chips, _ in four["leases"].values())) == set(four["chips"])


def failing_4_chip_test(required_ok=True):
    files = swap_prepare()
    files["configs/4/serve_and_compare_container.py"] = "import sys\nprint('server never healthy')\nsys.exit(4)\n"
    return {(4, "prepare"): files,
            (4, "finish"): {"result.json": {"configs": [swap_entry(1), swap_entry(2), swap_entry(4, ok=False)]}}}


def test_a_failed_4_chip_test_still_releases_its_board_and_restores_the_coder(rig):
    rig.args.required_chips = (2,)                 # here the 4-chip configuration is optional
    rig.script = lambda r: bringup(r, overrides=failing_4_chip_test())
    assert rig.run() == EXIT_READY
    es = [e for e in rig.entries() if e["stage"] == 4]
    seq = [(e["event"], e["data"].get("step") or e["data"].get("decision"), e["data"].get("config")) for e in es]
    test = seq.index(("decision", "hardware test started", 4))
    after = seq[test:]
    assert ("decision", "test lease released", None) in after
    # The coder comes back before the finish step, and through the restore itself: a restore left
    # for the next stage's recovery would also bring it back, one stage later.
    assert after.index(("restore", "canary", None)) < after.index(("restore", "resumed", None))
    assert after.index(("restore", "resumed", None)) < after.index(("decision", "agent step", None))
    assert not [x for x in seq if x[1] == "recover after restart; the machine wins"]
    rec = json.loads((rig.run_dir / "stages" / "4" / "tests" / "4" / "test-result.json").read_text())
    assert rec["returncode"] == 4
    assert [d["result"] for d in rig.ends(4)] == ["pass"] and rig.ends(5)[0]["result"] == "pass"


def test_a_pass_claimed_for_a_configuration_whose_test_never_ran_fails_the_gate(rig):
    # The agent drops the 4-chip test from the list and writes the files a passing test would
    # leave. Only the supervisor's record is missing.
    files = swap_prepare()
    files["hw_tests.json"] = {"tests": [t for t in files["hw_tests.json"]["tests"] if t["chips"] != 4]}
    files["configs/4/evidence/swap-check.json"] = {"result_draft": {"top1_agreement": 0.94}}
    files["tests/4/output.txt"] = "a test that never ran"
    rig.script = escalation_aware(4, {(4, "prepare"): files})
    assert rig.run() == EXIT_READY
    first = next(e["data"] for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    assert first["reasons"] == ["the 4-chip configuration claims a pass, but the supervisor has no "
                                "record of its test (tests/4/test-result.json)"]


def test_a_list_whose_configurations_share_a_cache_fails_before_any_hardware_is_used(rig):
    shared = str(rig.tmp / "orchard-cache" / "one-cache")
    rig.script = escalation_aware(4, {(4, "prepare"): swap_prepare(caches={1: shared, 2: shared})})
    assert rig.run() == EXIT_READY
    first = next(e for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    assert first["data"]["reasons"] == [f"tests[1]: tt_cache {shared} is also the 1-chip "
                                        "configuration's; each configuration needs its own"]
    before = [e for e in rig.entries() if e["seq"] < first["seq"] and e["stage"] == 4
              and e["data"].get("decision") in ("hardware test started", "test lease taken")]
    assert before == []


def test_a_list_missing_a_required_configuration_fails_before_any_hardware_is_used(rig):
    rig.args.required_chips = (2, 4)
    files = swap_prepare()
    files["hw_tests.json"] = {"tests": [t for t in files["hw_tests.json"]["tests"] if t["chips"] != 4]}
    rig.script = escalation_aware(4, {(4, "prepare"): files})
    assert rig.run() == EXIT_READY
    first = next(e["data"] for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    assert first["reasons"] == ["the 4-chip configuration is required and hw_tests.json has no test for it"]


def test_a_container_a_test_left_running_is_stopped_before_its_lease_goes_back(rig):
    rig.containers = FakeContainers([[], ["c1"], []])      # found after the 2-chip test, then gone
    assert rig.run() == EXIT_READY
    notes = [e for e in rig.entries() if e["event"] == "notice" and e["stage"] == 4]
    assert [n["data"]["containers"] for n in notes] == [["c1"]]
    assert ("remove", "c1") in rig.containers.calls
    seq = [(e["event"], e["data"].get("decision"), e["data"].get("lease_id")) for e in rig.entries()
           if e["stage"] == 4]
    note_at = next(i for i, x in enumerate(seq) if x[0] == "notice")
    released = [i for i, x in enumerate(seq) if x[1] == "test lease released"]
    assert released[0] < note_at < released[1]           # after the 1-chip release, before the 2-chip one


def test_a_container_that_will_not_stop_blocks_before_any_lease_is_released_or_the_coder_restored(rig):
    rig.containers = FakeContainers([[], [], ["c9"]])       # the 4-chip test's container stays
    at_pause = {}

    def paused():
        at_pause.update(leases=len(rig.m.leases), coder=rig.m.coder_running)
        raise Stop()
    rig.on_sleep = paused
    with pytest.raises(Stop):
        rig.run()
    blocked = [e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked[-1]["reason"].startswith("test containers are still running after docker stop and rm")
    assert blocked[-1]["evidence"]["containers"] == ["c9"]
    es = [e for e in rig.entries() if e["stage"] == 4]
    test = max(i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started")
    assert not [e for e in es[test:] if e["event"] == "restore"]
    assert not [e for e in es[test:] if e["data"].get("decision") == "test lease released"]
    assert at_pause == {"leases": 2, "coder": False}       # the coder's lease and the further board's


def test_a_test_record_the_agent_wrote_fails_the_gate(rig):
    # The 4-chip test exits 4. The finish step claims a pass and rewrites the record to match.
    forged = {"config": 4, "returncode": 0, "timed_out": False, "chips": ["a", "b", "c", "d"]}
    bad = failing_4_chip_test()
    bad[(4, "finish")] = {"tests/4/test-result.json": forged,
                          "result.json": {"configs": [swap_entry(1), swap_entry(2), swap_entry(4)]}}
    bad[(4, "prepare")]["configs/4/evidence/swap-check.json"] = {"result_draft": {"top1_agreement": 0.94}}
    rig.script = escalation_aware(4, bad)
    assert rig.run() == EXIT_READY
    first = next(e["data"] for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    assert first["reasons"] == ["stages/4/tests/4/test-result.json was not written by the supervisor: "
                                "its sha256 is not the one the ledger recorded"]
