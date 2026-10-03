"""End to end: a scripted Hemmingway-1 bring-up from stage 0 to the operator bundle, against a fake
machine and fake model servers, with the supervisor killed after every ledger event (spec section 13).
The scripted delta is weights-only, so stage 2 runs the weights swap check and stage 3 is skipped.

The two machine layouts are the operator's: the coder on all four chips (every hardware stage
parks it) and the coder on two chips (every hardware stage uses the free board).
"""

import pytest

from fake_model import FakeModel
from fakes import Crash
from orchard.ledger import Ledger, replay_state
from orchard.stages import run_progress
from orchard.supervisor import EXIT_READY, build, parse
from run_fakes import (BOARDS, SWAP_LOW, CrashingLedger, FakeContainers, Machine, MachineAdapter,
                       MachineCoder, argv, bringup, clock, feedback_aware, plenty,
                       test_fails_until_escalated, write_tiers)

FIRST, SECOND = 100, 200          # supervisor pids before and after the kill


@pytest.fixture(scope="module")
def servers():
    with FakeModel(bringup, models=["qwen-27b"]) as chips, FakeModel(bringup, models=["cpu-model"]) as cpu:
        yield chips, cpu


def run(base, servers, machine, *, chips, pid, crash_after=0):
    chip_server, cpu_server = servers
    tiers = base / "tiers.toml"
    base.mkdir(parents=True, exist_ok=True)
    if not tiers.exists():
        write_tiers(tiers, chip_server.endpoint, cpu_server.endpoint)
    args = parse(argv(base, tiers, chip_server.endpoint, chips=chips))
    machine.owner_pid = pid
    c = clock()
    path = base / "run" / "ledger.jsonl"
    with (CrashingLedger(path, crash_after) if crash_after else Ledger(path)) as led:
        return build(args, led, adapter=MachineAdapter(machine, owner_pid=pid),
                     coder=MachineCoder(machine), versions={"tt_model": "test"}, clock=c,
                     sleep=c.sleep, disk_usage=plenty, home=base / "operator-home",
                     containers=FakeContainers()).run()


def entries(base):
    with Ledger(base / "run" / "ledger.jsonl") as led:
        return led.read()


def final_state(base):
    """What must be the same however many times the supervisor died on the way."""
    es = entries(base)
    p = run_progress(es)
    last = {}
    for e in es:
        if e["event"] == "stage_end":
            last[e["stage"]] = e["data"]["result"]
    bundle = base / "run" / "stages" / "8" / "bundle"
    return {"finished": p.finished, "done": p.done, "parked": replay_state(es)["parked"],
            "paused": p.paused, "last_result": last,
            "bundle": {f.name: f.read_text() for f in sorted(bundle.iterdir()) if f.name != "ledger.jsonl"},
            # The stage 6 result numbers (they carry an evidence key); handoff timings vary with kills.
            "numbers": sorted({e["data"]["name"] for e in es if e["event"] == "measurement"
                               and e["stage"] == 6 and "evidence" in e["data"]})}


def test_with_the_coder_on_four_chips_every_hardware_stage_parks_it(tmp_path, servers):
    m = Machine()
    assert run(tmp_path, servers, m, chips=4, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    for n in (2, 5, 6):                     # stage 3 is skipped on the weights-only path
        seq = [(e["event"], e["data"].get("step") or e["data"].get("decision")) for e in es if e["stage"] == n]
        test = seq.index(("decision", "hardware test started"))
        assert seq.index(("park", "reset")) < test < seq.index(("restore", "reset")), n
        assert ("restore", "resumed") in seq[test:], n
        env = (tmp_path / "run" / "stages" / str(n) / "evidence" / "devices.txt").read_text()
        assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B0'])}\n" in env       # one board of the four chips
    # Stage 4 runs one test per configuration, each between its own park's reset and restore's reset.
    marks = (("park", "reset"), ("decision", "hardware test started"), ("restore", "reset"))
    seq = [(e["event"], e["data"].get("step") or e["data"].get("decision")) for e in es if e["stage"] == 4]
    assert [x for x in seq if x in marks] == list(marks) * 3
    for n, chips in ((1, BOARDS["B0"][:1]), (2, BOARDS["B0"]), (4, BOARDS["B0"] + BOARDS["B1"])):
        env = (tmp_path / "run" / "stages" / "4" / "configs" / str(n) / "evidence" / "devices.txt").read_text()
        assert f"TT_VISIBLE_DEVICES={','.join(chips)}\n" in env, n
        assert f"ORCHARD_DEVICE_IDS={','.join(str(i) for i in range(n))}\n" in env, n
    s = final_state(tmp_path)
    assert s["finished"] and s["done"] == (0, 1, 2, 3, 4, 5, 6, 7, 8) and not s["parked"]
    assert s["last_result"][3] == "skipped" and s["last_result"][7] == "skipped"
    assert not m.coder_running and m.leases == {}           # the finished run gave the hardware back


def test_with_the_coder_on_two_chips_only_the_4_chip_test_parks_it(tmp_path, servers):
    m = Machine()
    assert run(tmp_path, servers, m, chips=2, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    handoff = [(e["stage"], e["event"], e["data"]["step"]) for e in es if e["event"] in ("park", "restore")]
    assert {s for s, _, _ in handoff} == {4}
    assert [x for x in handoff if x[2] in ("note", "resumed")] == [(4, "park", "note"), (4, "restore", "resumed")]
    taken = [(e["stage"], e["data"].get("config"), e["data"]["test_lease"]["chips"]) for e in es
             if e["data"].get("decision") == "test lease taken"]
    b1 = list(BOARDS["B1"])
    # Stages 2, 5 and 6 and the 1- and 2-chip configurations use the free board; the 4-chip
    # configuration takes the free board too, then parks the coder for its board.
    assert taken == [(2, None, b1), (4, 1, b1), (4, 2, b1), (4, 4, b1), (5, None, b1), (6, None, b1)]
    seq = [(e["data"].get("decision") or e["data"].get("step"), e["data"].get("config")) for e in es if e["stage"] == 4]
    assert seq.index(("test lease taken", 4)) < seq.index(("note", None))      # the further board first
    env = (tmp_path / "run" / "stages" / "4" / "configs" / "4" / "evidence" / "devices.txt").read_text()
    assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B0'] + BOARDS['B1'])}\n" in env
    released = [e for e in es if e["data"].get("decision") == "test lease released"]
    assert len(released) == 6 and m.leases == {}
    assert final_state(tmp_path)["finished"]


def test_a_weights_only_run_serves_and_compares_in_stage_2_skips_3_and_goes_on_to_4(tmp_path, servers):
    m = Machine()
    assert run(tmp_path, servers, m, chips=2, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    assert next(e for e in es if e["event"] == "stage_end" and e["stage"] == 0)["data"]["path"] == "weights-only"
    skills = [e["data"]["skill"] for e in es if e["stage"] == 2 and e["data"].get("decision") == "agent step"]
    assert [s.rsplit("/", 1)[1] for s in skills] == ["weights-swap-check.md"] * 2     # prepare, finish
    assert [e["stage"] for e in es if e["event"] == "stage_start"][:6] == [0, 1, 2, 3, 4, 5]
    s3 = [(e["event"], e["data"].get("result")) for e in es if e["stage"] == 3]
    assert s3 == [("stage_start", None), ("stage_end", "skipped")]
    assert final_state(tmp_path)["last_result"] == {0: "pass", 1: "pass", 2: "pass", 3: "skipped",
                                                    4: "pass", 5: "pass", 6: "pass", 7: "skipped", 8: "pass"}


@pytest.mark.parametrize("chips", [4, 2])
def test_a_kill_after_any_ledger_event_reaches_the_same_final_state(tmp_path, servers, chips):
    ref = tmp_path / "reference"
    assert run(ref, servers, Machine(), chips=chips, pid=FIRST) == EXIT_READY
    want = final_state(ref)
    # The reference walks the weights-only path, so the kills below include one after stage 3's
    # skip start and one after its skip end.
    assert want["last_result"][3] == "skipped"
    total = len(entries(ref))
    for k in range(1, total + 1):
        base = tmp_path / f"kill-{k:03d}"
        m = Machine()
        with pytest.raises(Crash):
            run(base, servers, m, chips=chips, pid=FIRST, crash_after=k)
        assert run(base, servers, m, chips=chips, pid=SECOND) == EXIT_READY, k
        assert final_state(base) == want, k
        # The hardware is given back, including any lease the killed supervisor held.
        assert not m.coder_running and m.leases == {}, (k, m.leases)


def test_the_kill_test_has_a_kill_point_inside_the_coder_boot(tmp_path, servers):
    """A kill between the container start and its first answer is the likeliest Ctrl-C (a cold
    boot takes about 30 minutes). The kill test kills only after ledger appends, so the boot needs
    one append while the container runs and before "coder started"."""
    m = Machine()
    assert run(tmp_path, servers, m, chips=4, pid=FIRST) == EXIT_READY
    names = [e["data"].get("decision") for e in entries(tmp_path)]
    first, started = names.index("coder starting"), names.index("coder started")
    assert "coder container started" in names[first + 1:started]


@pytest.mark.parametrize("stage,phase,bad,chips", [
    (1, "run", {"evidence/notes.txt": "work done, no reference.json"}, 2),
    (2, "finish", SWAP_LOW, 4),
])
def test_a_kill_around_the_gate_feedback_continuation_resumes_to_the_same_final_state(
        tmp_path, stage, phase, bad, chips):
    """A stage whose first step ends without a passing gate and whose continuation fixes it. The
    supervisor is killed after each ledger event of that stage, including between the "gate
    feedback" decision and the continuation's result. A resumed stage starts a fresh step (the
    killed conversation is gone), which may use its own one continuation."""
    script = feedback_aware(stage, phase, bad)
    with FakeModel(script, models=["qwen-27b"]) as chip, FakeModel(script, models=["cpu-model"]) as cpu:
        ref = tmp_path / "reference"
        assert run(ref, (chip, cpu), Machine(), chips=chips, pid=FIRST) == EXIT_READY
        want = final_state(ref)
        es = entries(ref)
        assert want["last_result"][stage] == "pass" and not [e for e in es if e["event"] == "escalate"]
        idx = [i for i, e in enumerate(es) if e["stage"] == stage]
        fb = next(i for i, e in enumerate(es) if e["data"].get("decision") == "gate feedback")
        assert idx[0] < fb < idx[-1]
        for k in range(idx[0] + 1, idx[-1] + 2):            # kill after each of the stage's events
            base = tmp_path / f"kill-{k:03d}"
            m = Machine()
            with pytest.raises(Crash):
                run(base, (chip, cpu), m, chips=chips, pid=FIRST, crash_after=k)
            assert run(base, (chip, cpu), m, chips=chips, pid=SECOND) == EXIT_READY, k
            assert final_state(base) == want, k
            assert not m.coder_running and m.leases == {}, (k, m.leases)
            # Each run of the stage body (one per stage_start) has at most one continuation.
            # The resumed stage still passed through its continuation, with no escalation.
            assert not [e for e in entries(base) if e["event"] == "escalate"], k
            per_start = []
            for e in entries(base):
                if e["stage"] != stage:
                    continue
                if e["event"] == "stage_start":
                    per_start.append(0)
                elif e["data"].get("decision") == "gate feedback":
                    per_start[-1] += 1
            assert per_start and max(per_start) <= 1, (k, per_start)


def test_a_kill_around_a_failed_hardware_test_resumes_to_the_same_final_state(tmp_path):
    """Stage 2's first hardware test exits with code 4 and the finish step records the failure. The
    supervisor gives no continuation, records a "no gate feedback" decision and escalates, and the
    escalated attempt tests again and passes. The supervisor is killed after each ledger event of
    stage 2, including between that decision and the stage_end, and each resumed run must reach the
    same final state with no continuation at all."""
    script = test_fails_until_escalated(2, "exit")
    with FakeModel(script, models=["qwen-27b"]) as chip, FakeModel(script, models=["cpu-model"]) as cpu:
        ref = tmp_path / "reference"
        assert run(ref, (chip, cpu), Machine(), chips=2, pid=FIRST) == EXIT_READY
        want = final_state(ref)
        es = entries(ref)
        assert want["last_result"][2] == "pass"
        assert [e["data"]["result"] for e in es if e["event"] == "stage_end" and e["stage"] == 2] == \
            ["escalate", "pass"]
        idx = [i for i, e in enumerate(es) if e["stage"] == 2]
        nofb = next(i for i, e in enumerate(es)
                    if str(e["data"].get("decision")).startswith("no gate feedback"))
        assert idx[0] < nofb < idx[-1]
        # crash_after=k kills after the k-th append, which is list index k - 1.
        after_escalate = next(i for i, e in enumerate(es) if e["event"] == "escalate") + 1
        for k in range(idx[0] + 1, idx[-1] + 2):            # kill after each of the stage's events
            base = tmp_path / f"kill-{k:03d}"
            m = Machine()
            with pytest.raises(Crash):
                run(base, (chip, cpu), m, chips=2, pid=FIRST, crash_after=k)
            assert run(base, (chip, cpu), m, chips=2, pid=SECOND) == EXIT_READY, k
            assert final_state(base) == want, k
            assert not m.coder_running and m.leases == {}, (k, m.leases)
            got = entries(base)
            assert not [e for e in got if e["data"].get("decision") == "gate feedback"], k
            # A kill between the escalate entry and the stage_end resumes the stage escalated. Its
            # failed test record is not reused, so the escalated attempt tests again.
            fresh = [e for e in got if e["data"].get("decision") == "not resuming from a failed hardware test"]
            assert len(fresh) == (1 if k == after_escalate else 0), k
