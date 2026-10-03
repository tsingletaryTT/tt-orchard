"""End to end: a scripted Hemmingway-1 bring-up from stage 0 to the operator bundle, against a fake
machine and fake model servers, with the supervisor killed after every ledger event (spec section 13).

The two machine layouts are the operator's: the coder on all four chips (every hardware stage
parks it) and the coder on two chips (every hardware stage uses the free board).
"""

import pytest

from fake_model import FakeModel
from fakes import Crash
from orchard.ledger import Ledger, replay_state
from orchard.stages import run_progress
from orchard.supervisor import EXIT_READY, build, parse
from run_fakes import (BOARDS, CrashingLedger, Machine, MachineAdapter, MachineCoder, argv, bringup,
                       clock, plenty, write_tiers)

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
                     sleep=c.sleep, disk_usage=plenty, home=base / "operator-home").run()


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
    for n in (2, 3, 4, 5, 6):
        seq = [(e["event"], e["data"].get("step") or e["data"].get("decision")) for e in es if e["stage"] == n]
        test = seq.index(("decision", "hardware test started"))
        assert seq.index(("park", "reset")) < test < seq.index(("restore", "reset")), n
        assert ("restore", "resumed") in seq[test:], n
        env = (tmp_path / "run" / "stages" / str(n) / "evidence" / "devices.txt").read_text()
        assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B0'])}\n" in env       # one board of the four chips
    s = final_state(tmp_path)
    assert s["finished"] and s["done"] == (0, 1, 2, 3, 4, 5, 6, 7, 8) and not s["parked"]
    assert s["last_result"][7] == "skipped"
    assert not m.coder_running and m.leases == {}           # the finished run gave the hardware back


def test_with_the_coder_on_two_chips_the_free_board_is_used_and_nothing_parks(tmp_path, servers):
    m = Machine()
    assert run(tmp_path, servers, m, chips=2, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    assert not [e for e in es if e["event"] in ("park", "restore")]
    taken = [e["data"]["test_lease"]["chips"] for e in es if e["data"].get("decision") == "test lease taken"]
    assert taken == [list(BOARDS["B1"])] * 5
    released = [e for e in es if e["data"].get("decision") == "test lease released"]
    assert len(released) == 5 and m.leases == {}
    assert final_state(tmp_path)["finished"]


@pytest.mark.parametrize("chips", [4, 2])
def test_a_kill_after_any_ledger_event_reaches_the_same_final_state(tmp_path, servers, chips):
    ref = tmp_path / "reference"
    assert run(ref, servers, Machine(), chips=chips, pid=FIRST) == EXIT_READY
    want = final_state(ref)
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
