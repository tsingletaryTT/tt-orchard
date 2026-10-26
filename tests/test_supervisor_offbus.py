# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 25, second half: a failed hardware test or coder boot is checked against the PCIe bus before it is called a model
failure. A board that left the bus ends the run as `hardware-unhealthy` with the chip and the cause, and the automatic retry
skips it. A failure with a healthy bus goes the way it always did.

The bus is a fake: each test replaces the default reader (orchard/bus.py `_default_read`) with one that returns a canned
`Bus` over the fake machine's chips. Nothing reads /sys or opens a chip."""
import json

import pytest

from orchard import blocked, bus, supervisor
from orchard.server import ServerError
from run_fakes import FakeContainers, MachineCoder, BOARDS, FAILING_TEST_SCRIPTS, SWAP_FAILED, hw_failing, test_fails_until_escalated  # noqa: F401
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")

ALL_CHIPS = [c for cs in BOARDS.values() for c in cs]


class FakeBus:
    """A bus over the fake machine's chips. `bad` maps a chip address to what is wrong with it."""

    def __init__(self, bad=None):
        self.bad = dict(bad or {})
        self.reads = 0

    def __call__(self):
        self.reads += 1
        chips = []
        for i, bdf in enumerate(ALL_CHIPS):
            state = self.bad.get(bdf, "D0")
            chips.append(bus.Chip(bdf=bdf, power_state=state, serial="0x0000046131924062", node=i, driver_bound=True))
        return bus.Bus(chips=chips, nodes=frozenset(range(len(ALL_CHIPS))))


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.args.unattended = True
    r.on_sleep = None
    yield r
    r.close()


@pytest.fixture
def fake_bus(monkeypatch):
    fb = FakeBus()
    monkeypatch.setattr(bus, "_default_read", fb)
    return fb


def decisions(rig, name):
    return [d for d in rig.decisions() if d.get("decision") == name]


def failing_stage_2(rig):
    rig.script = test_fails_until_escalated(2, "exit")


def test_a_failed_test_on_a_dead_board_ends_the_run_as_hardware_unhealthy_naming_the_chip(rig, fake_bus):
    failing_stage_2(rig)
    fake_bus.bad = {ALL_CHIPS[0]: "D3cold", ALL_CHIPS[1]: "D3cold", ALL_CHIPS[2]: "D3cold", ALL_CHIPS[3]: "D3cold"}
    assert rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert end["code"] == "hardware-unhealthy"
    assert bus.OFF_BUS in end["reason"] and "D3cold" in end["reason"]
    assert any(c in end["reason"] for c in ALL_CHIPS), "the block names a chip"
    assert "exited 4" in end["reason"], "the block says which failure came first"
    assert blocked.retryable(end["code"], end["reason"]) is False


def test_the_block_goes_into_blocked_md_with_the_cause(rig, fake_bus):
    failing_stage_2(rig)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    rig.run()
    text = (rig.run_dir / "BLOCKED.md").read_text()
    assert "hardware-unhealthy" in text and "D3cold" in text
    facts = json.loads((rig.run_dir / "blocked.json").read_text())
    assert bus.OFF_BUS in facts["reason"]


def test_the_leases_are_released_when_the_run_ends_on_a_dead_board(rig, fake_bus):
    failing_stage_2(rig)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    rig.run()
    assert rig.m.leases == {} and not rig.m.coder_running


def test_a_failed_test_on_a_healthy_bus_is_still_a_model_failure(rig, fake_bus):
    failing_stage_2(rig)
    rig.run()
    assert fake_bus.reads >= 1, "the failed test was checked against the bus"
    assert not [d for d in decisions(rig, "blocked") if bus.OFF_BUS in d.get("reason", "")]
    assert [d["result"] for d in rig.ends(2)][:1] == ["escalate"], "the failure took the normal path: the stage escalated"


def test_a_passing_test_does_not_read_the_bus(rig, fake_bus):
    assert rig.run() == supervisor.EXIT_READY
    assert fake_bus.reads == 0


def test_a_timed_out_test_is_checked_too(rig, fake_bus):
    rig.script = test_fails_until_escalated(2, "hang")
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    assert rig.run() == supervisor.EXIT_BLOCKED
    assert "timed out" in decisions(rig, "blocked")[-1]["reason"]


def test_only_the_chips_the_test_holds_are_judged(rig, fake_bus, monkeypatch):
    failing_stage_2(rig)
    real = supervisor.spawn_checked
    judged = []

    def spawn(command, run_dir, env, timeout, out):
        mine = set(env["TT_VISIBLE_DEVICES"].split(","))
        fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS if c not in mine}     # every other chip is dead
        judged.append(bool(fake_bus.bad))
        return real(command, run_dir, env, timeout, out)
    monkeypatch.setattr(supervisor, "spawn_checked", spawn)
    rig.run()
    assert any(judged), "at least one test held only part of the box"
    assert not [d for d in decisions(rig, "blocked") if bus.OFF_BUS in d.get("reason", "")]


def test_a_coder_that_does_not_start_on_a_dead_board_is_reported_as_off_the_bus(rig, fake_bus):
    rig.m.never_ready = True
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    assert rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert end["code"] == "hardware-unhealthy" and bus.OFF_BUS in end["reason"]
    assert blocked.retryable(end["code"], end["reason"]) is False


def test_a_coder_that_does_not_start_on_a_healthy_bus_keeps_its_own_block(rig, fake_bus):
    rig.m.never_ready = True
    assert rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert end["code"] == "coder-unusable" and bus.OFF_BUS not in end["reason"]
    assert fake_bus.reads >= 1


def test_the_reset_window_is_waited_out_before_the_board_is_called_dead(rig, fake_bus):
    failing_stage_2(rig)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    start = rig.clock()
    rig.run()
    blocked_at = [e for e in rig.entries() if e["event"] == "decision" and e["data"].get("decision") == "blocked"]
    assert blocked_at and fake_bus.reads > 1, "the bus was read again during the window"
    assert rig.clock() - start >= bus.SETTLE_S


def test_a_board_that_comes_back_inside_the_window_is_not_dead(rig, fake_bus):
    failing_stage_2(rig)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    rig.on_sleep = lambda: fake_bus.bad.clear()           # the first wait ends the reset
    rig.run()
    assert not [d for d in decisions(rig, "blocked") if bus.OFF_BUS in d.get("reason", "")]


# ---- Copilot review of PR 79, finding 2: in lab mode the coder is still local, so its chips are still checked -------------------

@pytest.fixture
def lab_rig(tmp_path):
    from test_supervisor_lab import LabRig
    r = LabRig(tmp_path)
    r.args.unattended = True
    r.on_sleep = None
    yield r
    r.close()


def test_a_lab_runs_local_coder_that_does_not_start_on_a_dead_board_is_off_the_bus(lab_rig, fake_bus):
    lab_rig.m.never_ready = True
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    assert lab_rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(lab_rig, "blocked")[-1]
    assert end["code"] == "hardware-unhealthy" and bus.OFF_BUS in end["reason"]
    assert blocked.retryable(end["code"], end["reason"]) is False


def test_a_failed_test_on_the_lab_is_not_judged_by_this_boxs_bus(lab_rig, fake_bus):
    lab_rig.script = test_fails_until_escalated(2, "exit")
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}              # this box is dead, but the test ran on the lab
    lab_rig.run()
    assert fake_bus.reads == 0
    assert not [d for d in decisions(lab_rig, "blocked") if bus.OFF_BUS in d.get("reason", "")]


def test_off_bus_reads_the_local_chips_whether_or_not_there_is_a_lab():
    class Fake:
        lab, coder_chips = object(), 2
        now = [0.0]
        sleep = staticmethod(lambda s: Fake.now.__setitem__(0, Fake.now[0] + s))
        clock = staticmethod(lambda: Fake.now[0])

        @staticmethod
        def bus_read():
            return FakeBus({ALL_CHIPS[0]: "D3cold"})()
    msg = supervisor.Supervisor._off_bus(Fake(), [ALL_CHIPS[0]])
    assert msg and ALL_CHIPS[0] in msg


def test_a_test_command_the_runner_refuses_never_reached_a_chip_so_the_bus_is_not_blamed(rig, fake_bus, monkeypatch):
    from orchard.runner import Denied

    def refuse(command, run_dir, env, timeout, out):
        raise Denied("test-rule", command)
    monkeypatch.setattr(supervisor, "spawn_checked", refuse)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    rig.run()
    assert not [d for d in decisions(rig, "blocked") if bus.OFF_BUS in d.get("reason", "")]
    assert fake_bus.reads == 0


def test_with_no_chips_named_the_coders_chip_count_decides():
    class Fake:
        lab, coder_chips = None, 4
        now = [0.0]
        sleep = staticmethod(lambda s: Fake.now.__setitem__(0, Fake.now[0] + s))
        clock = staticmethod(lambda: Fake.now[0])

        @staticmethod
        def bus_read():
            chips = [bus.Chip(bdf=b, power_state="D0", serial="0x0000046131924062", node=i, driver_bound=True)
                     for i, b in enumerate(ALL_CHIPS[:2])]
            return bus.Bus(chips=chips, nodes=frozenset({0, 1}))
    msg = supervisor.Supervisor._off_bus(Fake(), [])
    assert msg and "2 of the 4" in msg


class RefusesToStart(MachineCoder):
    """The container never starts at all (the start call itself raises), as opposed to starting and not answering."""

    def start(self, lease):
        raise ServerError("docker run failed")


def test_a_coder_whose_start_call_fails_on_a_dead_board_is_reported_as_off_the_bus(rig, fake_bus):
    rig.coder_cls = RefusesToStart
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    assert rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert end["code"] == "hardware-unhealthy" and bus.OFF_BUS in end["reason"] and "docker run failed" in end["reason"]


def test_a_coder_whose_start_call_fails_on_a_healthy_bus_keeps_its_own_block(rig, fake_bus):
    rig.coder_cls = RefusesToStart
    assert rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert bus.OFF_BUS not in end["reason"] and "docker run failed" in end["reason"]
    assert fake_bus.reads >= 1


# ---- Copilot review of PR 79, finding 1: an off-bus block inside a listed test still sweeps the test containers -------------------

FLAKY = "import sys\nprint('Fabric Router Sync: Timeout')\nsys.exit(4)\n"


class ContainerAfterTest(FakeContainers):
    """No container until the 4-chip test has run, then `c1` until it is removed (or for ever when `stubborn`)."""

    def __init__(self, stubborn=False):
        super().__init__()
        self.armed, self.gone, self.stubborn = False, False, stubborn

    def list(self, label):
        self.calls.append(("list", label))
        return ["c1"] if self.armed and not self.gone else []

    def remove(self, cid):
        self.calls.append(("remove", cid))
        self.gone = not self.stubborn


def stage4_failing_on_a_dead_board(rig, fake_bus, containers, chips=4):
    from run_fakes import SWAP_SCRIPTS, bringup, swap_entry, swap_prepare
    rig.containers = containers
    files = swap_prepare()
    script = f"configs/{chips}/{SWAP_SCRIPTS[chips]}"
    files[script] = FLAKY
    finish = {"result.json": {"configs": [swap_entry(n, ok=n != chips) for n in (1, 2, 4)]}}
    rig.script = lambda r: bringup(r, overrides={(4, "prepare"): files, (4, "finish"): finish})
    real = supervisor.spawn_checked

    def spawn(command, run_dir, env, timeout, out):
        got = real(command, run_dir, env, timeout, out)
        if command.endswith(script):
            fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}              # the board is gone once the 4-chip test has failed
            containers.armed = True                                      # and the test left its container running
        return got
    return spawn


@pytest.mark.parametrize("chips", [2, 4])          # 2 uses a free board, 4 parks the coder: two code paths
def test_a_dead_board_found_after_a_listed_test_still_stops_its_containers(rig, fake_bus, monkeypatch, chips):
    monkeypatch.setattr(supervisor, "spawn_checked", stage4_failing_on_a_dead_board(rig, fake_bus, ContainerAfterTest(), chips))
    assert rig.run() == supervisor.EXIT_BLOCKED
    assert ("remove", "c1") in rig.containers.calls, "the labelled containers were stopped before the block ended the stage"
    end = decisions(rig, "blocked")[-1]
    assert bus.OFF_BUS in end["reason"], "the off-bus reason is kept"


@pytest.mark.parametrize("chips", [2, 4])
def test_a_container_that_survives_keeps_the_test_lease_when_the_block_is_off_bus(rig, fake_bus, monkeypatch, chips):
    monkeypatch.setattr(supervisor, "spawn_checked", stage4_failing_on_a_dead_board(rig, fake_bus, ContainerAfterTest(stubborn=True), chips))
    assert rig.run() == supervisor.EXIT_BLOCKED
    assert rig.m.leases, "a lease a live container runs under was not given back"
    end = decisions(rig, "blocked")[-1]
    assert bus.OFF_BUS in end["reason"] or "still running" in end["reason"]


# ---- Copilot review of PR 79, finding 3: a coder that does not come back after a park is checked against the bus too -------------

def restore_fails_after_the_4_chip_test(rig, fake_bus, dead):
    """The 4-chip test parks the coder. When it has run, the coder stops coming back (never ready), so the restore fails."""
    from run_fakes import SWAP_SCRIPTS, bringup, swap_entry, swap_prepare
    rig.script = lambda r: bringup(r, overrides={(4, "prepare"): swap_prepare(),
                                                 (4, "finish"): {"result.json": {"configs": [swap_entry(n) for n in (1, 2, 4)]}}})
    real = supervisor.spawn_checked
    script = f"configs/4/{SWAP_SCRIPTS[4]}"

    def spawn(command, run_dir, env, timeout, out):
        got = real(command, run_dir, env, timeout, out)
        if command.endswith(script):
            rig.m.never_ready = True                                   # the restart that follows never answers
            if dead:
                fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
        return got
    return spawn


def test_a_coder_that_does_not_come_back_after_a_park_on_a_dead_board_is_off_the_bus(rig, fake_bus, monkeypatch):
    monkeypatch.setattr(supervisor, "spawn_checked", restore_fails_after_the_4_chip_test(rig, fake_bus, dead=True))
    assert rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert end["code"] == "hardware-unhealthy" and bus.OFF_BUS in end["reason"]
    assert "cold-boot budget" in end["reason"], "the failure that came first is kept"
    assert blocked.retryable(end["code"], end["reason"]) is False


def test_a_coder_that_does_not_come_back_after_a_park_on_a_healthy_bus_keeps_its_own_block(rig, fake_bus, monkeypatch):
    monkeypatch.setattr(supervisor, "spawn_checked", restore_fails_after_the_4_chip_test(rig, fake_bus, dead=False))
    assert rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert bus.OFF_BUS not in end["reason"] and "cold-boot budget" in end["reason"]
    assert fake_bus.reads >= 1
    assert not rig.ends(4), "stage 4 did not end as if the restore had worked"


def test_a_recovery_after_a_restart_checks_the_bus_when_the_coder_does_not_come_back(rig, fake_bus):
    from run_fakes import Crash
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "park" and e["data"].get("step") == "stopped")
    rig.m.never_ready = True
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    assert rig.run(pid=200) == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert end["code"] == "hardware-unhealthy" and bus.OFF_BUS in end["reason"]


def die_after_the_first_test(rig, fake_bus, monkeypatch):
    """Stage 2's test is the first command the supervisor spawns. Once it has run, the coder stops coming back and the board is dead."""
    real = supervisor.spawn_checked

    def spawn(command, run_dir, env, timeout, out):
        got = real(command, run_dir, env, timeout, out)
        rig.m.never_ready = True
        fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
        return got
    monkeypatch.setattr(supervisor, "spawn_checked", spawn)


def test_a_coder_that_does_not_come_back_after_the_stage_2_park_is_off_the_bus(tmp_path, fake_bus, monkeypatch):
    r = Rig(tmp_path, chips=4)                   # a coder on both boards is parked for every hardware test
    r.args.unattended = True
    r.on_sleep = None
    try:
        die_after_the_first_test(r, fake_bus, monkeypatch)
        assert r.run() == supervisor.EXIT_BLOCKED
        assert [d for d in decisions(r, "hardware phase") if d.get("action") == "park"]
        end = decisions(r, "blocked")[-1]
        assert bus.OFF_BUS in end["reason"] and "cold-boot budget" in end["reason"]
    finally:
        r.close()


def test_a_coder_that_does_not_come_back_after_a_wide_container_test_is_off_the_bus(tmp_path, fake_bus, monkeypatch):
    from test_supervisor_container_base import wide_rig
    r = wide_rig(tmp_path)
    r.args.unattended = True
    r.on_sleep = None
    try:
        die_after_the_first_test(r, fake_bus, monkeypatch)
        assert r.run() == supervisor.EXIT_BLOCKED
        end = decisions(r, "blocked")[-1]
        assert bus.OFF_BUS in end["reason"] and "cold-boot budget" in end["reason"]
    finally:
        r.close()


def test_only_the_coders_chips_are_judged_when_the_restore_fails(rig, fake_bus, monkeypatch):
    from run_fakes import BOARDS
    spawn = restore_fails_after_the_4_chip_test(rig, fake_bus, dead=False)

    def with_the_other_board_dead(command, run_dir, env, timeout, out):
        got = spawn(command, run_dir, env, timeout, out)
        if rig.m.never_ready:
            fake_bus.bad = {c: "D3cold" for c in BOARDS["B1"]}         # the board the coder does not use
        return got
    monkeypatch.setattr(supervisor, "spawn_checked", with_the_other_board_dead)
    assert rig.run() == supervisor.EXIT_BLOCKED
    end = decisions(rig, "blocked")[-1]
    assert bus.OFF_BUS not in end["reason"], "a board the coder does not use is not why the coder did not return"
