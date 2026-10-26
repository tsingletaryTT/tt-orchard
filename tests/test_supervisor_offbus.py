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
from run_fakes import MachineCoder, BOARDS, FAILING_TEST_SCRIPTS, SWAP_FAILED, hw_failing, test_fails_until_escalated  # noqa: F401
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


def test_a_lab_run_never_reads_this_boxs_bus():
    class Boom:
        def __call__(self):
            raise AssertionError("the lab's chips are not on this bus")

    class Fake:
        lab, bus_read, sleep, clock, coder_chips = object(), Boom(), None, None, 2
    assert supervisor.Supervisor._off_bus(Fake(), ["x"]) is None


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
