# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The runtime half of issue 25: after a run has started, a failed hardware test, coder boot or benchmark phase is
checked against the PCIe bus, so a board that left the bus is reported as `hardware-unhealthy` and is not retried.

Every test reads a fake sysfs tree in a temporary directory, or a reader that returns canned `Bus` values. Nothing opens
a chip or touches /dev/tenstorrent. The sleep and the clock are fakes, so the settle window costs no time."""
from __future__ import annotations

import pytest

from orchard import blocked, bus
from test_bus_preflight import BDFS, SERIAL, box


class Time:
    """A clock and a sleep that move together."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def run(read, **kw):
    t = Time()
    kw.setdefault("settle_s", 30.0)
    kw.setdefault("poll_s", 5.0)
    return bus.diagnose(read=read, sleep=t.sleep, clock=t.clock, **kw), t


def test_a_healthy_bus_is_no_finding_and_is_read_once(tmp_path):
    pci, dev = box(tmp_path, 4)
    reads = []
    msg, t = run(lambda: reads.append(1) or bus.read_bus(pci, dev), held=BDFS[:2])
    assert msg is None
    assert len(reads) == 1 and t.sleeps == []


def test_a_held_chip_in_d3cold_is_reported_with_its_address(tmp_path):
    pci, dev = box(tmp_path, 4)
    (pci / BDFS[1] / "power_state").write_text("D3cold\n")
    msg, _ = run(lambda: bus.read_bus(pci, dev), held=BDFS[:2])
    assert msg.startswith(bus.OFF_BUS)
    assert BDFS[1] in msg and "D3cold" in msg
    assert BDFS[0] not in msg


def test_the_message_is_classed_hardware_unhealthy_and_is_not_retried(tmp_path):
    pci, dev = box(tmp_path, 4)
    (pci / BDFS[0] / "power_state").write_text("D3cold\n")
    msg, _ = run(lambda: bus.read_bus(pci, dev), held=BDFS[:2])
    pause = {"decision": "pause", "reason": f"blocked: {msg}"}
    assert blocked.block_code(pause) == "hardware-unhealthy"
    assert blocked.retryable("hardware-unhealthy", msg) is False


def test_other_hardware_blocks_and_codes_are_still_retried():
    assert blocked.retryable("hardware-unhealthy", "resetting the chips failed: device still open") is True
    assert blocked.retryable("coder-unusable", "the coder did not start") is True
    assert blocked.retryable("agent-stuck", "no file written") is True
    assert blocked.retryable("disk-full", "needs 110 GB") is False


def test_a_serial_that_clears_inside_the_settle_window_is_not_a_finding(tmp_path):
    pci, dev = box(tmp_path, 2)
    serial = pci / BDFS[0] / "tenstorrent" / "tenstorrent!0" / "tt_serial"
    serial.write_text("0x0000000000000000\n")                # the board being reset, right after a release
    t = Time()

    def read():
        if t.now >= 15:                                      # the reset ends 15 s in
            serial.write_text(SERIAL + "\n")
        return bus.read_bus(pci, dev)
    assert bus.diagnose(held=BDFS[:2], read=read, sleep=t.sleep, clock=t.clock, settle_s=30.0, poll_s=5.0) is None
    assert t.now == pytest.approx(15.0)


def test_a_serial_that_never_clears_is_a_finding_after_the_window(tmp_path):
    pci, dev = box(tmp_path, 2)
    (pci / BDFS[1] / "tenstorrent" / "tenstorrent!1" / "tt_serial").write_text("0xffffffffffffffff\n")
    reads = []

    def read():
        reads.append(1)
        return bus.read_bus(pci, dev)
    t = Time()
    msg = bus.diagnose(held=BDFS[:2], read=read, sleep=t.sleep, clock=t.clock, settle_s=30.0, poll_s=5.0)
    assert msg and BDFS[1] in msg and "serial" in msg
    assert 30.0 <= t.now <= 35.0, "it waited the settle window and no longer"
    assert len(reads) == 7                                   # at 0, 5, ..., 30 s


def test_a_held_chip_that_is_missing_from_the_bus_is_named(tmp_path):
    pci, dev = box(tmp_path, 4)
    msg, _ = run(lambda: bus.read_bus(pci, dev), held=BDFS[:2] + ["0000:09:00.0"])
    assert "0000:09:00.0" in msg and "not on the PCIe bus" in msg


def test_a_bad_chip_the_run_does_not_hold_is_ignored(tmp_path):
    pci, dev = box(tmp_path, 4)
    (pci / BDFS[3] / "power_state").write_text("D3cold\n")
    msg, t = run(lambda: bus.read_bus(pci, dev), held=BDFS[:2])
    assert msg is None and t.sleeps == [], "a board the run does not use is not this run's problem"


def test_a_held_chip_with_no_device_node_is_a_finding(tmp_path):
    pci, dev = box(tmp_path, 2)
    (dev / "1").unlink()
    msg, _ = run(lambda: bus.read_bus(pci, dev), held=BDFS[:2])
    assert BDFS[1] in msg and "/dev/tenstorrent/1" in msg


def test_without_held_chips_the_count_decides(tmp_path):
    pci, dev = box(tmp_path, 2)
    msg, _ = run(lambda: bus.read_bus(pci, dev), needed=4)
    assert msg and "2 of the 4" in msg
    ok, _ = run(lambda: bus.read_bus(pci, dev), needed=2)
    assert ok is None


def test_a_bus_that_cannot_be_listed_gives_no_finding(tmp_path):
    msg, t = run(lambda: bus.read_bus(tmp_path / "nowhere", tmp_path / "dev"), held=BDFS[:2])
    assert msg is None and t.sleeps == [], "the check cannot tell, so the original failure stands"


def test_a_missing_device_directory_is_a_finding(tmp_path):
    pci, dev = box(tmp_path, 2)
    for p in dev.iterdir():
        p.unlink() if p.is_file() else p.rmdir()
    dev.rmdir()
    msg, _ = run(lambda: bus.read_bus(pci, dev), held=BDFS[:2])
    assert msg and "/dev/tenstorrent" in msg


def test_a_power_state_that_is_not_reported_is_not_a_finding(tmp_path):
    pci, dev = box(tmp_path, 2, state=None)
    msg, _ = run(lambda: bus.read_bus(pci, dev), held=BDFS[:2])
    assert msg is None


def test_the_settle_window_is_a_stated_choice_that_covers_a_reset():
    assert bus.SETTLE_S >= 60, "a reset's unreadable serial clears in under a minute"
    assert 0 < bus.POLL_S < bus.SETTLE_S


def test_the_phrase_blocked_looks_for_is_the_phrase_the_bus_check_writes():
    assert blocked.OFF_BUS_MARK in bus.OFF_BUS


def test_the_unblock_steps_and_the_readme_name_the_runtime_phrase():
    from pathlib import Path
    from orchard import narrate
    steps = " ".join(narrate.how_to_unblock("hardware-unhealthy", "x/y"))
    assert bus.OFF_BUS in steps and "does not retry" in steps
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert bus.OFF_BUS in readme and "never\nretried" in readme
