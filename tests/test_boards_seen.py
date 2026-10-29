# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `boards` preflight row: do the tools that list the machine's boards agree with what the configuration expects? (issue 25)

The row compares `gozer topology`, `gozer status` and a `tt-smi -s` snapshot with the number of chips the configuration needs,
and names a board that has dropped off. It only reads. Every input is a fake in these tests, so no hardware is needed."""
from __future__ import annotations

import json

from orchard import boards_seen as bs
from orchard import preflight as pf
from test_preflight import cfg, signals


def topo(boards):
    """gozer topology text for {serial: [(chip, bdf)]}."""
    lines = ["grain: board"]
    for serial, chips in boards.items():
        lines.append(f"board {serial}  (p300c)")
        lines += [f"  chip {n}  {bdf}  asic AAAA{n}" for n, bdf in chips]
    return "\n".join(lines) + "\n"


def smi(boards):
    """tt-smi -s JSON text for {serial: [bdf]}."""
    return json.dumps({"time": "t", "device_info": [
        {"board_info": {"bus_id": bdf, "board_type": "p300c", "board_id": serial}, "telemetry": {}}
        for serial, bdfs in boards.items() for bdf in bdfs]})


B1, B2 = "0000046131924062", "0000046131924055"
FULL = {B1: [(0, "0000:01:00.0"), (1, "0000:02:00.0")], B2: [(2, "0000:03:00.0"), (3, "0000:04:00.0")]}
FULL_SMI = {B1: ["0000:01:00.0", "0000:02:00.0"], B2: ["0000:03:00.0", "0000:04:00.0"]}
HALF = {B1: FULL[B1]}
HALF_SMI = {B1: FULL_SMI[B1]}


def test_every_source_agrees_with_the_configuration():
    c = bs.check_boards(topo(FULL), smi(FULL_SMI), "", needed=4)
    assert c.status == pf.OK and "4 chips on 2 boards" in c.detail


def test_a_board_missing_from_the_topology_is_named_and_blocks():
    c = bs.check_boards(topo(HALF), smi(HALF_SMI), "", needed=4)
    assert c.status == pf.BLOCK and c.reason == "hardware-unhealthy"
    assert "2 chips on 1 board" in c.detail and "needs 4 chips (2 boards)" in c.detail
    assert "a board is missing" in c.detail


def test_a_board_that_tt_smi_cannot_see_is_named_by_serial():
    c = bs.check_boards(topo(FULL), smi(HALF_SMI), "", needed=4)
    assert c.status == pf.BLOCK
    assert B2 in c.detail and "0000:03:00.0" in c.detail and "tt-smi" in c.detail


def test_a_board_that_tt_smi_lists_and_gozer_does_not_is_a_warning():
    c = bs.check_boards(topo(HALF), smi(FULL_SMI), "", needed=2)
    assert c.status == pf.WARN and B2 in c.detail and "gozer" in c.detail


def test_status_that_lacks_a_topology_board_is_a_warning():
    c = bs.check_boards(topo(FULL), smi(FULL_SMI), topo(HALF), needed=4)
    assert c.status == pf.WARN and B2 in c.detail


def test_no_source_answering_is_a_warning_that_says_nothing_was_checked():
    c = bs.check_boards("", None, "", needed=4)
    assert c.status == pf.WARN and "not checked" in c.detail


def test_a_tt_smi_snapshot_that_is_not_json_is_reported_and_topology_still_counts():
    c = bs.check_boards(topo(FULL), "Traceback: no devices", "", needed=4)
    assert c.status == pf.WARN and "tt-smi" in c.detail and "not JSON" in c.detail


def test_without_a_tt_smi_run_the_topology_alone_decides():
    assert bs.check_boards(topo(FULL), None, "", needed=4).status == pf.OK
    assert bs.check_boards(topo(HALF), None, "", needed=4).status == pf.BLOCK


def test_the_row_never_runs_a_reset_and_the_runner_is_injected():
    seen = []

    def run(argv, timeout):
        seen.append(argv)
        return 0, smi(FULL_SMI) if argv[0] == "tt-smi" else topo(FULL), ""
    got = bs.read_sources("gozer", "tt-smi", run)
    assert got.topology and got.smi and seen == [["gozer", "topology"], ["tt-smi", "-s"], ["gozer", "status"]]
    assert not any("reset" in " ".join(a) or "-r" in a for a in seen)


def test_a_runner_that_fails_gives_none_and_never_raises():
    def run(argv, timeout):
        raise OSError("no such file")
    got = bs.read_sources("gozer", "tt-smi", run)
    assert got.topology == "" and got.smi is None


def test_run_preflight_adds_the_row_and_a_missing_board_blocks(tmp_path):
    sig = signals(boards=lambda: bs.Sources(topology=topo(HALF), smi=smi(HALF_SMI), status=""))
    rows = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=sig)
    row = [r for r in rows if r.name == "boards"][0]
    assert row.status == pf.BLOCK and row.reason == "hardware-unhealthy"


def test_run_preflight_leaves_the_row_out_without_the_signal(tmp_path):
    rows = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert all(r.name != "boards" for r in rows)


def test_tt_smi_is_not_run_when_no_binary_is_named():
    seen = []

    def run(argv, timeout):
        seen.append(argv[0])
        return 0, topo(FULL), ""
    got = bs.read_sources("gozer", None, run)
    assert got.smi is None and seen == ["gozer", "gozer"]
