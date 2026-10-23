# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `bus` row of the bringup preflight: is every chip the configuration needs on the PCIe bus, awake, readable and
bound to a device node? (issue 25)

On 2026-10-22 both chips of one board went to PCIe power state D3cold while idle. `tt-smi -r` did not bring them back
and a reboot was needed. gozer then refused to lease any chip, healthy ones included, because it could not read the
board's serial, and a benchmark session started anyway and died 70 seconds in with `Read 0xffffffff over PCIe ID 2`.
The row reads what the kernel already shows (world-readable sysfs files and the device nodes), so it needs no lease and
never opens a chip. Every test builds a fake sysfs tree in a temporary directory: nothing here touches hardware."""
from __future__ import annotations

from pathlib import Path

import pytest

from orchard import bus
from orchard import preflight as pf
from test_preflight import FREE, cfg, signals

SERIAL = "0x0000046131924062"
BDFS = ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]


def add_chip(root: Path, bdf: str, *, node: int, vendor: str = "0x1e52", state: str | None = "D0",
             serial: str | None = SERIAL, driver: bool = True, kmd: bool = True) -> Path:
    d = root / bdf
    d.mkdir(parents=True)
    (d / "vendor").write_text(vendor + "\n")
    if state is not None:
        (d / "power_state").write_text(state + "\n")
    if driver:
        (root.parent / "drivers" / "tenstorrent").mkdir(parents=True, exist_ok=True)
        (d / "driver").symlink_to(root.parent / "drivers" / "tenstorrent")
    if kmd:
        k = d / "tenstorrent" / f"tenstorrent!{node}"
        k.mkdir(parents=True)
        if serial is not None:
            (k / "tt_serial").write_text(serial + "\n")
    return d


def add_nodes(dev_root: Path, nodes) -> Path:
    dev_root.mkdir(parents=True, exist_ok=True)
    for n in nodes:
        (dev_root / str(n)).write_text("")
    (dev_root / "by-id").mkdir(exist_ok=True)
    return dev_root


def box(tmp_path: Path, chips: int = 4, **over) -> tuple[Path, Path]:
    pci, dev = tmp_path / "pci", tmp_path / "dev"
    for i in range(chips):
        add_chip(pci, BDFS[i], node=i, **over)
    add_nodes(dev, range(chips))
    return pci, dev


def read(tmp_path, chips=4, **over) -> bus.Bus:
    pci, dev = box(tmp_path, chips, **over)
    return bus.read_bus(pci, dev)


# ---- reading the bus ---------------------------------------------------------------------------

def test_only_tenstorrent_devices_are_chips(tmp_path):
    pci, dev = box(tmp_path, 2)
    other = pci / "0000:00:1f.0"
    other.mkdir()
    (other / "vendor").write_text("0x8086\n")
    got = bus.read_bus(pci, dev)
    assert [c.bdf for c in got.chips] == BDFS[:2]


def test_a_chip_reading_gives_its_state_serial_node_and_driver(tmp_path):
    got = read(tmp_path, 2)
    c = got.chips[1]
    assert (c.bdf, c.power_state, c.serial, c.node, c.driver_bound) == (BDFS[1], "D0", SERIAL, 1, True)


def test_the_node_number_comes_from_the_kernel_directory_name(tmp_path):
    pci, dev = tmp_path / "pci", tmp_path / "dev"
    add_chip(pci, BDFS[0], node=7)
    add_nodes(dev, [7])
    assert bus.read_bus(pci, dev).chips[0].node == 7


def test_device_nodes_are_the_numbered_entries_and_nothing_else(tmp_path):
    pci, dev = box(tmp_path, 2)
    assert bus.read_bus(pci, dev).nodes == frozenset({0, 1})


def test_a_missing_power_state_or_serial_is_none_not_a_guess(tmp_path):
    pci, dev = tmp_path / "pci", tmp_path / "dev"
    add_chip(pci, BDFS[0], node=0, state=None, serial=None)
    add_nodes(dev, [0])
    c = bus.read_bus(pci, dev).chips[0]
    assert c.power_state is None and c.serial is None


def test_a_chip_with_no_driver_or_no_kernel_directory_says_so(tmp_path):
    pci, dev = tmp_path / "pci", tmp_path / "dev"
    add_chip(pci, BDFS[0], node=0, driver=False, kmd=False)
    add_nodes(dev, [])
    c = bus.read_bus(pci, dev).chips[0]
    assert c.driver_bound is False and c.node is None


def test_no_pci_directory_means_the_bus_could_not_be_read(tmp_path):
    got = bus.read_bus(tmp_path / "nope", tmp_path / "dev")
    assert got.readable is False and got.chips == []


def test_a_box_with_no_device_directory_still_reads_its_chips(tmp_path):
    pci, _ = box(tmp_path, 2)
    got = bus.read_bus(pci, tmp_path / "nodev")
    assert len(got.chips) == 2 and got.nodes == frozenset()


def test_the_default_roots_are_the_kernels_own(tmp_path):
    import inspect
    sig = inspect.signature(bus.read_bus).parameters
    assert str(sig["pci_root"].default) == "/sys/bus/pci/devices"
    assert str(sig["dev_root"].default) == "/dev/tenstorrent"
    assert bus.TENSTORRENT_VENDOR == "0x1e52"


# ---- the row -----------------------------------------------------------------------------------

def check(tmp_path, needed=4, gozer=FREE, chips=4, **over):
    return bus.check_bus(read(tmp_path, chips, **over), gozer, needed)


def test_four_awake_readable_chips_with_nodes_pass(tmp_path):
    c = check(tmp_path)
    assert (c.name, c.status) == ("bus", "ok")
    assert "4" in c.detail and "D0" in c.detail


def test_fewer_chips_than_the_configuration_needs_blocks_with_the_counts_and_the_next_step(tmp_path):
    c = check(tmp_path, needed=4, chips=2, gozer=FREE)
    assert (c.status, c.reason) == ("block", "hardware-unhealthy")
    assert "2 of the 4" in c.detail
    assert "reboot" in c.detail or "power" in c.detail


def test_more_chips_than_needed_is_fine(tmp_path):
    assert check(tmp_path, needed=2).status == "ok"


@pytest.mark.parametrize("state", ["D3cold", "D3hot", "D1"])
def test_a_chip_that_is_not_awake_blocks_and_names_the_chip_and_state(tmp_path, state):
    pci, dev = box(tmp_path)
    (pci / BDFS[2] / "power_state").write_text(state + "\n")
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert (c.status, c.reason) == ("block", "hardware-unhealthy")
    assert BDFS[2] in c.detail and state in c.detail
    assert BDFS[0] not in c.detail


def test_d3cold_gets_the_incident_text_that_a_reset_did_not_help(tmp_path):
    pci, dev = box(tmp_path)
    (pci / BDFS[0] / "power_state").write_text("D3cold\n")
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert "tt-smi -r" in c.detail and "reboot" in c.detail


def test_an_unreadable_power_state_is_a_warning_because_old_kernels_have_none(tmp_path):
    pci, dev = tmp_path / "pci", tmp_path / "dev"
    for i in range(4):
        add_chip(pci, BDFS[i], node=i, state=None)
    add_nodes(dev, range(4))
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert c.status == "warn" and BDFS[0] in c.detail


@pytest.mark.parametrize("serial", [None, "", "0x", "0x0", "0x0000000000000000", "0xffffffffffffffff"])
def test_a_chip_whose_serial_cannot_be_read_blocks_because_gozer_will_refuse_to_lease(tmp_path, serial):
    pci, dev = box(tmp_path)
    k = pci / BDFS[1] / "tenstorrent" / "tenstorrent!1" / "tt_serial"
    if serial is None:
        k.unlink()
    else:
        k.write_text(serial + "\n")
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert (c.status, c.reason) == ("block", "hardware-unhealthy")
    assert BDFS[1] in c.detail and "serial" in c.detail and "gozer" in c.detail


def test_a_chip_with_no_device_node_blocks(tmp_path):
    pci, dev = box(tmp_path)
    (dev / "3").unlink()
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert c.status == "block" and BDFS[3] in c.detail and "/dev/tenstorrent/3" in c.detail


def test_a_node_with_no_chip_behind_it_is_a_warning(tmp_path):
    pci, dev = box(tmp_path)
    (dev / "9").write_text("")
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert c.status == "warn" and "9" in c.detail


def test_a_chip_with_no_driver_blocks(tmp_path):
    pci, dev = tmp_path / "pci", tmp_path / "dev"
    for i in range(4):
        add_chip(pci, BDFS[i], node=i, driver=(i != 2), kmd=(i != 2))
    add_nodes(dev, [0, 1, 3])
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert c.status == "block" and BDFS[2] in c.detail and "driver" in c.detail


def test_a_chip_gozer_lists_that_the_bus_does_not_have_blocks(tmp_path):
    c = check(tmp_path, needed=2, chips=2, gozer=FREE)       # the bus has 2 chips; gozer lists 4
    assert c.status == "block" and BDFS[2] in c.detail and "gozer" in c.detail
    assert "gozer lists" in c.detail


def test_gozer_listing_fewer_chips_than_the_bus_is_a_warning(tmp_path):
    first_board = "\n".join(FREE.splitlines()[:4])           # one complete board: only the count differs from the bus
    c = check(tmp_path, needed=4, chips=4, gozer=first_board)
    assert c.status == "warn" and "gozer lists 2 chips and the PCIe bus shows 4" in c.detail
    assert "half" not in c.detail


def test_a_board_listing_fewer_chips_than_the_others_is_a_warning(tmp_path):
    half = "\n".join(line for line in FREE.splitlines() if "0000:04" not in line)
    c = check(tmp_path, needed=3, chips=4, gozer=half)
    assert c.status == "warn" and "0000046131924055" in c.detail and "half" in c.detail
    assert "0000046131924062" not in c.detail                 # the complete board is not blamed


def test_a_block_still_lists_the_warnings_beside_it(tmp_path):
    pci, dev = box(tmp_path)
    (pci / BDFS[0] / "power_state").write_text("D3cold\n")
    (dev / "9").write_text("")
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert c.status == "block" and BDFS[0] in c.detail and "/dev/tenstorrent/9" in c.detail


def test_no_gozer_text_leaves_the_bus_findings_alone(tmp_path):
    assert check(tmp_path, gozer="").status == "ok"


def test_a_bus_that_cannot_be_read_is_a_warning_and_never_a_block(tmp_path):
    c = bus.check_bus(bus.read_bus(tmp_path / "nope", tmp_path / "dev"), FREE, 4)
    assert (c.name, c.status) == ("bus", "warn") and "could not read" in c.detail


def test_every_problem_is_listed_not_only_the_first(tmp_path):
    pci, dev = box(tmp_path)
    (pci / BDFS[0] / "power_state").write_text("D3cold\n")
    (dev / "3").unlink()
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert BDFS[0] in c.detail and BDFS[3] in c.detail


def test_a_box_with_no_device_directory_is_not_read_as_every_node_missing(tmp_path):
    pci, _ = box(tmp_path, 4)
    c = bus.check_bus(bus.read_bus(pci, tmp_path / "nodev"), FREE, 4)
    assert c.status == "ok"


def test_a_chip_with_no_driver_is_not_also_blamed_for_its_serial(tmp_path):
    pci, dev = tmp_path / "pci", tmp_path / "dev"
    for i in range(4):
        add_chip(pci, BDFS[i], node=i, driver=(i != 2), kmd=(i != 2))
    add_nodes(dev, [0, 1, 3])
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert "serial" not in c.detail and "device node entry" not in c.detail


def test_a_chip_with_a_driver_but_no_kernel_directory_says_the_driver_made_no_node_entry(tmp_path):
    pci, dev = tmp_path / "pci", tmp_path / "dev"
    for i in range(4):
        add_chip(pci, BDFS[i], node=i, kmd=(i != 1))
    add_nodes(dev, [0, 2, 3])
    c = bus.check_bus(bus.read_bus(pci, dev), FREE, 4)
    assert c.status == "block" and BDFS[1] in c.detail and "no device node entry" in c.detail


def test_the_check_reads_files_and_never_runs_anything(tmp_path, monkeypatch):
    import subprocess
    def boom(*a, **k):
        raise AssertionError("the bus row must not run a command")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    assert check(tmp_path).status == "ok"


# ---- how many chips the configuration needs -----------------------------------------------------

def test_the_coder_plus_a_test_board_is_what_a_two_chip_coder_needs(tmp_path):
    c = cfg(tmp_path)
    c.coder.chips = 2
    assert bus.needed_chips(c) == 4


def test_a_coder_on_the_whole_box_needs_the_whole_box(tmp_path):
    assert bus.needed_chips(cfg(tmp_path)) == 4               # the helper's coder holds 4 chips


def test_the_largest_required_configuration_counts(tmp_path):
    c = cfg(tmp_path)
    c.coder.chips = 2
    c.required_chips = "2,4"
    assert bus.needed_chips(c) == 4
    c.required_chips = "1,2"
    assert bus.needed_chips(c) == 4                           # still the coder and a test board


def test_a_required_count_above_the_coder_plus_board_raises_the_need(tmp_path):
    c = cfg(tmp_path)
    c.coder.chips = 1
    c.required_chips = "4"
    assert bus.needed_chips(c) == 4
    c.required_chips = "8"
    assert bus.needed_chips(c) == 8


def test_a_required_count_that_is_not_a_number_is_ignored(tmp_path):
    c = cfg(tmp_path)
    c.coder.chips = 2
    c.required_chips = "2,abc"
    assert bus.needed_chips(c) == 4


def test_a_lab_run_needs_only_the_coders_chips_here(tmp_path):
    from orchard.bringup_config import Lab
    c = cfg(tmp_path)
    c.coder.chips = 2
    c.lab, c.mode = Lab(host="node4", root=Path("/srv/orchard")), "lab"
    assert bus.needed_chips(c) == 2


# ---- in the preflight ---------------------------------------------------------------------------

def healthy(tmp_path, chips=4):
    pci, dev = box(tmp_path, chips)
    return lambda: bus.read_bus(pci, dev)


def run(tmp_path, config=None, **over):
    return pf.run_preflight(config or cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                            signals=signals(**over))


def by(out):
    return {c.name: c for c in out}


def test_the_bus_row_sits_right_after_gozer_and_only_when_the_signal_exists(tmp_path):
    names = [c.name for c in run(tmp_path, bus=healthy(tmp_path))]
    assert names == ["hub", "disk", "credentials", "tiers", "port", "gozer", "bus", "base", "reference"]
    assert "bus" not in [c.name for c in run(tmp_path)]


def test_the_row_judges_the_bus_against_what_the_configuration_needs(tmp_path):
    out = run(tmp_path, bus=healthy(tmp_path, chips=2))                # cfg's coder holds 4 chips
    row = by(out)["bus"]
    assert (row.status, row.reason) == ("block", "hardware-unhealthy") and "2 of the 4" in row.detail
    assert pf.blocked(out) == [row]


def test_the_row_reads_the_same_gozer_text_once(tmp_path):
    calls = []
    run(tmp_path, bus=healthy(tmp_path), gozer_status=lambda: calls.append(1) or FREE)
    assert calls == [1]


def test_the_row_compares_with_gozers_chips(tmp_path):
    gone = FREE.replace("0000:04:00.0", "0000:09:00.0")
    row = by(run(tmp_path, bus=healthy(tmp_path), gozer_status=lambda: gone))["bus"]
    assert row.status == "block" and "0000:09:00.0" in row.detail


def test_a_gozer_that_fails_leaves_the_bus_row_standing_alone(tmp_path):
    def boom():
        raise OSError("gozer gone")
    out = by(run(tmp_path, bus=healthy(tmp_path), gozer_status=boom))
    assert out["gozer"].status == "block" and out["bus"].status == "ok"


def test_a_bus_signal_that_fails_is_a_block_with_the_hardware_reason(tmp_path):
    def boom():
        raise OSError("sysfs gone")
    row = by(run(tmp_path, bus=boom))["bus"]
    assert (row.status, row.reason) == ("block", "hardware-unhealthy") and "sysfs gone" in row.detail


def test_a_lab_run_needs_only_the_coders_chips_on_this_box(tmp_path):
    from test_preflight import lab_cfg
    c = lab_cfg(tmp_path)
    c.coder.chips = 2
    one_board = "\n".join(FREE.splitlines()[:4])             # this box has one board; the tests run on the lab
    out = run(tmp_path, config=c, bus=healthy(tmp_path / "lab", chips=2), gozer_status=lambda: one_board,
              lab_status=lambda: (True, FREE, ""))
    assert by(out)["bus"].status == "ok"
    local = cfg(tmp_path)
    local.coder.chips = 2                                     # the same box without a lab needs a test board as well
    assert by(run(tmp_path, config=local, bus=healthy(tmp_path / "local", chips=2), gozer_status=lambda: one_board))["bus"].status == "block"


def test_the_default_signals_read_the_real_bus(tmp_path, monkeypatch):
    marker = bus.Bus(chips=[], readable=False)
    monkeypatch.setattr(bus, "read_bus", lambda *a, **k: marker)
    s = pf.default_signals(cfg(tmp_path))
    assert s.bus is not None and s.bus() is marker


def test_the_ui_shows_the_bus_row_and_refuses_a_launch_while_it_blocks(tmp_path, monkeypatch):
    from orchard import cli
    from test_webui import make_app
    pci, dev = box(tmp_path, 4)
    (pci / BDFS[1] / "power_state").write_text("D3cold\n")
    monkeypatch.setattr(cli.preflight, "default_signals",
                        lambda c, research_options=None: signals(bus=lambda: bus.read_bus(pci, dev)))
    rows = cli._ui_preflight(cfg(tmp_path))("Cloudflare/clef", None)
    root = tmp_path / "orchard-runs"
    root.mkdir()
    app = make_app(root, preflight=lambda m, b: rows)
    shown = app.preflight("Cloudflare/clef")
    row = next(r for r in shown["checks"] if r["name"] == "bus")
    assert shown["ok"] is False and row["status"] == "block" and BDFS[1] in row["detail"]
    from orchard.webui import Conflict
    with pytest.raises(Conflict, match=r"bus \(hardware-unhealthy\)"):
        app.bringup("Cloudflare/clef")


# ---- what the operator is told -------------------------------------------------------------------

def test_the_unblock_steps_for_hardware_say_what_to_do_when_the_bus_row_blocks():
    from orchard import narrate
    line = next((x for x in narrate.UNBLOCK["hardware-unhealthy"] if "`bus`" in x), "")
    assert "PCIe" in line and "D3cold" in line and "reboot" in line and "tt-smi -r" in line and "gozer status" in line


def test_the_readme_documents_the_bus_row_between_gozer_and_base():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    rows = [line for line in readme.splitlines() if line.startswith("| ")]
    names = [line.split("|")[1].strip() for line in rows]
    assert names.index("gozer") < names.index("bus") < names.index("base")
    row = rows[names.index("bus")]
    assert "D3cold" in row and "`hardware-unhealthy`" in row and "reads files" in row
