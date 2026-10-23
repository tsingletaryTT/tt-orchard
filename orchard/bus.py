# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Is every chip the configuration needs on the PCIe bus, awake, readable and bound to a device node? (preflight row `bus`)

On 2026-10-22 both chips of one board went to PCIe power state D3cold while idle ("device inaccessible", "Failed to set
initial power state: -5"). `tt-smi -r` did not bring them back and a reboot was needed. gozer then refused to lease any
chip, healthy ones included, because it could not read the board's serial, and a benchmark session started anyway and
died 70 seconds in with `Read 0xffffffff over PCIe ID 2: the board should be reset` (issue 25).

The row reads what the kernel already shows: the PCI directory (`/sys/bus/pci/devices/<address>/`: vendor, power state,
whether a driver is attached) and the driver's class directory under it (`tenstorrent/tenstorrent!<N>/tt_serial`), plus the
device nodes in `/dev/tenstorrent`. These are ordinary files that anyone can read. Nothing here opens a chip, takes a lease,
runs a command or resets anything, so the row cannot make a sick board worse and cannot collide with another agent's work.

It can only report. A board that has left the bus is not brought back by a reset, so the row names the cause and says what
the operator does (power the box off and on, or reboot) and stops the run before it spends a coder boot on it."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from orchard.preflight import BLOCK, OK, WARN, Check

TENSTORRENT_VENDOR = "0x1e52"
PCI_DEVICES = Path("/sys/bus/pci/devices")
DEV_NODES = Path("/dev/tenstorrent")
TEST_BOARD_CHIPS = 2        # a model's tests take one board; the coder keeps the other (QuietBox 2: two boards of two chips)
FULL_BOX_CHIPS = 4          # a coder on this many chips holds the whole box and is parked for every hardware stage
AWAKE = "D0"
_KMD_DIR = re.compile(r"tenstorrent!(\d+)$")
_GOZER_CHIP = re.compile(r"^\s*chip\s+(\d+)\s+(\S+)", re.MULTILINE)
_GOZER_BOARD = re.compile(r"^board\s+(\S+)", re.MULTILINE)
RESET_DOES_NOT_HELP = ("A board that has left the bus does not come back with `tt-smi -r`: on 2026-10-22 both chips of one "
                       "board went to D3cold while idle, a reset did not wake them and a reboot did. Power the box off and "
                       "on (or reboot), then run again")


@dataclass(frozen=True)
class Chip:
    bdf: str
    power_state: str | None        # D0 is awake; None when the kernel gives no such file
    serial: str | None             # tt_serial, None when it cannot be read
    node: int | None               # N of /dev/tenstorrent/N, from the driver's directory name; None with no driver directory
    driver_bound: bool


@dataclass
class Bus:
    chips: list[Chip] = field(default_factory=list)
    nodes: frozenset = frozenset()
    readable: bool = True          # False when the PCI directory itself could not be listed
    nodes_readable: bool = True    # False when /dev/tenstorrent is missing or cannot be listed: check_bus blocks on that


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def read_bus(pci_root: Path = PCI_DEVICES, dev_root: Path = DEV_NODES) -> Bus:
    """The Tenstorrent chips the kernel shows, and the numbered device nodes. Reads files only."""
    pci_root, dev_root = Path(pci_root), Path(dev_root)
    try:
        dirs = sorted(pci_root.iterdir())
    except OSError:
        return Bus(readable=False)
    chips = []
    for d in dirs:
        vendor = _text(d / "vendor")
        if vendor is None or vendor.lower() != TENSTORRENT_VENDOR:
            continue
        kmd_dirs = sorted((d / "tenstorrent").glob("*")) if (d / "tenstorrent").is_dir() else []
        kmd = kmd_dirs[0] if kmd_dirs else None
        found = _KMD_DIR.search(kmd.name) if kmd else None
        chips.append(Chip(bdf=d.name, power_state=_text(d / "power_state"),
                          serial=_text(kmd / "tt_serial") if kmd else None,
                          node=int(found.group(1)) if found else None,
                          driver_bound=(d / "driver").exists()))
    try:
        nodes = frozenset(int(p.name) for p in dev_root.iterdir() if p.name.isdigit())
        nodes_readable = True
    except OSError:
        nodes, nodes_readable = frozenset(), False
    return Bus(chips=chips, nodes=nodes, nodes_readable=nodes_readable)


def needed_chips(cfg) -> int:
    """How many chips this box must show for the configuration. A lab run's tests run on the lab, so only the coder's chips
    are needed here. Otherwise the coder keeps its chips and a model's tests need a board beside it (a coder that holds the
    whole box is parked instead), and stage 4's largest required chip count can ask for more."""
    chips = int(cfg.coder.chips)
    if getattr(cfg, "mode", "local") == "lab":
        return chips
    need = chips if chips >= FULL_BOX_CHIPS else chips + TEST_BOARD_CHIPS
    for part in str(getattr(cfg, "required_chips", None) or "").split(","):
        if part.strip().isdigit():
            need = max(need, int(part))
    return need


def _bad_serial(serial: str | None) -> bool:
    """No serial, or the all-zero or all-one value a chip returns when the read did not reach it."""
    if serial is None:
        return True
    digits = serial.lower().removeprefix("0x")           # an empty value is a subset of both sets below
    return set(digits) <= {"0"} or set(digits) <= {"f"}


def _gozer_boards(text: str) -> dict[str, list[tuple[str, str]]]:
    """{board serial: [(chip number, PCI address)]} from `gozer status`, in the order it prints them."""
    boards: dict[str, list[tuple[str, str]]] = {}
    current = None
    for line in text.splitlines():
        b = _GOZER_BOARD.match(line)
        if b:
            current = b.group(1)
            boards[current] = []
            continue
        c = _GOZER_CHIP.match(line)
        if c and current is not None:
            boards[current].append((c.group(1), c.group(2)))
    return boards


def check_bus(bus: Bus, gozer_text: str, needed: int) -> Check:
    """`gozer_text` is `gozer status` (empty when it gave none). Blocks as `hardware-unhealthy` with every cause it found."""
    if not bus.readable:
        return Check("bus", WARN, f"could not read the PCIe bus ({PCI_DEVICES}); the chips were not checked")
    blocks: list[str] = []
    warns: list[str] = []
    if len(bus.chips) < needed:
        blocks.append(f"only {len(bus.chips)} of the {needed} chips the configuration needs are on the PCIe bus. "
                      + RESET_DOES_NOT_HELP)
    if not bus.nodes_readable and needed > 0:
        # Without the device directory no chip can be opened, whatever sysfs shows. This is one finding for the box; the
        # per-chip check below needs a readable directory to compare with, so it stays quiet here.
        blocks.append(f"{DEV_NODES} cannot be read (it is missing or cannot be listed), so no chip can be opened. Check that the tenstorrent "
                      "driver is loaded (`lsmod | grep tenstorrent`) and that this user can list the directory")
    for c in bus.chips:
        if c.power_state is None:
            warns.append(f"{c.bdf}: its power state could not be read")
        elif c.power_state != AWAKE:
            note = (" (D3cold: the chip is powered down and inaccessible. " + RESET_DOES_NOT_HELP + ")"
                    if c.power_state == "D3cold" else "")
            blocks.append(f"{c.bdf} is in PCIe power state {c.power_state} (a chip that can be used is in {AWAKE}){note}")
        if not c.driver_bound:
            blocks.append(f"{c.bdf}: no tenstorrent driver is attached")
        elif c.node is None:
            blocks.append(f"{c.bdf}: the driver made no device node entry for it")
        elif bus.nodes_readable and c.node not in bus.nodes:
            blocks.append(f"{c.bdf}: /dev/tenstorrent/{c.node} is missing")
        if c.driver_bound and c.node is not None and _bad_serial(c.serial):
            blocks.append(f"{c.bdf}: its serial cannot be read. gozer reads the board serial to lease a chip and refuses "
                          "every lease, healthy chips included, while one cannot be read")
    # With no /dev/tenstorrent to read, `nodes` is empty and so is this list.
    for n in sorted(bus.nodes - {c.node for c in bus.chips if c.node is not None}):
        warns.append(f"/dev/tenstorrent/{n} has no chip on the PCIe bus behind it")
    boards = _gozer_boards(gozer_text)
    listed = [(n, bdf) for chips in boards.values() for n, bdf in chips]
    on_bus = {c.bdf for c in bus.chips}
    for n, bdf in listed:
        if bdf not in on_bus:
            blocks.append(f"gozer lists chip {n} at {bdf}, but the PCIe bus has no Tenstorrent device there")
    if listed and len(listed) < len(bus.chips):
        warns.append(f"gozer lists {len(listed)} chip{'s' if len(listed) != 1 else ''} and the PCIe bus shows {len(bus.chips)}")
    sizes = {k: len(v) for k, v in boards.items()}
    if len(set(sizes.values())) > 1:
        top = max(sizes.values())
        for k, n in sizes.items():
            if n < top:
                warns.append(f"gozer lists {n} chip{'s' if n != 1 else ''} for board {k} and {top} for another: "
                             "the board looks half present")
    if blocks:
        return Check("bus", BLOCK, "; ".join(blocks + warns), "hardware-unhealthy")
    if warns:
        return Check("bus", WARN, "; ".join(warns))
    return Check("bus", OK, f"{len(bus.chips)} Tenstorrent chips on the PCIe bus, all in {AWAKE} with a readable serial "
                 f"and a device node (the configuration needs {needed})")
