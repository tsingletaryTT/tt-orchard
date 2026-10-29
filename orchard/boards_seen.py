# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Do the tools that list the machine's boards agree with what the configuration expects? (preflight row `boards`, issue 25)

The `bus` row (orchard/bus.py) reads the kernel's own files. This row reads what the tools report: `gozer topology` (the boards
and chips gozer found), `gozer status` and, when a path to tt-smi is configured, a `tt-smi -s` JSON snapshot. It compares them
with the number of chips the configuration needs (two chips make a board) and says plainly when a board is missing, naming the
board's serial and the PCI addresses of its chips when a tool can still name them.

It only reads. It never runs a reset and never takes a lease. `gozer topology` and `gozer status` read gozer's own state.
`tt-smi -s` opens every device on the box, which can collide with another agent's work, so the real signal runs it only when
`ORCHARD_TT_SMI` names the binary (see `orchard.preflight.default_signals`). Every command goes through an injected runner
`run(argv, timeout) -> (returncode, stdout, stderr)`, so tests use fakes and need no hardware."""
from __future__ import annotations

import json
from dataclasses import dataclass

from orchard.bus import TEST_BOARD_CHIPS, _gozer_boards
from orchard.preflight import BLOCK, OK, WARN, Check

TOOL_TIMEOUT_S = 20


@dataclass
class Sources:
    topology: str = ""            # `gozer topology` text, "" when it gave none
    smi: str | None = None        # `tt-smi -s` stdout, None when it was not run or gave nothing
    status: str = ""              # `gozer status` text, "" when it gave none


def _ask(argv, run) -> str | None:
    try:
        code, out, _ = run(argv, TOOL_TIMEOUT_S)
    except Exception:                                 # a missing binary or a timeout: the tool gave no answer
        return None
    return out if code == 0 and out else None


def read_sources(gozer: str, tt_smi: str | None, run) -> Sources:
    """The three readings, each from the injected runner. A command that fails or does not exist gives an empty reading."""
    return Sources(topology=_ask([gozer, "topology"], run) or "",
                   smi=_ask([tt_smi, "-s"], run) if tt_smi else None,
                   status=_ask([gozer, "status"], run) or "")


def _smi_boards(text: str) -> dict[str, list[str]] | None:
    """{board serial: [PCI address]} from a `tt-smi -s` snapshot, or None when the text is not a snapshot."""
    try:
        doc = json.loads(text)
    except ValueError:
        return None
    info = doc.get("device_info") if isinstance(doc, dict) else None
    if not isinstance(info, list):
        return None
    boards: dict[str, list[str]] = {}
    for d in info:
        b = d.get("board_info") if isinstance(d, dict) else None
        if isinstance(b, dict) and isinstance(b.get("bus_id"), str):
            boards.setdefault(str(b.get("board_id") or "unknown board"), []).append(b["bus_id"])
    return boards


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def check_boards(topology: str, smi: str | None, status: str, needed: int) -> Check:
    """Blocks as `hardware-unhealthy` when the boards the tools list are fewer than the configuration needs, or when tt-smi
    cannot see a chip gozer lists. Warns when the tools disagree in a way that does not stop a run."""
    topo = _gozer_boards(topology)
    stat = _gozer_boards(status)
    snap = _smi_boards(smi) if smi is not None else None
    blocks: list[str] = []
    warns: list[str] = []
    if smi is not None and snap is None:
        warns.append("the tt-smi -s output was not JSON, so tt-smi was not compared")
    if not topo and snap is None:
        return Check("boards", WARN, "gozer topology and tt-smi gave no board list; the boards were not checked. "
                     + ("; ".join(warns) if warns else ""))
    chips = {s: [bdf for _, bdf in c] for s, c in topo.items()} if topo else snap
    source = "gozer topology" if topo else "tt-smi"
    have = sum(len(v) for v in chips.values())
    boards_needed = -(-needed // TEST_BOARD_CHIPS)
    if have < needed:
        blocks.append(f"a board is missing: {source} lists {_plural(have, 'chip')} on {_plural(len(chips), 'board')} and the "
                      f"configuration needs {_plural(needed, 'chip')} ({_plural(boards_needed, 'board')}). A board that has left "
                      "the bus does not come back with `tt-smi -r`; power the box off and on, or reboot")
    if topo and snap is not None:
        seen = {bdf for v in snap.values() for bdf in v}
        for serial, pairs in topo.items():
            lost = [(n, bdf) for n, bdf in pairs if bdf not in seen]
            if lost:
                where = ", ".join(f"chip {n} at {bdf}" for n, bdf in lost)
                blocks.append(f"tt-smi cannot see {where} of board {serial}, which gozer topology lists")
        known = {bdf for v in chips.values() for bdf in v}
        for serial, bdfs in snap.items():
            extra = [b for b in bdfs if b not in known]
            if extra:
                warns.append(f"tt-smi lists board {serial} ({', '.join(extra)}) and gozer topology does not")
    if topo and stat:
        for serial in topo:
            if serial not in stat:
                warns.append(f"gozer topology lists board {serial} and gozer status does not")
    if blocks:
        return Check("boards", BLOCK, "; ".join(blocks + warns), "hardware-unhealthy")
    if warns:
        return Check("boards", WARN, "; ".join(warns))
    used = ["gozer topology" if topo else None, "gozer status" if stat else None, "tt-smi" if snap is not None else None]
    return Check("boards", OK, f"{_plural(have, 'chip')} on {_plural(len(chips), 'board')}, as listed by "
                 f"{', '.join(u for u in used if u)} (the configuration needs {needed})")
