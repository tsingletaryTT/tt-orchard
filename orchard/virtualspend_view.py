# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the web page shows about the virtual local spend (orchard/virtualspend.py, README 5.9g).

Read only. The facts are built from the agent logs under the runs folder (each run, and each kernel-probe output under
`<runs>/kprobe`). Nothing here returns a path, opens a chip, writes a file or sends a request. The figure is virtual and is
kept apart from the real helper spend: no field here is added to it, and the meter text says "virtual".

`GET /api/virtual-spend` answers with every listed row and the total for what is shown: the selected run (`?run=NAME`, which
must be the name of a listed row) or, with no run, every row. Scans are cached for CACHE_S seconds because the page asks every
few seconds and a scan reads the log files.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from orchard import virtualspend as vs

CACHE_S = 30.0
MAX_CANDIDATES = 120                 # directories scanned for the table and the all-runs total
MAX_NAMES = 5000                     # directories listed (a name is cheap; a scan is not)
MAX_ROWS = 60                        # rows returned
NAME_MAX = 80
PROBE_PARENT = "kprobe"              # kernel-probe outputs live in <runs>/kprobe/<name>


def usd(value) -> str:
    """Dollars with enough digits that a small cost never shows as zero (the same rule as telegraph_view.usd)."""
    value = float(value or 0)
    if value == 0:
        return "$0.00"
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir() and not p.is_symlink()
    except OSError:
        return False


def candidates(runs_root: Path) -> list[tuple[str, Path]]:
    """(row name, directory) for each run folder and each kernel-probe output. Links are left out."""
    out: list[tuple[str, Path]] = []
    root = Path(runs_root)
    if not _is_dir(root):
        return out
    try:
        children = sorted(root.iterdir())
    except OSError:
        return out
    for child in children:
        if not _is_dir(child):
            continue
        if child.name != PROBE_PARENT:                               # the kernel-probe folder is a container: its outputs are the rows
            out.append((child.name, child))
        if child.name == PROBE_PARENT:
            try:
                inner = sorted(child.iterdir())
            except OSError:
                continue
            out.extend((f"{PROBE_PARENT}/{p.name}", p) for p in inner if _is_dir(p))
    return out[:MAX_NAMES]


class Spender:
    """Scans the first MAX_CANDIDATES directories once per CACHE_S seconds and answers requests from that. A selected run that is
    among the listed directories but past the cap is scanned on its own (and cached on its own), so its figure is never lost to the
    cap. The selected name is only compared with the listed names; it is never turned into a path."""

    def __init__(self, runs_root, cfg: vs.Config, *, redact=lambda t: t, clock=time.time):
        self.runs_root, self.cfg, self.redact, self.clock = Path(runs_root), cfg, redact, clock
        self._lock = threading.Lock()
        self._cache = None                       # (time, rows, {name: path} of every listed directory, count scanned)
        self._extra: dict = {}                   # name -> (time, row or None)

    def _scan_one(self, name, path):
        try:
            return {"name": name, **vs.priced(vs.scan(path), self.cfg)}
        except Exception:                                           # noqa: BLE001 - one bad folder must not hide the others
            return None

    def _rows(self):
        """(rows, listed names, number scanned). One refresh runs at a time: a request that arrives during a scan waits for it and then
        reuses its result. The cache is dated when the scan ends, so a slow scan is still reused for the full cache lifetime."""
        with self._lock:
            if self._cache is not None and self.clock() - self._cache[0] < CACHE_S:
                return self._cache[1], self._cache[2], self._cache[3]
            listed = candidates(self.runs_root)
            rows = []
            for name, path in listed[:MAX_CANDIDATES]:
                row = self._scan_one(name, path)
                if row is not None:
                    rows.append(row)
            names = dict(listed)
            scanned = min(len(listed), MAX_CANDIDATES)
            self._cache = (self.clock(), rows, names, scanned)
            self._extra = {}
            return rows, names, scanned

    def _extra_row(self, name, path):
        with self._lock:
            hit = self._extra.get(name)
            if hit is not None and self.clock() - hit[0] < CACHE_S:
                return hit[1]
            row = self._scan_one(name, path)
            self._extra[name] = (self.clock(), row)
            return row

    def build(self, run: str | None = None) -> dict:
        base_rows, names, scanned = self._rows()
        all_rows, extra_scanned = list(base_rows), 0
        listed_past_cap = run is not None and run in names and run not in {r["name"] for r in base_rows}
        if listed_past_cap and list(names).index(run) >= MAX_CANDIDATES:
            extra = self._extra_row(run, names[run])
            extra_scanned = 1
            if extra is not None:
                all_rows.append(extra)
        not_scanned = max(0, len(names) - scanned - extra_scanned)
        rows = [r for r in all_rows if r["turns"] > 0]
        rows.sort(key=lambda r: (-r["virtual"][0]["cost_usd"], r["name"]))
        total_rows = rows if run is None else [r for r in all_rows if r["name"] == run]
        total = {"turns": sum(r["turns"] for r in total_rows),
                 "prompt_tokens": sum(r["prompt_tokens"] for r in total_rows),
                 "completion_tokens": sum(r["completion_tokens"] for r in total_rows)}
        refs = [{"name": self.redact(n)[:NAME_MAX], "price_in": pi, "price_out": po,
                 "cost_usd": vs.cost(total["prompt_tokens"], total["completion_tokens"], pi, po)}
                for n, pi, po in self.cfg.references()]
        total["virtual"] = refs
        head = refs[0]
        meter = None
        if total["turns"]:
            meter = {"text": f"Local ≈ {usd(head['cost_usd'])} (virtual, at {head['name']})",
                     "title": (f"Virtual spend: {total['turns']} local turns, {total['prompt_tokens']} prompt and "
                               f"{total['completion_tokens']} completion tokens priced at {head['name']}. Nobody was charged and "
                               "this is not part of the helper spend. "
                               + (f"{not_scanned} runs were not scanned. " if not_scanned else "") + "Click for the table.")}
        shown = [{"name": self.redact(r["name"])[:NAME_MAX], "turns": r["turns"], "prompt_tokens": r["prompt_tokens"],
                  "completion_tokens": r["completion_tokens"], "agent_wall_s": r["agent_wall_s"],
                  "virtual": [{"name": self.redact(v["name"])[:NAME_MAX], "cost_usd": v["cost_usd"]} for v in r["virtual"]],
                  "left_out": self.redact(vs.skipped_text(r["skipped"]))[:300]} for r in rows[:MAX_ROWS]]
        return {"label": vs.LABEL, "virtual": True, "scope": "run" if run is not None else "all", "run": self.redact(run)[:NAME_MAX] if run is not None else None,
                "references": [{"name": r["name"], "price_in": r["price_in"], "price_out": r["price_out"]} for r in refs],
                "total": total, "meter": meter, "rows": shown, "not_scanned": not_scanned, "more_rows": max(0, len(rows) - MAX_ROWS)}
