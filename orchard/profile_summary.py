# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard profile`: run a command under the tt-metal profiler on a leased board and write a bounded `profile-summary.json` (issue 113, step 1).

    tt-orchard profile [--out DIR] [--top N] [--chips N] [--timeout S] [--report-timeout S] [--exclude BOARD] [--wait-s S]
                       [--gozer PATH] [--perf-report PATH] [--who W] [--reason R] -- PYTHON [-m MODULE | SCRIPT] [ARGS...]
    tt-orchard profile --from DIR [--out DIR] [--top N] [--perf-report PATH]       (summarise a profile that already exists; no hardware)

What it does, in order:

  0. Checks that the Python given can find `tracy` and that `$TT_METAL_HOME` (or `--tt-metal-home`) has the Tracy tools, before any lease is taken.
  1. Picks a board whose chips are all FREE (`bench_control.choose_boards`, the same call the benchmark launcher uses) and runs
     `gozer run --exact N -- PYTHON -m tracy -r --no-web-server -o OUT/tracy ARGS...` so the board stays leased for the whole run and is reset
     on release. The words after `--` are what `python -m tracy` takes: a script or `-m module`, then its arguments.
  2. Finds `ops_perf_results*.csv` under `OUT/tracy` (the raw per-op file the profiler exports) and runs `tt-perf-report` on it to get the
     report's own table, utilization columns and matmul advice (`--csv`).
  3. Reads both files through orchard/safeio.py (size caps, no links, no FIFOs) and writes `OUT/profile-summary.json`.

The summary is bounded. `SUMMARY_MAX_BYTES` caps the file whatever the input holds: top ops, per-op-code rows and per-layer rows are cut to fit,
every string is cut to a fixed length, and `limits` says what was cut. Neither CSV and no profiler log is ever copied into it or handed to a
model. A figure the exported files do not hold is `null` and is named in `could_not_measure`; a missing or unreadable column never becomes 0.

Op groups. The raw CSV holds signpost rows (a name the workload emitted with `tracy_signpost`) between the op rows. A signpost named
`layer<N>/boundary`, `layer<N>/mixer`, `layer<N>/moe` or `layer<N>/mlp` (separator `/`, `:` or `.`; `layer_<N>` also works) starts that group of
layer N; `outside_layers` starts the group of ops outside the layers; ops before the first signpost are outside the layers. Any other signpost is
counted and does not change the group. When no signpost matches there are no groups and `could_not_measure` says so: the summary never guesses a
group from an op name. Each group gives its op count, time in ops (sum of device kernel durations), time in gaps (sum of op-to-op latencies) and
the floor the rule of thumb gives (3 to 5 microseconds per op including its gap, from published Blackhole fused-decode work; an estimate that no device reading backs).

What it cannot see. The ops CSV carries per-op DRAM and network columns only when the profiler's performance counters ran (`DRAM BW UTIL PER CTRL (%)`,
`NPE CONG IMPACT (%)`). When those columns are absent or empty the summary says "not measured". `DRAM BW UTIL PER CTRL (%)` is a percent of peak per
DRAM controller; the CSV holds no peak, so the summary does not turn it into GB/s.

Several devices in one CSV. Totals and groups are for the device with the most op rows; `devices_seen` says how many there were. Device times of
different devices overlap in wall time, so adding them would overstate the run.

Exit codes: 0 summary written from measured data; 1 the profiled command failed (the summary covers whatever it exported); 2 refused (usage, a bad
`--from` directory); 3 nothing measured (no profiler in the Python given, no free board, no `ops_perf_results` file, the CSV unreadable). A summary file
is written for 0, 1 and 3 whenever the output directory could be created.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import errno
import hashlib
import heapq
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from orchard import safeio

SCHEMA = "tt-orchard-profile-summary/1"
SUMMARY_FILE = "profile-summary.json"
RUN_LOG = "profile-run.log"
REPORT_CSV = "perf-report.csv"
RAW_SNAPSHOT = "ops-snapshot.csv"          # the private copy of the ops CSV that both the totals and tt-perf-report read
TRACY_DIR = "tracy"

SUMMARY_MAX_BYTES = 48 * 1024          # choice: a few thousand tokens at most; a model reads this, the CSV it came from can be hundreds of MB
RAW_MAX_BYTES = 256 << 20              # choice: the raw ops CSV of a long run; a larger file is refused and never cut
REPORT_MAX_BYTES = 64 << 20
MAX_LINE_BYTES = 1 << 20               # choice: one CSV line (a row with its attribute and input text) is a few KB; a line this long is not an ops row
MAX_ROWS = 2_000_000                   # rows read from one CSV; a file with more is refused
MAX_TOP = 50
DEFAULT_TOP = 10
MAX_LAYERS_LISTED = 64
MAX_OP_CODES_LISTED = 15
MAX_NAME = 80
MAX_ADVICE_ITEMS = 5
MAX_ADVICE_CHARS = 240
MAX_REASON = 300
MAX_BANKS = 16
RULE_OF_THUMB_US = (3.0, 5.0)          # per op including its gap (issue 113, comment)
CONGESTION_FLAG_PCT = 10.0             # choice: an op whose network congestion impact is at least this is counted as congested
MAX_SEARCH_ENTRIES = 5000

EXIT_OK, EXIT_COMMAND_FAILED, EXIT_REFUSED, EXIT_NOTHING = 0, 1, 2, 3

GROUP_BOUNDARY, GROUP_MIXER, GROUP_MOE_MLP, GROUP_OUTSIDE = "layer_boundary", "mixer", "moe_or_mlp", "outside_layers"
GROUPS = (GROUP_BOUNDARY, GROUP_MIXER, GROUP_MOE_MLP, GROUP_OUTSIDE)
_SIGNPOST_GROUP = {"boundary": GROUP_BOUNDARY, "mixer": GROUP_MIXER, "moe": GROUP_MOE_MLP, "mlp": GROUP_MOE_MLP}
_LAYER_SIGNPOST = re.compile(r"^layer[_-]?(\d{1,4})[/:.](boundary|mixer|moe|mlp)$", re.IGNORECASE)
_OUTSIDE_SIGNPOST = re.compile(r"^outside[_-]?layers$", re.IGNORECASE)
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d{1,3})?$")

# Columns of the raw ops CSV (tools/tracy/process_ops_logs.py in tt-metal) and of tt-perf-report's `--csv` file that the summary reads.
RAW_COLUMNS = {"code": "OP CODE", "type": "OP TYPE", "device": "DEVICE ID", "arch": "DEVICE ARCH", "dur": "DEVICE KERNEL DURATION [ns]",
               "gap": "OP TO OP LATENCY [ns]", "dram_bank": "DRAM BW UTIL PER CTRL (%)", "noc_cong": "NPE CONG IMPACT (%)"}
REPORT_COLUMNS = ("OP Code", "Device Time", "Op-to-Op Gap", "Cores", "DRAM", "DRAM %", "FLOPs %", "Bound", "Advice", "Total %")


class ProfileError(Exception):
    """A refusal or a failure that stops the run; the message is a plain sentence with no file content."""


# ---------------------------------------------------------------------------------------------------------------- cell readers

def _clip(text, n: int = MAX_NAME) -> str:
    """`text` as a printable string of at most `n` characters. Control characters and lone surrogates become a space."""
    s = text if isinstance(text, str) else ""
    s = "".join(c if (c.isprintable() and not 0xD800 <= ord(c) <= 0xDFFF) else " " for c in s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _num(cell, *, lo: float = 0.0, hi: float = 1e15):
    """The finite number in a CSV cell, or None when the cell is blank, not a plain decimal, out of range or not text.
    Never raises, so one bad cell is a figure that was not measured and the rest of the row is still read."""
    if not isinstance(cell, str) or len(cell) > 40:
        return None
    t = cell.strip()
    if not t or not _NUMBER.match(t):
        return None
    try:
        return safeio.finite_number(float(t), lo=lo, hi=hi, what="cell")
    except (safeio.BadValue, OverflowError, ValueError):
        return None


def _num_list(cell, *, lo: float = 0.0, hi: float = 1000.0):
    """A list of numbers from a cell like `[12.5, 13.0]` or `12.5 13.0`, at most MAX_BANKS of them. None when there is none or any element is bad."""
    if not isinstance(cell, str) or len(cell) > 2000:
        return None
    parts = [p for p in re.split(r"[\s,;\[\]]+", cell) if p]
    if not parts or len(parts) > MAX_BANKS:
        return None
    out = [_num(p, lo=lo, hi=hi) for p in parts]
    return None if any(v is None for v in out) else out


def _round(x, nd: int = 3):
    return None if x is None else round(x, nd)


def _rows(source, what: str):
    """(header list, row iterator) of CSV `source`: text, or an iterable of lines (what safeio.iter_lines gives). A cell longer than the csv module
    allows, or more than MAX_ROWS rows, ends the read with a ProfileError."""
    reader = csv.DictReader(io.StringIO(source) if isinstance(source, str) else source)
    try:
        header = list(reader.fieldnames or [])
    except csv.Error as exc:
        raise ProfileError(f"{what} is not readable as CSV ({type(exc).__name__})") from None
    if not header:
        raise ProfileError(f"{what} has no header row")

    def gen():
        n = 0
        try:
            for row in reader:
                n += 1
                if n > MAX_ROWS:
                    raise ProfileError(f"{what} has more than {MAX_ROWS} rows")
                yield row
        except csv.Error as exc:
            raise ProfileError(f"{what} is not readable as CSV ({type(exc).__name__})") from None
    return header, gen()


def _cell(row: dict, col: str):
    v = row.get(col) if isinstance(row, dict) else None
    return v if isinstance(v, str) else None


# ------------------------------------------------------------------------------------------------------------- the raw ops CSV

def _empty_bucket() -> dict:
    return {"ops": 0, "time_in_ops_us": 0.0, "gap_us": 0.0, "ops_without_duration": 0, "ops_without_gap": 0, "long_gap_ops": 0, "long_gap_us": 0.0}


def _add(bucket: dict, dur_ns, gap_ns) -> None:
    bucket["ops"] += 1
    if dur_ns is None:
        bucket["ops_without_duration"] += 1
    else:
        bucket["time_in_ops_us"] += dur_ns / 1000.0
    if gap_ns is None:
        bucket["ops_without_gap"] += 1
    elif gap_ns > LONG_GAP_NS:
        # The host was idle or busy elsewhere (a sync, a compile, the first op after a device open); this is host time between ops. Kept apart so
        # one multi-second gap does not hide the 3 to 5 microsecond gaps the rule of thumb is about.
        bucket["long_gap_ops"] += 1
        bucket["long_gap_us"] += gap_ns / 1000.0
    else:
        bucket["gap_us"] += gap_ns / 1000.0


def _finish_bucket(b: dict) -> dict:
    lo, hi = RULE_OF_THUMB_US
    # A bucket whose every op lacked a duration (or a gap) did not measure that time: it is None, never 0.
    no_dur = b["ops"] > 0 and b["ops_without_duration"] == b["ops"]
    no_gap = b["ops"] > 0 and b["ops_without_gap"] == b["ops"]
    return {"ops": b["ops"], "time_in_ops_us": None if no_dur else _round(b["time_in_ops_us"]), "gap_us": None if no_gap else _round(b["gap_us"]),
            "rule_of_thumb_floor_us": [_round(b["ops"] * lo, 1), _round(b["ops"] * hi, 1)],
            "ops_without_duration": b["ops_without_duration"], "ops_without_gap": b["ops_without_gap"],
            "long_gap_ops": b["long_gap_ops"], "long_gap_us": _round(b["long_gap_us"])}


LONG_GAP_NS = 1_000_000                # choice: a gap over 1 ms is host idle time; it is reported as long_gap_us and left out of gap_us
MAX_OP_CODES_TRACKED = 5000            # choice: distinct op codes kept per device; a CSV with more folds the rest into one "(other op codes)" row


class _Device:
    """What one device's rows add up to, kept as running sums so a CSV of millions of rows needs memory for the distinct codes and layers only."""

    def __init__(self) -> None:
        self.total = _empty_bucket()
        self.by_code: dict[str, dict] = {}
        self.by_group: dict[str, dict] = {}
        self.by_layer: dict[int, dict[str, dict]] = {}
        self.banks_ops = 0
        self.banks_unparsed = 0
        self.cong_unparsed = 0
        self.banks_ctrl = 0
        self.banks_busiest: list[float] | None = None
        self.banks_max = 0.0
        self.cong_ops = 0
        self.cong_max = 0.0
        self.cong_over = 0

    def add(self, layer, group, code, dur, gap, bank_cell, cong, cong_cell=None) -> None:
        _add(self.total, dur, gap)
        if code not in self.by_code and len(self.by_code) >= MAX_OP_CODES_TRACKED:
            code = "(other op codes)"
        _add(self.by_code.setdefault(code, _empty_bucket()), dur, gap)
        _add(self.by_group.setdefault(group, _empty_bucket()), dur, gap)
        if layer is not None:
            _add(self.by_layer.setdefault(layer, {}).setdefault(group, _empty_bucket()), dur, gap)
        banks = _num_list(bank_cell) if isinstance(bank_cell, str) and bank_cell.strip() else None
        if banks is None and isinstance(bank_cell, str) and bank_cell.strip():
            self.banks_unparsed += 1
        if banks:
            self.banks_ops += 1
            self.banks_ctrl = max(self.banks_ctrl, len(banks))
            self.banks_max = max(self.banks_max, max(banks))
            if self.banks_busiest is None or max(banks) > max(self.banks_busiest):
                self.banks_busiest = banks
        if cong is None and cong_cell is not None and cong_cell.strip():
            self.cong_unparsed += 1
        if cong is not None:
            self.cong_ops += 1
            self.cong_max = max(self.cong_max, cong)
            self.cong_over += cong >= CONGESTION_FLAG_PCT


def analyze_raw(source) -> dict:
    """Totals, op groups, per-op-code rows and the counter columns of the raw ops CSV (text or an iterable of lines). Pure; raises ProfileError for a CSV it cannot read.

    Op rows are the rows whose OP TYPE is not `signpost` and that carry a DEVICE ID. Rows with no device id are host-only rows and are counted in
    `host_only_rows`. Rows are folded into running sums per device as they are read (`_Device`), so memory does not grow with the row count."""
    header, rows = _rows(source, "the ops CSV")
    if "OP CODE" not in header:
        raise ProfileError("the ops CSV has no OP CODE column")
    missing = [v for v in RAW_COLUMNS.values() if v not in header]
    has_device = RAW_COLUMNS["device"] in header
    has_arch = RAW_COLUMNS["arch"] in header
    group, layer = GROUP_OUTSIDE, None
    signposts = matched = host_only = 0
    arch = None
    devices: dict[int, _Device] = {}
    for row in rows:
        code = _clip(_cell(row, "OP CODE") or "", MAX_NAME)
        kind = (_cell(row, "OP TYPE") or "").strip().lower()
        if kind == "signpost":
            signposts += 1
            m = _LAYER_SIGNPOST.match(code)
            if m:
                matched += 1
                group, layer = _SIGNPOST_GROUP[m.group(2).lower()], int(m.group(1))
            elif _OUTSIDE_SIGNPOST.match(code):
                matched += 1
                group, layer = GROUP_OUTSIDE, None
            continue
        dev = _num(_cell(row, RAW_COLUMNS["device"]) or "", lo=0, hi=1e6) if has_device else None
        if dev is None:
            host_only += 1
            continue
        if arch is None and has_arch:
            arch = _clip(_cell(row, RAW_COLUMNS["arch"]) or "", 24) or None
        dev = int(dev)
        if dev not in devices and len(devices) >= 4096:
            host_only += 1                  # a CSV naming thousands of devices is hostile; the extra ones are not tracked
            continue
        devices.setdefault(dev, _Device()).add(
            layer, group, code or "(unnamed)", _num(_cell(row, RAW_COLUMNS["dur"]) or ""), _num(_cell(row, RAW_COLUMNS["gap"]) or ""),
            _cell(row, RAW_COLUMNS["dram_bank"]), _num(_cell(row, RAW_COLUMNS["noc_cong"]) or "", hi=1000.0), _cell(row, RAW_COLUMNS["noc_cong"]))
    could: list[str] = []
    out: dict = {"signposts_seen": signposts, "signposts_matching_layer_groups": matched, "host_only_rows": host_only,
                 "columns_missing": missing, "arch": arch, "devices_seen": len(devices), "device": None}
    if not devices:
        could.append("device ops: the ops CSV holds no row with a device id, so no device time was read")
        out.update(total=None, top_by_code=[], top_by_device_time=[], groups=None, per_layer=[], layers_seen=0, dram_per_bank=None, noc_congestion=None, could_not_measure=could)
        return out
    device = min(devices, key=lambda d: (-devices[d].total["ops"], d))
    d = devices[device]
    out["device"] = device
    if len(devices) > 1:
        could.append(f"other devices: {len(devices) - 1} more device(s) are in the CSV; totals and groups cover device {device} only")
    if RAW_COLUMNS["dur"] not in header:
        could.append(f"device time: the ops CSV has no {RAW_COLUMNS['dur']} column")
    elif d.total["ops_without_duration"]:
        could.append(f"device time of {d.total['ops_without_duration']} op(s): the cell was empty or not a number; they count as ops and add no time")
    if RAW_COLUMNS["gap"] not in header:
        could.append(f"gaps between ops: the ops CSV has no {RAW_COLUMNS['gap']} column")
    elif d.total["ops_without_gap"]:
        could.append(f"gap of {d.total['ops_without_gap']} op(s): the cell was empty or not a number; they count as ops and add no gap")
    out["total"] = _finish_bucket(d.total)
    # Ranked by raw sums (a missing column adds 0 to the order only); the rows themselves report a missing time as None.
    ranked = sorted(d.by_code.items(), key=lambda kv: (-(kv[1]["time_in_ops_us"] + kv[1]["gap_us"]), kv[0]))
    out["top_by_code"] = [{"op_code": c, **_finish_bucket(b)} for c, b in ranked[:MAX_OP_CODES_LISTED]]
    by_time = sorted(d.by_code.items(), key=lambda kv: (-kv[1]["time_in_ops_us"], kv[0]))     # the fallback for the top ops: device time alone
    out["top_by_device_time"] = [{"op_code": c, **_finish_bucket(b)} for c, b in by_time[:MAX_TOP]]
    out["op_codes_seen"] = len(d.by_code)
    if matched:
        out["groups"] = {g: _finish_bucket(d.by_group[g]) if g in d.by_group else None for g in GROUPS}
        layers = sorted(d.by_layer)
        out["layers_seen"] = len(layers)
        out["per_layer"] = [{"layer": n, **{g: _finish_bucket(d.by_layer[n][g]) for g in GROUPS if g in d.by_layer[n]}} for n in layers[:MAX_LAYERS_LISTED]]
    else:
        out["groups"], out["layers_seen"], out["per_layer"] = None, 0, []
        could.append("op groups: no signpost in the CSV is named layer<N>/boundary, layer<N>/mixer, layer<N>/moe or layer<N>/mlp "
                     f"({signposts} signpost(s) seen); the workload must emit them with tracy_signpost, and no group is guessed from an op name")
    if d.banks_ops:
        out["dram_per_bank"] = {"ops_with_counter": d.banks_ops, "controllers": d.banks_ctrl, "unit": "percent of peak per DRAM controller",
                                "busiest_op_per_controller_pct": [_round(v, 1) for v in d.banks_busiest or []], "max_pct": _round(d.banks_max, 1)}
    else:
        out["dram_per_bank"] = None
        if d.banks_unparsed:
            could.append(f"per-bank DRAM bandwidth: {d.banks_unparsed} op row(s) have a value in {RAW_COLUMNS['dram_bank']} that is not a list of percentages; none was read")
        else:
            could.append(f"per-bank DRAM bandwidth: no op row has a value in {RAW_COLUMNS['dram_bank']}; NoC event capture was not on (profile with --noc-traces)")
    if d.cong_ops:
        out["noc_congestion"] = {"ops_with_counter": d.cong_ops, "max_pct": _round(d.cong_max, 1),
                                 f"ops_at_or_over_{int(CONGESTION_FLAG_PCT)}_pct": d.cong_over}
    else:
        out["noc_congestion"] = None
        if d.cong_unparsed:
            could.append(f"network congestion: {d.cong_unparsed} op row(s) have a value in {RAW_COLUMNS['noc_cong']} that is not a number; none was read")
        else:
            could.append(f"network congestion: no op row has a value in {RAW_COLUMNS['noc_cong']}; NoC event capture was not on (profile with --noc-traces)")
    out["could_not_measure"] = could
    return out


# ----------------------------------------------------------------------------------------------------- the tt-perf-report CSV

def read_report(source, top_n: int, device: int | None = None) -> tuple[list[dict], list[str]]:
    """The `top_n` rows of tt-perf-report's `--csv` file with the largest device time, and the reasons something was not read. Pure.

    Every field is typed: a number cell that is not a plain decimal is None, advice is split on the bullet the tool joins it with, cut to
    MAX_ADVICE_ITEMS items of MAX_ADVICE_CHARS characters. Columns the file lacks are named in the reasons and give None for every row. Each entry
    carries the report's `Device`. With `device` given, only that device's rows are ranked, so the list describes the device the summary does; a
    report with no Device column cannot be tied to it and gives no rows."""
    header, rows = _rows(source, "the tt-perf-report CSV")
    could = [f"report column {c}: absent from the tt-perf-report CSV" for c in REPORT_COLUMNS if c not in header]
    if "OP Code" not in header or "Device Time" not in header:
        raise ProfileError("the tt-perf-report CSV lacks the OP Code or Device Time column, so it cannot rank ops")
    if device is not None and "Device" not in header:
        could.append(f"top ops: the tt-perf-report CSV has no Device column, so its rows cannot be tied to device {device}, the device of this summary")
        return [], could
    heap: list[tuple] = []                      # the top_n rows so far; the smallest time (the later row among equals) is at the root
    for seq, row in enumerate(rows):
        dt = _num(_cell(row, "Device Time") or "")
        if dt is None:
            continue
        if device is not None and _num(_cell(row, "Device") or "", lo=0, hi=1e6) != device:
            continue
        item = (dt, -seq, row)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    kept = [(dt, row) for dt, _, row in sorted(heap, key=lambda t: (-t[0], -t[1]))]
    out = []
    for dt, row in kept:
        advice_cell = _cell(row, "Advice") or ""
        advice = [_clip(a, MAX_ADVICE_CHARS) for a in advice_cell.split("•")[:MAX_ADVICE_ITEMS * 4] if a.strip()][:MAX_ADVICE_ITEMS]
        out.append({"op_code": _clip(_cell(row, "OP Code") or "", MAX_NAME) or "(unnamed)",
                    "id": _num(_cell(row, "ID") or "", hi=1e12), "device": _num(_cell(row, "Device") or "", lo=0, hi=1e6),
                    "device_time_us": _round(dt), "op_to_op_gap_us": _round(_num(_cell(row, "Op-to-Op Gap") or "")),
                    "share_of_total_pct": _round(_num(_cell(row, "Total %") or ""), 1), "bound": _clip(_cell(row, "Bound") or "", 16) or None,
                    "cores": _num(_cell(row, "Cores") or "", hi=1e6), "dram_gb_s": _round(_num(_cell(row, "DRAM") or ""), 1),
                    "dram_pct_of_peak": _round(_num(_cell(row, "DRAM %") or ""), 1), "flops_pct_of_peak": _round(_num(_cell(row, "FLOPs %") or ""), 1),
                    "advice": advice})
    if not out:
        could.append("top ops: the tt-perf-report CSV has no row with a readable Device Time")
    return out, could


def top_from_raw(raw: dict) -> list[dict]:
    """When tt-perf-report cannot run, the top rows come from the raw CSV: op codes by device time alone (the gap before an op is not time the op ran,
    and `--top` promises the largest device time first), with no utilization and no advice."""
    return [{"op_code": r["op_code"], "ops": r["ops"], "device_time_us": r["time_in_ops_us"], "op_to_op_gap_us": r["gap_us"],
             "advice": []} for r in raw.get("top_by_device_time", [])]


# ------------------------------------------------------------------------------------------------------------- the summary

def _dump(summary: dict) -> str:
    return json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def bound(summary: dict) -> dict:
    """`summary` cut until its JSON is at most SUMMARY_MAX_BYTES (a `limits.cut` already in it is kept, so a second call does not lose the record of the first). The lists that grow with the input go first (layers, then top ops, then op codes,
    then reasons); `limits.cut` names what was cut. If the fixed part alone is too large the lists are emptied. Returns a new dict."""
    s = json.loads(_dump(summary))
    earlier = (s.get("limits") or {}).get("cut") if isinstance(s.get("limits"), dict) else None
    cut: list[str] = [c for c in earlier if isinstance(c, str)] if isinstance(earlier, list) else []
    s["limits"] = {"max_bytes": SUMMARY_MAX_BYTES, "cut": cut}

    def size() -> int:
        return len(_dump(s).encode("utf-8"))
    for key, floor in (("per_layer", 0), ("top_ops", 3), ("by_op_code", 3), ("could_not_measure", 5)):
        while size() > SUMMARY_MAX_BYTES and isinstance(s.get(key), list) and len(s[key]) > floor:
            s[key] = s[key][: max(floor, len(s[key]) // 2)]
            if key not in cut:
                cut.append(key)
    for key in ("per_layer", "top_ops", "by_op_code", "could_not_measure"):
        if size() > SUMMARY_MAX_BYTES and isinstance(s.get(key), list):
            s[key] = []
            if key not in cut:
                cut.append(key)
    if size() > SUMMARY_MAX_BYTES:                  # unreachable with the caps above; kept so the bound holds if a cap is raised
        for key in ("groups", "dram_per_bank", "noc_congestion"):
            s[key] = None
            cut.append(key)
    return s


def build_summary(*, raw: dict | None, top_ops: list[dict] | None, top_source: str | None, extra_could: list[str], run: dict | None,
                  command_sha: str | None, status_reason: str | None, top_n: int) -> dict:
    could = [_clip(c, MAX_REASON) for c in extra_could]
    if raw is not None:
        could += [_clip(c, MAX_REASON) for c in raw.get("could_not_measure", [])]
    measured = raw is not None and raw.get("total") is not None
    summary = {
        "schema": SCHEMA,
        "status": ("measured" if measured and not extra_could else "partial" if measured else "not_measured"),
        "status_reason": _clip(status_reason, MAX_REASON) if status_reason else None,
        "command_sha256": command_sha, "run": run,
        "device": None if not measured else {"id": raw["device"], "arch": raw["arch"], "devices_seen": raw["devices_seen"]},
        "total": raw["total"] if measured else None,
        "rule_of_thumb": {"per_op_us_low_high": list(RULE_OF_THUMB_US), "basis": "published Blackhole fused-decode cases: about 3 to 5 microseconds per op "
                          "including its gap; an estimate to compare with the measured times, with no device reading behind it"},
        "groups": raw.get("groups") if measured else None, "layers_seen": raw.get("layers_seen") if measured else None,
        "per_layer": raw.get("per_layer", []) if measured else [],
        "top_ops_source": top_source if top_ops else None, "top_ops_requested": top_n, "top_ops": top_ops or [],
        "by_op_code": raw.get("top_by_code", []) if measured else [],
        "op_codes_seen": raw.get("op_codes_seen") if measured else None,
        "dram_per_bank": raw.get("dram_per_bank") if measured else None,
        "noc_congestion": raw.get("noc_congestion") if measured else None,
        "could_not_measure": could,
    }
    return bound(summary)


# --------------------------------------------------------------------------------------------------------- running things

def command_digest(argv: list[str]) -> str:
    """A hash of the profiled command, so two summaries can be told to come from the same command without storing its text (it can hold paths)."""
    return hashlib.sha256("\0".join(argv).encode("utf-8", "surrogateescape")).hexdigest()


def tracy_argv(user: list[str], out_dir: Path, noc_traces: bool = False) -> list[str]:
    """`PYTHON -m tracy -r --no-web-server -o OUT/tracy ARGS...` from the words the operator gave after `--`. The first word must be a Python."""
    if not user or not re.match(r"^python[0-9.]*$", os.path.basename(user[0])):
        raise ProfileError("the command after -- must start with a Python interpreter (for example `python3 script.py` or `python3 -m pytest FILE`): "
                           "the profiler front end is `python -m tracy`")
    if len(user) < 2:
        raise ProfileError("the command after -- needs a script or `-m module` after the interpreter")
    return [user[0], "-m", "tracy", "-r", "--no-web-server", *(["--collect-noc-traces"] if noc_traces else []), "-o", str(out_dir / TRACY_DIR), *user[1:]]


def has_profiler(python: str, run=subprocess.run) -> str | None:
    """None when `python -m tracy` can be found by that interpreter; otherwise one sentence. Uses find_spec, so nothing is imported and no device is opened.
    The probe keeps the environment (PYTHONPATH included) because the profiled run inherits it: a tt-metal setup that exposes tools/tracy through
    PYTHONPATH must pass, and a probe in isolated mode would refuse it."""
    try:
        res = run([python, "-c", "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tracy') else 3)"],
                  capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the Python given cannot be run ({type(exc).__name__})"
    if res.returncode != 0:
        return ("the Python given has no `tracy` module (tt-metal's profiler front end, tools/tracy). Use the Python of a tt-metal build with ENABLE_TRACY "
                "on, with that tools directory on its path")
    return None


TRACY_TOOLS = ("tracy-capture", "tracy-csvexport")      # what tools/tracy looks for under $TT_METAL_HOME/build/tools/profiler/bin


def has_tracy_tools(tt_metal_home: str | None) -> str | None:
    """None when both Tracy command-line tools are under `tt_metal_home`; otherwise one sentence. `python -m tracy` finds them under
    $TT_METAL_HOME/build/tools/profiler/bin and falls back to the current directory when TT_METAL_HOME is unset, so a run started from anywhere
    else stops with "Tracy tools were not found" after it holds a lease. Only looks at file names; opens nothing."""
    if not tt_metal_home or not isinstance(tt_metal_home, str) or "\0" in tt_metal_home:
        return ("TT_METAL_HOME is not set and --tt-metal-home was not given; `python -m tracy` looks for its capture tool under "
                "$TT_METAL_HOME/build/tools/profiler/bin (the current directory when unset), so the run would stop after it took a lease")
    bin_dir = Path(tt_metal_home) / "build" / "tools" / "profiler" / "bin"
    absent = []
    for tool in TRACY_TOOLS:
        try:
            ok = (bin_dir / tool).is_file()
        except (OSError, ValueError):
            ok = False
        if not ok:
            absent.append(tool)
    if absent:
        return f"{' and '.join(absent)} not found under build/tools/profiler/bin of the tt-metal directory given; the build has no Tracy tools"
    return None


def run_group(argv: list[str], *, timeout_s: float, log_path: Path, env: dict | None = None, term_grace_s: float = 150.0) -> tuple[int | None, bool]:
    """Run `argv` in its own process group with stdout and stderr in `log_path`. Returns (exit code, timed_out). A timeout, or an interrupt of this
    process, sends SIGTERM to the group, waits `term_grace_s` (gozer's own stop takes up to 120 s and resets the chips), then SIGKILL. The process group
    is always reaped before return."""
    with safeio.create_for_write(log_path) as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        try:
            try:
                return proc.wait(timeout=timeout_s), False
            except subprocess.TimeoutExpired:
                _signal_group(proc.pid, signal.SIGTERM)
                _wait_group_gone(proc, term_grace_s)
                return proc.returncode, True
        except BaseException:
            # Ctrl+C or another exit from this function: the child is in its own session and did not get the signal. SIGTERM first, so gozer can
            # release the lease and reset the chips, then the grace time; a second interrupt skips the wait and goes straight to the kill below.
            _signal_group(proc.pid, signal.SIGTERM)
            _wait_group_gone(proc, term_grace_s)
            raise
        finally:
            # The leader's exit says nothing about its descendants: a child that ignores SIGTERM, or one left running by a command that
            # exited on its own, still holds the chips. Killing a group with no member is a no-op.
            _signal_group(proc.pid, signal.SIGKILL)
            proc.wait()


@contextlib.contextmanager
def exit_on_sigterm():
    """While the block runs, SIGTERM to this process raises SystemExit(143) instead of ending it at once, so `run_group`'s cleanup runs (SIGTERM to the
    child's session, the grace time, the kill) and the workload does not keep running under gozer's lease with no timeout. The old handler is put
    back on the way out. A thread other than the main one cannot install a handler; the block then runs as it is."""
    try:
        old = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128 + signum)))
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def _wait_group_gone(proc, grace_s: float) -> None:
    """Wait up to `grace_s` seconds for the leader to exit and for no other member of its group to remain."""
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            try:
                os.killpg(proc.pid, 0)
            except (ProcessLookupError, PermissionError):
                return
        time.sleep(0.1)
    proc.poll()


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)             # the group id is the pid of the child started with start_new_session
    except (ProcessLookupError, PermissionError):
        pass


def locate_ops_csv(pin) -> tuple[str, ...] | None:
    """The path parts, beneath the pinned directory `pin`, of the newest `ops_perf_results*.csv` (the profiler names it with a timestamp, so the last
    name in sort order is the newest). Walks with safeio.scan_beneath: links are never entered, and the walk is cut at MAX_SEARCH_ENTRIES entries."""
    try:
        found: list[tuple[str, ...]] = []

        def visit(entry):
            name = entry.parts[-1] if entry.parts else ""
            if entry.kind == "file" and name.startswith("ops_perf_results") and name.endswith(".csv"):
                found.append(tuple(entry.parts))
        safeio.scan_beneath(pin, visit, max_entries=MAX_SEARCH_ENTRIES)
    except (safeio.SafeReadError, OSError, ValueError):
        return None
    return max(found, key=lambda p: (p[-1], p)) if found else None


def find_ops_csv(root: Path) -> Path | None:
    """The path of the CSV `locate_ops_csv` finds beneath `root`, or None. For a caller that only needs the name; `summarise_dir` reads the file
    through the pin it found it under, so a directory swapped for a link in between is refused."""
    try:
        with safeio.pinned_root(root) as pin:
            best = locate_ops_csv(pin)
    except (safeio.SafeReadError, OSError, ValueError):
        return None
    return root.joinpath(*best) if best else None


def _lines(path, cap: int, anchor: int | None = None):
    return safeio.iter_lines(path, max_bytes=cap, max_line_bytes=MAX_LINE_BYTES, anchor=anchor)


def analyze_raw_file(path, anchor: int | None = None) -> dict:
    """`analyze_raw` of the file at `path` (relative to the directory descriptor `anchor` when one is given), streamed through safeio.iter_lines with
    the RAW_MAX_BYTES cap. Raises safeio.SafeReadError or ProfileError, nothing else."""
    lines = _lines(path if anchor is not None else Path(path), RAW_MAX_BYTES, anchor)
    try:
        return analyze_raw(lines)
    finally:
        lines.close()


def read_report_file(path, top_n: int, anchor: int | None = None, device: int | None = None) -> tuple[list[dict], list[str]]:
    """`read_report` of the file at `path` (relative to `anchor` when given), streamed through safeio.iter_lines with the REPORT_MAX_BYTES cap.
    Raises safeio.SafeReadError or ProfileError, nothing else."""
    lines = _lines(path if anchor is not None else Path(path), REPORT_MAX_BYTES, anchor)
    try:
        return read_report(lines, top_n, device)
    finally:
        lines.close()


def _clear_name(path: Path) -> None:
    """Remove whatever is at `path` (a file, or a link that tt-perf-report would write through, which opens its --csv path with a plain open(...,"w")),
    so the tool creates a new file. A directory there is an IsADirectoryError, which the caller reports."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except PermissionError as exc:
        if os.path.isdir(path) and not os.path.islink(path):
            raise IsADirectoryError(errno.EISDIR, "the path is a directory") from exc
        raise


def run_perf_report(perf_report: str, raw_csv: Path, out_csv: Path, *, timeout_s: float, log_path: Path, run=run_group) -> str | None:
    """None when `tt-perf-report` wrote `out_csv`; otherwise one sentence. Its terminal output goes to `log_path` and is not read."""
    argv = [perf_report, str(raw_csv), "--csv", str(out_csv), "--no-color", "--no-summary", "--ignore-signposts"]
    try:
        _clear_name(out_csv)
        code, timed_out = run(argv, timeout_s=timeout_s, log_path=log_path, term_grace_s=10.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"tt-perf-report could not be run ({type(exc).__name__})"
    if timed_out:
        return f"tt-perf-report did not finish in {int(timeout_s)} s"
    if code != 0:
        return f"tt-perf-report exited with code {code}"
    return None


# ------------------------------------------------------------------------------------------------------------- the command

def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=add_help, description="profile a command on a leased board and write a bounded profile-summary.json")
    p.add_argument("--out", default="profile", help="output directory (default ./profile): holds profile-summary.json, the profiler's files under tracy/, perf-report.csv and profile-run.log")
    p.add_argument("--top", type=int, default=DEFAULT_TOP, help=f"ops listed in the summary, largest device time first (1 to {MAX_TOP}; default {DEFAULT_TOP})")
    p.add_argument("--from", dest="from_dir", metavar="DIR", help="summarise the ops_perf_results CSV under DIR instead of running anything (no board, no lease)")
    p.add_argument("--chips", type=int, default=1, help="chips to lease (default 1; a board with at least this many free chips is chosen)")
    p.add_argument("--timeout", type=float, default=1800.0, metavar="S", help="seconds the profiled command may run (default 1800)")
    p.add_argument("--report-timeout", type=float, default=300.0, metavar="S", help="seconds tt-perf-report may run (default 300)")
    p.add_argument("--exclude", action="append", default=[], metavar="BOARD", help="a board id not to use (repeat for several)")
    p.add_argument("--wait-s", type=float, default=0.0, metavar="S", help="wait up to this long for a free board (default 0: refuse at once)")
    p.add_argument("--noc-traces", action="store_true", help="also pass --collect-noc-traces to the profiler, which fills the per-DRAM-controller and network-congestion "
                   "columns (slower; not yet run on hardware by this command)")
    p.add_argument("--tt-metal-home", default=None, metavar="DIR", help="the tt-metal checkout whose build has the Tracy tools (default: $TT_METAL_HOME); "
                   "passed to the profiled command as TT_METAL_HOME")
    p.add_argument("--gozer", default="gozer", help="the gozer command (default gozer)")
    p.add_argument("--perf-report", default=None, metavar="PATH", help="the tt-perf-report command (default: the one on PATH)")
    p.add_argument("--who", default="tt-orchard:profile", help="the name the lease carries (default tt-orchard:profile)")
    p.add_argument("--reason", default="profile run (tt-orchard profile)", help="the reason the lease carries")
    p.add_argument("command", nargs=argparse.REMAINDER, help="after --: PYTHON [-m MODULE | SCRIPT] [ARGS...]")
    return p


def _write_summary(out_dir: Path, summary: dict) -> None:
    safeio.atomic_write(out_dir / SUMMARY_FILE, _dump(summary))


def _top_n(args) -> int:
    n = args.top
    return n if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= MAX_TOP else -1


def _earlier_profile(tracy_dir: Path) -> str | None:
    """A sentence when `tracy_dir` already holds anything, which a new run would add to and `summarise_dir` could then read as the new run's export."""
    try:
        with os.scandir(tracy_dir) as it:
            if next(it, None) is not None:
                return f"{TRACY_DIR}/ in the output directory is not empty"
    except FileNotFoundError:
        return None
    except (NotADirectoryError, OSError):
        return f"{TRACY_DIR} in the output directory is not a readable directory"
    return None


def summarise_dir(profile_dir: Path, out_dir: Path, *, top_n: int, perf_report: str | None, report_timeout_s: float, run=run_group,
                  run_info: dict | None = None, command_sha: str | None = None, command_failed: bool = False) -> tuple[dict, int]:
    """Find, run tt-perf-report on, read and summarise the profile under `profile_dir`; write the summary to `out_dir`. Returns (summary, exit code)."""
    extra: list[str] = []
    with contextlib.ExitStack() as stack:
        try:
            ppin = stack.enter_context(safeio.pinned_root(profile_dir))
            csv_parts = locate_ops_csv(ppin)
            opin = stack.enter_context(safeio.pinned_root(out_dir))
        except (safeio.SafeReadError, OSError, ValueError):
            csv_parts, opin = None, None
        if csv_parts is not None and opin is not None:
            return _summarise_found(ppin, csv_parts, profile_dir, opin, out_dir, extra, top_n=top_n, perf_report=perf_report, report_timeout_s=report_timeout_s,
                                    run=run, run_info=run_info, command_sha=command_sha, command_failed=command_failed)
    s = build_summary(raw=None, top_ops=None, top_source=None, extra_could=["everything: no ops_perf_results CSV was found under the profile directory; "
                      "the run exported no per-op file (the command may have opened no device, or the profiler build lacks device profiling)"],
                      run=run_info, command_sha=command_sha, status_reason="no ops_perf_results CSV found", top_n=top_n)
    return s, EXIT_NOTHING


def _device_to_filter(raw: dict):
    """The device id the report's rows are filtered to: the summary's device when the profile has more than one, else None (nothing to tell apart)."""
    dev = raw.get("device")
    return int(dev) if isinstance(dev, (int, float)) and not isinstance(dev, bool) and (raw.get("devices_seen") or 0) > 1 else None


def _copy_beneath(pin, rel: str, dest: Path) -> None:
    """Copy the file `rel` beneath the pinned directory to a new file at `dest`, through safeio.iter_lines (the RAW_MAX_BYTES cap, a link refused,
    the descriptor held open once), so the totals and tt-perf-report read the same bytes whatever happens to the profile directory afterwards."""
    lines = _lines(rel, RAW_MAX_BYTES, pin.fd)
    try:
        with safeio.create_for_write(dest) as f:
            for line in lines:
                f.write(line.encode("utf-8", "replace"))
    finally:
        lines.close()


def _summarise_found(ppin, csv_parts, profile_dir, opin, out_dir, extra, *, top_n, perf_report, report_timeout_s, run, run_info, command_sha, command_failed):
    """The rest of `summarise_dir`, once the CSV is found. It is read through `ppin` (the pinned profile directory), and tt-perf-report's CSV is
    read through `opin` (the pinned output directory), so a subdirectory or file swapped for a link after the walk is refused and never followed."""
    snapshot = out_dir / RAW_SNAPSHOT                    # tt-perf-report opens its input by path, so it gets the private copy too
    try:
        _copy_beneath(ppin, "/".join(csv_parts), snapshot)
        raw = analyze_raw_file(RAW_SNAPSHOT, anchor=opin.fd)
    except (safeio.SafeReadError, ProfileError, OSError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        s = build_summary(raw=None, top_ops=None, top_source=None, extra_could=[f"everything: the ops CSV could not be read ({_clip(reason, 120)})"],
                          run=run_info, command_sha=command_sha, status_reason="ops CSV unreadable", top_n=top_n)
        return s, EXIT_NOTHING
    top_ops, top_source = None, None
    tool = perf_report or shutil.which("tt-perf-report")
    if not tool:
        extra.append("utilization and advice: tt-perf-report was not found on PATH (pass --perf-report); the top rows come from the raw CSV and carry neither")
    else:
        report_csv = out_dir / REPORT_CSV
        problem = run_perf_report(tool, snapshot, report_csv, timeout_s=report_timeout_s, log_path=out_dir / "perf-report.log", run=run)
        if problem:
            extra.append(f"utilization and advice: {problem}; the top rows come from the raw CSV and carry neither")
        else:
            try:
                top_ops, rep_could = read_report_file(REPORT_CSV, top_n, anchor=opin.fd, device=_device_to_filter(raw))
                extra += rep_could
                top_source = "tt-perf-report"
                if not top_ops:                                   # nothing usable in the report: the raw rows below are the fallback
                    top_ops, top_source = None, None
                    extra.append("utilization and advice: the tt-perf-report CSV gave no ranked row; the top rows come from the raw CSV and carry neither")
            except (safeio.SafeReadError, ProfileError) as exc:
                reason = getattr(exc, "reason", None) or str(exc)
                extra.append(f"utilization and advice: the tt-perf-report CSV could not be read ({_clip(reason, 120)}); the top rows come from the raw CSV")
    if top_ops is None and raw.get("total") is not None:
        top_ops, top_source = top_from_raw(raw)[:top_n], "raw ops csv (one row per op code)"
    s = build_summary(raw=raw, top_ops=top_ops, top_source=top_source, extra_could=extra, run=run_info, command_sha=command_sha,
                      status_reason=None if raw.get("total") is not None else "the ops CSV holds no device op row", top_n=top_n)
    code = EXIT_NOTHING if raw.get("total") is None else (EXIT_COMMAND_FAILED if command_failed else EXIT_OK)
    return s, code


def main(args, *, out, run=run_group, probe=has_profiler, tools=has_tracy_tools, choose=None, now=time.monotonic, environ=None) -> int:
    from orchard import bench_control
    n = _top_n(args)
    if n < 0:
        print(f"tt-orchard profile: --top must be a whole number from 1 to {MAX_TOP}", file=out)
        return EXIT_REFUSED
    for flag, value, positive in (("--timeout", args.timeout, True), ("--report-timeout", args.report_timeout, True), ("--wait-s", args.wait_s, False)):
        # argparse takes "nan" and "inf" as floats; a NaN deadline is never reached and an infinite one never ends
        if not safeio.fits_float(value) or value < 0 or (positive and value == 0):
            print(f"tt-orchard profile: {flag} must be a finite number of seconds{' above 0' if positive else ', 0 or more'}", file=out)
            return EXIT_REFUSED
    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"tt-orchard profile: cannot create the output directory ({type(exc).__name__})", file=out)
        return EXIT_REFUSED
    words = list(args.command)
    if words and words[0] == "--":
        words = words[1:]
    if args.from_dir:
        if words:
            print("tt-orchard profile: --from reads an existing profile and takes no command", file=out)
            return EXIT_REFUSED
        src = Path(args.from_dir)
        if not src.is_dir():
            print("tt-orchard profile: --from must name a directory", file=out)
            return EXIT_REFUSED
        summary, code = summarise_dir(src, out_dir, top_n=n, perf_report=args.perf_report, report_timeout_s=args.report_timeout, run=run,
                                      run_info={"source": "existing profile directory", "exit_code": None, "wall_s": None})
    else:
        try:
            argv = tracy_argv(words, out_dir, noc_traces=bool(args.noc_traces))
        except ProfileError as exc:
            print(f"tt-orchard profile: {exc}", file=out)
            return EXIT_REFUSED
        if isinstance(args.chips, bool) or not isinstance(args.chips, int) or args.chips < 1:
            print("tt-orchard profile: --chips must be a whole number of 1 or more", file=out)
            return EXIT_REFUSED
        stale = _earlier_profile(out_dir / TRACY_DIR)
        if stale:
            print(f"tt-orchard profile: {stale}; the summary would otherwise be built from the earlier run's CSV. Use a new --out directory or remove it", file=out)
            return EXIT_REFUSED
        sha = command_digest(words)
        nothing = lambda why, text: (build_summary(raw=None, top_ops=None, top_source=None, extra_could=[text], run={"exit_code": None, "wall_s": None, "leased_chips": 0},
                                                   command_sha=sha, status_reason=why, top_n=n), EXIT_NOTHING)
        environ = os.environ if environ is None else environ
        home = args.tt_metal_home or environ.get("TT_METAL_HOME")
        problem = probe(words[0]) or tools(home)
        if problem:
            summary, code = nothing("no profiler", f"everything: {problem}")
        else:
            try:
                plan = (choose or bench_control.choose_boards)([args.chips], gozer=args.gozer, exclude=tuple(args.exclude), wait_s=args.wait_s)[0]
            except bench_control.NoFreeBoard as exc:
                # The exception text can hold board ids and gozer's stderr; a model reads the summary, so only a fixed sentence goes there.
                print(f"tt-orchard profile: no board was free to lease ({_clip(str(exc), 300)})", file=out)
                summary, code = nothing("no free board", "everything: no board was free to lease (the detail is on the console of the run); nothing was run")
                plan = None
            except (OSError, subprocess.SubprocessError) as exc:
                summary, code = nothing("gozer unavailable", f"everything: gozer could not be run ({type(exc).__name__}); nothing was run")
                plan = None
            if plan is not None:
                full = bench_control.gozer_run_args(plan, who=args.who, reason=args.reason, chips=args.chips, gozer=args.gozer) + argv
                print(f"tt-orchard profile: running under a lease on a free board ({args.chips} chip(s)); output in {out_dir}", file=out)
                t0 = now()
                try:
                    with exit_on_sigterm():
                        exit_code, timed_out = run(full, timeout_s=args.timeout, log_path=out_dir / RUN_LOG, env={**environ, "TT_METAL_HOME": home})
                except (OSError, subprocess.SubprocessError) as exc:
                    summary, code = nothing("run failed to start", f"everything: the profiled command could not be started ({type(exc).__name__})")
                else:
                    info = {"exit_code": exit_code, "wall_s": round(now() - t0, 1), "timed_out": timed_out, "leased_chips": args.chips}
                    failed = timed_out or exit_code != 0
                    summary, code = summarise_dir(out_dir / TRACY_DIR, out_dir, top_n=n, perf_report=args.perf_report, report_timeout_s=args.report_timeout,
                                                  run=run, run_info=info, command_sha=sha, command_failed=failed)
                    if timed_out:
                        summary["status_reason"] = f"the command was stopped after {int(args.timeout)} s; the summary covers what it exported"
                    elif exit_code == 10:
                        summary["status_reason"] = "gozer queued the run (exit 10); nothing was profiled"
                    elif failed and summary.get("status_reason") is None:
                        summary["status_reason"] = f"the command exited with code {exit_code}; the summary covers what it exported"
                    summary = bound(summary)                  # the reason was added after the first bound
    try:
        _write_summary(out_dir, summary)
    except OSError as exc:
        print(f"tt-orchard profile: cannot write the summary ({type(exc).__name__})", file=out)
        return EXIT_REFUSED
    print(f"tt-orchard profile: {summary['status']}; {SUMMARY_FILE} is {len(_dump(summary).encode())} bytes "
          f"({len(summary['could_not_measure'])} thing(s) not measured)", file=out)
    return code
