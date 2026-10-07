# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Append-only run ledger.

One JSON object per line. Each line carries the sha256 of the previous line's text, so editing
an earlier line breaks the chain and is detected on the next open. (The last line has no
successor, so a change to it alone cannot be detected. The chain covers every line except the last.)

A line only counts once its trailing newline is on disk. Opening the ledger sets aside any bytes
after the last newline (a write cut off by a crash) in a `.torn-<time>` file next to the ledger.
Nothing is deleted: each recovery writes its own sidecar, created exclusively so it can never
replace an earlier one.

If an append fails partway (disk full, I/O error), the file is cut back to where it was before
the attempt, so the next append cannot merge with a half-written line. If even that cut fails, the
writer refuses all further appends. After close() the ledger refuses appends, because the lock
is gone.

The current state of a run is computed from the ledger by `replay_state`. No other state file
exists, so nothing can disagree with the ledger.

One process may hold a ledger open at a time. Two writers would break the chain, so a second
opener gets LedgerLocked.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

GENESIS = "0" * 64

EVENTS = frozenset({
    "run_start", "stage_start", "stage_end", "park", "restore",
    "retry", "escalate", "notice", "measurement", "decision",
    # An evidence file and its sha256 (plan 4: agent steps and hardware tests write them).
    "evidence",
})

# Every number in a measurement entry says whether it was measured or is still to do.
LABELS = frozenset({"measured", "TODO"})


class LedgerCorrupt(Exception):
    """The file is not a valid chain. A person has to look at it."""


class LedgerLocked(Exception):
    """Another process has this ledger open."""


def _digest(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _complete_lines(path: Path) -> list[str]:
    """The complete lines of the file (those that end in a newline). Bytes after the last newline
    are a write in progress or cut off by a crash; they are not part of the chain and are ignored
    here. Only a writer's `_recover` sets them aside."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # Not a line the ledger wrote (it writes ASCII JSON), so a person has to look.
        raise LedgerCorrupt(f"{path} is not valid UTF-8 ({exc.reason} at byte {exc.start})") from exc
    return text.split("\n")[:-1]


def read_entries(path) -> list[dict]:
    """Read a ledger file and verify its whole hash chain. Raises LedgerCorrupt on any defect.

    This takes no lock and writes nothing, so it is safe next to a running supervisor and for
    tools that must not touch a run (orchard/status.py). A line the writer is still appending
    may be missing from the result; a complete line is never missing.
    """
    entries, prev = [], GENESIS
    for i, line in enumerate(_complete_lines(Path(path)), start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerCorrupt(f"line {i} is not JSON") from exc
        # Valid JSON is not enough: replay_state indexes these fields, so a wrong shape must be
        # reported here as corruption and not surface later as a KeyError or AttributeError.
        if not isinstance(entry, dict):
            raise LedgerCorrupt(f"line {i} is not a JSON object")
        seq = entry.get("seq")
        if type(seq) is not int or seq != i:   # `true` equals 1 in Python, so check the type
            raise LedgerCorrupt(f"line {i}: expected sequence {i}, found {seq!r}")
        if entry.get("prev") != prev:
            raise LedgerCorrupt(f"line {i}: hash chain broken")
        if not isinstance(entry.get("event"), str):
            raise LedgerCorrupt(f"line {i}: event must be a string")
        if "stage" not in entry:
            raise LedgerCorrupt(f"line {i} has no stage field")
        if not isinstance(entry.get("data"), dict):
            raise LedgerCorrupt(f"line {i}: data must be an object")
        entries.append(entry)
        prev = _digest(line)
    return entries


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # flock belongs to the open file, so it is released when close() runs or the
        # process dies. That is what lets a restarted supervisor reopen a crashed run.
        self._lock_file = open(self.path.with_name(self.path.name + ".lock"), "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise LedgerLocked(f"another process holds {self.path}") from None
        self._seq = 0
        self._prev = GENESIS
        self._closed = False
        self._broken = False  # set if a failed append could not be undone
        try:
            self._recover()
        except BaseException:
            # __init__ is failing, so the caller never gets an object to close. Release the
            # lock here, or a retry in the same process would report LedgerLocked.
            self._lock_file.close()
            raise

    def close(self) -> None:
        # Safe to call twice. Appends after this point are refused (see append).
        self._closed = True
        self._lock_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _lines(self) -> list[str]:
        return _complete_lines(self.path)

    def _recover(self) -> None:
        if self.path.exists():
            raw = self.path.read_bytes()
            good_end = raw.rfind(b"\n") + 1  # offset just past the last complete line
            torn = raw[good_end:]
            if torn:
                # Keep the cut-off bytes for a person to inspect, then drop them from the chain.
                # Nanosecond name plus exclusive create ("xb"): two recoveries in the same
                # instant get different files, and an existing sidecar is never overwritten.
                stamp = time.time_ns()
                n = 0
                while True:
                    suffix = "" if n == 0 else f"-{n}"
                    sidecar = self.path.with_name(f"{self.path.name}.torn-{stamp}{suffix}")
                    try:
                        with open(sidecar, "xb") as sf:
                            sf.write(torn)
                            # The sidecar must be on disk before the ledger is cut, or a crash
                            # between the two would lose the bytes this recovery promises to keep.
                            sf.flush()
                            os.fsync(sf.fileno())
                        break
                    except FileExistsError:
                        n += 1
                with open(self.path, "r+b") as f:
                    f.truncate(good_end)
                    f.flush()
                    os.fsync(f.fileno())
        entries = self.read()  # verifies the whole chain
        if entries:
            self._seq = entries[-1]["seq"]
            self._prev = _digest(self._lines()[-1])

    def read(self) -> list[dict]:
        return read_entries(self.path)

    def append(self, event: str, stage: int | None = None, **data) -> dict:
        if self._closed:
            raise ValueError("ledger is closed; the lock is released, so writing is not safe")
        if self._broken:
            raise ValueError("ledger writer is unusable after a failed append; reopen it")
        if event not in EVENTS:
            raise ValueError(f"unknown ledger event {event!r}")
        if event == "measurement" and data.get("label") not in LABELS:
            raise ValueError("a measurement entry needs label='measured' or label='TODO'")
        entry = {
            "seq": self._seq + 1,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prev": self._prev,
            "event": event,
            "stage": stage,
            "data": data,
        }
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        with open(self.path, "ab") as f:
            start = f.tell()  # append mode: tell() is the file size before this write
            try:
                f.write(line.encode("utf-8") + b"\n")
                f.flush()
                os.fsync(f.fileno())  # the entry exists only once this returns
            except BaseException:
                # Some or all of the line may be in the file while _seq and _prev still
                # describe the old tail. Cut back so the next append starts clean.
                try:
                    f.truncate(start)
                except OSError:
                    self._broken = True
                raise
        self._seq += 1
        self._prev = _digest(line)
        return entry


def replay_state(entries: list[dict]) -> dict:
    """Current run state, computed by walking the entries in order."""
    state = {"stage": None, "stage_status": None, "parked": False, "completed": []}
    for e in entries:
        event, stage = e["event"], e["stage"]
        if event == "stage_start":
            state["stage"], state["stage_status"] = stage, "running"
        elif event == "stage_end":
            result = e["data"].get("result")
            state["stage_status"] = result
            # A retried stage ends more than once; list it once, in the order it first passed.
            if result == "pass" and stage not in state["completed"]:
                state["completed"].append(stage)
        elif event == "park":
            # replay_state is the authority on "parked"; orchard/handoff.py progress() reads the
            # same entries and agrees. A park closed before the coder was told to stop ends
            # with step "abandoned": the coder never left.
            state["parked"] = e["data"].get("step") != "abandoned"
        elif event == "restore":
            # A restore runs in several steps (orchard/handoff.py). The coder is back only after
            # the last one. An entry with no step (the plan 1 form) still ends the park.
            if e["data"].get("step") in (None, "resumed"):
                state["parked"] = False
    return state


def file_evidence(path) -> dict:
    """The path and sha256 that an evidence entry records."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}
