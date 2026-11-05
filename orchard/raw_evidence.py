# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The raw evidence files a gate recomputes a number from, and the ledger record that binds each one (issue 445).

A hardware test writes a file that holds the raw measurements: the sidecar parity records of stage 2 and the boot check's replies of
stage 7. The gates recompute the headline numbers from those records. The agent that finishes the stage can write in the same directory,
so a record and its summary could be edited together. When the supervisor's hardware test ends, `Supervisor._run_test` hashes each such
file and appends an `evidence` entry with `what = RAW_EVIDENCE_WHAT`. A gate that finds the entry requires the file to hash to the same
value, and requires the raw records to be there (the run is "bound"). A run with no entry (a run from before this check, or a stage whose
test did not write the file) is judged as before, and the gate says the number is the agent's when it cannot recompute it.

Everything that reads an outside file goes through `safeio`. Nothing here raises for a hostile file: the readers return reasons.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from orchard import safeio

RAW_EVIDENCE_WHAT = "raw evidence hashed after the hardware test"
# stage number -> raw evidence files, relative to the stage directory. Only these are hashed.
RAW_FILES: dict[int, tuple[str, ...]] = {
    2: ("evidence/sidecar-parity.json",),
    7: ("verify/evidence/verify.json",),
}
RAW_MAX_BYTES = 16 * 1024 * 1024        # the same cap as the other stage records (stages.JSON_MAX_BYTES)


def _rel(run_dir, path) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(run_dir))


def record_after_test(run_dir, stage_dir, n: int) -> list[dict]:
    """[{path, sha256}] for each raw file stage `n` names that exists and can be read within the cap. A file that cannot be read gets
    no record (the gate then treats the run as unbound for that file, as it would any file the test did not write)."""
    out = []
    for name in RAW_FILES.get(n, ()):
        path = Path(stage_dir) / name
        if not os.path.lexists(path):
            continue
        try:
            raw = safeio.read_bytes(path, max_bytes=RAW_MAX_BYTES)
        except safeio.SafeReadError:
            continue
        out.append({"path": _rel(run_dir, path), "sha256": hashlib.sha256(raw).hexdigest()})
    return out


def recorded_sha(run_dir, n: int, rel: str) -> str | None:
    """The sha256 of the last ledger entry the supervisor wrote for stage `n`'s raw file `rel` (a run-relative path), or None when there is
    none or the ledger cannot be read."""
    from orchard.ledger import LedgerCorrupt, read_entries
    try:
        entries = read_entries(Path(run_dir) / "ledger.jsonl")
    except (LedgerCorrupt, OSError):
        return None
    found = None
    for e in entries:
        data = e.get("data")
        if e.get("event") == "evidence" and e.get("stage") == n and isinstance(data, dict) \
                and data.get("what") == RAW_EVIDENCE_WHAT and data.get("path") == rel and isinstance(data.get("sha256"), str):
            found = data["sha256"]
    return found


def read(run_dir, stage_dir, n: int, name: str) -> tuple[object, list[str], bool]:
    """(document, reasons, bound) for the raw file `name` of stage `n`. `bound` says the ledger holds the file's hash. `document` is the
    parsed JSON, or None when it is missing or unreadable. With a ledger record the file must exist, be readable and hash to the recorded
    value; each failure is a reason. Without one a missing or unreadable file gives no reason here (the caller decides)."""
    path = Path(stage_dir) / name
    rel = _rel(run_dir, path)
    recorded = recorded_sha(run_dir, n, rel)
    bound = recorded is not None
    shown = os.path.basename(name)
    try:
        raw = safeio.read_bytes(path, max_bytes=RAW_MAX_BYTES)
    except safeio.SafeReadError as exc:
        if bound:
            return None, [f"{shown} cannot be read ({exc.reason}), though the supervisor hashed it when the hardware test ended"], True
        return None, [], False
    reasons = []
    if bound and hashlib.sha256(raw).hexdigest() != recorded:
        reasons.append(f"{shown} is not the file the supervisor hashed when the hardware test ended (its sha256 differs), so its records "
                       "were changed after the test")
    try:
        doc = safeio.parse_json_bytes(raw, max_bytes=RAW_MAX_BYTES)
    except safeio.SafeReadError as exc:
        if bound:
            reasons.append(f"{shown} is not readable JSON ({exc.reason})")
        return None, reasons, bound
    return doc, reasons, bound
