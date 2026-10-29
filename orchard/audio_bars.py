# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The agreement bars of audio stage 2, from a measured noise floor when a floor record exists (issue 173).

`orchard/noise_floor.py` measures the spread between CPU runs of the same code, sets each bar from it and checks the bars against a
known-bad control. Its JSON result (`python -m orchard.noise_floor ... --out FILE`) is kept in the reference stage's directory as
`noise_floor.json`. `load_floor` reads that file and says which bars a comparison uses:

    no file                    "no floor measured": the provisional bars in `audio_reference.DEFAULT_BARS` decide, and the label says so
    a file whose bars the tool itself called unusable (they cannot tell the good run from the damaged control, or the good run fails
                               them), a file that cannot be read, or a floor grouped other than by kind
                               the provisional bars decide, and the label says why the floor was not used
    a usable file              the measured bars decide, for every kind the floor covers; a kind with no measured bar keeps its provisional
                               bar and the label names it

The stage records `floor.record()` in its result (the file's sha256 included), and the gate loads the file again and refuses a result whose
record differs, so the bars a result cites are the bars the gate applies.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from orchard import audio_reference as ar
from orchard import noise_floor as nf

FLOOR_FILE = "noise_floor.json"
MAX_BYTES = 8 * 1024 * 1024
NO_FLOOR = "no floor measured: the provisional bars in audio_reference.DEFAULT_BARS decide this (a choice that was never measured)"


@dataclass(frozen=True)
class Floor:
    """Which bars a comparison uses and what to say about them. `bars` is None for the provisional bars."""
    bars: dict | None
    label: str
    sha256: str | None = None

    @property
    def measured(self) -> bool:
        return self.bars is not None

    def record(self) -> dict:
        """What a result file holds about its bars."""
        return {"measured": self.measured, "label": self.label, "file": FLOOR_FILE, "sha256": self.sha256}

    def describe(self, kinds) -> str:
        """The label for a comparison of arrays of these kinds: the measured floor's, with the kinds it does not cover named."""
        if not self.measured:
            return self.label
        uncovered = sorted(set(kinds) - set(self.bars))
        if not uncovered:
            return self.label
        return f"{self.label}; no measured bar for {', '.join(uncovered)}, which keep their provisional bars"


def load_floor(stage_dir) -> Floor:
    """The floor record in `stage_dir` (the reference stage's directory), as a `Floor`. Never raises: every way the file can fail to give
    measured bars becomes a provisional-bars `Floor` whose label says why."""
    path = Path(stage_dir) / FLOOR_FILE
    if not path.exists():
        return Floor(None, NO_FLOOR)
    try:
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("it is larger than 8 MiB")
        raw = path.read_bytes()
        res = json.loads(raw)
    except (OSError, ValueError) as exc:
        return Floor(None, f"the floor record {FLOOR_FILE} cannot be read ({type(exc).__name__}): the provisional bars decide this", None)
    sha = hashlib.sha256(raw).hexdigest()
    try:
        bars, label = nf.bars_or_fallback(res)
    except nf.NoiseFloorError as exc:
        return Floor(None, f"the floor record {FLOOR_FILE} is not usable ({str(exc)[:300]}): the provisional bars decide this", sha)
    group = res.get("group")
    if group != "kind":
        return Floor(None, f"the floor record groups its bars by {group!r} and stage 2 applies bars by kind: the provisional bars decide this", sha)
    return Floor(bars, label, sha)


def limit_line(floor: Floor | None) -> str:
    """The line of a result's `limits` about the bars."""
    if floor is None or not floor.measured:
        return "No floor measured: the agreement bars are provisional choices that were never set from a measured noise floor."
    return ("The agreement bars were set from a noise floor measured between CPU runs, with a known-bad control that they reject; the floor holds no "
            "chip arithmetic, and the margin leaves the room for it that the control run checked.")
