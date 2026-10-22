# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Research for a bring-up: which models could a new model be based on, and what already exists for them.

Each module reads one kind of source and returns evidence or an explicit "unavailable". `report.py` ranks
the candidates. Nothing here decides with a model, and nothing here writes outside the run directory.
Design: docs/superpowers/specs/2026-10-12-base-research-design.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Options:
    """How the research runs. `cache_dir` None keeps no cache (a dry run). `progress` gets one line of text before
    each slow read (research took 81 s for a 27B model, mostly reading weight headers); None says nothing."""
    cache_dir: Path | None = None
    offline: bool = False
    no_github: bool = False
    refresh: bool = False
    progress: Callable[[str], None] | None = None
