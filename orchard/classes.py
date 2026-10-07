# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The outcome class of a bring-up: what kind of work this model needs, decided once by stage 0.

    weights-only     the nearest supported model's architecture, only the weights differ
    weights+sidecar  weights-only for the backbone, plus extra weight files the supported runtime does not
                     load (a task head, for example). The backbone takes the weights-only path; the sidecar
                     is checked by its own gate (orchard/stages.py, gate_sidecar_parity)
    full-port        the model needs new model code
    unknown          stage 0 could not decide (an unreadable or overlapping sidecar, headers that would not
                     parse). Nothing unproven runs as weights-only, so it takes the full-port path

Every class maps to one of the two paths the stage machine already knows, so a run that records a class
changes nothing for the stages that do not care about it. `path` stays in delta.json and in stage 0's
ledger entry, and a run from before classes (a path and no class) gets the class its path implies.
"""
from __future__ import annotations

CLASSES = ("weights-only", "weights+sidecar", "full-port", "unknown")

_PATH_OF = {"weights-only": "weights-only", "weights+sidecar": "weights-only",
            "full-port": "full-port", "unknown": "full-port"}


def path_of(cls) -> str | None:
    """The path a class takes, or None for a value that is not a class."""
    return _PATH_OF.get(cls) if isinstance(cls, str) else None


def class_of_path(path) -> str | None:
    """The class a path alone implies, for a run that recorded no class. Never `weights+sidecar` or
    `unknown`: those only come from stage 0's own answer."""
    return path if path in ("weights-only", "full-port") else None
