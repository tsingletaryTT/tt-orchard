# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Draft stage 2's `swap_config.json` from facts the run already holds.

The weights-swap-check and weights-sidecar-check skills start with this config. Every field is known
before the agent starts: the run's paths, stage 0's `delta.json` (the new model with its revision and
the nearest model), the run's inputs (`base=` and `model=` snapshots) and the installed bundles (the v6
bundle that serves the nearest model on the stage's chips). On the first lab run Coder-Next spent whole
attempts re-deriving them, and kept investigating why the bundle's architecture name differs from the
new model's (expected: prepare_swap.py copies the nearest model's config) until the watchdog stopped it.

`draft` returns (config, []) or (None, problems). It never guesses: two bundles that both fit, a missing
input or a missing snapshot is a problem, and the agent then finds the facts as the skill describes.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

from orchard.bringup_config import slug
from orchard.nearest import read_installed

KEYS = ("run_dir", "bundle_dir", "nearest_model_id", "base_snapshot", "new_snapshot", "new_model_id",
        "tt_cache", "hf_home", "port")
FIRST_PORT = 8100            # the skills' example port; the next free one is used when it is taken
PORT_TRIES = 100


def _repo(model_id: str) -> str:
    return str(model_id).split("@", 1)[0]


def _free_port() -> int | None:
    for port in range(FIRST_PORT, FIRST_PORT + PORT_TRIES):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return None


def draft(*, run_dir, tt_model_root, cache_root, hf_home, inputs: dict, chips: int) -> tuple[dict | None, list[str]]:
    run_dir = Path(run_dir)
    problems: list[str] = []
    try:
        delta = json.loads((run_dir / "stages" / "0" / "delta.json").read_text(encoding="utf-8"))
        model, nearest = str(delta["model"]), str(delta["nearest_model"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, [f"stages/0/delta.json has no model and nearest_model to draft from ({exc})"]

    snapshots = {}
    for key, what in (("base", "the nearest model's snapshot"), ("model", "the new model's snapshot")):
        value = inputs.get(key)
        if not value:
            problems.append(f"the run has no {key}= input ({what})")
        elif not Path(value).is_dir():
            problems.append(f"the {key}= input {value} does not exist ({what})")
        else:
            snapshots[key] = str(value)

    want = _repo(nearest)
    fits = [b for b in read_installed(tt_model_root)
            if b.weights_repo == want and b.schema == "6" and b.chips == chips
            and (Path(tt_model_root) / b.name / "run.sh").is_file()]
    if not fits:
        problems.append(f"no installed v6 bundle under {tt_model_root} serves {want} on {chips} chips")
    elif len(fits) > 1:
        problems.append(f"more than one bundle serves {want} on {chips} chips: "
                        + ", ".join(b.name for b in fits))

    port = _free_port()
    if port is None:
        problems.append(f"no free port from {FIRST_PORT} to {FIRST_PORT + PORT_TRIES - 1}")
    if problems:
        return None, problems
    return {"run_dir": str(run_dir),
            "bundle_dir": str(Path(tt_model_root) / fits[0].name),
            "nearest_model_id": want,
            "base_snapshot": snapshots["base"],
            "new_snapshot": snapshots["model"],
            "new_model_id": model,
            "tt_cache": str(Path(cache_root) / slug(_repo(model)) / "tt_cache"),
            "hf_home": str(hf_home),
            "port": port}, []
