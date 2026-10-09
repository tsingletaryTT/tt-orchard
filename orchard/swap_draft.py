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


def _facts(run_dir: Path, inputs: dict) -> tuple[str | None, str | None, dict, list[str]]:
    """The new model (stage 0's id, with its revision), the nearest model, and both snapshots."""
    try:
        delta = json.loads((run_dir / "stages" / "0" / "delta.json").read_text(encoding="utf-8"))
        model, nearest = str(delta["model"]), str(delta["nearest_model"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, None, {}, [f"stages/0/delta.json has no model and nearest_model to draft from ({exc})"]
    problems, snapshots = [], {}
    for key, what in (("base", "the nearest model's snapshot"), ("model", "the new model's snapshot")):
        value = inputs.get(key)
        if not value:
            problems.append(f"the run has no {key}= input ({what})")
        elif not Path(value).is_dir():
            problems.append(f"the {key}= input {value} does not exist ({what})")
        else:
            snapshots[key] = str(value)
    return model, nearest, snapshots, problems


def draft(*, run_dir, tt_model_root, cache_root, hf_home, inputs: dict, chips: int) -> tuple[dict | None, list[str]]:
    run_dir = Path(run_dir)
    model, nearest, snapshots, problems = _facts(run_dir, inputs)
    if model is None:
        return None, problems

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


# ---- stage 4: one configuration per chip count --------------------------------------------------------

CONTAINER_KEYS = ("run_dir", "package", "profile", "chips", "nearest_model_id", "base_snapshot", "new_snapshot",
                  "new_model_id", "tt_cache", "hf_home", "operator_home", "port")
TEST_DEADLINE_S = 3600


def installed_containers(operator_home) -> list[dict]:
    """tt-model's container packages (installed.json), each with the repo and chip count its manifest names."""
    path = Path(operator_home) / ".cache" / "tt-model" / "installed.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for name, e in (data.items() if isinstance(data, dict) else ()):
        if not isinstance(e, dict) or not e.get("container") or not e.get("image"):
            continue
        try:
            m = json.loads(Path(e["manifest"]).read_text(encoding="utf-8"))
            out.append({"name": str(name), "repo": str(m["weights"]["repo_id"]),
                        "chips": int(m.get("device_count") or 0), "profile": e.get("profile")})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def draft_stage4(*, run_dir, tt_model_root, cache_root, hf_home, operator_home, inputs: dict, counts,
                 four_chip_package: str | None = None) -> tuple[dict[int, dict], dict | None, list[str]]:
    """Stage 4's configs/<N>/swap_config.json for each chip count, and hw_tests.json when every count got
    one. A count is served by the one v6 bundle of the nearest model with that many chips, or else by a
    container package (installed.json): `four_chip_package` when it names one, or the only one. The
    tensor cache of each is <cache_root>/<model slug>/<N>chip-<package slug>/tt_cache, the same in every
    run, so a later run of the same model reuses it. Returns (configs, hw_tests or None, notes)."""
    run_dir = Path(run_dir)
    model, nearest, snapshots, problems = _facts(run_dir, inputs)
    if problems:
        return {}, None, problems
    want = _repo(nearest)
    home = Path(cache_root) / slug(_repo(model))
    common = {"run_dir": str(run_dir), "nearest_model_id": want, "base_snapshot": snapshots["base"],
              "new_snapshot": snapshots["model"], "new_model_id": model, "hf_home": str(hf_home)}
    bundles = [b for b in read_installed(tt_model_root) if b.weights_repo == want and b.schema == "6"
               and (Path(tt_model_root) / b.name / "run.sh").is_file()]
    containers = [c for c in installed_containers(operator_home) if c["repo"] == want]
    configs, notes = {}, []
    for n in sorted(set(counts)):
        cache = lambda name: str(home / f"{n}chip-{name.lower().replace('/', '--')}" / "tt_cache")   # noqa: E731
        fit = [b for b in bundles if b.chips == n]
        if len(fit) == 1:
            configs[n] = {**common, "bundle_dir": str(Path(tt_model_root) / fit[0].name),
                          "tt_cache": cache(fit[0].name), "port": FIRST_PORT + n}
            continue
        if len(fit) > 1:
            notes.append(f"the {n}-chip configuration: more than one bundle serves {want}: "
                         + ", ".join(b.name for b in fit))
            continue
        cfit = [c for c in containers if c["chips"] == n]
        if n == 4 and four_chip_package:
            pick = next((c for c in cfit if c["name"] == four_chip_package), None)
            if pick is None:
                notes.append(f"four_chip_package {four_chip_package} is not an installed container package that "
                             f"serves {want} on 4 chips")
                continue
        elif len(cfit) == 1:
            pick = cfit[0]
        elif cfit:
            notes.append(f"the {n}-chip configuration: several container packages serve {want} ("
                         + ", ".join(c["name"] for c in cfit) + "); set four_chip_package in bringup.toml to choose")
            continue
        else:
            notes.append(f"the {n}-chip configuration: no installed bundle or container package serves {want} "
                         f"on {n} chips")
            continue
        if not pick.get("profile"):
            notes.append(f"the {n}-chip configuration: {pick['name']} names no default profile")
            continue
        configs[n] = {**common, "package": pick["name"], "profile": pick["profile"], "chips": n,
                      "operator_home": str(operator_home), "tt_cache": cache(pick["name"]), "port": FIRST_PORT + n}
    tests = None
    if configs and not notes:
        tests = {"tests": [{"chips": n, "script": "serve_and_compare.py" if "bundle_dir" in c
                            else "serve_and_compare_container.py", "deadline_s": TEST_DEADLINE_S}
                           for n, c in sorted(configs.items())]}
    return configs, tests, notes
