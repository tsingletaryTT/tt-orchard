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
import os
import socket
from pathlib import Path

from orchard import graft_trial
from orchard.bringup_config import slug
from orchard.nearest import read_installed

KEYS = ("run_dir", "bundle_dir", "nearest_model_id", "base_snapshot", "new_snapshot", "new_model_id",
        "tt_cache", "hf_home", "port", "prompt_suite")
FIRST_PORT = 8100            # the skills' example port; the next free one is used when it is taken
PORT_TRIES = 100


def _repo(model_id: str) -> str:
    return str(model_id).split("@", 1)[0]


def nearest_problem(run_dir, config_path, *, tt_model_root=None, operator_home=None) -> str | None:
    """Why an agent-written swap config does not serve stage 0's nearest model, or None. The weights are swapped into a bundle or package of
    the nearest model; a config that names another model serves the wrong architecture (issue 67: a Qwen3.8-27B bundle for a Qwen3-Coder model).
    Repository ids are compared exactly, without any `@revision`. A missing config, a config with no nearest_model_id (prepare_swap.py refuses that itself) or a run with no stage 0 record is no problem here."""
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        want = _repo(json.loads((Path(run_dir) / "stages" / "0" / "delta.json").read_text(encoding="utf-8"))["nearest_model"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    named = cfg.get("nearest_model_id") if isinstance(cfg, dict) else None
    if not isinstance(named, str) or not named:
        return None                      # prepare_swap.py requires the key and refuses a config without it
    if _repo(named) != want:
        return (f"{Path(config_path).name} names {_repo(named)} as the nearest model, but stage 0's nearest model is {want}. "
                f"Weights are swapped into an installed bundle or package of {want}; one for another model serves the wrong architecture")
    # The label alone proves nothing: the package or bundle the config points at must itself serve the nearest model.
    if cfg.get("package") and cfg.get("bundle_dir"):
        return (f"{Path(config_path).name} names both a package and a bundle_dir; prepare_swap.py builds from the bundle, so the package could hide "
                f"another model's bundle. Name one")
    named_home = cfg.get("operator_home")
    if named_home and not Path(str(named_home)).is_absolute():
        named_home = str(Path(run_dir) / str(named_home))            # the container test runs from the run directory
    if (cfg.get("package") and operator_home is not None and named_home
            and os.path.realpath(str(named_home)) != os.path.realpath(str(operator_home))):
        # the container template reads installed.json and runs tt-model under the config's operator_home, so the registry checked here has to be that one
        return (f"{Path(config_path).name} names operator_home {named_home}, but the run's is {operator_home}; the package registry the test "
                f"reads has to be the one that was checked")
    served, what = _served_repo(cfg, run_dir, operator_home)
    if served is not None and _repo(served) != want:
        return (f"{Path(config_path).name} uses {what}, which serves {_repo(served)}, but stage 0's nearest model is {want}. "
                f"Weights are swapped into an installed bundle or package of {want}")
    return None


def _served_repo(cfg: dict, run_dir, operator_home) -> tuple[str | None, str]:
    """The weights repository the config's container package or v6 bundle serves, read from its manifest, and a name for the message. (None, "")
    when the config names neither or the manifest cannot be found (the template refuses a missing package itself)."""
    pkg = cfg.get("package")
    if isinstance(pkg, str) and pkg and operator_home is not None:
        for c in installed_containers(operator_home):
            if c["name"] == pkg:
                return c["repo"], f"the container package {pkg}"
    bundle = cfg.get("bundle_dir")
    if isinstance(bundle, str) and bundle:
        try:
            path = Path(bundle) if Path(bundle).is_absolute() else Path(run_dir) / bundle       # the test runs from the run directory
            m = json.loads((path / "tt_kernel_manifest.json").read_text(encoding="utf-8"))
            return str(m["weights"]["repo_id"]), f"the bundle {path.parent.name}/{path.name}"
        except (OSError, ValueError, KeyError, TypeError):
            return None, ""
    return None, ""


def stage4_nearest_problems(run_dir, stage_dir, *, tt_model_root=None, operator_home=None, counts=None) -> list[str]:
    """nearest_problem for each configs/<N>/swap_config.json of stage 4, each message led by its chip count. `counts`, when given, are the chip
    counts of the hardware-test plan: a configuration that is not in the plan is never run, so a leftover directory is not judged."""
    out = []
    for path in sorted(Path(stage_dir).glob("configs/*/swap_config.json")):
        if counts is not None and path.parent.name not in {str(c) for c in counts}:
            continue
        msg = nearest_problem(run_dir, path, tt_model_root=tt_model_root, operator_home=operator_home)
        if msg:
            out.append(f"the {path.parent.name}-chip configuration: {msg}")
    return out


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


def _container_fits(containers: list[dict], want: str, chips: int) -> list[dict]:
    """The installed container packages for `want` that run on exactly `chips` chips; when none does, those at the smallest larger count.
    Nearly every catalog package reports 2 boards (4 chips), so the stage's own chip count rarely matches; the supervisor leases the
    package's chips for the test (parking the coder when it needs its board), as stage 4 does for its 4-chip test."""
    mine = [c for c in containers if c["repo"] == want]
    exact = [c for c in mine if c["chips"] == chips]
    if exact:
        return exact
    larger = sorted({c["chips"] for c in mine if c["chips"] > chips})
    return [c for c in mine if larger and c["chips"] == larger[0]]


def draft(*, run_dir, tt_model_root, cache_root, hf_home, inputs: dict, chips: int,
          operator_home=None) -> tuple[dict | None, list[str]]:
    """Stage 2's swap_config.json. The one v6 bundle that serves the nearest model on `chips` chips is used when there is one. When
    there is none and `operator_home` is given, the one installed container (v5.1) package that serves it on that many chips is used
    instead (issue 60), with the keys the container template reads (CONTAINER_KEYS). Two that fit, or a package with no default
    profile, are left to the agent."""
    run_dir = Path(run_dir)
    model, nearest, snapshots, problems = _facts(run_dir, inputs)
    if model is None:
        return None, problems

    want = _repo(nearest)
    fits = [b for b in read_installed(tt_model_root)
            if b.weights_repo == want and b.schema == "6" and b.chips == chips
            and (Path(tt_model_root) / b.name / "run.sh").is_file()]
    boxes = _container_fits(installed_containers(operator_home), want, chips) if operator_home is not None else []
    box_chips = boxes[0]["chips"] if boxes else chips
    pick = None
    if len(fits) > 1:
        problems.append(f"more than one bundle serves {want} on {chips} chips: "
                        + ", ".join(b.name for b in fits))
    elif not fits and not boxes:
        problems.append(f"no installed v6 bundle under {tt_model_root} serves {want} on {chips} chips"
                        + (" and no installed container package does" if operator_home is not None else ""))
    elif not fits and len(boxes) > 1:
        problems.append(f"more than one container package serves {want} on {box_chips} chips: "
                        + ", ".join(c["name"] for c in boxes))
    elif not fits:
        pick = boxes[0]
        if not pick.get("profile"):
            problems.append(f"the container package {pick['name']} names no default profile")

    port = _free_port()
    if port is None:
        problems.append(f"no free port from {FIRST_PORT} to {FIRST_PORT + PORT_TRIES - 1}")
    if problems:
        return None, problems
    common = {"run_dir": str(run_dir), "nearest_model_id": want, "base_snapshot": snapshots["base"],
              "new_snapshot": snapshots["model"], "new_model_id": model, "hf_home": str(hf_home), "port": port,
              "prompt_suite": True}        # the swap test sends the prompt suite too (issue 15)
    if pick is not None:
        name = pick["name"].lower().replace("/", "--")
        return {**common, "package": pick["name"], "profile": pick["profile"], "chips": box_chips,
                "operator_home": str(operator_home),
                "tt_cache": str(Path(cache_root) / slug(_repo(model)) / f"{box_chips}chip-{name}" / "tt_cache")}, []
    return {**common, "bundle_dir": str(Path(tt_model_root) / fits[0].name),
            "tt_cache": str(Path(cache_root) / slug(_repo(model)) / "tt_cache")}, []


# ---- stage 4: one configuration per chip count --------------------------------------------------------

CONTAINER_KEYS = ("run_dir", "package", "profile", "chips", "nearest_model_id", "base_snapshot", "new_snapshot",
                  "new_model_id", "tt_cache", "hf_home", "operator_home", "port", "prompt_suite")
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
                 four_chip_package: str | None = None,
                 missing: list[int] | None = None) -> tuple[dict[int, dict], dict | None, list[str]]:
    """Stage 4's configs/<N>/swap_config.json for each chip count, and hw_tests.json when every count got
    one. A count is served by the one v6 bundle of the nearest model with that many chips, or else by a
    container package (installed.json): `four_chip_package` when it names one, or the only one. The
    tensor cache of each is <cache_root>/<model slug>/<N>chip-<package slug>/tt_cache, the same in every
    run, so a later run of the same model reuses it. Returns (configs, hw_tests or None, notes). `missing`, when given, receives the chip
    counts that no installed bundle or container package serves at all (not the ones left to a choice between several), issue 68."""
    run_dir = Path(run_dir)
    model, nearest, snapshots, problems = _facts(run_dir, inputs)
    if problems:
        return {}, None, problems
    want = _repo(nearest)
    grafted = graft_trial.chosen(run_dir)
    home = Path(cache_root) / slug(_repo(model))
    common = {"run_dir": str(run_dir), "nearest_model_id": want, "base_snapshot": snapshots["base"],
              "new_snapshot": snapshots["model"], "new_model_id": model, "hf_home": str(hf_home),
              "prompt_suite": True}
    bundles = [b for b in read_installed(tt_model_root) if b.weights_repo == want and b.schema == "6"
               and (Path(tt_model_root) / b.name / "run.sh").is_file()]
    containers = [c for c in installed_containers(operator_home) if c["repo"] == want]
    configs, notes = {}, []
    for n in sorted(set(counts)):
        cache = lambda name: str(home / f"{n}chip-{name.lower().replace('/', '--')}" / "tt_cache")   # noqa: E731
        fit = [b for b in bundles if b.chips == n]
        if len(fit) == 1:
            configs[n] = {**common, "bundle_dir": str(Path(tt_model_root) / fit[0].name),
                          "tt_cache": cache(fit[0].name), "port": FIRST_PORT + n,
                          # Stage 2's graft decision (orchard/graft_trial.py) covers the nearest model's own bundle, the server the trial
                          # measured. A container package below is another server and keeps its configuration.
                          **({"graft_mtp": True} if grafted else {})}
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
                if not cfit and missing is not None:      # nothing at all serves 4 chips: the count is missing, whatever was named
                    missing.append(n)
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
            if missing is not None:
                missing.append(n)
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
