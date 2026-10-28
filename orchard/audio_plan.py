# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The standard-library half of the audio stages (orchard/audio_stages.py holds the half that needs numpy and soundfile).

The supervisor and the stage table import this module; the numeric half runs in the interpreter that has numpy (orchard/audio_remote.py). It holds
the names, settings and plans of the runs, the commands they are started with, the config file of the chip test and the small file readers.
"""
from __future__ import annotations

import hashlib
import json
import math
import shlex
from pathlib import Path

from orchard import safeio
from orchard.audio_runtime import REFERENCE_FILE, RESULT_FILE, STAGE_DISK_GB

SCHEMA = 1
PROMPT = "A warm acoustic guitar melody with soft percussion, 100 BPM"
DURATION_S = 10.0
SEED = 42
SAMPLE_SIZE = 5292032                  # the library's own default for `generate`
SAMPLE_RATE = 44100

# Library registry names and the settings each is run with. The fine-tune is the post-trained 8-step model (ping-pong sampler, guidance scale 1);
# its base is the control and needs the base's own settings (the audio design's section 5).
LIBRARY_NAMES = {"stabilityai/stable-audio-3-small-music": "small-music", "stabilityai/stable-audio-3-small-music-base": "small-music-base"}
MODEL_SETTINGS = {"small-music": {"steps": 8, "cfg_scale": 1.0, "sampler": "pingpong"},
                  "small-music-base": {"steps": 50, "cfg_scale": 7.0, "sampler": "euler"}}

ROLE_REFERENCE, ROLE_CONTROL_REFERENCE, ROLE_HOST = "reference", "control_reference", "host_same_env"
ROLE_CHIP, ROLE_CONTROL_CHIP = "chip", "control_chip"


class AudioStageError(Exception):
    """A problem a stage reports as its failure reason."""




def library_name(repo) -> str | None:
    """The library's registry name for a repository id (a revision after `@` is ignored), or None for a model this module does not know."""
    return LIBRARY_NAMES.get(str(repo or "").split("@", 1)[0])


def settings_for(name: str) -> dict:
    """The recorded settings (audio_reference.MATCH_KEYS) of the run of library model `name`."""
    s = MODEL_SETTINGS[name]
    return {"prompt": PROMPT, "duration": DURATION_S, "steps": s["steps"], "cfg_scale": s["cfg_scale"], "sampler": s["sampler"], "seed": SEED,
            "sample_size": SAMPLE_SIZE}


def load_json(path, what: str) -> dict:
    try:
        return safeio.read_json(path, max_bytes=4 << 20, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        raise AudioStageError(f"{what} cannot be read: {exc}") from exc


def inside(run_dir, rel) -> Path | None:
    """`rel` as a path inside `run_dir`, or None for anything that leaves it: an absolute path, a `..` part, or a link on the way."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return None
    p = Path(run_dir)
    for part in Path(rel).parts:
        p = p / part
        if p.is_symlink():
            return None
    return p


def agree(recorded, fresh) -> bool:
    """A recorded comparison summary equals one computed again now (numbers to 1e-9, everything else exactly)."""
    if isinstance(recorded, dict) and isinstance(fresh, dict):
        return recorded.keys() == fresh.keys() and all(agree(recorded[k], fresh[k]) for k in recorded)
    if isinstance(recorded, (int, float)) and isinstance(fresh, (int, float)) and not isinstance(recorded, bool) and not isinstance(fresh, bool):
        return math.isclose(recorded, fresh, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(recorded, list) and isinstance(fresh, list):
        return len(recorded) == len(fresh) and all(agree(a, b) for a, b in zip(recorded, fresh))
    return type(recorded) is type(fresh) and recorded == fresh


def reference_plan(model_id, base_id) -> list[dict]:
    """The runs stage 1 makes: [{role, name (library name), env ('reference' or 'chip' interpreter), dir}]. The control and the same-environment
    run exist only when the base is a model this module knows."""
    name = library_name(model_id)
    if name is None:
        raise AudioStageError(f"{model_id} is not a model the audio runtime knows ({sorted(LIBRARY_NAMES)})")
    plan = [{"role": ROLE_REFERENCE, "name": name, "interpreter": "reference", "dir": f"reference/{name}"},
            {"role": ROLE_HOST, "name": name, "interpreter": "chip", "dir": f"host/{name}"}]
    base = library_name(base_id)
    if base is not None and base != name:
        plan.append({"role": ROLE_CONTROL_REFERENCE, "name": base, "interpreter": "reference", "dir": f"reference/{base}"})
    return plan


def reference_command(item: dict, paths, stage_rel_dir: Path, script: Path) -> tuple[list[str], dict]:
    """argv and extra environment for one CPU run. The reference interpreter finds the library through its editable install; the chip interpreter
    through PYTHONPATH. `--expect-src` makes the script refuse a library from anywhere else."""
    s = settings_for(item["name"])
    python = paths.reference_python if item["interpreter"] == "reference" else paths.chip_python
    argv = [python, str(script), "--model", item["name"], "--out", str(stage_rel_dir / item["dir"]), "--prompt", s["prompt"],
            "--duration", str(s["duration"]), "--steps", str(s["steps"]), "--cfg-scale", str(s["cfg_scale"]), "--seed", str(s["seed"]),
            "--sampler", s["sampler"], "--sample-size", str(s["sample_size"]), "--expect-src", paths.src_dir]
    env = {"PYTHONPATH": paths.src_dir} if item["interpreter"] == "chip" else {}
    return argv, env


def write_json(path, value) -> None:
    """Write a JSON file whole or not at all."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def chip_plan(model_id, base_id) -> list[dict]:
    name = library_name(model_id)
    if name is None:
        raise AudioStageError(f"{model_id} is not a model the audio runtime knows")
    plan = [{"role": ROLE_CHIP, "name": name}]
    base = library_name(base_id)
    if base is not None and base != name:
        plan.append({"role": ROLE_CONTROL_CHIP, "name": base})
    return plan


def snapshot_identity(snapshot_dir) -> dict:
    """Which snapshot a run was started on: its revision (the snapshot directory's name) and the sha256 of its `model_config.json` (the file behind
    the cache's link). The library's registry name does not say which snapshot a script loads; a run's manifest records the same two values, and the
    stages compare them."""
    d = Path(snapshot_dir)
    config = (d / "model_config.json").resolve()          # a Hugging Face snapshot holds the file as a link into the cache's blobs
    try:
        data = safeio.read_bytes(config, max_bytes=1 << 20)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        raise AudioStageError(f"the snapshot's model_config.json cannot be read: {exc}") from exc
    return {"revision": d.name, "config_sha256": hashlib.sha256(data).hexdigest()}


def repeat_name(name: str, k: int) -> str:
    """The directory name of repeat `k` of the run of library model `name`: the run itself for 0, `<name>-r<k>` after it."""
    return name if k == 0 else f"{name}-r{k}"


REPEATS = 3          # runs of each model on the chip; the first compiles kernels, so a timing needs more than one


def write_chip_config(stage_dir, paths, hf_home: str, plan: list[dict], repeats: int = REPEATS) -> Path:
    """`audio_chip_config.json`, the input of audio_chip_test.py: shared settings and one entry per run. Each model runs `repeats` times (the same
    settings and seed, so the runs agree with one another and only their timings differ)."""
    runs = [{"name": repeat_name(item["name"], k), "model": item["name"],
             **{key: MODEL_SETTINGS[item["name"]][key] for key in ("steps", "cfg_scale", "sampler")}}
            for item in plan for k in range(repeats)]
    cfg = {"prompt": PROMPT, "duration": DURATION_S, "seed": SEED, "sample_size": SAMPLE_SIZE, "src_dir": paths.src_dir, "hf_home": hf_home,
           "out_dir": "chip", "repeats": repeats, "runs": runs}
    path = Path(stage_dir) / CONFIG_FILE
    write_json(path, cfg)
    return path


def chip_test_command(paths, stage_dir_abs: Path, script: Path) -> str:
    """The one command the hardware phase runs under the lease."""
    return shlex.join([str(paths.chip_python), str(script), "--config", str(stage_dir_abs / CONFIG_FILE)])


CONFIG_FILE = "audio_chip_config.json"
