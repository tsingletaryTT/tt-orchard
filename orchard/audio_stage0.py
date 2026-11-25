# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 of an audio run as supervisor code (issue 143).

A model that counts as audio (orchard/audio_facts.py) needs no model to triage: the audio branch of `delta_triage.py` reads safetensors headers and small
JSON files. The agent stage 0 still booted the local coder, which held a board and took a cold start for a run that never asks it anything. When the
command knows the model is audio it records two run_start inputs:

    audio_model   "1"                        selects the harness stage 0 (`stages.AUDIO_STAGE_0`, `needs_coder` False)
    audio_base    "<owner>/<name>"           the nearest model's repository id (the audio base row's base); its snapshot is the input `base`

This module reads the selection from the ledger the way a resume must (the first `run_start` decides) and prepares the stage directory: it copies the shipped
template and its side file and writes `triage_config.json` from the run's inputs, which is what the agent did by hand. The template remains the judge of whether
the model is audio: the supervisor's body fails the stage when delta.json does not say so. Nothing here imports ttnn or opens a device."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from orchard import safeio
from orchard.stages import SHIPPED_TEMPLATES

MODEL_INPUT = "audio_model"
BASE_INPUT = "audio_base"
TIMEOUT_S = 600.0                         # the template reads headers and samples values; the same limit as the dry classification
TEMPLATE = "delta_triage.py"
SIDE_FILES = ("audio_config_categories.json",)       # files the template reads from its own directory when present
READ_MAX = 4 * 1024 * 1024                # the template and its side file are far smaller


def selected(entries) -> bool:
    """True when the first `run_start` holds the input `audio_model` equal to the string "1". Only that entry counts: the command writes it once, a resume keeps
    it, and nothing the agent writes is read. A ledger that does not say so, or whose entry is malformed, selects nothing (the agent stage 0 runs)."""
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and e.get("event") == "run_start":
            data = e.get("data")
            inputs = data.get("inputs") if isinstance(data, dict) else None
            return isinstance(inputs, dict) and inputs.get(MODEL_INPUT) == "1"
    return False


def input_problems(inputs: dict) -> list[str]:
    """Why the stage cannot be prepared from the run's inputs, or []: the model's snapshot, the base's snapshot and the base's repository id are needed."""
    out = []
    for key, what in (("model", "the model's snapshot directory"), ("base", "the nearest model's snapshot directory"),
                      (BASE_INPUT, "the nearest model's repository id")):
        value = inputs.get(key)
        if not isinstance(value, str) or not value.strip() or "\n" in value or "\0" in value:
            out.append(f"an audio run needs the input {key} ({what}); the run's inputs have none")
    return out


def prepare(stage_dir, run_dir, *, model_id: str, inputs: dict) -> Path:
    """Copy the shipped template (and its side file) into `stage_dir` and write `triage_config.json` from the inputs. Returns the template's path. The gate compares
    the copy with the shipped file, so the copy is byte for byte the shipped one. Raises OSError or safeio.SafeReadError when a file cannot be read or written."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    shipped = Path(SHIPPED_TEMPLATES[TEMPLATE])
    script = stage_dir / TEMPLATE
    safeio.atomic_write(script, safeio.read_bytes(shipped, max_bytes=READ_MAX), mode=0o600)
    for name in SIDE_FILES:
        side = shipped.parent / name
        if side.is_file():
            safeio.atomic_write(stage_dir / name, safeio.read_bytes(side, max_bytes=READ_MAX), mode=0o600)
    config = {"run_dir": str(Path(run_dir).resolve()), "model_id": model_id, "model_snapshot": inputs["model"],
              "nearest_model_id": inputs[BASE_INPUT], "nearest_snapshot": inputs["base"]}
    safeio.atomic_write(stage_dir / "triage_config.json", json.dumps(config, indent=2).encode("utf-8") + b"\n", mode=0o600)
    return script


def run_template(script, run_dir, *, env: dict, timeout_s: float | None = None) -> tuple[bool, str]:
    """Run the copied template in the run directory with `env` (the agent's environment: no token, HOME inside the run). (True, "") when it exits 0, else
    (False, the reason): the last line of its error output, or that it did not finish. Never raises for a failed or missing interpreter."""
    limit = TIMEOUT_S if timeout_s is None else timeout_s
    try:
        done = subprocess.run([sys.executable, str(script)], cwd=str(run_dir), capture_output=True, text=True, timeout=limit, env=env,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False, f"delta_triage.py did not finish within {limit:g} s"
    except OSError as exc:
        return False, f"delta_triage.py could not start ({type(exc).__name__})"
    if done.returncode != 0:
        last = [ln for ln in (done.stderr or done.stdout or "").splitlines() if ln.strip()]
        return False, f"delta_triage.py exited {done.returncode}: {(last[-1] if last else 'no output')[:300]}"
    return True, ""
