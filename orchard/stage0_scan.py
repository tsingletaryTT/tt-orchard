# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 writes the hazard scan (issue 183): `stages/0/hazard-scan.json`.

`orchard/hazards.py` is the pure scan. This module is the part that finds the two configs a run has, runs the scan and writes the
record. The supervisor calls `write_scan` when stage 0 passes, so every route to a pass (the agent stage with the triage template,
the stock-runtime stage, the text and audio runtimes) gets the record, and no template has to carry a copy of the table.

Where the configs come from. A stage 0 directory that holds `triage_config.json` (the delta-triage template's input) names both
snapshots. Any other stage 0 gets the model's directory from the run's `model` input and has no nearest config, and the scan then skips
the comparisons that need one and says so in `notes`.

The scan never stops a run. A model directory without a readable `config.json` (an audio or diffusion repository, a hostile file) gives a
record with `"scan": "skipped"` and the reason. Each hit carries the targeted check stage 2 should run first; stage 2's skill and the
operator bundle read the record from this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from orchard import hazards, model_config as mc, safeio

FILE = "hazard-scan.json"
MAX_TRIAGE_CONFIG_BYTES = 1 << 20


def _triage_dirs(stage_dir: Path) -> tuple[str | None, str | None]:
    """(model snapshot, nearest snapshot) named by the stage's triage_config.json, each None when absent or not a string."""
    try:
        cfg = safeio.read_json(stage_dir / "triage_config.json", max_bytes=MAX_TRIAGE_CONFIG_BYTES, expect=dict)
    except safeio.SafeReadError:
        return None, None
    pick = lambda k: cfg[k] if isinstance(cfg.get(k), str) and cfg[k] else None      # noqa: E731
    return pick("model_snapshot"), pick("nearest_snapshot")


def _config_of(directory) -> dict:
    # a Hugging Face cache snapshot's config.json is a link into blobs/, so the path is resolved before the guarded read
    try:
        resolved = os.path.realpath(Path(directory) / "config.json")
    except (ValueError, OSError) as exc:            # an embedded null byte, or a link loop
        raise mc.ModelConfigError("model config not usable: the path cannot be resolved") from exc
    return mc.read_model_config(resolved)


def build_record(stage_dir, model_dir=None, nearest_dir=None, *, dies: int = 2) -> dict:
    """The scan record for a stage directory. Never raises for an unreadable input: that is a record with `scan: skipped`."""
    stage_dir = Path(stage_dir)
    t_model, t_nearest = _triage_dirs(stage_dir)
    model_dir = model_dir or t_model
    nearest_dir = nearest_dir or t_nearest
    skipped = lambda why: {"label": "prediction from the config", "scan": "skipped", "reason": why, "hazards": [],   # noqa: E731
                           "facts": {}, "notes": [], "dies": dies}
    if not model_dir:
        return skipped("no model directory is known for this stage")
    try:
        model = _config_of(model_dir)
    except mc.ModelConfigError as exc:
        return skipped(f"the model's config.json could not be used: {exc}")
    nearest, note = None, None
    if nearest_dir:
        try:
            nearest = _config_of(nearest_dir)
        except mc.ModelConfigError as exc:
            note = f"the nearest model's config.json could not be used ({exc}), so the comparisons with it were skipped"
    if nearest is not None:
        try:
            mc.dims(nearest)
        except mc.ModelConfigError as exc:
            nearest, note = None, f"the nearest model's config.json has unusable dimensions ({exc}), so the comparisons with it were skipped"
    try:
        scan = hazards.scan_model_hazards(model, nearest, dies=dies)
    except hazards.HazardInputError as exc:
        return skipped(f"the model's config.json could not be used: {exc}")
    rec = scan.as_record()
    rec["scan"] = "ran"
    if note:
        rec["notes"].append(note)
    return rec


def write_scan(stage_dir, model_dir=None, nearest_dir=None, *, dies: int = 2) -> dict:
    """Build the record and write it to `stage_dir/hazard-scan.json` (all or nothing). Returns the record."""
    rec = build_record(stage_dir, model_dir, nearest_dir, dies=dies)
    safeio.atomic_write(Path(stage_dir) / FILE, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec
