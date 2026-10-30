# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Decode speed as a share of the DRAM-bandwidth floor, in stages 0, 2 and 4 and the operator bundle (issue 181).

`orchard/decode_floor.py` does the arithmetic. This module puts the pieces of a run together:

  * Stage 0 (`write_floor`) writes `stages/0/decode-floor.json`: the floor for 1, 2 and 4 dies of the board type in `BOARD`, with the
    model's counted shape, the dtype policy and the bandwidth, so a reader can redo it.
  * Stages 2 and 4 (`write_share`) write `decode-share.json` in the stage directory: each decode speed the stage measured, as a share of
    the floor for the dies it ran on.
  * The bundle (`share_rows`) recomputes the same rows from the raw measurement files, so a summary that someone wrote into a file is not
    what it reports.

What is an estimate. The floor is the datasheet bandwidth and the weights' stored dtype (`policy_for`). A bundle that keeps weights
on the chip in a smaller dtype has a higher floor, so the share here is then too high and never too low. A speed measured over a
whole request includes prefill and HTTP time, so its share is too low. Each row says which of these applies (`basis`). A speed above the
floor means an input is wrong or the run emits several tokens per weight read (a drafter).

Speeds come from files the template stages wrote and are read through `orchard/safeio.py`; an unreadable or malformed file gives no row.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from orchard import decode_floor as df
from orchard import defaults, model_config as mc, safeio, sampling_probe

FLOOR_FILE = "decode-floor.json"
SHARE_FILE = "decode-share.json"
BOARD = "p300"
DIES_OPTIONS = (1, 2, 4)
# Below this share the bundle adds the hint that names the likely causes. Choice: nobody measured a "good" share. The Qwen3.5-35B-A3B
# package reached 29.6 percent and MiniCPM3-4B 7 percent (docs/run-logs); 20 percent sits between them. The hint is text only.
LOW_SHARE_THRESHOLD = 0.20
LOW_SHARE_SOURCE = "choice (nobody measured a good share): between 7 percent (MiniCPM3-4B) and 29.6 percent (Qwen3.5-35B-A3B package), docs/run-logs"
MAX_RECORD_BYTES = 32 << 20          # a swap check holds reply texts; a megabyte would refuse real ones
MAX_SMALL_BYTES = 1 << 20

_DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
PROBE_BASIS = ("completion tokens over request seconds for a 96-token greedy request: it includes prefill and HTTP time, so the share is "
               "too low")
GRAFT_BASIS = "the graft trial's decode speed, median of four prompts"
GRAFT_DRAFTER_BASIS = GRAFT_BASIS + "; the grafted variant keeps the drafter on, which can emit several tokens per weight read"


# ---- the dtype policy ------------------------------------------------------------------------------------------------------

def policy_for(config) -> tuple[dict, str]:
    """(dtype policy, where it came from). The policy is the weights' stored dtype from the config's `torch_dtype` (or `dtype`), or
    bf16 when the config names none or one the floor has no width for (an FP8 checkpoint, for one). The source says which."""
    try:
        flat = mc.text_config(config)
    except mc.ModelConfigError:
        flat = {}
    for key in ("torch_dtype", "dtype"):
        v = flat.get(key) if isinstance(flat.get(key), str) else (config.get(key) if isinstance(config, dict) and isinstance(config.get(key), str) else None)
        if v in _DTYPES:
            return {"default": _DTYPES[v]}, f"the config's {key} {v}"
    return {"default": "bf16"}, "assumed bf16: the config names no weight dtype the floor has a width for"


# ---- stage 0 ---------------------------------------------------------------------------------------------------------------

def _triage_model_dir(stage_dir) -> str | None:
    try:
        cfg = safeio.read_json(Path(stage_dir) / "triage_config.json", max_bytes=MAX_SMALL_BYTES, expect=dict)
    except safeio.SafeReadError:
        return None
    v = cfg.get("model_snapshot")
    return v if isinstance(v, str) and v else None


def _skipped(why: str) -> dict:
    return {"label": "estimate", "scan": "skipped", "reason": why, "floors": {}}


def build_record(*, model_config=None, model_dir=None, stage_dir=None) -> dict:
    """The stage 0 record. Never raises for an input it cannot count: that is a record with `scan: skipped` and the reason."""
    if model_config is None:
        model_dir = model_dir or (_triage_model_dir(stage_dir) if stage_dir else None)
        if not model_dir:
            return _skipped("no model directory is known for this stage")
        try:
            # a Hugging Face cache snapshot's config.json is a link into blobs/, so the path is resolved before the guarded read
            model_config = mc.read_model_config(os.path.realpath(Path(model_dir) / "config.json"))
        except mc.ModelConfigError as exc:
            return _skipped(f"the model's config.json could not be used: {exc}")
    policy, policy_source = policy_for(model_config)
    board = df.BOARDS[BOARD]
    floors = {}
    try:
        for dies in DIES_OPTIONS:
            floors[str(dies)] = df.decode_floor(model_config, policy, dies, board.per_die_gb_s, board=BOARD).as_dict()
    except (df.FloorInputError, mc.ModelConfigError) as exc:
        return _skipped(f"the decode floor could not be counted: {exc}")
    return {"label": "estimate", "scan": "ran", "board": BOARD, "dtype_policy": policy, "dtype_policy_source": policy_source,
            "bandwidth_source": board.source, "floors": floors, "assumptions": df.ASSUMPTIONS}


def write_floor(stage_dir, model_dir=None) -> dict:
    """Build the stage 0 record and write it to `stage_dir/decode-floor.json` (all or nothing). Returns the record."""
    rec = build_record(model_dir=model_dir, stage_dir=stage_dir)
    safeio.atomic_write(Path(stage_dir) / FLOOR_FILE, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def read_floor(run_dir) -> dict | None:
    """The stage 0 record of a run when it ran and every floor in it is usable, else None."""
    try:
        rec = safeio.read_json(Path(run_dir) / "stages" / "0" / FLOOR_FILE, max_bytes=MAX_RECORD_BYTES, expect=dict)
    except safeio.SafeReadError:
        return None
    floors = rec.get("floors")
    if rec.get("scan") != "ran" or not isinstance(floors, dict) or not floors:
        return None
    for v in floors.values():
        t = v.get("floor_tok_s") if isinstance(v, dict) else None
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
            return None
    return rec


# ---- the measured speeds ---------------------------------------------------------------------------------------------------

def _good(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _json(path: Path, limit: int = MAX_RECORD_BYTES):
    try:
        return safeio.read_json(path, max_bytes=limit, expect=dict)
    except safeio.SafeReadError:
        return None


def _chips(path: Path, default: int) -> int:
    cfg = _json(path, MAX_SMALL_BYTES)
    n = cfg.get("chips") if cfg else None
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else default


def _probe_speed(path: Path):
    report = _json(path)
    if not report:
        return None
    try:
        s = sampling_probe.summarize(report.get("sampling_probe"))
    except Exception:           # a malformed record must give no row, whatever shape it has
        return None
    side = s.get("greedy") if isinstance(s, dict) else None
    v = side.get("tokens_per_s") if isinstance(side, dict) else None
    return v if _good(v) else None


def measured(run_dir) -> list[dict]:
    """Every decode speed the run measured that has a floor to compare with: stage, source, evidence path, dies, tok/s and what the speed is."""
    run = Path(run_dir)
    out = []
    rel = "stages/2/evidence/swap-check.json"
    v = _probe_speed(run / rel)
    dies2 = _chips(run / "stages/2/swap_config.json", defaults.CHIPS_PER_BOARD)
    if v:
        out.append({"stage": 2, "source": "stage 2 sampling probe, greedy request", "evidence": rel, "dies": dies2, "tok_s": v, "basis": PROBE_BASIS})
    configs = run / "stages" / "4" / "configs"
    try:
        names = sorted((p.name for p in configs.iterdir() if p.name.isdigit()), key=int) if configs.is_dir() else []
    except OSError:
        names = []
    for name in names:
        rel = f"stages/4/configs/{name}/evidence/swap-check.json"
        v = _probe_speed(run / rel)
        if v:
            out.append({"stage": 4, "source": f"stage 4 sampling probe, greedy request, {int(name)} chips", "evidence": rel,
                        "dies": _chips(run / "stages/4/configs" / name / "swap_config.json", int(name)), "tok_s": v, "basis": PROBE_BASIS})
    decision = _json(run / "stages/2/graft-decision.json")
    nums = decision.get("numbers") if decision and isinstance(decision.get("numbers"), dict) else {}
    for key, label, basis in (("safe_decode_tok_s", "stage 2 graft trial, safe configuration", GRAFT_BASIS),
                              ("variant_decode_tok_s", "stage 2 graft trial, grafted variant", GRAFT_DRAFTER_BASIS)):
        if _good(nums.get(key)):
            out.append({"stage": 2, "source": label, "evidence": "stages/2/graft-decision.json", "dies": dies2, "tok_s": nums[key], "basis": basis})
    return out


def share_rows(run_dir, *, threshold: float = LOW_SHARE_THRESHOLD) -> list[dict]:
    """The measured speeds as shares of the stage 0 floor for their die counts. No rows when the run has no usable floor record. A speed
    on a die count the record has no floor for is left out (`share_notes` says so)."""
    rec = read_floor(run_dir)
    if rec is None:
        return []
    rows = []
    for m in measured(run_dir):
        f = rec["floors"].get(str(m["dies"]))
        if f is None:
            continue
        share = m["tok_s"] / f["floor_tok_s"]
        note = ("the speed is above the floor, so an input is wrong (a smaller weight dtype on the chip, more dies) or the run emits several tokens "
                "per weight read (a drafter)" if share > 1.0 else None)
        rows.append({**m, "floor_tok_s": f["floor_tok_s"], "floor_ms": f["floor_ms"], "bytes_per_token": f["bytes_per_token"], "share": share,
                     "note": note, "hint": df.low_share_hint(share, threshold=threshold)})
    return rows


def share_notes(run_dir) -> list[str]:
    """Sentences about speeds that got no row."""
    rec = read_floor(run_dir)
    if rec is None:
        return []
    have = set(rec["floors"])
    return [f"{m['source']} ran on {m['dies']} dies, and the stage 0 record has floors only for {', '.join(sorted(have, key=int))}; it has no share."
            for m in measured(run_dir) if str(m["dies"]) not in have]


def write_share(run_dir, stage_dir, stage: int) -> dict | None:
    """Write `stage_dir/decode-share.json` for stage 2 or 4 from the run's measurement files. None (nothing written) when the stage
    has no speed with a floor."""
    rows = [r for r in share_rows(run_dir) if r["stage"] == stage]
    if not rows:
        return None
    floor = read_floor(run_dir)
    rec = {"label": "estimate", "stage": stage, "rows": rows, "inputs": {k: floor[k] for k in ("board", "dtype_policy", "dtype_policy_source",
                                                                                              "bandwidth_source")},
           "threshold": LOW_SHARE_THRESHOLD, "threshold_source": LOW_SHARE_SOURCE, "assumptions": floor["assumptions"],
           "floor_record": f"stages/0/{FLOOR_FILE}"}
    safeio.atomic_write(Path(stage_dir) / SHARE_FILE, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec
