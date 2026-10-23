# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The graft trial: stage 2 tries the grafted drafter once and keeps it only when a measurement says it is better.

Stage 0 records a `graft_candidate` when the new model has no `mtp.*` tensors and the nearest model has a head that fits (delta.json). The
nearest bundle's drafter needs that head, so a model without one is served with the drafter off and sampling on the host. Grafting the
nearest model's head back in (prepare_swap.py's `graft_mtp`) turns the drafter and on-device sampling on. Whether that helps a given
model is a measurement, not something analysis can say: the head was trained against the nearest model's hidden states.

After the safe configuration passes the stage 2 gate, the supervisor runs `skills/weights-swap-templates/graft_trial.py` under a lease. It
times the safe configuration, builds the grafted variant in `graft-trial/variant/`, checks it against the CPU reference the way stage 2 does
and times it. This module reads what that script wrote (`graft-trial/trial-result.json`), decides, writes `graft-decision.json`, and checks
a decision again from the files on disk. The decision file is the only place later stages (4, 7 and 8) learn the choice from.

The decision never turns the graft on from analysis alone, and a trial that fails or times out is not a stage failure: the safe result
stands and the reason is recorded.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from orchard.defaults import GRAFT_MIN_SPEEDUP, GRAFT_TOP1_MARGIN, SWAP_TOP1_MIN

DECISION_FILE = "graft-decision.json"        # in stages/2/
TRIAL_DIR = "graft-trial"                    # in stages/2/
TRIAL_RESULT = "trial-result.json"           # in TRIAL_DIR, written by the template script
CHOICES = ("graft", "safe")


def candidate(delta) -> dict | None:
    """Stage 0's graft candidate when it is worth a trial: the tensors are graftable and their width matches the model's hidden size.
    A candidate whose fit is unknown (`shapes_match` null) is not tried."""
    c = delta.get("graft_candidate") if isinstance(delta, dict) else None
    if isinstance(c, dict) and c.get("graftable") is True and c.get("shapes_match") is True:
        return c
    return None


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _positive(v) -> bool:
    return _number(v) and v > 0


def _thresholds() -> dict:
    return {"min_speedup": GRAFT_MIN_SPEEDUP, "top1_margin": GRAFT_TOP1_MARGIN, "top1_min": SWAP_TOP1_MIN}


def _text(v, limit: int = 300) -> str:
    return " ".join(str(v).split())[:limit]


def decide(trial, *, safe_top1) -> dict:
    """{"choice": "graft" | "safe", "reason", "numbers", "thresholds"} from the trial's result and the safe run's top-1 agreement.

    The graft is chosen when the variant served, was coherent, agreed with the CPU reference at least as well as stage 2's bar, lost no more
    than the margin against the safe run, and decoded at least GRAFT_MIN_SPEEDUP times faster. The first failing check is the reason."""
    safe = trial.get("safe") if isinstance(trial, dict) else None
    var = trial.get("variant") if isinstance(trial, dict) else None
    safe = safe if isinstance(safe, dict) else {}
    var = var if isinstance(var, dict) else {}
    safe_speed, var_speed = safe.get("decode_tok_s"), var.get("decode_tok_s")
    top1 = var.get("top1_agreement")
    speedup = var_speed / safe_speed if _positive(safe_speed) and _positive(var_speed) else None
    numbers = {"safe_top1_agreement": safe_top1 if _number(safe_top1) else None,
               "variant_top1_agreement": top1 if _number(top1) else None,
               "safe_decode_tok_s": safe_speed if _positive(safe_speed) else None,
               "variant_decode_tok_s": var_speed if _positive(var_speed) else None,
               "speedup": speedup,
               "safe_ttft_s": safe.get("ttft_s") if _positive(safe.get("ttft_s")) else None,
               "variant_ttft_s": var.get("ttft_s") if _positive(var.get("ttft_s")) else None,
               "variant_server_ready_s": var.get("server_ready_s") if _positive(var.get("server_ready_s")) else None}
    reason = None
    if not var:
        reason = "the graft trial wrote no result for the grafted variant"
    elif var.get("served") is not True:
        why = f": {_text(var['failure'])}" if var.get("failure") else ""
        reason = f"the grafted variant did not serve{why}"
    elif var.get("coherent") is not True:
        reason = "the grafted variant's free-run text was not coherent"
    elif not _number(top1) or not 0 <= top1 <= 1:
        reason = f"the grafted variant's top-1 agreement is not a fraction from 0 to 1 ({top1!r})"
    elif top1 < SWAP_TOP1_MIN:
        reason = f"the grafted variant's top-1 agreement {top1} is below stage 2's minimum of {SWAP_TOP1_MIN}"
    elif _number(safe_top1) and top1 < safe_top1 - GRAFT_TOP1_MARGIN:
        reason = (f"the grafted variant's top-1 agreement {top1} is more than the margin of {GRAFT_TOP1_MARGIN} below the safe "
                  f"configuration's {safe_top1}")
    elif not _positive(safe_speed):
        why = f": {_text(safe['failure'])}" if safe.get("failure") else ""
        reason = f"the safe configuration's speed was not measured{why}, so the graft cannot be compared with it"
    elif not _positive(var_speed):
        reason = "the grafted variant's decode speed was not measured"
    elif speedup < GRAFT_MIN_SPEEDUP:
        reason = (f"the grafted variant decodes only {speedup:.2f} times as fast as the safe configuration ({var_speed:g} against "
                  f"{safe_speed:g} tokens/s); the graft is kept at {GRAFT_MIN_SPEEDUP:g} times or more")
    if reason is None:
        reason = (f"the grafted variant decodes {speedup:.1f} times as fast as the safe configuration ({var_speed:g} against {safe_speed:g} "
                  f"tokens/s) with top-1 agreement {top1}" + (f" (safe configuration {safe_top1})" if _number(safe_top1) else ""))
        return {"choice": "graft", "reason": reason, "numbers": numbers, "thresholds": _thresholds()}
    return {"choice": "safe", "reason": reason, "numbers": numbers, "thresholds": _thresholds()}


def write_decision(stage_dir, decision: dict, *, candidate: dict | None, trial: bool = True) -> Path:
    """Write stages/2/graft-decision.json whole or not at all. It carries the candidate stage 0 recorded and where the trial's numbers are
    (`trial` is None when no trial ran: the run opted out, or the trial cannot run there)."""
    sd = Path(stage_dir)
    record = {**decision, "candidate": candidate, "trial": f"{TRIAL_DIR}/{TRIAL_RESULT}" if trial else None}
    path = sd / DECISION_FILE
    tmp = sd / (DECISION_FILE + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_decision(stage_dir) -> dict | None:
    """The decision file, or None when it is missing, unreadable, not an object or names no known choice."""
    try:
        d = json.loads((Path(stage_dir) / DECISION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("choice") in CHOICES else None


def chosen(run_dir) -> bool:
    """Did stage 2 choose the graft? False for a missing or malformed decision, so nothing downstream grafts without a recorded decision."""
    d = read_decision(Path(run_dir) / "stages" / "2")
    return d is not None and d["choice"] == "graft"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def decision_problems(stage_dir, run_dir) -> list[str]:
    """What is wrong with stage 2's decision file, for the stage 2 gate. No file is no problem (the trial is skipped when there is no
    candidate, or the run opted out). A graft decision is recomputed from the trial's result on disk and the safe run's top-1 agreement in
    result.json, so a file that says graft while its numbers say otherwise is refused. A kept-safe decision needs a reason."""
    sd = Path(stage_dir)
    path = sd / DECISION_FILE
    if not path.exists():
        return []
    d = _read_json(path)
    if not isinstance(d, dict):
        return [f"{DECISION_FILE} is not a JSON object"]
    if d.get("choice") not in CHOICES:
        return [f"{DECISION_FILE} choice must be one of {', '.join(CHOICES)}, got {d.get('choice')!r}"]
    reason = d.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return [f"{DECISION_FILE} needs a reason for the choice"]
    if d["choice"] == "safe":
        return []
    trial_path = sd / TRIAL_DIR / TRIAL_RESULT
    trial = _read_json(trial_path)
    if not isinstance(trial, dict):
        return [f"{DECISION_FILE} chose the graft, but the trial's {TRIAL_DIR}/{TRIAL_RESULT} is missing or unreadable"]
    result = _read_json(sd / "result.json")
    safe_top1 = result.get("top1_agreement") if isinstance(result, dict) else None
    again = decide(trial, safe_top1=safe_top1)
    if again["choice"] != "graft":
        return [f"{DECISION_FILE} chose the graft, but the trial's numbers do not support it: {again['reason']}"]
    return []
