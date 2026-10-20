# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The plain sentences the operator view shows above the raw numbers.

A number with no yardstick tells the operator nothing. Each verdict here is one sentence plus the yardstick: the bar from the code
(orchard/defaults.py) and the earlier runs on this machine. Two rules keep the sentences honest:

* A number is compared only with numbers that can be compared. A boot time is ranked only against runs of the same package on the
  same chip count. Stage 2's boot time is stated and never ranked, because its caches are sometimes warm and the result does not say.
* The reference line reads the evidence the reference template wrote (`summary.json`, `load-report.json`) and sets it beside the
  verdict in `reference.json`, the file the stage agent writes. On the pplx-decider run the template said `fail` (every weight
  missing, so the CPU model ran on uninitialised weights) and the agent wrote `pass`; the gate took the agent's word.

`build` is pure: it takes `run_view.results` for this run, the same for the other runs, and the reference check, and returns a list of
{id, tone, headline, detail}. `reference_check` is the only part that reads files.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from orchard.defaults import SWAP_MIN_TOKENS, SWAP_TOP1_MIN

TONES = ("bad", "warn", "good", "info")      # the order the page lists them: worst news first
MARGIN = 0.05                                # choice: a pass this close to the bar is called a thin margin (one token of 32 is 0.03)
TYPICAL_FACTOR = 1.15                        # choice: up to 15% over the median of earlier runs still counts as the same
SLOW_FACTOR = 1.5                            # choice: more than 50% over the median is called slow
MIN_PEERS = 2                                # choice: the fewest comparable earlier runs a boot time is ranked against
MAX_PEERS_SHOWN = 4
MAX_FILE_BYTES = 2 * 1024 * 1024


def _num(x) -> bool:
    """A real measurement: an int or a finite float. `json.loads` accepts NaN and Infinity, and a number that is not finite
    would print as "nan" in a sentence or make `round` raise. An int of any size counts (`math.isfinite` overflows on a huge one)."""
    if isinstance(x, bool):
        return False
    return isinstance(x, int) or (isinstance(x, float) and math.isfinite(x))


def _short(model) -> str:
    return str(model or "an earlier run").rsplit("/", 1)[-1].split("@", 1)[0]


def _list(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _secs(x: float) -> str:
    return f"{x:,.0f} s"


def _item(id_: str, tone: str, headline: str, detail: str = "") -> dict:
    return {"id": id_, "tone": tone, "headline": headline, "detail": detail}


def _worst(*tones: str) -> str:
    return min(tones, key=TONES.index)


# ---- the reference check: evidence against the agent's say-so ---------------------------------------------------------------------

def _read_json(run: Path, rel: str):
    """A small JSON file inside the run. A link that leads outside the run, a file over the cap or bad JSON reads as absent."""
    path = run / rel
    try:
        if not path.is_file() or not path.resolve().is_relative_to(run.resolve()) or path.stat().st_size > MAX_FILE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _missing_keys(report) -> tuple[int | None, int | None]:
    """(missing backbone keys, model keys). Keys under one of the report's own ignore patterns do not count."""
    if not isinstance(report, dict) or not isinstance(report.get("missing_keys"), list):
        return None, None
    patterns = []
    for p in report.get("ignore_patterns") or []:
        try:
            patterns.append(re.compile(p))
        except (re.error, TypeError):
            continue
    missing = [k for k in report["missing_keys"] if isinstance(k, str) and not any(p.search(k) for p in patterns)]
    keys = report.get("model_key_count")
    return len(missing), keys if isinstance(keys, int) and not isinstance(keys, bool) else None


def reference_check(run_dir) -> dict | None:
    """What stage 1 says of itself and what its template measured, or None when stage 1 left nothing to read."""
    run = Path(run_dir)
    ref = _read_json(run, "stages/1/reference.json")
    measured = _read_json(run, "stages/1/evidence/summary.json")
    if ref is None and measured is None:
        return None
    missing, keys = _missing_keys(_read_json(run, "stages/1/evidence/load-report.json"))
    verdict = lambda d: d.get("verdict") if isinstance(d, dict) and isinstance(d.get("verdict"), str) else None   # noqa: E731
    return {"claimed": verdict(ref), "measured": verdict(measured), "missing": missing, "keys": keys}


def _reference_item(ref: dict | None) -> dict | None:
    if not ref:
        return None
    claimed, measured, missing, keys = ref["claimed"], ref["measured"], ref["missing"], ref["keys"]
    counts = f"{missing} of {keys}" if missing is not None and keys is not None else None
    if measured == "fail" and claimed == "pass":
        detail = (f"{counts} model keys were not found in the checkpoint, so the CPU model may have run on uninitialised weights. "
                  if counts else "") + "Read stage 1's evidence before trusting any later comparison with this reference."
        return _item("reference", "bad", "The reference's own check failed, but reference.json says pass (evidence/summary.json says fail).", detail)
    if measured == "fail":
        return _item("reference", "bad", "The CPU reference failed its own checks.",
                     (f"{counts} model keys were not found in the checkpoint. " if counts else "") + "See stage 1's evidence/summary.json.")
    if measured == "pass":
        if missing:
            return _item("reference", "warn", "The CPU reference passed, but some weights did not load.",
                         f"{counts} model keys were not found in the checkpoint (outside its ignore patterns).")
        return _item("reference", "good", "The CPU reference loaded every weight.",
                     f"{keys} model keys, none missing." if keys is not None else "The load report lists no missing key.")
    return _item("reference", "warn", "Only the agent's reference.json exists; the template's own evidence is missing.",
                 "Without evidence/summary.json the reference's verdict cannot be checked.")


# ---- the sentences ----------------------------------------------------------------------------------------------------------------

def _agreement_tone(top1: float) -> str:
    return "bad" if top1 < SWAP_TOP1_MIN else "warn" if top1 < SWAP_TOP1_MIN + MARGIN else "good"


def _swap_items(swap: dict, peers: list[dict]) -> list[dict]:
    top1, n, ready = swap.get("top1_agreement"), swap.get("n_tokens"), swap.get("server_ready_s")
    items = []
    if _num(top1):
        n_ok = isinstance(n, int) and not isinstance(n, bool) and n > 0
        matches = f"{round(top1 * n)} of {n} tokens" if n_ok else "its tokens"
        earlier = [f"{p['results']['swap']['top1_agreement']:.2f} ({_short(p.get('model'))})" for p in peers
                   if _num(((p.get("results") or {}).get("swap") or {}).get("top1_agreement"))][:MAX_PEERS_SHOWN]
        yard = f"The bar is {SWAP_TOP1_MIN:.2f}. " + (f"Earlier runs here scored {_list(earlier)}." if earlier
                                                     else "No earlier run here to compare with.")
        rests = (f" This rests on one prompt of {n} tokens, so one token moves the score by {1 / n:.2f}." if n_ok else "")
        if swap.get("coherent") is False:
            items.append(_item("swap", "bad", f"The chips' output is not readable (agreement {top1:.2f}).", yard))
        elif top1 < SWAP_TOP1_MIN:
            items.append(_item("swap", "bad", f"The chips match the CPU reference on {matches} ({top1:.2f}), below the bar of {SWAP_TOP1_MIN:.2f}.", yard))
        elif n_ok and n < SWAP_MIN_TOKENS:
            items.append(_item("swap", "warn", f"The chips match the CPU reference on {matches} ({top1:.2f}).",
                               f"Only {n} tokens were compared; the gate asks for at least {SWAP_MIN_TOKENS}. {yard}"))
        elif top1 < SWAP_TOP1_MIN + MARGIN:
            items.append(_item("swap", "warn", f"The chips match the CPU reference on {matches} ({top1:.2f}), with a thin margin over the bar.", yard + rests))
        else:
            items.append(_item("swap", "good", f"The chips match the CPU reference on {matches} ({top1:.2f}).", yard + rests))
    if _num(ready) and ready > 0:
        items.append(_item("swap-boot", "info", f"Server ready in {_secs(ready)} (stage 2).",
                           "Not compared with earlier runs: stage 2 boots mix cold and warm tensor caches."))
    return items


def _boot(chips, package, ready, peers: list[dict]) -> tuple[str, str]:
    """(tone, sentence) for one configuration's boot time, against earlier runs of the same package on the same chip count."""
    if package is None:
        return "good", (f"The package for this configuration was not recorded, so {_secs(ready)} on {chips} chips is not ranked "
                        "against earlier runs.")
    same = []
    for p in peers:
        for c in (p.get("results") or {}).get("configs") or []:
            if c.get("chips") == chips and c.get("package") == package and c.get("pass") is True and _num(c.get("server_ready_s")):
                same.append(c["server_ready_s"])
    if len(same) < MIN_PEERS:
        return "good", f"Not enough earlier runs of this package on {chips} chips to say whether {_secs(ready)} is typical ({len(same)} found)."
    med = statistics.median(same)
    word = ("quicker than" if ready < med / TYPICAL_FACTOR else "about the same as" if ready <= med * TYPICAL_FACTOR
            else "somewhat slower than" if ready <= med * SLOW_FACTOR else "slower than")
    shown = _list([_secs(x) for x in same[:MAX_PEERS_SHOWN]])
    return ("warn" if ready > med * SLOW_FACTOR else "good"), \
        f"The boot is {word} the {len(same)} earlier runs of this package on {chips} chips ({shown}; median {_secs(med)})."


def _config_item(c: dict, peers: list[dict]) -> dict | None:
    chips = c.get("chips")
    if c.get("pass") is None:
        return None                                               # not run (an optional configuration)
    if c.get("pass") is not True:
        return _item(f"config-{chips}", "bad", f"{chips} chips: failed.", str(c.get("reason") or "No reason was recorded."))
    top1, ready = c.get("top1_agreement"), c.get("server_ready_s")
    parts, detail, tones = [], [], []
    if _num(top1):
        parts.append(f"top-1 agreement {top1:.2f}")
        detail.append(f"The bar is {SWAP_TOP1_MIN:.2f}.")
        tones.append(_agreement_tone(top1))
    if _num(ready) and ready > 0:
        parts.append(f"ready in {_secs(ready)}")
        tone, sentence = _boot(chips, c.get("package"), ready, peers)
        detail.append(sentence)
        tones.append(tone)
    if not parts:
        return None
    return _item(f"config-{chips}", _worst(*tones) if tones else "info", f"{chips} chips: " + ", ".join(parts) + ".", " ".join(detail))


def _triage_item(t: dict) -> dict | None:
    cls = t.get("class")
    if cls in ("weights-only", "weights+sidecar"):
        return _item("triage", "info", "Only the weights differ: the installed bundle can serve this model." if cls == "weights-only"
                     else "The weights differ and the model has a sidecar head the bundle does not serve.", f"Class {cls}.")
    if cls:
        return _item("triage", "warn", f"Needs new model code: class {cls}, path {t.get('path') or 'unknown'}.",
                     "An unattended run stops here; a full port is a person's decision.")
    return None


def build(results: dict, peers: list[dict], reference: dict | None = None) -> list[dict]:
    """The verdict lines for one run, worst news first. `peers` are the other runs: {"model", "results"} each."""
    items: list[dict] = []
    ref = _reference_item(reference)
    if ref:
        items.append(ref)
    if isinstance(results.get("triage"), dict):
        t = _triage_item(results["triage"])
        if t:
            items.append(t)
    swap = results.get("swap") if isinstance(results.get("swap"), dict) else {}
    items.extend(_swap_items(swap, peers))
    configs = [c for c in results.get("configs") or [] if isinstance(c, dict)]
    for c in configs:
        item = _config_item(c, peers)
        if item:
            items.append(item)
    ran_on_a_chip = _num(swap.get("top1_agreement")) or any(c.get("pass") is True for c in configs)
    if ran_on_a_chip and "speed" not in results:
        items.append(_item("speed", "info", "Not measured: generation speed (tokens per second).",
                           "These checks show the output is right. They do not time generation."))
    return sorted(items, key=lambda i: TONES.index(i["tone"]))
