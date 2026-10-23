#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Time the safe configuration, then serve the grafted variant and time it too. The supervisor runs this under a lease.

Stage 0 found that the new model has no `mtp.*` tensors and the nearest model has a head that fits (delta.json `graft_candidate`).
Stage 2 has already shown the safe configuration serves. This script answers one question for the supervisor: is the nearest model's head,
grafted onto the new model, better? It never decides. It writes `trial-result.json` next to itself, and orchard/graft_trial.py decides.

The supervisor copies this file into `stages/2/graft-trial/` and writes `trial_config.json` there: {"stage_dir": the stage 2 directory,
"orchard_dir": the tt-orchard checkout}. Everything else comes from the stage's `swap_config.json`. The shipped scripts are read from the
checkout, never from the stage directory, because an agent can edit the stage's copies.

Steps, each a child process whose output goes to `graft-trial/evidence/`:
  1. safe-speed: `orchard/measure_serving.py` on the stage's own run.sh, model directory and (warm) tensor cache.
  2. prepare: `prepare_swap.py` in `graft-trial/variant/`, with the stage's config plus `"graft_mtp": true` and a new tensor cache beside the
     stage's (`<tt_cache>-graft`). The cache marker rule of serve_and_compare.py then applies to it.
  3. serve: `serve_and_compare.py` in the same directory: the stage 2 check (top-1 agreement with the CPU reference, coherence), cold cache.
  4. variant-speed: `measure_serving.py` on the variant's run.sh and model directory.
The two speeds are taken on the same board in the same lease, with the same prompts (measure_serving's fixed set, 3 repetitions, greedy).

Exit 0 when `trial-result.json` was written, even if the variant failed: a failed trial is a result, and the safe configuration stands.
Exit 2 when the inputs are missing or unreadable, with no result written.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = "trial-result.json"
REPS = 3
TAIL_LINES = 12
SHIPPED = ("prepare_swap.py", "serve_and_compare.py")


def fail(message: str) -> None:
    print(f"graft_trial: {message}", file=sys.stderr)
    sys.exit(2)


def read_json(path: Path, what: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"{what} ({path}) cannot be read: {exc}")


def tail(path: Path, n: int = TAIL_LINES) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return " | ".join(x.strip() for x in lines[-n:] if x.strip())[:600]


def run_step(name: str, argv: list, cwd: Path, steps: list) -> tuple[int, Path]:
    """Run one child to completion (stdin closed) and record it. The child stays in this script's process group: the supervisor stops a
    test that passes its deadline by killing that group, and a child in its own session would outlive the stop. (The child's own server
    has its own session, as stage 2's does.)"""
    log = HERE / "evidence" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(log, "wb") as out:
        proc = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        code = proc.wait()
    steps.append({"name": name, "returncode": code, "seconds": round(time.monotonic() - t0, 1)})
    return code, log


def speed_summary(record: dict) -> dict:
    """measure_serving's record as the numbers the decision uses: each prompt's median decode speed and first-token time, and the median
    of the prompts' medians. The values are rounded (decode speed to 0.01 tokens/s, first-token time to 1 ms) so the card can show exactly
    what the file holds."""
    def rounded(v, key):
        digits = 2 if key == "decode_tok_s" else 3
        return round(v, digits) if isinstance(v, (int, float)) and not isinstance(v, bool) else v

    per = {}
    for pid, r in (record.get("prompts") or {}).items():
        s = r.get("summary") or {}
        per[pid] = {k: rounded((s.get(k) or {}).get("median"), k) for k in ("decode_tok_s", "ttft_s")}

    def med(key):
        vals = [v[key] for v in per.values() if isinstance(v[key], (int, float)) and not isinstance(v[key], bool)]
        return rounded(statistics.median(vals), key) if vals else None
    return {"decode_tok_s": med("decode_tok_s"), "ttft_s": med("ttft_s"), "server_ready_s": record.get("server_ready_s"),
            "per_prompt": per, "prompt_set_sha256": record.get("prompt_set_sha256"), "reps": record.get("reps")}


def measure(label: str, orchard: Path, cfg: dict, run_sh: Path, model_dir: Path, cache: Path, steps: list) -> dict:
    out = HERE / f"speed-{label}.json"
    out.unlink(missing_ok=True)
    code, log = run_step(f"{label}-speed", [sys.executable, orchard / "orchard" / "measure_serving.py", "--label", label,
                                            "--model-id", cfg["new_model_id"], "--run-sh", run_sh, "--model-dir", model_dir,
                                            "--tt-cache", cache, "--hf-home", cfg["hf_home"], "--port", cfg["port"],
                                            "--reps", REPS, "--out", out], HERE, steps)
    try:
        record = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if code != 0 or not isinstance(record, dict):
        return {"failure": f"measure_serving exited with code {code}" + (f": {tail(log)}" if tail(log) else "")}
    return speed_summary(record)


def build_variant(orchard: Path, cfg: dict, cache: Path, steps: list) -> tuple[Path | None, str | None]:
    """The variant directory: the shipped prepare_swap.py and serve_and_compare.py, and the stage's config with the graft on and the new cache."""
    variant = HERE / "variant"
    shutil.rmtree(variant, ignore_errors=True)
    variant.mkdir(parents=True)
    templates = orchard / "orchard" / "skills" / "weights-swap-templates"
    for name in SHIPPED:
        shutil.copy2(templates / name, variant / name)
    (variant / "swap_config.json").write_text(json.dumps({**cfg, "graft_mtp": True, "tt_cache": str(cache)}, indent=2) + "\n",
                                              encoding="utf-8")
    code, log = run_step("prepare", [sys.executable, variant / "prepare_swap.py"], variant, steps)
    if code != 0:
        return None, f"prepare_swap.py refused the graft (exit {code}): {tail(log)}"
    return variant, None


def check_variant(variant: Path, steps: list) -> dict:
    """The stage 2 check on the variant. Its report is evidence/swap-check.json; its exit code says whether the server started."""
    code, log = run_step("serve", [sys.executable, variant / "serve_and_compare.py"], variant, steps)
    try:
        draft = json.loads((variant / "evidence" / "swap-check.json").read_text(encoding="utf-8"))["result_draft"]
    except (OSError, ValueError, KeyError, TypeError):
        draft = None
    if code != 0 or not isinstance(draft, dict) or draft.get("serves") is not True:
        return {"served": False, "coherent": None, "top1_agreement": None, "n_tokens": None, "server_ready_s": None,
                "failure": f"serve_and_compare.py exited with code {code}" + (f": {tail(log)}" if tail(log) else "")}
    return {"served": True, "coherent": draft.get("coherent"), "top1_agreement": draft.get("top1_agreement"),
            "n_tokens": draft.get("n_tokens"), "server_ready_s": draft.get("server_ready_s"), "failure": None}


def main() -> int:
    conf = read_json(HERE / "trial_config.json", "trial_config.json")
    stage = Path(str(conf.get("stage_dir") or "")).resolve()
    orchard = Path(str(conf.get("orchard_dir") or "")).resolve()
    cfg = read_json(stage / "swap_config.json", "the stage's swap_config.json")
    for key in ("new_model_id", "hf_home", "port", "tt_cache", "run_dir"):
        if cfg.get(key) in (None, ""):
            fail(f"the stage's swap_config.json has no {key}")
    for path in (stage / "run.sh", stage / "model-dir", orchard / "orchard" / "measure_serving.py",
                 *(orchard / "orchard" / "skills" / "weights-swap-templates" / n for n in SHIPPED)):
        if not path.exists():
            fail(f"{path} does not exist")
    run_dir = Path(cfg["run_dir"]).resolve()
    # A killed earlier run can leave files that this run's evidence list would name without having made them.
    shutil.rmtree(HERE / "variant", ignore_errors=True)
    for stale in (HERE / "speed-safe.json", HERE / "speed-variant.json", HERE / RESULT):
        stale.unlink(missing_ok=True)
    steps: list = []
    result: dict = {"label": "measured", "new_model_id": cfg["new_model_id"], "reps": REPS, "steps": steps,
                    "unit_notes": {"decode_tok_s": "median over the prompts of each prompt's median (tokens after the first, over the time "
                                                   "after the first), greedy, one request at a time",
                                   "ttft_s": "median over the prompts of each prompt's median seconds to the first chunk"}}
    safe = measure("safe", orchard, cfg, stage / "run.sh", stage / "model-dir", Path(cfg["tt_cache"]), steps)
    result["safe"] = safe
    if "failure" in safe:
        result["variant"] = {"served": False, "coherent": None, "top1_agreement": None, "n_tokens": None, "server_ready_s": None,
                             "failure": "not tried: the safe configuration's speed was not measured"}
    else:
        cache = Path(cfg["tt_cache"])
        cache = cache.with_name(cache.name + "-graft")
        variant, problem = build_variant(orchard, cfg, cache, steps)
        if variant is None:
            result["variant"] = {"served": False, "coherent": None, "top1_agreement": None, "n_tokens": None,
                                 "server_ready_s": None, "failure": problem}
        else:
            v = check_variant(variant, steps)
            if v["served"]:
                speed = measure("variant", orchard, cfg, variant / "run.sh", variant / "model-dir", cache, steps)
                if "failure" in speed:
                    v = {**v, "decode_tok_s": None, "ttft_s": None, "failure": speed["failure"]}
                else:
                    v = {**v, "decode_tok_s": speed["decode_tok_s"], "ttft_s": speed["ttft_s"], "per_prompt": speed["per_prompt"]}
            result["variant"] = {**v, "tt_cache": str(cache)}
    rel = lambda p: os.path.relpath(p, run_dir)
    result["evidence"] = sorted(rel(p) for p in (HERE / "speed-safe.json", HERE / "speed-variant.json",
                                                 HERE / "variant" / "evidence" / "swap-check.json",
                                                 HERE / "variant" / "evidence" / "server.log",
                                                 HERE / "variant" / "model-dir-sources.json") if p.is_file())
    tmp = HERE / (RESULT + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, HERE / RESULT)                 # the result appears whole or not at all
    print(json.dumps({"safe": {k: safe.get(k) for k in ("decode_tok_s", "ttft_s", "failure")},
                      "variant": {k: result["variant"].get(k) for k in ("served", "coherent", "top1_agreement", "decode_tok_s", "ttft_s",
                                                                       "failure")}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
