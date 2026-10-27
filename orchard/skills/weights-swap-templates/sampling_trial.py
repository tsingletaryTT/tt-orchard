#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Serve the drafter-off variant of stage 2's configuration once and run the stage 2 check on it, sampling probe included. The supervisor runs this
under a lease.

Stage 2 served a configuration with the speculative drafter on, and its sampling probe did not show that sampling is applied (the drafter verifies by
argmax, so a request with temperature above 0 can come back greedy, issue 107). The model needs sampling. This script answers one question for the
supervisor: how does the same bundle behave with the drafter off and the host sampling? It never decides. It writes `trial-result.json` next to
itself, and orchard/sampling_choice.py decides from that and from the variant's own `swap-check.json`.

The supervisor copies this file into `stages/2/sampling-trial/` and writes `trial_config.json` there: {"stage_dir": the stage 2 directory,
"orchard_dir": the tt-orchard checkout}. Everything else comes from the stage's `swap_config.json`. The shipped scripts are read from the checkout, never
from the stage directory, because an agent can edit the stage's copies.

Steps, each a child process whose output goes to `sampling-trial/evidence/`:
  1. prepare: `prepare_swap.py` in `sampling-trial/variant/`, with the stage's config plus `"drafter": "off"` (no `graft_mtp`: prepare_swap refuses
     both), `"sampling_probe": true` and a new tensor cache beside the stage's (`<tt_cache>-nodraft`).
  2. serve: `serve_and_compare.py` in the same directory: the stage 2 check (top-1 agreement with the CPU reference, coherence, the prompt suite and
     the sampling probe), cold cache.

Exit 0 when `trial-result.json` was written, even if the variant failed: a failed trial is a result. Exit 2 when the inputs are missing or unreadable,
with no result written. This script imports nothing from orchard.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = "trial-result.json"
TAIL_LINES = 12
SHIPPED = ("prepare_swap.py", "serve_and_compare.py")
SUITE_FILE = "prompt_suite.json"
CACHE_SUFFIX = "-nodraft"


def shipped(cfg: dict) -> tuple[str, ...]:
    """The files the variant needs next to its scripts: the two swap scripts, and the prompt suite when the stage's config asks for it."""
    return SHIPPED + ((SUITE_FILE,) if cfg.get("prompt_suite") else ())


def fail(message: str) -> None:
    print(f"sampling_trial: {message}", file=sys.stderr)
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
    """Run one child to completion (stdin closed) and record it. The child stays in this script's process group: the supervisor stops a test that
    passes its deadline by killing that group, and a child in its own session would outlive the stop. (The child's own server has its own session,
    as stage 2's does.)"""
    log = HERE / "evidence" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(log, "wb") as out:
        proc = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        code = proc.wait()
    steps.append({"name": name, "returncode": code, "seconds": round(time.monotonic() - t0, 1)})
    return code, log


def build_variant(orchard: Path, cfg: dict, cache: Path, steps: list) -> tuple[Path | None, str | None]:
    variant = HERE / "variant"
    shutil.rmtree(variant, ignore_errors=True)
    variant.mkdir(parents=True)
    templates = orchard / "orchard" / "skills" / "weights-swap-templates"
    for name in shipped(cfg):
        shutil.copy2(templates / name, variant / name)
    keep = {k: v for k, v in cfg.items() if k != "graft_mtp"}
    (variant / "swap_config.json").write_text(
        json.dumps({**keep, "drafter": "off", "sampling_probe": True, "tt_cache": str(cache)}, indent=2) + "\n", encoding="utf-8")
    code, log = run_step("prepare", [sys.executable, variant / "prepare_swap.py"], variant, steps)
    if code != 0:
        return None, f"prepare_swap.py refused the drafter-off variant (exit {code}): {tail(log)}"
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
    for path in (*(orchard / "orchard" / "skills" / "weights-swap-templates" / n for n in shipped(cfg)),):
        if not path.exists():
            fail(f"{path} does not exist")
    run_dir = Path(cfg["run_dir"]).resolve()
    steps: list = []
    cache = Path(cfg["tt_cache"])
    cache = cache.with_name(cache.name + CACHE_SUFFIX)
    variant, problem = build_variant(orchard, cfg, cache, steps)
    if variant is None:
        v = {"served": False, "coherent": None, "top1_agreement": None, "n_tokens": None, "server_ready_s": None, "failure": problem}
    else:
        v = check_variant(variant, steps)
    result = {"label": "measured", "new_model_id": cfg["new_model_id"], "steps": steps, "variant": {**v, "tt_cache": str(cache)}}
    rel = lambda p: os.path.relpath(p, run_dir)
    result["evidence"] = sorted(rel(p) for p in (HERE / "variant" / "evidence" / "swap-check.json", HERE / "variant" / "evidence" / "server.log",
                                                 HERE / "variant" / "model-dir-sources.json") if p.is_file())
    tmp = HERE / (RESULT + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, HERE / RESULT)                 # the result appears whole or not at all
    print(json.dumps({"variant": {k: result["variant"].get(k) for k in ("served", "coherent", "top1_agreement", "failure")}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
