# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The numeric half of the text-model route's stages (the stage table is in orchard/stages.py; the adapter in orchard/text_runtime.py).

    1  CPU reference        supervisor code, no chip: the float32 run (it chooses the greedy path), two bfloat16 runs that follow the path (the floor), and the
                            float32 perplexity of a fixed text window
    2  one chip             supervisor code with one hardware test: for each variant, the teacher-forced chip run and the chip's perplexity on the same window;
                            then the comparison with stage 1 under the rule in orchard/text_bars.py
    6  benchmark            supervisor code with one hardware test: for each variant, one warm-up run and at least three measured runs in one process
                            (docs/benchmark-rules.md), and the decode share of the DRAM-bandwidth floor
    3 to 5, 7              skipped or built elsewhere, see orchard/stages.py

Everything a gate relies on is re-derived from the saved arrays and JSON files. The result files hold the numbers for people to read and for the bundle; a
gate recomputes each one and refuses a file that disagrees. This module needs numpy; the supervisor reaches it through orchard/text_remote.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from orchard import audio_reference as ar
from orchard import decode_floor as df
from orchard import model_config as mc
from orchard import resident, safeio
from orchard import text_bars as tb
from orchard import text_reference as tr
from orchard import text_runtime as rt
from orchard.text_runtime import (BENCH_FILE, CHIP_CONFIG_FILE, REFERENCE_DIRS, REFERENCE_FILE, RESULT_FILE, ROLE_FLOOR, ROLE_FLOOR_NATIVE, ROLE_REFERENCE,
                                  SCHEMA, STAGE_CONFIG_FILE, TextStageError, load_json, write_json)

PPL_REFERENCE_FILE = "ppl_reference.json"
MIN_DISTINCT_TOKENS = 2        # a greedy path with fewer distinct tokens is a loop or a stuck model, not a reference to compare with


def agree(recorded, fresh) -> bool:
    """A recorded summary equals one computed again now (numbers to 1e-9, everything else exactly)."""
    if isinstance(recorded, dict) and isinstance(fresh, dict):
        return recorded.keys() == fresh.keys() and all(agree(recorded[k], fresh[k]) for k in recorded)
    if isinstance(recorded, (int, float)) and isinstance(fresh, (int, float)) and not isinstance(recorded, bool) and not isinstance(fresh, bool):
        return math.isclose(recorded, fresh, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(recorded, list) and isinstance(fresh, list):
        return len(recorded) == len(fresh) and all(agree(a, b) for a, b in zip(recorded, fresh))
    return type(recorded) is type(fresh) and recorded == fresh


def inside(root, rel) -> Path | None:
    """`rel` as a path inside `root`, or None for an absolute path, a `..` part, or a link on the way."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return None
    p = Path(root)
    for part in Path(rel).parts:
        p = p / part
        if p.is_symlink():
            return None
    return p


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---- reading a run ----------------------------------------------------------------------------------------------------------------

def run_facts(run_dir, rel: str) -> dict:
    """What a record notes about one recorded run: where it is, its model identity, settings and the sha256 of its manifest."""
    d = Path(run_dir) / rel
    with ar.load_run(d) as run:
        m = run.manifest
        return {"dir": rel, "model": m["model"]["name"], "config_sha256": m["model"].get("config_sha256"), "library": m["library"]["name"],
                "settings": dict(m["settings"]), "manifest_sha256": sha256_file(d / "manifest.json"), "arrays": len(run.entries)}


def expected_arrays(n_layers: int, decode_tokens: int) -> list[tuple[str, str]]:
    """The arrays every run of the contract (orchard/text_reference.py) must hold: each block's output, the prefill logits, each decode step's logits and the token path."""
    names = [("layer", tr.block_name(i)) for i in range(n_layers)]
    names += [("other", tr.PREFILL_LOGITS)] + [("other", tr.dec_logits(k)) for k in range(decode_tokens)] + [("other", tr.TOKENS)]
    return names


def run_problems(run, label: str, cfg: dict, *, dtype: str, tokens: list[int] | None) -> list[str]:
    """Reasons a loaded run is not a run of this stage's model and settings: another configuration, another dtype or decode length, a missing array, an
    array with a value that is not finite, or another token path."""
    out = []
    if run.manifest["model"].get("config_sha256") != cfg["config_sha256"]:
        out.append(f"{label}: its model configuration is not this run's")
    if run.settings.get("dtype") != dtype:
        out.append(f"{label}: it ran in {run.settings.get('dtype')!r}; the expected dtype is {dtype!r}")
    if run.settings.get("steps") != cfg["decode_tokens"] or run.settings.get("sampler") != "greedy":
        out.append(f"{label}: its decode length or sampler is not the stage's ({cfg['decode_tokens']} greedy steps)")
    if run.settings.get("duration") != cfg["prompt_tokens"] and cfg.get("prompt_tokens") is not None:
        out.append(f"{label}: its prompt has {run.settings.get('duration')} tokens; the stage's has {cfg['prompt_tokens']}")
    if run.settings.get("prompt") is not None and cfg.get("prompt_sha256") and tr.prompt_sha256(str(run.settings["prompt"])) != cfg["prompt_sha256"]:
        out.append(f"{label}: its prompt is not the stage's prompt")
    missing = [f"{k}/{n}" for (k, n) in expected_arrays(cfg["n_layers"], cfg["decode_tokens"]) if (k, n) not in run.entries]
    if missing:
        out.append(f"{label}: it lacks {', '.join(missing[:4])}{' and more' if len(missing) > 4 else ''}")
        return out
    bad = [f"{k}/{n}" for (k, n), e in run.entries.items() if not e.get("finite", True)]
    if bad:
        out.append(f"{label}: these arrays hold a value that is not finite: {', '.join(bad[:4])}")
    if tokens is not None and [int(v) for v in run.array("other", tr.TOKENS)] != tokens:
        out.append(f"{label}: its token path is not the reference's")
    return out


# ---- stage 1 ------------------------------------------------------------------------------------------------------------------------

def stage_config(stage_dir) -> dict:
    cfg = load_json(Path(stage_dir) / STAGE_CONFIG_FILE, STAGE_CONFIG_FILE)
    for key in ("runtime", "model_id", "config_sha256", "n_layers", "decode_tokens", "prompt_sha256", "model_dir"):
        if key not in cfg:
            raise TextStageError(f"{STAGE_CONFIG_FILE} has no {key}")
    return cfg


def load_reference_runs(stage_dir, run_dir, rec):
    """(reference, {role: run}) loaded from the roles a reference record lists. The caller closes them."""
    runs = {}
    for role in (ROLE_REFERENCE, ROLE_FLOOR, ROLE_FLOOR_NATIVE):
        entry = (rec.get("runs") or {}).get(role)
        if not isinstance(entry, dict):
            continue
        p = inside(run_dir, entry.get("dir"))
        if p is None or not p.is_dir():
            raise TextStageError(f"{role}: its directory {entry.get('dir')!r} is not a directory inside the run directory")
        runs[role] = ar.load_run(p)
    return runs


def reference_record(stage_dir, run_dir, cfg: dict, runtime_name: str) -> dict:
    """The record of stage 1 from the runs on disk: each run's facts, the greedy path, the floor, the perplexity reference. Raises TextStageError when a run is
    missing or is not a run of this stage."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    rel_stage = stage_dir.resolve().relative_to(run_dir.resolve())
    runs, facts = {}, {}
    try:
        for role, sub in REFERENCE_DIRS.items():
            d = stage_dir / "reference" / sub
            if not (d / "manifest.json").is_file():
                if role == ROLE_FLOOR_NATIVE:
                    continue                       # the second floor run is a cross-check; a runtime may not have one
                raise TextStageError(f"the reference command left no {sub} run (stages/{rel_stage.name}/reference/{sub}/manifest.json)")
            facts[role] = run_facts(run_dir, str(rel_stage / "reference" / sub))
            runs[role] = ar.load_run(d)
        tokens = [int(v) for v in runs[ROLE_REFERENCE].array("other", tr.TOKENS)]
        problems = run_problems(runs[ROLE_REFERENCE], "the float32 reference", {**cfg, "prompt_tokens": facts[ROLE_REFERENCE]["settings"]["duration"]},
                                dtype="float32", tokens=tokens)
        prompt_tokens = facts[ROLE_REFERENCE]["settings"]["duration"]
        for role in (ROLE_FLOOR, ROLE_FLOOR_NATIVE):
            if role in runs:
                problems += run_problems(runs[role], f"the {role} run", {**cfg, "prompt_tokens": prompt_tokens}, dtype="bfloat16", tokens=tokens)
        if len(set(tokens)) < MIN_DISTINCT_TOKENS:
            problems.append(f"the reference's greedy path has {len(set(tokens))} distinct token(s); there is nothing to compare with")
        if problems:
            raise TextStageError("; ".join(problems))
        floors = {r: runs[r] for r in (ROLE_FLOOR, ROLE_FLOOR_NATIVE) if r in runs}
        rec = {"schema": SCHEMA, "runtime": runtime_name, "model_id": cfg["model_id"], "config_sha256": cfg["config_sha256"], "n_layers": cfg["n_layers"],
               "prompt_sha256": cfg["prompt_sha256"], "prompt_tokens": prompt_tokens, "decode_tokens": cfg["decode_tokens"], "runs": facts, "tokens": tokens,
               "floor": tb.floor_summary(runs[ROLE_REFERENCE], floors), "ppl_reference": ppl_reference_entry(stage_dir, cfg),
               "limits": ["The reference is the plain-torch float32 CPU run of the runtime's reference code. The bfloat16 runs are the same code in the chip's dtype on the CPU "
                          "(and, when present, independent library code); the chip adds its own accumulation order on top.",
                          *tb.CANNOT_SHOW]}
        return rec
    finally:
        for r in runs.values():
            r.close()


def ppl_reference_entry(stage_dir, cfg: dict) -> dict:
    path = Path(stage_dir) / PPL_REFERENCE_FILE
    data = load_json(path, PPL_REFERENCE_FILE)
    return {"file": PPL_REFERENCE_FILE, "tokens": data.get("tokens"), "perplexity": data.get("perplexity"), "sha256": sha256_file(path)}


def reference_problems(stage_dir, run_dir) -> list[str]:
    """The gate of stage 1, from the files. Every run listed verifies and is this stage's model and settings, the floor and the greedy path are the arrays'
    own, and the perplexity reference is the file the record names."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    try:
        cfg = stage_config(stage_dir)
        rec = load_json(stage_dir / REFERENCE_FILE, REFERENCE_FILE)
    except TextStageError as exc:
        return [str(exc)]
    try:
        fresh = reference_record(stage_dir, run_dir, cfg, rec.get("runtime"))
    except (TextStageError, ar.ReferenceError, OSError, KeyError) as exc:
        return [str(exc)]
    problems = []
    if rec.get("runtime") != cfg["runtime"]:
        problems.append(f"the record is for runtime {rec.get('runtime')!r}; the stage is for {cfg['runtime']!r}")
    for key in ("runs", "tokens", "floor", "ppl_reference", "config_sha256", "n_layers", "prompt_sha256", "prompt_tokens", "decode_tokens", "model_id"):
        if not agree(rec.get(key), fresh.get(key)):
            problems.append(f"the recorded {key} is not what the runs give now")
    ppl = fresh["ppl_reference"]
    expect_tokens = int(cfg.get("ppl_tokens", rt.PPL_TOKENS)) - 1
    p = ppl.get("perplexity")
    if ppl.get("tokens") != expect_tokens:
        problems.append(f"the perplexity reference covers {ppl.get('tokens')} tokens; the window is {expect_tokens}")
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or p <= 0:
        problems.append("the perplexity reference is not a positive finite number")
    return problems


def subprocess_runner(base_env: dict):
    """A runner for `run_references`: each command is a child in its own session with `base_env` plus the command's extra variables, its output in the log
    file. A timeout kills the whole session. Returns the exit status, None for a timeout."""
    import os
    import signal
    import subprocess

    def run(argv, env_extra, timeout_s, log_path):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(argv, env={**base_env, **env_extra}, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                return proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return None
            except BaseException:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
    return run


def run_references(stage_dir, run_dir, rtm, ctx: rt.TextContext, runner, *, timeout_s: float = rt.REFERENCE_TIMEOUT_S, say=lambda text: None) -> dict:
    """Make the weights file (when the adapter says it is missing), the reference runs and the perplexity reference, then write `text_reference.json`.
    `runner(argv, env_extra, timeout_s, log_path)` runs one command and returns its exit status (None for a timeout). Raises TextStageError when a command
    fails or leaves runs that are not this stage's."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    cfg = stage_config(stage_dir)
    (stage_dir / "log").mkdir(parents=True, exist_ok=True)
    steps = []
    prep = rtm.prepare_argv(ctx)
    if prep:
        steps.append(("prepare", prep, None))
    ref_out = stage_dir / "reference"
    if not (ref_out / "float32" / "manifest.json").is_file():
        # a half-made reference directory from an interrupted run is moved aside: a recorder never writes into an existing run
        if ref_out.exists():
            ref_out.rename(stage_dir / f"reference.stale-{int(time.time())}")
        steps.append(("reference", rtm.reference_argv(ctx), None))
    if not (stage_dir / PPL_REFERENCE_FILE).is_file():
        steps.append(("ppl_reference", rtm.ppl_reference_argv(ctx, str(stage_dir / PPL_REFERENCE_FILE)), None))
    for name, argv, env in steps:
        say(f"{name}: running on the CPU")
        code = runner(argv, env or {}, timeout_s, stage_dir / "log" / f"{name}.log")
        if code != 0:
            raise TextStageError(f"the {name} command {'timed out' if code is None else f'exited {code}'}; see {stage_dir.name}/log/{name}.log")
    rec = reference_record(stage_dir, run_dir, cfg, cfg["runtime"])
    write_json(stage_dir / REFERENCE_FILE, rec)
    return rec


# ---- stage 2 ------------------------------------------------------------------------------------------------------------------------

def summarise(report: dict) -> dict:
    """The numbers of a `text_bars.chip_report` that people read and the bundle prints. The per-array entries stay in the evidence file."""
    return {"ok": report["ok"], "reasons": report["reasons"], "margin": report["margin"], "arrays": len(report["blocks"]) + len(report["ops"]) + len(report["other"]),
            "first_block_leaving": report["first_block_leaving"], "first_op_leaving": report["first_op_leaving"],
            "worst_ratio_to_floor": report["worst_ratio_to_floor"], "agreement": report["agreement"],
            "block_rel_l2": {"first": report["blocks"][0]["rel_l2"], "last": report["blocks"][-1]["rel_l2"], "max": max(e["rel_l2"] or 0.0 for e in report["blocks"])},
            "rule": report["rule"]}


def compute_variant(stage_dir, run_dir, ref_stage_dir, ref_rec: dict, rtm, variant) -> tuple[dict, dict]:
    """(summary entry, full report) for one variant's chip run and perplexity, from the files."""
    stage_dir, run_dir, ref_stage_dir = Path(stage_dir), Path(run_dir), Path(ref_stage_dir)
    rel_stage = stage_dir.resolve().relative_to(run_dir.resolve())
    d = stage_dir / "chip" / variant.name
    if not (d / "manifest.json").is_file():
        raise TextStageError(f"the chip run for variant {variant.name} left no manifest")
    runs = load_reference_runs(ref_stage_dir, run_dir, ref_rec)
    try:
        facts = run_facts(run_dir, str(rel_stage / "chip" / variant.name))
        chip = ar.load_run(d)
        try:
            cfg = {"config_sha256": ref_rec["config_sha256"], "decode_tokens": ref_rec["decode_tokens"], "n_layers": ref_rec["n_layers"],
                   "prompt_tokens": ref_rec["prompt_tokens"], "prompt_sha256": ref_rec["prompt_sha256"]}
            bad = run_problems(chip, f"the chip run ({variant.name})", cfg, dtype="bfloat16", tokens=ref_rec["tokens"])
            if bad:
                raise TextStageError("; ".join(bad))
            floors = {r: runs[r] for r in (ROLE_FLOOR, ROLE_FLOOR_NATIVE) if r in runs}
            report = tb.chip_report(runs[ROLE_REFERENCE], floors, chip)
        finally:
            chip.close()
    finally:
        for r in runs.values():
            r.close()
    ppl_ref = load_json(ref_stage_dir / PPL_REFERENCE_FILE, PPL_REFERENCE_FILE)
    ppl_file = stage_dir / f"ppl_chip_{variant.name}.json"
    ppl = tb.perplexity_report(ppl_ref, load_json(ppl_file, ppl_file.name))
    entry = {"variant": variant.name, "description": variant.description, "run": facts, "numeric": summarise(report), "perplexity": ppl,
             "ppl_sha256": sha256_file(ppl_file), "ok": bool(report["ok"] and ppl["ok"])}
    return entry, report


def finish_chip(stage_dir, run_dir, ref_stage_dir, runtime_name: str) -> dict:
    """After the hardware test: compare each variant's chip run with stage 1, write the per-array report of each (evidence) and `text_result.json`."""
    stage_dir = Path(stage_dir)
    rtm = rt.get(runtime_name)
    if rtm is None:
        raise TextStageError(f"no text runtime named {runtime_name!r}")
    ref_rec = load_json(Path(ref_stage_dir) / REFERENCE_FILE, REFERENCE_FILE)
    variants = {}
    (stage_dir / "evidence").mkdir(exist_ok=True)
    for v in rtm.variants:
        entry, report = compute_variant(stage_dir, run_dir, ref_stage_dir, ref_rec, rtm, v)
        write_json(stage_dir / "evidence" / f"numeric_report_{v.name}.json", report)
        entry["report_sha256"] = sha256_file(stage_dir / "evidence" / f"numeric_report_{v.name}.json")
        variants[v.name] = entry
    default = rtm.default_variant().name
    result = {"schema": SCHEMA, "runtime": runtime_name, "model_id": ref_rec["model_id"], "config_sha256": ref_rec["config_sha256"], "default_variant": default,
              "variants": variants, "reference_runs": ref_rec["runs"], "floor": ref_rec["floor"], "rule": tb.RULE,
              "limits": [*tb.CANNOT_SHOW, "The default variant decides the stage; another variant's result is reported and does not fail the stage.", *rtm.limits]}
    write_json(stage_dir / RESULT_FILE, result)
    return result


def chip_problems(stage_dir, run_dir, ref_stage_dir) -> list[str]:
    """The gate of stage 2, from the files: every comparison is recomputed and a result that differs is refused. A pass needs the default variant to meet the
    rule: no array leaves, agreement above its bar, perplexity within its bar."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    try:
        res = load_json(stage_dir / RESULT_FILE, RESULT_FILE)
        ref_rec = load_json(Path(ref_stage_dir) / REFERENCE_FILE, REFERENCE_FILE)
        cfg = stage_config(Path(ref_stage_dir))
    except TextStageError as exc:
        return [str(exc)]
    rtm = rt.get(res.get("runtime"))
    if rtm is None or res.get("runtime") != cfg["runtime"]:
        return [f"the result is for runtime {res.get('runtime')!r}; the run's is {cfg['runtime']!r}"]
    problems = []
    if res.get("config_sha256") != ref_rec.get("config_sha256") or res.get("reference_runs") != ref_rec.get("runs"):
        problems.append("the result's model configuration or reference runs are not stage 1's")
    variants = res.get("variants")
    if not isinstance(variants, dict) or set(variants) != {v.name for v in rtm.variants}:
        return problems + [f"the result needs one entry for each variant of the runtime ({', '.join(v.name for v in rtm.variants)})"]
    if res.get("default_variant") != rtm.default_variant().name:
        problems.append("the result names a default variant that is not the runtime's")
    for v in rtm.variants:
        try:
            fresh, report = compute_variant(stage_dir, run_dir, ref_stage_dir, ref_rec, rtm, v)
        except (TextStageError, ar.ReferenceError, OSError, KeyError) as exc:
            problems.append(f"{v.name}: {exc}")
            continue
        recorded = variants[v.name]
        fresh["report_sha256"] = recorded.get("report_sha256")
        if not agree(recorded, fresh):
            problems.append(f"{v.name}: the recorded comparison is not what the runs give now")
        rp = stage_dir / "evidence" / f"numeric_report_{v.name}.json"
        try:
            if sha256_file(rp) != recorded.get("report_sha256"):
                problems.append(f"{v.name}: the per-array report file is not the recorded one")
        except OSError:
            problems.append(f"{v.name}: the per-array report file is missing")
        if v.name == rtm.default_variant().name and not fresh["ok"]:
            problems += [f"{v.name}: {r}" for r in fresh["numeric"]["reasons"] + fresh["perplexity"]["reasons"]] or [f"{v.name}: the chip run does not meet the bar"]
    return problems


# ---- stage 6 ------------------------------------------------------------------------------------------------------------------------

def bench_entry(stage_dir, rtm, variant, model_config: dict) -> dict:
    """The record of one variant's benchmark file, with the floor and the share of it recomputed here."""
    path = Path(stage_dir) / f"bench_{variant.name}.json"
    data = load_json(path, path.name)
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise TextStageError(f"{path.name} lists no runs")
    measured = [r for r in runs if isinstance(r, dict) and r.get("warmup") is False]
    warm = [r for r in runs if isinstance(r, dict) and r.get("warmup") is True]
    timings = [{k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "repeat"} for r in measured]
    summary = resident.summarise_runs(timings, warmup=len(warm))
    floor = df.floor_for_board(model_config, rtm.dtype_policy(variant), "p300", dies=1)
    parts = summary["parts"].get("decode_tok_s")
    median = parts["median"] if parts else None
    return {"variant": variant.name, "description": variant.description, "file": path.name, "sha256": sha256_file(path), "label": summary["label"],
            "warmup_runs": len(warm), "repeats": summary["repeats"], "summary": summary,
            "decode_tok_s": parts, "prefill_s": summary["parts"].get("prefill_s"), "prompt_tokens": measured[0].get("prefill_tokens") if measured else None,
            "steps": measured[0].get("steps") if measured else None, "floor": floor.as_dict(),
            "share_of_floor_median": df.share_of_floor(median, floor) if median else None, "share_text": df.share_text(median, floor) if median else None,
            "recorded_config": data.get("config")}


def finish_bench(stage_dir, run_dir, ref_stage_dir, runtime_name: str) -> dict:
    stage_dir = Path(stage_dir)
    rtm = rt.get(runtime_name)
    if rtm is None:
        raise TextStageError(f"no text runtime named {runtime_name!r}")
    cfg = stage_config(Path(ref_stage_dir))
    model_config = mc.read_model_config(Path(cfg["model_dir"]) / "config.json")
    variants = {v.name: bench_entry(stage_dir, rtm, v, model_config) for v in rtm.variants}
    result = {"schema": SCHEMA, "runtime": runtime_name, "model_id": cfg["model_id"], "default_variant": rtm.default_variant().name, "variants": variants,
              "rules": "docs/benchmark-rules.md: one discarded warm-up run, then at least three measured runs, in one process; spread is min, median and max",
              "limits": ["Decode speed is free-running greedy decode of a fixed number of tokens after one prompt, batch size one, one chip.",
                         "The floor is the weight-read time at the datasheet DRAM bandwidth (orchard/decode_floor.py); it ignores the KV cache and activations.",
                         "The host's load average is recorded with each run; other agents share this machine."]}
    write_json(stage_dir / BENCH_FILE, result)
    return result


def bench_problems(stage_dir, run_dir, ref_stage_dir) -> list[str]:
    """The gate of stage 6: each variant's record is what its benchmark file gives now, and the default variant is a benchmark by the rules (a warm-up run and
    at least MIN_REPEATS measured runs, decode speed consistent with its steps and time, a floor share that is a share)."""
    stage_dir = Path(stage_dir)
    try:
        res = load_json(stage_dir / BENCH_FILE, BENCH_FILE)
        cfg = stage_config(Path(ref_stage_dir))
        model_config = mc.read_model_config(Path(cfg["model_dir"]) / "config.json")
    except (TextStageError, mc.ModelConfigError) as exc:
        return [str(exc)]
    rtm = rt.get(res.get("runtime"))
    if rtm is None or res.get("runtime") != cfg["runtime"]:
        return [f"the result is for runtime {res.get('runtime')!r}; the run's is {cfg['runtime']!r}"]
    variants = res.get("variants")
    if not isinstance(variants, dict) or set(variants) != {v.name for v in rtm.variants}:
        return [f"the result needs one entry for each variant of the runtime ({', '.join(v.name for v in rtm.variants)})"]
    problems = []
    for v in rtm.variants:
        try:
            fresh = bench_entry(stage_dir, rtm, v, model_config)
        except (TextStageError, OSError, KeyError, df.FloorInputError, ValueError) as exc:
            problems.append(f"{v.name}: {exc}")
            continue
        if not agree(variants[v.name], fresh):
            problems.append(f"{v.name}: the recorded benchmark is not what its file gives now")
        if fresh["warmup_runs"] < resident.WARMUP_RUNS:
            problems.append(f"{v.name}: {fresh['warmup_runs']} warm-up run(s); the rule is {resident.WARMUP_RUNS}")
        if fresh["repeats"] < resident.MIN_REPEATS:
            problems.append(f"{v.name}: {fresh['repeats']} measured run(s); the rule is {resident.MIN_REPEATS}")
        if fresh["decode_tok_s"] is None:
            problems.append(f"{v.name}: the runs report no decode speed")
            continue
        data = load_json(stage_dir / fresh["file"], fresh["file"])
        for i, r in enumerate(data.get("runs", [])):
            if not isinstance(r, dict) or not all(isinstance(r.get(k), (int, float)) for k in ("steps", "decode_s", "decode_tok_s")) or r["decode_s"] <= 0:
                problems.append(f"{v.name} run {i}: it has no steps, decode time and speed that are numbers")
            elif not math.isclose(r["steps"] / r["decode_s"], r["decode_tok_s"], rel_tol=1e-6):
                problems.append(f"{v.name} run {i}: its decode speed is not its steps over its decode time")
        share = fresh["share_of_floor_median"]
        if not isinstance(share, (int, float)) or not 0 < share <= 1.5:
            problems.append(f"{v.name}: the share of the floor ({share}) is not a share between 0 and 1.5")
    return problems
