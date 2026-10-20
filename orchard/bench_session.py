# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One benchmark session for one model: start its server, run the benchmark phases through a recording proxy, stop the server, write a summary.

    python3 -m orchard.bench_session --label pi --model-id bytkim/Qwen3.8-27B-pi@34d3f21 --run-sh .../run.sh --model-dir DIR \\
        --tt-cache DIR --hf-home DIR --out-dir OUT --phases gsm8k,humaneval,ifeval,bfcl --limit 20 --thinking off

The caller holds the hardware lease (`gozer run --chips 2 ... -- timeout N python3 -m orchard.bench_session ...`). With `--url` the session
uses a server that is already running and starts nothing; the tests use that.

Phases: `gsm8k`, `humaneval` (`humaneval_chat`, from orchard/lm_eval_tasks), `mbpp` (`mbpp_chat`), `ifeval` run lm-evaluation-harness in its own venv as a
child process with a scrubbed environment (no tokens, a home inside the output directory) under the stall watcher. `bfcl` runs the
tool-call scorer in this process. Every phase goes through a recording proxy, which adds `chat_template_kwargs` when thinking is set and
records tokens, finish reasons and latency per request.

The summary records what two runs must share to be compared: the settings, the harness version, the task versions and sample counts, the
bundle's run.sh hash, and the served model name. A comparison across sessions is the reader's to make from those fields.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import bench_proxy, toolcall_bench
from .stallwatch import StallWatcher, run_watched

HERE = Path(__file__).resolve().parent
SWAP_TEMPLATE = HERE / "skills" / "weights-swap-templates" / "serve_and_compare.py"
DEFAULT_LM_EVAL = Path.home() / "orchard-venvs" / "bench" / "bin" / "lm_eval"
TASKS_DIR = HERE / "lm_eval_tasks"          # humaneval_chat and mbpp_chat: the harness's own *_instruct tasks cannot score a chat reply (see the task files)
LM_EVAL_TASKS = {"gsm8k": "gsm8k", "humaneval": "humaneval_chat", "mbpp": "mbpp_chat", "ifeval": "ifeval"}
PHASES = tuple(LM_EVAL_TASKS) + ("bfcl",)
CODE_TASKS = {"humaneval_chat", "mbpp_chat"}
BFCL_SETS = ("simple", "multiple")
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
ENV_KEEP = ("PATH", "LANG", "LC_ALL", "LC_CTYPE")
NOT_METRICS = {"name", "alias", "sample_len"}
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


# ---- the lm-evaluation-harness child ------------------------------------------------------------------------------------------------

def lm_eval_command(*, lm_eval, model: str, base_url: str, task: str, limit, concurrency: int, max_gen_toks: int, output_path, include_path) -> list[str]:
    margs = f"model={model},base_url={base_url},num_concurrent={concurrency},max_retries=1,tokenized_requests=False,timeout=3600"
    argv = [str(lm_eval), "--model", "local-chat-completions", "--model_args", margs, "--tasks", task, "--apply_chat_template",
            "--gen_kwargs", f"max_gen_toks={max_gen_toks},temperature=0", "--output_path", str(output_path), "--log_samples",
            "--include_path", str(include_path)]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if task == "gsm8k":
        argv += ["--num_fewshot", "5", "--fewshot_as_multiturn"]
    if task in CODE_TASKS:
        argv += ["--confirm_run_unsafe_code"]
    return argv


def lm_eval_env(task: str, out_dir) -> dict:
    """The environment of the harness child: the search path and locale, a home and Hugging Face directory inside the output directory (so
    the operator's token file is not found), no token variables, and the code-execution gate only for code tasks. The datasets download,
    so offline mode is not set."""
    out_dir = Path(out_dir)
    env = {k: os.environ[k] for k in ENV_KEEP if k in os.environ}
    env.update({"HOME": str(out_dir / "home"), "HF_HOME": str(out_dir / "hf-datasets"), "TOKENIZERS_PARALLELISM": "false"})
    if task in CODE_TASKS:
        env["HF_ALLOW_CODE_EVAL"] = "1"
    return env


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_lm_eval_results(root) -> dict:
    """The newest results file under `root`, as {tasks: {task: {version, n_samples, n_shot, metrics: [{metric, filter, value, stderr}]}}, ...}."""
    root = Path(root)
    found = sorted(root.rglob("results_*.json"), key=lambda p: p.name)
    if not found:
        raise FileNotFoundError(f"no results_*.json under {root}")
    data = json.loads(found[-1].read_text(encoding="utf-8"))
    tasks = {}
    for task, entry in data.get("results", {}).items():
        metrics = []
        for key, value in entry.items():
            metric, _, filt = key.partition(",")
            if metric in NOT_METRICS or metric.endswith("_stderr") or not _number(value):
                continue
            se = entry.get(f"{metric}_stderr,{filt}")
            metrics.append({"metric": metric, "filter": filt, "value": value, "stderr": se if _number(se) else None})
        tasks[task] = {"version": data.get("versions", {}).get(task), "n_samples": data.get("n-samples", {}).get(task),
                       "n_shot": data.get("n-shot", {}).get(task), "metrics": metrics}
    return {"tasks": tasks, "lm_eval_version": data.get("lm_eval_version"),
            "total_evaluation_time_seconds": data.get("total_evaluation_time_seconds")}


# ---- the drafter's acceptance, from the server's own log ------------------------------------------------------------------------------

# The DFlash2 serving code logs one line when a request on a slot ends, for example
#   [dflash2-serve] slot 1 session: 56 iters, 276 committed tokens, accept 3.95/7 -> 4.93 tok/iter, 59.0 tok/s in-model (incl. seed); histogram [7, 10, 3, 5, 6, 3, 3, 19]
# `accept A/K` is the mean number of drafted tokens accepted per iteration out of K drafted, and the histogram counts iterations by how many were accepted.
SESSION_LINE = re.compile(r"\[dflash2-serve\] slot \d+ session: (\d+) iters, (\d+) committed tokens, accept [0-9.]+/(\d+) -> [0-9.]+ tok/iter, "
                          r"([0-9.]+) tok/s in-model \(incl\. seed\); histogram \[([0-9, ]+)\]")


def drafter_stats(text: str):
    """Totals over the drafter's session lines in `text`, weighted by iterations, or None when there are none. A line that does not parse, or whose
    histogram does not have K + 1 counts that add up to its iterations, is skipped."""
    sessions = iters = committed = 0
    hist, k_seen, speeds = None, None, []
    for m in SESSION_LINE.finditer(text or ""):
        n_iters, n_committed, k, speed = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        counts = [int(x) for x in m.group(5).split(",")]
        if len(counts) != k + 1 or sum(counts) != n_iters or (k_seen not in (None, k)):
            continue
        k_seen = k
        hist = counts if hist is None else [a + b for a, b in zip(hist, counts)]
        sessions, iters, committed = sessions + 1, iters + n_iters, committed + n_committed
        if n_committed > 0:
            speeds.append(speed)
    if not sessions:
        return None
    accepted = sum(i * c for i, c in enumerate(hist))
    return {"sessions": sessions, "iters": iters, "committed_tokens": committed,
            "tokens_per_iter": committed / iters if iters else None, "draft_length": k_seen, "histogram": hist,
            "accepted_per_iter": accepted / iters if iters else None, "accept_rate": accepted / (iters * k_seen) if iters and k_seen else None,
            "tok_s_in_model_median": statistics.median(speeds) if speeds else None}


# ---- the tool-call phase ------------------------------------------------------------------------------------------------------------

def run_bfcl_phase(*, chat_url: str, served: str, bfcl_dir, out_dir, limit, concurrency: int, max_tokens: int, seed: int) -> dict:
    bfcl_dir, out_dir = Path(bfcl_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rev_file = bfcl_dir / "REVISION"
    dataset = f"{BFCL_REPO}@{rev_file.read_text().strip() if rev_file.is_file() else 'unknown'}"
    scores = {}
    for kind in BFCL_SETS:
        items = toolcall_bench.choose(toolcall_bench.load_items(bfcl_dir / f"{kind}_questions.jsonl", bfcl_dir / f"{kind}_answers.jsonl"),
                                      limit=limit, seed=seed)
        result = toolcall_bench.run(chat_url, served, items, concurrency=concurrency, max_tokens=max_tokens, seed=seed, dataset=dataset)
        (out_dir / f"bfcl-{kind}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        scores[kind] = {k: result[k] for k in ("n", "passed", "failed", "score", "stderr", "errors", "cut_off", "completion_tokens_total", "stopped")}
    return scores


# ---- the session ---------------------------------------------------------------------------------------------------------------------

def _swap():
    spec = importlib.util.spec_from_file_location("serve_and_compare_for_bench", str(SWAP_TEMPLATE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _served_name(base: str, fallback: str) -> str:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as resp:
            return (json.loads(resp.read()).get("data") or [{}])[0].get("id") or fallback
    except (OSError, ValueError):
        return fallback


def _inject(thinking: str) -> dict:
    return {} if thinking == "default" else {"chat_template_kwargs": {"enable_thinking": thinking == "on"}}


def _phase(name: str, a, *, upstream: str, served: str, out_dir: Path, server_log=None) -> dict:
    phase_dir = out_dir / name
    phase_dir.mkdir(parents=True, exist_ok=True)
    proxy = bench_proxy.Proxy(upstream, inject=_inject(a.thinking), log_path=phase_dir / "requests.jsonl")
    proxy.start()
    chat_url = f"http://127.0.0.1:{proxy.port}/v1/chat/completions"
    rec = {"name": name, "ok": False, "problem": None, "command": None, "drafter": None}
    log_start = server_log.stat().st_size if server_log is not None else None
    t0 = time.monotonic()
    try:
        if name == "bfcl":
            rec["scores"] = run_bfcl_phase(chat_url=chat_url, served=served, bfcl_dir=a.bfcl_dir, out_dir=phase_dir, limit=a.limit,
                                           concurrency=a.concurrency, max_tokens=a.max_gen_toks, seed=a.seed)
            stopped = [s["stopped"] for s in rec["scores"].values() if s["stopped"]]
            rec["ok"], rec["problem"] = not stopped, (stopped[0] if stopped else None)
        else:
            task = LM_EVAL_TASKS[name]
            argv = lm_eval_command(lm_eval=a.lm_eval, model=served, base_url=chat_url, task=task, limit=a.limit, concurrency=a.concurrency,
                                   max_gen_toks=a.max_gen_toks, output_path=phase_dir / "results", include_path=TASKS_DIR)
            rec["command"] = argv
            (phase_dir / "work").mkdir(exist_ok=True)
            watched = run_watched(argv, cwd=phase_dir / "work", env=lm_eval_env(task, out_dir), log_path=phase_dir / "lm_eval.log",
                                  watcher=StallWatcher(quiet_s=a.quiet_s, repeat_lines=10 ** 9), backstop_s=a.phase_backstop_s)
            rec["watch"] = {"max_quiet_s": watched.max_quiet_s}
            if watched.killed:
                rec["problem"] = watched.reason
            elif watched.returncode != 0:
                rec["problem"] = f"exit {watched.returncode}"
            else:
                try:
                    rec["scores"] = parse_lm_eval_results(phase_dir / "results")
                    rec["ok"] = True
                except (FileNotFoundError, ValueError) as exc:
                    rec["problem"] = f"no results: {exc}"
    finally:
        rec["wall_s"] = round(time.monotonic() - t0, 1)
        rec["proxy"] = proxy.summary()
        proxy.stop()
        if server_log is not None:
            with open(server_log, "rb") as f:
                f.seek(log_start)
                rec["drafter"] = drafter_stats(f.read().decode("utf-8", errors="replace"))
    return rec


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", help="use a server that is already running (http://127.0.0.1:PORT); starts nothing")
    ap.add_argument("--model-id", help="the id the tensor cache marker must name (the weights served)")
    ap.add_argument("--run-sh")
    ap.add_argument("--model-dir")
    ap.add_argument("--tt-cache")
    ap.add_argument("--hf-home")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--health-timeout", type=float, default=2400.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lm-eval", default=str(DEFAULT_LM_EVAL))
    ap.add_argument("--phases", default="gsm8k,humaneval,ifeval,bfcl")
    ap.add_argument("--bfcl-dir")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    ap.add_argument("--max-gen-toks", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet-s", type=float, default=900.0, help="stop a phase after this many seconds with no output and no CPU use")
    ap.add_argument("--phase-backstop-s", type=float, default=4 * 3600.0)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    phases = [p for p in a.phases.split(",") if p]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        ap.error(f"unknown phase {unknown}; choose from {', '.join(PHASES)}")
    if "bfcl" in phases and not a.bfcl_dir:
        ap.error("the bfcl phase needs --bfcl-dir")
    if not a.url:
        missing = [n for n in ("model_id", "run_sh", "model_dir", "tt_cache", "hf_home") if not getattr(a, n)]
        if missing:
            ap.error("starting a server needs " + ", ".join("--" + m.replace("_", "-") for m in missing) + " (or pass --url)")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary = {"label": a.label, "model_id": a.model_id, "served_name": None, "server_ready_s": None, "run_sh": a.run_sh, "run_sh_sha256": None,
               "settings": {"phases": phases, "limit": a.limit, "concurrency": a.concurrency, "thinking": a.thinking, "max_gen_toks": a.max_gen_toks,
                            "seed": a.seed, "temperature": 0},
               "lm_eval_version": None, "started_at": started, "phases": []}

    def write():
        summary["wall_s"] = round(time.time() - started, 1)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    proc = sc = None
    try:
        if a.url:
            base = a.url.rstrip("/")
        else:
            sc = _swap()
            cache = Path(a.tt_cache)
            sc.guard_cache(cache, a.model_id)
            summary["run_sh_sha256"] = hashlib.sha256(Path(a.run_sh).read_bytes()).hexdigest()
            env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS}
            env.update(TT_CACHE_PATH=str(cache), TT_CACHE_HOME=str(cache), HF_HOME=a.hf_home, HF_HUB_OFFLINE="1", MODEL_WEIGHTS_DIR=a.model_dir,
                       HF_MODEL=a.model_dir)
            log_path = out_dir / "server.log"
            with open(log_path, "wb") as log:
                proc = subprocess.Popen(["bash", a.run_sh, "--port", str(a.port)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
            summary["server_ready_s"] = round(sc.wait_healthy(proc, a.port, a.health_timeout, log_path), 1)
            base = f"http://127.0.0.1:{a.port}"
        summary["served_name"] = _served_name(base, a.model_dir or a.label)
        write()
        for name in phases:
            rec = _phase(name, a, upstream=base, served=summary["served_name"], out_dir=out_dir, server_log=None if a.url else out_dir / "server.log")
            summary["phases"].append(rec)
            if summary["lm_eval_version"] is None and rec.get("scores", {}).get("lm_eval_version"):
                summary["lm_eval_version"] = rec["scores"]["lm_eval_version"]
            write()
    finally:
        if proc is not None:
            sc.stop_server(proc)
        write()
    return 0 if all(p["ok"] for p in summary["phases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
