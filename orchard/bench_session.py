# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One benchmark session for one model: start its server, run the benchmark phases through a recording proxy, stop the server, write a summary.

    python3 -m orchard.bench_session --label pi --model-id bytkim/Qwen3.8-27B-pi@34d3f21 --run-sh .../run.sh --model-dir DIR \\
        --tt-cache DIR --hf-home DIR --out-dir OUT --phases gsm8k,humaneval,ifeval,bfcl --limit 20 --thinking off

The caller holds the hardware lease (`gozer run --chips 2 ... -- timeout N python3 -m orchard.bench_session ...`). With `--url` the session
uses a server that is already running and starts nothing; the tests use that.

Phases: `gsm8k`, `humaneval` (`humaneval_chat`, from orchard/lm_eval_tasks), `mbpp` (`mbpp_chat`), `ifeval` run lm-evaluation-harness in its own venv as a
child process under the stall watcher. Its environment has no token variables and a HOME and HF_HOME inside the output directory. That does not
stop code from opening a file by its absolute path or a socket, so the environment is not what protects the operator's token file from a model's
code. The code phases (`humaneval`, `mbpp`) score by running each reply in `orchard/lm_eval_tasks/code_sandbox.py`: a child that restricts itself with
Landlock and seccomp to a scratch directory and no sockets, and never runs a reply inside the harness's own process. Before the server starts, the
session runs the sandbox's self-check and refuses a code phase when it fails; `--allow-unsandboxed-code` runs the replies without isolation, and the
summary says so. `bfcl` runs the tool-call scorer in this process. Every phase goes through a recording proxy, which adds `chat_template_kwargs` when thinking is set and
records tokens, finish reasons and latency per request.

The summary records what two runs must share to be compared: the settings (including `stop`, the stop-string override), the harness version, the task
versions and sample counts, the bundle's run.sh hash, and the served model name. A comparison across sessions is the reader's to make from those fields.

It also counts `stop_truncated_suspects`: replies that stopped with reasoning text and no answer. lm-eval's gsm8k task sends "Question:" as a stop
string, vLLM matches it inside the reasoning, and a thinking model that restates the question while thinking is cut off there; the score then measures
how often the model avoids the phrase. `--stop` replaces the task's stop strings with one string.
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

def validate_stop(stop: str) -> str:
    """The `--stop` value, or ValueError. The harness reads --gen_kwargs as comma-separated key=value pairs, so a comma would end the value and
    start another pair, and an empty or blank string would send no usable stop. One string is all the option can carry."""
    if not stop.strip():
        raise ValueError("--stop must not be empty")
    if "," in stop:
        raise ValueError("--stop cannot contain a comma: the harness reads --gen_kwargs as comma-separated pairs, so it takes one stop string")
    return stop


def lm_eval_command(*, lm_eval, model: str, base_url: str, task: str, limit, concurrency: int, max_gen_toks: int, output_path, include_path,
                    samples=None, stop=None) -> list[str]:
    """The harness command. `stop` replaces the task's own stop strings (the harness's `until`); gsm8k's include "Question:", which vLLM also
    matches inside a thinking model's reasoning text, so a model that restates "Question:" while thinking is stopped with no answer."""
    margs = f"model={model},base_url={base_url},num_concurrent={concurrency},max_retries=1,tokenized_requests=False,timeout=3600"
    gen_kwargs = f"max_gen_toks={max_gen_toks},temperature=0" + (f",until={stop}" if stop is not None else "")
    argv = [str(lm_eval), "--model", "local-chat-completions", "--model_args", margs, "--tasks", task, "--apply_chat_template",
            "--gen_kwargs", gen_kwargs, "--output_path", str(output_path), "--log_samples",
            "--include_path", str(include_path)]
    if samples is not None:
        argv += ["--samples", json.dumps({task: sorted(samples)})]     # the harness refuses --samples together with --limit
    elif limit is not None:
        argv += ["--limit", str(limit)]
    if task == "gsm8k":
        argv += ["--num_fewshot", "5", "--fewshot_as_multiturn"]
    if task in CODE_TASKS:
        argv += ["--confirm_run_unsafe_code"]
    return argv


def lm_eval_env(task: str, out_dir, *, unsandboxed: bool = False) -> dict:
    """The environment of the harness child: the search path and locale, a home and Hugging Face directory inside the output directory (so
    a token file under the real home is not found by the usual lookup), and no token variables. The datasets download, so offline mode is not set.

    This does not isolate anything. Code that opens an absolute path or a socket is not stopped by it; the code tasks score through
    code_sandbox.py, which is. HF_ALLOW_CODE_EVAL is never set: it would let Hugging Face's `code_eval` run a reply inside the harness's own
    process. `unsandboxed` (the operator's --allow-unsandboxed-code) tells the code tasks to run replies without the sandbox."""
    out_dir = Path(out_dir)
    env = {k: os.environ[k] for k in ENV_KEEP if k in os.environ}
    env.update({"HOME": str(out_dir / "home"), "HF_HOME": str(out_dir / "hf-datasets"), "TOKENIZERS_PARALLELISM": "false"})
    if unsandboxed and task in CODE_TASKS:
        env["ORCHARD_CODE_UNSANDBOXED"] = "1"
    return env


def code_sandbox_check() -> dict:
    """The sandbox's own check (code_sandbox.self_check): {"ok", "detail", "seen"}. ok is True only when the kernel enforced every restriction."""
    spec = importlib.util.spec_from_file_location("code_sandbox_for_bench", str(TASKS_DIR / "code_sandbox.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.self_check()


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
                                   max_gen_toks=a.max_gen_toks, output_path=phase_dir / "results", include_path=TASKS_DIR,
                                   samples=(a.items or {}).get(name), stop=a.stop)
            rec["command"] = argv
            (phase_dir / "work").mkdir(exist_ok=True)
            watched = run_watched(argv, cwd=phase_dir / "work", env=lm_eval_env(task, out_dir, unsandboxed=a.allow_unsandboxed_code), log_path=phase_dir / "lm_eval.log",
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
        failed = rec["proxy"]["errors"]
        if rec["ok"] and failed:
            # The harness retries once and then scores a failed request as an empty reply, so it exits 0 and the score includes the failures.
            rec["ok"] = False
            rec["problem"] = (f"{failed} of {rec['proxy']['requests']} requests to the server failed (the proxy recorded them); "
                              "the scores count those items as failures")
        proxy.stop()
        if server_log is not None:
            with open(server_log, "rb") as f:
                f.seek(log_start)
                rec["drafter"] = drafter_stats(f.read().decode("utf-8", errors="replace"))
    return rec


STOP_SUSPECT_NOTE = ("These replies stopped with reasoning text and no answer. When the harness's stop strings include one that the model writes while "
                     "thinking (gsm8k's \"Question:\"), the server stops the reply there, and the harness scores the item as empty. The score then "
                     "measures how often the model avoids that phrase. Re-run the phase with --stop set to a string the model does not write, "
                     "for example --stop '</s>'.")


def stop_suspects(phases: list[dict]) -> dict:
    """{"total", "by_phase"} from each phase's recorded requests, and a `note` when any phase has one."""
    by_phase = {p["name"]: p["proxy"]["stop_truncated_suspects"] for p in phases
                if isinstance(p.get("proxy"), dict) and p["proxy"].get("stop_truncated_suspects")}
    out = {"total": sum(by_phase.values()), "by_phase": by_phase}
    if by_phase:
        out["note"] = STOP_SUSPECT_NOTE
    return out


def load_items(path) -> dict:
    """The `--items` file: {phase: [document ids]}, each list non-empty, ids non-negative integers. Raises ValueError with the reason."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}")
    if not isinstance(data, dict) or not data:
        raise ValueError("the items file must be a JSON object of phase -> list of document ids")
    for phase, ids in data.items():
        if not isinstance(ids, list) or not ids or not all(type(i) is int and i >= 0 for i in ids):
            raise ValueError(f"items for {phase!r} must be a non-empty list of non-negative integers")
    return {phase: sorted(set(ids)) for phase, ids in data.items()}


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
    ap.add_argument("--items", dest="items_file", help="JSON file {phase: [document ids]}: run only those items of each lm-eval phase (not with --limit; not for bfcl)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    ap.add_argument("--max-gen-toks", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop", help="one stop string that replaces the task's own (the harness's `until`); gsm8k's own list includes \"Question:\", which vLLM "
                                   "also matches in a thinking model's reasoning, so a reply that restates it while thinking is cut off with no answer. "
                                   "One string only: the harness reads --gen_kwargs as comma-separated pairs. Default: the tasks' own")
    ap.add_argument("--quiet-s", type=float, default=900.0, help="stop a phase after this many seconds with no output and no CPU use")
    ap.add_argument("--phase-backstop-s", type=float, default=4 * 3600.0)
    ap.add_argument("--allow-unsandboxed-code", action="store_true",
                    help="run the humaneval and mbpp replies without the sandbox when its self-check fails (the replies run as you, with your files and network)")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    phases = [p for p in a.phases.split(",") if p]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        ap.error(f"unknown phase {unknown}; choose from {', '.join(PHASES)}")
    if a.stop is not None:
        try:
            validate_stop(a.stop)
        except ValueError as exc:
            ap.error(str(exc))
    a.items = None
    if a.items_file:
        if a.limit is not None:
            ap.error("--items and --limit cannot be used together")
        try:
            a.items = load_items(a.items_file)
        except ValueError as exc:
            ap.error(str(exc))
        if "bfcl" in phases:
            ap.error("the bfcl phase cannot take --items")
        absent = [p for p in phases if p not in a.items]
        if absent:
            ap.error(f"the items file has no entry for phase {', '.join(absent)}")
    if "bfcl" in phases and not a.bfcl_dir:
        ap.error("the bfcl phase needs --bfcl-dir")
    if not a.url:
        missing = [n for n in ("model_id", "run_sh", "model_dir", "tt_cache", "hf_home") if not getattr(a, n)]
        if missing:
            ap.error("starting a server needs " + ", ".join("--" + m.replace("_", "-") for m in missing) + " (or pass --url)")
    code_sandbox = None
    if any(LM_EVAL_TASKS.get(p) in CODE_TASKS for p in phases):
        check = code_sandbox_check()
        if check["ok"]:
            code_sandbox = {"mode": "landlock+seccomp", "detail": check["detail"]}
        elif a.allow_unsandboxed_code:
            code_sandbox = {"mode": "none", "reason": check["detail"], "allowed_by": "--allow-unsandboxed-code"}
        else:
            ap.error(f"the code phases need the sandbox, and its check failed: {check['detail']}. "
                     "Pass --allow-unsandboxed-code to run the replies as yourself, with your files and network.")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary = {"label": a.label, "model_id": a.model_id, "served_name": None, "server_ready_s": None, "run_sh": a.run_sh, "run_sh_sha256": None,
               "settings": {"phases": phases, "limit": a.limit, "items": a.items, "concurrency": a.concurrency, "thinking": a.thinking, "max_gen_toks": a.max_gen_toks,
                            "seed": a.seed, "temperature": 0, "stop": a.stop},
               "lm_eval_version": None, "started_at": started, "phases": []}
    if code_sandbox is not None:
        summary["code_sandbox"] = code_sandbox

    def write():
        summary["wall_s"] = round(time.time() - started, 1)
        summary["stop_truncated_suspects"] = stop_suspects(summary["phases"])
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
