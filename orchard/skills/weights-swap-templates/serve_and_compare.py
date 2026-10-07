#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Serve the swapped weights with the bundle and compare the chip's tokens with the CPU reference.

The weights-swap-check skill copies this file into the stage directory. The supervisor runs it as
the stage's hardware test, on a leased board, with that board's chips in TT_VISIBLE_DEVICES. It
reads `swap_config.json` from its own directory (the stage directory) and uses `run.sh` and
`model-dir/` that prepare_swap.py built there.

Steps:
(a) Cache guard. The bundle converts weights into a tensor cache keyed only by layer name, so a
    cache made for another model is read without complaint and serves that model's weights. A
    non-empty tt_cache must hold `.orchard-model` whose text equals new_model_id, or this exits 3
    before any server starts. A missing or empty cache is created and the marker written.
(b) Start `bash run.sh --port <port>` in its own session, with TT_CACHE_PATH, TT_CACHE_HOME,
    HF_HOME, HF_HUB_OFFLINE=1, MODEL_WEIGHTS_DIR and HF_MODEL added to the inherited environment.
    The TT runtime takes its weights directory from MODEL_WEIGHTS_DIR, then HF_MODEL, then the
    config path, and the bundle's run.sh sets HF_MODEL to the nearest model. Without these two the
    server loads the nearest model's weights from the HF cache. Both are set to model-dir, so a
    later change in that order cannot fall back to the base weights. Output goes to
    evidence/server.log. The `try` starts on the line after Popen, and its `finally` stops the
    whole process group: SIGTERM, up to 60 s for the group to go, then SIGKILL.
(c) Poll /health every 2 s for up to HEALTH_TIMEOUT_S (config key health_timeout_s overrides it,
    for tests). If the server exits first, print the last 60 lines of server.log and exit 4.
(d) Free run: 32 greedy tokens from the stage 1 prompt ids.
(e) Teacher forced, k = 0..31: prompt_ids + generated_ids[:k], one greedy token. The returned
    text is re-tokenized with model-dir/tokenizer.json and its FIRST id is compared with
    generated_ids[k]. top1_agreement = matches / 32.
(f) coherent: the free-run text is non-empty, at least 80 percent of its characters are letters,
    digits, whitespace or common punctuation, and no 6-word phrase appears more than 3 times.
(g) Write evidence/swap-check.json (every number, the texts, ids, mismatches, a `serving` record of the
    drafter and sampling the run script sets, and a `result_draft`
    holding exactly the fields the skill's result.json needs) and print the draft.

Exit codes: 0 when the measurements completed, whatever they say. 3 cache guard. 4 the server
exited, or never became healthy. 5 the server answered a request with an HTTP error (its body is
printed). Any other error in this script exits non-zero through Python's traceback; the finally
still stops the server.

The server rejects logprobs and sampling parameters, so requests carry only model, prompt,
max_tokens and temperature.

serve_and_compare_container.py (stage 4) is copied next to this file and imports guard_cache,
load_reference, measure, write_report and ServerHTTPError from it, so both templates measure the
same way.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HEALTH_TIMEOUT_S = 3300.0     # a first boot with cold caches: weight conversion (about 5 min for 27B here) plus a
                              # cold kernel compile cache (more than 26 min, measured 2026-10-03). Warm: about 2 min.
HEALTH_POLL_S = 2.0
STOP_WAIT_S = 60.0
REQUEST_TIMEOUT_S = 600.0
N_TOKENS = 32
MARKER = ".orchard-model"
LOG_TAIL_LINES = 60
COHERENT_CHAR_SHARE = 0.8
MAX_PHRASE_REPEATS = 3
PHRASE_WORDS = 6
PUNCTUATION = set(".,;:!?'\"-()[]{}/\\&%$#@*+=<>_`~|^‘’“”–—…")

STAGE_DIR = Path(__file__).resolve().parent


class ServerHTTPError(Exception):
    """The server answered with an HTTP error status. Carries the body for the message."""


def load_config() -> dict:
    return json.loads((STAGE_DIR / "swap_config.json").read_text(encoding="utf-8"))


def guard_cache(cache: Path, model_id: str) -> None:
    """Step (a). Exits 3 for a non-empty cache whose marker is missing or names another model."""
    if cache.is_dir() and any(cache.iterdir()):
        marker = cache / MARKER
        found = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
        if found != model_id:
            what = "has no marker file" if found is None else f"is marked for {found!r}"
            print(f"serve_and_compare: the tensor cache {cache} is not empty and {what} "
                  f"({MARKER} must contain {model_id!r}). A cache without that marker may belong "
                  "to another model, and the server would serve that model's weights without any "
                  "error. Use a new empty directory for tt_cache.", file=sys.stderr)
            sys.exit(3)
        return
    cache.mkdir(parents=True, exist_ok=True)
    (cache / MARKER).write_text(model_id, encoding="utf-8")


def stop_server(proc: subprocess.Popen) -> None:
    """Stop the server's whole process group. With start_new_session=True the child's pid is its
    process group id (Popen has no pgid attribute)."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + STOP_WAIT_S
    try:
        proc.wait(timeout=STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        pass
    while time.monotonic() < deadline and group_alive(proc.pid):
        time.sleep(0.5)
    if group_alive(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def log_tail(path: Path, n: int = LOG_TAIL_LINES) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def wait_healthy(proc, port: int, timeout_s: float, log_path: Path) -> float:
    """Step (c). Returns the seconds until /health answered 200. Exits 4 otherwise."""
    t0 = time.monotonic()
    url = f"http://127.0.0.1:{port}/health"
    while True:
        if proc.poll() is not None:
            print(f"serve_and_compare: the server exited with code {proc.returncode} before it was "
                  f"healthy. Last {LOG_TAIL_LINES} lines of {log_path}:\n{log_tail(log_path)}")
            sys.exit(4)
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return time.monotonic() - t0
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() - t0 > timeout_s:
            print(f"serve_and_compare: the server did not answer {url} within {timeout_s} s. Last "
                  f"{LOG_TAIL_LINES} lines of {log_path}:\n{log_tail(log_path)}")
            sys.exit(4)
        time.sleep(HEALTH_POLL_S)


def complete(port: int, model: str, prompt: list[int], max_tokens: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ServerHTTPError(f"HTTP {exc.code}: "
                              f"{exc.read().decode('utf-8', errors='replace')}") from exc
    return data["choices"][0]["text"]


def coherence(text: str) -> dict:
    """Step (f). Returns the numbers and the verdict."""
    ok_chars = sum(1 for c in text if c.isalpha() or c.isdigit() or c.isspace() or c in PUNCTUATION)
    share = ok_chars / len(text) if text else 0.0
    words = text.split()
    phrases = Counter(tuple(words[i:i + PHRASE_WORDS]) for i in range(len(words) - PHRASE_WORDS + 1))
    top = phrases.most_common(1)[0][1] if phrases else 0
    coherent = bool(text.strip()) and share >= COHERENT_CHAR_SHARE and top <= MAX_PHRASE_REPEATS
    return {"coherent": coherent, "readable_char_share": round(share, 4),
            "max_6word_phrase_repeats": top}


def measure(cfg: dict, proc, log_path: Path, tokenizer, prompt_ids, generated_ids) -> dict:
    port = int(cfg["port"])
    model = str(STAGE_DIR / "model-dir")
    ready_s = wait_healthy(proc, port, float(cfg.get("health_timeout_s", HEALTH_TIMEOUT_S)), log_path)
    if cfg.get("test_raise_after_ready"):
        raise RuntimeError("test_raise_after_ready is set: raising after the server became healthy")
    free_text = complete(port, model, prompt_ids, N_TOKENS)
    matches, mismatches, forced = 0, [], []
    for k in range(N_TOKENS):
        got_text = complete(port, model, prompt_ids + generated_ids[:k], 1)
        ids = tokenizer.encode(got_text, add_special_tokens=False).ids
        got_id = ids[0] if ids else None
        forced.append({"position": k, "text": got_text, "first_id": got_id})
        if got_id == generated_ids[k]:
            matches += 1
        else:
            mismatches.append({"position": k, "expected_id": generated_ids[k], "got_id": got_id,
                               "expected_text": tokenizer.decode([generated_ids[k]])[:300],
                               "got_text": got_text[:300]})
    return {"server_ready_s": round(ready_s, 1), "free_run_text": free_text,
            "matches": matches, "forced": forced, "mismatches": mismatches}


def load_reference(run_dir: Path) -> tuple[list[int], list[int], str | None]:
    """The stage 1 prompt ids, generated ids and generated text. Exits 2 when the reference holds
    fewer than N_TOKENS generated ids."""
    ref = run_dir / "stages" / "1" / "evidence" / "reference"
    prompt_ids = json.loads((ref / "prompt-ids.json").read_text(encoding="utf-8"))["prompt_ids"]
    gen = json.loads((ref / "generated-ids.json").read_text(encoding="utf-8"))
    if len(gen["generated_ids"]) < N_TOKENS:
        print(f"serve_and_compare: the reference has {len(gen['generated_ids'])} generated ids; "
              f"{N_TOKENS} are needed", file=sys.stderr)
        sys.exit(2)
    return prompt_ids, gen["generated_ids"], gen.get("generated_text")


def describe_sampling(config_json: str | None) -> str:
    """Where sampling happens, from the text of one --additional-config argument."""
    if config_json is None:
        return "not set by the run script"
    try:
        tt = json.loads(config_json).get("tt")
    except (ValueError, AttributeError):
        return "unknown"
    mode = tt.get("sample_on_device_mode") if isinstance(tt, dict) else None
    return f"on device ({mode})" if mode else "host"


def serving_state(run_sh_text: str) -> dict:
    """How the bundle's run.sh serves, for the operator bundle: whether a speculative drafter is on, and
    where sampling happens. Both change what a speed figure means. The last `export QWEN36_DRAFTER=` line
    wins, as in bash; a commented line does not count."""
    drafter = "not set by the run script"
    for line in run_sh_text.splitlines():
        m = re.match(r"\s*export\s+QWEN36_DRAFTER=(.*)$", line)
        if m:
            value = m.group(1).strip().strip("\"'")
            drafter = "off" if value == "" else f"on ({value})"
    m = re.search(r"--additional-config\s+'([^']*)'", run_sh_text)
    return {"drafter": drafter, "sampling": describe_sampling(m.group(1) if m else None)}


def write_report(cfg: dict, run_dir: Path, cache: Path, weights_env: dict, m: dict, tokenizer,
                 reference, log_path: Path, extra: dict | None = None) -> dict:
    """Step (g). Writes swap-check.json next to log_path and prints the result draft. `extra` adds
    keys to the report; serve_and_compare_container.py records its docker command there."""
    prompt_ids, generated_ids, generated_text = reference
    top1 = m["matches"] / N_TOKENS
    coh = coherence(m["free_run_text"])
    swap_json = log_path.parent / "swap-check.json"
    draft = {"serves": True,
             "server_ready_s": m["server_ready_s"],
             "coherent": coh["coherent"],
             "free_run_text": m["free_run_text"][:200],
             "top1_agreement": top1,
             "n_tokens": N_TOKENS,
             "cache_dir": str(cache),
             "evidence": [os.path.relpath(swap_json, run_dir), os.path.relpath(log_path, run_dir)]}
    report = {"label": "measured", "new_model_id": cfg["new_model_id"],
              "model_dir": weights_env["MODEL_WEIGHTS_DIR"], "port": cfg["port"],
              "weights_dir_env": weights_env, "hf_model_env": weights_env["HF_MODEL"],
              "server_ready_s": m["server_ready_s"], "n_tokens": N_TOKENS,
              "matches": m["matches"], "top1_agreement": top1, **coh,
              "prompt_ids": prompt_ids, "reference_generated_ids": generated_ids[:N_TOKENS],
              "reference_generated_text": generated_text, "free_run_text": m["free_run_text"],
              "free_run_ids": tokenizer.encode(m["free_run_text"], add_special_tokens=False).ids,
              "teacher_forced": m["forced"], "mismatches": m["mismatches"], **(extra or {}),
              "result_draft": draft}
    swap_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(draft, indent=2, ensure_ascii=False))
    return draft


def main() -> int:
    from tokenizers import Tokenizer      # imported here so a missing package fails before Popen

    cfg = load_config()
    run_dir = Path(cfg["run_dir"]).resolve()
    reference = load_reference(run_dir)
    tokenizer = Tokenizer.from_file(str(STAGE_DIR / "model-dir" / "tokenizer.json"))
    cache = Path(cfg["tt_cache"])
    guard_cache(cache, cfg["new_model_id"])

    evidence = STAGE_DIR / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    log_path = evidence / "server.log"
    model_dir = str(STAGE_DIR / "model-dir")
    weights_env = {"MODEL_WEIGHTS_DIR": model_dir, "HF_MODEL": model_dir}
    env = dict(os.environ, TT_CACHE_PATH=str(cache), TT_CACHE_HOME=str(cache),
               HF_HOME=str(cfg["hf_home"]), HF_HUB_OFFLINE="1", **weights_env)
    log = open(log_path, "wb")
    proc = subprocess.Popen(["bash", str(STAGE_DIR / "run.sh"), "--port", str(cfg["port"])],
                            env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        m = measure(cfg, proc, log_path, tokenizer, reference[0], reference[1])
    except ServerHTTPError as exc:
        print(f"serve_and_compare: the server returned an error: {exc}")
        sys.exit(5)
    finally:
        stop_server(proc)
        log.close()
    write_report(cfg, run_dir, cache, weights_env, m, tokenizer, reference, log_path,
                 extra={"serving": serving_state((STAGE_DIR / "run.sh").read_text(encoding="utf-8"))})
    return 0

if __name__ == "__main__":
    sys.exit(main())
