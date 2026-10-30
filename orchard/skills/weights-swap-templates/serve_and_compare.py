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
(h) The prompt suite (issue 15), only when swap_config.json has `"prompt_suite": true`, which the supervisor
    writes into the configs it drafts. After the 32-token check the script sends the fixed prompts of
    `prompt_suite.json` (copied next to this file) and records each raw reply under `prompt_suite` in
    swap-check.json. The prompt format follows the tokenizer's chat template: a template in model-dir/
    chat_template.jinja or tokenizer_config.json means /v1/chat/completions with the suite's messages (the tool
    prompt only when the template mentions tools); none means /v1/completions with the suite's plain text. The
    format, the template's sha256 and the tokenizer's special tokens are recorded. The script judges nothing:
    the stage gate reads the replies with orchard/prompt_suite.py, so a copy of this file cannot change a verdict.
    `suite_control` in the config names the swap-check.json of an earlier run of the same suite on the nearest
    model (a path, relative to the run directory or absolute). Its record is embedded, and the gate reads the
    new model's findings against it. A missing suite file or an unusable control exits 2 before any server starts.
    A request the server refuses is recorded as that prompt's error and the run goes on.
(i) The sampling probe (issue 107), only when swap_config.json has `"sampling_probe": true`. After the 32-token
    check, and after the prompt suite when that ran, the script sends one greedy request and two sampled requests
    (temperature 1.0, top-p 0.9, seeds that differ) and one hot request (temperature 2.0, top-p 0.99) with one short plain prompt to /v1/completions, and records each
    one raw under `sampling_probe`: whether the server accepted it, its error text when it refused, the seconds the
    script measured, the completion tokens the server reported, the text, and the drafter's session lines that
    the request added to server.log. The script judges nothing; orchard/sampling_probe.py reads the record. A refused
    or cut-off request is that request's error and the run goes on. For a container, whose log reaches server.log only
    when `save_log()` runs, the probe calls it before it takes each starting offset and after each request. These requests carry `top_p` and `seed`, which
    the 32-token check does not send.

Exit codes: 0 when the measurements completed, whatever they say. 3 cache guard. 4 the server
exited, or never became healthy. 5 the server answered a request with an HTTP error (its body is
printed). Any other error in this script exits non-zero through Python's traceback; the finally
still stops the server.

The 32-token check carries only model, prompt, max_tokens and temperature. Only the sampling probe (i)
sends top_p and seed, and records what the server did with them.

serve_and_compare_container.py (stage 4) is copied next to this file and imports guard_cache,
load_reference, measure, write_report and ServerHTTPError from it, so both templates measure the
same way.
"""
from __future__ import annotations

import hashlib
import http.client
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
SUITE_FILE_NAME = "prompt_suite.json"
SUITE_MAX_TOKENS = 512        # the same number as SUITE_MAX_TOKENS in orchard/defaults.py (a test holds them equal)
PROBE_PROMPT = "Write the opening paragraph of a short story about a lighthouse keeper who finds an unmarked boat on the rocks."
PROBE_MAX_TOKENS = 96         # long enough for a speed figure that is mostly decoding, short enough to cost a few seconds
PROBE_TEMPERATURE = 1.0
PROBE_TOP_P = 0.9
PROBE_SEEDS = (1234, 4321)    # two seeds: a server that applies sampling gives different text for them
PROBE_HOT = {"temperature": 2.0, "top_p": 0.99, "seed": 2468}   # one hot request, recorded for a person to read; the judge does not use it (a peaked distribution matches greedy even at this temperature)
PROBE_LOG_SETTLE_S = 0.5      # the drafter logs a session line when a request ends; wait this long before reading the log
PROBE_LOG_LINES = 4           # keep at most this many session lines per request
PROBE_LOG_LINE_CHARS = 600
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


def error_body(exc) -> str:
    """The body of an HTTP error answer, as text. A body that is cut short raises IncompleteRead, which is not an OSError; the
    status code is worth more than the body, so the failure is said in its place."""
    try:
        return exc.read().decode("utf-8", errors="replace")
    except (http.client.HTTPException, OSError) as read_error:
        return f"(the error body could not be read: {type(read_error).__name__})"


def complete(port: int, model: str, prompt: list[int], max_tokens: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ServerHTTPError(f"HTTP {exc.code}: {error_body(exc)}") from exc
    return data["choices"][0]["text"]


# ---- the prompt suite (step h) -----------------------------------------------------------------------------------

def post_json(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ServerHTTPError(f"HTTP {exc.code}: {error_body(exc)}") from exc


def suite_problems(suite) -> list[str]:
    """Only what the script needs to send the prompts. The full rules are in orchard/prompt_suite.py."""
    if not isinstance(suite, dict) or not isinstance(suite.get("prompts"), list) or not suite["prompts"]:
        return ["the suite has no prompts list"]
    out, seen = [], set()
    for i, p in enumerate(suite["prompts"]):
        ok = (isinstance(p, dict) and isinstance(p.get("id"), str) and isinstance(p.get("text"), str)
              and isinstance(p.get("messages"), list) and p["messages"])
        if not ok:
            out.append(f"prompt {i} needs a string id, a string text and a non-empty messages list")
            continue
        if p["id"] in seen:
            out.append(f"duplicate prompt id {p['id']!r}")
        seen.add(p["id"])
    return out


_REVISION = re.compile(r"[0-9a-f]{40}")


def _split_id(model_id: str) -> tuple[str, str | None]:
    repo, sep, rev = model_id.rpartition("@")
    if sep and _REVISION.fullmatch(rev):
        return repo, rev
    return model_id, None


def control_identity_problem(expected, actual) -> str | None:
    """Why a control served from `actual` is not a control of the nearest model `expected`, or None when it is. The repository
    ids must be equal; when `expected` pins a revision, the control's id must carry the same one. The same rule, word for word, is
    in orchard/prompt_suite.py (this script imports nothing from that package); a test keeps the two equal."""
    if not isinstance(expected, str) or not expected.strip():
        return "the nearest model is not known, so the control cannot be checked against it"
    if not isinstance(actual, str) or not actual.strip():
        return f"the control's record has no model id, so it cannot be checked against the nearest model {expected}"
    want_repo, want_rev = _split_id(expected.strip())
    got_repo, got_rev = _split_id(actual.strip())
    if want_repo != got_repo:
        return f"it was served from {actual.strip()}, and the nearest model is {want_repo}, so it explains nothing"
    if want_rev and got_rev is None:
        return (f"the nearest model is pinned to revision {want_rev[:12]} and the control's model id carries no revision, so its "
                "revision cannot be checked")
    if want_rev and got_rev != want_rev:
        return f"it was served at revision {got_rev[:12]}, and the nearest model is pinned to revision {want_rev[:12]}"
    return None


def load_control(cfg: dict, run_dir: Path) -> dict | None:
    """The control named by `suite_control`: {source, sha256, record}. Exits 2 when it cannot be used, which includes a control
    that was not served from the nearest model (`nearest_model_id`, with its revision when the config pins one)."""
    value = cfg.get("suite_control")
    if not value:
        return None
    path = Path(value) if os.path.isabs(str(value)) else run_dir / str(value)
    try:
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8")).get("prompt_suite")
    except (OSError, ValueError, AttributeError) as exc:
        print(f"serve_and_compare: cannot read the control {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(record, dict) or not isinstance(record.get("entries"), list):
        print(f"serve_and_compare: the control {path} has no prompt_suite record. It must be the swap-check.json "
              "of a run of the same suite on the nearest model.", file=sys.stderr)
        sys.exit(2)
    problem = control_identity_problem(cfg.get("nearest_model_id"), record.get("model_id"))
    if problem:
        print(f"serve_and_compare: the control {path} is not a control of the nearest model: {problem}.", file=sys.stderr)
        sys.exit(2)
    limit = int(cfg.get("suite_max_tokens", SUITE_MAX_TOKENS))
    got = record.get("max_tokens")
    if isinstance(got, bool) or got != limit:
        print(f"serve_and_compare: the control {path} ran with a token limit of {got!r} and this run uses {limit}. Replies cut off at "
              "different lengths cannot be compared, so regenerate the control with this run's limit (tt-orchard suite-control run).",
              file=sys.stderr)
        sys.exit(2)
    record = {k: v for k, v in record.items() if k != "control"}
    try:
        source = str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        source = str(path)
    return {"source": source, "sha256": hashlib.sha256(raw).hexdigest(), "record": record}


def suite_preflight(cfg: dict) -> dict | None:
    """None when the config does not ask for the suite. Otherwise the suite and the control, read before any
    server starts: a stage that lacks prompt_suite.json or names an unusable control exits 2."""
    if not cfg.get("prompt_suite"):
        return None
    path = STAGE_DIR / SUITE_FILE_NAME
    if not path.is_file():
        print(f"serve_and_compare: swap_config.json asks for the prompt suite, but {path} is missing. Copy "
              f"{SUITE_FILE_NAME} from the skill's templates directory next to this script.", file=sys.stderr)
        sys.exit(2)
    raw = path.read_bytes()
    try:
        suite = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        print(f"serve_and_compare: {path} is not JSON: {exc}", file=sys.stderr)
        sys.exit(2)
    problems = suite_problems(suite)
    if problems:
        print(f"serve_and_compare: {path} has problems: " + "; ".join(problems), file=sys.stderr)
        sys.exit(2)
    return {"suite": suite, "suite_sha256": hashlib.sha256(raw).hexdigest(),
            "control": load_control(cfg, Path(cfg["run_dir"]).resolve())}


def chat_template_info(model_dir: Path) -> dict:
    """Whether the tokenizer has a chat template, where it is, its sha256 and whether it mentions tools.
    model-dir/chat_template.jinja wins; else tokenizer_config.json's `chat_template` (a string, or a list of
    {name, template} of which the entry named "default" or else the first is used). A blank template is none."""
    text, source = None, None
    jinja = model_dir / "chat_template.jinja"
    if jinja.is_file():
        found = jinja.read_text(encoding="utf-8")
        if found.strip():
            text, source = found, "chat_template.jinja"
    if text is None and (model_dir / "tokenizer_config.json").is_file():
        try:
            value = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template")
        except (ValueError, AttributeError):
            value = None
        if isinstance(value, list):
            named = {t.get("name"): t.get("template") for t in value if isinstance(t, dict)}
            value = named.get("default") or next((t for t in named.values() if isinstance(t, str)), None)
        if isinstance(value, str) and value.strip():
            text, source = value, "tokenizer_config.json"
    if text is None:
        return {"present": False, "source": None, "sha256": None, "mentions_tools": False}
    return {"present": True, "source": source, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "mentions_tools": "tools" in text}


SERVED_INPUT_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json", "chat_template.jinja")


def served_files(model_dir: Path) -> dict:
    """{file name: sha256 of its bytes, or None when absent} for the tokenizer, stop-id and template files of the served directory, read through
    links. orchard/served_chat.py compares these digests with the files as they are when stage 2 or 4 ends, so a file changed after this test
    (the replies below were made with the old one) is caught."""
    out = {}
    for name in SERVED_INPUT_FILES:
        try:
            h = hashlib.sha256()
            with open(model_dir / name, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[name] = h.hexdigest()
        except OSError:
            out[name] = None
    return out


def special_tokens(model_dir: Path) -> list[str]:
    """The strings of the tokenizer's special added tokens, in file order."""
    try:
        added = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8")).get("added_tokens") or []
    except (OSError, ValueError, AttributeError):
        return []
    return [t["content"] for t in added if isinstance(t, dict) and t.get("special") is True
            and isinstance(t.get("content"), str)]


def ask_suite_prompt(port: int, model: str, prompt: dict, mode: str, max_tokens: int) -> dict:
    """One prompt, one reply, recorded raw."""
    entry = {"id": prompt["id"], "mode": mode, "content": "", "reasoning": "", "tool_calls": [],
             "finish_reason": None, "error": None}
    try:
        if mode == "chat":
            body = {"model": model, "messages": prompt["messages"], "max_tokens": max_tokens, "temperature": 0}
            if prompt.get("needs_tools"):
                body["tools"] = prompt["tools"]
            choice = post_json(port, "/v1/chat/completions", body)["choices"][0]
            msg = choice["message"]
            entry["content"] = msg.get("content") or ""
            entry["reasoning"] = msg.get("reasoning_content") or msg.get("reasoning") or ""
            entry["tool_calls"] = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments")}
                                   for c in msg.get("tool_calls") or []]
        else:
            body = {"model": model, "prompt": prompt["text"], "max_tokens": max_tokens, "temperature": 0}
            choice = post_json(port, "/v1/completions", body)["choices"][0]
            entry["content"] = choice["text"]
        entry["finish_reason"] = choice.get("finish_reason")
    except ServerHTTPError as exc:
        entry["error"] = str(exc)[:300]
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        # http.client.HTTPException (IncompleteRead, BadStatusLine) is not an OSError: one cut-off answer must not end the test.
        entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return entry


def run_suite(cfg: dict, port: int, model: str, model_dir: Path, setup: dict) -> dict:
    """Step (h): send the suite and return the record that goes under `prompt_suite` in swap-check.json."""
    template = chat_template_info(model_dir)
    mode = "chat" if template["present"] else "completion"
    max_tokens = int(cfg.get("suite_max_tokens", SUITE_MAX_TOKENS))
    entries, skipped = [], []
    for prompt in setup["suite"]["prompts"]:
        if prompt.get("needs_tools") and not template["mentions_tools"]:
            skipped.append({"id": prompt["id"], "reason": "the chat template does not mention tools"
                            if template["present"] else "there is no chat template, so no tool format"})
            continue
        entries.append(ask_suite_prompt(port, model, prompt, mode, max_tokens))
    return {"ran": True, "model_id": cfg.get("new_model_id"), "suite_sha256": setup["suite_sha256"], "mode": mode,
            "max_tokens": max_tokens,
            "chat_template": template, "special_tokens": special_tokens(model_dir), "served_files": served_files(model_dir), "entries": entries,
            "skipped": skipped, "control": setup["control"]}


def log_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def session_lines_since(path: Path, offset: int) -> str:
    """The drafter's `[dflash2-serve] ... session:` lines the server wrote after byte `offset`, at most PROBE_LOG_LINES of them."""
    try:
        with open(path, "rb") as f:
            f.seek(max(offset, 0))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    # Only the session summaries count (the judge reads nothing else). Per-step lines come first and would use up the limit before the summary.
    lines = [ln[:PROBE_LOG_LINE_CHARS] for ln in text.splitlines() if "[dflash2-serve]" in ln and " session: " in ln]
    return "\n".join(lines[:PROBE_LOG_LINES])


def refresh_log(refresh) -> None:
    """Copy the server's log to server.log now, when the process can (a container: docker logs are saved to the file only by Container.save_log).
    A failure is ignored: the probe then reads whatever the file holds."""
    if refresh is None:
        return
    try:
        refresh()
    except Exception:                                  # noqa: BLE001  docker can fail in many ways; the request itself must still be recorded
        pass


def ask_probe(port: int, model: str, params: dict, log_path: Path, settle_s: float, refresh=None) -> dict:
    """One probe request, recorded raw. `seconds` is the time the script measured around the request; `completion_tokens` is the count the
    server reports in `usage`, or None when it reports none."""
    entry = {"params": params, "accepted": False, "error": None, "seconds": None, "completion_tokens": None, "text": "",
             "finish_reason": None, "drafter_log": ""}
    refresh_log(refresh)
    offset = log_size(log_path)
    started = time.monotonic()
    try:
        body = {"model": model, "prompt": PROBE_PROMPT, "max_tokens": PROBE_MAX_TOKENS, **params}
        data = post_json(port, "/v1/completions", body)
        entry["seconds"] = round(time.monotonic() - started, 4)
        choice = data["choices"][0]
        entry["text"] = choice["text"]
        entry["finish_reason"] = choice.get("finish_reason")
        usage = data.get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        entry["completion_tokens"] = tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None
        entry["accepted"] = True
    except ServerHTTPError as exc:
        entry["error"] = str(exc)[:300]
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
    time.sleep(settle_s)
    refresh_log(refresh)
    entry["drafter_log"] = session_lines_since(log_path, offset)
    return entry


def run_sampling_probe(cfg: dict, port: int, model: str, log_path: Path, refresh=None) -> dict:
    """Step (i): one greedy, two sampled and one hot request; the record that goes under `sampling_probe` in swap-check.json."""
    settle = float(cfg.get("sampling_log_settle_s", PROBE_LOG_SETTLE_S))
    sampled = {"temperature": PROBE_TEMPERATURE, "top_p": PROBE_TOP_P}
    requests = {"greedy": ask_probe(port, model, {"temperature": 0}, log_path, settle, refresh),
                "sampled_a": ask_probe(port, model, {**sampled, "seed": PROBE_SEEDS[0]}, log_path, settle, refresh),
                "sampled_b": ask_probe(port, model, {**sampled, "seed": PROBE_SEEDS[1]}, log_path, settle, refresh),
                "sampled_hot": ask_probe(port, model, dict(PROBE_HOT), log_path, settle, refresh)}
    return {"ran": True, "model_id": cfg.get("new_model_id"), "prompt": PROBE_PROMPT, "max_tokens": PROBE_MAX_TOKENS,
            "requests": requests}


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


def measure(cfg: dict, proc, log_path: Path, tokenizer, prompt_ids, generated_ids, suite_setup: dict | None = None) -> dict:
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
    out = {"server_ready_s": round(ready_s, 1), "free_run_text": free_text,
           "matches": matches, "forced": forced, "mismatches": mismatches}
    if suite_setup is not None:
        out["prompt_suite"] = run_suite(cfg, port, model, STAGE_DIR / "model-dir", suite_setup)
    if cfg.get("sampling_probe"):
        out["sampling_probe"] = run_sampling_probe(cfg, port, model, log_path, refresh=getattr(proc, "save_log", None))
    return out


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
    if "prompt_suite" in m:
        report["prompt_suite"] = m["prompt_suite"]
    if "sampling_probe" in m:
        report["sampling_probe"] = m["sampling_probe"]
    swap_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(draft, indent=2, ensure_ascii=False))
    return draft


def main() -> int:
    from tokenizers import Tokenizer      # imported here so a missing package fails before Popen

    cfg = load_config()
    run_dir = Path(cfg["run_dir"]).resolve()
    reference = load_reference(run_dir)
    tokenizer = Tokenizer.from_file(str(STAGE_DIR / "model-dir" / "tokenizer.json"))
    suite_setup = suite_preflight(cfg)
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
        m = measure(cfg, proc, log_path, tokenizer, reference[0], reference[1], suite_setup)
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
