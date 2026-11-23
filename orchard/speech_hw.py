# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The command stage 2 of a speech run executes under the lease (orchard/speech_route.py).

    python3 orchard/speech_hw.py --config STAGE_DIR/speech_config.json

It starts the published bundle with `tt-model serve BUNDLE`, reads the address the command printed, runs the checks of orchard/speech_check.py against that
server, saves every answer in `speech_result.json` (and, for text to speech, each prompt's audio under `audio/`), and runs `tt-model stop BUNDLE` in a
`finally`, so a failure in the middle never leaves the bundle running on the leased chips. It uses the standard library and this repository only.

Exit codes: 0 when the bundle started, answered the checks (whether the answers pass is the gate's question) and stopped; 1 when it did not start, no address was
found, the checks could not run or the stop failed; 2 when the config is refused before anything started. The result file is written in every case but 2.

The command never packages, uploads or republishes anything: its only external commands are `tt-model serve` and `tt-model stop` with the bundle id from the
bundle table (orchard/speech_models.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):                       # run as a script: the repository root is the parent of this file's directory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchard import safeio, speech_check, speech_models  # noqa: E402

SCHEMA = 1
RESULT_FILE = "speech_result.json"
EVIDENCE_DIR = "evidence"
SERVE_OUTPUT = "serve-output.txt"
AUDIO_DIR = "audio"
CONFIG_LIMIT = 1 << 20
OUTPUT_LIMIT = 1 << 20
_ADDRESS = re.compile(r"http://(?:127\.0\.0\.1|localhost|\[::1\]):[0-9]{1,5}")
_BUNDLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,40}\Z")


def _seconds(x) -> bool:
    return not isinstance(x, bool) and safeio.is_finite_number(x) and 0 < x <= 86400


def config_problems(cfg) -> list[str]:
    """Why a config is refused, or an empty list."""
    if not isinstance(cfg, dict):
        return ["the config is not a JSON object"]
    out = []
    if cfg.get("schema") != SCHEMA:
        out.append(f"schema is {cfg.get('schema')!r}; {SCHEMA} is the one this version reads")
    task = cfg.get("task")
    if task not in speech_models.TASKS:
        out.append(f"task {task!r} is not one of {speech_models.TASKS}")
    bundle = cfg.get("bundle")
    if not isinstance(bundle, str) or not _BUNDLE.match(bundle):
        out.append(f"bundle {bundle!r} is not a repository id of the form owner/name")
    elif not any(b.bundle == bundle and b.task == task for b in speech_models.PUBLISHED):
        out.append(f"bundle {bundle} is not a known bundle for {task}")
    stage = cfg.get("stage_dir")
    if not isinstance(stage, str) or not Path(stage).is_absolute():
        out.append("stage_dir must be an absolute path")
    for key in ("serve_timeout_s", "timeout_s"):
        if not _seconds(cfg.get(key)):
            out.append(f"{key} must be a number of seconds above 0 and at most 86400")
    if "tt_model" in cfg and (not isinstance(cfg["tt_model"], str) or not cfg["tt_model"]):
        out.append("tt_model, when given, must be the name or path of the tt-model executable")
    if task == "speech-to-text":
        clips = cfg.get("clips")
        if not isinstance(clips, list) or not clips:
            out.append("clips must be a non-empty list for speech to text")
        else:
            for c in clips:
                if not (isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("path"), str)
                        and isinstance(c.get("text"), str) and speech_check.words(c["text"])):
                    out.append(f"a clip needs a name, a path and a transcript with words, got {str(c)[:120]}")
    elif task == "text-to-speech":
        prompts = cfg.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            out.append("prompts must be a non-empty list for text to speech")
        else:
            for p in prompts:
                if not (isinstance(p, dict) and isinstance(p.get("id"), str) and _ID.match(p["id"]) and isinstance(p.get("text"), str)
                        and speech_check.words(p["text"])):
                    out.append(f"a prompt needs an id of letters, digits, _ and - and text with words, got {str(p)[:120]}")
    return out


def _run(argv, timeout):
    """(exit code or None, output text, a sentence when the command could not be run to its end)."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, errors="replace")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        return None, out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out), f"{Path(argv[0]).name} {argv[1]} ran past its deadline of {timeout} s"
    except OSError as exc:
        return None, "", f"{argv[0]} could not be run: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), ""


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safeio.atomic_write(path, data)


def run(config_path) -> int:
    try:
        cfg = safeio.read_json(Path(config_path), max_bytes=CONFIG_LIMIT)
    except (safeio.SafeReadError, OSError, ValueError) as exc:
        print(f"the config cannot be read: {exc}", file=sys.stderr)
        return 2
    problems = config_problems(cfg)
    if problems:
        print("the config is refused: " + "; ".join(problems), file=sys.stderr)
        return 2
    task, bundle, stage = cfg["task"], cfg["bundle"], Path(cfg["stage_dir"])
    tt_model = cfg.get("tt_model", "tt-model")
    result = {"schema": SCHEMA, "task": task, "bundle": bundle, "url": None, "served": {"serve_exit": None, "stop_exit": None}}
    error = ""
    try:
        code, output, why = _run([tt_model, "serve", bundle], cfg["serve_timeout_s"])
        result["served"]["serve_exit"] = code
        _write(stage / EVIDENCE_DIR / SERVE_OUTPUT, output[-OUTPUT_LIMIT:])
        if why:
            error = why
        elif code != 0:
            error = f"tt-model serve exited with code {code}"
        else:
            found = _ADDRESS.search(output)
            if found is None:
                error = "no local address (http://127.0.0.1:PORT) in the output of tt-model serve"
            else:
                try:
                    speech_check.parse_base_url(found.group(0))
                    result["url"] = found.group(0)
                except speech_check.SpeechCheckError as exc:
                    error = f"the address in the serve output is refused: {exc}"
        if not error:
            try:
                _collect(cfg, result, stage)
            except speech_check.SpeechCheckError as exc:
                error = f"the checks could not run: {exc}"
    except Exception as exc:                                  # the stop below still runs; the result says what happened
        error = f"{type(exc).__name__}: {exc}"
    finally:
        code, output, why = _run([tt_model, "stop", bundle], cfg["timeout_s"])
        result["served"]["stop_exit"] = code
        if why or code != 0:
            error = (error + "; " if error else "") + (why or f"tt-model stop exited with code {code}")
    if error:
        result["error"] = error
    _write(stage / RESULT_FILE, json.dumps(result, indent=1, sort_keys=True) + "\n")
    return 0 if not error else 1


def _collect(cfg, result, stage: Path) -> None:
    url, timeout = result["url"], float(cfg["timeout_s"])      # finite: ok config_problems checked it with safeio.is_finite_number
    if cfg["task"] == "speech-to-text":
        results = speech_check.check_speech_to_text(url, [(c["path"], c["text"]) for c in cfg["clips"]], timeout=timeout)
        result["checks"] = [r.as_dict() for r in results]
        return
    result["prompts"] = []
    for p in cfg["prompts"]:
        saved = {}

        def keep(name, data, p=p, saved=saved):
            if name == "speech":
                rel = f"{AUDIO_DIR}/{p['id']}.wav"
                _write(stage / rel, data)
                saved["audio"] = {"file": rel, "sha256": hashlib.sha256(data).hexdigest()}

        results = speech_check.check_text_to_speech(url, p["text"], timeout=timeout, on_audio=keep)
        result["prompts"].append({"id": p["id"], "checks": [r.as_dict() for r in results], "audio": saved.get("audio"), "heard": None})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="speech_hw", description="serve a speech bundle, check it, save the answers and stop it")
    ap.add_argument("--config", required=True)
    return run(ap.parse_args(argv).config)


if __name__ == "__main__":
    sys.exit(main())
