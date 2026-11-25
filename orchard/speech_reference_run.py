# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The CPU reference producer of the speech route's stage 1. It runs in the reference interpreter (torch and transformers, no ttnn), as a child process of the
supervisor, and writes one JSON file.

    python -I speech_reference_run.py --mode asr|score --config CONFIG --out OUT

  asr    transcribe each file with the Voxtral model in `model_dir` (or `model_id`) on the CPU, greedy decoding, the library's own transcription request
         (`processor.apply_transcription_request`, as the upstream model card shows)
  score  transcribe each file with the speech-to-text pipeline of `model_id` (Whisper) on the CPU. Stage 1 of a text-to-speech run uses it to check that the
         scorer hears the operator's own clips, and stage 2 uses it on the bundle's audio

CONFIG is {"schema": 1, "model_id": str, "model_dir": str|null, "files": [{"name", "path"}], "language": str|null, "max_new_tokens": int}. OUT is
{"schema": 1, "engine": {"python", "transformers", "torch", "model_id", "decoding"}, "texts": {name: text}}.

This file uses only the standard library, torch and transformers, so a bare reference environment runs it. Exit codes: 0 done, 1 a file could not be
transcribed (the message names it), 2 the config is refused, 3 torch or transformers cannot be imported. The code is written against the public transformers API
and was tested with stand-in modules only; the first run on real weights is owed.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys

SCHEMA = 1
MODES = ("asr", "score")
MAX_NEW_TOKENS_LIMIT = 4096
CONFIG_LIMIT = 1 << 20
MAX_DEPTH = 32


def read_config(path):
    """The parsed config at `path`. Opened without following a link and without blocking on a pipe; a file that is not a regular file, is larger than CONFIG_LIMIT,
    is not UTF-8 JSON or nests deeper than the parser allows raises ValueError, as does any OS error."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as exc:
        raise ValueError(f"the config cannot be opened: {exc}") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("the config is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as f:
            raw = f.read(CONFIG_LIMIT + 1)
    except OSError as exc:
        raise ValueError(f"the config cannot be read: {exc}") from None
    finally:
        os.close(fd)
    if len(raw) > CONFIG_LIMIT:
        raise ValueError(f"the config is larger than {CONFIG_LIMIT} bytes")
    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=_no_constant, parse_float=_finite_float, parse_int=_bounded_int)
    except (ValueError, RecursionError) as exc:           # UnicodeDecodeError is a ValueError
        raise ValueError(f"the config is not readable JSON: {exc}") from None
    _shape_problem(doc)
    return doc


def _no_constant(name):
    raise ValueError(f"{name} is not allowed")


def _finite_float(text):
    value = float(text)      # finite: ok the next line refuses NaN and infinity
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"the number {text} is not finite")
    return value


def _bounded_int(text):
    if len(text.lstrip("-")) > 30:
        raise ValueError("an integer has too many digits")
    return int(text)


def _shape_problem(doc) -> None:
    """Raise ValueError for a config nested deeper than MAX_DEPTH or holding text that cannot be encoded as UTF-8 (a lone surrogate)."""
    stack = [(doc, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise ValueError(f"the config is nested more than {MAX_DEPTH} levels deep")
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("the config holds text that is not valid Unicode") from None
        elif isinstance(item, dict):
            for key, value in item.items():
                stack.append((key, depth + 1))
                stack.append((value, depth + 1))
        elif isinstance(item, list):
            stack.extend((x, depth + 1) for x in item)


def _config_problems(cfg) -> list[str]:
    if not isinstance(cfg, dict):
        return ["the config is not a JSON object"]
    out = []
    if cfg.get("schema") != SCHEMA:
        out.append(f"schema is {cfg.get('schema')!r}; this code reads schema {SCHEMA}")
    if not isinstance(cfg.get("model_id"), str) or not cfg["model_id"]:
        out.append("model_id must be a non-empty string")
    for key in ("model_dir", "language"):
        if cfg.get(key) is not None and not isinstance(cfg[key], str):
            out.append(f"{key} must be a string or null")
    n = cfg.get("max_new_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_NEW_TOKENS_LIMIT:
        out.append(f"max_new_tokens must be a whole number from 1 to {MAX_NEW_TOKENS_LIMIT}")
    files = cfg.get("files")
    if not isinstance(files, list) or not files:
        out.append("files must be a non-empty list")
    else:
        for f in files:
            if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"] or "/" in f["name"] or f["name"] in (".", ".."):
                out.append(f"a file needs a plain name, got {str(f)[:100]}")
            elif not isinstance(f.get("path"), str) or not f["path"]:
                out.append(f"the file {f['name']} needs a path")
    return out


def _engine(cfg) -> dict:
    import torch
    import transformers
    return {"python": sys.executable, "transformers": transformers.__version__, "torch": torch.__version__, "model_id": cfg["model_id"], "decoding": "greedy"}


def _asr(cfg) -> dict:
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    repo = cfg.get("model_dir") or cfg["model_id"]
    processor = AutoProcessor.from_pretrained(repo)
    model = VoxtralForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cpu")
    texts = {}
    for f in cfg["files"]:
        try:
            inputs = processor.apply_transcription_request(language=cfg.get("language") or "en", audio=f["path"], model_id=cfg["model_id"])
            inputs = inputs.to("cpu", dtype=torch.bfloat16)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
            texts[f["name"]] = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        except Exception as exc:
            raise _FileError(f["name"], exc) from exc
    return texts


def _score(cfg) -> dict:
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=cfg["model_id"], device="cpu")
    kwargs = {"generate_kwargs": {"language": cfg["language"]}} if cfg.get("language") else {}
    texts = {}
    for f in cfg["files"]:
        try:
            texts[f["name"]] = str(asr(f["path"], **kwargs)["text"]).strip()
        except Exception as exc:
            raise _FileError(f["name"], exc) from exc
    return texts


class _FileError(Exception):
    def __init__(self, name, exc):
        super().__init__(f"{name}: {type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="speech_reference_run")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.mode not in MODES:
        print(f"mode {args.mode!r} is not one of {MODES}", file=sys.stderr)
        return 2
    try:
        cfg = read_config(args.config)
    except ValueError as exc:
        print(f"the config cannot be read: {exc}", file=sys.stderr)
        return 2
    problems = _config_problems(cfg)
    if problems:
        print("the config is refused: " + "; ".join(problems), file=sys.stderr)
        return 2
    try:
        engine = _engine(cfg)
    except ImportError as exc:
        print(f"torch and transformers are needed in this interpreter: {exc}", file=sys.stderr)
        return 3
    try:
        texts = _asr(cfg) if args.mode == "asr" else _score(cfg)
    except ImportError as exc:
        print(f"torch and transformers are needed in this interpreter: {exc}", file=sys.stderr)
        return 3
    except _FileError as exc:
        print(f"transcription failed for {exc}", file=sys.stderr)
        return 1
    tmp = f"{args.out}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"schema": SCHEMA, "engine": engine, "texts": texts}, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
