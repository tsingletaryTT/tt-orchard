# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The speech route: records and gates for a run on a published speech bundle (docs/superpowers/specs/2026-11-23-speech-route-design.md).

A run started with the input `speech_route=<task>` (`speech-to-text` or `text-to-speech`) takes the stage table `SPEECH_TABLE` in orchard/stages.py. The
route never boots the coder, never copies or uploads a model's weights and never writes a port. It serves the bundle its author published, under a
lease, and compares what the bundle answers with a CPU reference.

This module holds what is the same for every speech run and has no side effects: which run takes the route, the licence record, the shape of the
stage 1 reference and of the stage 2 result, and the problems each gate finds. Every number a gate relies on is recomputed from the saved clips, the saved
audio and the transcripts (the word error rate is `speech_check.wer`). A recorded number that differs from the recomputed one is a problem.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

from orchard import safeio, speech_check, speech_models

ROUTE_INPUT = "speech_route"          # the run_start input that selects the route; its value is a task in speech_models.TASKS
CLIPS_INPUT = "speech_clips"          # the operator's directory of clips: NAME.wav|flac|mp3|ogg with NAME.txt holding the expected text
PROMPTS_INPUT = "speech_prompts"      # a text file with one prompt per line (text to speech); DEFAULT_PROMPTS when absent
SCHEMA = 1
REFERENCE_FILE = "speech_reference.json"     # in stage 1's directory
RESULT_FILE = "speech_result.json"    # in stage 2's directory
CLIP_DIR = "clips"                    # stage 1's copy of the operator's clips, so the hashes can be checked later
AUDIO_DIR = "audio"                   # stage 2's saved speech, one WAV per prompt
CLIP_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
CLIP_LIMIT = speech_check.CLIP_LIMIT
JSON_LIMIT = 8 << 20

# Bars. Provisional until the first run on a chip (the design note gives the evidence for each).
MAX_ASR_VS_REFERENCE = 0.05           # the card's longest clip differed from the reference in 2 of 521 words (0.004)
MAX_ASR_VS_EXPECTED = speech_check.DEFAULT_MAX_WER_TEXT
MAX_TTS_WER = speech_check.DEFAULT_MAX_WER_SPEECH
MAX_SCORER_WER = 0.10                 # a scorer that mishears the operator's own clips more than this says nothing about the bundle's speech

DEFAULT_PROMPTS = (
    "The lighthouse keeper wrote in his journal every night.",
    "She tuned the old radio slowly until the music came through.",
    "The expedition mapped the cave system for three long weeks.",
)

MAX_CLIPS = 8
MAX_PROMPTS = 16
EXPECTED_LIMIT = speech_check.TEXT_LIMIT
SCORER_MODEL = "openai/whisper-large-v3-turbo"      # the scorer the Voxtral TTS card names for its word error rates
PRODUCER = Path(__file__).with_name("speech_reference_run.py")
HW_SCRIPT = Path(__file__).with_name("speech_hw.py")
CONFIG_FILE = "speech_config.json"       # in stage 2's directory: the in-lease command's config
SPEECH_ROUTE_SELECTED = "speech route selected"
STDERR_TAIL = 2000
PRODUCER_ENV_KEYS = ("PATH", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")      # nothing else reaches the reference interpreter; no token
PRODUCER_MODES = ("asr", "score")

REFERENCE_DEADLINE_S = 3600.0         # stage 1's CPU run (a 3B or 4B model and a Whisper scorer on the host's cores)
CHIP_DEADLINE_S = 3600.0              # stage 2's lease: the bundle's first start, the checks and the stop
SCORE_DEADLINE_S = 1800.0
STAGE_DISK_GB = 40


class SpeechRouteError(ValueError):
    """An input of the route that is refused, with a sentence that says what to change."""


def _text(x) -> bool:
    return isinstance(x, str) and bool(speech_check.words(x))


# ---- which run takes the route --------------------------------------------------------------------------------------


def selected(entries) -> str | None:
    """The task in the first `run_start`'s inputs, when it is one of speech_models.TASKS. The run's choice is read from the ledger, so a resume keeps it."""
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and e.get("event") == "run_start":
            data = e.get("data")
            inputs = data.get("inputs") if isinstance(data, dict) else None
            value = inputs.get(ROUTE_INPUT) if isinstance(inputs, dict) else None
            return value if isinstance(value, str) and value in speech_models.TASKS else None
    return None


# ---- the bundle and the licence ------------------------------------------------------------------------------------------


def bundle_for_task(task, model_id) -> speech_models.Bundle | None:
    """The known bundle that serves `model_id` for `task`, or None (no bundle, or the model is for the other task)."""
    b = speech_models.bundle_for(model_id)
    return b if b is not None and b.task == task else None


def license_line(bundle: speech_models.Bundle) -> str:
    """The sentence every speech run carries. It names the licence as the bundle's card states it and says the run served the bundle and copied no weights.
    It gives no opinion about what the licence allows."""
    name = bundle.upstream.split("/", 1)[-1]
    kind = f"{bundle.license} (non-commercial use only)" if bundle.non_commercial else bundle.license
    return (f"{name} is used under its licence, {kind}, as the bundle's card states it. This run served the bundle {bundle.bundle} and did not copy or "
            "republish any weights.")


def license_record(bundle: speech_models.Bundle) -> dict:
    return {"id": bundle.license, "non_commercial": bundle.non_commercial, "line": license_line(bundle)}


# ---- stage 1: the reference record --------------------------------------------------------------------------------------


def _file_problems(stage_dir, entry, what) -> list[str]:
    """Problems with one recorded clip: its name, its file under stage_dir/clips, and the hash and size recorded for it."""
    name = entry.get("name")
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in (".", "..") or "\x00" in name:
        return [f"{what} has a name that is not a plain file name: {name!r}"]
    try:
        data = safeio.read_bytes(Path(stage_dir) / CLIP_DIR / name, max_bytes=CLIP_LIMIT)
    except (safeio.SafeReadError, OSError, ValueError) as exc:
        return [f"{what} {name}: the saved clip cannot be read ({exc})"]
    out = []
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry.get("sha256"):
        out.append(f"{what} {name}: the saved clip's sha256 is {digest[:12]}, the record says {str(entry.get('sha256'))[:12]}")
    if len(data) != entry.get("bytes") or isinstance(entry.get("bytes"), bool):
        out.append(f"{what} {name}: the saved clip is {len(data)} bytes, the record says {entry.get('bytes')!r} bytes")
    return out


def _license_problems(doc, bundle) -> list[str]:
    if doc.get("license") != license_record(bundle):
        return [f"the licence record is {doc.get('license')!r}; the bundle table gives {license_record(bundle)!r}"]
    return []


def _engine_problems(doc) -> list[str]:
    ref = doc.get("reference")
    if not isinstance(ref, dict):
        return ["the record has no reference object (the interpreter, library versions and model that made it)"]
    return [f"reference.{k} is missing or not text" for k in ("python", "transformers", "torch", "model_id") if not isinstance(ref.get(k), str) or not ref[k]]


def reference_problems(doc, stage_dir) -> list[str]:
    """What is wrong with stage 1's `reference.json`, or an empty list. `stage_dir` holds the saved clips."""
    if not isinstance(doc, dict):
        return ["reference.json is not a JSON object"]
    out = []
    if doc.get("schema") != SCHEMA:
        out.append(f"schema is {doc.get('schema')!r}; this code reads schema {SCHEMA}")
    task = doc.get("task")
    if task not in speech_models.TASKS:
        return out + [f"task {task!r} is not one of {speech_models.TASKS}"]
    model = doc.get("model")
    if not isinstance(model, str):
        return out + ["model is missing or not text"]
    bundle = bundle_for_task(task, model)
    if bundle is None:
        return out + [f"for task {task}, no known bundle serves {model!r}"]
    out += _license_problems(doc, bundle) + _engine_problems(doc)
    if task == "speech-to-text":
        out += _asr_reference_problems(doc, stage_dir)
    else:
        out += _tts_reference_problems(doc, stage_dir)
    return out


def _clips_list(doc, key):
    clips = doc.get(key)
    return clips if isinstance(clips, list) and clips else None


def _asr_reference_problems(doc, stage_dir) -> list[str]:
    clips = _clips_list(doc, "clips")
    if clips is None:
        return ["clips is missing or empty: a speech-to-text reference needs at least one clip with its transcript"]
    out, seen = [], set()
    for entry in clips:
        if not isinstance(entry, dict):
            out.append("a clip record is not an object")
            continue
        name = entry.get("name")
        if name in seen:
            out.append(f"the clip {name!r} is listed twice")
        seen.add(name)
        out += _file_problems(stage_dir, entry, "clip")
        if not _text(entry.get("text")):
            out.append(f"clip {name!r} has no reference transcript with words")
        if entry.get("expected") is not None and not _text(entry.get("expected")):
            out.append(f"clip {name!r} has an expected text with no words")
    return out


def _tts_reference_problems(doc, stage_dir) -> list[str]:
    out = []
    prompts = _clips_list(doc, "prompts")
    if prompts is None:
        out.append("prompts is missing or empty: a text-to-speech run needs at least one prompt")
    else:
        ids = set()
        for p in prompts:
            if not isinstance(p, dict) or not isinstance(p.get("id"), str) or not p["id"] or not _text(p.get("text")):
                out.append(f"a prompt needs an id and text with words, got {p!r}"[:200])
                continue
            if p["id"] in ids:
                out.append(f"the prompt id {p['id']!r} is listed twice")
            ids.add(p["id"])
    cal = _clips_list(doc, "calibration")
    if cal is None:
        return out + ["calibration is missing or empty: the scorer has to be shown to hear the operator's clips before it judges the bundle"]
    for entry in cal:
        if not isinstance(entry, dict):
            out.append("a calibration record is not an object")
            continue
        name = entry.get("name")
        out += _file_problems(stage_dir, entry, "calibration clip")
        if not _text(entry.get("expected")):
            out.append(f"calibration clip {name!r} has no expected text with words")
            continue
        if not isinstance(entry.get("text"), str):
            out.append(f"calibration clip {name!r} has no scorer transcript")
            continue
        errors, n = speech_check.wer(entry["expected"], entry["text"])
        rate = errors / n
        recorded = entry.get("wer")
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or abs(recorded - rate) > 1e-9:
            out.append(f"calibration clip {name!r}: the recorded wer {recorded!r} differs from the recomputed {rate:.4f}")
        if rate > MAX_SCORER_WER:
            out.append(f"the scorer's word error rate on calibration clip {name!r} is {rate:.2f}, above {MAX_SCORER_WER}: it cannot judge the bundle's speech")
    return out


# ---- stage 2: the result record -----------------------------------------------------------------------------------------

ASR_STRUCTURE = ("health", "models", "bad request")
TTS_STRUCTURE = ("health", "models", "speech", "bad voice", "missing input")


def _by_name(checks) -> dict | None:
    """{name: check} for a list of check objects, or None when it is not that."""
    if not isinstance(checks, list) or not all(isinstance(c, dict) and isinstance(c.get("name"), str) for c in checks):
        return None
    return {c["name"]: c for c in checks}


def _check_problems(by_name, names, where) -> list[str]:
    out = []
    for name in names:
        c = by_name.get(name)
        if c is None:
            out.append(f"{where}no {name} check in the result")
        elif c.get("status") != "pass":
            out.append(f"{where}the {name} check status is {c.get('status')!r}: {str(c.get('detail'))[:160]}")
    return out


def result_problems(result, ref, stage_dir) -> list[str]:
    """What is wrong with stage 2's `speech_result.json`, judged against stage 1's reference record, or an empty list. `stage_dir` holds the saved audio."""
    if not isinstance(result, dict):
        return ["speech_result.json is not a JSON object"]
    if not isinstance(ref, dict) or ref.get("task") not in speech_models.TASKS:
        return ["stage 1's reference record is not usable, so stage 2's answers have nothing to be compared with"]
    task = ref["task"]
    out = []
    if result.get("schema") != SCHEMA:
        out.append(f"schema is {result.get('schema')!r}; this code reads schema {SCHEMA}")
    if result.get("task") != task:
        out.append(f"task is {result.get('task')!r}; stage 1's reference is for {task}")
    bundle = bundle_for_task(task, ref.get("model"))
    if bundle is None:
        out.append(f"for task {task}, no known bundle serves {ref.get('model')!r}")
    elif result.get("bundle") != bundle.bundle:
        out.append(f"bundle is {result.get('bundle')!r}; the bundle table gives {bundle.bundle} for {ref.get('model')}")
    try:
        speech_check.parse_base_url(result.get("url"))
    except speech_check.SpeechCheckError as exc:
        out.append(f"url is not an address the checker accepts: {exc}")
    served = result.get("served")
    if not isinstance(served, dict):
        out.append("served is missing: it records the exit codes of `tt-model serve` and `tt-model stop`")
    else:
        if served.get("serve_exit") != 0 or isinstance(served.get("serve_exit"), bool):
            out.append(f"served.serve_exit is {served.get('serve_exit')!r}; the bundle's start has to exit 0")
        if served.get("stop_exit") != 0 or isinstance(served.get("stop_exit"), bool):
            out.append(f"served.stop_exit is {served.get('stop_exit')!r}; the bundle has to be stopped, and the stop has to exit 0")
    if result.get("task") != task:
        return out
    return out + (_asr_result_problems(result, ref) if task == "speech-to-text" else _tts_result_problems(result, ref, stage_dir))


def _asr_result_problems(result, ref) -> list[str]:
    by_name = _by_name(result.get("checks"))
    if by_name is None:
        return ["checks is not a list of check objects"]
    out = _check_problems(by_name, ASR_STRUCTURE, "")
    for clip in ref.get("clips") if isinstance(ref.get("clips"), list) else []:
        name = clip.get("name")
        where = f"clip {name}: "
        c = by_name.get(f"transcribe {name}")
        if c is None:
            out.append(f"{where}no transcribe check in the result")
            continue
        if c.get("status") != "pass":
            out.append(f"{where}the transcribe check status is {c.get('status')!r}: {str(c.get('detail'))[:160]}")
        data = c.get("data")
        heard = data.get("text") if isinstance(data, dict) else None
        if not isinstance(heard, str):
            out.append(f"{where}the check holds no transcript text")
            continue
        errors, n = speech_check.wer(clip["text"], heard)
        rate = errors / n
        recorded = data.get("wer")
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or abs(recorded - rate) > 1e-9:
            out.append(f"{where}the recorded wer {recorded!r} differs from the recomputed {rate:.4f}")
        if rate > MAX_ASR_VS_REFERENCE:
            out.append(f"{where}word error rate {rate:.3f} against the reference transcript is above {MAX_ASR_VS_REFERENCE} "
                       f"(reference {clip['text'][:80]!r}, heard {heard[:80]!r})")
        if clip.get("expected") is not None:
            e_errors, e_n = speech_check.wer(clip["expected"], heard)
            if e_errors / e_n > MAX_ASR_VS_EXPECTED:
                out.append(f"{where}word error rate {e_errors / e_n:.3f} against the expected text is above {MAX_ASR_VS_EXPECTED}")
    return out


def _tts_result_problems(result, ref, stage_dir) -> list[str]:
    prompts = result.get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(p, dict) for p in prompts):
        return ["prompts is not a list of objects"]
    got = {p.get("id"): p for p in prompts}
    out = []
    for want in ref.get("prompts") if isinstance(ref.get("prompts"), list) else []:
        pid, where = want.get("id"), f"prompt {want.get('id')}: "
        p = got.get(pid)
        if p is None:
            out.append(f"{where}not in the result")
            continue
        by_name = _by_name(p.get("checks"))
        if by_name is None:
            out.append(f"{where}checks is not a list of check objects")
            continue
        out += _check_problems(by_name, TTS_STRUCTURE, where)
        out += _audio_problems(p, by_name.get("speech"), want["text"], stage_dir, where)
        heard = p.get("heard")
        if not isinstance(heard, str):
            out.append(f"{where}no scorer transcript of the saved audio")
            continue
        errors, n = speech_check.wer(want["text"], heard)
        if errors / n > MAX_TTS_WER:
            out.append(f"{where}word error rate {errors / n:.3f} of the scorer's transcript against the prompt is above {MAX_TTS_WER}")
    return out


def _audio_problems(p, speech_check_result, text, stage_dir, where) -> list[str]:
    audio = p.get("audio")
    rel = audio.get("file") if isinstance(audio, dict) else None
    parts = Path(rel).parts if isinstance(rel, str) and rel else ()
    if not parts or Path(rel).is_absolute() or ".." in parts or parts[0] != AUDIO_DIR or len(parts) != 2:
        return [f"{where}the audio file path {rel!r} is not a file directly under {AUDIO_DIR}/"]
    try:
        data = safeio.read_bytes(Path(stage_dir) / rel, max_bytes=speech_check.AUDIO_LIMIT)
    except (safeio.SafeReadError, OSError, ValueError) as exc:
        return [f"{where}the saved audio {rel} cannot be read ({exc})"]
    digest = hashlib.sha256(data).hexdigest()
    if digest != audio.get("sha256"):
        return [f"{where}the saved audio's sha256 is {digest[:12]}, the record says {str(audio.get('sha256'))[:12]}"]
    ok, detail, got = speech_check._judge_wav(data, "audio/wav", text, speech_check.DEFAULT_SAMPLE_RATE)
    if not ok:
        return [f"{where}the saved audio fails the WAV checks: {detail}"]
    recorded = ((speech_check_result or {}).get("data") or {}).get("seconds")
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or abs(recorded - got["seconds"]) > 1e-6:
        return [f"{where}the recorded seconds {recorded!r} differ from the saved audio's {got['seconds']:.4f}"]
    return []


# ---- stage 1: assembling the reference -----------------------------------------------------------------------------------


def discover_clips(directory) -> list[dict]:
    """The operator's clips in `directory`: [{"name", "path", "expected"}] sorted by name. A file with a suffix in CLIP_SUFFIXES is a clip. NAME.txt next to
    it, when there is one, holds the expected text. A link, a clip over CLIP_LIMIT and an expected text that is not UTF-8 text with words (at most EXPECTED_LIMIT
    characters) are refused. Other files are ignored."""
    d = Path(str(directory))
    if os.path.islink(d) or not d.is_dir():
        raise SpeechRouteError(f"the clips directory {d} is not a directory (a link is not followed)")
    found = []
    for entry in sorted(os.scandir(d), key=lambda e: e.name):
        if Path(entry.name).suffix.lower() not in CLIP_SUFFIXES:
            continue
        st = os.lstat(entry.path)
        if not stat.S_ISREG(st.st_mode):
            raise SpeechRouteError(f"the clip {entry.name} is not a regular file (a link is not followed)")
        if st.st_size > CLIP_LIMIT:
            raise SpeechRouteError(f"the clip {entry.name} is {st.st_size} bytes, above the limit of {CLIP_LIMIT}")
        found.append({"name": entry.name, "path": str(Path(entry.path)), "expected": _expected_for(d, entry.name)})
    if not found:
        raise SpeechRouteError(f"no clips in {d}: it needs files ending in {', '.join(CLIP_SUFFIXES)}")
    if len(found) > MAX_CLIPS:
        raise SpeechRouteError(f"{len(found)} clips in {d}; at most {MAX_CLIPS} are used")
    return found


def _expected_for(d: Path, clip_name: str):
    txt = d / (Path(clip_name).stem + ".txt")
    if not os.path.lexists(txt):
        return None
    try:
        raw = safeio.read_bytes(txt, max_bytes=EXPECTED_LIMIT * 4)
        text = raw.decode("utf-8").strip()
    except (safeio.SafeReadError, OSError, ValueError) as exc:          # UnicodeDecodeError is a ValueError
        raise SpeechRouteError(f"the expected text {txt.name} cannot be read as UTF-8 text ({exc})") from None
    if not _text(text) or len(text) > EXPECTED_LIMIT:
        raise SpeechRouteError(f"the expected text {txt.name} must have words and at most {EXPECTED_LIMIT} characters")
    return text


def save_clips(clips, stage_dir) -> list[dict]:
    """Copy each clip into stage_dir/clips and return [{"name", "sha256", "bytes", "expected"}] for the reference record."""
    out = []
    target = Path(stage_dir) / CLIP_DIR
    target.mkdir(parents=True, exist_ok=True)
    for c in clips:
        try:
            data = safeio.read_bytes(c["path"], max_bytes=CLIP_LIMIT)
        except (safeio.SafeReadError, OSError, ValueError) as exc:
            raise SpeechRouteError(f"the clip {c['name']} cannot be read: {exc}") from None
        safeio.atomic_write(target / c["name"], data)
        out.append({"name": c["name"], "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "expected": c.get("expected")})
    return out


def read_prompts(path) -> list[dict]:
    """The prompts of a text-to-speech run: one per non-blank line of `path`, or DEFAULT_PROMPTS when `path` is None. Each is {"id", "text", "voice"}."""
    if path is None:
        lines = list(DEFAULT_PROMPTS)
    else:
        try:
            lines = [ln.strip() for ln in safeio.read_bytes(path, max_bytes=EXPECTED_LIMIT * 16).decode("utf-8").splitlines() if ln.strip()]
        except (safeio.SafeReadError, OSError, ValueError) as exc:
            raise SpeechRouteError(f"the prompts file cannot be read as UTF-8 text ({exc})") from None
    if not lines or not all(_text(ln) and len(ln) <= EXPECTED_LIMIT for ln in lines):
        raise SpeechRouteError("the prompts must be lines of text with words, one prompt per line, and at least one")
    if len(lines) > MAX_PROMPTS:
        raise SpeechRouteError(f"{len(lines)} prompts; at most {MAX_PROMPTS} are used")
    return [{"id": f"p{i}", "text": t, "voice": None} for i, t in enumerate(lines)]


def producer_config(model_id, model_dir, files, *, language=None, max_new_tokens=512) -> dict:
    """The config orchard/speech_reference_run.py reads. `files` is [{"name", "path"}]."""
    return {"schema": SCHEMA, "model_id": model_id, "model_dir": model_dir, "files": [{"name": f["name"], "path": f["path"]} for f in files],
            "language": language, "max_new_tokens": max_new_tokens}


def producer_command(python, mode, config_path, out_path) -> list[str]:
    """The argument list that runs the producer in `python` (the reference interpreter), isolated from the working directory and the user's site."""
    if mode not in PRODUCER_MODES:
        raise SpeechRouteError(f"producer mode {mode!r} is not one of {PRODUCER_MODES}")
    return [str(python), "-I", str(PRODUCER), "--mode", mode, "--config", str(config_path), "--out", str(out_path)]


def build_reference(task, model_id, entries, engine, texts, prompts=None) -> dict:
    """The reference record of stage 1. `entries` come from save_clips, `texts` is the producer's {clip name: transcript}, `engine` its engine record. For text to
    speech the texts are the scorer's and each clip must have an expected text, because the record states the scorer's word error rate on it."""
    bundle = bundle_for_task(task, model_id)
    if bundle is None:
        raise SpeechRouteError(f"for task {task}, no known bundle serves {model_id!r}")
    doc = {"schema": SCHEMA, "task": task, "model": model_id, "reference": dict(engine), "license": license_record(bundle)}
    rows = []
    for e in entries:
        text = texts.get(e["name"]) if isinstance(texts, dict) else None
        if not isinstance(text, str):
            raise SpeechRouteError(f"the producer gave no transcript for the clip {e['name']}")
        rows.append({"name": e["name"], "sha256": e["sha256"], "bytes": e["bytes"], "expected": e.get("expected"), "text": text})
    if task == "speech-to-text":
        doc["clips"] = rows
        return doc
    for r in rows:
        if not _text(r["expected"]):
            raise SpeechRouteError(f"the clip {r['name']} has no expected text (a NAME.txt next to it): the scorer is calibrated on it")
        errors, n = speech_check.wer(r["expected"], r["text"])
        r["wer"] = errors / n
    doc["prompts"] = list(prompts or [])
    doc["calibration"] = rows
    return doc


# ---- stage 2: around the in-lease command ----------------------------------------------------------------------------------


def selection(entries) -> dict | None:
    """The run's last `SPEECH_ROUTE_SELECTED` decision (written by stage 0: the task, the interpreter, the script hashes), or None."""
    found = None
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and e.get("event") == "decision" and isinstance(e.get("data"), dict) and e["data"].get("decision") == SPEECH_ROUTE_SELECTED:
            found = e["data"]
    return found


def hw_config(ref, ref_dir, stage_dir, *, serve_timeout_s=1800, timeout_s=600) -> dict:
    """The config of orchard/speech_hw.py for a run whose stage 1 recorded `ref`. Clip paths point at stage 1's saved copies."""
    task, model = ref.get("task"), ref.get("model")
    bundle = bundle_for_task(task, model)
    if bundle is None:
        raise SpeechRouteError(f"for task {task}, no known bundle serves {model!r}")
    cfg = {"schema": SCHEMA, "task": task, "bundle": bundle.bundle, "stage_dir": str(Path(stage_dir).resolve()), "serve_timeout_s": serve_timeout_s,
           "timeout_s": timeout_s}
    if task == "speech-to-text":
        cfg["clips"] = [{"name": c["name"], "path": str((Path(ref_dir) / CLIP_DIR / c["name"]).resolve()), "text": c["text"]} for c in ref["clips"]]
    else:
        cfg["prompts"] = [{"id": p["id"], "text": p["text"]} for p in ref["prompts"]]
    return cfg


def hw_command(stage_dir) -> str:
    """The one command the hardware phase runs under the lease."""
    return shlex.join([sys.executable, str(HW_SCRIPT), "--config", str(Path(stage_dir).resolve() / CONFIG_FILE)])


def score_config(result, stage_dir) -> dict:
    """The producer config that scores the audio of a text-to-speech result with the scorer. Prompts without audio are left out."""
    files = []
    for p in result.get("prompts") or []:
        audio = p.get("audio") if isinstance(p, dict) else None
        if isinstance(audio, dict) and isinstance(audio.get("file"), str):
            files.append({"name": Path(audio["file"]).name, "path": str((Path(stage_dir) / audio["file"]).resolve())})
    if not files:
        raise SpeechRouteError("the result holds no audio to score")
    return producer_config(SCORER_MODEL, None, files, max_new_tokens=448)


def merge_scores(result, texts) -> dict:
    """A copy of `result` with each prompt's `heard` set from the scorer's {audio file name: transcript}. A prompt that has audio and no transcript is an error."""
    out = copy.deepcopy(result)
    for p in out.get("prompts") or []:
        audio = p.get("audio")
        if not isinstance(audio, dict):
            continue
        text = texts.get(Path(audio["file"]).name) if isinstance(texts, dict) else None
        if not isinstance(text, str):
            raise SpeechRouteError(f"the scorer gave no transcript for the audio of prompt {p.get('id')}")
        p["heard"] = text
    return out


def producer_env(environ) -> dict:
    """The environment for the reference interpreter: the keys in PRODUCER_ENV_KEYS that are set, and nothing else (no token, no home directory)."""
    return {k: environ[k] for k in PRODUCER_ENV_KEYS if k in environ}


def run_producer(argv, *, timeout_s, env) -> tuple[int | None, str]:
    """(exit code or None, the end of stderr). None means the program could not start or ran past its deadline, and the text says which."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout_s, env=env, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, f"the reference producer ran past its deadline of {timeout_s} s and was stopped"
    except OSError as exc:
        return None, f"the reference producer cannot be started: {exc}"
    return proc.returncode, (proc.stderr or "")[-STDERR_TAIL:]
