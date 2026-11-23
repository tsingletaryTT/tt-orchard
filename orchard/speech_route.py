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

import hashlib
import json
from pathlib import Path

from orchard import safeio, speech_check, speech_models

ROUTE_INPUT = "speech_route"          # the run_start input that selects the route; its value is a task in speech_models.TASKS
CLIPS_INPUT = "speech_clips"          # the operator's directory of clips: NAME.wav|flac|mp3|ogg with NAME.txt holding the expected text
PROMPTS_INPUT = "speech_prompts"      # a text file with one prompt per line (text to speech); DEFAULT_PROMPTS when absent
SCHEMA = 1
REFERENCE_FILE = "reference.json"     # in stage 1's directory
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

REFERENCE_DEADLINE_S = 3600.0         # stage 1's CPU run (a 3B or 4B model and a Whisper scorer on the host's cores)
CHIP_DEADLINE_S = 3600.0              # stage 2's lease: the bundle's first start, the checks and the stop
SCORE_DEADLINE_S = 1800.0
STAGE_DISK_GB = 40


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


def selection_from_run_dir(run_dir) -> str | None:
    """`selected` for the ledger file of a run directory (a gate is given only directories). None when the ledger cannot be read."""
    entries = []
    try:
        with open(Path(run_dir) / "ledger.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return None
    return selected(entries)


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
        out.append(f"schema is {doc.get('schema')!r}; {SCHEMA} is the one this version reads")
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
        out.append(f"schema is {result.get('schema')!r}; {SCHEMA} is the one this version reads")
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
