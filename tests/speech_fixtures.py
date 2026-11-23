# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Plain-dict builders for the speech route's records (orchard/speech_route.py). They write the dictionaries out by hand, so a test that uses them
checks the module against the documented layout and not against the module's own builder."""
import copy
import hashlib
import json
from pathlib import Path

from orchard import speech_check
from test_speech_check import wav_bytes

ASR_MODEL = "mistralai/Voxtral-Mini-3B-2507"
TTS_MODEL = "mistralai/Voxtral-4B-TTS-2603"
ASR_BUNDLE = "apande-TT/mistralai-voxtral-mini-3b-2507-tt"
TTS_BUNDLE = "apande-TT/mistralai-voxtral-4b-tts-2603-tt"
ENGINE = {"python": "ref-python", "transformers": "5.24.0", "torch": "2.15.0+cpu", "model_id": "mistralai/Voxtral-Mini-3B-2507", "decoding": "greedy"}
SCORER = {"python": "ref-python", "transformers": "5.24.0", "torch": "2.15.0+cpu", "model_id": "openai/whisper-large-v3-turbo", "decoding": "greedy"}
ASR_LICENSE = {"id": "apache-2.0", "non_commercial": False,
               "line": "Voxtral-Mini-3B-2507 is used under its licence, apache-2.0, as the bundle's card states it. This run served the bundle apande-TT/mistralai-voxtral-mini-3b-2507-tt and did not copy or republish any weights."}
TTS_LICENSE = {"id": "cc-by-nc-4.0", "non_commercial": True,
               "line": "Voxtral-4B-TTS-2603 is used under its licence, cc-by-nc-4.0 (non-commercial use only), as the bundle's card states it. This run served the bundle apande-TT/mistralai-voxtral-4b-tts-2603-tt and did not copy or republish any weights."}
PROMPTS = [("p0", "The lighthouse keeper wrote in his journal every night."), ("p1", "She tuned the old radio slowly until the music came through.")]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_json(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return path


def clip_entry(stage_dir: Path, name: str, expected, text, seconds=1.0) -> dict:
    data = wav_bytes(seconds, rate=16000)
    d = Path(stage_dir) / "clips"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    return {"name": name, "sha256": sha(data), "bytes": len(data), "expected": expected, "text": text}


def asr_reference(stage_dir, clips=(("mlk.wav", "I have a dream that one day", "I have a dream that one day"),)) -> dict:
    doc = {"schema": 1, "task": "speech-to-text", "model": ASR_MODEL, "reference": copy.deepcopy(ENGINE), "license": copy.deepcopy(ASR_LICENSE),
           "clips": [clip_entry(stage_dir, n, e, t) for n, e, t in clips]}
    put_json(Path(stage_dir) / "reference.json", doc)
    return doc


def tts_reference(stage_dir, prompts=PROMPTS, calibration=(("mlk.wav", "I have a dream that one day", "I have a dream that one day", 0.0),)) -> dict:
    doc = {"schema": 1, "task": "text-to-speech", "model": TTS_MODEL, "reference": copy.deepcopy(SCORER), "license": copy.deepcopy(TTS_LICENSE),
           "prompts": [{"id": i, "text": t, "voice": None} for i, t in prompts],
           "calibration": [dict(clip_entry(stage_dir, n, e, t), wer=w) for n, e, t, w in calibration]}
    put_json(Path(stage_dir) / "reference.json", doc)
    return doc


def ok(name, **data):
    return {"name": name, "status": "pass", "detail": "fine", "data": data}


def asr_result(ref, texts=None, **over) -> dict:
    texts = texts or {}
    checks = [ok("health", status=200), ok("models", status=200, ids=["m"])]
    for c in ref["clips"]:
        heard = texts.get(c["name"], c["text"])
        errors, n = speech_check.wer(c["text"], heard)
        checks.append(ok(f"transcribe {c['name']}", text=heard, wer=errors / n, errors=errors, reference_words=n, max_wer=0.1, seconds=1.0))
    checks.append(ok("bad request", status=400))
    doc = {"schema": 1, "task": "speech-to-text", "bundle": ASR_BUNDLE, "url": "http://127.0.0.1:20000", "served": {"serve_exit": 0, "stop_exit": 0},
           "checks": checks}
    doc.update(over)
    return doc


def tts_result(stage_dir, ref, heard=None, seconds=3.0, **over) -> dict:
    heard = heard or {}
    prompts = []
    for p in ref["prompts"]:
        data = wav_bytes(seconds)
        rel = f"audio/{p['id']}.wav"
        (Path(stage_dir) / "audio").mkdir(parents=True, exist_ok=True)
        (Path(stage_dir) / rel).write_bytes(data)
        checks = [ok("health", status=200), ok("models", status=200, ids=["m"]),
                  ok("speech", sample_rate=24000, channels=1, sample_width=2, seconds=seconds, peak=8000, seconds_per_character=seconds / len(p["text"])),
                  ok("bad voice", status=400), ok("missing input", status=400)]
        prompts.append({"id": p["id"], "checks": checks, "audio": {"file": rel, "sha256": sha(data)}, "heard": heard.get(p["id"], p["text"])})
    doc = {"schema": 1, "task": "text-to-speech", "bundle": TTS_BUNDLE, "url": "http://127.0.0.1:20000", "served": {"serve_exit": 0, "stop_exit": 0},
           "prompts": prompts}
    doc.update(over)
    return doc
