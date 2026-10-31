# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The audio-job boot check of the canning step (`[check] mode = "audio-job"`).

A model that makes audio clips behind a job server has no chat endpoint, so `orchard/can.py` hands a booted server to `audio_checks` and sends no
chat requests. The checks follow the job API of the Stable Audio 3 job server (`pkgsrc/server/stable_audio_jobs.py`):

    POST /v1/audio/jobs {prompt, duration, steps, seed}  -> 202 {id}
    GET  /v1/audio/jobs/<id>                             -> {status, stats: {audio_s, total_s, ...}, error}
    GET  /v1/audio/jobs/<id>/audio                       -> the clip (FLAC or WAV)

Two jobs with the same settings run one after the other. The first runs against a kernel cache that the boot check's copy has never filled, so
its time includes compiling; the second is warm. Both times go in the evidence, so a card can state them separately. The checks:

    audio_job          the job finished with status `done` before the timeout (the server's error text is in the detail when it did not)
    audio_file         the clip is a FLAC file (fLaC header) or a RIFF/WAVE file
    audio_length       the clip's length, read from its own header, is within 5 percent of the seconds asked for
    audio_rate         the second (warm) job's server time divided by its audio length is at most `max_rtf` (only when the spec gives `max_rtf`)
    audio_second_job   the second job also finished, and its clip is the same length

`rtf` here is generation time divided by audio length. A value below 1 means a clip takes less time to make than to play.
"""
from __future__ import annotations

import io
import struct
import time
import wave

from orchard import safeio

MAX_JSON_BYTES = 1024 * 1024        # choice: a job view is a few kilobytes
LENGTH_TOLERANCE = 0.05             # choice: the length in the file header must be within 5 percent of the seconds asked for
POLL_S = 0.5


def clip_seconds(data: bytes) -> float:
    """The length in seconds from the clip's own header: FLAC STREAMINFO or a WAV header. Raises ValueError for anything else."""
    if data[:4] == b"fLaC":
        if len(data) < 42 or data[4] & 0x7F != 0:
            raise ValueError("a FLAC file with no STREAMINFO block first")
        packed = struct.unpack(">Q", data[18:26])[0]
        rate, total = packed >> 44, packed & ((1 << 36) - 1)
        if rate == 0 or total == 0:
            raise ValueError("the FLAC header gives no sample rate or no length")
        return total / rate
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / w.getframerate()
    raise ValueError("not a FLAC or WAV file")


def _json_body(raw: bytes):
    """A server answer as a dict, or None when it is not one. The server is the bundle under test, so its answers are read with the guarded parser."""
    try:
        return safeio.parse_json_bytes(raw, max_bytes=MAX_JSON_BYTES, expect=dict)
    except safeio.SafeReadError:
        return None


def run_job(http, base: str, settings: dict, *, timeout: float) -> dict:
    """Submit one job and wait for it. `http(method, url, body, timeout=)` returns (status, bytes). Returns what happened, never raises on a bad
    answer: {ok, detail, job_s (wall time from the submit to the finished status), view, audio (bytes or None)}."""
    t0 = time.time()
    code, raw = http("POST", f"{base}/v1/audio/jobs", settings, timeout=30.0)
    body = _json_body(raw)
    if code != 202 or not body or not isinstance(body.get("id"), str):
        return {"ok": False, "detail": f"the submit answered {code}: {raw[:200]!r}", "job_s": None, "view": None, "audio": None}
    jid, view = body["id"], None
    while time.time() - t0 < timeout:
        code, raw = http("GET", f"{base}/v1/audio/jobs/{jid}", None, timeout=30.0)
        view = _json_body(raw)
        if code == 200 and view and view.get("status") in ("done", "error", "cancelled"):
            break
        time.sleep(POLL_S)
    job_s = round(time.time() - t0, 3)
    if not view or view.get("status") != "done":
        status = (view or {}).get("status", "no answer")
        return {"ok": False, "detail": f"the job ended as {status!r} after {job_s} s" + (f": {view.get('error')}" if view and view.get("error") else ""),
                "job_s": job_s, "view": view, "audio": None}
    code, audio = http("GET", f"{base}/v1/audio/jobs/{jid}/audio", None, timeout=60.0)
    if code != 200:
        return {"ok": False, "detail": f"the clip answered {code}", "job_s": job_s, "view": view, "audio": None}
    return {"ok": True, "detail": f"done in {job_s} s", "job_s": job_s, "view": view, "audio": audio}


def _summary(result: dict, asked: float) -> dict:
    view = result["view"] or {}
    stats = view.get("stats") or {}
    audio_s = stats.get("audio_s")
    total = stats.get("total_s")
    rtf = round(total / audio_s, 3) if isinstance(total, (int, float)) and isinstance(audio_s, (int, float)) and audio_s > 0 else None
    return {"settings": view.get("settings"), "audio_s": audio_s, "server_total_s": total, "wall_s": result["job_s"], "rtf": rtf,
            "stages": stats.get("stages"), "audio_format": stats.get("audio_format")}


def audio_checks(http, base: str, ck, evidence: dict, check, *, job_timeout: float) -> None:
    """Run the audio-job checks against a server that answered /health. `check(name, ok, detail)` records one. Fills evidence["audio"]."""
    settings = {"prompt": ck.prompt, "duration": ck.audio_seconds, "steps": ck.audio_steps, "seed": ck.audio_seed}
    evidence["audio"] = {}
    first = run_job(http, base, settings, timeout=job_timeout)
    if not check("audio_job", first["ok"], first["detail"]):
        evidence["audio"]["first"] = _summary(first, ck.audio_seconds)
        return
    evidence["audio"]["first"] = _summary(first, ck.audio_seconds)
    try:
        got = clip_seconds(first["audio"])
        kind_ok = True
    except ValueError as exc:
        got, kind_ok = None, False
        check("audio_file", False, str(exc))
    if kind_ok:
        check("audio_file", True, f"{len(first['audio'])} bytes")
        near = abs(got - ck.audio_seconds) <= LENGTH_TOLERANCE * ck.audio_seconds
        check("audio_length", near, f"{got:.3f} s in the file header; {ck.audio_seconds} s asked for")
        evidence["audio"]["first"]["file_s"] = round(got, 3)
    second = run_job(http, base, settings, timeout=job_timeout)
    evidence["audio"]["second"] = _summary(second, ck.audio_seconds)
    second_ok = second["ok"]
    if second_ok:
        try:
            second_s = clip_seconds(second["audio"])
            evidence["audio"]["second"]["file_s"] = round(second_s, 3)
            second_ok = abs(second_s - ck.audio_seconds) <= LENGTH_TOLERANCE * ck.audio_seconds
        except ValueError:
            second_ok = False
    check("audio_second_job", second_ok, second["detail"] if not second["ok"] else "the warm job finished with a clip of the length asked for")
    rtf = evidence["audio"]["second"]["rtf"]
    if ck.max_rtf is not None:
        check("audio_rate", isinstance(rtf, (int, float)) and rtf <= ck.max_rtf,
              f"{rtf} s of generation per second of audio on the warm job; at most {ck.max_rtf} wanted")
