# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Asks a running server for a clip and checks it. Needs numpy; opens no device.

    python tools/smoke.py [--port 20120] [--duration 30] [--out clip.wav] [--prompt TEXT] [--lyrics TEXT] [--seed 42]

Checks, on the WAV that comes back: 48 kHz, two channels, a length within 5 percent of the request, at least 40 percent of 100 ms windows louder than
-40 dBFS, and at most 1 percent of samples at full scale. These are the sound checks of the tt-orchard audio stages. They measure length and level; they
do not say whether the music is good. Prints one JSON line and exits 0 when every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import struct

import numpy as np

PROMPT = "warm lo-fi hip hop, mellow rhodes piano, soft vinyl crackle, relaxed drums"


def read_wav(data: bytes):
    """(sample rate, channels, float64 array [channels, samples]) of a RIFF WAV with 16-bit integer or 32-bit float samples. ACE-Step writes 32-bit float."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        tag, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if tag == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif tag == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("the WAV has no fmt or data chunk")
    tag, ch, rate, _, _, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(pcm[:len(pcm) // 4 * 4], dtype="<f4").astype(np.float64)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2").astype(np.float64) / 32768.0
    else:
        raise ValueError(f"unsupported WAV format tag {tag} with {bits} bits")
    return rate, ch, x[:len(x) // ch * ch].reshape(-1, ch).T


def sound_checks(wav_bytes: bytes, requested_s: float) -> dict:
    try:
        rate, ch, x = read_wav(wav_bytes)
    except ValueError as exc:
        return {"ok": False, "reasons": [str(exc)]}
    if rate <= 0 or ch <= 0 or x.size == 0:
        return {"ok": False, "reasons": ["the WAV has no samples or a zero rate"]}
    if not np.isfinite(x).all():
        return {"ok": False, "reasons": ["the WAV has a sample that is not finite"]}
    win = max(1, int(rate * 0.1))
    k = x.shape[1] // win
    rms = np.sqrt((x[:, :k * win].reshape(ch, k, win) ** 2).mean(axis=(0, 2))) if k else np.zeros(0)
    loud = float((20 * np.log10(np.maximum(rms, 1e-12)) > -40.0).mean()) if k else 0.0
    full = float((np.abs(x) >= 0.9999).mean())
    duration = x.shape[1] / rate
    reasons = []
    if rate != 48000:
        reasons.append(f"sample rate {rate}, expected 48000")
    if ch != 2:
        reasons.append(f"{ch} channels, expected 2")
    if abs(duration - requested_s) > 0.05 * requested_s:
        reasons.append(f"duration {duration:.2f} s is more than 5% from the requested {requested_s} s")
    if loud < 0.40:
        reasons.append(f"only {loud:.1%} of 100 ms windows are above -40 dBFS")
    if full > 0.01:
        reasons.append(f"{full:.2%} of samples are at full scale")
    return {"ok": not reasons, "reasons": reasons, "duration_s": round(duration, 3), "sample_rate": rate, "channels": ch,
            "loud_window_fraction": round(loud, 4), "full_scale_fraction": round(full, 6)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=20120)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--lyrics", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="clip.wav")
    a = p.parse_args(argv)
    body = json.dumps({"prompt": a.prompt, "lyrics": a.lyrics, "duration": a.duration, "seed": a.seed}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/music", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data, timings = r.read(), json.loads(r.headers.get("X-Timings", "{}"))
    except urllib.error.HTTPError as exc:
        print(json.dumps({"ok": False, "http_status": exc.code, "body": exc.read(2000).decode("utf-8", "replace")}))
        return 1
    open(a.out, "wb").write(data)
    checks = sound_checks(data, a.duration)
    print(json.dumps({**checks, "out": a.out, "timings": timings}))
    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
