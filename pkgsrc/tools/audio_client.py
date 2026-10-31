#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A command-line client for the Stable Audio 3 job server in this bundle (standard library only).

    python tools/audio_client.py --url http://127.0.0.1:20170 --prompt "A warm acoustic guitar melody" --duration 10 --out clip.flac
    python tools/audio_client.py --url http://127.0.0.1:20170 --prompt "..." --duration 30 --repeat 4 --out-dir clips --json timings.json

It submits a job, polls it until it ends, downloads the clip and prints one line per job: the wall time from the submit to the finished status, the
server's own time for the clip, the clip's length from its header, and the stage times the server reports. With `--repeat N` the same job runs N times
one after the other (the first of a fresh server compiles kernels, so it is slower than the rest). `--json FILE` writes every job's numbers, the host's
load average before and after each job (read from /proc/loadavg on the machine running this client) and the server's /health document.

The exit status is 0 when every job produced a clip, 1 when one did not (a 429 queue-full answer, an error job, a timeout).
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
import io
from pathlib import Path


def http(method, url, body=None, timeout=60.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def clip_seconds(data: bytes) -> float:
    """The clip's length from its own header (FLAC STREAMINFO or WAV)."""
    if data[:4] == b"fLaC":
        packed = struct.unpack(">Q", data[18:26])[0]
        return (packed & ((1 << 36) - 1)) / (packed >> 44)
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / w.getframerate()
    raise ValueError("not a FLAC or WAV file")


def loadavg():
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError):
        return None


def run_job(base, settings, timeout):
    """One job. Returns a dict with ok, wall_s, server_total_s, audio_s, rtf, stages, error, and the clip bytes under `audio` (None on failure)."""
    out = {"ok": False, "settings": settings, "host_load_before": loadavg()}
    t0 = time.time()
    code, raw = http("POST", f"{base}/v1/audio/jobs", settings)
    if code != 202:
        out["error"] = f"submit answered {code}: {raw[:200]!r}"
        return out
    jid = json.loads(raw)["id"]
    view = {}
    while time.time() - t0 < timeout:
        code, raw = http("GET", f"{base}/v1/audio/jobs/{jid}")
        view = json.loads(raw) if code == 200 else {}
        if view.get("status") in ("done", "error", "cancelled"):
            break
        time.sleep(0.25)
    out["wall_s"] = round(time.time() - t0, 3)
    out["host_load_after"] = loadavg()
    if view.get("status") != "done":
        out["error"] = f"the job ended as {view.get('status', 'no answer')!r}: {view.get('error', '')}"
        return out
    stats = view["stats"]
    code, audio = http("GET", f"{base}/v1/audio/jobs/{jid}/audio")
    if code != 200:
        out["error"] = f"the clip answered {code}"
        return out
    out.update(ok=True, audio=audio, id=jid, server_total_s=stats["total_s"], audio_s=stats["audio_s"], file_s=round(clip_seconds(audio), 3),
               rtf=round(stats["total_s"] / stats["audio_s"], 4), stages=stats.get("stages"), seed=view["settings"]["seed"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("AUDIO_URL", "http://127.0.0.1:20170"))
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1, help="run the same job this many times, one after the other")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", help="write the (last) clip here")
    ap.add_argument("--out-dir", help="write every clip here as clip-<n>.flac")
    ap.add_argument("--json", help="write every job's numbers and the server's /health here")
    args = ap.parse_args(argv)
    base = args.url.rstrip("/")
    settings = {"prompt": args.prompt, "duration": args.duration, "steps": args.steps, "cfg_scale": args.cfg_scale, "seed": args.seed}
    jobs, failed = [], False
    for n in range(1, args.repeat + 1):
        r = run_job(base, settings, args.timeout)
        audio = r.pop("audio", None)
        jobs.append(r)
        if not r["ok"]:
            failed = True
            print(f"job {n}: FAILED {r.get('error')}", file=sys.stderr)
            continue
        print(f"job {n}: wall {r['wall_s']} s, server {r['server_total_s']} s for {r['audio_s']} s of audio (rtf {r['rtf']}), clip header {r['file_s']} s, stages {r['stages']}")
        if args.out_dir:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.out_dir) / f"clip-{n}.flac").write_bytes(audio)
        if args.out and n == args.repeat:
            Path(args.out).write_bytes(audio)
    if args.json:
        code, raw = http("GET", f"{base}/v1/health")
        Path(args.json).write_text(json.dumps({"url_path": "/v1/audio/jobs", "settings": settings, "jobs": jobs,
                                               "health": json.loads(raw) if code == 200 else None}, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
