# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A local HTTP server that keeps ACE-Step 1.5 turbo loaded on one Blackhole chip and makes a clip per request.

    python acestep_serve.py --bundle-code CODE_DIR --snapshot SNAPSHOT_DIR --root WORK_DIR [--host 127.0.0.1] [--port 20120]

    GET  /health      {"status": "ok"}
    POST /v1/music    {"prompt": str, "lyrics": str (optional), "duration": seconds, "seed": int (optional)}
                      -> 200 audio/wav (48 kHz stereo, peak-normalised by ACE-Step), headers X-Timings (JSON) and X-Sample-Rate.

One request runs at a time; a second gets 503 with Retry-After. The server binds the address it is given (127.0.0.1 by default), has no sign-in and
no TLS. The planner language model is not loaded (no thinking mode). The first request at a new duration compiles kernels and takes several
seconds longer than later ones.

On start it prints one JSON line with `"event": "ready"` after the chip is open and the models are loaded. Standard library only apart from the model
code: the request handling does not import torch, so `make_server` can be tested with a stand-in session.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 20120
SAMPLE_RATE = 48000
MIN_SECONDS, MAX_SECONDS = 5.0, 480.0        # the longest clip measured on the chip is 480 s
MAX_PROMPT, MAX_LYRICS = 2000, 8000
MAX_BODY = 64 * 1024
CLIENT_TIMEOUT_S = 30


def validate(body) -> dict:
    """The generate parameters of a request body, or a ValueError whose text is sent back as the 400 message."""
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    unknown = sorted(set(body) - {"prompt", "lyrics", "duration", "seed"})
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT:
        raise ValueError(f"prompt must be a non-empty string of at most {MAX_PROMPT} characters")
    lyrics = body.get("lyrics", "")
    if not isinstance(lyrics, str) or len(lyrics) > MAX_LYRICS:
        raise ValueError(f"lyrics must be a string of at most {MAX_LYRICS} characters")
    duration = body.get("duration", 30)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not (MIN_SECONDS <= duration <= MAX_SECONDS):
        raise ValueError(f"duration must be a number from {MIN_SECONDS:g} to {MAX_SECONDS:g} seconds")
    seed = body.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed < 2 ** 31):
        raise ValueError("seed must be an integer from 0 to 2147483647")
    return {"prompt": prompt, "lyrics": lyrics, "duration": float(duration), "seed": seed}


def make_server(session, host: str, port: int, work_dir) -> ThreadingHTTPServer:
    """An HTTP server around `session.one(save_dir, duration, prompt, lyrics, seed) -> (wav_path, timings)`."""
    busy = threading.Lock()
    work_dir = Path(work_dir)

    class Handler(BaseHTTPRequestHandler):
        timeout = CLIENT_TIMEOUT_S
        server_version = "acestep-serve"

        def log_message(self, fmt, *args):           # one line per request on stderr, no client body
            import sys
            print(json.dumps({"event": "request", "line": fmt % args}), file=sys.stderr, flush=True)

        def _send(self, code, payload: bytes, ctype="application/json", headers=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, code, message, headers=()):
            self._send(code, json.dumps({"error": message}).encode(), headers=headers)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b'{"status": "ok"}')
            else:
                self._error(404, "not found")

        def do_POST(self):
            if self.path != "/v1/music":
                return self._error(404, "not found")
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._error(400, "Content-Length is required")
            if n < 0 or n > MAX_BODY:
                return self._error(400, f"the body must be at most {MAX_BODY} bytes")
            try:
                params = validate(json.loads(self.rfile.read(n)))
            except (ValueError, UnicodeDecodeError) as exc:
                return self._error(400, str(exc))
            if not busy.acquire(blocking=False):
                return self._error(503, "a clip is being made; try again", headers=[("Retry-After", "5")])
            tmp = None
            try:
                tmp = Path(tempfile.mkdtemp(prefix="clip-", dir=work_dir))
                t0 = time.perf_counter()
                try:
                    wav, timings = session.one(tmp, params["duration"], params["prompt"], params["lyrics"], params["seed"])
                    payload = Path(wav).read_bytes()
                except Exception as exc:                  # a failed generate is a 500 with its type, never the model's text
                    return self._error(500, f"generation failed ({type(exc).__name__})")
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)    # the clip is in memory; its directory is gone before the reply is sent
                timings = {**timings, "request_s": round(time.perf_counter() - t0, 4)}
                self._send(200, payload, "audio/wav", [("X-Timings", json.dumps(timings)), ("X-Sample-Rate", str(SAMPLE_RATE))])
            finally:
                busy.release()

    class Server(ThreadingHTTPServer):
        daemon_threads = False         # server_close() waits for a request in flight, so the chip worker is not closed under it

    return Server((host, port), Handler)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bundle-code", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--root", required=True, help="a writable directory for ACE-Step's project root and the clips in flight")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--fast", default="all", help="the request-level savings and chip steps of acestep_fast.py: all (default), none (the bundle path of version 0.1.0) or a comma-separated list")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    import acestep_fast                                  # names of the steps; checked before the model loads
    steps = acestep_fast.parse_steps(a.fast)
    import acestep_run                                   # the model session; imports torch
    t0 = time.perf_counter()
    session = acestep_run.Session(device="chip", bundle_code=a.bundle_code, snapshot=a.snapshot, root=a.root, fast=steps)
    work = Path(a.root) / "clips-in-flight"
    work.mkdir(parents=True, exist_ok=True)
    srv = make_server(session, a.host, a.port, work)
    print(json.dumps({"event": "ready", "model": "ACE-Step/Ace-Step1.5", "revision": session.WEIGHTS_REVISION, "host": a.host, "port": srv.server_address[1], "fast": list(session.fast),
                      "load_s": round(time.perf_counter() - t0, 2)}), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
