# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A local job server for Stable Audio 3 small music on a Tenstorrent chip (standard library only).

    python audio_server.py --generator chip --ui          # the model on the leased chip, then the Gradio page
    python audio_server.py --generator fake --ui          # no chip: a fake generator that makes a plain tone, for trying the page

The server makes one clip at a time and keeps a short queue. When the queue is full a new job gets HTTP 429, and the client and the page say so.
The audio comes from a `Generator`; the server knows nothing about the model. Two generators ship here:

  * `FakeGenerator` makes a tone of the length asked for. It exists for tests and for trying the page without a chip. It says it is fake in `/v1/health`.
  * `SubprocessChipGenerator` runs the harness's `chip_generate.py` once per clip under the lease the server's own process holds. The model is loaded
    again for each clip (the clip's own timings say how long that took), because `chip_generate.py` loads and runs in one call. A generator that keeps the
    model resident only has to implement `load` and `generate` below.

The API (JSON bodies; the audio is the response body of its own endpoint):

    GET    /v1/health                    status ok | loading | error, the model, the device, the queue and the limits the server accepts
    POST   /v1/audio/jobs                {prompt, duration?, steps?, cfg_scale?, seed?} -> 202 {id}; 400 bad input; 429 queue full; 503 not ready
    GET    /v1/audio/jobs/<id>           status queued | running | done | error | cancelled, queue_position, stage, settings, stats, error
    GET    /v1/audio/jobs/<id>/audio     the clip (409 until the job is done)
    DELETE /v1/audio/jobs/<id>           a queued job is dropped; a running one cannot be stopped (the chip keeps running) and its clip is discarded

The server binds 127.0.0.1 unless told otherwise. It signs nobody in: bind 0.0.0.0 only on a network you trust.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import random
import re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

MIN_SECONDS = 1
MAX_SECONDS = 120            # the model's own limit: its sample_size is 5,292,032 samples at 44,100 Hz. Only 10 s clips have been checked on the chip.
MAX_STEPS = 100
MIN_CFG, MAX_CFG = 1.0, 20.0
MAX_PROMPT_CHARS = 2000
MAX_SEED = 2**32 - 1
MAX_BODY_BYTES = 64 * 1024
DEFAULT_SECONDS = 10
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_PORT = 20100
DEFAULT_UI_PORT = 7872
SETTING_KEYS = {"prompt", "duration", "steps", "cfg_scale", "seed"}


class QueueFull(Exception):
    """Every place for a job (the running one and the waiting ones) is taken."""


class NotReady(Exception):
    """The model is still loading, or failed to load."""


@dataclass(frozen=True)
class JobRequest:
    prompt: str
    seconds: float
    steps: int
    cfg_scale: float
    seed: int

    def settings(self) -> dict:
        return {"prompt": self.prompt, "duration": self.seconds, "steps": self.steps, "cfg_scale": self.cfg_scale, "seed": self.seed}


@dataclass
class Clip:
    """What a generator returns: the audio file's bytes, its format ("flac" or "wav"), its length measured from the file, and named stage times."""
    audio: bytes
    audio_format: str
    audio_s: float
    stages: dict = field(default_factory=dict)


class Generator:
    """The interface the server calls. `load` runs once, before the server accepts jobs; `generate` runs for one job at a time, never twice at once.
    `on_stage(name)` tells the job's status what the generator is doing. Both may raise: the message reaches the job's `error` (or `/v1/health`)."""
    model_name = "unnamed model"
    device_name = "unknown device"
    min_seconds = MIN_SECONDS
    max_seconds = MAX_SECONDS
    max_steps = MAX_STEPS
    default_steps = 8             # the settings the model's authors use for small music: 8 steps, guidance scale 1.0
    default_cfg_scale = 1.0

    def load(self) -> None:
        pass

    def generate(self, req: JobRequest, on_stage: Callable[[str], None]) -> Clip:
        raise NotImplementedError


# ---- audio file helpers ------------------------------------------------------------------------------------

def flac_seconds(data: bytes) -> float:
    """The length in seconds from a FLAC file's STREAMINFO block (sample rate in 20 bits, total samples in 36 bits)."""
    if data[:4] != b"fLaC" or len(data) < 42 or data[4] & 0x7F != 0:
        raise ValueError("not a FLAC file with a STREAMINFO block first")
    packed = struct.unpack(">Q", data[18:26])[0]           # sample rate (20), channels-1 (3), bits-1 (5), total samples (36)
    rate, total = packed >> 44, packed & ((1 << 36) - 1)
    if rate == 0 or total == 0:
        raise ValueError("the FLAC header gives no sample rate or no length")
    return total / rate


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


# ---- generators ---------------------------------------------------------------------------------------------

class FakeGenerator(Generator):
    """No model and no chip: a quiet two-tone arpeggio of the length asked for, seeded so the same request gives the same bytes."""
    model_name = "fake generator (no model, plain tones)"
    device_name = "no chip"

    def __init__(self, delay_s: float = 2.0):
        self.delay_s = delay_s

    def generate(self, req: JobRequest, on_stage: Callable[[str], None]) -> Clip:
        on_stage("sampling")
        time.sleep(self.delay_s)
        on_stage("decoding")
        rate = 22050
        rng = random.Random(req.seed)
        base = rng.choice([220.0, 246.9, 261.6, 293.7])
        notes = [base * r for r in (1.0, 1.25, 1.5, 2.0)]
        frames = int(rate * req.seconds)
        out = bytearray()
        for i in range(frames):
            t = i / rate
            f = notes[int(t * 2) % len(notes)]
            env = 0.5 * (1 - (t * 2) % 1.0)
            out += struct.pack("<h", int(9000 * env * math.sin(2 * math.pi * f * t)))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(out))
        data = buf.getvalue()
        return Clip(data, "wav", wav_seconds(data), {"fake_tone_s": self.delay_s})


class SubprocessChipGenerator(Generator):
    """Runs the harness's `chip_generate.py` for each clip, in a child process, and reads the clip and its timings from the run directory.

    The caller holds the lease (run the server under `gozer run --chips 1 --`); the child inherits the environment, so it sees the leased chip.
    Each clip loads the model again (about the load time `chip_generate.py` records as `model_load_s`). That cost is part of every clip made this way.
    """
    model_name = "stable-audio-3 small-music, diffusion transformer on one Blackhole chip"
    device_name = "one Blackhole chip"

    def __init__(self, chip_python: str, script: str, src_dir: str, model: str = "small-music", work_dir: Optional[str] = None,
                 timeout_s: float = 900.0, max_seconds: int = MAX_SECONDS):
        self.chip_python, self.script, self.src_dir, self.model = chip_python, script, src_dir, model
        self.work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="audio-demo-"))
        self.timeout_s = timeout_s
        self.max_seconds = max_seconds

    def load(self) -> None:
        """Check what the child needs. The model itself is loaded by the child for each clip."""
        missing = [str(p) for p in (self.chip_python, self.script) if not os.path.exists(p)]
        if not (Path(self.src_dir) / "stable_audio_3").is_dir():
            missing.append(f"{self.src_dir}/stable_audio_3")
        if missing:
            raise RuntimeError("these files are missing: " + ", ".join(missing))
        self.work.mkdir(parents=True, exist_ok=True)

    def command(self, req: JobRequest, run_dir: Path) -> list[str]:
        # --no-layers: the per-layer read backs are for the harness's agreement check and cost time
        return [self.chip_python, self.script, "--model", self.model, "--out", str(run_dir), "--prompt", req.prompt,
                "--duration", repr(float(req.seconds)), "--steps", str(req.steps), "--cfg-scale", repr(float(req.cfg_scale)),
                "--seed", str(req.seed), "--no-layers"]

    def generate(self, req: JobRequest, on_stage: Callable[[str], None]) -> Clip:
        run_dir = self.work / f"run-{uuid.uuid4().hex[:12]}"
        env = dict(os.environ)
        env["PYTHONPATH"] = self.src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        on_stage("loading the model and sampling on the chip")
        log = run_dir.with_suffix(".log")
        self.work.mkdir(parents=True, exist_ok=True)
        with open(log, "wb") as out:
            proc = subprocess.Popen(self.command(req, run_dir), stdout=out, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
            try:
                rc = proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError(f"the chip run did not finish in {self.timeout_s:.0f} s and was stopped; the chip may need a reset") from None
        if rc != 0:
            tail = log.read_text(errors="replace").strip().splitlines()[-3:]
            raise RuntimeError(f"the chip run exited with code {rc}: " + " | ".join(tail))
        clip_path = run_dir / "clips" / "audio.flac"
        if not clip_path.is_file():
            raise RuntimeError("the chip run finished without writing a clip")
        data = clip_path.read_bytes()
        stages = {}
        try:
            timings = json.loads((run_dir / "chip.json").read_text()).get("timings", {})
            stages = {k: float(v) for k, v in timings.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        except (OSError, ValueError):
            pass                                  # the clip is good without its timings
        return Clip(data, "flac", flac_seconds(data), stages)


# ---- the job queue --------------------------------------------------------------------------------------------

class Job:
    def __init__(self, req: JobRequest):
        self.id = uuid.uuid4().hex
        self.req = req
        self.status = "queued"
        self.stage = ""
        self.error = ""
        self.cancel_requested = False
        self.clip: Optional[Clip] = None
        self.total_s: Optional[float] = None
        self.started: Optional[float] = None


class JobServer:
    """Holds the generator, the queue and the jobs. One worker thread makes the clips, one at a time.

    A job fits while fewer than `max_waiting` jobs are queued: in all, one running job and up to `max_waiting` waiting ones. Finished jobs are kept for `keep` entries, newest last.
    """
    def __init__(self, generator: Generator, max_waiting: int = 2, keep: int = 16):
        self.gen = generator
        self.max_waiting = max_waiting
        self.keep = keep
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.jobs: dict[str, Job] = {}
        self.queue: collections.deque[Job] = collections.deque()
        self.finished: collections.deque[str] = collections.deque()
        self.running: Optional[Job] = None
        self.state = "loading"
        self.load_error = ""
        self.stopping = False
        self.threads: list[threading.Thread] = []

    # -- life cycle
    def start(self) -> None:
        t = threading.Thread(target=self._run, name="audio-worker", daemon=True)
        self.threads.append(t)
        t.start()

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
            self.wake.notify_all()

    def _run(self) -> None:
        try:
            self.gen.load()
        except Exception as e:                      # a load failure is reported through /v1/health
            with self.lock:
                self.state, self.load_error = "error", f"{type(e).__name__}: {e}"
            return
        with self.lock:
            self.state = "ok"
        while True:
            with self.lock:
                while not self.queue and not self.stopping:
                    self.wake.wait(0.5)
                if self.stopping:
                    return
                job = self.queue.popleft()
                job.status, job.started = "running", time.monotonic()
                self.running = job
            self._make(job)

    def _make(self, job: Job) -> None:
        def on_stage(name: str) -> None:
            job.stage = name
        clip, err = None, ""
        try:
            clip = self.gen.generate(job.req, on_stage)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        with self.lock:
            job.total_s = time.monotonic() - (job.started or time.monotonic())
            if job.cancel_requested:
                job.status, job.clip = "cancelled", None
            elif err:
                job.status, job.error = "error", err
            else:
                job.status, job.clip = "done", clip
            self.running = None
            self._remember(job)

    def _remember(self, job: Job) -> None:
        """Keep the newest `keep` finished jobs; the rest (and their audio) are dropped. Called with the lock held."""
        self.finished.append(job.id)
        while len(self.finished) > self.keep:
            self.jobs.pop(self.finished.popleft(), None)

    # -- what the HTTP layer calls
    def validate(self, body: object) -> JobRequest:
        """Turn a request body into a JobRequest or raise ValueError with a plain message. Exact types: a bool or a string is not a number."""
        if not isinstance(body, dict):
            raise ValueError("the body must be a JSON object")
        extra = set(body) - SETTING_KEYS
        if extra:
            raise ValueError(f"unknown field(s): {', '.join(sorted(extra))}")
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt must be at most {MAX_PROMPT_CHARS} characters")

        def number(key, default):
            v = body.get(key, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{key} must be a number")
            return v

        def whole(key, default):
            v = body.get(key, default)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{key} must be a whole number")
            return v

        seconds = number("duration", DEFAULT_SECONDS)
        if not self.gen.min_seconds <= seconds <= self.gen.max_seconds:
            raise ValueError(f"duration must be between {self.gen.min_seconds} and {self.gen.max_seconds} seconds")
        steps = whole("steps", self.gen.default_steps)
        if not 1 <= steps <= self.gen.max_steps:
            raise ValueError(f"steps must be a whole number from 1 to {self.gen.max_steps}")
        cfg = number("cfg_scale", self.gen.default_cfg_scale)
        if not MIN_CFG <= cfg <= MAX_CFG:
            raise ValueError(f"cfg_scale must be between {MIN_CFG} and {MAX_CFG}")
        seed = whole("seed", random.randint(0, 2**31 - 1))        # a seed that was not asked for is chosen here and comes back in the settings
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be a whole number from 0 to {MAX_SEED}")
        return JobRequest(prompt, seconds, steps, float(cfg), seed)

    def submit(self, req: JobRequest) -> Job:
        with self.lock:
            if self.state == "loading":
                raise NotReady("the model is still loading; try again in a minute")
            if self.state == "error":
                raise NotReady(f"the model did not load: {self.load_error}")
            if len(self.queue) >= self.max_waiting:
                raise QueueFull("the queue is full: a clip is being made and others are waiting")
            job = Job(req)
            self.jobs[job.id] = job
            self.queue.append(job)
            self.wake.notify()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[dict]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.status == "queued":
                self.queue.remove(job)
                job.status = "cancelled"
                self._remember(job)
            elif job.status == "running":
                job.cancel_requested = True
                return {"id": job.id, "status": "running", "cancel_requested": True,
                        "detail": "a running clip cannot be stopped: the chip keeps running until it ends, and the clip is then discarded"}
            return {"id": job.id, "status": job.status, "cancel_requested": job.cancel_requested}

    def job_view(self, job: Job) -> dict:
        with self.lock:
            pos = (list(self.queue).index(job) + 1) if job.status == "queued" else 0
            view = {"id": job.id, "status": job.status, "queue_position": pos, "stage": job.stage, "settings": job.req.settings(),
                    "error": job.error, "cancel_requested": job.cancel_requested, "stats": None}
            if job.status == "done" and job.clip is not None:
                view["stats"] = {"audio_s": round(job.clip.audio_s, 3), "total_s": round(job.total_s or 0.0, 3),
                                 "audio_format": job.clip.audio_format, "stages": job.clip.stages}
            return view

    def health(self) -> dict:
        with self.lock:
            h = {"status": self.state, "model": self.gen.model_name, "device_name": self.gen.device_name,
                 "queue_depth": len(self.queue), "queue_max": self.max_waiting, "busy": self.running is not None,
                 "limits": {"min_seconds": self.gen.min_seconds, "max_seconds": self.gen.max_seconds, "max_steps": self.gen.max_steps,
                            "min_cfg_scale": MIN_CFG, "max_cfg_scale": MAX_CFG},
                 "defaults": {"duration": DEFAULT_SECONDS, "steps": self.gen.default_steps, "cfg_scale": self.gen.default_cfg_scale}}
            if self.state == "error":
                h["error"] = self.load_error
            return h


# ---- HTTP ----------------------------------------------------------------------------------------------------

def _strict_json(raw: bytes):
    def refuse(name):
        raise ValueError(f"{name} is not valid JSON")
    return json.loads(raw.decode("utf-8"), parse_constant=refuse)       # NaN and Infinity are refused


def make_http_server(js: JobServer, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):         # the default writes every request to stderr
            pass

        def _send(self, code: int, payload, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code: int, detail: str) -> None:
            self._send(code, {"detail": detail})

        def do_GET(self):
            if self.path == "/v1/health":
                return self._send(200, js.health())
            m = re.fullmatch(r"/v1/audio/jobs/([0-9a-f]{32})(/audio)?", self.path)
            job = js.get(m.group(1)) if m else None
            if job is None:
                return self._err(404, "no such job")
            if not m.group(2):
                return self._send(200, js.job_view(job))
            clip = job.clip if job.status == "done" else None
            if clip is None:
                return self._err(409, f"the job is {job.status}; there is no clip")
            self._send(200, clip.audio, "audio/flac" if clip.audio_format == "flac" else "audio/wav")

        def do_POST(self):
            if self.path != "/v1/audio/jobs":
                return self._err(404, "no such endpoint")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                return self._err(400, "bad Content-Length")
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                return self._err(413, f"the body is larger than {MAX_BODY_BYTES} bytes")
            try:
                req = js.validate(_strict_json(self.rfile.read(length)))
            except (ValueError, UnicodeDecodeError) as e:
                return self._err(400, str(e))
            try:
                job = js.submit(req)
            except QueueFull as e:
                return self._err(429, str(e))
            except NotReady as e:
                return self._err(503, str(e))
            self._send(202, {"id": job.id, "status": "queued"})

        def do_DELETE(self):
            m = re.fullmatch(r"/v1/audio/jobs/([0-9a-f]{32})", self.path)
            result = js.cancel(m.group(1)) if m else None
            if result is None:
                return self._err(404, "no such job")
            self._send(200, result)

    ThreadingHTTPServer.daemon_threads = True
    return ThreadingHTTPServer((host, port), Handler)


# ---- command line --------------------------------------------------------------------------------------------

def default_script(here: Path) -> Path:
    """chip_generate.py beside this file when a package carries it, else the harness's copy (this file is orchard/audio_demo/audio_server.py in the repo)."""
    beside = here / "chip_generate.py"
    return beside if beside.exists() else here.parent / "skills" / "audio-chip-templates" / "chip_generate.py"


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", choices=("chip", "fake"), default="chip")
    ap.add_argument("--host", default=env("AUDIO_SERVER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("AUDIO_SERVER_PORT", str(DEFAULT_PORT))))
    ap.add_argument("--ui", action="store_true", help="run the Gradio page in this process after the model is ready")
    ap.add_argument("--ui-host", default=env("AUDIO_UI_HOST", "127.0.0.1"))
    ap.add_argument("--ui-port", type=int, default=int(env("AUDIO_UI_PORT", str(DEFAULT_UI_PORT))))
    ap.add_argument("--max-waiting", type=int, default=2, help="jobs that may wait while one is running")
    ap.add_argument("--max-seconds", type=int, default=MAX_SECONDS, help="longest clip the chip generator accepts (the model's limit is 120)")
    ap.add_argument("--model", default=env("AUDIO_MODEL", "small-music"))
    ap.add_argument("--chip-python", default=env("ORCHARD_TTNN_PYTHON", sys.executable), help="the interpreter that has ttnn")
    ap.add_argument("--src-dir", default=env("ORCHARD_AUDIO_SRC", ""), help="the stable-audio-3 library checkout")
    ap.add_argument("--script", default=env("AUDIO_CHIP_GENERATE", str(default_script(here))), help="the harness's chip_generate.py")
    ap.add_argument("--work-dir", default=env("AUDIO_WORK_DIR"), help="where each clip's run directory goes (default: a new temporary directory)")
    ap.add_argument("--fake-delay", type=float, default=2.0, help="seconds the fake generator pretends to work")
    return ap.parse_args(argv)


def build_generator(args) -> Generator:
    if args.generator == "fake":
        return FakeGenerator(delay_s=args.fake_delay)
    return SubprocessChipGenerator(args.chip_python, args.script, args.src_dir, model=args.model, work_dir=args.work_dir,
                                   max_seconds=args.max_seconds)


def main(argv=None) -> int:
    args = parse_args(argv)
    js = JobServer(build_generator(args), max_waiting=args.max_waiting)
    http = make_http_server(js, args.host, args.port)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    js.start()
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    print(f"audio server on http://{args.host}:{http.server_address[1]} ({js.gen.model_name}); loading", flush=True)
    try:
        while js.health()["status"] == "loading":
            time.sleep(0.2)
        h = js.health()
        if h["status"] == "error":
            print(f"audio_server: the model did not load: {h['error']}", file=sys.stderr)
            return 2
        print("model ready", flush=True)
        if args.ui:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import audio_app
            audio_app.launch(args.ui_host, args.ui_port, f"http://{args.host}:{http.server_address[1]}")
        else:
            while True:
                time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        js.stop()
        http.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
