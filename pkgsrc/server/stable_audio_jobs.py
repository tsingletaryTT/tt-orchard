# A copy of serve/serve_audio.py from the tt-stable-audio-3-small-music repository (commit 59724fd, sha256 9cf9bd06...), the job server of the Stable Audio 3
# small-music package. It is kept unchanged so that stable_audio_asgi.py and the can's tests run from this repository alone. That repository's file is the source.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""A local job server for Stable Audio 3 small music on a Tenstorrent chip (standard library only).

    python serve/serve_audio.py --generator chip --ui          # the model on the leased chip, then the Gradio page
    python serve/serve_audio.py --generator fake --ui          # no chip: a fake generator that makes a plain tone, for trying the page

The server makes one clip at a time and keeps a short queue. When the queue is full a new job gets HTTP 429, and the client and the page say so.
The audio comes from a `Generator`; the server knows nothing about the model. Two generators ship here:

  * `FakeGenerator` makes a tone of the length asked for. It exists for tests and for trying the page without a chip. It says it is fake in `/v1/health`.
  * `SubprocessChipGenerator` runs the chip runtime script `chip_generate.py` (from `--script` or AUDIO_CHIP_GENERATE) once per clip under the lease the server's own process holds. The model is loaded
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
import shutil
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
    @property
    def default_seconds(self) -> float:
        """The length used when a request gives none: 10 seconds, or the nearest length the generator accepts."""
        return min(max(DEFAULT_SECONDS, self.min_seconds), self.max_seconds)

    default_steps = 8             # the settings the model's authors use for small music: 8 steps, guidance scale 1.0
    default_cfg_scale = 1.0

    def load(self) -> None:
        pass

    def generate(self, req: JobRequest, on_stage: Callable[[str], None]) -> Clip:
        raise NotImplementedError

    def close(self) -> None:
        """Called once when the server stops. A generator that owns a child process or files ends and removes them here, so nothing outlives the
        process that holds the lease."""


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
    """Runs the chip runtime script `chip_generate.py` for each clip, in a child process, and reads the clip and its timings from the run directory.

    The caller holds the lease (run the server under `gozer run --chips 1 --`); the child inherits the environment, so it sees the leased chip.
    Each clip loads the model again (about the load time `chip_generate.py` records as `model_load_s`). That cost is part of every clip made this way.

    Disk and processes: each request's run directory and log are removed when the request ends, whether it succeeded, failed or timed out (`keep_runs`
    keeps them for debugging). `close` stops a child that is still running and removes a work directory this object made. The child is also stopped
    if anything goes wrong while waiting for it.
    """
    device_name = "one Blackhole chip"

    def __init__(self, chip_python: str, script: str, src_dir: str, model: str = "small-music", work_dir: Optional[str] = None,
                 timeout_s: float = 900.0, max_seconds: int = MAX_SECONDS, keep_runs: bool = False):
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, int) or not MIN_SECONDS <= max_seconds <= MAX_SECONDS:
            raise ValueError(f"max_seconds must be a whole number from {MIN_SECONDS} to {MAX_SECONDS} (the model's own limit), got {max_seconds!r}")
        self.model_name = f"stable-audio-3 {model}, diffusion transformer on one Blackhole chip"
        self.chip_python, self.script, self.src_dir, self.model = chip_python, script, src_dir, model
        self.own_work = work_dir is None
        self.work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="audio-demo-"))
        self.timeout_s = timeout_s
        self.max_seconds = max_seconds
        self.keep_runs = keep_runs
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._closed = False

    def load(self) -> None:
        """Check what the child needs. The model itself is loaded by the child for each clip."""
        missing = [str(p) for p in (self.chip_python, self.script) if not os.path.exists(p)]
        if not (Path(self.src_dir) / "stable_audio_3").is_dir():
            missing.append(f"{self.src_dir}/stable_audio_3")
        if missing:
            raise RuntimeError("these files are missing: " + ", ".join(missing))
        self.work.mkdir(parents=True, exist_ok=True)

    def command(self, req: JobRequest, run_dir: Path) -> list[str]:
        # --no-layers: the per-layer read backs are for the runtime's agreement check and cost time
        return [self.chip_python, self.script, "--model", self.model, "--out", str(run_dir), f"--prompt={req.prompt}",      # one token: a prompt that starts with a dash is not read as an option
                "--duration", repr(float(req.seconds)), "--steps", str(req.steps), "--cfg-scale", repr(float(req.cfg_scale)),
                "--seed", str(req.seed), "--no-layers"]

    @staticmethod
    def _stop(proc: subprocess.Popen) -> None:
        """Kill a child that is still running and reap it."""
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            proc = self._proc
        if proc is not None:
            self._stop(proc)                       # the request that waits on it sees "stopped" and cleans up its own files
        if self.own_work and not self.keep_runs:          # --keep-runs asks for the run directories to be left for inspection
            shutil.rmtree(self.work, ignore_errors=True)

    def generate(self, req: JobRequest, on_stage: Callable[[str], None]) -> Clip:
        run_dir = self.work / f"run-{uuid.uuid4().hex[:12]}"
        log = run_dir.with_suffix(".log")
        try:
            return self._run(req, run_dir, log, on_stage)
        finally:
            if not self.keep_runs:
                shutil.rmtree(run_dir, ignore_errors=True)
                log.unlink(missing_ok=True)

    def _run(self, req: JobRequest, run_dir: Path, log: Path, on_stage: Callable[[str], None]) -> Clip:
        env = dict(os.environ)
        env["PYTHONPATH"] = self.src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        on_stage("loading the model and sampling on the chip")
        self.work.mkdir(parents=True, exist_ok=True)
        with open(log, "wb") as out:
            with self._lock:
                if self._closed:
                    raise RuntimeError("the generator was stopped")
                proc = self._proc = subprocess.Popen(self.command(req, run_dir), stdout=out, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
            try:
                rc = proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                self._stop(proc)
                raise RuntimeError(f"the chip run did not finish in {self.timeout_s:.0f} s and was stopped; the chip may need a reset") from None
            finally:
                self._stop(proc)                   # a no-op for a child that has ended; stops one if anything else went wrong while waiting
                with self._lock:
                    self._proc = None
        if self._closed:
            raise RuntimeError("the chip run was stopped because the server is shutting down")
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
            # only the entries that are times (named *_s); a count such as the number of transformer passes is not a time
            stages = {k: float(v) for k, v in timings.items() if k.endswith("_s") and isinstance(v, (int, float)) and not isinstance(v, bool)}
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

    In all, one running job and up to `max_waiting` waiting ones fit; with `max_waiting` 0 an idle server takes one clip and refuses the next until it ends. Finished jobs are kept for `keep` entries, newest last.
    """
    def __init__(self, generator: Generator, max_waiting: int = 2, keep: int = 16):
        if max_waiting < 0 or keep < 1:
            raise ValueError("max_waiting must be 0 or more and keep at least 1")
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
        """Stop the worker and close the generator (which ends a clip that is running)."""
        with self.lock:
            self.stopping = True
            self.wake.notify_all()
        self.gen.close()

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
            # only a float can be NaN or infinite; a huge whole number is finite and the range checks below refuse it (math.isfinite would overflow on it)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
                raise ValueError(f"{key} must be a number")
            return v

        def whole(key, default):
            v = body.get(key, default)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{key} must be a whole number")
            return v

        seconds = number("duration", self.gen.default_seconds)
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
            # the running clip and the waiting ones together: a job just queued and not yet picked up by the worker counts, so the answer does not
            # depend on the worker's timing, and with no waiting slots an idle server still takes one clip
            if len(self.queue) + (1 if self.running is not None else 0) >= self.max_waiting + 1:
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
                 "defaults": {"duration": self.gen.default_seconds, "steps": self.gen.default_steps, "cfg_scale": self.gen.default_cfg_scale}}
            if self.state == "error":
                h["error"] = self.load_error
            return h


# ---- HTTP ----------------------------------------------------------------------------------------------------

def _strict_json(raw: bytes):
    def refuse(name):
        raise ValueError(f"{name} is not valid JSON")
    return json.loads(raw.decode("utf-8"), parse_constant=refuse)       # NaN and Infinity are refused


LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")


def host_allowed(host_header: Optional[str], bound_host: str) -> bool:
    """DNS rebinding guard. A server bound to the loopback address only answers requests whose Host names the loopback address, so a web page on
    another site (whose own name resolves to 127.0.0.1) cannot talk to it through the browser. A server bound to another address is open to
    whatever names reach it; the operator chose that."""
    if bound_host not in LOOPBACK_NAMES:
        return True
    if not host_header:
        return False
    name = host_header.strip()
    if name.startswith("["):                                  # [::1]:port
        name = name[1:].split("]", 1)[0]
    elif name.count(":") == 1:                                # name:port
        name = name.split(":", 1)[0]
    return name.lower() in LOOPBACK_NAMES


# ---- the API description -------------------------------------------------------------------------------------

API_VERSION = "0.1.0"
DOCS_PAGE = (b"<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>Stable Audio 3 small music job server</title></head><body>\n"
             b"<h1>Stable Audio 3 small music job server</h1>\n"
             b"<p>The machine-readable description of this API is <a href=\"/openapi.json\">/openapi.json</a> (OpenAPI 3.1).</p>\n"
             b"<p>Routes: GET /v1/health, POST /v1/audio/jobs, GET /v1/audio/jobs/{id}, GET /v1/audio/jobs/{id}/audio, DELETE /v1/audio/jobs/{id}.</p>\n"
             b"</body></html>\n")


def openapi_document(js: JobServer) -> dict:
    """The OpenAPI 3.1 description of every route the handler answers. The limits and defaults are read from the same constants and generator
    attributes that `JobServer.validate` and `JobServer.health` use, so the document and the validation cannot drift. tests/test_openapi.py drives
    each route and compares what comes back with this document."""
    gen = js.gen

    def ref(name):
        return {"$ref": f"#/components/schemas/{name}"}

    def error(description):
        return {"description": description, "content": {"application/json": {"schema": ref("Error")}}}

    job_id = {"name": "id", "in": "path", "required": True, "description": "the job id returned by POST /v1/audio/jobs",
              "schema": {"type": "string", "pattern": "^[0-9a-f]{32}$"}}
    host_refused = error("the Host header does not name the loopback address (a server bound to the loopback address answers its own name only)")
    no_job = error("no job has this id (it never existed or its record was dropped)")
    return {
        "openapi": "3.1.0",
        "info": {"title": "Stable Audio 3 small music job server", "version": API_VERSION,
                 "description": "One clip at a time with a short queue. A full queue answers 429; a model that is still loading answers 503."},
        "paths": {
            "/openapi.json": {"get": {"summary": "This document", "operationId": "openapi", "responses": {
                "200": {"description": "the OpenAPI document", "content": {"application/json": {"schema": {"type": "object"}}}}, "403": host_refused}}},
            "/docs": {"get": {"summary": "A small page that points at the document", "operationId": "docs", "responses": {
                "200": {"description": "an HTML page", "content": {"text/html": {"schema": {"type": "string"}}}}, "403": host_refused}}},
            "/v1/health": {"get": {"summary": "Server state, queue and the limits it accepts", "operationId": "health", "responses": {
                "200": {"description": "the state (status ok, loading or error), the model, the queue and the limits",
                        "content": {"application/json": {"schema": ref("Health")}}}, "403": host_refused}}},
            "/v1/audio/jobs": {"post": {
                "summary": "Queue a clip", "operationId": "submitJob",
                "requestBody": {"required": True, "content": {"application/json": {"schema": ref("JobRequest")}}},
                "responses": {
                    "202": {"description": "the job is queued", "content": {"application/json": {"schema": ref("JobAccepted")}}},
                    "400": error("the body is not valid JSON, or a field is missing, of the wrong type or out of range"),
                    "403": host_refused,
                    "413": error(f"the body is larger than {MAX_BODY_BYTES} bytes"),
                    "415": error("the Content-Type is not application/json"),
                    "429": error("the queue is full: a clip is being made and others are waiting"),
                    "503": error("the model is still loading, or failed to load")}}},
            "/v1/audio/jobs/{id}": {
                "get": {"summary": "A job's status, queue position, settings and stats", "operationId": "getJob", "parameters": [job_id], "responses": {
                    "200": {"description": "the job", "content": {"application/json": {"schema": ref("JobView")}}}, "403": host_refused, "404": no_job}},
                "delete": {"summary": "Cancel a job: a queued job is dropped; a running one cannot be stopped and its clip is discarded",
                           "operationId": "cancelJob", "parameters": [job_id], "responses": {
                    "200": {"description": "the job's status after the request", "content": {"application/json": {"schema": ref("CancelResult")}}},
                    "403": host_refused, "404": no_job}}},
            "/v1/audio/jobs/{id}/audio": {"get": {"summary": "The finished clip", "operationId": "getAudio", "parameters": [job_id], "responses": {
                "200": {"description": "the clip: FLAC from the chip generator, WAV from the fake generator",
                        "content": {"audio/flac": {"schema": {"type": "string", "format": "binary"}},
                                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}}},
                "403": host_refused, "404": no_job, "409": error("the job is not done, so there is no clip yet")}}},
        },
        "components": {"schemas": {
            "Error": {"type": "object", "required": ["detail"], "properties": {"detail": {"type": "string"}}},
            "JobRequest": {
                "type": "object", "required": ["prompt"], "additionalProperties": False,
                "properties": {
                    "prompt": {"type": "string", "minLength": 1, "maxLength": MAX_PROMPT_CHARS, "description": "a non-blank text describing the music"},
                    "duration": {"type": "number", "minimum": gen.min_seconds, "maximum": gen.max_seconds, "default": gen.default_seconds,
                                 "description": "seconds of audio"},
                    "steps": {"type": "integer", "minimum": 1, "maximum": gen.max_steps, "default": gen.default_steps},
                    "cfg_scale": {"type": "number", "minimum": MIN_CFG, "maximum": MAX_CFG, "default": gen.default_cfg_scale},
                    "seed": {"type": "integer", "minimum": 0, "maximum": MAX_SEED, "description": "a random seed is chosen when none is given, and comes back in settings"}}},
            "JobAccepted": {"type": "object", "required": ["id", "status"], "properties": {"id": {"type": "string"}, "status": {"enum": ["queued"]}}},
            "Settings": {"type": "object", "required": ["prompt", "duration", "steps", "cfg_scale", "seed"],
                         "properties": {"prompt": {"type": "string"}, "duration": {"type": "number"}, "steps": {"type": "integer"},
                                        "cfg_scale": {"type": "number"}, "seed": {"type": "integer"}}},
            "JobView": {"type": "object", "required": ["id", "status", "queue_position", "stage", "settings", "error", "cancel_requested", "stats"],
                        "properties": {
                            "id": {"type": "string"},
                            "status": {"enum": ["queued", "running", "done", "error", "cancelled"]},
                            "queue_position": {"type": "integer", "minimum": 0, "description": "1 for the next job to run; 0 when not queued"},
                            "stage": {"type": "string", "description": "what the generator reports it is doing; empty before it starts"},
                            "settings": ref("Settings"),
                            "error": {"type": ["string", "null"]},
                            "cancel_requested": {"type": "boolean"},
                            "stats": {"type": ["object", "null"], "description": "set once the job is done: audio_s, total_s, audio_format and stage times",
                                      "properties": {"audio_s": {"type": "number"}, "total_s": {"type": "number"},
                                                     "audio_format": {"enum": ["flac", "wav"]}, "stages": {"type": "object"}}}}},
            "CancelResult": {"type": "object", "required": ["id", "status", "cancel_requested"],
                             "properties": {"id": {"type": "string"}, "status": {"type": "string"}, "cancel_requested": {"type": "boolean"},
                                            "detail": {"type": "string"}}},
            "Health": {"type": "object", "required": ["status", "model", "device_name", "queue_depth", "queue_max", "busy", "limits", "defaults"],
                       "properties": {
                           "status": {"enum": ["ok", "loading", "error"]}, "model": {"type": "string"}, "device_name": {"type": "string"},
                           "queue_depth": {"type": "integer", "minimum": 0}, "queue_max": {"type": "integer", "minimum": 0}, "busy": {"type": "boolean"},
                           "error": {"type": ["string", "null"], "description": "present when status is error"},
                           "limits": {"type": "object", "required": ["min_seconds", "max_seconds", "max_steps", "min_cfg_scale", "max_cfg_scale"],
                                      "properties": {"min_seconds": {"type": "number", "const": gen.min_seconds},
                                                     "max_seconds": {"type": "number", "const": gen.max_seconds},
                                                     "max_steps": {"type": "integer", "const": gen.max_steps},
                                                     "min_cfg_scale": {"type": "number", "const": MIN_CFG},
                                                     "max_cfg_scale": {"type": "number", "const": MAX_CFG}}},
                           "defaults": {"type": "object", "required": ["duration", "steps", "cfg_scale"],
                                        "properties": {"duration": {"type": "number"}, "steps": {"type": "integer"}, "cfg_scale": {"type": "number"}}}}},
        }},
    }


def make_http_server(js: JobServer, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _host_ok(self) -> bool:
            if host_allowed(self.headers.get("Host"), host):
                return True
            self.close_connection = True
            self._err(403, "this server answers requests for its own loopback address only")
            return False

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
            if not self._host_ok():
                return
            if self.path == "/v1/health":
                return self._send(200, js.health())
            if self.path == "/openapi.json":
                return self._send(200, openapi_document(js))
            if self.path == "/docs":
                return self._send(200, DOCS_PAGE, "text/html; charset=utf-8")
            m = re.fullmatch(r"/v1/audio/jobs/([0-9a-f]{32})(/audio)?", self.path)
            if m is None:
                return self._err(404, "not found")
            job = js.get(m.group(1))
            if job is None:
                return self._err(404, "no such job")
            if not m.group(2):
                return self._send(200, js.job_view(job))
            clip = job.clip if job.status == "done" else None
            if clip is None:
                return self._err(409, f"the job is {job.status}; there is no clip")
            self._send(200, clip.audio, "audio/flac" if clip.audio_format == "flac" else "audio/wav")

        def do_POST(self):
            if not self._host_ok():
                return
            if self.path != "/v1/audio/jobs":
                return self._err(404, "not found")
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                # a page on another site can send other content types without the browser asking first; refusing them stops that page queueing work
                self.close_connection = True
                return self._err(415, "send the job as application/json")
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
            if not self._host_ok():
                return
            m = re.fullmatch(r"/v1/audio/jobs/([0-9a-f]{32})", self.path)
            if m is None:
                return self._err(404, "not found")
            result = js.cancel(m.group(1))
            if result is None:
                return self._err(404, "no such job")
            self._send(200, result)

    ThreadingHTTPServer.daemon_threads = True
    return ThreadingHTTPServer((host, port), Handler)


# ---- command line --------------------------------------------------------------------------------------------

def default_script(here: Path) -> str:
    """The chip runtime script: a `chip_generate.py` beside this file when one is there, else an empty string. The runtime is not part of this repository,
    so the chip generator needs `--script` or AUDIO_CHIP_GENERATE and refuses to start without one."""
    beside = here / "chip_generate.py"
    return str(beside) if beside.exists() else ""


def nonnegative(text: str) -> int:
    n = int(text)
    if n < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return n


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
    ap.add_argument("--max-waiting", type=nonnegative, default=2, help="jobs that may wait while one is running (0 or more)")
    ap.add_argument("--max-seconds", type=int, default=MAX_SECONDS, help="longest clip the chip generator accepts (the model's limit is 120)")
    ap.add_argument("--model", default=env("AUDIO_MODEL", "small-music"))
    ap.add_argument("--chip-python", default=env("ORCHARD_TTNN_PYTHON", sys.executable), help="the interpreter that has ttnn")
    ap.add_argument("--src-dir", default=env("ORCHARD_AUDIO_SRC", ""), help="the stable-audio-3 library checkout")
    ap.add_argument("--script", default=env("AUDIO_CHIP_GENERATE", default_script(here)), help="the chip runtime script, chip_generate.py (required for --generator chip)")
    ap.add_argument("--work-dir", default=env("AUDIO_WORK_DIR"), help="where each clip's run directory goes (default: a new temporary directory, removed on exit)")
    ap.add_argument("--keep-runs", action="store_true", help="keep each clip's run directory and log (for debugging; they are removed by default)")
    ap.add_argument("--fake-delay", type=float, default=2.0, help="seconds the fake generator pretends to work")
    return ap.parse_args(argv)


def build_generator(args) -> Generator:
    if args.generator == "fake":
        return FakeGenerator(delay_s=args.fake_delay)
    if not args.script:
        raise ValueError("--generator chip needs the chip runtime script: pass --script PATH or set AUDIO_CHIP_GENERATE")
    return SubprocessChipGenerator(args.chip_python, args.script, args.src_dir, model=args.model, work_dir=args.work_dir,
                                   max_seconds=args.max_seconds, keep_runs=args.keep_runs)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        generator = build_generator(args)
    except ValueError as e:
        print(f"serve_audio: {e}", file=sys.stderr)
        return 2
    js = JobServer(generator, max_waiting=args.max_waiting)
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
            print(f"serve_audio: the model did not load: {h['error']}", file=sys.stderr)
            return 2
        print("model ready", flush=True)
        if args.ui:
            here = Path(__file__).resolve().parent
            sys.path[:0] = [str(here), str(here.parent / "demo")]      # the page is in ../demo in this repository, and beside this file in a staged package
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
