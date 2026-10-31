# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ASGI front end of the Stable Audio 3 small-music package, for a v6 thin bundle of kind `tt-dit-server`.

`tt-model` starts that kind with `uvicorn <module>:app`, so the standard-library job server cannot be started directly. This file serves the same
routes with the same request checks, the same queue and the same job views. It imports `JobServer`, `Generator`, `FakeGenerator`, `host_allowed`,
`openapi_document` and the limits from the job server (shipped in the bundle as `extra_code/stable_audio_jobs.py`; nothing is copied):

    GET    /health                    200 when the model is loaded, 503 while it loads or after a failed load (a health check for tt-model)
    GET    /v1/health                 the job server's own health document (always 200; the state is in the body)
    POST   /v1/audio/jobs             {prompt, duration?, steps?, cfg_scale?, seed?} -> 202 {id}; 400, 413, 415, 429 queue full, 503 not ready
    GET    /v1/audio/jobs/<id>        status, stage, settings, stats, error
    GET    /v1/audio/jobs/<id>/audio  the clip (409 until the job is done)
    DELETE /v1/audio/jobs/<id>        a queued job is dropped; a running one cannot be stopped and its clip is discarded
    GET    /openapi.json, /docs

What differs from the stand-alone server:

  * The model stays loaded. The stand-alone server starts a new process for each clip and loads the model again each time. This front end keeps one
    resident runtime (`ResidentChipGenerator`, which uses `audio_runtimes.ChipRuntime`) on the chip, so a clip does not pay the load.
  * The device opens in the ASGI lifespan startup and closes in the shutdown. uvicorn accepts no connection until startup returns, so `/health` answers
    only after the weights are on the chip. Shutdown waits for a running clip to end before it closes the device.
  * The job server already has its own queue: a clip runs in the job server's worker thread, the request handlers only take its lock for a moment, and
    a submit to a full queue gets 429 at once. Nothing here waits for the chip. A job is not tied to the connection that submitted it, so a client that
    leaves does not stop its clip; `DELETE` does that for a queued job.
  * The Host check of the stand-alone server stays: with the server bound to the loopback address (HOST unset or 127.0.0.1), a request whose Host names
    anything else gets 403.

uvicorn passes no flags, so the settings come from the environment:

    HOST, PORT                    only used for the Host check (run.sh passes them to uvicorn; default 127.0.0.1 and 20170)
    AUDIO_GENERATOR               chip (default) or fake (a tone, no chip; for trying the routes)
    AUDIO_MODEL                   the library's model name (default small-music)
    AUDIO_MAX_WAITING             jobs that may wait while one runs, 0 or more (default 2)
    AUDIO_MAX_SECONDS             longest clip accepted, 1 to 120 (default 120)
    AUDIO_HOST_DECODER=1          keep the whole autoencoder decoder on the host (default: its transformer layers run on the chip)
    AUDIO_WARMUP_SECONDS          run one clip of this many seconds when the server starts, so the first request does not compile kernels (default 0: off)
    AUDIO_PKG_CACHE               where the private cache of the pinned weights is made (default: under XDG_CACHE_HOME or ~/.cache)
    HF_MODEL, TT_MODEL_WEIGHTS_REVISION   the weights repository and its revision (run.sh sets both from the manifest)
    HF_TOKEN                      the account that accepted the model's terms (the weights are gated)
"""
import asyncio
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
MODEL_ID = "stabilityai/stable-audio-3-small-music"
REVISION = "8bdfb53c6b0fbc62113cd571fd37373efe15fa14"
SAMPLE_SIZE = 5292032                  # the model's own sample_size (120 s at 44,100 Hz); chip_generate.py's default
DEFAULT_PORT = 20170
JOB_PATH = re.compile(r"/v1/audio/jobs/([0-9a-f]{32})(/audio)?")
SHUTDOWN_WAIT_S = 300.0                # choice: the longest a running clip is waited for at shutdown (a 120 s clip takes about 5 s on the chip)


def _core():
    """The job server's module. In a bundle it is `stable_audio_jobs`, next to this file in extra_code/; in the repository it is beside this file."""
    try:
        return importlib.import_module("stable_audio_jobs")
    except ImportError:
        spec = importlib.util.spec_from_file_location("stable_audio_jobs", HERE / "stable_audio_jobs.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["stable_audio_jobs"] = mod
        spec.loader.exec_module(mod)
        return mod


jobs = _core()
STATE: dict = {}                       # js (the JobServer), gen (its generator), host (the address uvicorn binds)


# ---- settings -------------------------------------------------------------------------------------------------------------
def _whole(environ, name, default, low, high=None):
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be {low} or more" + (f" and at most {high}" if high is not None else "") + f", got {value}")
    return value


def settings(environ=None):
    """The server's settings from the environment (uvicorn passes no flags). A bad value raises ValueError that names the variable."""
    e = os.environ if environ is None else environ
    generator = e.get("AUDIO_GENERATOR", "chip")
    if generator not in ("chip", "fake"):
        raise ValueError(f"AUDIO_GENERATOR must be chip or fake, got {generator!r}")
    return SimpleNamespace(
        generator=generator, model=e.get("AUDIO_MODEL", "small-music"), max_waiting=_whole(e, "AUDIO_MAX_WAITING", 2, 0),
        max_seconds=_whole(e, "AUDIO_MAX_SECONDS", jobs.MAX_SECONDS, 1, jobs.MAX_SECONDS), host=e.get("HOST", "127.0.0.1") or "127.0.0.1",
        port=_whole(e, "PORT", DEFAULT_PORT, 1, 65535), host_decoder=e.get("AUDIO_HOST_DECODER") == "1",
        warmup_seconds=_whole(e, "AUDIO_WARMUP_SECONDS", 0, 0, jobs.MAX_SECONDS))


# ---- the pinned weights ---------------------------------------------------------------------------------------------------
def _roots():
    """Directories that hold `orchard/skills/audio-chip-templates/`: next to this file (a bundle's extra_code/) or the repository root."""
    found = [b for b in (HERE, HERE.parents[1]) if (b / "orchard" / "skills" / "audio-chip-templates" / "audio_runtimes.py").is_file()]
    if not found:
        raise RuntimeError("the chip runtime (orchard/skills/audio-chip-templates/audio_runtimes.py) was not found next to this file or in the repository")
    return found[0]


def _load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_weights(environ, download, *, repo=None, revision=None):
    """Download the pinned revision and make a cache directory that holds only it, so the model library (which resolves the checkpoint and the text
    encoder through the cache's `main` reference) cannot load any other revision. Sets HF_HUB_CACHE in `environ` and returns the directory.

    `download(repo, revision)` returns the snapshot directory (huggingface_hub.snapshot_download in the server). Its name must be the revision."""
    repo = repo or environ.get("HF_MODEL") or MODEL_ID
    revision = revision or environ.get("TT_MODEL_WEIGHTS_REVISION") or REVISION
    snapshot = Path(download(repo, revision))
    if snapshot.name != revision:
        raise RuntimeError(f"the download of {repo} landed on revision {snapshot.name}, and this bundle is pinned to revision {revision}")
    cache = environ.get("AUDIO_PKG_CACHE") or str(Path(environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "stable-audio-3-small-music-blackhole")
    private = Path(os.path.abspath(cache)) / revision / "hub"
    check = _load_file(_roots() / "orchard" / "skills" / "audio-chip-templates" / "check_revision.py", "audio_check_revision")
    check.build_private_cache(private, snapshot.parents[1].name, revision, Path(os.path.abspath(snapshot)))
    environ["HF_HUB_CACHE"] = str(private)
    return private


def _download(repo, revision):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision)


# ---- the generator --------------------------------------------------------------------------------------------------------
class ResidentChipGenerator(jobs.Generator):
    """Makes each clip with one resident chip runtime (audio_runtimes.ChipRuntime): the model, the chip's weights and its kernel cache stay loaded between
    clips. `load` runs once (it opens the device); `generate` runs in the job server's worker thread, one clip at a time; `close` closes the device."""
    device_name = "one Blackhole chip"

    def __init__(self, model="small-music", *, chip_decoder=True, max_seconds=jobs.MAX_SECONDS, work_dir=None, warmup_seconds=0, runtime_factory=None):
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, int) or not jobs.MIN_SECONDS <= max_seconds <= jobs.MAX_SECONDS:
            raise ValueError(f"max_seconds must be a whole number from {jobs.MIN_SECONDS} to {jobs.MAX_SECONDS} (the model's own limit), got {max_seconds!r}")
        self.model, self.chip_decoder, self.max_seconds, self.warmup_seconds = model, chip_decoder, max_seconds, warmup_seconds
        self.model_name = f"stable-audio-3 {model}, diffusion transformer" + (" and decoder layers" if chip_decoder else "") + " on one Blackhole chip"
        self.work = Path(work_dir) if work_dir else None
        self._factory = runtime_factory
        self.runtime = None
        self.load_timings: dict = {}
        self._closed = False

    def _runtime(self):
        if self._factory is not None:
            return self._factory()
        resolve_weights(os.environ, _download)         # the weights are gated: HF_TOKEN must belong to an account that accepted the terms
        root = _roots()
        sys.path.insert(0, str(root))
        rts = _load_file(root / "orchard" / "skills" / "audio-chip-templates" / "audio_runtimes.py", "audio_runtimes")
        # record_layers=False: the per-layer read backs are for the agreement check and cost time
        return rts.ChipRuntime(self.model, device_id=0, record_layers=False, chip_decoder=self.chip_decoder)

    def load(self):
        if self.runtime is not None:
            return
        if self.work is None:
            import tempfile
            self.work = Path(tempfile.mkdtemp(prefix="audio-clips-"))
        self.work.mkdir(parents=True, exist_ok=True)
        runtime = self._runtime()
        self.load_timings = runtime.load() or {}
        self.runtime = runtime
        if self.warmup_seconds:
            t0 = time.time()
            self.generate(jobs.JobRequest("a short warm-up clip", self.warmup_seconds, self.default_steps, self.default_cfg_scale, 0), lambda stage: None)
            self.load_timings["warmup_s"] = round(time.time() - t0, 3)

    def generate(self, req, on_stage):
        if self.runtime is None:
            raise RuntimeError("the generator is not loaded")
        on_stage("sampling and decoding on the chip")
        run_dir = self.work / f"run-{uuid.uuid4().hex[:12]}"
        try:
            self.runtime.generate({"prompt": req.prompt, "duration": req.seconds, "steps": req.steps, "cfg_scale": req.cfg_scale, "seed": req.seed,
                                   "sample_size": SAMPLE_SIZE}, out_dir=run_dir)
            clip_path = run_dir / "clips" / "audio.flac"
            if not clip_path.is_file():
                raise RuntimeError("the chip run finished without writing a clip")
            data = clip_path.read_bytes()
            stages = {}
            try:
                timings = json.loads((run_dir / "chip.json").read_text()).get("timings", {})
                # only the entries that are times (named *_s); the load average and the pass count are not
                stages = {k: float(v) for k, v in timings.items() if k.endswith("_s") and isinstance(v, (int, float)) and not isinstance(v, bool)}
            except (OSError, ValueError):
                pass                                   # the clip is good without its timings
            return jobs.Clip(data, "flac", jobs.flac_seconds(data), stages)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.runtime is not None:
            self.runtime.close()


def build_generator(a):
    if a.generator == "fake":
        return jobs.FakeGenerator(delay_s=float(os.environ.get("AUDIO_FAKE_DELAY", "0.2")))
    return ResidentChipGenerator(a.model, chip_decoder=not a.host_decoder, max_seconds=a.max_seconds, work_dir=os.environ.get("AUDIO_WORK_DIR") or None,
                                 warmup_seconds=a.warmup_seconds)


# ---- life cycle -----------------------------------------------------------------------------------------------------------
def startup(a):
    """Build the generator, load it (this opens the device), start the job server. Raises if the load fails. Runs in a worker thread."""
    gen = build_generator(a)
    try:
        gen.load()
    except BaseException:
        try:
            gen.close()
        except Exception:  # noqa: BLE001  the load error is the one to report
            pass
        raise
    js = jobs.JobServer(gen, max_waiting=a.max_waiting)
    js.start()
    t0 = time.time()
    while js.health()["status"] == "loading" and time.time() - t0 < 30:     # the generator is loaded, so this is the worker thread starting
        time.sleep(0.01)
    STATE.update(js=js, gen=gen, host=a.host)
    print(json.dumps({"ready": True, "model": gen.model_name, "load": getattr(gen, "load_timings", {}), "max_waiting": a.max_waiting,
                      "decoder_on_chip": getattr(gen, "chip_decoder", None)}), flush=True)

    def close():
        """Stop taking jobs, wait for a clip that is running, then close the device (closing it under a running clip would crash the runtime)."""
        with js.lock:
            js.stopping = True
            js.wake.notify_all()
        for t in js.threads:
            t.join(SHUTDOWN_WAIT_S)
        gen.close()
    return close


# ---- HTTP -----------------------------------------------------------------------------------------------------------------
async def _reply(send, code, payload, ctype="application/json"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": code, "headers": [(b"content-type", ctype.encode()), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def _err(send, code, detail):
    await _reply(send, code, {"detail": detail})


async def _read_body(receive, limit):
    """The request body, or None when it is larger than `limit` (or the client left)."""
    body = b""
    while True:
        msg = await receive()
        if msg["type"] == "http.disconnect":
            return None
        body += msg.get("body", b"")
        if len(body) > limit:
            return None
        if not msg.get("more_body"):
            return body


async def _submit(headers, receive, send):
    ctype = (headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        # a page on another site can send other content types without the browser asking first; refusing them stops that page queueing work
        return await _err(send, 415, "send the job as application/json")
    try:
        length = int(headers.get("content-length") or 0)
    except ValueError:
        length = -1
    if length < 0:
        return await _err(send, 400, "bad Content-Length")
    if length > jobs.MAX_BODY_BYTES:
        return await _err(send, 413, f"the body is larger than {jobs.MAX_BODY_BYTES} bytes")
    body = await _read_body(receive, jobs.MAX_BODY_BYTES)
    if body is None:
        return await _err(send, 413, f"the body is larger than {jobs.MAX_BODY_BYTES} bytes")
    js = STATE["js"]
    try:
        req = js.validate(jobs._strict_json(body))
    except (ValueError, UnicodeDecodeError) as exc:
        return await _err(send, 400, str(exc))
    try:
        job = js.submit(req)
    except jobs.QueueFull as exc:
        return await _err(send, 429, str(exc))
    except jobs.NotReady as exc:
        return await _err(send, 503, str(exc))
    await _reply(send, 202, {"id": job.id, "status": "queued"})


async def _http(scope, receive, send):
    method, path = scope["method"], scope["path"]
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
    js = STATE["js"]
    if not jobs.host_allowed(headers.get("host"), STATE.get("host", "127.0.0.1")):
        return await _err(send, 403, "this server answers requests for its own loopback address only")
    if method == "GET" and path == "/health":
        h = js.health()
        return await _reply(send, 200 if h["status"] == "ok" else 503, h)
    if method == "GET" and path == "/v1/health":
        return await _reply(send, 200, js.health())
    if method == "GET" and path == "/openapi.json":
        return await _reply(send, 200, jobs.openapi_document(js))
    if method == "GET" and path == "/docs":
        return await _reply(send, 200, jobs.DOCS_PAGE, "text/html; charset=utf-8")
    if method == "POST" and path == "/v1/audio/jobs":
        return await _submit(headers, receive, send)
    m = JOB_PATH.fullmatch(path)
    if m and method == "GET":
        job = js.get(m.group(1))
        if job is None:
            return await _err(send, 404, "no such job")
        if not m.group(2):
            return await _reply(send, 200, js.job_view(job))
        clip = job.clip if job.status == "done" else None
        if clip is None:
            return await _err(send, 409, f"the job is {job.status}; there is no clip")
        return await _reply(send, 200, clip.audio, "audio/flac" if clip.audio_format == "flac" else "audio/wav")
    if m and method == "DELETE" and not m.group(2):
        result = js.cancel(m.group(1))
        if result is None:
            return await _err(send, 404, "no such job")
        return await _reply(send, 200, result)
    await _err(send, 404, "not found")


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        close = None
        loop = asyncio.get_running_loop()
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                try:
                    close = await loop.run_in_executor(None, startup, settings())
                except Exception as exc:  # noqa: BLE001
                    await send({"type": "lifespan.startup.failed", "message": f"{type(exc).__name__}: {exc}"})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                if close:
                    await loop.run_in_executor(None, close)
                await send({"type": "lifespan.shutdown.complete"})
                return
    elif scope["type"] == "http":
        await _http(scope, receive, send)
