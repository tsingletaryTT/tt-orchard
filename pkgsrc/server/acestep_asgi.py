# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ASGI front end of the ACE-Step 1.5 turbo package, for a v6 thin bundle of kind `tt-dit-server`.

`tt-model` starts that kind with `uvicorn <module>:app`, so the standard-library server in acestep_serve.py cannot be started directly. This file
serves the same two endpoints with the same request checks (it imports `validate` and the limits from acestep_serve.py; nothing is copied) and
makes the clip with the same `acestep_run.Session`:

    GET  /health      200 {"status": "ok"}, or 503 {"status": "unusable", ...} after a failed chip call
    POST /v1/music    {"prompt": str, "lyrics": str (optional), "duration": seconds, "seed": int (optional)}
                      -> 200 audio/wav (48 kHz stereo), headers X-Timings (JSON) and X-Sample-Rate.

One clip is made at a time. A second request gets 503 with Retry-After at once. The chip work runs in a worker thread, so /health answers while a
clip is made. ACE-Step has no way to stop a clip half way, so a client that leaves does not end the generation: the slot stays taken until the
clip is finished, and the worker releases it then.

acestep_serve.py takes flags. uvicorn does not pass them here, so the same settings come from the environment:

    ACESTEP_WORK          a writable directory for the fetched model code, ACE-Step's project root and the clips in flight
                          (default: $XDG_CACHE_HOME or ~/.cache, then ace-step-1.5-turbo-blackhole)
    ACESTEP_BUNDLE_CODE   the `code/` directory of the community bundle, if you already have it as real files (default: fetched into ACESTEP_WORK)
    ACESTEP_SNAPSHOT      the weights snapshot directory, if you already have it (default: the pinned revision in the Hugging Face cache)
    ACESTEP_FAST          all (default), none, or a comma-separated list of the steps of acestep_fast.py
    HF_MODEL, TT_MODEL_WEIGHTS_REVISION   the weights repository and revision (run.sh sets both from the manifest)

The bundle code (sunandbean/acestep-1.5-turbo-p100a) and the weights are not in the bundle. At start-up both are fetched at the pinned revisions and
checked against acestep_pins.json: the code as real files (a link into the Hugging Face cache makes ACE-Step run other model code without an error,
issue 270) and the weights by size. `acestep_run.Session` then builds its own checkpoint tree under the work directory, because ACE-Step rewrites
files in it (issue 264). The device opens when the server starts (the ASGI lifespan) and closes when it stops.
"""
import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent       # in the repository: pkgsrc/server/ -> the repository root


def _load(module: str, *candidates: Path):
    """`module` if it is importable (in a bundle every file sits in extra_code/, next to this one), else the first candidate file."""
    try:
        return importlib.import_module(module)
    except ImportError:
        for path in candidates:
            if path.is_file():
                spec = importlib.util.spec_from_file_location(module, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module] = mod
                spec.loader.exec_module(mod)
                return mod
        raise


serve = _load("acestep_serve", REPO / "scripts" / "acestep" / "acestep_serve.py")
check_bundle = _load("check_bundle", REPO / "pkgsrc" / "acestep" / "tools" / "check_bundle.py")


def _pins() -> dict:
    for path in (HERE / "acestep_pins.json", REPO / "pkgsrc" / "can" / "acestep-pins.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise RuntimeError("acestep_pins.json was not found next to this file or in pkgsrc/can")


PINS = _pins()
MODEL_ID = PINS["model"]
LOCK = threading.Lock()         # one clip at a time; taken without waiting, released by the worker
STATE: dict = {}                # session, work: filled by startup()
WEIGHT_PATTERNS = ["acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*", "config.json"]    # as `run.sh fetch` of the recipe package


def settings(environ=None):
    """The server's settings from the environment, as the attributes acestep_serve.py's flags would have."""
    e = os.environ if environ is None else environ
    cache = e.get("XDG_CACHE_HOME") or str(Path(e.get("HOME") or Path.home()) / ".cache")
    return SimpleNamespace(
        work=Path(e.get("ACESTEP_WORK") or Path(cache) / "ace-step-1.5-turbo-blackhole"), bundle_code=e.get("ACESTEP_BUNDLE_CODE") or None,
        snapshot=e.get("ACESTEP_SNAPSHOT") or None, fast=e.get("ACESTEP_FAST", "all"), model=e.get("HF_MODEL") or MODEL_ID,
        revision=e.get("TT_MODEL_WEIGHTS_REVISION") or PINS["model_revision"], host=e.get("HOST", "127.0.0.1"), port=int(e.get("PORT", serve.PORT)))


# ---- inputs ---------------------------------------------------------------------------------------------------------------
def problems_of(code, snapshot, hash_weights=False) -> list:
    """What differs between the code directory and the weights snapshot and the pinned ones (an empty list is a match). Opens no device."""
    return check_bundle.problems(code, snapshot, PINS, hash_weights)


def download_code(work: Path, repo: str, revision: str) -> Path:
    """The bundle's `code/` directory as real files under `work`. A directory that already matches the pin is kept and nothing is fetched."""
    target = Path(work) / "bundle"
    code = target / "code"
    if code.is_dir():
        links = any(p.is_symlink() for p in code.rglob("*"))
        digest, n = check_bundle.tree_hash(code)
        if not links and digest == PINS["bundle_tree_sha256"] and n == PINS["bundle_files"]:
            return code
        shutil.rmtree(code)                       # a half download or a different revision: start again
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=revision, allow_patterns=["code/*"], local_dir=str(target))   # local_dir writes real files
    return code


def download_weights(model: str, revision: str) -> Path:
    """The weights snapshot directory (in the Hugging Face cache; its name is the revision). Only what the turbo path reads is fetched."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, revision=revision, allow_patterns=WEIGHT_PATTERNS))


def resolve_inputs(a):
    """(code directory, weights snapshot directory), fetched when not given, and checked against the pins. Raises RuntimeError on a mismatch."""
    code = Path(a.bundle_code) if a.bundle_code else download_code(a.work, PINS["bundle_repo"], PINS["bundle_revision"])
    snapshot = Path(a.snapshot) if a.snapshot else download_weights(a.model, a.revision)
    found = problems_of(code, snapshot)
    if found:
        raise RuntimeError("the model inputs do not match the pins: " + "; ".join(found))
    return code, snapshot


def startup(a):
    """Fetch and check the inputs, open the chip and load the models; fill STATE. Returns a function that closes the chip."""
    import acestep_fast                                       # the names of the steps, checked before anything is fetched or loaded
    steps = acestep_fast.parse_steps(a.fast)
    code, snapshot = resolve_inputs(a)
    import acestep_run                                        # imports torch
    root = Path(a.work) / "root"
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    session = acestep_run.Session(device="chip", bundle_code=code, snapshot=snapshot, root=root, fast=steps)
    work = Path(a.work) / "clips-in-flight"
    work.mkdir(parents=True, exist_ok=True)
    STATE.update(session=session, work=work)
    print(json.dumps({"event": "ready", "model": MODEL_ID, "revision": session.WEIGHTS_REVISION, "fast": list(session.fast),
                      "load_s": round(time.perf_counter() - t0, 2)}), flush=True)
    return session.close


# ---- HTTP -----------------------------------------------------------------------------------------------------------------
async def _reply(send, code, payload, ctype="application/json", headers=()):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": code,
                "headers": [(k.lower().encode(), v.encode()) for k, v in (*headers, ("Content-Type", ctype), ("Content-Length", str(len(body))))]})
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive):
    """The request body, or None when it is empty, larger than serve.MAX_BODY, or the client left."""
    body = b""
    while True:
        msg = await receive()
        if msg["type"] == "http.disconnect":
            return None
        body += msg.get("body", b"")
        if len(body) > serve.MAX_BODY:
            return None
        if not msg.get("more_body"):
            return body or None


def _unusable():
    """Why the chip cannot serve, or None. A chip call that raised set `failed` on the worker; the card may be unhealthy until the server restarts."""
    worker = getattr(STATE.get("session"), "worker", None)
    return "a chip call failed" if worker is not None and getattr(worker, "failed", False) else None


def _make_clip(params):
    """One clip as (wav bytes, timings). Runs in a worker thread and releases the slot in a `finally`, whatever happens."""
    tmp = None
    try:
        tmp = Path(tempfile.mkdtemp(prefix="clip-", dir=STATE["work"]))
        t0 = time.perf_counter()
        wav, timings = STATE["session"].one(tmp, params["duration"], params["prompt"], params["lyrics"], params["seed"])
        payload = Path(wav).read_bytes()
        return payload, {**timings, "request_s": round(time.perf_counter() - t0, 4)}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)       # the clip is in memory; its directory is gone before the reply is sent
        LOCK.release()


async def _music(receive, send):
    body = await _read_body(receive)
    if body is None:
        return await _reply(send, 400, {"error": f"the body must be 1 to {serve.MAX_BODY} bytes"})
    try:
        params = serve.validate(json.loads(body))
    except (ValueError, UnicodeDecodeError) as exc:        # json.JSONDecodeError is a ValueError
        return await _reply(send, 400, {"error": str(exc)})
    bad = _unusable()
    if bad:
        return await _reply(send, 503, {"error": f"the chip is unusable until the server is restarted: {bad}"})
    if not LOCK.acquire(blocking=False):                  # one clip at a time; a second request is told so at once and never waits
        return await _reply(send, 503, {"error": "a clip is being made; try again"}, headers=[("Retry-After", "5")])
    bad = _unusable()                                      # look again under the lock: the request that held it may have failed a chip call
    if bad:
        LOCK.release()
        return await _reply(send, 503, {"error": f"the chip is unusable until the server is restarted: {bad}"})
    try:
        payload, timings = await asyncio.get_running_loop().run_in_executor(None, _make_clip, params)
    except Exception as exc:  # noqa: BLE001  a failed generate is a 500 with its type, never the model's text
        sys.stderr.write(f"serve: generation failed: {type(exc).__name__}\n")
        return await _reply(send, 500, {"error": f"generation failed ({type(exc).__name__})"})
    await _reply(send, 200, payload, "audio/wav", [("X-Timings", json.dumps(timings)), ("X-Sample-Rate", str(serve.SAMPLE_RATE))])


async def _http(scope, receive, send):
    path, method = scope["path"].rstrip("/"), scope["method"]
    if method == "GET" and path == "/health":
        bad = _unusable()
        if bad:                                            # health checks must see it, so the instance is restarted
            return await _reply(send, 503, {"status": "unusable", "reason": bad})
        return await _reply(send, 200, {"status": "ok"})
    if method == "POST" and path == "/v1/music":
        return await _music(receive, send)
    await _reply(send, 404, {"error": "not found"})


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        close = None
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                try:
                    close = await asyncio.get_running_loop().run_in_executor(None, startup, settings())
                except Exception as exc:  # noqa: BLE001
                    await send({"type": "lifespan.startup.failed", "message": f"{type(exc).__name__}: {exc}"})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                if close:
                    await asyncio.get_running_loop().run_in_executor(None, close)
                await send({"type": "lifespan.shutdown.complete"})
                return
    elif scope["type"] == "http":
        await _http(scope, receive, send)
