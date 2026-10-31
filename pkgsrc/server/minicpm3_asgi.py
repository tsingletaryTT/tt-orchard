# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ASGI front end of the MiniCPM3-4B package, for a v6 thin bundle of kind `tt-dit-server`.

`tt-model` starts that kind with `uvicorn <module>:app`, so the standard-library server in scripts/minicpm3/serve.py cannot be started directly. This file
serves the same endpoints with the same request checks and the same generation loop (it imports `parse_request`, `encode_prompt`, `generate_ids`, `STATE` and
the lock from serve.py; nothing is copied):

    GET  /health               200 {"status": "ok", ...}, or 503 {"status": "unusable", ...} after a failure on the chip
    GET  /v1/models
    POST /v1/chat/completions  messages, max_tokens, stream. Decoding is greedy only: temperature and top_p are accepted and ignored.

One request runs at a time. A second request gets 503 with Retry-After at once. The chip work runs in a worker thread, so /health answers while a
request generates. A client that leaves (streaming or not) ends the generation before its next chip step and frees the slot. A failure inside a chip step
marks the server unusable: /health and every later request answer 503 until the server is restarted, because the recipe's own check found that a chip
that failed once (MMIO timeout) fails every later request.

serve.py takes flags. uvicorn does not pass them here, so the same settings come from the environment:

    MINICPM3_DIR             a directory with config.json, the tokenizer files and pytorch_model.bin (default: the pinned revision, downloaded into the
                             Hugging Face cache with HF_MODEL and TT_MODEL_WEIGHTS_REVISION, which the bundle's run.sh sets)
    MINICPM3_WEIGHTS         the weights as one safetensors file (default: a file converted once from pytorch_model.bin, kept in MINICPM3_CACHE)
    MINICPM3_CACHE           where that converted file goes (default: tt-orchard-minicpm3 inside HF_HOME)
    MINICPM3_MAX_LEN         context length, 128 to 32768 (default 2048; rounded up to a multiple of 128)
    MINICPM3_MLP_DTYPE       bf16 (default), bfp8 or bfp4
    MINICPM3_MLP_FIDELITY    hifi2 (default), lofi or hifi4

The device is opened when the server starts (the ASGI lifespan) and closed when it stops. Importing this module loads neither ttnn nor the chip model.
"""
import asyncio
import importlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REVISION = "d6b14ddaefdb11c624dd75c3c779549bc90b08cb"      # the revision of openbmb/MiniCPM3-4B that the stages measured and the card names
MLP_DTYPES = ("bf16", "bfp8", "bfp4")
MLP_FIDELITIES = ("hifi2", "lofi", "hifi4")


def _core():
    """serve.py's module. In a bundle it is `minicpm3_serve`, next to this file in extra_code/; in the repository it is scripts/minicpm3/serve.py."""
    try:
        return importlib.import_module("minicpm3_serve")
    except ImportError:
        spec = importlib.util.spec_from_file_location("minicpm3_serve", HERE.parent.parent / "scripts" / "minicpm3" / "serve.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["minicpm3_serve"] = mod
        spec.loader.exec_module(mod)
        return mod


serve = _core()
import convert_weights  # noqa: E402  serve.py put its own directory (extra_code/ in a bundle, scripts/minicpm3 here) on sys.path


def settings(environ=None):
    """The server's settings from the environment, as the attributes serve.py's flags would have. A value the chip model cannot take raises ValueError."""
    e = os.environ if environ is None else environ

    def choice(name, default, allowed):
        v = e.get(name, default)
        if v not in allowed:
            raise ValueError(f"{name} is {v!r}; it must be one of {', '.join(allowed)}")
        return v
    try:
        max_len = int(e.get("MINICPM3_MAX_LEN", 2048))
    except ValueError:
        raise ValueError(f"MINICPM3_MAX_LEN is {e.get('MINICPM3_MAX_LEN')!r}; it must be a whole number") from None
    if not 128 <= max_len <= 32768:
        raise ValueError(f"MINICPM3_MAX_LEN is {max_len}; it must be from 128 to 32768")
    max_len = -(-max_len // 128) * 128                  # rounded once here: the model builds its rotary tables from this value and the server advertises it
    return SimpleNamespace(model_dir=e.get("MINICPM3_DIR") or None, weights=e.get("MINICPM3_WEIGHTS") or None, max_len=max_len,
                           mlp_dtype=choice("MINICPM3_MLP_DTYPE", "bf16", MLP_DTYPES), mlp_fidelity=choice("MINICPM3_MLP_FIDELITY", "hifi2", MLP_FIDELITIES))


def cache_dir(environ=None):
    """Where the converted weights file goes."""
    e = os.environ if environ is None else environ
    if e.get("MINICPM3_CACHE"):
        return Path(e["MINICPM3_CACHE"])
    hf_home = e.get("HF_HOME")
    return (Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface") / "tt-orchard-minicpm3"


def weights_file(model_dir, cache):
    """The weights as one safetensors file: the model directory's own `model.safetensors`, or a file converted once from `pytorch_model.bin` into `cache`.
    The cache file's name holds the directory's name and a hash of its full path, so two directories never share a converted file."""
    import hashlib
    d = Path(model_dir).resolve()
    if (d / "model.safetensors").is_file():
        return d / "model.safetensors"
    out = Path(cache) / f"{d.name}-{hashlib.sha256(str(d).encode('utf-8')).hexdigest()[:12]}.safetensors"
    if not out.is_file():
        print(f"serve: converting pytorch_model.bin to {out.name} (first start only)", file=sys.stderr, flush=True)
        convert_weights.convert(d, out)
    return out


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir)


def _roots():
    """Where chip_model.py and ref_model.py are: next to this file (a bundle's extra_code/ holds both) or in scripts/minicpm3."""
    for base in (HERE, HERE.parent.parent / "scripts" / "minicpm3"):
        if (base / "chip_model.py").is_file():
            return base
    raise RuntimeError("chip_model.py was not found next to this file or in scripts/minicpm3")


def load_chip(a, weights):
    """Open device 0 and build the traced model. Returns (device, model, seconds). The caller closes the device."""
    sys.path.insert(0, str(_roots()))
    import ttnn
    dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
    t0 = time.time()
    try:
        model = serve.build_single_chip(dev, a.model_dir, str(weights), a.max_len, a.mlp_dtype, a.mlp_fidelity)
    except BaseException:
        ttnn.close_device(dev)
        raise
    return dev, model, time.time() - t0


def close_chip(dev, model):
    import ttnn
    ttnn.close_device(dev)


def startup(a):
    """Resolve the weights, load the tokenizer and the chip, fill serve.STATE. Returns a function that closes the chip."""
    if not a.model_dir:
        from huggingface_hub import snapshot_download
        a.model_dir = snapshot_download(os.environ.get("HF_MODEL") or serve.MODEL_ID,
                                        revision=os.environ.get("TT_MODEL_WEIGHTS_REVISION") or REVISION)
    weights = a.weights or weights_file(a.model_dir, cache_dir())
    serve.STATE.clear()
    serve.STATE["tok"] = load_tokenizer(a.model_dir)
    serve.STATE["max_len"] = a.max_len
    serve.STATE["batched_prefill"] = False         # the batched prefill hung the chip on the second request after a trace (see serve.py)
    dev, model, load_s = load_chip(a, weights)
    serve.STATE["model"] = model
    print(json.dumps({"event": "ready", "model": serve.MODEL_ID, "context": serve.STATE["max_len"], "mlp_dtype": a.mlp_dtype, "mlp_fidelity": a.mlp_fidelity,
                      "traced": True, "greedy_only": True, "load_s": round(load_s, 1)}), flush=True)
    return lambda: close_chip(dev, model)


# ---- HTTP -----------------------------------------------------------------------------------------------------------------
async def _reply(send, code, obj, headers=()):
    body = json.dumps(obj).encode()
    await send({"type": "http.response.start", "status": code,
                "headers": [(k.lower().encode(), v.encode()) for k, v in (*headers, ("Content-Type", "application/json"),
                                                                           ("Content-Length", str(len(body))))]})
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive):
    """The request body, or None when it is empty or larger than serve.MAX_BODY."""
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
    return serve.STATE.get("unusable")


def _chunk(cid, created, delta, fin=None):
    return ("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": serve.MODEL_ID,
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}) + "\n\n").encode()


def _mark_unusable(exc):
    """A step on the chip failed. Later requests would fail too (the recipe's check found a chip that timed out once failing every request), so the server says
    so on /health and refuses them until it is restarted."""
    serve.STATE["unusable"] = f"{type(exc).__name__}: {exc}"
    sys.stderr.write(f"serve: generation failed: {serve.STATE['unusable']}\n")


def _complete(ids, max_tokens, cid, created, cancelled):
    """The whole answer as a chat.completion object. Runs in a worker thread and releases the lock."""
    pieces, t0, t_first, t_end = [], time.perf_counter(), None, None
    try:
        for piece in serve.generate_ids(ids, max_tokens, should_stop=cancelled.is_set):
            t_end = time.perf_counter()
            t_first = t_first or t_end
            pieces.append(piece)
        t_end = time.perf_counter()
        # Read the shared state before the slot is released: the next request overwrites it as soon as it takes the slot.
        pt, ct = serve.STATE.get("last_counts", (len(ids), 0))
        finish = serve.STATE.get("finish", "stop")
    except Exception as exc:  # noqa: BLE001
        _mark_unusable(exc)
        raise
    finally:
        serve.LOCK.release()
    # Each answer token is followed by one chip step, so the steps after the first piece number about `ct` (a token split across pieces moves this by one).
    timings = {"first_token_s": round(t_first - t0, 4) if t_first else None,
               "decode_tok_s": round(ct / (t_end - t_first), 2) if t_first and t_end > t_first else None}
    return {"id": cid, "object": "chat.completion", "created": created, "model": serve.MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)}, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}, "timings": timings}


def _stream_worker(ids, max_tokens, cid, created, put, cancelled):
    """Produce the event stream's bytes with `put`. Runs in a worker thread and releases the lock. `put(None)` ends the stream."""
    gen = serve.generate_ids(ids, max_tokens, should_stop=cancelled.is_set)
    try:
        put(_chunk(cid, created, {"role": "assistant"}))
        for piece in gen:
            put(_chunk(cid, created, {"content": piece}))
        if cancelled.is_set():
            sys.stderr.write("serve: the client left before the response was complete\n")
            return
        put(_chunk(cid, created, {}, serve.STATE.get("finish", "stop")) + b"data: [DONE]\n\n")
    except Exception as exc:  # noqa: BLE001  a failure on the chip must reach the log; the stream then ends without [DONE]
        _mark_unusable(exc)
    finally:
        gen.close()
        serve.LOCK.release()
        put(None)


async def _chat(receive, send):
    body = await _read_body(receive)
    if body is None:
        return await _reply(send, 400, {"error": {"message": "request body must be 1 byte to 1 MiB"}})
    loop = asyncio.get_running_loop()
    try:
        messages, max_tokens, stream = serve.parse_request(body)
        ids = await loop.run_in_executor(None, serve.encode_prompt, messages)     # the tokenizer is not chip work, so it runs before the slot is taken
    except Exception as exc:  # noqa: BLE001  a template or tokenizer error is the client's input; ValueError is the usual case
        return await _reply(send, 400, {"error": {"message": str(exc) if isinstance(exc, ValueError) else f"bad request: {type(exc).__name__}: {exc}"}})
    bad = _unusable()
    if bad:
        return await _reply(send, 503, {"error": {"message": f"the chip is unusable until the server is restarted: {bad}"}})
    cid, created = f"chatcmpl-{int(time.time() * 1000)}", int(time.time())
    if not serve.LOCK.acquire(blocking=False):        # one request runs at a time; a second one is told so at once and never waits
        return await _reply(send, 503, {"error": {"message": "the server is busy with another request; retry shortly"}}, [("Retry-After", "5")])
    bad = _unusable()                                   # look again under the lock: the request that held it may have failed meanwhile
    if bad:
        serve.LOCK.release()
        return await _reply(send, 503, {"error": {"message": f"the chip is unusable until the server is restarted: {bad}"}})
    queue, cancelled = asyncio.Queue(), threading.Event()
    put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)  # noqa: E731

    async def watch_disconnect():
        while True:
            if (await receive())["type"] == "http.disconnect":
                cancelled.set()
                return

    watcher = asyncio.ensure_future(watch_disconnect())
    try:
        if not stream:
            try:
                out = await loop.run_in_executor(None, _complete, ids, max_tokens, cid, created, cancelled)
            except Exception as exc:  # noqa: BLE001  _complete released the lock and marked the server unusable
                return await _reply(send, 500, {"error": {"message": f"generation failed: {type(exc).__name__}"}})
            return await _reply(send, 200, out)
        worker = loop.run_in_executor(None, _stream_worker, ids, max_tokens, cid, created, put, cancelled)
        try:
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
            while (item := await queue.get()) is not None:
                await send({"type": "http.response.body", "body": item, "more_body": True})
            await send({"type": "http.response.body", "body": b""})
        finally:
            cancelled.set()
            await worker
    finally:
        cancelled.set()
        watcher.cancel()


async def _http(scope, receive, send):
    path, method = scope["path"].rstrip("/"), scope["method"]
    if method == "GET" and path == "/v1/models":
        return await _reply(send, 200, {"object": "list", "data": [{"id": serve.MODEL_ID, "object": "model", "owned_by": "openbmb"}]})
    if method == "GET" and path == "/health":
        bad = _unusable()
        if bad:                                         # a failed chip step: health checks must see it so the instance is restarted
            return await _reply(send, 503, {"status": "unusable", "reason": bad, "model": serve.MODEL_ID})
        return await _reply(send, 200, {"status": "ok", "model": serve.MODEL_ID, "context": serve.STATE["max_len"]})
    if method == "POST" and path == "/v1/chat/completions":
        return await _chat(receive, send)
    await _reply(send, 404, {"error": {"message": "not found"}})


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
