# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ASGI front end of the Granite-4.0-H-1B package, for a v6 thin bundle of kind `tt-dit-server`.

`tt-model` starts that kind with `uvicorn <module>:app`, so the standard-library server in serve.py cannot be started directly. This file
serves the same endpoints with the same request checks and the same generation loop (it imports `parse_request`, `encode_prompt`,
`generate`, `ChipEngine` and the lock from serve.py; nothing is copied):

    GET  /health               200 {"status": "ok", ...}, or 503 {"status": "unusable", ...} after a failed trace capture
    GET  /v1/models
    POST /v1/chat/completions  messages, max_tokens or max_completion_tokens, stream, stop, temperature, top_p, seed

One request runs at a time. A second request gets 503 with Retry-After at once. The chip work runs in a worker thread, so /health answers
while a request generates. A streaming client that leaves ends the generation at the next piece and frees the slot.

serve.py takes flags. uvicorn does not pass them here, so the same settings come from the environment:

    GRANITE_MODEL_DIR            a directory with config.json, model.safetensors, tokenizer.json and chat_template.jinja
                                 (default: the pinned revision, downloaded into the Hugging Face cache with HF_MODEL and
                                 TT_MODEL_WEIGHTS_REVISION, which the bundle's run.sh sets)
    GRANITE_MAX_LEN              context length (default 2048)
    GRANITE_SSD                  composed (default) or another mode of the runtime
    GRANITE_SSD_PRECISION        fp32 (default) or bf16
    GRANITE_NO_TRACE=1           run the step eagerly (slower; for comparison only)
    GRANITE_CHUNKED_PREFILL=0    prefill with one decode step per token (default: chunked prefill)

The device is opened when the server starts (the ASGI lifespan) and closed when it stops.
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


def _core():
    """serve.py's module. In a bundle it is `granite_serve`, next to this file in extra_code/; in the repository it is serve.py beside this file."""
    try:
        return importlib.import_module("granite_serve")
    except ImportError:
        spec = importlib.util.spec_from_file_location("granite_serve", HERE / "serve.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["granite_serve"] = mod
        spec.loader.exec_module(mod)
        return mod


serve = _core()


def settings(environ=None):
    """The server's settings from the environment, as the attributes serve.py's flags would have."""
    e = os.environ if environ is None else environ
    return SimpleNamespace(
        model_dir=e.get("GRANITE_MODEL_DIR") or None, host=e.get("HOST", "127.0.0.1"), port=int(e.get("PORT", 20110)),
        max_len=int(e.get("GRANITE_MAX_LEN", 2048)), ssd=e.get("GRANITE_SSD", "composed"),
        ssd_precision=e.get("GRANITE_SSD_PRECISION", "fp32"), no_trace=e.get("GRANITE_NO_TRACE") == "1",
        no_chunked_prefill=e.get("GRANITE_CHUNKED_PREFILL") == "0")


def _roots():
    """Where the runtime and its kernels are: next to this file (a bundle's extra_code/ holds all of them), or in the recipe package's
    runtime/ (HERE/../runtime)."""
    for base in (HERE, HERE / "extra_code", HERE.parent / "runtime"):
        if (base / "granite_chip.py").is_file():
            return base, base / "kernels"
    raise RuntimeError("granite_chip.py was not found next to this file or in ../runtime")


def load_chip(a):
    """Open device 0 and build the model. Returns (device, model, seconds). The caller closes both."""
    runtime, kernels = _roots()
    sys.path.insert(0, str(runtime))
    import ttnn
    from granite_chip import GraniteChip
    dev = ttnn.open_device(device_id=0, trace_region_size=400 * 1024 * 1024)
    t0 = time.time()
    try:
        model = GraniteChip(dev, a.model_dir, max_len=a.max_len, ssd=a.ssd, kernel_dir=kernels, trace=not a.no_trace,
                            ssd_precision=a.ssd_precision, chunked_prefill=not a.no_chunked_prefill)
    except BaseException:
        ttnn.close_device(dev)
        raise
    return dev, model, time.time() - t0


def startup(a):
    """Load the tokenizer, the chat template and the chip; fill serve.STATE. Returns a function that closes the chip."""
    if not a.model_dir:
        from huggingface_hub import snapshot_download
        a.model_dir = snapshot_download(os.environ.get("HF_MODEL") or serve.MODEL_ID,
                                        revision=os.environ.get("TT_MODEL_WEIGHTS_REVISION") or serve.REVISION)
    serve.STATE["tok"] = serve.load_tokenizer(a.model_dir)
    serve.STATE["template"] = serve.make_template(Path(a.model_dir) / "chat_template.jinja")
    serve.STATE["stop_ids"] = serve.stop_token_ids(a.model_dir, serve.STATE["tok"])
    dev, model, load_s = load_chip(a)
    serve.STATE["engine"] = serve.ChipEngine(model)
    print(json.dumps(serve.ready_event(model, a, load_s)), flush=True)

    def close():
        import ttnn
        model.close()
        ttnn.close_device(dev)
    return close


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
    return getattr(serve.STATE["engine"], "unusable", None)


def _chunk(cid, created, delta, fin=None):
    return ("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": serve.MODEL_ID,
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}) + "\n\n").encode()


def _complete(ids, args, cid, created):
    """The whole answer as a chat.completion object. Runs in a worker thread and releases the lock."""
    timings = args["timings"]
    pieces, finish = [], "length"
    try:
        for piece, fin in serve.generate(serve.STATE["engine"], serve.STATE["tok"], ids, **args):
            pieces.append(piece)
            finish = fin or finish
    finally:
        serve.LOCK.release()
    return {"id": cid, "object": "chat.completion", "created": created, "model": serve.MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)}, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": timings["completion_tokens"],
                      "total_tokens": len(ids) + timings["completion_tokens"]},
            "timings": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in timings.items()}}


def _stream_worker(ids, args, cid, created, put, cancelled):
    """Produce the event stream's bytes with `put`. Runs in a worker thread and releases the lock. `put(None)` ends the stream."""
    gen = serve.generate(serve.STATE["engine"], serve.STATE["tok"], ids, **args)
    try:
        put(_chunk(cid, created, {"role": "assistant"}))
        finish = "length"
        for piece, fin in gen:
            if cancelled.is_set():
                sys.stderr.write("serve: the client left before the response was complete\n")
                return
            if piece:
                put(_chunk(cid, created, {"content": piece}))
            finish = fin or finish
        put(_chunk(cid, created, {}, finish) + b"data: [DONE]\n\n")
    except Exception as exc:  # noqa: BLE001  a failure on the chip must reach the log; the stream then ends without [DONE]
        sys.stderr.write(f"serve: generation failed: {type(exc).__name__}: {exc}\n")
    finally:
        gen.close()
        serve.LOCK.release()
        put(None)


async def _chat(receive, send):
    body = await _read_body(receive)
    if body is None:
        return await _reply(send, 400, {"error": {"message": "request body must be 1 byte to 1 MiB"}})
    try:
        r = serve.parse_request(body)
        ids = serve.encode_prompt(serve.STATE["tok"], serve.STATE["template"], r["messages"])
    except (serve.RequestError, ValueError, serve.jinja2.TemplateError) as exc:       # json.JSONDecodeError is a ValueError
        return await _reply(send, 400, {"error": {"message": f"bad request: {exc}"}})
    bad = _unusable()
    if bad:
        return await _reply(send, 503, {"error": {"message": f"the chip is unusable until the server is restarted: {bad}"}})
    cid, created, timings = f"chatcmpl-{int(time.time() * 1000)}", int(time.time()), {}
    args = dict(max_tokens=r["max_tokens"], stop_ids=serve.STATE["stop_ids"], stop_strings=r["stop"], temperature=r["temperature"],
                top_p=r["top_p"], seed=r["seed"], timings=timings)
    if not serve.LOCK.acquire(blocking=False):        # one request runs at a time; a second one is told so at once and never waits
        return await _reply(send, 503, {"error": {"message": "the server is busy with another request; retry shortly"}},
                            [("Retry-After", "5")])
    bad = _unusable()                                   # look again under the lock: the request that held it may have failed a recapture
    if bad:
        serve.LOCK.release()
        return await _reply(send, 503, {"error": {"message": f"the chip is unusable until the server is restarted: {bad}"}})
    loop = asyncio.get_running_loop()
    if not r["stream"]:
        try:
            out = await loop.run_in_executor(None, _complete, ids, args, cid, created)
        except serve.RequestError as exc:
            return await _reply(send, 400, {"error": {"message": str(exc)}})
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"serve: generation failed: {type(exc).__name__}: {exc}\n")
            return await _reply(send, 500, {"error": {"message": f"generation failed: {type(exc).__name__}"}})
        return await _reply(send, 200, out)
    if len(ids) >= serve.STATE["engine"].max_len:      # refuse before any header is sent, so the client gets a JSON 400
        serve.LOCK.release()
        return await _reply(send, 400, {"error": {"message": f"the prompt is {len(ids)} tokens and the context limit is "
                                                             f"{serve.STATE['engine'].max_len}"}})
    queue, cancelled = asyncio.Queue(), threading.Event()
    put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)  # noqa: E731

    async def watch_disconnect():
        while True:
            if (await receive())["type"] == "http.disconnect":
                cancelled.set()
                return

    watcher = asyncio.ensure_future(watch_disconnect())
    worker = loop.run_in_executor(None, _stream_worker, ids, args, cid, created, put, cancelled)
    try:
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        while (item := await queue.get()) is not None:
            await send({"type": "http.response.body", "body": item, "more_body": True})
        await send({"type": "http.response.body", "body": b""})
    finally:
        cancelled.set()
        watcher.cancel()
        await worker


async def _http(scope, receive, send):
    path, method = scope["path"].rstrip("/"), scope["method"]
    if method == "GET" and path == "/v1/models":
        return await _reply(send, 200, {"object": "list", "data": [{"id": serve.MODEL_ID, "object": "model", "owned_by": "ibm-granite"}]})
    if method == "GET" and path == "/health":
        bad = _unusable()
        if bad:                                         # a failed trace capture: health checks must see it so the instance is restarted
            return await _reply(send, 503, {"status": "unusable", "reason": bad, "model": serve.MODEL_ID})
        return await _reply(send, 200, {"status": "ok", "model": serve.MODEL_ID, "context": serve.STATE["engine"].max_len})
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
