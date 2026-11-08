# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ASGI front end of the Olmo-Hybrid-7B package, for a v6 thin bundle of kind `tt-dit-server`.

`tt-model` starts that kind with `uvicorn <module>:app`. The model runs on a mesh of four Blackhole chips through `olmo_hybrid_port.py` (the port
that passed the orchard full-port stages; it is shipped next to this file in `extra_code/`). The weights stay on the Hugging Face Hub: the app
downloads the pinned revision on first start (`HF_MODEL` and `TT_MODEL_WEIGHTS_REVISION` are set by the bundle's run.sh).

    GET  /health               200 {"status": "ok", ...}
    GET  /v1/models
    POST /v1/completions       prompt (text), max_tokens, stream, stop, temperature, top_p, seed
    POST /v1/chat/completions  messages, max_tokens or max_completion_tokens, stream, stop, temperature, top_p, seed

Olmo-Hybrid-7B is a pretrained base model. It ships no chat template, so the chat route writes the messages as `role: content` lines that end with
`assistant:`; /v1/completions is the natural route. Sampling is greedy unless `temperature` is above 0. One request runs at a time: a second one
gets 503 with Retry-After at once. The chip work runs in a worker thread, so /health answers while a request generates. A streaming client that
leaves ends the generation at the next token and frees the slot. Prompt and answer together may not pass OLMO_MAX_LEN tokens.

Settings come from the environment (uvicorn passes no flags):

    OLMO_MODEL_DIR   a directory with config.json, model.safetensors and tokenizer.json (default: the pinned revision, downloaded into the cache)
    OLMO_MAX_LEN     context limit in tokens (default 16384; the model's own limit is 65536)
    OLMO_CHIPS       chips in the mesh (default 4: a 1x4 mesh with the 1-D fabric)

The device opens when the server starts (the ASGI lifespan) and closes when it stops.
"""
import asyncio
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = "allenai/Olmo-Hybrid-7B"
MAX_BODY = 1 << 20
LOCK = threading.Lock()
STATE = {}


def settings(environ=None):
    e = os.environ if environ is None else environ
    return {"model_dir": e.get("OLMO_MODEL_DIR") or None, "max_len": int(e.get("OLMO_MAX_LEN", 16384)), "chips": int(e.get("OLMO_CHIPS", 4))}


class RequestError(ValueError):
    pass


def parse_request(body: bytes, chat: bool) -> dict:
    """The checked fields of a request body. Raises RequestError with a message for the client."""
    try:
        obj = json.loads(body)
    except ValueError as exc:
        raise RequestError(f"the body is not JSON: {exc}") from None
    if not isinstance(obj, dict):
        raise RequestError("the body must be a JSON object")
    if chat:
        msgs = obj.get("messages")
        if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in msgs):
            raise RequestError("messages must be a non-empty list of objects with string content")
        prompt = "".join(f"{m.get('role', 'user')}: {m['content']}\n" for m in msgs) + "assistant:"
    else:
        prompt = obj.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RequestError("prompt must be a non-empty string")
    mt = obj.get("max_completion_tokens", obj.get("max_tokens", 128))
    if not isinstance(mt, int) or isinstance(mt, bool) or not 1 <= mt <= 65536:
        raise RequestError("max_tokens must be a whole number from 1 to 65536")
    temp = obj.get("temperature", 0)
    top_p = obj.get("top_p", 1)
    seed = obj.get("seed")
    stop = obj.get("stop") or []
    stop = [stop] if isinstance(stop, str) else stop
    if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not 0 <= temp <= 5:
        raise RequestError("temperature must be a number from 0 to 5")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise RequestError("top_p must be a number above 0 and at most 1")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise RequestError("seed must be a whole number")
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(s, str) and s for s in stop):
        raise RequestError("stop must be up to 4 non-empty strings")
    return {"prompt": prompt, "max_tokens": mt, "temperature": float(temp), "top_p": float(top_p), "seed": seed, "stop": stop,
            "stream": obj.get("stream") is True, "return_token_ids": obj.get("return_token_ids") is True}


def _load_port():
    """The port module: `olmo_hybrid_port` in a bundle's extra_code/, or runtime/olmo_hybrid_port.py in the repository."""
    for base in (HERE, HERE.parent / "runtime"):
        f = base / "olmo_hybrid_port.py"
        if f.is_file():
            spec = importlib.util.spec_from_file_location("olmo_hybrid_port", f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["olmo_hybrid_port"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("olmo_hybrid_port.py was not found next to this file or in ../runtime")


def startup(a):
    """Download the weights if needed, load the tokenizer, open the mesh and load the port. Returns a function that closes everything."""
    if not a["model_dir"]:
        from huggingface_hub import snapshot_download
        a["model_dir"] = snapshot_download(os.environ.get("HF_MODEL") or MODEL_ID, revision=os.environ.get("TT_MODEL_WEIGHTS_REVISION") or None)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a["model_dir"]) / "tokenizer.json"))
    cfg = json.loads((Path(a["model_dir"]) / "config.json").read_text(encoding="utf-8"))
    eos = cfg.get("eos_token_id")
    STATE["eos"] = {int(eos)} if isinstance(eos, int) else set(int(x) for x in (eos or []))
    STATE["tok"], STATE["max_len"] = tok, min(a["max_len"], int(cfg.get("max_position_embeddings", 65536)))
    import ttnn
    n = a["chips"]
    if n > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, n))
    else:
        dev = ttnn.open_device(device_id=0)
    port = _load_port()
    t0 = time.time()
    try:
        model = port.load(a["model_dir"], dev)
    except BaseException:
        _close_device(ttnn, dev, n)
        raise
    load_s = time.time() - t0
    STATE["port"], STATE["model"] = port, model
    print(json.dumps({"event": "ready", "model": MODEL_ID, "chips": n, "mesh": f"1x{n}", "max_len": STATE["max_len"],
                      "load_seconds": round(load_s, 1), "decoding": "greedy by default"}), flush=True)

    def close():
        try:
            port.close(model)
        finally:
            _close_device(ttnn, dev, n)
    return close


def _close_device(ttnn, dev, n):
    if n > 1:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    else:
        ttnn.close_device(dev)


def _pick(logits, temperature, top_p, gen):
    """The next token id from the port's logits: greedy at temperature 0, otherwise a sample (nucleus when top_p is below 1)."""
    import torch
    x = logits.reshape(-1).float()
    if temperature <= 0:
        return int(x.argmax())
    p = torch.softmax(x / temperature, dim=-1)
    if top_p < 1:
        sp, si = torch.sort(p, descending=True)
        keep = torch.cumsum(sp, dim=-1) - sp < top_p
        sp = sp * keep
        return int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen)])
    return int(torch.multinomial(p, 1, generator=gen))


def generate(prompt_ids, r, timings, cancelled=None):
    """Yield (text piece, finish reason or None). Fills `timings`. Stops on an end token, a stop string, the token budget or the context limit."""
    import torch
    port, model, tok = STATE["port"], STATE["model"], STATE["tok"]
    gen = torch.Generator().manual_seed(r["seed"] if r["seed"] is not None else int(time.time_ns() % (1 << 31)))
    limit = min(r["max_tokens"], STATE["max_len"] - len(prompt_ids))
    out, emitted = [], 0
    t0 = time.time()
    logits = port.prefill(model, prompt_ids)
    t1 = time.time()
    timings.update(prompt_tokens=len(prompt_ids), prefill_s=t1 - t0, completion_tokens=0)
    finish = "length"
    for step in range(limit):
        if cancelled is not None and cancelled.is_set():
            return
        nxt = _pick(logits, r["temperature"], r["top_p"], gen)
        if nxt in STATE["eos"]:
            finish = "stop"
            break
        out.append(nxt)
        timings["completion_tokens"] = len(out)
        full = tok.decode(out)
        if not full.endswith("\ufffd"):          # an unfinished multi-byte character waits for the next token
            cut = min((full.find(s) for s in r["stop"] if s in full), default=-1)
            if cut >= 0:
                if cut > emitted:
                    yield full[emitted:cut], None
                finish = "stop"
                break
            if len(full) > emitted:
                yield full[emitted:], None
                emitted = len(full)
        if step + 1 < limit:
            logits = port.decode_step(model, nxt, len(prompt_ids) + step)
    t2 = time.time()
    n = timings["completion_tokens"]
    timings["decode_s"] = t2 - t1
    timings["_ids"] = list(out)
    timings["decode_tok_s"] = (n - 1) / (t2 - t1) if n > 1 and t2 > t1 else None
    yield "", finish


async def _reply(send, code, obj, headers=()):
    body = json.dumps(obj).encode()
    await send({"type": "http.response.start", "status": code,
                "headers": [(k.lower().encode(), v.encode()) for k, v in (*headers, ("Content-Type", "application/json"),
                                                                           ("Content-Length", str(len(body))))]})
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive):
    body = b""
    while True:
        msg = await receive()
        if msg["type"] == "http.disconnect":
            return None
        body += msg.get("body", b"")
        if len(body) > MAX_BODY:
            return None
        if not msg.get("more_body"):
            return body or None


def _obj(kind, cid, created, choice, extra=None):
    return {"id": cid, "object": kind, "created": created, "model": MODEL_ID, "choices": [choice], **(extra or {})}


def _choice(chat, text, finish, delta=None):
    if chat:
        return {"index": 0, "delta": delta if delta is not None else {"content": text}, "finish_reason": finish}
    return {"index": 0, "text": text, "finish_reason": finish}


def _round(timings):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in timings.items()}


def _complete(ids, r, cid, created, chat):
    timings, pieces, finish = {}, [], "length"
    try:
        for piece, fin in generate(ids, r, timings):
            pieces.append(piece)
            finish = fin or finish
    finally:
        LOCK.release()
    text = "".join(pieces)
    ids_out = timings.pop("_ids", None)
    usage = {"prompt_tokens": len(ids), "completion_tokens": timings.get("completion_tokens", 0),
             "total_tokens": len(ids) + timings.get("completion_tokens", 0)}
    choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish} if chat
              else {"index": 0, "text": text, "finish_reason": finish})
    if r["return_token_ids"] and ids_out is not None:        # an extension for checking the port against a reference, as vLLM has one
        choice["token_ids"] = ids_out
    return _obj("chat.completion" if chat else "text_completion", cid, created, choice, {"usage": usage, "timings": _round(timings)})


def _stream_worker(ids, r, cid, created, chat, put, cancelled):
    kind = "chat.completion.chunk" if chat else "text_completion"
    sse = lambda ch: ("data: " + json.dumps(_obj(kind, cid, created, ch)) + "\n\n").encode()  # noqa: E731
    gen = generate(ids, r, {}, cancelled)
    try:
        if chat:
            put(sse(_choice(True, "", None, {"role": "assistant"})))
        finish = "length"
        for piece, fin in gen:
            if cancelled.is_set():
                sys.stderr.write("serve: the client left before the response was complete\n")
                return
            if piece:
                put(sse(_choice(chat, piece, None)))
            finish = fin or finish
        put(sse(_choice(chat, "", finish, {} if chat else None)) + b"data: [DONE]\n\n")
    except Exception as exc:  # noqa: BLE001  a failure on the chip must reach the log; the stream then ends without [DONE]
        sys.stderr.write(f"serve: generation failed: {type(exc).__name__}: {exc}\n")
    finally:
        gen.close()
        LOCK.release()
        put(None)


async def _complete_route(receive, send, chat):
    body = await _read_body(receive)
    if body is None:
        return await _reply(send, 400, {"error": {"message": "request body must be 1 byte to 1 MiB"}})
    try:
        r = parse_request(body, chat)
        ids = STATE["tok"].encode(r["prompt"]).ids
    except RequestError as exc:
        return await _reply(send, 400, {"error": {"message": f"bad request: {exc}"}})
    if len(ids) >= STATE["max_len"]:
        return await _reply(send, 400, {"error": {"message": f"the prompt is {len(ids)} tokens and the context limit is {STATE['max_len']}"}})
    cid, created = f"{'chatcmpl' if chat else 'cmpl'}-{int(time.time() * 1000)}", int(time.time())
    if not LOCK.acquire(blocking=False):
        return await _reply(send, 503, {"error": {"message": "the server is busy with another request; retry shortly"}}, [("Retry-After", "5")])
    loop = asyncio.get_running_loop()
    if not r["stream"]:
        try:
            out = await loop.run_in_executor(None, _complete, ids, r, cid, created, chat)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"serve: generation failed: {type(exc).__name__}: {exc}\n")
            return await _reply(send, 500, {"error": {"message": f"generation failed: {type(exc).__name__}"}})
        return await _reply(send, 200, out)
    queue, cancelled = asyncio.Queue(), threading.Event()
    put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)  # noqa: E731

    async def watch_disconnect():
        while True:
            if (await receive())["type"] == "http.disconnect":
                cancelled.set()
                return

    watcher = asyncio.ensure_future(watch_disconnect())
    worker = loop.run_in_executor(None, _stream_worker, ids, r, cid, created, chat, put, cancelled)
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
        return await _reply(send, 200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "allenai"}]})
    if method == "GET" and path == "/health":
        return await _reply(send, 200, {"status": "ok", "model": MODEL_ID, "context": STATE.get("max_len")})
    if method == "POST" and path == "/v1/chat/completions":
        return await _complete_route(receive, send, True)
    if method == "POST" and path == "/v1/completions":
        return await _complete_route(receive, send, False)
    await _reply(send, 404, {"error": {"message": "not found"}})


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        close = None
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                try:
                    close = await asyncio.get_running_loop().run_in_executor(None, startup, settings())
                except BaseException as exc:  # noqa: BLE001
                    sys.stderr.write(f"serve: start-up failed: {type(exc).__name__}: {exc}\n")
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
