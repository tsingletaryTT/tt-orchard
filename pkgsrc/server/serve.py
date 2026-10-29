# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""An OpenAI-compatible chat server for ibm-granite/granite-4.0-h-1b on one Blackhole chip (standard library, `tokenizers` and `jinja2`).

    python serve.py --model-dir <directory with config.json, model.safetensors, tokenizer.json, chat_template.jinja> --port 20110

Endpoints: GET /v1/models, GET /health, POST /v1/chat/completions (messages, max_tokens or max_completion_tokens, stream, stop, temperature,
top_p, seed). One request runs at a time. The server binds to 127.0.0.1 and has no sign-in.

Behaviour that a client can rely on:
  * The prompt is the model's own chat template (`chat_template.jinja` in the model directory) applied to `messages`, with the generation
    prompt added. A system message is optional (the template adds a default one when there is none).
  * Generation ends at the model's end-of-text token, at a `stop` string, at `max_tokens`, or when the context is full.
  * `finish_reason` is `stop` when the loop ended on the end-of-text token or a stop string, and `length` when it ended on `max_tokens` or
    on the context limit. It is taken from the token that ended the loop.
  * temperature is greedy (argmax) when it is absent or 0. A temperature above 0 samples on the host, with `top_p` and `seed`; sampled
    output was not evaluated. Tool calls and document inputs of the template are not supported and are refused.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jinja2
import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment

MODEL_ID = "ibm-granite/granite-4.0-h-1b"
REVISION = "d18cca4c121edb87d022116d281ce212c9136f57"       # the revision the package was measured with
MAX_BODY = 1 << 20
MAX_NEW_CAP = 4096
LOCK = threading.Lock()
STATE = {}


class RequestError(ValueError):
    """A problem with the request itself; the client gets a 400 with this message."""


# ---- prompt ---------------------------------------------------------------------------------------------------------------
def make_template(path):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise RequestError(message)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(Path(path).read_text())


def render_prompt(template, messages):
    return template.render(messages=messages, add_generation_prompt=True, tools=None, documents=None, available_tools=None)


def encode_prompt(tokenizer, template, messages):
    text = render_prompt(template, messages)
    return tokenizer.encode(text, add_special_tokens=False).ids


# ---- generation (engine independent: tests drive it with a fake engine) ---------------------------------------------------
def sample(logits, temperature, top_p, gen):
    if not temperature:
        return int(logits.argmax())
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    if top_p < 1.0:
        p, idx = torch.sort(probs, descending=True)
        keep = torch.cumsum(p, 0) - p < top_p
        p = p * keep
        probs = torch.zeros_like(probs).scatter_(0, idx, p)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=gen))


def generate(engine, tokenizer, ids, max_tokens, stop_ids, stop_strings=(), temperature=0.0, top_p=1.0, seed=None, timings=None):
    """Yield (text_piece, finish_reason). finish_reason is None except on the last item, whose piece may be ''.

    `engine.prefill(ids)` returns the logits after the prompt and `engine.step(token)` the logits after one more token."""
    room = engine.max_len - len(ids)
    if room < 1:
        raise RequestError(f"the prompt is {len(ids)} tokens and the context limit is {engine.max_len}")
    budget = min(max_tokens, room)
    gen = torch.Generator().manual_seed(seed if seed is not None else 0)
    t0 = time.time()
    engine.reset()
    logits = engine.prefill(ids)
    t_prefill = time.time() - t0
    out, shown, finish, matched = [], "", "length", False
    hold = max((len(s) for s in stop_strings), default=1) - 1       # characters kept back while a stop string could be forming
    t_first = None
    for k in range(budget):
        nxt = sample(logits, temperature, top_p, gen)
        if t_first is None:
            t_first = time.time() - t0
        if nxt in stop_ids:
            finish = "stop"
            break
        out.append(nxt)
        text = tokenizer.decode(out)
        cut = min((text.find(s) for s in stop_strings if s in text), default=-1)
        if cut >= 0:
            if len(text[:cut]) > len(shown):
                yield text[len(shown):cut], None
            shown = text[:cut]
            finish, matched = "stop", True
            break
        safe = text[: max(len(text) - hold, 0)] if hold else text
        if safe.endswith("�"):
            safe = safe.rstrip("�")                            # an incomplete UTF-8 character: wait for the next token
        if len(safe) > len(shown):
            yield safe[len(shown):], None
            shown = safe
        if k + 1 < budget:
            logits = engine.step(nxt)
    if not matched:                                                 # release what was held back for a stop string that did not form
        text = tokenizer.decode(out)
        if len(text) > len(shown):
            yield text[len(shown):], None
    if timings is not None:
        t_total = time.time() - t0
        timings.update({"prompt_tokens": len(ids), "completion_tokens": len(out), "prefill_s": t_prefill, "first_token_s": t_first,
                        "decode_tok_s": (len(out) - 1) / (t_total - t_first) if len(out) > 1 and t_total > t_first else None})
    yield "", finish


# ---- HTTP -----------------------------------------------------------------------------------------------------------------
def parse_request(body):
    req = json.loads(body)
    if not isinstance(req, dict):
        raise RequestError("the request body must be a JSON object")
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages or not all(
            isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str) for m in messages):
        raise RequestError("messages must be a non-empty list of {role, content} objects with string content")
    if req.get("tools") or req.get("documents"):
        raise RequestError("tool and document inputs are not supported by this server")
    mt = req.get("max_completion_tokens", req.get("max_tokens"))
    try:
        mt = 256 if mt is None else int(mt)
    except (TypeError, ValueError):
        raise RequestError("max_tokens must be an integer")
    if mt < 1:
        raise RequestError("max_tokens must be at least 1")
    stop = req.get("stop") or []
    stop = [stop] if isinstance(stop, str) else stop
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(s, str) and s for s in stop):
        raise RequestError("stop must be a string or a list of up to 4 non-empty strings")
    try:
        temperature = float(req.get("temperature") or 0.0)
        top_p = float(req.get("top_p") if req.get("top_p") is not None else 1.0)
        seed = None if req.get("seed") is None else int(req["seed"])
    except (TypeError, ValueError):
        raise RequestError("temperature, top_p and seed must be numbers")
    if temperature < 0 or not 0 < top_p <= 1:
        raise RequestError("temperature must be 0 or more and top_p in (0, 1]")
    return {"messages": messages, "max_tokens": min(mt, MAX_NEW_CAP), "stop": stop, "stream": bool(req.get("stream")),
            "temperature": temperature, "top_p": top_p, "seed": seed}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "ibm-granite"}]})
        elif path == "/health":
            self._json(200, {"status": "ok", "model": MODEL_ID, "context": STATE["engine"].max_len})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= MAX_BODY:
                return self._json(400, {"error": {"message": "request body must be 1 byte to 1 MiB"}})
            r = parse_request(self.rfile.read(n))
            ids = encode_prompt(STATE["tok"], STATE["template"], r["messages"])
        except (RequestError, ValueError, json.JSONDecodeError, jinja2.TemplateError) as exc:
            return self._json(400, {"error": {"message": f"bad request: {exc}"}})
        cid, created, timings = f"chatcmpl-{int(time.time() * 1000)}", int(time.time()), {}
        args = dict(max_tokens=r["max_tokens"], stop_ids=STATE["stop_ids"], stop_strings=r["stop"], temperature=r["temperature"],
                    top_p=r["top_p"], seed=r["seed"], timings=timings)
        self._started = False                                 # set once response headers have gone out
        with LOCK:
            try:
                if r["stream"]:
                    self._stream(cid, created, ids, args)
                else:
                    self._started = False
                    pieces, finish = [], "length"
                    for piece, fin in generate(STATE["engine"], STATE["tok"], ids, **args):
                        pieces.append(piece)
                        finish = fin or finish
                    self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)}, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": len(ids), "completion_tokens": timings["completion_tokens"],
                                               "total_tokens": len(ids) + timings["completion_tokens"]},
                                     "timings": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in timings.items()}})
            except RequestError as exc:
                if not self._started:
                    self._json(400, {"error": {"message": str(exc)}})
            except Exception as exc:  # noqa: BLE001  a failure on the chip must reach the client and the log
                sys.stderr.write(f"serve: generation failed: {type(exc).__name__}: {exc}\n")
                if not self._started:
                    self._json(500, {"error": {"message": f"generation failed: {type(exc).__name__}"}})

    def _stream(self, cid, created, ids, args):
        chunk = lambda delta, fin=None: ("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,  # noqa: E731
                                                                 "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}) + "\n\n").encode()
        if len(ids) >= STATE["engine"].max_len:             # refuse before any header is sent, so the client gets a JSON 400
            raise RequestError(f"the prompt is {len(ids)} tokens and the context limit is {STATE['engine'].max_len}")
        gen = generate(STATE["engine"], STATE["tok"], ids, **args)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._started = True
        self.wfile.write(chunk({"role": "assistant"}))
        self.wfile.flush()
        finish = "length"
        for piece, fin in gen:                              # lazily: each piece is written and flushed as it is produced
            if piece:
                self.wfile.write(chunk({"content": piece}))
                self.wfile.flush()
            finish = fin or finish
        self.wfile.write(chunk({}, finish) + b"data: [DONE]\n\n")

    def log_message(self, fmt, *args):
        sys.stderr.write("serve: " + fmt % args + "\n")


class ChipEngine:
    """Adapts GraniteChip to the engine interface used by `generate`."""

    def __init__(self, model):
        self.m, self.max_len = model, model.L

    def reset(self):
        self.m.reset()

    def prefill(self, ids):
        return self.m.prefill(ids)

    def step(self, tok):
        return self.m.step(tok)


def load_tokenizer(model_dir):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))


def stop_token_ids(model_dir, tokenizer):
    cfg = json.load(open(Path(model_dir) / "config.json"))
    eos = cfg["eos_token_id"]
    return {eos} if isinstance(eos, int) else set(eos)


def ready_event(model, a, load_s):
    """The ready line. It reports what the constructed model does (`model.f32`), not what was requested."""
    return {"event": "ready", "model": MODEL_ID, "host": a.host, "port": a.port, "context": model.L, "ssd": a.ssd,
            "traced": not a.no_trace, "ssd_precision": "fp32" if model.f32 else "bf16", "load_s": round(load_s, 1), "pid": os.getpid()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("GRANITE_MODEL_DIR"),
                    help="directory with config.json, model.safetensors, tokenizer.json and chat_template.jinja; default: the pinned revision from the Hugging Face cache")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 20110)))
    ap.add_argument("--max-len", type=int, default=int(os.environ.get("GRANITE_MAX_LEN", 2048)))
    ap.add_argument("--ssd", default=os.environ.get("GRANITE_SSD", "composed"))
    ap.add_argument("--ssd-precision", default=os.environ.get("GRANITE_SSD_PRECISION", "fp32"), choices=["fp32", "bf16"])
    ap.add_argument("--no-trace", action="store_true", help="run the step eagerly (slower); for comparison only")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.port in (8000, 8011, 8012, 8780) or 8100 <= a.port <= 8103:
        sys.exit("refused: that port belongs to other work on this machine")
    if a.host != "127.0.0.1":
        sys.exit("refused: this server has no sign-in and binds 127.0.0.1 only")
    if not a.model_dir:
        from huggingface_hub import snapshot_download
        a.model_dir = snapshot_download(MODEL_ID, revision=REVISION)        # downloads the weights into your Hugging Face cache on the first run
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime"))
    import ttnn
    from granite_chip import GraniteChip
    STATE["tok"] = load_tokenizer(a.model_dir)
    STATE["template"] = make_template(Path(a.model_dir) / "chat_template.jinja")
    STATE["stop_ids"] = stop_token_ids(a.model_dir, STATE["tok"])
    dev = ttnn.open_device(device_id=0, trace_region_size=400 * 1024 * 1024)
    model = None
    try:
        t0 = time.time()
        model = GraniteChip(dev, a.model_dir, max_len=a.max_len, ssd=a.ssd, kernel_dir=Path(__file__).resolve().parent.parent / "runtime" / "kernels",
                            trace=not a.no_trace, ssd_precision=a.ssd_precision)
        STATE["engine"] = ChipEngine(model)
        srv = ThreadingHTTPServer((a.host, a.port), Handler)
        print(json.dumps(ready_event(model, a, time.time() - t0)), flush=True)
        srv.serve_forever()
    finally:
        if model is not None:
            model.close()
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
