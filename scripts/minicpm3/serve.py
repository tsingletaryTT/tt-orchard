# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A small OpenAI-compatible chat server for MiniCPM3-4B on the chip (standard library only). Run it under a gozer lease and stop it when you are done; it holds the
chip while it runs.

    MINICPM3_DIR=... MINICPM3_WEIGHTS=... python serve.py --port 8210 [--tp 2]

Endpoints: GET /v1/models, POST /v1/chat/completions (messages, max_tokens, stream). One request runs at a time (a lock); sampling is greedy only (temperature and top_p
are accepted and ignored: the traced step returns logits to the host, and a sampler on the host is a few lines, but greedy is what was checked). The prompt is processed one token
per step (about 30 ms each; `--batched-prefill` is off by default, see its help), then tokens stream at decode speed. The server binds to 127.0.0.1 only. Do not use ports 8000, 8100 to 8103, 8780 or 8011.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

EOS = (2, 73440)
MODEL_ID = "openbmb/MiniCPM3-4B"
MAX_BODY = 1 << 20
LOCK = threading.Lock()
STATE = {}


def parse_request(raw):
    """(messages, max_tokens, stream) from a request body. Raises ValueError with a message for the client when the body is not a valid chat request."""
    if not 0 < len(raw) <= MAX_BODY:
        raise ValueError("request body must be 1 byte to 1 MiB")
    try:
        req = json.loads(raw)
        if not isinstance(req, dict):
            raise ValueError("the body must be a JSON object")
        messages = req["messages"]
        if not isinstance(messages, list) or not messages or not all(isinstance(x, dict) and isinstance(x.get("content"), str) for x in messages):
            raise ValueError("messages must be a non-empty list of {role, content} with string content")
        max_tokens = min(max(int(req.get("max_tokens") or 256), 1), 4096)
    except (KeyError, TypeError) as exc:                  # a ValueError (also json.JSONDecodeError) already has a message for the client
        raise ValueError(f"bad request: {type(exc).__name__}: {exc}") from exc
    return messages, max_tokens, bool(req.get("stream"))


def encode_prompt(messages):
    """The prompt's token ids. Raises ValueError when the prompt leaves no room in the server's context."""
    ids = STATE["tok"].apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = (ids if hasattr(ids, "shape") else ids["input_ids"])[0].tolist()
    if STATE["max_len"] - len(ids) - 1 < 1:
        raise ValueError(f"the prompt is {len(ids)} tokens; the server's context is {STATE['max_len']}")
    return ids


def generate(messages, max_tokens):
    """Yield decoded text pieces for a chat (`encode_prompt`, then `generate_ids`)."""
    yield from generate_ids(encode_prompt(messages), max_tokens)


def build_single_chip(dev, model_dir, weights, max_len, mlp_dtype, mlp_fidelity):
    """The one-chip model with its decode step traced: the configuration the stages measured (absorbed attention, paged cache, opt 3)."""
    from chip_model import ChipMiniCPM3
    model = ChipMiniCPM3(dev, model_dir, weights, attn="mla", max_len=max_len, upd="paged", opt=3, mlp_dtype=mlp_dtype, mlp_fidelity=mlp_fidelity)
    model.enable_trace()
    return model


def generate_ids(ids, max_tokens, should_stop=None):
    """Yield decoded text pieces for a prompt; positions restart at 0 (rows past the position are masked by the attention op). `should_stop`, when given, is called
    before every chip step (prompt and answer). When it returns true the generator ends at once and sets neither `finish` nor `last_counts`."""
    tok, m = STATE["tok"], STATE["model"]
    stop = should_stop or (lambda: False)
    room = STATE["max_len"] - len(ids) - 1
    if STATE.get("batched_prefill") and hasattr(m, "prefill"):
        logits = m.prefill(ids)                       # all prompt rows at once; off by default (see --batched-prefill)
    else:
        logits = None
        for p, t in enumerate(ids):                   # one traced step per prompt token (about 30 ms each)
            if stop():
                return
            logits = m.traced_step(t, p)
    out, shown = [], ""
    finish = "length"                                 # the loop ends on the token budget unless an end-of-sequence token ends it first
    for k in range(min(max_tokens, room)):
        nxt = int(logits.argmax())
        if nxt in EOS:
            finish = "stop"
            break
        out.append(nxt)
        text = tok.decode(out)
        piece, shown = text[len(shown):], text
        if piece and not piece.endswith("�"):
            yield piece
        else:
            shown = text[: len(shown) - len(piece)]
        if stop():
            return
        logits = m.traced_step(nxt, len(ids) + k)
    STATE["last_counts"] = (len(ids), len(out))
    STATE["finish"] = finish


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "openbmb"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= 1 << 20:
                return self._json(400, {"error": {"message": "request body must be 1 byte to 1 MiB"}})
            messages, max_tokens, stream = parse_request(self.rfile.read(n))
        except ValueError as exc:
            return self._json(400, {"error": {"message": str(exc)}})
        cid = f"chatcmpl-{int(time.time() * 1000)}"
        with LOCK:
            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunk = lambda delta, fin=None: ("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "model": MODEL_ID, "choices": [  # noqa: E731
                        {"index": 0, "delta": delta, "finish_reason": fin}]}) + "\n\n").encode()
                    self.wfile.write(chunk({"role": "assistant"}))
                    for piece in generate(messages, max_tokens):
                        self.wfile.write(chunk({"content": piece}))
                        self.wfile.flush()
                    self.wfile.write(chunk({}, STATE.get("finish", "stop")) + b"data: [DONE]\n\n")
                else:
                    text = "".join(generate(messages, max_tokens))
                    pt, ct = STATE["last_counts"]
                    self._json(200, {"id": cid, "object": "chat.completion", "model": MODEL_ID,
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": STATE.get("finish", "stop")}],
                                     "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})
            except ValueError as exc:
                if not stream:
                    self._json(400, {"error": {"message": str(exc)}})
            except Exception as exc:  # noqa: BLE001  a failure on the chip or in the tokenizer must reach the client and the log
                sys.stderr.write(f"serve: generation failed: {type(exc).__name__}: {exc}\n")
                if not stream:
                    self._json(500, {"error": {"message": f"generation failed: {type(exc).__name__}"}})

    def log_message(self, fmt, *args):
        sys.stderr.write("serve: " + fmt % args + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8210)
    ap.add_argument("--tp", type=int, default=1, choices=[1, 2])
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batched-prefill", action="store_true", help="process the prompt with the model's batched prefill. Off by default: it allocates device buffers while the decode "
                    "trace is live, and on 2026-10-29 the second request after it hung the chip (MMIO timeout). The benchmark uses it before any trace exists")
    ap.add_argument("--mlp-dtype", default="bf16", choices=["bf16", "bfp8", "bfp4"], help="MLP weight dtype (single-chip model); bfp8 with --mlp-fidelity lofi is the faster variant")
    ap.add_argument("--mlp-fidelity", default="hifi2", choices=["hifi2", "lofi", "hifi4"])
    ap.add_argument("--model-dir", default=os.environ.get("MINICPM3_DIR"), required="MINICPM3_DIR" not in os.environ)
    ap.add_argument("--weights", default=os.environ.get("MINICPM3_WEIGHTS"), required="MINICPM3_WEIGHTS" not in os.environ)
    a = ap.parse_args()
    if a.port in (8000, 8011, 8780) or 8100 <= a.port <= 8103:
        sys.exit("refused: that port belongs to other work on this machine")
    if a.tp == 2:
        from chip_model_tp import refuse_unsupported_tp_args
        refuse_unsupported_tp_args(ap, a)                        # before any device is opened or the fabric is configured
    import ttnn                                                  # here and not at the top: the ASGI front end imports this file and must not load ttnn
    from transformers import AutoTokenizer
    STATE["tok"] = AutoTokenizer.from_pretrained(a.model_dir)
    STATE["max_len"] = -(-a.max_len // 128) * 128
    STATE["batched_prefill"] = a.batched_prefill
    if a.tp == 2:
        from chip_model_tp import ChipMiniCPM3TP
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=256 * 1024 * 1024)
    else:
        dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
    try:
        if a.tp == 2:
            STATE["model"] = ChipMiniCPM3TP(dev, a.model_dir, a.weights, max_len=a.max_len, mlp_dtype=a.mlp_dtype, mlp_fidelity=a.mlp_fidelity)
            STATE["model"].enable_trace()
        else:
            STATE["model"] = build_single_chip(dev, a.model_dir, a.weights, a.max_len, a.mlp_dtype, a.mlp_fidelity)
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
        print(f"serving {MODEL_ID} on 127.0.0.1:{a.port} (pid {os.getpid()})", flush=True)
        srv.serve_forever()
    finally:
        if a.tp == 2:
            ttnn.close_mesh_device(dev)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        else:
            ttnn.close_device(dev)


if __name__ == "__main__":
    main()
