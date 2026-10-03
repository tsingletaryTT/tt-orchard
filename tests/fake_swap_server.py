"""A stand-in for the bundle's vLLM server, for tests/test_weights_swap_templates.py.

It opens no device and loads no model. The test writes a JSON config and a run.sh that execs this
file with `--config <path>`; serve_and_compare.py then adds `--port <port>`. It answers:

- GET /health with 200.
- POST /v1/completions with words from a small vocabulary. A free run (max_tokens > 1) gets the
  whole reference continuation. A teacher-forced request (max_tokens 1) gets the reference word
  at position k = len(prompt) - len(prompt_ids), with a leading space as a real tokenizer's
  decode gives.

Modes (config key "mode"):
- "perfect": every teacher-forced answer is the reference word.
- "every4": positions 3, 7, 11, ... answer the next vocabulary word, so 8 of 32 are wrong.
- "http500": /v1/completions answers 500 with a plain body.
- "die": print 70 log lines and exit 1 before serving anything.

Like the real server, it answers 400 to a request that carries any key other than model, prompt,
max_tokens and temperature (the real one rejects logprobs and sampling parameters), and to a
request whose model is not the expected model directory.

On start it writes {"pid", "pgid", "env"} to the config's "pid_file", so the test can check the
environment it was given and, afterwards, that its process group is gone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED_KEYS = {"model", "prompt", "max_tokens", "temperature"}
ENV_KEYS = ("TT_CACHE_PATH", "TT_CACHE_HOME", "HF_HOME", "HF_HUB_OFFLINE")


class Handler(BaseHTTPRequestHandler):
    cfg: dict = {}

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"")
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        cfg = self.cfg
        if self.path != "/v1/completions":
            self._send(404, b'{"error": "not found"}')
            return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if cfg["mode"] == "http500":
            self._send(500, b"fake server failure", "text/plain")
            return
        extra = set(req) - ALLOWED_KEYS
        if extra:
            self._send(400, json.dumps({"error": f"unsupported keys {sorted(extra)}"}).encode())
            return
        if req.get("model") != cfg["model"]:
            self._send(400, json.dumps({"error": f"unknown model {req.get('model')!r}"}).encode())
            return
        vocab, prompt_ids, gen = cfg["vocab"], cfg["prompt_ids"], cfg["generated_ids"]
        prompt = req["prompt"]
        if req["max_tokens"] > 1:
            text = " " + " ".join(vocab[i] for i in gen[:req["max_tokens"]])
        else:
            k = len(prompt) - len(prompt_ids)
            wanted = gen[k]
            if cfg["mode"] == "every4" and k % 4 == 3:
                wanted = (wanted + 1) % len(vocab)
            text = " " + vocab[wanted]
        self._send(200, json.dumps({"choices": [{"text": text, "index": 0}]}).encode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(cfg["pid_file"], "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "pgid": os.getpgid(0),
                   "env": {k: os.environ.get(k) for k in ENV_KEYS}}, f)
    if cfg["mode"] == "die":
        for i in range(70):
            print(f"fake server log line {i}", flush=True)
        return 1
    Handler.cfg = cfg
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
