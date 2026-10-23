# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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
- "slow": like "perfect", but each completion waits the config's "delay_s" first, so a check can run past its deadline.

The prompt suite (issue 15) adds two things, both off unless the config asks:
- POST /v1/chat/completions answers with the first of the config's "chat_replies" ([[substring, reply], ...], reply
  being {"content", "reasoning_content" or "reasoning", "tool_calls", "finish_reason"}) whose substring is in the last
  user message, else with "chat_default" (a string, default a short healthy sentence). "chat_error_for" is a substring
  that makes the server answer 400 to a chat request containing it. A completion whose prompt is a string (the suite in
  completion mode) is answered from "text_replies" the same way, else "text_default".
- "request_log" is a file to which every POST appends {"path", "body"} as one JSON line, so a test can see what was sent.

Like the real server, it answers 400 to a request that carries any key other than model, prompt,
max_tokens and temperature (the real one rejects logprobs and sampling parameters), and to a
request whose model is not the expected model directory: the config's "model", or else the
`--model` it was started with. GET /v1/models lists that one model, as vLLM does. Arguments it
does not know (the rest of a vLLM command line) are ignored.

On start it writes {"pid", "pgid", "env", "model_arg"} to the config's "pid_file", so the test can check the
environment it was given and, afterwards, that its process group is gone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED_KEYS = {"model", "prompt", "max_tokens", "temperature"}
CHAT_ALLOWED_KEYS = {"model", "messages", "max_tokens", "temperature", "tools"}
ENV_KEYS = ("TT_CACHE_PATH", "TT_CACHE_HOME", "HF_HOME", "HF_HUB_OFFLINE", "MODEL_WEIGHTS_DIR", "HF_MODEL")


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
        elif self.path == "/v1/models":
            self._send(200, json.dumps({"data": [{"id": self.cfg["model"]}]}).encode())
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        cfg = self.cfg
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self._send(404, b'{"error": "not found"}')
            return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if cfg.get("request_log"):
            with open(cfg["request_log"], "a", encoding="utf-8") as log:
                log.write(json.dumps({"path": self.path, "body": req}) + "\n")
        if cfg["mode"] == "http500":
            self._send(500, b"fake server failure", "text/plain")
            return
        if cfg["mode"] == "slow":
            time.sleep(float(cfg.get("delay_s", 1)))
        chat = self.path == "/v1/chat/completions"
        extra = set(req) - (CHAT_ALLOWED_KEYS if chat else ALLOWED_KEYS)
        if extra:
            self._send(400, json.dumps({"error": f"unsupported keys {sorted(extra)}"}).encode())
            return
        if req.get("model") != cfg["model"]:
            self._send(400, json.dumps({"error": f"unknown model {req.get('model')!r}"}).encode())
            return
        if chat:
            self._chat(req)
            return
        vocab, prompt_ids, gen = cfg["vocab"], cfg["prompt_ids"], cfg["generated_ids"]
        prompt = req["prompt"]
        if isinstance(prompt, str):
            text = next((r for sub, r in cfg.get("text_replies", []) if sub in prompt), cfg.get("text_default", " Paris is the capital."))
        elif req["max_tokens"] > 1:
            text = " " + " ".join(vocab[i] for i in gen[:req["max_tokens"]])
        else:
            k = len(prompt) - len(prompt_ids)
            wanted = gen[k]
            if cfg["mode"] == "every4" and k % 4 == 3:
                wanted = (wanted + 1) % len(vocab)
            text = " " + vocab[wanted]
        self._send(200, json.dumps({"choices": [{"text": text, "index": 0}]}).encode())

    def _chat(self, req):
        cfg = self.cfg
        last = req["messages"][-1]["content"]
        if cfg.get("chat_error_for") and cfg["chat_error_for"] in last:
            self._send(400, json.dumps({"error": "the chat request was refused"}).encode())
            return
        reply = next((r for sub, r in cfg.get("chat_replies", []) if sub in last),
                     cfg.get("chat_default", "That is a short and healthy reply."))
        if isinstance(reply, str):
            reply = {"content": reply}
        message = {"role": "assistant", "content": reply.get("content")}
        for key in ("reasoning_content", "reasoning"):
            if key in reply:
                message[key] = reply[key]
        if reply.get("tool_calls"):
            message["tool_calls"] = [{"id": "call_0", "type": "function", "function": c} for c in reply["tool_calls"]]
        self._send(200, json.dumps({"choices": [{"index": 0, "message": message,
                                                 "finish_reason": reply.get("finish_reason", "stop")}]}).encode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model")            # passed when the test runs a prepare_swap-built run.sh
    args, _ = ap.parse_known_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("model", args.model)
    with open(cfg["pid_file"], "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "pgid": os.getpgid(0),
                   "env": {k: os.environ.get(k) for k in ENV_KEYS}, "model_arg": args.model}, f)
    if cfg["mode"] == "die":
        for i in range(70):
            print(f"fake server log line {i}", flush=True)
        return 1
    Handler.cfg = cfg
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
