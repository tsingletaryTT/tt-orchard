# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A small OpenAI-shaped HTTP server that opens no device.

orchard/park_check.py runs two of these, one standing in for the coder and one for the CPU
stand-in, so the park sequence can run on a real board without loading a model. Tests run it too.
It answers GET /health, GET /v1/models and POST /v1/chat/completions, always with the same text.
It imports nothing from orchard, so it can run as a plain script.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    answer = ""
    model = ""

    def log_message(self, *args):        # keep test output quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": self.model}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        json.loads(self.rfile.read(n) or b"{}")          # a malformed body fails loudly
        self._send(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": self.answer}}]})


def make_server(port: int, answer: str, model: str, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"answer": answer, "model": model})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fake_server", description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--answer", required=True)
    p.add_argument("--model", default="fake")
    a = p.parse_args(argv)
    srv = make_server(a.port, a.answer, a.model)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
