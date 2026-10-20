# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A recording proxy between a benchmark client and the model server.

lm-evaluation-harness cannot send `chat_template_kwargs` (the way thinking is turned off), and it keeps no record of tokens per item or of
replies cut off at the length limit. This proxy sits between the harness and the server on the same machine. It adds the fields a run asks
for (`inject`) to each JSON request body, forwards everything else unchanged, and records one line per completion request: status, token
counts, finish reason, the length of any reasoning text, and how long the server took.

It listens on 127.0.0.1 only and forwards only to a loopback server.
"""
from __future__ import annotations

import http.client
import json
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

COMPLETION_PATHS = ("/v1/chat/completions", "/v1/completions")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def apply_injection(body, inject: dict):
    """The request body with the injected fields applied. A dict value is merged into a dict the client sent, with the injection winning."""
    if not inject or not isinstance(body, dict):
        return body
    out = dict(body)
    for key, value in inject.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _record_from(path: str, status: int, payload, latency: float) -> dict:
    rec = {"t": round(time.time(), 3), "path": path, "status": status, "error": status >= 400, "latency_s": round(latency, 3),
           "prompt_tokens": None, "completion_tokens": None, "finish_reason": None, "reasoning_chars": 0}
    if isinstance(payload, dict):
        usage = payload.get("usage") or {}
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        choice = (payload.get("choices") or [{}])[0] or {}
        rec["finish_reason"] = choice.get("finish_reason")
        message = choice.get("message") or {}
        rec["reasoning_chars"] = len(message.get("reasoning_content") or message.get("reasoning") or "")
    return rec


class Proxy:
    def __init__(self, upstream: str, *, inject: dict, log_path, timeout: float = 3600.0):
        parts = urlsplit(upstream)
        if parts.hostname not in LOOPBACK:
            raise ValueError(f"the proxy forwards only to a loopback server, not {parts.hostname}")
        self.host, self.port_up = parts.hostname, parts.port or 80
        self.inject, self.timeout = inject, timeout
        self.log_path = Path(log_path)
        self.records: list[dict] = []
        self._lock = threading.Lock()
        self.server = None
        self.port = None
        self._thread = None

    # -- the forwarding -----------------------------------------------------------------------------------------------------------

    def _forward(self, method: str, path: str, data):
        conn = http.client.HTTPConnection(self.host, self.port_up, timeout=self.timeout)
        try:
            conn.request(method, path, data, {"Content-Type": "application/json"} if data is not None else {})
            resp = conn.getresponse()
            return resp.status, resp.read(), resp.getheader("Content-Type") or "application/json"
        finally:
            conn.close()

    def _record(self, rec: dict) -> None:
        with self._lock:
            self.records.append(rec)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def start(self) -> int:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status: int, raw: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                try:
                    status, raw, ctype = proxy._forward("GET", self.path, None)
                except OSError as exc:
                    status, raw, ctype = 502, json.dumps({"error": {"message": f"upstream unreachable: {exc}"}}).encode(), "application/json"
                self._reply(status, raw, ctype)

            def do_POST(self):
                data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    body = json.loads(data)
                except ValueError:
                    body = None
                if body is not None:
                    data = json.dumps(apply_injection(body, proxy.inject)).encode()
                sent = time.monotonic()
                try:
                    status, raw, ctype = proxy._forward("POST", self.path, data)
                except OSError as exc:
                    status, raw, ctype = 502, json.dumps({"error": {"message": f"upstream unreachable: {exc}"}}).encode(), "application/json"
                if self.path in COMPLETION_PATHS:
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        payload = None
                    proxy._record(_record_from(self.path, status, payload, time.monotonic() - sent))
                self._reply(status, raw, ctype)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def summary(self) -> dict:
        with self._lock:
            recs = list(self.records)
        ok = [r for r in recs if not r["error"]]
        comp = [r["completion_tokens"] for r in ok if isinstance(r["completion_tokens"], int)]
        prompt = [r["prompt_tokens"] for r in ok if isinstance(r["prompt_tokens"], int)]
        lat = [r["latency_s"] for r in recs]
        return {"requests": len(recs), "errors": sum(1 for r in recs if r["error"]),
                "cut_off": sum(1 for r in recs if r["finish_reason"] == "length"),
                "completion_tokens_total": sum(comp), "completion_tokens_median": statistics.median(comp) if comp else None,
                "prompt_tokens_total": sum(prompt), "latency_s_median": statistics.median(lat) if lat else None,
                "latency_s_max": max(lat) if lat else None}
