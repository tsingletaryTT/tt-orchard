#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A logging pass-through proxy for one diagnostic: what does OpenCode send to the model server?

    python3 oc_request_proxy.py LISTEN_PORT UPSTREAM_PORT LOGFILE

Forwards every request to 127.0.0.1:UPSTREAM_PORT unchanged and streams the reply back. For each POST that has a JSON body it appends one
line to LOGFILE: the path, every top-level field except `messages` and `tools`, the number of messages and tools, and the character
count of the first (system) message. Message text is not written. Not part of the harness.
"""
import http.client
import http.server
import json
import sys
import time

LISTEN, UPSTREAM, LOG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _forward(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        if self.command == "POST" and body:
            try:
                d = json.loads(body)
                rec = {k: v for k, v in d.items() if k not in ("messages", "tools")}
                msgs = d.get("messages") or []
                rec.update({"_path": self.path, "_t": time.strftime("%FT%TZ", time.gmtime()), "_messages": len(msgs),
                            "_tools": len(d.get("tools") or []), "_tool_names": [t.get("function", {}).get("name") for t in d.get("tools") or []],
                            "_first_message_chars": len(json.dumps(msgs[0])) if msgs else 0, "_body_bytes": len(body)})
                with open(LOG, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except ValueError:
                pass
        conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=1800)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        conn.request(self.command, self.path, body=body, headers=headers)
        r = conn.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            chunk = r.read1(8192) if hasattr(r, "read1") else r.read(8192)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        conn.close()

    do_GET = do_POST = _forward


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler).serve_forever()
