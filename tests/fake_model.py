"""A deterministic OpenAI-shaped model server for tests. No model, no network beyond 127.0.0.1.

`FakeModel(script)` serves GET /v1/models and POST /v1/chat/completions. For each chat request
it calls `script(request)`, which returns an assistant message dict, or an int to answer with
that HTTP status. The answer depends only on the request, so a restarted supervisor that sends
the same conversation gets the same answer, as from a greedy server.

An answer dict may carry two extra keys that are not part of the message: `finish_reason`
(default "stop") and `completion_tokens` (default 20). `truncated()` builds a reply that ran
into max_tokens. `empty()` builds a reply with no text and no tool calls that ended normally.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def call(name: str, call_id: str = "c1", **arguments) -> dict:
    """An assistant message that asks for one tool call."""
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def final(text: str = "done") -> dict:
    return {"role": "assistant", "content": text}


def truncated(tokens: int = 8192, **extra) -> dict:
    """A reply cut off at max_tokens: empty content, no tool calls, finish_reason "length"."""
    return {"role": "assistant", "content": "", "finish_reason": "length",
            "completion_tokens": tokens, **extra}


def empty(tokens: int = 159, content="", **extra) -> dict:
    """A reply with no text and no tool calls that still ended with finish_reason "stop". The live
    Qwen3.8 run returned one: 159 completion tokens, all reasoning, and content ""."""
    return {"role": "assistant", "content": content, "finish_reason": "stop",
            "completion_tokens": tokens, **extra}


def turn(request: dict) -> int:
    """How many assistant messages the conversation already holds: 0 on the first request."""
    return sum(1 for m in request["messages"] if m.get("role") == "assistant")


class FakeModel:
    def __init__(self, script, models=("fake",)):
        self.script, self.models = script, list(models)
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, {"data": [{"id": m} for m in owner.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(n))
                owner.requests.append(request)
                answer = owner.script(request)
                if isinstance(answer, int):
                    self._send(answer, {"error": "scripted failure"})
                    return
                answer = dict(answer)
                reason = answer.pop("finish_reason", "stop")
                tokens = answer.pop("completion_tokens", 20)
                self._send(200, {"choices": [{"index": 0, "message": answer,
                                              "finish_reason": reason}],
                                 "usage": {"prompt_tokens": 100 + 10 * len(request["messages"]),
                                           "completion_tokens": tokens}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        # A short poll interval keeps shutdown (and so each test) fast.
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
