"""The Granite package's ASGI front end (pkgsrc/server/granite_asgi.py), driven without a server process or a chip.

A small ASGI driver calls `app` with a scope, a body and a `receive`/`send` pair. The engine is a fake that returns scripted logits, so
these tests check the HTTP behaviour (routes, status codes, streaming, busy and unusable answers, client disconnect) and not the model."""
import asyncio
import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("granite_asgi_under_test", ROOT / "pkgsrc" / "server" / "granite_asgi.py")
# serve.py is next to the file under test, which is where `_core()` finds it when `granite_serve` is not on the path
asgi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asgi)
serve = asgi.serve
EOS = 15


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[1, 2, 3])

    def decode(self, ids):
        return "".join(chr(97 + i) for i in ids)


class FakeEngine:
    max_len = 64
    unusable = None

    def __init__(self, script):
        self.script, self.i = list(script), 0

    def reset(self):
        self.i = 0

    def _logits(self):
        v = torch.zeros(16)
        v[self.script[min(self.i, len(self.script) - 1)]] = 1.0
        self.i += 1
        return v

    def prefill(self, ids):
        return self._logits()

    def step(self, tok):
        return self._logits()


@pytest.fixture(autouse=True)
def state(tmp_path):
    t = tmp_path / "t.jinja"
    t.write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
    serve.STATE.update(tok=FakeTok(), template=serve.make_template(t), stop_ids={EOS}, engine=FakeEngine([0, 1, 2, EOS]))
    yield
    if serve.LOCK.locked():
        serve.LOCK.release()


async def call(method, path, body=b"", disconnect_after=None):
    """Returns (status, headers, body bytes). `disconnect_after`: send http.disconnect once this many body chunks have been sent."""
    sent, parts = [], []
    inbox = [{"type": "http.request", "body": body, "more_body": False}]
    gate = asyncio.Event()

    async def receive():
        if inbox:
            return inbox.pop(0)
        await gate.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)
        if msg["type"] == "http.response.body":
            parts.append(msg.get("body", b""))
            if disconnect_after is not None and len(parts) >= disconnect_after:
                gate.set()

    await asyncio.wait_for(asgi.app({"type": "http", "method": method, "path": path}, receive, send), 20)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}, b"".join(parts)


def post(payload, **kw):
    return call("POST", "/v1/chat/completions", json.dumps(payload).encode() if not isinstance(payload, bytes) else payload, **kw)


def run(coro):
    return asyncio.run(coro)


MSG = {"messages": [{"role": "user", "content": "hi"}]}


def test_health_models_and_unknown_path():
    status, _, body = run(call("GET", "/health"))
    assert status == 200 and json.loads(body) == {"status": "ok", "model": serve.MODEL_ID, "context": 64}
    status, _, body = run(call("GET", "/v1/models"))
    assert status == 200 and json.loads(body)["data"][0]["id"] == serve.MODEL_ID
    assert run(call("GET", "/nope"))[0] == 404
    assert run(call("GET", "/v1/chat/completions"))[0] == 404


def test_chat_completion_gives_text_usage_and_finish_stop():
    status, headers, body = run(post(MSG))
    out = json.loads(body)
    assert status == 200 and headers["content-type"] == "application/json"
    assert out["choices"][0] == {"index": 0, "message": {"role": "assistant", "content": "abc"}, "finish_reason": "stop"}
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
    assert not serve.LOCK.locked()


def test_max_tokens_gives_finish_length():
    out = json.loads(run(post({**MSG, "max_tokens": 2}))[2])
    assert out["choices"][0]["message"]["content"] == "ab" and out["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("payload", [b"not json", b"[]", {"messages": []}, {**MSG, "tools": [{"x": 1}]}, {**MSG, "max_tokens": 0},
                                     {**MSG, "stop": ["a"] * 5}, {**MSG, "top_p": 0}])
def test_bad_requests_get_a_json_400_and_free_the_slot(payload):
    status, _, body = run(post(payload))
    assert status == 400 and "error" in json.loads(body)
    assert not serve.LOCK.locked()


def test_empty_and_oversized_bodies_are_refused():
    assert run(post(b""))[0] == 400
    assert run(post(b"x" * (serve.MAX_BODY + 1)))[0] == 400


def test_streaming_sends_a_role_chunk_content_chunks_a_finish_chunk_and_done():
    status, headers, body = run(post({**MSG, "stream": True}))
    assert status == 200 and headers["content-type"] == "text/event-stream"
    events = [e[len("data: "):] for e in body.decode().split("\n\n") if e]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "abc"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert not serve.LOCK.locked()


def test_oversized_prompt_with_stream_true_gets_a_json_400_before_any_event():
    serve.STATE["engine"].max_len = 3
    status, headers, body = run(post({**MSG, "stream": True}))
    assert status == 400 and headers["content-type"] == "application/json" and b"context limit" in body
    assert not serve.LOCK.locked()


def test_a_second_request_while_one_runs_gets_503_with_retry_after():
    assert serve.LOCK.acquire(blocking=False)          # stands in for the running request
    status, headers, _ = run(post(MSG))
    assert status == 503 and headers["retry-after"] == "5"


def test_unusable_chip_gives_503_on_health_and_on_chat():
    serve.STATE["engine"].unusable = "trace capture failed"
    status, _, body = run(call("GET", "/health"))
    assert status == 503 and json.loads(body)["status"] == "unusable"
    status, _, body = run(post(MSG))
    assert status == 503 and b"unusable" in body
    assert not serve.LOCK.locked()


def test_a_failure_on_the_chip_gives_500_and_frees_the_slot():
    class Boom(FakeEngine):
        def prefill(self, ids):
            raise RuntimeError("device error")
    serve.STATE["engine"] = Boom([0])
    status, _, body = run(post(MSG))
    assert status == 500 and b"RuntimeError" in body
    assert not serve.LOCK.locked()


def test_client_disconnect_mid_stream_frees_the_slot():
    class SlowStep(FakeEngine):
        def step(self, tok):
            time.sleep(0.03)            # gives the event loop time to deliver the disconnect, even on a busy host
            return super().step(tok)
    engine = SlowStep([0] * 50)
    serve.STATE["engine"] = engine
    status, _, _ = run(post({**MSG, "stream": True, "max_tokens": 40}, disconnect_after=3))
    assert status == 200
    assert not serve.LOCK.locked()
    assert engine.i < 20          # the generation stopped soon after the client left; it did not run its 40 tokens


def test_health_answers_while_a_request_generates():
    """The chip work is in a worker thread, so the event loop is free."""
    release, entered = threading.Event(), threading.Event()

    class Slow(FakeEngine):
        def prefill(self, ids):
            entered.set()
            release.wait(10)
            return super().prefill(ids)
    serve.STATE["engine"] = Slow([0, EOS])

    async def both():
        chat = asyncio.ensure_future(post(MSG))
        await asyncio.get_running_loop().run_in_executor(None, entered.wait, 10)
        health = await call("GET", "/health")
        release.set()
        return health, await chat
    health, chat = run(both())
    assert health[0] == 200 and chat[0] == 200


def test_settings_come_from_the_environment():
    a = asgi.settings({"GRANITE_MAX_LEN": "512", "GRANITE_NO_TRACE": "1", "GRANITE_CHUNKED_PREFILL": "0", "GRANITE_SSD_PRECISION": "bf16"})
    assert (a.max_len, a.no_trace, a.no_chunked_prefill, a.ssd_precision, a.ssd) == (512, True, True, "bf16", "composed")
    d = asgi.settings({})
    assert (d.max_len, d.no_trace, d.no_chunked_prefill, d.ssd_precision) == (2048, False, False, "fp32")
