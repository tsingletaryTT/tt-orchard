# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The recording proxy between a benchmark client and the model server (orchard/bench_proxy.py).

lm-evaluation-harness cannot send `chat_template_kwargs`, which is how thinking is turned off, and it keeps no record of tokens per item or of
replies cut off at the length limit. The proxy adds the fields the run asks for to each request body, passes everything else through
unchanged, and records one line per request.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from orchard import bench_proxy as bp


class Upstream:
    def __init__(self, answer=None):
        self.seen = []
        self.answer = answer or (lambda path, body: (200, {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                                                           "usage": {"prompt_tokens": 7, "completion_tokens": 3}}))
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, payload):
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = raw
                outer.seen.append({"path": self.path, "body": body})
                self._send(*outer.answer(self.path, body))

            def do_GET(self):
                outer.seen.append({"path": self.path, "body": None})
                self._send(200, {"data": [{"id": "served-model"}]})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def post(port, path, body, raw=False):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = body if raw else json.dumps(body)
    conn.request("POST", path, data, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = (resp.status, resp.read())
    conn.close()
    return out


@pytest.fixture
def rig(tmp_path):
    made = []

    def make(inject=None, answer=None):
        up = Upstream(answer)
        proxy = bp.Proxy(f"http://127.0.0.1:{up.port}", inject=inject or {}, log_path=tmp_path / "proxy.jsonl")
        proxy.start()
        made.append((up, proxy))
        return up, proxy
    yield make
    for up, proxy in made:
        proxy.stop()
        up.close()


def test_a_request_is_passed_through_with_its_body_and_the_answer_comes_back_unchanged(rig):
    up, proxy = rig()
    status, raw = post(proxy.port, "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0})
    assert status == 200 and json.loads(raw)["choices"][0]["message"]["content"] == "hi"
    assert up.seen == [{"path": "/v1/chat/completions", "body": {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0}}]


def test_thinking_off_is_added_to_the_body_and_other_fields_stay(rig):
    up, proxy = rig(inject={"chat_template_kwargs": {"enable_thinking": False}})
    post(proxy.port, "/v1/chat/completions", {"model": "m", "messages": [], "max_tokens": 5})
    body = up.seen[0]["body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 5 and body["model"] == "m"


def test_an_injected_template_option_is_merged_with_one_the_client_sent_and_the_injection_wins(rig):
    up, proxy = rig(inject={"chat_template_kwargs": {"enable_thinking": False}})
    post(proxy.port, "/v1/chat/completions", {"chat_template_kwargs": {"enable_thinking": True, "other": 1}})
    assert up.seen[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False, "other": 1}


def test_an_injected_plain_field_replaces_the_clients_value(rig):
    up, proxy = rig(inject={"max_tokens": 9})
    post(proxy.port, "/v1/chat/completions", {"max_tokens": 2})
    assert up.seen[0]["body"]["max_tokens"] == 9


def test_nothing_is_added_when_no_injection_is_asked_for(rig):
    up, proxy = rig()
    post(proxy.port, "/v1/chat/completions", {"model": "m"})
    assert up.seen[0]["body"] == {"model": "m"}


def test_a_body_that_is_not_json_is_forwarded_as_it_came(rig):
    up, proxy = rig(inject={"chat_template_kwargs": {"enable_thinking": False}})
    post(proxy.port, "/v1/other", b"plain text", raw=True)
    assert up.seen[0]["body"] == b"plain text"
    assert proxy.records == []                                  # not a completion path, so not a recorded request


def test_a_get_is_forwarded_so_the_client_can_ask_for_the_model_list(rig):
    up, proxy = rig()
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("GET", "/v1/models")
    resp = conn.getresponse()
    assert resp.status == 200 and json.loads(resp.read())["data"][0]["id"] == "served-model"
    conn.close()
    assert up.seen[0]["path"] == "/v1/models"


def test_an_error_status_from_the_server_reaches_the_client_and_is_recorded(rig):
    up, proxy = rig(answer=lambda p, b: (500, {"error": {"message": "boom"}}))
    status, raw = post(proxy.port, "/v1/chat/completions", {"model": "m"})
    assert status == 500 and json.loads(raw)["error"]["message"] == "boom"
    assert proxy.records[0]["status"] == 500 and proxy.records[0]["error"] is True


def test_a_server_that_is_not_there_gives_the_client_a_502_and_a_record(tmp_path):
    proxy = bp.Proxy("http://127.0.0.1:9", inject={}, log_path=tmp_path / "p.jsonl")
    proxy.start()
    try:
        status, raw = post(proxy.port, "/v1/chat/completions", {"model": "m"})
    finally:
        proxy.stop()
    assert status == 502 and "upstream" in json.loads(raw)["error"]["message"]
    assert proxy.records[0]["status"] == 502 and proxy.records[0]["error"] is True


def test_each_request_is_recorded_with_tokens_finish_reason_and_latency(rig, tmp_path):
    def slow(path, body):
        time.sleep(0.1)
        return 200, {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": None, "reasoning_content": "abcdef"}}],
                     "usage": {"prompt_tokens": 11, "completion_tokens": 64}}
    up, proxy = rig(answer=slow)
    post(proxy.port, "/v1/chat/completions", {"model": "m"})
    rec = proxy.records[0]
    assert rec["path"] == "/v1/chat/completions" and rec["status"] == 200 and rec["error"] is False
    assert rec["prompt_tokens"] == 11 and rec["completion_tokens"] == 64 and rec["finish_reason"] == "length"
    assert rec["reasoning_chars"] == 6 and rec["latency_s"] >= 0.1
    lines = (tmp_path / "proxy.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == rec


def test_the_reasoning_field_may_be_called_reasoning(rig):
    up, proxy = rig(answer=lambda p, b: (200, {"choices": [{"finish_reason": "stop", "message": {"content": "x", "reasoning": "12345"}}], "usage": {}}))
    post(proxy.port, "/v1/chat/completions", {})
    assert proxy.records[0]["reasoning_chars"] == 5


def test_a_reply_without_usage_is_recorded_with_none(rig):
    up, proxy = rig(answer=lambda p, b: (200, {"choices": [{"message": {"content": "x"}}]}))
    post(proxy.port, "/v1/chat/completions", {})
    rec = proxy.records[0]
    assert rec["completion_tokens"] is None and rec["finish_reason"] is None and rec["reasoning_chars"] == 0


def test_only_posts_to_the_completion_paths_are_recorded_as_requests(rig):
    up, proxy = rig()
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("GET", "/v1/models")
    conn.getresponse().read()
    conn.close()
    assert proxy.records == []


def test_summary_counts_requests_errors_cut_offs_and_tokens(rig):
    state = {"n": 0}

    def varied(path, body):
        state["n"] += 1
        n = state["n"]
        time.sleep(0.05 * n)
        if n == 3:
            return 500, {"error": {"message": "x"}}
        return 200, {"choices": [{"finish_reason": "length" if n == 2 else "stop", "message": {"content": "x"}}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": n * 10}}
    up, proxy = rig(answer=varied)
    for _ in range(4):
        post(proxy.port, "/v1/chat/completions", {})
    s = proxy.summary()
    assert s["requests"] == 4 and s["errors"] == 1 and s["cut_off"] == 1
    assert s["completion_tokens_total"] == 10 + 20 + 40 and s["completion_tokens_median"] == 20
    assert s["prompt_tokens_total"] == 30
    assert s["latency_s_max"] >= 0.2 > s["latency_s_median"] > 0


def test_summary_of_nothing_is_zeroes_and_none(rig):
    up, proxy = rig()
    s = proxy.summary()
    assert s["requests"] == 0 and s["completion_tokens_total"] == 0 and s["completion_tokens_median"] is None and s["latency_s_max"] is None


def test_concurrent_requests_are_all_recorded(rig):
    up, proxy = rig()
    threads = [threading.Thread(target=post, args=(proxy.port, "/v1/chat/completions", {"i": i})) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert proxy.summary()["requests"] == 12 and len(up.seen) == 12


def test_the_proxy_listens_only_on_loopback(rig):
    up, proxy = rig()
    assert proxy.server.server_address[0] == "127.0.0.1"


def test_an_upstream_that_is_not_loopback_is_refused(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        bp.Proxy("http://example.com:8100", inject={}, log_path=tmp_path / "p.jsonl")
