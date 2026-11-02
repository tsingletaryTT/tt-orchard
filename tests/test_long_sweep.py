# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/long_sweep.py (issue 18) against fakes: an injected send function for the arithmetic and a small streaming HTTP server for the
real request path. No board is used."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from orchard import agreement_core as core
from orchard import long_sweep as ls


def fake_send(first_s=0.5, decode_per_token=0.05, prompt_tokens="words", completion_tokens=None, fail_for=None, text="Alpha beta gamma.",
              tail_s=0.0):
    """A send function that returns invented timings (no sleep): the first text after `first_s` per 1000 prompt words, then one token per
    `decode_per_token`, then `tail_s` more seconds before the stream ends. `fail_for` is a substring of the prompt that makes it raise."""
    def send(prompt, max_tokens):
        if fail_for and fail_for in prompt:
            raise OSError("connection refused")
        words = len(prompt.split())
        t_first = first_s * words / 1000
        comp = completion_tokens or max_tokens
        t_last = t_first + (comp - 1) * decode_per_token
        pt = int(words * 1.3) if prompt_tokens == "words" else prompt_tokens
        return {"t_first": t_first, "t_last": t_last, "t_end": t_last + tail_s, "text": text, "prompt_tokens": pt, "completion_tokens": comp}
    return send


CELL = {"input_tokens": 1000, "concurrency": 1, "reference": "cpu"}


def test_a_cell_gets_prefill_and_decode_speeds():
    cell = ls.measure_cell(fake_send(), CELL, 21, "n")
    assert cell["error"] is None
    assert cell["decode_tok_s"] == pytest.approx(20.0)                     # 20 tokens in 20 * 0.05 s
    assert cell["prefill_tok_s"] > 0 and cell["prompt_tokens"] > 500
    assert cell["output_head"] == "Alpha beta gamma." and len(cell["output_sha256"]) == 64 and len(cell["prompt_sha256"]) == 64


def test_the_wait_for_the_end_of_the_stream_is_not_charged_to_decode():
    fast = ls.measure_cell(fake_send(), CELL, 21, "n")
    slow_tail = ls.measure_cell(fake_send(tail_s=30.0), CELL, 21, "n")
    assert fast["decode_tok_s"] == slow_tail["decode_tok_s"]


def test_concurrent_requests_leave_together_each_with_its_own_prompt(monkeypatch):
    class SlowStart(threading.Thread):
        """A thread that takes 30 ms to start, as a busy machine's threads can: without a barrier the requests would leave 30 ms apart."""
        def start(self):
            time.sleep(0.03)
            super().start()
    monkeypatch.setattr(ls.threading, "Thread", SlowStart)
    seen, started = [], []
    base = fake_send()

    def send(prompt, k):
        seen.append(prompt.splitlines()[0])
        started.append(time.monotonic())
        time.sleep(0.05)
        return base(prompt, k)
    ls.measure_cell(send, {"input_tokens": 500, "concurrency": 4, "reference": "cpu"}, 16, "run1")
    assert len(set(seen)) == 4                                              # a repeat would be answered from a prefix cache
    assert max(started) - min(started) < 0.04                              # the four were in flight together, not one after another


def test_the_prompt_asks_for_a_sustained_reply_and_two_runs_with_one_nonce_send_the_same_prompts():
    assert "numbered list of 40" in ls._prompt(500, "x")
    a = ls.measure_cell(fake_send(), CELL, 16, "same")
    b = ls.measure_cell(fake_send(), CELL, 16, "same")
    c = ls.measure_cell(fake_send(), CELL, 16, "other")
    assert a["prompt_sha256"] == b["prompt_sha256"] != c["prompt_sha256"]


def test_a_failing_request_gives_the_cell_an_error_and_the_sweep_goes_on():
    plan = core.sweep_plan([500, 1000], [1])
    rec = ls.run_sweep(fake_send(fail_for="-1000-"), plan, 16, nonce_base="z")
    assert [c["error"] is None for c in rec["cells"]] == [True, False]
    assert rec["summary"]["missing_cells"] == 1 and rec["label"] == "measured"


def test_a_short_reply_makes_the_cell_incomplete_and_gives_no_decode_speed():
    for tokens in (1, 2, 9):
        cell = ls.measure_cell(fake_send(completion_tokens=tokens), CELL, 64, "n")
        assert cell["decode_tok_s"] is None and "at least 32 are needed" in cell["error"]


def test_a_server_that_reports_no_usage_gives_no_rate_and_says_why():
    cell = ls.measure_cell(fake_send(prompt_tokens=None), CELL, 16, "n")
    assert cell["prefill_tok_s"] is None and cell["decode_tok_s"] is None and "no token usage" in cell["error"]
    assert cell["prompt_tokens"] is None and cell["output_head"]               # the output is still kept for the comparison


def test_the_record_names_the_requested_model_and_says_it_was_not_verified():
    rec = ls.run_sweep(fake_send(), core.sweep_plan([500], [1]), 16, model="some/model", api="chat")
    assert rec["model"]["requested"] == "some/model" and rec["model"]["verified"] is False and rec["api"] == "chat"


def test_the_sweep_summary_in_the_record_flags_a_long_context_slowdown():
    def send(prompt, k):
        words = len(prompt.split())
        return fake_send(decode_per_token=0.05 if words < 3000 else 0.5)(prompt, k)
    rec = ls.run_sweep(send, core.sweep_plan([1000, 8192], [1]), 16, nonce_base="q")
    assert [d["input_tokens"] for d in rec["summary"]["decode_drops"]] == [8192]


def test_compare_reports_the_shared_prefix_of_each_cell_and_refuses_different_prompts():
    plan = core.sweep_plan([500, 1000], [1])
    new = ls.run_sweep(fake_send(text="The answer is violet."), plan, 16, nonce_base="a")
    same = ls.run_sweep(fake_send(text="The answer is violet."), plan, 16, nonce_base="a")
    other = ls.run_sweep(fake_send(text="The answer is blue."), plan, 16, nonce_base="a")
    unpaired = ls.run_sweep(fake_send(text="The answer is violet."), plan, 16, nonce_base="b")
    assert all(r["identical_output"] and r["common_prefix_share"] == 1.0 for r in ls.compare_sweeps(new, same)["cells"])
    row = ls.compare_sweeps(new, other)["cells"][0]
    assert row["identical_output"] is False and 0.5 < row["common_prefix_share"] < 1.0
    rows = ls.compare_sweeps(new, unpaired)["cells"]
    assert all(r["different_prompts"] and r["common_prefix_share"] is None and r["identical_output"] is False for r in rows)
    assert ls.compare_sweeps({}, None)["cells"] == []


def test_an_ipv6_host_is_bracketed_in_the_url():
    assert ls._netloc("::1", 8000) == "[::1]:8000" and ls._netloc("127.0.0.1", 8000) == "127.0.0.1:8000" and ls._netloc("localhost", 1) == "localhost:1"


# ---- the real request path against a streaming fake -------------------------------------------------------------------------------

class Stream(BaseHTTPRequestHandler):
    """Streams `max_tokens` pieces. The class attribute `mode` picks a behaviour; `paths` records what was asked for."""
    mode = "ok"
    paths: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).paths.append(self.path)
        chat = self.path == "/v1/chat/completions"
        if self.path not in ("/v1/chat/completions", "/v1/completions") or (self.mode == "chat-only" and not chat):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        assert req["stream"] is True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(req["max_tokens"]):
            if self.mode == "error-after-3" and i == 3:
                self.wfile.write(b"data: " + json.dumps({"error": {"message": "backend died"}}).encode() + b"\n\n")
                return
            time.sleep(0.01)
            piece = {"delta": {"content": f" w{i}"}} if chat else {"text": f" w{i}"}
            self.wfile.write(b"data: " + json.dumps({"choices": [piece]}).encode() + b"\n\n")
            self.wfile.flush()
        if self.mode != "no-usage":
            self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 777, "completion_tokens": req["max_tokens"]}}).encode() + b"\n\n")
        if self.mode != "no-done":
            self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():
    Stream.mode, Stream.paths = "ok", []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stream)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_send_reads_a_chat_stream_and_the_usage_record(server):
    r = ls.http_send("127.0.0.1", server, "m", "hello", 12, "chat", timeout=20)
    assert r["prompt_tokens"] == 777 and r["completion_tokens"] == 12 and r["text"].startswith(" w0 w1")
    assert 0 < r["t_first"] < r["t_last"] <= r["t_end"] and Stream.paths == ["/v1/chat/completions"]


def test_http_send_reads_a_completions_stream_too(server):
    r = ls.http_send("127.0.0.1", server, "m", "hello", 5, "completions", timeout=20)
    assert r["completion_tokens"] == 5 and Stream.paths == ["/v1/completions"]


def test_the_chat_api_works_against_a_server_that_has_no_completions_endpoint(server):
    Stream.mode = "chat-only"
    assert ls.http_send("127.0.0.1", server, "m", "hello", 5, timeout=20)["completion_tokens"] == 5
    with pytest.raises(OSError):                                           # urllib raises HTTPError (an OSError) for the 404
        ls.http_send("127.0.0.1", server, "m", "hello", 5, "completions", timeout=20)


def test_an_error_event_in_the_stream_is_a_failure_not_a_partial_measurement(server):
    Stream.mode = "error-after-3"
    with pytest.raises(ValueError, match="error event"):
        ls.http_send("127.0.0.1", server, "m", "hello", 12, timeout=20)
    cell = ls.measure_cell(lambda p, k: ls.http_send("127.0.0.1", server, "m", p, k, timeout=20), CELL, 12, "n")
    assert cell["decode_tok_s"] is None and "error event" in cell["error"] and cell["output_sha256"] is None


def test_a_stream_that_ends_without_the_done_marker_is_a_failure(server):
    Stream.mode = "no-done"
    with pytest.raises(ValueError, match="DONE"):
        ls.http_send("127.0.0.1", server, "m", "hello", 12, timeout=20)


def test_http_send_gives_no_counts_when_the_server_reports_no_usage(server):
    Stream.mode = "no-usage"
    r = ls.http_send("127.0.0.1", server, "m", "hello", 12, timeout=20)
    assert r["prompt_tokens"] is None and r["completion_tokens"] is None


def test_a_bad_api_name_is_refused():
    with pytest.raises(ValueError):
        ls.http_send("127.0.0.1", 1, "m", "x", 5, "grpc")


def test_the_command_writes_a_record_from_a_running_server(server, tmp_path, capsys):
    out = tmp_path / "sweep.json"
    rc = ls.main(["run", "--port", str(server), "--model", "m", "--out", str(out), "--lengths", "200,400", "--concurrency", "1,2",
                  "--decode-tokens", "10"])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert len(rec["cells"]) == 4 and all(c["error"] is None and c["decode_tok_s"] > 0 for c in rec["cells"])
    assert rec["model"]["requested"] == "m" and rec["api"] == "chat"
    assert capsys.readouterr().out.count("input_tokens") == 4


@pytest.mark.parametrize("argv", [["run", "--port", "1", "--model", "m", "--out", "x", "--host", "10.0.0.5"],
                                  ["run", "--port", "1", "--model", "m", "--out", "x", "--lengths", "0"],
                                  ["run", "--port", "1", "--model", "m", "--out", "x", "--decode-tokens", "2"]])
def test_the_command_refuses_a_remote_host_a_bad_length_and_a_tiny_decode(argv, tmp_path, capsys):
    argv = [a if a != "x" else str(tmp_path / "o.json") for a in argv]
    assert ls.main(argv) == 2
    assert not (tmp_path / "o.json").exists()


def test_compare_names_an_unreadable_record_and_exits_2(tmp_path, capsys):
    assert ls.main(["compare", str(tmp_path / "missing.json"), str(tmp_path / "other.json")]) == 2
    assert "cannot read a sweep record" in capsys.readouterr().err
