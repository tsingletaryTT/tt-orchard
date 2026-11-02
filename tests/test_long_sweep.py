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


def fake_send(first_s=0.5, decode_per_token=0.05, prompt_tokens=None, completion_tokens=None, fail_for=None, text="Alpha beta gamma."):
    """A send function that returns invented timings (no sleep): the first token after `first_s` per 1000 prompt words, then one token per
    `decode_per_token`. `fail_for` is a substring of the prompt that makes it raise."""
    def send(prompt, max_tokens):
        if fail_for and fail_for in prompt:
            raise OSError("connection refused")
        words = len(prompt.split())
        t_first = first_s * words / 1000
        comp = completion_tokens or max_tokens
        return {"t_first": t_first, "t_end": t_first + (comp - 1) * decode_per_token, "text": text,
                "prompt_tokens": prompt_tokens if prompt_tokens is not None else int(words * 1.3), "completion_tokens": comp}
    return send


def test_a_cell_gets_prefill_and_decode_speeds():
    cell = ls.measure_cell(fake_send(), {"input_tokens": 1000, "concurrency": 1, "reference": "cpu"}, 21, "n")
    assert cell["error"] is None
    assert cell["decode_tok_s"] == pytest.approx(20.0)                     # 20 tokens in 20 * 0.05 s
    assert cell["prefill_tok_s"] > 0 and cell["prompt_tokens"] > 500
    assert cell["output_head"] == "Alpha beta gamma." and len(cell["output_sha256"]) == 64


def test_concurrent_requests_each_get_a_unique_prompt():
    seen = []
    base = fake_send()

    def send(prompt, k):
        seen.append(prompt.splitlines()[0])
        return base(prompt, k)
    ls.measure_cell(send, {"input_tokens": 500, "concurrency": 4, "reference": "cpu"}, 16, "run1")
    assert len(set(seen)) == 4                                              # a repeat would be answered from a prefix cache


def test_a_failing_request_gives_the_cell_an_error_and_the_sweep_goes_on():
    plan = core.sweep_plan([500, 1000], [1])
    rec = ls.run_sweep(fake_send(fail_for="-1000-"), plan, 16, nonce_base="z")
    assert [c["error"] is None for c in rec["cells"]] == [True, False]
    assert rec["summary"]["missing_cells"] == 1 and rec["label"] == "measured"


def test_a_cell_with_one_completion_token_has_no_decode_speed():
    cell = ls.measure_cell(fake_send(completion_tokens=1), {"input_tokens": 500, "concurrency": 1, "reference": "cpu"}, 16, "n")
    assert cell["decode_tok_s"] is None and cell["error"]


def test_the_sweep_summary_in_the_record_flags_a_long_context_slowdown():
    def send(prompt, k):
        words = len(prompt.split())
        per_token = 0.05 if words < 3000 else 0.5          # the long input decodes ten times slower
        return fake_send(decode_per_token=per_token)(prompt, k)
    rec = ls.run_sweep(send, core.sweep_plan([1000, 8192], [1]), 16, nonce_base="q")
    assert [d["input_tokens"] for d in rec["summary"]["decode_drops"]] == [8192]


def test_compare_reports_the_shared_prefix_of_each_cell():
    plan = core.sweep_plan([500, 1000], [1])
    new = ls.run_sweep(fake_send(text="The answer is violet."), plan, 16, nonce_base="a")
    same = ls.run_sweep(fake_send(text="The answer is violet."), plan, 16, nonce_base="b")
    other = ls.run_sweep(fake_send(text="The answer is blue."), plan, 16, nonce_base="c")
    assert all(r["identical_output"] and r["common_prefix_share"] == 1.0 for r in ls.compare_sweeps(new, same)["cells"])
    row = ls.compare_sweeps(new, other)["cells"][0]
    assert row["identical_output"] is False and 0.5 < row["common_prefix_share"] < 1.0
    assert ls.compare_sweeps({}, None)["cells"] == []


# ---- the real request path against a streaming fake -------------------------------------------------------------------------------

class Stream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert req["stream"] is True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(req["max_tokens"]):
            time.sleep(0.01)
            self.wfile.write(b"data: " + json.dumps({"choices": [{"text": f" w{i}"}]}).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 777, "completion_tokens": req["max_tokens"]}}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stream)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_send_reads_a_stream_and_the_usage_record(server):
    r = ls.http_send("127.0.0.1", server, "m", "hello", 12, timeout=20)
    assert r["prompt_tokens"] == 777 and r["completion_tokens"] == 12 and r["text"].startswith(" w0 w1")
    assert 0 < r["t_first"] < r["t_end"]


def test_the_command_writes_a_record_from_a_running_server(server, tmp_path, capsys):
    out = tmp_path / "sweep.json"
    rc = ls.main(["run", "--port", str(server), "--model", "m", "--out", str(out), "--lengths", "200,400", "--concurrency", "1,2",
                  "--decode-tokens", "10"])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert len(rec["cells"]) == 4 and all(c["error"] is None and c["decode_tok_s"] > 0 for c in rec["cells"])
    assert capsys.readouterr().out.count("input_tokens") == 4


@pytest.mark.parametrize("argv", [["run", "--port", "1", "--model", "m", "--out", "x", "--host", "10.0.0.5"],
                                  ["run", "--port", "1", "--model", "m", "--out", "x", "--lengths", "0"],
                                  ["run", "--port", "1", "--model", "m", "--out", "x", "--decode-tokens", "2"]])
def test_the_command_refuses_a_remote_host_a_bad_length_and_a_tiny_decode(argv, tmp_path, capsys):
    argv = [a if a != "x" else str(tmp_path / "o.json") for a in argv]
    assert ls.main(argv) == 2
    assert not (tmp_path / "o.json").exists()
