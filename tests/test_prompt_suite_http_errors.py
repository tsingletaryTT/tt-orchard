# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A broken HTTP answer to one suite request is recorded for that prompt and the run goes on (PR 49 review, finding 2).

A truncated body raises http.client.IncompleteRead, a malformed status line raises http.client.BadStatusLine, and an error answer
whose body is cut short raises IncompleteRead from `HTTPError.read()`. None of these is a URLError or an OSError, so they used to leave
`ask_suite_prompt()` and end the hardware test before `write_report()` saved the 32-token measurement and the other replies.
These tests talk to a real socket that sends exactly those bytes, so the exceptions are the ones http.client raises."""
import http.client
import json
import socketserver
import threading

import pytest

pytest.importorskip("tokenizers")

from orchard import prompt_suite as ps  # noqa: E402
from test_weights_swap_templates import load_swap_template  # noqa: E402

GOOD = json.dumps({"choices": [{"message": {"content": "Paris is the capital of France."}, "finish_reason": "stop"}]}).encode()
OK = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(GOOD), GOOD)
TRUNCATED_BODY = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n" + GOOD[:20]
BAD_STATUS = b"this is not http\r\n\r\n"
TRUNCATED_ERROR_BODY = b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 500\r\n\r\nboom"
CUT_BEFORE_ANY_ANSWER = b""


class RawServer(socketserver.ThreadingTCPServer):
    """Answers the n-th request with payloads[n] (the last one repeats), as raw bytes, then closes the connection."""
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, payloads):
        self.payloads, self.count, self.lock = list(payloads), 0, threading.Lock()
        super().__init__(("127.0.0.1", 0), self.Handler)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            length = 0
            while True:
                line = self.rfile.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1])
            self.rfile.read(length)
            with self.server.lock:
                n = self.server.count
                self.server.count += 1
            self.wfile.write(self.server.payloads[min(n, len(self.server.payloads) - 1)])
            self.wfile.flush()


@pytest.fixture
def raw():
    servers = []

    def start(*payloads):
        srv = RawServer(payloads)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


PROMPT = {"id": "chat-capital", "text": "What is the capital of France?", "messages": [{"role": "user", "content": "What is the capital of France?"}]}


@pytest.mark.parametrize("payload, name", [(TRUNCATED_BODY, "IncompleteRead"), (BAD_STATUS, "BadStatusLine"),
                                           (CUT_BEFORE_ANY_ANSWER, "RemoteDisconnected")])
def test_a_broken_answer_is_recorded_as_that_prompts_error(raw, payload, name):
    srv = raw(payload)
    tpl = load_swap_template()
    entry = tpl.ask_suite_prompt(srv.server_address[1], "m", PROMPT, "chat", 16)
    assert name in (entry["error"] or ""), entry
    assert entry["content"] == "" and entry["finish_reason"] is None


def test_an_error_answer_whose_body_is_cut_short_keeps_its_status_in_the_record(raw):
    srv = raw(TRUNCATED_ERROR_BODY)
    tpl = load_swap_template()
    entry = tpl.ask_suite_prompt(srv.server_address[1], "m", PROMPT, "chat", 16)
    assert entry["error"] and "500" in entry["error"], entry


def test_the_completion_endpoint_is_covered_the_same_way(raw):
    srv = raw(TRUNCATED_BODY)
    tpl = load_swap_template()
    entry = tpl.ask_suite_prompt(srv.server_address[1], "m", PROMPT, "completion", 16)
    assert "IncompleteRead" in (entry["error"] or "")


def test_post_json_reports_a_cut_error_body_as_a_server_error_with_the_status(raw):
    srv = raw(TRUNCATED_ERROR_BODY)
    tpl = load_swap_template()
    with pytest.raises(tpl.ServerHTTPError) as exc:
        tpl.post_json(srv.server_address[1], "/v1/chat/completions", {})
    assert "HTTP 500" in str(exc.value)


def test_the_suite_goes_on_after_one_broken_answer_and_records_every_prompt(raw, tmp_path):
    suite = ps.load_suite()
    srv = raw(TRUNCATED_BODY, OK)                      # the first request is cut short, every later one is fine
    (tmp_path / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}{% if tools %}tools{% endif %}")
    tpl = load_swap_template()
    record = tpl.run_suite({"new_model_id": "x/y"}, srv.server_address[1], "m", tmp_path,
                           {"suite": suite, "suite_sha256": ps.suite_sha256(), "control": None})
    entries = record["entries"]
    assert len(entries) == len(suite["prompts"])
    errors = [e for e in entries if e["error"]]
    assert len(errors) == 1 and "IncompleteRead" in errors[0]["error"]
    assert all(e["content"] for e in entries if not e["error"])


def test_the_32_token_check_reports_a_cut_error_body_as_a_server_error_with_the_status_too(raw):
    srv = raw(TRUNCATED_ERROR_BODY)
    tpl = load_swap_template()
    with pytest.raises(tpl.ServerHTTPError) as exc:
        tpl.complete(srv.server_address[1], "m", [1, 2, 3], 1)
    assert "HTTP 500" in str(exc.value)


@pytest.mark.parametrize("error", [ConnectionResetError("reset"), TimeoutError("timed out"), http.client.IncompleteRead(b"x", 9)])
def test_an_error_body_that_cannot_be_read_is_said_in_its_place(error):
    class Broken:
        def read(self):
            raise error

    tpl = load_swap_template()
    assert type(error).__name__ in tpl.error_body(Broken())


def test_the_suite_record_holds_the_digests_of_the_served_tokenizer_and_stop_files(raw, tmp_path):
    """orchard/served_chat.py ties the recorded replies to the files they were made with (issue 17, review of PR 294)."""
    import hashlib
    suite = ps.load_suite()
    srv = raw(OK)
    (tmp_path / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 7}')
    tpl = load_swap_template()
    record = tpl.run_suite({"new_model_id": "x/y"}, srv.server_address[1], "m", tmp_path,
                           {"suite": suite, "suite_sha256": ps.suite_sha256(), "control": None})
    got = record["served_files"]
    assert got["generation_config.json"] == hashlib.sha256(b'{"eos_token_id": 7}').hexdigest()
    assert got["tokenizer.json"] is None and set(got) == {"tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json", "chat_template.jinja"}
