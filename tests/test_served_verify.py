# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/served_verify.py: read-only probes of a live OpenAI-shaped server, tested against a fake HTTP server (no chip, no model).

Each probe answers `pass`, `fail` or `not_checked`. `not_checked` is what a probe returns when the server does not offer the thing
asked about (no /metrics, no /v1/completions); it is never a pass. The mutation table for these tests is in the PR text.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from orchard import served_verify as sv


# ---- a fake server whose answers a test can change ---------------------------------------------------------------------------

class Fake:
    """Routes: GET path -> (status, bytes); POST path -> a function(body dict) -> (status, bytes). `counter` is what /metrics reports."""

    def __init__(self):
        self.get = {}
        self.post = {}
        self.counter = 0.0
        self.metrics = True
        self.metrics_body = None
        self.log = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, data):
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.log.append(("GET", self.path))
                if self.path == "/metrics":
                    if not outer.metrics:
                        return self._send(404, b"no")
                    text = outer.metrics_body or f'vllm:generation_tokens_total{{engine="0",model_name="m"}} {outer.counter}\n'
                    return self._send(200, text.encode())
                code, data = outer.get.get(self.path, (404, b"no"))
                self._send(code, data)

            def do_POST(self):
                outer.log.append(("POST", self.path))
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                fn = outer.post.get(self.path)
                if fn is None:
                    return self._send(404, b"no")
                code, data = fn(body)
                self._send(code, data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.close()


def models_body(name="org/m", max_len=4096):
    return json.dumps({"object": "list", "data": [{"id": name, "max_model_len": max_len}]}).encode()


def completion(text, finish="stop", tokens=3):
    return json.dumps({"choices": [{"text": text, "finish_reason": finish}], "usage": {"completion_tokens": tokens}}).encode()


# ---- the listing: served name and context length (V2) -------------------------------------------------------------------------

def test_a_listing_that_matches_the_declared_name_and_context_passes():
    r = sv.check_listing(200, models_body("org/m", 4096), declared_name="org/m", declared_context=4096)
    assert r["state"] == "pass", r
    assert r["served"] == {"id": "org/m", "max_model_len": 4096}


def test_a_served_name_other_than_the_declared_one_fails():
    r = sv.check_listing(200, models_body("other/model"), declared_name="org/m")
    assert r["state"] == "fail" and "other/model" in r["detail"] and "org/m" in r["detail"]


def test_a_context_length_other_than_the_declared_one_fails():
    r = sv.check_listing(200, models_body(max_len=2048), declared_context=4096)
    assert r["state"] == "fail" and "2048" in r["detail"] and "4096" in r["detail"]


def test_an_empty_listing_is_a_fail_because_a_server_that_answers_health_must_list_its_model():
    r = sv.check_listing(200, json.dumps({"data": []}).encode())
    assert r["state"] == "fail" and "lists no model" in r["detail"]


def test_a_listing_that_cannot_be_parsed_or_has_the_wrong_shape_is_a_fail_not_an_exception():
    for body in (b"not json", b"[]", json.dumps({"data": "x"}).encode(), json.dumps({"data": [None]}).encode(),
                 json.dumps({"data": [{"id": 5}]}).encode(), json.dumps({"data": [{"id": ""}]}).encode(), b"\xff\xfe", b""):
        assert sv.check_listing(200, body)["state"] == "fail", body


def test_a_non_200_listing_fails():
    assert sv.check_listing(503, b"{}")["state"] == "fail"


@pytest.mark.parametrize("bad", [None, [], {}, "4096", 4096.5, True, -1, 0, 10 ** 30, float("nan")])
def test_a_served_context_length_of_the_wrong_type_is_recorded_as_unreported_not_trusted(bad):
    body = json.dumps({"data": [{"id": "m", "max_model_len": bad}]}).encode() if bad == bad else b'{"data":[{"id":"m","max_model_len":NaN}]}'
    r = sv.check_listing(200, body, declared_context=4096)
    assert r["state"] in ("fail", "not_checked") and r["state"] != "pass"


def test_no_declared_context_means_the_context_is_reported_not_compared():
    r = sv.check_listing(200, models_body("m", 4096))
    assert r["state"] == "pass" and "context was not compared" in r["detail"]


def test_the_native_context_gives_one_of_three_outcomes_and_never_a_failure():
    below = sv.check_listing(200, models_body(max_len=2048), native_context=4096)
    equal = sv.check_listing(200, models_body(max_len=4096), native_context=4096)
    above = sv.check_listing(200, models_body(max_len=8192), native_context=4096)
    assert [r["state"] for r in (below, equal, above)] == ["pass"] * 3
    assert "below native" in below["detail"] and "equal to native" in equal["detail"] and "above native" in above["detail"]


# ---- the reply: the model it names, and an answer eaten by thinking (V17) -----------------------------------------------------

def chat(content, finish="stop", model="org/m"):
    return json.dumps({"model": model, "choices": [{"message": {"content": content}, "finish_reason": finish}]}).encode()


def test_a_reply_from_the_asked_model_with_content_passes():
    assert sv.check_reply(200, chat("Paris"), asked_model="org/m")["state"] == "pass"


def test_a_reply_that_names_another_model_fails():
    r = sv.check_reply(200, chat("Paris", model="elsewhere"), asked_model="org/m")
    assert r["state"] == "fail" and "elsewhere" in r["detail"]


def test_a_reply_without_a_model_field_is_not_a_failure():
    body = json.dumps({"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}]}).encode()
    assert sv.check_reply(200, body, asked_model="org/m")["state"] == "pass"


def test_empty_content_with_finish_reason_length_is_named_as_a_spent_budget():
    r = sv.check_reply(200, chat("", finish="length"), asked_model="org/m")
    assert r["state"] == "fail" and "token budget" in r["detail"]


def test_empty_content_that_stopped_is_a_different_finding_from_a_spent_budget():
    r = sv.check_reply(200, chat("", finish="stop"), asked_model="org/m")
    assert r["state"] == "fail" and "token budget" not in r["detail"]


def test_a_reply_of_the_wrong_shape_fails_without_raising():
    for body in (b"x", b"[]", json.dumps({"choices": []}).encode(), json.dumps({"choices": [None]}).encode(),
                 json.dumps({"choices": [{"message": None}]}).encode()):
        assert sv.check_reply(200, body, asked_model="m")["state"] == "fail", body


# ---- raw completions at temperature 0: arithmetic and sequential state (V4, V5) ------------------------------------------------

def test_the_arithmetic_and_sequence_probes_pass_a_correct_server(fake):
    fake.post["/v1/completions"] = lambda b: (200, completion("408") if "17*24" in b["prompt"] else completion(" 9 10 11 12"))
    got = {r["name"]: r for r in sv.probe_completions(fake.base, "m")}
    assert got["arithmetic"]["state"] == "pass" and got["sequence"]["state"] == "pass"


def test_every_probe_request_is_greedy_and_bounded(fake):
    seen = []

    def answer(body):
        seen.append(body)
        return 200, completion("408 9 10 11")
    fake.post["/v1/completions"] = answer
    sv.probe_completions(fake.base, "m")
    assert len(seen) == 2 and all(b["temperature"] == 0 and 0 < b["max_tokens"] <= 16 and b["model"] == "m" for b in seen)


def test_a_wrong_product_fails_the_arithmetic_probe(fake):
    fake.post["/v1/completions"] = lambda b: (200, completion("407") if "17*24" in b["prompt"] else completion(" 9 10 11"))
    assert {r["name"]: r["state"] for r in sv.probe_completions(fake.base, "m")}["arithmetic"] == "fail"


def test_a_digit_run_that_merely_contains_the_product_does_not_pass(fake):
    """4080 and 1408 contain 408 as text and are wrong."""
    for text in ("4080", "1408"):
        fake.post["/v1/completions"] = lambda b, t=text: (200, completion(t) if "17*24" in b["prompt"] else completion(" 9 10 11"))
        assert {r["name"]: r["state"] for r in sv.probe_completions(fake.base, "m")}["arithmetic"] == "fail", text


def test_a_repeated_token_fails_the_sequence_probe(fake):
    fake.post["/v1/completions"] = lambda b: (200, completion("408") if "17*24" in b["prompt"] else completion(" 9 9 10 10 1"))
    assert {r["name"]: r["state"] for r in sv.probe_completions(fake.base, "m")}["sequence"] == "fail"


def test_a_sequence_that_skips_or_starts_late_fails(fake):
    for text in (" 10 11 12", " 9 11 12", " 8 9 10", "", " 11 10 9", " 9 10 12 11", " 9 10 110"):
        fake.post["/v1/completions"] = lambda b, t=text: (200, completion("408") if "17*24" in b["prompt"] else completion(t))
        assert {r["name"]: r["state"] for r in sv.probe_completions(fake.base, "m")}["sequence"] == "fail", repr(text)


def test_a_server_with_no_completions_route_is_not_checked_rather_than_failed(fake):
    got = sv.probe_completions(fake.base, "m")
    assert [r["state"] for r in got] == ["not_checked", "not_checked"]
    assert all("404" in r["detail"] for r in got)


def test_a_server_error_on_completions_is_a_fail_and_a_dead_server_is_not_checked_as_pass(fake):
    fake.post["/v1/completions"] = lambda b: (500, b"boom")
    assert [r["state"] for r in sv.probe_completions(fake.base, "m")] == ["fail", "fail"]
    fake.close()
    got = sv.probe_completions(fake.base, "m", timeout=2)
    assert all(r["state"] == "fail" and "no answer" in r["detail"] for r in got), got


def test_a_malformed_completion_body_fails_without_raising(fake):
    for body in (b"x", b"[]", json.dumps({"choices": [{}]}).encode(), json.dumps({"choices": [{"text": 5}]}).encode()):
        fake.post["/v1/completions"] = lambda b, x=body: (200, x)
        assert all(r["state"] == "fail" for r in sv.probe_completions(fake.base, "m")), body


# ---- the server's own counters (V8) --------------------------------------------------------------------------------------------

METRICS_BEFORE = 'vllm:generation_tokens_total{engine="0",model_name="m"} 100.0\n# HELP x y\n'
METRICS_AFTER = 'vllm:generation_tokens_total{engine="0",model_name="m"} 164.0\n'


def test_the_token_counter_rising_by_the_request_tokens_passes():
    r = sv.check_counter(METRICS_BEFORE, METRICS_AFTER, completion_tokens=64)
    assert r["state"] == "pass" and r["delta"] == 64


def test_a_counter_that_did_not_move_fails():
    r = sv.check_counter(METRICS_BEFORE, METRICS_BEFORE, completion_tokens=64)
    assert r["state"] == "fail" and "did not rise" in r["detail"]


def test_a_counter_that_rose_by_less_than_the_request_generated_fails():
    assert sv.check_counter(METRICS_BEFORE, 'vllm:generation_tokens_total{engine="0"} 105.0\n', completion_tokens=64)["state"] == "fail"


def test_other_traffic_may_raise_the_counter_by_more_than_the_request():
    assert sv.check_counter(METRICS_BEFORE, 'vllm:generation_tokens_total{engine="0"} 400.0\n', completion_tokens=64)["state"] == "pass"


def test_counters_of_several_engines_are_added():
    before = 'vllm:generation_tokens_total{engine="0"} 10\nvllm:generation_tokens_total{engine="1"} 10\n'
    after = 'vllm:generation_tokens_total{engine="0"} 40\nvllm:generation_tokens_total{engine="1"} 40\n'
    assert sv.check_counter(before, after, completion_tokens=60)["delta"] == 60


def test_a_missing_counter_or_missing_token_count_is_not_checked():
    assert sv.check_counter(None, METRICS_AFTER, completion_tokens=64)["state"] == "not_checked"
    assert sv.check_counter(METRICS_BEFORE, "other_metric 1\n", completion_tokens=64)["state"] == "not_checked"
    for n in (None, 0, -3, True, "64", 1.5):
        assert sv.check_counter(METRICS_BEFORE, METRICS_AFTER, completion_tokens=n)["state"] == "not_checked", n


def test_a_counter_value_that_is_not_a_finite_number_is_not_trusted():
    for v in ("nan", "inf", "1e999999", "-5"):
        text = f'vllm:generation_tokens_total{{engine="0"}} {v}\n'
        assert sv.check_counter(METRICS_BEFORE, text, completion_tokens=64)["state"] != "pass", v


def test_scrape_reads_the_metrics_route_and_says_none_when_there_is_none(fake):
    fake.counter = 7
    assert "vllm:generation_tokens_total" in sv.scrape(fake.base)
    fake.metrics = False
    assert sv.scrape(fake.base) is None


def test_scrape_refuses_a_body_over_the_size_bound(fake, monkeypatch):
    monkeypatch.setattr(sv, "MAX_METRICS_BYTES", 100)
    fake.metrics_body = "x" * 5000
    assert sv.scrape(fake.base) is None


# ---- the mesh the server reports in its own log (V7) ---------------------------------------------------------------------------

def test_a_log_that_reports_the_expected_device_count_passes():
    r = sv.check_mesh("INFO multidevice with 4 devices and grid (1, 4) is created\n", expected_devices=4)
    assert r["state"] == "pass" and r["devices"] == 4


def test_a_log_that_reports_fewer_devices_than_declared_fails():
    r = sv.check_mesh("multidevice with 2 devices and grid (1, 2) is created", expected_devices=4)
    assert r["state"] == "fail" and "2" in r["detail"] and "4" in r["detail"]


def test_a_log_with_no_mesh_line_or_no_declared_count_is_not_checked():
    assert sv.check_mesh("nothing here", expected_devices=4)["state"] == "not_checked"
    assert sv.check_mesh("multidevice with 4 devices and grid (1, 4)", expected_devices=None)["state"] == "not_checked"


def test_the_mesh_line_is_matched_in_full_so_a_count_in_other_text_does_not_pass():
    for text in ("multidevice with 4 devicesXX", "a multidevice with 4 devices", "devices 4 grid (1, 4)"):
        assert sv.check_mesh(text, expected_devices=4)["state"] == "not_checked", text


def test_the_last_mesh_line_wins_when_a_log_holds_two_sessions():
    log = "multidevice with 2 devices and grid (1, 2)\nmultidevice with 4 devices and grid (1, 4)\n"
    assert sv.check_mesh(log, expected_devices=4)["state"] == "pass"


@pytest.mark.parametrize("bad", [True, False, 0, -1, 10 ** 30, 4096.0, "4096"])
def test_a_context_value_of_the_wrong_type_or_range_is_never_taken_as_the_served_length(bad):
    """Even when it happens to equal the declared value (True is 1), it must not be reported as the served context."""
    r = sv.check_listing(200, json.dumps({"data": [{"id": "m", "max_model_len": bad}]}).encode(), declared_context=1)
    assert r["served"]["max_model_len"] is None
    assert r["state"] != "pass"
