# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Concurrency and isolation probes of orchard/served_probes.py (Lottolabs study, 2026-11-24, learnings 20 and 21), against a local fake
server that streams. No chip and no model. The Lottolabs cards list these checks: each of 16 concurrent requests equals the same request alone, an
A-B-A check with other streams running, a cancel mid-decode that leaves the next request exact, speculation on equals off, a streamed tool call that
never arrives with `{}` arguments, and a 400 for min_p, logit_bias and min_tokens under speculation. The first chip run of these probes is owed.

The fake's replies are a function of the prompt, so a reply that differs from the one for the same prompt alone is crosstalk. Each fault is a switch.
Overlap of concurrent requests comes from every request waiting `delay` seconds inside the handler, so a burst of threads started together overlaps
by a wide margin."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from orchard import served_probes as sp

TOOL_ARGS = {"city": "Paris", "unit": "celsius"}


class Server:
    def __init__(self, **faults):
        self.faults = faults
        self.lock = threading.Lock()
        self.in_flight = 0
        self.cancelled = 0
        self.broken = False
        self.calls = 0
        self.seen_bodies: list = []
        self.models_gets = 0
        self.unhealthy = False
        self.stale_after = None
        self.active: set = set()
        self.crowded: set = set()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, doc):
                data = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    outer.models_gets += 1
                    return self._json(500 if (outer.broken or outer.unhealthy) else 200, {"data": [{"id": "m"}]})
                self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.seen_bodies.append(body)
                if self.path != "/v1/chat/completions":
                    return self._json(404, {})
                me = object()
                with outer.lock:
                    outer.in_flight += 1
                    outer.calls += 1
                    crowd = outer.in_flight
                    outer.active.add(me)
                    if len(outer.active) >= 3:
                        outer.crowded |= outer.active           # every request that was in flight when three overlapped
                try:
                    if outer.faults.get("overload500") and crowd >= 3:
                        return self._json(500, {"error": "overloaded"})
                    if outer.broken:
                        return self._json(500, {"error": "broken"})
                    for key in ("min_p", "logit_bias", "min_tokens"):
                        if key in body and key in outer.faults.get("param_status", {}):
                            code = outer.faults["param_status"][key]
                            return self._json(code, {"error": {"message": f"{key} is not supported with speculative decoding"}} if code == 400 else {"error": "x"})
                    if body.get("tools"):
                        return self._tools(body)
                    if body.get("stream"):
                        return self._stream_text(body)
                    time.sleep(outer.faults.get("delay", 0.06))
                    self._json(200, {"choices": [{"message": {"role": "assistant", "content": outer.reply(body, me in outer.crowded)}, "finish_reason": "stop"}],
                                     "usage": {"completion_tokens": 5, "prompt_tokens": 7}})
                finally:
                    with outer.lock:
                        outer.in_flight -= 1
                        outer.active.discard(me)

            def _stream_text(self, body):
                if outer.faults.get("no_stream"):
                    return self._json(200, {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for i in range(int(body.get("max_tokens", 50))):
                        chunk = {"choices": [{"delta": {"content": f"w{i} "}, "index": 0}]}
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                        self.wfile.flush()
                        time.sleep(0.02)
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    with outer.lock:
                        outer.cancelled += 1
                        if outer.faults.get("cancel_breaks"):
                            outer.broken = True
                        if outer.faults.get("unhealthy_after_cancel"):
                            outer.unhealthy = True
                        if outer.faults.get("stale_soon_after_cancel"):
                            outer.stale_after = time.monotonic() + outer.faults["stale_soon_after_cancel"]

            def _tools(self, body):
                f = outer.faults
                args = dict(TOOL_ARGS)
                if f.get("tool_differs") and body.get("stream"):
                    args = {"city": "Rome", "unit": "celsius"}
                if f.get("no_tool"):
                    return self._json(200, {"choices": [{"message": {"role": "assistant", "content": "I do not need a tool"}, "finish_reason": "stop"}]})
                if not body.get("stream"):
                    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(TOOL_ARGS)}}
                    return self._json(200, {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                text = json.dumps(args)
                if f.get("stream_no_tool"):
                    self.wfile.write(b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n')
                    return
                if f.get("tool_bad_json"):
                    pieces = ["{not json"]
                elif f.get("tool_missing_key"):
                    pieces = ['{"unit": "celsius"}']
                elif f.get("tool_empty"):
                    pieces = ["{}"]
                elif f.get("tool_one_event"):
                    pieces = [text]
                else:
                    pieces = [text[:9], text[9:20], text[20:]]
                tool_name = "get_time" if f.get("tool_name_differs") else "get_weather"
                events = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": tool_name, "arguments": ""}}]}}]}]
                events += [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": p}}]}}]} for p in pieces]
                events.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
                for e in events:
                    self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def reply(self, body, crowded):
        prompt = body["messages"][-1]["content"]
        text = "echo " + hashlib.sha256(prompt.encode()).hexdigest()[:10]
        f = self.faults
        if f.get("same_for_all"):
            return "echo constant"
        if f.get("nondeterministic"):
            return text + f" {time.monotonic_ns()}"
        if f.get("crosstalk") and crowded:
            return text + " (mixed)"
        if f.get("spec_differs") and "plain" not in body:
            return text + " spec"
        if self.stale_after is not None and time.monotonic() > self.stale_after:
            return text + " stale"
        if f.get("sticky") and self.calls > f["sticky"]:
            return text + " stale"
        return text

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def make():
    made = []

    def build(**faults):
        s = Server(**faults)
        made.append(s)
        return s
    yield build
    for s in made:
        s.close()


# ---- the judges, on text alone ------------------------------------------------------------------------------------

def test_the_judge_passes_when_every_concurrent_reply_equals_its_alone_reply():
    r = sp.judge_concurrent_equals_alone(["a", "b", "c"], ["a", "b"], ["a", "b", "c"])
    assert r["state"] == "pass" and r["advisory"] is True and r["name"] == "concurrent_equals_alone"


def test_the_judge_names_each_request_that_changed():
    r = sp.judge_concurrent_equals_alone(["a", "b", "c", "d"], ["a", "b"], ["a", "X", "c", "Y"])
    assert r["state"] == "fail" and r["differing"] == [1, 3] and "2 of 4" in r["detail"]
    assert "'X'" in r["detail"] and "'b'" in r["detail"]


def test_a_server_that_does_not_repeat_itself_alone_gets_no_verdict():
    r = sp.judge_concurrent_equals_alone(["a", "b", "c"], ["a", "Z"], ["a", "b", "c"])
    assert r["state"] == "not_checked" and "repeat" in r["detail"]


def test_the_same_reply_for_every_prompt_gets_no_verdict():
    r = sp.judge_concurrent_equals_alone(["same", "same", "same"], ["same", "same"], ["same", "same", "same"])
    assert r["state"] == "not_checked" and "different" in r["detail"]


@pytest.mark.parametrize("alone,repeat,conc", [([None, "b"], ["x"], ["a", "b"]), (["a", "b"], ["a"], [None, "b"]), (["a", "b"], [None], ["a", "b"])])
def test_a_missing_reply_is_a_fail_for_the_concurrent_ones_and_no_verdict_for_the_alone_ones(alone, repeat, conc):
    r = sp.judge_concurrent_equals_alone(alone, repeat, conc)
    assert r["state"] in ("fail", "not_checked")
    assert r["state"] == ("fail" if None in conc else "not_checked")
    if None in alone or None in repeat:
        assert "asked alone did not get a reply" in r["detail"]        # named for what it is, not as a repeat failure


def test_the_judge_clips_a_long_reply_in_its_detail():
    r = sp.judge_concurrent_equals_alone(["a" * 500, "b"], ["a" * 500], ["c" * 500, "b"])
    assert len(r["detail"]) < 600


def test_the_aba_judge_needs_the_reference_to_repeat_and_both_later_replies_to_match():
    assert sp.judge_aba("r", "r", "r", "r")["state"] == "pass"
    f = sp.judge_aba("r", "r", "x", "r")
    assert f["state"] == "fail" and "under load" in f["detail"]
    f = sp.judge_aba("r", "r", "r", "x")
    assert f["state"] == "fail" and "afterwards" in f["detail"]
    assert sp.judge_aba("r", "q", "r", "r")["state"] == "not_checked"
    assert sp.judge_aba(None, "r", "r", "r")["state"] == "not_checked"
    f = sp.judge_aba("r", "r", None, "r")
    assert f["state"] == "fail" and "no reply while other streams" in f["detail"]
    f = sp.judge_aba("r", "r", "r", None)
    assert f["state"] == "fail" and "no reply when it was asked alone again" in f["detail"]


def test_the_cancel_judge_needs_a_healthy_server_and_an_unchanged_reply():
    assert sp.judge_after_cancel("r", "r", ["r", "r"], True)["state"] == "pass"
    assert sp.judge_after_cancel("r", "r", ["r", "x"], True)["state"] == "fail"       # the settled reply counts as much as the immediate one
    assert sp.judge_after_cancel("r", "r", ["x", "r"], True)["state"] == "fail"
    assert sp.judge_after_cancel("r", "r", ["r", None], True)["state"] == "fail"
    assert sp.judge_after_cancel("r", "r", "r", True)["state"] == "pass"
    none = sp.judge_after_cancel("r", "r", ["r", None], True)
    assert none["state"] == "fail" and "no reply after the stream was cut" in none["detail"]
    assert sp.judge_after_cancel("r", "r", "x", True)["state"] == "fail"
    assert sp.judge_after_cancel("r", "r", "r", False)["state"] == "fail"
    assert sp.judge_after_cancel("r", "r", None, True)["state"] == "fail"
    assert sp.judge_after_cancel("r", "q", "r", True)["state"] == "not_checked"


def test_the_speculation_judge_compares_each_prompt_with_and_without():
    assert sp.judge_spec_equals_plain(["a", "b"], ["a", "b"], ["a", "b"])["state"] == "pass"
    r = sp.judge_spec_equals_plain(["a", "X"], ["a", "b"], ["a", "X"])
    assert r["state"] == "fail" and r["differing"] == [1]
    assert sp.judge_spec_equals_plain(["a", "b"], ["a", "b"], ["a", "Q"])["state"] == "not_checked"
    assert sp.judge_spec_equals_plain(["a", None], ["a", "b"], ["a", None])["state"] == "not_checked"


# ---- concurrent_equals_alone against the fake ---------------------------------------------------------------------

def test_a_server_that_answers_each_prompt_the_same_alone_and_together_passes(make):
    s = make()
    r = sp.probe_concurrent_equals_alone(s.base, "m", n=6, timeout=30)
    assert r["state"] == "pass", r
    assert r["requests"] == 6


def test_crosstalk_under_load_is_a_fail_that_names_the_requests(make):
    s = make(crosstalk=True)
    r = sp.probe_concurrent_equals_alone(s.base, "m", n=6, timeout=30)
    assert r["state"] == "fail" and r["differing"], r


def test_a_nondeterministic_server_gets_no_verdict(make):
    r = sp.probe_concurrent_equals_alone(make(nondeterministic=True).base, "m", n=4, timeout=30)
    assert r["state"] == "not_checked" and "repeat" in r["detail"], r


def test_a_server_that_ignores_the_prompt_gets_no_verdict(make):
    r = sp.probe_concurrent_equals_alone(make(same_for_all=True).base, "m", n=4, timeout=30)
    assert r["state"] == "not_checked", r


@pytest.mark.parametrize("n", [0, 1, -3, True, 2.5, "4", None])
def test_fewer_than_two_requests_or_a_bad_count_is_not_checked_and_sends_nothing(make, n):
    s = make()
    r = sp.probe_concurrent_equals_alone(s.base, "m", n=n, timeout=5)
    assert r["state"] == "not_checked" and s.calls == 0


def test_the_burst_is_capped(make):
    s = make(delay=0.0)
    r = sp.probe_concurrent_equals_alone(s.base, "m", n=500, timeout=60)
    assert r["requests"] == sp.MAX_CONCURRENT and s.calls <= sp.MAX_CONCURRENT * 2 + sp.CONTROL_REPEATS


def test_requests_are_greedy_bounded_and_carry_the_models_extra(make):
    s = make(delay=0.0)
    sp.probe_concurrent_equals_alone(s.base, "the-model", {"seed": 7}, n=3, max_tokens=11, timeout=30)
    assert s.seen_bodies and all(b["temperature"] == 0 and b["max_tokens"] == 11 and b["model"] == "the-model" and b["seed"] == 7 for b in s.seen_bodies)


def test_a_server_that_answers_500_under_load_is_a_fail(make):
    r = sp.probe_concurrent_equals_alone(make(overload500=True).base, "m", n=5, timeout=30)
    assert r["state"] == "fail" and "no reply" in r["detail"], r


def test_a_dead_server_is_not_checked_not_a_crash():
    r = sp.probe_concurrent_equals_alone("http://127.0.0.1:1", "m", n=3, timeout=2)
    assert r["state"] == "not_checked" and r["advisory"] is True


# ---- aba ----------------------------------------------------------------------------------------------------------

def test_aba_passes_on_a_clean_server_and_fails_on_a_server_that_goes_stale(make):
    s = make()
    assert sp.probe_aba_isolation(s.base, "m", others=3, timeout=30)["state"] == "pass"
    stale = make(sticky=6)
    r = sp.probe_aba_isolation(stale.base, "m", others=3, timeout=30)
    assert r["state"] == "fail", r


def test_aba_fails_when_a_busy_server_mixes_the_reply(make):
    r = sp.probe_aba_isolation(make(crosstalk=True).base, "m", others=4, timeout=30)
    assert r["state"] == "fail" and "under load" in r["detail"], r


@pytest.mark.parametrize("others", [0, -1, True, 1.5, "3", None, 99])
def test_aba_bounds_the_number_of_other_streams(make, others):
    s = make(delay=0.0)
    r = sp.probe_aba_isolation(s.base, "m", others=others, timeout=30)
    if others in (0, -1, True, 1.5, "3", None):
        assert r["state"] == "not_checked" and s.calls == 0
    else:
        assert s.calls <= sp.MAX_CONCURRENT + 6


# ---- cancel mid-decode --------------------------------------------------------------------------------------------

def test_a_cancel_mid_decode_leaves_the_next_request_exact(make):
    s = make()
    r = sp.probe_cancel_mid_decode(s.base, "m", settle=0.5, timeout=30)
    assert r["state"] == "pass", r
    assert s.models_gets >= 1                       # the probe looked at the server's health afterwards


def test_a_server_that_breaks_after_a_cancel_is_a_fail(make):
    s = make(cancel_breaks=True)
    r = sp.probe_cancel_mid_decode(s.base, "m", settle=0.5, timeout=30)
    assert r["state"] == "fail", r


def test_a_server_that_does_not_stream_is_not_checked(make):
    r = sp.probe_cancel_mid_decode(make(no_stream=True).base, "m", settle=0.0, timeout=30)
    assert r["state"] == "not_checked" and "stream" in r["detail"]


def test_the_cancel_really_closes_the_stream_early(make):
    s = make(delay=0.0)
    sp.probe_cancel_mid_decode(s.base, "m", settle=0.3, timeout=30)
    time.sleep(0.5)
    assert s.cancelled >= 1


# ---- speculation on against off -----------------------------------------------------------------------------------

def test_speculation_on_equals_off_passes_and_names_the_differing_prompts_otherwise(make):
    s = make(delay=0.0)
    r = sp.probe_spec_equals_plain(s.base, "m", spec_extra={}, plain_extra={"plain": True}, timeout=30)
    assert r["state"] == "pass", r
    r = sp.probe_spec_equals_plain(make(spec_differs=True, delay=0.0).base, "m", spec_extra={}, plain_extra={"plain": True}, timeout=30)
    assert r["state"] == "fail" and r["differing"], r


def test_without_a_per_request_switch_the_probe_does_not_guess(make):
    s = make()
    r = sp.probe_spec_equals_plain(s.base, "m", spec_extra=None, plain_extra=None, timeout=30)
    assert r["state"] == "not_checked" and "switch" in r["detail"] and s.calls == 0


def test_the_two_extras_must_differ_or_the_probe_compares_a_server_with_itself(make):
    s = make()
    r = sp.probe_spec_equals_plain(s.base, "m", spec_extra={"a": 1}, plain_extra={"a": 1}, timeout=30)
    assert r["state"] == "not_checked" and s.calls == 0


# ---- streamed tool calls ------------------------------------------------------------------------------------------

def test_a_streamed_tool_call_with_its_arguments_equals_the_plain_one(make):
    r = sp.probe_streamed_tool_call(make().base, "m", timeout=30)
    assert r["state"] == "pass", r


def test_a_tool_call_that_arrives_whole_in_one_event_still_passes(make):
    assert sp.probe_streamed_tool_call(make(tool_one_event=True).base, "m", timeout=30)["state"] == "pass"


def test_a_streamed_tool_call_with_empty_arguments_is_a_fail_that_says_so(make):
    r = sp.probe_streamed_tool_call(make(tool_empty=True).base, "m", timeout=30)
    assert r["state"] == "fail" and "{}" in r["detail"], r


def test_streamed_arguments_that_differ_from_the_plain_ones_are_a_fail(make):
    r = sp.probe_streamed_tool_call(make(tool_differs=True).base, "m", timeout=30)
    assert r["state"] == "fail" and "differ" in r["detail"], r


def test_a_model_that_calls_no_tool_gets_no_verdict(make):
    r = sp.probe_streamed_tool_call(make(no_tool=True).base, "m", timeout=30)
    assert r["state"] == "not_checked", r


def test_tool_calls_are_collected_by_index_from_fragments():
    events = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f", "arguments": ""}},
                                                      {"index": 1, "id": "b", "function": {"name": "g", "arguments": "{"}}]}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "}"}}, {"index": 0, "function": {"arguments": "{\"x\": 1}"}}]}}]}]
    assert sp.tool_calls_of_stream(events) == [{"name": "f", "arguments": "{\"x\": 1}"}, {"name": "g", "arguments": "{}"}]


@pytest.mark.parametrize("events", [[], [{}], [{"choices": []}], [{"choices": [{"delta": None}]}], [{"choices": [{"delta": {"tool_calls": "x"}}]}],
                                    [{"choices": [{"delta": {"tool_calls": [{"index": "0", "function": {"arguments": "{}"}}]}}]}],
                                    [{"choices": [{"delta": {"tool_calls": [{"index": 10**9, "function": {"arguments": "{}"}}]}}]}],
                                    [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 5}}]}}]}],
                                    ["not a dict", 5, None]])
def test_malformed_stream_events_give_no_calls_and_no_crash(events):
    assert sp.tool_calls_of_stream(events) == []


def test_the_stream_reader_stops_at_its_event_cap_and_closes(make):
    s = make(delay=0.0)
    status, events, done = sp.stream_events(s.base + "/v1/chat/completions", {"model": "m", "stream": True, "max_tokens": 300,
                                                                                 "messages": [{"role": "user", "content": "x"}]}, 10, max_events=3)
    assert status == 200 and len(events) == 3 and done is False


def test_the_stream_reader_reports_a_refusal_without_events(make):
    s = make(delay=0.0)
    status, events, done = sp.stream_events(s.base + "/nope", {}, 5)
    assert status == 404 and events == []


# ---- sampling options under speculation ---------------------------------------------------------------------------

def test_a_400_that_names_the_option_is_the_documented_answer_and_passes(make):
    s = make(param_status={"min_p": 400, "logit_bias": 400, "min_tokens": 400})
    r = sp.probe_sampling_params_under_speculation(s.base, "m", timeout=30)
    assert r["state"] == "pass" and r["statuses"] == {"min_p": 400, "logit_bias": 400, "min_tokens": 400}, r
    assert "speculative" in r["label"].lower()


def test_a_200_is_also_listed_and_passes(make):
    r = sp.probe_sampling_params_under_speculation(make().base, "m", timeout=30)
    assert r["state"] == "pass" and set(r["statuses"].values()) == {200}


def test_a_500_is_a_fail(make):
    r = sp.probe_sampling_params_under_speculation(make(param_status={"min_p": 500}).base, "m", timeout=30)
    assert r["state"] == "fail" and "min_p" in r["detail"], r


def test_the_probe_names_each_option_in_the_request(make):
    s = make()
    sp.probe_sampling_params_under_speculation(s.base, "m", timeout=30)
    keys = [k for b in s.seen_bodies for k in ("min_p", "logit_bias", "min_tokens") if k in b]
    assert sorted(keys) == ["logit_bias", "min_p", "min_tokens"]


# ---- more guards --------------------------------------------------------------------------------------------------

def test_a_server_that_is_unhealthy_after_a_cancel_is_a_fail_even_when_the_replies_are_right(make):
    s = make(unhealthy_after_cancel=True)
    r = sp.probe_cancel_mid_decode(s.base, "m", settle=0.5, timeout=30)
    assert r["state"] == "fail" and "health" in r["detail"], r


def test_the_other_streams_of_the_aba_probe_run_longer_than_the_request_between_them(make):
    s = make(delay=0.0)
    sp.probe_aba_isolation(s.base, "m", others=3, max_tokens=10, timeout=30)
    sizes = sorted({b["max_tokens"] for b in s.seen_bodies})
    assert sizes == [10, 20], sizes


def test_a_stream_without_a_tool_call_when_the_plain_answer_has_one_is_a_fail(make):
    r = sp.probe_streamed_tool_call(make(stream_no_tool=True).base, "m", timeout=30)
    assert r["state"] == "fail" and "no tool call" in r["detail"], r


def test_arguments_that_are_not_json_are_a_fail(make):
    r = sp.probe_streamed_tool_call(make(tool_bad_json=True).base, "m", timeout=30)
    assert r["state"] == "fail" and "not a JSON object" in r["detail"], r


def test_a_missing_required_argument_is_a_fail_that_names_it(make):
    r = sp.probe_streamed_tool_call(make(tool_missing_key=True).base, "m", timeout=30)
    assert r["state"] == "fail" and "missing the required city" in r["detail"], r


def test_the_tool_request_forces_the_tool_and_carries_a_schema_only(make):
    s = make()
    sp.probe_streamed_tool_call(s.base, "m", timeout=30)
    assert len(s.seen_bodies) == 2 and s.seen_bodies[0].get("stream") is None and s.seen_bodies[1]["stream"] is True
    b = s.seen_bodies[0]
    assert b["tool_choice"]["function"]["name"] == "get_weather" and b["tools"][0]["function"]["parameters"]["required"] == ["city"]


@pytest.mark.parametrize("status,state", [(422, "pass"), (404, "fail"), (401, "fail"), (503, "fail"), (200, "pass")])
def test_only_a_200_a_400_or_a_422_is_a_clean_answer_to_the_sampling_options(make, status, state):
    r = sp.probe_sampling_params_under_speculation(make(param_status={"logit_bias": status}).base, "m", timeout=30)
    assert r["state"] == state and r["statuses"]["logit_bias"] == status, r


def test_the_stream_reader_stops_at_its_byte_cap(make, monkeypatch):
    s = make(delay=0.0)
    monkeypatch.setattr(sp, "MAX_STREAM_BYTES", 400)
    status, events, done = sp.stream_events(s.base + "/v1/chat/completions", {"model": "m", "stream": True, "max_tokens": 200,
                                                                               "messages": [{"role": "user", "content": "x"}]}, 20)
    assert status == 200 and 0 < len(events) < 20 and done is False


def test_the_stream_reader_skips_lines_that_are_not_json_objects():
    import io
    import urllib.request

    class R(io.BytesIO):
        status = 200

    raw = b'data: not json\n\ndata: [1, 2]\n\ndata: "x"\n\n: comment\n\ndata: {"a": 1}\n\ndata: [DONE]\n\n'

    def fake_open(req, timeout=None):
        return R(raw)
    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_open
    try:
        status, events, done = sp.stream_events("http://127.0.0.1:1/x", {}, 5)
    finally:
        urllib.request.urlopen = orig
    assert events == [{"a": 1}] and done is True and status == 200


def test_tool_arguments_are_capped_per_call(monkeypatch):
    monkeypatch.setattr(sp, "MAX_TOOL_ARGS_CHARS", 10)
    frags = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "0123456789"}}]}}]},
             {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "X"}}]}}]}]
    assert sp.tool_calls_of_stream(frags) == [{"name": "f", "arguments": "0123456789"}]


def test_the_first_name_of_a_call_is_kept_and_an_index_at_the_limit_is_skipped():
    frags = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "first", "arguments": "{"}}]}}]},
             {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "second", "arguments": "}"}},
                                                    {"index": sp.MAX_TOOL_INDEX, "function": {"name": "x", "arguments": "{}"}},
                                                    {"index": sp.MAX_TOOL_INDEX - 1, "function": {"name": "y", "arguments": "{}"}},
                                                    {"index": True, "function": {"name": "z", "arguments": "{}"}},
                                                    {"index": -1, "function": {"name": "w", "arguments": "{}"}}]}}]}]
    assert sp.tool_calls_of_stream(frags) == [{"name": "first", "arguments": "{}"}, {"name": "y", "arguments": "{}"}]


def test_the_settled_request_catches_a_server_that_goes_wrong_a_moment_after_the_abort(make):
    """A server whose replies are right at once and wrong 0.3 s after the cancel: only the request sent after the settle time sees it."""
    s = make(stale_soon_after_cancel=0.3)
    r = sp.probe_cancel_mid_decode(s.base, "m", settle=0.8, timeout=30)
    assert r["state"] == "fail" and "changed after the stream was cut" in r["detail"], r


def test_a_streamed_tool_call_with_another_name_is_a_fail(make):
    r = sp.probe_streamed_tool_call(make(tool_name_differs=True).base, "m", timeout=30)
    assert r["state"] == "fail" and "differs" in r["detail"], r


HOSTILE_LINES = [b'data: {"a": NaN}', b'data: {"a": Infinity}', b'data: {"a": -Infinity}', b'data: {"a": 1e999}', b'data: {"a": 1, "a": 2}',
                 b'data: ' + b'[' * 200 + b']' * 200, b'data: {"a": "\\ud800"}', b'data: {"a": ' + b"9" * 400 + b"}",
                 b'data: {"a": 1} trailing', b'data: \xff\xfe']


@pytest.mark.parametrize("line", HOSTILE_LINES)
def test_a_hostile_stream_line_is_skipped_and_the_good_line_after_it_still_counts(line):
    import io
    import urllib.request

    class R(io.BytesIO):
        status = 200

    raw = line + b'\n\ndata: {"ok": 1}\n\ndata: [DONE]\n\n'
    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=None: R(raw)
    try:
        status, events, done = sp.stream_events("http://127.0.0.1:1/x", {}, 5)
    finally:
        urllib.request.urlopen = orig
    assert events == [{"ok": 1}] and done is True


@pytest.mark.parametrize("args", ['{"city": NaN}', '{"city": Infinity}', '{"city": "Paris", "city": "Rome"}', '{"city": 1e999}', '[1, 2]', '"x"', "7"])
def test_tool_arguments_that_are_not_a_strict_json_object_are_a_fail(args):
    plain = [{"name": "get_weather", "arguments": '{"city": "Paris"}'}]
    r = sp.judge_streamed_tool_call([{"name": "get_weather", "arguments": args}], plain)
    assert r["state"] == "fail" and "not a JSON object" in r["detail"], r


def test_the_module_parses_no_json_itself_so_the_hostile_reader_registry_needs_no_entry():
    from pathlib import Path
    text = (Path(sp.__file__)).read_text()
    assert "json.loads(" not in text and "json.load(" not in text


def test_the_json_object_reader_enforces_its_size_bound_and_the_object_type():
    assert sp._json_object('{"a": 1}', 8) == {"a": 1}
    assert sp._json_object('{"a": 12}', 8) is None            # nine bytes against a bound of eight
    assert sp._json_object("[1]", 100) is None and sp._json_object("", 100) is None and sp._json_object("{", 100) is None
