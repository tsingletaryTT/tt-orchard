"""telegraph: an opt-in call to a model outside the box, with caps, a cache and a measurement for every call.

Nothing here reaches a network: the transport is a fake that records each request."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from orchard import telegraph as tg

REPO = Path(tg.__file__).resolve().parents[1]
KEY = "k3y-VALUE-0123456789"


class Wire:
    """A fake transport. `reply` is a function of the decoded request body; every request is kept."""

    def __init__(self, reply=None, status=200):
        self.requests, self.status = [], status
        self.reply = reply or (lambda body: openai_reply("use ttl.block.fill for constants", 120, 30))

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": json.loads(body), "timeout": timeout})
        out = self.reply(self.requests[-1]["body"])
        return self.status, (out if isinstance(out, bytes) else json.dumps(out).encode())


def openai_reply(text, tin, tout):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout}}


def anthropic_reply(text, tin, tout):
    return {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": tin, "output_tokens": tout}}


def table(**kw):
    base = {"enabled": True, "endpoint": "https://llm.example.org/v1", "models": ["small", "big"],
            "key_env": "TG_KEY"}
    base.update(kw)
    return base


def make(tmp_path, wire=None, environ=None, now=None, **kw):
    cfg = tg.load(table(**kw), base=tmp_path)
    wire = wire or Wire()
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=wire, environ={"TG_KEY": KEY} if environ is None else environ,
                       clock=now or (lambda: 1000.0), hostname="node9", home="/home/someone")
    return tel, wire


def records(tmp_path):
    """The ledger's call, refusal and report lines. Budget reservations are internal and are left out."""
    p = tmp_path / "tg" / "telegraph.jsonl"
    rows = []
    for line in (p.read_text().splitlines() if p.exists() else []):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "reserve" not in row:
            rows.append(row)
    return rows


def worst_cost(body, price, max_out):
    """The ceiling telegraph reserves for a request: its size in bytes plus framing as input, max_out as output."""
    return ((len(json.dumps(body).encode()) + tg.REQUEST_FRAMING_TOKENS) * price[0] + max_out * price[1]) / 1e6


def request_body_of(tmp_path, **kw):
    """The body telegraph sends for ("p", "t", "f1") under these settings, taken from an uncapped run."""
    probe, wire = make(tmp_path / "probe", **kw)
    probe.ask("p", "t", "f1")
    return wire.requests[0]["body"]


# ------------------------------------------------------------------------------------------------ configuration

def test_it_is_off_by_default_and_a_missing_table_is_off():
    assert tg.load(None, base=Path("/x")).enabled is False
    assert tg.load({}, base=Path("/x")).enabled is False


def test_an_unknown_key_is_refused_with_a_suggestion():
    with pytest.raises(tg.ConfigError, match="unknown key 'modles'.*models"):
        tg.load(table(modles=["a"]), base=Path("/x"))


@pytest.mark.parametrize("bad, text", [
    ({"endpoint": ""}, "endpoint"),
    ({"endpoint": "ftp://x/v1"}, "https"),
    ({"endpoint": "http://llm.example.org/v1"}, "https"),
    ({"endpoint": "https://user:pw@llm.example.org/v1"}, "credentials in the endpoint"),
    ({"models": []}, "models"),
    ({"models": ["a", "a"]}, "twice"),
    ({"models": "a"}, "list"),
    ({"api": "grpc"}, "api"),
    ({"key_env": "TG_KEY", "key_file": "k"}, "only one"),
    ({"after_failures": 0}, "after_failures"),
    ({"max_calls": -1}, "max_calls"),
    ({"max_calls": True}, "max_calls"),
    ({"max_cost_usd": 1.0}, "prices"),
    ({"max_prompt_chars": 100}, "max_prompt_chars"),
    ({"timeout_s": 0}, "timeout_s"),
])
def test_a_bad_setting_is_refused_and_named(bad, text):
    with pytest.raises(tg.ConfigError, match=text):
        tg.load(table(**bad), base=Path("/x"))


@pytest.mark.parametrize("url", ["http://127.0.0.1:4000/v1", "http://localhost:4000/v1", "http://192.168.1.20:4000/v1",
                                 "http://10.1.2.3/v1", "http://gpu-box:8000/v1"])
def test_plain_http_is_allowed_only_on_the_machine_or_the_private_network(url):
    assert tg.load(table(endpoint=url), base=Path("/x")).endpoint == url


def test_a_key_file_inside_the_checkout_is_refused_and_one_outside_is_resolved_against_the_config(tmp_path):
    with pytest.raises(tg.ConfigError, match="inside the checkout"):
        tg.load(table(key_env=None, key_file=str(REPO / "config" / "key")), base=tmp_path)
    cfg = tg.load(table(key_env=None, key_file="keys/k"), base=tmp_path)
    assert cfg.key_file == (tmp_path / "keys" / "k")


def test_defaults_are_small_numbers():
    cfg = tg.load(table(), base=Path("/x"))
    assert (cfg.after_failures, cfg.max_calls, cfg.max_output_tokens, cfg.max_prompt_chars) == (2, 5, 1500, 12000)
    assert cfg.max_total_tokens == 40000 and cfg.max_cost_usd is None and cfg.api == "openai"


def test_a_disabled_table_is_not_checked_for_the_rest():
    assert tg.load({"enabled": False, "endpoint": ""}, base=Path("/x")).enabled is False


# ------------------------------------------------------------------------------------------------ the credential

def test_the_key_comes_from_the_environment_at_call_time(tmp_path):
    tel, wire = make(tmp_path)
    assert tel.ask("p", "task", "failure").ok
    assert wire.requests[0]["headers"]["Authorization"] == f"Bearer {KEY}"


def test_the_key_comes_from_a_file_and_a_trailing_newline_is_dropped(tmp_path):
    keyfile = tmp_path / "k"
    keyfile.write_text(KEY + "\n")
    tel, wire = make(tmp_path, key_env=None, key_file=str(keyfile), environ={})
    assert tel.ask("p", "t", "f").ok
    assert wire.requests[0]["headers"]["Authorization"] == f"Bearer {KEY}"


def test_a_missing_key_is_a_refusal_with_no_request_and_no_traceback(tmp_path):
    tel, wire = make(tmp_path, environ={})
    got = tel.ask("p", "t", "f")
    assert not got.ok and "TG_KEY" in got.reason and wire.requests == []
    assert records(tmp_path)[0]["outcome"] == "no-key"


def test_an_empty_or_unreadable_key_file_is_a_refusal(tmp_path):
    keyfile = tmp_path / "k"
    keyfile.write_text("  \n")
    tel, wire = make(tmp_path, key_env=None, key_file=str(keyfile), environ={})
    assert not tel.ask("p", "t", "f").ok and wire.requests == []
    tel2, _ = make(tmp_path, key_env=None, key_file=str(tmp_path / "absent"), environ={})
    assert "cannot read" in tel2.ask("p", "t", "f").reason


def test_the_key_never_reaches_the_prompt_the_ledger_the_cache_or_the_reason(tmp_path):
    wire = Wire(lambda body: openai_reply(f"the answer mentions {KEY}", 10, 5))
    tel, wire = make(tmp_path, wire=wire)
    tel.ask("p", f"task with {KEY} inside", f"failure Bearer {KEY}")
    for path in (tmp_path / "tg").rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text(), path
    assert KEY not in json.dumps(wire.requests[0]["body"])


def test_a_keyless_endpoint_sends_no_authorization_header(tmp_path):
    tel, wire = make(tmp_path, endpoint="http://127.0.0.1:4000/v1", key_env=None, environ={})
    assert tel.ask("p", "t", "f").ok
    assert "Authorization" not in wire.requests[0]["headers"] and "x-api-key" not in wire.requests[0]["headers"]


# ------------------------------------------------------------------------------------------------ when to ask

@pytest.mark.parametrize("failures, want", [(0, False), (1, False), (2, True), (7, True)])
def test_it_asks_only_after_the_configured_number_of_repeated_failures(tmp_path, failures, want):
    tel, _ = make(tmp_path)
    assert tel.should_ask(failures) is want


def test_a_disabled_telegraph_never_asks_and_never_touches_the_network(tmp_path):
    wire = Wire()
    tel = tg.Telegraph(tg.load(None, base=tmp_path), tmp_path / "tg", transport=wire, environ={})
    assert tel.should_ask(99) is False
    got = tel.ask("p", "t", "f")
    assert not got.ok and got.reason == "telegraph is not enabled" and wire.requests == []
    assert not (tmp_path / "tg").exists()


# ------------------------------------------------------------------------------------------------ the request

def test_the_openai_request_is_small_and_greedy(tmp_path):
    tel, wire = make(tmp_path, max_output_tokens=900)
    tel.ask("rmsnorm", "write a kernel", "Deadlock detected")
    req = wire.requests[0]
    assert req["url"] == "https://llm.example.org/v1/chat/completions"
    body = req["body"]
    assert body["model"] == "small" and body["max_tokens"] == 900 and body["temperature"] == 0
    assert body["messages"][0]["role"] == "system" and body["messages"][-1]["role"] == "user"
    assert "write a kernel" in body["messages"][-1]["content"] and "Deadlock detected" in body["messages"][-1]["content"]
    assert req["timeout"] == 120


def test_the_anthropic_request_uses_its_own_path_and_headers(tmp_path):
    wire = Wire(lambda body: anthropic_reply("try fill", 50, 10))
    tel, wire = make(tmp_path, wire=wire, api="anthropic", endpoint="https://llm.example.org")
    got = tel.ask("p", "t", "f")
    req = wire.requests[0]
    assert got.ok and got.text == "try fill" and (got.tokens_in, got.tokens_out) == (50, 10)
    assert req["url"] == "https://llm.example.org/v1/messages"
    assert req["headers"]["x-api-key"] == KEY and req["headers"]["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in req["headers"]
    assert req["body"]["system"] and req["body"]["messages"][0]["role"] == "user" and req["body"]["max_tokens"] == 1500


def test_the_prompt_is_redacted_of_paths_host_and_tokens(tmp_path):
    tel, wire = make(tmp_path)
    text = ("failed at /home/someone/orchard-runs/x/kernel.py on node9 with hf_" + "a" * 34 +
            " and ghp_" + "b" * 36 + " and Authorization: Bearer abcdef123456")
    tel.ask("p", text, "-----BEGIN PRIVATE KEY-----")
    sent = wire.requests[0]["body"]["messages"][-1]["content"]
    for leak in ("/home/someone", "node9", "hf_" + "a" * 34, "ghp_" + "b" * 36, "abcdef123456", "PRIVATE KEY"):
        assert leak not in sent, leak
    assert "orchard-runs/x/kernel.py" in sent


def test_a_long_prompt_is_cut_to_the_cap_keeping_the_start_and_the_end(tmp_path):
    tel, wire = make(tmp_path, max_prompt_chars=1000)
    tel.ask("p", "HEAD" + "x" * 5000, "y" * 5000 + "TAIL")
    sent = wire.requests[0]["body"]["messages"][-1]["content"]
    assert len(sent) <= 1000 and "HEAD" in sent and "TAIL" in sent and "[cut" in sent


# ------------------------------------------------------------------------------------------------ the measurement

def test_every_call_leaves_one_record_with_the_numbers(tmp_path):
    ticks = iter([1000.0, 1002.5])
    tel, _ = make(tmp_path, now=lambda: next(ticks), prices={"small": [1.0, 3.0], "big": [5.0, 15.0]})
    got = tel.ask("rmsnorm", "task", "failure")
    rec = records(tmp_path)[0]
    assert got.call_id == rec["call_id"] and rec["problem"] == "rmsnorm" and rec["model"] == "small"
    assert (rec["tokens_in"], rec["tokens_out"], rec["outcome"], rec["cached"]) == (120, 30, "ok", False)
    assert rec["latency_s"] == 2.5 and rec["cost_usd"] == pytest.approx(120e-6 + 90e-6)
    assert rec["prompt_chars"] > 0 and rec["prompt_sha256"] and rec["reply_file"]
    assert (tmp_path / "tg" / rec["reply_file"]).read_text() == "use ttl.block.fill for constants"
    assert "task" not in json.dumps(rec) and "failure" not in json.dumps({k: v for k, v in rec.items() if k != "problem"})


def test_a_model_with_no_price_has_an_unknown_cost_not_zero(tmp_path):
    tel, _ = make(tmp_path)
    tel.ask("p", "t", "f")
    assert records(tmp_path)[0]["cost_usd"] is None


def test_the_outcome_is_reported_back_and_summarised(tmp_path):
    tel, _ = make(tmp_path)
    first = tel.ask("p", "t1", "f1")
    second = tel.ask("p", "t2", "f2")
    tel.report(first.call_id, helped=False, note="same deadlock")
    tel.report(second.call_id, helped=True)
    s = tg.summarise(tmp_path / "tg")
    assert (s["calls"], s["ok"], s["helped"], s["not_helped"], s["unreported"]) == (2, 2, 1, 1, 0)
    assert s["tokens_in"] == 240 and s["tokens_out"] == 60 and s["by_model"]["small"]["calls"] == 2
    assert s["by_problem"]["p"] == {"calls": 2, "helped": 1}


def test_reporting_an_unknown_call_is_refused(tmp_path):
    tel, _ = make(tmp_path)
    with pytest.raises(KeyError):
        tel.report("nope", helped=True)


def test_the_summary_of_an_empty_directory_is_zeros(tmp_path):
    s = tg.summarise(tmp_path / "none")
    assert s["calls"] == 0 and s["cost_usd"] == 0.0 and s["cost_unknown_calls"] == 0


def test_a_torn_last_line_does_not_stop_the_summary(tmp_path):
    tel, _ = make(tmp_path)
    tel.ask("p", "t", "f")
    with open(tmp_path / "tg" / "telegraph.jsonl", "a") as f:
        f.write('{"call_id": "half')
    assert tg.summarise(tmp_path / "tg")["calls"] == 1


# ------------------------------------------------------------------------------------------------ cost control

def test_a_repeated_question_is_answered_from_the_cache_with_no_request(tmp_path):
    tel, wire = make(tmp_path)
    a = tel.ask("p", "same task", "same failure")
    b = tel.ask("p", "same task", "same failure")
    assert len(wire.requests) == 1 and b.cached and b.text == a.text and b.tokens_in == 0
    recs = records(tmp_path)
    assert [r["cached"] for r in recs] == [False, True] and recs[1]["outcome"] == "ok"
    assert tg.summarise(tmp_path / "tg")["calls"] == 1 and tg.summarise(tmp_path / "tg")["cache_hits"] == 1


def test_a_failed_answer_is_not_cached(tmp_path):
    wire = Wire(status=500)
    tel, wire = make(tmp_path, wire=wire)
    tel.ask("p", "t", "f")
    wire.status = 200
    assert tel.ask("p", "t", "f").ok and len(wire.requests) == 2


def test_the_cache_is_per_problem_text_and_does_not_cross_a_changed_failure(tmp_path):
    tel, wire = make(tmp_path)
    tel.ask("p", "t", "failure one")
    tel.ask("p", "t", "failure two")
    assert len(wire.requests) == 2


def test_the_cheapest_model_is_asked_first_and_a_stronger_one_after_an_answer_that_did_not_help(tmp_path):
    tel, wire = make(tmp_path)
    first = tel.ask("p", "t", "f1")
    tel.report(first.call_id, helped=False)
    second = tel.ask("p", "t", "f2")
    tel.report(second.call_id, helped=False)
    third = tel.ask("p", "t", "f3")
    assert [r["body"]["model"] for r in wire.requests] == ["small", "big", "big"]
    assert third.model == "big"


def test_an_answer_that_helped_keeps_the_cheap_model_for_the_next_question(tmp_path):
    tel, wire = make(tmp_path)
    first = tel.ask("p", "t", "f1")
    tel.report(first.call_id, helped=True)
    tel.ask("p", "t", "f2")
    assert [r["body"]["model"] for r in wire.requests] == ["small", "small"]


def test_the_ladder_is_per_problem(tmp_path):
    tel, wire = make(tmp_path)
    a = tel.ask("one", "t", "f1")
    tel.report(a.call_id, helped=False)
    tel.ask("two", "t", "f1")
    assert [r["body"]["model"] for r in wire.requests] == ["small", "small"]


def test_the_call_cap_is_kept_across_processes_and_a_refusal_sends_nothing(tmp_path):
    tel, wire = make(tmp_path, max_calls=2)
    tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f2")
    tel2, wire2 = make(tmp_path, max_calls=2)                       # a new process reads the same ledger
    got = tel2.ask("p", "t", "f3")
    assert not got.ok and "call cap" in got.reason and wire2.requests == []
    assert records(tmp_path)[-1]["outcome"] == "budget"


def test_a_cache_hit_does_not_count_against_the_call_cap(tmp_path):
    tel, wire = make(tmp_path, max_calls=1)
    tel.ask("p", "t", "f")
    assert tel.ask("p", "t", "f").ok and len(wire.requests) == 1


def test_the_token_cap_stops_the_next_call_once_the_total_is_reached(tmp_path):
    tel, wire = make(tmp_path, max_total_tokens=200)                # each call here is 150 tokens
    assert tel.ask("p", "t", "f1").ok
    assert tel.ask("p", "t", "f2").ok                               # 150 < 200 before the call
    got = tel.ask("p", "t", "f3")                                   # 300 >= 200
    assert not got.ok and "token cap" in got.reason and len(wire.requests) == 2


def test_the_cost_cap_needs_a_price_and_stops_when_spent(tmp_path):
    prices = {"small": [1.0, 3.0], "big": [5.0, 15.0]}
    w = worst_cost(request_body_of(tmp_path, max_output_tokens=100, prices=prices), prices["small"], 100)
    first_cost = 120e-6 + 90e-6
    cap = w + first_cost / 2                                        # room for the first call, not for a second
    tel, wire = make(tmp_path, max_cost_usd=cap, max_output_tokens=100, prices=prices)
    assert tel.ask("p", "t", "f1").ok
    got = tel.ask("p", "t", "f2")
    assert not got.ok and "cost cap" in got.reason and len(wire.requests) == 1


def test_a_model_with_no_price_is_refused_under_a_cost_cap(tmp_path):
    tel, wire = make(tmp_path, max_cost_usd=1.0, prices={"big": [5.0, 15.0]})
    got = tel.ask("p", "t", "f")
    assert not got.ok and "no price for small" in got.reason and wire.requests == []


def test_the_worst_case_cost_of_a_call_is_checked_before_it_is_sent(tmp_path):
    tel, wire = make(tmp_path, max_cost_usd=0.001, max_output_tokens=1500, prices={"small": [1.0, 3.0], "big": [1, 1]})
    tel.ask("p", "t", "f1")
    # 1500 output tokens at $3 per million is 0.0045 on its own: the call may not start.
    assert wire.requests == [] and "cost cap" in records(tmp_path)[0]["reason"]


# ------------------------------------------------------------------------------------------------ failure never reaches the harness

@pytest.mark.parametrize("raiser", [OSError("down"), TimeoutError("slow"), ValueError("odd"), RuntimeError("x")])
def test_a_transport_error_is_a_result_and_not_an_exception(tmp_path, raiser):
    def boom(url, headers, body, timeout):
        raise raiser
    cfg = tg.load(table(), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=boom, environ={"TG_KEY": KEY})
    got = tel.ask("p", "t", "f")
    assert not got.ok and got.text == "" and records(tmp_path)[0]["outcome"] == "error"


@pytest.mark.parametrize("status, body", [(401, b'{"error": "bad key"}'), (429, b"slow down"), (500, b"oops"),
                                          (200, b"not json"), (200, b'{"choices": []}'),
                                          (200, json.dumps({"choices": [{"message": {"content": None}}]}).encode()),
                                          (200, json.dumps({"choices": [{"message": {"content": "  "}}]}).encode())])
def test_an_unusable_answer_is_a_failed_result_with_a_reason(tmp_path, status, body):
    tel, _ = make(tmp_path, wire=Wire(lambda b: body, status=status))
    got = tel.ask("p", "t", "f")
    assert not got.ok and got.reason and got.text == ""
    assert records(tmp_path)[0]["outcome"] == "error"


def test_the_error_reason_never_echoes_the_key_even_if_the_server_does(tmp_path):
    body = json.dumps({"error": f"invalid key {KEY}"}).encode()
    tel, _ = make(tmp_path, wire=Wire(lambda b: body, status=401))
    got = tel.ask("p", "t", "f")
    assert KEY not in got.reason and "401" in got.reason
    assert KEY not in (tmp_path / "tg" / "telegraph.jsonl").read_text()


def test_the_reply_is_data_with_a_marker_that_says_it_is_not_to_be_run(tmp_path):
    tel, _ = make(tmp_path)
    got = tel.ask("p", "t", "f")
    assert got.as_hint().startswith("Advice from a helper model (untrusted text") and got.text in got.as_hint()


# ------------------------------------------------------------------------------------------------ standing rules

def test_the_module_imports_no_hardware_package_and_opens_no_device():
    src = Path(tg.__file__).read_text()
    for word in ("import ttnn", "import ttml", "/dev/tenstorrent", "gozer"):
        assert word not in src


def test_the_ledger_and_cache_files_are_private(tmp_path):
    tel, _ = make(tmp_path)
    tel.ask("p", "t", "f")
    for path in (tmp_path / "tg").rglob("*"):
        if path.is_file():
            assert (path.stat().st_mode & 0o077) == 0, path


def test_a_redirect_is_not_followed_so_the_key_is_not_sent_to_another_host():
    import http.server, threading
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        status, _ = tg.http_post(f"http://127.0.0.1:{srv.server_port}/v1/chat/completions",
                                 {"Authorization": "Bearer abc"}, b"{}", 5)
    finally:
        srv.shutdown()
    assert status == 302 and seen == [("/v1/chat/completions", "Bearer abc")]


def test_the_token_cap_stops_a_call_when_the_total_equals_it(tmp_path):
    tel, wire = make(tmp_path, max_total_tokens=300)                # two calls of 150 tokens reach it exactly
    tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f2")
    assert not tel.ask("p", "t", "f3").ok and len(wire.requests) == 2


def test_a_home_directory_outside_the_usual_places_is_still_removed(tmp_path):
    tel, wire = make(tmp_path)                                      # this telegraph's home is /home/someone
    tel2 = tg.Telegraph(tg.load(table(), base=tmp_path), tmp_path / "tg2", transport=wire, environ={"TG_KEY": KEY},
                        hostname="node9", home="/mnt/people/zed")
    tel2.ask("p", "see /mnt/people/zed/work/a.py and /home/other/b.py", "f")
    sent = wire.requests[-1]["body"]["messages"][-1]["content"]
    assert "/mnt/people/zed" not in sent and "/home/other" not in sent and "~/work/a.py" in sent and "~/b.py" in sent


@pytest.mark.parametrize("url", ["http://8.8.8.8/v1", "http://example.com/v1", "http://[2001:4860::1]/v1"])
def test_plain_http_to_a_public_address_is_refused(url):
    with pytest.raises(tg.ConfigError, match="https"):
        tg.load(table(endpoint=url), base=Path("/x"))


def test_an_anthropic_endpoint_that_already_ends_in_v1_gets_the_path_once(tmp_path):
    wire = Wire(lambda body: anthropic_reply("ok", 5, 5))
    tel, wire = make(tmp_path, wire=wire, api="anthropic", endpoint="https://llm.example.org/v1")
    tel.ask("p", "t", "f")
    assert wire.requests[0]["url"] == "https://llm.example.org/v1/messages"
    assert wire.requests[0]["body"]["temperature"] == 0


def test_an_anthropic_answer_with_no_text_is_a_failed_result(tmp_path):
    wire = Wire(lambda body: {"content": [{"type": "text", "text": "  "}], "usage": {"input_tokens": 5, "output_tokens": 1}})
    tel, _ = make(tmp_path, wire=wire, api="anthropic")
    got = tel.ask("p", "t", "f")
    assert not got.ok and "empty" in got.reason and records(tmp_path)[0]["outcome"] == "error"


@pytest.mark.parametrize("prices", [{"small": [1]}, {"small": "cheap"}, {"small": [1, "x"]}, {"small": [-1, 2]},
                                    {"small": [True, 2]}])
def test_a_malformed_price_is_refused(prices):
    with pytest.raises(tg.ConfigError, match="prices.small"):
        tg.load(table(prices=prices), base=Path("/x"))


# ------------------------------------------------------------------------------------------------ thinking models

def test_a_think_block_is_removed_from_the_answer(tmp_path):
    wire = Wire(lambda b: openai_reply("<think>\nlet me see\n</think>\n\nUse ttl.block.fill.", 50, 40))
    tel, _ = make(tmp_path, wire=wire)
    got = tel.ask("p", "t", "f")
    assert got.ok and got.text == "Use ttl.block.fill." and "think" not in got.as_hint().split("\n", 1)[1]
    assert (tmp_path / "tg" / records(tmp_path)[0]["reply_file"]).read_text() == "Use ttl.block.fill."
    assert records(tmp_path)[0]["tokens_out"] == 40                 # the reasoning was paid for and is counted


def test_an_answer_cut_off_inside_its_reasoning_is_a_failed_call_and_is_not_cached(tmp_path):
    cut_off = {"choices": [{"message": {"content": "<think>\nOkay, let's see. The user"}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": 188, "completion_tokens": 400}}
    wire = Wire(lambda b: cut_off)
    tel, wire = make(tmp_path, wire=wire)
    got = tel.ask("p", "t", "f")
    assert not got.ok and "cut off inside its reasoning" in got.reason and "extra_body" in got.reason
    rec = records(tmp_path)[0]
    assert rec["outcome"] == "error" and (rec["tokens_in"], rec["tokens_out"]) == (188, 400)
    assert not (tmp_path / "tg" / "cache").exists()
    tel.ask("p", "t", "f")
    assert len(wire.requests) == 2                                  # nothing was cached, so it asked again


def test_the_tokens_of_a_failed_answer_count_against_the_token_cap(tmp_path):
    cut_off = {"choices": [{"message": {"content": "<think>x"}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": 188, "completion_tokens": 400}}
    tel, wire = make(tmp_path, wire=Wire(lambda b: cut_off), max_total_tokens=500)
    tel.ask("p", "t", "f1")
    got = tel.ask("p", "t", "f2")
    assert not got.ok and "token cap" in got.reason and len(wire.requests) == 1


def test_a_think_block_with_nothing_after_it_is_an_empty_answer(tmp_path):
    wire = Wire(lambda b: openai_reply("<think>only thoughts</think>", 10, 10))
    tel, _ = make(tmp_path, wire=wire)
    got = tel.ask("p", "t", "f")
    assert not got.ok and "only reasoning, no answer" in got.reason


def test_a_failed_answer_is_costed_when_the_model_has_a_price(tmp_path):
    cut_off = {"choices": [{"message": {"content": "<think>x"}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 400}}
    tel, _ = make(tmp_path, wire=Wire(lambda b: cut_off), prices={"small": [1.0, 3.0], "big": [5.0, 15.0]})
    tel.ask("p", "t", "f")
    assert records(tmp_path)[0]["cost_usd"] == pytest.approx(100e-6 + 1200e-6)
    assert tg.summarise(tmp_path / "tg")["cost_usd"] == pytest.approx(1300e-6)


def test_extra_body_is_sent_with_the_request_for_both_api_styles(tmp_path):
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    tel, wire = make(tmp_path, extra_body=extra)
    tel.ask("p", "t", "f")
    assert wire.requests[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert wire.requests[0]["body"]["model"] == "small" and wire.requests[0]["body"]["max_tokens"] == 1500
    tel2, wire2 = make(tmp_path / "b", wire=Wire(lambda b: anthropic_reply("ok", 5, 5)), api="anthropic",
                       extra_body={"thinking": {"type": "disabled"}})
    tel2.ask("p", "t", "f")
    assert wire2.requests[0]["body"]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("key", ["model", "messages", "max_tokens", "temperature", "system", "stream"])
def test_extra_body_cannot_replace_what_telegraph_sets(key):
    with pytest.raises(tg.ConfigError, match=f"extra_body.*{key}"):
        tg.load(table(extra_body={key: 1}), base=Path("/x"))


@pytest.mark.parametrize("bad", ["x", [1], 3])
def test_extra_body_must_be_a_table(bad):
    with pytest.raises(tg.ConfigError, match="extra_body must be a table"):
        tg.load(table(extra_body=bad), base=Path("/x"))


def test_a_cache_entry_that_still_holds_reasoning_is_not_used(tmp_path):
    tel, wire = make(tmp_path)
    tel.ask("p", "t", "f")
    entry = next((tmp_path / "tg" / "cache").glob("*.json"))
    data = json.loads(entry.read_text())
    data["text"] = "<think>an entry written by an earlier version"
    entry.write_text(json.dumps(data))
    got = tel.ask("p", "t", "f")
    assert len(wire.requests) == 2 and not got.cached and "<think>" not in got.text


# ------------------------------------------------------------------------------------------------ a cap for one problem

def test_defaults_for_one_problem_are_a_few_calls_and_no_other_cap():
    cfg = tg.load(table(), base=Path("/x"))
    assert cfg.max_calls_per_problem == 3 and cfg.max_tokens_per_problem is None and cfg.max_cost_per_problem_usd is None
    assert tg.Config().max_calls_per_problem == 3


@pytest.mark.parametrize("bad, text", [({"max_calls_per_problem": 0}, "max_calls_per_problem"),
                                       ({"max_calls_per_problem": True}, "max_calls_per_problem"),
                                       ({"max_tokens_per_problem": 0}, "max_tokens_per_problem"),
                                       ({"max_cost_per_problem_usd": 0.5}, "prices"),
                                       ({"max_cost_per_problem_usd": -1, "prices": {"small": [1, 1], "big": [1, 1]}},
                                        "max_cost_per_problem_usd")])
def test_a_bad_per_problem_setting_is_refused(bad, text):
    with pytest.raises(tg.ConfigError, match=text):
        tg.load(table(**bad), base=Path("/x"))


def test_the_calls_for_one_problem_stop_at_the_cap_while_another_problem_is_still_served(tmp_path):
    tel, wire = make(tmp_path, max_calls_per_problem=2, max_calls=10)
    tel.ask("scan", "t", "f1")
    tel.ask("scan", "t", "f2")
    got = tel.ask("scan", "t", "f3")
    assert not got.ok and "problem scan" in got.reason and "call cap" in got.reason and len(wire.requests) == 2
    assert tel.ask("other", "t", "f1").ok and len(wire.requests) == 3
    assert records(tmp_path)[2]["outcome"] == "budget"


def test_a_cache_hit_and_a_failed_request_are_counted_as_the_ledger_counts_them(tmp_path):
    tel, wire = make(tmp_path, max_calls_per_problem=2)
    tel.ask("p", "t", "same")
    tel.ask("p", "t", "same")                                       # a cache hit: not a call
    assert tel.ask("p", "t", "other").ok                            # the second call
    assert not tel.ask("p", "t", "third").ok and len(wire.requests) == 2


def test_the_token_cap_for_one_problem_counts_that_problems_tokens_only(tmp_path):
    tel, wire = make(tmp_path, max_tokens_per_problem=200, max_calls_per_problem=10)    # 150 tokens a call
    tel.ask("a", "t", "f1")
    assert tel.ask("a", "t", "f2").ok                               # 150 < 200 before the call
    got = tel.ask("a", "t", "f3")
    assert not got.ok and "problem a" in got.reason and "token cap" in got.reason
    assert tel.ask("b", "t", "f1").ok


def test_the_token_cap_for_a_problem_stops_when_the_total_equals_it(tmp_path):
    tel, wire = make(tmp_path, max_tokens_per_problem=300, max_calls_per_problem=10)
    tel.ask("a", "t", "f1")
    tel.ask("a", "t", "f2")
    assert not tel.ask("a", "t", "f3").ok and len(wire.requests) == 2


def test_the_cost_cap_for_one_problem_looks_at_the_worst_case_of_the_next_call(tmp_path):
    prices = {"small": [1.0, 3.0], "big": [5.0, 15.0]}
    w = worst_cost(request_body_of(tmp_path, max_output_tokens=100, prices=prices), prices["small"], 100)
    cap = w + (120e-6 + 90e-6) / 2
    tel, wire = make(tmp_path, max_cost_per_problem_usd=cap, max_output_tokens=100, max_calls_per_problem=10, prices=prices)
    assert tel.ask("a", "t", "f1").ok                               # 210 micro-dollars spent; the next could cost up to w
    got = tel.ask("a", "t", "f2")
    assert not got.ok and "problem a" in got.reason and "cost cap" in got.reason and len(wire.requests) == 1
    assert tel.ask("b", "t", "f1").ok


def test_a_model_with_no_price_is_refused_under_a_per_problem_cost_cap(tmp_path):
    tel, wire = make(tmp_path, max_cost_per_problem_usd=1.0, prices={"big": [5.0, 15.0]})
    got = tel.ask("a", "t", "f")
    assert not got.ok and "no price for small" in got.reason and wire.requests == []


def test_the_per_problem_cap_is_kept_across_processes(tmp_path):
    tel, _ = make(tmp_path, max_calls_per_problem=1)
    tel.ask("a", "t", "f1")
    tel2, wire2 = make(tmp_path, max_calls_per_problem=1)
    assert not tel2.ask("a", "t", "f2").ok and wire2.requests == []


def test_the_summary_counts_refusals_for_a_problem_cap(tmp_path):
    tel, _ = make(tmp_path, max_calls_per_problem=1)
    tel.ask("a", "t", "f1")
    tel.ask("a", "t", "f2")
    assert tg.summarise(tmp_path / "tg")["refused"] == 1


# ------------------------------------------------------------------------------------------------ review of PR 106, group A

@pytest.mark.parametrize("key", ["n", "best_of", "stream_options"])
def test_extra_body_cannot_ask_for_more_than_one_completion(key):
    with pytest.raises(tg.ConfigError, match=f"extra_body cannot set {key}"):
        tg.load(table(extra_body={key: 10}), base=Path("/x"))


@pytest.mark.parametrize("value", ["false", "true", 1, 0, "yes", [True]])
def test_enabled_must_be_a_toml_boolean(value):
    with pytest.raises(tg.ConfigError, match="enabled must be true or false"):
        tg.load({"enabled": value, "endpoint": "https://llm.example.org/v1", "models": ["a"], "key_env": "K"},
                base=Path("/x"))


@pytest.mark.parametrize("bad", ["on", 3, ["enabled"]])
def test_the_telegraph_table_must_be_a_table(bad):
    with pytest.raises(tg.ConfigError, match="telegraph must be a table"):
        tg.load(bad, base=Path("/x"))


def test_a_complete_pem_block_is_removed_before_anything_is_sent(tmp_path):
    pem = ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAxyz0123456789abcdefABCDEF\nsecondline0123456789\n"
           "-----END RSA PRIVATE KEY-----")
    tel, wire = make(tmp_path)
    tel.ask("p", f"before\n{pem}\nafter", "f")
    sent = wire.requests[0]["body"]["messages"][-1]["content"]
    for leak in ("MIIEowIBAAKCAQEAxyz", "secondline0123456789", "END RSA PRIVATE KEY", "BEGIN RSA PRIVATE KEY"):
        assert leak not in sent, leak
    assert "before" in sent and "after" in sent


def test_an_unterminated_pem_block_is_removed_to_the_end_of_the_text(tmp_path):
    tel, wire = make(tmp_path)
    tel.ask("p", "task\n-----BEGIN PRIVATE KEY-----\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7", "f")
    sent = wire.requests[0]["body"]["messages"][-1]["content"]
    assert "MIIBVQIBADANBgkq" not in sent and "BEGIN PRIVATE KEY" not in sent and "task" in sent


def test_a_problem_id_that_holds_the_key_is_cleaned_before_the_ledger_and_the_budget(tmp_path):
    tel, wire = make(tmp_path, max_calls_per_problem=1)
    problem = f"scan-{KEY}"
    tel.ask(problem, "t", "f1")
    for path in (tmp_path / "tg").rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text(), path
    got = tel.ask(problem, "t", "f2")                              # the cap still applies to the cleaned id
    assert not got.ok and "call cap" in got.reason and KEY not in got.reason and len(wire.requests) == 1


def test_a_problem_id_with_a_home_path_is_cleaned_in_the_ledger(tmp_path):
    tel, _ = make(tmp_path)
    tel.ask("fix /home/someone/orchard-runs/x", "t", "f")
    rec = records(tmp_path)[0]
    assert "/home/someone" not in rec["problem"] and rec["problem"].startswith("fix ~/")


def test_a_report_note_that_holds_the_key_is_not_written_to_the_ledger(tmp_path):
    tel, _ = make(tmp_path)
    got = tel.ask("p", "t", "f")
    tel.report(got.call_id, helped=False, note=f"the answer repeated {KEY} back")
    assert KEY not in (tmp_path / "tg" / "telegraph.jsonl").read_text()


# ------------------------------------------------------------------------------------------------ review of PR 106, group B

def test_a_changed_request_option_does_not_return_the_answer_cached_for_the_old_one(tmp_path):
    tel, wire = make(tmp_path)
    tel.ask("p", "t", "f")
    tel2, wire2 = make(tmp_path, extra_body={"response_format": {"type": "json_object"}})
    got = tel2.ask("p", "t", "f")
    assert not got.cached and len(wire2.requests) == 1 and wire2.requests[0]["body"]["response_format"]


def test_a_changed_endpoint_or_api_style_does_not_return_a_cached_answer(tmp_path):
    tel, wire = make(tmp_path)
    tel.ask("p", "t", "f")
    tel2, wire2 = make(tmp_path, endpoint="https://other.example.org/v1")
    assert not tel2.ask("p", "t", "f").cached and len(wire2.requests) == 1
    tel3, wire3 = make(tmp_path, wire=Wire(lambda b: anthropic_reply("x", 5, 5)), api="anthropic")
    assert not tel3.ask("p", "t", "f").cached and len(wire3.requests) == 1


def test_the_same_configuration_still_hits_the_cache_across_processes(tmp_path):
    tel, _ = make(tmp_path)
    tel.ask("p", "t", "f")
    tel2, wire2 = make(tmp_path)
    assert tel2.ask("p", "t", "f").cached and wire2.requests == []


def test_a_cached_answers_id_can_be_reported_and_the_verdict_counts_for_the_original_call(tmp_path):
    tel, wire = make(tmp_path)
    first = tel.ask("p", "t", "f")
    again = tel.ask("p", "t", "f")
    assert again.cached
    tel.report(again.call_id, helped=False)                          # the CLI tells the user to report this id
    s = tg.summarise(tmp_path / "tg")
    assert (s["calls"], s["not_helped"], s["unreported"]) == (1, 1, 0)
    nxt = tel.ask("p", "t", "different failure")
    assert nxt.model == "big"                                        # the ladder moved on after the verdict
    assert [r.get("origin") for r in records(tmp_path) if r.get("cached")] == [first.call_id]


def test_reporting_a_cached_id_and_its_original_id_is_one_verdict_for_one_call(tmp_path):
    tel, _ = make(tmp_path)
    first = tel.ask("p", "t", "f")
    again = tel.ask("p", "t", "f")
    tel.report(first.call_id, helped=True)
    tel.report(again.call_id, helped=False)                          # the latest verdict for the call wins
    s = tg.summarise(tmp_path / "tg")
    assert (s["helped"], s["not_helped"]) == (0, 1)


def test_reporting_the_same_answer_twice_moves_the_ladder_one_step_only(tmp_path):
    tel, wire = make(tmp_path, models=["small", "big", "third"])
    first = tel.ask("p", "t", "f1")
    tel.report(first.call_id, helped=False)
    tel.report(first.call_id, helped=False)
    tel.ask("p", "t", "f2")
    assert [r["body"]["model"] for r in wire.requests] == ["small", "big"]


def test_a_later_helped_verdict_for_a_call_undoes_its_earlier_not_helped(tmp_path):
    tel, wire = make(tmp_path)
    first = tel.ask("p", "t", "f1")
    tel.report(first.call_id, helped=False)
    tel.report(first.call_id, helped=True)
    tel.ask("p", "t", "f2")
    assert [r["body"]["model"] for r in wire.requests] == ["small", "small"]


# ------------------------------------------------------------------------------------------------ review of PR 106, group C

def test_a_call_in_flight_is_counted_against_the_caps_of_a_second_process(tmp_path):
    other = {}

    def transport(url, headers, body, timeout):
        # while this request is in flight, a second process asks: the first call's reservation must already count
        other["got"] = other["tel"].ask("p", "t", "second question")
        return 200, json.dumps(openai_reply("ok", 10, 10)).encode()
    wire_b = Wire()
    tel_b, _ = make(tmp_path, wire=wire_b, max_calls=1)
    other["tel"] = tel_b
    cfg = tg.load(table(max_calls=1), base=tmp_path)
    tel_a = tg.Telegraph(cfg, tmp_path / "tg", transport=transport, environ={"TG_KEY": KEY})
    assert tel_a.ask("p", "t", "first question").ok
    assert not other["got"].ok and "call cap" in other["got"].reason and wire_b.requests == []


def test_a_process_that_dies_during_a_request_still_leaves_its_reservation(tmp_path):
    def dies(url, headers, body, timeout):
        raise SystemExit(1)
    cfg = tg.load(table(max_calls=1), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=dies, environ={"TG_KEY": KEY})
    with pytest.raises(SystemExit):
        tel.ask("p", "t", "f")
    tel2, wire2 = make(tmp_path, max_calls=1)
    got = tel2.ask("p", "t", "f2")
    assert not got.ok and "call cap" in got.reason and wire2.requests == []
    s = tg.summarise(tmp_path / "tg")
    assert s["unsettled"] == 1


def test_a_request_with_unknown_usage_counts_its_worst_case_against_a_cost_cap(tmp_path):
    def times_out(url, headers, body, timeout):
        raise TimeoutError("slow")
    cfg = tg.load(table(max_cost_usd=0.005, max_output_tokens=1000, prices={"small": [1.0, 3.0], "big": [5.0, 15.0]}),
                  base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=times_out, environ={"TG_KEY": KEY})
    assert not tel.ask("p", "t", "f1").ok                           # the provider may have processed it and billed it
    got = tel.ask("p", "t", "f2")
    assert not got.ok and "cost cap" in got.reason
    assert tg.summarise(tmp_path / "tg")["unknown_usage_calls"] == 1


def test_a_request_the_provider_rejected_before_processing_does_not_use_up_the_budget(tmp_path):
    tel, wire = make(tmp_path, wire=Wire(status=401), max_cost_usd=0.005, max_output_tokens=1000,
                     prices={"small": [1.0, 3.0], "big": [5.0, 15.0]})
    assert not tel.ask("p", "t", "f1").ok
    wire.status = 200
    assert tel.ask("p", "t", "f2").ok


def test_the_token_cap_counts_a_calls_reservation_when_its_usage_is_unknown(tmp_path):
    def times_out(url, headers, body, timeout):
        raise TimeoutError("slow")
    cfg = tg.load(table(max_total_tokens=500, max_output_tokens=1000), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=times_out, environ={"TG_KEY": KEY})
    tel.ask("p", "t", "f1")
    got = tel.ask("p", "t", "f2")
    assert not got.ok and "token cap" in got.reason


def test_a_torn_last_line_is_not_glued_to_the_next_record(tmp_path):
    tel, _ = make(tmp_path)
    tel.ask("p", "t", "f1")
    with open(tmp_path / "tg" / "telegraph.jsonl", "a") as f:
        f.write('{"call_id": "half')                                # a write cut short
    got = tel.ask("p", "t", "f2")
    assert got.ok
    s = tg.summarise(tmp_path / "tg")
    assert s["calls"] == 2                                          # the call after the torn line is counted


def test_an_unwritable_ledger_refuses_the_call_before_any_request(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    wire = Wire()
    cfg = tg.load(table(), base=tmp_path)
    tel = tg.Telegraph(cfg, blocker / "tg", transport=wire, environ={"TG_KEY": KEY})
    got = tel.ask("p", "t", "f")
    assert not got.ok and "cannot be written" in got.reason and wire.requests == []


def test_a_read_only_ledger_directory_refuses_the_call(tmp_path):
    import os
    d = tmp_path / "tg"
    d.mkdir()
    d.chmod(0o500)
    try:
        if os.access(d, os.W_OK):
            pytest.skip("this user can write to a read-only directory")
        wire = Wire()
        tel = tg.Telegraph(tg.load(table(), base=tmp_path), d, transport=wire, environ={"TG_KEY": KEY})
        got = tel.ask("p", "t", "f")
        assert not got.ok and "cannot be written" in got.reason and wire.requests == []
    finally:
        d.chmod(0o700)


def test_the_input_bound_for_cost_counts_bytes_so_non_english_text_cannot_slip_under_the_cap(tmp_path):
    # 3,000 four-byte characters: characters divided by three is about 1,000 tokens, the real bound is the 12,000 bytes
    tel, wire = make(tmp_path, max_cost_usd=5.0, max_output_tokens=1, prices={"small": [1000.0, 0.0], "big": [1.0, 1.0]})
    got = tel.ask("p", "\U0001F600" * 3000, "f")
    assert not got.ok and "cost cap" in got.reason and wire.requests == []


def test_the_input_bound_counts_extra_body_too(tmp_path):
    tel, wire = make(tmp_path, max_cost_usd=0.01, max_output_tokens=1, prices={"small": [1000.0, 0.0], "big": [1.0, 1.0]},
                     extra_body={"metadata": {"note": "x" * 4000}})
    got = tel.ask("p", "t", "f")
    assert not got.ok and "cost cap" in got.reason and wire.requests == []


def test_reservations_and_settlements_are_one_call_in_the_summary(tmp_path):
    tel, _ = make(tmp_path)
    tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f2")
    s = tg.summarise(tmp_path / "tg")
    assert (s["calls"], s["unsettled"], s["unknown_usage_calls"]) == (2, 0, 0)
    assert s["tokens_in"] == 240 and s["tokens_out"] == 60


def test_two_processes_racing_for_the_last_allowed_call_cannot_both_send(tmp_path):
    import threading, time

    class Slow(tg.Telegraph):
        def _over_budget(self, *a, **kw):                           # widen the window between the check and the reservation
            out = super()._over_budget(*a, **kw)
            time.sleep(0.05)
            return out
    wire = Wire()
    cfg = tg.load(table(max_calls=1), base=tmp_path)
    results = []

    def go(i):
        t = Slow(cfg, tmp_path / "tg", transport=wire, environ={"TG_KEY": KEY})
        results.append(t.ask("p", "t", f"question {i}").ok)
    threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results.count(True) == 1 and len(wire.requests) == 1


def test_a_ledger_that_cannot_take_a_record_refuses_the_call(tmp_path):
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").mkdir()                   # the file the ledger needs is a directory
    wire = Wire()
    tel = tg.Telegraph(tg.load(table(), base=tmp_path), tmp_path / "tg", transport=wire, environ={"TG_KEY": KEY})
    got = tel.ask("p", "t", "f")
    assert not got.ok and "cannot be written" in got.reason and wire.requests == []


def test_records_from_before_reservations_count_by_their_own_numbers(tmp_path):
    (tmp_path / "tg").mkdir()
    old = {"call_id": "old1", "problem": "p", "model": "small", "outcome": "ok", "cached": False,
           "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.001, "ts": 1.0, "latency_s": 1.0}
    (tmp_path / "tg" / "telegraph.jsonl").write_text(json.dumps(old) + "\n")
    tel, wire = make(tmp_path, max_calls=1)
    got = tel.ask("p", "t", "f")
    assert not got.ok and "call cap" in got.reason and wire.requests == []
    tel2, _ = make(tmp_path, max_total_tokens=150, max_calls=5)
    assert not tel2.ask("p", "t", "f").ok                           # 100 + 50 tokens are counted as they were written


def test_an_answer_with_no_usage_counts_its_worst_case_against_a_cost_cap(tmp_path):
    prices = {"small": [1.0, 3.0], "big": [5.0, 15.0]}
    no_usage = {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]}
    w = worst_cost(request_body_of(tmp_path, max_output_tokens=100, prices=prices), prices["small"], 100)
    tel, wire = make(tmp_path, wire=Wire(lambda b: no_usage), max_cost_usd=w * 1.5, max_output_tokens=100, prices=prices)
    assert tel.ask("p", "t", "f1").ok
    assert tg.summarise(tmp_path / "tg")["unknown_usage_calls"] == 1
    got = tel.ask("p", "t", "f2")                                    # the server billed something; w is the safe amount
    assert not got.ok and "cost cap" in got.reason and len(wire.requests) == 1


def test_a_reservation_written_after_a_torn_line_is_not_lost(tmp_path):
    def dies(url, headers, body, timeout):
        raise SystemExit(1)
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").write_text('{"call_id": "half')          # a write cut short, no line break
    cfg = tg.load(table(max_calls=2), base=tmp_path)                # the torn fragment counts as one call
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=dies, environ={"TG_KEY": KEY})
    with pytest.raises(SystemExit):
        tel.ask("p", "t", "f1")                                      # leaves only a reservation behind
    tel2, wire2 = make(tmp_path, max_calls=2)
    got = tel2.ask("p", "t", "f2")
    assert not got.ok and "call cap" in got.reason and wire2.requests == []


def test_an_unreadable_answer_to_a_processed_request_counts_its_worst_case(tmp_path):
    prices = {"small": [1.0, 3.0], "big": [5.0, 15.0]}
    w = worst_cost(request_body_of(tmp_path, max_output_tokens=100, prices=prices), prices["small"], 100)
    tel, wire = make(tmp_path, wire=Wire(lambda b: b"not json at all"), max_cost_usd=w * 1.5, max_output_tokens=100,
                     prices=prices)
    assert not tel.ask("p", "t", "f1").ok                            # the provider answered, so it may have billed
    assert tg.summarise(tmp_path / "tg")["unknown_usage_calls"] == 1
    got = tel.ask("p", "t", "f2")
    assert not got.ok and "cost cap" in got.reason and len(wire.requests) == 1


# ------------------------------------------------------------------------------------------------ review of PR 106, round 2

def _unpriced_ledger(tmp_path, problem="p"):
    (tmp_path / "tg").mkdir(parents=True, exist_ok=True)
    old = {"call_id": "old1", "problem": problem, "model": "small", "outcome": "ok", "cached": False, "tokens_in": 9000,
           "tokens_out": 900, "cost_usd": None, "usage_known": True, "ts": 1.0, "latency_s": 1.0}
    with open(tmp_path / "tg" / "telegraph.jsonl", "a") as f:
        f.write(json.dumps(old) + "\n")


def test_earlier_calls_with_no_known_cost_block_a_later_cost_cap(tmp_path):
    _unpriced_ledger(tmp_path)
    tel, wire = make(tmp_path, max_cost_usd=100.0, prices={"small": [1.0, 3.0], "big": [5.0, 15.0]})
    got = tel.ask("other", "t", "f")
    assert not got.ok and "no known cost" in got.reason and wire.requests == []


def test_earlier_calls_with_no_known_cost_block_a_per_problem_cost_cap_for_that_problem_only(tmp_path):
    _unpriced_ledger(tmp_path, problem="p")
    tel, wire = make(tmp_path, max_cost_per_problem_usd=100.0, prices={"small": [1.0, 3.0], "big": [5.0, 15.0]})
    got = tel.ask("p", "t", "f")
    assert not got.ok and "no known cost" in got.reason and "problem p" in got.reason
    assert tel.ask("another", "t", "f").ok


def test_unknown_costs_do_not_matter_when_no_cost_cap_is_set(tmp_path):
    _unpriced_ledger(tmp_path)
    tel, _ = make(tmp_path)
    assert tel.ask("p", "t", "f").ok


def test_a_response_body_that_echoes_the_key_across_the_old_cut_point_is_fully_redacted(tmp_path):
    body = ("x" * 155 + KEY + " given").encode()                   # the old 160-byte cut left five characters of the key
    tel, _ = make(tmp_path, wire=Wire(lambda b: body, status=401))
    got = tel.ask("p", "t", "f")
    assert KEY[:4] not in got.reason and "401" in got.reason
    assert KEY[:4] not in (tmp_path / "tg" / "telegraph.jsonl").read_text()


def test_a_short_write_of_the_reservation_refuses_the_call(tmp_path, monkeypatch):
    real = os.write

    def half(fd, data):
        return real(fd, data[: len(data) // 2])
    wire = Wire()
    tel, wire = make(tmp_path, wire=wire)
    monkeypatch.setattr(tg.os, "write", half)
    got = tel.ask("p", "t", "f")
    assert not got.ok and "cannot be written" in got.reason and wire.requests == []


@pytest.mark.parametrize("bad, text", [({"key_env": ["TG_KEY"]}, "key_env"), ({"key_env": ""}, "key_env"),
                                       ({"key_env": None, "key_file": 1}, "key_file"),
                                       ({"key_env": None, "key_file": ""}, "key_file")])
def test_a_credential_source_of_the_wrong_type_is_a_config_error(bad, text):
    with pytest.raises(tg.ConfigError, match=text):
        tg.load(table(**bad), base=Path("/x"))


@pytest.mark.parametrize("prices", [[1, 2], "cheap", 3])
def test_prices_must_be_a_table(prices):
    with pytest.raises(tg.ConfigError, match="telegraph.prices must be a table"):
        tg.load(table(prices=prices), base=Path("/x"))


def test_a_verdict_that_cannot_be_written_is_an_error(tmp_path, monkeypatch):
    tel, _ = make(tmp_path)
    got = tel.ask("p", "t", "f")
    monkeypatch.setattr(tel, "_append", lambda rec, **kw: False)
    with pytest.raises(tg.LedgerUnavailable):
        tel.report(got.call_id, helped=True)


def test_the_cache_is_published_only_after_the_call_is_recorded(tmp_path, monkeypatch):
    tel, wire = make(tmp_path)
    real = tel._append

    def fail_settle(rec, **kw):
        if rec.get("outcome") == "ok":
            return False
        return real(rec, **kw)
    monkeypatch.setattr(tel, "_append", fail_settle)
    got = tel.ask("p", "t", "f")
    assert got.ok and "could not record" in got.reason               # the paid answer is returned with a note
    assert not (tmp_path / "tg" / "cache").exists()
    monkeypatch.setattr(tel, "_append", real)
    again = tel.ask("p", "t", "f")
    assert not again.cached and len(wire.requests) == 2              # nothing cached, so the question was asked again


def test_existing_files_are_made_private_before_anything_is_written(tmp_path):
    import stat
    (tmp_path / "tg").mkdir()
    ledger = tmp_path / "tg" / "telegraph.jsonl"
    ledger.write_text("")
    ledger.chmod(0o644)
    (tmp_path / "tg" / "telegraph.lock").write_text("")
    (tmp_path / "tg" / "telegraph.lock").chmod(0o644)
    tel, _ = make(tmp_path)
    got = tel.ask("p", "t", "f")
    for path in [ledger, tmp_path / "tg" / "telegraph.lock", tmp_path / "tg" / "replies" / f"{got.call_id}.txt"]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path
    victim = tmp_path / "tg" / "cache" / "old.json"                  # an existing file is rewritten in place
    victim.write_text("{}")
    victim.chmod(0o644)
    tel._write(victim, '{"text": "x"}')
    assert stat.S_IMODE(victim.stat().st_mode) == 0o600


# ------------------------------------------------------------------------------------------------ review of PR 106, round 3

def test_a_ledger_that_cannot_be_read_refuses_the_call_and_is_not_an_empty_history(tmp_path, monkeypatch):
    tel, wire = make(tmp_path, max_calls=1)
    assert tel.ask("p", "t", "f1").ok
    real = tg.Path.read_text

    def broken(self, *a, **kw):
        if self.name == "telegraph.jsonl":
            raise PermissionError("denied")
        return real(self, *a, **kw)
    monkeypatch.setattr(tg.Path, "read_text", broken)
    got = tel.ask("p", "t", "f2")
    assert not got.ok and "cannot be written or read" in got.reason and len(wire.requests) == 1


def test_a_missing_ledger_is_an_empty_history(tmp_path):
    tel, wire = make(tmp_path)
    assert tel.ask("p", "t", "f").ok and len(wire.requests) == 1


def test_a_credential_in_extra_body_is_removed_from_the_outbound_request(tmp_path):
    extra = {"metadata": {"note": f"key {KEY} here", "list": [f"x{KEY}", {"deep": KEY}]}, "user": "/home/someone/me"}
    tel, wire = make(tmp_path, extra_body=extra)
    tel.ask("p", "t", "f")
    sent = json.dumps(wire.requests[0]["body"])
    assert KEY not in sent and "/home/someone" not in sent and "metadata" in sent
    assert KEY not in "".join(p.read_text() for p in (tmp_path / "tg").rglob("*") if p.is_file())


def test_the_reservation_is_flushed_to_disk_before_the_request_is_sent(tmp_path, monkeypatch):
    synced = []
    real = os.fsync
    monkeypatch.setattr(tg.os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    seen = {}

    def transport(url, headers, body, timeout):
        seen["synced_before_send"] = len(synced)
        return 200, json.dumps(openai_reply("ok", 10, 10)).encode()
    cfg = tg.load(table(), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=transport, environ={"TG_KEY": KEY})
    assert tel.ask("p", "t", "f").ok
    assert seen["synced_before_send"] >= 1 and len(synced) >= 2


def test_a_reservation_that_cannot_be_flushed_refuses_the_call(tmp_path, monkeypatch):
    def failing(fd):
        raise OSError("disk full")
    monkeypatch.setattr(tg.os, "fsync", failing)
    tel, wire = make(tmp_path)
    got = tel.ask("p", "t", "f")
    assert not got.ok and "cannot be written" in got.reason and wire.requests == []


def test_an_earlier_call_and_a_torn_fragment_both_count_toward_the_call_cap(tmp_path):
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").write_text(json.dumps(
        {"call_id": "old", "problem": "p", "outcome": "ok", "cached": False, "tokens_in": 1, "tokens_out": 1,
         "cost_usd": 0.0, "usage_known": True}) + '\n{"reserve": "half')
    tel, wire = make(tmp_path, max_calls=2)
    got = tel.ask("p", "t", "f")                                     # old call + torn fragment = 2 of 2
    assert not got.ok and "call cap" in got.reason and wire.requests == []


def test_a_torn_tail_is_moved_to_a_private_sidecar_file_and_the_ledger_is_whole_again(tmp_path):
    import stat
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").write_text('{"reserve": "half')
    tel, _ = make(tmp_path)
    tel.ask("p", "t", "f")
    sidecars = list((tmp_path / "tg").glob("telegraph.jsonl.torn-*"))
    assert len(sidecars) == 1 and sidecars[0].read_text() == '{"reserve": "half'
    assert stat.S_IMODE(sidecars[0].stat().st_mode) == 0o600
    for line in (tmp_path / "tg" / "telegraph.jsonl").read_text().splitlines():
        json.loads(line)                                             # every line parses now


def test_a_quarantined_fragment_counts_against_the_call_cap(tmp_path):
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").write_text('{"reserve": "half')
    tel, wire = make(tmp_path, max_calls=1)
    got = tel.ask("p", "t", "f")
    assert not got.ok and "call cap" in got.reason and wire.requests == []


def test_a_quarantined_fragment_blocks_a_cost_cap_until_it_is_reconciled(tmp_path):
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").write_text('{"reserve": "half')
    tel, wire = make(tmp_path, max_cost_usd=100.0, prices={"small": [1.0, 3.0], "big": [5.0, 15.0]})
    got = tel.ask("p", "t", "f")
    assert not got.ok and "no known cost" in got.reason and wire.requests == []


def test_a_torn_tail_whose_note_cannot_be_written_refuses_the_call(tmp_path, monkeypatch):
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").write_text('{"reserve": "half')
    tel, wire = make(tmp_path)
    real = tel._append
    monkeypatch.setattr(tel, "_append", lambda rec, **kw: False if "torn" in rec else real(rec, **kw))
    got = tel.ask("p", "t", "f")
    assert not got.ok and "cannot be written" in got.reason and wire.requests == []


def test_an_append_after_a_torn_line_starts_on_its_own_line(tmp_path):
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").write_text('{"call_id": "half')   # another process died mid-write
    tel, _ = make(tmp_path)
    assert tel._append({"report": "x", "helped": True})
    lines = (tmp_path / "tg" / "telegraph.jsonl").read_text().splitlines()
    assert lines[-1] == '{"helped": true, "report": "x"}' and lines[0] == '{"call_id": "half'


# ------------------------------------------------------------------------------------------------ review of PR 106, round 4

@pytest.mark.parametrize("url", ["http://134744072/v1", "http://2130706433/v1", "http://0x7f000001/v1",
                                 "http://0177.0.0.1/v1", "http://127.1/v1", "http://1/v1", "http://0/v1"])
def test_legacy_numeric_ipv4_forms_are_not_a_private_host_name(url):
    with pytest.raises(tg.ConfigError, match="https"):
        tg.load(table(endpoint=url), base=Path("/x"))


@pytest.mark.parametrize("url", ["http://gpu-box:8000/v1", "http://localhost/v1", "http://node4/v1", "http://a1/v1"])
def test_a_plain_machine_name_that_starts_with_a_letter_is_still_allowed(url):
    assert tg.load(table(endpoint=url), base=Path("/x")).endpoint == url


def test_an_orphaned_torn_sidecar_is_accounted_for_exactly_once(tmp_path):
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl.torn-abc12345").write_text('{"reserve": "half')    # a crash after the cut
    (tmp_path / "tg" / "telegraph.jsonl").write_text("")
    tel, wire = make(tmp_path, max_calls=1)
    got = tel.ask("p", "t", "f")
    assert not got.ok and "call cap" in got.reason and wire.requests == []
    tel2, wire2 = make(tmp_path, max_calls=2)
    assert tel2.ask("p", "t", "f").ok                                # one note for the sidecar, not one per process
    notes = [json.loads(l) for l in (tmp_path / "tg" / "telegraph.jsonl").read_text().splitlines()
             if '"torn"' in l]
    assert [n["torn"] for n in notes] == ["telegraph.jsonl.torn-abc12345"]


def test_a_sidecar_whose_note_was_written_is_not_counted_again(tmp_path):
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl.torn-abc12345").write_text("x")
    (tmp_path / "tg" / "telegraph.jsonl").write_text(json.dumps({"torn": "telegraph.jsonl.torn-abc12345", "tokens": 10}) + "\n")
    tel, wire = make(tmp_path, max_calls=2)
    assert tel.ask("p", "t", "f").ok                                 # the fragment is one call of the two, and this is the other


def test_every_ledger_append_waits_for_the_recovery_lock(tmp_path):
    import threading, time
    tel, _ = make(tmp_path)
    tel.ask("p", "t", "f0")
    done = threading.Event()

    def other():
        tel._append({"report": "nope", "helped": True})
        done.set()
    with tel._ledger_lock():
        t = threading.Thread(target=other)
        t.start()
        time.sleep(0.15)
        assert not done.is_set()                                     # blocked while the lock is held
    t.join(timeout=5)
    assert done.is_set()


def test_a_locked_caller_can_append_without_waiting_on_its_own_lock(tmp_path):
    tel, _ = make(tmp_path)
    with tel._ledger_lock():
        assert tel._append({"report": "x", "helped": True}, held=True)


def test_a_settlement_cannot_slip_in_between_recovery_reading_and_cutting_the_ledger(tmp_path, monkeypatch):
    import threading
    (tmp_path / "tg").mkdir()
    (tmp_path / "tg" / "telegraph.jsonl").write_text('{"call_id": "half')
    tel, _ = make(tmp_path)
    appended_during_recovery = []
    real_truncate = os.truncate

    def slow_truncate(path, length):
        t = threading.Thread(target=lambda: tel._append({"report": "late", "helped": False}))
        t.start()
        t.join(timeout=0.2)                                          # the other writer must still be waiting
        appended_during_recovery.append(t.is_alive())
        return real_truncate(path, length)
    monkeypatch.setattr(tg.os, "truncate", slow_truncate)
    tel.ask("p", "t", "f")
    assert appended_during_recovery == [True]
    assert '"late"' in (tmp_path / "tg" / "telegraph.jsonl").read_text()    # and it landed after the recovery
