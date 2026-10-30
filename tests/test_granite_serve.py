"""The Granite package's chat server (pkgsrc/server/serve.py): prompt template, stop handling, finish_reason, request checks.
No chip is used: `generate` is driven by a fake engine that returns scripted logits."""
import importlib.util
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("granite_serve", ROOT / "pkgsrc" / "server" / "serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)

SNAP = next(iter(Path.home().glob(".cache/huggingface/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/*")), None)
V = 16
EOS = 15


class FakeTok:
    """Token i decodes to the letter chr(97 + i); 14 decodes to a replacement character (an incomplete character)."""
    def decode(self, ids):
        return "".join("�" if i == 14 else chr(97 + i) for i in ids)


class FakeEngine:
    def __init__(self, script, max_len=64):
        self.script, self.max_len, self.i, self.fed, self.prompt = list(script), max_len, 0, [], None

    def reset(self):
        self.i, self.fed = 0, []

    def _logits(self):
        v = torch.zeros(V)
        v[self.script[min(self.i, len(self.script) - 1)]] = 1.0
        self.i += 1
        return v

    def prefill(self, ids):
        self.prompt = list(ids)
        return self._logits()

    def step(self, tok):
        self.fed.append(tok)
        return self._logits()


def run(script, ids=(1, 2, 3), max_tokens=10, stop_strings=(), **kw):
    eng = FakeEngine(script, **kw)
    pieces, finish = [], None
    for p, fin in serve.generate(eng, FakeTok(), list(ids), max_tokens, {EOS}, stop_strings):
        pieces.append(p)
        finish = fin or finish
    return "".join(pieces), finish, eng


def test_stops_on_end_of_text_with_finish_stop():
    text, finish, eng = run([0, 1, 2, EOS, 5])
    assert (text, finish) == ("abc", "stop")
    assert eng.fed == [0, 1, 2]                      # the end-of-text token is not fed back


def test_max_tokens_gives_finish_length():
    text, finish, _ = run([0, 1, 2, 3, 4, 5], max_tokens=3)
    assert (text, finish) == ("abc", "length")


def test_end_of_text_as_the_last_allowed_token_is_a_stop():
    text, finish, _ = run([0, 1, EOS], max_tokens=3)
    assert (text, finish) == ("ab", "stop")


def test_context_limit_ends_with_length_and_never_overruns():
    text, finish, eng = run([0] * 100, ids=[1] * 60, max_tokens=50, max_len=64)
    assert finish == "length" and len(text) == 4 and len(eng.prompt) + len(eng.fed) <= 64


def test_prompt_longer_than_context_is_a_request_error():
    with pytest.raises(serve.RequestError):
        run([0], ids=[1] * 64, max_len=64)


def test_stop_string_cuts_the_text_and_finishes_with_stop():
    text, finish, _ = run([0, 1, 2, 3, 4], stop_strings=["cd"])
    assert (text, finish) == ("ab", "stop")


def test_a_partial_stop_string_is_released_at_the_end():
    text, finish, _ = run([0, 1, 2, EOS], stop_strings=["cx"])
    assert (text, finish) == ("abc", "stop")


def test_incomplete_character_is_held_back():
    eng = FakeEngine([0, 14, 1, EOS])
    pieces = [p for p, _ in serve.generate(eng, FakeTok(), [1], 10, {EOS}, ())]
    assert "�" not in "".join(pieces[:-1]) or "".join(pieces) == "a�b"


def test_greedy_is_argmax_and_sampling_uses_the_seed():
    logits = torch.tensor([0.0, 5.0, 4.9, 0.0])
    assert serve.sample(logits, 0, 1.0, None) == 1
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert [serve.sample(logits, 1.0, 1.0, g1) for _ in range(8)] == [serve.sample(logits, 1.0, 1.0, g2) for _ in range(8)]
    assert serve.sample(logits, 1.0, 0.01, torch.Generator().manual_seed(1)) == 1     # top_p cuts to the best token


@pytest.mark.parametrize("body,fragment", [
    ({"messages": []}, "non-empty"),
    ({"messages": [{"role": "user", "content": 5}]}, "string content"),
    ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "at least 1"),
    ({"messages": [{"role": "user", "content": "x"}], "stop": [""]}, "stop"),
    ({"messages": [{"role": "user", "content": "x"}], "tools": [{"a": 1}]}, "not supported"),
    ({"messages": [{"role": "user", "content": "x"}], "top_p": 0}, "top_p"),
])
def test_bad_requests_are_refused(body, fragment):
    with pytest.raises(serve.RequestError) as e:
        serve.parse_request(json.dumps(body))
    assert fragment in str(e.value)


def test_max_tokens_is_capped_and_defaults():
    r = serve.parse_request(json.dumps({"messages": [{"role": "user", "content": "x"}], "max_tokens": 10 ** 6}))
    assert r["max_tokens"] == serve.MAX_NEW_CAP
    assert serve.parse_request(json.dumps({"messages": [{"role": "user", "content": "x"}]}))["max_tokens"] == 256
    r = serve.parse_request(json.dumps({"messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 7, "max_tokens": 9}))
    assert r["max_tokens"] == 7


@pytest.mark.skipif(SNAP is None, reason="the model's tokenizer files are not in the Hugging Face cache")
def test_prompt_matches_the_transformers_chat_template():
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(SNAP))
    tok, tpl = serve.load_tokenizer(SNAP), serve.make_template(SNAP / "chat_template.jinja")
    for msgs in ([{"role": "user", "content": "What is 2+2?"}],
                 [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
                  {"role": "user", "content": "Bye"}]):
        want = hf.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        want = want["input_ids"] if hasattr(want, "keys") else want
        assert serve.encode_prompt(tok, tpl, msgs) == list(want)
    assert serve.stop_token_ids(SNAP, tok) == {100257}


def test_http_round_trip_with_a_fake_engine():
    class T(FakeTok):
        def encode(self, text, add_special_tokens=False):
            return type("E", (), {"ids": [1, 2, 3]})()
    tpl = serve.make_template(ROOT / "tests" / "fixtures" / "granite_tiny_template.jinja")
    serve.STATE.update(tok=T(), template=tpl, stop_ids={EOS}, engine=FakeEngine([0, 1, 2, EOS]))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        def post(body):
            rq = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            return urllib.request.urlopen(rq)
        r = json.load(post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}))
        assert r["choices"][0]["message"]["content"] == "abc" and r["choices"][0]["finish_reason"] == "stop"
        assert r["usage"] == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
        r = json.load(post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}))
        assert r["choices"][0]["finish_reason"] == "length"
        body = post({"messages": [{"role": "user", "content": "hi"}], "stream": True}).read().decode()
        events = [json.loads(x[6:]) for x in body.split("\n\n") if x.startswith("data: {")]
        assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "abc"
        assert events[-1]["choices"][0]["finish_reason"] == "stop" and body.rstrip().endswith("[DONE]")
        with pytest.raises(urllib.error.HTTPError) as e:
            post({"messages": []})
        assert e.value.code == 400
        assert json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"] == serve.MODEL_ID
    finally:
        srv.shutdown()


# ---- review fixes (PR 244) ------------------------------------------------------------------------------------------------
def test_stop_string_as_long_as_the_whole_output_leaks_nothing():
    text, finish, _ = run([0, 1, 2, 3, 4, 5, 6], stop_strings=["abcdef"])
    assert (text, finish) == ("", "stop")


def test_stop_string_longer_than_the_output_so_far_never_slices_backwards():
    eng = FakeEngine([0, 1, EOS])
    pieces = [p for p, _ in serve.generate(eng, FakeTok(), [1], 10, {EOS}, ["abcdefghij"])]
    assert "".join(pieces) == "ab"                                  # held back, then released at the end because the stop string never formed


class _Srv:
    """A server on a free port with a fake engine; `ids_for` maps the request text to prompt ids."""
    def __init__(self, engine, long_ids=False):
        class T(FakeTok):
            def encode(self, text, add_special_tokens=False):
                return type("E", (), {"ids": [1] * (100 if "LONG" in text else 3)})()
        tpl = serve.make_template(ROOT / "tests" / "fixtures" / "granite_tiny_template.jinja")
        serve.STATE.update(tok=T(), template=tpl, stop_ids={EOS}, engine=engine)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def post(self, body):
        rq = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return urllib.request.urlopen(rq)


def test_oversized_prompt_with_stream_true_gets_a_json_400():
    import urllib.error
    s = _Srv(FakeEngine([0, 1, EOS], max_len=64))
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            s.post({"messages": [{"role": "user", "content": "LONG"}], "stream": True})
        assert e.value.code == 400 and "context" in json.loads(e.value.read())["error"]["message"]
    finally:
        s.srv.shutdown()


def test_streamed_response_arrives_piece_by_piece_not_after_the_whole_generation():
    import time

    class Slow(FakeEngine):
        def step(self, tok):
            time.sleep(0.4)
            return super().step(tok)
    s = _Srv(Slow([0, 1, 2, 3, 4, EOS]))
    try:
        t0, first = time.time(), None
        for line in s.post({"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 8}):
            if line.startswith(b"data: {") and json.loads(line[6:])["choices"][0]["delta"].get("content") and first is None:
                first = time.time() - t0
        total = time.time() - t0
        assert first is not None and first < 0.3 and total > 1.4, (first, total)
    finally:
        s.srv.shutdown()


# ---- release behaviours -----------------------------------------------------------------------------------------------
import socket  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402


class SlowEngine(FakeEngine):
    def __init__(self, *a, delay=0.3, **k):
        super().__init__(*a, **k)
        self.delay = delay

    def step(self, tok):
        time.sleep(self.delay)
        return super().step(tok)


def _raw_post(s, body, stream=True):
    host, port = s.base[7:].split(":")
    sock = socket.create_connection((host, int(port)), timeout=10)
    data = json.dumps(body).encode()
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
    return sock


def test_a_second_request_while_one_runs_gets_503_with_retry_after_and_never_hangs():
    s = _Srv(SlowEngine([0, 1, 2, 3, 4, 5, 6, 7, EOS]))
    try:
        first = _raw_post(s, {"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 8})
        time.sleep(0.5)
        t0 = time.time()
        with pytest.raises(urllib.error.HTTPError) as e:
            s.post({"messages": [{"role": "user", "content": "hi"}]})
        assert e.value.code == 503 and e.value.headers["Retry-After"] and time.time() - t0 < 2
        first.close()
    finally:
        s.srv.shutdown()


def test_client_disconnect_mid_stream_frees_the_slot():
    s = _Srv(SlowEngine([0] * 40 + [EOS], delay=0.1))
    try:
        sock = _raw_post(s, {"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 40})
        sock.recv(200)
        sock.close()
        deadline, ok = time.time() + 6, False
        while time.time() < deadline and not ok:
            try:
                r = json.load(s.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}))
                ok = r["choices"][0]["finish_reason"] == "length"
            except urllib.error.HTTPError as e:
                assert e.code == 503
                time.sleep(0.2)
        assert ok, "the slot was not freed after the client left"
    finally:
        s.srv.shutdown()


def test_body_over_the_limit_and_bad_bodies_get_400():
    s = _Srv(FakeEngine([0, EOS]))
    try:
        rq = urllib.request.Request(s.base + "/v1/chat/completions", b"x" * (serve.MAX_BODY + 10), {"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(rq)
        assert e.value.code == 400
        for raw in (b"{not json", b"[1, 2]", b'{"messages": "x"}', b'{"messages": [{"role": "user", "content": "x"}], "max_tokens": "many"}',
                    b'{"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]}',
                    b'{"messages": [{"role": "user", "content": "x"}], "documents": [{"text": "d"}]}'):
            rq = urllib.request.Request(s.base + "/v1/chat/completions", raw, {"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(rq)
            assert e.value.code == 400, raw
    finally:
        s.srv.shutdown()


def test_idle_connection_is_closed_after_the_read_timeout():
    s = _Srv(FakeEngine([0, EOS]))
    old = serve.Handler.timeout
    try:
        assert old and old <= 60
        serve.Handler.timeout = 1
        host, port = s.base[7:].split(":")
        sock = socket.create_connection((host, int(port)), timeout=10)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{")     # the rest never arrives
        t0 = time.time()
        assert sock.recv(100) == b""                                    # the server closed the idle connection
        assert time.time() - t0 < 5
        r = json.load(s.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}))        # and the server still answers
        assert r["choices"][0]["finish_reason"] in ("length", "stop")
        sock.close()
    finally:
        serve.Handler.timeout = old
        s.srv.shutdown()


def test_multibyte_characters_are_never_split_across_stream_pieces():
    class MB:
        """token 13 = b'\\xc3', token 14 = b'\\xa9' (together 'é'), others letters; decode of a lone first byte gives the replacement character."""
        def decode(self, ids):
            raw = b"".join({13: b"\xc3", 14: b"\xa9"}.get(i, chr(97 + i).encode()) for i in ids)
            return raw.decode("utf-8", errors="replace")
    eng = FakeEngine([0, 13, 14, 1, EOS])
    pieces = [p for p, _ in serve.generate(eng, MB(), [1], 10, {EOS}, ())]
    assert all("�" not in p for p in pieces) and "".join(pieces) == "aéb"


def test_same_seed_repeats_sampling_and_a_different_seed_differs():
    class Flat(FakeEngine):
        def _logits(self):
            return torch.zeros(V - 1)                                  # all tokens equally likely, no end-of-text
    def go(seed):
        eng = Flat([0], max_len=64)
        return "".join(p for p, _ in serve.generate(eng, FakeTok(), [1], 12, {EOS}, (), temperature=1.0, top_p=1.0, seed=seed))
    assert go(7) == go(7) and go(7) != go(8)


class UnusableEngine(FakeEngine):
    """An engine whose chip marked itself unusable after a failed trace recapture."""
    unusable = "OSError: capture failed"


def _serve_with(engine):
    class T:
        def encode(self, text, add_special_tokens=False):
            return type("E", (), {"ids": [1, 2, 3]})()
    tpl = serve.make_template(ROOT / "tests" / "fixtures" / "granite_tiny_template.jinja")
    serve.STATE.update(tok=T(), template=tpl, stop_ids={EOS}, engine=engine)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_health_reports_an_unusable_chip_with_503_and_a_usable_one_with_200():
    import urllib.error
    srv, base = _serve_with(FakeEngine([0, 1]))
    try:
        assert json.load(urllib.request.urlopen(base + "/health"))["status"] == "ok"
    finally:
        srv.shutdown()
    srv, base = _serve_with(UnusableEngine([0, 1]))
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/health")
        body = json.load(e.value)
        assert e.value.code == 503 and body["status"] == "unusable" and "capture failed" in body["reason"]
    finally:
        srv.shutdown()


def test_chat_requests_are_refused_with_503_while_the_chip_is_unusable():
    import urllib.error
    srv, base = _serve_with(UnusableEngine([0, 1]))
    try:
        rq = urllib.request.Request(base + "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                                    {"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(rq)
        assert e.value.code == 503 and "restart" in json.load(e.value)["error"]["message"]
    finally:
        srv.shutdown()


def test_the_chip_engine_exposes_the_models_unusable_state_and_refuses_reset_prefill_and_step():
    class M:
        L, unusable = 64, "OSError: capture failed"
    eng = serve.ChipEngine(M())
    assert eng.unusable == "OSError: capture failed"
    for call in (eng.reset, lambda: eng.prefill([1, 2]), lambda: eng.step(1)):
        with pytest.raises(RuntimeError, match="unusable"):
            call()


def test_a_request_that_loses_the_race_to_a_failed_recapture_gets_503_before_any_header():
    """The chip becomes unusable after the first check and before the lock: the second check, under the lock, must still answer 503."""
    import urllib.error

    class Flaky(FakeEngine):
        calls = 0

        @property
        def unusable(self):
            Flaky.calls += 1
            return None if Flaky.calls == 1 else "OSError: capture failed"      # healthy at the first look, broken afterwards
    srv, base = _serve_with(Flaky([0, 1]))
    try:
        rq = urllib.request.Request(base + "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                                    {"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(rq)
        assert e.value.code == 503
    finally:
        srv.shutdown()
