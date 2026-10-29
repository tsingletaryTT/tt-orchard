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
