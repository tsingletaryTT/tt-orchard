"""The canary call, the exact comparison, and the fake server it is tested against."""
import threading

import pytest

from orchard.canary import CanaryError, ask, compare
from orchard.fake_server import make_server


def test_ask_sends_a_greedy_chat_request():
    seen = {}

    def http(url, body, timeout):
        seen.update(url=url, body=body, timeout=timeout)
        return {"choices": [{"message": {"role": "assistant", "content": "4"}}]}

    assert ask("http://127.0.0.1:8000/", "qwen", "2+2?", http=http, timeout=5) == "4"
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["body"]["temperature"] == 0 and seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [{"role": "user", "content": "2+2?"}]


def test_ask_turns_a_connection_error_into_canary_error():
    def http(url, body, timeout):
        raise ConnectionRefusedError("refused")

    with pytest.raises(CanaryError, match="refused"):
        ask("http://127.0.0.1:1", "m", "p", http=http)


def test_ask_refuses_a_reply_without_content():
    with pytest.raises(CanaryError, match="choices"):
        ask("http://x", "m", "p", http=lambda u, b, t: {"error": "overloaded"})


def test_identical_answers_match():
    r = compare("Paris", "Paris")
    assert r.match and not r.whitespace_only


def test_whitespace_only_difference_is_flagged():
    r = compare("Paris\n", "Paris")
    assert not r.match and r.whitespace_only


def test_a_different_word_is_not_whitespace_only():
    r = compare("Paris", "Lyon")
    assert not r.match and not r.whitespace_only


def test_fake_server_answers_health_models_and_chat():
    srv = make_server(0, "forty-two", "fake")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        endpoint = f"http://127.0.0.1:{srv.server_address[1]}"
        assert ask(endpoint, "fake", "anything", timeout=5) == "forty-two"
    finally:
        srv.shutdown()
        srv.server_close()
