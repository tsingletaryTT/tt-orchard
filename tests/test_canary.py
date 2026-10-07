# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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


def test_ask_switches_thinking_off_in_the_request_body():
    seen = {}

    def http(url, body, timeout):
        seen.update(body=body)
        return {"choices": [{"message": {"role": "assistant", "content": "4"}}]}

    ask("http://x", "m", "p", http=http)
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_ask_explains_a_reply_cut_off_while_the_model_was_still_reasoning():
    reply = {"choices": [{"finish_reason": "length",
                          "message": {"role": "assistant", "content": None,
                                      "reasoning_content": "Let me think about " * 5}}]}
    with pytest.raises(CanaryError) as err:
        ask("http://x", "m", "p", http=lambda u, b, t: reply, max_tokens=16)
    text = str(err.value)
    assert "cut off while the model was still reasoning" in text
    assert "finish_reason=length" in text and "max_tokens=16" in text
    assert "reasoning chars=95" in text
    assert "thinking may not be switchable" in text


def test_ask_names_null_content_without_a_length_cut():
    reply = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None}}]}
    with pytest.raises(CanaryError, match=r"content was null.*finish_reason='stop'"):
        ask("http://x", "m", "p", http=lambda u, b, t: reply)


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
