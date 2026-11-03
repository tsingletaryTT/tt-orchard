# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 360 and 361 in the role-fit test: does the server reuse a repeated long prefix, and how many tokens go to reasoning and to the answer?

The server is an injected function (the test's `http` argument), so no network and no chip.
"""
import json

import pytest

from orchard import rolefit


def chat(content="ok", completion=10, reasoning=None, details=None, prompt=1500, cached=None, finish="stop"):
    message = {"role": "assistant", "content": content, **({"reasoning_content": reasoning} if reasoning else {})}
    usage = {"completion_tokens": completion, "prompt_tokens": prompt}
    if details:
        usage["completion_tokens_details"] = details
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def cached_server(clock, *, caches: bool, report: bool, cold=3.0, warm=0.2):
    seen = set()

    def http(url, body, timeout):
        text = body["messages"][0]["content"]
        hit = caches and text in seen
        seen.add(text)
        clock.t += warm if hit else cold
        return chat(prompt=1500, cached=(1400 if hit else 0) if report else None, completion=1)
    return http


@pytest.mark.parametrize("report", [True, False])
def test_a_server_that_caches_the_prefix_is_recorded_as_reusing_it(report):
    clock = Clock()
    r = rolefit.prefix_reuse("http://x/v1", "m", http=cached_server(clock, caches=True, report=report), clock=clock)
    assert r["state"] == "pass", r
    assert r["basis"].startswith("the server") is report


@pytest.mark.parametrize("report", [True, False])
def test_a_server_without_prefix_caching_is_recorded_not_failed(report):
    clock = Clock()
    r = rolefit.prefix_reuse("http://x/v1", "m", http=cached_server(clock, caches=False, report=report), clock=clock)
    assert r["state"] == "not_checked" and "was not served faster" in r["detail"], r


def test_the_three_requests_are_one_token_greedy_and_the_repeat_equals_the_first():
    sent = []
    clock = Clock()

    def http(url, body, timeout):
        sent.append(body)
        clock.t += 1
        return chat()
    rolefit.prefix_reuse("http://x/v1", "m", http=http, clock=clock, chat_template_kwargs={"enable_thinking": False})
    assert len(sent) == 3
    assert all(b["max_tokens"] == 1 and b["temperature"] == 0 and b["chat_template_kwargs"] == {"enable_thinking": False} for b in sent)
    texts = [b["messages"][0]["content"] for b in sent]
    assert texts[0] == texts[2] and texts[0] != texts[1]


def test_a_failing_request_is_not_checked_not_an_exception():
    def http(url, body, timeout):
        raise OSError("refused")
    r = rolefit.prefix_reuse("http://x/v1", "m", http=http)
    assert r["state"] == "not_checked" and "refused" in r["detail"]


def test_a_replayed_reply_records_its_reasoning_and_answer_tokens():
    turn = rolefit.Turn(source="s", turn=1, messages=[{"role": "user", "content": "go"}], recorded={}, finish_reason=None, prompt_chars=2)
    http = lambda url, body, timeout: chat("408", completion=100, reasoning="x" * 90, details={"reasoning_tokens": 90})
    (row,) = rolefit.replay("http://x/v1", "m", [turn], http=http)
    assert (row["completion_tokens"], row["reasoning_tokens"], row["answer_tokens"], row["token_basis"]) == (100, 90, 10, "reported")


def test_a_failed_replay_has_no_token_counts():
    turn = rolefit.Turn(source="s", turn=1, messages=[{"role": "user", "content": "go"}], recorded={}, finish_reason=None, prompt_chars=2)

    def http(url, body, timeout):
        raise OSError("down")
    (row,) = rolefit.replay("http://x/v1", "m", [turn], http=http)
    assert row["verdict"] == "error" and row["reasoning_tokens"] is None and row["token_basis"] is None


def test_the_token_report_gives_a_distribution_the_basis_of_each_split_and_a_cap():
    rows = [{"verdict": "ok", "completion_tokens": c, "reasoning_tokens": r, "answer_tokens": c - r, "token_basis": b}
            for c, r, b in ((100, 80, "reported"), (300, 250, "estimated by characters"), (40, 0, "no reasoning text"))]
    rows.append({"verdict": "error", "completion_tokens": None, "reasoning_tokens": None, "answer_tokens": None, "token_basis": None})
    rep = rolefit.token_report(rows)
    assert rep["completion_tokens"]["max"] == 300 and rep["reasoning_tokens"]["median"] == 80 and rep["answer_tokens"]["n"] == 3
    assert rep["split_basis"] == {"reported": 1, "estimated by characters": 1, "no reasoning text": 1}
    assert rep["cap_evidence"]["supported_cap"] == 450 and "estimate" in rep["note"]


def test_a_model_that_does_not_reason_has_no_cap_evidence():
    rows = [{"verdict": "ok", "completion_tokens": 20, "reasoning_tokens": 0, "answer_tokens": 20, "token_basis": "no reasoning text"}]
    assert "cap_evidence" not in rolefit.token_report(rows)


def test_main_records_the_prefix_and_token_parts_and_they_do_not_decide_the_verdict(tmp_path):
    from test_rolefit import write_log, log_line
    write_log(tmp_path, "run-a", [log_line(1)])
    clock_calls = []

    def http(url, body, timeout):
        clock_calls.append(body)
        if body.get("tools"):
            return chat(content="", completion=5) | {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "shell", "arguments": json.dumps({"command": "ls"})}}]}, "finish_reason": "tool_calls"}]}
        if body["max_tokens"] == 64 and "7 times 6" in body["messages"][0]["content"]:
            return chat("42")
        return chat("fine", completion=5)
    out = tmp_path / "result.json"
    rc = rolefit.main(["--endpoint", "http://127.0.0.1:1/v1", "--model", "m", "--logs", str(tmp_path), "--out", str(out), "--sizes", "64"], http=http)
    result = json.loads(out.read_text())
    assert "prefix_reuse" in result and "tokens" in result and result["tokens"]["completion_tokens"]["n"] >= 1
    assert result["verdict"]["failed"] == [] and rc == 0
