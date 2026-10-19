# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/measure_serving.py: time a served model on a fixed prompt set, so a performance claim has a number and a context.

The arithmetic is tested here with a fake clock and a fake event stream: time to first token, decode rate (tokens after the first, over the
time after the first), the median over repetitions and the hash of the prompt set. The HTTP and server parts are measured on the chips.
"""
import json

import pytest

from orchard import measure_serving as m


def sse(*chunks):
    return [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]


def delta(**kw):
    return {"choices": [{"delta": kw, "finish_reason": None}]}


def test_the_first_token_is_the_first_chunk_with_content_reasoning_or_a_tool_call():
    lines = sse(delta(role="assistant"), delta(reasoning_content="We"), delta(content="x"),
                {"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 7}})
    ticks = iter([1.0, 1.5, 2.0, 3.0, 4.0])
    ev = m.read_stream(lines, clock=lambda: next(ticks))
    assert ev["t_first"] == 1.5 and ev["t_last"] == 3.0 and ev["completion_tokens"] == 3 and ev["prompt_tokens"] == 7
    assert ev["text"] == "Wex"


def test_a_tool_call_chunk_counts_as_the_first_token_and_its_arguments_are_kept():
    lines = sse(delta(role="assistant"), delta(tool_calls=[{"index": 0, "function": {"name": "run_tests", "arguments": "{\"c"}}]),
                delta(tool_calls=[{"index": 0, "function": {"arguments": "md\": 1}"}}]))
    ticks = iter([0.0, 1.0, 2.0])
    ev = m.read_stream(lines, clock=lambda: next(ticks))
    assert ev["t_first"] == 1.0 and ev["tool_calls"] == [{"name": "run_tests", "arguments": "{\"cmd\": 1}"}]


def test_an_error_the_server_sends_inside_the_stream_is_raised_with_its_message():
    lines = sse({"error": {"message": "This block-output model does not support min_tokens=128", "code": 400}})
    with pytest.raises(m.StreamError) as e:
        m.read_stream(lines, clock=iter([0.0, 1.0]).__next__)
    assert "min_tokens=128" in str(e.value)


def test_a_stream_with_no_token_is_an_error():
    with pytest.raises(m.NoTokens):
        m.read_stream(sse(delta(role="assistant")), clock=iter([0.0, 1.0]).__next__)


def test_timing_is_time_to_first_token_and_decode_rate_over_the_tokens_after_the_first():
    t = m.timing(sent=10.0, t_first=12.0, t_last=17.0, completion_tokens=51)
    assert t["ttft_s"] == pytest.approx(2.0)
    assert t["decode_tok_s"] == pytest.approx(50 / 5.0)              # 50 tokens after the first, in 5 s
    assert t["completion_tokens"] == 51


def test_a_single_token_has_no_decode_rate():
    assert m.timing(sent=0.0, t_first=1.0, t_last=1.0, completion_tokens=1)["decode_tok_s"] is None
    assert m.timing(sent=0.0, t_first=1.0, t_last=1.0, completion_tokens=None)["decode_tok_s"] is None


def test_the_summary_gives_the_median_and_the_range_of_the_repetitions():
    runs = [{"ttft_s": 1.0, "decode_tok_s": 10.0, "completion_tokens": 128},
            {"ttft_s": 8.0, "decode_tok_s": 80.0, "completion_tokens": 130},
            {"ttft_s": 3.0, "decode_tok_s": 30.0, "completion_tokens": 129}]
    s = m.summarize(runs)                                                       # a median of 3.0; the mean would be 4.0
    assert s["ttft_s"] == {"median": 3.0, "min": 1.0, "max": 8.0}
    assert s["decode_tok_s"] == {"median": 30.0, "min": 10.0, "max": 80.0}
    assert s["completion_tokens"]["min"] == 128 and s["runs"] == 3


def test_a_run_without_a_decode_rate_is_left_out_of_that_statistic_only():
    s = m.summarize([{"ttft_s": 1.0, "decode_tok_s": None, "completion_tokens": 1}, {"ttft_s": 2.0, "decode_tok_s": 8.0, "completion_tokens": 20}])
    assert s["decode_tok_s"] == {"median": 8.0, "min": 8.0, "max": 8.0} and s["ttft_s"]["median"] == 1.5


def test_the_prompt_set_hash_changes_with_any_prompt_or_setting():
    base = m.prompt_set_hash(m.PROMPTS)
    assert base == m.prompt_set_hash(m.PROMPTS) and len(base) == 64
    changed = [dict(p) for p in m.PROMPTS]
    changed[0] = {**changed[0], "max_tokens": changed[0]["max_tokens"] + 1}
    assert m.prompt_set_hash(changed) != base


def test_every_prompt_asks_for_greedy_decoding_and_the_set_has_the_kinds_the_operator_asked_for():
    kinds = {p["id"] for p in m.PROMPTS}
    assert kinds == {"pi-coding", "tool-call", "tool-call-write", "chat"}           # the second tool prompt has a long call (the first is 55 tokens)
    write = next(p for p in m.PROMPTS if p["id"] == "tool-call-write")
    assert [t["function"]["name"] for t in write["tools"]] == ["write_file"] and write["max_tokens"] >= 2 * m.MIN_TOKENS_ASKED
    assert all(p["temperature"] == 0 for p in m.PROMPTS)
    tool = next(p for p in m.PROMPTS if p["id"] == "tool-call")
    assert tool["tools"]
    # The TT block-output server owns its sampling and rejects min_tokens ("accepted: 0"); a forced minimum would also push a tool call past
    # its end. The prompts have to make the model write 100 tokens on its own, and the file records whether they did.
    assert all("min_tokens" not in p for p in m.PROMPTS)


def test_the_request_body_carries_the_prompt_settings_and_asks_for_usage():
    body = m.request_body("served-name", m.PROMPTS[0])
    assert body["model"] == "served-name" and body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0 and body["max_tokens"] == m.PROMPTS[0]["max_tokens"]


def test_spec_decode_counters_are_read_from_the_metrics_text_when_the_server_has_them():
    text = ("# HELP x\nvllm:spec_decode_num_accepted_tokens_total 40\nvllm:spec_decode_num_draft_tokens_total 100\n"
            "vllm:generation_tokens_total 500\n")
    assert m.spec_decode_counters(text) == {"vllm:spec_decode_num_accepted_tokens_total": 40.0, "vllm:spec_decode_num_draft_tokens_total": 100.0}
    assert m.spec_decode_counters("vllm:generation_tokens_total 5\n") == {}
    assert m.acceptance(before={"a_accepted": 0}, after={}) is None


def test_acceptance_is_accepted_over_drafted_between_two_scrapes():
    b = {"vllm:spec_decode_num_accepted_tokens_total": 10.0, "vllm:spec_decode_num_draft_tokens_total": 40.0}
    a = {"vllm:spec_decode_num_accepted_tokens_total": 70.0, "vllm:spec_decode_num_draft_tokens_total": 140.0}
    assert m.acceptance(before=b, after=a) == pytest.approx(0.6)
    assert m.acceptance(before=a, after=a) is None                    # nothing was drafted between the scrapes


def test_the_warmup_request_is_short_and_the_server_will_accept_it():
    """A min_tokens above max_tokens is a 400 from the server (found on the first run: the warm-up kept the chat prompt's min_tokens of 128)."""
    body = m.request_body("served", m.warmup_prompt())
    assert body["max_tokens"] <= 16 and "min_tokens" not in body
    for p in m.PROMPTS:
        b = m.request_body("served", p)
        assert b["max_tokens"] >= 2 * m.MIN_TOKENS_ASKED, p["id"]            # room to write 100 tokens and finish
