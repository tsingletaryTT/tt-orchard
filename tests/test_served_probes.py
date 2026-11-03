# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/served_probes.py: the multi-request probes (code, batching, long prefill, reasoning, prefix reuse) against a local fake HTTP server.

No chip and no model. The fake's behaviour is switched per test (a serialising server, a batching one, a server that drops thinking).
Timing in these tests has a wide margin (a request takes 0.3 s, so a serialised burst of four takes about four times a batched one), and
the batching judge is also tested on numbers alone, so no verdict rests on a clock in the fake.
"""
import json
import threading
import time

import pytest

from orchard import served_probes as sp
from test_served_verify import Fake


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.close()


def chat_reply(content, tokens=20, finish="stop", prompt_tokens=10, **message):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": finish}],
                       "usage": {"completion_tokens": tokens, "prompt_tokens": prompt_tokens}}).encode()


GOOD_CODE = "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"


# ---- 369: generated code is parsed, never run --------------------------------------------------------------------------------------

def test_good_code_in_a_fence_passes_and_the_result_says_the_check_is_static():
    r = sp.check_generated_code(GOOD_CODE)
    assert r["state"] == "pass" and r["static"] is True and "never run" in r["detail"], r


def test_good_code_without_a_fence_passes():
    assert sp.check_generated_code(GOOD_CODE.strip("`").replace("python\n", "", 1))["state"] == "pass"


@pytest.mark.parametrize("code,words", [
    ("def is_prime(n)\n    return True\n", "does not parse"),
    ("def is_odd(n):\n    return n % 2\n", "no function named is_prime"),
    ("def is_prime(n):\n    print(n)\n", "no `return` with a value"),
    ("def is_prime(n):\n    return\n", "no `return` with a value"),
    ("def is_prime(n):\n    return helper(n)\n", "helper"),
    ("def is_prime(n):\n    while True:\n        n += 1\n    return n\n", "while True"),
    ("def is_prime(n):\n    for x in range(n):\n        while True:\n            break\n    while 1:\n        pass\n    return n\n", "while True"),
    ("", "no text"),
    ("   \n", "no text"),
    ("I cannot write that.", "does not parse"),
    ("\x00def is_prime(n): return 1", "does not parse"),
    ("def is_prime(n):\n" + "    if n:\n" * 3000 + "        return 1\n", "does not parse"),
    ("def is_prime(n):\n    return " + "-" * 9000 + "n\n", "does not parse"),          # the parser runs out of stack and raises MemoryError
])
def test_each_code_fault_is_named(code, words):
    r = sp.check_generated_code(code)
    assert r["state"] == "fail" and words in r["detail"], r
    assert r["static"] is True


def test_a_while_true_with_an_exit_is_fine_and_a_break_in_an_inner_loop_is_not_an_exit():
    ok = "def is_prime(n):\n    while True:\n        if n:\n            return True\n"
    assert sp.check_generated_code(ok)["state"] == "pass"
    ok2 = "def is_prime(n):\n    while True:\n        n += 1\n        if n > 5:\n            break\n    return n\n"
    assert sp.check_generated_code(ok2)["state"] == "pass"
    bad = "def is_prime(n):\n    while True:\n        for i in range(3):\n            break\n    return n\n"
    assert sp.check_generated_code(bad)["state"] == "fail"


def test_names_from_imports_comprehensions_and_module_level_are_defined():
    code = ("import math\nLIMIT = 10\n\ndef helper(x):\n    return x\n\n"
            "def is_prime(n):\n    return all(n % d for d in range(2, int(math.sqrt(n)) + 1)) and helper(n) < LIMIT * 1000\n")
    assert sp.check_generated_code(code)["state"] == "pass"


def test_the_reply_is_never_executed():
    marker = "orchard_probe_ran_the_reply"
    code = f"import builtins\nbuiltins.{marker} = True\ndef is_prime(n):\n    return True\n"
    sp.check_generated_code(code)
    import builtins
    assert not hasattr(builtins, marker)


def test_the_probe_sends_a_greedy_bounded_request_with_the_can_extra_and_judges_the_reply(fake):
    seen = []
    fake.post["/v1/chat/completions"] = lambda b: (seen.append(b), (200, chat_reply(GOOD_CODE)))[1]
    r = sp.probe_code(fake.base, "m", {"chat_template_kwargs": {"enable_thinking": False}})
    assert r["state"] == "pass", r
    assert seen[0]["temperature"] == 0 and seen[0]["max_tokens"] == 256 and seen[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_empty_content_with_finish_reason_length_is_named_as_a_spent_budget(fake):
    fake.post["/v1/chat/completions"] = lambda b: (200, chat_reply("", finish="length"))
    r = sp.probe_code(fake.base, "m")
    assert r["state"] == "fail" and "token budget" in r["detail"]


def test_probe_code_handles_a_missing_route_an_error_and_a_dead_server(fake):
    assert sp.probe_code(fake.base, "m")["state"] == "not_checked"             # no chat route: 404
    fake.post["/v1/chat/completions"] = lambda b: (500, b"boom")
    assert sp.probe_code(fake.base, "m")["state"] == "fail"
    fake.close()
    assert sp.probe_code(fake.base, "m", timeout=2)["state"] == "fail"


# ---- 371: batching ------------------------------------------------------------------------------------------------------------------

def test_the_judge_reads_a_ratio_near_one_as_serialised_and_well_above_one_as_batching():
    kw = dict(single_rate=10.0, n=4, max_running=None)
    assert sp.judge_batching(tokens=128, seconds=12.8, **kw)["state"] == "fail"        # 10 tok/s aggregate: ratio 1.0
    assert sp.judge_batching(tokens=128, seconds=3.2, **kw)["state"] == "pass"         # 40 tok/s: ratio 4.0
    assert sp.judge_batching(tokens=128, seconds=9.0, **kw)["state"] == "not_checked"  # ratio 1.42: between the two thresholds


def test_the_engines_own_running_count_decides_when_it_is_reported():
    kw = dict(single_rate=10.0, tokens=128, seconds=12.8, n=4)
    assert sp.judge_batching(max_running=1, **kw)["state"] == "fail"
    assert sp.judge_batching(max_running=3, **kw)["state"] == "pass"


@pytest.mark.parametrize("single", [None, 0, -1, "x", True])
def test_no_single_stream_rate_means_not_checked(single):
    assert sp.judge_batching(single_rate=single, tokens=100, seconds=1, n=4, max_running=None)["state"] == "not_checked"


@pytest.mark.parametrize("tokens,seconds", [(0, 1), (None, 1), (True, 1), (10, 0), (10, -1), (10, None)])
def test_a_burst_with_no_token_count_or_no_time_is_not_checked(tokens, seconds):
    assert sp.judge_batching(single_rate=10, tokens=tokens, seconds=seconds, n=4, max_running=None)["state"] == "not_checked"


@pytest.mark.parametrize("seqs", [None, 0, 1, True, "4"])
def test_the_probe_runs_only_when_the_can_declares_max_num_seqs_above_one(fake, seqs):
    fake.post["/v1/chat/completions"] = lambda b: (200, chat_reply("x"))
    r = sp.probe_batching(fake.base, "m", None, single_rate=10, max_num_seqs=seqs)
    assert r["state"] == "not_checked" and ("POST", "/v1/chat/completions") not in fake.log


def _burst_fake(fake, *, serial: bool, per_request_s: float = 0.3, tokens: int = 32):
    lock = threading.Lock()
    seen = []

    def handler(body):
        seen.append(body)
        if serial:
            with lock:
                time.sleep(per_request_s)
        else:
            time.sleep(per_request_s)
        return 200, chat_reply("1\n2\n3", tokens=tokens)
    fake.post["/v1/chat/completions"] = handler
    return seen


def test_a_serialising_server_fails_and_a_batching_one_passes(fake):
    single = 32 / 0.3                                             # the rate one request alone would show
    seen = _burst_fake(fake, serial=True)
    r = sp.probe_batching(fake.base, "m", None, single_rate=single, max_num_seqs=4)
    assert r["state"] == "fail" and r["ratio"] < 1.2, r
    _burst_fake(fake, serial=False)
    r = sp.probe_batching(fake.base, "m", None, single_rate=single, max_num_seqs=4)
    assert r["state"] == "pass" and r["ratio"] > 1.5, r
    assert len(seen) == 4


def test_the_burst_is_bounded_in_requests_and_sends_ignore_eos_with_a_fixed_token_count(fake):
    seen = _burst_fake(fake, serial=False, per_request_s=0.0)
    sp.probe_batching(fake.base, "m", {"chat_template_kwargs": {"enable_thinking": False}}, single_rate=10, max_num_seqs=20)
    assert len(seen) == sp.MAX_BURST
    assert all(b["ignore_eos"] is True and b["max_tokens"] == sp.BURST_TOKENS and b["temperature"] == 0 for b in seen)
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in seen)


def test_a_failed_request_in_the_burst_fails_the_probe_and_a_missing_usage_is_not_checked(fake):
    fake.post["/v1/chat/completions"] = lambda b: (500, b"x")
    assert sp.probe_batching(fake.base, "m", None, single_rate=10, max_num_seqs=2)["state"] == "fail"
    fake.post["/v1/chat/completions"] = lambda b: (200, json.dumps({"choices": []}).encode())
    assert sp.probe_batching(fake.base, "m", None, single_rate=10, max_num_seqs=2)["state"] == "not_checked"


def test_the_burst_waits_no_longer_than_the_timeout(fake):
    fake.post["/v1/chat/completions"] = lambda b: (time.sleep(3), (200, chat_reply("x")))[1]
    t0 = time.monotonic()
    r = sp.probe_batching(fake.base, "m", None, single_rate=10, max_num_seqs=2, timeout=0.5)
    assert r["state"] == "fail" and time.monotonic() - t0 < 2.5


def test_the_running_count_is_read_from_metrics_during_the_burst(fake):
    fake.metrics_body = "vllm:num_requests_running{engine=\"0\"} 3.0\n"
    _burst_fake(fake, serial=True)
    r = sp.probe_batching(fake.base, "m", None, single_rate=32 / 0.3, max_num_seqs=4)
    assert r["state"] == "pass" and r["max_running"] == 3          # the engine's count decides; the serialised timing does not


# ---- 371: a long prompt -------------------------------------------------------------------------------------------------------------

def _long_fake(fake, *, answer="tangerine", per_token=0.2):
    seen = []

    def handler(body):
        text = body["messages"][0]["content"]
        seen.append(body)
        tokens = int(len(text.split()) * per_token * 10)                # two tokens a word
        return 200, chat_reply(answer, prompt_tokens=tokens)
    fake.post["/v1/chat/completions"] = handler
    return seen


def test_the_prompt_is_built_from_the_measured_token_count_not_a_word_count(fake):
    seen = _long_fake(fake)
    r = sp.probe_long_prefill(fake.base, "m", None, target_tokens=2000)
    assert r["state"] == "pass" and r["answered"] is True and r["recalled"] is True, r
    assert 1700 <= r["prompt_tokens"] <= 2300, r                          # the fake counts two tokens a word, so a word count would be far off
    assert len(seen) == 2 and seen[0]["max_tokens"] == 1 and seen[1]["max_tokens"] == 32      # a calibration request, then the long one


def test_a_prompt_the_declared_context_cannot_hold_is_not_sent(fake):
    seen = _long_fake(fake)
    r = sp.probe_long_prefill(fake.base, "m", None, max_model_len=1024)
    assert r["state"] == "not_checked" and not seen


def test_an_empty_reply_to_the_long_prompt_fails_and_a_wrong_word_still_counts_as_reached(fake):
    _long_fake(fake, answer="")
    assert sp.probe_long_prefill(fake.base, "m", None)["state"] == "fail"
    _long_fake(fake, answer="banana")
    r = sp.probe_long_prefill(fake.base, "m", None)
    assert r["state"] == "pass" and r["recalled"] is False


def test_a_server_that_reports_no_prompt_tokens_is_not_checked_and_an_error_is_a_fail(fake):
    fake.post["/v1/chat/completions"] = lambda b: (200, json.dumps({"choices": [{"message": {"content": "x"}}]}).encode())
    assert sp.probe_long_prefill(fake.base, "m", None)["state"] == "not_checked"
    fake.post["/v1/chat/completions"] = lambda b: (500, b"x")
    assert sp.probe_long_prefill(fake.base, "m", None)["state"] == "not_checked"       # the calibration failed: nothing to size the prompt with
    fake.close()
    assert sp.probe_long_prefill(fake.base, "m", None, timeout=2)["state"] == "fail"


def test_a_timeout_on_the_long_request_is_a_fail(fake):
    def handler(body):
        if body["max_tokens"] == 1:
            return 200, chat_reply("x", prompt_tokens=400)
        time.sleep(3)
        return 200, chat_reply("tangerine", prompt_tokens=2000)
    fake.post["/v1/chat/completions"] = handler
    r = sp.probe_long_prefill(fake.base, "m", None, timeout=0.5)
    assert r["state"] == "fail"


# ---- 372: reasoning -----------------------------------------------------------------------------------------------------------------

def _reasoning_fake(fake, mode="populated"):
    seen = []

    def handler(body):
        seen.append(body)
        thinking_off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
        small = body["max_tokens"] < 100
        if thinking_off:
            return 200, chat_reply("408", tokens=3)
        if small:
            return 200, chat_reply("" if mode != "leaking" else "<think>hm", tokens=body["max_tokens"], finish="length",
                                   **({"reasoning_content": "thinking"} if mode == "populated" else {}))
        if mode == "populated":
            return 200, chat_reply("408", tokens=300, reasoning_content="17*24 = 408")
        if mode == "leaking":
            return 200, chat_reply("<think>17*24</think>408", tokens=300)
        if mode == "dropped":
            return 200, chat_reply("408", tokens=300)
        if mode == "no_switch":
            return 200, chat_reply("408", tokens=3, reasoning_content="x")
        return 200, chat_reply("", tokens=body["max_tokens"], finish="length")
    fake.post["/v1/chat/completions"] = handler
    return seen


def states(results):
    return {r["name"]: r["state"] for r in results}


def test_a_populated_reasoning_field_a_working_switch_and_a_budget_sweep_pass(fake):
    seen = _reasoning_fake(fake)
    out = sp.probe_reasoning(fake.base, "m", {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
                             log_text="Namespace(reasoning_parser='qwen3')")
    assert states(out) == {"reasoning_field": "pass", "reasoning_parser": "pass", "thinking_switch": "pass", "budget_sweep": "pass"}, out
    assert len(seen) == 3 and [b["max_tokens"] for b in seen] == [sp.BUDGETS[0], sp.BUDGETS[1], sp.BUDGETS[1]]
    assert all("enable_thinking" not in (b.get("chat_template_kwargs") or {}) for b in seen[:2])        # the can's thinking-off switch is removed for the on requests
    assert seen[2]["chat_template_kwargs"]["enable_thinking"] is False and all(b["top_k"] == 1 for b in seen)
    sweep = [r for r in out if r["name"] == "budget_sweep"][0]
    assert sweep["small_answered"] is False and sweep["large_answered"] is True


@pytest.mark.parametrize("mode,field", [("leaking", "fail"), ("dropped", "fail")])
def test_leaked_and_dropped_thinking_are_failures_with_their_class_named(fake, mode, field):
    _reasoning_fake(fake, mode)
    out = sp.probe_reasoning(fake.base, "m", None)
    first = out[0]
    assert first["state"] == field and first["kind"] == mode, first


def test_a_switch_that_does_not_cut_the_tokens_fails(fake):
    def handler(body):
        return 200, chat_reply("408", tokens=40, reasoning_content="x")
    fake.post["/v1/chat/completions"] = handler
    out = {r["name"]: r for r in sp.probe_reasoning(fake.base, "m", None)}
    assert out["thinking_switch"]["state"] == "fail"


def test_no_answer_even_at_the_large_budget_fails_the_sweep(fake):
    _reasoning_fake(fake, "never")
    out = {r["name"]: r for r in sp.probe_reasoning(fake.base, "m", None)}
    assert out["budget_sweep"]["state"] == "fail" and out["reasoning_field"]["state"] == "fail"


def test_the_parser_check_is_not_checked_without_a_log_line_and_fails_when_the_log_shows_none(fake):
    _reasoning_fake(fake)
    out = {r["name"]: r for r in sp.probe_reasoning(fake.base, "m", None, log_text="INFO nothing relevant")}
    assert out["reasoning_parser"]["state"] == "not_checked"
    out = {r["name"]: r for r in sp.probe_reasoning(fake.base, "m", None, log_text="args reasoning_parser=None")}
    assert out["reasoning_parser"]["state"] == "fail"
    out = {r["name"]: r for r in sp.probe_reasoning(fake.base, "m", None, log_text=None)}
    assert out["reasoning_parser"]["state"] == "not_checked"


def test_no_chat_route_is_not_checked_and_a_dead_server_is_a_fail(fake):
    assert states(sp.probe_reasoning(fake.base, "m", None)) == {"reasoning_field": "not_checked"}
    fake.close()
    assert states(sp.probe_reasoning(fake.base, "m", None, timeout=2)) == {"reasoning_field": "fail"}


# ---- 360: a repeated prefix ---------------------------------------------------------------------------------------------------------

def _prefix_fake(fake, *, cache: bool, report_cached: bool, slow=0.3):
    seen_prompts = set()

    def handler(body):
        text = body["messages"][0]["content"]
        hit = cache and text in seen_prompts
        if body["max_tokens"] == 1 and "Document" in text:
            seen_prompts.add(text)
        time.sleep(0.02 if hit else slow)
        usage = {"completion_tokens": 1, "prompt_tokens": 1500}
        if report_cached:
            usage["prompt_tokens_details"] = {"cached_tokens": 1400 if hit else 0}
        return 200, json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "length"}], "usage": usage}).encode()
    fake.post["/v1/chat/completions"] = handler


@pytest.mark.parametrize("report", [True, False])
def test_a_server_that_caches_the_prefix_passes_and_one_that_does_not_fails_when_declared(fake, report):
    _prefix_fake(fake, cache=True, report_cached=report)
    r = sp.probe_prefix_reuse(fake.base, "m", None, declared=True)
    assert r["state"] == "pass", r
    _prefix_fake(fake, cache=False, report_cached=report)
    r = sp.probe_prefix_reuse(fake.base, "m", None, declared=True)
    assert r["state"] == "fail", r
    assert sp.probe_prefix_reuse(fake.base, "m", None, declared=False)["state"] == "not_checked"


def test_the_repeat_is_byte_for_byte_the_first_prompt_and_the_control_differs(fake):
    prompts = []

    def handler(body):
        prompts.append(body["messages"][0]["content"])
        return 200, chat_reply("x", prompt_tokens=1500)
    fake.post["/v1/chat/completions"] = handler
    sp.probe_prefix_reuse(fake.base, "m", None)
    assert len(prompts) == 4                                    # calibration, first, control, repeat
    assert prompts[1] == prompts[3] and prompts[1] != prompts[2]


def test_prefix_probe_handles_a_failed_request_and_a_dead_server(fake):
    fake.post["/v1/chat/completions"] = lambda b: (500, b"x")
    assert sp.probe_prefix_reuse(fake.base, "m", None)["state"] == "not_checked"
    fake.close()
    assert sp.probe_prefix_reuse(fake.base, "m", None, timeout=2)["state"] == "fail"
