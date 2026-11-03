# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/served_verify.py additions for issues 360, 361, 370, 372, 373 and 385: judges that read text or a parsed reply, no server needed.

The log fixtures use the line formats of vLLM 0.10 (the v1 engine: `GPU KV cache size`, `Maximum concurrency`) and of the older engine
(`# GPU blocks`). The formats differ between releases, so a fixture names the release it follows.
"""
import json

import pytest

from orchard import served_verify as sv


def models(*names, max_len=4096):
    return json.dumps({"object": "list", "data": [{"id": n, "max_model_len": max_len} for n in names]}).encode()


# ---- 385: a --served-model-name alias set only through extra_args ------------------------------------------------------------------

@pytest.mark.parametrize("args,want", [
    (("--served-model-name", "alias"), ("alias",)),
    (("--served-model-name", "a", "b", "--enable-prefix-caching"), ("a", "b")),
    (("--served_model_name=alias",), ("alias",)),
    (("--served-model-name=a", "--served_model_name", "b"), ("b",)),             # the last use wins
    (("--other", "x"), ()),
    ((), ()),
    (("--served-model-name",), ()),
    (("--served-model-name", ""), ()),
    (("--served-model-name", "x" * 600), ()),
    (("--served-model-names", "a"), ()),                                       # a longer option name is another option
    ((None, 3, "--served-model-name", "a"), ("a",)),
    (None, ()),
])
def test_served_names_reads_the_alias_options(args, want):
    assert sv.served_names(args) == want


def test_a_name_that_starts_with_a_dash_ends_the_list():
    assert sv.served_names(("--served-model-name", "a", "-x", "b")) == ("a",)


def test_the_request_carries_the_declared_alias_when_the_server_lists_two_names():
    body = models("org/real", "alias")
    assert sv.request_model(body, ("alias",)) == "alias"
    assert sv.request_model(body, ("missing", "alias")) == "alias"       # the first declared name that is listed
    assert sv.request_model(body, ()) == "org/real"                       # nothing declared: the first listed id
    assert sv.request_model(body, ("missing",)) is None
    assert sv.request_model(b"not json", ("alias",)) is None


def test_the_listing_passes_on_a_declared_alias_that_is_not_the_first_listed_name():
    r = sv.check_listing(200, models("org/real", "alias"), declared_names=("alias",), declared_context=4096)
    assert r["state"] == "pass" and r["served"]["id"] == "alias", r


def test_the_listing_fails_with_the_declared_names_when_it_lists_none_of_them():
    r = sv.check_listing(200, models("org/real", "other"), declared_names=("alias", "alias2"))
    assert r["state"] == "fail" and "alias" in r["detail"] and "alias2" in r["detail"], r


def test_the_old_single_declared_name_still_works_with_the_new_list():
    assert sv.check_listing(200, models("a", "b"), declared_name="b")["served"]["id"] == "b"
    assert sv.check_listing(200, models("a", "b"), declared_name="c")["state"] == "fail"


# ---- 370: layers loaded and the KV cache, from the server log ------------------------------------------------------------------------

V1_LOG = """\
INFO 11-02 10:00:00 [loader.py:211] loaded 36 layers on 4 devices
INFO 11-02 10:00:20 [kv_cache_utils.py:861] GPU KV cache size: 131,072 tokens
INFO 11-02 10:00:20 [kv_cache_utils.py:865] Maximum concurrency for 8,192 tokens per request: 16.00x
"""
V0_LOG = "INFO 11-02 [worker.py:100] # GPU blocks: 512, # CPU blocks: 0\n"


def by_name(results):
    return {r["name"]: r for r in results}


def test_a_log_that_matches_config_and_holds_a_full_sequence_passes():
    r = by_name(sv.check_engine_log(V1_LOG, config={"num_hidden_layers": 36}, max_model_len=8192, max_num_seqs=8))
    assert r["layers_loaded"]["state"] == "pass" and r["kv_cache"]["state"] == "pass", r
    assert r["kv_cache"]["capacity_tokens"] == 131072 and r["kv_cache"]["concurrency"] == 16.0
    assert "max_num_seqs is 8" in r["kv_cache"]["detail"]


def test_a_layer_count_that_differs_from_config_fails():
    r = by_name(sv.check_engine_log(V1_LOG, config={"num_hidden_layers": 40}, max_model_len=8192))
    assert r["layers_loaded"]["state"] == "fail" and "36" in r["layers_loaded"]["detail"] and "40" in r["layers_loaded"]["detail"]


def test_a_nested_text_config_supplies_the_layer_count():
    assert sv.layers_in_config({"text_config": {"num_hidden_layers": 36}}) == 36
    r = by_name(sv.check_engine_log(V1_LOG, config={"text_config": {"num_hidden_layers": 36}}))
    assert r["layers_loaded"]["state"] == "pass"


@pytest.mark.parametrize("config", [None, {}, {"num_hidden_layers": "36"}, {"num_hidden_layers": True}, {"num_hidden_layers": 0}, [], "x"])
def test_no_usable_num_hidden_layers_is_not_checked(config):
    assert by_name(sv.check_engine_log(V1_LOG, config=config))["layers_loaded"]["state"] == "not_checked"


def test_a_log_with_no_layer_line_is_not_checked_not_a_pass():
    r = by_name(sv.check_engine_log("INFO nothing here\n", config={"num_hidden_layers": 36}))
    assert r["layers_loaded"]["state"] == "not_checked" and r["kv_cache"]["state"] == "not_checked"


def test_a_cache_smaller_than_one_full_length_sequence_is_a_finding():
    log = "GPU KV cache size: 2,048 tokens\nMaximum concurrency for 8,192 tokens per request: 0.25x\n"
    r = by_name(sv.check_engine_log(log, max_model_len=8192))
    assert r["kv_cache"]["state"] == "fail" and "2048" in r["kv_cache"]["detail"]


def test_a_stated_concurrency_below_one_fails_even_without_a_declared_length():
    r = by_name(sv.check_engine_log("Maximum concurrency for 8,192 tokens per request: 0.50x\n"))
    assert r["kv_cache"]["state"] == "fail"


def test_the_older_block_line_needs_the_declared_block_size():
    assert by_name(sv.check_engine_log(V0_LOG, max_model_len=8192))["kv_cache"]["state"] == "not_checked"
    r = by_name(sv.check_engine_log(V0_LOG, max_model_len=8192, block_size=64))
    assert r["kv_cache"]["state"] == "pass" and r["kv_cache"]["capacity_tokens"] == 512 * 64
    assert by_name(sv.check_engine_log(V0_LOG, max_model_len=65536, block_size=64))["kv_cache"]["state"] == "fail"


def test_the_last_line_wins_and_an_unreadable_log_is_not_checked():
    log = "loaded 12 layers\n" + V1_LOG
    assert by_name(sv.check_engine_log(log, config={"num_hidden_layers": 36}))["layers_loaded"]["state"] == "pass"
    r = by_name(sv.check_engine_log(None, config={"num_hidden_layers": 36}))
    assert {x["state"] for x in r.values()} == {"not_checked"}


def test_a_layer_count_inside_other_text_is_not_matched():
    r = by_name(sv.check_engine_log("loaded 3600 layers_total ids\nthe model has loaded 36 layersX\n", config={"num_hidden_layers": 36}))
    assert r["layers_loaded"]["state"] == "not_checked"


# ---- 373: the quiet-warning filter ---------------------------------------------------------------------------------------------------

NOISE_LOG = """\
INFO 11-02 10:00:01 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs
WARNING 11-02 10:00:02 [__init__.py:30] Failed to import from vllm._C with ImportError('libcuda.so.1: cannot open shared object file')
INFO 11-02 10:00:03 [triton_utils.py:5] Triton is installed but 0 active driver(s) found. Disabling Triton to prevent runtime errors.
WARNING 11-02 10:00:04 [rocm.py:1] ROCm is not available: falling back
INFO 11-02 10:00:05 [server.py:9] Application startup complete.
"""
REAL = "WARNING 11-02 10:00:10 [platform.py:77] Unrecognised device name 'P150b'; falling back to the generic profile"


def test_the_filter_leaves_a_real_warning_among_real_noise():
    out = sv.log_warnings(NOISE_LOG + REAL + "\n")
    assert [w["text"] for w in out["lines"]] == [REAL], out
    assert out["dropped_as_noise"] >= 3 and out["truncated"] is False


def test_a_log_of_noise_only_leaves_nothing():
    out = sv.log_warnings(NOISE_LOG)
    assert out["lines"] == [] and out["hits"] == 0


def test_lines_that_differ_only_in_numbers_are_one_line_with_a_count():
    log = "\n".join(f"WARNING 11-02 10:00:{i:02d} [x.py:1] request {i} ignored option max_foo=3" for i in range(10))
    out = sv.log_warnings(log)
    assert len(out["lines"]) == 1 and out["lines"][0]["count"] == 10 and out["hits"] == 10


def test_the_result_is_bounded_in_lines_and_characters_and_says_so():
    log = "\n".join(f"WARNING deprecated option kind{chr(97 + i % 26)}{chr(97 + i // 26)}x{'y' * 600}" for i in range(60))
    out = sv.log_warnings(log)
    assert len(out["lines"]) == sv.MAX_WARNING_LINES and out["truncated"] is True
    assert all(len(w["text"]) <= sv.MAX_WARNING_CHARS for w in out["lines"])


def test_color_codes_are_stripped_and_an_unreadable_log_gives_no_lines():
    out = sv.log_warnings("\x1b[33mWARNING\x1b[0m mismatch between a and b\n")
    assert out["lines"][0]["text"] == "WARNING mismatch between a and b"
    assert sv.log_warnings(None)["lines"] is None and sv.log_warnings(b"x")["lines"] is None


@pytest.mark.parametrize("line", ["fell back to host sampling", "option --foo is ignored", "deprecated flag --bar", "model name not recognised",
                                  "dtype mismatch: bf16 vs fp32", "unsupported device grid", "overriding block_size to 64"])
def test_the_keyword_list_is_broad(line):
    assert sv.log_warnings("INFO " + line)["lines"], line


# ---- 361 and 372: reasoning tokens and the reply classes ----------------------------------------------------------------------------

def reply(content=None, reasoning=None, tokens=50, finish="stop", details=None, field="reasoning_content"):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message[field] = reasoning
    usage = {"completion_tokens": tokens, **({"completion_tokens_details": details} if details is not None else {})}
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


def test_a_reported_reasoning_token_count_is_used_as_reported():
    s = sv.reasoning_split(reply("408", "think think", tokens=100, details={"reasoning_tokens": 90}))
    assert (s["reasoning_tokens"], s["answer_tokens"], s["basis"]) == (90, 10, "reported")


def test_without_a_reported_count_the_split_is_an_estimate_by_characters_and_says_so():
    s = sv.reasoning_split(reply("ab", "x" * 8, tokens=100))
    assert s["reasoning_tokens"] == 80 and s["basis"] == "estimated by characters"


def test_a_reply_with_no_reasoning_text_has_no_reasoning_tokens():
    s = sv.reasoning_split(reply("hello", None, tokens=7))
    assert (s["reasoning_tokens"], s["answer_tokens"], s["basis"]) == (0, 7, "no reasoning text")


@pytest.mark.parametrize("body", [{}, {"usage": {}}, {"usage": {"completion_tokens": True}}, {"usage": {"completion_tokens": -1}}, b"junk", None])
def test_a_reply_without_a_token_count_has_no_split(body):
    assert sv.reasoning_split(body) is None


def test_an_over_large_reported_reasoning_count_is_not_trusted():
    s = sv.reasoning_split(reply("a", "b", tokens=10, details={"reasoning_tokens": 99}))
    assert s["basis"] == "estimated by characters"


def test_the_new_reasoning_field_name_is_read_too():
    assert sv.classify_reasoning(reply("408", "thinking", field="reasoning"))["kind"] == "populated"


@pytest.mark.parametrize("body,kind", [
    (reply("408", "17*24 is 408 because", tokens=40), "populated"),
    (reply("", "still thinking", tokens=48, finish="length"), "populated"),
    (reply("<think>hmm</think>408", None, tokens=30), "leaking"),
    (reply("408", "x</think>", tokens=30), "leaking"),
    (reply("408", None, tokens=400), "dropped"),
    (reply("408", None, tokens=4), "none"),
    (reply("408", "   ", tokens=4), "none"),
    ({"choices": []}, "unusable"),
    ({"choices": [{"message": None}]}, "unusable"),
    (b"junk", "unusable"),
])
def test_each_reasoning_reply_class(body, kind):
    assert sv.classify_reasoning(body)["kind"] == kind


def test_the_dropped_class_needs_the_count_far_above_the_visible_answer():
    assert sv.classify_reasoning(reply("a" * 30, None, tokens=40))["kind"] == "none"
    assert sv.classify_reasoning(reply("a" * 30, None, tokens=200))["kind"] == "dropped"


def test_the_parser_name_comes_from_the_engine_arguments_line_and_the_last_one_wins():
    log = "args: Namespace(model='m', reasoning_parser='qwen3', port=1)\n"
    assert sv.reasoning_parser_in_log(log) == "qwen3"
    assert sv.reasoning_parser_in_log("--reasoning-parser deepseek_r1 ... reasoning_parser='qwen3'") == "qwen3"
    assert sv.reasoning_parser_in_log("reasoning_parser=None") is None
    assert sv.reasoning_parser_in_log("nothing") is None and sv.reasoning_parser_in_log(None) is None


def test_distribution_skips_what_is_not_a_whole_number():
    assert sv.distribution([5, 1, 3, True, "x", None, -2, 2.5]) == {"n": 3, "min": 1, "median": 3, "max": 5}
    assert sv.distribution([2, 4]) == {"n": 2, "min": 2, "median": 3.0, "max": 4}
    assert sv.distribution([]) is None and sv.distribution(None) is None


# ---- 360: a repeated prefix ---------------------------------------------------------------------------------------------------------

def req(seconds, prompt=1500, cached=None):
    return {"seconds": seconds, "prompt_tokens": prompt, "cached_tokens": cached}


def test_a_reported_cached_count_decides_whatever_the_clock_says():
    r = sv.check_prefix_reuse(req(2.0), req(2.0, cached=1400), req(2.0), declared=True)
    assert r["state"] == "pass" and "cached-token count" in r["basis"]
    r = sv.check_prefix_reuse(req(2.0), req(0.1, cached=0), req(2.0), declared=True)
    assert r["state"] == "fail"


def test_without_a_count_the_repeat_must_beat_the_control_and_the_basis_says_timing():
    r = sv.check_prefix_reuse(req(3.0), req(0.5), req(2.0), declared=True)
    assert r["state"] == "pass" and "timing" in r["basis"]
    assert sv.check_prefix_reuse(req(3.0), req(1.9), req(2.0), declared=True)["state"] == "fail"


def test_the_first_cold_request_is_not_the_baseline_because_kernels_warm_during_it():
    # the repeat is faster than the first request only because the first compiled kernels; the control shows the warm cost
    assert sv.check_prefix_reuse(req(30.0), req(2.0), req(2.0), declared=True)["state"] == "fail"


def test_a_can_that_does_not_declare_prefix_caching_is_not_failed_for_lacking_it():
    r = sv.check_prefix_reuse(req(3.0), req(2.0), req(2.0), declared=False)
    assert r["state"] == "not_checked" and "does not declare" in r["detail"]


@pytest.mark.parametrize("bad", [None, {}, {"seconds": "x"}, {"seconds": True}, {"seconds": -1}])
def test_missing_or_malformed_numbers_are_not_checked(bad):
    assert sv.check_prefix_reuse(req(1.0), bad, bad, declared=True)["state"] == "not_checked"
