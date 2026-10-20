# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The tool-call scorer for the model-card benchmark (orchard/toolcall_bench.py).

It scores the Berkeley Function Calling Leaderboard `simple` and `multiple` sets (one question, one expected call). The fixtures in
tests/fixtures/bfcl are the first lines of the real files (Apache-2.0), so the loader is checked against the shape the dataset really has:
names with dots, the type words `dict`, `float`, `tuple` and `any`, and a ground truth that lists the accepted values for each parameter,
with `""` meaning the parameter may be left out.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from orchard import toolcall_bench as tb

FIX = Path(__file__).parent / "fixtures" / "bfcl"


def items(kind="simple", **kw):
    return tb.load_items(FIX / f"{kind}_questions.jsonl", FIX / f"{kind}_answers.jsonl", **kw)


def call(name, **args):
    return {"name": name, "arguments": args}


# ---- loading ----------------------------------------------------------------------------------------------------------------------

def test_the_real_fixtures_load_with_messages_functions_and_one_expected_call():
    got = items()
    assert [i["id"] for i in got] and all(i["id"].startswith("simple_") for i in got)
    first = next(i for i in got if i["id"] == "simple_0")
    assert first["messages"] == [{"role": "user", "content": "Find the area of a triangle with a base of 10 units and height of 5 units."}]
    assert first["functions"][0]["name"] == "calculate_triangle_area"
    assert first["truth"] == {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}


def test_a_question_and_its_answer_must_have_the_same_id(tmp_path):
    q = tmp_path / "q.jsonl"
    a = tmp_path / "a.jsonl"
    q.write_text((FIX / "simple_questions.jsonl").read_text())
    lines = (FIX / "simple_answers.jsonl").read_text().splitlines()
    a.write_text("\n".join(reversed(lines)) + "\n")
    with pytest.raises(ValueError, match="simple_0|id"):
        tb.load_items(q, a)


def test_an_item_with_more_than_one_expected_call_is_refused(tmp_path):
    q = tmp_path / "q.jsonl"
    a = tmp_path / "a.jsonl"
    q.write_text((FIX / "simple_questions.jsonl").read_text().splitlines()[0] + "\n")
    row = json.loads((FIX / "simple_answers.jsonl").read_text().splitlines()[0])
    row["ground_truth"] = row["ground_truth"] * 2
    a.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="one expected call|parallel"):
        tb.load_items(q, a)


def test_a_multi_turn_question_is_refused(tmp_path):
    q = tmp_path / "q.jsonl"
    a = tmp_path / "a.jsonl"
    row = json.loads((FIX / "simple_questions.jsonl").read_text().splitlines()[0])
    row["question"] = [row["question"][0], row["question"][0]]
    q.write_text(json.dumps(row) + "\n")
    a.write_text((FIX / "simple_answers.jsonl").read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="turn"):
        tb.load_items(q, a)


def test_limit_takes_a_fixed_seeded_subset_that_does_not_depend_on_file_order(tmp_path):
    full = items("multiple") + items("simple")
    a = [i["id"] for i in tb.choose(full, limit=3, seed=7)]
    b = [i["id"] for i in tb.choose(list(reversed(full)), limit=3, seed=7)]
    c = [i["id"] for i in tb.choose(full, limit=3, seed=8)]
    by_key = sorted((i["id"] for i in full), key=lambda x: tb.order_key(7, x))
    assert a == b == by_key[:3] and len(set(a)) == 3
    assert c == sorted((i["id"] for i in full), key=lambda x: tb.order_key(8, x))[:3]
    assert [i["id"] for i in tb.choose(full, limit=None, seed=7)] == by_key
    assert tb.order_key(7, "simple_0") != tb.order_key(8, "simple_0") and tb.order_key(7, "simple_0") == tb.order_key(7, "simple_0")


# ---- the request ------------------------------------------------------------------------------------------------------------------

def test_tools_use_openai_type_words_and_names_without_dots_and_map_back():
    item = next(i for i in items() if i["id"] == "simple_1")           # math.factorial
    tools, names = tb.openai_tools(item["functions"])
    fn = tools[0]["function"]
    assert tools[0]["type"] == "function" and "." not in fn["name"] and names[fn["name"]] == "math.factorial"
    assert fn["parameters"]["type"] == "object"
    assert fn["parameters"]["properties"]["number"]["type"] == "integer"


def test_type_words_are_mapped_everywhere_in_the_schema():
    fns = [{"name": "f", "description": "d", "parameters": {"type": "dict", "required": ["a"], "properties": {
        "a": {"type": "float"}, "b": {"type": "tuple", "items": {"type": "float"}}, "c": {"type": "any"},
        "d": {"type": "array", "items": {"type": "dict", "properties": {"e": {"type": "float"}}}}}}}]
    tools, _ = tb.openai_tools(fns)
    p = tools[0]["function"]["parameters"]
    assert p["type"] == "object" and p["required"] == ["a"]
    assert p["properties"]["a"]["type"] == "number"
    assert p["properties"]["b"]["type"] == "array" and p["properties"]["b"]["items"]["type"] == "number"
    assert p["properties"]["c"]["type"] == "string"
    assert p["properties"]["d"]["items"]["type"] == "object" and p["properties"]["d"]["items"]["properties"]["e"]["type"] == "number"


def test_a_property_called_type_is_not_mistaken_for_a_type_word():
    fns = [{"name": "f", "description": "d", "parameters": {"type": "dict", "properties": {"type": {"type": "string", "description": "kind"}}}}]
    tools, _ = tb.openai_tools(fns)
    assert tools[0]["function"]["parameters"]["properties"]["type"]["type"] == "string"


def test_the_input_functions_are_not_changed():
    fns = [{"name": "a.b", "description": "d", "parameters": {"type": "dict", "properties": {"x": {"type": "float"}}}}]
    before = json.dumps(fns)
    tb.openai_tools(fns)
    assert json.dumps(fns) == before


def test_request_body_is_greedy_non_streaming_with_tools_and_a_token_limit():
    item = items()[0]
    body = tb.request_body("m", item, max_tokens=512)
    assert body["model"] == "m" and body["temperature"] == 0 and body["max_tokens"] == 512 and body["stream"] is False
    assert body["tool_choice"] == "auto" and body["tools"][0]["type"] == "function" and body["messages"] == item["messages"]


def test_request_body_can_turn_thinking_off():
    body = tb.request_body("m", items()[0], max_tokens=64, thinking=False)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in tb.request_body("m", items()[0], max_tokens=64)


# ---- reading the reply ------------------------------------------------------------------------------------------------------------

def test_calls_are_read_from_the_message_and_names_map_back():
    names = {"math_factorial": "math.factorial"}
    msg = {"content": None, "tool_calls": [{"type": "function", "function": {"name": "math_factorial", "arguments": "{\"number\": 5}"}}]}
    assert tb.parse_calls(msg, names) == [{"name": "math.factorial", "arguments": {"number": 5}}]


def test_unreadable_arguments_are_kept_as_an_error_not_dropped():
    msg = {"tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]}
    got = tb.parse_calls(msg, {"f": "f"})
    assert got == [{"name": "f", "arguments": None}]


def test_arguments_that_are_not_an_object_are_unreadable():
    msg = {"tool_calls": [{"function": {"name": "f", "arguments": "[1, 2]"}}]}
    assert tb.parse_calls(msg, {"f": "f"}) == [{"name": "f", "arguments": None}]


def test_a_reply_with_no_tool_calls_gives_no_calls():
    assert tb.parse_calls({"content": "I would call it"}, {}) == []
    assert tb.parse_calls({"content": "x", "tool_calls": None}, {}) == []


# ---- grading ----------------------------------------------------------------------------------------------------------------------

def graded(item_id, calls, kind="simple"):
    item = next(i for i in items(kind) if i["id"] == item_id)
    return tb.grade(item, calls)


def test_the_expected_call_passes():
    assert graded("simple_0", [call("calculate_triangle_area", base=10, height=5, unit="units")]) == {"passed": True, "reason": "ok"}


def test_an_optional_parameter_may_be_left_out_when_the_truth_allows_an_empty_value():
    assert graded("simple_0", [call("calculate_triangle_area", base=10, height=5)])["passed"] is True


def test_a_wrong_value_fails_with_bad_value():
    got = graded("simple_0", [call("calculate_triangle_area", base=10, height=6)])
    assert got["passed"] is False and got["reason"].startswith("bad_value") and "height" in got["reason"]


def test_a_missing_required_parameter_fails():
    got = graded("simple_0", [call("calculate_triangle_area", base=10)])
    assert got["passed"] is False and got["reason"].startswith("missing_required") and "height" in got["reason"]


def test_a_parameter_the_function_does_not_declare_fails():
    got = graded("simple_0", [call("calculate_triangle_area", base=10, height=5, colour="red")])
    assert got["passed"] is False and got["reason"].startswith("extra_param") and "colour" in got["reason"]


def test_a_wrong_function_name_fails():
    got = graded("simple_0", [call("calculate_circle_area", radius=3)])
    assert got["passed"] is False and got["reason"].startswith("wrong_name")


def test_no_call_fails_with_no_call():
    assert graded("simple_0", []) == {"passed": False, "reason": "no_call"}


def test_two_calls_fail_even_when_one_is_right():
    right = call("calculate_triangle_area", base=10, height=5)
    got = graded("simple_0", [right, right])
    assert got["passed"] is False and got["reason"].startswith("too_many_calls")


def test_unreadable_arguments_fail_with_bad_arguments():
    got = graded("simple_0", [{"name": "calculate_triangle_area", "arguments": None}])
    assert got["passed"] is False and got["reason"] == "bad_arguments"


def test_an_integer_is_accepted_where_the_truth_has_a_float_and_the_reverse():
    # simple_2: math.hypot x=4, y=5, z in ["", 0]
    assert graded("simple_2", [call("math.hypot", x=4.0, y=5, z=0)])["passed"] is True
    assert graded("simple_2", [call("math.hypot", x=4, y=5.0)])["passed"] is True


def test_a_string_is_not_a_number_and_a_bool_is_not_a_number():
    assert graded("simple_2", [call("math.hypot", x="4", y=5)])["passed"] is False
    assert graded("simple_2", [call("math.hypot", x=True, y=5)])["passed"] is False      # True == 1 in Python; here the truth is 4, so also guards bool as int


def test_a_bool_value_is_not_accepted_for_the_number_one():
    item = {"id": "t", "messages": [], "functions": [{"name": "f", "description": "", "parameters": {"type": "dict", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}],
            "truth": {"f": {"n": [1]}}}
    assert tb.grade(item, [call("f", n=True)])["passed"] is False
    assert tb.grade(item, [call("f", n=1)])["passed"] is True


def test_strings_compare_without_case_spaces_or_the_punctuation_the_leaderboard_ignores():
    item = {"id": "t", "messages": [], "functions": [{"name": "f", "description": "", "parameters": {"type": "dict", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
            "truth": {"f": {"city": ["San Francisco, CA"]}}}
    assert tb.grade(item, [call("f", city="san francisco ca")])["passed"] is True
    assert tb.grade(item, [call("f", city="San Francisco, CA")])["passed"] is True
    assert tb.grade(item, [call("f", city="Los Angeles, CA")])["passed"] is False


def test_lists_compare_element_by_element_and_length_matters():
    item = {"id": "t", "messages": [], "functions": [{"name": "f", "description": "", "parameters": {"type": "dict", "properties": {"v": {"type": "array", "items": {"type": "integer"}}}, "required": ["v"]}}],
            "truth": {"f": {"v": [[1, 2, 3]]}}}
    assert tb.grade(item, [call("f", v=[1, 2, 3])])["passed"] is True
    assert tb.grade(item, [call("f", v=[1, 2])])["passed"] is False
    assert tb.grade(item, [call("f", v=[1, 2, 3, 4])])["passed"] is False
    assert tb.grade(item, [call("f", v=[1, 2, 4])])["passed"] is False
    assert tb.grade(item, [call("f", v=[3, 2, 1])])["passed"] is False


def test_nested_objects_compare_by_key_with_accepted_values_per_key():
    item = {"id": "t", "messages": [], "functions": [{"name": "f", "description": "", "parameters": {"type": "dict", "properties": {"o": {"type": "dict"}}, "required": ["o"]}}],
            "truth": {"f": {"o": [{"a": [1, 2], "b": ["x"]}]}}}
    assert tb.grade(item, [call("f", o={"a": 2, "b": "x"})])["passed"] is True
    assert tb.grade(item, [call("f", o={"a": 3, "b": "x"})])["passed"] is False
    assert tb.grade(item, [call("f", o={"a": 1})])["passed"] is False
    assert tb.grade(item, [call("f", o={"a": 1, "b": "x", "c": 1})])["passed"] is False


def test_the_multiple_set_needs_the_right_function_among_several():
    item = items("multiple")[0]
    assert len(item["functions"]) > 1
    (name, params), = item["truth"].items()
    args = {k: next(a for a in alts if a != "") for k, alts in params.items() if any(a != "" for a in alts)}
    assert tb.grade(item, [call(name, **args)])["passed"] is True
    other = next(f["name"] for f in item["functions"] if f["name"] != name)
    assert tb.grade(item, [call(other, **args)])["reason"].startswith("wrong_name")


def test_every_real_fixture_item_passes_with_its_own_ground_truth():
    for kind in ("simple", "multiple"):
        for item in items(kind):
            (name, params), = item["truth"].items()
            args = {k: next(a for a in alts if a != "") for k, alts in params.items() if any(a != "" for a in alts)}
            assert tb.grade(item, [call(name, **args)]) == {"passed": True, "reason": "ok"}, item["id"]


def test_a_dotted_name_in_the_truth_matches_the_name_the_model_returns_after_mapping():
    item = next(i for i in items() if i["id"] == "simple_1")
    tools, names = tb.openai_tools(item["functions"])
    msg = {"tool_calls": [{"function": {"name": tools[0]["function"]["name"], "arguments": "{\"number\": 5}"}}]}
    assert tb.grade(item, tb.parse_calls(msg, names))["passed"] is True


# ---- running against a server -----------------------------------------------------------------------------------------------------

class Fake:
    """A chat server that answers each request from a function of the request body, and records what it was sent."""
    def __init__(self, answer):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append(body)
                status, payload = answer(body)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def tool_reply(name, args, usage=(30, 12)):
    return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def oracle(body):
    """Answers the way the ground truth says, by reading the tool the request offers."""
    fn = body["tools"][0]["function"]
    return 200, tool_reply(fn["name"], {"base": 10, "height": 5} if "triangle" in fn["name"] else {"number": 5})


def test_run_sends_every_item_and_reports_a_score_with_counts():
    fake = Fake(oracle)
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", items()[:2], concurrency=2, max_tokens=256)
    finally:
        fake.close()
    assert out["n"] == 2 and len(fake.seen) == 2 and out["passed"] + out["failed"] == 2
    assert {r["id"] for r in out["items"]} == {"simple_0", "simple_1"}
    assert out["items"][0]["completion_tokens"] == 12 and out["items"][0]["prompt_tokens"] == 30
    assert out["score"] == out["passed"] / out["n"]
    assert out["stderr"] is not None


def test_run_gives_results_in_item_order_whatever_order_the_server_finishes():
    def slow_first(body):
        import time
        if "triangle" in body["tools"][0]["function"]["name"]:
            time.sleep(0.3)
        return oracle(body)
    fake = Fake(slow_first)
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", items()[:2], concurrency=2, max_tokens=64)
    finally:
        fake.close()
    assert [r["id"] for r in out["items"]] == ["simple_0", "simple_1"]


def test_a_server_error_is_recorded_for_that_item_and_counts_as_a_failure():
    fake = Fake(lambda body: (500, {"error": {"message": "boom"}}))
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", items()[:1], concurrency=1, max_tokens=64, max_consecutive_errors=5)
    finally:
        fake.close()
    assert out["items"][0]["passed"] is False and out["items"][0]["reason"].startswith("error")
    assert out["errors"] == 1 and out["stopped"] is None


def test_three_errors_in_a_row_stop_the_run_with_a_reason_and_the_rest_are_not_sent():
    fake = Fake(lambda body: (500, {"error": {"message": "boom"}}))
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", items("simple") + items("multiple"), concurrency=1, max_tokens=64,
                     max_consecutive_errors=3)
    finally:
        fake.close()
    assert len(fake.seen) == 3 and out["stopped"].startswith("3 errors in a row")
    assert len(out["items"]) == 3 and out["n"] == 3


def test_a_success_resets_the_count_of_errors_in_a_row():
    state = {"n": 0}

    def flaky(body):
        state["n"] += 1
        return (500, {"error": {"message": "x"}}) if state["n"] in (1, 2, 4, 5) else oracle(body)
    fake = Fake(flaky)
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", (items("simple") + items("multiple"))[:5], concurrency=1, max_tokens=64,
                     max_consecutive_errors=3)
    finally:
        fake.close()
    assert out["stopped"] is None and len(fake.seen) == 5


def test_an_empty_reply_from_a_cut_off_generation_is_a_no_call_that_says_so():
    def cut(body):
        return 200, {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": None}}], "usage": {"prompt_tokens": 9, "completion_tokens": 64}}
    fake = Fake(cut)
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", items()[:1], concurrency=1, max_tokens=64)
    finally:
        fake.close()
    assert out["items"][0]["reason"] == "no_call" and out["items"][0]["finish_reason"] == "length" and out["cut_off"] == 1


def test_stderr_is_the_binomial_standard_error():
    from math import sqrt
    assert tb.stderr(passed=50, n=100) == pytest.approx(sqrt(0.25 / 100))
    assert tb.stderr(passed=0, n=10) == 0.0
    assert tb.stderr(passed=0, n=0) is None


def test_the_result_records_what_was_asked_so_two_runs_can_be_compared():
    fake = Fake(oracle)
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", items()[:2], concurrency=1, max_tokens=300, thinking=False, seed=3,
                     dataset="gorilla-llm/Berkeley-Function-Calling-Leaderboard@61fc060")
    finally:
        fake.close()
    assert out["settings"] == {"max_tokens": 300, "thinking": False, "temperature": 0, "concurrency": 1, "seed": 3,
                               "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard@61fc060"}
    assert out["item_ids_sha256"] and fake.seen[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_two_functions_that_collide_once_dots_are_replaced_are_refused():
    fns = [{"name": "a.b", "description": "", "parameters": {"type": "dict", "properties": {}}},
           {"name": "a_b", "description": "", "parameters": {"type": "dict", "properties": {}}}]
    with pytest.raises(ValueError, match="a_b"):
        tb.openai_tools(fns)


def one_function_item(*, properties, required, truth):
    return {"id": "t", "messages": [], "truth": {"f": truth},
            "functions": [{"name": "f", "description": "", "parameters": {"type": "dict", "properties": properties, "required": required}}]}


def test_a_parameter_the_function_declares_but_the_truth_does_not_list_is_extra():
    item = one_function_item(properties={"a": {"type": "integer"}, "b": {"type": "integer"}}, required=["a"], truth={"a": [1]})
    assert tb.grade(item, [call("f", a=1)])["passed"] is True
    got = tb.grade(item, [call("f", a=1, b=2)])
    assert got["passed"] is False and got["reason"] == "extra_param: b"


def test_a_parameter_the_truth_lists_but_the_function_does_not_declare_is_extra():
    item = one_function_item(properties={"a": {"type": "integer"}}, required=["a"], truth={"a": [1], "z": [2]})
    got = tb.grade(item, [call("f", a=1, z=2)])
    assert got["passed"] is False and got["reason"] == "extra_param: z"


def test_a_required_parameter_is_needed_even_when_the_truth_also_allows_an_empty_value():
    item = one_function_item(properties={"a": {"type": "integer"}}, required=["a"], truth={"a": [1, ""]})
    got = tb.grade(item, [call("f")])
    assert got["passed"] is False and got["reason"] == "missing_required: a"


def test_a_parameter_that_is_not_required_is_still_needed_when_the_truth_has_no_empty_value():
    item = one_function_item(properties={"a": {"type": "integer"}, "b": {"type": "integer"}}, required=["a"], truth={"a": [1], "b": [2]})
    got = tb.grade(item, [call("f", a=1)])
    assert got["passed"] is False and got["reason"] == "missing_required: b"
    assert tb.grade(item, [call("f", a=1, b=2)])["passed"] is True


def test_the_leaderboard_reads_a_single_quote_as_a_double_quote_in_strings():
    item = one_function_item(properties={"s": {"type": "string"}}, required=["s"], truth={"s": ["say 'hi'"]})
    assert tb.grade(item, [call("f", s='say "hi"')])["passed"] is True
    assert tb.grade(item, [call("f", s="say hi")])["passed"] is False


def test_fewer_answers_than_questions_is_refused_not_silently_cut_short(tmp_path):
    q = tmp_path / "q.jsonl"
    a = tmp_path / "a.jsonl"
    q.write_text((FIX / "simple_questions.jsonl").read_text())
    a.write_text("\n".join((FIX / "simple_answers.jsonl").read_text().splitlines()[:2]) + "\n")
    with pytest.raises(ValueError, match="3 questions but 2 answers"):
        tb.load_items(q, a)


def test_the_score_is_the_share_that_passed():
    def half(body):
        fn = body["tools"][0]["function"]
        return 200, tool_reply(fn["name"], {"base": 10, "height": 5} if "triangle" in fn["name"] else {"number": 6})
    fake = Fake(half)
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", items()[:2], concurrency=1, max_tokens=64)
    finally:
        fake.close()
    assert out["passed"] == 1 and out["failed"] == 1 and out["score"] == 0.5
    assert [r["reason"] for r in out["items"]] == ["ok", "bad_value: number"]


def test_the_reason_for_stopping_is_the_error_that_crossed_the_limit_not_a_later_one():
    import time
    state = {"n": 0}
    lock = threading.Lock()

    def staggered(body):
        with lock:
            state["n"] += 1
            mine = state["n"]
        time.sleep(0.15 * mine)                                    # completions arrive in the order the requests did
        return 500, {"error": {"message": f"err {mine}"}}
    fake = Fake(staggered)
    try:
        out = tb.run(f"http://127.0.0.1:{fake.port}/v1/chat/completions", "m", (items("simple") + items("multiple"))[:4], concurrency=4, max_tokens=64,
                     max_consecutive_errors=3)
    finally:
        fake.close()
    assert "err 3" in out["stopped"] and "err 4" not in out["stopped"]
