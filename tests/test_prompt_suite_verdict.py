# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The verdict of the prompt suite: what a swap-check.json record says, read against its control (issue 15).

The template records raw replies. `evaluate` reads them. A finding fails a gate only when its kind is blocking and the
control does not show the same kind on the same prompt. With no control for a prompt, no control explains the finding, so it
counts. The stage 1 free run has a control that is always there: the CPU reference's own text."""
import copy

import pytest

from orchard import prompt_suite as ps

SUITE = ps.load_suite()
HEALTHY = {
    "chat-capital": "The capital of France is Paris. One famous building there is the Eiffel Tower.",
    "code-fizzbuzz": "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        out.append('FizzBuzz' if i % 15 == 0 else str(i))\n    return out\n",
    "arithmetic": "Seven pens cost 21 dollars and four notebooks cost 20 dollars, so the total is 41 dollars.",
    "multi-turn": "Your name is Zephyrine and you like to drink green tea.",
    "non-english": "Столицей Франции является Париж, город на берегах Сены.",
    "summarize": "Every Tuesday the harbor town of Lindfors holds a fish market that is over by nine.",
    "list-format": "Red\nBlue\nYellow",
}


def make_entry(pid, content=None, **extra):
    return {"id": pid, "mode": "chat", "content": HEALTHY.get(pid, "") if content is None else content, "reasoning": "",
            "tool_calls": [{"name": "get_weather", "arguments": '{"city": "Lisbon"}'}] if pid == "tool-call" else [],
            "finish_reason": "stop", "error": None, **extra}


def make_record(**change):
    rec = {"ran": True, "model_id": "Altworld/Hemmingway-1@1a5f363a", "suite_sha256": ps.suite_sha256(), "mode": "chat", "max_tokens": 64,
           "chat_template": {"present": True, "source": "chat_template.jinja", "sha256": "a" * 64, "mentions_tools": True},
           "special_tokens": ["<|im_end|>", "<|im_start|>"],
           "entries": [make_entry(p["id"]) for p in SUITE["prompts"]], "skipped": [], "control": None}
    rec.update(change)
    return rec


def make_report(record=None, free="Paris is the capital of France and a large city", reference="Paris is the capital of France and a large city"):
    rep = {"free_run_text": free, "reference_generated_text": reference}
    if record is not None:
        rep["prompt_suite"] = record
    return rep


def with_entry(record, pid, **change):
    rec = copy.deepcopy(record)
    for e in rec["entries"]:
        if e["id"] == pid:
            e.update(change)
    return rec


def control_of(record, source="evidence/control-swap-check.json", differ=True):
    """A control record. The control ran the nearest model, so it has another model id and, by default, a different reply to one
    prompt (a fine-tune and its base are never identical on every prompt; `differ=False` makes them so)."""
    ctl = copy.deepcopy(record)
    if "model_id" in ctl:
        ctl["model_id"] = "Qwen/Qwen3.8-27B"
    if differ:
        for e in ctl["entries"]:
            if e["id"] == "list-format":
                e["content"] = "Red\nGreen\nBlue"
    return {"source": source, "sha256": "b" * 64, "record": ctl}


LOOP = "Sure. " + "the quick brown fox jumps over " * 5


def test_a_report_without_the_suite_has_no_verdict_and_no_reason():
    v = ps.evaluate(make_report())
    assert v.ran is False and v.reasons == [] and v.advisory == []


def test_a_healthy_record_passes_with_no_reason_and_no_advisory():
    v = ps.evaluate(make_report(make_record()))
    assert v.ran is True and v.reasons == [] and v.advisory == []
    assert v.summary["prompts"] == len(SUITE["prompts"]) and v.summary["control"] == "none"


def test_a_blocking_finding_with_no_control_fails_and_names_the_prompt_the_kind_and_the_missing_control():
    v = ps.evaluate(make_report(with_entry(make_record(), "arithmetic", content=LOOP)))
    assert len(v.reasons) == 1
    r = v.reasons[0]
    assert "arithmetic" in r and "mechanical_repetition" in r and "no control" in r


def test_the_same_kind_in_the_control_is_shared_and_does_not_fail():
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    rep = make_report(rec)
    rep["prompt_suite"]["control"] = control_of(rec)
    v = ps.evaluate(rep)
    assert v.reasons == []
    shared = [f for f in v.summary["findings"] if f["shared"]]
    assert [(f["prompt"], f["kind"]) for f in shared] == [("arithmetic", "mechanical_repetition")]
    assert v.summary["control"] == "file"


def test_a_control_that_shows_another_kind_does_not_explain_the_finding():
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    ctl = with_entry(make_record(), "arithmetic", content="ok <|im_end|>")
    rec["control"] = control_of(ctl)
    v = ps.evaluate(make_report(rec))
    assert len(v.reasons) == 1 and "mechanical_repetition" in v.reasons[0] and "control does not show it" in v.reasons[0]


def test_a_control_for_other_prompts_does_not_explain_a_finding_on_this_one():
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    ctl = with_entry(make_record(), "chat-capital", content=LOOP)
    rec["control"] = control_of(ctl)
    assert len(ps.evaluate(make_report(rec)).reasons) == 1


def test_an_advisory_finding_is_listed_and_does_not_fail():
    v = ps.evaluate(make_report(with_entry(make_record(), "chat-capital", content="")))   # an empty answer
    assert v.reasons == []
    assert len(v.advisory) == 1 and "chat-capital" in v.advisory[0] and "empty_output" in v.advisory[0]


def test_a_failed_request_is_advisory():
    v = ps.evaluate(make_report(with_entry(make_record(), "tool-call", content="", tool_calls=[], error="HTTP 400: bad tools")))
    assert v.reasons == [] and any("request_failed" in a for a in v.advisory)


def test_a_leak_between_requests_is_found_with_the_markers_of_the_suite():
    v = ps.evaluate(make_report(with_entry(make_record(), "multi-turn", content="Your name is Zephyrine. Lindfors is a harbor town.")))
    assert any("cross_request_leak" in r and "multi-turn" in r for r in v.reasons)


def test_a_special_token_the_template_recorded_is_searched_for():
    rec = with_entry(make_record(special_tokens=["[SEP]"]), "chat-capital", content="Paris is the capital. [SEP] more")
    assert any("control_token_leak" in r for r in ps.evaluate(make_report(rec)).reasons)


# ---- the record has to describe the committed suite -----------------------------------------------------------------

def test_a_record_of_other_prompts_fails_even_when_every_reply_is_healthy():
    v = ps.evaluate(make_report(make_record(suite_sha256="0" * 64)))
    assert any("not the committed prompt suite" in r for r in v.reasons)


def test_a_prompt_format_that_contradicts_the_chat_template_fails():
    rec = make_record(mode="completion")      # the template is present
    assert any("prompt format" in r for r in ps.evaluate(make_report(rec)).reasons)
    rec = make_record(chat_template={"present": False, "source": None, "sha256": None, "mentions_tools": False})
    assert any("prompt format" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_an_unknown_mode_fails():
    assert any("mode" in r for r in ps.evaluate(make_report(make_record(mode="chatty"))).reasons)


def test_a_prompt_with_no_reply_fails():
    rec = make_record()
    rec["entries"] = [e for e in rec["entries"] if e["id"] != "arithmetic"]
    assert any("arithmetic" in r and "no reply" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_a_reply_for_a_prompt_that_is_not_in_the_suite_fails():
    rec = make_record()
    rec["entries"].append(make_entry("made-up", content="hello there"))
    assert any("made-up" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_the_tool_prompt_may_be_skipped_only_when_the_template_has_no_tools():
    no_tools = {"present": True, "source": "chat_template.jinja", "sha256": "a" * 64, "mentions_tools": False}
    rec = make_record(chat_template=no_tools)
    rec["entries"] = [e for e in rec["entries"] if e["id"] != "tool-call"]
    rec["skipped"] = [{"id": "tool-call", "reason": "the chat template does not mention tools"}]
    assert ps.evaluate(make_report(rec)).reasons == []
    rec2 = copy.deepcopy(rec)
    rec2["chat_template"]["mentions_tools"] = True
    assert any("tool-call" in r and "skipped" in r for r in ps.evaluate(make_report(rec2)).reasons)


def test_any_other_prompt_may_not_be_skipped():
    rec = make_record()
    rec["entries"] = [e for e in rec["entries"] if e["id"] != "arithmetic"]
    rec["skipped"] = [{"id": "arithmetic", "reason": "slow"}]
    assert any("arithmetic" in r and "skipped" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_a_record_that_is_not_an_object_or_has_no_entries_fails_without_crashing():
    for bad in ("text", 7, [], {"ran": True}):
        v = ps.evaluate(make_report(bad))
        assert v.reasons, bad


def test_a_record_that_did_not_run_fails():
    assert ps.evaluate(make_report(make_record(ran=False))).reasons


def test_an_entry_that_is_not_an_object_is_a_reason_not_a_crash():
    rec = make_record()
    rec["entries"][0] = "oops"
    assert ps.evaluate(make_report(rec)).reasons


# ---- the control has to be comparable --------------------------------------------------------------------------------

def test_a_control_run_on_other_prompts_is_refused_and_does_not_explain_anything():
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    ctl = with_entry(make_record(suite_sha256="0" * 64), "arithmetic", content=LOOP)
    rec["control"] = control_of(ctl)
    v = ps.evaluate(make_report(rec))
    assert any("control" in r and "suite" in r for r in v.reasons)
    assert any("mechanical_repetition" in r for r in v.reasons)


def test_a_control_in_another_prompt_format_is_refused():
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    ctl = with_entry(make_record(mode="completion"), "arithmetic", content=LOOP)
    rec["control"] = control_of(ctl)
    v = ps.evaluate(make_report(rec))
    assert any("control" in r and "format" in r for r in v.reasons)


def test_a_control_with_a_different_chat_template_is_noted_and_still_used():
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    ctl = with_entry(make_record(), "arithmetic", content=LOOP)
    ctl["chat_template"]["sha256"] = "c" * 64
    rec["control"] = control_of(ctl)
    v = ps.evaluate(make_report(rec))
    assert v.reasons == []
    assert any("chat template" in a and "differs" in a for a in v.advisory)


def test_a_control_with_no_record_is_refused():
    rec = make_record()
    rec["control"] = {"source": "x", "sha256": "b" * 64}
    assert any("control" in r for r in ps.evaluate(make_report(rec)).reasons)


# ---- the stage 1 free run and the CPU reference ----------------------------------------------------------------------

def test_a_loop_in_the_free_run_that_the_cpu_reference_does_not_have_fails():
    v = ps.evaluate(make_report(make_record(), free=LOOP, reference="Paris is the capital of France and a large city on the Seine"))
    assert len(v.reasons) == 1 and "stage 1" in v.reasons[0] and "mechanical_repetition" in v.reasons[0]


def test_a_loop_that_the_cpu_reference_shows_too_is_the_models_own_behaviour():
    v = ps.evaluate(make_report(make_record(), free=LOOP, reference=LOOP))
    assert v.reasons == []


def test_a_free_run_loop_with_no_reference_text_has_no_control_and_fails():
    v = ps.evaluate(make_report(make_record(), free=LOOP, reference=None))
    assert len(v.reasons) == 1 and "no control" in v.reasons[0]


def test_the_free_run_is_not_read_when_the_suite_did_not_run():
    v = ps.evaluate(make_report(None, free=LOOP, reference=None))
    assert v.ran is False and v.reasons == []


# ---- the length of the answer ---------------------------------------------------------------------------------------

def test_many_findings_give_a_bounded_number_of_reasons_and_one_line_counts_the_rest():
    rec = make_record()
    for e in rec["entries"]:
        e["content"] = LOOP + " <|im_end|>"          # two blocking findings on every prompt
    v = ps.evaluate(make_report(rec))
    total = len([f for f in v.summary["findings"] if f["blocking"] and not f["shared"]])
    assert total > ps.MAX_REASONS
    assert len(v.reasons) == ps.MAX_REASONS
    assert v.reasons[-1].startswith(f"{total - (ps.MAX_REASONS - 1)} more")


def test_exactly_the_most_reasons_are_all_shown():
    rec = make_record()
    for e in rec["entries"][:ps.MAX_REASONS]:
        e["content"] = LOOP
    v = ps.evaluate(make_report(rec))
    assert len(v.reasons) == ps.MAX_REASONS and not any("more prompt suite" in r for r in v.reasons)


@pytest.mark.parametrize("kind", ["control_token_leak", "cross_request_leak", "corrupt_first_token", "repeated_first_token",
                                  "mechanical_repetition", "gibberish"])
def test_each_blocking_kind_fails_a_gate(kind):
    content = {"control_token_leak": "Paris <|im_end|>", "cross_request_leak": "Paris and Lindfors",
               "corrupt_first_token": "�Paris is it", "repeated_first_token": "no no no no Paris",
               "mechanical_repetition": LOOP,
               "gibberish": "krtp wxzq bdfgh qwrtp zzxkd mnbvc ghjkl rtlps xcvbn lkjhg The capital"}[kind]
    v = ps.evaluate(make_report(with_entry(make_record(), "chat-capital", content=content)))
    assert any(kind in r for r in v.reasons), (kind, v.reasons)


@pytest.mark.parametrize("kind,content", [("prompt_echo", "What is the capital of France, and what is one famous building there? Paris."),
                                          ("doubled_subwords", "ParisParis and buildingbuilding and EiffelEiffel and towertower"),
                                          ("wrong_language", "巴黎是法国的首都，埃菲尔铁塔是那里著名的建筑。")])
def test_each_advisory_kind_is_listed_and_does_not_fail(kind, content):
    v = ps.evaluate(make_report(with_entry(make_record(), "chat-capital", content=content)))
    assert v.reasons == [] and any(kind in a for a in v.advisory), (kind, v.advisory)


# ---- cases added after the first mutation run -----------------------------------------------------------------------

def test_a_leak_of_the_controls_own_special_token_is_shared():
    rec = with_entry(make_record(special_tokens=["[SEP]"]), "chat-capital", content="Paris [SEP] again")
    rec["control"] = control_of(rec)
    assert ps.evaluate(make_report(rec)).reasons == []


def test_a_leak_between_requests_in_the_control_is_shared():
    rec = with_entry(make_record(), "multi-turn", content="Your name is Zephyrine. Lindfors is a harbor town.")
    rec["control"] = control_of(rec)
    assert ps.evaluate(make_report(rec)).reasons == []


def test_an_entry_with_no_id_is_a_reason():
    rec = make_record()
    rec["entries"][0]["id"] = None
    assert ps.evaluate(make_report(rec)).reasons


def test_two_entries_for_one_prompt_are_a_reason():
    rec = make_record()
    rec["entries"].append(copy.deepcopy(rec["entries"][0]))
    assert any("share one" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_no_prompt_may_be_skipped_when_the_chat_template_has_no_tools_either():
    rec = make_record(chat_template={"present": True, "source": "chat_template.jinja", "sha256": "a" * 64, "mentions_tools": False})
    rec["entries"] = [e for e in rec["entries"] if e["id"] != "arithmetic"]
    rec["skipped"] = [{"id": "arithmetic", "reason": "slow"}]
    assert any("arithmetic" in r and "skipped" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_an_extra_entry_that_is_not_an_object_is_a_reason_even_when_every_prompt_has_its_reply():
    rec = make_record()
    rec["entries"].append("oops")
    assert any("not an object" in r for r in ps.evaluate(make_report(rec)).reasons)


# ---- a run whose replies all equal the control's may be serving the control's weights -------------------------------------
#
# Issue 15: on the first prototype the base model's weights stood in by mistake and scored 0.78 against the fine-tune's
# reference, close to the bar. A healthy base model gives healthy replies, so no degenerate-output check can see that. A
# fine-tune usually differs from its base within 64 greedy tokens on some of the prompts; a run that is identical on all of them
# has probably not served the new weights.

def distinct_replies(record):
    rec = copy.deepcopy(record)
    for i, e in enumerate(rec["entries"]):
        e["content"] = f"{e['content']} (the fine-tune adds sentence {i})" if e["content"] else e["content"]
    return rec


def test_replies_identical_to_the_control_on_every_prompt_fail_the_run_and_say_why():
    rec = make_record()
    rec["control"] = control_of(rec, differ=False)
    v = ps.evaluate(make_report(rec))
    assert len(v.reasons) == 1
    assert "identical" in v.reasons[0] and "control" in v.reasons[0] and "weights" in v.reasons[0]


def test_one_different_reply_is_enough_to_show_the_weights_differ():
    rec = distinct_replies(make_record())
    ctl = make_record()
    one_more = copy.deepcopy(ctl)
    # all replies equal except the first
    for e_new, e_old in zip(rec["entries"][1:], one_more["entries"][1:]):
        e_new["content"] = e_old["content"]
    rec["control"] = control_of(one_more)
    assert ps.evaluate(make_report(rec)).reasons == []


def test_a_run_whose_replies_differ_from_the_control_passes():
    rec = distinct_replies(make_record())
    rec["control"] = control_of(make_record())
    assert ps.evaluate(make_report(rec)).reasons == []


def test_too_few_comparable_replies_say_nothing():
    rec = make_record()
    for e in rec["entries"][ps_min():]:
        e.update(content="", tool_calls=[])     # no reply to compare
    rec["control"] = control_of(rec, differ=False)
    assert not any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def ps_min():
    from orchard.defaults import SUITE_IDENTICAL_MIN_PROMPTS
    return SUITE_IDENTICAL_MIN_PROMPTS - 1


def test_exactly_the_minimum_of_identical_replies_counts():
    from orchard.defaults import SUITE_IDENTICAL_MIN_PROMPTS
    rec = make_record()
    for e in rec["entries"][SUITE_IDENTICAL_MIN_PROMPTS:]:
        e["content"] = ""
    rec["control"] = control_of(rec, differ=False)
    assert any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_a_failed_request_is_not_a_reply_that_matches():
    rec = make_record()
    for e in rec["entries"]:
        e.update(content="", error="HTTP 500")
    rec["control"] = control_of(rec, differ=False)
    assert not any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_a_tool_call_reply_that_matches_counts_as_a_match():
    from orchard.defaults import SUITE_IDENTICAL_MIN_PROMPTS as need
    rec = make_record()
    keep = [e["id"] for e in rec["entries"] if e["id"] != "tool-call"][:need - 1] + ["tool-call"]
    for e in rec["entries"]:
        if e["id"] not in keep:
            e.update(content="", tool_calls=[])
    rec["control"] = control_of(rec, differ=False)
    assert any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_a_control_with_fewer_replies_than_the_run_is_compared_on_the_prompts_it_has():
    rec = make_record()
    ctl = control_of(rec, differ=False)
    ctl["record"]["entries"] = ctl["record"]["entries"][:3]
    rec["control"] = ctl
    reasons = ps.evaluate(make_report(rec)).reasons
    assert not any("identical" in r for r in reasons)


def test_a_control_served_from_the_same_model_id_is_not_a_control():
    rec = distinct_replies(make_record())
    ctl = control_of(make_record(), differ=False)
    ctl["record"]["model_id"] = rec["model_id"]
    rec["control"] = ctl
    assert any("same model" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_records_without_a_model_id_are_not_held_to_the_same_model_rule():
    rec = distinct_replies(make_record())
    del rec["model_id"]
    ctl = control_of(make_record())
    ctl["record"].pop("model_id", None)
    rec["control"] = ctl
    assert ps.evaluate(make_report(rec)).reasons == []


def test_an_identical_control_that_cannot_be_compared_is_not_reported_as_identical():
    rec = make_record()
    ctl = control_of(rec, differ=False)
    ctl["record"]["suite_sha256"] = "0" * 64
    rec["control"] = ctl
    reasons = ps.evaluate(make_report(rec)).reasons
    assert any("explains nothing" in r for r in reasons) and not any("identical" in r for r in reasons)


def test_the_identical_reasoning_text_counts_when_there_is_no_answer():
    rec = make_record()
    for e in rec["entries"]:
        e.update(content="", reasoning="We need to think about this for a while and then answer.")
    rec["control"] = control_of(rec, differ=False)
    assert any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_a_reply_with_an_error_recorded_is_not_compared_even_if_it_has_text():
    rec = make_record()
    for e in rec["entries"]:
        e["error"] = "HTTP 500"
    rec["control"] = control_of(rec, differ=False)
    assert not any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)
    # and the same for the control's side
    rec = make_record()
    ctl = control_of(rec, differ=False)
    for e in ctl["record"]["entries"]:
        e["error"] = "HTTP 500"
    rec["control"] = ctl
    assert not any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_replies_with_the_same_answer_and_different_reasoning_are_not_identical():
    rec = make_record()
    ctl = control_of(rec, differ=False)
    for e in rec["entries"]:
        e["reasoning"] = "thinking A"
    for e in ctl["record"]["entries"]:
        e["reasoning"] = "thinking B"
    rec["control"] = ctl
    assert not any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_a_tool_call_with_other_arguments_is_not_identical():
    rec = make_record()
    ctl = control_of(rec, differ=False)
    next(e for e in ctl["record"]["entries"] if e["id"] == "tool-call")["tool_calls"] = [
        {"name": "get_weather", "arguments": '{"city": "Porto"}'}]
    rec["control"] = ctl
    assert not any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_whitespace_alone_is_not_a_reply_to_compare():
    from orchard.defaults import SUITE_IDENTICAL_MIN_PROMPTS as need
    rec = make_record()
    for e in rec["entries"][need - 1:]:
        e.update(content="  \n", tool_calls=[])
    rec["control"] = control_of(rec, differ=False)
    assert not any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_prompts_one_side_failed_on_are_left_out_and_the_rest_are_still_compared():
    from orchard.defaults import SUITE_IDENTICAL_MIN_PROMPTS as need
    # the model failed two requests; the other prompts match the control exactly
    rec = make_record()
    ctl = control_of(rec, differ=False)
    for e in rec["entries"][:2]:
        e.update(content="", error="HTTP 500")
    rec["control"] = ctl
    assert len(rec["entries"]) - 2 >= need
    assert any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)
    # and the control failed two
    rec = make_record()
    ctl = control_of(rec, differ=False)
    for e in ctl["record"]["entries"][:2]:
        e.update(content="", error="HTTP 500")
    rec["control"] = ctl
    assert any("identical" in r for r in ps.evaluate(make_report(rec)).reasons)
