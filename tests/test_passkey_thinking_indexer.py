# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 641 and 638 (found on the Qwen3.8-27B-QSA bring-up).

641: stage 5's passkey check asked a thinking model for 12 tokens. The replies were thinking text ('We need answer user's prompt...') cut off
before any key, the corpus agreement was 80 of 80, and the gate said 'a server that cannot find a key in its own context is not serving the model'.
Now the template turns thinking off when the chat template has a switch, otherwise reads the answer after the thinking block with a larger limit,
and a reply that is still inside the thinking block is 'not applicable', never 'broken'. A non-thinking model, or a thinking-off request, still
fails a missing key.

638: stage 2's agreement prompts were shorter than the indexer budget, so a port that ignored the indexer passed. The facts now carry
`sparse_attention` and stage 2's result says whether the check exercised it. No chip, no model."""
import importlib.util
import json

import pytest

from orchard import full_port, full_port_result, full_port_serve, port_facts

THINKING = {"template_thinks": True, "switch": None, "off": False, "plan_max_tokens": 512}
SWITCHED_OFF = {"template_thinks": True, "switch": "enable_thinking", "off": True, "plan_max_tokens": 64}
PLAIN = {"template_thinks": False, "switch": None, "off": False, "plan_max_tokens": 12}


def runs(*answers, key="94554", finish="length"):
    """Passkey runs. A reply is cut by the answer limit unless `finish` says it stopped: the default is a truncated reply (finish_reason 'length', the
    reply's token count equal to its limit of 1 here), which is the only evidence that earns the not-applicable status."""
    return [{"length": 256 * (i + 1), "key": key, "answer_text": a, "prompt_tokens": 256, "ids_first": [1], "ids_second": [1], "finish_reason": finish,
             "limit": 1 if finish == "length" else 1000} for i, a in enumerate(answers)]


def template():
    spec = importlib.util.spec_from_file_location("port_serve_under_test_641", full_port_serve.TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- 641: the gate's reading of the answers --------------------------------------------------------------------------------------

def test_a_non_thinking_model_that_loses_the_key_still_fails():
    answered, status = full_port_serve.passkey_answers(runs("The key is 94554.", "I do not know."), PLAIN)
    assert answered == [True, False] and status == "failed"


def test_a_thinking_reply_cut_inside_the_thinking_block_is_not_applicable_and_not_a_failure():
    answered, status = full_port_serve.passkey_answers(runs("We need answer user's prompt. User: repeated text,", "We need answer user's request"), THINKING)
    assert answered == [False, False] and status == "not-applicable"


def test_a_thinking_reply_is_read_after_the_closing_tag():
    answered, status = full_port_serve.passkey_answers(runs("We need the key.</think>\n\nThe key is 94554.", "hmm</think>94554"), THINKING)
    assert answered == [True, True] and status == "answered"


def test_the_key_repeated_inside_the_thinking_block_does_not_count():
    answered, status = full_port_serve.passkey_answers(runs("The key might be 94554, let me check.</think>\n\nI cannot tell."), THINKING)
    assert answered == [False] and status == "failed"


def test_a_missing_key_with_thinking_switched_off_is_a_failure():
    answered, status = full_port_serve.passkey_answers(runs("The key is 11111."), SWITCHED_OFF)
    assert answered == [False] and status == "failed"
    answered, status = full_port_serve.passkey_answers(runs("The key is 94554."), SWITCHED_OFF)
    assert answered == [True] and status == "answered"


def test_one_answered_and_one_cut_reply_is_not_applicable_but_one_wrong_closed_reply_fails():
    assert full_port_serve.passkey_answers(runs("ok</think>94554", "still thinking"), THINKING)[1] == "not-applicable"
    assert full_port_serve.passkey_answers(runs("done</think>no idea", "still thinking"), THINKING)[1] == "failed"


def test_a_record_without_the_thinking_section_is_read_as_a_non_thinking_model():
    assert full_port_serve.thinking_of({"runs": runs("x")}) == full_port_serve.NO_THINKING
    assert full_port_serve.thinking_of({"thinking": {"template_thinks": "yes"}, "runs": []}) == full_port_serve.NO_THINKING
    assert full_port_serve.thinking_of({"thinking": THINKING, "runs": []}) == THINKING


def test_the_failure_message_names_the_lengths_and_does_not_blame_the_server_for_a_cut_reply():
    reasons = []
    full_port_serve._result_reasons({"serves": True, "coherent": False, "passkey_answered": [False, False], "passkey_lengths": [256, 512]}, None, None,
                                    ([False, False], True, "failed"), False, None, reasons, {"passkey": {"runs": runs("a", "b")}})
    text = " ".join(reasons)
    assert "256" in text and "512" in text and "holds no key" in text
    cut = []
    full_port_serve._result_reasons({"serves": True, "coherent": None, "passkey_answered": [False, False], "passkey_lengths": [256, 512],
                                     "passkey_status": "not-applicable"}, None, None, ([False, False], True, "not-applicable"), False, None, cut,
                                    {"passkey": {"runs": runs("a", "b"), "thinking": THINKING}})
    assert not any("coherent" in r or "serving the model" in r for r in cut), cut


def test_coherent_must_be_null_when_the_passkey_is_not_applicable_and_a_bool_otherwise():
    reasons = []
    full_port_serve._result_reasons({"serves": True, "coherent": True, "passkey_answered": [False], "passkey_lengths": [256], "passkey_status": "not-applicable"},
                                    None, None, ([False], True, "not-applicable"), False, None, reasons, {"passkey": {"runs": runs("a"), "thinking": THINKING}})
    assert any("coherent" in r for r in reasons)


# ---- 641: the template ---------------------------------------------------------------------------------------------------------

def _support(mod, tmp_path, text):
    (tmp_path / "chat_template.jinja").write_text(text)
    return mod.renderer_and_thinking(tmp_path)[2]


def test_the_template_finds_a_thinking_switch_from_the_generation_rendering(tmp_path):
    mod = template()
    assert mod.renderer_and_thinking(tmp_path)[2] == {"template_thinks": False, "switch": None}
    assert _support(mod, tmp_path, "{% for m in messages %}{{ m.content }}{% endfor %}{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n"
                                   "{% else %}<think>\n{% endif %}") == {"template_thinks": True, "switch": "enable_thinking"}
    assert _support(mod, tmp_path, "{{ messages[0].content }}<think>\n") == {"template_thinks": True, "switch": None}


def test_a_think_marker_in_a_jinja_comment_does_not_make_a_model_a_thinking_model(tmp_path):
    mod = template()
    assert _support(mod, tmp_path, "{# the model writes <think> blocks and tests enable_thinking #}{% for m in messages %}{{ m.content }}{% endfor %}<assistant>") == {
        "template_thinks": False, "switch": None}


def test_a_think_marker_that_only_formats_history_does_not_make_the_generation_turn_a_thinking_turn(tmp_path):
    mod = template()
    text = ("{% for m in messages %}{% if m.role == 'assistant' %}<think>{{ m.reasoning }}</think>{{ m.content }}{% else %}{{ m.content }}{% endif %}{% endfor %}"
            "<assistant>")
    assert _support(mod, tmp_path, text) == {"template_thinks": False, "switch": None}


def test_a_closed_empty_think_block_in_the_generation_prompt_is_not_a_thinking_turn(tmp_path):
    mod = template()
    assert _support(mod, tmp_path, "{{ messages[0].content }}<assistant><think>\n\n</think>\n\n")["template_thinks"] is False


def test_the_switch_is_only_reported_when_it_changes_the_rendering(tmp_path):
    mod = template()
    text = "{{ messages[0].content }}{% if enable_thinking is defined %}[x]{% endif %}<assistant><think>\n"
    assert _support(mod, tmp_path, text)["switch"] == "enable_thinking"
    assert _support(mod, tmp_path, "{# enable_thinking #}{{ messages[0].content }}<assistant><think>\n")["switch"] is None


def test_the_renderer_reads_the_chat_template_jinja_file_and_passes_the_switch(tmp_path):
    mod = template()
    (tmp_path / "chat_template.jinja").write_text("{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}<assistant>"
                                                  "{% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}")
    render, name = mod.chat_renderer(tmp_path)
    assert name == "chat_template.jinja" and render([{"role": "user", "content": "hi"}]) == "<user>hi<assistant>"
    assert render([{"role": "user", "content": "hi"}], enable_thinking=False).endswith("<think></think>")


def test_the_passkey_plan_turns_thinking_off_with_a_switch_and_reads_after_the_block_without_one():
    mod = template()
    off = mod.passkey_plan({"template_thinks": True, "switch": "enable_thinking"})
    assert off["off"] is True and off["kwargs"] == {"enable_thinking": False} and off["max_tokens"] <= 64
    read = mod.passkey_plan({"template_thinks": True, "switch": None})
    assert read["off"] is False and read["kwargs"] == {} and read["max_tokens"] >= 256
    plain = mod.passkey_plan({"template_thinks": False, "switch": None})
    assert plain["off"] is False and plain["max_tokens"] == 12


# ---- 638: the facts and the result --------------------------------------------------------------------------------------------

def test_the_facts_carry_the_indexer_budget_from_the_config_table():
    assert port_facts.sparse_attention({"qsa": {"format_version": 1, "budget": 2048, "block_size": 4, "indexer_heads": 4, "runtime": "reference_sdpa"}}, {}) == {
        "kind": "qsa-indexer", "budget": 2048, "block_size": 4, "source": "config.qsa"}
    assert port_facts.sparse_attention({}, {"sliding_window": 512, "use_sliding_window": True}) == {"kind": "sliding-window", "window": 512, "source": "text_config.sliding_window"}
    assert port_facts.sparse_attention({}, {}) is None


@pytest.mark.parametrize("bad", [{"budget": "2048"}, {"budget": 0}, {"budget": True}, {"budget": -4}, {"budget": 10 ** 40}, "qsa", [], {}])
def test_a_budget_that_is_not_a_plain_positive_number_gives_no_field(bad):
    assert port_facts.sparse_attention({"qsa": bad}, {}) is None


def _qsa(budget=2048, block=4):
    return {"sparse_attention": {"kind": "qsa-indexer", "budget": budget, "block_size": block, "source": "config.qsa"}}


def test_the_indexer_is_exercised_only_when_the_evaluated_input_exceeds_the_budget_plus_the_block():
    """The check feeds the prompt and every generated token but the last, so it evaluates prompt + generated - 1 inputs. QSA keeps the current
    block beyond its budget, so it selects only once the evaluated length exceeds budget + block size (2052 here)."""
    facts = _qsa()
    for prompt, generated, want in ((2017, 32, False),      # 2048 evaluated: the budget exactly
                                    (2021, 32, False),      # 2052 evaluated: budget plus block, still every key
                                    (2022, 32, True),       # 2053 evaluated: one above
                                    (64, 32, False), (4200, 32, True)):
        got = full_port_result.indexer_record(facts, prompt_tokens=prompt, generated=generated)
        assert got["indexer_exercised"] is want, (prompt, generated, got)
        assert str(prompt + generated - 1) in got["indexer_basis"]
    assert "not exercised" in full_port_result.indexer_record(facts, prompt_tokens=64, generated=32)["indexer_basis"]
    assert full_port_result.indexer_record({}, prompt_tokens=64, generated=32) == {}
    assert full_port_result.indexer_record({"sparse_attention": {"kind": "qsa-indexer", "budget": "x"}}, prompt_tokens=64, generated=32) == {}


def test_a_sliding_window_is_exercised_when_the_evaluated_input_exceeds_the_window():
    facts = {"sparse_attention": {"kind": "sliding-window", "window": 512, "source": "text_config.sliding_window"}}
    assert full_port_result.indexer_record(facts, prompt_tokens=480, generated=33)["indexer_exercised"] is False      # 512 evaluated
    assert full_port_result.indexer_record(facts, prompt_tokens=481, generated=33)["indexer_exercised"] is True       # 513


@pytest.mark.parametrize("kind", [["qsa-indexer"], {"a": 1}, 7, None, True, ""])
def test_a_kind_that_is_not_a_known_string_gives_no_record_and_raises_nothing(kind):
    facts = {"sparse_attention": {"kind": kind, "budget": 2048, "window": 512}}
    assert full_port.sparse_budget(facts) is None
    assert full_port_result.indexer_record(facts, prompt_tokens=64, generated=32) == {}
    assert full_port.indexer_reasons({}, facts, prompt_tokens=64, generated=32) == []


def test_the_gate_recomputes_the_indexer_field_and_refuses_a_false_claim():
    facts = _qsa()
    ok = full_port.indexer_reasons({"indexer_exercised": False, "indexer_basis": full_port_result.indexer_record(facts, prompt_tokens=64, generated=32)["indexer_basis"]},
                                   facts, prompt_tokens=64, generated=32)
    assert ok == []
    claimed = full_port.indexer_reasons({"indexer_exercised": True, "indexer_basis": "x"}, facts, prompt_tokens=64, generated=32)
    assert claimed and "indexer_exercised" in claimed[0]
    missing = full_port.indexer_reasons({}, facts, prompt_tokens=64, generated=32)
    assert missing and "indexer_exercised" in missing[0]
    assert full_port.indexer_reasons({}, {}, prompt_tokens=64, generated=32) == []


# ---- 641 end to end: the real template as a subprocess on the CPU, a fake port, the real gate -----------------------------------------

def _serve(tmp_path, chat_template: str):
    import port_serve_fakes as fk
    b = fk.build(tmp_path)
    (b["snap"] / "chat_template.jinja").write_text(chat_template)
    done = fk.run_serve(b)
    assert done.returncode == 0, done.stdout + done.stderr
    record = json.loads((b["stage"] / full_port_serve.RECORD_FILE).read_text())
    return fk, b, record


def test_a_model_with_a_thinking_switch_is_asked_with_thinking_off_and_the_record_says_so(tmp_path):
    fk, b, record = _serve(tmp_path, "{% for m in messages %}{{ m.content }}{% endfor %}{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}")
    assert record["passkey"]["thinking"] == {"template_thinks": True, "switch": "enable_thinking", "off": True, "plan_max_tokens": 64}
    assert all(r["answer_text"] for r in record["passkey"]["runs"])
    result, why = fk.derive(b)
    assert result["passkey_status"] == "answered" and result["coherent"] is True, why
    reasons, _, _ = fk.gate(b)
    assert reasons == [], reasons


def test_a_thinking_model_without_a_switch_gets_the_longer_limit_and_an_unclosed_reply_is_not_applicable(tmp_path):
    fk, b, record = _serve(tmp_path, "{% for m in messages %}{{ m.content }}{% endfor %}<think>\n")
    assert record["passkey"]["thinking"] == {"template_thinks": True, "switch": None, "off": False, "plan_max_tokens": 512}
    result, why = fk.derive(b)
    assert result["coherent"] is None and result["passkey_status"] == "not-applicable", why
    assert result["checks"]["passkey"]["pass"] is True
    reasons, _, waived = fk.gate(b)
    assert reasons == [] and waived == [], reasons


def test_a_forged_coherent_true_on_an_unjudged_passkey_is_refused(tmp_path):
    fk, b, _ = _serve(tmp_path, "{% for m in messages %}{{ m.content }}{% endfor %}<think>\n")
    result, _ = fk.derive(b)
    result["coherent"] = True
    (b["stage"] / "result.json").write_text(json.dumps(result))
    reasons, _, _ = fk.gate(b)
    assert any("coherent" in r for r in reasons)


def test_a_model_that_does_not_think_still_needs_its_key_at_every_length(tmp_path):
    import port_serve_fakes as fk
    b = fk.build(tmp_path, fk.RANDOM_WEIGHTS)
    done = fk.run_serve(b)
    assert done.returncode == 0, done.stdout + done.stderr
    result, _ = fk.derive(b)
    assert result["coherent"] is False and result["passkey_status"] == "failed"
    reasons, _, _ = fk.gate(b)
    assert any("holds no key" in r for r in reasons), reasons


# ---- 638 end to end: stage 2's derive and gate with a facts file that carries a sparse attention -----------------------------------------

def _stage2_with_facts(tmp_path, budget):
    from test_full_port import build
    stage, run = build(tmp_path)
    (stage / "result.json").unlink()
    (stage / "port_facts.json").write_text(json.dumps({"sparse_attention": {"kind": "qsa-indexer", "budget": budget, "block_size": 4, "source": "config.qsa"}}))
    result, why = full_port_result.derive(stage, run)
    assert result is not None, why
    return stage, run, result


def test_stage_2_says_the_indexer_was_not_exercised_when_the_prompt_is_far_below_the_budget(tmp_path):
    stage, run, result = _stage2_with_facts(tmp_path, 2048)
    assert result["indexer_exercised"] is False and "not exercised" in result["indexer_basis"]
    (stage / "result.json").write_text(json.dumps(result))
    assert full_port.check(stage, run)[0] == []                  # informative: the run still passes, and says what it did not show


def test_stage_2_says_the_indexer_was_exercised_when_the_context_passes_the_budget(tmp_path):
    stage, run, result = _stage2_with_facts(tmp_path, 4)
    assert result["indexer_exercised"] is True
    (stage / "result.json").write_text(json.dumps(result))
    assert full_port.check(stage, run)[0] == []


def test_a_result_that_claims_the_indexer_was_exercised_when_it_was_not_is_refused(tmp_path):
    stage, run, result = _stage2_with_facts(tmp_path, 2048)
    result["indexer_exercised"] = True
    (stage / "result.json").write_text(json.dumps(result))
    assert any("indexer_exercised" in r for r in full_port.check(stage, run)[0])


def test_a_result_without_the_field_is_refused_when_the_facts_carry_a_sparse_attention(tmp_path):
    stage, run, result = _stage2_with_facts(tmp_path, 2048)
    for key in ("indexer_exercised", "indexer_basis"):
        result.pop(key)
    (stage / "result.json").write_text(json.dumps(result))
    assert any("indexer_exercised" in r for r in full_port.check(stage, run)[0])


def test_facts_without_a_sparse_attention_change_nothing(tmp_path):
    from test_full_port import build
    stage, run = build(tmp_path)
    (stage / "result.json").unlink()
    (stage / "port_facts.json").write_text(json.dumps({"layer_schedule": None}))
    result, _ = full_port_result.derive(stage, run)
    assert "indexer_exercised" not in result
    (stage / "result.json").write_text(json.dumps(result))
    assert full_port.check(stage, run)[0] == []


def test_a_passkey_status_that_differs_from_the_recomputed_one_is_refused():
    reasons = []
    full_port_serve._result_reasons({"serves": True, "coherent": True, "passkey_answered": [True], "passkey_lengths": [256], "passkey_status": "not-applicable"},
                                    None, None, ([True], True, "answered"), False, None, reasons, {"passkey": {"runs": runs("94554")}})
    assert any("passkey_status" in r for r in reasons), reasons


# ---- review of PR 652 ----------------------------------------------------------------------------------------------------------

def test_a_template_that_falls_back_to_the_plain_renderer_gives_non_thinking_facts(tmp_path):
    """A chat template with <think> and no switch that jinja cannot render is not used: the prompt is the plain form. Reading the replies as a
    thinking model's would turn a wrong plain-rendered reply into 'not applicable'."""
    mod = template()
    (tmp_path / "chat_template.jinja").write_text("{{ raise_exception('no') }}<think>\n")
    render, name, support = mod.renderer_and_thinking(tmp_path)
    assert name == "plain" and support == {"template_thinks": False, "switch": None}
    (tmp_path / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}<think>\n")
    render, name, support = mod.renderer_and_thinking(tmp_path)
    assert name == "chat_template.jinja" and support == {"template_thinks": True, "switch": None}


def test_the_reply_is_decoded_a_second_time_with_special_tokens_kept():
    mod = template()
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "the": 1, "key": 2, "</think>": 3, "94554": 4}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["</think>"])
    clean, raw = mod.decode_reply(tok, [1, 3, 4])
    assert "</think>" not in clean and "</think>" in raw and "94554" in raw


def test_a_wrong_answer_after_a_special_closing_tag_is_a_failure_read_from_the_raw_text():
    wrong = [{"length": 256, "key": "94554", "answer_text": " no idea", "answer_raw": "hmm</think> no idea", "prompt_tokens": 256, "ids_first": [1], "ids_second": [1]}]
    assert full_port_serve.passkey_answers(wrong, THINKING) == ([False], "failed")
    right = [dict(wrong[0], answer_text=" 94554", answer_raw="hmm</think> 94554")]
    assert full_port_serve.passkey_answers(right, THINKING) == ([True], "answered")
    cut = [dict(wrong[0], answer_text="still thinking", answer_raw="still thinking", finish_reason="length", limit=1)]
    assert full_port_serve.passkey_answers(cut, THINKING)[1] == "not-applicable"
    legacy = [{"length": 256, "key": "94554", "answer_text": "x</think>94554", "prompt_tokens": 256, "ids_first": [1], "ids_second": [1]}]
    assert full_port_serve.passkey_answers(legacy, THINKING) == ([True], "answered")          # a record without answer_raw reads answer_text


def test_the_answer_limit_is_clamped_against_the_rendered_prompt_not_the_requested_filler():
    mod = template()

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class E:
                ids = list(range(len(text)))
            return E()
    render = lambda messages, **kw: "<m>" * 10 + messages[0]["content"]          # 30 marker characters count as tokens in this fake
    assert mod.answer_limit(Tok(), render, "x" * 60, {}, plan_max=512, max_len=100) == 9           # 90 prompt tokens, room 100 - 90 - 1
    assert mod.answer_limit(Tok(), render, "x" * 60, {}, plan_max=4, max_len=100) == 4
    with pytest.raises(mod.ServeFailed, match="context"):
        mod.answer_limit(Tok(), render, "x" * 80, {}, plan_max=512, max_len=100)


def test_an_answer_raw_that_is_not_text_is_a_reason_not_an_error(tmp_path):
    import port_serve_fakes as fk
    b = fk.build(tmp_path)
    assert fk.run_serve(b).returncode == 0
    assert fk.derive(b)[0] is not None
    path = b["stage"] / full_port_serve.RECORD_FILE
    record = json.loads(path.read_text())
    record["passkey"]["runs"][0]["answer_raw"] = ["x"]
    path.write_text(json.dumps(record))
    reasons, _, _ = fk.gate(b)
    assert any("answer_raw" in r for r in reasons)


# ---- review of PR 652, round 2 ----------------------------------------------------------------------------------------------------

def test_a_thinking_reply_that_stopped_before_closing_its_block_is_a_failed_answer_not_a_truncated_one():
    stopped = runs("We need to find the key. I give up", finish="stop")
    assert full_port_serve.passkey_answers(stopped, THINKING) == ([False], "failed")
    assert full_port_serve.passkey_answers(runs("We need to find", finish="length"), THINKING)[1] == "not-applicable"


def test_the_token_count_reaching_the_limit_is_also_evidence_of_truncation():
    at_limit = [{"length": 256, "key": "94554", "answer_text": "thinking", "prompt_tokens": 256, "ids_first": [1, 2, 3], "ids_second": [1, 2, 3],
                 "finish_reason": "stop", "limit": 3}]
    assert full_port_serve.passkey_answers(at_limit, THINKING)[1] == "not-applicable"
    below = [dict(at_limit[0], limit=4)]
    assert full_port_serve.passkey_answers(below, THINKING)[1] == "failed"
    no_evidence = [{k: v for k, v in at_limit[0].items() if k not in ("finish_reason", "limit")}]
    assert full_port_serve.passkey_answers(no_evidence, THINKING)[1] == "failed"


def test_the_record_names_the_plan_ceiling_and_each_run_records_its_finish_reason_and_limit(tmp_path):
    fk, b, record = _serve(tmp_path, "{% for m in messages %}{{ m.content }}{% endfor %}<think>\n")
    assert record["passkey"]["thinking"]["plan_max_tokens"] == 512 and "max_tokens" not in record["passkey"]["thinking"]
    for r in record["passkey"]["runs"]:
        assert r["finish_reason"] in ("stop", "length") and isinstance(r["limit"], int) and 1 <= r["limit"] <= 512


def test_a_stopped_thinking_reply_fails_the_gate_end_to_end(tmp_path):
    fk, b, record = _serve(tmp_path, "{% for m in messages %}{{ m.content }}{% endfor %}<think>\n")
    path = b["stage"] / full_port_serve.RECORD_FILE
    for r in record["passkey"]["runs"]:
        r["finish_reason"], r["limit"], r["ids_first"] = "stop", 10 ** 6, r["ids_first"][:2]
        r["ids_second"] = r["ids_first"]
    path.write_text(json.dumps(record))
    assert fk.derive(b)[0] is not None
    reasons, _, _ = fk.gate(b)
    assert any("holds no key" in r for r in reasons), reasons


def test_the_diagnostic_lists_only_the_runs_that_failed_on_their_own():
    mixed = runs("done</think>no idea", "still thinking")                    # one closed and wrong, one cut
    answered, status = full_port_serve.passkey_answers(mixed, THINKING)
    assert status == "failed"
    reasons = []
    full_port_serve._result_reasons({"serves": True, "coherent": False, "passkey_answered": answered, "passkey_lengths": [256, 512], "passkey_status": "failed"},
                                    None, None, (answered, True, status), False, None, reasons, {"passkey": {"runs": mixed, "thinking": THINKING}})
    text = " ".join(reasons)
    assert "[256]" in text and "512" not in text, text


def test_the_run_statuses_are_answered_cut_or_failed():
    got = full_port_serve.passkey_run_statuses(runs("ok</think>94554", "thinking...", "x</think>no"), THINKING)
    assert got == ["answered", "cut", "failed"]


def test_the_servers_length_finish_reason_alone_is_evidence_of_truncation():
    run = [{"length": 256, "key": "94554", "answer_text": "thinking", "prompt_tokens": 256, "ids_first": [1, 2], "ids_second": [1, 2], "finish_reason": "length", "limit": 1000}]
    assert full_port_serve.passkey_answers(run, THINKING)[1] == "not-applicable"


@pytest.mark.parametrize("ceiling", [0, -1, 4097, "512", True, None, 1.5])
def test_a_plan_ceiling_that_is_not_a_whole_number_from_1_to_4096_reads_as_no_thinking(ceiling):
    section = {"thinking": {"template_thinks": True, "switch": None, "off": False, "plan_max_tokens": ceiling}, "runs": []}
    assert full_port_serve.thinking_of(section) == full_port_serve.NO_THINKING
    assert full_port_serve.thinking_of({"thinking": {"template_thinks": True, "switch": None, "off": False, "max_tokens": 512}, "runs": []}) == full_port_serve.NO_THINKING
