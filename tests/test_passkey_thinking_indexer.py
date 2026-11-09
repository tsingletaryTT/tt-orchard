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

THINKING = {"template_thinks": True, "switch": None, "off": False, "max_tokens": 512}
SWITCHED_OFF = {"template_thinks": True, "switch": "enable_thinking", "off": True, "max_tokens": 64}
PLAIN = {"template_thinks": False, "switch": None, "off": False, "max_tokens": 12}


def runs(*answers, key="94554"):
    return [{"length": 256 * (i + 1), "key": key, "answer_text": a, "prompt_tokens": 256, "ids_first": [1], "ids_second": [1]} for i, a in enumerate(answers)]


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

def test_the_template_finds_a_thinking_switch_in_the_chat_template_file(tmp_path):
    mod = template()
    assert mod.thinking_support(tmp_path) == {"template_thinks": False, "switch": None}
    (tmp_path / "chat_template.jinja").write_text("{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}")
    assert mod.thinking_support(tmp_path) == {"template_thinks": True, "switch": "enable_thinking"}
    (tmp_path / "chat_template.jinja").write_text("{{ messages[0].content }}<think>\n")
    assert mod.thinking_support(tmp_path) == {"template_thinks": True, "switch": None}


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


def test_the_indexer_is_exercised_only_when_the_context_exceeds_the_budget():
    facts = {"sparse_attention": {"kind": "qsa-indexer", "budget": 2048, "block_size": 4, "source": "config.qsa"}}
    short = full_port_result.indexer_record(facts, prompt_tokens=64, generated=32)
    assert short["indexer_exercised"] is False and "not exercised" in short["indexer_basis"] and "2048" in short["indexer_basis"]
    long = full_port_result.indexer_record(facts, prompt_tokens=4200, generated=32)
    assert long["indexer_exercised"] is True and "4232" in long["indexer_basis"]
    assert full_port_result.indexer_record({}, prompt_tokens=64, generated=32) == {}
    assert full_port_result.indexer_record({"sparse_attention": {"kind": "qsa-indexer", "budget": "x"}}, prompt_tokens=64, generated=32) == {}


def test_the_gate_recomputes_the_indexer_field_and_refuses_a_false_claim():
    facts = {"sparse_attention": {"kind": "qsa-indexer", "budget": 2048, "block_size": 4, "source": "config.qsa"}}
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
    assert record["passkey"]["thinking"] == {"template_thinks": True, "switch": "enable_thinking", "off": True, "max_tokens": 64}
    assert all(r["answer_text"] for r in record["passkey"]["runs"])
    result, why = fk.derive(b)
    assert result["passkey_status"] == "answered" and result["coherent"] is True, why
    reasons, _, _ = fk.gate(b)
    assert reasons == [], reasons


def test_a_thinking_model_without_a_switch_gets_the_longer_limit_and_an_unclosed_reply_is_not_applicable(tmp_path):
    fk, b, record = _serve(tmp_path, "{% for m in messages %}{{ m.content }}{% endfor %}<think>\n")
    assert record["passkey"]["thinking"] == {"template_thinks": True, "switch": None, "off": False, "max_tokens": 512}
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
