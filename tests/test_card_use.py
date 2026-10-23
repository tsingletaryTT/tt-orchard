# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What a package card can say about how to use a model, read from the model's own files (card_use.py).

The fixtures are cut from real files: AREX-2's chat template (a reasoning effort list, thinking on unless
`enable_thinking` is false) and the CMD and export lines of the package's run.sh."""
import json

from orchard.card_use import (UseFacts, authors_intended_use, flag_facts, has_facts, read_use_facts, template_facts,
                              tool_call_evidence)

TEMPLATE = """\
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort) }}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\\n' }}
{%- endif %}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}
{%- else %}
    {{- '<think>\\n' }}
{%- endif %}
"""

RUN_SH = """\
#!/usr/bin/env bash
export QWEN36_SKIP_VISION="1"
export QWEN36_DRAFTER="dflash2"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "$HERE/model-dir" --max_num_seqs 4 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning_parser qwen3 --no-async-scheduling "$@")
"""

README = """\
---
license: apache-2.0
---
# Model

AREX-2 is a 27B long-horizon agent model. It learns to improve a solution over rounds.

## Intended use
AREX-2 is intended for research on long-horizon agents, iterative problem
solving, machine-learning engineering, algorithmic coding, and tool-augmented
deep research.

## License
Apache 2.0.
"""


# ---- the template -----------------------------------------------------------------------------------------------

def test_thinking_is_on_by_default_when_the_template_turns_it_off_only_for_an_explicit_false():
    assert template_facts(TEMPLATE)["thinking_default"] is True


def test_a_template_that_never_mentions_thinking_says_nothing_about_it():
    assert template_facts("{{ messages }}")["thinking_default"] is None


def test_thinking_is_off_by_default_when_the_template_turns_it_on_only_for_an_explicit_true():
    t = "{%- if enable_thinking is defined and enable_thinking is true %}<think>{%- endif %}"
    assert template_facts(t)["thinking_default"] is False


def test_the_reasoning_effort_values_and_default_come_from_the_template_list():
    f = template_facts(TEMPLATE)
    assert f["effort_values"] == ("xhigh", "medium", "low")
    assert f["effort_default"] == "xhigh"


def test_a_template_without_a_reasoning_effort_lists_none():
    f = template_facts("{%- if enable_thinking is undefined or enable_thinking is true %}x{%- endif %}")
    assert f["effort_values"] == () and f["effort_default"] is None


def test_the_template_says_whether_thinking_can_be_turned_off():
    assert template_facts(TEMPLATE)["can_disable_thinking"] is True
    assert template_facts("{{ messages }}")["can_disable_thinking"] is False


def test_the_template_says_whether_it_renders_tools():
    assert template_facts(TEMPLATE)["tools_in_template"] is True
    assert template_facts("{{ messages }}")["tools_in_template"] is False


def test_a_word_that_only_contains_tools_does_not_count_as_tool_support():
    assert template_facts("{{ toolset }} {{ bulletools }}")["tools_in_template"] is False


# ---- run.sh -------------------------------------------------------------------------------------------------------

def test_the_flags_come_from_the_run_sh_command_line():
    f = flag_facts(RUN_SH)
    assert (f["reasoning_parser"], f["tool_parser"], f["auto_tool_choice"]) == ("qwen3", "qwen3_coder", True)


def test_a_run_sh_that_skips_the_vision_tower_is_text_only():
    assert flag_facts(RUN_SH)["text_only"] is True


def test_a_run_sh_that_sets_the_skip_flag_to_zero_is_not_text_only():
    assert flag_facts(RUN_SH.replace('QWEN36_SKIP_VISION="1"', 'QWEN36_SKIP_VISION="0"'))["text_only"] is False


def test_a_commented_out_skip_flag_does_not_count():
    assert flag_facts(RUN_SH.replace('export QWEN36_SKIP_VISION="1"', '# export QWEN36_SKIP_VISION="1"'))["text_only"] is False


def test_a_run_sh_with_no_parsers_names_none():
    f = flag_facts("CMD=(python -m vllm.entrypoints.openai.api_server --model x)")
    assert (f["reasoning_parser"], f["tool_parser"], f["auto_tool_choice"], f["text_only"]) == (None, None, False, False)


def test_the_reasoning_parser_flag_is_read_with_a_hyphen_too():
    assert flag_facts("CMD=(x --reasoning-parser deepseek_r1)")["reasoning_parser"] == "deepseek_r1"


def test_a_flag_in_a_comment_does_not_count():
    assert flag_facts("# --tool-call-parser hermes\nCMD=(x)")["tool_parser"] is None


# ---- the authors' words -----------------------------------------------------------------------------------------

def test_the_authors_intended_use_paragraph_is_returned_as_one_line():
    text = authors_intended_use(README)
    assert text == ("AREX-2 is intended for research on long-horizon agents, iterative problem solving, "
                    "machine-learning engineering, algorithmic coding, and tool-augmented deep research.")


def test_the_description_paragraph_is_not_taken_as_the_intended_use():
    assert "27B" not in (authors_intended_use(README) or "")


def test_a_card_with_no_intended_use_section_gives_none():
    assert authors_intended_use("# Model\n\nSome text.\n") is None


def test_a_section_called_intended_uses_or_uses_counts():
    assert authors_intended_use("## Intended Uses\nFor chat.\n\n## License\nx") == "For chat."


def test_the_section_stops_at_the_next_heading_of_any_level():
    assert authors_intended_use("# Intended use\nFor chat.\n### Out of scope\nNothing.\n") == "For chat."


def test_a_paragraph_with_a_performance_number_is_not_quoted():
    assert authors_intended_use("## Intended use\nRuns at 50 tokens/s and is 99% accurate.\n") is None


def test_an_empty_section_gives_none():
    assert authors_intended_use("## Intended use\n\n## License\nx\n") is None


def test_a_very_long_section_is_cut_at_a_paragraph_boundary():
    para = "Word " * 100
    text = authors_intended_use("## Intended use\n" + "\n\n".join([para.strip()] * 6) + "\n")
    assert text is not None and len(text) <= 1200 and text.count("Word") % 100 == 0


def test_a_single_paragraph_longer_than_the_limit_is_not_quoted():
    assert authors_intended_use("## Intended use\n" + "Word " * 400 + "\n") is None


# ---- evidence that a tool-call prompt returned a call ------------------------------------------------------------

REV = "d4e3502f92d9e889031c2d04e387ab2eb3520268"
ID = ("BAAI/AREX-2", REV)                      # the package's model, as the card step knows it
OTHER_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def _speed(run, calls, model_id=f"BAAI/AREX-2@{REV}"):
    p = run / "stages/7/benchmark/speed.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"prompts": {"tool-call": {"runs": [{"tool_calls": calls}]}}}
    if model_id is not None:
        body["model_id"] = model_id
    p.write_text(json.dumps(body))
    return p


def test_a_speed_file_with_a_returned_tool_call_is_evidence(tmp_path):
    _speed(tmp_path, [{"name": "run_tests", "arguments": "{}"}])
    assert tool_call_evidence(tmp_path, ID) == ("stages/7/benchmark/speed.json", ("run_tests",))


def test_empty_tool_call_lists_are_no_evidence(tmp_path):
    _speed(tmp_path, [])
    assert tool_call_evidence(tmp_path, ID) is None


def test_no_speed_file_is_no_evidence(tmp_path):
    assert tool_call_evidence(tmp_path, ID) is None


def test_a_file_that_is_not_json_is_no_evidence(tmp_path):
    p = tmp_path / "stages/7/benchmark/speed.json"
    p.parent.mkdir(parents=True)
    p.write_text("not json")
    assert tool_call_evidence(tmp_path, ID) is None


def test_the_verify_evidence_counts_too_and_names_are_sorted_and_unique(tmp_path):
    p = tmp_path / "stages/7/verify/evidence/verify.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model_id": "BAAI/AREX-2", "revision": REV, "tool_calls": [{"name": "b"}, {"name": "a"}, {"name": "a"}]}))
    assert tool_call_evidence(tmp_path, ID) == ("stages/7/verify/evidence/verify.json", ("a", "b"))


def test_a_call_without_a_name_is_not_evidence(tmp_path):
    _speed(tmp_path, [{"arguments": "{}"}])
    assert tool_call_evidence(tmp_path, ID) is None


def test_a_link_out_of_the_run_is_not_followed(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"model_id": f"BAAI/AREX-2@{REV}", "tool_calls": [{"name": "x"}]}))
    run = tmp_path / "run"
    (run / "stages/7/benchmark").mkdir(parents=True)
    (run / "stages/7/benchmark/speed.json").symlink_to(outside)
    assert tool_call_evidence(run, ID) is None


# ---- all of it -----------------------------------------------------------------------------------------------------

def _snapshot(tmp_path, readme=README, template=TEMPLATE):
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    if readme is not None:
        (snap / "README.md").write_text(readme)
    if template is not None:
        (snap / "chat_template.jinja").write_text(template)
    return snap


def test_read_use_facts_joins_the_template_the_flags_and_the_authors(tmp_path):
    u = read_use_facts(_snapshot(tmp_path), RUN_SH, tmp_path / "run")
    assert isinstance(u, UseFacts)
    assert u.authors_use.startswith("AREX-2 is intended for research")
    assert u.thinking_default is True and u.effort_values == ("xhigh", "medium", "low")
    assert u.tools_in_template and u.tool_parser == "qwen3_coder" and u.auto_tool_choice
    assert u.reasoning_parser == "qwen3" and u.text_only and u.tool_evidence is None


def test_read_use_facts_reads_the_template_from_tokenizer_config_when_there_is_no_jinja_file(tmp_path):
    snap = _snapshot(tmp_path, template=None)
    (snap / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE}))
    assert read_use_facts(snap, "CMD=(x)", tmp_path / "run").thinking_default is True


def test_a_tool_call_in_the_speed_file_is_carried_into_the_facts(tmp_path):
    run = tmp_path / "run"
    _speed(run, [{"name": "read_file"}])
    assert read_use_facts(_snapshot(tmp_path), RUN_SH, run, identity=ID).tool_evidence == ("stages/7/benchmark/speed.json", ("read_file",))


def test_a_model_with_no_readable_source_has_no_facts(tmp_path):
    assert read_use_facts(_snapshot(tmp_path, readme=None, template=None), "CMD=(x)", tmp_path / "run") is None
    assert read_use_facts(tmp_path / "missing", "", tmp_path) is None


def test_has_facts_needs_at_least_one_fact():
    assert not has_facts(UseFacts(None, None, (), None, False, False, None, None, False, False, None))
    assert has_facts(UseFacts("x", None, (), None, False, False, None, None, False, False, None))
    assert has_facts(UseFacts(None, None, (), None, False, False, None, None, False, True, None))


def test_tool_support_needs_the_template_the_parser_and_auto_choice():
    base = dict(authors_use=None, thinking_default=None, effort_values=(), effort_default=None, can_disable_thinking=False,
                tools_in_template=True, reasoning_parser=None, tool_parser="p", auto_tool_choice=True, text_only=False,
                tool_evidence=None)
    assert UseFacts(**base).tools_ok
    assert not UseFacts(**{**base, "tools_in_template": False}).tools_ok
    assert not UseFacts(**{**base, "tool_parser": None}).tools_ok
    assert not UseFacts(**{**base, "auto_tool_choice": False}).tools_ok


# ---- cases the first mutation run left alive -------------------------------------------------------------------------

def test_the_last_assignment_of_the_skip_flag_decides():
    on_then_off = 'export QWEN36_SKIP_VISION="1"\nexport QWEN36_SKIP_VISION="0"\n'
    off_then_on = 'export QWEN36_SKIP_VISION="0"\nexport QWEN36_SKIP_VISION="1"\n'
    assert flag_facts(on_then_off)["text_only"] is False
    assert flag_facts(off_then_on)["text_only"] is True


def test_a_skip_flag_value_other_than_on_is_not_text_only():
    assert flag_facts('export QWEN36_SKIP_VISION="no"')["text_only"] is False
    assert flag_facts("export QWEN36_SKIP_VISION=true")["text_only"] is True


def test_effort_values_in_double_quotes_are_read():
    t = '{%- if resolved_reasoning_effort not in ["high", "low"] %}x{%- endif %}'
    assert template_facts(t)["effort_values"] == ("high", "low")


def test_the_tool_parser_flag_is_read_with_underscores_and_with_an_equals_sign():
    assert flag_facts("CMD=(x --tool_call_parser=hermes)")["tool_parser"] == "hermes"


def test_a_tool_call_file_over_the_size_limit_is_not_read(tmp_path):
    p = tmp_path / "stages/7/benchmark/speed.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model_id": f"BAAI/AREX-2@{REV}", "tool_calls": [{"name": "x"}], "pad": "a" * (2 * 1024 * 1024)}))
    assert tool_call_evidence(tmp_path, ID) is None


def test_the_openai_shaped_tool_call_names_are_read(tmp_path):
    p = tmp_path / "stages/7/benchmark/speed.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model_id": f"BAAI/AREX-2@{REV}", "choices": [{"message": {"tool_calls": [{"function": {"name": "search"}}]}}]}))
    assert tool_call_evidence(tmp_path, ID) == ("stages/7/benchmark/speed.json", ("search",))


def test_the_default_template_of_a_list_in_tokenizer_config_is_used(tmp_path):
    snap = _snapshot(tmp_path, template=None)
    other = "{%- if enable_thinking is defined and enable_thinking is true %}x{%- endif %}"
    (snap / "tokenizer_config.json").write_text(json.dumps({"chat_template": [
        {"name": "rag", "template": other}, {"name": "default", "template": TEMPLATE}]}))
    assert read_use_facts(snap, "CMD=(x)", tmp_path / "run").thinking_default is True


def test_a_malformed_tokenizer_config_gives_no_template(tmp_path):
    snap = _snapshot(tmp_path, template=None)
    (snap / "tokenizer_config.json").write_text("{not json")
    assert read_use_facts(snap, RUN_SH, tmp_path / "run").thinking_default is None


def test_a_heading_inside_the_section_ends_it_and_a_later_section_is_not_taken():
    assert authors_intended_use("## Intended use\nFor chat.\n\n## Citation\nA paper.\n") == "For chat."


def test_html_only_lines_in_the_section_are_dropped():
    assert authors_intended_use("## Intended use\n<br>\nFor chat.\n") == "For chat."


def test_either_phrase_alone_shows_thinking_is_on_by_default():
    assert template_facts("{%- if enable_thinking is undefined or enable_thinking is true %}x{%- endif %}")["thinking_default"] is True
    assert template_facts("{%- if enable_thinking is defined and enable_thinking is false %}x{%- endif %}")["thinking_default"] is True


def test_a_default_effort_with_no_list_of_values_is_not_reported():
    f = template_facts("{{ reasoning_effort|default('high') }}")
    assert f["effort_values"] == () and f["effort_default"] is None


def test_a_bare_if_tools_shows_the_template_renders_tools():
    assert template_facts("{%- if tools %}x{%- endif %}")["tools_in_template"] is True


def test_the_auto_tool_choice_flag_is_read_with_underscores():
    assert flag_facts("CMD=(x --enable_auto_tool_choice)")["auto_tool_choice"] is True


# ---- the evidence must be about this package's model (review finding on PR 51) -------------------------------------------

def test_a_speed_report_of_another_model_is_not_evidence(tmp_path):
    _speed(tmp_path, [{"name": "run_tests"}], model_id=f"Qwen/Qwen3.8-27B@{OTHER_REV}")
    assert tool_call_evidence(tmp_path, ID) is None


def test_the_same_repo_at_another_revision_is_not_evidence(tmp_path):
    _speed(tmp_path, [{"name": "run_tests"}], model_id=f"BAAI/AREX-2@{OTHER_REV}")
    assert tool_call_evidence(tmp_path, ID) is None


def test_a_report_that_names_no_model_is_not_evidence(tmp_path):
    _speed(tmp_path, [{"name": "run_tests"}], model_id=None)
    assert tool_call_evidence(tmp_path, ID) is None


def test_a_repo_with_no_revision_in_the_report_is_not_evidence(tmp_path):
    _speed(tmp_path, [{"name": "run_tests"}], model_id="BAAI/AREX-2")
    assert tool_call_evidence(tmp_path, ID) is None


def test_no_expected_identity_means_no_evidence(tmp_path):
    _speed(tmp_path, [{"name": "run_tests"}])
    assert tool_call_evidence(tmp_path) is None
    assert tool_call_evidence(tmp_path, None) is None
    assert tool_call_evidence(tmp_path, ("BAAI/AREX-2", None)) is None


def test_a_report_with_several_models_counts_only_the_matching_entry(tmp_path):
    p = tmp_path / "stages/7/benchmark/speed.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"note": "x", "models": {
        "base": {"model_id": f"Qwen/Qwen3.8-27B@{OTHER_REV}", "prompts": {"t": {"runs": [{"tool_calls": [{"name": "base_only"}]}]}}},
        "graft": {"model_id": f"BAAI/AREX-2@{REV}", "prompts": {"t": {"runs": [{"tool_calls": [{"name": "read_file"}]}]}}}}}))
    assert tool_call_evidence(tmp_path, ID) == ("stages/7/benchmark/speed.json", ("read_file",))


def test_a_report_with_several_models_none_matching_is_not_evidence(tmp_path):
    p = tmp_path / "stages/7/benchmark/speed.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"models": {"base": {"model_id": f"Qwen/Qwen3.8-27B@{OTHER_REV}",
                                                  "prompts": {"t": {"runs": [{"tool_calls": [{"name": "x"}]}]}}}}}))
    assert tool_call_evidence(tmp_path, ID) is None


def test_a_verify_record_names_the_model_and_the_revision_apart(tmp_path):
    p = tmp_path / "stages/7/verify/evidence/verify.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model_id": "BAAI/AREX-2", "revision": OTHER_REV, "tool_calls": [{"name": "a"}]}))
    assert tool_call_evidence(tmp_path, ID) is None


def test_read_use_facts_passes_the_identity_to_the_evidence_reader(tmp_path):
    run = tmp_path / "run"
    _speed(run, [{"name": "read_file"}], model_id=f"Qwen/Qwen3.8-27B@{OTHER_REV}")
    assert read_use_facts(_snapshot(tmp_path), RUN_SH, run, identity=ID).tool_evidence is None
    assert read_use_facts(_snapshot(tmp_path), RUN_SH, run).tool_evidence is None
    _speed(run, [{"name": "read_file"}])
    assert read_use_facts(_snapshot(tmp_path), RUN_SH, run, identity=ID).tool_evidence is not None


def test_another_repo_at_the_same_revision_is_not_evidence(tmp_path):
    _speed(tmp_path, [{"name": "run_tests"}], model_id=f"Someone/Else@{REV}")
    assert tool_call_evidence(tmp_path, ID) is None


def test_an_empty_expected_revision_matches_nothing(tmp_path):
    p = tmp_path / "stages/7/verify/evidence/verify.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model_id": "BAAI/AREX-2", "revision": "", "tool_calls": [{"name": "a"}]}))
    assert tool_call_evidence(tmp_path, ("BAAI/AREX-2", "")) is None


def test_other_models_under_a_matching_top_record_do_not_lend_their_calls(tmp_path):
    p = tmp_path / "stages/7/benchmark/speed.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model_id": f"BAAI/AREX-2@{REV}", "prompts": {"t": {"runs": [{"tool_calls": [{"name": "mine"}]}]}},
                             "models": {"base": {"model_id": f"Qwen/Qwen3.8-27B@{OTHER_REV}",
                                                  "prompts": {"t": {"runs": [{"tool_calls": [{"name": "theirs"}]}]}}}}}))
    assert tool_call_evidence(tmp_path, ID) == ("stages/7/benchmark/speed.json", ("mine",))
