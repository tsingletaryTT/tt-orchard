# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package card and its license (plan 5, stage 7)."""
import json

import pytest

from orchard.package_card import (CardFacts, Number, Reported, card_problems, non_commercial, read_license,
                                  render_card, sampling_disclosures)
from orchard.package_card import REPORTED_HEAD

REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf"


def run_with_evidence(tmp_path):
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    (run / "stages" / "2" / "result.json").write_text(json.dumps({"top1_agreement": 0.94,
                                                                  "server_ready_s": 280.5}))
    (run / "stages" / "2" / "evidence").mkdir()
    (run / "stages" / "2" / "evidence" / "swap-check.json").write_text("{}")
    return run


def facts(**over):
    base = dict(
        name="hemmingway-1-p300", namespace="episod", model_id="Altworld/Hemmingway-1",
        revision=REV, nearest_model="Qwen/Qwen3.8-27B", source_name="qwen3.8-27b-dflash2-p300",
        license_id="cc-by-nc-4.0", chips=2, mesh="P150x2", arch="blackhole", max_model_len=262144,
        max_num_seqs=4, drafter="incoai/Qwen3.8-27B-DFlash2", verified=True,
        numbers=(Number("top1 agreement with the CPU reference (2 chips, stage 2)", 0.94, "fraction",
                        "measured", ("stages/2/result.json", "stages/2/evidence/swap-check.json")),
                 Number("time to first token", None, "ms", "TODO", ())),
        not_measured=("the drafter's acceptance rate on this model",))
    base.update(over)
    return CardFacts(**base)


def test_a_rendered_card_passes_its_own_check(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts())
    assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=run) == []
    assert card.startswith("---\nlicense: cc-by-nc-4.0\nbase_model: Altworld/Hemmingway-1\n")
    for tag in ("tt-model-cache", "blackhole", "vllm", "thin", "p150x2"):
        assert f"\n- {tag}\n" in card
    assert "tt-model serve episod/hemmingway-1-p300" in card
    assert "| time to first token | TODO | TODO | - |" in card


def test_a_non_commercial_license_is_shown_and_says_non_commercial(tmp_path):
    card = render_card(facts())
    assert "CC BY-NC 4.0" in card and "Non-commercial use only." in card
    assert "Commercial use is not permitted by that license." in card


def test_a_permissive_license_card_has_no_non_commercial_line(tmp_path):
    card = render_card(facts(license_id="apache-2.0"))
    assert "Apache 2.0" in card and "Non-commercial" not in card
    assert card_problems(card, license_id="apache-2.0", run_dir=run_with_evidence(tmp_path)) == []


@pytest.mark.parametrize("claim", [
    "Commercial use is permitted.", "Ready for commercial deployments.", "Use it commercially.",
    "Production-ready for your customers.", "Licensed for enterprise use."])
def test_a_card_that_implies_commercial_use_of_a_non_commercial_model_is_refused(tmp_path, claim):
    card = render_card(facts()).replace("## What it runs", claim + "\n\n## What it runs")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path))
    assert any("implies commercial use" in p for p in problems), problems


def test_a_card_that_lost_its_license_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts())
    assert any("license" in p for p in card_problems(card.replace("license: cc-by-nc-4.0\n", ""),
                                                     license_id="cc-by-nc-4.0", run_dir=run))
    no_line = card.replace("Non-commercial use only. ", "")
    assert any("Non-commercial use only." in p
               for p in card_problems(no_line, license_id="cc-by-nc-4.0", run_dir=run))
    other = render_card(facts(license_id="apache-2.0"))
    assert any("apache-2.0" in p and "cc-by-nc-4.0" in p
               for p in card_problems(other, license_id="cc-by-nc-4.0", run_dir=run))


def test_a_measured_number_needs_its_evidence_and_its_value_in_the_first_file(tmp_path):
    run = run_with_evidence(tmp_path)
    missing = facts(numbers=(Number("ready", 280.5, "s", "measured", ("stages/2/nope.json",)),))
    assert any("stages/2/nope.json" in p for p in
               card_problems(render_card(missing), license_id="cc-by-nc-4.0", run_dir=run))
    wrong = facts(numbers=(Number("ready", 99.0, "s", "measured", ("stages/2/result.json",)),))
    assert any("99.0" in p for p in
               card_problems(render_card(wrong), license_id="cc-by-nc-4.0", run_dir=run))


def test_a_number_in_the_prose_is_refused(tmp_path):
    card = render_card(facts()).replace("## How to serve", "It decodes at 80 tok/s.\n\n## How to serve")
    assert any("80 tok/s" in p for p in
               card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path)))


def test_a_row_labelled_measured_by_hand_without_a_value_is_refused(tmp_path):
    card = render_card(facts()).replace("| time to first token | TODO | TODO | - |",
                                        "| time to first token | fast | measured | - |")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path))
    assert any("time to first token" in p for p in problems), problems


def test_the_license_is_read_from_the_snapshot_card(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: cc-by-nc-4.0\nbase_model:\n- Qwen/Qwen3.8-27B\n"
                                        "---\n# Hemmingway-1\n")
    assert read_license(tmp_path) == "cc-by-nc-4.0"
    (tmp_path / "README.md").write_text("---\nlicense: 'Apache-2.0'\n---\n")
    assert read_license(tmp_path) == "apache-2.0"
    (tmp_path / "README.md").write_text("# no front matter\nlicense: mit\n")
    assert read_license(tmp_path) is None
    (tmp_path / "README.md").unlink()
    assert read_license(tmp_path) is None


@pytest.mark.parametrize("license_id, nc", [("cc-by-nc-4.0", True), ("cc-by-nc-sa-4.0", True),
                                            ("cc-by-nc-nd-4.0", True), ("apache-2.0", False),
                                            ("mit", False), ("cc-by-4.0", False), ("other", True)])
def test_non_commercial_licenses_are_recognised(license_id, nc):
    assert non_commercial(license_id) is nc


def test_the_card_has_the_sections_the_catalog_standard_names(tmp_path):
    card = render_card(facts(sidecars=("joint_head.safetensors",), drafter_off=True))
    for head in ("## Intended use", "## Expected performance", "## Limitations",
                 "## Risks and safety considerations"):
        assert head in card, head
    limits = card.split("## Limitations")[1].split("\n## ")[0]
    assert "joint_head.safetensors" in limits and "speculative decoding is off" in limits.lower()
    assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path)) == []


@pytest.mark.parametrize("head", ["## Intended use", "## Limitations", "## Risks and safety considerations"])
def test_a_card_without_a_required_section_is_a_problem(tmp_path, head):
    card = render_card(facts()).replace(head, "## Other")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path))
    assert any(head in p for p in problems), problems


# The card says which files the package takes from the nearest model. A model that ships no
# generation_config.json is served with the nearest model's stop tokens and sampling defaults (BAAI/AREX-2).

BASE_FILES = ("config.json", "preprocessor_config.json")


def test_the_card_lists_the_files_it_takes_from_the_nearest_model():
    card = render_card(facts(base_files=BASE_FILES))
    assert "(`config.json`, `preprocessor_config.json`, shipped in `base_config/`)" in card


def test_a_card_for_a_model_with_no_generation_config_says_whose_sampling_defaults_it_serves():
    card = render_card(facts(base_files=(*BASE_FILES, "generation_config.json"), sampling_from_base=True))
    assert "ships no generation_config.json" in card
    assert "stop tokens and the sampling defaults" in card
    assert "those of Qwen/Qwen3.8-27B" in card
    assert "Default sampling settings are those of Qwen/Qwen3.8-27B" in card.split("## Limitations")[1].split("##")[0]


def test_a_card_for_a_model_that_ships_its_own_generation_config_does_not_say_so():
    card = render_card(facts(base_files=BASE_FILES))
    assert "ships no generation_config.json" not in card
    assert "Default sampling settings" not in card


def test_the_check_accepts_the_rendered_card_for_both_cases(tmp_path):
    run = run_with_evidence(tmp_path)
    own = render_card(facts(base_files=BASE_FILES))
    assert card_problems(own, license_id="cc-by-nc-4.0", run_dir=run, base_files=BASE_FILES) == []
    names = (*BASE_FILES, "generation_config.json")
    borrowed = render_card(facts(base_files=names, sampling_from_base=True))
    f = facts(base_files=names, sampling_from_base=True)
    assert card_problems(borrowed, license_id="cc-by-nc-4.0", run_dir=run, base_files=names,
                         model_id=f.model_id, nearest_model=f.nearest_model) == []


def test_a_borrowed_generation_config_needs_both_sentences_naming_the_nearest_model(tmp_path):
    run = run_with_evidence(tmp_path)
    names = (*BASE_FILES, "generation_config.json")
    f = facts(base_files=names, sampling_from_base=True)
    card = render_card(f)
    def problems(c, **kw):
        return card_problems(c, license_id="cc-by-nc-4.0", run_dir=run, base_files=names,
                             **{"model_id": f.model_id, "nearest_model": f.nearest_model, **kw})
    first, second = sampling_disclosures(f.model_id, f.nearest_model)
    assert problems(card.replace(first, f"{f.model_id} ships no generation_config.json."))
    assert problems(card.replace(second, ""))
    assert problems(card.replace(f"those of {f.nearest_model}", "those of other/model"))
    assert problems(card, model_id="", nearest_model="")        # the gate must say whose defaults it expects
    assert "those of Qwen/Qwen3.8-27B." in card                  # the rendered text itself names the model
    assert problems(card.replace(", so the stop tokens", ",\nso the   stop tokens")) == []     # a re-wrapped card


def test_a_card_that_leaves_out_the_borrowed_sampling_defaults_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    names = (*BASE_FILES, "generation_config.json")
    card = render_card(facts(base_files=BASE_FILES))                   # rendered as if the model shipped its own
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, base_files=names)
    assert any("generation_config.json" in p and "sampling defaults" in p for p in problems)


def test_a_card_that_does_not_name_a_base_file_the_package_ships_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts(base_files=("config.json",)))
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, base_files=BASE_FILES)
    assert any("preprocessor_config.json" in p for p in problems)


def test_a_file_name_inside_other_text_does_not_count_as_named(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts(base_files=("config.json",))) + "\nsee my_preprocessor_config.json.bak\n"
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, base_files=BASE_FILES)
    assert any("preprocessor_config.json" in p for p in problems)


# ---- numbers the authors reported, kept apart from the ones measured here (#19) ---------------------------------

REPORTED = (Reported("Frontier-CS (agent track)", "70.7", "188 tasks"),
            Reported("MLE-Lite", "81.8", "Any Medal, mean of 3 seeds"),
            Reported("DeepSearchQA", "93.8%", ""))
SOURCE = "the model's card on Hugging Face"


def reported_card(**over):
    return render_card(facts(reported=REPORTED, reported_source=SOURCE, **over))


def test_a_card_with_reported_scores_has_a_section_that_says_they_were_not_measured_here():
    card = reported_card()
    sec = card.split("## Reported by the authors")[1].split("\n## ")[0]
    assert f"These scores come from {SOURCE}." in sec and "were not measured here" in sec
    assert "| Frontier-CS (agent track) | 70.7 | 188 tasks |" in sec
    assert "| DeepSearchQA | 93.8% |  |" in sec


def test_the_reported_section_comes_after_the_measured_numbers_and_before_the_limitations():
    card = reported_card()
    assert card.index("## Expected performance") < card.index("## Reported by the authors") < card.index("## Limitations")


def test_a_card_without_reported_scores_has_no_such_section():
    assert "Reported by the authors" not in render_card(facts())


def test_a_card_with_reported_scores_passes_its_own_check(tmp_path):
    run = run_with_evidence(tmp_path)
    assert card_problems(reported_card(), license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED, reported_source=SOURCE) == []


def test_a_percent_in_the_reported_section_is_not_a_number_outside_the_numbers_table(tmp_path):
    run = run_with_evidence(tmp_path)
    problems = card_problems(reported_card(), license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED, reported_source=SOURCE)
    assert not any("outside the Numbers table" in p for p in problems)


def test_a_percent_in_prose_outside_the_reported_section_is_still_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = reported_card().replace("## Limitations", "## Limitations\n\nIt is 93.8% good.\n")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED, reported_source=SOURCE)
    assert any("outside the Numbers table" in p and "93.8%" in p for p in problems)


@pytest.mark.parametrize("cut, word", [
    ("## Reported by the authors", "section"),
    ("| MLE-Lite | 81.8 | Any Medal, mean of 3 seeds |", "MLE-Lite"),
    ("were not measured here", "not measured here"),
    ("| DeepSearchQA | 93.8% |  |", "DeepSearchQA"),
])
def test_a_card_that_leaves_reported_scores_or_the_disclaimer_out_is_refused(tmp_path, cut, word):
    run = run_with_evidence(tmp_path)
    card = reported_card()
    assert cut in card
    problems = card_problems(card.replace(cut, "x"), license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED, reported_source=SOURCE)
    assert any(word in p for p in problems), problems


def test_a_card_that_changes_a_reported_value_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = reported_card().replace("| MLE-Lite | 81.8 |", "| MLE-Lite | 91.8 |")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED, reported_source=SOURCE)
    assert any("MLE-Lite" in p and "81.8" in p for p in problems)


def test_a_reported_score_listed_as_measured_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    measured = Number("MLE-Lite", 0.5, "fraction", "measured", ("stages/2/result.json",))
    card = render_card(facts(reported=REPORTED, reported_source=SOURCE,
                             numbers=(*facts().numbers, measured)))
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED, reported_source=SOURCE)
    assert any("MLE-Lite" in p and "measured" in p for p in problems)


def test_a_cell_cannot_break_the_table_or_the_card():
    rows = (Reported("a | b", "1\n2", "x\r\n## Heading in a note"),)
    card = render_card(facts(reported=rows, reported_source=SOURCE))
    sec = card.split("## Reported by the authors")[1].split("\n## ")[0]
    table = [ln for ln in sec.splitlines() if ln.startswith("| a")]
    assert len(table) == 1 and table[0].count("|") == 4 and "## Heading" in table[0]
    assert "\n## Heading" not in card


def test_a_reported_section_on_a_card_whose_run_has_no_reported_scores_is_refused(tmp_path):
    """Adding the heading and a claim to a card must not pass because the run had nothing to report."""
    run = run_with_evidence(tmp_path)
    card = render_card(facts()).replace("## Limitations", f"{REPORTED_HEAD}\n\nIt decodes at 80 tok/s.\n\n## Limitations")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run)
    assert any(REPORTED_HEAD in x and "reported-results.json" in x for x in problems), problems
    assert any("80 tok/s" in x for x in problems), problems       # and the claim is no longer exempt from the number check


def _reported_problems(card, tmp_path, **over):
    kw = {"reported": REPORTED, "reported_source": SOURCE, **over}
    return card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path), **kw)


def test_a_card_that_deletes_the_source_sentence_is_refused(tmp_path):
    card = reported_card()
    cut = f"These scores come from {SOURCE}. "
    assert cut in card
    problems = _reported_problems(card.replace(cut, ""), tmp_path)
    assert any("source" in p and SOURCE in p for p in problems), problems


def test_a_card_that_changes_the_source_is_refused(tmp_path):
    problems = _reported_problems(reported_card().replace(SOURCE, "a blog post"), tmp_path)
    assert any("source" in p and SOURCE in p for p in problems), problems


def test_a_gate_that_expects_another_source_than_the_card_cites_refuses_the_card(tmp_path):
    problems = _reported_problems(reported_card(), tmp_path, reported_source="the paper")
    assert any("source" in p and "the paper" in p for p in problems), problems


def test_a_card_with_no_source_cites_the_model_authors_and_passes_when_the_gate_has_none(tmp_path):
    card = render_card(facts(reported=REPORTED))
    assert "These scores come from the model authors." in card
    assert _reported_problems(card, tmp_path, reported_source="") == []


@pytest.mark.parametrize("old, new", [
    ("Any Medal, mean of 3 seeds", "Any Medal, best of 3 seeds"),
    ("Any Medal, mean of 3 seeds", ""),
    ("188 tasks", "190 tasks"),
])
def test_a_card_that_changes_or_drops_a_reported_note_is_refused(tmp_path, old, new):
    card = reported_card()
    assert old in card
    problems = _reported_problems(card.replace(old, new), tmp_path)
    assert any("note" in p for p in problems), problems


def test_a_card_that_adds_a_note_to_a_row_that_had_none_is_refused(tmp_path):
    card = reported_card().replace("| DeepSearchQA | 93.8% |  |", "| DeepSearchQA | 93.8% | with search |")
    assert any("DeepSearchQA" in p for p in _reported_problems(card, tmp_path))


def test_a_card_that_adds_a_row_or_a_line_to_the_reported_section_is_refused(tmp_path):
    card = reported_card()
    extra_row = card.replace("| DeepSearchQA | 93.8% |  |", "| DeepSearchQA | 93.8% |  |\n| Invented | 99 | |")
    assert any("Invented" in p for p in _reported_problems(extra_row, tmp_path / "row"))
    extra_line = card.replace("\n| Benchmark |", "\nIt is the best model.\n| Benchmark |")
    assert any("best model" in p for p in _reported_problems(extra_line, tmp_path / "line"))


def test_a_reflowed_reported_section_still_passes(tmp_path):
    card = reported_card().replace("They were not", "They\n  were   not")
    assert _reported_problems(card, tmp_path) == [], card


def test_a_line_after_the_reported_table_is_refused(tmp_path):
    card = reported_card().replace("| DeepSearchQA | 93.8% |  |", "| DeepSearchQA | 93.8% |  |\n\nIt beats every rival.")
    assert any("beats every rival" in p for p in _reported_problems(card, tmp_path))


def test_a_reported_table_with_other_spacing_still_passes(tmp_path):
    card = reported_card().replace("| MLE-Lite | 81.8 | Any Medal", "|   MLE-Lite  |  81.8   |  Any   Medal")
    assert _reported_problems(card, tmp_path) == []


# ---- a specific "Intended use" and "How to call it", from facts (card_use.py) ------------------------------------

from orchard.card_use import UNTESTED_SENTENCE, UseFacts


def use_facts(**over):
    base = dict(authors_use="It is intended for research on long-horizon agents and tool-augmented deep research.",
                thinking_default=True, effort_values=("xhigh", "medium", "low"), effort_default="xhigh",
                can_disable_thinking=True, tools_in_template=True, reasoning_parser="qwen3", tool_parser="qwen3_coder",
                auto_tool_choice=True, text_only=True, tool_evidence=None)
    base.update(over)
    return UseFacts(**base)


def cutoff_number(value=3):
    return Number("GSM8K replies cut off at the token limit, this model", value, "items", "measured",
                  ("stages/2/result.json",))


def section(card, head):
    after = card.split(head + "\n", 1)[1]
    return after.split("\n## ", 1)[0]


def test_a_card_with_no_use_facts_keeps_the_generic_sentence_and_has_no_how_to_call_section():
    card = render_card(facts())
    assert "Text generation with Altworld/Hemmingway-1 through the OpenAI-compatible server" in card
    assert "## How to call it" not in card


def test_a_card_with_use_facts_does_not_say_text_generation_in_general():
    assert "Text generation with" not in render_card(facts(use=use_facts()))


def test_the_authors_words_are_quoted_and_labelled_as_theirs_and_not_measured():
    text = section(render_card(facts(use=use_facts())), "## Intended use")
    assert "> It is intended for research on long-horizon agents and tool-augmented deep research." in text
    assert "their words" in text and "not measured here" in text
    assert "Altworld/Hemmingway-1" in text.split(">")[0]


def test_without_the_authors_words_there_is_no_quote():
    text = section(render_card(facts(use=use_facts(authors_use=None))), "## Intended use")
    assert "\n> " not in text and "their words" not in text


def test_a_line_break_or_heading_in_the_authors_words_cannot_start_a_section():
    card = render_card(facts(use=use_facts(authors_use="Chat.\n## Fake heading\nMore.")))
    assert "\n## Fake heading" not in card


def test_thinking_on_by_default_is_said_with_the_reasoning_field():
    text = section(render_card(facts(use=use_facts())), "## Intended use")
    assert "Thinking is on by default" in text and "`reasoning_content`" in text and "`content`" in text


def test_without_a_reasoning_parser_the_card_does_not_name_reasoning_content():
    card = render_card(facts(use=use_facts(reasoning_parser=None)))
    assert "reasoning_content" not in card and "Thinking is on by default" in card


def test_thinking_off_by_default_is_said():
    text = section(render_card(facts(use=use_facts(thinking_default=False))), "## Intended use")
    assert "Thinking is off by default" in text and "on by default" not in text


def test_a_template_that_does_not_say_whether_it_thinks_gets_no_thinking_sentence():
    text = section(render_card(facts(use=use_facts(thinking_default=None, effort_values=(), can_disable_thinking=False,
                                                    reasoning_parser=None))), "## Intended use")
    assert "Thinking is" not in text and "enable_thinking" not in text


def test_the_reasoning_effort_values_are_listed_in_the_templates_order_with_the_default():
    text = section(render_card(facts(use=use_facts())), "## Intended use")
    assert "`xhigh`, `medium`, `low`" in text and "default `xhigh`" in text and "`chat_template_kwargs`" in text


def test_no_effort_values_no_effort_sentence():
    assert "reasoning_effort" not in render_card(facts(use=use_facts(effort_values=(), effort_default=None)))


def test_the_thinking_switch_is_named_only_when_the_template_has_one():
    assert '"enable_thinking": false' in render_card(facts(use=use_facts()))
    assert "enable_thinking" not in render_card(facts(use=use_facts(can_disable_thinking=False)))


def test_tool_calls_without_evidence_are_marked_as_untested():
    line = [ln for ln in render_card(facts(use=use_facts())).splitlines() if "`tool_calls`" in ln][0]
    assert UNTESTED_SENTENCE in line and "qwen3_coder" in line and "prompts returned tool calls" not in line


def test_tool_calls_with_evidence_name_the_file_and_the_tools_and_are_not_marked_untested():
    u = use_facts(tool_evidence=("stages/7/benchmark/speed.json", ("read_file", "run_tests")))
    line = [ln for ln in render_card(facts(use=u)).splitlines() if "`tool_calls`" in ln][0]
    assert "prompts returned tool calls" in line and "`read_file`, `run_tests`" in line
    assert "`stages/7/benchmark/speed.json`" in line and UNTESTED_SENTENCE not in line


def test_a_package_that_cannot_return_tool_calls_does_not_mention_them():
    for over in ({"tools_in_template": False}, {"tool_parser": None}, {"auto_tool_choice": False}):
        card = render_card(facts(use=use_facts(**over)))
        assert "tool_calls" not in card and '"tools"' not in card


def test_text_only_is_said_when_run_sh_skips_vision():
    text = section(render_card(facts(use=use_facts())), "## Intended use")
    assert "Text only" in text and "images" in text and UNTESTED_SENTENCE in text


def test_a_package_that_does_not_skip_vision_does_not_say_text_only():
    assert "Text only" not in render_card(facts(use=use_facts(text_only=False)))


def test_a_cut_off_warning_needs_thinking_a_reasoning_field_and_measured_cut_offs():
    marker = "reaches `max_tokens` while the model is still thinking"
    with_cut = facts(use=use_facts(), numbers=(cutoff_number(3),))
    assert marker in render_card(with_cut)
    assert marker not in render_card(facts(use=use_facts(), numbers=(cutoff_number(0),)))
    assert marker not in render_card(facts(use=use_facts(reasoning_parser=None), numbers=(cutoff_number(3),)))
    assert marker not in render_card(facts(use=use_facts(thinking_default=False), numbers=(cutoff_number(3),)))
    assert marker not in render_card(facts(use=use_facts()))


def test_how_to_call_it_sits_between_intended_use_and_the_numbers():
    card = render_card(facts(use=use_facts()))
    assert card.index("## Intended use") < card.index("## How to call it") < card.index("## Expected performance")


def test_how_to_call_it_reads_the_model_name_from_the_server():
    text = section(render_card(facts(use=use_facts())), "## How to call it")
    assert "/v1/models" in text and "/v1/chat/completions" in text and "--port" in text and '"$MODEL"' in text


def test_the_example_has_a_generic_tool_when_the_server_returns_tool_calls():
    text = section(render_card(facts(use=use_facts())), "## How to call it")
    assert '"tools"' in text and '"run_tests"' in text and '"type": "function"' in text


def test_the_example_has_no_tool_when_it_cannot_return_tool_calls():
    text = section(render_card(facts(use=use_facts(tool_parser=None))), "## How to call it")
    assert '"tools"' not in text and '"messages"' in text


def test_a_thinking_model_gets_a_large_max_tokens_in_the_example_and_others_a_small_one():
    assert '"max_tokens": 32768' in render_card(facts(use=use_facts()))
    assert '"max_tokens": 1024' in render_card(facts(use=use_facts(thinking_default=False, tool_parser=None)))


def test_the_sidecar_out_of_scope_line_stays_with_use_facts():
    text = section(render_card(facts(use=use_facts(), sidecars=("joint_head.safetensors",))), "## Intended use")
    assert "Out of scope: the sidecar head" in text


def test_a_cut_off_count_that_is_not_measured_does_not_trigger_the_warning():
    todo = Number("GSM8K replies cut off at the token limit, this model", 3, "items", "TODO", ())
    assert "reaches `max_tokens`" not in render_card(facts(use=use_facts(), numbers=(todo,)))


def test_the_thinking_off_switch_is_named_only_for_a_model_that_thinks_by_default():
    for default in (False, None):
        card = render_card(facts(use=use_facts(thinking_default=default, can_disable_thinking=True)))
        assert "enable_thinking" not in card


def test_the_example_without_tools_asks_a_plain_question_and_not_the_tool_one():
    text = section(render_card(facts(use=use_facts(tool_parser=None))), "## How to call it")
    assert "mutex" in text and "Run the tests" not in text
    assert "Run the tests" in section(render_card(facts(use=use_facts())), "## How to call it")


# ---- the check: the section must exist and must not claim what the sources contradict ---------------------------

def use_problems(card, use, tmp_path, **over):
    run = tmp_path / "run"
    if not run.exists():
        run_with_evidence(tmp_path)
    return card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, use=use, **over)


def test_a_rendered_card_with_use_facts_passes_its_own_check(tmp_path):
    for u in (use_facts(), use_facts(tool_evidence=("stages/7/benchmark/speed.json", ("run_tests",))),
              use_facts(authors_use=None, thinking_default=False, effort_values=(), tool_parser=None, text_only=False)):
        assert use_problems(render_card(facts(use=u, numbers=(cutoff_number(2),))), u, tmp_path) == []


def test_a_card_with_no_use_facts_is_not_asked_for_the_sections(tmp_path):
    assert use_problems(render_card(facts()), None, tmp_path) == []


def test_a_card_that_dropped_how_to_call_it_is_refused(tmp_path):
    u = use_facts()
    card = render_card(facts(use=u))
    head, rest = card.split("## How to call it", 1)
    card = head + "## Expected performance" + rest.split("## Expected performance", 1)[1]
    assert any("How to call it" in p for p in use_problems(card, u, tmp_path))


@pytest.mark.parametrize("cut, word", [
    ("Thinking is on by default", "thinking"),
    ("`reasoning_effort`", "reasoning effort"),
    ("`reasoning_content`", "reasoning_content"),
    ("`xhigh`, `medium`, `low`", "reasoning effort"),
    ('`{"enable_thinking": false}`', "enable_thinking"),
    ("`tool_calls`", "tool_calls"),
    ("Text only", "text only"),
    (UNTESTED_SENTENCE, "not tested"),
    ("> It is intended for research on long-horizon agents and tool-augmented deep research.", "authors"),
    ("reaches `max_tokens` while the model is still thinking", "cut off")])
def test_a_card_that_leaves_out_a_fact_the_sources_give_is_refused(tmp_path, cut, word):
    u = use_facts()
    card = render_card(facts(use=u, numbers=(cutoff_number(2),)))
    assert cut in card
    problems = use_problems(card.replace(cut, "xx"), u, tmp_path)
    assert any(word in p.lower() for p in problems), problems


def test_a_card_that_changes_the_authors_words_is_refused(tmp_path):
    u = use_facts()
    card = render_card(facts(use=u)).replace("long-horizon agents", "short-horizon agents")
    assert any("authors" in p for p in use_problems(card, u, tmp_path))


def test_the_authors_words_may_be_rewrapped(tmp_path):
    u = use_facts(authors_use="One two three four five six seven eight.")
    card = render_card(facts(use=u)).replace("One two three four", "One two\n> three four")
    assert use_problems(card, u, tmp_path) == []


def test_a_card_that_says_thinking_is_off_when_the_template_turns_it_on_is_refused(tmp_path):
    card = render_card(facts(use=use_facts(thinking_default=False)))
    assert any("says thinking is off by default" in p for p in use_problems(card, use_facts(thinking_default=True), tmp_path))


def test_a_card_that_says_thinking_is_on_when_the_template_turns_it_off_is_refused(tmp_path):
    card = render_card(facts(use=use_facts(thinking_default=True)))
    assert any("says thinking is on by default" in p for p in use_problems(card, use_facts(thinking_default=False), tmp_path))


def test_a_card_that_names_reasoning_content_without_a_reasoning_parser_is_refused(tmp_path):
    card = render_card(facts(use=use_facts()))
    assert any("reasoning_content" in p for p in use_problems(card, use_facts(reasoning_parser=None), tmp_path))


def test_a_card_that_names_effort_values_the_template_does_not_have_is_refused(tmp_path):
    card = render_card(facts(use=use_facts()))
    assert any("reasoning_effort" in p for p in use_problems(card, use_facts(effort_values=(), effort_default=None), tmp_path))


def test_a_card_that_names_the_thinking_switch_the_template_lacks_is_refused(tmp_path):
    card = render_card(facts(use=use_facts()))
    assert any("enable_thinking" in p for p in use_problems(card, use_facts(can_disable_thinking=False), tmp_path))


@pytest.mark.parametrize("over", [{"tools_in_template": False}, {"tool_parser": None}, {"auto_tool_choice": False}])
def test_a_card_that_claims_tool_calls_the_server_cannot_return_is_refused(tmp_path, over):
    card = render_card(facts(use=use_facts()))
    assert any("tool" in p for p in use_problems(card, use_facts(**over), tmp_path))


def test_a_card_that_says_the_prompts_returned_tool_calls_without_evidence_is_refused(tmp_path):
    card = render_card(facts(use=use_facts(tool_evidence=("stages/7/benchmark/speed.json", ("run_tests",)))))
    assert any("returned tool calls" in p for p in use_problems(card, use_facts(), tmp_path))


def test_a_card_that_names_a_tool_the_evidence_does_not_hold_is_refused(tmp_path):
    card = render_card(facts(use=use_facts(tool_evidence=("stages/7/benchmark/speed.json", ("run_tests", "invented")))))
    u = use_facts(tool_evidence=("stages/7/benchmark/speed.json", ("run_tests",)))
    assert any("invented" in p for p in use_problems(card, u, tmp_path))


def test_a_card_with_evidence_that_still_says_untested_for_tool_calls_is_refused(tmp_path):
    card = render_card(facts(use=use_facts()))
    u = use_facts(tool_evidence=("stages/7/benchmark/speed.json", ("run_tests",)))
    assert any("returned tool calls" in p for p in use_problems(card, u, tmp_path))


def test_a_card_that_claims_text_only_for_a_package_that_does_not_skip_vision_is_refused(tmp_path):
    card = render_card(facts(use=use_facts(text_only=True)))
    assert any("text only" in p.lower() for p in use_problems(card, use_facts(text_only=False), tmp_path))


@pytest.mark.parametrize("claim", ["The package accepts images.", "It supports video input.", "Handles vision tasks.",
                                   "Send an image_url in the message.", "Takes videos as input."])
def test_a_card_that_claims_vision_for_a_text_only_package_is_refused(tmp_path, claim):
    u = use_facts()
    card = render_card(facts(use=u)).replace("## How to call it", claim + "\n\n## How to call it")
    assert any("vision" in p or "image" in p for p in use_problems(card, u, tmp_path)), claim


def test_a_card_that_says_it_does_not_take_images_is_not_a_vision_claim(tmp_path):
    u = use_facts()
    assert use_problems(render_card(facts(use=u)), u, tmp_path) == []


def test_a_file_name_with_video_in_it_is_not_a_vision_claim(tmp_path):
    u = use_facts()
    card = render_card(facts(use=u, base_files=("config.json", "video_preprocessor_config.json")))
    assert use_problems(card, u, tmp_path, base_files=("config.json", "video_preprocessor_config.json")) == []


def test_a_cut_off_warning_for_a_model_that_does_not_think_is_refused(tmp_path):
    card = render_card(facts(use=use_facts(), numbers=(cutoff_number(2),)))
    assert any("max_tokens" in p for p in use_problems(card, use_facts(thinking_default=False), tmp_path))


def test_a_card_whose_use_sections_hold_a_number_with_a_unit_is_still_refused(tmp_path):
    u = use_facts()
    card = render_card(facts(use=u)).replace("## How to call it", "It decodes at 50 tokens/s.\n\n## How to call it")
    assert any("number outside the Numbers table" in p for p in use_problems(card, u, tmp_path))


def test_a_card_that_states_a_thinking_default_the_template_does_not_show_is_refused(tmp_path):
    u = use_facts(thinking_default=None, effort_values=(), can_disable_thinking=False)
    card = render_card(facts(use=use_facts()))
    assert any("says thinking is on by default" in p for p in use_problems(card, u, tmp_path))


def test_a_card_that_leaves_out_thinking_off_by_default_is_refused(tmp_path):
    u = use_facts(thinking_default=False, can_disable_thinking=False, effort_values=(), reasoning_parser=None, text_only=False,
                  tool_parser=None)
    card = render_card(facts(use=u)).replace("Thinking is off by default", "xx")
    assert any("'Thinking is off by default'" in p for p in use_problems(card, u, tmp_path))


def test_a_card_that_claims_tool_calls_in_words_alone_or_in_the_example_alone_is_refused(tmp_path):
    u = use_facts(tool_parser=None)
    base = render_card(facts(use=use_facts()))
    words_only = base.replace('"tools"', '"xtools"')
    example_only = base.replace("`tool_calls`", "xcalls")
    for card in (words_only, example_only):
        assert any("claims tool calls" in p for p in use_problems(card, u, tmp_path))


@pytest.mark.parametrize("old, new", [("returned tool calls", "gave back nothing"),
                                      ("stages/7/benchmark/speed.json", "somewhere"),
                                      ("`run_tests`", "`xx`")])
def test_a_card_that_leaves_out_the_file_the_tools_or_the_sentence_the_evidence_gives_is_refused(tmp_path, old, new):
    u = use_facts(tool_evidence=("stages/7/benchmark/speed.json", ("run_tests",)))
    card = render_card(facts(use=u))
    bad = card.replace("tool-use prompts returned tool calls (`run_tests`)", "tool-use prompts returned tool calls (`run_tests`)")
    bad = bad.replace(old, new)
    assert any("returned tool calls" in p for p in use_problems(bad, u, tmp_path)), old


def only(**kw):
    base = dict(authors_use=None, thinking_default=None, effort_values=(), effort_default=None, can_disable_thinking=False,
                tools_in_template=False, reasoning_parser=None, tool_parser=None, auto_tool_choice=False, text_only=False,
                tool_evidence=None)
    base.update(kw)
    return UseFacts(**base)


@pytest.mark.parametrize("u", [only(thinking_default=True), only(thinking_default=False), only(text_only=True),
                               only(tools_in_template=True, tool_parser="p", auto_tool_choice=True),
                               only(effort_values=("low",), effort_default="low")])
def test_each_fact_taken_from_the_template_or_run_sh_needs_the_not_tested_note(tmp_path, u):
    card = render_card(facts(use=u))
    assert UNTESTED_SENTENCE in card and use_problems(card, u, tmp_path) == []
    bad = card.replace(UNTESTED_SENTENCE, "xx")
    assert any("not tested here" in p for p in use_problems(bad, u, tmp_path))


def test_a_zero_cut_off_count_needs_no_warning(tmp_path):
    u = use_facts()
    assert use_problems(render_card(facts(use=u, numbers=(cutoff_number(0),))), u, tmp_path) == []


def test_a_cut_off_row_that_is_not_measured_needs_no_warning(tmp_path):
    u = use_facts()
    card = render_card(facts(use=u, numbers=(cutoff_number(0),)))
    card = card.replace("| 0 items | measured |", "| 2 items | TODO |")
    assert "| 2 items | TODO |" in card
    assert not any("must warn" in p for p in use_problems(card, u, tmp_path))


def test_a_cut_off_warning_needs_a_reasoning_parser_in_the_check_too(tmp_path):
    u = use_facts(reasoning_parser=None)
    assert use_problems(render_card(facts(use=u, numbers=(cutoff_number(2),))), u, tmp_path) == []


# ---- the evidence file the tool-call sentence cites ships with the package -----------------------------------------

def test_the_file_the_tool_call_sentence_cites_is_listed_with_the_numbers_evidence():
    from orchard.package_card import card_evidence_paths
    u = use_facts(tool_evidence=("stages/7/benchmark/speed.json", ("run_tests",)))
    paths = card_evidence_paths(render_card(facts(use=u)))
    assert paths[-1] == "stages/7/benchmark/speed.json" and "stages/2/result.json" in paths


def test_a_file_cited_by_the_numbers_and_the_tool_call_sentence_is_listed_once():
    from orchard.package_card import card_evidence_paths
    u = use_facts(tool_evidence=("stages/2/result.json", ("run_tests",)))
    assert card_evidence_paths(render_card(facts(use=u))).count("stages/2/result.json") == 1


def test_a_card_without_the_tool_call_sentence_cites_only_its_numbers():
    from orchard.package_card import card_evidence_paths
    assert card_evidence_paths(render_card(facts(use=use_facts()))) == ["stages/2/result.json", "stages/2/evidence/swap-check.json"]


def test_a_tool_call_sentence_outside_intended_use_does_not_add_evidence():
    from orchard.package_card import card_evidence_paths
    card = render_card(facts(use=use_facts())).replace(
        "## Limitations", "## Limitations\n\nIn the runs recorded in `stages/9/other.json` the tool-use prompts returned tool calls.\n")
    assert "stages/9/other.json" not in card_evidence_paths(card)
