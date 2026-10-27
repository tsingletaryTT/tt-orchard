# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 records the lines of a model card that warn against greedy decoding (issue 107), because a model that its authors say must not be decoded greedily
needs sampling, and a speculative drafter that verifies by argmax serves such a model badly. The card is prose, so the reader is a set of patterns and
the record quotes each line it found; `orchard/sampling_choice.py` reads the quotes."""
import pytest

from test_delta_triage_template import evidence, run_triage, set_model_card, template_module, text_case, tokenizer_json  # noqa: F401


@pytest.mark.parametrize("text", [
    "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.",
    "Please avoid greedy decoding.",
    "Never use greedy sampling with this model.",
    "Don't use greedy decoding.",
    "Greedy decoding is not recommended for this model.",
    "We do not recommend greedy decoding.",
    "We advise against greedy decoding here.",
    "Greedy decoding can lead to endless repetitions.",
    "Greedy decoding may cause repetition loops.",
    "> **Note:** do not use greedy decoding.",
    "Tips: Greedy decoding is discouraged.",
])
def test_a_line_that_warns_against_greedy_decoding_is_found(text):
    found = template_module().card_greedy_warnings("# A model\n\nsome text\n" + text + "\nmore text\n")
    assert len(found) == 1 and found[0]["line"] == 4 and "greedy" in found[0]["quote"].lower()


@pytest.mark.parametrize("text", [
    "Greedy decoding gives deterministic output.",
    "The model is evaluated with greedy decoding.",
    "We use greedy decoding for the benchmarks.",
    "We do not use greedy decoding in the evaluation.",
    "Set temperature to 0 for greedy decoding.",
    "This is a greedy algorithm for the knapsack problem.",
    "Do not use this model for medical advice.",
    "Avoid long prompts.",
    "",
])
def test_a_line_that_only_mentions_greedy_decoding_is_not_a_warning(text):
    assert template_module().card_greedy_warnings("# A model\n" + text + "\n") == []


def test_each_found_line_is_quoted_with_its_number_and_the_quote_is_bounded():
    long_line = "Do not use greedy decoding " + "x" * 600
    found = template_module().card_greedy_warnings("a\nb\n" + long_line + "\n")
    assert found[0]["line"] == 3 and len(found[0]["quote"]) <= 300


def test_at_most_three_lines_are_kept():
    text = "\n".join(f"Avoid greedy decoding ({i})." for i in range(10))
    assert [w["line"] for w in template_module().card_greedy_warnings(text)] == [1, 2, 3]


def test_only_the_part_of_the_card_the_hints_read_is_searched():
    t = template_module()
    text = "x" * t.CARD_READ_CHARS + "\nDo not use greedy decoding.\n"
    assert t.card_greedy_warnings(text) == []


def test_the_warnings_are_written_into_the_genconfig_evidence(text_case):
    stage = text_case()
    set_model_card(stage, "Do not use greedy decoding.\n")
    assert run_triage(stage).returncode == 0
    g = evidence(stage, "genconfig-license.json")
    assert len(g["card_warns_greedy"]) == 1
    assert "greedy" in g["card_warns_greedy"][0]["quote"].lower()


def test_a_card_with_no_warning_writes_an_empty_list(text_case):
    stage = text_case()
    set_model_card(stage, "Use temperature=0.6.\n")
    assert run_triage(stage).returncode == 0
    assert evidence(stage, "genconfig-license.json")["card_warns_greedy"] == []


def test_a_missing_card_writes_an_empty_list(text_case):
    import json
    from pathlib import Path
    stage = text_case()
    cfg = json.loads((stage / "triage_config.json").read_text())
    (Path(cfg["model_snapshot"]) / "README.md").unlink()
    assert run_triage(stage).returncode == 0
    assert evidence(stage, "genconfig-license.json")["card_warns_greedy"] == []
