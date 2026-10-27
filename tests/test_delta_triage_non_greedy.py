# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fourth review of PR 114, finding C (issue 107): a card that warns against non-greedy decoding recommends greedy decoding. The warning patterns must not match
`greedy` inside `non-greedy`, because a warning takes precedence over `do_sample: false` and `--sampling auto` would reverse the card's recommendation."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_delta_triage_template import template_module, tokenizer_json  # noqa: E402,F401

from orchard import sampling_choice as sc  # noqa: E402


@pytest.mark.parametrize("line", [
    "Do not use non-greedy decoding.",
    "Please avoid non-greedy sampling.",
    "- **Never use non-greedy decoding**.",
    "Non-greedy decoding is not recommended for this model.",
    "We do not recommend non-greedy decoding.",
    "We advise against non-greedy decoding here.",
    "Non-greedy decoding can lead to endless repetitions.",
    "Do not use non greedy decoding.",
    "Don't use nongreedy decoding.",
    "Avoid semi-greedy decoding.",
    "DO NOT use Non-Greedy decoding.",
])
def test_a_warning_about_non_greedy_decoding_is_not_a_warning_against_greedy_decoding(line):
    assert template_module().card_greedy_warnings("# Tips\n\n" + line + "\n") == []


@pytest.mark.parametrize("line", [
    "Do not use greedy decoding, only non-greedy.",
    "Avoid greedy decoding; non-greedy decoding is fine.",
    "Greedy decoding is not recommended, unlike non-greedy decoding.",
])
def test_a_line_that_warns_against_greedy_and_mentions_non_greedy_is_still_found(line):
    assert len(template_module().card_greedy_warnings(line + "\n")) == 1


def test_a_non_greedy_recommendation_does_not_turn_sampling_on_through_the_need(tmp_path):
    warns = template_module().card_greedy_warnings("Do not use non-greedy decoding.\n")
    g = {"model_generation_config": {"do_sample": False}, "model_generation_config_present": True, "card_sampling_hints": {}, "card_warns_greedy": warns}
    p = tmp_path / "stages" / "0" / "evidence"
    p.mkdir(parents=True)
    (p / "genconfig-license.json").write_text(json.dumps(g))
    n = sc.need(tmp_path)
    assert n["needed"] is False and n["source"] == "generation_config"
