# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package card of a vocabulary trim (issue 56).

The first package of ProprietaryLegal/Qwen3.8-27B-CJK-Trim (stage 7 on the finished run, 2026-10-25: boot check 30 of 32 tokens, gate ok)
said "Speculative decoding is off. The drafter of Qwen/Qwen3.8-27B needs the MTP tensors of the model, and the model has none." That is
false for this model, which ships 15 mtp.* tensors. The drafter is off because the vocabulary differs. The card now says so, gives both
sizes and says the token ids are renumbered, and the gate refuses the old sentence on a trim."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchard.package_card import card_problems, render_card  # noqa: E402
from test_package_card import facts, run_with_evidence  # noqa: E402

TRIM = (181760, 248320)


def trim_card(**over):
    return render_card(facts(license_id="apache-2.0", drafter=None, drafter_off=True, host_sampling=True, vocab=TRIM, **over))


def flat(card):
    return " ".join(card.split())


def test_the_card_gives_the_vocabulary_as_the_reason_the_drafter_is_off():
    c = flat(trim_card())
    assert "Speculative decoding is off." in c
    assert "181,760" in c and "248,320" in c
    assert "works on the token ids of Qwen/Qwen3.8-27B" in c
    assert "has none" not in c and "needs the MTP tensors of the model" not in c


def test_the_card_says_the_token_ids_are_renumbered_and_what_the_rows_are():
    c = flat(trim_card())
    assert "token ids are renumbered" in c
    assert "every row of its embedding and lm_head is a row of Qwen/Qwen3.8-27B's" in c


def test_the_limitations_say_the_ids_differ_from_the_nearest_models():
    limits = trim_card().split("## Limitations", 1)[1].split("## Risks", 1)[0]
    assert "token ids" in limits and "Qwen/Qwen3.8-27B" in limits


def test_a_model_without_a_vocabulary_record_keeps_the_mtp_sentence():
    c = flat(render_card(facts(license_id="apache-2.0", drafter=None, drafter_off=True)))
    assert "needs the MTP tensors of the model, and Altworld/Hemmingway-1 has none" in c
    assert "token ids are renumbered" not in c


def test_a_trim_card_passes_its_own_check(tmp_path):
    run = run_with_evidence(tmp_path)
    assert card_problems(trim_card(), license_id="apache-2.0", run_dir=run, vocab=TRIM, model_id="Altworld/Hemmingway-1",
                         nearest_model="Qwen/Qwen3.8-27B") == []


def test_the_check_refuses_the_old_mtp_sentence_on_a_trim(tmp_path):
    run = run_with_evidence(tmp_path)
    old = render_card(facts(license_id="apache-2.0", drafter=None, drafter_off=True, host_sampling=True))
    problems = card_problems(old, license_id="apache-2.0", run_dir=run, vocab=TRIM, model_id="Altworld/Hemmingway-1",
                             nearest_model="Qwen/Qwen3.8-27B")
    assert any("vocabulary" in p and "181,760" in p for p in problems), problems
    assert any("says the model has no mtp.* tensors" in p and "vocabulary differs" in p for p in problems), problems


def test_the_check_wants_each_size_the_ids_sentence_and_the_limitation(tmp_path):
    run = tmp_path / "x"
    run = run_with_evidence(tmp_path)
    card = trim_card()
    for cut, word in (("181,760", "181,760"), ("248,320", "248,320"), ("token ids are renumbered", "renumbered"),
                      ("Token ids differ", "Limitations")):
        problems = card_problems(card.replace(cut, "REDACTED"), license_id="apache-2.0", run_dir=run, vocab=TRIM,
                                 model_id="Altworld/Hemmingway-1", nearest_model="Qwen/Qwen3.8-27B")
        assert any(word in p for p in problems), (cut, problems)


def test_no_vocabulary_asks_nothing_extra(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts(license_id="apache-2.0"))
    assert card_problems(card, license_id="apache-2.0", run_dir=run) == []
