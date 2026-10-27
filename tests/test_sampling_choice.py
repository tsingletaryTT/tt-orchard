# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 chooses the served configuration by what the model is for (issue 107, orchard/sampling_choice.py).

The rule reads whether sampling is needed (from files), the sampling probe's verdict for the configuration stage 2 serves, the measured speed of
each configuration and a floor. These tests cover each row of the design note's table. No hardware: sides are written by hand."""
import json

import pytest

from orchard import defaults, sampling_choice as sc


def side(drafter="on", verdict="sampled-ignored", greedy=60.0, sampled=59.0, served=True):
    return {"drafter": drafter, "verdict": verdict, "greedy_tok_s": greedy, "sampled_tok_s": sampled, "served": served}


NEEDED = {"needed": True, "source": "generation_config", "override": "auto", "evidence": ["generation_config.json: temperature 0.6"]}
NOT_NEEDED = {"needed": False, "source": "none", "override": "auto", "evidence": []}


# ---- the floor --------------------------------------------------------------------------------------------------------------------

def test_the_floor_is_a_labelled_choice_below_the_slowest_measured_drafter_off_speed():
    assert defaults.SAMPLING_MIN_TOKS_PER_S == 10.0
    assert defaults.SAMPLING_MIN_TOKS_PER_S < 16.5          # the slowest drafter-off sampled speed measured on 2 chips, 2026-10-27
    src = open(defaults.__file__, encoding="utf-8").read()
    line = next(ln for ln in src.splitlines() if ln.startswith("SAMPLING_MIN_TOKS_PER_S"))
    assert "choice" in line


# ---- when the variant is needed -----------------------------------------------------------------------------------------------------

def test_the_variant_is_needed_only_when_sampling_is_needed_the_drafter_is_on_and_sampling_is_not_shown_to_work():
    assert sc.needs_variant(NEEDED, side()) is True
    assert sc.needs_variant(NOT_NEEDED, side()) is False
    assert sc.needs_variant(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=18, sampled=17)) is False
    assert sc.needs_variant(NEEDED, side(verdict="sampled-full-speed")) is False
    assert sc.needs_variant(NEEDED, side(verdict="sampled-slower")) is False
    assert sc.needs_variant(NEEDED, side(drafter="off")) is False            # nothing to switch off


@pytest.mark.parametrize("verdict", ["inconclusive", "not-run", "malformed", "greedy-refused", "sampled-refused", "sampled-incoherent"])
def test_an_unknown_or_bad_verdict_with_the_drafter_on_needs_the_variant(verdict):
    assert sc.needs_variant(NEEDED, side(verdict=verdict)) is True


# ---- sampling not needed -----------------------------------------------------------------------------------------------------------

def test_when_sampling_is_not_needed_the_served_configuration_stays_even_if_it_ignores_sampling():
    d = sc.decide(NOT_NEEDED, side())
    assert d["choice"] == "keep" and d["flag"] is None
    assert "not needed" in d["reason"]


def test_not_needed_with_no_probe_keeps_the_old_behaviour():
    d = sc.decide(NOT_NEEDED, side(verdict="not-run"))
    assert d["choice"] == "keep" and d["flag"] is None


# ---- sampling needed, the served configuration applies it ------------------------------------------------------------------------------

def test_a_drafter_off_configuration_above_the_floor_is_kept_without_a_flag():
    d = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=18.4, sampled=17.2))
    assert d["choice"] == "keep" and d["flag"] is None
    assert d["numbers"]["served_tok_s"] == pytest.approx(17.2)


def test_a_configuration_that_applies_sampling_below_the_floor_is_kept_and_flagged():
    d = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=9.0, sampled=8.0))
    assert d["choice"] == "keep" and d["flag"] == "below-floor"
    assert "8" in d["reason"] and "10" in d["reason"]


def test_a_speed_exactly_at_the_floor_clears_it():
    d = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=10.0, sampled=10.0))
    assert d["flag"] is None


def test_the_floor_reads_the_sampled_speed_and_falls_back_to_the_greedy_one():
    low_sampled = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=18.0, sampled=9.0))
    assert low_sampled["flag"] == "below-floor"
    no_sampled = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=18.0, sampled=None))
    assert no_sampled["flag"] is None and no_sampled["numbers"]["served_tok_s"] == 18.0


def test_a_sampled_slower_verdict_still_counts_as_applied():
    d = sc.decide(NEEDED, side(drafter="off", verdict="sampled-slower", greedy=30.0, sampled=12.0))
    assert d["choice"] == "keep" and d["flag"] is None


def test_a_drafter_on_configuration_that_applies_sampling_a_future_bundle_is_kept():
    d = sc.decide(NEEDED, side(verdict="sampled-full-speed", greedy=60.0, sampled=59.0))
    assert d["choice"] == "keep" and d["flag"] is None
    assert "applies sampling" in d["reason"] or "applied" in d["reason"]


def test_an_applied_configuration_with_no_measured_speed_is_flagged_unverified():
    d = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=None, sampled=None))
    assert d["choice"] == "keep" and d["flag"] == "unverified"


# ---- sampling needed, the served configuration ignores it ------------------------------------------------------------------------------

def variant(verdict="sampled-full-speed", greedy=18.4, sampled=17.2, served=True):
    return {"drafter": "off", "verdict": verdict, "greedy_tok_s": greedy, "sampled_tok_s": sampled, "served": served}


def test_ignored_sampling_with_a_variant_that_applies_it_above_the_floor_chooses_drafter_off():
    d = sc.decide(NEEDED, side(), variant())
    assert d["choice"] == "drafter-off" and d["flag"] is None
    n = d["numbers"]
    assert n["served_tok_s"] == pytest.approx(59.0) and n["variant_tok_s"] == pytest.approx(17.2)
    assert "ignored" in d["reason"] and "17.2" in d["reason"]


def test_ignored_sampling_and_a_variant_below_the_floor_keeps_the_fast_one_and_says_neither_serves_the_purpose():
    d = sc.decide(NEEDED, side(), variant(greedy=9.0, sampled=8.0))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"
    assert "neither" in d["reason"]


def test_ignored_sampling_and_a_variant_that_does_not_serve_keeps_the_fast_one_with_the_flag():
    d = sc.decide(NEEDED, side(), variant(served=False, verdict="not-run", greedy=None, sampled=None))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"


@pytest.mark.parametrize("verdict", ["sampled-ignored", "sampled-refused", "sampled-incoherent"])
def test_a_variant_that_also_fails_sampling_does_not_replace_the_served_configuration(verdict):
    d = sc.decide(NEEDED, side(), variant(verdict=verdict))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"


def test_a_variant_with_an_unknown_verdict_still_replaces_a_measured_ignoring_configuration_but_is_flagged_unverified():
    d = sc.decide(NEEDED, side(), variant(verdict="inconclusive"))
    assert d["choice"] == "drafter-off" and d["flag"] == "unverified"


def test_a_variant_with_no_measured_speed_is_chosen_unverified_when_the_served_configuration_ignores_sampling():
    d = sc.decide(NEEDED, side(), variant(greedy=None, sampled=None))
    assert d["choice"] == "drafter-off" and d["flag"] == "unverified"


def test_the_trial_not_run_keeps_the_served_configuration_and_says_the_variant_was_not_tried():
    d = sc.decide(NEEDED, side(), None, variant_note="the sampling trial was skipped: the run uses a lab")
    assert d["choice"] == "keep" and d["flag"] == "variant-not-tried"
    assert "lab" in d["reason"]


def test_without_a_drafter_to_switch_off_ignored_sampling_is_flagged_and_kept():
    d = sc.decide(NEEDED, side(drafter="off", verdict="sampled-ignored", greedy=18.0, sampled=18.0))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"


# ---- sampling needed, the probe gave no verdict ---------------------------------------------------------------------------------------

def test_no_probe_verdict_for_a_drafter_on_model_that_needs_sampling_chooses_the_safe_drafter_off_configuration_when_it_serves_and_is_fast_enough():
    d = sc.decide(NEEDED, side(verdict="not-run", greedy=None, sampled=None), variant(verdict="not-run", greedy=18.0, sampled=None))
    assert d["choice"] == "drafter-off" and d["flag"] == "unverified"


def test_no_probe_verdict_and_a_variant_that_does_not_serve_keeps_the_served_one_unverified():
    d = sc.decide(NEEDED, side(verdict="not-run", greedy=None, sampled=None), variant(served=False, verdict="not-run", greedy=None, sampled=None))
    assert d["choice"] == "keep" and d["flag"] == "unverified"


def test_no_probe_verdict_and_a_variant_below_the_floor_keeps_the_served_one_and_flags_it():
    d = sc.decide(NEEDED, side(verdict="not-run", greedy=None, sampled=None), variant(verdict="not-run", greedy=7.0, sampled=None))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"


def test_no_verdict_with_the_drafter_off_keeps_it_unverified():
    d = sc.decide(NEEDED, side(drafter="off", verdict="inconclusive", greedy=18.0, sampled=None))
    assert d["choice"] == "keep" and d["flag"] == "unverified"


def test_an_unknown_drafter_state_never_triggers_a_variant():
    assert sc.needs_variant(NEEDED, side(drafter=None)) is False


# ---- the record --------------------------------------------------------------------------------------------------------------

def test_the_decision_carries_need_numbers_thresholds_and_the_evidence_it_read():
    d = sc.decide(NEEDED, side(), variant())
    assert d["needed"] is True and d["need_source"] == "generation_config"
    assert d["need_evidence"] == ["generation_config.json: temperature 0.6"]
    assert d["thresholds"] == {"min_tok_s": defaults.SAMPLING_MIN_TOKS_PER_S}
    assert d["numbers"]["served_verdict"] == "sampled-ignored" and d["numbers"]["variant_verdict"] == "sampled-full-speed"
    assert d["numbers"]["served_drafter"] == "on"


def test_the_card_sentence_states_the_choice_and_the_reason_without_a_number_in_prose():
    """The card's Numbers table is the only place for a measured number, and the card gate refuses a speed in prose."""
    from orchard.card_use import PERF_IN_PROSE
    chosen = sc.decide(NEEDED, side(), variant())
    s = sc.card_sentence(chosen)
    assert "chose the drafter-off configuration" in s and "needs sampling" in s and "ignored" in s
    assert "generation_config.json" in s and "Numbers table" in s
    kept = sc.decide(NOT_NEEDED, side())
    assert "kept the configuration it served" in sc.card_sentence(kept) and "not needed" in sc.card_sentence(kept)
    applied = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=18, sampled=17))
    assert "applies it" in sc.card_sentence(applied)
    flagged = sc.decide(NEEDED, side(), variant(greedy=9.0, sampled=8.0))
    assert "does not do what the model needs" in sc.card_sentence(flagged)
    for d in (chosen, kept, applied, flagged, sc.decide(NEEDED, side(), None, variant_note="x")):
        assert not PERF_IN_PROSE.search(sc.card_sentence(d)), sc.card_sentence(d)


def test_each_need_source_is_named_in_plain_words_in_the_card_sentence():
    for source, words in (("operator", "operator"), ("card-warning", "warns against greedy"), ("generation_config", "generation_config.json"),
                          ("card-hints", "recommends")):
        n = {**NEEDED, "source": source}
        assert words in sc.card_sentence(sc.decide(n, side(), variant()))


def test_an_unverified_or_untried_choice_says_so_in_the_card_sentence():
    unverified = sc.decide(NEEDED, side(), variant(verdict="inconclusive"))
    assert "not fully verified" in sc.card_sentence(unverified)
    untried = sc.decide(NEEDED, side(), None, variant_note="skipped")
    assert "not fully verified" in sc.card_sentence(untried)


def test_a_below_floor_keep_says_the_configuration_decodes_below_the_floor():
    s = sc.card_sentence(sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=9.0, sampled=8.0)))
    assert "below the usability floor" in s


def test_the_marker_phrases_are_the_ones_the_card_sentence_uses():
    for dec in (sc.decide(NEEDED, side(), variant()), sc.decide(NOT_NEEDED, side())):
        assert sc.CARD_MARKERS[dec["choice"]] in sc.card_sentence(dec)


# ---- need --------------------------------------------------------------------------------------------------------------------------

def write_genconfig(run, **over):
    g = {"model_generation_config": {}, "model_generation_config_present": False, "base_generation_config": {"temperature": 0.7},
         "base_generation_config_present": True, "card_sampling_hints": {}, "card_warns_greedy": []}
    g.update(over)
    p = run / "stages" / "0" / "evidence"
    p.mkdir(parents=True, exist_ok=True)
    (p / "genconfig-license.json").write_text(json.dumps(g), encoding="utf-8")


def test_the_operator_override_wins_over_every_file(tmp_path):
    write_genconfig(tmp_path, model_generation_config={"temperature": 0.6}, model_generation_config_present=True)
    n = sc.need(tmp_path, "not-needed")
    assert n["needed"] is False and n["source"] == "operator"
    write_genconfig(tmp_path, card_warns_greedy=[])
    assert sc.need(tmp_path, "needed")["needed"] is True and sc.need(tmp_path, "needed")["source"] == "operator"


def test_an_unknown_override_is_refused(tmp_path):
    with pytest.raises(ValueError):
        sc.need(tmp_path, "maybe")


def test_a_card_that_warns_against_greedy_decoding_means_sampling_is_needed(tmp_path):
    write_genconfig(tmp_path, card_warns_greedy=[{"line": 12, "quote": "DO NOT use greedy decoding, it can lead to endless repetitions."}])
    n = sc.need(tmp_path)
    assert n["needed"] is True and n["source"] == "card-warning"
    assert "greedy" in n["evidence"][0] and "12" in n["evidence"][0]


@pytest.mark.parametrize("cfg", [{"do_sample": True}, {"temperature": 0.6}, {"top_p": 0.95}, {"top_k": 20}, {"do_sample": False, "temperature": 0.3}])
def test_the_models_own_generation_config_with_a_sampling_default_means_sampling_is_needed(tmp_path, cfg):
    write_genconfig(tmp_path, model_generation_config=cfg, model_generation_config_present=True)
    n = sc.need(tmp_path)
    assert n["needed"] is True and n["source"] == "generation_config"
    assert "generation_config.json" in n["evidence"][0]


@pytest.mark.parametrize("cfg", [{"temperature": 0}, {"top_p": 1.0}, {"top_k": 0}, {"top_k": -1}, {"eos_token_id": 1}])
def test_a_generation_config_whose_values_do_not_sample_is_not_evidence_of_need(tmp_path, cfg):
    write_genconfig(tmp_path, model_generation_config=cfg, model_generation_config_present=True)
    assert sc.need(tmp_path)["needed"] is False


def test_the_nearest_models_copied_generation_config_is_not_evidence_about_this_model(tmp_path):
    write_genconfig(tmp_path, model_generation_config_present=False, base_generation_config={"temperature": 0.7, "top_p": 0.8})
    n = sc.need(tmp_path)
    assert n["needed"] is False and n["source"] == "none"


def test_card_hints_with_a_sampling_setting_mean_sampling_is_needed(tmp_path):
    write_genconfig(tmp_path, card_sampling_hints={"temperature": [0.7], "top_p": [0.8]})
    n = sc.need(tmp_path)
    assert n["needed"] is True and n["source"] == "card-hints"
    assert "temperature" in n["evidence"][0]


def test_card_hints_that_ask_for_greedy_settings_are_not_evidence_of_need(tmp_path):
    write_genconfig(tmp_path, card_sampling_hints={"temperature": [0.0], "top_p": [1.0], "top_k": [0.0]})
    assert sc.need(tmp_path)["needed"] is False


def test_an_explicit_do_sample_false_with_no_other_evidence_says_sampling_is_not_needed(tmp_path):
    write_genconfig(tmp_path, model_generation_config={"do_sample": False}, model_generation_config_present=True)
    n = sc.need(tmp_path)
    assert n["needed"] is False and n["source"] == "generation_config"


def test_card_warning_beats_a_do_sample_false_default(tmp_path):
    write_genconfig(tmp_path, model_generation_config={"do_sample": False}, model_generation_config_present=True,
                    card_warns_greedy=[{"line": 3, "quote": "Avoid greedy decoding."}])
    assert sc.need(tmp_path)["needed"] is True


def test_no_evidence_means_not_needed_and_the_record_says_so(tmp_path):
    write_genconfig(tmp_path)
    n = sc.need(tmp_path)
    assert n["needed"] is False and n["source"] == "none" and "no evidence" in n["evidence"][0]


def test_a_missing_or_unreadable_evidence_file_means_not_needed_and_says_so(tmp_path):
    n = sc.need(tmp_path)
    assert n["needed"] is False and n["source"] == "none" and "genconfig-license.json" in n["evidence"][0]
    p = tmp_path / "stages" / "0" / "evidence"
    p.mkdir(parents=True)
    (p / "genconfig-license.json").write_text("{not json", encoding="utf-8")
    assert sc.need(tmp_path)["needed"] is False


def test_the_override_is_recorded_in_the_need(tmp_path):
    write_genconfig(tmp_path)
    assert sc.need(tmp_path, "auto")["override"] == "auto"
    assert sc.need(tmp_path, "needed")["override"] == "needed"
