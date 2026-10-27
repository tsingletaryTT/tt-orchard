# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The sampling probe's judge (issue 107). The swap template sends one greedy and two sampled requests and records them raw; orchard/sampling_probe.py
reads the record and says what the configuration supports. The judge recomputes every speed from tokens and seconds, so a summary field written
by anyone is never believed. The speculative-drafter log line has the exact shape of a real AREX-2 server log (2026-10-23)."""
import copy

import pytest

from orchard import sampling_probe as sp

LINE = ("[dflash2-serve] slot 0 session: 4 iters, 18 committed tokens, accept 3.75/7 -> 4.50 tok/iter, 21.3 tok/s in-model (incl. seed); "
        "histogram [0, 2, 0, 0, 0, 0, 1, 1]\n")
STORY = "The lighthouse keeper climbed the long spiral stairs every evening, counting the steps and listening to the sea below him."
GREEDY = "The lighthouse keeper climbed the stairs one more time that night and found the lamp already burning, though nobody had lit it."
OTHER = "On the island there was a bell that rang by itself whenever the fog came in, and nobody remembered who had hung it there."


def entry(text=GREEDY, tokens=96, seconds=2.0, error=None, accepted=True, log="", finish="length", params=None):
    return {"params": params or {}, "accepted": accepted, "error": error, "seconds": seconds, "completion_tokens": tokens,
            "text": text, "finish_reason": finish, "drafter_log": log}


def refused(error="HTTP 400: no"):
    return entry(text="", tokens=None, seconds=None, accepted=False, error=error)


GREEDY_PARAMS = {"temperature": 0}
SAMPLED_PARAMS = ({"temperature": 1.0, "top_p": 0.9, "seed": 1234}, {"temperature": 1.0, "top_p": 0.9, "seed": 4321})


def record(greedy=None, a=None, b=None):
    """A record shaped like the template's: the greedy request has temperature 0 and the two sampled requests carry temperature, top-p and different seeds."""
    reqs = {"greedy": greedy or entry(text=GREEDY, seconds=2.0), "sampled_a": a or entry(text=STORY, seconds=2.0),
            "sampled_b": b or entry(text=OTHER, seconds=2.0)}
    for name, params in zip(sp.ENTRY_NAMES, (GREEDY_PARAMS, *SAMPLED_PARAMS)):
        if not reqs[name]["params"]:
            reqs[name]["params"] = dict(params)
    return {"ran": True, "prompt": "p", "max_tokens": 96, "requests": reqs}


def test_a_healthy_pair_is_full_speed_and_the_speeds_are_tokens_over_seconds():
    s = sp.summarize(record(greedy=entry(seconds=2.0), a=entry(text=STORY, seconds=2.0), b=entry(text=OTHER, seconds=3.0)))
    assert s["verdict"] == "sampled-full-speed"
    assert s["greedy"]["tokens_per_s"] == pytest.approx(48.0)
    assert s["sampled"]["tokens_per_s"] == pytest.approx((48.0 + 32.0) / 2)      # the mean of the two sampled requests


def test_a_summary_written_into_the_record_is_not_believed():
    rec = record()
    rec["verdict"] = "sampled-ignored"
    rec["requests"]["greedy"]["tokens_per_s"] = 9999
    s = sp.summarize(rec)
    assert s["verdict"] == "sampled-full-speed" and s["greedy"]["tokens_per_s"] == pytest.approx(48.0)


def test_two_seeds_with_the_same_text_are_a_suspected_match_with_greedy_decoding():
    s = sp.summarize(record(a=entry(text=STORY), b=entry(text=STORY)))
    assert s["verdict"] == "sampled-matches-greedy"
    assert "greedy" in s["sentence"] and "seed" in s["sentence"]


def test_a_sampled_text_equal_to_the_greedy_text_is_a_suspected_match_even_when_the_seeds_differ_in_text():
    """Measured on the DFlash2 bundle with the drafter on (2026-10-27): every sampled request returned the greedy text."""
    for a, b in ((GREEDY, OTHER), (OTHER, GREEDY)):
        s = sp.summarize(record(a=entry(text=a), b=entry(text=b)))
        assert s["verdict"] == "sampled-matches-greedy", (a, b)
        assert "greedy text" in s["sentence"]


@pytest.mark.parametrize("which", ["greedy", "sampled_a", "sampled_b"])
@pytest.mark.parametrize("blank", ["", "   ", "\n\n", " \n \t"])
def test_a_reply_with_no_visible_text_is_inconclusive_whatever_the_token_count(which, blank):
    rec = record()
    rec["requests"][which]["text"] = blank
    s = sp.summarize(rec)
    assert s["verdict"] == "inconclusive", (which, blank, s["verdict"])
    assert "empty" in s["card_sentence"] and "empty" in s["sentence"]


def test_two_different_blank_sampled_replies_are_not_a_sampling_result():
    s = sp.summarize(record(a=entry(text="  "), b=entry(text="\n\n")))
    assert s["verdict"] == "inconclusive"


def test_two_equal_blank_sampled_replies_are_not_called_ignored_sampling():
    assert sp.summarize(record(a=entry(text="\n"), b=entry(text="\n")))["verdict"] == "inconclusive"


def test_a_blank_reply_does_not_hide_a_refusal():
    assert sp.summarize(record(a=refused("HTTP 400: no"), b=entry(text="")))["verdict"] == "sampled-refused"


@pytest.mark.parametrize("which", ["greedy", "sampled_a", "sampled_b"])
@pytest.mark.parametrize("damage", [{"completion_tokens": None}, {"seconds": None}, {"completion_tokens": 4}, {"seconds": 0}])
def test_a_missing_speed_in_any_of_the_three_requests_is_inconclusive_not_full_speed(which, damage):
    rec = record()
    rec["requests"][which].update(damage)
    s = sp.summarize(rec)
    assert s["verdict"] == "inconclusive", (which, damage, s["verdict"])
    assert s["card_sentence"] == sp.CARD_SENTENCES["inconclusive"]
    assert "speed" in s["card_sentence"]


def test_a_missing_speed_does_not_hide_a_match_with_greedy():
    rec = record(a=entry(text=GREEDY), b=entry(text=OTHER))
    rec["requests"]["greedy"]["completion_tokens"] = None
    assert sp.summarize(rec)["verdict"] == "sampled-matches-greedy"


def test_equal_seeds_texts_are_a_match_even_when_the_greedy_speed_is_missing():
    rec = record(a=entry(text=STORY), b=entry(text=STORY))
    rec["requests"]["greedy"]["completion_tokens"] = None
    assert sp.summarize(rec)["verdict"] == "sampled-matches-greedy"


def test_the_card_sentence_for_a_suspected_match_is_true_for_both_ways_it_is_found():
    """One sampled reply equal to the greedy reply and the other different is a match, and the card must not say the two sampled replies were
    identical. The sentence is the same for both branches, so it names both."""
    by_seeds = sp.summarize(record(a=entry(text=STORY), b=entry(text=STORY)))
    by_greedy = sp.summarize(record(a=entry(text=GREEDY), b=entry(text=OTHER)))
    assert by_seeds["verdict"] == by_greedy["verdict"] == "sampled-matches-greedy"
    assert by_seeds["card_sentence"] == by_greedy["card_sentence"]
    text = by_greedy["card_sentence"]
    assert "greedy text" in text and "different seeds" in text and "did not confirm" in text
    assert "two requests" not in text and "the two sampled replies" not in text


def test_the_detailed_sentence_names_the_branch_that_was_found():
    by_seeds = sp.summarize(record(a=entry(text=STORY), b=entry(text=STORY)))["sentence"]
    by_greedy = sp.summarize(record(a=entry(text=GREEDY), b=entry(text=OTHER)))["sentence"]
    assert "differ only in seed returned the same text" in by_seeds and "a sampled request returned the greedy text" not in by_seeds
    assert "a sampled request returned the greedy text" in by_greedy and "differ only in seed" not in by_greedy


HOT_PARAMS = {"temperature": 2.0, "top_p": 0.99, "seed": 2468}


def with_hot(rec, text=GREEDY, **kw):
    rec["requests"]["sampled_hot"] = entry(text=text, params=dict(HOT_PARAMS), **kw)
    return rec


def test_matching_text_alone_is_suspected_and_not_confirmed_as_ignored_sampling():
    """With top-p 0.9 and a most likely token above 0.9 at every step, one token is eligible, so correctly applied sampling also returns the greedy text."""
    for a, b in ((STORY, STORY), (GREEDY, OTHER), (OTHER, GREEDY), (GREEDY, GREEDY)):
        s = sp.summarize(record(a=entry(text=a), b=entry(text=b)))
        assert s["verdict"] == "sampled-matches-greedy", (a, b, s["verdict"])
        assert "did not confirm" in s["card_sentence"] and "did not confirm" in s["sentence"]
        assert "sampling was not applied" not in s["card_sentence"].lower().replace("not applied:", "")


def with_session_lines(rec, names=("sampled_a", "sampled_b")):
    """The drafter's own session line in the log slice of these requests: independent runtime evidence that the DFlash2 speculative loop served them."""
    for n in names:
        rec["requests"][n]["drafter_log"] = LINE
    return rec


def test_a_hot_request_that_matches_greedy_is_not_evidence_a_peaked_distribution_gives_the_same_text():
    """A logit gap of 20 at temperature 2.0 leaves the leading token a probability of about 0.99995, so top-p 0.99 keeps only that token and correctly
    applied sampling returns the greedy text at the hot temperature too."""
    for a, b in ((STORY, STORY), (GREEDY, OTHER)):
        s = sp.summarize(with_hot(record(a=entry(text=a), b=entry(text=b))))
        assert s["verdict"] == "sampled-matches-greedy", (a, b, s["verdict"])


@pytest.mark.parametrize("a, b", [(STORY, STORY), (GREEDY, OTHER), (OTHER, GREEDY), (GREEDY, GREEDY)])
def test_matching_text_with_the_drafters_session_line_on_both_sampled_requests_is_ignored_sampling(a, b):
    s = sp.summarize(with_session_lines(record(a=entry(text=a), b=entry(text=b))))
    assert s["verdict"] == "sampled-ignored", (a, b, s["verdict"])
    assert "session line" in s["sentence"] and "argmax" in s["sentence"] and "greedy" in s["sentence"]
    assert "not applied" in s["card_sentence"] and "argmax" in s["card_sentence"]


def test_the_evidence_does_not_depend_on_a_hot_request():
    with_one = sp.summarize(with_hot(with_session_lines(record(a=entry(text=STORY), b=entry(text=STORY))), text=OTHER))
    without = sp.summarize(with_session_lines(record(a=entry(text=STORY), b=entry(text=STORY))))
    assert with_one["verdict"] == without["verdict"] == "sampled-ignored"


@pytest.mark.parametrize("names", [("sampled_a",), ("sampled_b",), ()])
def test_the_session_line_on_only_some_of_the_sampled_requests_is_not_enough(names):
    s = sp.summarize(with_session_lines(record(a=entry(text=STORY), b=entry(text=STORY)), names))
    assert s["verdict"] == "sampled-matches-greedy"


def test_a_log_line_that_is_not_the_drafters_session_line_is_not_evidence():
    rec = with_session_lines(record(a=entry(text=STORY), b=entry(text=STORY)))
    rec["requests"]["sampled_b"]["drafter_log"] = "[dflash2-serve] step: rows=[0] iters=1 12.0 ms carry=[3]"
    assert sp.summarize(rec)["verdict"] == "sampled-matches-greedy"


def test_a_session_line_on_the_greedy_request_alone_is_not_evidence_about_sampled_requests():
    rec = record(greedy=entry(log=LINE), a=entry(text=STORY), b=entry(text=STORY))
    assert sp.summarize(rec)["verdict"] == "sampled-matches-greedy"


def test_the_session_lines_do_not_make_different_texts_ignored_sampling():
    assert sp.summarize(with_session_lines(record()))["verdict"] == "sampled-full-speed"


def test_a_high_temperature_request_that_differs_from_greedy_leaves_matching_text_unconfirmed():
    s = sp.summarize(with_hot(record(a=entry(text=STORY), b=entry(text=STORY)), text=OTHER))
    assert s["verdict"] == "sampled-matches-greedy"


@pytest.mark.parametrize("hot", [dict(accepted=False, error="HTTP 400: no", text=""), dict(accepted=False, error="HTTP 500: x", text=GREEDY), dict(text="")])
def test_a_refused_or_blank_high_temperature_request_confirms_nothing(hot):
    rec = record(a=entry(text=STORY), b=entry(text=STORY))
    rec["requests"]["sampled_hot"] = entry(params=dict(HOT_PARAMS), **hot)
    assert sp.summarize(rec)["verdict"] == "sampled-matches-greedy"


def test_a_high_temperature_request_does_not_change_a_verdict_when_the_replies_differ():
    s = sp.summarize(with_hot(record(), text=GREEDY))
    assert s["verdict"] == "sampled-full-speed"


@pytest.mark.parametrize("params", [{"temperature": 1.0, "top_p": 0.99, "seed": 1}, {"temperature": 2.0, "top_p": 1.0, "seed": 1},
                                    {"temperature": 2.0, "top_p": 0.99}, {"temperature": 2.0, "top_p": 0.99, "seed": 1, "prompt": "x"}, "hot", None])
def test_a_present_high_temperature_entry_is_validated(params):
    rec = with_hot(record())
    rec["requests"]["sampled_hot"]["params"] = params
    assert sp.problems(rec) and sp.summarize(rec)["verdict"] == "malformed"


def test_a_high_temperature_entry_that_is_not_an_object_never_raises():
    for bad in ("x", [], 5, None):
        rec = record()
        rec["requests"]["sampled_hot"] = bad
        assert sp.problems(rec) and sp.summarize(rec)["verdict"] == "malformed"


def test_the_high_temperature_seed_may_equal_a_sampled_seed_without_error_and_is_not_part_of_the_seed_rule():
    rec = with_hot(record())
    rec["requests"]["sampled_hot"]["params"]["seed"] = 1234
    assert sp.problems(rec) == []


def test_a_match_with_greedy_wins_over_slower():
    s = sp.summarize(record(greedy=entry(seconds=1.0), a=entry(text=STORY, seconds=9.0), b=entry(text=STORY, seconds=9.0)))
    assert s["verdict"] == "sampled-matches-greedy"
    s = sp.summarize(with_session_lines(record(greedy=entry(seconds=1.0), a=entry(text=STORY, seconds=9.0), b=entry(text=STORY, seconds=9.0))))
    assert s["verdict"] == "sampled-ignored"


def test_short_identical_replies_are_inconclusive_not_ignored():
    s = sp.summarize(record(greedy=entry(tokens=96), a=entry(text="Yes.", tokens=2), b=entry(text="Yes.", tokens=2)))
    assert s["verdict"] == "inconclusive"


def test_a_refused_sampled_request_is_reported_with_its_error():
    s = sp.summarize(record(a=refused("HTTP 400: temperature is not supported")))
    assert s["verdict"] == "sampled-refused"
    assert "temperature is not supported" in s["sentence"]


def test_one_refused_sampled_request_of_two_still_counts_as_refused():
    assert sp.summarize(record(b=refused("HTTP 500: boom")))["verdict"] == "sampled-refused"


def test_a_refused_greedy_request_is_reported_first():
    s = sp.summarize(record(greedy=refused("HTTP 500: boom"), a=refused("HTTP 500: boom")))
    assert s["verdict"] == "greedy-refused"


def test_sampled_decoding_below_the_ratio_is_slower():
    slow = round(2.0 / (sp.SLOW_RATIO - 0.05), 3)
    s = sp.summarize(record(a=entry(text=STORY, seconds=slow), b=entry(text=OTHER, seconds=slow)))
    assert s["verdict"] == "sampled-slower"
    assert s["ratio"] == pytest.approx(2.0 / slow)


def test_sampled_decoding_just_above_the_ratio_is_not_slower():
    ok = round(2.0 / (sp.SLOW_RATIO + 0.05), 3)
    assert sp.summarize(record(a=entry(text=STORY, seconds=ok), b=entry(text=OTHER, seconds=ok)))["verdict"] == "sampled-full-speed"


def test_a_looping_sampled_reply_is_incoherent():
    loop = "the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the"
    assert sp.summarize(record(a=entry(text=loop), b=entry(text=OTHER)))["verdict"] == "sampled-incoherent"


def test_drafter_acceptance_is_read_from_the_log_slice_of_each_request():
    s = sp.summarize(record(greedy=entry(log=LINE), a=entry(text=STORY, log=LINE), b=entry(text=OTHER, log="")))
    assert s["greedy"]["drafter_accept_rate"] == pytest.approx((2 + 6 + 7) / (4 * 7))
    assert s["sampled"]["drafter_accept_rate"] == pytest.approx((2 + 6 + 7) / (4 * 7))     # the one sampled request that has a line
    assert sp.summarize(record())["greedy"]["drafter_accept_rate"] is None


def test_no_record_is_not_run_and_has_no_reasons():
    s = sp.summarize(None)
    assert s["verdict"] == "not-run" and sp.problems(None) == []
    assert sp.summarize({"ran": False})["verdict"] == "not-run"


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: r["requests"].pop("greedy"), "greedy"),
    (lambda r: r["requests"]["sampled_a"].__setitem__("seconds", "fast"), "seconds"),
    (lambda r: r["requests"]["sampled_a"].__setitem__("seconds", float("nan")), "seconds"),
    (lambda r: r["requests"]["sampled_b"].__setitem__("completion_tokens", True), "completion_tokens"),
    (lambda r: r["requests"]["greedy"].__setitem__("accepted", "yes"), "accepted"),
    (lambda r: r["requests"]["greedy"].__setitem__("text", 7), "text"),
    (lambda r: r.__setitem__("requests", []), "requests"),
])
def test_a_malformed_record_gives_a_reason(mutate, fragment):
    rec = copy.deepcopy(record())
    mutate(rec)
    reasons = sp.problems(rec)
    assert reasons and any(fragment in r for r in reasons)


@pytest.mark.parametrize("key, bad", [("drafter_log", 7), ("drafter_log", ["x"]), ("error", 7), ("error", {"a": 1}), ("finish_reason", 3)])
def test_a_field_that_must_be_a_string_or_null_gives_a_reason_and_no_crash(key, bad):
    rec = record(a=refused("HTTP 400: no"))
    rec["requests"]["sampled_a"][key] = bad
    reasons = sp.problems(rec)
    assert any(key in r for r in reasons), reasons
    assert sp.summarize(rec)["verdict"] == "malformed"


def test_string_and_null_values_of_those_fields_are_fine():
    rec = record()
    for n in sp.ENTRY_NAMES:
        rec["requests"][n].update({"drafter_log": None, "error": None, "finish_reason": None})
    assert sp.problems(rec) == []


def damaged(name, params):
    rec = record()
    if params is _MISSING:
        del rec["requests"][name]["params"]
    else:
        rec["requests"][name]["params"] = params
    return rec


_MISSING = object()


@pytest.mark.parametrize("name, params", [
    ("greedy", {"temperature": 0.7}), ("greedy", {"temperature": 0, "seed": 1}), ("greedy", {"temperature": "0"}), ("greedy", {}),
    ("greedy", _MISSING), ("greedy", "greedy"), ("greedy", {"temperature": True}),
    ("sampled_a", {"temperature": 0, "seed": 1234}), ("sampled_a", {"temperature": 0, "top_p": 0.9, "seed": 1234}),
    ("sampled_a", {"temperature": float("inf"), "top_p": 0.9, "seed": 1234}), ("sampled_a", {"temperature": 1.0, "seed": 1234}),
    ("sampled_a", {"temperature": 1.0, "top_p": 1.0, "seed": 1234}), ("sampled_a", {"temperature": 1.0, "top_p": 0.9}),
    ("sampled_a", {"temperature": 1.0, "top_p": 0.9, "seed": "1234"}), ("sampled_a", {"temperature": 1.0, "top_p": 0.9, "seed": 1.5}),
    ("sampled_a", {"temperature": 1.0, "top_p": 0.9, "seed": True}), ("sampled_a", {"temperature": -1.0, "top_p": 0.9, "seed": 1}),
    ("sampled_a", {"temperature": 1.0, "top_p": 0.0, "seed": 1}), ("sampled_b", _MISSING), ("sampled_b", {}),
    ("sampled_b", {"temperature": float("nan"), "top_p": 0.9, "seed": 4321}),
    ("sampled_b", {"temperature": 1.0, "top_p": 0.9, "seed": 1234}),                 # the same seed as sampled_a
])
def test_the_recorded_request_parameters_must_say_what_the_probe_claims(name, params):
    rec = damaged(name, params)
    reasons = sp.problems(rec)
    assert reasons and any("params" in r or "seed" in r for r in reasons), (name, params, reasons)
    assert sp.summarize(rec)["verdict"] == "malformed"


BAD_ENTRIES = ["invalid", [], [1], 5, 1.5, None, True]


@pytest.mark.parametrize("name", ["greedy", "sampled_a", "sampled_b"])
@pytest.mark.parametrize("bad", BAD_ENTRIES)
def test_a_request_entry_that_is_not_an_object_gives_reasons_and_never_raises(name, bad):
    rec = record()
    rec["requests"][name] = bad
    reasons = sp.problems(rec)
    assert any(name in r for r in reasons), reasons
    s = sp.summarize(rec)
    assert s["verdict"] == "malformed" and s["card_sentence"] is None


@pytest.mark.parametrize("bad", BAD_ENTRIES)
def test_a_params_value_that_is_not_an_object_never_raises_in_the_seed_check(bad):
    rec = record()
    rec["requests"]["sampled_a"]["params"] = bad
    rec["requests"]["sampled_b"]["params"] = bad
    assert any("params" in r for r in sp.problems(rec))


def test_both_sampled_entries_broken_in_different_ways_still_reports_both():
    rec = record()
    rec["requests"]["sampled_a"] = "invalid"
    rec["requests"]["sampled_b"] = None
    reasons = sp.problems(rec)
    assert any("sampled_a" in r for r in reasons) and any("sampled_b" in r for r in reasons)


@pytest.mark.parametrize("name, params", [
    ("sampled_b", {"temperature": 0.5, "top_p": 0.9, "seed": 4321}),                              # another temperature
    ("sampled_b", {"temperature": 1.0, "top_p": 0.8, "seed": 4321}),                              # another top-p
    ("sampled_b", {"temperature": 1.0, "top_p": 0.9, "seed": 4321, "prompt": "something else"}),   # an extra key
    ("sampled_a", {"temperature": 1.0, "top_p": 0.9, "seed": 1234, "max_tokens": 1}),
    ("sampled_a", {"temperature": 1.0, "top_p": 0.9, "seed": 1234, "top_k": 5}),
    ("sampled_a", {"temperature": 1.0, "top_p": 0.9}),                                            # no seed
])
def test_the_two_sampled_requests_may_differ_only_in_seed_and_carry_no_extra_keys(name, params):
    rec = record()
    rec["requests"][name]["params"] = params
    reasons = sp.problems(rec)
    assert any("params" in r or "seed" in r for r in reasons), reasons
    assert sp.summarize(rec)["verdict"] == "malformed"


@pytest.mark.parametrize("extra", [{"prompt": "x"}, {"top_k": 5}, {"max_tokens": 1}, {"presence_penalty": 0.5}])
def test_an_extra_key_present_in_both_sampled_requests_is_still_refused(extra):
    rec = record()
    for n, seed in (("sampled_a", 1234), ("sampled_b", 4321)):
        rec["requests"][n]["params"] = {"temperature": 1.0, "top_p": 0.9, "seed": seed, **extra}
    reasons = sp.problems(rec)
    assert any(next(iter(extra)) in r for r in reasons), reasons


def test_both_sampled_requests_with_the_same_other_parameters_and_different_seeds_are_fine():
    rec = record()
    for n, seed in (("sampled_a", 7), ("sampled_b", 8)):
        rec["requests"][n]["params"] = {"temperature": 0.7, "top_p": 0.5, "seed": seed}
    assert sp.problems(rec) == []


@pytest.mark.parametrize("bad", [96.5, -1, -0.5, 96.0, True, "96", float("nan")])
def test_a_completion_token_count_must_be_a_non_negative_integer_or_null(bad):
    rec = record()
    rec["requests"]["sampled_a"]["completion_tokens"] = bad
    assert any("completion_tokens" in r for r in sp.problems(rec)), bad
    assert sp.summarize(rec)["verdict"] == "malformed"


@pytest.mark.parametrize("ok", [0, 1, 96, None])
def test_a_zero_positive_or_null_token_count_is_well_formed(ok):
    rec = record()
    rec["requests"]["sampled_a"]["completion_tokens"] = ok
    assert not any("completion_tokens" in r for r in sp.problems(rec))


def test_the_refusal_sentence_reports_the_settings_the_refused_request_carried():
    rec = record(a=refused("HTTP 400: no"))
    rec["requests"]["sampled_a"]["params"] = {"temperature": 0.7, "top_p": 0.5, "seed": 1234}
    rec["requests"]["sampled_b"]["params"] = {"temperature": 0.7, "top_p": 0.5, "seed": 4321}
    text = sp.summarize(rec)["sentence"]
    assert "temperature 0.7" in text and "top-p 0.5" in text and "seed 1234" in text
    assert "temperature 1.0" not in text and "top-p 0.9" not in text


def test_the_refusal_sentence_names_the_request_that_was_refused_when_only_the_second_was():
    rec = record(b=refused("HTTP 400: no"))
    text = sp.summarize(rec)["sentence"]
    assert "seed 4321" in text and "seed 1234" not in text


def test_the_real_probe_parameters_are_valid():
    assert sp.problems(record()) == []


def test_all_three_requests_at_temperature_zero_with_different_texts_are_not_a_sampling_result():
    rec = record()
    for n in sp.ENTRY_NAMES:
        rec["requests"][n]["params"] = {"temperature": 0}
    assert sp.summarize(rec)["verdict"] == "malformed"


@pytest.mark.parametrize("ran", [0, [], "false", "true", "", None, 1])
def test_a_ran_marker_that_is_not_a_boolean_is_malformed_and_never_skipped(ran):
    rec = record()
    rec["ran"] = ran
    assert any("ran" in r for r in sp.problems(rec)), ran
    s = sp.summarize(rec)
    assert s["verdict"] == "malformed" and s["card_sentence"] is None


def test_a_present_record_with_no_ran_marker_is_malformed():
    rec = record()
    del rec["ran"]
    assert sp.summarize(rec)["verdict"] == "malformed"
    assert sp.problems({}) and sp.summarize({})["verdict"] == "malformed"


def test_only_an_explicit_false_skips_a_present_record():
    assert sp.problems({"ran": False}) == [] and sp.summarize({"ran": False})["verdict"] == "not-run"


def test_a_malformed_record_is_not_summarised_as_a_verdict():
    rec = record()
    rec["requests"]["greedy"]["seconds"] = "fast"
    assert sp.summarize(rec)["verdict"] == "malformed"


def test_a_healthy_record_has_no_reasons():
    assert sp.problems(record()) == []


def test_a_refused_request_may_have_no_seconds_or_tokens():
    assert sp.problems(record(a=refused("HTTP 400"))) == []


def test_every_sentence_is_one_plain_line_without_the_forbidden_framing():
    sentences = [sp.summarize(record())["sentence"],
                 sp.summarize(record(a=entry(text=STORY), b=entry(text=STORY)))["sentence"],
                 sp.summarize(record(a=refused()))["sentence"],
                 sp.summarize(record(a=entry(text=STORY, seconds=9.0), b=entry(text=OTHER, seconds=9.0)))["sentence"]]
    for text in sentences:
        assert text and text == " ".join(text.split())
        for bad in (", not ", "honest", "reads as"):
            assert bad not in text


# ---- the card's sentence: the same verdict with no number in it (the card's Numbers table is the only place for one) -----------------

def card_sentences():
    return {
        "full": sp.summarize(record()),
        "ignored": sp.summarize(with_session_lines(record(a=entry(text=STORY), b=entry(text=STORY)))),
        "matches": sp.summarize(record(a=entry(text=STORY), b=entry(text=STORY))),
        "refused": sp.summarize(record(a=refused("HTTP 400: no sampling here"))),
        "slower": sp.summarize(record(a=entry(text=STORY, seconds=9.0), b=entry(text=OTHER, seconds=9.0))),
        "incoherent": sp.summarize(record(a=entry(text="the " * 40), b=entry(text=OTHER))),
        "short": sp.summarize(record(a=entry(text="Yes.", tokens=2), b=entry(text="No.", tokens=2))),
        "greedy": sp.summarize(record(greedy=refused("HTTP 500: boom"))),
    }


def test_every_verdict_has_a_card_sentence_with_no_performance_number():
    from orchard.card_use import PERF_IN_PROSE
    for name, s in card_sentences().items():
        text = s["card_sentence"]
        assert text and text == " ".join(text.split()), name
        assert not PERF_IN_PROSE.search(text), (name, text)


def test_the_card_sentences_state_what_the_verdict_means():
    s = card_sentences()
    assert "different text" in s["full"]["card_sentence"]
    assert "not applied" in s["ignored"]["card_sentence"] and "greedy" in s["ignored"]["card_sentence"]
    assert "did not confirm" in s["matches"]["card_sentence"] and s["matches"]["card_sentence"] != s["ignored"]["card_sentence"]
    assert "refused" in s["refused"]["card_sentence"]
    assert "more slowly" in s["slower"]["card_sentence"]
    assert "repeats itself" in s["incoherent"]["card_sentence"]


def test_the_card_sentence_of_a_refusal_does_not_copy_the_error_text():
    assert "no sampling here" not in card_sentences()["refused"]["card_sentence"]     # the error text is in the evidence file


def test_a_run_with_no_record_has_no_card_sentence():
    assert sp.summarize(None).get("card_sentence") is None


def test_a_greedy_reply_of_a_few_tokens_has_no_speed():
    assert sp.summarize(record(greedy=entry(tokens=4, seconds=0.001)))["greedy"]["tokens_per_s"] is None
