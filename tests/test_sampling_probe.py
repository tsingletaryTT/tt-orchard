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


def record(greedy=None, a=None, b=None):
    return {"ran": True, "prompt": "p", "max_tokens": 96,
            "requests": {"greedy": greedy or entry(text=GREEDY, seconds=2.0),
                         "sampled_a": a or entry(text=STORY, seconds=2.0),
                         "sampled_b": b or entry(text=OTHER, seconds=2.0)}}


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


def test_two_seeds_with_the_same_text_mean_sampling_was_ignored():
    s = sp.summarize(record(a=entry(text=STORY), b=entry(text=STORY)))
    assert s["verdict"] == "sampled-ignored"
    assert "greedy" in s["sentence"] and "seed" in s["sentence"]


def test_a_sampled_text_equal_to_the_greedy_text_means_sampling_was_ignored_even_when_the_seeds_differ_in_text():
    """Measured on the DFlash2 bundle with the drafter on (2026-10-27): every sampled request returned the greedy text."""
    for a, b in ((GREEDY, OTHER), (OTHER, GREEDY)):
        s = sp.summarize(record(a=entry(text=a), b=entry(text=b)))
        assert s["verdict"] == "sampled-ignored", (a, b)
        assert "greedy text" in s["sentence"]


def test_ignored_wins_over_slower():
    s = sp.summarize(record(greedy=entry(seconds=1.0), a=entry(text=STORY, seconds=9.0), b=entry(text=STORY, seconds=9.0)))
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
        "ignored": sp.summarize(record(a=entry(text=STORY), b=entry(text=STORY))),
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
    assert "refused" in s["refused"]["card_sentence"]
    assert "more slowly" in s["slower"]["card_sentence"]
    assert "repeats itself" in s["incoherent"]["card_sentence"]


def test_the_card_sentence_of_a_refusal_does_not_copy_the_error_text():
    assert "no sampling here" not in card_sentences()["refused"]["card_sentence"]     # the error text is in the evidence file


def test_a_run_with_no_record_has_no_card_sentence():
    assert sp.summarize(None).get("card_sentence") is None


def test_a_greedy_reply_of_a_few_tokens_has_no_speed():
    assert sp.summarize(record(greedy=entry(tokens=4, seconds=0.001)))["greedy"]["tokens_per_s"] is None
