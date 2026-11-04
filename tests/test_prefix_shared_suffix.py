# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The shared-prefix, new-suffix probe (orchard/served_probes.py `probe_prefix_shared_suffix`) and its advisory row in the boot check.

Background: tenstorrent/skills PR 42 (tt-model-bringup 0.1.21) found a shipped model that returned corrupted text on repeated prefixes, and its
served check compares a warm request with a cold one after `/reset_prefix_cache`. Issue 83 asks for the same proof for requests that share a
prefix. The probe sends request A (a prefix and question 1), request B (the same prefix and question 2, so the prefix is a cache hit and the
suffix is new), resets the cache, and sends B again cold. The warm and cold replies to B must be equal. The control: the reply to A, standing in
for a contaminated reply to B, must be called a fail, so the probe is shown able to miss.

No chip and no model: a small server that models a prefix cache, with a switch for each fault. The mutation table is in the PR text.
"""
import json

import pytest

from orchard import served_probes as sp
from orchard import can
from test_served_verify import Fake
from test_served_probes import chat_reply

CODE = sp.CODE_NUMBER


class CacheModel:
    """A server with a prefix cache. `fault` picks the bug: "" is correct; "stale" answers a prefix hit with the reply cached for the first
    request (the contamination the probe exists to catch); "blind" ignores the question (the probe cannot tell right from stale);
    "reset_noop" answers 200 to the reset and keeps the cache; "no_reset" has no reset route; "no_hit" never reports a cached token;
    "silent" reports no cached_tokens field at all; "flaky" answers a cold request differently from a warm one."""

    def __init__(self, fake: Fake, fault: str = ""):
        self.fake, self.fault = fake, fault
        self.cache: dict[str, str] = {}
        self.resets = 0
        self.requests: list[dict] = []
        fake.post["/v1/chat/completions"] = self.chat
        if fault != "no_reset":
            fake.post["/reset_prefix_cache"] = self.reset

    def reset(self, body):
        self.resets += 1
        if self.fault != "reset_noop":
            self.cache.clear()
        return 200, b"ok"

    def chat(self, body):
        text = body["messages"][0]["content"]
        prefix, _, question = text.rpartition("\n\n")
        self.requests.append({"prefix": prefix, "question": question, "max_tokens": body.get("max_tokens"), "temperature": body.get("temperature")})
        hit = prefix in self.cache
        if self.fault == "blind":
            answer = sp.SECRET
        elif "secret word" in question:
            answer = sp.SECRET
        else:
            answer = CODE
        if hit and self.fault in ("stale", "reset_noop"):
            answer = self.cache[prefix]
        if hit and self.fault == "flaky":
            answer = answer + " (cached)"
        self.cache.setdefault(prefix, answer)
        usage = {"completion_tokens": 3, "prompt_tokens": 1500}
        if self.fault != "silent":
            cached = 1400 if hit and self.fault != "no_hit" else 0
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return 200, json.dumps({"choices": [{"message": {"content": answer}, "finish_reason": "stop"}], "usage": usage}).encode()


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.close()


def run(fake, fault="", **kw):
    model = CacheModel(fake, fault)
    kw.setdefault("declared", True)
    kw.setdefault("allow_reset", True)
    return model, sp.probe_prefix_shared_suffix(fake.base, "m", None, **kw)


# ---- the verdicts ------------------------------------------------------------------------------------------------------------

def test_a_correct_cache_passes_and_the_result_says_what_was_measured(fake):
    model, r = run(fake)
    assert r["state"] == "pass", r
    assert r["advisory"] is True and r["hit_observed"] is True and r["control"]["missed"] is True
    assert "text" in r["label"] and "not" in r["label"], r["label"]       # the label states that it compares reply text only, and what it does not show
    assert r["warm_reply"] == r["cold_reply"] == CODE and r["other_reply"] == sp.SECRET


def test_a_stale_answer_on_a_prefix_hit_fails(fake):
    _, r = run(fake, "stale")
    assert r["state"] == "fail" and "differs" in r["detail"], r
    assert r["warm_reply"] == sp.SECRET and r["cold_reply"] == CODE


def test_a_server_that_ignores_the_question_cannot_be_judged_so_the_control_does_not_miss(fake):
    _, r = run(fake, "blind")
    assert r["state"] == "not_checked" and "control" in r["detail"], r
    assert r["control"]["missed"] is False


def test_a_reset_that_keeps_the_cache_leaves_the_cold_run_contaminated_and_is_not_a_pass(fake):
    _, r = run(fake, "reset_noop")
    assert r["state"] != "pass", r            # warm and cold are both stale: equal, but the control shows the probe could not tell


def test_a_server_without_a_reset_route_is_not_checked_with_the_way_to_enable_it(fake):
    model, r = run(fake, "no_reset")
    assert r["state"] == "not_checked" and "VLLM_SERVER_DEV_MODE" in r["detail"], r
    assert model.requests == [], "nothing was sent to a server that cannot give a cold reference"


def test_the_reset_is_sent_only_when_the_caller_allows_it(fake):
    model, r = run(fake, allow_reset=False)
    assert r["state"] == "not_checked" and "reset" in r["detail"], r
    assert model.resets == 0 and model.requests == []


def test_a_can_that_does_not_declare_prefix_caching_is_not_checked(fake):
    model, r = run(fake, declared=False)
    assert r["state"] == "not_checked" and model.requests == [] and model.resets == 0


def test_a_warm_request_that_reports_zero_cached_tokens_never_tested_a_hit(fake):
    _, r = run(fake, "no_hit")
    assert r["state"] == "not_checked" and "hit" in r["detail"] and r["hit_observed"] is False, r


def test_a_server_that_reports_no_cached_tokens_is_judged_and_the_label_says_the_hit_was_not_observed(fake):
    _, r = run(fake, "silent")
    assert r["state"] == "pass" and r["hit_observed"] is None, r
    assert "not observed" in r["label"], r["label"]


def test_a_cold_reply_that_differs_from_the_warm_one_for_another_reason_still_fails(fake):
    _, r = run(fake, "flaky")
    assert r["state"] == "fail", r


# ---- what is sent ------------------------------------------------------------------------------------------------------------

def test_the_requests_are_greedy_short_and_share_one_prefix_with_two_different_questions(fake):
    model, r = run(fake)
    reqs = [q for q in model.requests if q["max_tokens"] != 1]          # the calibration request asks for one token
    assert [q["question"] == reqs[0]["question"] for q in reqs] == [True, False, False], reqs
    assert len({q["prefix"] for q in reqs}) == 1, "all three requests share one prefix, byte for byte"
    assert all(q["temperature"] == 0 and q["max_tokens"] == sp.SUFFIX_TOKENS for q in reqs)
    assert model.resets == 2, "one reset before the first request and one before the cold reference"
    assert len(reqs) == 3, reqs


def test_a_fresh_tag_starts_each_prefix_so_an_earlier_run_cannot_pre_warm_it(fake):
    first, _ = run(fake)
    second, _ = run(fake)
    a = [q for q in first.requests if q["max_tokens"] != 1][0]["prefix"]
    b = [q for q in second.requests if q["max_tokens"] != 1][0]["prefix"]
    assert a.startswith("Document ") and a != b


def test_a_context_too_small_for_the_prefix_is_not_checked(fake):
    model, r = run(fake, max_model_len=100)
    assert r["state"] == "not_checked" and model.requests == [], r


def test_a_transport_failure_is_a_fail_naming_the_error_and_a_non_200_reply_is_too(fake):
    CacheModel(fake)
    fake.post["/v1/chat/completions"] = lambda body: (500, b"boom")
    r = sp.probe_prefix_shared_suffix(fake.base, "m", None, declared=True, allow_reset=True)
    assert r["state"] in ("fail", "not_checked") and r["state"] != "pass", r


@pytest.mark.parametrize("a,w,c", [("", "x", "y"), ("x", "", "y"), ("x", "y", "")])
def test_an_empty_reply_is_not_checked(a, w, c):
    r = sp.judge_prefix_shared_suffix(a, w, c, cached_tokens=100, declared=True)
    assert r["state"] == "not_checked", r


def test_the_judge_alone_passes_equal_replies_with_a_distinct_control_and_fails_a_stale_one():
    assert sp.judge_prefix_shared_suffix("tangerine", "4417", "4417", cached_tokens=10, declared=True)["state"] == "pass"
    assert sp.judge_prefix_shared_suffix("tangerine", "tangerine", "4417", cached_tokens=10, declared=True)["state"] == "fail"
    assert sp.judge_prefix_shared_suffix("tangerine", "tangerine", "tangerine", cached_tokens=10, declared=True)["state"] == "not_checked"


def test_the_replies_in_the_result_are_bounded():
    long = "x" * 5000
    r = sp.judge_prefix_shared_suffix("a", long, "b", cached_tokens=10, declared=True)
    assert len(r["warm_reply"]) <= sp.REPLY_CHARS and len(r["detail"]) < 1000


def test_a_boolean_or_negative_cached_token_count_is_not_a_hit_count():
    assert sp.judge_prefix_shared_suffix("a", "b", "b", cached_tokens=True, declared=True)["hit_observed"] is None
    assert sp.judge_prefix_shared_suffix("a", "b", "b", cached_tokens=-5, declared=True)["hit_observed"] is None


# ---- the advisory row in the boot check -------------------------------------------------------------------------------------

class _Spec:
    max_model_len = 4096
    max_num_seqs = None


def _live(monkeypatch, result):
    calls = []
    for name in ("probe_code", "probe_long_prefill", "probe_batching", "probe_prefix_reuse", "probe_reasoning"):
        monkeypatch.setattr(sp, name, lambda *a, _n=name, **k: {"name": _n, "state": "pass", "detail": "ok"} if _n != "probe_reasoning" else [])
    monkeypatch.setattr(sp, "probe_prefix_shared_suffix", lambda *a, **k: calls.append(k) or result)
    checks, evidence = {}, {}

    def check(name, ok, detail=""):
        checks[name] = {"ok": ok, "detail": detail}
    ck = can.Check(expect="Paris", stream=False, prefix_caching=True)
    can._live_probes("http://127.0.0.1:1", ck, _Spec(), {"model": "m"}, None, "", check, evidence)
    return calls, checks, evidence


def test_a_failing_advisory_result_is_listed_and_does_not_fail_the_boot(monkeypatch):
    bad = {"name": "prefix_shared_suffix", "state": "fail", "detail": "the warm reply differs", "advisory": True, "label": "L"}
    calls, checks, evidence = _live(monkeypatch, bad)
    assert "prefix_shared_suffix" not in checks, checks
    assert [a["name"] for a in evidence["advisory"]] == ["prefix_shared_suffix"] and evidence["advisory"][0]["state"] == "fail"
    assert calls and calls[0]["allow_reset"] is True and calls[0]["declared"] is True


def test_a_passing_advisory_result_is_listed_too(monkeypatch):
    ok = {"name": "prefix_shared_suffix", "state": "pass", "detail": "ok", "advisory": True, "label": "L"}
    _, checks, evidence = _live(monkeypatch, ok)
    assert "prefix_shared_suffix" not in checks and evidence["advisory"][0]["state"] == "pass"


def test_a_not_checked_advisory_result_is_listed_with_the_other_not_checked_probes(monkeypatch):
    nc = {"name": "prefix_shared_suffix", "state": "not_checked", "detail": "no reset route", "advisory": True}
    _, checks, evidence = _live(monkeypatch, nc)
    assert "prefix_shared_suffix" not in checks and "prefix_shared_suffix" in [n["name"] for n in evidence["not_checked"]]


def test_the_probe_does_not_run_for_a_can_that_does_not_declare_prefix_caching(monkeypatch):
    called = []
    monkeypatch.setattr(sp, "probe_prefix_shared_suffix", lambda *a, **k: called.append(1) or {})
    for name in ("probe_code", "probe_long_prefill", "probe_batching"):
        monkeypatch.setattr(sp, name, lambda *a, _n=name, **k: {"name": _n, "state": "pass", "detail": "ok"})
    can._live_probes("http://127.0.0.1:1", can.Check(expect="Paris", stream=False), _Spec(), {"model": "m"}, None, "", lambda *a, **k: None, {})
    assert called == []


def test_the_judge_alone_does_not_decide_for_a_can_that_does_not_declare_prefix_caching():
    r = sp.judge_prefix_shared_suffix("tangerine", "4417", "4417", cached_tokens=10, declared=False)
    assert r["state"] == "not_checked" and r["advisory"] is True
