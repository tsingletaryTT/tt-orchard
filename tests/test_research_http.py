# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The research fetcher (orchard/research/http.py): the one place the research package touches the network.

Every guard here is a rule from the design: no token, fixed hosts, a size cap, byte ranges answered with
206, a cache a resume reads, and no network at all when offline."""
import http.client
import urllib.error
import urllib.request
from email.message import Message

import pytest

from orchard.research import types
from orchard.research.http import FetchError, Fetcher, _Redirects, host_allowed

HF = "https://huggingface.co/api/models/a/b"


class Resp:
    def __init__(self, body=b"{}", status=200):
        self.body, self.status, self.headers = body, status, Message()

    def read(self, n=-1):
        self.asked = n
        return self.body if n is None or n < 0 else self.body[:n]

    def close(self):
        pass


class FakeOpener:
    def __init__(self, routes):
        self.routes, self.requests = routes, []

    def open(self, req, timeout=None):
        self.requests.append(req)
        got = self.routes[req.full_url]
        if isinstance(got, Exception):
            raise got
        return got


def http_error(code, **headers):
    msg = Message()
    for key, value in headers.items():
        msg[key.replace("_", "-")] = value
    return urllib.error.HTTPError(HF, code, "x", msg, None)


def fetcher(tmp_path, routes, **kw):
    return Fetcher(tmp_path / "cache", opener=FakeOpener(routes), now=lambda: "2026-10-12T00:00:00Z", **kw)


def test_no_authorization_header_is_ever_sent_even_with_tokens_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 34)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "b" * 36)
    f = fetcher(tmp_path, {HF: Resp(b'{"id": "a/b"}')})
    assert f.json(HF) == {"id": "a/b"}
    sent = {k.lower() for k in f.opener.requests[0].headers}
    assert "authorization" not in sent and "x-api-key" not in sent


@pytest.mark.parametrize("url", ["https://evil.example/x", "http://huggingface.co/api/models/a/b",
                                 "https://huggingface.co.evil.example/x", "ftp://huggingface.co/x"])
def test_only_https_on_the_allowed_hosts_is_fetched(tmp_path, url):
    f = fetcher(tmp_path, {})
    with pytest.raises(FetchError, match="not allowed"):
        f.get(url)
    assert f.opener.requests == []


def test_host_allow_list():
    assert host_allowed("huggingface.co") and host_allowed("api.github.com")
    assert host_allowed("raw.githubusercontent.com") and host_allowed("us.aws.cdn.hf.co")
    assert not host_allowed("evilhf.co") and not host_allowed("hf.co.evil.example") and not host_allowed(None)


def test_a_redirect_to_another_host_is_refused_and_a_hub_cdn_redirect_is_followed():
    req = urllib.request.Request(HF)
    with pytest.raises(FetchError, match="redirect"):
        _Redirects().redirect_request(req, None, 302, "Found", Message(), "https://evil.example/x")
    ok = _Redirects().redirect_request(req, None, 302, "Found", Message(), "https://us.aws.cdn.hf.co/x?sig=1")
    assert ok.full_url.startswith("https://us.aws.cdn.hf.co/")


def test_a_redirect_from_https_down_to_http_is_refused():
    req = urllib.request.Request(HF)
    with pytest.raises(FetchError, match="redirect"):
        _Redirects().redirect_request(req, None, 302, "Found", Message(), "http://huggingface.co/x")


def test_the_production_opener_carries_the_guarded_redirect_handler():
    handlers = Fetcher(None).opener.handlers
    assert any(isinstance(h, _Redirects) for h in handlers)


def test_after_one_connection_failure_later_requests_to_that_host_make_no_request(tmp_path):
    other = "https://huggingface.co/api/models/c/d"
    f = fetcher(tmp_path, {HF: OSError("network is unreachable"), other: Resp(b"{}")})
    with pytest.raises(FetchError, match="unreachable"):
        f.get(HF)
    with pytest.raises(FetchError, match="network unreachable: huggingface.co"):
        f.get(other)
    assert len(f.opener.requests) == 1


def test_a_connection_failure_on_one_host_does_not_stop_requests_to_another(tmp_path):
    gh = "https://api.github.com/x"
    f = fetcher(tmp_path, {HF: OSError("timed out"), gh: Resp(b"{}")})
    with pytest.raises(FetchError):
        f.get(HF)
    assert f.get(gh) == b"{}"


def test_a_value_error_does_not_mark_the_host_unreachable(tmp_path):
    other = "https://huggingface.co/api/models/c/d"
    f = fetcher(tmp_path, {HF: ValueError("bad url"), other: Resp(b"{}")})
    with pytest.raises(FetchError, match="bad url"):
        f.get(HF)
    assert f.get(other) == b"{}"


def test_an_http_error_does_not_mark_the_host_unreachable(tmp_path):
    other = "https://huggingface.co/api/models/c/d"
    f = fetcher(tmp_path, {HF: http_error(404), other: Resp(b"{}")})
    with pytest.raises(FetchError, match="404"):
        f.get(HF)
    assert f.get(other) == b"{}"


def test_a_range_must_be_answered_with_206(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"12345678", status=200)})
    with pytest.raises(FetchError, match="range"):
        f.get(HF, byte_range=(0, 7))


def test_a_range_is_sent_and_a_206_body_is_returned(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"12345678", status=206)})
    assert f.get(HF, byte_range=(0, 7)) == b"12345678"
    assert f.opener.requests[0].get_header("Range") == "bytes=0-7"


def test_a_body_over_the_cap_is_refused(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"x" * 100)})
    with pytest.raises(FetchError, match="larger"):
        f.get(HF, max_bytes=50)


def test_the_read_itself_is_bounded_by_the_cap(tmp_path):
    resp = Resp(b"x" * 10)
    f = fetcher(tmp_path, {HF: resp})
    f.get(HF, max_bytes=50)
    assert resp.asked == 51              # one byte over the cap is enough to know it was exceeded


def test_a_range_body_longer_than_the_range_is_refused(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"x" * 100, status=206)})
    with pytest.raises(FetchError, match="larger"):
        f.get(HF, byte_range=(0, 7))


def test_the_second_get_comes_from_the_cache_and_a_new_fetcher_reads_it_too(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b'{"n": 1}')})
    f.json(HF), f.json(HF)
    assert len(f.opener.requests) == 1
    g = fetcher(tmp_path, {HF: Resp(b'{"n": 2}')})
    assert g.json(HF) == {"n": 1} and g.opener.requests == []


def test_refresh_ignores_the_cache(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b'{"n": 1}')})
    f.json(HF)
    g = fetcher(tmp_path, {HF: Resp(b'{"n": 2}')}, refresh=True)
    assert g.json(HF) == {"n": 2}


def test_without_a_cache_directory_nothing_is_written(tmp_path):
    f = Fetcher(None, opener=FakeOpener({HF: Resp(b"{}")}))
    f.json(HF)
    assert list(tmp_path.iterdir()) == []


def test_offline_makes_no_request_even_when_the_cache_has_the_answer(tmp_path):
    fetcher(tmp_path, {HF: Resp(b"{}")}).json(HF)
    g = fetcher(tmp_path, {HF: Resp(b"{}")}, offline=True)
    with pytest.raises(FetchError, match="offline"):
        g.json(HF)
    assert g.opener.requests == []


@pytest.mark.parametrize("err,reason", [
    (http_error(401), "HTTP 401"), (http_error(404), "HTTP 404"), (http_error(429), "rate limited"),
    (http_error(403, x_ratelimit_remaining="0"), "rate limited"), (http_error(403), "HTTP 403")])
def test_http_errors_become_reasons(tmp_path, err, reason):
    f = fetcher(tmp_path, {HF: err})
    with pytest.raises(FetchError) as info:
        f.get(HF)
    assert reason in info.value.reason


def test_a_failure_is_not_cached(tmp_path):
    f = fetcher(tmp_path, {HF: http_error(500)})
    with pytest.raises(FetchError):
        f.get(HF)
    g = fetcher(tmp_path, {HF: Resp(b'{"ok": 1}')})
    assert g.json(HF) == {"ok": 1}      # not {}: a cached failure placeholder would show here


def test_a_body_that_is_not_json_is_a_reason_not_a_crash(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"<html>")})
    with pytest.raises(FetchError, match="JSON"):
        f.json(HF)


def test_a_network_error_is_a_reason(tmp_path):
    f = fetcher(tmp_path, {HF: OSError("timed out")})
    with pytest.raises(FetchError, match="timed out"):
        f.get(HF)


class BrokenBody(Resp):
    """An answer whose body fails part way: the connection opened, then the read raised."""
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def read(self, n=-1):
        raise self.exc


@pytest.mark.parametrize("exc", [TimeoutError("read timed out"), http.client.IncompleteRead(b"ab"),
                                 ConnectionResetError("reset by peer")])
def test_a_body_that_fails_while_reading_is_a_fetch_error(tmp_path, exc):
    f = fetcher(tmp_path, {HF: BrokenBody(exc)})
    with pytest.raises(FetchError):
        f.get(HF)


def test_a_malformed_status_line_is_a_fetch_error(tmp_path):
    f = fetcher(tmp_path, {HF: http.client.BadStatusLine("x")})
    with pytest.raises(FetchError):
        f.get(HF)


def test_a_failed_body_read_does_not_mark_the_host_unreachable(tmp_path):
    other = "https://huggingface.co/api/models/c/d"
    f = fetcher(tmp_path, {HF: BrokenBody(TimeoutError("read timed out")), other: Resp(b"{}")})
    with pytest.raises(FetchError):
        f.get(HF)
    assert f.get(other) == b"{}"


def test_one_timeout_does_not_mark_the_host_unreachable(tmp_path):
    other = "https://huggingface.co/api/models/c/d"
    f = fetcher(tmp_path, {HF: TimeoutError("timed out"), other: Resp(b"{}")})
    with pytest.raises(FetchError):
        f.get(HF)
    assert f.get(other) == b"{}"


def test_repeated_timeouts_mark_the_host_unreachable(tmp_path):
    urls = [f"https://huggingface.co/api/models/t/{i}" for i in range(3)]
    f = fetcher(tmp_path, {u: TimeoutError("timed out") for u in urls})
    for u in urls[:2]:
        with pytest.raises(FetchError, match="timed out"):
            f.get(u)
    with pytest.raises(FetchError, match="network unreachable"):
        f.get(urls[2])
    assert len(f.opener.requests) == 2


def test_a_timeout_wrapped_in_a_url_error_counts_as_a_timeout(tmp_path):
    other = "https://huggingface.co/api/models/c/d"
    f = fetcher(tmp_path, {HF: urllib.error.URLError(TimeoutError("timed out")), other: Resp(b"{}")})
    with pytest.raises(FetchError):
        f.get(HF)
    assert f.get(other) == b"{}"


# ---- types ---------------------------------------------------------------------------------------

def test_clean_makes_one_short_line_and_removes_a_home_path_or_token():
    assert types.clean("a\x00b\n  c") == "a b c"
    assert len(types.clean("x" * 1000)) == 300
    assert types.clean("see /home/someone/.cache/x").startswith("[removed:")
    assert types.clean("tok " + "hf_" + "a" * 34).startswith("[removed:")


def test_a_token_that_starts_before_the_length_limit_and_ends_after_it_is_not_kept_even_in_part():
    token = "hf_" + "a" * 34
    text = "x" * 279 + " " + token + " tail"
    out = types.clean(text)
    assert "hf_" not in out and "aaaa" not in out
    assert out.startswith("[removed:")


def test_clean_strips_c1_controls_and_bidi_overrides():
    out = types.clean("a\u009b31m \u202eevil\u2066 x\u2069 y\u202a")
    for ch in "\u009b\u202e\u2066\u2069\u202a":
        assert ch not in out
    assert "evil" in out


def test_an_unavailable_reason_is_cleaned_when_it_is_written_out():
    row = types.Unavailable("lineage", "boom\x1b[31m in /home/someone/.cache/x").as_dict()
    assert "\x1b" not in row["reason"] and "/home/someone" not in row["reason"]
    assert row["source"] == "lineage"
    assert types.Unavailable("s", "plain reason").as_dict() == {"source": "s", "reason": "plain reason"}


@pytest.mark.parametrize("value,ok", [
    ("google/gemma-4-12B", True), ("Qwen/Qwen3.8-27B", True), ("a/b", True),
    ("x/../../y?z", False), ("a/b c", False), ("a/b\n", False), ("nogroup", False), ("a/b/c", False),
    ("", False), (None, False), (7, False), ("a/" + "b" * 200, False)])
def test_valid_id(value, ok):
    assert types.valid_id(value) is ok


def test_result_extend_merges_lists():
    a = types.Result(evidence=[1], unavailable=[2])
    a.extend(types.Result(evidence=[3], unavailable=[4]))
    assert a.evidence == [1, 3] and a.unavailable == [2, 4]


# ---- the budget: a research run makes a bounded number of requests in a bounded time ------------------

def test_after_the_request_budget_further_requests_fail_at_once_without_a_request(tmp_path):
    routes = {f"{HF}{i}": Resp(b"{}") for i in range(4)}
    f = fetcher(tmp_path, routes, max_requests=2)
    f.json(HF + "0")
    f.json(HF + "1")
    with pytest.raises(FetchError, match="request budget"):
        f.json(HF + "2")
    assert len(f.opener.requests) == 2


def test_after_the_deadline_further_requests_fail_at_once_without_a_request(tmp_path):
    clock = iter([0.0, 10.0, 500.0]).__next__
    f = fetcher(tmp_path, {HF + "0": Resp(b"{}"), HF + "1": Resp(b"{}"), HF + "2": Resp(b"{}")},
                deadline_s=100, clock=clock)
    f.json(HF + "0")
    f.json(HF + "1")
    with pytest.raises(FetchError, match="time budget"):
        f.json(HF + "2")
    assert len(f.opener.requests) == 2


def test_cached_answers_do_not_spend_the_budget(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"{}")}, max_requests=1)
    f.json(HF)
    assert f.json(HF) == {} and len(f.opener.requests) == 1


def test_a_spent_budget_does_not_mark_the_host_unreachable_or_get_cached(tmp_path):
    f = fetcher(tmp_path, {HF + "0": Resp(b"{}"), HF + "1": Resp(b"{}")}, max_requests=1)
    f.json(HF + "0")
    with pytest.raises(FetchError):
        f.json(HF + "1")
    assert "huggingface.co" not in f._unreachable
    assert fetcher(tmp_path, {HF + "1": Resp(b'{"a": 1}')}).json(HF + "1") == {"a": 1}


def test_the_budget_has_named_defaults():
    from orchard.research import http
    f = Fetcher(None)
    assert (f.max_requests, f.deadline_s) == (http.MAX_REQUESTS, http.DEADLINE_S)
    assert 100 <= http.MAX_REQUESTS and 60 <= http.DEADLINE_S <= 600


def test_a_spent_budget_makes_the_class_estimate_unknown_not_a_guess(tmp_path):
    pytest.importorskip("tokenizers", reason="the class rules live in the delta-triage template")
    import json as _json
    import struct
    from orchard.research import headers
    blob = _json.dumps({"a.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    data = struct.pack("<Q", len(blob)) + blob

    class Ranged(FakeOpener):
        def open(self, req, timeout=None):
            self.requests.append(req)
            start, end = (int(x) for x in req.headers["Range"].split("=")[1].split("-"))
            return Resp(data[start:end + 1], status=206)
    files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    full = Fetcher(None, opener=Ranged({}))
    cand = headers.read_model(full, "g/base", files)
    short = Fetcher(None, opener=Ranged({}), max_requests=2)          # one shard's two ranges, not the second's
    new = headers.read_model(short, "g/new", files)
    assert any("budget" in e for e in new.data["errors"])
    cfg = {"model_type": "g", "num_hidden_layers": 1}
    assert headers.estimate_class(cfg, new.data, cfg, cand.data)["class"] == "unknown"


# ---- the authority is the host and nothing else -------------------------------------------------------------
# urlsplit and urllib.request read these differently: urlsplit's hostname is an allowed host, while urllib would
# connect to the whole authority (for example "evil.com\\@huggingface.co"). Only a bare allowed host passes.

ODD_AUTHORITIES = ["https://evil.com\\@huggingface.co/x", "https://evil.com%2f@huggingface.co/x",
                   "https://user@huggingface.co/x", "https://user:pw@huggingface.co/x",
                   "https://huggingface.co:9999/x", "https://huggingface.co:443/x",
                   "https://cdn.hf.co:8443/x", "https://[::1]/x"]


@pytest.mark.parametrize("url", ODD_AUTHORITIES)
def test_a_url_whose_authority_is_more_than_an_allowed_host_is_not_fetched(tmp_path, url):
    f = fetcher(tmp_path, {})
    with pytest.raises(FetchError, match="not allowed"):
        f.get(url)
    assert f.opener.requests == []


@pytest.mark.parametrize("url", ODD_AUTHORITIES)
def test_a_redirect_whose_location_carries_more_than_an_allowed_host_is_refused(url):
    with pytest.raises(FetchError, match="redirect"):
        _Redirects().redirect_request(urllib.request.Request(HF), None, 302, "Found", Message(), url)


def test_a_host_in_capitals_is_still_the_allowed_host():
    from orchard.research.http import url_allowed
    assert url_allowed("https://HuggingFace.co/api/models/a/b") and url_allowed("https://us.aws.cdn.hf.co/x?sig=1")


# ---- threads: the research reads weight headers in parallel, so the fetcher's counters must hold up ---------------

def test_the_budget_check_and_the_start_time_are_taken_under_the_lock(tmp_path):
    """Wiring: _spend reads the clock inside its critical section, so a clock that notes the lock shows what is held."""
    f = fetcher(tmp_path, {HF: Resp(b"{}")})
    held = []
    f._clock = lambda: (held.append(f._lock.locked()), 0.0)[1]
    f._spend()
    assert held == [True]


def test_connection_failure_bookkeeping_is_done_under_the_lock(tmp_path):
    f = fetcher(tmp_path, {})
    seen = []

    class WatchedDict(dict):
        def __setitem__(self, key, value):
            seen.append(("timeouts", f._lock.locked()))
            super().__setitem__(key, value)

    class WatchedSet(set):
        def add(self, item):
            seen.append(("unreachable", f._lock.locked()))
            super().add(item)

    f._timeouts, f._unreachable = WatchedDict(), WatchedSet()
    f._note_connection_failure("huggingface.co", TimeoutError("slow"))
    f._note_connection_failure("huggingface.co", TimeoutError("slow"))
    f._note_connection_failure("api.github.com", ConnectionRefusedError("no"))
    assert seen == [("timeouts", True), ("timeouts", True), ("unreachable", True), ("unreachable", True)]


def test_many_threads_never_spend_more_than_the_budget():
    """The check and the increment are one step: 48 threads ask for 48 distinct URLs at once, five requests are
    allowed, and exactly five reach the network. Several rounds with a fast thread switch give a race room."""
    import sys
    import threading
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(30):
            urls = [HF + str(i) for i in range(48)]
            f = Fetcher(None, opener=FakeOpener({u: Resp(b"{}") for u in urls}), max_requests=5)
            start, results = threading.Barrier(len(urls)), []

            def one(u):
                start.wait()
                try:
                    f.get(u)
                    results.append("ok")
                except FetchError as exc:
                    results.append(exc.reason)

            threads = [threading.Thread(target=one, args=(u,)) for u in urls]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert len(f.opener.requests) == 5 and results.count("ok") == 5 and f._requests == 5
            assert all("budget" in r for r in results if r != "ok")
    finally:
        sys.setswitchinterval(old)
