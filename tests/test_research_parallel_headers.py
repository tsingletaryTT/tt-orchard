# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Weight headers are read by several shards at once (headers.read_model). A repo has up to 64 shards and each
header is two requests, so one after another took 20 to 26 s a model on a real run. The parallel read must give the
same answer as the sequential one, stay inside the worker bound, and leave the fetcher's request budget exact."""
import json
import struct
import sys
import threading
import time
import urllib.request
from email.message import Message

import pytest

pytest.importorskip("tokenizers", reason="the class rules live in the delta-triage template")

from orchard.research import headers  # noqa: E402
from orchard.research.http import FetchError, Fetcher  # noqa: E402


def header_bytes(name: str) -> bytes:
    blob = json.dumps({f"{name}.w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 0]}}).encode()
    return struct.pack("<Q", len(blob)) + blob


def shard_names(n: int) -> list[str]:
    return [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]


class SlowFetch:
    """Serves headers by range. `delay(shard)` seconds per request, `fail` names shards that fail, and it records the
    largest number of requests in flight at once."""

    def __init__(self, shards, delay=lambda s: 0.0, fail=None, on_first=None):
        self.files = {s: header_bytes(s) for s in shards}
        self.delay, self.fail, self.on_first = delay, fail or {}, on_first
        self.lock, self.active, self.peak = threading.Lock(), 0, 0

    def get(self, url, *, byte_range=None, max_bytes=0):
        name = url.rsplit("/", 1)[1]
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.on_first is not None:
                self.on_first(name)
            time.sleep(self.delay(name))
            if name in self.fail:
                raise self.fail[name]
            start, end = byte_range
            return self.files[name][start:end + 1]
        finally:
            with self.lock:
                self.active -= 1

    def json(self, url, **kw):
        raise FetchError("HTTP 404")

    def now(self):
        return "2026-10-22T00:00:00Z"


def read(fetch, shards):
    return headers.read_model(fetch, "g/it", list(shards))


def test_the_shards_of_one_model_are_read_at_the_same_time():
    """Three shards wait for each other at a barrier. A sequential read never gets a second shard to the barrier."""
    barrier, first_time, lock = threading.Barrier(3, timeout=5), set(), threading.Lock()

    def meet(name):
        with lock:
            fresh = name not in first_time
            first_time.add(name)
        if fresh:
            barrier.wait()                  # raises BrokenBarrierError when the other shards never arrive

    r = read(SlowFetch(shard_names(3), on_first=meet), shard_names(3))
    assert r.data["errors"] == [] and len(r.data["tensors"]) == 3


def test_the_number_of_workers_is_bounded():
    shards = shard_names(24)
    f = SlowFetch(shards, delay=lambda s: 0.02)
    r = read(f, shards)
    assert r.data["errors"] == []
    assert 2 <= f.peak <= headers.HEADER_WORKERS


def test_the_worker_bound_is_a_small_number():
    assert 2 <= headers.HEADER_WORKERS <= 8


def test_results_are_in_shard_order_whatever_order_the_reads_finish_in():
    """Later shards answer first. Tensors, shards and errors still come back in shard order."""
    shards = shard_names(8)
    f = SlowFetch(shards, delay=lambda s: 0.002 * (9 - int(s.split("-")[1])),
                  fail={shards[1]: FetchError("HTTP 500"), shards[5]: FetchError("HTTP 404")})
    r = read(f, shards)
    assert r.data["shards"] == shards
    assert list(r.data["tensors"]) == [f"{s}.w" for i, s in enumerate(shards) if i not in (1, 5)]
    assert [t["shard"] for t in r.data["tensors"].values()] == [s for i, s in enumerate(shards) if i not in (1, 5)]
    assert r.data["errors"] == [f"{shards[1]}: HTTP 500", f"{shards[5]}: HTTP 404"]


def test_a_failed_shard_does_not_stop_the_others_and_a_value_error_is_recorded_like_before():
    shards = shard_names(6)
    f = SlowFetch(shards, fail={shards[0]: ValueError("shorter than 8 bytes")})
    r = read(f, shards)
    assert r.data["errors"] == [f"{shards[0]}: shorter than 8 bytes"] and len(r.data["tensors"]) == 5


def test_a_header_that_is_not_utf8_is_recorded_as_an_error_and_the_others_are_read():
    shards = shard_names(4)
    f = SlowFetch(shards)
    f.files[shards[2]] = struct.pack("<Q", 4) + b"\xff\xfe\xfd\xfc"
    r = read(f, shards)
    assert len(r.data["tensors"]) == 3
    assert r.data["errors"] == [f"{shards[2]}: invalid start byte"]


def test_the_same_answer_as_reading_one_after_another(monkeypatch):
    shards = shard_names(10)
    parallel = read(SlowFetch(shards, fail={shards[3]: FetchError("HTTP 500")}), shards).data
    monkeypatch.setattr(headers, "HEADER_WORKERS", 1)
    serial = read(SlowFetch(shards, fail={shards[3]: FetchError("HTTP 500")}), shards).data
    assert parallel == serial and list(parallel["tensors"]) == list(serial["tensors"])


# ---- with the real fetcher: the request budget, the deadline and the host state hold under threads ----------

class Resp:
    def __init__(self, body, status=206):
        self.body, self.status, self.headers = body, status, Message()

    def read(self, n=-1):
        return self.body if n is None or n < 0 else self.body[:n]

    def close(self):
        pass


class RangeOpener:
    """Answers a range request for a shard's header, like the hub does, and counts every request."""

    def __init__(self, shards, error=None):
        self.files = {s: header_bytes(s) for s in shards}
        self.requests, self.lock, self.error = [], threading.Lock(), error

    def open(self, req, timeout=None):
        with self.lock:
            self.requests.append(req.full_url)
        time.sleep(0.001)
        if self.error is not None:
            raise self.error
        name = req.full_url.rsplit("/", 1)[1]
        start, end = (int(x) for x in req.get_header("Range").removeprefix("bytes=").split("-"))
        return Resp(self.files[name][start:end + 1])


def test_a_spent_request_budget_is_exact_under_the_parallel_read():
    """24 shards need 48 requests. With 7 allowed, exactly 7 reach the network, in every one of 20 rounds."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(20):
            shards = shard_names(24)
            opener = RangeOpener(shards)
            f = Fetcher(None, opener=opener, max_requests=7)
            r = read(f, shards)
            assert len(opener.requests) == 7 and f._requests == 7
            assert any("request budget is spent" in e for e in r.data["errors"])
            assert len(r.data["errors"]) + len(r.data["tensors"]) == 24
    finally:
        sys.setswitchinterval(old)


def test_a_spent_time_budget_fails_every_shard_and_keeps_shard_order():
    shards = shard_names(6)
    opener = RangeOpener(shards)
    ticks = iter([0.0] + [999.0] * 100)
    f = Fetcher(None, opener=opener, deadline_s=10, clock=lambda: next(ticks))
    r = read(f, shards)
    assert len(opener.requests) == 1        # the first request starts the clock; none is made after the deadline
    assert [e.split(":")[0] for e in r.data["errors"]] == sorted(e.split(":")[0] for e in r.data["errors"])
    # The shard whose first request went through fails on its second, the other five on their first.
    assert len(r.data["errors"]) == 6 and all("time budget is spent" in e for e in r.data["errors"])


def test_a_dead_network_marks_the_host_and_every_shard_reports_a_reason():
    """Under threads the first connection failure marks the host. A shard that asked before the mark was set reports
    the connection error, a later one the unreachable host; each shard has one error either way, in shard order."""
    shards = shard_names(12)
    opener = RangeOpener(shards, error=urllib.request.URLError(ConnectionRefusedError("no route")))
    f = Fetcher(None, opener=opener)
    r = read(f, shards)
    assert "huggingface.co" in f._unreachable
    assert [e.split(":")[0] for e in r.data["errors"]] == shards and r.data["tensors"] == {}
    assert len(opener.requests) <= headers.HEADER_WORKERS         # no request after the mark, at most one per worker
