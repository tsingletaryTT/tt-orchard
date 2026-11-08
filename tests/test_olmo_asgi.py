# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Olmo-Hybrid-7B ASGI adapter (pkgsrc/server/olmo_asgi.py) and its card generator, against fakes. No device, no weights.

Review of PR 633: (1) the request parser must reject what it cannot honour, (2) a stop string that spans tokens must not leak its first
piece, (3) a non-streaming request must watch for a disconnect and free the single slot, (4) the card must not explain a divergence the
records contradict."""
import asyncio
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("olmo_asgi_under_test", ROOT / "pkgsrc" / "server" / "olmo_asgi.py")
oa = importlib.util.module_from_spec(SPEC)
sys.modules["olmo_asgi_under_test"] = oa
SPEC.loader.exec_module(oa)


def parse(**kw):
    return oa.parse_request(json.dumps({"prompt": "hi", **kw}).encode(), chat=False)


# ---- 1: the parser -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [("stream", "true"), ("stream", 1), ("stop", False), ("stop", 5), ("stop", [1]), ("n", 2), ("n", "1"),
                                          ("logprobs", 3), ("tools", []), ("response_format", {}), ("best_of", 2), ("echo", True),
                                          ("presence_penalty", 0.5), ("stream_options", "x"), ("return_token_ids", "yes"), ("surprise", 1)])
def test_a_request_the_server_cannot_honour_is_a_400(field, value):
    with pytest.raises(oa.RequestError):
        parse(**{field: value})


def test_the_supported_fields_are_accepted_and_n_of_one_is_fine():
    r = parse(stream=True, stop="x", temperature=0.5, top_p=0.9, seed=3, n=1, max_tokens=4, return_token_ids=True, model="m", stream_options={"include_usage": True})
    assert r["stream"] is True and r["stop"] == ["x"] and r["return_token_ids"] is True
    assert parse()["stream"] is False and parse(stop=None)["stop"] == []


def test_the_chat_route_checks_the_same_fields():
    with pytest.raises(oa.RequestError):
        oa.parse_request(json.dumps({"messages": [{"role": "user", "content": "x"}], "n": 2}).encode(), chat=True)


# ---- 2: stop strings across tokens ---------------------------------------------------------------------------------------------

class Tok:
    """ids are indexes into `pieces`; decode joins them."""
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


class Port:
    """prefill and decode_step return one-hot logits that choose the next id of a fixed script."""
    def __init__(self, script, vocab):
        self.script, self.vocab, self.i = script, vocab, 0

    def _logits(self):
        import torch
        x = torch.zeros(self.vocab)
        x[self.script[min(self.i, len(self.script) - 1)]] = 5.0
        self.i += 1
        return x

    def prefill(self, model, ids):
        self.i = 0
        return self._logits()

    def decode_step(self, model, tok, pos):
        return self._logits()


def run_generate(pieces, script, stop, max_tokens=32, eos=(99,)):
    pytest.importorskip("torch")
    oa.STATE.update(port=Port(script, 100), model=None, tok=Tok(pieces), eos=set(eos), max_len=1000)
    r = {"max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "seed": 1, "stop": stop}
    timings = {}
    out = list(oa.generate([1, 2, 3], r, timings))
    return out, timings


def text_of(out):
    return "".join(p for p, _ in out)


def test_a_stop_string_that_spans_tokens_never_leaks_its_first_piece():
    pieces = {0: "x", 1: "a", 2: "b", 3: "c"}
    out, _ = run_generate([pieces.get(i, "") for i in range(100)], [0, 1, 2, 3], ["ab"])
    assert text_of(out) == "x" and out[-1][1] == "stop"


def test_held_back_text_is_flushed_when_the_stop_string_never_completes():
    pieces = {0: "x", 1: "a", 2: "c"}
    out, _ = run_generate([pieces.get(i, "") for i in range(100)], [0, 1, 2], ["ab"], max_tokens=3)
    assert text_of(out) == "xac" and out[-1][1] == "length"


def test_held_back_text_is_flushed_at_an_end_token():
    pieces = {0: "x", 1: "a"}
    out, _ = run_generate([pieces.get(i, "") for i in range(100)], [0, 1, 99], ["ab"])
    assert text_of(out) == "xa" and out[-1][1] == "stop"


def test_a_stop_string_inside_one_token_still_cuts_there():
    pieces = {0: "he", 1: "llo ab cd"}
    out, _ = run_generate([pieces.get(i, "") for i in range(100)], [0, 1], ["ab"], max_tokens=2)
    assert text_of(out) == "hello " and out[-1][1] == "stop"


# ---- 3: a disconnect in a non-streaming request --------------------------------------------------------------------------------

def test_a_non_streaming_request_watches_for_a_disconnect_and_frees_the_slot(monkeypatch):
    seen = {"cancelled": None, "stopped_early": False}

    def slow_generate(ids, r, timings, cancelled=None):
        seen["cancelled"] = cancelled
        for _ in range(200):                      # about 10 s of work if nobody cancels
            if cancelled is not None and cancelled.is_set():
                seen["stopped_early"] = True
                return
            time.sleep(0.05)
            yield "x", None
        yield "", "length"

    class T:
        def encode(self, text):
            return type("E", (), {"ids": [1, 2, 3]})()

    oa.STATE.update(tok=T(), max_len=1000)
    monkeypatch.setattr(oa, "generate", slow_generate)
    sent = []
    inbox = asyncio.Queue()

    async def receive():
        return await inbox.get()

    async def send(msg):
        sent.append(msg)

    async def drive():
        await inbox.put({"type": "http.request", "body": json.dumps({"prompt": "hi", "max_tokens": 5}).encode(), "more_body": False})
        task = asyncio.ensure_future(oa._complete_route(receive, send, False))
        await asyncio.sleep(0.4)
        assert not oa.LOCK.acquire(blocking=False), "the request should hold the slot while it generates"
        await inbox.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, timeout=5)

    t0 = time.time()
    asyncio.run(drive())
    assert time.time() - t0 < 5 and seen["cancelled"] is not None and seen["stopped_early"]
    assert oa.LOCK.acquire(blocking=False), "the slot must be free once the request is over"
    oa.LOCK.release()


# ---- 4: the card ----------------------------------------------------------------------------------------------------------------

def diag(tmp_path, **over):
    row = {"prompt": "The capital of France is", "compared": 32, "leading_match": 1, "matching_positions": 1, "matched_fully": False,
           "first_difference_at_token_index": 1,
           "reference_top2": [{"id": 13, "text": ".", "logit": 9.3125}, {"id": 11, "text": ",", "logit": 9.25}], "reference_gap": 0.0625,
           "chip_top2": [{"id": 13, "text": ".", "logit": 9.25}, {"id": 11, "text": ",", "logit": 9.1875}], "chip_gap": 0.0625,
           "near_tie": True, "reference_generate_choice": {"id": 11, "text": ","}, "reference_generate_choice_differs_from_its_full_forward_top1": True,
           "served_repeats": {"runs": 6, "token_at_index": [13] * 6, "all_identical": True}}
    row.update(over)
    other = {"prompt": "Water", "compared": 32, "leading_match": 32, "matching_positions": 32, "matched_fully": True}
    p = tmp_path / "diag.json"
    p.write_text(json.dumps({"near_tie_logits": 0.25, "prompts": [row, other]}))
    return p


def render(tmp_path, diag_path):
    bench = {"load_before": [1, 1, 1], "load_after": [1, 1, 1], "repeats": 3,
             "decode": {"prompts": 5, "new_tokens_each": 64, "runs": 15, "tok_s": {"n": 15, "min": 1.0, "median": 2.0, "max": 3.0}},
             "prefill": [{"prompt_tokens": 10, "seconds": {"n": 3, "min": 1.0, "median": 1.0, "max": 1.0}, "tok_s_median": 10.0}],
             "concurrency": {"batching": "x"}}
    (tmp_path / "bench.json").write_text(json.dumps(bench))
    (tmp_path / "q.json").write_text(json.dumps({"top1": 1.0, "ppl": 3.0, "n": 32}))
    out = tmp_path / "card.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "olmo_hybrid" / "make_card.py"), "--warm", str(tmp_path / "bench.json"), "--quality",
                    str(tmp_path / "q.json"), "--diag", str(diag_path), "--out", str(out)], check=True)
    return out.read_text()


def test_the_near_tie_explanation_needs_the_records_to_back_it(tmp_path):
    good = render(tmp_path, diag(tmp_path))
    assert "bf16 near tie" in good and "took the other side of the tie" in good


@pytest.mark.parametrize("over", [
    {"chip_top2": [{"id": 11, "text": ",", "logit": 9.3}, {"id": 13, "text": ".", "logit": 9.2}]},                  # the chip ranks them the other way
    {"chip_top2": [{"id": 13, "text": ".", "logit": 9.25}, {"id": 99, "text": "!", "logit": 9.1875}]},                # a different second token
    {"reference_generate_choice": {"id": 13, "text": "."}, "reference_generate_choice_differs_from_its_full_forward_top1": False},   # no reversal
    {"reference_generate_choice": {"id": 77, "text": "?"}, "reference_generate_choice_differs_from_its_full_forward_top1": True},   # a third token
])
def test_a_contradicted_explanation_is_reported_as_unexplained(tmp_path, over):
    card = render(tmp_path, diag(tmp_path, **over))
    assert "bf16 near tie" not in card and "took the other side of the tie" not in card
    assert "not explained" in card


# ---- review round 3: the worker outlives a cancelled request; stop strings are bounded and matched incrementally ----------------------

def test_a_cancelled_request_task_still_waits_for_the_inference_thread(monkeypatch):
    """Cancelling the ASGI task cancels the executor future, but the thread keeps running. Cleanup must wait for the thread, so the slot and
    a later close of the model never overlap with inference."""
    gate = threading.Event()
    entered = threading.Event()

    def blocking_generate(ids, r, timings, cancelled=None):
        entered.set()
        gate.wait(30)                                  # a long chip step that does not look at the cancel event
        yield "x", None
        yield "", "length"

    class T:
        def encode(self, text):
            return type("E", (), {"ids": [1, 2, 3]})()

    oa.STATE.update(tok=T(), max_len=1000)
    monkeypatch.setattr(oa, "generate", blocking_generate)
    inbox = asyncio.Queue()

    async def receive():
        return await inbox.get()

    async def send(msg):
        pass

    async def drive():
        await inbox.put({"type": "http.request", "body": json.dumps({"prompt": "hi", "max_tokens": 5}).encode(), "more_body": False})
        task = asyncio.ensure_future(oa._complete_route(receive, send, False))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.2)
        task.cancel()                                  # a second cancellation while cleanup waits must not cut the wait short either
        await asyncio.sleep(0.3)
        assert not task.done(), "the request task must not finish while the inference thread is still running"
        assert not oa.LOCK.acquire(blocking=False), "the slot must stay taken while the thread runs"
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

    asyncio.run(drive())
    assert oa.LOCK.acquire(blocking=False), "the slot is free once the thread has finished"
    oa.LOCK.release()


def test_shutdown_waits_for_a_running_request_before_it_closes_the_model():
    closed = []
    oa.LOCK.acquire()
    t = threading.Thread(target=oa._close_when_idle, args=(lambda: closed.append(time.time()),), daemon=True)
    t.start()
    time.sleep(0.3)
    assert closed == [], "close ran while a request still held the slot"
    oa.LOCK.release()
    t.join(5)
    assert len(closed) == 1


def test_the_lifespan_shutdown_goes_through_the_wait(monkeypatch):
    closed = []
    monkeypatch.setattr(oa, "startup", lambda a: (lambda: closed.append(1)))
    inbox = asyncio.Queue()
    sent = []

    async def receive():
        return await inbox.get()

    async def send(msg):
        sent.append(msg["type"])

    async def drive():
        task = asyncio.ensure_future(oa.app({"type": "lifespan"}, receive, send))
        await inbox.put({"type": "lifespan.startup"})
        await asyncio.sleep(0.1)
        oa.LOCK.acquire()
        await inbox.put({"type": "lifespan.shutdown"})
        await asyncio.sleep(0.5)
        assert closed == [] and "lifespan.shutdown.complete" not in sent
        oa.LOCK.release()
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(drive())
    assert closed == [1] and "lifespan.shutdown.complete" in sent


def test_a_stop_string_over_the_length_limit_is_a_400():
    assert parse(stop="x" * oa.MAX_STOP_CHARS)["stop"] == ["x" * oa.MAX_STOP_CHARS]
    with pytest.raises(oa.RequestError, match="characters"):
        parse(stop="x" * (oa.MAX_STOP_CHARS + 1))
    with pytest.raises(oa.RequestError):
        parse(stop=["ok", "y" * (1 << 20)])


def test_the_stop_matcher_does_a_bounded_amount_of_work_per_character():
    stops = ["a" * 255 + "b", "ab" * 120 + "c", "z" * 256, "q" * 100]
    m = oa.StopMatcher(stops)
    text = ("a" * 200 + "ab" * 50) * 40                  # input that makes a naive matcher backtrack
    for k in range(0, len(text), 7):
        assert m.feed(text[k:k + 7], k) is None
    assert m.ops <= 2 * len(text) * len(stops) + 4 * sum(len(s) for s in stops), (m.ops, len(text))


def test_the_stop_matcher_finds_overlapping_starts_and_reports_the_end():
    m = oa.StopMatcher(["aab"])
    assert m.feed("aa", 0) is None and m.hold == 2
    assert m.feed("ab", 2) == (4, 3)


def test_generate_matches_incrementally_and_never_rescans_the_text(monkeypatch):
    pytest.importorskip("torch")
    made = []

    class Counting(oa.StopMatcher):
        def __init__(self, stops):
            super().__init__(stops)
            made.append(self)

    monkeypatch.setattr(oa, "StopMatcher", Counting)
    pieces = [chr(ord("a") + i % 26) for i in range(100)]
    stop = ["q" * 256]                                   # never matches
    out, _ = run_generate(pieces, list(range(60)), stop, max_tokens=60)
    chars = len(text_of(out))
    assert chars == 60 and made and made[0].ops <= 2 * chars + 4 * 256


def test_the_stop_matcher_agrees_with_a_brute_force_search_on_random_input():
    import random
    rng = random.Random(7)
    for _ in range(300):
        stops = ["".join(rng.choice("ab") for _ in range(rng.randint(1, 6))) for _ in range(rng.randint(1, 3))]
        text = "".join(rng.choice("ab") for _ in range(rng.randint(0, 30)))
        # the first stop string to complete: the smallest end offset over all occurrences of all stop strings
        ends = [(text.find(x) + len(x), len(x)) for x in stops if x in text]
        want = min(ends) if ends else None
        m = oa.StopMatcher(stops)
        got = None
        for k in range(0, len(text), 3):
            got = m.feed(text[k:k + 3], k)
            if got:
                break
        assert (got[0] if got else None) == (want[0] if want else None), (stops, text, got, want)
        if want is None:
            longest = max((k for x in stops for k in range(len(x)) if text.endswith(x[:k])), default=0)
            assert m.hold == longest, (stops, text)
