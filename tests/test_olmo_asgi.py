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
