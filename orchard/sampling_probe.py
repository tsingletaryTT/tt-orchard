# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What a served configuration does with a sampled request (issue 107).

Some fast decoding paths only work with greedy decoding: the speculative drafter of the Qwen3.8 bundles verifies by argmax, and the
plugin neutralises or ignores sampling controls on those steps. A request with temperature above 0 is then slower, refused, or
silently answered greedily. The swap template (`serve_and_compare.py`, step i) sends one greedy request and two sampled requests that
differ only in seed, to the server that has just passed the 32-token check, and records each one raw under `sampling_probe` in
swap-check.json. This module reads that record. It recomputes every speed from the recorded tokens and seconds and never reads a
summary field, so a record that somebody edited cannot say more than its raw numbers do.

The record is a capability statement. A sampled request that is refused or slow does not fail a stage. `problems` is only the check
that the record is well formed.
"""
from __future__ import annotations

import math

from . import defaults, drafter_log, prompt_suite

ENTRY_NAMES = ("greedy", "sampled_a", "sampled_b")
VERDICTS = ("not-run", "malformed", "greedy-refused", "sampled-refused", "sampled-incoherent", "inconclusive", "sampled-ignored",
            "sampled-slower", "sampled-full-speed")

SLOW_RATIO = 0.7        # choice: a sampled request that decodes at less than this share of the greedy speed is called slower. 1 would call
                        # every 5 % wobble slower; the AREX-2 safe configuration against the grafted one differed by 2.95 times, so 0.7
                        # leaves room for noise of one short request and still separates a path that falls back to the host.
MIN_TOKENS = 16         # choice: a reply shorter than this says too little about speed, or about whether two seeds gave different text.


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def problems(record) -> list[str]:
    """Why the record is not well formed, or [] for a record that is well formed or absent."""
    if record is None or (isinstance(record, dict) and not record.get("ran")):
        return []
    if not isinstance(record, dict):
        return [f"sampling_probe must be an object, got {record!r}"]
    reqs = record.get("requests")
    if not isinstance(reqs, dict):
        return [f"sampling_probe.requests must be an object, got {reqs!r}"]
    out = []
    for name in ENTRY_NAMES:
        e = reqs.get(name)
        if not isinstance(e, dict):
            out.append(f"sampling_probe.requests.{name} is missing")
            continue
        if not isinstance(e.get("accepted"), bool):
            out.append(f"sampling_probe.requests.{name}.accepted must be true or false")
        if not isinstance(e.get("text"), str):
            out.append(f"sampling_probe.requests.{name}.text must be a string")
        for key in ("drafter_log", "error", "finish_reason"):
            if e.get(key) is not None and not isinstance(e[key], str):
                out.append(f"sampling_probe.requests.{name}.{key} must be a string or null")
        for key in ("seconds", "completion_tokens"):
            if e.get(key) is not None and not _number(e[key]):
                out.append(f"sampling_probe.requests.{name}.{key} must be a number or null")
    return out


def _speed(e: dict):
    tokens, seconds = e.get("completion_tokens"), e.get("seconds")
    if _number(tokens) and _number(seconds) and seconds > 0 and tokens >= MIN_TOKENS:
        return tokens / seconds
    return None


def _rate(entries):
    """The drafter's accepted share over the log slices of these requests, or None when none of them holds a session line."""
    stats = drafter_log.drafter_stats("\n".join(e.get("drafter_log") or "" for e in entries))
    return stats["accept_rate"] if stats else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _side(entries: list[dict]) -> dict:
    return {"tokens_per_s": _mean([_speed(e) for e in entries]), "drafter_accept_rate": _rate(entries),
            "requests": len(entries)}


def _loops(text: str) -> bool:
    found = prompt_suite.findings({"id": "sampled", "content": text, "error": None})
    return any(f["kind"] in defaults.SUITE_BLOCKING_KINDS for f in found)


def _fmt(speed) -> str:
    return "an unmeasured speed" if speed is None else f"{speed:.1f} tokens/s"


# The card's wording for each verdict. It has no number: the card's Numbers table is the only place for one, and the error text of a refusal
# stays in the evidence file.
CARD_SENTENCES = {
    "greedy-refused": "The server refused the greedy request of the sampling probe, so this configuration was not measured for sampling.",
    "sampled-refused": "The server refused a sampled request (temperature above 0, top-p below 1, with a seed). The error text is in the evidence file.",
    "sampled-incoherent": "The server accepted a sampled request, and its text repeats itself or is unreadable.",
    "inconclusive": ("The probe could not compare the speed of greedy and sampled requests: a request had no usable token count or timing, "
                     "or a reply was too short."),
    "sampled-ignored": ("The server accepted sampled requests, but sampling was not applied: two requests that differ only in seed returned the "
                        "same text, so decoding is greedy."),
    "sampled-slower": "Sampled requests decoded more slowly than greedy requests. Both speeds are in the table above.",
    "sampled-full-speed": ("Sampled requests (temperature above 0, top-p below 1, with a seed) were served and two seeds gave different text. "
                           "Their speed is in the table above next to the speed of greedy requests."),
}


def summarize(record) -> dict:
    """`_summarize` plus `card_sentence`: the verdict's card wording, or None when the record is absent or malformed."""
    out = _summarize(record)
    return {**out, "card_sentence": CARD_SENTENCES.get(out["verdict"])}


def _summarize(record) -> dict:
    """The verdict for one record: {verdict, sentence, greedy, sampled, ratio, reasons}. Checks run in this order: a refused greedy
    request, a refused sampled request, incoherent sampled text, replies too short to judge, two seeds that gave the same text, a
    slower sampled speed, and otherwise full speed."""
    empty = {"greedy": None, "sampled": None, "ratio": None, "reasons": []}
    if record is None or (isinstance(record, dict) and not record.get("ran")):
        return {**empty, "verdict": "not-run", "sentence": "The sampling probe did not run."}
    reasons = problems(record)
    if reasons:
        return {**empty, "verdict": "malformed", "sentence": "The sampling probe record is malformed.", "reasons": reasons}
    reqs = record["requests"]
    g, a, b = (reqs[n] for n in ENTRY_NAMES)
    sampled = [e for e in (a, b) if e["accepted"]]
    out = {"greedy": _side([g]) if g["accepted"] else None, "sampled": _side(sampled) if sampled else None, "ratio": None, "reasons": []}
    gs, ss = (out["greedy"] or {}).get("tokens_per_s"), (out["sampled"] or {}).get("tokens_per_s")
    if gs and ss:
        out["ratio"] = ss / gs
    if not g["accepted"]:
        return {**out, "verdict": "greedy-refused", "sentence": f"The server refused the greedy request: {(g.get('error') or 'no error text')[:200]}"}
    refused = [e for e in (a, b) if not e["accepted"]]
    if refused:
        return {**out, "verdict": "sampled-refused",
                "sentence": f"The server refused a sampled request (temperature 1.0, top-p 0.9, with a seed): {(refused[0].get('error') or 'no error text')[:200]}"}
    if any(_loops(e["text"]) for e in (a, b)):
        return {**out, "verdict": "sampled-incoherent",
                "sentence": "The sampled request was accepted and its text repeats itself or is unreadable."}
    long_enough = all(_number(e.get("completion_tokens")) and e["completion_tokens"] >= MIN_TOKENS for e in (a, b))
    if not long_enough:
        return {**out, "verdict": "inconclusive",
                "sentence": f"A sampled reply was shorter than {MIN_TOKENS} tokens, which says too little about speed or about whether sampling was applied."}
    if a["text"] == b["text"]:
        return {**out, "verdict": "sampled-ignored",
                "sentence": ("The server accepted the sampled requests, but two requests that differ only in seed returned the same text. "
                             f"Sampling was not applied, so decoding was greedy. Greedy decoded at {_fmt(gs)}.")}
    if g["text"] in (a["text"], b["text"]):
        # At temperature 1 and top-p 0.9 over about 96 tokens, a sampled text that equals the greedy text is not a chance event.
        return {**out, "verdict": "sampled-ignored",
                "sentence": ("The server accepted the sampled requests, but a sampled request returned the greedy text. "
                             f"Sampling was not applied, so decoding was greedy. Greedy decoded at {_fmt(gs)}.")}
    if any(_speed(e) is None for e in (g, a, b)):
        return {**out, "verdict": "inconclusive",
                "sentence": ("The probe could not compare speeds: a request has no usable token count or timing, or a reply was shorter "
                             f"than {MIN_TOKENS} tokens.")}
    if out["ratio"] is not None and out["ratio"] < SLOW_RATIO:
        return {**out, "verdict": "sampled-slower",
                "sentence": (f"Sampled requests decoded at {_fmt(ss)} against {_fmt(gs)} for greedy requests, "
                             f"{1 / out['ratio']:.2f} times slower.")}
    return {**out, "verdict": "sampled-full-speed",
            "sentence": f"Sampled requests decoded at {_fmt(ss)} against {_fmt(gs)} for greedy requests, and two seeds gave different text."}
