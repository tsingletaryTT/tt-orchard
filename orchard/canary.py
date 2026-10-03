"""Canary prompts: ask a model server one fixed question and compare the answers.

This module owns the chat call the park sequence uses to check a server (spec section 6, steps 2
and 5) and the comparison rule. Decoding is greedy (temperature 0), so a restarted coder with the
same weights should give the same text, and the comparison is exact. A difference in whitespace
alone still fails. The result says so, so a person can judge it quickly. Whether the answer is
identical across a real restart is not yet measured (spec section 12).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from orchard.defaults import CANARY_MAX_TOKENS, CANARY_TIMEOUT_S


class CanaryError(Exception):
    """The server did not give a usable answer."""


def post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise CanaryError(f"{url} returned JSON that is not an object")
    return data


def ask(endpoint: str, model: str, prompt: str, *, http=post_json,
        timeout: float = CANARY_TIMEOUT_S, max_tokens: int = CANARY_MAX_TOKENS) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    try:
        data = http(url, body, timeout)
    except (OSError, ValueError) as exc:     # URLError and socket errors are OSError
        raise CanaryError(f"canary request to {url} failed: {exc}") from exc
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CanaryError(f"{url} gave no choices[0].message.content: {str(data)[:300]}") from exc
    if not isinstance(text, str):
        raise CanaryError(f"{url} gave content that is not text: {text!r}")
    return text


@dataclass(frozen=True)
class CanaryResult:
    match: bool
    whitespace_only: bool     # the texts differ, and only in whitespace
    before: str
    after: str


def compare(before: str, after: str) -> CanaryResult:
    match = before == after
    return CanaryResult(match=match, whitespace_only=(not match and before.split() == after.split()),
                        before=before, after=after)
