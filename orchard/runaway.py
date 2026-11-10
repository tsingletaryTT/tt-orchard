# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runaway output from a coder model: what to refuse, and how to retry a reply that was cut off (issues 458 and 456).

On 2026-11-03 two full-port bring-ups on a new model ended in stage 2 because the coder's replies were cut off at `max_tokens` (16,384
tokens each, four times in one stage). The text of each capped reply was one paragraph repeated about 60 times: 15 distinct lines out of
757. The step had run greedy decoding (temperature 0) on a context of 35,000 to 42,000 tokens. The retry after a capped reply sent the
same context with the same sampling, so it produced the same loop, and the step ended after two.

This module holds the pure parts: a test for degenerate repetition, the size caps for a file an agent writes, and the sampling that a
retry uses once a reply has been capped. Nothing here changes a normal request: the retry profile applies only after a capped reply, a
case that otherwise ends the step.

How the thresholds were chosen (measured on 389 coder replies from three runs and on 750 files of this repository):
- The 8 capped replies compress (zlib level 6) to 0.014 to 0.025 of their size.
- The only ordinary reply of more than 2,000 characters compressed to 0.11. The most repetitive of 750 source, test and doc files
  compressed to 0.175.
- The largest reply that ended normally was 5,463 tokens; the 99th percentile was 5,203 tokens (a whole port file written in one call).
So a text of at least `MIN_CHARS` characters that compresses to under `RATIO_MAX` is a loop, and a retry limited to `UNSTICK_MAX_TOKENS`
still has room for the largest legitimate reply.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

# ---- what an agent may write to a file ----
MAX_FILE_BYTES = 400_000        # choice: a real port is tens of KB (the rehearsal's was 21.8 KB); this is 18 times that
MAX_FILE_LINES = 6_000          # choice: about 12 times the longest port written so far (493 lines)

# ---- degenerate repetition ----
MIN_CHARS = 8_000               # below this a short reply is never called a loop: compression is noisy on small texts
RATIO_MAX = 0.06                # measured: loops 0.014 to 0.025; ordinary text 0.11 and up; repository files 0.175 and up
FILE_MIN_CHARS = 20_000         # a file is judged from this size
FILE_RATIO_MAX = 0.04           # and at this ratio: stricter than for a reply (see file_problem)
DISTINCT_MIN_LINES = 100        # line test: at least this many non-empty lines
DISTINCT_MAX = 0.20             # line test: at most this share of the lines are different from each other


def repetition(text: str) -> dict:
    """Measurements of how repetitive `text` is: its length, its zlib ratio, and the share of different lines among the non-empty ones."""
    raw = text.encode("utf-8", "replace")
    ratio = len(zlib.compress(raw, 6)) / len(raw) if raw else 1.0
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    distinct = len(set(lines)) / len(lines) if lines else 1.0
    return {"chars": len(text), "lines": len(lines), "zlib_ratio": round(ratio, 4), "distinct_lines": round(distinct, 4)}


def loop_reason(text: str, *, min_chars: int = MIN_CHARS, ratio_max: float = RATIO_MAX) -> str | None:
    """Why `text` is a repetition loop, or None. Only a text of at least `min_chars` characters is judged."""
    if not isinstance(text, str) or len(text) < min_chars:
        return None
    m = repetition(text)
    if m["zlib_ratio"] < ratio_max:
        return (f"it is degenerate repetition: {m['chars']:,} characters compress to {m['zlib_ratio']:.3f} of their size "
                f"(ordinary text and code compress to 0.1 or more)")
    if m["lines"] >= DISTINCT_MIN_LINES and m["distinct_lines"] <= DISTINCT_MAX:
        return (f"it is degenerate repetition: {m['lines']:,} non-empty lines, only {m['distinct_lines']:.0%} of them different")
    return None


def file_problem(content: str) -> str | None:
    """Why a file an agent writes must be refused, or None. The caps are on bytes, lines, and repetition."""
    if not isinstance(content, str):
        return None
    size = len(content.encode("utf-8", "replace"))
    if size > MAX_FILE_BYTES:
        return f"it is {size:,} bytes; the limit for a file you write is {MAX_FILE_BYTES:,}"
    lines = content.count("\n") + 1
    if lines > MAX_FILE_LINES:
        return f"it is {lines:,} lines; the limit for a file you write is {MAX_FILE_LINES:,}"
    # A file is judged from a larger size and a stricter ratio than a reply: refusing a legitimate file costs the agent a rewrite, a reply
    # that is cut off costs minutes of decoding. All 8 measured loops compressed to under 0.03; a regular generated table is near 0.05.
    return loop_reason(content, min_chars=FILE_MIN_CHARS, ratio_max=FILE_RATIO_MAX)


# ---- the retry after a capped reply ----

@dataclass(frozen=True)
class Unstick:
    """Sampling for the request that follows a reply cut off at `max_tokens`. A value of None is not sent. `temperature` replaces greedy
    decoding (which repeats exactly the loop it just produced); `max_tokens` bounds the cost of a second loop; the penalties are sent only
    when set, because not every server accepts them (a request the server refuses is sent again without the profile)."""
    temperature: float = 0.5
    max_tokens: int = 8_192
    frequency_penalty: float | None = 0.2
    repetition_penalty: float | None = None

    def apply(self, request: dict) -> dict:
        """A copy of `request` with this profile applied."""
        out = dict(request)
        out["temperature"] = self.temperature
        out["max_tokens"] = min(int(out.get("max_tokens") or self.max_tokens), self.max_tokens)
        if self.frequency_penalty is not None:
            out["frequency_penalty"] = self.frequency_penalty
        if self.repetition_penalty is not None:
            out["repetition_penalty"] = self.repetition_penalty
        return out

    def record(self) -> dict:
        """The profile as the ledger holds it."""
        return {k: v for k, v in (("temperature", self.temperature), ("max_tokens", self.max_tokens),
                                  ("frequency_penalty", self.frequency_penalty),
                                  ("repetition_penalty", self.repetition_penalty)) if v is not None}


def _number(value, low: float, high: float):
    return None if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high else float(value)


def unstick_from_tier(tier: dict) -> Unstick:
    """The retry profile for a tier: the defaults, with any `unstick_*` value the tier sets (validated by orchard/tiers.py)."""
    base = Unstick()
    pick = lambda key, low, high, default: (lambda v: default if v is None else v)(_number(tier.get(key), low, high))
    tokens = tier.get("unstick_max_tokens")
    return Unstick(temperature=pick("unstick_temperature", 0.0, 2.0, base.temperature),
                   max_tokens=tokens if isinstance(tokens, int) and not isinstance(tokens, bool) and 256 <= tokens <= 65_536 else base.max_tokens,
                   frequency_penalty=pick("unstick_frequency_penalty", -2.0, 2.0, base.frequency_penalty),
                   repetition_penalty=pick("unstick_repetition_penalty", 0.5, 2.0, base.repetition_penalty))


# The fixed start of the text an agent step ends with when the coder's replies ran away across a stage (issue 458, item 4). The supervisor blocks the
# run with it, and orchard/blocked.py reads it before any other word: the text that follows names the chip tier and the coder, which other rows
# look for. It lives here, a module both the agent and the block classifier can import.
REPLIES_CAPPED_MARK = "the coder's replies keep running away"
