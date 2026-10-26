# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The speculative drafter's acceptance, read from the server's own log.

The DFlash2 serving code logs one line when a request on a slot ends, for example (the log prefix is left out)
  [dflash2-serve] slot 1 session: 56 iters, 276 committed tokens, accept 3.95/7 -> 4.93 tok/iter, 59.0 tok/s in-model (incl. seed); histogram [7, 10, 3, 5, 6, 3, 3, 19]
`accept A/K` is the mean number of drafted tokens accepted per iteration out of K drafted. The histogram counts iterations by how many drafted
tokens were accepted, so `accepted / (iterations * K)` is the share of drafted tokens the model accepted. A server of another kind prints no such
line, and then there is no number.

Used by bench_session.py (per benchmark phase) and measure_serving.py (per timed run, which the graft trial reads).
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

SESSION_LINE = re.compile(r"\[dflash2-serve\] slot \d+ session: (\d+) iters, (\d+) committed tokens, accept [0-9.]+/(\d+) -> [0-9.]+ tok/iter, "
                          r"([0-9.]+) tok/s in-model \(incl\. seed\); histogram \[([0-9, ]+)\]")

# The fields the graft trial copies into its result.
SUMMARY_KEYS = ("accept_rate", "accepted_per_iter", "draft_length", "iters", "sessions")


def drafter_stats(text: str):
    """Totals over the drafter's session lines in `text`, weighted by iterations, or None when there are none. A line that does not parse, or whose
    histogram does not have K + 1 counts that add up to its iterations, is skipped."""
    sessions = iters = committed = 0
    hist, k_seen, speeds = None, None, []
    for m in SESSION_LINE.finditer(text or ""):
        n_iters, n_committed, k, speed = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        counts = [int(x) for x in m.group(5).split(",")]
        if len(counts) != k + 1 or sum(counts) != n_iters or (k_seen not in (None, k)):
            continue
        k_seen = k
        hist = counts if hist is None else [a + b for a, b in zip(hist, counts)]
        sessions, iters, committed = sessions + 1, iters + n_iters, committed + n_committed
        if n_committed > 0:
            speeds.append(speed)
    if not sessions:
        return None
    accepted = sum(i * c for i, c in enumerate(hist))
    return {"sessions": sessions, "iters": iters, "committed_tokens": committed,
            "tokens_per_iter": committed / iters if iters else None, "draft_length": k_seen, "histogram": hist,
            "accepted_per_iter": accepted / iters if iters else None, "accept_rate": accepted / (iters * k_seen) if iters and k_seen else None,
            "tok_s_in_model_median": statistics.median(speeds) if speeds else None}


def size(path) -> int:
    """Bytes in the log now, or 0 when it does not exist. Taken before a measured stretch, it marks where the stretch starts."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def read_since(path, offset: int) -> str:
    """The log text from byte `offset` on, or "" when the file is missing, unreadable or shorter than the offset."""
    try:
        with open(path, "rb") as f:
            f.seek(max(int(offset), 0))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def summary(stats):
    """The compact record the graft trial keeps: the five SUMMARY_KEYS, or None when there are no stats."""
    if not isinstance(stats, dict) or not stats:
        return None
    return {k: stats.get(k) for k in SUMMARY_KEYS}
