# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Shared shapes for the research collectors, and the two functions that make untrusted text safe to keep."""
from __future__ import annotations

import re
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orchard.scrub import scrub_text

STATED, INFERRED, WEAK = "stated", "inferred", "weak"
CONFIDENCE_RANK = {STATED: 0, INFERRED: 1, WEAK: 2}
MAX_CLAIM = 300
# Written into a tt-metal demo's claim by ports.demo_facts and read back by preflight._code_note.
LACKS_BLACKHOLE = "no Blackhole board named"

# C0 and C1 controls, and the bidirectional overrides and isolates that can reorder displayed text.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


@dataclass(frozen=True)
class Evidence:
    source: str          # "hub", "catalog", "tt-metal", "github", "bundle", "config", "headers"
    claim: str           # one cleaned line
    url: str
    confidence: str      # STATED, INFERRED or WEAK
    fetched_at: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Unavailable:
    source: str
    reason: str

    def as_dict(self) -> dict:
        # The reason is often the text of an exception or an HTTP answer, so it is cleaned here, once.
        return {"source": self.source, "reason": clean(self.reason)}


@dataclass
class Result:
    evidence: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def extend(self, other: "Result") -> "Result":
        self.evidence += other.evidence
        self.unavailable += other.unavailable
        return self


def clean(text, limit: int = MAX_CLAIM) -> str:
    """Fetched text is untrusted data: one line, no control characters, a length limit, and nothing the
    scrub would refuse (a home path, a host name, a token). A hit replaces the whole text."""
    one = " ".join(_CONTROL.sub(" ", str(text)).split())
    # Scrub the whole line before cutting it, so a token that straddles the limit cannot leave a prefix behind.
    hits = scrub_text(one, hostname=socket.gethostname(), home=str(Path.home()))
    return f"[removed: {', '.join(hits)}]" if hits else one[:limit]


def valid_id(value) -> bool:
    """A Hugging Face `org/name`. Ids from a card, a search or a listing are checked before they enter a URL."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None
