"""The bundle scrub check (spec section 10): search for the hostname, tokens and home paths.

The first p150 repo exposed a hostname in an old revision (spec section 3), so the operator bundle
is searched before it is called ready. A hit blocks the bundle. The supervisor runs this check
itself; the stage 8 agent's own scrub notes are not trusted for it.

`ledger.jsonl` is skipped. It is the operator's copy of the run's record, it is never published,
and it holds absolute evidence paths written by the park sequence (orchard/handoff.py). The
operator-bundle skill tells the agent to say so in RESULTS.md.

What this does not find: a token in a format not listed below, a hostname written in another form
(an IP address, a fully qualified name that differs from `socket.gethostname()`), or anything
inside a binary file.
"""
from __future__ import annotations

import os
import re
import socket
from pathlib import Path

TOKEN_PATTERNS = (
    ("a Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
HOME_PATH = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|/root)(?=/|\b)")
SKIP_NAMES = frozenset({"ledger.jsonl"})


def scrub_text(text: str, *, hostname: str, home: str) -> list[str]:
    hits = []
    if hostname and hostname != "localhost" and re.search(rf"\b{re.escape(hostname)}\b", text):
        hits.append(f"the hostname {hostname!r}")
    for what, pattern in TOKEN_PATTERNS:
        if pattern.search(text):
            hits.append(what)
    if (home and home != "/" and home in text) or HOME_PATH.search(text):
        hits.append("an absolute home path")
    return hits


def scrub_bundle(bundle, *, hostname: str | None = None, home: str | None = None) -> list[str]:
    """Every hit in the bundle's text files, as 'relative/path: what was found'."""
    hostname = socket.gethostname() if hostname is None else hostname
    home = os.path.expanduser("~") if home is None else home
    root = Path(bundle)
    hits = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name in SKIP_NAMES:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        hits += [f"{f.relative_to(root)}: {what}" for what in scrub_text(text, hostname=hostname, home=home)]
    return hits
