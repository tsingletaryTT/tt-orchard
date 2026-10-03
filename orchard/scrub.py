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



# ---- the package scrub (plan 5, stage 7) --------------------------------------------------------
# A staged package is uploaded as it is, so it gets a stricter check than the operator bundle:
# besides the text search above, nothing produced by installing or serving may be in it. A tensor
# cache is keyed by layer name only, so a cache built for another model would serve that model's
# weights without an error. Weights are a pointer in the manifest and are never shipped. Wheels are
# binary and are not searched; orchard/package.py checks that each one is byte-identical to the
# wheel in the source bundle it came from.

FORBIDDEN_DIRS = {".tt_cache": "a tensor cache directory", "tensors": "a tensor cache directory",
                  "venv": "an installed venv", ".venv": "an installed venv",
                  ".python": "an installed interpreter", ".uv": "an installed uv",
                  ".hf": "a Hugging Face cache", ".cache": "a runtime cache",
                  "model-dir": "a built model directory"}
FORBIDDEN_SUFFIXES = {".tensorbin": "a tensor cache file", ".safetensors": "a weights file",
                      ".bin": "a weights or cache file", ".pt": "a weights file",
                      ".pth": "a weights file", ".gguf": "a weights file"}
BINARY_SUFFIXES = (".whl",)
NAMESPACE_FILES = frozenset({"README.md"})       # where the operator's repo id is meant to appear


def scrub_package(root, *, hostname: str | None = None, home: str | None = None,
                  namespace: str | None = None) -> list[str]:
    """Every hit in a staged package, as 'relative/path: what was found', sorted by path.

    A forbidden directory is reported once and not searched. `namespace` is the operator's Hugging
    Face namespace: it belongs in the card's serve and publish lines and nowhere else."""
    hostname = socket.gethostname() if hostname is None else hostname
    home = os.path.expanduser("~") if home is None else home
    root = Path(root)
    hits: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        for d in sorted(dirnames):
            rel = (here / d).relative_to(root).as_posix()
            if (here / d).is_symlink():
                hits.append((rel, "a symbolic link"))
            elif d in FORBIDDEN_DIRS:
                hits.append((rel, FORBIDDEN_DIRS[d]))
        dirnames[:] = [d for d in dirnames if d not in FORBIDDEN_DIRS and not (here / d).is_symlink()]
        for name in filenames:
            f = here / name
            rel = f.relative_to(root).as_posix()
            if f.is_symlink():
                hits.append((rel, "a symbolic link"))
                continue
            suffix = f.suffix.lower()
            if suffix in FORBIDDEN_SUFFIXES:
                hits.append((rel, FORBIDDEN_SUFFIXES[suffix]))
                continue
            if suffix in BINARY_SUFFIXES:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            hits += [(rel, what) for what in scrub_text(text, hostname=hostname, home=home)]
            if (namespace and rel not in NAMESPACE_FILES
                    and re.search(rf"(?<![\w.-]){re.escape(namespace)}/", text)):
                hits.append((rel, f"the operator's namespace {namespace!r} outside README.md"))
    return [f"{rel}: {what}" for rel, what in sorted(hits)]
