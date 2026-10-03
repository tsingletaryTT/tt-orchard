#!/usr/bin/env python3
"""Write the sanitised event signature of one qwen-code transcript.

Usage: python3 scripts/extract_qwen_signature.py TRANSCRIPT OUT

OUT gets one JSON object per event: kinds, times, token counts, durations, tool names and
sha256 hashes. No message text is written. Every hash is salted with 16 random bytes that are
discarded when the script ends: equal content within one file still has equal hashes, which is
all the detectors compare, and nobody can test a guess against a committed hash. Prints the
number of events written.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The script runs from a checkout without installing the package, so the repo root goes on the
# import path. This affects only this script's own process.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchard.transcripts import qwen_events, write_signature  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src, out = argv
    with open(src, encoding="utf-8", errors="replace") as f:
        salt = os.urandom(16)                   # never written anywhere
        n = write_signature(qwen_events(f, salt=salt), out)
    print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
