# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 of a full port, finish phase: `result.json` written by the supervisor from the check's own record (issue 456).

The finish phase used to be an agent step: the coder read `test-result.json` and `evidence/port-check.json` and wrote `result.json` from
them. Every field of that file is a copy of, or a number computed from, the record, and the gate (`orchard/full_port.py`) recomputes them
from the record anyway. The agent added nothing but one more reply that could go wrong: on 2026-11-03 a bring-up whose hardware test had
passed ended as `stage-failed` because the coder's finish replies were cut off at `max_tokens` four times.

`derive` builds the result only when the test passed and the record is complete. One field takes a judgement: `coherent`, whether the
free-run text is readable. `readable` answers it with a stated floor that checks form (printable text, mostly words, no word holding more than
a quarter of the text). The file says so, and a text that does not pass the floor is left to the agent, who reads it as before. A failed or
missing record is also left to the agent. Nothing here weakens the gate: the supervisor runs `full_port.check` on what it wrote, and a
result that does not pass is deleted before the agent step runs.
"""
from __future__ import annotations

import os
import zlib
from collections import Counter
from pathlib import Path

from orchard import full_port

DERIVED_BY = "supervisor"
# `readable` floor, with the numbers stated so a reader can judge it.
MIN_WORDS = 8                   # fewer words than this is not enough text to call readable (the gate itself asks for 4 different words)
MIN_PRINTABLE = 0.97            # share of non-space characters that are printable
MIN_WORDY = 0.80                # share of the words that hold at least one letter
MIN_DISTINCT = 0.35             # share of the words that differ from each other
MAX_TOP_WORD = 0.25             # largest share of the words that one word may hold
WORD_LENGTH = (2.0, 12.0)       # mean word length, in characters
MIN_ZLIB = 0.20                 # compression ratio of a text of 400 or more characters (loops compress far below this)
SAMPLE_CHARS = 200_000     # a longer text is judged on its first this many characters: the record may hold up to 16 MiB, and every step below is linear in the sample


def readable(text) -> tuple[bool, str]:
    """(readable, basis): whether `text` passes the floor above, and a sentence that says what was measured. The floor checks form only."""
    if not isinstance(text, str):
        return False, "the free-run text is not text"
    sample = text[:SAMPLE_CHARS]
    words = sample.split()
    if len(words) < MIN_WORDS:
        return False, f"{len(words)} words, fewer than the {MIN_WORDS} the floor needs"
    chars = [c for c in sample if not c.isspace()]
    printable = sum(1 for c in chars if c.isprintable()) / len(chars)
    wordy = sum(1 for w in words if any(c.isalpha() for c in w)) / len(words)
    counts = Counter(words)                                   # one pass; the sample bounds the number of distinct words
    distinct = len(counts) / len(words)
    top = counts.most_common(1)[0][1] / len(words)
    mean = sum(len(w) for w in words) / len(words)
    raw = sample.encode("utf-8", "replace")
    ratio = len(zlib.compress(raw, 6)) / max(1, len(raw))
    scope = f" (first {SAMPLE_CHARS:,} characters of {len(text):,})" if len(text) > SAMPLE_CHARS else ""
    facts = (f"{len(words)} words{scope}; {printable:.0%} printable, {wordy:.0%} hold a letter, {distinct:.0%} different, the commonest word "
             f"{top:.0%}, mean length {mean:.1f}, compresses to {ratio:.2f}")
    if printable < MIN_PRINTABLE:
        return False, f"{facts}: below the printable floor of {MIN_PRINTABLE:.0%}"
    if wordy < MIN_WORDY:
        return False, f"{facts}: fewer than {MIN_WORDY:.0%} of the words hold a letter"
    if distinct < MIN_DISTINCT:
        return False, f"{facts}: fewer than {MIN_DISTINCT:.0%} of the words differ"
    if top > MAX_TOP_WORD:
        return False, f"{facts}: one word holds more than {MAX_TOP_WORD:.0%} of the text"
    if not WORD_LENGTH[0] <= mean <= WORD_LENGTH[1]:
        return False, f"{facts}: the mean word length is outside {WORD_LENGTH[0]} to {WORD_LENGTH[1]}"
    if len(sample) >= 400 and ratio < MIN_ZLIB:
        return False, f"{facts}: it repeats itself (a ratio under {MIN_ZLIB})"
    return True, facts


def derive(stage_dir, run_dir, *, rehearsal: bool = False) -> tuple[dict | None, str]:
    """(result, why): the content of `result.json` for a stage 2 directory whose hardware test passed and whose record is complete,
    or (None, why not). The numbers are the record's; `top1_agreement` is recomputed from its ids exactly as the gate does. `rehearsal` is
    the run's `--full-port-rehearsal` (the caller reads it from the ledger): a model with random weights never writes readable text, the
    gate waives `coherent` for it, so the supervisor writes the result with `coherent` false and the agent is not asked."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    problems: list[str] = []
    marker = full_port._json(stage_dir / "test-result.json", "test-result.json", problems)       # the gate's own bounded reader
    if not isinstance(marker, dict):
        return None, "test-result.json is missing or unreadable"
    code = marker.get("returncode")
    if not (type(code) is int and code == 0 and marker.get("timed_out") is False):
        return None, "the hardware test did not pass"
    record = full_port._json(stage_dir / full_port.RECORD_FILE, full_port.RECORD_FILE, problems)
    if not isinstance(record, dict):
        return None, f"{full_port.RECORD_FILE} is missing or unreadable"
    ref, chip, load, text = record.get("reference_ids"), record.get("chip_ids"), record.get("load_s"), record.get("free_run_text")
    if not (full_port._ids(ref) and full_port._ids(chip)) or not chip or len(ref) != len(chip):
        return None, "the record's reference_ids and chip_ids are not matching lists of whole numbers"
    if not full_port._num(load) or load <= 0:
        return None, "the record's load_s is not a positive number"
    if not isinstance(text, str) or not text.strip():
        return None, "the record has no free-run text"
    ok, basis = readable(text)
    if not ok and not rehearsal:
        return None, f"the free-run text did not pass the readable floor ({basis}); the agent reads it"
    if not ok:
        basis = f"not passed ({basis}); this is a rehearsal model with random weights and the gate waives coherent"
    evidence = [os.path.relpath(stage_dir / full_port.RECORD_FILE, run_dir), os.path.relpath(stage_dir / "evidence" / "hw-test-output.txt", run_dir)]
    return ({"runs": True, "load_s": load, "coherent": ok, "free_run_text": text,
             "top1_agreement": full_port.agreement(ref, chip), "n_tokens": len(chip), "evidence": evidence,
             "derived_by": DERIVED_BY,
             "coherent_basis": f"supervisor floor (form only: printable text, words, repetition): {basis}"}, "derived")
