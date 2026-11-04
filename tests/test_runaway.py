# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 458: runaway coder output. The pure parts of orchard/runaway.py: the repetition test, the file caps and the retry profile."""
import glob
import random
from pathlib import Path

import pytest

from orchard import runaway

ROOT = Path(__file__).resolve().parent.parent

# The paragraph a real capped reply repeated 63 times (run of 2026-11-03; paraphrased, same shape and length).
PARAGRAPH = ("Looking at the structure:\n- Position 5350: `model = {` (opens model dictionary)\n"
             "- Position 17628: `}` (closes something, but not the model dictionary)\n"
             "- Position 17630-17632: `model[\"layers\"].append(layer)` (no closing brace for model)\n")


def test_a_paragraph_repeated_sixty_times_is_a_loop():
    text = "Now I can see the issue clearly.\n" + PARAGRAPH * 60
    assert len(text) >= runaway.MIN_CHARS
    reason = runaway.loop_reason(text)
    assert reason and "degenerate repetition" in reason and "compress" in reason


def test_a_loop_is_caught_by_the_line_test_when_compression_cannot_see_it():
    # 15 random lines of 3,000 characters repeat every 45 KB, which is outside zlib's 32 KB window: the ratio stays near 1 and only the
    # line test can see it.
    rnd = random.Random(7)
    cycle = ["".join(rnd.choice("0123456789abcdef") for _ in range(3000)) for _ in range(15)]
    text = "\n".join(cycle * 8)
    assert runaway.repetition(text)["zlib_ratio"] > runaway.RATIO_MAX
    reason = runaway.loop_reason(text)
    assert reason and "only" in reason and "different" in reason


def test_a_text_below_the_minimum_is_never_called_a_loop():
    assert runaway.loop_reason(PARAGRAPH * 20) is None            # about 3,000 characters of pure repetition
    assert runaway.loop_reason("x" * (runaway.MIN_CHARS - 1)) is None
    assert runaway.loop_reason(None) is None and runaway.loop_reason(b"x" * 100000) is None


def test_no_source_test_or_doc_file_of_this_repository_is_called_a_loop():
    """The calibration: the thresholds were chosen from these files (lowest zlib ratio 0.175) and the real replies."""
    checked = 0
    for pattern in ("orchard/**/*.py", "tests/*.py", "docs/**/*.md", "README.md"):
        for path in glob.glob(str(ROOT / pattern), recursive=True):
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            if len(text) >= runaway.MIN_CHARS:
                checked += 1
                assert runaway.loop_reason(text) is None, path
    assert checked > 300


def test_prose_and_json_of_realistic_shapes_are_not_loops():
    rnd = random.Random(3)
    words = [f"w{rnd.randrange(5000)}" for _ in range(4000)]
    assert runaway.loop_reason(" ".join(words)) is None
    rows = ",\n".join(f'  {{"name": "layers.{i}.mlp.w{i % 7}.weight", "shape": [{i}, {i * 3}], "dtype": "BF16"}}' for i in range(400))
    assert runaway.loop_reason("[\n" + rows + "\n]") is None      # a tensor table: repetitive, but with different numbers on every row


# ---- the caps on a file ----

def test_a_file_over_the_byte_cap_is_refused_with_its_size_and_the_limit():
    reason = runaway.file_problem("a" * (runaway.MAX_FILE_BYTES + 1))
    assert reason and f"{runaway.MAX_FILE_BYTES + 1:,} bytes" in reason and f"{runaway.MAX_FILE_BYTES:,}" in reason


def test_the_byte_cap_counts_bytes_not_characters():
    rnd = random.Random(5)
    letters = [chr(c) for c in range(0x100, 0x2c0)]                 # two bytes each in UTF-8, so the text does not compress to a loop
    text = "".join(rnd.choice(letters) for _ in range(runaway.MAX_FILE_BYTES // 2 + 1))
    assert len(text) < runaway.MAX_FILE_BYTES and len(text.encode()) > runaway.MAX_FILE_BYTES
    assert runaway.loop_reason(text, min_chars=20_000, ratio_max=runaway.FILE_RATIO_MAX) is None      # only the byte cap can refuse it
    reason = runaway.file_problem(text)
    assert reason and "bytes" in reason


def test_a_file_over_the_line_cap_is_refused():
    reason = runaway.file_problem("\n" * runaway.MAX_FILE_LINES)
    assert reason and "lines" in reason


def test_a_file_at_each_cap_passes_and_a_looped_file_is_refused():
    assert runaway.file_problem("x = 1\n" * 1000) is None
    table = "\n".join(f"line {i} " + "y" * 40 for i in range(runaway.MAX_FILE_LINES - 1))     # regular, compresses to about 0.05
    assert runaway.file_problem(table) is None
    assert runaway.loop_reason(table) is not None                  # the same text as a REPLY is judged by the looser threshold
    loop = "def f():\n    return 1\n" * 2000                       # 44,000 characters, one block
    assert "degenerate repetition" in runaway.file_problem(loop)
    assert runaway.file_problem(12345) is None


# ---- the retry profile ----

def test_the_retry_profile_replaces_greedy_decoding_and_lowers_the_cap():
    request = {"model": "m", "messages": [], "temperature": 0, "max_tokens": 16384, "stream": False}
    out = runaway.Unstick().apply(request)
    assert out["temperature"] == 0.5 and out["max_tokens"] == 8192 and out["frequency_penalty"] == 0.2
    assert "repetition_penalty" not in out
    assert request["temperature"] == 0 and request["max_tokens"] == 16384           # the original is not changed


def test_the_retry_never_raises_a_cap_the_step_already_lowered():
    assert runaway.Unstick(max_tokens=8192).apply({"max_tokens": 2048})["max_tokens"] == 2048


def test_penalties_are_sent_only_when_set():
    out = runaway.Unstick(frequency_penalty=None, repetition_penalty=1.1).apply({"max_tokens": 100})
    assert "frequency_penalty" not in out and out["repetition_penalty"] == 1.1
    assert runaway.Unstick().record() == {"temperature": 0.5, "max_tokens": 8192, "frequency_penalty": 0.2}


def test_a_tier_can_set_the_profile_and_a_bad_value_falls_back_to_the_default():
    base = runaway.Unstick()
    assert runaway.unstick_from_tier({}) == base
    tier = {"unstick_temperature": 0.7, "unstick_max_tokens": 4096, "unstick_frequency_penalty": 0.4, "unstick_repetition_penalty": 1.05}
    assert runaway.unstick_from_tier(tier) == runaway.Unstick(0.7, 4096, 0.4, 1.05)
    bad = {"unstick_temperature": True, "unstick_max_tokens": 99999999, "unstick_frequency_penalty": "high", "unstick_repetition_penalty": 99}
    assert runaway.unstick_from_tier(bad) == base


def test_the_largest_reply_that_ended_normally_fits_in_the_retry_cap():
    """5,463 tokens was the largest ordinary reply in 389 measured replies (a whole port file in one call)."""
    assert runaway.Unstick().max_tokens > 5463 * 1.4
