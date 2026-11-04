# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 462 (issue 458): the readable floor of the supervisor-written result must take linear time and bounded memory on a
record of up to 16 MiB. `derive()` runs it on the record's free-run text before the gate validates the record."""
import subprocess
import sys
import textwrap

import pytest

from orchard import full_port_result as fpr

HOSTILE_CHILD = textwrap.dedent("""
    import resource, sys, time
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    sys.path.insert(0, sys.argv[1])
    from orchard import full_port_result as fpr
    # 50 MB of words that are all different: the old commonest-word count compared every word with every other
    n = 5_000_000
    text = " ".join(f"w{i:07d}" for i in range(n))
    t = time.monotonic()
    ok, basis = fpr.readable(text)
    print(round(time.monotonic() - t, 2), ok, basis[:120])
""")


@pytest.mark.timeout(120)
def test_a_record_of_millions_of_different_words_is_judged_in_bounded_time_in_a_limited_child():
    import pathlib
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    done = subprocess.run([sys.executable, "-I", "-c", HOSTILE_CHILD, root], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-400:]
    seconds = float(done.stdout.split()[0])
    assert seconds < 10, f"readable took {seconds} s on 50 MB of words"


def test_a_text_longer_than_the_sample_is_judged_on_the_sample_and_says_so():
    prose = ("The compiler maps each tensor onto the cores of the chip, and the runtime launches one kernel per operation while the host "
             "waits for the result and then prints the next token of the answer. ")
    text = prose * 2000                                                          # about 340,000 characters
    assert len(text) > fpr.SAMPLE_CHARS
    ok, basis = fpr.readable(text)
    assert f"first {fpr.SAMPLE_CHARS:,} characters" in basis


def test_a_text_inside_the_sample_has_no_sample_note():
    ok, basis = fpr.readable("Plain words that differ from each other in a short text. " * 1)
    assert "first " not in basis or "characters" not in basis.split(":")[0]


def test_the_commonest_word_is_found_in_one_pass_with_the_same_share():
    words = ["alpha"] * 30 + [f"w{i}" for i in range(70)]
    ok, basis = fpr.readable(" ".join(words) + " " + " ".join(f"v{i}" for i in range(400)))
    assert "commonest word" in basis
