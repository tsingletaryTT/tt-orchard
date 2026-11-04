# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 425, Copilot round 4: the URL user-information pattern in `skillsfresh.scrub_secrets` scanned a long run of letters quadratically (every
start position read to the end of the run before it failed to find `://`), and `port_repair.safe_text` runs it over a whole captured check
output, which can be 16 MiB, outside the watchdog. The pattern now limits the scheme to 32 characters, as the port check's own copy does.

Each hostile text runs in a child with a wall-clock limit, so a mutation that brings the quadratic scan back fails the test in seconds."""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = str(Path(__file__).resolve().parents[1])
LIMIT_S = 20

SHAPES = {
    "letters": '"a" * 2_000_000',
    "scheme_chars": '"a1+.-" * 400_000',
    "many_scheme_starts": '"a:/" * 700_000',
    "bare_slashes": '"://" * 700_000',
    "long_userinfo": '"http://" + "u" * 2_000_000',
    "at_signs": '"x://" + "@" * 1_000_000',
}


def child(entry: str, expr: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {REPO!r})
        from orchard import skillsfresh, port_repair
        text = {expr}
        t0 = time.time()
        out = {entry}(text)
        print(round(time.time() - t0, 2), len(out))
    """)
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=LIMIT_S + 10)


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("entry", ["skillsfresh.scrub_secrets", "port_repair.safe_text"])
def test_a_hostile_capture_is_scrubbed_in_bounded_time(entry, shape):
    try:
        got = child(entry, SHAPES[shape])
    except subprocess.TimeoutExpired:
        pytest.fail(f"{entry} on {shape} ran past {LIMIT_S + 10} s: a pattern scans quadratically")
    assert got.returncode == 0, got.stderr[-300:]
    assert float(got.stdout.split()[0]) < LIMIT_S, got.stdout


def test_the_bounded_scheme_still_removes_user_information():
    from orchard import skillsfresh
    assert skillsfresh.scrub_secrets("see https://user:pw@example.org/x") == "see https://example.org/x"
    assert skillsfresh.scrub_secrets("git+ssh://me@host/y") == "git+ssh://host/y"
    assert "pw" not in skillsfresh.scrub_secrets("x" * 100 + "://user:pw@h")      # a scheme longer than the bound is cut, the secret still goes
