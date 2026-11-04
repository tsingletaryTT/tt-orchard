# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The limits of orchard/skip_mask_helper.py that the hostile-input corpus cannot tell apart (PR 422).

The corpus checks that the helper answers every hostile input with one short object. A request that is too large gets such an answer
whether or not the cap exists, so the cap is asserted here: the helper asks stdin for at most REQUEST_MAX + 1 bytes, and a request that
needs more is refused by name."""
import json
import sys

from orchard import skip_mask_helper as helper


class _Buffer:
    """stdin.buffer that records every read and serves blanks. An unbounded read is an error, not a long wait."""

    def __init__(self):
        self.asked = []

    def read(self, n=-1):
        self.asked.append(n)
        if n is None or n < 0:
            raise AssertionError("an unbounded read of stdin")
        return b" " * n


class _Stdin:
    def __init__(self):
        self.buffer = _Buffer()


def _run(monkeypatch, capsys):
    stdin = _Stdin()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "argv", ["skip_mask_helper.py", "unused"])
    assert helper.main() == 0
    return stdin.buffer.asked, json.loads(capsys.readouterr().out.splitlines()[-1])


def test_the_request_cap_bounds_the_read_of_stdin(monkeypatch, capsys):
    asked, _ = _run(monkeypatch, capsys)
    assert asked == [helper.REQUEST_MAX + 1]


def test_a_request_over_the_request_cap_is_refused_by_name(monkeypatch, capsys):
    _, answer = _run(monkeypatch, capsys)
    assert answer["ok"] is False and "the request is too large" in answer["error"]
    assert "masks" not in answer
