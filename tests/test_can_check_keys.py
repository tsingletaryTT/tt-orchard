# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `[check]` keys `reasoning`, `prefix_caching` and `quirks` (issues 360 and 372) are read and validated when the spec loads."""
import pytest

from orchard import can
from test_can_vllm import base, edit  # noqa: F401  (the fixture and the helper of the vllm spec tests)

ANCHOR = 'request_extra = { chat_template_kwargs = { enable_thinking = false } }'


def test_the_keys_load():
    pass


def test_reasoning_prefix_caching_and_quirks_load(base):
    s = edit(base, ANCHOR, ANCHOR + '\nreasoning = true\nprefix_caching = true\nquirks = [{ model = "someone/toy-moe", server_release = "0.26.0", note = "n" }]')
    assert s.check.reasoning is True and s.check.prefix_caching is True
    assert s.check.quirks == ({"model": "someone/toy-moe", "server_release": "0.26.0", "note": "n"},)


def test_they_default_to_off_and_empty(base):
    s = can.load_spec(base / "can.toml", base=base)
    assert (s.check.reasoning, s.check.prefix_caching, s.check.quirks) == (False, False, ())


@pytest.mark.parametrize("line,words", [
    ('reasoning = "yes"', "reasoning must be true or false"),
    ('reasoning = 1', "reasoning must be true or false"),
    ('prefix_caching = "true"', "prefix_caching must be true or false"),
    ('quirks = "x"', "quirks must be a list"),
    ('quirks = [{ model = "m" }]', "quirks must be a list"),
    ('quirks = [{ model = "m", server_release = "r", note = "n", extra = "x" }]', "quirks must be a list"),
    ('quirks = [{ model = "m", server_release = "r", note = "" }]', "quirks must be a list"),
    ('quirks = [{ model = "m", server_release = "r", note = 3 }]', "quirks must be a list"),
])
def test_a_bad_value_is_refused_with_the_key_named(base, line, words):
    with pytest.raises(can.CanError, match=words):
        edit(base, ANCHOR, ANCHOR + "\n" + line)


def test_more_than_fifty_quirks_or_a_long_note_is_refused(base):
    many = ", ".join('{ model = "m", server_release = "r", note = "n" }' for _ in range(51))
    with pytest.raises(can.CanError, match="quirks"):
        edit(base, ANCHOR, ANCHOR + f"\nquirks = [{many}]")
    with pytest.raises(can.CanError, match="quirks"):
        edit(base, ANCHOR, ANCHOR + '\nquirks = [{ model = "m", server_release = "r", note = "' + "n" * 501 + '" }]')
