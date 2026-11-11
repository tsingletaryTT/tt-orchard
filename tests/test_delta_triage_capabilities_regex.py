# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 671: a tensor pattern in a bundle's orchard-capabilities.json must not be able to hang stage 0.

The template compiles `layers.pattern`, `shapes[].tensor` and `optional_tensors[].tensor` and runs `fullmatch` on every tensor name. A pattern
with nested quantifiers (`(a|aa)+$`, `(a+)+b`) takes exponential time on a name that almost matches. The template now refuses such a pattern
with a named problem, by looking at the parsed pattern: no quantifier inside a quantifier, no alternation or lookaround or back-reference
under a quantifier, no two unbounded quantifiers in a row, at most two unbounded quantifiers in all. A name over `CAP_MAX_NAME` characters
takes no part in a match. Every refusal test here runs under a time limit so that a regression fails in a second and does not hang."""
from __future__ import annotations

import time

import pytest

from test_delta_triage_capabilities import LAYER, T, caps

LONG_A = "a" * 40 + "!"
BAD = [
    r"^(a|aa)+$",                      # the issue's example: alternation under a quantifier
    r"^(a+)+b$",                       # nested quantifier
    r"^(a*)*$",
    r"^(a|a)*b$",
    r"^(\d+)*x$",
    r"^model\.(layers|blocks)+\.x$",
    r"^(?:a{1,5}){1,5}b$",             # nested bounded quantifiers still multiply
    r"^.*.*x$",                        # two unbounded quantifiers in a row
    r"^\d+\d+x$",
    r"^a+\.b+\.c+x$",                  # three unbounded quantifiers, none next to another
    r"^(?=a)a+$",                      # a lookaround
    r"^(?P<x>a)(?P=x)+$",              # a back-reference
]
GOOD = [
    r"^model\.layers\.\d+\.mlp\.up_proj\.weight$",
    r"^model\.layers\.\d+\.mlp\.(gate|up)_proj\.weight$",
    r"^model\.layers\.\d+\.self_attn\.[kv]_proj\.weight$",
    r"^model\.layers\.\d+\.self_attn\.indexer\..+$",
    r"^model\.layers\.(?P<layer>\d+)\.(?P<rest>.+)$",
    r"^model\.layers\.(?P<layer>\d+)\..+$",
    r"^model\.embed_tokens\.weight$",
    r"^model\.layers\.\d{1,3}\.norm\.weight$",
]


def refused(pattern: str, field: str = "shapes") -> list:
    start = time.monotonic()
    if field == "shapes":
        data = caps(shapes=[{"tensor": pattern, "dims": {"0": "hidden_size"}}])
    elif field == "optional":
        data = caps(optional_tensors=[{"tensor": pattern, "feature": "f"}])
    else:
        data = caps(layers={"pattern": pattern, "count_key": "num_hidden_layers"})
    _, problems = T.validate_capabilities(data)
    assert time.monotonic() - start < 1.0, "validation itself must not take long"
    return problems


@pytest.mark.timeout(10)
@pytest.mark.parametrize("pattern", BAD)
def test_a_pattern_that_can_backtrack_badly_is_refused_with_a_named_problem(pattern):
    problems = refused(pattern)
    assert any("shapes[0]" in p and "backtrack" in p for p in problems), problems


@pytest.mark.timeout(10)
@pytest.mark.parametrize("field", ["shapes", "optional", "layers"])
def test_the_rule_applies_to_all_three_pattern_fields(field):
    pattern = r"^(?P<layer>a|aa)+$" if field == "layers" else r"^(a|aa)+$"
    problems = refused(pattern, field)
    assert any("backtrack" in p for p in problems), problems


@pytest.mark.timeout(10)
@pytest.mark.parametrize("pattern", GOOD)
def test_the_patterns_a_real_bundle_uses_are_still_accepted(pattern):
    problems = refused(pattern)
    assert not any("backtrack" in p for p in problems), problems


@pytest.mark.timeout(10)
def test_the_layer_pattern_of_the_shipped_fixtures_is_accepted():
    _, problems = T.validate_capabilities(caps())
    assert problems == []
    assert T.validate_capabilities(caps(layers={"pattern": LAYER, "count_key": "num_hidden_layers"}))[1] == []


@pytest.mark.timeout(10)
def test_the_refusal_names_what_is_allowed():
    problem = next(p for p in refused(r"^(a|aa)+$") if "backtrack" in p)
    assert "quantifier" in problem and "at most two unbounded" in problem


@pytest.mark.timeout(10)
def test_a_501_character_pattern_is_refused():
    """The length bound had no test: removing the check left the file's other tests green (issue 671, last paragraph)."""
    problems = refused("a" * (T.CAP_MAX_REGEX + 1))
    assert any("1 to 500 characters" in p for p in problems), problems
    assert not any("1 to 500 characters" in p for p in refused("a" * T.CAP_MAX_REGEX))


@pytest.mark.timeout(10)
def test_the_catastrophic_pattern_would_hang_without_the_rule():
    """Evidence that the patterns above are the real thing: compiled directly, the issue's pattern is exponential in the length of the name."""
    import re
    rx = re.compile(r"^(a|aa)+$")
    small = time.monotonic()
    rx.fullmatch("a" * 22 + "!")
    t22 = time.monotonic() - small
    small = time.monotonic()
    rx.fullmatch("a" * 26 + "!")
    t26 = time.monotonic() - small
    assert t26 > 4 * t22 > 0, (t22, t26)


@pytest.mark.timeout(10)
def test_a_name_longer_than_the_bound_takes_no_part_in_a_match():
    rx = T.re.compile(r"^model\.layers\.\d+\..+$")
    assert T.cap_fullmatch(rx, "model.layers.3.x") is not None
    at_bound = "model.layers.3." + "x" * (T.CAP_MAX_NAME - len("model.layers.3."))
    assert len(at_bound) == T.CAP_MAX_NAME and T.cap_fullmatch(rx, at_bound) is not None
    assert T.cap_fullmatch(rx, at_bound + "x") is None
