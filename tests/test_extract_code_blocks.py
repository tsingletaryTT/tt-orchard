"""A reply with an explanation block and one function block is usable (arm E8: five repairs were refused for 'the reply holds 2 code blocks')."""
from __future__ import annotations

import pytest

from orchard import stub_prompts as sp

FN = "```python\ndef _rms_norm(model, prefix, x):\n    return x\n```"


def test_one_block_with_a_def_is_taken_when_the_others_define_nothing():
    reply = "Example call:\n```python\ny = _rms_norm(model, 'a', x)\n```\nFix:\n" + FN
    assert sp.extract_code(reply).lstrip().startswith("def _rms_norm")


def test_two_blocks_that_both_define_functions_are_still_refused():
    other = "```python\ndef _other(model, x):\n    return x\n```"
    with pytest.raises(sp.SpliceError, match="2 code blocks"):
        sp.extract_code(FN + "\n" + other)


def test_three_blocks_with_no_def_are_refused():
    b = "```python\ny = 1\n```"
    with pytest.raises(sp.SpliceError, match="3 code blocks"):
        sp.extract_code(b + b + b)
