"""Hint shapes from the arm E7 run (issue 677): unary minus on a ttnn tensor, five indices on a four-dimensional tensor, a slice outside the
padded shape, and a concat whose pieces disagree (partial rotary). The failed lines are the check's, with no path."""
from __future__ import annotations

import pytest

from orchard import thin_repair as tr

PORT = "import ttnn\n\n\ndef _rms_norm(model, prefix, x):\n    return x\n"


def failed(code: str, etype: str, message: str) -> str:
    return f"FAILED: the port failed at 'prefill': '[port_model.py:12: {code}] {etype}: {message}'\n"


CASES = [
    ("x2 = -x[..., :d]", "TypeError", "bad operand type for unary -: 'ttnn._ttnn.tensor.Tensor'", "ttnn.neg"),
    ("g = g_ttnn[:, :, t:t+1, :, :]", "IndexError", "Too many indices for tensor of dimension 4", "4-D"),
    ("y = x[:, :, :, :64]", "RuntimeError", "TT_FATAL @ slice_op.cpp: input_tensor.padded_shape()[i] >= slice_start[i] (slice_start[i] < padded_shape)",
     "rotary"),
    ("o = ttnn.concat([x_rot_emb, x_pass], dim=-1)", "RuntimeError", "TT_FATAL: Input tensors must have the same shape except in the concat dim",
     "rotary"),
]


@pytest.mark.parametrize("code,etype,message,needle", CASES)
def test_shape_gets_a_hint(code, etype, message, needle):
    failure = tr.parse_failed(failed(code, etype, message))
    assert failure is not None
    hint = tr.hint_for(failure, PORT)
    assert hint and needle.lower() in hint.lower(), hint


@pytest.mark.parametrize("code,etype,message", [CASES[0][:3], CASES[2][:3]])
def test_rotary_hints_point_at_the_skill_example(code, etype, message):
    hint = tr.hint_for(tr.parse_failed(failed(code, etype, message)), PORT)
    assert "ttnn.concat([ttnn.neg(x2), x1], dim=-1)" in hint and "graph-fusing" in hint


E9 = [
    ("x = ttnn.repeat(x, 1, 1, 1, CONFIG['hc_count'])", "TypeError", "\\nttnn.repeat(): incompatible function arguments.\\n\\nCalled with:\\n  repeat(ttnn.Tensor, int, int, int, int)", "ttnn.Shape"),
    ("A_log = model['dev'][f\"model.layers.{i}.linear_attn.A_log.weight\"]", "KeyError", "'model.layers.0.linear_attn.A_log.weight'", "suffix"),
]


@pytest.mark.parametrize("code,etype,message,needle", E9)
def test_e9_shape_gets_a_hint(code, etype, message, needle):
    hint = tr.hint_for(tr.parse_failed(failed(code, etype, message)), PORT)
    assert hint and needle.lower() in hint.lower(), hint
