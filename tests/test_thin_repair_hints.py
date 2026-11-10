"""More error shapes for the thin repair hint table (issue 677, arm E3). Each shape is a failure the arm E2 run showed on the chip: a relative
import in a generated port, a torch call on a ttnn tensor, a ttnn.rms_norm whose weight is narrower than the input, and a lookup of a pattern
string that still holds {i}. The failed lines are the check's, with no path."""
from __future__ import annotations

import pytest

from orchard import stub_prompts, thin_repair as tr

PORT = "import ttnn\n\n\ndef _rms_norm(model, prefix, x):\n    return x\n\n\ndef _layer_A(model, i, x, ids):\n    return x\n"


def failed(op: str, code: str, etype: str, message: str, line: int = 12) -> str:
    return f"FAILED: the port failed at '{op}': '[port_model.py:{line}: {code}] {etype}: {message}'\nPORT_CHECK_FAILURE {{\"kind\": \"port\", \"op\": \"{op}\"}}\n"


CASES = [
    ("init", "from .helpers import _upload", "ImportError", "attempted relative import with no known parent package",
     "one module", "no relative import"),
    ("prefill", "position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).view(1, 1, -1)", "TypeError",
     "arange() received an invalid combination of arguments - got (int, device=nanobind.nb_bound_method, dtype=torch.dtype)",
     "ttnn tensor", "no torch device"),
    ("prefill", "return ttnn.rms_norm(x, epsilon=epsilon, weight=weight)", "RuntimeError",
     "TT_FATAL: a.padded_shape()[-1] == gamma.value().padded_shape()[-1] info: Input and gamma padded widths must match",
     "group_size", "reshape"),
    ("prefill", "epsilon = NORM_EPS[_pattern(prefix)]", "KeyError", "'model.layers.{i}.attn_hyper_connection.hc_norm'",
     "{i}", "real name"),
]


@pytest.mark.parametrize("op,code,etype,message,needle_a,needle_b", CASES)
def test_each_new_error_shape_gets_a_hint(op, code, etype, message, needle_a, needle_b):
    failure = tr.parse_failed(failed(op, code, etype, message))
    assert failure is not None and failure.etype == etype
    hint = tr.hint_for(failure, PORT)
    assert hint, "no hint for " + etype
    low = hint.lower()
    assert needle_a.lower() in low and needle_b.lower() in low, hint
    assert "/home/" not in hint


def test_a_shape_that_has_no_entry_still_gets_no_hint():
    failure = tr.parse_failed(failed("prefill", "y = x + 1", "ZeroDivisionError", "division by zero"))
    assert tr.hint_for(failure, PORT) is None


def test_the_hint_for_a_key_error_names_the_helper_that_makes_the_pattern_key():
    failure = tr.parse_failed(failed("prefill", "epsilon = NORM_EPS[_pattern(prefix)]", "KeyError", "'model.layers.{i}.mlp.up_proj'"))
    hint = tr.hint_for(failure, PORT)
    assert "_pattern" in hint and "NORM_EPS" in hint


def test_the_rules_say_the_port_is_one_module_and_an_activation_is_a_ttnn_tensor():
    text = stub_prompts.RULES.lower()
    assert "relative import" in text and "ttnn tensor" in text and "no torch device" in text
