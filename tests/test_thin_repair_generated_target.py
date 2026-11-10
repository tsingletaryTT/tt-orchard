"""A failing line inside a function the generator always writes is not sent to the coder (system review 2026-11-10, issue 698): arm E8 spent 11
rounds on _rms_norm for a table entry that a constant above it lacked."""
from __future__ import annotations

from pathlib import Path

from orchard import port_template as pt, thin_repair as tr

PORT = ("import ttnn\n\n\ndef _rms_norm(model, prefix, x):\n    return ttnn.rms_norm(x, epsilon=NORM_EPS[prefix])\n\n\n"
        "def _A_linear_attn(model, i, x):\n    return x\n")


def _stage(tmp_path: Path) -> Path:
    d = tmp_path / "stage"
    (d / "port").mkdir(parents=True)
    (d / "port" / "port_model.py").write_text(PORT)
    return d


def _raw(line: int, code: str) -> str:
    return f"FAILED: the port failed at 'prefill': '[port_model.py:{line}: {code}] KeyError: \\'x\\''\n"


def _round(tmp_path, line, code):
    def post(*a, **k):
        raise AssertionError("no request may be sent for a generated helper")
    return tr.run_round(stage_dir=_stage(tmp_path), k=1, raw_output=_raw(line, code), post=post, endpoint="http://x/v1", model="m", sampling={})


def test_a_failure_inside_a_fixed_helper_falls_through_to_the_agent_loop_without_a_request(tmp_path):
    r = _round(tmp_path, 5, "return ttnn.rms_norm(x, epsilon=NORM_EPS[prefix])")
    assert r.status == "fallback" and r.function == "_rms_norm" and r.requests == 0
    assert "generated helper" in r.reason


def test_a_failure_inside_a_stub_function_is_still_repaired(tmp_path):
    sent = []

    def post(url, body, timeout=None):
        sent.append(body)
        return {"choices": [{"message": {"content": "```python\ndef _A_linear_attn(model, i, x):\n    return x\n```"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}}
    r = tr.run_round(stage_dir=_stage(tmp_path), k=1, raw_output=_raw(9, "return x"), post=post, endpoint="http://x/v1", model="m", sampling={})
    assert sent and r.status == "spliced"


def test_the_fixed_helper_names_are_in_the_template():
    assert all(name in pt.FIXED_HELPERS for name in ("_rms_norm", "_linear", "_embedding", "_upload", "_to_host", "_pattern"))
