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


def test_a_failure_inside_a_function_a_block_family_wrote_is_not_sent_to_the_coder(tmp_path):
    """The generated hyper-connection functions carry GENERATED_BLOCK_MARK: a failure in one of them goes to the agent loop (issue 698)."""
    port = ("import ttnn\n\n\ndef _A_attn_hyper_connection(model, i, x, ids):\n    # GENERATED (not a stub): hyper_connection\n"
            "    return _hyper_connection(model, 'p', x)\n\n\ndef _A_linear_attn(model, i, x, ids):\n    return x\n")
    d = tmp_path / "stage"
    (d / "port").mkdir(parents=True)
    (d / "port" / "port_model.py").write_text(port)

    def post(*a, **k):
        raise AssertionError("no request may be sent for a generated block")
    r = tr.run_round(stage_dir=d, k=1, raw_output=_raw(5, "return _hyper_connection(model, 'p', x)"), post=post, endpoint="http://x/v1", model="m", sampling={})
    assert r.status == "fallback" and r.function == "_A_attn_hyper_connection" and r.requests == 0 and "generated" in r.reason


# ---- attribution to the calling stub (issue 698, item 2) ---------------------------------------------------------------------------------

CALLER_PORT = ("import ttnn\n\n\ndef _rms_norm(model, prefix, x):\n    return ttnn.rms_norm(x, epsilon=1e-6, weight=model['dev'][prefix])\n\n\n"
               "def _C_self_attn(model, i, x, ids):\n    q = _rms_norm(model, 'q_norm', x)\n    return q\n\n\n"
               "def _A_linear_attn(model, i, x, ids):\n    # GENERATED (not a stub): gated_delta_net\n    return _rms_norm(model, 'n', x)\n")


def _caller_stage(tmp_path, text):
    d = tmp_path / "stage"
    (d / "port").mkdir(parents=True)
    (d / "port" / "port_model.py").write_text(text)
    return d


def test_a_failure_in_a_fixed_helper_is_sent_to_the_one_stub_that_calls_it(tmp_path):
    sent = []

    def post(url, body, timeout=None):
        sent.append(body)
        return {"choices": [{"message": {"content": "```python\ndef _C_self_attn(model, i, x, ids):\n    return x\n```"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}}
    raw = "FAILED: the port failed at 'prefill': '[port_model.py:5: return ttnn.rms_norm(x, epsilon=1e-6, weight=model[\\'dev\\'][prefix])] RuntimeError: TT_FATAL: padded widths must match'\n"
    r = tr.run_round(stage_dir=_caller_stage(tmp_path, CALLER_PORT), k=1, raw_output=raw, post=post, endpoint="http://x/v1", model="m", sampling={})
    assert r.status == "spliced" and r.function == "_C_self_attn"
    text = sent[0]["messages"][-1]["content"]
    assert "_rms_norm" in text and "called by `_C_self_attn`" in text.replace("\n", " ")


def test_with_two_calling_stubs_the_round_still_falls_through(tmp_path):
    two = CALLER_PORT + "\n\ndef _B_ple(model, i, x, ids):\n    return _rms_norm(model, 'p', x)\n"

    def post(*a, **k):
        raise AssertionError("no request when the caller is not unique")
    raw = "FAILED: the port failed at 'prefill': '[port_model.py:5: return ttnn.rms_norm(x)] RuntimeError: x'\n"
    r = tr.run_round(stage_dir=_caller_stage(tmp_path, two), k=1, raw_output=raw, post=post, endpoint="http://x/v1", model="m", sampling={})
    assert r.status == "fallback" and "calls it" in r.reason


def test_a_generated_function_that_calls_the_helper_is_not_a_calling_stub(tmp_path):
    only_generated = CALLER_PORT.replace("def _C_self_attn(model, i, x, ids):\n    q = _rms_norm(model, 'q_norm', x)\n    return q",
                                         "def _C_self_attn(model, i, x, ids):\n    return x")

    def post(*a, **k):
        raise AssertionError("a generated caller is never a target")
    raw = "FAILED: the port failed at 'prefill': '[port_model.py:5: return ttnn.rms_norm(x)] RuntimeError: x'\n"
    r = tr.run_round(stage_dir=_caller_stage(tmp_path, only_generated), k=1, raw_output=raw, post=post, endpoint="http://x/v1", model="m", sampling={})
    assert r.status == "fallback"


def test_a_function_the_template_wrote_is_not_a_calling_stub_when_the_stub_names_are_recorded(tmp_path):
    """_mlp calls _linear in every port (the template wrote it); only a recorded stub counts, so one stub is a unique caller."""
    port = ("import ttnn\n\n\ndef _linear(model, prefix, x):\n    return ttnn.linear(x, model['dev'][prefix])\n\n\n"
            "def _mlp(model, prefix, x):\n    return _linear(model, prefix + '.up', x)\n\n\n"
            "def _C_self_attn(model, i, x, ids):\n    return _linear(model, 'attn', x)\n")
    stage = _caller_stage(tmp_path, port)
    (stage / "thin_stubs.json").write_text('["_C_self_attn"]')
    sent = []

    def post(url, body, timeout=None):
        sent.append(body)
        return {"choices": [{"message": {"content": "```python\ndef _C_self_attn(model, i, x, ids):\n    return x\n```"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}}
    raw = "FAILED: the port failed at 'prefill': '[port_model.py:5: return ttnn.linear(x, model[\\'dev\\'][prefix])] KeyError: \\'attn\\''\n"
    r = tr.run_round(stage_dir=stage, k=1, raw_output=raw, post=post, endpoint="http://x/v1", model="m", sampling={})
    assert r.status == "spliced" and r.function == "_C_self_attn"
    # without the record, _mlp is a second caller and the round falls through (the earlier behaviour)
    (stage / "thin_stubs.json").unlink()
    (stage / "port" / "port_model.py").write_text(port)          # the first round spliced a reply; start again from the same port
    r2 = tr.run_round(stage_dir=stage, k=1, raw_output=raw, post=post, endpoint="http://x/v1", model="m", sampling={})
    assert r2.status == "fallback"


def test_a_malformed_stub_record_is_ignored(tmp_path):
    stage = _caller_stage(tmp_path, CALLER_PORT)
    (stage / "thin_stubs.json").write_text('{"not": "a list"}')
    assert tr.recorded_stubs(stage) is None
    (stage / "thin_stubs.json").write_text('[1, 2]')
    assert tr.recorded_stubs(stage) is None
