"""The stub request for an attention stub says how to get the rotary tables and how to apply them (issue 696, slice 4; arm E13 stubs read
`model['host']['cos_cached']`, which does not exist)."""
from __future__ import annotations

from orchard import op_cheats, port_template as pt, stub_prompts as sp
from test_port_template import _load, tiny_qwen4, std_generate

SHEET = op_cheats.load_sheet(op_cheats.TTNN_SHEET)


def test_an_attention_stub_request_names_the_rotary_helpers_and_the_layout(monkeypatch):
    monkeypatch.delitem(pt.BLOCK_FAMILIES, "qsa_attention")          # the indexer attention is written by a family; a stub is left for an attention no family recognizes
    text, plan = tiny_qwen4()
    stub = next(s for s in plan.stubs if s.function == "_C_self_attn")
    user = sp.stub_request(text, stub, sheet=SHEET, excerpts=None, inventory=_load("tiny-qwen4-ops-inventory.json"))["user"]
    assert "_rope_tables(model, S)" in user and "_apply_rotary(model, x, cos, sin)" in user
    assert "[1, heads, S, head_dim]" in user and "ROTARY['rot']" in user
    assert "cos_cached" in user                                 # the request says that no such table exists


def test_a_port_without_the_helpers_gets_no_rotary_advice():
    text = "import ttnn\n\n\ndef _linear(model, prefix, x):\n    return x\n"
    assert "_apply_rotary" not in sp.model_text(text)
