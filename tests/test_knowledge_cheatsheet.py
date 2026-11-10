"""What arms C to E14 of the tiny Qwen4 stage 2 rehearsal taught, in the places a small coder reads (issue 677): the verified ttnn sheet and its
mistakes list, decoder_idioms.md, the attention stub request and the hint table. The catalog behind it: 106 distinct failure shapes over 18 arms."""
from __future__ import annotations

import ast
import re

import pytest

from orchard import full_port, op_cheats as oc, stub_prompts as sp, thin_repair as tr
from test_port_template import _load, tiny_qwen4

SHEET = oc.load_sheet(oc.TTNN_SHEET)
RENDERED = oc.render(SHEET)


# ---- the sheet -----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,needle", [
    ("ttnn.neg", "ttnn.neg(x)"),
    ("ttnn.repeat", "repeat_dims"),
    ("ttnn.permute", "one entry per axis"),
])
def test_the_sheet_has_an_entry_for_each_op_the_arms_guessed_wrong(name, needle):
    entry = oc.find_entry(SHEET, name)
    assert entry is not None, name
    assert needle in (entry["usage"] + " " + entry["notes"] + " " + entry["signature"])


@pytest.mark.parametrize("needle", [
    "has no attribute 'view'",                      # a ttnn tensor has no tensor methods but reshape and to
    "bad operand type for unary -",
    "incompatible function arguments",              # ttnn.repeat with one argument per dimension
    "must be Tensor, not ttnn",                     # a torch call on a ttnn tensor
    "padded widths must match",                     # a per-head norm weight against the full width
    "_to_host() missing",                           # the helpers' argument order
    "KeyError: 'model.layers.3.self_attn.weight'",  # a block path given where a projection path belongs
    "KeyError: 'head_k_dim'",                       # a config key that does not exist
    "KeyError: 'mesh_device'",                      # a key of model looked up in model['dev']
    "Too many indices for tensor of dimension 4",
    "attempted relative import",
])
def test_the_rendered_sheet_lists_each_recurring_failure_with_its_fix(needle):
    assert needle in RENDERED, needle


def test_every_mistake_has_a_fix_and_the_sheet_stays_well_formed_and_in_budget():
    assert oc.shape_problems(SHEET) == []
    assert len(RENDERED.splitlines()) < oc.RENDER_LINE_BUDGET


def test_the_repeat_note_shows_a_tuple_of_repeats_as_the_tree_uses_it_and_never_ttnn_shape():
    entry = oc.find_entry(SHEET, "ttnn.repeat")
    assert re.search(r"ttnn\.repeat\(\w+, \(1, 1, 1, \w+\)\)", entry["usage"] + entry["notes"])
    assert "ttnn.Shape" not in entry["usage"] + entry["notes"] + RENDERED


# ---- decoder_idioms.md names only helpers the generated port has ----------------------------------------------------------------------------

def _idiom_helper_names() -> list[str]:
    text = full_port.IDIOMS_SOURCE.read_text(encoding="utf-8")
    m = re.search(r"<!-- generated-port-functions: ([\w\s]+) -->", text)
    assert m, "decoder_idioms.md lists the generated port's functions in a `generated-port-functions` comment"
    return m.group(1).split()


def test_every_function_the_idioms_tell_the_reader_to_call_exists_in_the_generated_port_with_the_documented_arguments():
    text, _plan = tiny_qwen4()
    defined = {n.name: [a.arg for a in n.args.args] for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    names = _idiom_helper_names()
    assert {"_rope_tables", "_apply_rotary", "_rms_norm", "_group_rms_norm", "_to_host", "_upload", "_linear", "_gated_delta_net", "_ple",
            "_C_self_attn"} <= set(names)
    assert [n for n in names if n not in defined] == []
    idioms = full_port.IDIOMS_SOURCE.read_text(encoding="utf-8")
    for call in ("_to_host(model, t)", "_upload(model, t, layout)", "_apply_rotary(model, x, cos, sin)", "_rope_tables(model, n)",
                 "_C_self_attn(model, i, x, ids)"):
        fn = call.split("(")[0]
        assert call in idioms, call
        assert call.split("(", 1)[1].rstrip(")").split(", ") == defined[fn], (call, defined[fn])
    assert "_C_self_attn(model, i, mixed, ids)" in text                  # the idioms say that layer C passes the attention connection's output as x


# ---- the attention stub request ---------------------------------------------------------------------------------------------------------

def _attention_request():
    text, plan = tiny_qwen4()
    stub = next(s for s in plan.stubs if s.function == "_C_self_attn")
    return sp.stub_request(text, stub, sheet=SHEET, excerpts=None, inventory=_load("tiny-qwen4-ops-inventory.json"))["user"]


def test_the_attention_stub_request_has_an_ordered_recipe_with_the_layouts():
    user = _attention_request()
    assert "## How to write this block" in user
    for needle in ("_linear(model, prefix + \".q_proj\", x)", "ttnn.permute(x, (0, 2, 1, 3))", "(1, 1, S * heads, head_dim)", "_rms_norm",
                   "_apply_rotary(model, x, cos, sin)", "the block output with no residual add", "no `.view`"):
        assert needle in user, needle
    assert len(user) < 30_000


def test_a_stub_that_is_not_an_attention_gets_no_attention_recipe():
    text, _plan = tiny_qwen4()
    ple = {"function": "_B_ple", "cls": "Qwen4ExpTextPLELayer", "path": "model.layers.{i}.ple"}      # a function the port has, for another class
    user = sp.stub_request(text, ple, sheet=SHEET, excerpts=None)["user"]
    assert "Write the attention in this order" not in user              # (the sheet's mistakes, which print in every request, may mention permute)
    delta = {"function": "_A_linear_attn", "cls": "Qwen4ExpTextGatedDeltaNet", "path": "model.layers.{i}.linear_attn"}
    assert "Write the attention in this order" not in sp.stub_request(text, delta, sheet=SHEET, excerpts=None)["user"]


# ---- the hint table ---------------------------------------------------------------------------------------------------------------------------

def _failed(code, etype, message):
    return f"FAILED: the port failed at 'prefill': '[port_model.py:12: {code}] {etype}: {message}'\n"


PORT = "import ttnn\n\n\ndef _C_self_attn(model, i, x, ids):\n    return x\n"


@pytest.mark.parametrize("code,etype,message,needles", [
    ("q_gate_host = _to_host(q_gate)", "TypeError", "_to_host() missing 1 required positional argument: \\'t\\'", ("_to_host(model, t)",)),
    ("gate = _upload(model, gate_host)", "TypeError", "_upload() missing 1 required positional argument: \\'layout\\'",
     ("_upload(model, t, layout)",)),
    ("a, b = torch.chunk(q_gate, 2, dim=-1)", "TypeError", "chunk(): argument \\'input\\' (position 1) must be Tensor, not ttnn._ttnn.tensor.Tensor",
     ("_to_host(model, t)", "torch")),
    ("x = ttnn.repeat(x, 1, 1, 1, 4)", "TypeError", "\\nttnn.repeat(): incompatible function arguments.", ("(1, 1, 1, n)",)),
    ("w = model['dev'][prefix + '.weight']", "KeyError", "\\'model.layers.3.self_attn.weight\\'", ("projection", "q_proj")),
])
def test_hint_for_the_arm_failures_names_the_fix(code, etype, message, needles):
    failure = tr.parse_failed(_failed(code, etype, message))
    assert failure is not None
    hint = tr.hint_for(failure, PORT)
    assert hint, (etype, message)
    for n in needles:
        assert n in hint, (n, hint)
    assert "ttnn.Shape" not in hint
