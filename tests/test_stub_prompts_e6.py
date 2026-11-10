"""Arm E6 (issue 677): the arm E5 run failed on two things the prompt can state. A layer stub called the stubs of its parts with the wrong
number of arguments (`_A_linear_attn() takes 4 positional arguments but 6 were given`), and stubs read CONFIG keys that the config does
not have (`CONFIG['head_k_dim']`: the key is linear_key_head_dim)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import op_cheats, port_template as pt, stub_prompts as sp, thin_repair as tr

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "port-generator"
SHEET = op_cheats.load_sheet(op_cheats.TTNN_SHEET)


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny():
    return pt.generate(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"),
                       tensors=_load("tiny-qwen4-tensors.json"), inventory=_load("tiny-qwen4-ops-inventory.json"))


def _user(tiny, function):
    text, plan = tiny
    stub = next(s for s in plan.stubs if s.function == function)
    return sp.stub_request(text, stub, sheet=SHEET, excerpts=None, inventory=_load("tiny-qwen4-ops-inventory.json"))["user"]


def test_the_signatures_of_the_functions_in_the_port_are_listed_with_their_exact_arguments(tiny):
    text, _ = tiny
    section = sp.signatures_text(text)
    assert "def _A_linear_attn(model, i, x, ids)" in section
    assert "def _layer_A(model, i, x, ids)" in section or "def _layer_A(model, i, x, ids, cos, sin)" in section
    assert "def _linear(" in section and "def _model_prelude(model, x, ids)" in section


def test_a_request_gives_the_signatures_so_a_caller_passes_the_right_arguments(tiny):
    user = _user(tiny, "_layer_A")
    assert "## Every function of the port, with its exact arguments" in user
    assert "def _A_linear_attn(model, i, x, ids)" in user
    assert "call each with exactly these arguments" in user


def test_the_signature_list_is_bounded():
    port = "\n".join(f"def f{i}(a, b):\n    return a\n" for i in range(500))
    assert len(sp.signatures_text(port)) < 6000


def test_the_config_keys_are_named_in_the_model_section(tiny):
    text, _ = tiny
    section = sp.model_text(text)
    assert "CONFIG has exactly these keys" in section
    assert "linear_key_head_dim" in section and "hidden_size" in section and "head_k_dim" not in section


def test_a_port_without_a_config_literal_has_no_key_list():
    assert "CONFIG has exactly these keys" not in sp.model_text("N_LAYERS = 2\n")


def test_a_key_error_on_a_name_that_config_lacks_gets_a_hint_with_the_keys(tiny):
    text, _ = tiny
    failure = tr.parse_failed("FAILED: the port failed at 'prefill': \"[port_model.py:304: head_k_dim = CONFIG['head_k_dim']] KeyError: 'head_k_dim'\"\n")
    hint = tr.hint_for(failure, text)
    assert hint and "head_k_dim" in hint and "linear_key_head_dim" in hint and "not a key of CONFIG" in hint


def test_a_key_error_on_a_name_config_has_gets_no_config_hint(tiny):
    text, _ = tiny
    failure = tr.parse_failed("FAILED: the port failed at 'prefill': \"[port_model.py:9: w = model['dev']['a.weight']] KeyError: 'a.weight'\"\n")
    assert tr.hint_for(failure, text) is None or "not a key of CONFIG" not in tr.hint_for(failure, text)


# ---- a layer function (arm E5: `_layer_A` called LAYERS[variant], which is `_layer_A`, and the run ended in a RecursionError) ----------------

def test_a_layer_request_lists_the_parts_of_that_layer_and_forbids_the_dispatch(tiny):
    user = _user(tiny, "_layer_A")
    assert "## The parts of this layer" in user
    part = user.split("## The parts of this layer", 1)[1].split("\n## ", 1)[0]
    assert "_A_linear_attn" in part and "_A_attn_hyper_connection" in part and "_A_mlp_hyper_connection" in part
    assert "_B_linear_attn" not in part and "_C_self_attn" not in part                      # the parts of another variant are not listed
    assert "do not call LAYERS or any _layer_ function" in part


def test_a_request_for_a_part_has_no_parts_section(tiny):
    assert "## The parts of this layer" not in _user(tiny, "_A_linear_attn")
    assert "## The parts of this layer" not in _user(tiny, "_model_prelude")
