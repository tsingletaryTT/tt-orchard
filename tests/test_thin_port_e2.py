"""Arm E2 of the stage 2 experiment (issue 677): the stub prompt names the `model` dict and the module constants of the generated port,
capped and refused replies are retried with a stronger sampling profile under a smaller token cap, and a failed board test is repaired by
one request per failing function (the thin repair). Arm E (PR 689) failed both board checks with `'dict' object has no attribute 'config'`:
the stubs treated `model` as an object."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from orchard import op_cheats, port_template as pt, stub_prompts as sp

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "port-generator"
SHEET = op_cheats.load_sheet(op_cheats.TTNN_SHEET)


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny():
    text, plan = pt.generate(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"),
                             tensors=_load("tiny-qwen4-tensors.json"), inventory=_load("tiny-qwen4-ops-inventory.json"))
    return text, plan


# ---- (1) the prompt says what the arguments are -------------------------------------------------------------------------------------------

def _model_keys_set_by_load(text: str) -> set:
    """The keys the generated load() puts in `model`: the dict literal it starts from and every `model["key"] = ...` after it."""
    load = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "load")
    keys = set()
    for node in ast.walk(load):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(getattr(t, "id", "") == "model" for t in node.targets):
            keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and getattr(t.value, "id", "") == "model" and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)):
                    keys.add(t.slice.value)
    return keys


def _module_constants(text: str) -> set:
    return {t.id for n in ast.parse(text).body if isinstance(n, ast.Assign) for t in n.targets
            if isinstance(t, ast.Name) and not t.id.startswith("_")}


def test_the_model_keys_in_the_prompt_are_exactly_the_keys_the_generated_load_sets(tiny):
    text, _ = tiny
    assert _model_keys_set_by_load(text) == set(pt.MODEL_KEYS)


def test_the_module_constants_in_the_prompt_are_the_ones_the_generated_port_defines(tiny):
    text, _ = tiny
    assert _module_constants(text) == set(pt.MODULE_CONSTANTS)


def test_a_stub_request_names_every_model_key_and_constant_and_says_there_is_no_attribute_access(tiny):
    text, plan = tiny
    stub = next(s for s in plan.stubs if s.function == "_model_prelude")
    user = sp.stub_request(text, stub, sheet=SHEET, excerpts=None)["user"]
    for key in pt.MODEL_KEYS:
        assert f"`{key}`" in user, key
    for name in pt.MODULE_CONSTANTS:
        assert f"`{name}`" in user, name
    assert "plain dict" in user
    for missing in ("model.config", "rotary_emb", "past_key_values"):
        assert missing in user                              # named as things that do not exist
    assert "build position tables from CONFIG" in user
    assert "`_rope_tables`" in user or "_rope_tables" in user or "`_linear`" in user       # the port's own helper names are listed


def test_a_repair_request_carries_the_same_section(tiny):
    text, _ = tiny
    req = sp.repair_request(text, "_A_linear_attn", last_check="FAILED: x", sheet=SHEET)
    assert "plain dict" in req["user"] and "`CONFIG`" in req["user"] and "`dev`" in req["user"]


def test_the_model_text_lists_only_the_constants_the_port_has():
    port = "CONFIG = {}\nN_LAYERS = 2\n\ndef _linear(model, prefix, x):\n    return x\n"
    text = sp.model_text(port)
    assert "`CONFIG`" in text and "`N_LAYERS`" in text and "`PAD_ID`" not in text and "`_linear`" in text
