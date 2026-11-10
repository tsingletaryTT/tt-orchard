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


# ---- (2) the token cap of a stub request, and the sampling of a retry --------------------------------------------------------------------------

from orchard import thin_port as tp                                                              # noqa: E402
from test_thin_port import Coder, run as run_step, SENTINEL                                      # noqa: E402


class Capped(Coder):
    """A coder whose n-th reply (n in `capped`) ends at the token cap; the others are good."""

    def __init__(self, capped=(), refuse=()):
        super().__init__()
        self.capped, self.refuse = set(capped), set(refuse)

    def __call__(self, url, body, timeout):
        self.bodies.append(body)
        n = len(self.bodies)
        from test_thin_port import target
        function, args = target(body["messages"][-1]["content"])
        if n in self.capped:
            return {"choices": [{"message": {"content": "x " * 50}, "finish_reason": "length"}], "usage": {"completion_tokens": 16384}}
        content = "no code here" if n in self.refuse else f"```python\ndef {function}({args}):\n    return x  # {SENTINEL}\n```"
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"completion_tokens": 40}}


def test_the_request_token_cap_is_the_smaller_of_the_tier_cap_and_the_stub_bound(tmp_path):
    coder = Capped()
    run_step(tmp_path, coder, sampling={"temperature": 0.6, "max_tokens": 16384})
    assert coder.bodies and {b["max_tokens"] for b in coder.bodies} == {tp.STUB_MAX_TOKENS}
    small = Capped()
    run_step(tmp_path / "b", small, sampling={"temperature": 0.6, "max_tokens": 1000})
    assert {b["max_tokens"] for b in small.bodies} == {1000}


def test_the_first_request_uses_the_tiers_sampling_unchanged(tmp_path):
    coder = Capped()
    run_step(tmp_path, coder, sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096})
    first = coder.bodies[0]
    assert first["temperature"] == 0.6 and first["top_p"] == 0.95 and first["top_k"] == 20 and "repetition_penalty" not in first


def test_a_retry_after_a_refusal_lowers_the_temperature_and_adds_no_penalty(tmp_path):
    coder = Capped(refuse={1})
    out, ledger, _ = run_step(tmp_path, coder, sampling={"temperature": 0.6, "max_tokens": 4096})
    assert coder.bodies[1]["temperature"] <= tp.RETRY_TEMPERATURE_MAX and "repetition_penalty" not in coder.bodies[1]
    retried = [d for d in ledger.decisions("thin port request") if d.get("attempt") == 2]
    assert retried and retried[0]["retry_sampling"] == {"temperature": tp.RETRY_TEMPERATURE_MAX}


def test_a_retry_never_raises_the_temperature(tmp_path):
    coder = Capped(refuse={1})
    run_step(tmp_path, coder, sampling={"temperature": 0.1, "max_tokens": 4096})
    assert coder.bodies[1]["temperature"] == 0.1


def test_a_retry_after_a_capped_reply_adds_a_repetition_penalty_when_the_tier_has_none(tmp_path):
    coder = Capped(capped={1})
    out, ledger, _ = run_step(tmp_path, coder, sampling={"temperature": 0.6, "max_tokens": 4096})
    assert coder.bodies[1]["repetition_penalty"] == tp.RETRY_REPETITION_PENALTY and coder.bodies[1]["temperature"] <= tp.RETRY_TEMPERATURE_MAX
    retried = [d for d in ledger.decisions("thin port request") if d.get("attempt") == 2]
    assert retried[0]["retry_sampling"] == {"temperature": tp.RETRY_TEMPERATURE_MAX, "repetition_penalty": tp.RETRY_REPETITION_PENALTY}


def test_a_tier_penalty_above_the_retry_penalty_is_kept(tmp_path):
    coder = Capped(capped={1})
    run_step(tmp_path, coder, sampling={"temperature": 0.6, "repetition_penalty": 1.3, "max_tokens": 4096})
    assert coder.bodies[1]["repetition_penalty"] == 1.3


def test_one_retry_only_and_the_second_failure_leaves_the_stub(tmp_path):
    coder = Capped(refuse={1, 2})
    out, ledger, _ = run_step(tmp_path, coder, sampling={"temperature": 0.6, "max_tokens": 4096})
    first = [d for d in ledger.decisions("thin port request") if d.get("function") == out.stubs_left[0]] if out.stubs_left else []
    assert out.stubs_left and len(first) == 2 and first[-1]["outcome"] == "left as a stub"
