"""Arm E5 (issue 677): what a stub request says about the part of the model it writes. The arm E4 run showed three things: the prelude stub was
given the whole forward of the text model (position ids, masks, rotary, the layer loop, the final mixer) and wrote torch code for all of it;
the layer stubs had no word on how the hyper-connection streams are laid out between the layers; and a delta-net stub ran to the token cap
four times because it tried to port the chunked recurrence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import op_cheats, port_template as pt, stub_prompts as sp

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "port-generator"
SHEET = op_cheats.load_sheet(op_cheats.TTNN_SHEET)

MODEL_EXCERPT = '''# Reference excerpts

## Qwen4ExpTextModel

Why: model body.

```python
class Qwen4ExpTextModel(Qwen4ExpPreTrainedModel):
    def forward(self, input_ids=None, position_ids=None, **kwargs):
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(hidden_states, position_embeddings=position_embeddings)

        hidden_states = self.hyper_connection_mixer(hidden_states)

        return Qwen4ExpModelOutputWithPast(last_hidden_state=hidden_states)
```
'''


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny():
    text, plan = pt.generate(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"),
                             tensors=_load("tiny-qwen4-tensors.json"), inventory=_load("tiny-qwen4-ops-inventory.json"))
    return text, plan


def _request(tiny, function, excerpts=MODEL_EXCERPT):
    text, plan = tiny
    stub = next(s for s in plan.stubs if s.function == function)
    return sp.stub_request(text, stub, sheet=SHEET, excerpts=excerpts, inventory=_load("tiny-qwen4-ops-inventory.json"))["user"]


# ---- (a) the part of the text model's forward before and after the layer loop --------------------------------------------------------------

def test_forward_part_splits_at_the_layer_loop():
    code = MODEL_EXCERPT.split("```python\n", 1)[1].rsplit("```", 1)[0]
    before = sp.forward_part(code, "before")
    after = sp.forward_part(code, "after")
    assert "hidden_states.repeat(1, 1, self.config.hc_count)" in before and "for layer_idx" not in before and "hyper_connection_mixer" not in before
    assert "hyper_connection_mixer" in after and "repeat(1, 1" not in after and "for layer_idx" not in after
    assert sp.forward_part("def forward(self):\n    return 1\n", "before") is None          # no layer loop: the caller keeps the whole excerpt


def test_the_prelude_request_holds_only_the_part_before_the_layer_loop(tiny):
    user = _request(tiny, "_model_prelude")
    assert "hidden_states.repeat(1, 1, self.config.hc_count)" in user
    assert "self.hyper_connection_mixer(hidden_states)" not in user
    assert "the part of forward before the layer loop" in user


def test_the_epilogue_request_holds_only_the_part_after_the_layer_loop(tiny):
    user = _request(tiny, "_model_hyper_connection_mixer")
    assert "self.hyper_connection_mixer(hidden_states)" in user
    assert "hidden_states.repeat(1, 1, self.config.hc_count)" not in user
    assert "the part of forward after the layer loop" in user


def test_a_stub_of_another_class_keeps_its_whole_excerpt(tiny):
    user = _request(tiny, "_A_mlp_hyper_connection", excerpts=MODEL_EXCERPT + "\n## Qwen4ExpTextGatedResidual\n\nWhy: gr.\n\n```python\nclass Qwen4ExpTextGatedResidual:\n    def forward(self, x):\n        return x\n```\n")
    assert "the part of forward" not in user


# ---- (b) how the streams are laid out between the layers -----------------------------------------------------------------------------------

def test_a_model_with_hyper_connections_says_how_the_streams_are_laid_out(tiny):
    text, _ = tiny
    section = sp.model_text(text)
    assert "hc_count" in section and "side by side" in section and "[1, 1, S, HC * HIDDEN]" in section
    assert "narrow" in section.lower()


def test_a_model_without_hyper_connections_has_no_such_paragraph():
    port = "CONFIG = {'hidden_size': 64}\nN_LAYERS = 2\nHIDDEN = 64\n\ndef _linear(model, prefix, x):\n    return x\n"
    assert "side by side" not in sp.model_text(port)


# ---- (c) a delta-net block ---------------------------------------------------------------------------------------------------------------

def test_a_delta_net_stub_is_told_to_write_a_plain_loop_over_tokens(tiny):
    user = _request(tiny, "_A_linear_attn", excerpts=MODEL_EXCERPT)
    assert "plain for loop over the tokens" in user and "no chunking" in user and "60 lines" in user


def test_other_stubs_do_not_get_the_delta_net_advice(tiny):
    assert "plain for loop over the tokens" not in _request(tiny, "_A_mlp_hyper_connection")
