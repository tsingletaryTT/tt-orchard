# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The generated port declares how its next token is chosen, and the claim holds (issue 183, PR 787).

`TOKEN_CHOICE = "host-row"` in a generated port says: the port returns the whole vocabulary row to the host and the caller picks the token
there. The stage 2 check records the vocabulary-per-die check as `not-applicable` for a port that says so and returned a row of the vocabulary's
width (tests/test_port_check_vocab_not_applicable.py). This file holds what makes the declaration true for the generator's output:

* the generated text has the constant and nothing else new (the old pins of the layer-rope tests hold once the line is removed);
* a generated port, on one and on two devices, returns exactly the vocabulary's width;
* the stage 5 server asks its worker only for `prefill` and `decode`, takes the worker's id as the next token, and never asks for `sample`,
  so no hook that the check could exercise stands between the logits and the token on the served path."""
import hashlib
import importlib.util

import pytest

import fake_ttnn_compute as fake
from orchard import full_port_serve, hazards
from test_port_template import std_generate, tiny_qwen4
from test_port_template_decode import tiny_model
from test_port_template_layer_rope import aion_rope_config, run_port

# The pins of tests/test_port_template_layer_rope.py before this change: the generated text of the standard small decoder and of the tiny Qwen4
# model, byte for byte as PR 791 left them.
OLD_STD = "c4ce91928d206703477caeef6ded6389f56d812e876ce32a95b6636ee07a1012"
OLD_QWEN4 = "21ff801d7aab080c0ef26c99f5481878d561ce0cc95acfb17c9760d5cdff36a0"


def without_marker(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.startswith("TOKEN_CHOICE = "))


@pytest.mark.parametrize("generate,old", [(std_generate, OLD_STD), (tiny_qwen4, OLD_QWEN4)])
def test_the_only_change_to_the_generated_text_is_the_one_constant_line(generate, old):
    text = generate()[0]
    lines = [line for line in text.split("\n") if line.startswith("TOKEN_CHOICE = ")]
    assert len(lines) == 1 and lines[0].startswith(f'TOKEN_CHOICE = "{hazards.TOKEN_CHOICE_HOST}"'), lines
    assert hashlib.sha256(without_marker(text).encode()).hexdigest() == old


def test_a_generated_port_declares_host_token_choice_and_returns_the_whole_row_on_one_and_two_devices(monkeypatch, tmp_path):
    cfg = aion_rope_config(hd=16, original=16, window=None)
    for n_devices in (1, 2):
        workdir = tmp_path / f"d{n_devices}"
        workdir.mkdir()
        mod, model, weights, got = run_port(monkeypatch, workdir, cfg, [(7 * i + 3) % 50 for i in range(12)], n_devices=n_devices)
        assert mod.TOKEN_CHOICE == hazards.TOKEN_CHOICE_HOST
        assert got.dim() == 1 and got.numel() == cfg["vocab_size"] == mod.VOCAB, (n_devices, tuple(got.shape))
        nxt = mod.decode_step(model, int(got.argmax()), 12)
        assert nxt.dim() == 1 and nxt.numel() == cfg["vocab_size"]
        assert not hasattr(mod, "sample")                     # nothing for the vocabulary check to call: the declaration is what it reads


def test_the_cached_decode_port_returns_the_whole_row_too(monkeypatch, tmp_path):
    """The generator has a second return for the cached decode (`_cached_decode_text`); the claim has to hold on that path as well."""
    mod, model, snap = tiny_model(monkeypatch, tmp_path)
    ref = mod.load(snap, fake.MeshDevice(1))
    ids = [(7 * i + 3) % 90 + 1 for i in range(14)]
    assert mod.TOKEN_CHOICE == hazards.TOKEN_CHOICE_HOST
    first = mod.prefill(model, ids[:9])
    assert first.dim() == 1 and first.numel() == mod.VOCAB
    step = mod.decode_step(model, ids[9], 9)                              # a cached step: one row through the layers with the kept state
    assert model["cache"]["n"] == 10 and step.dim() == 1 and step.numel() == mod.VOCAB
    assert step.numel() == mod._forward(ref, ids[:10]).numel()            # the same width as the whole-sequence path


# ---- the served path ------------------------------------------------------------------------------------------------------------------

def serve_module():
    spec = importlib.util.spec_from_file_location("port_serve_token_choice", full_port_serve.TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RecordingWorker:
    def __init__(self):
        self.ops = []

    def ask(self, **req):
        self.ops.append(req["op"])
        return 100 if req["op"] == "prefill" else req["token"] + 1


def test_the_server_takes_the_workers_id_as_the_token_and_asks_for_nothing_but_prefill_and_decode():
    mod = serve_module()
    worker = RecordingWorker()
    engine = mod.Engine(worker, 64, None, [], lambda m: "")
    ids, finish = engine.run([1, 2, 3], 5)
    assert ids == [100, 101, 102, 103, 104] and finish == "length"          # the worker's ids, with nothing chosen again here
    assert worker.ops == ["prefill", "decode", "decode", "decode", "decode"]
    assert "sample" not in worker.ops


def test_the_serving_template_never_names_the_sample_op():
    text = full_port_serve.TEMPLATE.read_text(encoding="utf-8")
    assert 'op="sample"' not in text and "op='sample'" not in text and '"sample"' not in text
