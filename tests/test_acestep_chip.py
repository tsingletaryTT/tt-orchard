# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Targeted tests for the wiring of scripts/acestep/acestep_chip.py: which calls go to the chip stack and which stay on the CPU path. The stack itself needs a chip; its
numbers are checked on hardware (docs/run-logs/2026-10-30-acestep-chip-path.md). No chip, no weights."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "acestep"
spec = importlib.util.spec_from_file_location("acestep_chip", SCRIPTS / "acestep_chip.py")
chip = importlib.util.module_from_spec(spec)
sys.modules["acestep_chip"] = chip
spec.loader.exec_module(chip)


class FakeStack:
    instances = []

    def __init__(self, dit, get, **kw):
        self.kw = kw
        self.forwards = []
        self.names = []
        FakeStack.instances.append(self)

    def forward(self, x):
        self.forwards.append(tuple(x.shape))
        return torch.ones(x.shape[0], x.shape[1])


@pytest.fixture(autouse=True)
def fake_stack(monkeypatch):
    FakeStack.instances.clear()
    monkeypatch.setattr(chip, "ChipEncoderStack", FakeStack)


def test_pad_to_rounds_up_to_the_bucket():
    assert chip.pad_to(1, 128) == 128 and chip.pad_to(128, 128) == 128 and chip.pad_to(129, 128) == 256


class Lyric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Linear(4, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.cpu_calls = 0

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kw):
        self.cpu_calls += 1
        return SimpleNamespace(last_hidden_state=torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], 8))


def _lyric_handler():
    le = Lyric()
    cfg = SimpleNamespace(num_lyric_encoder_hidden_layers=2, hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=16,
                          rms_norm_eps=1e-6, rope_theta=1e6, layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=128, use_sliding_window=True)
    return SimpleNamespace(model=SimpleNamespace(encoder=SimpleNamespace(lyric_encoder=le), config=cfg)), le


def _worker():
    return SimpleNamespace(model=SimpleNamespace(dit=object()))


def test_lyric_encoder_runs_on_the_chip_for_one_item_with_every_token_valid():
    h, le = _lyric_handler()
    chip.install_lyric_on_chip(_worker(), h)
    stack = FakeStack.instances[-1]
    assert stack.kw["fidelity"] == "HiFi2"
    assert stack.kw["causal"] is False and stack.kw["layer_types"] == ["sliding_attention", "full_attention"] and stack.kw["sliding_window"] == 128
    out = le.forward(inputs_embeds=torch.zeros(1, 5, 4), attention_mask=torch.ones(1, 5))
    assert tuple(out.last_hidden_state.shape) == (1, 5, 8)
    assert (5, 8) in stack.forwards[1:], "the request went through the chip stack"     # forwards[0] is the warm-up at install
    assert le.cpu_calls == 0


@pytest.mark.parametrize("kind", ["batch2", "padded", "attentions"])
def test_lyric_encoder_falls_back_to_the_cpu_path_when_the_chip_stack_does_not_apply(kind):
    h, le = _lyric_handler()
    chip.install_lyric_on_chip(_worker(), h)
    stack = FakeStack.instances[-1]
    n = len(stack.forwards)
    if kind == "batch2":
        le.forward(inputs_embeds=torch.zeros(2, 5, 4), attention_mask=torch.ones(2, 5))
    elif kind == "padded":
        m = torch.ones(1, 5)
        m[0, 4] = 0
        le.forward(inputs_embeds=torch.zeros(1, 5, 4), attention_mask=m)
    else:
        le.forward(inputs_embeds=torch.zeros(1, 5, 4), attention_mask=torch.ones(1, 5), output_attentions=True)
    assert le.cpu_calls == 1 and len(stack.forwards) == n


class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(10, 6)
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=6, num_attention_heads=2, num_key_value_heads=1, head_dim=3, intermediate_size=12,
                                      rms_norm_eps=1e-6, rope_theta=1e6)
        self.norm = torch.nn.LayerNorm(6)


def test_text_encoder_runs_on_the_chip_for_batch_one_only():
    te = TextEncoder()
    calls = []
    h = SimpleNamespace(text_encoder=te, infer_text_embeddings=lambda ids: calls.append(tuple(ids.shape)) or torch.zeros(ids.shape[0], ids.shape[1], 6))
    chip.install_text_on_chip(_worker(), h)
    stack = FakeStack.instances[-1]
    assert stack.kw["causal"] is True and stack.kw["fidelity"] == "HiFi4"
    out = h.infer_text_embeddings(torch.tensor([[1, 2, 3]]))
    assert tuple(out.shape) == (1, 3, 6) and calls == [] and (3, 6) in stack.forwards
    h.infer_text_embeddings(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert calls == [(2, 3)], "a batch of two keeps the CPU path"
