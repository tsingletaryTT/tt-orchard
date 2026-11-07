# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fakes for the T5Gemma encoder pin (issue 162): no transformers, no weights. The shapes match what orchard/audio_encoder_pin.py reads: the library's
`StableAudioModel.model.conditioner.conditioners[...]` holds a `T5GemmaConditioner` whose `.model` is the encoder (parameters, `.float()`, a config that
holds `_attn_implementation`)."""
from __future__ import annotations

from types import SimpleNamespace

import torch


class FakeEncoder(torch.nn.Module):
    """A T5GemmaEncoderModel stand-in. Like the real one after `from_pretrained` it starts in the checkpoint's bfloat16 with the attention
    implementation the installed transformers picks (`sdpa` in 5.19.0, `eager` in 5.21.0)."""

    def __init__(self, dtype=torch.bfloat16, attn="sdpa"):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2).to(dtype)
        self.config = SimpleNamespace(_attn_implementation=attn)
        self.attn_calls = []

    def set_attn_implementation(self, name):
        self.attn_calls.append(name)
        self.config._attn_implementation = name


class T5GemmaConditioner:
    """Named like the library's class: the pin finds the encoder by that name."""

    def __init__(self, encoder):
        self.model = encoder


def library_model(dtype=torch.bfloat16, attn="sdpa", encoders=1):
    """A stand-in for a freshly loaded library model, with `encoders` T5Gemma conditioners (0: none)."""
    conds = {f"prompt{i}": T5GemmaConditioner(FakeEncoder(dtype, attn)) for i in range(encoders)}
    return SimpleNamespace(model=SimpleNamespace(conditioner=SimpleNamespace(conditioners=conds), diffusion_objective="rf_denoiser"))
