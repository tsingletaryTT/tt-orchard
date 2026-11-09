#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Load a Qwen4-shaped test checkpoint ("qwen-faux") on the CPU and run it. See docs/qwen4-faux.md.

    python3 qwen4_faux_quickstart.py SNAPSHOT_DIR [--new-tokens N]

SNAPSHOT_DIR is a directory that holds config.json and model.safetensors: the output of scripts/tiny_qwen4/make_checkpoint.py or
scripts/tiny_qwen4/make_franken.py. Run it with an interpreter that has torch and transformers 5.19.0 (the class `Qwen4ExpForCausalLM`
lives there). Nothing is downloaded and no device is opened.

Why the shim: transformers builds a mixture-of-experts block in every decoder layer of `qwen4_exp`, and its config refuses
`num_experts <= 0`. These checkpoints have a plain gated MLP in every layer, so the layers are built through
orchard/skills/reference-gate-templates/qwen4_dense.py, which swaps each dense layer's MLP for the model's own `Qwen4ExpTextMLP`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SHIM = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "reference-gate-templates" / "qwen4_dense.py"


def load_shim(path: Path = SHIM):
    """The dense-layer shim as a module (it holds `prepare_config` and the `dense_layers` context manager)."""
    spec = importlib.util.spec_from_file_location("qwen4_dense", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(snapshot: Path, torch, transformers, shim=None):
    """`Qwen4ExpForCausalLM` built through the shim and loaded from `snapshot` in float32, in eval mode."""
    shim = shim or load_shim()
    raw = json.loads((Path(snapshot) / "config.json").read_text())
    config = shim.prepare_config(transformers, raw)
    with shim.dense_layers():
        model = transformers.Qwen4ExpForCausalLM.from_pretrained(str(snapshot), config=config, torch_dtype=torch.float32)
    return model.eval()


def greedy(model, torch, ids: list[int], new_tokens: int) -> list[int]:
    """`new_tokens` greedy ids, each from a full forward pass over the sequence so far (no cache)."""
    seq = list(ids)
    with torch.no_grad():
        for _ in range(new_tokens):
            logits = model(torch.tensor([seq]), use_cache=False).logits
            seq.append(int(logits[0, -1].argmax()))
    return seq[len(ids):]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--new-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    import torch
    import transformers
    model = load_model(args.snapshot, torch, transformers)
    cfg = model.config
    print(f"layers: {cfg.layer_types}")
    print(f"hidden_size {cfg.hidden_size}, hc_count {cfg.hc_count}, parameters {sum(p.numel() for p in model.parameters()):,}")
    ids = [(7 * i + 3) % cfg.vocab_size for i in range(8)]
    with torch.no_grad():
        logits = model(torch.tensor([ids]), use_cache=False).logits
    print(f"input ids {ids}")
    print(f"logits shape {tuple(logits.shape)}, next-token id {int(logits[0, -1].argmax())}")
    print(f"greedy continuation of {args.new_tokens}: {greedy(model, torch, ids, args.new_tokens)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
