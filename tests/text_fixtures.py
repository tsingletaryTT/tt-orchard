# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Synthetic recorded text runs for the tests of the text-model route (no model, no chip)."""
import json
from pathlib import Path

import numpy as np

from orchard import audio_reference as ar
from orchard import text_reference as tr

N_LAYERS, N_PROMPT, HIDDEN, VOCAB, N_DEC = 4, 5, 16, 24, 4


def base_arrays(seed=0):
    """A reference's arrays. The decode logits have a clear winner at each step so a small perturbation keeps the argmax."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(N_LAYERS):
        out[("layer", tr.block_name(i))] = rng.normal(size=(N_PROMPT, HIDDEN)).astype(np.float32)
        out[("layer", tr.block_name(i) + ".attn")] = rng.normal(size=(N_PROMPT, HIDDEN)).astype(np.float32)
        out[("other", "dec0_" + tr.block_name(i))] = rng.normal(size=(1, HIDDEN)).astype(np.float32)
    toks = [3, 7, 11, 2]
    pre = rng.normal(size=VOCAB).astype(np.float32)
    pre[toks[0]] += 20
    out[("other", tr.PREFILL_LOGITS)] = pre
    for k in range(N_DEC):
        lg = rng.normal(size=VOCAB).astype(np.float32)
        if k + 1 < len(toks):
            lg[toks[k + 1]] += 20
        out[("other", tr.dec_logits(k))] = lg
    out[("other", "final_norm")] = rng.normal(size=(N_PROMPT, HIDDEN)).astype(np.float32)
    out[("other", tr.TOKENS)] = np.array(toks, dtype=np.float32)
    return out


def perturb(arrays, rel, seed=1, only=None):
    """A copy with relative L2 noise `rel` on each array (or only the arrays whose name is in `only`); the token path is kept."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, a in arrays.items():
        if key == ("other", tr.TOKENS) or (only is not None and key[1] not in only):
            out[key] = a.copy()
            continue
        noise = rng.normal(size=a.shape).astype(np.float32)
        out[key] = a + noise * (rel * np.linalg.norm(a) / max(np.linalg.norm(noise), 1e-12))
    return out


def write_run(root, arrays, *, dtype="float32", model="example/text-model", prompt="hello", device=None, config_dir=None, decode_tokens=N_DEC):
    root = Path(root)
    cfg = Path(config_dir or root.parent) / "config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps({"num_hidden_layers": N_LAYERS}))
    rec = tr.make_recorder(root, model_id=model, config_path=cfg, weights_bytes=123, runtime="fake-runtime", prompt=prompt, prompt_tokens=N_PROMPT,
                           decode_tokens=decode_tokens, dtype=dtype, device=device)
    for (kind, name), a in arrays.items():
        rec.record(kind, name, a)
    rec.finish()
    return root


def load(path):
    return ar.load_run(path)
