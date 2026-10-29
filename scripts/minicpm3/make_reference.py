# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Record a CPU reference run of MiniCPM3-4B in the layout of orchard/audio_reference.py (read by orchard/first_divergence.py).

Three runs of the same prompt and the same greedy token path (the path comes from the float32 plain-torch run, so every run is teacher-forced
on it):
  fp32   plain-torch reference, float32           the reference
  bf16   plain-torch reference, bfloat16          floor run 0 (the chip's own dtype, CPU numerics)
  native transformers-native MiniCPM3, bfloat16   floor run 1 (the same dtype from independent library code)
`floor = (bf16, native)` is the environment floor next to every chip number. It is a floor from two independent bfloat16 implementations; the
chip adds its own matmul accumulation order on top.

Arrays: kind `layer`: `embed`-free names `layer_NN` (block output) and `layer_NN.attn` (stream after attention) for the prefill, every position.
Kind `other`: `dec0_layer_NN` (decode step 0 stream), `prefill_logits`
(last position), `dec_logits_K` (K-th decode step), `final_norm`, `tokens` (the greedy path).
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 3)[0])
from ref_model import RefModel  # noqa: E402
from orchard import audio_reference as ar  # noqa: E402

PROMPT = ("Write a short paragraph about how a lighthouse keeper spends a stormy night, mentioning the lamp, the logbook, "
          "and a ship in trouble.")


def prompt_ids(tok):
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt")
    return ids if torch.is_tensor(ids) else ids["input_ids"]


def make_recorder(out, kind_name, dtype, n_prompt, n_dec, cfg_path, weights):
    cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
    model = {"name": "openbmb/MiniCPM3-4B", "revision": "local-snapshot", "config_sha256": cfg_sha, "weights_bytes": str(weights.stat().st_size) if hasattr(weights, "stat") else "0"}
    lib = {"name": "minicpm3-reference", "commit": "n/a"}
    settings = {"prompt": PROMPT, "duration": n_prompt, "steps": n_dec, "cfg_scale": 0.0, "sampler": "greedy", "seed": 0, "dtype": dtype,
                "sample_size": 0}
    return ar.Recorder(out, model=model, library=lib, settings=settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True, help="a new directory; fp32, bf16 and native runs go under it")
    ap.add_argument("--native-model", help="a directory the transformers-native class can load (config with rope_parameters plus weights); "
                    "by default the native model is built from --model-dir and --weights")
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--prompt-file", help="use this text (read as UTF-8) in place of the built-in prompt")
    a = ap.parse_args()
    global PROMPT
    if a.prompt_file:
        PROMPT = open(a.prompt_file, encoding="utf-8").read().strip()
    from pathlib import Path
    out = Path(a.out_dir)
    weights = Path(a.weights)
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids = prompt_ids(tok)
    n = ids.shape[1]
    cfg_path = f"{a.model_dir}/config.json"
    path = None

    def run_plain(dtype_name, force=None):
        dt = getattr(torch, dtype_name)
        rec = make_recorder(out / dtype_name, "plain", "float32" if dtype_name == "float32" else "bfloat16", n, a.decode, cfg_path, weights)
        m = RefModel(a.model_dir, a.weights, dtype=dt)
        step = {"k": -1}

        def tap(name, t):
            if step["k"] == -1:
                if name.startswith("layer_") and "." not in name or name.endswith(".attn"):
                    rec.record("layer", name, t[0].float().numpy())
                elif name == "final_norm":
                    rec.record("other", "final_norm", t[0].float().numpy())
            elif step["k"] == 0 and name.startswith("layer_") and "." not in name:
                rec.record("other", "dec0_" + name, t[0].float().numpy())
        lg = m.forward(ids, 0, tap)
        rec.record("other", "prefill_logits", lg[0, -1].numpy())
        toks = []
        nxt = lg[0, -1].argmax().item() if force is None else force[0]
        for k in range(a.decode):
            toks.append(nxt)
            step["k"] = k
            lg = m.forward(torch.tensor([[nxt]]), n + k, tap)
            rec.record("other", f"dec_logits_{k}", lg[0, -1].numpy())
            nxt = lg[0, -1].argmax().item() if force is None else (force[k + 1] if k + 1 < len(force) else 0)
        rec.record("other", "tokens", np.array(toks, dtype=np.float32))
        rec.finish()
        return toks

    toks = run_plain("float32")
    print("greedy path", toks, tok.decode(toks))
    run_plain("bfloat16", force=toks)

    # transformers-native bfloat16, hooks on the blocks (stream after each block, prefill, every position)
    if a.native_model:
        from transformers import AutoModelForCausalLM
        nm = AutoModelForCausalLM.from_pretrained(a.native_model, dtype=torch.bfloat16)
    else:
        from native_model import build_native_model
        nm = build_native_model(a.model_dir, a.weights, torch.bfloat16)
    rec = make_recorder(out / "native", "native", "bfloat16", n, a.decode, cfg_path, weights)
    mode = {"k": -1}

    def hook_for(i):
        def h(mod, args, output):
            o = output[0] if isinstance(output, tuple) else output
            if mode["k"] == -1:
                rec.record("layer", f"layer_{i:02d}", o[0].float().numpy())
            elif mode["k"] == 0:
                rec.record("other", f"dec0_layer_{i:02d}", o[0].float().numpy())
        return h
    for i, layer in enumerate(nm.model.layers):
        layer.register_forward_hook(hook_for(i))
    with torch.no_grad():
        o = nm(ids, use_cache=True)
        pkv = o.past_key_values
        rec.record("other", "prefill_logits", o.logits[0, -1].float().numpy())
        for k, t in enumerate(toks):
            mode["k"] = k
            o = nm(torch.tensor([[t]]), past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            rec.record("other", f"dec_logits_{k}", o.logits[0, -1].float().numpy())
    rec.record("other", "tokens", np.array(toks, dtype=np.float32))
    rec.finish()
    print("done")


if __name__ == "__main__":
    main()
